use crate::{
    config::{
        INSECURE_DIST_CLIENT_TOKEN,
        dist::scheduler::{Auth as ClientAuth, JWTDecode, ProxyTokenDecode},
    },
    dist::http::{ClientAuthCheck, ClientClaims},
    errors::*,
    util::new_reqwest_client,
};
use anyhow::{Context, bail};
use async_trait::async_trait;
use futures::{TryFutureExt, lock::Mutex};
use itertools::Itertools;
use jwt::jwk::JwkSet;

use std::{
    collections::{BTreeMap, HashMap},
    net::{IpAddr, SocketAddr},
    time::{Duration, Instant},
};

// Check a token is equal to a fixed string
pub struct EqCheck {
    s: String,
}

#[async_trait]
impl ClientAuthCheck for EqCheck {
    async fn check(&self, _: &SocketAddr, token: &str) -> Result<ClientClaims> {
        if self.s == token {
            Ok(BTreeMap::new())
        } else {
            Err(anyhow!("Fixed token mismatch"))
        }
    }
}

impl EqCheck {
    pub fn new(s: String) -> Self {
        Self { s }
    }
}

enum ProxyTokenDecoder {
    None,
    Jwt(ValidJWTCheck),
}

impl ProxyTokenDecoder {
    async fn decode(&self, response: &str) -> Result<ClientClaims> {
        match self {
            Self::Jwt(jwt) => jwt.check_jwt_validity(response).await,
            _ => Ok(BTreeMap::new()),
        }
    }
}

struct ProxyTokenErrorRateLimiter {
    errors: Mutex<HashMap<IpAddr, Vec<Instant>>>,
    rate_limit_on_error_count: usize,
    rate_limit_on_error_window_size_secs: Duration,
}

impl ProxyTokenErrorRateLimiter {
    pub fn new(
        rate_limit_on_error_count: usize,
        rate_limit_on_error_window_size_secs: Duration,
    ) -> Self {
        Self {
            errors: Default::default(),
            rate_limit_on_error_window_size_secs,
            rate_limit_on_error_count,
        }
    }

    pub async fn check(&self, remote_addr: &IpAddr) -> Result<()> {
        let now = Instant::now();
        let mut errors = self.errors.lock().await;
        if let Some(window) = errors.get_mut(remote_addr) {
            self.truncate_window(&now, window);
            if window.is_empty() {
                errors.remove(remote_addr);
            } else if window.len() >= self.rate_limit_on_error_count {
                bail!("Too many unauthenticated requests within time window")
            }
        }
        Ok(())
    }

    pub async fn record(&self, remote_addr: IpAddr) {
        let now = Instant::now();
        let mut errors = self.errors.lock().await;
        let window = errors.entry(remote_addr).or_default();
        self.truncate_window(&now, window);
        window.push(now);
    }

    fn truncate_window(&self, now: &Instant, window: &mut Vec<Instant>) {
        // Drop all error timestamps older than `rate_limit_on_error_window_size_secs`
        window.retain(|then| now.duration_since(*then) < self.rate_limit_on_error_window_size_secs);
        // Drop all but the last `rate_limit_on_error_count` number of error timestamps
        window.drain(0..window.len().saturating_sub(self.rate_limit_on_error_count));
    }
}

// Don't check a token is valid (it may not even be a JWT) just forward it to
// an API and check for success
pub struct ProxyTokenCheck {
    cache: Mutex<HashMap<String, (Instant, ClientClaims)>>,
    client: reqwest::Client,
    decoder: ProxyTokenDecoder,
    rate_limiter: ProxyTokenErrorRateLimiter,
    token_ttl: Option<Duration>,
    url: String,
}

#[async_trait]
impl ClientAuthCheck for ProxyTokenCheck {
    async fn check(&self, remote_addr: &SocketAddr, token: &str) -> Result<ClientClaims> {
        let remote_addr = remote_addr.ip();
        match self
            .rate_limiter
            .check(&remote_addr)
            .and_then(|_| self.check_token_with_forwarding(token))
            .await
        {
            Ok(claims) => Ok(claims),
            Err(err) => {
                tracing::warn!("Proxying token validation failed: {err}");
                self.rate_limiter.record(remote_addr).await;
                Err(err)
            }
        }
    }
}

impl ProxyTokenCheck {
    pub async fn new(
        url: String,
        token_ttl: Option<u64>,
        rate_limit_on_error_count: Option<usize>,
        rate_limit_on_error_window_size_secs: Option<u64>,
        decode: Option<ProxyTokenDecode>,
    ) -> Result<Self> {
        let decoder = match decode {
            Some(ProxyTokenDecode::Jwt(JWTDecode {
                audience,
                claims,
                issuer,
                jwks_url,
                leeway,
            })) => ProxyTokenDecoder::Jwt(
                ValidJWTCheck::new(audience, issuer, jwks_url, claims, leeway)
                    .await
                    .context("Failed to create a checker for valid JWTs")?,
            ),
            _ => ProxyTokenDecoder::None,
        };

        Ok(Self {
            cache: Mutex::new(HashMap::new()),
            client: new_reqwest_client(None),
            decoder,
            rate_limiter: ProxyTokenErrorRateLimiter::new(
                rate_limit_on_error_count.unwrap_or(5),
                Duration::from_secs(rate_limit_on_error_window_size_secs.unwrap_or(60)),
            ),
            token_ttl: token_ttl.map(Duration::from_secs),
            url,
        })
    }

    async fn check_token_with_forwarding(&self, token: &str) -> Result<ClientClaims> {
        if token.is_empty() {
            bail!("Token must not be empty");
        }

        // If the token is cached and not cache has not expired, return it
        if let Some(token_ttl) = self.token_ttl {
            let mut cache = self.cache.lock().await;
            let entry = cache
                .get(token)
                .filter(|(cached_at, _)| cached_at.elapsed() < token_ttl)
                .map(|(_, claims)| claims);
            if let Some(claims) = entry {
                return Ok(claims.clone());
            } else {
                cache.remove(token);
            }
        }

        tracing::trace!("Validating token by forwarding to {}", self.url);

        // Make a request to another API, which as a side effect should actually check the token
        let res = self
            .client
            .get(&self.url)
            .bearer_auth(token)
            .send()
            .await
            .context("Failed to make request to proxying url")?;

        if !res.status().is_success() {
            bail!("Token forwarded to {} returned {}", self.url, res.status());
        }

        let mime = res
            .headers()
            .get(http::header::CONTENT_TYPE)
            .map(|t| t.to_str().unwrap_or("application/text"));

        let auth = match mime.unwrap_or("application/text") {
            "application/json" => {
                let json = res.json::<serde_json::Value>().await?;
                json.as_object()
                    .and_then(|o| o.get("token"))
                    .and_then(|s| s.as_str())
                    .map(|s| s.to_owned())
                    .unwrap_or_default()
            }
            _ => res.text().await?,
        };

        let claims = self
            .decoder
            .decode(&auth)
            .await
            .or_else(|_| Result::Ok(BTreeMap::new()))?;

        // Cache the token
        if self.token_ttl.is_some() {
            self.cache
                .lock()
                .await
                .insert(token.to_owned(), (Instant::now(), claims.clone()));
        }

        Ok(claims)
    }
}

// Check a JWT is valid
pub struct ValidJWTCheck {
    audience: Vec<String>,
    claims_to_check: Vec<(String, String)>,
    claims_to_log: Vec<(String, String)>,
    issuer: Vec<String>,
    keys: HashMap<String, jwt::DecodingKey>,
    leeway: u64,
}

#[async_trait]
impl ClientAuthCheck for ValidJWTCheck {
    async fn check(&self, _: &SocketAddr, token: &str) -> Result<ClientClaims> {
        self.check_jwt_validity(token).await
    }
}

impl ValidJWTCheck {
    pub async fn new(
        audience: Vec<String>,
        issuer: Vec<String>,
        jwks_url: Vec<String>,
        claims: HashMap<String, String>,
        leeway: u64,
    ) -> Result<Self> {
        let mut keys = HashMap::new();

        crate::util::install_crypto_provider();
        for url in jwks_url {
            let res = reqwest::get(url)
                .await
                .context("Failed to make request to JWKs url")?;

            if !res.status().is_success() {
                bail!("Could not retrieve JWKs, HTTP error: {}", res.status())
            }

            res.json::<JwkSet>()
                .await
                .context("Failed to parse JWKs json")?
                .keys
                .into_iter()
                .filter_map(|jwk| {
                    jwk.common.key_id.as_ref().and_then(|kid| {
                        jwt::DecodingKey::from_jwk(&jwk)
                            .ok()
                            .map(|key| (kid.clone(), key))
                    })
                })
                .for_each(|(kid, key)| {
                    keys.insert(kid, key);
                });
        }

        let (claims_to_log, claims_to_check): (Vec<_>, Vec<_>) =
            claims.into_iter().partition_map(|(key, val)| {
                if matches!(val.as_str(), "*" | "") {
                    itertools::Either::Left((key, val))
                } else {
                    itertools::Either::Right((key, val))
                }
            });

        trace!("claims_to_log: {claims_to_log:?}, claims_to_check: {claims_to_check:?}");

        Ok(Self {
            audience,
            claims_to_check,
            claims_to_log,
            issuer,
            keys,
            leeway,
        })
    }

    async fn check_jwt_validity(&self, token: &str) -> Result<ClientClaims> {
        tracing::trace!("Validating JWT");

        if token.is_empty() {
            bail!("Token must not be empty");
        }

        let header = jwt::decode_header(token).context("Could not decode jwt header")?;

        // Prepare validation
        let mut validation = jwt::Validation::new(header.alg);
        validation.set_audience(self.audience.as_slice());
        validation.set_issuer(self.issuer.as_slice());
        validation.leeway = self.leeway;

        let value_to_string = |v: &serde_json::Value| {
            if let Some(s) = v.as_str() {
                s.to_owned()
            } else {
                v.to_string()
            }
        };

        let decode = |key: &jwt::DecodingKey| {
            let jwt::TokenData { claims, .. } =
                jwt::decode::<serde_json::Value>(token, key, &validation)?;

            let mut valid_claims = BTreeMap::new();

            if let Some(claims) = claims.as_object() {
                let mut passed = self.claims_to_check.is_empty();
                for (key, req) in self.claims_to_check.iter() {
                    let val = claims
                        .get(key.as_str())
                        .map(value_to_string)
                        .unwrap_or_default();
                    if &val == req {
                        passed = true;
                        valid_claims.insert(key.to_owned(), val);
                    }
                }
                if !passed {
                    bail!("Failed to validate required claims");
                }
                for (key, _) in self.claims_to_log.iter() {
                    if let Some(val) = claims.get(key.as_str()).map(value_to_string) {
                        valid_claims.insert(key.to_owned(), val);
                    }
                }
            }

            Ok(valid_claims)
        };

        // Decode the JWT and return the claims

        if let Some(key) = header.kid.as_ref().and_then(|kid| self.keys.get(kid)) {
            decode(key)
        } else {
            self.keys
                .values()
                .try_for_each(|key| {
                    if let Ok(claims) = decode(key) {
                        std::ops::ControlFlow::Break(claims)
                    } else {
                        std::ops::ControlFlow::Continue(())
                    }
                })
                .map_break(Ok)
                .break_value()
                .unwrap_or_else(|| bail!("Unable to decode or validate JWT"))
        }
    }
}

pub async fn new_client_auth_check(client_auth: ClientAuth) -> Result<Box<dyn ClientAuthCheck>> {
    Ok(match client_auth {
        ClientAuth::Insecure => Box::new(EqCheck::new(INSECURE_DIST_CLIENT_TOKEN.to_owned())),
        ClientAuth::Token { token } => Box::new(EqCheck::new(token)),
        ClientAuth::Jwt(JWTDecode {
            audience,
            issuer,
            jwks_url,
            claims,
            leeway,
        }) => Box::new(
            ValidJWTCheck::new(audience, issuer, jwks_url, claims, leeway)
                .await
                .context("Failed to create a checker for valid JWTs")?,
        ) as Box<dyn ClientAuthCheck>,
        ClientAuth::ProxyToken {
            url,
            cache_secs,
            decode,
            rate_limit_on_error_count,
            rate_limit_on_error_window_size_secs,
        } => Box::new(
            ProxyTokenCheck::new(
                url,
                cache_secs,
                rate_limit_on_error_count,
                rate_limit_on_error_window_size_secs,
                decode,
            )
            .await?,
        ) as Box<dyn ClientAuthCheck>,
    })
}

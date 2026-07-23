// Copyright 2016 Mozilla Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#[cfg(feature = "dist-client")]
pub use self::client::Client;

#[cfg(feature = "dist-server")]
pub use self::scheduler::{ClientAuthCheck, ClientClaims, Scheduler};

#[cfg(any(feature = "dist-client", feature = "dist-server"))]
pub use self::common::bincode_req_fut;

pub use self::common::{bincode_deserialize, bincode_serialize};

mod common {

    use bytes::Buf;
    use reqwest::header;

    use crate::errors::*;

    pub async fn bincode_deserialize<T>(bytes: opendal::Buffer) -> Result<T>
    where
        T: for<'de> serde::Deserialize<'de> + Send + 'static,
    {
        tokio::task::spawn_blocking(move || bincode::deserialize_from(bytes.reader()))
            .await
            .map_err(anyhow::Error::new)?
            .map_err(anyhow::Error::new)
    }

    pub async fn bincode_serialize<T>(value: T) -> Result<Vec<u8>>
    where
        T: serde::Serialize + Send + 'static,
    {
        tokio::task::spawn_blocking(move || bincode::serialize(&value))
            .await
            .map_err(anyhow::Error::new)?
            .map_err(anyhow::Error::new)
    }

    // Note that content-length is necessary due to https://github.com/tiny-http/tiny-http/issues/147
    pub trait ReqwestRequestBuilderExt: Sized {
        fn bincode<T: serde::Serialize + ?Sized>(self, bincode: &T) -> Result<Self>;
        fn bytes(self, bytes: Vec<u8>) -> Self;
    }

    impl ReqwestRequestBuilderExt for reqwest::RequestBuilder {
        fn bincode<T: serde::Serialize + ?Sized>(self, bincode: &T) -> Result<Self> {
            let bytes =
                bincode::serialize(bincode).context("Failed to serialize body to bincode")?;
            Ok(self.bytes(bytes))
        }
        fn bytes(self, bytes: Vec<u8>) -> Self {
            self.header(
                header::CONTENT_TYPE,
                mime::APPLICATION_OCTET_STREAM.to_string(),
            )
            .header(header::CONTENT_LENGTH, bytes.len())
            .body(bytes)
        }
    }

    #[cfg(any(feature = "dist-client", feature = "dist-server"))]
    pub async fn bincode_req_fut<T: serde::de::DeserializeOwned + 'static>(
        req: reqwest::RequestBuilder,
    ) -> Result<T> {
        let res = match req.send().await {
            Ok(res) => res,
            Err(err) => {
                trace!(
                    "Request error: {}",
                    [
                        err.url().map(|u| format!("url={u:?}")).unwrap_or_default(),
                        err.status()
                            .map(|u| format!("status={u:?}"))
                            .unwrap_or_default(),
                        format!("err={err:?}"),
                    ]
                    .join(", ")
                );
                return Err(err.into());
            }
        };

        let url = res.url().clone();
        let status = res.status();
        let bytes = match res.bytes().await {
            Ok(b) => b,
            Err(err) => {
                debug!("Request body error: url={url}, status={status}, err={err}");
                return Err(err.into());
            }
        };
        if !status.is_success() {
            let errmsg = format!(
                "Error {}: {}",
                status.as_u16(),
                String::from_utf8_lossy(&bytes)
            );
            if status.is_client_error() {
                anyhow::bail!(HttpClientError(errmsg));
            } else {
                anyhow::bail!(errmsg);
            }
        } else {
            Ok(bincode::deserialize(&bytes)?)
        }
    }
}

pub mod urls {
    pub fn scheduler_status(scheduler_url: &reqwest::Url) -> reqwest::Url {
        scheduler_url
            .join("/api/v2/status")
            .expect("failed to create alloc job url")
    }
    pub fn scheduler_new_job(scheduler_url: &reqwest::Url) -> reqwest::Url {
        scheduler_url
            .join("/api/v2/jobs/new")
            .expect("failed to create new job url")
    }
    pub fn scheduler_put_job(scheduler_url: &reqwest::Url, job_id: &str) -> reqwest::Url {
        scheduler_url
            .join(&format!("/api/v2/job/{job_id}"))
            .expect("failed to create put job url")
    }
    pub fn scheduler_run_job(scheduler_url: &reqwest::Url, job_id: &str) -> reqwest::Url {
        scheduler_url
            .join(&format!("/api/v2/job/{job_id}"))
            .map(|mut url| {
                url.set_query(Some("v=2"));
                url
            })
            .expect("failed to create run job url")
    }
    pub fn scheduler_del_job(scheduler_url: &reqwest::Url, job_id: &str) -> reqwest::Url {
        scheduler_url
            .join(&format!("/api/v2/job/{job_id}"))
            .expect("failed to create put job url")
    }
    pub fn scheduler_submit_toolchain(
        scheduler_url: &reqwest::Url,
        archive_id: &str,
    ) -> reqwest::Url {
        scheduler_url
            .join(&format!("/api/v2/toolchain/{archive_id}"))
            .expect("failed to create submit toolchain url")
    }
}

#[cfg(feature = "dist-server")]
mod scheduler {

    use async_trait::async_trait;

    use axum::{
        RequestExt, RequestPartsExt, Router,
        body::{Body, Bytes},
        extract::{
            ConnectInfo, Extension, FromRequest, FromRequestParts, MatchedPath, Path, Request,
        },
        http::{StatusCode, request::Parts},
        response::{IntoResponse, Response},
        routing,
    };

    use axum_extra::{
        TypedHeader,
        headers::{Authorization, ContentType, Header, HeaderValue, authorization::Bearer},
    };

    use futures::{TryFutureExt, TryStreamExt};
    use memmap2::Mmap;
    use serde_json::json;

    use std::{
        collections::BTreeMap,
        io::{self, Seek},
        net::SocketAddr,
        str::FromStr,
        sync::Arc,
        time::{Duration, Instant},
    };

    use tokio::{io::AsyncReadExt, task_local};
    use tokio_util::{
        compat::TokioAsyncReadCompatExt,
        io::{ReaderStream, StreamReader, SyncIoBridge},
    };
    use tower::ServiceBuilder;
    use tower_http::{
        request_id::{MakeRequestUuid, PropagateRequestIdLayer, SetRequestIdLayer},
        sensitive_headers::{SetSensitiveRequestHeadersLayer, SetSensitiveResponseHeadersLayer},
        trace::{DefaultMakeSpan, DefaultOnResponse, TraceLayer},
    };

    use crate::{
        config::DistNetworkingKeepalive,
        dist::{
            RunJobRequest, RunJobRequestV2, SchedulerService, Toolchain,
            http::{bincode_deserialize, bincode_serialize},
            metrics::{GaugeRecorder, Metrics},
        },
        errors::*,
    };

    pub type ClientClaims = BTreeMap<String, String>;

    #[async_trait]
    pub trait ClientAuthCheck: Send + Sync {
        async fn check(&self, remote_addr: &SocketAddr, token: &str) -> Result<ClientClaims>;
    }

    // Make our own error that wraps `anyhow::Error`.
    struct AppError(anyhow::Error);

    // Tell axum how to convert `AppError` into a response.
    impl IntoResponse for AppError {
        fn into_response(self) -> Response {
            tracing::error!("AppError: {:?}", self.0);
            (StatusCode::INTERNAL_SERVER_ERROR, format!("{}", self.0)).into_response()
        }
    }

    // This enables using `?` on functions that return `Result<_, anyhow::Error>` to turn them into
    // `Result<_, AppError>`. That way you don't need to do that manually.
    impl<E> From<E> for AppError
    where
        E: Into<anyhow::Error>,
    {
        fn from(err: E) -> Self {
            Self(err.into())
        }
    }

    #[derive(Clone, Debug, PartialEq)]
    struct Accepted(pub Vec<String>);

    impl Accepted {
        pub async fn serialize<T>(&self, res: T) -> Result<Response>
        where
            T: serde::Serialize + Send + 'static,
        {
            let Self(mimetypes) = self;
            for mimetype in mimetypes {
                match mimetype.as_ref() {
                    "application/json" => {
                        return Ok(axum::response::Json(json!(res)).into_response());
                    }
                    "application/octet-stream" | "*/*" => {
                        return Ok(match bincode_serialize(res).await {
                            Ok(body) => (StatusCode::OK, body),
                            Err(err) => {
                                tracing::error!("Failed to serialize response body: {err:?}");
                                (
                                    StatusCode::INTERNAL_SERVER_ERROR,
                                    format!("Failed to serialize response body: {err:#}")
                                        .into_bytes(),
                                )
                            }
                        }
                        .into_response());
                    }
                    _ => continue,
                }
            }

            let msg = "Request must accept application/json or application/octet-stream";
            tracing::error!(msg);

            Ok((StatusCode::BAD_REQUEST, msg.to_string().into_bytes()).into_response())
        }
    }

    #[derive(Clone, Debug, PartialEq)]
    struct AcceptedMimeTypes(pub Accepted);

    impl Header for AcceptedMimeTypes {
        fn name() -> &'static http::header::HeaderName {
            &http::header::ACCEPT
        }

        fn decode<'i, I: Iterator<Item = &'i HeaderValue>>(
            values: &mut I,
        ) -> std::result::Result<Self, axum_extra::headers::Error> {
            use itertools::Itertools;

            Ok(Self(Accepted(
                values
                    .flat_map(|value| {
                        value
                            .to_str()
                            .unwrap_or("/*")
                            .split(",")
                            .map(|s| s.trim())
                            .filter_map(|s| {
                                let mimetype;
                                let mut quality = 1.0;
                                if s.contains(";") {
                                    let mut parts = s.split(";");
                                    if let Some(m) = parts.next() {
                                        mimetype = m.to_owned();
                                    } else {
                                        return None;
                                    }
                                    if let Some(q) = parts.next()
                                        && let Some(q) = q.split("=").last()
                                        && let Ok(q) = q.parse()
                                    {
                                        quality = q;
                                    }
                                } else {
                                    mimetype = s.to_owned();
                                }
                                Some((mimetype, quality))
                            })
                    })
                    .sorted_by(|(lhs, lhs_q), (rhs, rhs_q)| {
                        if rhs_q > lhs_q {
                            return std::cmp::Ordering::Greater;
                        }
                        match (lhs.ends_with("/*"), rhs.ends_with("/*")) {
                            (true, true) => std::cmp::Ordering::Equal,
                            (true, false) => std::cmp::Ordering::Greater,
                            (false, true) => std::cmp::Ordering::Less,
                            (false, false) => std::cmp::Ordering::Equal,
                        }
                    })
                    .map(|(mimetype, _)| mimetype)
                    .unique()
                    .collect::<Vec<_>>(),
            )))
        }

        fn encode<E: Extend<HeaderValue>>(&self, values: &mut E) {
            let Self(Accepted(accepts)) = self;
            values.extend(
                accepts
                    .iter()
                    .filter_map(|s| HeaderValue::from_str(s.as_str()).ok()),
            );
        }
    }

    task_local! {
        static CLIENT_AUTH: std::result::Result<RequireAuth, (StatusCode, String)>;
    }

    // Verify authenticated sccache clients
    #[derive(Clone, Debug, Default)]
    struct RequireAuth(ClientClaims);

    impl<S> FromRequestParts<S> for RequireAuth
    where
        S: Send + Sync,
    {
        type Rejection = (StatusCode, String);

        async fn from_request_parts(
            parts: &mut Parts,
            _state: &S,
        ) -> std::result::Result<Self, Self::Rejection> {
            if let Ok(auth_result) = CLIENT_AUTH.try_get() {
                return auth_result;
            }

            let TypedHeader(Authorization(bearer)) = parts
                .extract::<TypedHeader<Authorization<Bearer>>>()
                .await
                .map_err(|_| {
                    (
                        StatusCode::UNAUTHORIZED,
                        "Missing authentication token".into(),
                    )
                })?;

            let ConnectInfo(remote_addr) = parts
                .extract::<ConnectInfo<SocketAddr>>()
                .await
                .map_err(|_| (StatusCode::BAD_REQUEST, "Bad request".into()))?;

            let Extension(this) = parts
                .extract::<Extension<Arc<SchedulerState>>>()
                .await
                .map_err(|_| {
                    (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "Internal server error".into(),
                    )
                })?;

            let mut warnings = vec![];

            for client_auth in this.client_auth.iter() {
                match client_auth.check(&remote_addr, bearer.token()).await {
                    Ok(res) => {
                        return Ok(RequireAuth(res));
                    }
                    Err(err) => warnings.push(format!(
                        "[RequireAuth({remote_addr})]: Authentication failure: {err}"
                    )),
                }
            }

            warnings.iter().for_each(|msg| tracing::warn!(msg));

            Err((StatusCode::UNAUTHORIZED, "Unauthorized".into()))
        }
    }

    struct Bincode<T>(T);

    impl<S, T> FromRequest<S> for Bincode<T>
    where
        Bytes: FromRequest<S>,
        S: Send + Sync,
        T: serde::de::DeserializeOwned + Send + 'static,
    {
        type Rejection = (StatusCode, String);

        async fn from_request(
            mut req: Request,
            state: &S,
        ) -> std::result::Result<Self, Self::Rejection> {
            let TypedHeader(content_type) =
                req.extract_parts::<TypedHeader<ContentType>>()
                    .await
                    .map_err(|_| (StatusCode::BAD_REQUEST, "Bad request".into()))?;

            let data = if content_type == ContentType::octet_stream() {
                Bytes::from_request(req, state)
                    .await
                    .map_err(|_| (StatusCode::BAD_REQUEST, "Bad request".into()))?
            } else {
                return Err((StatusCode::BAD_REQUEST, "Wrong content type".into()));
            };

            let data = bincode_deserialize::<T>(data.into())
                .await
                .map_err(|err| (StatusCode::BAD_REQUEST, err.to_string()))?;

            Ok(Self(data))
        }
    }

    struct RunJobBody(RunJobRequestV2);

    impl<S> FromRequest<S> for RunJobBody
    where
        Bytes: FromRequest<S>,
        S: Send + Sync,
    {
        type Rejection = (StatusCode, String);

        async fn from_request(
            mut req: Request,
            _state: &S,
        ) -> std::result::Result<Self, Self::Rejection> {
            let Path::<String>(job_id) =
                req.extract_parts::<Path<String>>().await.map_err(|_| {
                    (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "Internal server error".into(),
                    )
                })?;

            let Extension::<Arc<HTTPMetrics>>(metrics) = req
                .extract_parts::<Extension<Arc<HTTPMetrics>>>()
                .await
                .map_err(|_| {
                    (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "Internal server error".into(),
                    )
                })?;

            let version_query_param =
                url::form_urlencoded::parse(req.uri().query().unwrap_or("").as_bytes())
                    .find_map(|(k, v)| if k == "v" { Some(v) } else { None });

            let mut job = match version_query_param {
                Some(std::borrow::Cow::Borrowed("2")) => {
                    req.extract::<Bincode<RunJobRequestV2>, _>().await?.0
                }
                _ => {
                    let job = req.extract::<Bincode<RunJobRequest>, _>().await?.0;
                    RunJobRequestV2 {
                        labels: None,
                        outputs: job.outputs,
                        toolchain: job.toolchain,
                        command: job.command,
                    }
                }
            };

            let labels = job.labels.get_or_insert_default();

            // Add labels from the auth middleware to `job.labels`
            if let Some(claims) = metrics.scoped_labels() {
                for (k, v) in claims.iter() {
                    labels.insert(k.clone(), v.clone());
                }
            }

            // Add `job_id` to  `job.labels`
            labels.insert("job_id".into(), job_id.clone());

            Ok(Self(job))
        }
    }

    struct RequestBodyTokioAsyncRead(Box<dyn tokio::io::AsyncRead + Send + Unpin>);

    impl<S> FromRequest<S> for RequestBodyTokioAsyncRead
    where
        S: Send + Sync,
    {
        type Rejection = StatusCode;

        async fn from_request(
            req: Request,
            _state: &S,
        ) -> std::result::Result<Self, Self::Rejection> {
            // Convert the request body into a `tokio::io::AsyncRead`
            Ok(Self(Box::new(StreamReader::new(
                req.into_body().into_data_stream().map_err(io::Error::other),
            ))))
        }
    }

    const REQUEST_TIME: &str = "sccache::scheduler::http::request_time";
    const REQUEST_COUNT: &str = "sccache::scheduler::http::request_count";
    const REQUESTS_ACTIVE: &str = "sccache::scheduler::http::requests_active";

    struct HTTPMetrics {
        active: std::sync::Mutex<BTreeMap<Arc<BTreeMap<String, String>>, Arc<GaugeRecorder>>>,
        metrics: Metrics,
    }

    impl HTTPMetrics {
        fn new(metrics: Metrics) -> Self {
            metrics::describe_histogram!(
                REQUEST_TIME,
                metrics::Unit::Seconds,
                "The time to handle each request in seconds."
            );
            metrics::describe_counter!(
                REQUEST_COUNT,
                metrics::Unit::Count,
                "The total number of requests."
            );
            metrics::describe_gauge!(
                REQUESTS_ACTIVE,
                metrics::Unit::Count,
                "The number of active requests."
            );

            Self {
                active: Default::default(),
                metrics,
            }
        }

        fn active(&self, labels: Arc<BTreeMap<String, String>>) -> Arc<GaugeRecorder> {
            self.active
                .lock()
                .unwrap()
                .entry(labels.clone())
                .or_insert_with(|| {
                    Arc::new(GaugeRecorder::new(REQUESTS_ACTIVE.into(), Some(labels)))
                })
                .clone()
        }

        fn render_to_write(&self, writer: Box<dyn io::Write>) -> io::Result<()> {
            self.metrics.render_to_write(writer)
        }

        fn scoped_labels(&self) -> Option<Arc<BTreeMap<String, String>>> {
            self.metrics.scoped_labels()
        }

        fn scope_with_labels<F>(
            &self,
            local_labels: &BTreeMap<String, String>,
            f: F,
        ) -> tokio::task::futures::TaskLocalFuture<Arc<BTreeMap<String, String>>, F>
        where
            F: Future,
        {
            self.metrics.scope_with_labels(local_labels, f)
        }
    }

    struct SchedulerState {
        service: Arc<dyn SchedulerService>,
        // Test whether clients are permitted to use the scheduler
        client_auth: Vec<Box<dyn ClientAuthCheck>>,
    }

    pub struct Scheduler {
        state: Arc<SchedulerState>,
    }

    impl Scheduler {
        pub fn new(
            service: Arc<dyn SchedulerService>,
            client_auth: Vec<Box<dyn ClientAuthCheck>>,
        ) -> Self {
            Self {
                state: Arc::new(SchedulerState {
                    service,
                    client_auth,
                }),
            }
        }

        fn metrics(app: Router, metrics: Metrics) -> Router {
            let app = if let Some(path) = metrics.listen_path() {
                app.route(
                    &path,
                    routing::get(|Extension::<Arc<HTTPMetrics>>(metrics)| async move {
                        let (reader, writer) = tokio::io::duplex(4096);
                        let writer = Box::new(SyncIoBridge::new(writer));
                        tokio::spawn(async move {
                            let _ = tokio::task::spawn_blocking(move || {
                                let _ = metrics.render_to_write(writer);
                            })
                            .await;
                        });
                        Body::from_stream(ReaderStream::new(reader))
                    }),
                )
            } else {
                app
            };

            async fn record_metrics(
                mut req: Request,
                next: axum::middleware::Next,
            ) -> impl IntoResponse {
                let start = Instant::now();

                // Check auth status before routing
                let auth = req.extract_parts::<RequireAuth>().await;

                // Labels include auth claims to support `GROUPBY "user_id"` etc.
                let mut labels = auth.as_ref().map(|auth| auth.0.clone()).unwrap_or_default();

                let authorized = auth.is_ok().to_string();
                let method = req.method().to_string();
                let path = req
                    .extensions()
                    .get::<MatchedPath>()
                    .map(|path| path.as_str().to_string())
                    .unwrap_or_else(|| req.uri().path().to_owned());

                labels.insert("method".into(), method.clone());
                labels.insert("path".into(), path.clone());

                let labels = Arc::new(labels);

                let res = req
                    .extract_parts::<Extension<Arc<HTTPMetrics>>>()
                    .await
                    .map(|Extension(metrics): Extension<Arc<HTTPMetrics>>| {
                        let labels = labels.clone();
                        CLIENT_AUTH.scope(auth, async move {
                            let counter = metrics.active(labels.clone());
                            let _handle = counter.increment();
                            metrics.scope_with_labels(&labels, next.run(req)).await
                        })
                    })
                    .map_err(|err| AppError(anyhow!(err)).into_response());

                let res = match res {
                    Ok(res) => res.await,
                    Err(res) => res,
                };

                let mut labels = Arc::unwrap_or_clone(labels);
                labels.insert("authorized".into(), authorized.clone());
                labels.insert("status".into(), res.status().as_u16().to_string());

                metrics::counter!(REQUEST_COUNT, &labels).increment(1);
                metrics::histogram!(REQUEST_TIME, &labels).record(start.elapsed().as_secs_f64());

                res
            }

            // Define this at the end so we also track metrics for the `/metrics` route
            app.route_layer(axum::middleware::from_fn(record_metrics))
                .layer(Extension(Arc::new(HTTPMetrics::new(metrics))))
        }

        fn routes() -> axum::Router {
            Router::new()
                .route(
                    "/api/v2/status",
                    routing::get(
                        |TypedHeader(AcceptedMimeTypes(mime)),
                         Extension::<Arc<SchedulerState>>(state)| async move {
                            state
                                .service
                                .get_status()
                                .and_then(|res| mime.serialize(res))
                                .map_err(AppError)
                                .await
                        },
                    ),
                )
                //
                // TOOLCHAIN OPERATIONS
                //
                // HEAD
                .route(
                    "/api/v2/toolchain/{archive_id}",
                    routing::head(
                        |Extension::<Arc<SchedulerState>>(state),
                         Path(archive_id): Path<String>| async move {
                            if state.service.has_toolchain(&Toolchain { archive_id }).await {
                                (StatusCode::OK).into_response()
                            } else {
                                (StatusCode::NOT_FOUND).into_response()
                            }
                        },
                    ),
                )
                // PUT
                .route(
                    "/api/v2/toolchain/{archive_id}",
                    routing::put(
                        |TypedHeader(AcceptedMimeTypes(mime)),
                         Extension::<Arc<SchedulerState>>(state),
                         Path(archive_id): Path<String>,
                         req: Request| async move {
                            let RequestBodyTokioAsyncRead(reader) = req
                                .extract::<RequestBodyTokioAsyncRead, _>()
                                .await
                                .map_err(|_| AppError(anyhow!("")).into_response())?;

                            // First read the toolchain into a tempfile
                            let mut toolchain = crate::util::tempfile()
                                .map_err(|_| AppError(anyhow!("")).into_response())?;

                            futures::io::copy(
                                reader.compat(),
                                &mut futures::io::AllowStdIo::new(toolchain.as_file_mut()),
                            )
                            .await
                            .map_err(|_| (StatusCode::BAD_REQUEST, "Invalid toolchain"))
                            .map_err(IntoResponse::into_response)?;

                            let service = state.service.clone();

                            // Once we've received the entire input, spawn a new task to write them
                            // to storage that won't be cancelled. This ensures we don't perform an
                            // incomplete write if the client disconnects prematurely, for example,
                            // if a user ctrl-C's their build.
                            tokio::spawn(async move {
                                let toolchain = Ok(toolchain.into_file())
                                    .and_then(|mut tc| tc.rewind().map(|_| tc))
                                    .and_then(|tc| unsafe { Mmap::map(&tc) }.map(Bytes::from_owner))
                                    .map_err(|_| AppError(anyhow!("")).into_response())?;
                                service
                                    .put_toolchain(&Toolchain { archive_id }, toolchain.into())
                                    .and_then(|res| mime.serialize(res))
                                    .map_err(|e| AppError(e).into_response())
                                    .await
                            })
                            .await
                            .map_err(|_| AppError(anyhow!("")).into_response())?
                        },
                    ),
                )
                // DELETE
                .route(
                    "/api/v2/toolchain/{archive_id}",
                    routing::delete(
                        |TypedHeader(AcceptedMimeTypes(mime)),
                         Extension::<Arc<SchedulerState>>(state),
                         Path(archive_id): Path<String>| async move {
                            state
                                .service
                                .del_toolchain(&Toolchain { archive_id })
                                .and_then(|res| mime.serialize(res))
                                .map_err(AppError)
                                .await
                        },
                    ),
                )
                //
                // JOB OPERATIONS
                //
                // CREATE
                .route(
                    "/api/v2/jobs/new",
                    routing::post(
                        |TypedHeader(AcceptedMimeTypes(mime)),
                         Extension::<Arc<SchedulerState>>(state),
                         RequestBodyTokioAsyncRead(mut body)| async move {
                            // First deserialize the job request and read all inputs into memory.
                            // The toolchain bincode is first, followed by the compressed inputs.
                            let (toolchain, inputs) = {
                                let bincode_len = body
                                    .read_u32()
                                    .await
                                    .map_err(|_| (StatusCode::BAD_REQUEST, "Invalid length"))
                                    .map_err(IntoResponse::into_response)?
                                    as u64;

                                let mut bincode_reader = body.take(bincode_len);
                                let mut bincode = vec![];
                                bincode_reader
                                    .read_to_end(&mut bincode)
                                    .await
                                    .map_err(|_| (StatusCode::BAD_REQUEST, "Invalid toolchain"))
                                    .map_err(IntoResponse::into_response)?;

                                let toolchain = bincode_deserialize(bincode.into())
                                    .await
                                    .map_err(|_| (StatusCode::BAD_REQUEST, "Invalid toolchain"))
                                    .map_err(IntoResponse::into_response)?;

                                let mut inputs_reader = bincode_reader.into_inner();
                                let mut inputs = vec![];
                                inputs_reader
                                    .read_to_end(&mut inputs)
                                    .await
                                    .map_err(|_| (StatusCode::BAD_REQUEST, "Invalid job inputs"))
                                    .map_err(IntoResponse::into_response)?;

                                (toolchain, inputs)
                            };

                            let service = state.service.clone();

                            // Once we've received the entire input, spawn a new task to write them
                            // to storage that won't be cancelled. This ensures we don't perform an
                            // incomplete write if the client disconnects prematurely, for example,
                            // if a user ctrl-C's their build.
                            tokio::spawn(async move {
                                service
                                    .new_job(toolchain, inputs.into())
                                    .and_then(|res| mime.serialize(res))
                                    .map_err(|e| AppError(e).into_response())
                                    .await
                            })
                            .await
                            .map_err(|_| AppError(anyhow!("")).into_response())?
                        },
                    ),
                )
                // PUT
                .route(
                    "/api/v2/job/{job_id}",
                    routing::put(
                        |TypedHeader(AcceptedMimeTypes(mime)),
                         Extension::<Arc<SchedulerState>>(state),
                         Path::<String>(job_id),
                         req: Request| async move {
                            let RequestBodyTokioAsyncRead(mut reader) = req
                                .extract::<RequestBodyTokioAsyncRead, _>()
                                .await
                                .map_err(|_| AppError(anyhow!("")))
                                .map_err(IntoResponse::into_response)?;

                            // First read all inputs into memory
                            let mut inputs = vec![];
                            reader
                                .read_to_end(&mut inputs)
                                .await
                                .map_err(|_| (StatusCode::BAD_REQUEST, "Invalid job inputs"))
                                .map_err(IntoResponse::into_response)?;

                            let service = state.service.clone();

                            // Once we've received the entire input, spawn a new task to write them
                            // to storage that won't be cancelled. This ensures we don't perform an
                            // incomplete write if the client disconnects prematurely, for example,
                            // if a user ctrl-C's their build.
                            tokio::spawn(async move {
                                service
                                    .put_job(&job_id, inputs.into())
                                    .and_then(|res| mime.serialize(res))
                                    .map_err(|e| AppError(e).into_response())
                                    .await
                            })
                            .await
                            .map_err(|_| AppError(anyhow!("")).into_response())?
                        },
                    ),
                )
                // POST
                .route(
                    "/api/v2/job/{job_id}",
                    routing::post(
                        |TypedHeader(AcceptedMimeTypes(mime)),
                         Extension::<Arc<SchedulerState>>(state),
                         Path::<String>(job_id),
                         RunJobBody(job): RunJobBody| async move {
                            state
                                .service
                                .run_job(&job_id, job)
                                .and_then(|res| mime.serialize(res))
                                .map_err(|err| AppError(err).into_response())
                                .await
                        },
                    ),
                )
                // DELETE
                .route(
                    "/api/v2/job/{job_id}",
                    routing::delete(
                        |TypedHeader(AcceptedMimeTypes(mime)),
                         Extension::<Arc<SchedulerState>>(state),
                         Path::<String>(job_id)| async move {
                            state
                                .service
                                .del_job(&job_id)
                                .and_then(|res| mime.serialize(res))
                                .map_err(AppError)
                                .await
                        },
                    ),
                )
                .fallback((StatusCode::NOT_FOUND, "404"))
        }

        fn tracing(app: Router) -> Router {
            // Mark these headers as sensitive so they don't show in logs
            let headers_to_redact: Arc<[_]> = Arc::new([
                http::header::AUTHORIZATION,
                http::header::PROXY_AUTHORIZATION,
                http::header::COOKIE,
                http::header::SET_COOKIE,
            ]);

            let request_id_header_name = http::HeaderName::from_str(
                std::env::var("SCCACHE_DIST_REQUEST_ID_HEADER_NAME")
                    .as_ref()
                    .map(|s| s.as_str())
                    // tower_http::request_id::X_REQUEST_ID inlined here because it is not public
                    .unwrap_or("x-request-id"),
            )
            .unwrap();

            app.layer(
                ServiceBuilder::new()
                    .layer(SetSensitiveRequestHeadersLayer::from_shared(Arc::clone(
                        &headers_to_redact,
                    )))
                    .layer(SetRequestIdLayer::new(
                        request_id_header_name,
                        MakeRequestUuid,
                    ))
                    .layer(
                        TraceLayer::new_for_http()
                            .make_span_with(DefaultMakeSpan::new().include_headers(true))
                            .on_response(DefaultOnResponse::new().include_headers(true)),
                    )
                    .layer(PropagateRequestIdLayer::x_request_id())
                    .layer(SetSensitiveResponseHeadersLayer::from_shared(
                        headers_to_redact,
                    )),
            )
        }

        pub fn serve(
            self,
            metrics: Metrics,
            bind_addr: SocketAddr,
            keepalive: DistNetworkingKeepalive,
            max_body_size: usize,
            max_concurrent_streams: impl Into<Option<u32>>,
        ) -> (
            axum_server::Handle<SocketAddr>,
            impl futures::Future<Output = Result<()>>,
        ) {
            let mut server = axum_server::bind(bind_addr);
            let mut builder = server.http_builder().http2();

            builder
                .adaptive_window(true)
                .max_concurrent_streams(max_concurrent_streams);

            if keepalive.enabled {
                builder
                    .timer(hyper_util::rt::TokioTimer::new())
                    .keep_alive_interval(Duration::from_secs(keepalive.interval))
                    .keep_alive_timeout(Duration::from_secs(keepalive.timeout));
            }

            // Create a handle for graceful shutdown
            let handle = axum_server::Handle::new();

            // Create the application router
            let app = Self::routes()
                // Authenticate the client before routing, but after capturing metrics and traces.
                .route_layer(axum::middleware::from_extractor::<RequireAuth>())
                // Limit request body size
                .layer(tower_http::limit::RequestBodyLimitLayer::new(max_body_size));

            // Capture metrics for all routes
            let app = Self::metrics(app, metrics);
            // Capture traces before metrics
            let app = Self::tracing(app);
            // Add SchedulerState layer last so it's available to all middleware
            let app = app.layer(Extension(self.state.clone()));
            // Make a service factory that captures client IP addresses
            let svc = app.into_make_service_with_connect_info::<SocketAddr>();
            // Serve the application service with the shutdown handle
            let svc = server.handle(handle.clone()).serve(svc).map_err(Into::into);

            (handle, svc)
        }
    }
}

#[cfg(feature = "dist-client")]
mod client {
    use async_trait::async_trait;

    use reqwest::Body;

    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio_util::compat::FuturesAsyncReadCompatExt;

    use std::path::{Path, PathBuf};
    use std::sync::Arc;
    use std::time::Duration;

    use super::super::cache;
    use super::common::{ReqwestRequestBuilderExt, bincode_req_fut};
    use super::{bincode_serialize, urls};
    use crate::{
        config,
        dist::{
            self, CompileCommand, NewJobResponse, RunJobRequestV2, RunJobResponse, SchedulerStatus,
            SubmitToolchainResult, Toolchain,
            pkg::{PackagedToolchain, ToolchainPackager},
        },
        errors::*,
        util::{AsyncMulticast, AsyncMulticastArgs, AsyncMulticastFunc, new_reqwest_client},
    };

    impl AsyncMulticastArgs for (Toolchain, Option<Arc<dyn PackagedToolchain + '_>>) {
        type Key = String;
        fn hash(&self) -> Self::Key {
            self.0.archive_id.clone()
        }
    }

    struct SubmitToolchainFn {
        client: Arc<ReqwestClients>,
        auth_token: String,
        scheduler_url: reqwest::Url,
        client_toolchains: Arc<cache::ClientToolchains>,
    }

    #[async_trait]
    impl AsyncMulticastFunc<(Toolchain, Option<Arc<dyn PackagedToolchain>>), SubmitToolchainResult>
        for SubmitToolchainFn
    {
        async fn call(
            &self,
            (tc, package): &(Toolchain, Option<Arc<dyn PackagedToolchain>>),
        ) -> Result<SubmitToolchainResult> {
            let id = &tc.archive_id;

            let Self {
                client,
                auth_token,
                scheduler_url,
                client_toolchains,
            } = self;

            let toolchain = if let Some(package) = package {
                client_toolchains.put_toolchain(tc, package.as_ref()).await
            } else {
                client_toolchains.get_toolchain(tc).await
            };

            let res = match toolchain {
                Err(err) => Err(err),
                Ok(None) => Err(anyhow!("Couldn't find toolchain locally")),
                Ok(Some(file)) => {
                    debug!("Uploading toolchain {id:?}");
                    let body = futures::io::AllowStdIo::new(file);
                    let body = tokio_util::io::ReaderStream::new(body.compat());
                    let req = client
                        .put(urls::scheduler_submit_toolchain(scheduler_url, id))
                        .header(http::header::ACCEPT, "application/octet-stream")
                        .bearer_auth(auth_token)
                        .body(Body::wrap_stream(body));
                    bincode_req_fut::<SubmitToolchainResult>(req).await
                }
            };

            match res {
                Ok(dist::SubmitToolchainResult::Success) => {
                    debug!("Successfully submitted toolchain {id:?}");
                    res
                }
                Ok(dist::SubmitToolchainResult::Error { ref message }) => {
                    warn!("Failed to submit toolchain {id:?}: {message}");
                    res
                }
                Err(err) => Err(err.context(format!("Could not submit toolchain {id:?}:"))),
            }
        }
    }

    struct ReqwestClients {
        clients: Vec<reqwest::Client>,
        index: std::sync::atomic::AtomicUsize,
    }

    impl ReqwestClients {
        fn new(net: &config::DistNetworking) -> Self {
            Self {
                clients: (0..net.max_connections.max(1))
                    .map(|_| new_reqwest_client(net))
                    .collect::<Vec<_>>(),
                index: Default::default(),
            }
        }

        fn next(&self) -> &reqwest::Client {
            use std::sync::atomic::Ordering::SeqCst;
            &self.clients[self
                .index
                .fetch_update(SeqCst, SeqCst, |i| Some((i + 1) % self.clients.len()))
                .unwrap_or(0)]
        }

        fn delete<U>(&self, url: U) -> reqwest::RequestBuilder
        where
            U: reqwest::IntoUrl,
        {
            self.next().delete(url)
        }

        fn get<U>(&self, url: U) -> reqwest::RequestBuilder
        where
            U: reqwest::IntoUrl,
        {
            self.next().get(url)
        }

        fn post<U>(&self, url: U) -> reqwest::RequestBuilder
        where
            U: reqwest::IntoUrl,
        {
            self.next().post(url)
        }

        fn put<U>(&self, url: U) -> reqwest::RequestBuilder
        where
            U: reqwest::IntoUrl,
        {
            self.next().put(url)
        }
    }

    pub struct Client {
        auth_token: String,
        client: Arc<ReqwestClients>,
        fallback_to_local_compile: bool,
        max_retries: f64,
        request_timeout: u32,
        rewrite_includes_only: bool,
        scheduler_url: reqwest::Url,
        submit_toolchain_reqs:
            AsyncMulticast<(Toolchain, Option<Arc<dyn PackagedToolchain>>), SubmitToolchainResult>,
        tc_cache: Arc<cache::ClientToolchains>,
    }

    impl Client {
        #[allow(clippy::too_many_arguments)]
        pub async fn new(
            scheduler_url: reqwest::Url,
            cache_dir: &Path,
            cache_size: u64,
            toolchain_configs: &[config::DistToolchainConfig],
            auth_token: String,
            fallback_to_local_compile: bool,
            max_retries: f64,
            rewrite_includes_only: bool,
            net: &config::DistNetworking,
        ) -> Result<Self> {
            let request_timeout = net.request_timeout;
            let client = Arc::new(ReqwestClients::new(net));
            let client_toolchains = Arc::new(
                cache::ClientToolchains::new(cache_dir, cache_size, toolchain_configs)
                    .context("failed to initialise client toolchains")?,
            );

            let submit_toolchain_reqs = AsyncMulticast::new(SubmitToolchainFn {
                client: client.clone(),
                auth_token: auth_token.clone(),
                scheduler_url: scheduler_url.clone(),
                client_toolchains: client_toolchains.clone(),
            });

            Ok(Self {
                auth_token: auth_token.clone(),
                client,
                fallback_to_local_compile,
                max_retries,
                request_timeout,
                rewrite_includes_only,
                scheduler_url: scheduler_url.clone(),
                submit_toolchain_reqs,
                tc_cache: client_toolchains,
            })
        }
    }

    #[async_trait]
    impl dist::Client for Client {
        async fn new_job(
            &self,
            toolchain: Toolchain,
            inputs: std::fs::File,
        ) -> Result<NewJobResponse> {
            bincode_req_fut(
                self.client
                    .post(urls::scheduler_new_job(&self.scheduler_url))
                    .header(http::header::ACCEPT, "application/octet-stream")
                    .bearer_auth(self.auth_token.clone())
                    .body(Body::wrap_stream({
                        let bincode = bincode_serialize(toolchain).await?;
                        let (body, mut writer) = tokio::io::simplex(bincode.len() + 4);
                        writer
                            .write_u32(bincode.len() as u32)
                            .await
                            .expect("Infallible write of bincode length to stream failed");
                        writer
                            .write_all(&bincode)
                            .await
                            .expect("Infallible write of bincode body to stream failed");
                        writer.shutdown().await?;
                        let inputs = tokio::fs::File::from_std(inputs);
                        let inputs = tokio::io::BufReader::new(inputs);
                        tokio_util::io::ReaderStream::new(body.chain(inputs))
                    })),
            )
            .await
        }

        async fn put_job(&self, job_id: &str, inputs: std::fs::File) -> Result<()> {
            let body = futures::io::AllowStdIo::new(inputs);
            let body = tokio_util::io::ReaderStream::new(body.compat());
            bincode_req_fut(
                self.client
                    .put(urls::scheduler_put_job(&self.scheduler_url, job_id))
                    .header(http::header::ACCEPT, "application/octet-stream")
                    .bearer_auth(self.auth_token.clone())
                    .body(Body::wrap_stream(body)),
            )
            .await
        }

        async fn run_job(
            &self,
            job_id: &str,
            timeout: Duration,
            toolchain: Toolchain,
            command: CompileCommand,
            outputs: Vec<String>,
        ) -> Result<RunJobResponse> {
            bincode_req_fut(
                self.client
                    .post(urls::scheduler_run_job(&self.scheduler_url, job_id))
                    .header(http::header::ACCEPT, "application/octet-stream")
                    .bearer_auth(self.auth_token.clone())
                    .timeout(timeout)
                    .bincode(&RunJobRequestV2 {
                        command,
                        outputs,
                        toolchain,
                        // TODO:
                        // Decide if there's any interesting client details worth
                        // sending to the server to include in metrics dimensions
                        labels: None,
                    })?,
            )
            .await
        }

        async fn del_job(&self, job_id: &str) -> Result<()> {
            bincode_req_fut(
                self.client
                    .delete(urls::scheduler_del_job(&self.scheduler_url, job_id))
                    .header(http::header::ACCEPT, "application/octet-stream")
                    .bearer_auth(self.auth_token.clone()),
            )
            .await
        }

        async fn get_status(&self) -> Result<SchedulerStatus> {
            bincode_req_fut(
                self.client
                    .get(urls::scheduler_status(&self.scheduler_url))
                    .header(http::header::ACCEPT, "application/octet-stream")
                    .bearer_auth(self.auth_token.clone()),
            )
            .await
        }

        async fn put_toolchain(
            &self,
            toolchain: Toolchain,
            packaged: Option<Arc<dyn PackagedToolchain>>,
        ) -> Result<SubmitToolchainResult> {
            self.submit_toolchain_reqs
                .call((toolchain, packaged))
                .await
                .map(|(_, res)| res)
        }

        async fn hash_toolchain(
            &self,
            compiler_path: &Path,
            weak_toolchain_key: &str,
            toolchain_packager: &dyn ToolchainPackager,
        ) -> Result<(
            Toolchain,
            Option<(String, PathBuf)>,
            Option<Arc<dyn PackagedToolchain>>,
        )> {
            self.tc_cache
                .hash_toolchain(compiler_path, weak_toolchain_key, toolchain_packager)
                .await
        }

        fn fallback_to_local_compile(&self) -> bool {
            self.fallback_to_local_compile
        }

        fn max_retries(&self) -> f64 {
            self.max_retries
        }

        fn request_timeout(&self) -> u32 {
            self.request_timeout
        }

        fn rewrite_includes_only(&self) -> bool {
            self.rewrite_includes_only
        }

        async fn get_custom_toolchain(&self, exe: &Path) -> Option<PathBuf> {
            match self.tc_cache.get_custom_toolchain(exe).await {
                Some(Ok((_, _, path))) => Some(path),
                _ => None,
            }
        }
    }
}

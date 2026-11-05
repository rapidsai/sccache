// Copyright 2016 Mozilla Foundation
// Copyright 2025 Mozilla Foundation
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

//! The server process: starting it, the stats it reports, its logs, and the
//! errors it reports for bad cache storage configs.

use crate::harness::client::SccacheClient;
#[cfg(not(target_os = "macos"))]
use crate::harness::sccache_command;
use anyhow::{Context, Result};
use assert_cmd::prelude::*;
#[cfg(any(unix, feature = "gcs", feature = "s3"))]
use predicates::prelude::*;
use regex::Regex;
#[cfg(not(target_os = "macos"))]
use std::net::TcpListener;
use std::path::Path;
use tempfile::TempDir;

#[test]
// test fails intermittently on macos:
// https://github.com/mozilla/sccache/issues/234
#[cfg(not(target_os = "macos"))]
fn test_server_port_in_use() {
    // Bind an arbitrary free port.
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let config = tempfile::NamedTempFile::new()
        .map(|p| p.into_temp_path())
        .unwrap();
    let output = sccache_command()
        .arg("--start-server")
        .env("SCCACHE_CONF", &config)
        .env(
            "SCCACHE_SERVER_PORT",
            listener.local_addr().unwrap().port().to_string(),
        )
        .output()
        .unwrap();
    assert!(!output.status.success());
    let s = String::from_utf8_lossy(&output.stderr);
    const MSG: &str = "Server startup failed:";
    assert!(
        s.contains(MSG),
        "Output did not contain '{MSG}':\n========\n{s}\n========"
    );
}

#[test]
fn test_stats_no_server() {
    let config = tempfile::NamedTempFile::new()
        .map(|p| p.into_temp_path())
        .unwrap();
    let client = SccacheClient::new(&config, &config);
    let _ = client.stats();
    assert!(
        !client.stop(),
        "Server shouldn't be running after --show-stats"
    );
}

#[test]
fn test_log_timestamp_format_without_millis() {
    // Test that without SCCACHE_LOG_MILLIS, timestamps have second precision
    let tempdir = TempDir::new().unwrap();

    let mut cmd = SccacheClient::new_no_cfg().cmd();
    cmd.arg("--show-stats")
        .env("SCCACHE_LOG", "debug")
        .env("SCCACHE_DIR", tempdir.path())
        .env_remove("SCCACHE_LOG_MILLIS");

    let output = cmd.output().unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr);

    // Pattern for timestamp without milliseconds: [2025-10-09T14:42:35Z DEBUG ...]
    let timestamp_regex = Regex::new(r"\[\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z\s+\w+\s+").unwrap();

    // Check that we have at least one log line with the expected format
    assert!(
        timestamp_regex.is_match(&stderr),
        "Expected timestamp format without milliseconds not found in stderr:\n{stderr}"
    );

    // Ensure NO milliseconds are present (no decimal point after seconds)
    let millis_regex = Regex::new(r"\[\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z").unwrap();
    assert!(
        !millis_regex.is_match(&stderr),
        "Unexpected milliseconds found in timestamp when SCCACHE_LOG_MILLIS not set:\n{stderr}"
    );
}

#[test]
fn test_log_timestamp_format_with_millis() {
    // Test that with SCCACHE_LOG_MILLIS, timestamps have millisecond precision
    let tempdir = TempDir::new().unwrap();

    let mut cmd = SccacheClient::new_no_cfg().cmd();
    cmd.arg("--show-stats")
        .env("SCCACHE_LOG", "debug")
        .env("SCCACHE_LOG_MILLIS", "1")
        .env("SCCACHE_DIR", tempdir.path());

    let output = cmd.output().unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr);

    // Pattern for timestamp with milliseconds: [2025-10-09T14:44:56.628Z DEBUG ...]
    let millis_regex =
        Regex::new(r"\[\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z\s+\w+\s+").unwrap();

    // Check that we have at least one log line with millisecond precision
    assert!(
        millis_regex.is_match(&stderr),
        "Expected timestamp format with milliseconds not found in stderr:\n{stderr}"
    );
}

#[test]
fn test_log_millis_flag_with_various_values() {
    // Test that SCCACHE_LOG_MILLIS accepts any value (not just "1")
    let tempdir = TempDir::new().unwrap();

    // Pattern for timestamp with milliseconds
    let millis_regex = Regex::new(r"\[\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z").unwrap();

    for value in &["1", "true", "yes", "anything"] {
        let mut cmd = SccacheClient::new_no_cfg().cmd();
        cmd.arg("--show-stats")
            .env("SCCACHE_LOG", "debug")
            .env("SCCACHE_LOG_MILLIS", value)
            .env("SCCACHE_DIR", tempdir.path());

        let output = cmd.output().unwrap();
        let stderr = String::from_utf8_lossy(&output.stderr);

        assert!(
            millis_regex.is_match(&stderr),
            "Expected timestamp with milliseconds for SCCACHE_LOG_MILLIS={value}, but not found in:\n{stderr}"
        );
    }
}

#[test]
#[cfg(unix)]
fn test_run_log_no_perm() -> Result<()> {
    trace!("sccache with log");
    let client = SccacheClient::new_no_cfg();
    let mut cmd = client.cmd();
    cmd.arg("gcc")
        .env("SCCACHE_ERROR_LOG", "/no-perm.log") // Should not work
        .env("SCCACHE_LOG", "debug");

    cmd.assert().failure().stderr(predicate::str::contains(
        "Cannot open/write log file '/no-perm.log'",
    ));
    Ok(())
}

#[test]
fn test_run_log() -> Result<()> {
    trace!("sccache with log");
    let client = SccacheClient::new_no_cfg();

    let tempdir = tempfile::Builder::new()
        .prefix("sccache_test_rust_cargo")
        .tempdir()
        .context("Failed to create tempdir")?;
    let tmppath = tempdir.path().join("perm.log");
    let config = tempfile::NamedTempFile::new_in(tempdir.path()).map(|p| p.into_temp_path())?;
    let mut cmd = client.cmd();
    cmd.arg("--start-server")
        .env("SCCACHE_CONF", &config)
        .env("SCCACHE_ERROR_LOG", &tmppath) // Should not work
        .env("SCCACHE_LOG", "debug");

    cmd.assert().success();
    client.stop();
    assert!(Path::new(&tmppath).is_file());
    Ok(())
}

#[test]
#[cfg(feature = "gcs")]
fn test_gcp_arg_check() -> Result<()> {
    trace!("sccache with log");
    let client = SccacheClient::new_no_cfg();

    let config = tempfile::NamedTempFile::new().map(|p| p.into_temp_path())?;

    let mut cmd = client.cmd();
    cmd.arg("--start-server")
        .env("SCCACHE_CONF", &config)
        .env("SCCACHE_LOG", "sccache=debug")
        .env("SCCACHE_GCS_KEY_PATH", "foo.json");

    cmd.assert().success().stderr(predicate::str::contains(
        "If setting GCS credentials, `bucket` and an auth mechanism need to be set.",
    ));

    client.stop();

    let mut cmd = client.cmd();
    cmd.arg("--start-server")
        .env("SCCACHE_CONF", &config)
        .env("SCCACHE_LOG", "sccache=debug")
        .env("SCCACHE_GCS_OAUTH_URL", "http://127.0.0.1");

    cmd.assert()
        .failure()
        .stderr(predicate::str::contains("`oauth_url` has been deprecated"));

    client.stop();
    let mut cmd = client.cmd();
    cmd.arg("--start-server")
        .env("SCCACHE_CONF", &config)
        .env("SCCACHE_LOG", "sccache=debug")
        .env("SCCACHE_GCS_BUCKET", "b")
        .env("SCCACHE_GCS_CREDENTIALS_URL", "not_valid_url//127.0.0.1")
        .env("SCCACHE_GCS_KEY_PATH", "foo.json");

    // This is just a warning
    cmd.assert()
        .failure()
        .stderr(predicate::str::contains("gcs credential url is invalid"));

    Ok(())
}

#[test]
#[cfg(feature = "s3")]
fn test_s3_invalid_args() -> Result<()> {
    let client = SccacheClient::new_no_cfg();

    let config = tempfile::NamedTempFile::new().map(|p| p.into_temp_path())?;

    let mut cmd = client.cmd();
    cmd.arg("--start-server")
        .env("SCCACHE_CONF", &config)
        .env("SCCACHE_LOG", "sccache=debug")
        .env("SCCACHE_BUCKET", "test")
        .env("SCCACHE_REGION", "us-east-1")
        .env("AWS_ACCESS_KEY_ID", "invalid_ak")
        .env("AWS_SECRET_ACCESS_KEY", "invalid_sk");

    cmd.assert()
        .failure()
        .stderr(predicate::str::contains("cache storage failed to read"));

    Ok(())
}

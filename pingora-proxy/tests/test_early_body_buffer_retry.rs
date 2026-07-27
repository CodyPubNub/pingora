// Copyright 2026 Cloudflare, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Regression coverage for early request body buffering across an upstream retry.
//!
//! `buffer_request_body_early()` drains the downstream body before
//! `enable_retry_buffering()`, so the retry buffer never sees those bytes and the proxy must
//! replay the buffered copy itself. These tests cover the h1 upstream path against a
//! self-contained origin; the selection logic is shared with h2 and unit-tested in
//! `proxy_common`.

#![cfg(feature = "early_body_buffer")]

use async_trait::async_trait;
use pingora_core::prelude::HttpPeer;
use pingora_core::server::configuration::ServerConf;

use pingora_error::Result;
use pingora_proxy::{ProxyHttp, Session};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

/// Per-harness rather than global so these tests stay independent under parallel runs.
#[derive(Default)]
struct Recorder {
    /// (head, decoded body) per request, in arrival order.
    requests: Mutex<Vec<(String, String)>>,
}

impl Recorder {
    fn bodies(&self) -> Vec<String> {
        self.requests
            .lock()
            .unwrap()
            .iter()
            .map(|(_, b)| b.clone())
            .collect()
    }

    fn heads(&self) -> Vec<String> {
        self.requests
            .lock()
            .unwrap()
            .iter()
            .map(|(h, _)| h.clone())
            .collect()
    }
}

/// The bug sends `Content-Length: N` with no body, so waiting forever would hang instead of
/// reporting a short read.
const BODY_READ_TIMEOUT: Duration = Duration::from_millis(600);

async fn read_one_request(stream: &mut TcpStream, recorder: &Recorder) -> Option<()> {
    let mut buf = Vec::new();
    let mut chunk = [0u8; 4096];

    let head_end = loop {
        if let Some(pos) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            break pos + 4;
        }
        let n = stream.read(&mut chunk).await.ok()?;
        if n == 0 {
            return None;
        }
        buf.extend_from_slice(&chunk[..n]);
    };

    let head = String::from_utf8_lossy(&buf[..head_end]).to_string();
    let head_lower = head.to_ascii_lowercase();

    let content_length = head_lower
        .lines()
        .find_map(|l| l.strip_prefix("content-length:"))
        .and_then(|v| v.trim().parse::<usize>().ok());
    let chunked = head_lower
        .lines()
        .any(|l| l.starts_with("transfer-encoding:") && l.contains("chunked"));

    let mut body_raw = buf[head_end..].to_vec();

    let body = if let Some(len) = content_length {
        while body_raw.len() < len {
            match tokio::time::timeout(BODY_READ_TIMEOUT, stream.read(&mut chunk)).await {
                Ok(Ok(0)) | Err(_) => break,
                Ok(Ok(n)) => body_raw.extend_from_slice(&chunk[..n]),
                Ok(Err(_)) => return None,
            }
        }
        String::from_utf8_lossy(&body_raw[..len.min(body_raw.len())]).to_string()
    } else if chunked {
        while !body_raw.windows(5).any(|w| w == b"0\r\n\r\n") {
            match tokio::time::timeout(BODY_READ_TIMEOUT, stream.read(&mut chunk)).await {
                Ok(Ok(0)) | Err(_) => break,
                Ok(Ok(n)) => body_raw.extend_from_slice(&chunk[..n]),
                Ok(Err(_)) => return None,
            }
        }
        dechunk(&body_raw)
    } else {
        String::new()
    };

    recorder.requests.lock().unwrap().push((head, body));
    Some(())
}

fn dechunk(raw: &[u8]) -> String {
    let mut out = Vec::new();
    let mut rest = raw;
    while let Some(pos) = rest.windows(2).position(|w| w == b"\r\n") {
        let size = usize::from_str_radix(
            String::from_utf8_lossy(&rest[..pos])
                .trim()
                .split(';')
                .next()
                .unwrap_or("0"),
            16,
        )
        .unwrap_or(0);
        if size == 0 {
            break;
        }
        let data_start = pos + 2;
        let data_end = (data_start + size).min(rest.len());
        out.extend_from_slice(&rest[data_start..data_end]);
        rest = &rest[(data_end + 2).min(rest.len())..];
    }
    String::from_utf8_lossy(&out).to_string()
}

/// With `fail_first`, the first connection answers request #1 so the proxy pools it, then
/// abandons request #2. That shape is required: the failure must land after connect (a connect
/// failure never reaches the code that forwards the buffered body) and on a reused connection
/// (`error_while_proxy` only retries when `decide_reuse(client_reused)` holds). The connection
/// must also stay open while idle, or the pool discards it on the FIN instead of retrying.
async fn spawn_origin(recorder: Arc<Recorder>, fail_first: bool) -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let first_conn = Arc::new(AtomicBool::new(true));

    tokio::spawn(async move {
        loop {
            let Ok((mut stream, _)) = listener.accept().await else {
                return;
            };
            let recorder = recorder.clone();
            let may_poison = fail_first && first_conn.swap(false, Ordering::SeqCst);
            tokio::spawn(async move {
                let mut seen = 0usize;
                loop {
                    if read_one_request(&mut stream, &recorder).await.is_none() {
                        return;
                    }
                    seen += 1;
                    if may_poison && seen == 2 {
                        let _ = stream.shutdown().await;
                        return;
                    }
                    let _ = stream
                        .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 3\r\n\r\nok\n")
                        .await;
                    let _ = stream.flush().await;
                }
            });
        }
    });

    port
}

struct EarlyBufferProxy {
    origin_port: u16,
    limit: usize,
    max_attempts: Arc<AtomicUsize>,
}

/// Attempts are counted per request, not globally, so a retry in one request cannot mask the
/// absence of a retry in the next.
#[derive(Default)]
struct Ctx {
    attempt: usize,
}

#[async_trait]
impl ProxyHttp for EarlyBufferProxy {
    type CTX = Ctx;
    fn new_ctx(&self) -> Ctx {
        Ctx::default()
    }

    fn early_request_body_buffer_limit(&self, _session: &Session, _ctx: &Ctx) -> Option<usize> {
        Some(self.limit)
    }

    async fn upstream_peer(&self, _session: &mut Session, ctx: &mut Ctx) -> Result<Box<HttpPeer>> {
        ctx.attempt += 1;
        self.max_attempts.fetch_max(ctx.attempt, Ordering::SeqCst);

        let mut peer = Box::new(HttpPeer::new(
            format!("127.0.0.1:{}", self.origin_port),
            false,
            String::new(),
        ));
        // Pool connections so the proxy reuses one instead of dialing fresh every attempt.
        peer.options.idle_timeout = Some(Duration::from_secs(60));
        Ok(peer)
    }
}

struct Harness {
    proxy_port: u16,
    max_attempts: Arc<AtomicUsize>,
    recorder: Arc<Recorder>,
}

async fn start_harness(limit: usize, fail_first: bool) -> Harness {
    let recorder = Arc::new(Recorder::default());
    let origin_port = spawn_origin(recorder.clone(), fail_first).await;
    let max_attempts = Arc::new(AtomicUsize::new(0));

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let proxy_port = listener.local_addr().unwrap().port();
    drop(listener);

    let app_attempts = max_attempts.clone();
    std::thread::spawn(move || {
        let mut server = pingora_core::server::Server::new(None).unwrap();
        server.bootstrap();
        let conf = Arc::new(ServerConf::default());
        let mut service = pingora_proxy::http_proxy_service(
            &conf,
            EarlyBufferProxy {
                origin_port,
                limit,
                max_attempts: app_attempts,
            },
        );
        service.add_tcp(&format!("127.0.0.1:{proxy_port}"));
        server.add_service(service);
        server.run_forever();
    });

    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    let mut listening = false;
    while std::time::Instant::now() < deadline {
        if TcpStream::connect(("127.0.0.1", proxy_port)).await.is_ok() {
            listening = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert!(
        listening,
        "proxy did not start listening on 127.0.0.1:{proxy_port} within 10s"
    );

    Harness {
        proxy_port,
        max_attempts,
        recorder,
    }
}

/// These tests break upstream connections on purpose; bound the client so a regression into
/// never responding fails instead of hanging CI.
const CLIENT_TIMEOUT: Duration = Duration::from_secs(15);

async fn post(port: u16, body: &str) -> String {
    let client = reqwest::Client::builder()
        .pool_max_idle_per_host(0)
        .timeout(CLIENT_TIMEOUT)
        .build()
        .unwrap();
    let res = client
        .post(format!("http://127.0.0.1:{port}/echo"))
        .body(body.to_string())
        .send()
        .await
        .expect("proxy did not answer the request");
    format!("{}", res.status().as_u16())
}

/// Hand-rolled because reqwest needs its `stream` feature to send a chunked request.
async fn post_chunked(port: u16, body: &str) -> String {
    tokio::time::timeout(CLIENT_TIMEOUT, async {
        let mut stream = TcpStream::connect(("127.0.0.1", port)).await.unwrap();
        let req = format!(
            "POST /echo HTTP/1.1\r\nHost: 127.0.0.1\r\nTransfer-Encoding: chunked\r\n\r\n{:x}\r\n{}\r\n0\r\n\r\n",
            body.len(),
            body
        );
        stream.write_all(req.as_bytes()).await.unwrap();
        stream.flush().await.unwrap();

        let mut resp = Vec::new();
        let mut chunk = [0u8; 4096];
        while !resp.windows(2).any(|w| w == b"\r\n") {
            let n = stream.read(&mut chunk).await.unwrap();
            if n == 0 {
                break;
            }
            resp.extend_from_slice(&chunk[..n]);
        }
        String::from_utf8_lossy(&resp)
            .split_whitespace()
            .nth(1)
            .unwrap_or("000")
            .to_string()
    })
    .await
    .expect("proxy did not answer the chunked request")
}

/// Asserts every recorded request carried `body` under framing that matches it. The
/// Content-Length check is not redundant: the origin stops at a short read, so an inflated
/// Content-Length still records the full payload and would pass the body check alone.
fn assert_all_requests_carried(recorder: &Recorder, body: &str) {
    let bodies = recorder.bodies();
    assert!(
        bodies.len() >= 3,
        "expected priming + failed attempt + retry, saw {bodies:?}"
    );
    for (i, seen) in bodies.iter().enumerate() {
        assert_eq!(
            seen, body,
            "origin request #{i} lost the early-buffered body; origin saw {bodies:?}"
        );
    }
    for (i, head) in recorder.heads().iter().enumerate() {
        if let Some(cl) = head
            .to_ascii_lowercase()
            .lines()
            .find_map(|l| l.strip_prefix("content-length:"))
            .and_then(|v| v.trim().parse::<usize>().ok())
        {
            assert_eq!(
                cl,
                body.len(),
                "origin request #{i} Content-Length disagrees with the body sent; head:\n{head}"
            );
        }
    }
}

/// The first POST pools an upstream connection; the second reuses it, the origin abandons it
/// mid-exchange, and the retry that follows must still carry the buffered body.
#[tokio::test]
async fn early_buffered_body_survives_upstream_retry() {
    let harness = start_harness(64 * 1024, true).await;
    let body = "cody-early-buffer-payload";

    assert_eq!(
        post(harness.proxy_port, body).await,
        "200",
        "priming request should succeed"
    );
    let attempts_after_priming = harness.max_attempts.load(Ordering::SeqCst);

    let status = post(harness.proxy_port, body).await;

    let attempts = harness.max_attempts.load(Ordering::SeqCst);
    assert!(
        attempts > attempts_after_priming,
        "test setup failed to force a retry: max attempts per request stayed at {attempts}"
    );
    assert_eq!(status, "200", "retried request should still succeed");

    assert_all_requests_carried(&harness.recorder, body);
}

/// Same, for a chunked body — the framing the early buffer has to reconstruct itself.
#[tokio::test]
async fn early_buffered_chunked_body_survives_upstream_retry() {
    let harness = start_harness(64 * 1024, true).await;
    let body = "chunked-payload-across-retry";

    assert_eq!(
        post_chunked(harness.proxy_port, body).await,
        "200",
        "priming request should succeed"
    );
    let attempts_after_priming = harness.max_attempts.load(Ordering::SeqCst);

    assert_eq!(
        post_chunked(harness.proxy_port, body).await,
        "200",
        "retried request should still succeed"
    );

    let attempts = harness.max_attempts.load(Ordering::SeqCst);
    assert!(
        attempts > attempts_after_priming,
        "test setup failed to force a retry: max attempts per request stayed at {attempts}"
    );

    assert_all_requests_carried(&harness.recorder, body);
}

/// Guards against "fixing" the retry by dropping the size limit.
#[tokio::test]
async fn oversized_body_is_rejected_not_silently_emptied() {
    let harness = start_harness(8, false).await;
    let status = post(
        harness.proxy_port,
        "this body is definitely longer than 8 bytes",
    )
    .await;
    assert_eq!(
        status, "413",
        "body over the early-buffer limit should be rejected with 413"
    );
    assert!(
        harness.recorder.bodies().is_empty(),
        "rejected request must never reach the origin, saw {:?}",
        harness.recorder.bodies()
    );
}

/// Baseline: replaying on retry must not double-send when there is no retry.
#[tokio::test]
async fn buffered_body_forwarded_once_without_retry() {
    let harness = start_harness(64 * 1024, false).await;
    let body = "single-attempt-body";

    assert_eq!(post(harness.proxy_port, body).await, "200");
    assert_eq!(
        harness.max_attempts.load(Ordering::SeqCst),
        1,
        "expected no retry"
    );

    let bodies = harness.recorder.bodies();
    assert_eq!(bodies, vec![body.to_string()], "body should be sent once");
}

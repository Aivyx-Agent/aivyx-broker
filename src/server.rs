use std::sync::Arc;
use std::time::Duration;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::Value;

use crate::Scheduler;

#[derive(Clone)]
pub struct AppState {
    pub scheduler: Scheduler,
    pub http: reqwest::Client,
    pub llama_server_url: String,
    pub kv_store: Arc<aivyx_kvcache::LlamaServerSlotStore>,
    pub queue_timeout: Duration,
}

pub fn build_router(state: AppState) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/status", get(status))
        .route("/slots", get(slots))
        .with_state(state)
}

async fn status(State(state): State<AppState>) -> impl IntoResponse {
    Json(serde_json::json!({ "llama_server_url": state.llama_server_url, "queue_depth": 0 }))
}

async fn slots(State(state): State<AppState>) -> impl IntoResponse {
    Json(serde_json::json!(state.scheduler.snapshot()))
}

async fn chat_completions(
    State(state): State<AppState>,
    Json(mut body): Json<Value>,
) -> axum::response::Response {
    let hint = body
        .as_object_mut()
        .and_then(|o| o.remove("aivyx_slot_hint"))
        .and_then(|v| serde_json::from_value::<crate::SlotHint>(v).ok());

    let admission = match state.scheduler.admit(hint.clone(), state.queue_timeout).await {
        Ok(a) => a,
        Err(crate::SchedulerError::Timeout) => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({"error": "timed out waiting for a free slot"})),
            )
                .into_response();
        }
    };

    // Owns a `Scheduler` clone (cheap -- an `Arc` internally) rather than
    // borrowing `&state`, specifically so it can be moved into the
    // streamed response body below and outlive this `async fn`'s own
    // stack frame, which returns as soon as the `Body` is *constructed*,
    // well before it's actually drained to the client.
    let guard = ReleaseGuard { scheduler: state.scheduler.clone(), slot_id: admission.slot_id };

    let result = handle_admitted(&state, &mut body, &hint, admission, guard).await;

    match result {
        Ok(response) => response,
        Err(err) => {
            tracing::warn!(error = %err, "chat_completions forwarding failed");
            (
                StatusCode::BAD_GATEWAY,
                Json(serde_json::json!({"error": err.to_string()})),
            )
                .into_response()
        }
    }
}

/// Releases `slot_id` back to the scheduler on drop.
///
/// Deliberately *not* released via an explicit `scheduler.release(..)`
/// call right after `handle_admitted` returns: that function returns as
/// soon as it has *constructed* the streaming response body, not once the
/// body has actually been drained to the client -- llama-server is still
/// generating on the slot for the entire streaming window after that
/// point. Releasing early would let a second request get admitted onto
/// the same physical slot while the first is still mid-generation,
/// defeating the entire point of admission control. Instead this guard is
/// moved into the response stream itself (see `release_on_stream_end`) so
/// it drops -- and releases -- only once the stream is fully drained or
/// the client disconnects mid-stream and the stream is dropped early. For
/// the non-streaming error paths (admission-adjacent HTTP calls failing
/// before any stream exists), it stays owned by `handle_admitted`'s local
/// scope and simply drops -- and releases -- when that function returns.
struct ReleaseGuard {
    scheduler: Scheduler,
    slot_id: u32,
}

impl Drop for ReleaseGuard {
    fn drop(&mut self) {
        self.scheduler.release(self.slot_id);
    }
}

/// Wraps `inner` so that `guard` is dropped -- releasing the slot --
/// exactly once, when the stream is exhausted (`None` observed) or when
/// the returned stream itself is dropped before exhaustion (client
/// disconnect). Preserves `inner`'s item type unchanged so it satisfies
/// whatever bound `axum::body::Body::from_stream` needs, identically to
/// passing `inner` directly.
fn release_on_stream_end<S>(inner: S, guard: ReleaseGuard) -> impl futures::Stream<Item = S::Item> + Send + 'static
where
    S: futures::Stream + Send + 'static,
{
    futures::stream::unfold((Box::pin(inner), Some(guard)), |(mut stream, mut guard)| async move {
        use futures::StreamExt;
        match stream.next().await {
            Some(item) => Some((item, (stream, guard))),
            None => {
                // Stream exhausted: drop the guard now (releasing the
                // slot) instead of waiting for the unfold combinator
                // itself to be dropped later.
                guard.take();
                None
            }
        }
    })
}

async fn handle_admitted(
    state: &AppState,
    body: &mut Value,
    hint: &Option<crate::SlotHint>,
    admission: crate::Admission,
    guard: ReleaseGuard,
) -> Result<axum::response::Response, anyhow::Error> {
    if !admission.cache_ready
        && let Some(h) = hint
    {
        warm_or_restore(state, body, h, admission.slot_id).await?;
    }

    if let Some(obj) = body.as_object_mut() {
        obj.insert("id_slot".to_string(), serde_json::json!(admission.slot_id));
    }

    let resp = state
        .http
        .post(format!("{}/v1/chat/completions", state.llama_server_url.trim_end_matches('/')))
        .json(body)
        .send()
        .await?;

    // Relay llama-server's response untouched: real status code, real
    // headers, real body -- whether it succeeded or not. Deliberately does
    // *not* call `.error_for_status()`, which would discard the upstream
    // status/body and collapse every non-2xx into a generic broker error.
    let status = resp.status();
    let headers = resp.headers().clone();

    if !status.is_success() {
        // Non-streaming error path: read the whole body so it can be
        // forwarded byte-for-byte, then release the slot (via `guard`
        // dropping at the end of this function) instead of holding it for
        // a stream that was never started.
        let bytes = resp.bytes().await?;
        let mut builder = axum::response::Response::builder().status(status);
        for (name, value) in headers.iter() {
            builder = builder.header(name, value);
        }
        return Ok(builder.body(axum::body::Body::from(bytes))?);
    }

    let mut builder = axum::response::Response::builder().status(status);
    for (name, value) in headers.iter() {
        builder = builder.header(name, value);
    }
    let axum_body = axum::body::Body::from_stream(release_on_stream_end(resp.bytes_stream(), guard));
    Ok(builder.body(axum_body)?)
}

async fn warm_or_restore(
    state: &AppState,
    body: &Value,
    hint: &crate::SlotHint,
    slot_id: u32,
) -> Result<(), anyhow::Error> {
    let key = aivyx_kvcache::CacheKey {
        backend_id: "aivyx-broker".to_string(),
        model_id: "aivyx-broker".to_string(),
        build_hash: "aivyx-broker".to_string(),
        prefix_hash: hint.prefix_hash.clone(),
    };

    let restored = state.kv_store.restore_into_slot(&key, slot_id).await.unwrap_or_else(|err| {
        tracing::warn!(error = %err, "kvcache: restore_into_slot failed");
        false
    });

    if !restored {
        let system_message = body
            .get("messages")
            .and_then(|m| m.as_array())
            .and_then(|arr| arr.iter().find(|m| m.get("role").and_then(|r| r.as_str()) == Some("system")))
            .cloned()
            .unwrap_or_else(|| serde_json::json!({"role": "system", "content": ""}));
        let tools = body.get("tools").cloned();

        let mut warm_up = serde_json::json!({
            "messages": [system_message, {"role": "user", "content": ""}],
            "max_tokens": 1,
            "id_slot": slot_id,
        });
        // The stable "prefix" that gets cached under `prefix_hash` is
        // system-prompt-plus-tools together (see design spec). Omitting
        // `tools` here would warm/save a prefix that doesn't match what a
        // real request for the same `prefix_hash` actually sends,
        // defeating the cache.
        if let Some(tools) = tools {
            warm_up["tools"] = tools;
        }
        state
            .http
            .post(format!("{}/v1/chat/completions", state.llama_server_url.trim_end_matches('/')))
            .json(&warm_up)
            .send()
            .await?
            .error_for_status()?;

        if let Err(err) = state
            .kv_store
            .save_from_slot(&key, slot_id, aivyx_kvcache::CacheMeta { size_bytes: 1, token_count: 1 })
            .await
        {
            tracing::warn!(error = %err, "kvcache: save_from_slot failed");
        }
    }

    state.scheduler.mark_resident(slot_id, hint.prefix_hash.clone());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use tower::ServiceExt;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    async fn test_state(llama_server_url: String) -> (AppState, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let kv_store = aivyx_kvcache::LlamaServerSlotStore::open(
            dir.path(),
            llama_server_url.clone(),
            10 * 1024 * 1024,
        )
        .unwrap();
        let state = AppState {
            scheduler: Scheduler::new(2),
            http: reqwest::Client::new(),
            llama_server_url,
            kv_store: Arc::new(kv_store),
            queue_timeout: Duration::from_secs(5),
        };
        (state, dir)
    }

    #[tokio::test]
    async fn status_reports_llama_server_url() {
        let (state, _dir) = test_state("http://127.0.0.1:1".to_string()).await;
        let app = build_router(state);
        let response = app
            .oneshot(Request::builder().uri("/status").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn slots_reports_scheduler_snapshot() {
        let (state, _dir) = test_state("http://127.0.0.1:1".to_string()).await;
        let app = build_router(state);
        let response = app
            .oneshot(Request::builder().uri("/slots").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json.as_array().unwrap().len(), 2);
    }

    #[tokio::test]
    async fn cold_request_warms_slot_then_forwards_and_saves() {
        let server = MockServer::start().await;
        // Warm-up call (max_tokens: 1) hits the completions endpoint once...
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "choices": [{"delta": {"content": ""}, "finish_reason": "length"}]
            })))
            .mount(&server)
            .await;
        // ...and the real forwarded request hits it again.
        // (wiremock's default un-scoped mock above serves both calls;
        // this test asserts on call count via server.received_requests().)

        let (mut state, _dir) = test_state(server.uri()).await;
        state.llama_server_url = server.uri();
        let app = build_router(state);

        let req_body = serde_json::json!({
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "hi"}
            ],
            "aivyx_slot_hint": {"prefix_hash": "abc123", "preferred_slot": null}
        });

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(req_body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let requests = server.received_requests().await.unwrap();
        // Warm-up + real forward = at least 2 calls to llama-server.
        assert!(requests.len() >= 2, "expected warm-up + forward, got {}", requests.len());
    }

    #[tokio::test]
    async fn llama_server_error_status_and_body_are_forwarded_untouched() {
        // Finding 1: the broker must forward llama-server's real non-2xx
        // status code and body, not collapse everything into a generic
        // broker 502 via `.error_for_status()`.
        let server = MockServer::start().await;
        let upstream_error_body = serde_json::json!({
            "error": {"message": "slot is busy", "type": "server_error"}
        });
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(
                ResponseTemplate::new(429).set_body_json(upstream_error_body.clone()),
            )
            .mount(&server)
            .await;

        let (mut state, _dir) = test_state(server.uri()).await;
        state.llama_server_url = server.uri();
        let app = build_router(state);

        // No `aivyx_slot_hint` -- goes straight to forwarding with no
        // warm-up call, so the mock above is hit exactly once and there's
        // no ambiguity about which response we're checking.
        let req_body = serde_json::json!({"messages": [{"role": "user", "content": "hi"}]});

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(req_body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json, upstream_error_body);
    }

    #[tokio::test]
    async fn llama_server_response_headers_are_forwarded() {
        // Finding 2: the broker must copy upstream response headers (and
        // the real status code) onto its own response, per the design's
        // "relay untouched" requirement -- not hardcode 200 with no
        // headers copied.
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("x-llama-server-marker", "from-upstream")
                    .set_body_json(serde_json::json!({
                        "choices": [{"delta": {"content": "hi"}, "finish_reason": "stop"}]
                    })),
            )
            .mount(&server)
            .await;

        let (mut state, _dir) = test_state(server.uri()).await;
        state.llama_server_url = server.uri();
        let app = build_router(state);

        // No `aivyx_slot_hint` -- forward call only, no warm-up call to
        // confuse which response's headers are being asserted on.
        let req_body = serde_json::json!({"messages": [{"role": "user", "content": "hi"}]});

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(req_body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response.headers().get("x-llama-server-marker").map(|v| v.to_str().unwrap()),
            Some("from-upstream"),
        );
    }

    #[tokio::test]
    async fn warm_up_request_includes_tools_from_the_real_client_request() {
        // Finding 3: the design's own definition of the stable "prefix"
        // that gets cached under `prefix_hash` is system-prompt-plus-tools
        // together. The synthetic warm-up request must include the
        // client's real `tools` array, or the content actually cached
        // won't match what real requests for that same `prefix_hash` send.
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "choices": [{"delta": {"content": ""}, "finish_reason": "length"}]
            })))
            .mount(&server)
            .await;

        let (mut state, _dir) = test_state(server.uri()).await;
        state.llama_server_url = server.uri();
        let app = build_router(state);

        let tools = serde_json::json!([
            {"type": "function", "function": {"name": "get_weather", "parameters": {}}}
        ]);
        let req_body = serde_json::json!({
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "hi"}
            ],
            "tools": tools,
            "aivyx_slot_hint": {"prefix_hash": "abc123", "preferred_slot": null}
        });

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(req_body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let requests = server.received_requests().await.unwrap();
        // The warm-up request is identifiable by its `max_tokens: 1` /
        // `id_slot` shape (see `warm_or_restore`); the real forwarded
        // request is the other one.
        let warm_up = requests
            .iter()
            .map(|r| r.body_json::<serde_json::Value>().unwrap())
            .find(|b| b.get("max_tokens") == Some(&serde_json::json!(1)))
            .expect("expected a warm-up request among those received");
        assert_eq!(
            warm_up.get("tools"),
            Some(&tools),
            "warm-up request must include the client's real `tools` array"
        );
    }

    #[tokio::test]
    async fn queue_timeout_returns_503_not_a_hang() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "choices": [{"delta": {"content": "hi"}, "finish_reason": "stop"}]
            })))
            .mount(&server)
            .await;

        let (mut state, _dir) = test_state(server.uri()).await;
        state.llama_server_url = server.uri();
        state.scheduler = Scheduler::new(1);
        state.queue_timeout = Duration::from_millis(50);
        // Occupy the only slot directly via the scheduler, bypassing HTTP,
        // so the next request has nothing free to admit to.
        let _held = state.scheduler.admit(None, Duration::from_secs(1)).await.unwrap();

        let app = build_router(state);
        let req_body = serde_json::json!({"messages": []});
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(req_body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    }
}

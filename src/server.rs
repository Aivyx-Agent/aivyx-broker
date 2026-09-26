use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

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
    pub gpu_lock: crate::gpu_lock::GpuLock,
    pub gpu_lock_queue_timeout: Duration,
    /// Host VRAM for `/v1/aivyx/residency`.
    pub gpu_probe: Arc<dyn crate::gpu::GpuProbe>,
    /// The last `gpu_probe` result, shared by concurrent residency requests.
    pub vram_cache: VramCache,
}

/// The longest a residency request waits on the GPU probe. Past it, `vram`
/// is `null`; the probe's own thread finishes (or is bounded by
/// `gpu::NVIDIA_SMI_TIMEOUT`) in the background.
pub const RESIDENCY_PROBE_DEADLINE: Duration = Duration::from_millis(2500);

/// How long a probe result is reused.
pub const VRAM_CACHE_TTL: Duration = Duration::from_secs(2);

/// How long a timed-out probe's `None` is kept before probing again: a
/// probe that blew its deadline points at a wedged driver, which a 2 s
/// retry cadence would only pile onto.
pub const VRAM_TIMEOUT_BACKOFF: Duration = Duration::from_secs(30);

/// The last GPU probe result and when it was taken, behind a single-flight
/// lock: the lock is held while probing, so concurrent residency requests
/// share one probe (one `nvidia-smi` fork) instead of each spawning their
/// own. A result is reused for [`VRAM_CACHE_TTL`], or for
/// [`VRAM_TIMEOUT_BACKOFF`] when the probe timed out. A waiter never waits
/// longer than the holder's [`RESIDENCY_PROBE_DEADLINE`].
///
/// `in_flight` guards against stacking probes: a probe outliving its
/// deadline (an `nvidia-smi` in uninterruptible sleep never reaps, so its
/// blocking thread stays pinned) keeps the flag set until it actually
/// returns, and while it is set no new probe starts — the last cached
/// value (or `None`) is served instead. At most one blocking thread is
/// ever pinned by the probe.
#[derive(Clone, Default)]
pub struct VramCache {
    last: Arc<tokio::sync::Mutex<Option<VramReading>>>,
    in_flight: Arc<AtomicBool>,
}

/// One probe result, when it was taken, and how long it stays fresh.
#[derive(Clone, Copy)]
struct VramReading {
    taken: Instant,
    vram: Option<crate::gpu::Vram>,
    fresh_for: Duration,
}

/// Clears the in-flight flag when the probe closure finishes (or panics).
struct InFlight(Arc<AtomicBool>);

impl Drop for InFlight {
    fn drop(&mut self) {
        self.0.store(false, Ordering::SeqCst);
    }
}

impl VramCache {
    /// The cached VRAM, or a fresh probe of `probe` bounded by
    /// [`RESIDENCY_PROBE_DEADLINE`].
    async fn get(&self, probe: &Arc<dyn crate::gpu::GpuProbe>) -> Option<crate::gpu::Vram> {
        let mut last = self.last.lock().await;
        if let Some(reading) = *last
            && reading.taken.elapsed() < reading.fresh_for
        {
            return reading.vram;
        }
        // A previous probe is still out (hung): don't stack another.
        if self.in_flight.swap(true, Ordering::SeqCst) {
            return last.and_then(|r| r.vram);
        }
        let guard = InFlight(Arc::clone(&self.in_flight));
        let probe = Arc::clone(probe);
        // nvidia-smi is a blocking subprocess.
        let task = tokio::task::spawn_blocking(move || {
            let _guard = guard;
            probe.vram()
        });
        let (vram, fresh_for) = match tokio::time::timeout(RESIDENCY_PROBE_DEADLINE, task).await {
            Ok(joined) => (joined.ok().flatten(), VRAM_CACHE_TTL),
            Err(_) => (None, VRAM_TIMEOUT_BACKOFF),
        };
        *last = Some(VramReading {
            taken: Instant::now(),
            vram,
            fresh_for,
        });
        vram
    }
}

pub fn build_router(state: AppState) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/status", get(status))
        .route("/slots", get(slots))
        .route("/v1/aivyx/residency", get(residency))
        .route("/gpu-lock/acquire", post(gpu_lock_acquire))
        .route("/gpu-lock/release", post(gpu_lock_release))
        .with_state(state)
}

async fn status(State(state): State<AppState>) -> impl IntoResponse {
    Json(serde_json::json!({ "llama_server_url": state.llama_server_url, "queue_depth": 0 }))
}

async fn slots(State(state): State<AppState>) -> impl IntoResponse {
    Json(serde_json::json!(state.scheduler.snapshot()))
}

/// Model routing Part 4 — read-only residency for `aivyx-route`: the
/// upstream's models (loaded or not), host VRAM, and slot pressure.
/// Every part is best-effort; this always answers.
async fn residency(State(state): State<AppState>) -> impl IntoResponse {
    let models = crate::llama_client::fetch_models(&state.http, &state.llama_server_url)
        .await
        .unwrap_or_default();
    let vram = state.vram_cache.get(&state.gpu_probe).await;
    let slots = state.scheduler.snapshot();
    let busy = slots.iter().filter(|s| s.busy).count() as u32;
    Json(serde_json::json!({
        "models": models,
        "vram": vram,
        "slots": { "busy": busy, "total": slots.len() as u32 },
    }))
}

async fn gpu_lock_acquire(State(state): State<AppState>) -> axum::response::Response {
    match state.gpu_lock.acquire(state.gpu_lock_queue_timeout).await {
        Ok(lease) => (
            StatusCode::OK,
            Json(serde_json::json!({ "lease_id": lease.to_string() })),
        )
            .into_response(),
        Err(crate::gpu_lock::GpuLockError::Timeout) => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({"error": "timed out waiting for the GPU lock"})),
        )
            .into_response(),
        Err(crate::gpu_lock::GpuLockError::UnknownLease) => unreachable!(
            "acquire() never returns UnknownLease -- only release()/reap_expired() paths do"
        ),
    }
}

#[derive(serde::Deserialize)]
struct ReleaseRequest {
    lease_id: String,
}

async fn gpu_lock_release(
    State(state): State<AppState>,
    Json(body): Json<ReleaseRequest>,
) -> axum::response::Response {
    let lease_id = match body.lease_id.parse::<uuid::Uuid>() {
        Ok(uuid) => crate::gpu_lock::LeaseId::from(uuid),
        Err(_) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": "lease_id is not a valid UUID"})),
            )
                .into_response();
        }
    };
    match state.gpu_lock.release(lease_id) {
        Ok(()) => StatusCode::OK.into_response(),
        Err(crate::gpu_lock::GpuLockError::UnknownLease) => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": "lease not found, already released, or expired"})),
        )
            .into_response(),
        Err(crate::gpu_lock::GpuLockError::Timeout) => {
            unreachable!("release() never returns Timeout -- only acquire() does")
        }
    }
}

async fn chat_completions(
    State(state): State<AppState>,
    Json(mut body): Json<Value>,
) -> axum::response::Response {
    let hint = body
        .as_object_mut()
        .and_then(|o| o.remove("aivyx_slot_hint"))
        .and_then(|v| serde_json::from_value::<crate::SlotHint>(v).ok());

    let admission = match state
        .scheduler
        .admit(hint.clone(), state.queue_timeout)
        .await
    {
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
    let guard = ReleaseGuard {
        scheduler: state.scheduler.clone(),
        slot_id: admission.slot_id,
    };

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
fn release_on_stream_end<S>(
    inner: S,
    guard: ReleaseGuard,
) -> impl futures::Stream<Item = S::Item> + Send + 'static
where
    S: futures::Stream + Send + 'static,
{
    futures::stream::unfold(
        (Box::pin(inner), Some(guard)),
        |(mut stream, mut guard)| async move {
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
        },
    )
}

async fn handle_admitted(
    state: &AppState,
    body: &mut Value,
    hint: &Option<crate::SlotHint>,
    admission: crate::Admission,
    guard: ReleaseGuard,
) -> Result<axum::response::Response, anyhow::Error> {
    match hint {
        Some(h) if !admission.cache_ready => {
            warm_or_restore(state, body, h, admission.slot_id).await?;
        }
        None => {
            // A hint-less request still gets forwarded onto whatever slot
            // admission picked, silently overwriting that slot's real KV
            // content -- clear the occupancy table's record of what used
            // to be resident there so a later request for that stale
            // prefix doesn't wrongly see `cache_ready: true` and skip
            // restoring content that's actually gone (Fix 7).
            state.scheduler.clear_resident(admission.slot_id);
        }
        Some(_) => {}
    }

    if let Some(obj) = body.as_object_mut() {
        obj.insert("id_slot".to_string(), serde_json::json!(admission.slot_id));
    }

    let resp = state
        .http
        .post(format!(
            "{}/v1/chat/completions",
            state.llama_server_url.trim_end_matches('/')
        ))
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
    let axum_body =
        axum::body::Body::from_stream(release_on_stream_end(resp.bytes_stream(), guard));
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

    let restored = state
        .kv_store
        .restore_into_slot(&key, slot_id)
        .await
        .unwrap_or_else(|err| {
            tracing::warn!(error = %err, "kvcache: restore_into_slot failed");
            false
        });

    if !restored {
        let system_message = body
            .get("messages")
            .and_then(|m| m.as_array())
            .and_then(|arr| {
                arr.iter()
                    .find(|m| m.get("role").and_then(|r| r.as_str()) == Some("system"))
            })
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
            .post(format!(
                "{}/v1/chat/completions",
                state.llama_server_url.trim_end_matches('/')
            ))
            .json(&warm_up)
            .send()
            .await?
            .error_for_status()?;

        if let Err(err) = state
            .kv_store
            .save_from_slot(
                &key,
                slot_id,
                aivyx_kvcache::CacheMeta {
                    size_bytes: 1,
                    token_count: 1,
                },
            )
            .await
        {
            tracing::warn!(error = %err, "kvcache: save_from_slot failed");
        }
    }

    state
        .scheduler
        .mark_resident(slot_id, hint.prefix_hash.clone());
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
            gpu_lock: crate::gpu_lock::GpuLock::new(Duration::from_secs(600)),
            gpu_lock_queue_timeout: Duration::from_secs(5),
            gpu_probe: Arc::new(FakeProbe(None)),
            vram_cache: VramCache::default(),
        };
        (state, dir)
    }

    struct FakeProbe(Option<crate::gpu::Vram>);

    impl crate::gpu::GpuProbe for FakeProbe {
        fn vram(&self) -> Option<crate::gpu::Vram> {
            self.0
        }
    }

    const CONTRACT: &str = include_str!("../tests/fixtures/broker_residency.json");

    async fn get_json(app: Router, uri: &str) -> serde_json::Value {
        let response = app
            .oneshot(Request::builder().uri(uri).body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        serde_json::from_slice(&body).unwrap()
    }

    #[tokio::test]
    async fn residency_matches_the_shared_contract() {
        let upstream = wiremock::MockServer::start().await;
        wiremock::Mock::given(wiremock::matchers::method("GET"))
            .and(wiremock::matchers::path("/models"))
            .respond_with(
                wiremock::ResponseTemplate::new(200).set_body_json(serde_json::json!({
                    "data": [
                        {"id": "qwen3-8b", "status": {"value": "loaded"}},
                        {"id": "gemma-3-4b", "status": {"value": "unloaded"}}
                    ]
                })),
            )
            .mount(&upstream)
            .await;
        let (mut state, _dir) = test_state(upstream.uri()).await;
        state.gpu_probe = Arc::new(FakeProbe(Some(crate::gpu::Vram {
            total_bytes: 25_769_803_776,
            used_bytes: 9_663_676_416,
        })));
        // One of the two slots busy.
        let _admission = state
            .scheduler
            .admit(None, Duration::from_secs(1))
            .await
            .unwrap();
        let got = get_json(build_router(state), "/v1/aivyx/residency").await;
        let want: serde_json::Value = serde_json::from_str(CONTRACT).unwrap();
        assert_eq!(got, want);
    }

    #[tokio::test]
    async fn residency_without_upstream_or_gpu_still_answers() {
        let (state, _dir) = test_state("http://127.0.0.1:1".to_string()).await;
        let got = get_json(build_router(state), "/v1/aivyx/residency").await;
        assert_eq!(
            got,
            serde_json::json!({"models": [], "vram": null, "slots": {"busy": 0, "total": 2}})
        );
    }

    /// A wedged `nvidia-smi`: sleeps far past the residency deadline.
    struct HungProbe;

    impl crate::gpu::GpuProbe for HungProbe {
        fn vram(&self) -> Option<crate::gpu::Vram> {
            std::thread::sleep(Duration::from_secs(5));
            Some(crate::gpu::Vram {
                total_bytes: 1,
                used_bytes: 1,
            })
        }
    }

    #[tokio::test]
    async fn a_hung_gpu_probe_still_answers_within_the_deadline() {
        let (mut state, _dir) = test_state("http://127.0.0.1:1".to_string()).await;
        state.gpu_probe = Arc::new(HungProbe);
        let start = std::time::Instant::now();
        let got = get_json(build_router(state), "/v1/aivyx/residency").await;
        let elapsed = start.elapsed();
        assert!(elapsed < Duration::from_secs(4), "took {elapsed:?}");
        assert_eq!(got["vram"], serde_json::Value::Null);
    }

    struct CountingProbe(std::sync::atomic::AtomicUsize);

    impl crate::gpu::GpuProbe for CountingProbe {
        fn vram(&self) -> Option<crate::gpu::Vram> {
            self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Some(crate::gpu::Vram {
                total_bytes: 2,
                used_bytes: 1,
            })
        }
    }

    /// A wedged `nvidia-smi` stuck in D state: blocks until the test drops
    /// the gate's sender (so the runtime can shut down promptly), and
    /// counts how often it was started.
    struct GatedHungProbe {
        calls: std::sync::atomic::AtomicUsize,
        gate: std::sync::Mutex<std::sync::mpsc::Receiver<()>>,
    }

    impl crate::gpu::GpuProbe for GatedHungProbe {
        fn vram(&self) -> Option<crate::gpu::Vram> {
            self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let _ = self
                .gate
                .lock()
                .unwrap()
                .recv_timeout(Duration::from_secs(60));
            None
        }
    }

    #[tokio::test]
    async fn a_hung_probe_is_never_stacked_past_the_cache_ttl() {
        let (mut state, _dir) = test_state("http://127.0.0.1:1".to_string()).await;
        let (release, gate) = std::sync::mpsc::channel();
        let probe = Arc::new(GatedHungProbe {
            calls: std::sync::atomic::AtomicUsize::new(0),
            gate: std::sync::Mutex::new(gate),
        });
        state.gpu_probe = probe.clone();
        let app = build_router(state);
        let got = get_json(app.clone(), "/v1/aivyx/residency").await;
        assert_eq!(got["vram"], serde_json::Value::Null);
        // Past the 2 s TTL; the first probe is still hung.
        tokio::time::sleep(Duration::from_millis(2600)).await;
        let got = get_json(app, "/v1/aivyx/residency").await;
        assert_eq!(got["vram"], serde_json::Value::Null);
        assert_eq!(probe.calls.load(std::sync::atomic::Ordering::SeqCst), 1);
        drop(release);
    }

    #[tokio::test]
    async fn an_outstanding_probe_blocks_a_new_one_even_after_the_backoff() {
        // The in-flight guard alone: nothing cached, a probe still out.
        let cache = VramCache::default();
        cache
            .in_flight
            .store(true, std::sync::atomic::Ordering::SeqCst);
        let probe = Arc::new(CountingProbe(std::sync::atomic::AtomicUsize::new(0)));
        let dyn_probe: Arc<dyn crate::gpu::GpuProbe> = probe.clone();
        assert_eq!(cache.get(&dyn_probe).await, None);
        assert_eq!(probe.0.load(std::sync::atomic::Ordering::SeqCst), 0);
        // Once it returns, probing resumes.
        cache
            .in_flight
            .store(false, std::sync::atomic::Ordering::SeqCst);
        assert!(cache.get(&dyn_probe).await.is_some());
        assert_eq!(probe.0.load(std::sync::atomic::Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn back_to_back_residency_requests_share_one_probe() {
        let (mut state, _dir) = test_state("http://127.0.0.1:1".to_string()).await;
        let probe = Arc::new(CountingProbe(std::sync::atomic::AtomicUsize::new(0)));
        state.gpu_probe = probe.clone();
        let app = build_router(state);
        for _ in 0..2 {
            let got = get_json(app.clone(), "/v1/aivyx/residency").await;
            assert_eq!(
                got["vram"],
                serde_json::json!({"total_bytes": 2, "used_bytes": 1})
            );
        }
        assert_eq!(probe.0.load(std::sync::atomic::Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn status_reports_llama_server_url() {
        let (state, _dir) = test_state("http://127.0.0.1:1".to_string()).await;
        let app = build_router(state);
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/status")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn slots_reports_scheduler_snapshot() {
        let (state, _dir) = test_state("http://127.0.0.1:1".to_string()).await;
        let app = build_router(state);
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/slots")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
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
        assert!(
            requests.len() >= 2,
            "expected warm-up + forward, got {}",
            requests.len()
        );
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
            .respond_with(ResponseTemplate::new(429).set_body_json(upstream_error_body.clone()))
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
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
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
            response
                .headers()
                .get("x-llama-server-marker")
                .map(|v| v.to_str().unwrap()),
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
        let _held = state
            .scheduler
            .admit(None, Duration::from_secs(1))
            .await
            .unwrap();

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

    #[tokio::test]
    async fn hint_less_admission_clears_stale_resident_prefix_on_the_slot_it_lands_on() {
        // Regression for Fix 7: a hint-less request still gets forwarded
        // onto whatever slot admission picked, silently overwriting that
        // slot's real KV content -- the occupancy table must stop naming
        // the old prefix, or a later request for it wrongly sees
        // `cache_ready: true` and skips restoring content that's gone.
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
        // A single slot forces the hint-less second request to land on
        // exactly the slot the first (hinted) request marked resident.
        state.scheduler = Scheduler::new(1);
        let app = build_router(state.clone());

        // First request: has a hint, so `warm_or_restore` marks the slot
        // resident with "abc123".
        let hinted_body = serde_json::json!({
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "hi"}
            ],
            "aivyx_slot_hint": {"prefix_hash": "abc123", "preferred_slot": null}
        });
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(hinted_body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        // Drain the streamed body so the `ReleaseGuard` drops and frees
        // the slot for the next request.
        let _ = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();

        let snap = state.scheduler.snapshot();
        assert_eq!(snap[0].resident_prefix.as_deref(), Some("abc123"));

        // Second request: no hint at all -- lands on the same (only)
        // slot, forwarding onto it and overwriting its real content.
        let hint_less_body =
            serde_json::json!({"messages": [{"role": "user", "content": "hi again"}]});
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(hint_less_body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let _ = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();

        let snap = state.scheduler.snapshot();
        assert_eq!(
            snap[0].resident_prefix, None,
            "stale resident prefix must be cleared once a hint-less request overwrites the slot"
        );
    }

    #[tokio::test]
    async fn gpu_lock_acquire_then_release_round_trips_over_http() {
        let (state, _dir) = test_state("http://127.0.0.1:1".to_string()).await;
        let app = build_router(state);

        let acquire_response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/gpu-lock/acquire")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(acquire_response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(acquire_response.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let lease_id = json["lease_id"]
            .as_str()
            .expect("lease_id must be present")
            .to_string();

        let release_response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/gpu-lock/release")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        serde_json::json!({"lease_id": lease_id}).to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(release_response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn gpu_lock_acquire_times_out_with_503_when_already_held() {
        let (mut state, _dir) = test_state("http://127.0.0.1:1".to_string()).await;
        state.gpu_lock_queue_timeout = Duration::from_millis(50);
        let app = build_router(state.clone());

        // Hold the lock directly via the GpuLock, bypassing HTTP, so the next
        // HTTP acquire has nothing free.
        let _held = state
            .gpu_lock
            .acquire(Duration::from_secs(1))
            .await
            .unwrap();

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/gpu-lock/acquire")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    }

    #[tokio::test]
    async fn gpu_lock_release_with_an_unknown_lease_id_is_a_clear_error_not_a_panic() {
        let (state, _dir) = test_state("http://127.0.0.1:1".to_string()).await;
        let app = build_router(state);

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/gpu-lock/release")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        serde_json::json!({"lease_id": uuid::Uuid::new_v4().to_string()})
                            .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn gpu_lock_release_with_a_malformed_lease_id_is_a_clear_400_not_a_panic() {
        let (state, _dir) = test_state("http://127.0.0.1:1".to_string()).await;
        let app = build_router(state);

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/gpu-lock/release")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        serde_json::json!({"lease_id": "not-a-uuid"}).to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }
}

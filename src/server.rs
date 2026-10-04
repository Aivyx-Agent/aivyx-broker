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
    /// The largest `/v1/chat/completions` request body accepted, in bytes
    /// (replaces axum's 2 MB default; see `--max-request-body-bytes`).
    pub max_request_body_bytes: usize,
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
/// [`VRAM_TIMEOUT_BACKOFF`] when the probe timed out (here, or inside
/// `nvidia-smi`'s own deadline). A waiter never waits
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
        // A previous probe is still out (hung): don't stack another. Serve
        // the last reading only while it's recent enough to still describe
        // the GPU.
        if self.in_flight.swap(true, Ordering::SeqCst) {
            return last
                .filter(|r| r.taken.elapsed() < VRAM_TIMEOUT_BACKOFF)
                .and_then(|r| r.vram);
        }
        let guard = InFlight(Arc::clone(&self.in_flight));
        let probe = Arc::clone(probe);
        // nvidia-smi is a blocking subprocess.
        let task = tokio::task::spawn_blocking(move || {
            let _guard = guard;
            probe.vram()
        });
        let started = Instant::now();
        let (vram, fresh_for) = match tokio::time::timeout(RESIDENCY_PROBE_DEADLINE, task).await {
            // Nothing, after running into nvidia-smi's own deadline: it was
            // killed as too slow, so back off like a hung probe rather than
            // pay that deadline on every poll.
            Ok(joined)
                if started.elapsed() >= crate::gpu::NVIDIA_SMI_TIMEOUT
                    && joined.as_ref().is_ok_and(Option::is_none) =>
            {
                (None, VRAM_TIMEOUT_BACKOFF)
            }
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
    // axum caps extracted bodies at 2 MB by default; llama-server accepts far
    // more, and a long agent conversation with tool output passes 2 MB.
    let body_limit = axum::extract::DefaultBodyLimit::max(state.max_request_body_bytes);
    Router::new()
        .route(
            "/v1/chat/completions",
            post(chat_completions).layer(body_limit),
        )
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

    // From here on the slot is held. Nothing between `admit` returning and
    // the guard moving into the spawned task awaits, so a client that hangs
    // up can't drop the handler in between and leak the slot.
    let guard = ReleaseGuard {
        scheduler: state.scheduler.clone(),
        slot_id: admission.slot_id,
    };
    let (head_tx, head_rx) = tokio::sync::oneshot::channel();
    let (chunk_tx, chunk_rx) = tokio::sync::mpsc::channel(FORWARD_CHANNEL_CHUNKS);
    tokio::spawn(forward(
        state, body, hint, admission, guard, head_tx, chunk_tx,
    ));

    // This handler only relays the task's output; if the client goes away,
    // dropping it (and `head_rx`/`chunk_rx`) never cuts the task short.
    let head = match head_rx.await {
        Ok(Ok(head)) => head,
        Ok(Err(err)) => {
            tracing::warn!(error = %err, "chat_completions forwarding failed");
            return (
                StatusCode::BAD_GATEWAY,
                Json(serde_json::json!({"error": err.to_string()})),
            )
                .into_response();
        }
        Err(_) => {
            return (
                StatusCode::BAD_GATEWAY,
                Json(serde_json::json!({"error": "forwarding task ended without a response"})),
            )
                .into_response();
        }
    };

    let mut builder = axum::response::Response::builder().status(head.status);
    for (name, value) in head.headers.iter() {
        builder = builder.header(name, value);
    }
    let chunks = futures::stream::unfold(chunk_rx, |mut rx| async move {
        rx.recv().await.map(|chunk| (chunk, rx))
    });
    builder
        .body(axum::body::Body::from_stream(chunks))
        .unwrap_or_else(|err| {
            (
                StatusCode::BAD_GATEWAY,
                Json(serde_json::json!({"error": err.to_string()})),
            )
                .into_response()
        })
}

/// How many upstream body chunks the forwarding task may buffer ahead of a
/// slow client before it waits for the client to catch up.
const FORWARD_CHANNEL_CHUNKS: usize = 16;

/// llama-server's response status and headers, relayed untouched.
struct UpstreamHead {
    status: reqwest::StatusCode,
    headers: reqwest::header::HeaderMap,
}

/// Releases `slot_id` back to the scheduler on drop.
///
/// Owned by the spawned [`forward`] task, never by the client-facing
/// handler: llama-server keeps working on the slot -- warming it up,
/// restoring it, generating -- whether or not the client is still
/// listening, so the slot must stay busy until that upstream work has
/// finished or failed. Were the guard owned by the handler, a client
/// hanging up would drop it and hand the slot to the next request while
/// llama-server is still busy with the abandoned one. Dropping (rather
/// than an explicit `release` call) also covers the task panicking.
struct ReleaseGuard {
    scheduler: Scheduler,
    slot_id: u32,
}

impl Drop for ReleaseGuard {
    fn drop(&mut self) {
        self.scheduler.release(self.slot_id);
    }
}

/// The slot's whole upstream job, run in its own task: warm-up or restore,
/// the forwarded request, and the response body. Sends the response head on
/// `head_tx` and the body on `chunk_tx`. When the client has gone away (a
/// send fails) it keeps reading llama-server's response to the end and
/// discards it, so the slot (`guard`) is released only once llama-server is
/// actually done with it. Every early return drops `guard` too.
async fn forward(
    state: AppState,
    mut body: Value,
    hint: Option<crate::SlotHint>,
    admission: crate::Admission,
    guard: ReleaseGuard,
    head_tx: tokio::sync::oneshot::Sender<Result<UpstreamHead, anyhow::Error>>,
    chunk_tx: tokio::sync::mpsc::Sender<Result<axum::body::Bytes, reqwest::Error>>,
) {
    use futures::StreamExt;

    let resp = match send_upstream(&state, &mut body, &hint, admission).await {
        Ok(resp) => resp,
        Err(err) => {
            let _ = head_tx.send(Err(err));
            return;
        }
    };
    // Relay llama-server's response untouched: real status code, real
    // headers, real body -- whether it succeeded or not. Deliberately does
    // *not* call `.error_for_status()`, which would discard the upstream
    // status/body and collapse every non-2xx into a generic broker error.
    let head = UpstreamHead {
        status: resp.status(),
        headers: resp.headers().clone(),
    };
    let mut client = head_tx.send(Ok(head)).is_ok().then_some(chunk_tx);
    let mut stream = resp.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let failed = chunk.is_err();
        if let Some(tx) = &client
            && tx.send(chunk).await.is_err()
        {
            client = None;
        }
        if failed {
            break;
        }
    }
    // llama-server is done with the slot: free it before the client sees
    // the end of the body, so its next request doesn't queue behind itself.
    drop(guard);
    drop(client);
}

/// Prepares `admission`'s slot (warm-up/restore for a hinted request) and
/// sends `body` to llama-server on it, returning llama-server's response
/// with its body not yet read.
async fn send_upstream(
    state: &AppState,
    body: &mut Value,
    hint: &Option<crate::SlotHint>,
    admission: crate::Admission,
) -> Result<reqwest::Response, anyhow::Error> {
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

    Ok(state
        .http
        .post(format!(
            "{}/v1/chat/completions",
            state.llama_server_url.trim_end_matches('/')
        ))
        .json(body)
        .send()
        .await?)
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
            max_request_body_bytes: crate::config::DEFAULT_MAX_REQUEST_BODY_BYTES,
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

    /// A slow `nvidia-smi`: killed at its own deadline, so the probe
    /// returns `None` after about [`crate::gpu::NVIDIA_SMI_TIMEOUT`].
    struct SlowProbe(std::sync::atomic::AtomicUsize);

    impl crate::gpu::GpuProbe for SlowProbe {
        fn vram(&self) -> Option<crate::gpu::Vram> {
            self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            std::thread::sleep(crate::gpu::NVIDIA_SMI_TIMEOUT + Duration::from_millis(50));
            None
        }
    }

    /// Re-review R3 — a probe the deadline killed backs off like a hung
    /// one, instead of costing every poll another two seconds.
    #[tokio::test]
    async fn a_probe_killed_at_its_deadline_backs_off() {
        let cache = VramCache::default();
        let probe = Arc::new(SlowProbe(std::sync::atomic::AtomicUsize::new(0)));
        let dyn_probe: Arc<dyn crate::gpu::GpuProbe> = probe.clone();
        assert_eq!(cache.get(&dyn_probe).await, None);
        // Past the ordinary 2 s TTL, well inside the 30 s backoff.
        tokio::time::sleep(VRAM_CACHE_TTL + Duration::from_millis(200)).await;
        assert_eq!(cache.get(&dyn_probe).await, None);
        assert_eq!(probe.0.load(std::sync::atomic::Ordering::SeqCst), 1);
    }

    /// Re-review R4 — while a probe is stuck, an old reading isn't served
    /// as current.
    #[tokio::test]
    async fn a_stale_reading_is_not_served_while_a_probe_is_stuck() {
        let cache = VramCache::default();
        *cache.last.lock().await = Some(VramReading {
            taken: Instant::now()
                .checked_sub(VRAM_TIMEOUT_BACKOFF + Duration::from_secs(1))
                .unwrap(),
            vram: Some(crate::gpu::Vram {
                total_bytes: 2,
                used_bytes: 1,
            }),
            fresh_for: VRAM_CACHE_TTL,
        });
        cache
            .in_flight
            .store(true, std::sync::atomic::Ordering::SeqCst);
        let probe: Arc<dyn crate::gpu::GpuProbe> =
            Arc::new(CountingProbe(std::sync::atomic::AtomicUsize::new(0)));
        assert_eq!(cache.get(&probe).await, None);
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

    fn chat_request(body: &serde_json::Value) -> Request<Body> {
        Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap()
    }

    #[tokio::test]
    async fn a_client_disconnect_keeps_the_slot_until_the_upstream_job_finishes() {
        // The fake llama-server takes 600 ms to answer each request.
        const UPSTREAM: Duration = Duration::from_millis(600);
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_delay(UPSTREAM)
                    .set_body_json(serde_json::json!({
                        "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}]
                    })),
            )
            .mount(&server)
            .await;
        let (mut state, _dir) = test_state(server.uri()).await;
        state.scheduler = Scheduler::new(1);
        let app = build_router(state.clone());
        let body = serde_json::json!({"messages": [{"role": "user", "content": "hi"}]});

        let start = Instant::now();
        // Client 1 gives up 100 ms in, while llama-server is still working.
        let first = tokio::spawn(app.clone().oneshot(chat_request(&body)));
        tokio::time::sleep(Duration::from_millis(100)).await;
        first.abort();
        let _ = first.await;
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(
            state.scheduler.snapshot()[0].busy,
            "slot was released while llama-server was still running the abandoned request"
        );

        // Client 2 must not reach llama-server's slot until job 1 is done,
        // so it can't finish before two full upstream round trips.
        let response = app.oneshot(chat_request(&body)).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let _ = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let elapsed = start.elapsed();
        assert!(
            elapsed >= UPSTREAM * 2 - Duration::from_millis(50),
            "second request finished after {elapsed:?}: it overlapped the abandoned one"
        );
        assert_eq!(server.received_requests().await.unwrap().len(), 2);
        assert!(!state.scheduler.snapshot()[0].busy);
    }

    #[tokio::test]
    async fn a_client_disconnect_mid_stream_keeps_the_slot_until_the_upstream_stream_ends() {
        // A real streaming upstream: headers at once, then body chunks over
        // 600 ms. The client reads the headers and hangs up.
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let upstream_done = Arc::new(AtomicBool::new(false));
        let done = upstream_done.clone();
        tokio::spawn(async move {
            use tokio::io::{AsyncReadExt, AsyncWriteExt};
            let (mut sock, _) = listener.accept().await.unwrap();
            let mut buf = vec![0u8; 64 * 1024];
            let _ = sock.read(&mut buf).await;
            let _ = sock
                .write_all(
                    b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ntransfer-encoding: chunked\r\n\r\n",
                )
                .await;
            for _ in 0..6 {
                tokio::time::sleep(Duration::from_millis(100)).await;
                let chunk = b"data: {}\n\n";
                let _ = sock
                    .write_all(format!("{:x}\r\n", chunk.len()).as_bytes())
                    .await;
                let _ = sock.write_all(chunk).await;
                let _ = sock.write_all(b"\r\n").await;
            }
            let _ = sock.write_all(b"0\r\n\r\n").await;
            done.store(true, Ordering::SeqCst);
        });

        let (mut state, _dir) = test_state(format!("http://{addr}")).await;
        state.scheduler = Scheduler::new(1);
        let app = build_router(state.clone());
        let body =
            serde_json::json!({"stream": true, "messages": [{"role": "user", "content": "hi"}]});
        let response = app.oneshot(chat_request(&body)).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        // Hang up without reading the stream.
        drop(response);
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert!(
            state.scheduler.snapshot()[0].busy,
            "slot was released while llama-server was still streaming the abandoned request"
        );
        // Released once the upstream stream ends.
        tokio::time::timeout(Duration::from_secs(3), async {
            while state.scheduler.snapshot()[0].busy {
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        })
        .await
        .expect("slot never released after the upstream stream ended");
        assert!(upstream_done.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn a_request_larger_than_axums_2mb_default_is_forwarded() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "choices": [{"delta": {"content": "hi"}, "finish_reason": "stop"}]
            })))
            .mount(&server)
            .await;
        let (state, _dir) = test_state(server.uri()).await;
        let app = build_router(state);

        let big = "x".repeat(3 * 1024 * 1024);
        let req_body = serde_json::json!({"messages": [{"role": "user", "content": big}]});
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
        assert_eq!(requests.len(), 1);
        let forwarded: serde_json::Value = requests[0].body_json().unwrap();
        assert_eq!(
            forwarded["messages"][0]["content"].as_str().unwrap().len(),
            big.len()
        );
    }

    #[tokio::test]
    async fn a_request_over_the_configured_body_limit_is_rejected() {
        let (mut state, _dir) = test_state("http://127.0.0.1:1".to_string()).await;
        state.max_request_body_bytes = 1024;
        let app = build_router(state);
        let req_body =
            serde_json::json!({"messages": [{"role": "user", "content": "x".repeat(2048)}]});
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
        assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
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
        // Drain the body: the slot is freed once the forwarding task has
        // read llama-server's whole response.
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

# GPU Lock Extension Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a generic, lease-based GPU-exclusion primitive to `aivyx-broker` —
`GpuLock`, plus `POST /gpu-lock/acquire` / `POST /gpu-lock/release` HTTP
routes — so Aivyx-Vision's `aivyx-vision-mold` backend can coordinate
exclusive GPU access with whatever local LLM inference is already sharing
the machine through this same broker. This is the prerequisite `aivyx-vision-mold`
depends on, per `aivyx-ecosystem/docs/superpowers/specs/
2026-09-18-aivyx-vision-v1-design.md` §5.

**Architecture:** A new module, `src/gpu_lock.rs`, holding `GpuLock` — a
single-holder, lease-based lock, **deliberately independent of** the
existing `Scheduler` (`src/scheduler.rs`): no KV-cache awareness, no
prefix hints, no multi-slot LRU spreading, no seeded-across-restart
reconciliation. Not because those features don't matter, but because
they're all specific to `llama-server`'s own slot semantics, which a
GPU-heavy image/3D generation job has no equivalent of — see the design
spec's own reasoning for why reusing the *mechanism* (not just the
*daemon*) would be wrong. The waiter-queueing shape (a `VecDeque` of bare
`oneshot::Sender<()>` wake signals, every waiter re-deriving admission
from scratch under a fresh lock on each wake, `release` waking the whole
queue rather than stopping at the first successful send) is deliberately
copied from `Scheduler`'s own, already-hardened design — that file's own
test suite documents four real concurrency bugs (Fix 1-4) this exact
pattern was built to avoid; there is no reason to risk re-discovering the
same class of bug in a second, parallel implementation when the proven
shape is sitting right there in the same crate.

The one thing `GpuLock` has that `Scheduler` doesn't: a **max-hold safety
expiry**. `Scheduler`'s holder is always inside one HTTP request's own
handler stack (`ReleaseGuard` ties release to that request's response
stream lifetime, so a crash or disconnect releases it automatically via
`Drop`). `GpuLock`'s acquire/release are two **separate** HTTP calls from
an external client (`aivyx-vision-mold`) — a crash between them would hold
the lock forever with nothing to `Drop`. A background reap task (mirroring
`main.rs`'s existing `reconcile_seeded_slots` pattern) periodically
force-releases a lease that's exceeded a configured `max_hold` duration.

## Global Constraints

- `cargo clippy --all-targets -- -D warnings` and `cargo fmt --check` must
  stay clean (this repo's own documented gate).
- `cargo test` must stay green.
- `GpuLock` must not import from or otherwise couple to `scheduler.rs`,
  `llama_client.rs`, or `aivyx_kvcache` — it is a new, independent
  primitive sharing only the process (the daemon), not the mechanism.
- Every new concurrency-sensitive behavior needs a real regression test,
  not just a happy-path test — in particular, the "wake every waiter, not
  just the first" behavior must be tested the same way `Scheduler`'s own
  `release_wakes_a_second_waiter_even_if_the_first_woken_one_is_dropped_unpolled`
  test proves it (re-read that test before writing this task's own
  equivalent).
- Default branch of this repo is `master`, not `main` — confirm before
  any git operation that assumes otherwise.
- Re-read every file cited by line number below before editing — accurate
  as of this plan's own research (2026-09-18) but the repo moves.

---

## Task 1: `GpuLock` primitive + unit tests

**Files:**
- Create: `/home/julian/Projects/Rust/aivyx-broker/src/gpu_lock.rs`
- Modify: `/home/julian/Projects/Rust/aivyx-broker/src/lib.rs` (add
  `pub mod gpu_lock;` and re-export `GpuLock`/`GpuLockError`/`LeaseId`
  alongside the existing `Scheduler`/`SchedulerError` re-exports)
- Modify: `/home/julian/Projects/Rust/aivyx-broker/Cargo.toml` (add a
  `uuid` dependency with the `v4` feature — check whether any sibling
  Aivyx repo already pins a specific `uuid` version, e.g. `aivyx-pa`'s
  webhook-secret generation from the 2026-09-16 security-audit-fixes work
  used `uuid` v4 — match that version if you can find it easily; otherwise
  let `cargo add uuid --features v4` resolve it)

**Interfaces:**
- Produces: `LeaseId` (a `Copy`, `PartialEq`, `serde::Serialize +
  Deserialize` newtype wrapping a `uuid::Uuid`), `GpuLockError` (`Timeout`,
  `UnknownLease`), `GpuLock::new(max_hold: Duration) -> Self`,
  `async fn acquire(&self, queue_timeout: Duration) -> Result<LeaseId, GpuLockError>`,
  `fn release(&self, lease: LeaseId) -> Result<(), GpuLockError>`,
  `fn reap_expired(&self)` (force-releases an over-held lease; call
  periodically from a background task, added in Task 3).

- [ ] **Step 1: Write the failing tests**

```rust
use std::time::Duration;

use super::*;

#[tokio::test]
async fn acquire_and_release_round_trips() {
    let lock = GpuLock::new(Duration::from_secs(60));
    let lease = lock.acquire(Duration::from_secs(1)).await.unwrap();
    lock.release(lease).unwrap();
}

#[tokio::test]
async fn a_second_acquire_waits_for_the_first_release() {
    let lock = GpuLock::new(Duration::from_secs(60));
    let first = lock.acquire(Duration::from_secs(1)).await.unwrap();

    let lock2 = lock.clone();
    let waiter = tokio::spawn(async move { lock2.acquire(Duration::from_secs(5)).await });

    tokio::time::sleep(Duration::from_millis(50)).await;
    lock.release(first).unwrap();

    let second = waiter.await.unwrap().unwrap();
    lock.release(second).unwrap();
}

#[tokio::test]
async fn acquire_times_out_clearly_when_never_released() {
    let lock = GpuLock::new(Duration::from_secs(60));
    let _held = lock.acquire(Duration::from_secs(1)).await.unwrap();

    let result = lock.acquire(Duration::from_millis(50)).await;
    assert!(matches!(result, Err(GpuLockError::Timeout)));
}

#[tokio::test]
async fn releasing_an_unknown_or_already_released_lease_is_a_clear_error() {
    let lock = GpuLock::new(Duration::from_secs(60));
    let lease = lock.acquire(Duration::from_secs(1)).await.unwrap();
    lock.release(lease).unwrap();

    // Releasing the same lease twice must not panic or silently succeed --
    // the second caller has no lock to release.
    let result = lock.release(lease);
    assert!(matches!(result, Err(GpuLockError::UnknownLease)));
}

#[tokio::test]
async fn release_wakes_every_queued_waiter_not_just_the_first() {
    // Same regression shape as Scheduler's own
    // `release_wakes_a_second_waiter_even_if_the_first_woken_one_is_dropped_unpolled`
    // (src/scheduler.rs) -- if `release` stopped after the first
    // successful wake `send()`, a waiter whose future is dropped
    // (e.g. an HTTP client disconnect) in the window between that send
    // succeeding and being re-polled would silently strand every waiter
    // behind it until their own timeout, even though the lock is free.
    let lock = GpuLock::new(Duration::from_secs(60));
    let held = lock.acquire(Duration::from_secs(5)).await.unwrap();

    // Waiter A: poll exactly once by hand so it registers itself in the
    // wait queue and suspends, without ever being polled again.
    let lock_a = lock.clone();
    let mut waiter_a = Box::pin(lock_a.acquire(Duration::from_secs(5)));
    assert!(futures::poll!(&mut waiter_a).is_pending());

    // Waiter B: queued for real on the runtime.
    let lock_b = lock.clone();
    let waiter_b = tokio::spawn(async move { lock_b.acquire(Duration::from_secs(5)).await });
    tokio::time::sleep(Duration::from_millis(20)).await;

    lock.release(held).unwrap();
    drop(waiter_a); // simulates A's client disconnecting before re-polling

    let admitted = tokio::time::timeout(Duration::from_secs(1), waiter_b)
        .await
        .expect("B must be admitted well before its timeout, not stranded by A's drop")
        .unwrap()
        .unwrap();
    lock.release(admitted).unwrap();
}

#[tokio::test]
async fn reap_expired_force_releases_a_lease_past_max_hold() {
    let lock = GpuLock::new(Duration::from_millis(50));
    let _held = lock.acquire(Duration::from_secs(1)).await.unwrap();

    tokio::time::sleep(Duration::from_millis(100)).await;
    lock.reap_expired();

    // The lock must now be free -- a fresh acquire should succeed
    // immediately rather than timing out.
    let result = tokio::time::timeout(Duration::from_millis(50), lock.acquire(Duration::from_secs(1))).await;
    assert!(result.is_ok(), "lock should be free after reap_expired force-released the stale lease");
}

#[tokio::test]
async fn reap_expired_is_a_no_op_when_nothing_is_held_or_the_hold_is_still_fresh() {
    let lock = GpuLock::new(Duration::from_secs(60));
    lock.reap_expired(); // nothing held -- must not panic

    let held = lock.acquire(Duration::from_secs(1)).await.unwrap();
    lock.reap_expired(); // held, but well within max_hold -- must not release it

    // Still held -- releasing the real lease must still succeed (proves
    // reap_expired did not already release it).
    lock.release(held).unwrap();
}

#[tokio::test]
async fn reap_expired_wakes_a_queued_waiter() {
    // Same reasoning as Scheduler's own
    // `reconcile_seeded_idle_wakes_a_queued_waiter` -- force-freeing the
    // lock via the reap path must wake queued waiters too, not just the
    // ordinary `release()` path.
    let lock = GpuLock::new(Duration::from_millis(50));
    let _held = lock.acquire(Duration::from_secs(1)).await.unwrap();

    let lock2 = lock.clone();
    let waiter = tokio::spawn(async move { lock2.acquire(Duration::from_secs(60)).await });
    tokio::time::sleep(Duration::from_millis(20)).await;

    tokio::time::sleep(Duration::from_millis(60)).await; // let the held lease exceed max_hold
    lock.reap_expired();

    let admitted = tokio::time::timeout(Duration::from_secs(1), waiter)
        .await
        .expect("waiter must be woken promptly by reap_expired, not stranded until its own timeout")
        .unwrap()
        .unwrap();
    lock.release(admitted).unwrap();
}
```

- [ ] **Step 2: Run to verify they fail**

```bash
cd /home/julian/Projects/Rust/aivyx-broker
cargo test gpu_lock -- --nocapture
```

Expected: compile error — `GpuLock` doesn't exist yet.

- [ ] **Step 3: Implement `GpuLock`**

```rust
//! A generic, lease-based exclusive lock for GPU-heavy local generation
//! work (Aivyx-Vision's mold backend). Deliberately independent of
//! `scheduler.rs`'s `Scheduler` -- no KV-cache awareness, no prefix
//! hints, no multi-slot spreading, no seeded-across-restart
//! reconciliation. Those are all specific to llama-server's own slot
//! semantics; a GPU generation job has no equivalent of any of them. See
//! `aivyx-ecosystem/docs/superpowers/specs/
//! 2026-09-18-aivyx-vision-v1-design.md` §5.
//!
//! The waiter-queueing shape (bare `oneshot::Sender<()>` wake signals,
//! every waiter re-deriving admission from scratch under a fresh lock on
//! each wake, `release`/`reap_expired` waking the *entire* queue rather
//! than stopping at the first successful send) is deliberately copied
//! from `Scheduler`'s own design -- that file's test suite documents the
//! exact bug class (Fix 3: stopping at the first successful wake strands
//! every waiter behind a dropped one) this shape exists to avoid.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use tokio::sync::oneshot;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct LeaseId(uuid::Uuid);

impl LeaseId {
    fn new() -> Self {
        LeaseId(uuid::Uuid::new_v4())
    }
}

impl std::fmt::Display for LeaseId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum GpuLockError {
    #[error("timed out waiting for the GPU lock")]
    Timeout,
    #[error("lease not found, already released, or expired")]
    UnknownLease,
}

struct Held {
    lease: LeaseId,
    acquired_at: Instant,
}

struct Inner {
    held: Option<Held>,
    waiters: VecDeque<oneshot::Sender<()>>,
}

#[derive(Clone)]
pub struct GpuLock {
    inner: Arc<Mutex<Inner>>,
    max_hold: Duration,
}

impl GpuLock {
    pub fn new(max_hold: Duration) -> Self {
        Self {
            inner: Arc::new(Mutex::new(Inner {
                held: None,
                waiters: VecDeque::new(),
            })),
            max_hold,
        }
    }

    /// Waits for and claims exclusive access, or returns `Timeout` if
    /// `queue_timeout` elapses first. Re-derives admission from scratch
    /// under a fresh lock on every wake -- never assumes ownership from
    /// the wake signal alone -- so a wake that goes to a since-dropped
    /// waiter is harmless (see module doc comment).
    pub async fn acquire(&self, queue_timeout: Duration) -> Result<LeaseId, GpuLockError> {
        let deadline = tokio::time::Instant::now() + queue_timeout;
        loop {
            let rx = {
                let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
                if inner.held.is_none() {
                    let lease = LeaseId::new();
                    inner.held = Some(Held {
                        lease,
                        acquired_at: Instant::now(),
                    });
                    return Ok(lease);
                }
                let (tx, rx) = oneshot::channel();
                inner.waiters.push_back(tx);
                rx
            };
            match tokio::time::timeout_at(deadline, rx).await {
                Ok(Ok(())) => continue,
                Ok(Err(_)) | Err(_) => return Err(GpuLockError::Timeout),
            }
        }
    }

    /// Releases `lease`. Returns `UnknownLease` if it doesn't match the
    /// current holder (already released, expired via `reap_expired`, or
    /// never valid) -- the caller has nothing to release either way, but
    /// distinguishing this from success lets a caller notice a double-
    /// release bug in its own code rather than silently no-op-ing.
    pub fn release(&self, lease: LeaseId) -> Result<(), GpuLockError> {
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        match &inner.held {
            Some(h) if h.lease == lease => {
                inner.held = None;
                Self::wake_all(&mut inner);
                Ok(())
            }
            _ => Err(GpuLockError::UnknownLease),
        }
    }

    /// Force-releases the current holder if it has held the lock longer
    /// than `max_hold` -- the safety valve against a crashed or
    /// disconnected client that never calls `release`. No-op if nothing
    /// is held or the current hold is still within `max_hold`. Intended
    /// to be called periodically from a background task (see `main.rs`'s
    /// `reconcile_seeded_slots` for the existing analogous pattern).
    pub fn reap_expired(&self) {
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let expired = inner
            .held
            .as_ref()
            .is_some_and(|h| h.acquired_at.elapsed() > self.max_hold);
        if expired {
            tracing::warn!(
                "gpu-lock: force-releasing a lease that exceeded max_hold -- \
                 the holder likely crashed or disconnected without releasing"
            );
            inner.held = None;
            Self::wake_all(&mut inner);
        }
    }

    /// Wakes every queued waiter with a bare signal, not just the first --
    /// see the module doc comment for why stopping early strands waiters.
    /// Ignores individual send failures (a `Receiver` already dropped
    /// from timing out or being cancelled).
    fn wake_all(inner: &mut Inner) {
        while let Some(tx) = inner.waiters.pop_front() {
            let _ = tx.send(());
        }
    }

    #[cfg(test)]
    pub(crate) fn is_held(&self) -> bool {
        self.inner.lock().unwrap().held.is_some()
    }
}

#[cfg(test)]
mod tests {
    // (Step 1's tests land here.)
}
```

(Confirm `uuid::Uuid::new_v4()` requires the `v4` feature on the `uuid`
crate — Step 1's Cargo.toml note already covers adding it. If a sibling
repo's own `uuid` usage pins a specific version, match it; there is no
existing `uuid` dependency in *this* repo to conflict with.)

- [ ] **Step 4: Run to verify they pass**

```bash
cd /home/julian/Projects/Rust/aivyx-broker
cargo test gpu_lock -- --nocapture
```

Expected: all 7 new tests pass.

- [ ] **Step 5: Run full crate check**

```bash
cargo test
cargo clippy --all-targets -- -D warnings
cargo fmt --check
```

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat: add GpuLock -- a generic, lease-based GPU-exclusion primitive

New, independent primitive for Aivyx-Vision's mold backend to coordinate
GPU access with local LLM inference sharing this same broker. Waiter-
queueing shape deliberately copied from Scheduler's own hardened design
(wake every queued waiter on release, never just the first -- see that
file's own Fix 3 regression test for why). Adds a max-hold safety expiry
Scheduler doesn't need (its holder lives inside one HTTP request's own
handler via ReleaseGuard's Drop; this lock's acquire/release are two
separate external calls, so a crashed client between them needs an
explicit reap path)."
```

---

## Task 2: `POST /gpu-lock/acquire` / `POST /gpu-lock/release` routes

**Files:**
- Modify: `/home/julian/Projects/Rust/aivyx-broker/src/server.rs`

**Interfaces:**
- Consumes: `crate::gpu_lock::{GpuLock, GpuLockError, LeaseId}` (Task 1).
- Produces: `AppState` gains a `gpu_lock: GpuLock` field and a
  `gpu_lock_queue_timeout: Duration` field; `build_router` gains two new
  routes returning JSON.

**Grounding for the code below:** the existing `chat_completions` handler
and its `queue_timeout`/`SERVICE_UNAVAILABLE`-on-timeout handling
(`server.rs`, already read during this plan's research) is the closest
real template for the error-mapping shape — match its style.

- [ ] **Step 1: Write the failing tests**

```rust
// Add to server.rs's existing `#[cfg(test)] mod tests` block, alongside
// the existing `test_state` helper -- extend that helper (or add a
// variant) to also construct a `GpuLock` with a short `max_hold` and
// `gpu_lock_queue_timeout` for these tests specifically.

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
    let lease_id = json["lease_id"].as_str().expect("lease_id must be present").to_string();

    let release_response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/gpu-lock/release")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::json!({"lease_id": lease_id}).to_string()))
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
    let _held = state.gpu_lock.acquire(Duration::from_secs(1)).await.unwrap();

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
                    serde_json::json!({"lease_id": uuid::Uuid::new_v4().to_string()}).to_string(),
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
                .body(Body::from(serde_json::json!({"lease_id": "not-a-uuid"}).to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}
```

- [ ] **Step 2: Run to verify they fail**

```bash
cd /home/julian/Projects/Rust/aivyx-broker
cargo test gpu_lock -- --nocapture
```

Expected: compile error — `AppState` has no `gpu_lock` field yet, routes
don't exist.

- [ ] **Step 3: Implement the routes**

In `server.rs`, extend `AppState`:

```rust
#[derive(Clone)]
pub struct AppState {
    pub scheduler: Scheduler,
    pub http: reqwest::Client,
    pub llama_server_url: String,
    pub kv_store: Arc<aivyx_kvcache::LlamaServerSlotStore>,
    pub queue_timeout: Duration,
    pub gpu_lock: crate::gpu_lock::GpuLock,
    pub gpu_lock_queue_timeout: Duration,
}
```

Extend `build_router`:

```rust
pub fn build_router(state: AppState) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/status", get(status))
        .route("/slots", get(slots))
        .route("/gpu-lock/acquire", post(gpu_lock_acquire))
        .route("/gpu-lock/release", post(gpu_lock_release))
        .with_state(state)
}
```

New handlers:

```rust
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
        Ok(uuid) => crate::gpu_lock::LeaseId::from(uuid), // see note below
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
        Err(crate::gpu_lock::GpuLockError::Timeout) => unreachable!(
            "release() never returns Timeout -- only acquire() does"
        ),
    }
}
```

(`crate::gpu_lock::LeaseId::from(uuid::Uuid)` isn't defined in Task 1's
sketch — Task 1's `LeaseId` only exposes a private `new()` constructor.
Add `impl From<uuid::Uuid> for LeaseId` to `gpu_lock.rs` in this task
(`LeaseId(uuid)`, a one-line trivial wrapping conversion), since it's
needed here to parse an inbound lease id from JSON; this is a small,
legitimate addition this task's own scope requires, not scope creep into
Task 1's already-closed work.)

Update `test_state` (or add a variant) to construct `AppState` with the
two new fields — a `GpuLock::new(Duration::from_secs(600))` and a
`gpu_lock_queue_timeout` matching whatever this task's own tests need
(short for the timeout test, generous for the round-trip test).

- [ ] **Step 4: Run to verify they pass**

```bash
cd /home/julian/Projects/Rust/aivyx-broker
cargo test gpu_lock -- --nocapture
```

- [ ] **Step 5: Run full crate check**

```bash
cargo test
cargo clippy --all-targets -- -D warnings
cargo fmt --check
```

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat: add POST /gpu-lock/acquire and /gpu-lock/release routes

Wires GpuLock into AppState/build_router, matching chat_completions'
own timeout-to-503 error-mapping style. Malformed/unknown lease ids get
a clear 400/404, never a panic."
```

---

## Task 3: `main.rs` wiring, config, reap task, docs, final verification

**Files:**
- Modify: `/home/julian/Projects/Rust/aivyx-broker/src/config.rs`
- Modify: `/home/julian/Projects/Rust/aivyx-broker/src/main.rs`
- Modify: `/home/julian/Projects/Rust/aivyx-broker/README.md`
- Modify: `/home/julian/Projects/Rust/aivyx-broker/CLAUDE.md`

**Interfaces:** none new — wiring + documentation only.

- [ ] **Step 1: Add config fields**

In `config.rs`'s `BrokerConfig`:

```rust
    /// How long a caller may wait in the GPU-lock queue before getting a
    /// clear timeout error instead of hanging. Distinct from
    /// `queue_timeout_secs` (the LLM chat-completion admission queue) --
    /// a GPU generation job can legitimately queue much longer than a
    /// chat turn.
    #[arg(long, env = "AIVYX_BROKER_GPU_LOCK_QUEUE_TIMEOUT_SECS", default_value_t = 300)]
    pub gpu_lock_queue_timeout_secs: u64,

    /// Safety valve: a GPU lock lease held longer than this is force-
    /// released by the background reap task, on the assumption its
    /// holder crashed or disconnected without releasing. Set generously
    /// above realistic generation time (an image/3D generation job can
    /// legitimately run for minutes).
    #[arg(long, env = "AIVYX_BROKER_GPU_LOCK_MAX_HOLD_SECS", default_value_t = 900)]
    pub gpu_lock_max_hold_secs: u64,
```

(Confirm these default values -- 300s queue timeout, 900s/15min max
hold -- are reasonable against real `mold` generation-time expectations
before finalizing; adjust if research into typical local FLUX/SDXL
generation time on consumer hardware suggests otherwise. This is a
judgment call this task should make with whatever information is
available, not leave as a placeholder.)

Add a test to `config.rs`'s existing test module confirming both new
fields parse with their defaults, following the exact shape of
`parses_required_and_default_fields`.

- [ ] **Step 2: Wire into `main.rs`**

```rust
let gpu_lock = aivyx_broker::gpu_lock::GpuLock::new(Duration::from_secs(
    config.gpu_lock_max_hold_secs,
));
```

Add to the `AppState` construction (alongside the existing `scheduler`/
`http`/etc. fields):

```rust
        gpu_lock: gpu_lock.clone(),
        gpu_lock_queue_timeout: Duration::from_secs(config.gpu_lock_queue_timeout_secs),
```

Add a background reap task, mirroring the existing
`tokio::spawn(reconcile_seeded_slots(...))` call's shape:

```rust
tokio::spawn(reap_expired_gpu_lock(gpu_lock));
```

with a new function near the existing `reconcile_seeded_slots`:

```rust
/// Periodically force-releases a GPU lock lease that's exceeded its
/// max_hold safety expiry -- see `GpuLock::reap_expired`'s own doc
/// comment for why this exists (a crashed client between its acquire
/// and release calls has no Drop guard to release it automatically,
/// unlike the chat-completions path's ReleaseGuard).
async fn reap_expired_gpu_lock(gpu_lock: aivyx_broker::gpu_lock::GpuLock) {
    let mut interval = tokio::time::interval(RECONCILE_INTERVAL);
    loop {
        interval.tick().await;
        gpu_lock.reap_expired();
    }
}
```

(Confirm `RECONCILE_INTERVAL` — already defined at the top of `main.rs`
for the existing reconciliation task — is an appropriate poll interval to
reuse here too, or whether the GPU lock's reap check should run on its
own, possibly different, interval; a shared constant is simpler and
`RECONCILE_INTERVAL`'s existing 10s value is short relative to any
reasonable `max_hold`, so reuse is likely fine, but confirm this
reasoning holds rather than assuming it.)

- [ ] **Step 3: Update `README.md` and `CLAUDE.md`**

Add a section to `README.md` documenting the new endpoints (`POST
/gpu-lock/acquire` → `{"lease_id": "<uuid>"}` or 503 on timeout; `POST
/gpu-lock/release` with `{"lease_id": "<uuid>"}` → 200, or 404/400 on an
unknown/malformed lease) and the two new config flags, following
whatever format the README already uses for the existing `/v1/chat/completions`
endpoint and `queue_timeout_secs`.

Update `CLAUDE.md`'s "What this is" section to mention the broker now
also arbitrates GPU-heavy non-LLM workloads (Aivyx-Vision's mold backend)
via this new, independent primitive — one sentence, pointing at
`gpu_lock.rs`'s own doc comment for the full rationale, matching this
file's existing terse-pointer style rather than duplicating detail here.

- [ ] **Step 4: Full workspace verification**

```bash
cd /home/julian/Projects/Rust/aivyx-broker
cargo build
cargo test 2>&1 | tail -40
cargo clippy --all-targets -- -D warnings
cargo fmt --check
```

Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: wire GpuLock into main.rs -- config, background reap task, docs"
```

---

## Final verification (after all 3 tasks land)

- [ ] Run the complete test suite once, not per-task:

```bash
cd /home/julian/Projects/Rust/aivyx-broker
cargo test 2>&1 | tail -30
```

- [ ] Run the documented clippy + fmt commands once more:

```bash
cargo clippy --all-targets -- -D warnings
cargo fmt --check
```

- [ ] This plan does not decide whether to push a branch / open a PR —
  follow `superpowers:finishing-a-development-branch` once all tasks are
  individually reviewed and a final whole-branch review has passed, same
  as every other plan executed this cycle. **This repo's default branch
  is `master`, not `main`** — confirm the finishing skill detects this
  correctly rather than assuming `main`.

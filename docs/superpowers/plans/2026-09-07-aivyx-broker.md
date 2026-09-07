# aivyx-broker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `aivyx-broker`, a standalone loopback-HTTP daemon that both
`aivyx` and `aivyx-coder` can point at instead of `llama-server` directly,
providing cache-locality-aware slot admission and FIFO queuing across
independent local processes sharing one GPU-backed `llama-server` — then
wire both client apps to use it as a new, purely additive opt-in mode.

**Architecture:** An Axum HTTP server exposing an OpenAI-compatible
`/v1/chat/completions` passthrough (plus `/status`/`/slots` admin
endpoints), backed by an in-memory scheduler core (occupancy table + FIFO
wait queues) and an `aivyx-kvcache`-driven restore/warm/save lifecycle that
fully replaces each client's own `ensure_kv_slot_checked_out`-style logic
on this path. Design spec:
`docs/superpowers/specs/2026-09-07-aivyx-broker-design.md` (this repo) —
read it in full before starting; this plan assumes it.

**Tech Stack:** Rust, Axum 0.8, Tokio, reqwest (rustls), `aivyx-kvcache`
(git dependency), `wiremock` for HTTP-mock tests.

## Global Constraints

- Loopback-only, no auth: binds `127.0.0.1` by default, same trust model as
  `llama-server` itself. Never add auth/remote-binding in this plan.
- FIFO scheduling only in v1 — no priority/weighting. Do not add either.
- The broker keeps no state that must outlive its own process. On startup
  it always rebuilds occupancy from `llama-server`'s real `GET /slots`.
  Never persist scheduler state to disk.
- `aivyx-kvcache`'s WAL-sqlite manifest remains the sole source of truth
  for saved KV content on disk — the broker drives it as a library
  dependency (`LlamaServerSlotStore`), never reimplements persistence.
- The `aivyx_slot_hint` field on incoming requests is additive/optional —
  a request without it must work exactly like a plain OpenAI-compatible
  call (any free slot, LRU-first).
- No test in this plan may require a real GPU, a real GGUF model, or a
  real `llama-server` process — this environment has none of those. Use
  `wiremock` to fake `llama-server`'s HTTP surface; use fabricated
  in-memory structs for scheduler-only tests.
- Dependency versions, once picked in Task 1, are fixed for the rest of
  the plan: `axum = "0.8.9"`, `tokio = "1.52.3"`, `reqwest = "0.13.4"`
  (`default-features = false`, `features = ["json", "stream", "rustls"]`),
  `clap = "4.6.1"` (`features = ["derive", "env"]`), `tracing = "0.1.44"`,
  `tracing-subscriber = "0.3.23"` (`features = ["env-filter"]`),
  `thiserror = "2.0.18"`, `wiremock = "0.6.5"` (dev-only) — all match
  existing pins elsewhere in the Aivyx ecosystem; do not introduce a
  different version of any of these.
- Repo scope: Tasks 1–5 live entirely in this repo (`aivyx-broker`).
  Task 6 lives in `/home/julian/Projects/Rust/aivyx` (a **different git
  repo** — branch there, don't touch this repo's history). Task 7 lives in
  `/home/julian/Projects/Rust/aivyx-coder` (also a different git repo,
  branch there). Each of the three repos gets its own branch and its own
  `finishing-a-development-branch` pass at the end — they are not one
  commit history.

---

## Part A — `aivyx-broker` (this repo)

### Task 1: Cargo scaffold, CLI, health endpoint

**Files:**
- Create: `Cargo.toml`
- Create: `src/main.rs`
- Create: `src/config.rs`
- Create: `src/lib.rs`
- Create: `CLAUDE.md`
- Create: `README.md` (stub — full docs land in Task 5)
- Test: inline `#[cfg(test)]` in `src/config.rs`

**Interfaces:**
- Produces: `aivyx_broker::config::BrokerConfig` — the parsed CLI/env
  config every later task's `main.rs` wiring consumes. Fields: `port: u16`
  (default `8899`), `llama_server_url: String` (required, `--llama-server-url`
  / env `AIVYX_BROKER_LLAMA_SERVER_URL`), `kvcache_store_path: PathBuf`
  (required, `--kvcache-store-path` / env `AIVYX_BROKER_KVCACHE_STORE_PATH`),
  `kvcache_max_bytes: u64` (default `10 * 1024 * 1024 * 1024`,
  `--kvcache-max-bytes`), `queue_timeout_secs: u64` (default `60`,
  `--queue-timeout-secs`).

- [ ] **Step 1: Write `Cargo.toml`**

```toml
[package]
name = "aivyx-broker"
description = "Multi-process GPU-slot scheduling broker for local llama-server deployments"
version = "0.1.0"
edition = "2024"
license = "MIT OR Apache-2.0"

[[bin]]
name = "aivyx-broker"
path = "src/main.rs"

[dependencies]
aivyx-kvcache = { git = "https://github.com/Aivyx-Agent/aivyx-kvcache", branch = "main" }
axum = "0.8.9"
clap = { version = "4.6.1", features = ["derive", "env"] }
futures = "0.3"
reqwest = { version = "0.13.4", default-features = false, features = ["json", "stream", "rustls"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
thiserror = "2.0.18"
tokio = { version = "1.52.3", features = ["rt-multi-thread", "macros", "sync", "time", "net"] }
tracing = "0.1.44"
tracing-subscriber = { version = "0.3.23", features = ["env-filter"] }

[dev-dependencies]
tempfile = "3.27.0"
tokio = { version = "1.52.3", features = ["rt-multi-thread", "macros", "test-util"] }
wiremock = "0.6.5"
```

- [ ] **Step 2: Write `src/config.rs`**

```rust
use std::path::PathBuf;

use clap::Parser;

#[derive(Debug, Clone, Parser)]
#[command(name = "aivyx-broker", about = "Multi-process GPU-slot scheduling broker")]
pub struct BrokerConfig {
    /// Port to bind on 127.0.0.1.
    #[arg(long, env = "AIVYX_BROKER_PORT", default_value_t = 8899)]
    pub port: u16,

    /// Base URL of the real llama-server this broker proxies to, e.g.
    /// http://127.0.0.1:8080
    #[arg(long, env = "AIVYX_BROKER_LLAMA_SERVER_URL")]
    pub llama_server_url: String,

    /// Directory for the shared kvcache store (same directory both
    /// aivyx and aivyx-coder should point their own kvcache_store_path
    /// at, per the multi-process sharing convention).
    #[arg(long, env = "AIVYX_BROKER_KVCACHE_STORE_PATH")]
    pub kvcache_store_path: PathBuf,

    /// Max bytes the kvcache store will hold before LRU eviction.
    #[arg(long, env = "AIVYX_BROKER_KVCACHE_MAX_BYTES", default_value_t = 10 * 1024 * 1024 * 1024)]
    pub kvcache_max_bytes: u64,

    /// How long a request may wait in the admission queue before it gets
    /// a clear timeout error instead of hanging.
    #[arg(long, env = "AIVYX_BROKER_QUEUE_TIMEOUT_SECS", default_value_t = 60)]
    pub queue_timeout_secs: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_required_and_default_fields() {
        let cfg = BrokerConfig::parse_from([
            "aivyx-broker",
            "--llama-server-url",
            "http://127.0.0.1:8080",
            "--kvcache-store-path",
            "/tmp/kv",
        ]);
        assert_eq!(cfg.port, 8899);
        assert_eq!(cfg.llama_server_url, "http://127.0.0.1:8080");
        assert_eq!(cfg.kvcache_store_path, PathBuf::from("/tmp/kv"));
        assert_eq!(cfg.kvcache_max_bytes, 10 * 1024 * 1024 * 1024);
        assert_eq!(cfg.queue_timeout_secs, 60);
    }

    #[test]
    fn overrides_apply() {
        let cfg = BrokerConfig::parse_from([
            "aivyx-broker",
            "--llama-server-url",
            "http://127.0.0.1:9090",
            "--kvcache-store-path",
            "/tmp/kv2",
            "--port",
            "9999",
            "--queue-timeout-secs",
            "5",
        ]);
        assert_eq!(cfg.port, 9999);
        assert_eq!(cfg.queue_timeout_secs, 5);
    }
}
```

- [ ] **Step 3: Write `src/lib.rs`** (module declarations only for now — later tasks add real content to each)

```rust
pub mod config;

pub use config::BrokerConfig;
```

- [ ] **Step 4: Write `src/main.rs`** — health-only server, just to prove the scaffold builds and binds

```rust
use axum::{routing::get, Router};
use clap::Parser;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();
    let config = aivyx_broker::BrokerConfig::parse();

    let app = Router::new().route("/status", get(status));

    let addr = format!("127.0.0.1:{}", config.port);
    tracing::info!(%addr, llama_server_url = %config.llama_server_url, "aivyx-broker starting");
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

async fn status() -> &'static str {
    "ok"
}
```

Add `anyhow = "1"` to `[dependencies]` in `Cargo.toml` for this step (used
only in `main.rs`'s `Result` return type).

- [ ] **Step 5: `cargo build` and `cargo test`**

Run: `cargo build && cargo test`
Expected: builds clean, both `config.rs` tests pass.

- [ ] **Step 6: Write `CLAUDE.md`**

```markdown
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`aivyx-broker` is a standalone local daemon that arbitrates access to a
shared `llama-server`'s finite KV-cache slots across multiple independent
local processes (e.g. `aivyx`'s daemon and a delegated `aivyx-coder`
subprocess pointed at the same `llama-server`). It sits fully in the
request path: both apps send OpenAI-compatible `/v1/chat/completions`
requests to the broker instead of to `llama-server` directly; the broker
assigns a physical slot (cache-locality-aware, via an additive
`aivyx_slot_hint` field), drives `aivyx-kvcache`'s restore/warm/save
lifecycle, forwards the request to the real `llama-server`, and streams
the response back untouched.

See `docs/superpowers/specs/2026-09-07-aivyx-broker-design.md` for the
full design rationale, including the real-code grounding (both the
client-side race and llama-server's own server-side defer-on-busy-slot
behavior, confirmed against llama.cpp's `server-context.cpp`) that shaped
this design.

## Build, run, test, lint

\`\`\`sh
cargo build
cargo test
cargo clippy --all-targets

cargo run -- --llama-server-url http://127.0.0.1:8080 --kvcache-store-path ~/.local/state/aivyx-broker/kvcache
\`\`\`

Loopback-only, no auth — same trust model as `llama-server` itself. Not
safe to expose beyond `127.0.0.1`.

## Where to look next

- `README.md` — setup and config reference.
- `docs/superpowers/specs/2026-09-07-aivyx-broker-design.md` — full design.
```

- [ ] **Step 7: Write `README.md` stub**

```markdown
# aivyx-broker

Multi-process GPU-slot scheduling broker for shared `llama-server`
deployments. Full documentation lands once the implementation is complete
(see `docs/superpowers/plans/2026-09-07-aivyx-broker.md`).
```

- [ ] **Step 8: Commit**

```bash
git add Cargo.toml src/ CLAUDE.md README.md
git commit -m "feat: scaffold aivyx-broker crate with config + health endpoint"
```

---

### Task 2: Scheduler core

**Files:**
- Create: `src/scheduler.rs`
- Modify: `src/lib.rs` (add `pub mod scheduler;` and re-export)

**Interfaces:**
- Consumes: nothing from earlier tasks beyond the module system.
- Produces: `Scheduler` — the type Task 4's HTTP handler drives directly.
  - `Scheduler::new(num_slots: u32) -> Self`
  - `async fn admit(&self, hint: Option<SlotHint>, timeout: Duration) -> Result<Admission, SchedulerError>`
  - `fn release(&self, slot_id: u32)`
  - `fn seed_busy(&self, slot_id: u32)` — used by Task 3's startup seeding to mark a slot busy-but-unowned per the spec's restart-recovery rule.
  - `SlotHint { prefix_hash: String, preferred_slot: Option<u32> }`
  - `Admission { slot_id: u32, cache_ready: bool }` — `cache_ready = true`
    means this exact `prefix_hash` was already the tracked occupant of
    `slot_id` (same-session continuation, no restore/warm/save needed by
    the caller); `false` means the caller must do the cold-start
    restore-or-warm-then-save flow before forwarding the real request.
  - `SchedulerError::Timeout`

- [ ] **Step 1: Write the failing tests** (append to `src/scheduler.rs`, written before the implementation below)

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[tokio::test]
    async fn admits_free_slot_with_no_hint() {
        let sched = Scheduler::new(2);
        let admission = sched.admit(None, Duration::from_secs(1)).await.unwrap();
        assert!(admission.slot_id < 2);
        assert!(!admission.cache_ready);
    }

    #[tokio::test]
    async fn second_admission_for_same_prefix_on_idle_slot_is_cache_ready() {
        let sched = Scheduler::new(2);
        let first = sched.admit(None, Duration::from_secs(1)).await.unwrap();
        sched.mark_resident(first.slot_id, "prefix-a".to_string());
        sched.release(first.slot_id);

        let hint = Some(SlotHint { prefix_hash: "prefix-a".to_string(), preferred_slot: None });
        let second = sched.admit(hint, Duration::from_secs(1)).await.unwrap();
        assert_eq!(second.slot_id, first.slot_id);
        assert!(second.cache_ready);
    }

    #[tokio::test]
    async fn preferred_slot_busy_queues_for_that_slot_specifically() {
        let sched = Scheduler::new(2);
        let hint_a = Some(SlotHint { prefix_hash: "a".to_string(), preferred_slot: Some(0) });
        let first = sched.admit(hint_a.clone(), Duration::from_secs(1)).await.unwrap();
        assert_eq!(first.slot_id, 0);

        // A second request preferring the same slot must wait, not steal
        // slot 1.
        let sched2 = sched.clone();
        let waiter = tokio::spawn(async move {
            sched2.admit(hint_a, Duration::from_secs(5)).await
        });

        tokio::time::sleep(Duration::from_millis(50)).await;
        sched.release(0);

        let admitted = waiter.await.unwrap().unwrap();
        assert_eq!(admitted.slot_id, 0);
    }

    #[tokio::test]
    async fn no_free_slots_and_no_preference_times_out_with_clear_error() {
        let sched = Scheduler::new(1);
        let _held = sched.admit(None, Duration::from_secs(1)).await.unwrap();

        let result = sched.admit(None, Duration::from_millis(50)).await;
        assert!(matches!(result, Err(SchedulerError::Timeout)));
    }

    #[tokio::test]
    async fn seed_busy_marks_slot_unavailable_until_explicitly_released() {
        let sched = Scheduler::new(1);
        sched.seed_busy(0);

        let result = sched.admit(None, Duration::from_millis(50)).await;
        assert!(matches!(result, Err(SchedulerError::Timeout)));

        sched.release(0);
        let admission = sched.admit(None, Duration::from_secs(1)).await.unwrap();
        assert_eq!(admission.slot_id, 0);
    }
}
```

- [ ] **Step 2: Run the tests to verify they fail to compile** (the types don't exist yet)

Run: `cargo test -p aivyx-broker scheduler`
Expected: compile error, `Scheduler`/`SlotHint`/etc. not found.

- [ ] **Step 3: Implement `Scheduler`**

```rust
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use tokio::sync::oneshot;

#[derive(Debug, Clone)]
pub struct SlotHint {
    pub prefix_hash: String,
    pub preferred_slot: Option<u32>,
}

#[derive(Debug, Clone, Copy)]
pub struct Admission {
    pub slot_id: u32,
    pub cache_ready: bool,
}

#[derive(Debug, thiserror::Error)]
pub enum SchedulerError {
    #[error("timed out waiting for a free slot")]
    Timeout,
}

#[derive(Debug, Default)]
struct SlotState {
    busy: bool,
    resident_prefix: Option<String>,
}

struct Inner {
    slots: Vec<SlotState>,
    /// FIFO waiters for a specific slot id.
    slot_waiters: HashMap<u32, VecDeque<oneshot::Sender<u32>>>,
    /// FIFO waiters for "any free slot".
    global_waiters: VecDeque<oneshot::Sender<u32>>,
}

/// In-memory, in-process (single `aivyx-broker` instance) slot admission
/// scheduler. Cheap to clone -- all state lives behind the shared `Arc`.
/// See `docs/superpowers/specs/2026-09-07-aivyx-broker-design.md`
/// "Scheduling & failure handling" for the admission rules this
/// implements.
#[derive(Clone)]
pub struct Scheduler {
    inner: Arc<Mutex<Inner>>,
}

impl Scheduler {
    pub fn new(num_slots: u32) -> Self {
        let slots = (0..num_slots).map(|_| SlotState::default()).collect();
        Self {
            inner: Arc::new(Mutex::new(Inner {
                slots,
                slot_waiters: HashMap::new(),
                global_waiters: VecDeque::new(),
            })),
        }
    }

    /// Marks `slot_id` busy with no known resident prefix -- used at
    /// startup to reflect a slot `llama-server` reports mid-generation
    /// across a broker restart (see spec's "Broker startup/restart").
    pub fn seed_busy(&self, slot_id: u32) {
        let mut inner = self.inner.lock().unwrap();
        if let Some(slot) = inner.slots.get_mut(slot_id as usize) {
            slot.busy = true;
        }
    }

    /// Records that `slot_id` now holds `prefix_hash`'s content -- called
    /// by the HTTP handler after a successful restore/warm-then-save, or
    /// on a fresh admission before forwarding the real request.
    pub fn mark_resident(&self, slot_id: u32, prefix_hash: String) {
        let mut inner = self.inner.lock().unwrap();
        if let Some(slot) = inner.slots.get_mut(slot_id as usize) {
            slot.resident_prefix = Some(prefix_hash);
        }
    }

    pub async fn admit(
        &self,
        hint: Option<SlotHint>,
        timeout: Duration,
    ) -> Result<Admission, SchedulerError> {
        let rx = {
            let mut inner = self.inner.lock().unwrap();
            match Self::try_admit_locked(&mut inner, &hint) {
                Some(admission) => return Ok(admission),
                None => {
                    let (tx, rx) = oneshot::channel();
                    match hint.as_ref().and_then(|h| h.preferred_slot) {
                        Some(slot_id) => {
                            inner.slot_waiters.entry(slot_id).or_default().push_back(tx)
                        }
                        None => inner.global_waiters.push_back(tx),
                    }
                    rx
                }
            }
        };

        let prefix_hash = hint.map(|h| h.prefix_hash);
        match tokio::time::timeout(timeout, rx).await {
            Ok(Ok(slot_id)) => {
                let cache_ready = {
                    let inner = self.inner.lock().unwrap();
                    prefix_hash.as_deref() == inner.slots[slot_id as usize].resident_prefix.as_deref()
                };
                Ok(Admission { slot_id, cache_ready })
            }
            Ok(Err(_)) | Err(_) => Err(SchedulerError::Timeout),
        }
    }

    /// Attempts admission immediately, without waiting. Marks the chosen
    /// slot busy on success. Returns `None` if nothing is admittable now.
    fn try_admit_locked(inner: &mut Inner, hint: &Option<SlotHint>) -> Option<Admission> {
        if let Some(h) = hint {
            if let Some(preferred) = h.preferred_slot {
                let slot = inner.slots.get_mut(preferred as usize)?;
                if !slot.busy {
                    slot.busy = true;
                    let cache_ready = slot.resident_prefix.as_deref() == Some(h.prefix_hash.as_str());
                    return Some(Admission { slot_id: preferred, cache_ready });
                }
                return None; // must queue for this specific slot
            }
        }
        // No preference: any free slot, prefer one already resident with
        // this exact prefix, else LRU-first among idle (first free found).
        if let Some(h) = hint {
            if let Some((idx, slot)) = inner
                .slots
                .iter_mut()
                .enumerate()
                .find(|(_, s)| !s.busy && s.resident_prefix.as_deref() == Some(h.prefix_hash.as_str()))
            {
                slot.busy = true;
                return Some(Admission { slot_id: idx as u32, cache_ready: true });
            }
        }
        let (idx, slot) = inner.slots.iter_mut().enumerate().find(|(_, s)| !s.busy)?;
        slot.busy = true;
        let cache_ready = hint
            .as_ref()
            .is_some_and(|h| slot.resident_prefix.as_deref() == Some(h.prefix_hash.as_str()));
        Some(Admission { slot_id: idx as u32, cache_ready })
    }

    pub fn release(&self, slot_id: u32) {
        let mut inner = self.inner.lock().unwrap();
        if let Some(slot) = inner.slots.get_mut(slot_id as usize) {
            slot.busy = false;
        } else {
            return;
        }

        // Wake this slot's specific waiters first (rule: don't let a
        // global waiter steal a slot someone is specifically waiting on).
        if let Some(queue) = inner.slot_waiters.get_mut(&slot_id) {
            if let Some(tx) = queue.pop_front() {
                inner.slots[slot_id as usize].busy = true;
                let _ = tx.send(slot_id);
                return;
            }
        }
        if let Some(tx) = inner.global_waiters.pop_front() {
            inner.slots[slot_id as usize].busy = true;
            let _ = tx.send(slot_id);
        }
    }
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test -p aivyx-broker scheduler`
Expected: all 5 tests pass.

- [ ] **Step 5: Wire into `src/lib.rs`**

```rust
pub mod config;
pub mod scheduler;

pub use config::BrokerConfig;
pub use scheduler::{Admission, Scheduler, SchedulerError, SlotHint};
```

- [ ] **Step 6: `cargo clippy --all-targets` clean, then commit**

```bash
cargo clippy --all-targets
git add src/scheduler.rs src/lib.rs
git commit -m "feat: add in-memory FIFO slot scheduler core"
```

---

### Task 3: llama-server `/slots` client + startup seeding

**Files:**
- Create: `src/llama_client.rs`
- Modify: `src/lib.rs` (add module)
- Modify: `src/main.rs` (seed the scheduler from real `/slots` at startup)

**Interfaces:**
- Consumes: `Scheduler::new`, `Scheduler::seed_busy` (Task 2).
- Produces: `fetch_slots(client: &reqwest::Client, base_url: &str) -> Result<Vec<LlamaSlot>, LlamaClientError>`, `LlamaSlot { id: u32, is_processing: bool }`, `seed_scheduler_from_llama_server(scheduler: &Scheduler, client: &reqwest::Client, base_url: &str) -> Result<u32, LlamaClientError>` (returns the real slot count, used by `main.rs` to construct the `Scheduler` — note this means `Scheduler::new` in `main.rs` must be constructed *after* this call, not before, unlike the standalone-scheduler tests in Task 2).

Real `GET /slots` response shape (confirmed against llama.cpp's own
`tools/server/README.md`): a JSON array of objects, each with (at least)
`"id": <u32>` and `"is_processing": <bool>` — other fields exist but are
not needed here.

- [ ] **Step 1: Write the failing tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[test]
    fn parses_real_slots_response_shape() {
        let body = serde_json::json!([
            {"id": 0, "id_task": 135, "is_processing": true, "n_ctx": 65536},
            {"id": 1, "id_task": -1, "is_processing": false, "n_ctx": 65536},
        ]);
        let slots = parse_slots(&body).expect("must parse a real llama-server /slots body");
        assert_eq!(slots.len(), 2);
        assert_eq!(slots[0], LlamaSlot { id: 0, is_processing: true });
        assert_eq!(slots[1], LlamaSlot { id: 1, is_processing: false });
    }

    #[tokio::test]
    async fn fetch_slots_hits_the_real_endpoint_path() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/slots"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!([
                {"id": 0, "is_processing": false},
            ])))
            .mount(&server)
            .await;

        let client = reqwest::Client::new();
        let slots = fetch_slots(&client, &server.uri()).await.unwrap();
        assert_eq!(slots, vec![LlamaSlot { id: 0, is_processing: false }]);
    }

    #[tokio::test]
    async fn seed_scheduler_marks_processing_slots_busy() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/slots"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!([
                {"id": 0, "is_processing": true},
                {"id": 1, "is_processing": false},
            ])))
            .mount(&server)
            .await;

        let client = reqwest::Client::new();
        let scheduler = crate::Scheduler::new(2);
        let count = seed_scheduler_from_llama_server(&scheduler, &client, &server.uri())
            .await
            .unwrap();
        assert_eq!(count, 2);

        // Slot 0 must be unavailable (busy-but-unowned); slot 1 must be
        // immediately admittable.
        let admission = scheduler.admit(None, std::time::Duration::from_millis(50)).await.unwrap();
        assert_eq!(admission.slot_id, 1);
    }
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo test -p aivyx-broker llama_client`
Expected: compile error (types don't exist yet).

- [ ] **Step 3: Implement `src/llama_client.rs`**

```rust
use serde::Deserialize;

use crate::Scheduler;

#[derive(Debug, thiserror::Error)]
pub enum LlamaClientError {
    #[error("request to llama-server failed: {0}")]
    Request(#[from] reqwest::Error),
    #[error("failed to parse llama-server /slots response")]
    Parse,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
pub struct LlamaSlot {
    pub id: u32,
    pub is_processing: bool,
}

pub(crate) fn parse_slots(json: &serde_json::Value) -> Option<Vec<LlamaSlot>> {
    let arr = json.as_array()?;
    arr.iter()
        .map(|v| {
            Some(LlamaSlot {
                id: v.get("id")?.as_u64()? as u32,
                is_processing: v.get("is_processing")?.as_bool()?,
            })
        })
        .collect()
}

pub async fn fetch_slots(
    client: &reqwest::Client,
    base_url: &str,
) -> Result<Vec<LlamaSlot>, LlamaClientError> {
    let url = format!("{}/slots", base_url.trim_end_matches('/'));
    let resp = client.get(&url).send().await?.error_for_status()?;
    let json: serde_json::Value = resp.json().await?;
    parse_slots(&json).ok_or(LlamaClientError::Parse)
}

/// Fetches the real slot count/state from `llama-server` and seeds
/// `scheduler` accordingly (any slot reported `is_processing: true` is
/// marked busy-but-unowned, per the spec's restart-recovery rule).
/// Returns the real slot count so the caller can construct the
/// `Scheduler` with the right size before this is called for the very
/// first time -- see `main.rs`.
pub async fn seed_scheduler_from_llama_server(
    scheduler: &Scheduler,
    client: &reqwest::Client,
    base_url: &str,
) -> Result<u32, LlamaClientError> {
    let slots = fetch_slots(client, base_url).await?;
    for slot in &slots {
        if slot.is_processing {
            scheduler.seed_busy(slot.id);
        }
    }
    Ok(slots.len() as u32)
}
```

- [ ] **Step 4: Run to verify pass**

Run: `cargo test -p aivyx-broker llama_client`
Expected: all 3 tests pass.

- [ ] **Step 5: Wire real startup seeding into `src/main.rs`**

Replace the placeholder health-only body with:

```rust
use axum::{routing::get, Router};
use clap::Parser;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();
    let config = aivyx_broker::BrokerConfig::parse();
    let http_client = reqwest::Client::new();

    // Fetch the real slot count first -- the Scheduler is constructed
    // once we know it, then seeded from the same response.
    let slots = aivyx_broker::llama_client::fetch_slots(&http_client, &config.llama_server_url)
        .await
        .map_err(|e| anyhow::anyhow!("failed to reach llama-server at startup: {e}"))?;
    let scheduler = aivyx_broker::Scheduler::new(slots.len() as u32);
    for slot in &slots {
        if slot.is_processing {
            scheduler.seed_busy(slot.id);
        }
    }
    tracing::info!(num_slots = slots.len(), "seeded scheduler from real llama-server /slots");

    let app = Router::new().route("/status", get(status));

    let addr = format!("127.0.0.1:{}", config.port);
    tracing::info!(%addr, llama_server_url = %config.llama_server_url, "aivyx-broker starting");
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

async fn status() -> &'static str {
    "ok"
}
```

- [ ] **Step 6: Update `src/lib.rs`**

```rust
pub mod config;
pub mod llama_client;
pub mod scheduler;

pub use config::BrokerConfig;
pub use scheduler::{Admission, Scheduler, SchedulerError, SlotHint};
```

- [ ] **Step 7: `cargo build && cargo test && cargo clippy --all-targets`, then commit**

```bash
git add src/llama_client.rs src/lib.rs src/main.rs
git commit -m "feat: fetch real llama-server /slots and seed scheduler at startup"
```

---

### Task 4: HTTP server — `/v1/chat/completions`, `/status`, `/slots`

This is the largest task: it wires the scheduler (Task 2), the llama-server
client (Task 3), and `aivyx-kvcache`'s `LlamaServerSlotStore` together
behind the real Axum routes.

**Files:**
- Create: `src/server.rs`
- Modify: `src/lib.rs` (add module, `AppState`)
- Modify: `src/main.rs` (build `AppState`, mount real routes)

**Interfaces:**
- Consumes: `Scheduler` (Task 2), `fetch_slots`/`LlamaSlot` (Task 3),
  `aivyx_kvcache::{LlamaServerSlotStore, CacheKey, CacheMeta, KvCacheStore}`
  (external dependency — `LlamaServerSlotStore::open`,
  `restore_into_slot(&self, key: &CacheKey, slot_id: u32) -> Result<bool, KvCacheError>`,
  `save_from_slot(&self, key: &CacheKey, slot_id: u32, meta: CacheMeta) -> Result<(), KvCacheError>`
  — confirmed exact signatures against `aivyx-kvcache/src/llama_server.rs`
  in this project's own grounding; re-read that file if anything here
  looks off before assuming this plan is stale).
- Produces: `AppState { scheduler: Scheduler, http: reqwest::Client, llama_server_url: String, kv_store: Arc<LlamaServerSlotStore>, queue_timeout: Duration }`, `pub fn build_router(state: AppState) -> Router`.

**Algorithm for `POST /v1/chat/completions`** (implements the spec's
admission semantics plus the restore/warm/save lifecycle correction — see
spec's "Architecture" and "Client integration" sections):

1. Parse the incoming JSON body. Extract the optional `aivyx_slot_hint`
   field (`{prefix_hash, preferred_slot}`) and remove it from the body
   before forwarding (llama-server doesn't know this field).
2. Call `scheduler.admit(hint, queue_timeout)`. On `SchedulerError::Timeout`,
   respond `503` with a clear JSON error body (`{"error": "..."}"`) — never
   hang silently.
3. If `admission.cache_ready` is `true`: forward the client's real request
   body as-is, pinned to `admission.slot_id` via the `id_slot` field,
   directly to `llama-server`, streaming the response back. Skip steps 4–5
   entirely.
4. If `admission.cache_ready` is `false` (cold path) — build a `CacheKey`
   from the hint's `prefix_hash` plus fixed `backend_id`/`model_id`/
   `build_hash` strings (the broker uses constant values here, e.g.
   `backend_id: "aivyx-broker"`, since — unlike each client app — it
   proxies for exactly one `llama-server` at a time, so there's no need to
   disambiguate multiple backends the way each client's own `CacheKey`
   construction does):
   - Call `kv_store.restore_into_slot(&key, admission.slot_id)`.
   - If it returns `Ok(true)` (hit): the slot's stable prefix is now
     loaded from disk. Call `scheduler.mark_resident(slot_id, prefix_hash)`.
   - If it returns `Ok(false)` or an error (miss/failure — log a `warn` on
     error, same fail-open posture as the existing client-side code):
     extract just the system message + `tools` array from the client's
     real request body, build a synthetic warm-up request
     (`{messages: [that system message, {"role": "user", "content": ""}],
     max_tokens: 1, id_slot: admission.slot_id}`), `POST` it to
     `llama-server`, drain the response. Then call
     `kv_store.save_from_slot(&key, admission.slot_id, CacheMeta { size_bytes: 1, token_count: 1 })`
     (log a `warn` on failure, don't fail the real request over it). Then
     call `scheduler.mark_resident(slot_id, prefix_hash)`.
5. Forward the client's real request body (with `id_slot` set to
   `admission.slot_id`, `aivyx_slot_hint` stripped) to `llama-server`,
   streaming the response back via `reqwest::Response::bytes_stream()`
   wrapped into an Axum `Body` — this is a direct, already-`'static`
   stream (unlike mistral.rs's borrowed in-process stream), so no
   mpsc-forwarding shim is needed here.
6. Whether step 3 or steps 4–5 ran: after the streamed response completes
   (success or error), call `scheduler.release(admission.slot_id)`. Use a
   guard/drop pattern so a client disconnect mid-stream still releases the
   slot.

`GET /status` returns `{llama_server_url, queue_depth: 0}` (queue depth
tracking beyond "0 or nonzero" is explicitly out of scope for this task —
YAGNI; add only if a later task's spec revision asks for it).

`GET /slots` returns the broker's own occupancy view: an array of
`{slot_id, busy, resident_prefix}` built from a new `Scheduler::snapshot(&self) -> Vec<SlotSnapshot>` method — add this small method to `src/scheduler.rs` as part of this task (`SlotSnapshot { slot_id: u32, busy: bool, resident_prefix: Option<String> }`, reading the same `Inner` fields `try_admit_locked` already uses).

- [ ] **Step 1: Add `Scheduler::snapshot` + its test to `src/scheduler.rs`**

```rust
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SlotSnapshot {
    pub slot_id: u32,
    pub busy: bool,
    pub resident_prefix: Option<String>,
}

impl Scheduler {
    pub fn snapshot(&self) -> Vec<SlotSnapshot> {
        let inner = self.inner.lock().unwrap();
        inner
            .slots
            .iter()
            .enumerate()
            .map(|(idx, s)| SlotSnapshot {
                slot_id: idx as u32,
                busy: s.busy,
                resident_prefix: s.resident_prefix.clone(),
            })
            .collect()
    }
}
```

Test (append to `scheduler.rs`'s existing test module):

```rust
#[tokio::test]
async fn snapshot_reflects_busy_and_resident_state() {
    let sched = Scheduler::new(2);
    let admission = sched.admit(None, Duration::from_secs(1)).await.unwrap();
    sched.mark_resident(admission.slot_id, "p".to_string());

    let snap = sched.snapshot();
    let this_slot = snap.iter().find(|s| s.slot_id == admission.slot_id).unwrap();
    assert!(this_slot.busy);
    assert_eq!(this_slot.resident_prefix.as_deref(), Some("p"));
}
```

Run: `cargo test -p aivyx-broker scheduler` — expect the new test to pass
alongside the existing five.

- [ ] **Step 2: Write the failing integration tests in `src/server.rs`**

```rust
use std::sync::Arc;
use std::time::Duration;

use axum::Router;
use axum::routing::{get, post};

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

// (handler stubs below get replaced by Step 3's real implementation)

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
```

Add `tower = { version = "0.5", features = ["util"] }` to `[dev-dependencies]`
in `Cargo.toml` for `ServiceExt::oneshot` in these tests.

- [ ] **Step 3: Run to verify failure**

Run: `cargo test -p aivyx-broker server`
Expected: compile errors (`chat_completions`/`status`/`slots` handlers
don't exist yet).

- [ ] **Step 4: Implement the real handlers in `src/server.rs`** (replace the stub comment from Step 2 with this)

```rust
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::Json;
use serde_json::Value;

async fn status(State(state): State<AppState>) -> impl IntoResponse {
    Json(serde_json::json!({ "llama_server_url": state.llama_server_url }))
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

    let result = handle_admitted(&state, &mut body, &hint, admission).await;
    state.scheduler.release(admission.slot_id);

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

async fn handle_admitted(
    state: &AppState,
    body: &mut Value,
    hint: &Option<crate::SlotHint>,
    admission: crate::Admission,
) -> Result<axum::response::Response, anyhow::Error> {
    if !admission.cache_ready {
        if let Some(h) = hint {
            warm_or_restore(state, body, h, admission.slot_id).await?;
        }
    }

    if let Some(obj) = body.as_object_mut() {
        obj.insert("id_slot".to_string(), serde_json::json!(admission.slot_id));
    }

    let resp = state
        .http
        .post(format!("{}/v1/chat/completions", state.llama_server_url.trim_end_matches('/')))
        .json(body)
        .send()
        .await?
        .error_for_status()?;

    let axum_body = axum::body::Body::from_stream(resp.bytes_stream());
    Ok(axum::response::Response::builder()
        .status(StatusCode::OK)
        .body(axum_body)?)
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

        let warm_up = serde_json::json!({
            "messages": [system_message, {"role": "user", "content": ""}],
            "max_tokens": 1,
            "id_slot": slot_id,
        });
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
```

Also derive `serde::{Serialize, Deserialize}` on `SlotHint` and
`serde::Serialize` on `SlotSnapshot` in `src/scheduler.rs` (needed for the
`Json` extractor and the `/slots` response above) — add
`#[derive(serde::Serialize, serde::Deserialize)]` to `SlotHint` and
`#[derive(serde::Serialize)]` to `SlotSnapshot`.

- [ ] **Step 5: Run to verify pass**

Run: `cargo test -p aivyx-broker server`
Expected: all 4 tests pass.

- [ ] **Step 6: Wire real routes into `src/main.rs`**, replacing the
  `/status`-only router:

```rust
let kv_store = aivyx_kvcache::LlamaServerSlotStore::open(
    &config.kvcache_store_path,
    config.llama_server_url.clone(),
    config.kvcache_max_bytes,
)?;
let state = aivyx_broker::server::AppState {
    scheduler,
    http: http_client,
    llama_server_url: config.llama_server_url.clone(),
    kv_store: std::sync::Arc::new(kv_store),
    queue_timeout: std::time::Duration::from_secs(config.queue_timeout_secs),
};
let app = aivyx_broker::server::build_router(state);
```

(Remove the old `status` fn and `get`-only `Router::new()` line from
`main.rs` — `build_router` replaces both.)

- [ ] **Step 7: Update `src/lib.rs`**

```rust
pub mod config;
pub mod llama_client;
pub mod scheduler;
pub mod server;

pub use config::BrokerConfig;
pub use scheduler::{Admission, Scheduler, SchedulerError, SlotHint, SlotSnapshot};
```

- [ ] **Step 8: `cargo build && cargo test && cargo clippy --all-targets`, then commit**

```bash
git add src/server.rs src/scheduler.rs src/lib.rs src/main.rs Cargo.toml
git commit -m "feat: wire scheduler + kvcache restore/warm/save into HTTP routes"
```

---

### Task 5: README.md + CLAUDE.md finalization

**Files:**
- Modify: `README.md` (replace stub with full docs)
- Modify: `CLAUDE.md` (already mostly complete from Task 1 — add a
  "Known limitations" section)

**Interfaces:** none — documentation only.

- [ ] **Step 1: Write the full `README.md`**

```markdown
# aivyx-broker

A standalone local daemon that arbitrates access to a shared
`llama-server`'s finite KV-cache slots across multiple independent local
processes — for example `aivyx`'s daemon and a delegated `aivyx-coder`
subprocess both pointed at the same GPU-backed `llama-server`.

## Why this exists

Both `aivyx` and `aivyx-coder` can be configured to talk to the same
`llama-server` (see each project's own kvcache-sharing docs). Each picks a
physical KV-cache slot for its requests via its own private, in-process
tracker, with zero awareness of the other process. Confirmed directly
against real code: two such processes will independently pick the same
slot id first, and while `llama-server` itself safely defers rather than
corrupting state on a same-slot collision (confirmed against llama.cpp's
own `server-context.cpp`), the result is silent head-of-line blocking and
KV-cache-locality thrash — not corruption, but a real, invisible
performance problem. See
`docs/superpowers/specs/2026-09-07-aivyx-broker-design.md` for the full
grounding and design rationale.

`aivyx-broker` sits between both apps and the real `llama-server`: it owns
live slot admission (cache-locality-aware, via an additive
`aivyx_slot_hint` field) and the KV-cache restore/warm/save lifecycle,
then forwards the actual completion to `llama-server` and streams the
response back untouched.

## Running

\`\`\`sh
cargo build --release
./target/release/aivyx-broker \
  --llama-server-url http://127.0.0.1:8080 \
  --kvcache-store-path ~/.local/state/aivyx-broker/kvcache
\`\`\`

Binds `127.0.0.1:8899` by default (`--port` to change). Loopback-only, no
auth — same trust model as `llama-server` itself; not safe to expose
beyond localhost.

Start it the same way you'd start `llama-server` itself: manually, before
the client apps that will use it. There is no auto-spawn in v1 — if the
broker isn't running, both client apps will simply see a connection-refused
error against its `base_url`, the same shape as `llama-server` being down.

## Configuration reference

| Flag | Env var | Default | Meaning |
|---|---|---|---|
| `--port` | `AIVYX_BROKER_PORT` | `8899` | Loopback port to bind |
| `--llama-server-url` | `AIVYX_BROKER_LLAMA_SERVER_URL` | *(required)* | The real `llama-server` this broker proxies to |
| `--kvcache-store-path` | `AIVYX_BROKER_KVCACHE_STORE_PATH` | *(required)* | Shared kvcache store directory — point both `aivyx` and `aivyx-coder`'s own `kvcache_store_path` at the same directory |
| `--kvcache-max-bytes` | `AIVYX_BROKER_KVCACHE_MAX_BYTES` | 10 GiB | LRU eviction budget for the kvcache store |
| `--queue-timeout-secs` | `AIVYX_BROKER_QUEUE_TIMEOUT_SECS` | `60` | How long a request may wait for a free slot before getting a clear `503` instead of hanging |

## Pointing a client at the broker

Both `aivyx` and `aivyx-coder` need a broker-aware backend mode to use
this — see each repo's own README for its own config once that work
lands. In short: point the client's `base_url` at the broker instead of
`llama-server` directly, and enable its broker mode so it stops doing its
own local slot-picking and kvcache restore/save (the broker now owns
that).

## API

- `POST /v1/chat/completions` — OpenAI-compatible, identical to
  `llama-server`'s own endpoint, plus an optional `aivyx_slot_hint` field:
  `{"prefix_hash": "...", "preferred_slot": <u32 or null>}`. Omit it for
  plain "any free slot" behavior.
- `GET /status` — broker health + which `llama-server` it's proxying to.
- `GET /slots` — the broker's own occupancy view (superset of
  `llama-server`'s own `/slots`, adding cross-process attribution
  `llama-server` has no notion of).

## Honest tradeoffs

- **A new required-up dependency.** Once a setup opts into the broker, it
  becomes as load-bearing as `llama-server` itself — if it's down, both
  client apps see connection-refused. This is an accepted cost of
  centralizing coordination, not a bug.
- **No priority/weighted scheduling in v1.** Pure FIFO by arrival order.
  Deferred as a future increment.
- **No real multi-process race test exists.** This project's own test
  suite has no way to run two genuine OS processes contending for one
  genuine `llama-server` — verified with fakes/mocks only. Manual
  verification against a real rig is a documented follow-up, not a merge
  blocker.
```

- [ ] **Step 2: Add a "Known limitations" section to `CLAUDE.md`**

```markdown

## Known, deliberately-undefended limitations

- Loopback-only, no auth. Not safe to expose beyond `127.0.0.1`.
- No priority/weighted scheduling — pure FIFO in v1.
- No persisted broker state across restarts (by design — see the spec's
  "Broker startup/restart" section); a restart loses cross-process
  fairness bookkeeping but never desyncs from `llama-server`'s own
  physical reality.
- No automatic lifecycle management (auto-spawn, systemd unit
  generation) — started manually, same as `llama-server` itself.
```

- [ ] **Step 3: Commit**

```bash
git add README.md CLAUDE.md
git commit -m "docs: write full README and CLAUDE.md for aivyx-broker"
```

---

## Part B — client integration in `aivyx`

**Repo:** `/home/julian/Projects/Rust/aivyx` — a separate git repo. Branch
before starting (see `superpowers:using-git-worktrees`); do not commit to
this repo's default branch directly.

### Task 6: `ProviderKind::Broker` in `aivyx`

**Files:**
- Modify: `crates/aivyx-config/src/lib.rs` (new `ProviderKind` variant +
  config fields, mirroring the existing `ProviderKind::MistralRs` pattern
  at line ~411)
- Modify: `crates/aivyx-cli/src/bin/aivyx.rs` (dispatch, mirroring the
  existing `ProviderKind::LlamaCpp` construction around line ~6291 —
  ground the exact current line range fresh before editing; this plan's
  earlier reads may have drifted)
- Modify: `crates/aivyx-core/src/llm_planner.rs` (skip local
  `KvSlotPool::checkout()`/restore/save when broker mode is active; attach
  `slot_hint` to outgoing requests instead of `id_slot` directly)
- Modify: `crates/aivyx-llm/src/*` — wherever the `ChatRequest`-equivalent
  struct with `id_slot: Option<u32>` lives (ground its exact current
  location and field name fresh — this plan's earlier grounding found
  `id_slot: Some(slot_id)` used at `llm_planner.rs` lines ~820/946 but did
  not pin the struct's own definition site) — add an additive
  `slot_hint: Option<SlotHint>` field alongside it, where
  `SlotHint { prefix_hash: String, preferred_slot: Option<u32> }` is a new
  small struct (serialized as the `aivyx_slot_hint` JSON key by whatever
  serializes the outgoing HTTP body today — likely the same file that
  already serializes `id_slot`).
- Test: alongside each modified file, following that file's existing test
  conventions.

**Interfaces:**
- Consumes: nothing new from `aivyx-broker` itself (it's a separate
  process reached over HTTP) — only needs `aivyx-broker`'s documented
  wire contract: `POST /v1/chat/completions` with an optional
  `aivyx_slot_hint: {prefix_hash, preferred_slot}` JSON field, response
  identical in shape to a plain OpenAI-compatible completion.
- Produces: nothing consumed by a later task in this plan — this is the
  last task touching `aivyx`.

- [ ] **Step 1: Ground the exact current state before editing**

Before writing any code, confirm (things may have drifted since this
plan's own grounding pass):
- The exact line range of `ProviderKind::MistralRs =>` in
  `crates/aivyx-cli/src/bin/aivyx.rs` (grep for it) — this is the pattern
  to mirror for the new `Broker` variant's own dispatch arm.
- The exact struct name and field list of the `ChatRequest`-equivalent
  type carrying `id_slot` in `aivyx-llm` (grep `id_slot: Option` across
  `crates/aivyx-llm/src/`).
- The exact current body of `ensure_kv_slot_checked_out`'s equivalent in
  `llm_planner.rs` (grep `KvSlotPool` / `.pool.checkout()` — this plan's
  own earlier grounding read real call sites around lines ~738–946 and
  ~1633, but confirm they still match before editing).

- [ ] **Step 2: Add `ProviderKind::Broker` + config fields**

In `crates/aivyx-config/src/lib.rs`, following the exact shape of the
existing `MistralRs` variant (serde aliases, the `is_in_process`-style
helper if the new variant needs an equivalent — it does not, since unlike
`MistralRs` this is still an HTTP backend, not in-process):

```rust
#[serde(alias = "broker", alias = "aivyx-broker", alias = "aivyx_broker")]
Broker,
```

Add a `broker_base_url: Option<String>` field to whichever settings struct
holds `mistralrs_options`-style per-provider config (mirror its placement
and doc-comment style exactly).

- [ ] **Step 3: Write config round-trip tests**

Follow the exact pattern of this file's own existing `ProviderKind`
serde tests (grep `ProviderKind::MistralRs` in this file's `#[cfg(test)]`
module for the pattern to copy) — one test parsing `"broker"` from TOML,
one confirming `"aivyx-broker"`/`"aivyx_broker"` aliases also parse.

- [ ] **Step 4: Wire dispatch in `aivyx.rs`**

Mirror the `ProviderKind::LlamaCpp` branch's `OpenAiProvider`/
`OpenAiConfig` construction exactly (the broker speaks the identical
OpenAI-compatible wire protocol — same struct, just a different
`base_url`, pointed at `broker_base_url` instead of the real
`llama-server`'s own address).

- [ ] **Step 5: Skip local slot-picking/persistence when broker mode is active**

In `llm_planner.rs`, guard the existing `kv.pool.checkout()` /
`restore_into_slot` / `save_from_slot` call sites so they never run when
`ProviderKind::Broker` is active (the broker owns that lifecycle now — see
`aivyx-broker`'s own design spec, "Architecture" section, for why). Attach
`slot_hint: Some(SlotHint { prefix_hash: compute_prefix_hash(...), preferred_slot: self.kv_slot_id })`
to the outgoing request instead of the direct `id_slot: Some(slot_id)`
pinning those call sites did before.

- [ ] **Step 6: Write a test proving the skip actually happens**

A test asserting that, with `ProviderKind::Broker` configured, no call to
`KvSlotPool::checkout()` occurs (e.g. via a pool with 0 slots that would
otherwise force a `None`/warn path — assert no such warning fires, or
assert the pool's checked-out set stays empty after a full turn) and that
the outgoing request carries a `slot_hint` rather than a raw `id_slot`.

- [ ] **Step 7: Run the full workspace suite and clippy**

Run: `cargo test --workspace && cargo clippy --workspace --all-targets`
Expected: all tests pass, clippy clean.

- [ ] **Step 8: Update docs**

Find wherever this repo documents its provider list today (grep for
`ProviderKind::MistralRs` or `"mistralrs"` across `docs/` and `README.md`
to locate it — likely a config reference table or a dedicated serving
guide) and add a `Broker` row/section following the same structure,
pointing at `aivyx-broker`'s own repo/README for setup.

- [ ] **Step 9: Commit**

```bash
git add crates/aivyx-config/src/lib.rs crates/aivyx-cli/src/bin/aivyx.rs crates/aivyx-core/src/llm_planner.rs crates/aivyx-llm/src/
git commit -m "feat: add ProviderKind::Broker for aivyx-broker multi-process slot coordination"
```

---

## Part C — client integration in `aivyx-coder`

**Repo:** `/home/julian/Projects/Rust/aivyx-coder` — a separate git repo.
Branch before starting; do not commit to this repo's default branch
directly.

### Task 7: `BackendKind::LlamaServerBroker` in `aivyx-coder`

**Files:**
- Modify: `crates/aivyx-config/src/lib.rs` (new `BackendKind` variant,
  mirroring `BackendKind::MistralRs` at line ~640; new
  `broker_base_url: Option<String>` field on `BackendSettings`, near
  `kvcache_store_path` at line ~597)
- Modify: `crates/aivyx/src/agent_builder.rs` (dispatch — mirror the
  existing `BackendKind::Generic | BackendKind::LlamaServer =>` arm at
  line ~101, since the broker is still a plain `OpenAiCompatBackend`
  pointed at a different `base_url`; also gate the `kv_cache_handles`
  construction at line ~578, which today only fires for
  `BackendKind::LlamaServer`, so it does **not** fire for
  `BackendKind::LlamaServerBroker` — the broker owns that lifecycle now)
- Modify: `crates/aivyx-core/src/agent/mod.rs` (`ensure_kv_slot_checked_out`,
  lines ~472–594 per this plan's own grounding — early-return when
  `BackendKind::LlamaServerBroker` is active, attach a `slot_hint` to the
  outgoing `ChatRequest` instead of pinning `id_slot` directly)
- Modify: `crates/aivyx-llm/src/backend.rs` (or wherever `ChatRequest` is
  defined — same struct this plan already confirmed has
  `id_slot: Option<u32>`, add an additive `slot_hint: Option<SlotHint>`
  field; `SlotHint { prefix_hash: String, preferred_slot: Option<u32> }`)
- Modify: `crates/aivyx-llm/src/openai/provider.rs` (or wherever
  `OpenAiCompatBackend` serializes `id_slot` into the outgoing JSON body —
  serialize `slot_hint` as the `aivyx_slot_hint` key alongside it, only
  when `Some`)
- Test: alongside each modified file, following that file's existing test
  conventions.

**Interfaces:**
- Consumes: `aivyx-broker`'s documented wire contract only (a separate
  process, reached over HTTP) — identical contract Task 6 consumed.
- Produces: nothing consumed by a later task — last task in this plan.

- [ ] **Step 1: Ground the exact current state before editing**

This plan's own earlier grounding read `agent_builder.rs`'s dispatch (lines
~99–130), `BackendSettings` (lines ~559–605+), and
`ensure_kv_slot_checked_out`'s full body (lines ~472–594) directly — but
re-confirm line numbers with a fresh grep before editing, since other work
may have landed on this repo's default branch since this plan was written.

- [ ] **Step 2: Add `BackendKind::LlamaServerBroker` + `broker_base_url`**

```rust
pub enum BackendKind {
    #[default]
    Generic,
    LlamaServer,
    LlamaServerBroker,
    MistralRs,
}
```

Add `pub broker_base_url: Option<String>` to `BackendSettings`, with a doc
comment mirroring `kvcache_store_path`'s own style, explaining: required
when `kind = "llama_server_broker"`; this replaces `base_url` as the
address the `OpenAiCompatBackend` actually connects to (llama-server's own
`base_url` is no longer contacted directly by this process on this path —
see `aivyx-broker`'s own README for why).

- [ ] **Step 3: Write config tests**

Mirror this file's existing `BackendKind::MistralRs` round-trip/default
tests exactly, substituting the new variant.

- [ ] **Step 4: Wire dispatch in `agent_builder.rs`**

```rust
BackendKind::Generic | BackendKind::LlamaServer => Ok(Arc::new(OpenAiCompatBackend::new(
    settings.backend.base_url.clone(),
    settings.backend.model.clone(),
    settings.backend.api_key.clone(),
    // ...(existing args, unchanged)
))),
BackendKind::LlamaServerBroker => {
    let broker_url = settings.backend.broker_base_url.clone().ok_or_else(|| {
        anyhow::anyhow!(
            "backend.kind = \"llama_server_broker\" but backend.broker_base_url is missing. \
             Set it to your running aivyx-broker's address, e.g. http://127.0.0.1:8899"
        )
    })?;
    Ok(Arc::new(OpenAiCompatBackend::new(
        broker_url,
        settings.backend.model.clone(),
        settings.backend.api_key.clone(),
        // ...(existing args, unchanged)
    )))
}
```

At line ~578's `kv_cache_handles` construction, change the guard from
`settings.backend.kind == BackendKind::LlamaServer` to explicitly exclude
`LlamaServerBroker` too (it must stay `false`/`None` for that variant — the
broker owns kvcache lifecycle, this process must not also try).

- [ ] **Step 5: Skip local slot-picking/persistence in `agent/mod.rs`**

At the top of `ensure_kv_slot_checked_out`, add an early return when
`BackendKind::LlamaServerBroker` is active (mirroring the existing
`if self.kv_slot_id.is_some() { return; }` / `let Some(kv) = &self.kv_cache else { return; }`
guards already there — since Step 4 already ensures `kv_cache_handles`
is `None` for this variant, the existing `let Some(kv) = &self.kv_cache else { return; }`
guard already causes this function to no-op correctly with **zero
additional code** — confirm this is actually sufficient before adding any
new guard; it likely is, since `kv_cache: None` already short-circuits the
whole function today for any backend that never constructed cache
handles).

Wherever `id_slot: Some(slot_id)` / `id_slot: self.kv_slot_id` is set on an
outgoing `ChatRequest` today (lines ~557, ~1525 per this plan's own
grounding), add the parallel `slot_hint` population when
`BackendKind::LlamaServerBroker` is active:
`slot_hint: (kind == BackendKind::LlamaServerBroker).then(|| SlotHint { prefix_hash: compute_prefix_hash(&system_text, &tools), preferred_slot: self.kv_slot_id })`.
Note `self.kv_slot_id` will always be `None` on this path per Step 5's
finding above — so `preferred_slot` starts `None` on a session's first
request and the broker's own occupancy tracking (not this client) is what
makes subsequent same-session requests fast, exactly as designed. This is
expected and correct, not a gap: the broker, not this client, remembers
which slot a session landed on.

- [ ] **Step 6: Add `slot_hint` field to `ChatRequest`, serialize it in `OpenAiCompatBackend`**

```rust
pub struct ChatRequest {
    // ...(existing fields, unchanged)
    pub id_slot: Option<u32>,
    pub slot_hint: Option<SlotHint>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct SlotHint {
    pub prefix_hash: String,
    pub preferred_slot: Option<u32>,
}
```

Wherever `OpenAiCompatBackend` builds the outgoing JSON body and inserts
`id_slot` conditionally, add the same pattern for `slot_hint`, serialized
under the `aivyx_slot_hint` key only when `Some`.

- [ ] **Step 7: Write a test proving the skip + hint attachment**

Mirror whatever existing test in `agent/mod.rs` already exercises
`ensure_kv_slot_checked_out`'s no-op-when-no-kvcache path (there should be
one, since `kv_cache: None` is already a tested case for other backend
kinds) — add a case for `BackendKind::LlamaServerBroker` confirming the
same no-op, plus a new test in wherever `ChatRequest`/`OpenAiCompatBackend`
already has serialization tests, confirming a `Some(slot_hint)` produces
the `aivyx_slot_hint` key in the serialized body and `None` omits it
entirely (mirroring however this repo's existing `id_slot` serialization
test is structured).

- [ ] **Step 8: Run the full workspace suite and clippy in both feature
  configurations** (with and without `provider-mistral-rs`, matching this
  repo's own established convention from the mistral.rs work)

Run: `cargo test --workspace && cargo test --workspace --features provider-mistral-rs && cargo clippy --workspace --all-targets && cargo clippy --workspace --all-targets --features provider-mistral-rs`
Expected: all green in both configurations.

- [ ] **Step 9: Update `README.md`** with a short "Multi-process GPU
  sharing (`aivyx-broker`)" section — point at `aivyx-broker`'s own repo
  and README for setup, document `backend.kind = "llama_server_broker"`
  and `backend.broker_base_url` in this repo's own config reference table.

- [ ] **Step 10: Commit**

```bash
git add crates/aivyx-config/src/lib.rs crates/aivyx/src/agent_builder.rs crates/aivyx-core/src/agent/mod.rs crates/aivyx-llm/src/ README.md
git commit -m "feat: add BackendKind::LlamaServerBroker for aivyx-broker multi-process slot coordination"
```

---

## After all tasks

Each of the three repos gets its own `finishing-a-development-branch` pass
— they are independent histories, independent merges, independent pushes.
Do not treat this as one branch across three repos. Update
`aivyx-ecosystem/ROADMAP.md` with one cross-repo entry once all three are
merged, following this session's established pattern for multi-repo work
(see the mistral.rs work's own roadmap entry for the shape to match).

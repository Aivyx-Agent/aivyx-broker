# aivyx-broker: multi-process GPU-slot scheduling design

**Status:** approved, ready for implementation planning
**Repo:** `aivyx-broker` (new standalone repo)

## Motivation

`aivyx` and `aivyx-coder` are two independent local processes that can both
be configured to point at the same `llama-server` (possible since the
kvcache-store-path-sharing work). Each picks a physical KV-cache slot for
its requests via its own private, in-process slot tracker
(`aivyx`'s `KvSlotPool`, `aivyx-coder`'s `ensure_kv_slot_checked_out`) and
pins it via `id_slot` on the request. Neither tracker has any awareness of
the other process.

This was investigated directly against real code before any design work
began, to separate confirmed fact from the original audit's assumption:

- **Confirmed real (client side):** both `KvSlotPool` (`aivyx`,
  `crates/aivyx-llm/src/kv_slot_pool.rs`) and `aivyx-coder`'s equivalent
  hand out the lowest free slot id starting from 0, with zero cross-process
  coordination. Two processes pointed at the same server will independently
  pick `id_slot: Some(0)` first.
- **Confirmed real (server side, verified against llama.cpp's own
  `server-context.cpp`):** when a request pins a busy `id_slot`,
  `get_available_slot` returns that slot anyway, but `process_single_task`
  separately checks `is_processing()` and **defers** the task in its own
  queue rather than running it concurrently or corrupting shared state.
  llama-server already serializes same-slot collisions safely.

**What this means for scope:** the race is real, but it is **not** a data-
corruption bug — it's silent head-of-line blocking (a second process's
request queues invisibly behind the first with no diagnosis available) and
cache-locality thrash (a process's private bookkeeping believes a slot still
holds its last prefix; another process's request may have silently evicted
it in between, so the next turn eats an unexpected full reprocess instead of
a cache hit). This is a real, worthwhile problem, but a narrower one than
"processes could corrupt each other's inference."

The user, informed of this narrower-than-assumed severity, chose to build
the full scheduling layer anyway (not just a collision-avoidance patch),
reasoning that a proper coordination layer is worth having regardless of
whether the current failure mode is "only" a performance problem.

## Out of scope for v1

- Priority/weighted scheduling across processes or requests. v1 is pure
  FIFO by arrival order. Explicitly deferred as a future increment once
  real usage shows starvation is worth solving.
- Remote/multi-host operation. The broker is a local-loopback-only process,
  same trust model as `llama-server` itself today — no auth, binds
  `127.0.0.1` only, not safe to expose beyond localhost.
- Persisting broker-owned scheduling state across restarts. `llama-server`'s
  own `GET /slots` remains the source of truth for physical occupancy on
  broker startup; `aivyx-kvcache`'s existing WAL-sqlite manifest remains
  the source of truth for saved KV content on disk. The broker adds only
  the missing piece — live arbitration of who uses a slot right now — and
  keeps no state that must outlive its own process.
- Automatic broker lifecycle management (auto-spawn, systemd unit
  generation) by either client app. The broker is started the same way
  `llama-server` itself is today: manually, by the operator.
- A real multi-process race test. This environment has no way to run two
  genuine OS processes contending for one genuine `llama-server` instance.
  Documented as an accepted gap, same category as the mistral.rs work's
  "no real GGUF model to test against" — manual verification against a
  real rig is a follow-up, not a merge blocker.

## Architecture

A standalone Rust daemon, `aivyx-broker`, reached over local loopback HTTP.
Both `aivyx` and `aivyx-coder` are configured to talk to it instead of
`llama-server` directly. It sits fully in the request path: it owns
admission + cache-locality-aware slot assignment, then forwards the actual
completion (streamed) to the real `llama-server` and relays the stream back
untouched.

Two alternatives were considered and rejected:

- **gRPC/custom binary protocol** — more efficient, but this ecosystem has
  zero precedent for it; every existing integration speaks OpenAI-compatible
  HTTP/JSON. Adopting a new toolchain for one component buys nothing
  concrete and breaks convention.
- **Embedded-in-process broker with leader election** (one of the two apps
  elects itself "primary") — rejected: contradicts the standalone-repo
  decision, and adds real distributed-systems complexity (leader election,
  handling the leader's own restart) that a plain separate process avoids
  entirely — `llama-server` itself is already precedent for "a separate
  reachable local process" in this ecosystem.

Components inside `aivyx-broker`:

- An Axum HTTP server exposing an OpenAI-compatible `/v1/chat/completions`
  endpoint (switching either client is a `base_url` config change, zero
  `LlmBackend` code changes) plus a small admin surface (`/status`,
  `/slots`).
- A scheduler core: in-memory occupancy table (slot → `{holder, prefix_hash,
  state}`) plus a FIFO wait queue per contested slot and one global FIFO
  queue for "any free slot" requests. Rebuilt from `llama-server`'s real
  `GET /slots` on every broker startup.
- A thin proxy/forwarder to the real `llama-server`, streaming chunks
  through as they arrive rather than buffering the full response (same
  shape as the mpsc-forwarding shim built for the mistral.rs work).

## Protocol

**`POST /v1/chat/completions`** — identical shape to `llama-server`'s own
endpoint, with one additive field:

```json
{
  "...": "standard OpenAI-compatible chat completion body",
  "aivyx_slot_hint": {
    "prefix_hash": "string, same fnv1a-based hash aivyx-kvcache already computes",
    "preferred_slot": 2
  }
}
```

`aivyx_slot_hint` is optional. Omitting it means "any free slot, prefer
LRU among idle ones" — the same fallback `KvSlotPool` already implements
today. `prefix_hash` must be computed with the **same** hash function
`aivyx-kvcache` already uses for its on-disk slot keys (its existing
`fnv1a`-based scheme in `llama_server.rs`, made `pub` and reused directly)
so "which prefix is this" has one canonical definition across the whole
ecosystem instead of a second, competing identity scheme.

Response is a plain passthrough of `llama-server`'s own SSE stream — the
broker does not reinterpret token content.

**`GET /status`** — broker health, which `llama-server` it's proxying to,
current queue depth.

**`GET /slots`** — the broker's own occupancy view (slot → holder process
label, prefix hash, busy/idle, queue position if contested) — a superset of
`llama-server`'s own `/slots`, adding the cross-process attribution
`llama-server` has no notion of.

**Admission semantics on `/v1/chat/completions`:**

1. `preferred_slot` set and free → assign immediately.
2. `preferred_slot` set but busy → queue FIFO for that specific slot. Do not
   steal a different free slot — that would silently defeat the hint's
   purpose (the client would get a slot without its cached prefix and eat a
   full reprocess anyway), so waiting briefly for the *right* slot is
   usually still the faster outcome.
3. No hint, or the hinted slot is unknown to the broker (never-seen
   prefix) → assign any free slot, LRU-first among idle ones.
4. No slots free at all, no preference → FIFO queue, first-come-first-served.
   No priority tiers in v1.

## Scheduling & failure handling

**Queue mechanics:** a queued request holds its HTTP connection open,
awaiting admission (no new pattern — same shape as a client blocked on a
permission-gate prompt). A slot release wakes the head of that slot's
specific wait queue; a general "any free slot" release wakes the head of
the global queue.

**Timeouts:** a request queued past a configurable ceiling (default 60s)
gets a clear error back rather than hanging forever — the whole reason this
project exists is to replace silent head-of-line blocking with something
diagnosable.

**Broker startup/restart:** seeds its occupancy table from `llama-server`'s
real `GET /slots`. A slot `llama-server` reports mid-generation is marked
busy-but-unowned (the broker doesn't know which client it belonged to across
its own restart) until a client with a matching prefix hash reclaims it —
but that slot still won't accept an unrelated new request until
`llama-server` itself reports it idle. A broker restart loses cross-process
fairness bookkeeping but never desyncs from physical reality.

**`llama-server` down/unreachable:** the broker forwards the real connection
error back to the client rather than masking it. Both client apps already
handle "backend unreachable" today (against `llama-server` directly) — this
needs no new client-side error path, only that the broker not swallow the
error.

**Broker itself down:** both apps' `OpenAiCompatBackend` see connection-
refused against the broker's `base_url`, identical in shape to "backend
unreachable" — no special-casing needed. This is a real, accepted cost of
centralizing that the README must state honestly: once a setup opts into
the broker, it becomes a required-up dependency, the same way `llama-server`
already is.

## Client integration (`aivyx` and `aivyx-coder`)

Additive and config-gated in both apps — a new backend mode (e.g.
`BackendKind::LlamaServerBroker` in each app's existing `BackendKind` enum)
so setups that don't need multi-process sharing see zero behavior change.
When enabled:

- `aivyx`'s `llm_planner.rs` and `aivyx-coder`'s `agent/mod.rs` stop calling
  their local `KvSlotPool::checkout()` (or equivalent) entirely on this
  path — they no longer pick a physical slot number themselves. They
  compute the prefix hash and let the broker decide. This is a net
  simplification: today's client-local slot-picking logic becomes dead
  code on the broker path, replaced by "ask the broker."
- The existing direct-to-`llama-server` mode (today's default) is
  unchanged — this is a new opt-in path, not a replacement.

## Testing strategy

Same environment constraint as the mistral.rs work: no real GPU or
`llama-server` available here.

- Scheduler core: pure unit tests against fabricated occupancy
  tables/queues, no HTTP or real server — same style as `KvSlotPool`'s
  existing tests.
- HTTP/proxy layer: integration tests against a small fake local server
  standing in for `llama-server` (canned SSE chunks), verifying passthrough
  shape, hint parsing, queueing, and timeout behavior end-to-end.
- Client-side hint-building + config wiring in both apps: unit tests using
  each repo's existing mock-backend patterns.
- Real two-process race behavior against a real `llama-server`: not
  testable in this environment, documented as an accepted gap. Manual
  verification against a real rig is a follow-up.

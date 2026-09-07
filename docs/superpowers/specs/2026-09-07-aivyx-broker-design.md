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

- Priority/weighted scheduling across processes or requests. v1 admits
  approximately FIFO by arrival order — under contention, a released
  slot's queued waiters race to reclaim it rather than being served in
  strict order, so there's no hard ordering guarantee beyond each
  request's own queue timeout. Priority/weighting is explicitly deferred
  as a future increment once real usage shows starvation is worth
  solving.
- Remote/multi-host operation. The broker is a local-loopback-only process,
  same trust model as `llama-server` itself today — no auth, binds
  `127.0.0.1` only, not safe to expose beyond localhost.
- Persisting broker-owned scheduling state across restarts. `llama-server`'s
  own `GET /slots` remains the source of truth for physical occupancy on
  broker startup; `aivyx-kvcache`'s existing WAL-sqlite manifest remains
  the source of truth for saved KV content on disk (the broker drives it
  as a library dependency, same as each client does today — it doesn't
  reimplement persistence). The broker's own occupancy/queue bookkeeping
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
- A thin proxy/forwarder to the real `llama-server`. `reqwest::Response`'s
  own `bytes_stream()` is already an owned, `'static` stream (unlike
  mistral.rs's in-process, borrowed `Stream<'_>`), so this is a direct
  stream-through, not another mpsc-forwarding shim — chunks pass to the
  client as they arrive, never buffered whole.
- **The broker owns the full KV-cache restore/warm/save lifecycle, not
  just live slot admission** — a late but important correction made
  during implementation planning. The original per-app flow
  (`ensure_kv_slot_checked_out`) does two things together: pick a slot
  number, then restore/warm/save that slot's on-disk content via
  `aivyx-kvcache`'s `LlamaServerSlotStore` — and it only learns which slot
  to restore into *after* picking one. Once admission moves inside the
  broker's single `/v1/chat/completions` call, the client can no longer
  do its own restore first — it doesn't know the slot number yet. So
  `aivyx-broker` depends on `aivyx-kvcache` directly and drives
  `restore_into_slot`/`save_from_slot` itself, using each client's
  `prefix_hash` hint as the `CacheKey`. Concretely, on each request: if
  the hint's `prefix_hash` is already the *current* occupant of some slot
  in the broker's own occupancy table (i.e. the same session's own prior
  turn, still resident), admit straight to that slot — no restore/save
  needed, since the slot's live content already matches and llama-server
  itself accumulates the growing turn history in place. Otherwise
  (first request for this `prefix_hash`, or the broker just restarted and
  lost its occupancy memory) — admit a slot per the rules below, call
  `restore_into_slot` before forwarding the request, and call
  `save_from_slot` after the response completes so a future session
  (this client's, or another's, restarted or not) can reuse it. This
  fully replaces `ensure_kv_slot_checked_out` on the broker path — clients
  using the broker never call `aivyx-kvcache` themselves.

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
today. `prefix_hash` is an **opaque string as far as the broker is
concerned** — each client already computes one locally today (`aivyx`'s
`compute_prefix_hash` in `llm_planner.rs`, `aivyx-coder`'s equivalent in
`agent/mod.rs`), already stable per system-prompt+tools combination,
already used to build their own `CacheKey`s for `aivyx-kvcache`. The
broker's only requirement is that the *same* client sends the *same*
string back for the *same* prefix — it never compares one app's hash
against the other's, so the two apps' hash algorithms never need to match.
(Earlier drafts of this spec proposed extracting `aivyx-kvcache`'s internal
`fnv1a` into a shared function for this — that was a mistake, caught during
implementation planning: that function is a private filename-disambiguation
detail with no cross-repo stability guarantee, not the actual prefix-hash
source. No change to `aivyx-kvcache` is needed for this field at all.)

Response is a plain passthrough of `llama-server`'s own SSE stream — the
broker does not reinterpret token content.

**`GET /status`** — broker health, which `llama-server` it's proxying to,
and a `queue_depth` field. In v1 this is a fixed placeholder (always `0`),
not real queue-depth tracking — an explicit YAGNI scope-cut from the plan,
not a bug; real tracking is deferred to a later increment.

**`GET /slots`** — the broker's own occupancy view (slot busy/idle plus the
currently-resident prefix hash the broker itself last admitted into that
slot) — a superset of `llama-server`'s own `/slots` in that sense, but not
a cross-process attribution: the broker doesn't track which process or
request owns a given slot, only what prefix it last placed there.

**Admission semantics on `/v1/chat/completions`:**

1. `preferred_slot` set and free → assign immediately.
2. `preferred_slot` set but busy → queue FIFO for that specific slot. Do not
   steal a different free slot — that would silently defeat the hint's
   purpose (the client would get a slot without its cached prefix and eat a
   full reprocess anyway), so waiting briefly for the *right* slot is
   usually still the faster outcome.
3. No hint, or the hinted slot is unknown to the broker (never-seen
   prefix) → assign any free slot, LRU-first among idle ones.
4. No slots free at all, no preference → FIFO queue, approximately
   first-come-first-served (see "Queue mechanics" below for why this isn't
   a strict ordering guarantee). No priority tiers in v1.

## Scheduling & failure handling

**Queue mechanics:** a queued request holds its HTTP connection open,
awaiting admission (no new pattern — same shape as a client blocked on a
permission-gate prompt). A slot release wakes every waiter queued for that
slot's specific wait queue, and every waiter in the global "any free slot"
queue too — not just the head of each. This is deliberate: stopping after
the first successfully-woken waiter left a narrow starvation window (if
that waiter's future was dropped without being re-polled before it could
reclaim the slot, nobody else queued behind it would ever be woken even
though the slot was sitting free). Every waiter re-validates by attempting
admission again itself upon waking, so waking more than one is safe — only
whichever one wins the race actually claims the slot; the rest re-queue and
keep waiting. The tradeoff: admission is approximately FIFO by arrival
order, not strictly ordered — under contention, a released slot's queued
waiters race to reclaim it rather than being served in strict order, so
there's no hard ordering guarantee beyond each request's own queue
timeout.

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

- `aivyx`'s `llm_planner.rs` and `aivyx-coder`'s `agent/mod.rs` skip
  `ensure_kv_slot_checked_out` (or equivalent) entirely on this path — no
  local `KvSlotPool::checkout()`, no `aivyx-kvcache` calls at all. They
  reuse the `prefix_hash` they already compute today for `CacheKey` (via
  each app's own existing `compute_prefix_hash`) as the hint's
  `prefix_hash`, attach it to the outgoing `ChatRequest`, and let the
  broker own slot assignment *and* restore/warm/save. This is a net
  simplification: today's client-local slot-picking-and-persistence logic
  becomes entirely dead code on the broker path, replaced by "ask the
  broker."
- Each client's own `kvcache_store_path`/`kvcache_max_bytes`-style config
  becomes irrelevant on the broker path — the broker takes its own
  `--kvcache-store-path`/`--kvcache-max-bytes` at startup and owns that
  store directly. Both apps should point their own `kvcache_store_path` at
  the same directory as the broker's, matching the multi-process sharing
  convention the kvcache-store-path-sharing work already established.
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

# aivyx-broker

[![CI](https://github.com/Aivyx-Agent/aivyx-broker/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/Aivyx-Agent/aivyx-broker/actions/workflows/ci.yml)
[![License: BUSL-1.1](https://img.shields.io/badge/license-BUSL--1.1-blue.svg)](LICENSE)

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

```sh
cargo build --release
./target/release/aivyx-broker \
  --llama-server-url http://127.0.0.1:8080 \
  --kvcache-store-path ~/.local/state/aivyx-broker/kvcache
```

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
| `--gpu-lock-queue-timeout-secs` | `AIVYX_BROKER_GPU_LOCK_QUEUE_TIMEOUT_SECS` | `300` | How long a caller may wait in the GPU-lock queue (see [`POST /gpu-lock/acquire`](#api) below) before getting a clear `503` instead of hanging — longer than `queue-timeout-secs` because a GPU generation job can legitimately queue much longer than a chat turn |
| `--gpu-lock-max-hold-secs` | `AIVYX_BROKER_GPU_LOCK_MAX_HOLD_SECS` | `900` | Safety valve: a GPU lock lease held longer than this is force-released by a background reap task, on the assumption its holder crashed or disconnected without releasing. If it fires on a holder that's still alive and running (just slow), the lock is handed to a second waiter while the first is still using the GPU -- set it above your worst-case generation time, not merely the typical one |
| `--vram-source` | `AIVYX_BROKER_VRAM_SOURCE` | `auto` | Where `GET /v1/aivyx/residency` reads host GPU memory from: `auto` (`nvidia-smi`, else AMD sysfs), `nvidia`, `amd`, or `none` to report no VRAM |

## Pointing a client at the broker

Both `aivyx-pa` and `aivyx-coder` have shipped a broker-aware backend
mode — see [`aivyx-pa`'s own
INSTALL.md](https://github.com/Aivyx-Agent/aivyx-pa/blob/main/docs/INSTALL.md#coordinating-gpu-slot-access-across-multiple-processes-aivyx-broker)
(`aivyx-pa`'s top-level README/CLAUDE.md don't mention it yet, so this is
the real place to look) and [`aivyx-coder`'s own
README](https://github.com/Aivyx-Agent/aivyx-coder#multi-process-gpu-sharing-aivyx-broker)
for each one's own config. In short: point the client's `base_url` at the
broker instead of `llama-server` directly, and enable its broker mode so
it stops doing its own local slot-picking and kvcache restore/save (the
broker now owns that).

## API

- `POST /v1/chat/completions` — OpenAI-compatible, identical to
  `llama-server`'s own endpoint, plus an optional `aivyx_slot_hint` field:
  `{"prefix_hash": "...", "preferred_slot": <u32 or null>}`. Omit it for
  plain "any free slot" behavior.
- `GET /status` — broker health + which `llama-server` it's proxying to.
- `GET /slots` — the broker's own occupancy view: `{slot_id, busy,
  resident_prefix}` per slot — a superset of `llama-server`'s own
  `/slots`, adding which `prefix_hash` is currently resident in each slot
  (something `llama-server`'s own `/slots` doesn't expose).
- `POST /gpu-lock/acquire` — a generic, lease-based exclusive lock for
  GPU-heavy non-LLM generation work (Aivyx-Vision's mold backend),
  deliberately independent of the `/v1/chat/completions` slot-scheduling
  path above (see `src/gpu_lock.rs`'s own doc comment for why). Waits for
  exclusive access up to `--gpu-lock-queue-timeout-secs`, then returns
  `200 {"lease_id": "<uuid>"}` or `503` on timeout.
- `POST /gpu-lock/release` — releases a lease acquired above:
  `{"lease_id": "<uuid>"}` → `200` on success, `404` if the lease is
  unknown/already released/expired, `400` if `lease_id` isn't a valid
  UUID. A lease not released within `--gpu-lock-max-hold-secs` is force-
  released by a background reap task, so a crashed holder can't wedge the
  lock forever.
- `GET /v1/aivyx/residency` — read-only residency for `aivyx-route` model
  routing: the upstream's models (loaded or not), host VRAM, and slot
  pressure. Every part is best-effort and the route always answers `200`,
  even with no upstream and no GPU reachable:
  ```json
  {
    "models": [
      {"id": "qwen3-8b", "loaded": true},
      {"id": "gemma-3-4b", "loaded": false}
    ],
    "vram": {"total_bytes": 25769803776, "used_bytes": 9663676416},
    "slots": {"busy": 1, "total": 2}
  }
  ```
  - `models` — from the upstream `GET /models`, reduced to
    `{id, loaded}`. Router-mode `status.value` of `loaded`/`loading`
    counts as loaded, `unloaded`/`sleeping` as not; a single-model
    `llama-server` with no `status` field at all is loaded. `[]` if the
    upstream is unreachable or unparseable.
  - `vram` — host-wide VRAM per `--vram-source` above, or `null` if
    unreadable.
  - `slots` — the broker's own scheduler pressure: how many of its slots
    are currently busy, out of the total.

## Honest tradeoffs

- **A new required-up dependency.** Once a setup opts into the broker, it
  becomes as load-bearing as `llama-server` itself — if it's down, both
  client apps see connection-refused. This is an accepted cost of
  centralizing coordination, not a bug.
- **No priority/weighted scheduling in v1.** Admission is approximately
  FIFO by arrival order — under contention, a released slot's queued
  waiters race to reclaim it rather than being served in strict order, so
  there's no hard ordering guarantee beyond each request's own queue
  timeout. Priority/weighting is deferred as a future increment.
- **No real multi-process race test exists.** This project's own test
  suite has no way to run two genuine OS processes contending for one
  genuine `llama-server` — verified with fakes/mocks only. Manual
  verification against a real rig is a documented follow-up, not a merge
  blocker.
- **The GPU lock is advisory, not enforced against `/v1/chat/completions`.**
  Holding a `gpu-lock` lease does not pause, throttle, or otherwise affect
  LLM slot scheduling in any way — the two paths are deliberately
  independent implementations (see `src/gpu_lock.rs`'s doc comment) with
  no shared state. A client that acquires the GPU lock expecting LLM
  inference to also be quiet during that window will be wrong; the lock
  only excludes other GPU-lock callers from each other.
- **No fairness guarantee among GPU-lock waiters**, same as the slot path
  above — released/reaped lock waiters race to reclaim it rather than
  being served in strict arrival order.
- **A crashed GPU-lock holder wedges the lock for up to
  `--gpu-lock-max-hold-secs`** before the background reap task frees it;
  there's no faster way to detect the crash from the broker's side.

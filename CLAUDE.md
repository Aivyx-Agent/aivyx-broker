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

As of 2026-09-18 the broker also arbitrates GPU-heavy non-LLM workloads
(Aivyx-Vision's mold backend) via a separate, independent lease-based
lock (`POST /gpu-lock/acquire` / `/gpu-lock/release`) — see
`src/gpu_lock.rs`'s own doc comment for the full rationale.

## Build, run, test, lint

```sh
cargo build
cargo test
cargo clippy --all-targets
cargo fmt --check

cargo run -- --llama-server-url http://127.0.0.1:8080 --kvcache-store-path ~/.local/state/aivyx-broker/kvcache
```

Loopback-only, no auth — same trust model as `llama-server` itself. Not
safe to expose beyond `127.0.0.1`.

## Known, deliberately-undefended limitations

- Loopback-only, no auth. Not safe to expose beyond `127.0.0.1`.
- No priority/weighted scheduling in v1. Admission is approximately FIFO
  by arrival order — under contention, a released slot's queued waiters
  race to reclaim it rather than being served in strict order, so there's
  no hard ordering guarantee beyond each request's own queue timeout.
- No persisted broker state across restarts (by design — see the spec's
  "Broker startup/restart" section); a restart loses cross-process
  fairness bookkeeping but never desyncs from `llama-server`'s own
  physical reality.
- No automatic lifecycle management (auto-spawn, systemd unit
  generation) — started manually, same as `llama-server` itself.
- The GPU lock (`POST /gpu-lock/acquire`/`/release`) is advisory only —
  holding a lease does not pause or affect `/v1/chat/completions`
  admission in any way; the two are independent implementations with no
  shared state. Same no-fairness-guarantee-among-waiters caveat as the
  slot path applies. A crashed holder wedges the lock for up to
  `--gpu-lock-max-hold-secs` before the background reap task frees it —
  and if that fires on a holder that's merely slow, not crashed, the lock
  is handed to a second caller while the first is still using the GPU.

## Where to look next

- `README.md` — setup and config reference.
- `docs/superpowers/specs/2026-09-07-aivyx-broker-design.md` — full design.

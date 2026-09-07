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

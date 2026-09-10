# Security Policy

`aivyx-broker` is a loopback-only local daemon — no external network
exposure by design (binds `127.0.0.1` only, no auth, same trust model as
`llama-server` itself). The security boundary that matters is exactly
that scope: a finding that this daemon accepts non-loopback connections,
leaks another local user's KV-cache data across process boundaries, or
can be coerced into forwarding requests somewhere other than the
configured `llama-server` is in scope. "The LLM produced a bad answer"
or similar model-quality issues are not.

This is not a bug bounty program.

## Reporting a vulnerability

Two channels:
- **GitHub Security Advisories** — use the "Report a vulnerability"
  button under this repo's Security tab (private by default, visible
  only to maintainers until you choose to publish).
- **Email** — **jccorbett67@gmail.com** with details.

This is a small, solo-maintained project: reports will be read and
acknowledged, and we aim to resolve or provide a remediation plan for a
confirmed vulnerability within 90 days of the report, or coordinate a
later disclosure date directly with the reporter if a fix genuinely
needs longer. Credit is offered in release notes at the reporter's
preference.

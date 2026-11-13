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

Report it privately on GitHub: [Report a vulnerability](https://github.com/Aivyx-Agent/aivyx-broker/security/advisories/new)
(this repo's **Security** tab). Only the maintainer sees the report
until a fix is published. Please don't open a public issue for a
vulnerability.

This is a small, solo-maintained project: reports will be read and
acknowledged, and we aim to resolve or provide a remediation plan for a
confirmed vulnerability within 90 days of the report, or coordinate a
later disclosure date directly with the reporter if a fix genuinely
needs longer. Credit is offered in release notes at the reporter's
preference.

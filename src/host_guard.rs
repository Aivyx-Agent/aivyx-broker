//! Loopback `Host` header validation.
//!
//! `aivyx-broker` is documented (`CLAUDE.md`, README) as loopback-only,
//! no-auth -- the same trust model as `llama-server` itself. But axum
//! (like most HTTP servers) never validates the `Host` header against
//! the socket it's actually bound to: it happily serves a request whose
//! `Host` names anything at all, as long as the bytes reached the right
//! port. A browser page on `https://evil.example` can exploit this via
//! DNS rebinding -- point a subdomain's DNS at `127.0.0.1`, fetch it from
//! JS, and the browser will send `Host: evil.example` (or whatever the
//! page's origin is) straight to this loopback-only broker, defeating
//! the "loopback-only" trust boundary entirely: no browser same-origin
//! check ever kicks in, because as far as the browser's networking stack
//! is concerned this *is* a same-origin request to `evil.example`.
//!
//! The fix mirrors what Grafana, Prometheus and others do for the same
//! class of service: validate `Host` against an explicit allowlist on
//! every route, rejecting anything else with `421 Misdirected Request`
//! before any handler runs. `localhost`, `127.0.0.0/8`, and `[::1]` are
//! always allowed (with any port, since the broker's own `--port` is
//! configurable); `--allowed-hosts`/`AIVYX_BROKER_ALLOWED_HOSTS` extends
//! the allowlist for anyone who deliberately binds the broker to a
//! non-loopback address.

use axum::extract::State;
use axum::http::{Request, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};

use crate::server::AppState;

/// Rejects any request whose `Host` header doesn't resolve to a loopback
/// address (`localhost`, `127.0.0.0/8`, `::1`, any port) or one of
/// `state.allowed_hosts`. Applied as a router-wide layer so every route
/// is covered, not just the ones an author remembers to annotate.
pub async fn enforce_allowed_host(
    State(state): State<AppState>,
    req: Request<axum::body::Body>,
    next: Next,
) -> Response {
    let host_header = req
        .headers()
        .get(axum::http::header::HOST)
        .and_then(|v| v.to_str().ok());
    match host_header {
        Some(host) if host_is_allowed(host, &state.allowed_hosts) => next.run(req).await,
        _ => (
            StatusCode::MISDIRECTED_REQUEST,
            [("content-type", "application/json")],
            serde_json::json!({
                "error": "Host header not allowed: this broker only serves loopback requests"
            })
            .to_string(),
        )
            .into_response(),
    }
}

/// Whether `host_header` (the raw `Host` header value, with or without a
/// port, bracketed for IPv6) names a loopback address, or one of
/// `extra_allowed_hosts` (hostnames/IPs, no port -- any port on them is
/// allowed, matching the loopback rule).
pub fn host_is_allowed(host_header: &str, extra_allowed_hosts: &[String]) -> bool {
    let host = strip_port(host_header);
    if is_loopback_host(host) {
        return true;
    }
    extra_allowed_hosts
        .iter()
        .any(|allowed| allowed.eq_ignore_ascii_case(host))
}

/// Strips an optional trailing `:port` from a `Host` header value.
/// Bracketed IPv6 literals (`[::1]` or `[::1]:8899`) have their brackets
/// stripped too; everything else is only split on the last `:` when what
/// follows is all digits, so a bare (non-bracketed, technically
/// non-conformant) IPv6 literal is returned unchanged rather than
/// mis-split.
fn strip_port(host: &str) -> &str {
    if let Some(rest) = host.strip_prefix('[') {
        return rest.split(']').next().unwrap_or(rest);
    }
    match host.rsplit_once(':') {
        Some((h, port)) if !port.is_empty() && port.bytes().all(|b| b.is_ascii_digit()) => h,
        _ => host,
    }
}

/// Whether `host` (already stripped of its port) is `localhost` or a
/// loopback IP (`127.0.0.0/8` or `::1`).
fn is_loopback_host(host: &str) -> bool {
    if host.eq_ignore_ascii_case("localhost") {
        return true;
    }
    host.parse::<std::net::IpAddr>()
        .is_ok_and(|ip| ip.is_loopback())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn evil_host_is_rejected() {
        assert!(!host_is_allowed("evil.example", &[]));
        assert!(!host_is_allowed("evil.example:8899", &[]));
    }

    #[test]
    fn localhost_with_any_port_is_allowed() {
        assert!(host_is_allowed("localhost", &[]));
        assert!(host_is_allowed("localhost:8899", &[]));
        assert!(host_is_allowed("LOCALHOST:1", &[]));
    }

    #[test]
    fn ipv4_loopback_range_with_any_port_is_allowed() {
        assert!(host_is_allowed("127.0.0.1", &[]));
        assert!(host_is_allowed("127.0.0.1:8899", &[]));
        assert!(host_is_allowed("127.255.255.255:1", &[]));
    }

    #[test]
    fn ipv4_outside_loopback_range_is_rejected() {
        assert!(!host_is_allowed("10.0.0.1:8899", &[]));
        assert!(!host_is_allowed("128.0.0.1", &[]));
    }

    #[test]
    fn bracketed_ipv6_loopback_with_any_port_is_allowed() {
        assert!(host_is_allowed("[::1]", &[]));
        assert!(host_is_allowed("[::1]:8899", &[]));
    }

    #[test]
    fn non_loopback_ipv6_is_rejected() {
        assert!(!host_is_allowed("[::2]:8899", &[]));
    }

    #[test]
    fn extra_allowed_host_is_accepted_with_any_port() {
        let extra = vec!["my-box.lan".to_string()];
        assert!(host_is_allowed("my-box.lan:8899", &extra));
        assert!(host_is_allowed("MY-BOX.LAN", &extra));
        assert!(!host_is_allowed("other.lan:8899", &extra));
    }
}

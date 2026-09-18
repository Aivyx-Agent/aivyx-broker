use std::path::PathBuf;

use clap::Parser;

#[derive(Debug, Clone, Parser)]
#[command(
    name = "aivyx-broker",
    about = "Multi-process GPU-slot scheduling broker"
)]
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

    /// How long a caller may wait in the GPU-lock queue before getting a
    /// clear timeout error instead of hanging. Distinct from
    /// `queue_timeout_secs` (the LLM chat-completion admission queue) --
    /// a GPU generation job can legitimately queue much longer than a
    /// chat turn.
    #[arg(
        long,
        env = "AIVYX_BROKER_GPU_LOCK_QUEUE_TIMEOUT_SECS",
        default_value_t = 300
    )]
    pub gpu_lock_queue_timeout_secs: u64,

    /// Safety valve: a GPU lock lease held longer than this is force-
    /// released by the background reap task, on the assumption its
    /// holder crashed or disconnected without releasing. Set generously
    /// above realistic generation time (an image/3D generation job can
    /// legitimately run for minutes). This is a genuine tradeoff, not a
    /// free safety net: if it fires while the holder is still alive and
    /// running (just slow), the freed lock is handed to a second waiter
    /// while the first is still using the GPU -- the exact double-
    /// occupancy this lock exists to prevent. Set it above your worst-case
    /// generation time, not merely the typical one.
    #[arg(
        long,
        env = "AIVYX_BROKER_GPU_LOCK_MAX_HOLD_SECS",
        default_value_t = 900
    )]
    pub gpu_lock_max_hold_secs: u64,
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
        assert_eq!(cfg.gpu_lock_queue_timeout_secs, 300);
        assert_eq!(cfg.gpu_lock_max_hold_secs, 900);
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

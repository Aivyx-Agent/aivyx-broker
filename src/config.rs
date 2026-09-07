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

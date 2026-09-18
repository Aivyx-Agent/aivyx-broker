use std::time::Duration;

use clap::Parser;

/// How often the background reconciliation task polls `llama-server`'s
/// real `/slots` to clear out seeded-busy-but-unowned slots that have
/// since gone idle. See `Scheduler::reconcile_seeded_idle` (Fix 1).
const RECONCILE_INTERVAL: Duration = Duration::from_secs(10);

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();
    let config = aivyx_broker::BrokerConfig::parse();

    // A wedged llama-server must not hold a slot forever, but a blanket
    // request timeout would abort legitimate long-running streamed
    // generations mid-stream. `connect_timeout` bounds only the initial
    // TCP/TLS handshake. `read_timeout` in reqwest bounds the time since
    // the *last byte was read* -- which includes the wait for the very
    // first response byte (headers included), not just inter-chunk gaps
    // during an already-started stream. For this broker that first wait
    // covers real prompt-processing time before llama-server sends
    // anything back at all, which routinely exceeds 30s for a large
    // system prompt + tools on a loaded local GPU -- especially on the
    // cold-cache warm-up path this broker itself issues. 300s is generous
    // enough for realistic prompt-processing + generation stalls without
    // being unbounded. One client, shared by the HTTP routes and the
    // reconciliation task below (Fix 5).
    let http_client = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(5))
        .read_timeout(Duration::from_secs(300))
        .build()?;

    // Fetch the real slot count first -- the Scheduler is constructed
    // once we know it, then seeded from the same response.
    let slots = aivyx_broker::llama_client::fetch_slots(&http_client, &config.llama_server_url)
        .await
        .map_err(|e| anyhow::anyhow!("failed to reach llama-server at startup: {e}"))?;
    let scheduler = aivyx_broker::Scheduler::new(slots.len() as u32);
    for slot in &slots {
        if slot.is_processing {
            scheduler.seed_busy(slot.id);
        }
    }
    tracing::info!(
        num_slots = slots.len(),
        "seeded scheduler from real llama-server /slots"
    );

    let kv_store = aivyx_kvcache::LlamaServerSlotStore::open(
        &config.kvcache_store_path,
        config.llama_server_url.clone(),
        config.kvcache_max_bytes,
    )?;
    let gpu_lock = aivyx_broker::GpuLock::new(Duration::from_secs(config.gpu_lock_max_hold_secs));
    let state = aivyx_broker::server::AppState {
        scheduler: scheduler.clone(),
        http: http_client.clone(),
        llama_server_url: config.llama_server_url.clone(),
        kv_store: std::sync::Arc::new(kv_store),
        queue_timeout: std::time::Duration::from_secs(config.queue_timeout_secs),
        gpu_lock: gpu_lock.clone(),
        gpu_lock_queue_timeout: Duration::from_secs(config.gpu_lock_queue_timeout_secs),
    };
    let app = aivyx_broker::server::build_router(state);

    let addr = format!("127.0.0.1:{}", config.port);
    tracing::info!(%addr, llama_server_url = %config.llama_server_url, "aivyx-broker starting");
    let listener = tokio::net::TcpListener::bind(&addr).await?;

    // Background reconciliation (Fix 1): a slot seeded busy-but-unowned at
    // startup has no `ReleaseGuard` -- the broker never admitted a
    // request onto it, so nothing will ever call `release()` for it. Poll
    // llama-server's real `/slots` periodically and clear any such slot
    // once llama-server itself reports it idle, so a broker restart that
    // catches llama-server mid-generation doesn't cause permanent
    // capacity loss.
    tokio::spawn(reconcile_seeded_slots(
        scheduler,
        http_client,
        config.llama_server_url.clone(),
    ));

    // Background reap: a GPU-lock holder that crashes or disconnects
    // between its acquire and release calls has no `ReleaseGuard` (unlike
    // the chat-completions path) to release it automatically, so the lock
    // would otherwise stay held forever. Poll periodically and force-
    // release any lease that's exceeded its `max_hold` safety expiry.
    tokio::spawn(reap_expired_gpu_lock(gpu_lock));

    axum::serve(listener, app).await?;
    Ok(())
}

async fn reconcile_seeded_slots(
    scheduler: aivyx_broker::Scheduler,
    http_client: reqwest::Client,
    llama_server_url: String,
) {
    let mut interval = tokio::time::interval(RECONCILE_INTERVAL);
    // The first tick fires immediately; skip it since startup already
    // seeded from a fresh `/slots` fetch moments ago.
    interval.tick().await;
    loop {
        interval.tick().await;
        match aivyx_broker::llama_client::fetch_slots(&http_client, &llama_server_url).await {
            Ok(slots) => {
                for slot in slots {
                    if !slot.is_processing {
                        scheduler.reconcile_seeded_idle(slot.id);
                    }
                }
            }
            Err(err) => {
                tracing::warn!(error = %err, "reconcile: failed to poll llama-server /slots, will retry next interval");
            }
        }
    }
}

/// Periodically force-releases a GPU lock lease that's exceeded its
/// max_hold safety expiry -- see `GpuLock::reap_expired`'s own doc
/// comment for why this exists (a crashed client between its acquire
/// and release calls has no Drop guard to release it automatically,
/// unlike the chat-completions path's ReleaseGuard). Reuses
/// `RECONCILE_INTERVAL`: it's already short (10s) relative to any
/// reasonable `max_hold` (900s by default), so a dedicated interval
/// would add a second tunable without meaningfully improving reap
/// promptness.
async fn reap_expired_gpu_lock(gpu_lock: aivyx_broker::gpu_lock::GpuLock) {
    let mut interval = tokio::time::interval(RECONCILE_INTERVAL);
    loop {
        interval.tick().await;
        gpu_lock.reap_expired();
    }
}

use clap::Parser;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();
    let config = aivyx_broker::BrokerConfig::parse();
    let http_client = reqwest::Client::new();

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
    tracing::info!(num_slots = slots.len(), "seeded scheduler from real llama-server /slots");

    let kv_store = aivyx_kvcache::LlamaServerSlotStore::open(
        &config.kvcache_store_path,
        config.llama_server_url.clone(),
        config.kvcache_max_bytes,
    )?;
    let state = aivyx_broker::server::AppState {
        scheduler,
        http: http_client,
        llama_server_url: config.llama_server_url.clone(),
        kv_store: std::sync::Arc::new(kv_store),
        queue_timeout: std::time::Duration::from_secs(config.queue_timeout_secs),
    };
    let app = aivyx_broker::server::build_router(state);

    let addr = format!("127.0.0.1:{}", config.port);
    tracing::info!(%addr, llama_server_url = %config.llama_server_url, "aivyx-broker starting");
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

use axum::{routing::get, Router};
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

    let app = Router::new().route("/status", get(status));

    let addr = format!("127.0.0.1:{}", config.port);
    tracing::info!(%addr, llama_server_url = %config.llama_server_url, "aivyx-broker starting");
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

async fn status() -> &'static str {
    "ok"
}

use axum::{routing::get, Router};
use clap::Parser;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();
    let config = aivyx_broker::BrokerConfig::parse();

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

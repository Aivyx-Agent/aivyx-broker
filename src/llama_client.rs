use serde::Deserialize;

use crate::Scheduler;

#[derive(Debug, thiserror::Error)]
pub enum LlamaClientError {
    #[error("request to llama-server failed: {0}")]
    Request(#[from] reqwest::Error),
    #[error("failed to parse llama-server /slots response")]
    Parse,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
pub struct LlamaSlot {
    pub id: u32,
    pub is_processing: bool,
}

pub(crate) fn parse_slots(json: &serde_json::Value) -> Option<Vec<LlamaSlot>> {
    let arr = json.as_array()?;
    arr.iter()
        .map(|v| {
            Some(LlamaSlot {
                id: v.get("id")?.as_u64()? as u32,
                is_processing: v.get("is_processing")?.as_bool()?,
            })
        })
        .collect()
}

pub async fn fetch_slots(
    client: &reqwest::Client,
    base_url: &str,
) -> Result<Vec<LlamaSlot>, LlamaClientError> {
    let url = format!("{}/slots", base_url.trim_end_matches('/'));
    let resp = client.get(&url).send().await?.error_for_status()?;
    let json: serde_json::Value = resp.json().await?;
    parse_slots(&json).ok_or(LlamaClientError::Parse)
}

/// Fetches the real slot count/state from `llama-server` and seeds
/// `scheduler` accordingly (any slot reported `is_processing: true` is
/// marked busy-but-unowned, per the spec's restart-recovery rule).
/// Returns the real slot count so the caller can construct the
/// `Scheduler` with the right size before this is called for the very
/// first time -- see `main.rs`.
pub async fn seed_scheduler_from_llama_server(
    scheduler: &Scheduler,
    client: &reqwest::Client,
    base_url: &str,
) -> Result<u32, LlamaClientError> {
    let slots = fetch_slots(client, base_url).await?;
    for slot in &slots {
        if slot.is_processing {
            scheduler.seed_busy(slot.id);
        }
    }
    Ok(slots.len() as u32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[test]
    fn parses_real_slots_response_shape() {
        let body = serde_json::json!([
            {"id": 0, "id_task": 135, "is_processing": true, "n_ctx": 65536},
            {"id": 1, "id_task": -1, "is_processing": false, "n_ctx": 65536},
        ]);
        let slots = parse_slots(&body).expect("must parse a real llama-server /slots body");
        assert_eq!(slots.len(), 2);
        assert_eq!(
            slots[0],
            LlamaSlot {
                id: 0,
                is_processing: true
            }
        );
        assert_eq!(
            slots[1],
            LlamaSlot {
                id: 1,
                is_processing: false
            }
        );
    }

    #[tokio::test]
    async fn fetch_slots_hits_the_real_endpoint_path() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/slots"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!([
                {"id": 0, "is_processing": false},
            ])))
            .mount(&server)
            .await;

        let client = reqwest::Client::new();
        let slots = fetch_slots(&client, &server.uri()).await.unwrap();
        assert_eq!(
            slots,
            vec![LlamaSlot {
                id: 0,
                is_processing: false
            }]
        );
    }

    #[tokio::test]
    async fn seed_scheduler_marks_processing_slots_busy() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/slots"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!([
                {"id": 0, "is_processing": true},
                {"id": 1, "is_processing": false},
            ])))
            .mount(&server)
            .await;

        let client = reqwest::Client::new();
        let scheduler = crate::Scheduler::new(2);
        let count = seed_scheduler_from_llama_server(&scheduler, &client, &server.uri())
            .await
            .unwrap();
        assert_eq!(count, 2);

        // Slot 0 must be unavailable (busy-but-unowned); slot 1 must be
        // immediately admittable.
        let admission = scheduler
            .admit(None, std::time::Duration::from_millis(50))
            .await
            .unwrap();
        assert_eq!(admission.slot_id, 1);
    }
}

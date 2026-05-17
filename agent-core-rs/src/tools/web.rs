use anyhow::anyhow;
use serde_json::{json, Value};
use std::sync::Arc;

use crate::registry::Handler;
use crate::types::ToolSpec;

pub fn spec() -> ToolSpec {
    ToolSpec {
        name: "web_fetch".into(),
        description: "HTTP GET a URL and return the response body as text. Useful for reading docs, raw files, or API responses.".into(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "max_bytes": {"type": "integer", "description": "Truncate response to this many bytes (default 200000)", "default": 200000}
            },
            "required": ["url"]
        }),
    }
}

pub fn handler() -> Handler {
    Arc::new(|args: Value| {
        Box::pin(async move {
            let url = args["url"].as_str().ok_or_else(|| anyhow!("url required"))?.to_string();
            let max = args["max_bytes"].as_u64().unwrap_or(200_000) as usize;
            let resp = reqwest::Client::new()
                .get(&url)
                .header("user-agent", "hermes-core/0.1")
                .send()
                .await?;
            let status = resp.status();
            let body = resp.text().await?;
            let trimmed = if body.len() > max {
                format!("{}\n... [{} bytes truncated]", &body[..max], body.len() - max)
            } else {
                body
            };
            Ok(format!("HTTP {status}\n\n{trimmed}"))
        })
    })
}

use anyhow::{anyhow, Context, Result};
use futures_util::StreamExt;
use serde_json::{json, Value};

use crate::provider::StreamCb;
use crate::types::{AssistantTurn, Block, Message, ToolSpec};

pub async fn call(
    api_key: &str,
    base_url: &str,
    model: &str,
    max_tokens: u32,
    system: &str,
    messages: &[Message],
    tools: &[ToolSpec],
    mut on_delta: StreamCb,
) -> Result<AssistantTurn> {
    let url = format!("{}/v1/messages", base_url.trim_end_matches('/'));
    let api_messages: Vec<Value> = messages
        .iter()
        .map(|m| match m {
            Message::User { content } => json!({"role": "user", "content": content}),
            Message::Assistant { content } => json!({"role": "assistant", "content": content}),
        })
        .collect();

    let mut body = json!({
        "model": model,
        "max_tokens": max_tokens,
        "messages": api_messages,
        "stream": true,
    });
    if !system.is_empty() {
        body["system"] = Value::String(system.to_string());
    }
    if !tools.is_empty() {
        body["tools"] = json!(tools
            .iter()
            .map(|t| json!({
                "name": t.name,
                "description": t.description,
                "input_schema": t.input_schema,
            }))
            .collect::<Vec<_>>());
    }

    let resp = reqwest::Client::new()
        .post(&url)
        .header("x-api-key", api_key)
        .header("anthropic-version", "2023-06-01")
        .header("content-type", "application/json")
        .json(&body)
        .send()
        .await
        .context("anthropic request failed")?;

    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        return Err(anyhow!("anthropic HTTP {status}: {text}"));
    }

    let mut blocks: Vec<Block> = Vec::new();
    let mut tool_buf: Vec<String> = Vec::new();
    let mut text_buf: Vec<String> = Vec::new();
    let mut think_buf: Vec<String> = Vec::new();
    let mut block_types: Vec<String> = Vec::new();
    let mut tool_ids: Vec<String> = Vec::new();
    let mut tool_names: Vec<String> = Vec::new();
    let mut stop_reason = String::new();

    let mut stream = resp.bytes_stream();
    let mut buf = String::new();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.context("anthropic stream error")?;
        buf.push_str(&String::from_utf8_lossy(&chunk));
        while let Some(idx) = buf.find("\n\n") {
            let event = buf[..idx].to_string();
            buf = buf[idx + 2..].to_string();
            for line in event.lines() {
                let data = match line.strip_prefix("data: ") {
                    Some(d) => d,
                    None => continue,
                };
                let v: Value = match serde_json::from_str(data) {
                    Ok(v) => v,
                    Err(_) => continue,
                };
                match v["type"].as_str().unwrap_or("") {
                    "content_block_start" => {
                        let blk = &v["content_block"];
                        let ty = blk["type"].as_str().unwrap_or("").to_string();
                        block_types.push(ty.clone());
                        tool_buf.push(String::new());
                        text_buf.push(String::new());
                        think_buf.push(String::new());
                        tool_ids.push(blk["id"].as_str().unwrap_or("").to_string());
                        tool_names.push(blk["name"].as_str().unwrap_or("").to_string());
                    }
                    "content_block_delta" => {
                        let i = block_types.len().saturating_sub(1);
                        let d = &v["delta"];
                        match d["type"].as_str().unwrap_or("") {
                            "text_delta" => {
                                if let Some(t) = d["text"].as_str() {
                                    text_buf[i].push_str(t);
                                    on_delta(t);
                                }
                            }
                            "input_json_delta" => {
                                if let Some(t) = d["partial_json"].as_str() {
                                    tool_buf[i].push_str(t);
                                }
                            }
                            "thinking_delta" => {
                                if let Some(t) = d["thinking"].as_str() {
                                    think_buf[i].push_str(t);
                                }
                            }
                            _ => {}
                        }
                    }
                    "content_block_stop" => {
                        let i = block_types.len().saturating_sub(1);
                        match block_types[i].as_str() {
                            "text" => blocks.push(Block::Text { text: std::mem::take(&mut text_buf[i]) }),
                            "tool_use" => {
                                let raw = std::mem::take(&mut tool_buf[i]);
                                let input: Value = serde_json::from_str(&raw).unwrap_or(Value::Object(Default::default()));
                                blocks.push(Block::ToolUse {
                                    id: std::mem::take(&mut tool_ids[i]),
                                    name: std::mem::take(&mut tool_names[i]),
                                    input,
                                });
                            }
                            "thinking" => blocks.push(Block::Thinking { thinking: std::mem::take(&mut think_buf[i]) }),
                            _ => {}
                        }
                    }
                    "message_delta" => {
                        if let Some(r) = v["delta"]["stop_reason"].as_str() {
                            stop_reason = r.to_string();
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    Ok(AssistantTurn { blocks, stop_reason })
}

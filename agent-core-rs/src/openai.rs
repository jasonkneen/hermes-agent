use anyhow::{anyhow, Context, Result};
use futures_util::StreamExt;
use serde_json::{json, Value};
use std::collections::BTreeMap;

use crate::provider::StreamCb;
use crate::types::{AssistantTurn, Block, Message, ToolSpec};

pub async fn call(
    api_key: &str,
    base_url: &str,
    model: &str,
    system: &str,
    messages: &[Message],
    tools: &[ToolSpec],
    mut on_delta: StreamCb,
) -> Result<AssistantTurn> {
    let url = format!("{}/chat/completions", base_url.trim_end_matches('/'));
    let mut api_msgs: Vec<Value> = Vec::new();
    if !system.is_empty() {
        api_msgs.push(json!({"role": "system", "content": system}));
    }
    for m in messages {
        match m {
            Message::User { content } => {
                let mut text_parts: Vec<String> = Vec::new();
                for b in content {
                    match b {
                        Block::Text { text } => text_parts.push(text.clone()),
                        Block::ToolResult { tool_use_id, content, .. } => {
                            if !text_parts.is_empty() {
                                api_msgs.push(json!({"role": "user", "content": text_parts.join("\n")}));
                                text_parts.clear();
                            }
                            api_msgs.push(json!({
                                "role": "tool",
                                "tool_call_id": tool_use_id,
                                "content": content,
                            }));
                        }
                        _ => {}
                    }
                }
                if !text_parts.is_empty() {
                    api_msgs.push(json!({"role": "user", "content": text_parts.join("\n")}));
                }
            }
            Message::Assistant { content } => {
                let mut text_parts: Vec<String> = Vec::new();
                let mut tool_calls: Vec<Value> = Vec::new();
                for b in content {
                    match b {
                        Block::Text { text } => text_parts.push(text.clone()),
                        Block::ToolUse { id, name, input } => tool_calls.push(json!({
                            "id": id,
                            "type": "function",
                            "function": {"name": name, "arguments": input.to_string()},
                        })),
                        _ => {}
                    }
                }
                let mut msg = json!({"role": "assistant"});
                msg["content"] = if text_parts.is_empty() { Value::Null } else { Value::String(text_parts.join("")) };
                if !tool_calls.is_empty() {
                    msg["tool_calls"] = Value::Array(tool_calls);
                }
                api_msgs.push(msg);
            }
        }
    }

    let mut body = json!({
        "model": model,
        "messages": api_msgs,
        "stream": true,
    });
    if !tools.is_empty() {
        body["tools"] = json!(tools
            .iter()
            .map(|t| json!({
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.input_schema,
                },
            }))
            .collect::<Vec<_>>());
    }

    let resp = reqwest::Client::new()
        .post(&url)
        .bearer_auth(api_key)
        .header("content-type", "application/json")
        .json(&body)
        .send()
        .await
        .context("openai request failed")?;

    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        return Err(anyhow!("openai HTTP {status}: {text}"));
    }

    let mut text_acc = String::new();
    let mut calls: BTreeMap<u64, (String, String, String)> = BTreeMap::new(); // idx -> (id, name, args)
    let mut finish_reason = String::new();

    let mut stream = resp.bytes_stream();
    let mut buf = String::new();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.context("openai stream error")?;
        buf.push_str(&String::from_utf8_lossy(&chunk));
        while let Some(idx) = buf.find('\n') {
            let line = buf[..idx].trim_end_matches('\r').to_string();
            buf = buf[idx + 1..].to_string();
            let data = match line.strip_prefix("data: ") {
                Some(d) => d,
                None => continue,
            };
            if data == "[DONE]" {
                break;
            }
            let v: Value = match serde_json::from_str(data) {
                Ok(v) => v,
                Err(_) => continue,
            };
            let choice = &v["choices"][0];
            if let Some(r) = choice["finish_reason"].as_str() {
                finish_reason = r.to_string();
            }
            let delta = &choice["delta"];
            if let Some(t) = delta["content"].as_str() {
                text_acc.push_str(t);
                on_delta(t);
            }
            if let Some(arr) = delta["tool_calls"].as_array() {
                for tc in arr {
                    let i = tc["index"].as_u64().unwrap_or(0);
                    let entry = calls.entry(i).or_insert_with(|| (String::new(), String::new(), String::new()));
                    if let Some(id) = tc["id"].as_str() {
                        entry.0.push_str(id);
                    }
                    if let Some(name) = tc["function"]["name"].as_str() {
                        entry.1.push_str(name);
                    }
                    if let Some(args) = tc["function"]["arguments"].as_str() {
                        entry.2.push_str(args);
                    }
                }
            }
        }
    }

    let mut blocks: Vec<Block> = Vec::new();
    if !text_acc.is_empty() {
        blocks.push(Block::Text { text: text_acc });
    }
    for (_, (id, name, args)) in calls {
        let input: Value = serde_json::from_str(&args).unwrap_or(Value::Object(Default::default()));
        blocks.push(Block::ToolUse { id, name, input });
    }
    Ok(AssistantTurn { blocks, stop_reason: finish_reason })
}

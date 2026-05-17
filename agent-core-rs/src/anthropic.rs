use anyhow::{anyhow, Context, Result};
use futures_util::{Stream, StreamExt};
use serde_json::{json, Value};

use crate::provider::StreamCb;
use crate::types::{AssistantTurn, Block, Message, ToolSpec};

pub struct AnthropicConfig<'a> {
    pub api_key: &'a str,
    pub base_url: &'a str,
    pub model: &'a str,
    pub max_tokens: u32,
}

pub async fn call(
    cfg: AnthropicConfig<'_>,
    system: &str,
    messages: &[Message],
    tools: &[ToolSpec],
    on_delta: StreamCb,
) -> Result<AssistantTurn> {
    let url = format!("{}/v1/messages", cfg.base_url.trim_end_matches('/'));
    let api_messages: Vec<Value> = messages
        .iter()
        .map(|m| match m {
            Message::User { content } => json!({"role": "user", "content": content}),
            Message::Assistant { content } => json!({"role": "assistant", "content": content}),
        })
        .collect();

    let mut body = json!({
        "model": cfg.model,
        "max_tokens": cfg.max_tokens,
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
        .header("x-api-key", cfg.api_key)
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

    parse_stream(resp.bytes_stream().map(|r| r.map_err(anyhow::Error::from)), on_delta).await
}

pub async fn parse_stream<S>(mut stream: S, mut on_delta: StreamCb) -> Result<AssistantTurn>
where
    S: Stream<Item = Result<bytes::Bytes>> + Unpin,
{
    let mut blocks: Vec<Block> = Vec::new();
    let mut tool_buf: Vec<String> = Vec::new();
    let mut text_buf: Vec<String> = Vec::new();
    let mut think_buf: Vec<String> = Vec::new();
    let mut block_types: Vec<String> = Vec::new();
    let mut tool_ids: Vec<String> = Vec::new();
    let mut tool_names: Vec<String> = Vec::new();
    let mut stop_reason = String::new();

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
                        block_types.push(ty);
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

#[cfg(test)]
mod tests {
    use super::*;
    use futures_util::stream;

    fn run_parse(chunks: Vec<&'static [u8]>) -> AssistantTurn {
        let s = stream::iter(chunks.into_iter().map(|c| Ok(bytes::Bytes::from_static(c))));
        let cb: StreamCb = Box::new(|_: &str| {});
        tokio::runtime::Runtime::new().unwrap().block_on(parse_stream(s, cb)).unwrap()
    }

    #[test]
    fn parses_text_and_tool_use() {
        let chunks: Vec<&'static [u8]> = vec![
            b"data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
            b"data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hello \"}}\n\n",
            b"data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"world\"}}\n\n",
            b"data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
            b"data: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"tool_use\",\"id\":\"tu_1\",\"name\":\"bash\",\"input\":{}}}\n\n",
            b"data: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"command\\\":\"}}\n\n",
            b"data: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"\\\"ls\\\"}\"}}\n\n",
            b"data: {\"type\":\"content_block_stop\",\"index\":1}\n\n",
            b"data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"}}\n\n",
        ];
        let turn = run_parse(chunks);
        assert_eq!(turn.stop_reason, "tool_use");
        assert_eq!(turn.blocks.len(), 2);
        match &turn.blocks[0] {
            Block::Text { text } => assert_eq!(text, "Hello world"),
            _ => panic!("expected text"),
        }
        match &turn.blocks[1] {
            Block::ToolUse { id, name, input } => {
                assert_eq!(id, "tu_1");
                assert_eq!(name, "bash");
                assert_eq!(input["command"], "ls");
            }
            _ => panic!("expected tool_use"),
        }
    }

    #[test]
    fn parses_split_event_boundaries() {
        let chunks: Vec<&'static [u8]> = vec![
            b"data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n",
            b"\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",",
            b"\"text\":\"chunked\"}}\n\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
        ];
        let turn = run_parse(chunks);
        assert_eq!(turn.blocks.len(), 1);
        match &turn.blocks[0] {
            Block::Text { text } => assert_eq!(text, "chunked"),
            _ => panic!("expected text"),
        }
    }
}

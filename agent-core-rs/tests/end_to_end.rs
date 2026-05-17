// End-to-end test of the agent loop against a local mock HTTP server that
// speaks the Anthropic Messages SSE protocol. Exercises: request build,
// HTTP transport, SSE parser, tool dispatch, multi-turn loop, transcript
// persistence — i.e. everything the real binary does, minus the live API.

use std::sync::Arc;

use hermes_core::agent::{run, AgentConfig};
use hermes_core::provider::Provider;
use hermes_core::registry::Registry;
use hermes_core::types::{Message, ToolSpec};
use serde_json::json;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

async fn start_mock(responses: Vec<String>) -> (String, tokio::task::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let url = format!("http://{}", addr);
    let responses = Arc::new(tokio::sync::Mutex::new(responses.into_iter()));

    let handle = tokio::spawn(async move {
        loop {
            let (mut sock, _) = match listener.accept().await {
                Ok(s) => s,
                Err(_) => break,
            };
            let responses = responses.clone();
            tokio::spawn(async move {
                let mut buf = [0u8; 8192];
                let mut acc = Vec::new();
                loop {
                    let n = match sock.read(&mut buf).await {
                        Ok(0) | Err(_) => return,
                        Ok(n) => n,
                    };
                    acc.extend_from_slice(&buf[..n]);
                    if let Some(headers_end) = find_headers_end(&acc) {
                        let content_length = parse_content_length(&acc[..headers_end]);
                        if acc.len() >= headers_end + content_length {
                            break;
                        }
                    }
                }

                let body = {
                    let mut it = responses.lock().await;
                    it.next().unwrap_or_else(|| "data: {}\n\n".into())
                };
                let resp = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body.len(),
                    body
                );
                let _ = sock.write_all(resp.as_bytes()).await;
                let _ = sock.shutdown().await;
            });
        }
    });

    (url, handle)
}

fn find_headers_end(b: &[u8]) -> Option<usize> {
    b.windows(4).position(|w| w == b"\r\n\r\n").map(|i| i + 4)
}

fn parse_content_length(headers: &[u8]) -> usize {
    let s = String::from_utf8_lossy(headers);
    for line in s.lines() {
        if let Some(v) = line.to_ascii_lowercase().strip_prefix("content-length:") {
            if let Ok(n) = v.trim().parse() {
                return n;
            }
        }
    }
    0
}

fn turn_tool_use_then_text() -> Vec<String> {
    vec![
        // Turn 1: assistant calls the `ping` tool
        concat!(
            "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"tu_1\",\"name\":\"ping\",\"input\":{}}}\n\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"msg\\\":\\\"hi\\\"}\"}}\n\n",
            "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
            "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"}}\n\n",
        ).to_string(),
        // Turn 2: assistant returns final text, no tool calls — loop exits
        concat!(
            "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"pong received\"}}\n\n",
            "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
            "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"}}\n\n",
        ).to_string(),
    ]
}

fn build_registry() -> Registry {
    let mut r = Registry::new();
    r.register(
        ToolSpec {
            name: "ping".into(),
            description: "test tool".into(),
            input_schema: json!({"type": "object"}),
        },
        Arc::new(|args: serde_json::Value| {
            Box::pin(async move {
                let msg = args["msg"].as_str().unwrap_or("").to_string();
                Ok(format!("pong: {msg}"))
            })
        }),
    );
    r
}

#[tokio::test]
async fn full_loop_with_tool_call_and_transcript() {
    let (base_url, _server) = start_mock(turn_tool_use_then_text()).await;

    let provider = Provider::Anthropic {
        api_key: "test-key".into(),
        base_url,
        model: "test-model".into(),
        max_tokens: 1024,
    };
    let registry = build_registry();

    let tmp_dir = std::env::temp_dir().join(format!("hermes-e2e-{}", std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
    tokio::fs::create_dir_all(&tmp_dir).await.unwrap();
    let transcript = tmp_dir.join("messages.jsonl");

    let cfg = AgentConfig { system: "", max_iterations: 5, transcript: Some(&transcript), quiet: true };
    let final_text = run(&provider, &registry, cfg, Vec::new(), "say hi").await.unwrap();

    assert_eq!(final_text, "pong received");

    // Transcript should now contain: user(ping) → assistant(tool_use) → user(tool_result) → assistant(text)
    let loaded = hermes_core::session::load(&transcript).await.unwrap();
    assert_eq!(loaded.len(), 4, "expected 4 transcript entries, got {}", loaded.len());

    match &loaded[0] {
        Message::User { content } => match &content[0] {
            hermes_core::types::Block::Text { text } => assert_eq!(text, "say hi"),
            _ => panic!("first entry should be user text"),
        },
        _ => panic!("first entry should be user"),
    }
    match &loaded[1] {
        Message::Assistant { content } => match &content[0] {
            hermes_core::types::Block::ToolUse { name, input, .. } => {
                assert_eq!(name, "ping");
                assert_eq!(input["msg"], "hi");
            }
            _ => panic!("second entry should be tool_use"),
        },
        _ => panic!("second entry should be assistant"),
    }
    match &loaded[2] {
        Message::User { content } => match &content[0] {
            hermes_core::types::Block::ToolResult { content: c, is_error, .. } => {
                assert_eq!(c, "pong: hi");
                assert!(!is_error);
            }
            _ => panic!("third entry should be tool_result"),
        },
        _ => panic!("third entry should be user"),
    }
    match &loaded[3] {
        Message::Assistant { content } => match &content[0] {
            hermes_core::types::Block::Text { text } => assert_eq!(text, "pong received"),
            _ => panic!("fourth entry should be text"),
        },
        _ => panic!("fourth entry should be assistant"),
    }

    let _ = tokio::fs::remove_dir_all(&tmp_dir).await;
}

#[tokio::test]
async fn session_resume_loads_prior_history() {
    let tmp = std::env::temp_dir().join(format!("hermes-resume-{}", std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
    tokio::fs::create_dir_all(&tmp).await.unwrap();
    let transcript = tmp.join("messages.jsonl");

    // First run
    let (url1, _s1) = start_mock(vec![concat!(
        "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"first response\"}}\n\n",
        "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
        "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"}}\n\n",
    ).to_string()]).await;
    let p1 = Provider::Anthropic { api_key: "k".into(), base_url: url1, model: "m".into(), max_tokens: 1024 };
    let reg = Registry::new();
    run(
        &p1,
        &reg,
        AgentConfig { system: "", max_iterations: 5, transcript: Some(&transcript), quiet: true },
        Vec::new(),
        "first turn",
    ).await.unwrap();

    // Reload from disk + second run
    let history = hermes_core::session::load(&transcript).await.unwrap();
    assert_eq!(history.len(), 2);

    let (url2, _s2) = start_mock(vec![concat!(
        "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"second response\"}}\n\n",
        "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
        "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"}}\n\n",
    ).to_string()]).await;
    let p2 = Provider::Anthropic { api_key: "k".into(), base_url: url2, model: "m".into(), max_tokens: 1024 };
    let out = run(
        &p2,
        &reg,
        AgentConfig { system: "", max_iterations: 5, transcript: Some(&transcript), quiet: true },
        history,
        "second turn",
    ).await.unwrap();
    assert_eq!(out, "second response");

    let final_loaded = hermes_core::session::load(&transcript).await.unwrap();
    assert_eq!(final_loaded.len(), 4, "transcript should accumulate both turns");

    let _ = tokio::fs::remove_dir_all(&tmp).await;
}

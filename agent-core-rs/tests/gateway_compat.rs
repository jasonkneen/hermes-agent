// End-to-end test of the gateway HTTP service. Wires the gateway's Provider
// at a local mock Anthropic SSE server (no network, no real key), boots
// the axum router on an ephemeral port, then hits it with HTTP requests
// the way an OpenAI-compatible client would.

use std::sync::Arc;

use hermes_core::gateway::{router, GatewayState};
use hermes_core::provider::Provider;
use hermes_core::registry::Registry;
use serde_json::{json, Value};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

async fn start_mock_llm(canned: Vec<String>) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let url = format!("http://{addr}");
    let resps = Arc::new(tokio::sync::Mutex::new(canned.into_iter()));
    tokio::spawn(async move {
        loop {
            let (mut sock, _) = match listener.accept().await {
                Ok(s) => s,
                Err(_) => break,
            };
            let resps = resps.clone();
            tokio::spawn(async move {
                let mut buf = [0u8; 8192];
                let mut acc = Vec::new();
                loop {
                    let n = match sock.read(&mut buf).await {
                        Ok(0) | Err(_) => return,
                        Ok(n) => n,
                    };
                    acc.extend_from_slice(&buf[..n]);
                    if let Some(end) = acc.windows(4).position(|w| w == b"\r\n\r\n") {
                        let cl = String::from_utf8_lossy(&acc[..end])
                            .lines()
                            .find_map(|l| {
                                l.to_ascii_lowercase()
                                    .strip_prefix("content-length:")
                                    .map(str::trim)
                                    .map(str::to_string)
                            })
                            .and_then(|v| v.parse::<usize>().ok())
                            .unwrap_or(0);
                        if acc.len() >= end + 4 + cl {
                            break;
                        }
                    }
                }
                let body = resps.lock().await.next().unwrap_or_else(|| "data: {}\n\n".into());
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
    url
}

async fn start_gateway(provider_url: String) -> (String, tokio::task::JoinHandle<()>) {
    let provider = Provider::Anthropic {
        api_key: "test".into(),
        base_url: provider_url,
        model: "test-model".into(),
        max_tokens: 1024,
    };
    let state = GatewayState::new(Ok(provider), Registry::new(), None, "test-model".into());

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let url = format!("http://{addr}");
    let app = router(Arc::new(state));
    let handle = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    (url, handle)
}

fn end_turn_sse(text: &str) -> String {
    format!(
        concat!(
            "data: {{\"type\":\"content_block_start\",\"index\":0,\"content_block\":{{\"type\":\"text\",\"text\":\"\"}}}}\n\n",
            "data: {{\"type\":\"content_block_delta\",\"index\":0,\"delta\":{{\"type\":\"text_delta\",\"text\":\"{}\"}}}}\n\n",
            "data: {{\"type\":\"content_block_stop\",\"index\":0}}\n\n",
            "data: {{\"type\":\"message_delta\",\"delta\":{{\"stop_reason\":\"end_turn\"}}}}\n\n",
        ),
        text,
    )
}

#[tokio::test]
async fn health_endpoint_returns_ok() {
    let mock = start_mock_llm(vec![end_turn_sse("ignored")]).await;
    let (gw, _h) = start_gateway(mock).await;
    let r = reqwest::get(format!("{gw}/health")).await.unwrap();
    assert_eq!(r.status(), 200);
    let v: Value = r.json().await.unwrap();
    assert_eq!(v["status"], "ok");
}

#[tokio::test]
async fn models_endpoint_lists_configured_model() {
    let mock = start_mock_llm(vec![end_turn_sse("ignored")]).await;
    let (gw, _h) = start_gateway(mock).await;
    let r = reqwest::get(format!("{gw}/v1/models")).await.unwrap();
    assert_eq!(r.status(), 200);
    let v: Value = r.json().await.unwrap();
    assert_eq!(v["object"], "list");
    assert_eq!(v["data"][0]["id"], "test-model");
    assert_eq!(v["data"][0]["object"], "model");
}

#[tokio::test]
async fn chat_completions_nonstream_returns_openai_envelope() {
    let mock = start_mock_llm(vec![end_turn_sse("pong")]).await;
    let (gw, _h) = start_gateway(mock).await;
    let client = reqwest::Client::new();
    let r = client
        .post(format!("{gw}/v1/chat/completions"))
        .json(&json!({
            "messages": [{"role": "user", "content": "ping"}],
            "stream": false,
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    let v: Value = r.json().await.unwrap();
    assert_eq!(v["object"], "chat.completion");
    assert!(v["id"].as_str().unwrap().starts_with("chatcmpl-"));
    assert_eq!(v["choices"][0]["message"]["role"], "assistant");
    assert_eq!(v["choices"][0]["message"]["content"], "pong");
    assert_eq!(v["choices"][0]["finish_reason"], "stop");
}

#[tokio::test]
async fn chat_completions_400_on_empty_messages() {
    let mock = start_mock_llm(vec![]).await;
    let (gw, _h) = start_gateway(mock).await;
    let client = reqwest::Client::new();
    let r = client
        .post(format!("{gw}/v1/chat/completions"))
        .json(&json!({"messages": [], "stream": false}))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 400);
    let v: Value = r.json().await.unwrap();
    assert!(v["error"]["message"].as_str().unwrap().contains("messages"));
}

#[tokio::test]
async fn chat_completions_400_when_no_user_message() {
    let mock = start_mock_llm(vec![]).await;
    let (gw, _h) = start_gateway(mock).await;
    let client = reqwest::Client::new();
    let r = client
        .post(format!("{gw}/v1/chat/completions"))
        .json(&json!({"messages": [{"role": "system", "content": "hi"}], "stream": false}))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 400);
}

#[tokio::test]
async fn chat_completions_stream_emits_chunks_then_done() {
    let mock = start_mock_llm(vec![end_turn_sse("hello")]).await;
    let (gw, _h) = start_gateway(mock).await;
    let client = reqwest::Client::new();
    let r = client
        .post(format!("{gw}/v1/chat/completions"))
        .json(&json!({
            "messages": [{"role": "user", "content": "say hello"}],
            "stream": true,
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);

    let body = r.text().await.unwrap();
    let lines: Vec<&str> = body.lines().filter(|l| l.starts_with("data: ")).collect();
    assert!(!lines.is_empty(), "expected SSE data lines");

    let mut saw_role = false;
    let mut saw_content = false;
    let mut saw_done = false;
    let mut saw_stop = false;
    for line in &lines {
        let data = line.strip_prefix("data: ").unwrap();
        if data == "[DONE]" {
            saw_done = true;
            continue;
        }
        let v: Value = serde_json::from_str(data).unwrap();
        assert_eq!(v["object"], "chat.completion.chunk");
        let delta = &v["choices"][0]["delta"];
        if delta["role"] == "assistant" {
            saw_role = true;
        }
        if let Some(t) = delta["content"].as_str() {
            if !t.is_empty() && t == "hello" {
                saw_content = true;
            }
        }
        if v["choices"][0]["finish_reason"] == "stop" {
            saw_stop = true;
        }
    }
    assert!(saw_role, "missing opening role chunk");
    assert!(saw_content, "missing content delta with 'hello'");
    assert!(saw_stop, "missing finish_reason: stop");
    assert!(saw_done, "missing terminating [DONE]");
}

#[tokio::test]
async fn session_continuity_via_header_loads_history() {
    let mock = start_mock_llm(vec![
        end_turn_sse("first reply"),
        end_turn_sse("teal"),
    ]).await;
    let (gw, _h) = start_gateway(mock).await;
    let sid = format!("test-{}", std::process::id());
    let client = reqwest::Client::new();

    // Turn 1
    let r1 = client
        .post(format!("{gw}/v1/chat/completions"))
        .header("X-Hermes-Session-Id", &sid)
        .json(&json!({
            "messages": [{"role": "user", "content": "remember teal"}],
            "stream": false,
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(r1.status(), 200);

    // Turn 2 — gateway should load the prior transcript and pass it
    let r2 = client
        .post(format!("{gw}/v1/chat/completions"))
        .header("X-Hermes-Session-Id", &sid)
        .json(&json!({
            "messages": [{"role": "user", "content": "what color"}],
            "stream": false,
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(r2.status(), 200);
    let v: Value = r2.json().await.unwrap();
    assert_eq!(v["choices"][0]["message"]["content"], "teal");

    // Verify the transcript actually accumulated 4 messages on disk
    let path = hermes_core::session::transcript_path(&sid);
    let loaded = hermes_core::session::load(&path).await.unwrap();
    assert_eq!(loaded.len(), 4, "expected 4 transcript entries (2 user, 2 assistant)");
    let _ = tokio::fs::remove_file(&path).await;
}

#[tokio::test]
async fn auth_rejects_request_without_bearer_when_key_set() {
    let mock = start_mock_llm(vec![end_turn_sse("ok")]).await;
    let provider = Provider::Anthropic {
        api_key: "test".into(),
        base_url: mock,
        model: "test-model".into(),
        max_tokens: 1024,
    };
    let state = GatewayState::new(Ok(provider), Registry::new(), Some("topsecret".into()), "test-model".into());
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let gw = format!("http://{addr}");
    let app = router(Arc::new(state));
    let _h = tokio::spawn(async move { axum::serve(listener, app).await.unwrap(); });

    let client = reqwest::Client::new();
    let r = client
        .post(format!("{gw}/v1/chat/completions"))
        .json(&json!({"messages": [{"role": "user", "content": "hi"}]}))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 401);

    let r = client
        .post(format!("{gw}/v1/chat/completions"))
        .header("Authorization", "Bearer topsecret")
        .json(&json!({"messages": [{"role": "user", "content": "hi"}]}))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
}

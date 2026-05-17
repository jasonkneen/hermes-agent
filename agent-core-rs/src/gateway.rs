// OpenAI-compatible HTTP gateway service. Speaks the same wire format as
// `gateway/platforms/api_server.py`:
//   - POST /v1/chat/completions    (OpenAI body, stream: bool)
//   - GET  /v1/models
//   - GET  /health
// Session continuity via `X-Hermes-Session-Id` header — when present, prior
// turns load from $HERMES_HOME/sessions/<id>/messages.jsonl before the new
// user message is appended.

use anyhow::{anyhow, Result};
use axum::{
    extract::{Json, State},
    http::{HeaderMap, StatusCode},
    response::{sse::Event, IntoResponse, Sse},
    routing::{get, post},
    Router,
};
use futures_util::StreamExt;
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::Mutex;

use crate::agent::{self, AgentConfig};
use crate::provider::{Provider, StreamCb};
use crate::registry::Registry;
use crate::session;
use crate::types::{Block, Message};

pub struct GatewayState {
    pub provider: Option<Provider>,
    pub provider_error: Option<String>,
    pub registry: Arc<Registry>,
    pub api_key: Option<String>,
    pub model_name: String,
    session_locks: Mutex<HashMap<String, Arc<Mutex<()>>>>,
}

impl GatewayState {
    pub fn new(
        provider: std::result::Result<Provider, String>,
        registry: Registry,
        api_key: Option<String>,
        model_name: String,
    ) -> Self {
        let (provider, provider_error) = match provider {
            Ok(p) => (Some(p), None),
            Err(e) => (None, Some(e)),
        };
        Self {
            provider,
            provider_error,
            registry: Arc::new(registry),
            api_key,
            model_name,
            session_locks: Mutex::new(HashMap::new()),
        }
    }

    async fn lock_for(&self, session_id: &str) -> Arc<Mutex<()>> {
        let mut map = self.session_locks.lock().await;
        map.entry(session_id.to_string()).or_insert_with(|| Arc::new(Mutex::new(()))).clone()
    }
}

pub fn router(state: Arc<GatewayState>) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(state)
}

async fn health() -> impl IntoResponse {
    Json(json!({"status": "ok"}))
}

async fn models(State(state): State<Arc<GatewayState>>) -> impl IntoResponse {
    Json(json!({
        "object": "list",
        "data": [{
            "id": state.model_name,
            "object": "model",
            "owned_by": "hermes-core",
        }],
    }))
}

#[derive(Deserialize)]
struct ChatRequest {
    messages: Vec<Value>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    model: Option<String>,
}

fn auth_check(headers: &HeaderMap, expected: &Option<String>) -> std::result::Result<(), (StatusCode, Json<Value>)> {
    let Some(want) = expected else { return Ok(()) };
    let got = headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .unwrap_or("");
    if got == want {
        Ok(())
    } else {
        Err(err(StatusCode::UNAUTHORIZED, "Unauthorized"))
    }
}

fn err(status: StatusCode, message: &str) -> (StatusCode, Json<Value>) {
    (status, Json(json!({"error": {"message": message, "type": "invalid_request_error"}})))
}

fn parse_messages(arr: &[Value]) -> (String, Vec<Message>, Option<String>) {
    let mut system = String::new();
    let mut history: Vec<Message> = Vec::new();

    for m in arr {
        let role = m["role"].as_str().unwrap_or("");
        let content = m["content"].as_str().unwrap_or("").to_string();
        match role {
            "system" => {
                if !system.is_empty() {
                    system.push('\n');
                }
                system.push_str(&content);
            }
            "user" => history.push(Message::User { content: vec![Block::Text { text: content }] }),
            "assistant" => history.push(Message::Assistant { content: vec![Block::Text { text: content }] }),
            _ => {}
        }
    }

    // Pop the last user message — it becomes `user_input` for the agent.
    let last_user_idx = history.iter().rposition(|m| matches!(m, Message::User { .. }));
    let last_user = last_user_idx.and_then(|i| {
        if let Message::User { content } = history.remove(i) {
            if let Some(Block::Text { text }) = content.into_iter().next() {
                Some(text)
            } else {
                None
            }
        } else {
            None
        }
    });

    (system, history, last_user)
}

async fn chat_completions(
    State(state): State<Arc<GatewayState>>,
    headers: HeaderMap,
    Json(body): Json<ChatRequest>,
) -> axum::response::Response {
    if let Err(e) = auth_check(&headers, &state.api_key) {
        return e.into_response();
    }
    if body.messages.is_empty() {
        return err(StatusCode::BAD_REQUEST, "Missing or invalid 'messages' field").into_response();
    }

    let (system, prior_history, last_user) = parse_messages(&body.messages);
    let Some(user_input) = last_user else {
        return err(StatusCode::BAD_REQUEST, "No user message found in messages").into_response();
    };

    let session_id = headers
        .get("x-hermes-session-id")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty());

    let (history, transcript_path) = match &session_id {
        Some(id) => {
            let path = session::transcript_path(id);
            let h = session::load(&path).await.unwrap_or_default();
            (h, Some(path))
        }
        None => (prior_history, None),
    };

    let completion_id = format!("chatcmpl-{}", random_id(29));
    let created = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
    let model_name = body.model.unwrap_or_else(|| state.model_name.clone());

    let lock = match &session_id {
        Some(id) => Some(state.lock_for(id).await),
        None => None,
    };

    if state.provider.is_none() {
        let msg = state
            .provider_error
            .clone()
            .unwrap_or_else(|| "LLM provider not configured".into());
        return err(StatusCode::SERVICE_UNAVAILABLE, &msg).into_response();
    }

    if body.stream {
        stream_response(state, history, user_input, system, transcript_path, completion_id, model_name, created, lock).await.into_response()
    } else {
        match nonstream_response(state, history, user_input, system, transcript_path, completion_id, model_name, created, lock).await {
            Ok(resp) => Json(resp).into_response(),
            Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, &format!("agent error: {e}")).into_response(),
        }
    }
}

#[allow(clippy::too_many_arguments)]
async fn nonstream_response(
    state: Arc<GatewayState>,
    history: Vec<Message>,
    user_input: String,
    system: String,
    transcript_path: Option<std::path::PathBuf>,
    completion_id: String,
    model_name: String,
    created: u64,
    lock: Option<Arc<Mutex<()>>>,
) -> Result<Value> {
    let _guard = match &lock {
        Some(l) => Some(l.lock().await),
        None => None,
    };
    let provider = state.provider.as_ref().ok_or_else(|| anyhow!("LLM provider not configured"))?;
    let text = agent::run(
        provider,
        &state.registry,
        AgentConfig { system: &system, max_iterations: 20, transcript: transcript_path.as_deref(), quiet: true },
        history,
        &user_input,
    )
    .await?;

    Ok(json!({
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": model_name,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop",
        }],
    }))
}

#[allow(clippy::too_many_arguments)]
async fn stream_response(
    state: Arc<GatewayState>,
    history: Vec<Message>,
    user_input: String,
    system: String,
    transcript_path: Option<std::path::PathBuf>,
    completion_id: String,
    model_name: String,
    created: u64,
    lock: Option<Arc<Mutex<()>>>,
) -> Sse<impl futures_util::Stream<Item = std::result::Result<Event, std::convert::Infallible>>> {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<Event>();

    let id_for_role = completion_id.clone();
    let model_for_role = model_name.clone();
    let _ = tx.send(Event::default().data(json!({
        "id": id_for_role,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_for_role,
        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": null}],
    }).to_string()));

    let id_for_close = completion_id.clone();
    let model_for_close = model_name.clone();
    let tx_for_cb = tx.clone();
    let cb_id = completion_id.clone();
    let cb_model = model_name.clone();

    let cb: StreamCb = Arc::new(move |delta_text: &str| {
        let payload = json!({
            "id": cb_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": cb_model,
            "choices": [{"index": 0, "delta": {"content": delta_text}, "finish_reason": null}],
        });
        let _ = tx_for_cb.send(Event::default().data(payload.to_string()));
    });

    tokio::spawn(async move {
        let _guard = match &lock {
            Some(l) => Some(l.lock().await),
            None => None,
        };
        let Some(provider) = state.provider.as_ref() else {
            let _ = tx.send(Event::default().data(json!({
                "error": {"message": "LLM provider not configured", "type": "service_unavailable"}
            }).to_string()));
            let _ = tx.send(Event::default().data("[DONE]"));
            return;
        };
        let result = agent::run_with_stream(
            provider,
            &state.registry,
            AgentConfig {
                system: &system,
                max_iterations: 20,
                transcript: transcript_path.as_deref(),
                quiet: true,
            },
            history,
            &user_input,
            cb,
        )
        .await;

        let final_delta = match result {
            Ok(_) => json!({}),
            Err(e) => json!({"content": format!("\n[error: {e}]")}),
        };
        let _ = tx.send(Event::default().data(json!({
            "id": id_for_close,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_for_close,
            "choices": [{"index": 0, "delta": final_delta, "finish_reason": "stop"}],
        }).to_string()));
        let _ = tx.send(Event::default().data("[DONE]"));
    });

    let stream = tokio_stream::wrappers::UnboundedReceiverStream::new(rx)
        .map(Ok::<_, std::convert::Infallible>);
    Sse::new(stream)
}

fn random_id(len: usize) -> String {
    let nanos = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    let mut s = format!("{nanos:x}");
    while s.len() < len {
        s.push('a');
    }
    s.truncate(len);
    s
}

pub async fn serve(bind: &str, state: GatewayState) -> Result<()> {
    let app = router(Arc::new(state));
    let listener = tokio::net::TcpListener::bind(bind).await?;
    let addr = listener.local_addr()?;
    eprintln!("hermes-core gateway listening on http://{addr}");
    axum::serve(listener, app).await.map_err(|e| anyhow!("axum serve: {e}"))?;
    Ok(())
}

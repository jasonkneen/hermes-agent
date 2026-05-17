use anyhow::Result;
use futures_util::future::BoxFuture;

use crate::types::{AssistantTurn, Message, ToolSpec};

pub enum Provider {
    Anthropic { api_key: String, base_url: String, model: String, max_tokens: u32 },
    OpenAI { api_key: String, base_url: String, model: String },
}

pub type StreamCb = Box<dyn FnMut(&str) + Send>;

impl Provider {
    pub fn call<'a>(
        &'a self,
        system: &'a str,
        messages: &'a [Message],
        tools: &'a [ToolSpec],
        on_delta: StreamCb,
    ) -> BoxFuture<'a, Result<AssistantTurn>> {
        match self {
            Provider::Anthropic { api_key, base_url, model, max_tokens } => Box::pin(
                crate::anthropic::call(api_key, base_url, model, *max_tokens, system, messages, tools, on_delta),
            ),
            Provider::OpenAI { api_key, base_url, model } => Box::pin(
                crate::openai::call(api_key, base_url, model, system, messages, tools, on_delta),
            ),
        }
    }
}

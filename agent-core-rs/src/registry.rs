use anyhow::Result;
use futures_util::future::BoxFuture;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;

use crate::types::ToolSpec;

pub type Handler = Arc<dyn Fn(Value) -> BoxFuture<'static, Result<String>> + Send + Sync>;

pub struct Tool {
    pub spec: ToolSpec,
    pub handler: Handler,
}

#[derive(Default)]
pub struct Registry {
    tools: HashMap<String, Tool>,
}

impl Registry {
    pub fn new() -> Self {
        Self { tools: HashMap::new() }
    }

    pub fn register(&mut self, spec: ToolSpec, handler: Handler) {
        self.tools.insert(spec.name.clone(), Tool { spec, handler });
    }

    pub fn specs(&self) -> Vec<ToolSpec> {
        self.tools.values().map(|t| t.spec.clone()).collect()
    }

    pub async fn dispatch(&self, name: &str, args: Value) -> (String, bool) {
        match self.tools.get(name) {
            None => (format!("error: unknown tool '{name}'"), true),
            Some(t) => match (t.handler)(args).await {
                Ok(s) => (s, false),
                Err(e) => (format!("error: {e}"), true),
            },
        }
    }
}

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

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::anyhow;
    use serde_json::json;

    fn mk_spec(name: &str) -> ToolSpec {
        ToolSpec {
            name: name.into(),
            description: "test".into(),
            input_schema: json!({"type": "object"}),
        }
    }

    #[tokio::test]
    async fn dispatch_unknown_tool_returns_error() {
        let r = Registry::new();
        let (out, err) = r.dispatch("nope", json!({})).await;
        assert!(err);
        assert!(out.contains("unknown tool"));
    }

    #[tokio::test]
    async fn dispatch_ok_returns_handler_output() {
        let mut r = Registry::new();
        r.register(
            mk_spec("echo"),
            Arc::new(|args: Value| Box::pin(async move { Ok(args["x"].as_str().unwrap_or("").to_string()) })),
        );
        let (out, err) = r.dispatch("echo", json!({"x": "hi"})).await;
        assert!(!err);
        assert_eq!(out, "hi");
    }

    #[tokio::test]
    async fn dispatch_handler_error_is_wrapped() {
        let mut r = Registry::new();
        r.register(
            mk_spec("boom"),
            Arc::new(|_| Box::pin(async move { Err(anyhow!("kaboom")) })),
        );
        let (out, err) = r.dispatch("boom", json!({})).await;
        assert!(err);
        assert!(out.contains("kaboom"));
    }
}

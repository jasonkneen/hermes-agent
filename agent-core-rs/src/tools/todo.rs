use anyhow::anyhow;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::sync::{Arc, Mutex};

use crate::registry::Handler;
use crate::types::ToolSpec;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Todo {
    content: String,
    status: String, // "pending" | "in_progress" | "completed"
}

#[derive(Default)]
pub struct TodoStore {
    items: Mutex<Vec<Todo>>,
}

pub fn spec() -> ToolSpec {
    ToolSpec {
        name: "todo_write".into(),
        description: "Overwrite the current task list. Use to plan multi-step work and track progress. Pass the full list each call.".into(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "todos": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": {"type": "string"},
                            "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]}
                        },
                        "required": ["content", "status"]
                    }
                }
            },
            "required": ["todos"]
        }),
    }
}

pub fn handler(store: Arc<TodoStore>) -> Handler {
    Arc::new(move |args: Value| {
        let store = store.clone();
        Box::pin(async move {
            let arr = args["todos"].as_array().ok_or_else(|| anyhow!("todos must be an array"))?;
            let mut items = Vec::new();
            for v in arr {
                let content = v["content"].as_str().ok_or_else(|| anyhow!("content required"))?.to_string();
                let status = v["status"].as_str().ok_or_else(|| anyhow!("status required"))?.to_string();
                items.push(Todo { content, status });
            }
            *store.items.lock().unwrap() = items.clone();
            let rendered: Vec<String> = items
                .iter()
                .map(|t| {
                    let mark = match t.status.as_str() {
                        "completed" => "[x]",
                        "in_progress" => "[~]",
                        _ => "[ ]",
                    };
                    format!("{mark} {}", t.content)
                })
                .collect();
            Ok(format!("updated:\n{}", rendered.join("\n")))
        })
    })
}

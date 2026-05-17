use anyhow::anyhow;
use serde_json::{json, Value};
use std::sync::Arc;

use crate::registry::Handler;
use crate::types::ToolSpec;

pub fn read_spec() -> ToolSpec {
    ToolSpec {
        name: "read_file".into(),
        description: "Read a file from disk. Returns its full text contents.".into(),
        input_schema: json!({
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"]
        }),
    }
}

pub fn read_handler() -> Handler {
    Arc::new(|args: Value| {
        Box::pin(async move {
            let path = args["path"].as_str().ok_or_else(|| anyhow!("path required"))?.to_string();
            let bytes = tokio::fs::read(&path).await?;
            Ok(String::from_utf8_lossy(&bytes).into_owned())
        })
    })
}

pub fn write_spec() -> ToolSpec {
    ToolSpec {
        name: "write_file".into(),
        description: "Write text content to a file, creating parent directories as needed. Overwrites existing files.".into(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"}
            },
            "required": ["path", "content"]
        }),
    }
}

pub fn write_handler() -> Handler {
    Arc::new(|args: Value| {
        Box::pin(async move {
            let path = args["path"].as_str().ok_or_else(|| anyhow!("path required"))?.to_string();
            let content = args["content"].as_str().ok_or_else(|| anyhow!("content required"))?.to_string();
            if let Some(parent) = std::path::Path::new(&path).parent() {
                if !parent.as_os_str().is_empty() {
                    tokio::fs::create_dir_all(parent).await.ok();
                }
            }
            tokio::fs::write(&path, content.as_bytes()).await?;
            Ok(format!("wrote {} bytes to {}", content.len(), path))
        })
    })
}

pub fn edit_spec() -> ToolSpec {
    ToolSpec {
        name: "edit_file".into(),
        description: "Replace an exact string in a file with a new one. The old_string must appear exactly once unless replace_all is true.".into(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old_string": {"type": "string"},
                "new_string": {"type": "string"},
                "replace_all": {"type": "boolean", "default": false}
            },
            "required": ["path", "old_string", "new_string"]
        }),
    }
}

pub fn edit_handler() -> Handler {
    Arc::new(|args: Value| {
        Box::pin(async move {
            let path = args["path"].as_str().ok_or_else(|| anyhow!("path required"))?.to_string();
            let old = args["old_string"].as_str().ok_or_else(|| anyhow!("old_string required"))?.to_string();
            let new = args["new_string"].as_str().ok_or_else(|| anyhow!("new_string required"))?.to_string();
            let all = args["replace_all"].as_bool().unwrap_or(false);
            let original = tokio::fs::read_to_string(&path).await?;
            let count = original.matches(&old).count();
            if count == 0 {
                return Err(anyhow!("old_string not found in {path}"));
            }
            if !all && count > 1 {
                return Err(anyhow!("old_string appears {count} times in {path}; set replace_all=true or provide more context"));
            }
            let updated = if all { original.replace(&old, &new) } else { original.replacen(&old, &new, 1) };
            tokio::fs::write(&path, updated.as_bytes()).await?;
            Ok(format!("edited {path} ({count} replacement{})", if count == 1 { "" } else { "s" }))
        })
    })
}

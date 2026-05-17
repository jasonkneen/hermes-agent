use anyhow::anyhow;
use serde_json::{json, Value};
use std::sync::Arc;
use tokio::process::Command;

use crate::registry::Handler;
use crate::types::ToolSpec;

pub fn spec() -> ToolSpec {
    ToolSpec {
        name: "bash".into(),
        description: "Execute a shell command via `sh -c`. Returns stdout, stderr, and exit code. Use for filesystem inspection, build/test, git, etc.".into(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to execute"},
                "timeout_secs": {"type": "integer", "description": "Optional timeout in seconds (default 120)", "default": 120}
            },
            "required": ["command"]
        }),
    }
}

pub fn handler() -> Handler {
    Arc::new(|args: Value| {
        Box::pin(async move {
            let cmd = args["command"].as_str().ok_or_else(|| anyhow!("command must be a string"))?;
            let timeout = args["timeout_secs"].as_u64().unwrap_or(120);
            let child = Command::new("sh")
                .arg("-c")
                .arg(cmd)
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::piped())
                .spawn()?;
            let out = tokio::time::timeout(std::time::Duration::from_secs(timeout), child.wait_with_output()).await;
            let output = match out {
                Ok(Ok(o)) => o,
                Ok(Err(e)) => return Err(anyhow!("spawn error: {e}")),
                Err(_) => return Ok(format!("error: command timed out after {timeout}s")),
            };
            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);
            let code = output.status.code().map(|c| c.to_string()).unwrap_or_else(|| "signaled".into());
            Ok(format!("exit_code: {code}\n--- stdout ---\n{}\n--- stderr ---\n{}", truncate(&stdout, 30_000), truncate(&stderr, 10_000)))
        })
    })
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max { s.to_string() } else { format!("{}\n... [{} bytes truncated]", &s[..max], s.len() - max) }
}

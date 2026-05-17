use anyhow::Result;
use std::path::Path;

use crate::provider::Provider;
use crate::registry::Registry;
use crate::session;
use crate::types::{Block, Message};

pub struct AgentConfig<'a> {
    pub system: &'a str,
    pub max_iterations: usize,
    pub transcript: Option<&'a Path>,
    pub quiet: bool,
}

impl<'a> Default for AgentConfig<'a> {
    fn default() -> Self {
        Self { system: "", max_iterations: 20, transcript: None, quiet: false }
    }
}

pub async fn run(
    provider: &Provider,
    registry: &Registry,
    cfg: AgentConfig<'_>,
    mut history: Vec<Message>,
    user_input: &str,
) -> Result<String> {
    let user_msg = Message::User { content: vec![Block::Text { text: user_input.to_string() }] };
    history.push(user_msg.clone());
    if let Some(path) = cfg.transcript {
        session::append(path, &user_msg).await?;
    }

    let specs = registry.specs();
    let mut final_text = String::new();
    let quiet = cfg.quiet;

    for iter in 0..cfg.max_iterations {
        let turn = provider
            .call(
                cfg.system,
                &history,
                &specs,
                Box::new(move |delta: &str| {
                    if quiet {
                        return;
                    }
                    use std::io::Write;
                    let _ = std::io::stderr().write_all(delta.as_bytes());
                    let _ = std::io::stderr().flush();
                }),
            )
            .await?;
        if !quiet {
            eprintln!();
        }

        let assistant = Message::Assistant { content: turn.blocks.clone() };
        history.push(assistant.clone());
        if let Some(path) = cfg.transcript {
            session::append(path, &assistant).await?;
        }
        final_text = turn.final_text();

        if !turn.has_tool_calls() {
            return Ok(final_text);
        }

        let mut results: Vec<Block> = Vec::new();
        for block in &turn.blocks {
            if let Block::ToolUse { id, name, input } = block {
                if !quiet {
                    eprintln!("→ {name}({})", compact(input));
                }
                let (out, is_error) = registry.dispatch(name, input.clone()).await;
                if !quiet {
                    eprintln!(
                        "← {} ({} bytes){}",
                        name,
                        out.len(),
                        if is_error { " [error]" } else { "" }
                    );
                }
                results.push(Block::ToolResult {
                    tool_use_id: id.clone(),
                    content: out,
                    is_error,
                });
            }
        }
        let user_results = Message::User { content: results };
        history.push(user_results.clone());
        if let Some(path) = cfg.transcript {
            session::append(path, &user_results).await?;
        }

        if iter + 1 == cfg.max_iterations && !quiet {
            eprintln!("(hit max_iterations={})", cfg.max_iterations);
        }
    }

    Ok(final_text)
}

fn compact(v: &serde_json::Value) -> String {
    let s = v.to_string();
    if s.len() > 120 { format!("{}…", &s[..120]) } else { s }
}

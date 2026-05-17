mod agent;
mod anthropic;
mod openai;
mod provider;
mod registry;
mod session;
mod tools;
mod types;

use anyhow::{anyhow, Result};
use std::sync::Arc;

use crate::provider::Provider;
use crate::registry::Registry;
use crate::tools::todo::TodoStore;

const HELP: &str = r#"hermes-core — minimal headless agent

Usage:
  hermes-core [options] "<prompt>"
  echo "<prompt>" | hermes-core [options]

Options:
  --provider <anthropic|openai>     default: anthropic
  --model <id>                      default: claude-sonnet-4-5 (anthropic) / gpt-4o-mini (openai)
  --base-url <url>                  override provider base URL
  --session <id>                    persist transcript at ~/.local/share/hermes-core/sessions/<id>/messages.jsonl
  --system <text>                   system prompt
  --max-iterations <n>              default: 20
  --max-tokens <n>                  anthropic only, default: 8192

Env:
  ANTHROPIC_API_KEY, ANTHROPIC_BASE_URL
  OPENAI_API_KEY, OPENAI_BASE_URL
"#;

struct Args {
    provider: String,
    model: Option<String>,
    base_url: Option<String>,
    session: Option<String>,
    system: String,
    max_iterations: usize,
    max_tokens: u32,
    prompt: Option<String>,
}

fn parse_args() -> Result<Args> {
    let mut a = Args {
        provider: "anthropic".into(),
        model: None,
        base_url: None,
        session: None,
        system: String::new(),
        max_iterations: 20,
        max_tokens: 8192,
        prompt: None,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-h" | "--help" => {
                println!("{HELP}");
                std::process::exit(0);
            }
            "--provider" => a.provider = it.next().ok_or_else(|| anyhow!("--provider needs value"))?,
            "--model" => a.model = Some(it.next().ok_or_else(|| anyhow!("--model needs value"))?),
            "--base-url" => a.base_url = Some(it.next().ok_or_else(|| anyhow!("--base-url needs value"))?),
            "--session" => a.session = Some(it.next().ok_or_else(|| anyhow!("--session needs value"))?),
            "--system" => a.system = it.next().ok_or_else(|| anyhow!("--system needs value"))?,
            "--max-iterations" => a.max_iterations = it.next().and_then(|s| s.parse().ok()).ok_or_else(|| anyhow!("bad --max-iterations"))?,
            "--max-tokens" => a.max_tokens = it.next().and_then(|s| s.parse().ok()).ok_or_else(|| anyhow!("bad --max-tokens"))?,
            other if other.starts_with("--") => return Err(anyhow!("unknown flag: {other}")),
            _ => {
                a.prompt = Some(arg);
            }
        }
    }
    Ok(a)
}

fn build_provider(a: &Args) -> Result<Provider> {
    match a.provider.as_str() {
        "anthropic" => {
            let api_key = std::env::var("ANTHROPIC_API_KEY").map_err(|_| anyhow!("ANTHROPIC_API_KEY not set"))?;
            let base_url = a
                .base_url
                .clone()
                .or_else(|| std::env::var("ANTHROPIC_BASE_URL").ok())
                .unwrap_or_else(|| "https://api.anthropic.com".into());
            let model = a.model.clone().unwrap_or_else(|| "claude-sonnet-4-5".into());
            Ok(Provider::Anthropic { api_key, base_url, model, max_tokens: a.max_tokens })
        }
        "openai" => {
            let api_key = std::env::var("OPENAI_API_KEY").map_err(|_| anyhow!("OPENAI_API_KEY not set"))?;
            let base_url = a
                .base_url
                .clone()
                .or_else(|| std::env::var("OPENAI_BASE_URL").ok())
                .unwrap_or_else(|| "https://api.openai.com/v1".into());
            let model = a.model.clone().unwrap_or_else(|| "gpt-4o-mini".into());
            Ok(Provider::OpenAI { api_key, base_url, model })
        }
        other => Err(anyhow!("unknown provider: {other}")),
    }
}

fn build_registry() -> Registry {
    let mut r = Registry::new();
    r.register(tools::bash::spec(), tools::bash::handler());
    r.register(tools::files::read_spec(), tools::files::read_handler());
    r.register(tools::files::write_spec(), tools::files::write_handler());
    r.register(tools::files::edit_spec(), tools::files::edit_handler());
    r.register(tools::web::spec(), tools::web::handler());
    let todos = Arc::new(TodoStore::default());
    r.register(tools::todo::spec(), tools::todo::handler(todos));
    r
}

async fn read_stdin() -> Result<String> {
    use tokio::io::AsyncReadExt;
    let mut s = String::new();
    tokio::io::stdin().read_to_string(&mut s).await?;
    Ok(s)
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = parse_args()?;
    let prompt = match &args.prompt {
        Some(p) => p.clone(),
        None => {
            let s = read_stdin().await?;
            if s.trim().is_empty() {
                eprintln!("{HELP}");
                std::process::exit(2);
            }
            s
        }
    };

    let provider = build_provider(&args)?;
    let registry = build_registry();

    let transcript = args.session.as_ref().map(|id| session::transcript_path(id));
    let history = match transcript.as_deref() {
        Some(p) => session::load(p).await?,
        None => Vec::new(),
    };

    let final_text = agent::run(
        &provider,
        &registry,
        agent::AgentConfig {
            system: &args.system,
            max_iterations: args.max_iterations,
            transcript: transcript.as_deref(),
        },
        history,
        prompt.trim(),
    )
    .await?;

    println!("{final_text}");
    Ok(())
}

use anyhow::{anyhow, Result};
use std::sync::Arc;

use hermes_core::provider::Provider;
use hermes_core::registry::Registry;
use hermes_core::tools::todo::TodoStore;
use hermes_core::{agent, session, tools};

const HELP: &str = r#"hermes-core — minimal headless agent (hermes -z compatible)

Usage:
  hermes-core [options] "<prompt>"
  echo "<prompt>" | hermes-core [options]

Options (hermes CLI-compatible):
  -z, --oneshot               oneshot mode (always-on; accepted for compat)
  -m, --model <id>            model id (env: HERMES_INFERENCE_MODEL)
      --provider <name>       anthropic | openai (env: HERMES_INFERENCE_PROVIDER)
      --base-url <url>        override provider base URL
  -c, --continue [<id>]       resume named session (default: "default")
  -r, --resume <id>           alias of --continue
  -t, --toolsets <list>       comma-separated; bash,file,todo,web (default: all)
  -q, --quiet                 suppress stderr streaming + tool traces
  -v, --verbose               (accepted; default behavior is already verbose on stderr)
      --system <text>         system prompt
      --max-iterations <n>    default: 20
      --max-tokens <n>        anthropic only, default: 8192
  -h, --help                  show this help

Session storage:
  $HERMES_HOME/sessions/<id>/messages.jsonl  (default: ~/.hermes/sessions/<id>/messages.jsonl)

Env:
  HERMES_HOME, HERMES_INFERENCE_MODEL, HERMES_INFERENCE_PROVIDER
  ANTHROPIC_API_KEY, ANTHROPIC_BASE_URL
  OPENAI_API_KEY, OPENAI_BASE_URL

Output:
  stdout = final assistant response
  stderr = streaming tokens + tool traces (silenced with -q)
"#;

struct Args {
    provider: Option<String>,
    model: Option<String>,
    base_url: Option<String>,
    session: Option<String>,
    system: String,
    max_iterations: usize,
    max_tokens: u32,
    quiet: bool,
    toolsets: Option<Vec<String>>,
    prompt: Option<String>,
}

fn parse_args() -> Result<Args> {
    let mut a = Args {
        provider: None,
        model: None,
        base_url: None,
        session: None,
        system: String::new(),
        max_iterations: 20,
        max_tokens: 8192,
        quiet: false,
        toolsets: None,
        prompt: None,
    };
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < argv.len() {
        let arg = &argv[i];
        match arg.as_str() {
            "-h" | "--help" => {
                println!("{HELP}");
                std::process::exit(0);
            }
            "-z" | "--oneshot" => {
                // Always-on. Hermes -z also accepts an optional value
                // (the prompt); we support both styles.
                if let Some(next) = argv.get(i + 1) {
                    if !next.starts_with('-') && a.prompt.is_none() {
                        a.prompt = Some(next.clone());
                        i += 1;
                    }
                }
            }
            "--provider" => {
                a.provider = Some(take_val(&argv, &mut i, "--provider")?);
            }
            "-m" | "--model" => {
                a.model = Some(take_val(&argv, &mut i, "--model")?);
            }
            "--base-url" => {
                a.base_url = Some(take_val(&argv, &mut i, "--base-url")?);
            }
            "-c" | "--continue" => {
                // Optional value (matches hermes nargs='?').
                let next = argv.get(i + 1).cloned();
                a.session = Some(match next {
                    Some(v) if !v.starts_with('-') => {
                        i += 1;
                        v
                    }
                    _ => "default".into(),
                });
            }
            "-r" | "--resume" | "--session" => {
                a.session = Some(take_val(&argv, &mut i, "--resume")?);
            }
            "-t" | "--toolsets" => {
                let raw = take_val(&argv, &mut i, "--toolsets")?;
                a.toolsets = Some(
                    raw.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect(),
                );
            }
            "-q" | "--quiet" => a.quiet = true,
            "-v" | "--verbose" => {}
            "--system" => {
                a.system = take_val(&argv, &mut i, "--system")?;
            }
            "--max-iterations" => {
                a.max_iterations = take_val(&argv, &mut i, "--max-iterations")?
                    .parse()
                    .map_err(|_| anyhow!("--max-iterations must be a number"))?;
            }
            "--max-tokens" => {
                a.max_tokens = take_val(&argv, &mut i, "--max-tokens")?
                    .parse()
                    .map_err(|_| anyhow!("--max-tokens must be a number"))?;
            }
            other if other.starts_with("--") => return Err(anyhow!("unknown flag: {other}")),
            _ => {
                a.prompt = Some(arg.clone());
            }
        }
        i += 1;
    }
    Ok(a)
}

fn take_val(argv: &[String], i: &mut usize, name: &str) -> Result<String> {
    *i += 1;
    argv.get(*i).cloned().ok_or_else(|| anyhow!("{name} needs a value"))
}

fn build_provider(a: &Args) -> Result<Provider> {
    let provider = a
        .provider
        .clone()
        .or_else(|| std::env::var("HERMES_INFERENCE_PROVIDER").ok())
        .unwrap_or_else(|| "anthropic".into())
        .to_lowercase();
    let env_model = std::env::var("HERMES_INFERENCE_MODEL").ok();

    match provider.as_str() {
        "anthropic" => {
            let api_key = std::env::var("ANTHROPIC_API_KEY")
                .map_err(|_| anyhow!("ANTHROPIC_API_KEY not set"))?;
            let base_url = a
                .base_url
                .clone()
                .or_else(|| std::env::var("ANTHROPIC_BASE_URL").ok())
                .unwrap_or_else(|| "https://api.anthropic.com".into());
            let model = a.model.clone().or(env_model).unwrap_or_else(|| "claude-sonnet-4-5".into());
            Ok(Provider::Anthropic { api_key, base_url, model, max_tokens: a.max_tokens })
        }
        "openai" => {
            let api_key = std::env::var("OPENAI_API_KEY")
                .map_err(|_| anyhow!("OPENAI_API_KEY not set"))?;
            let base_url = a
                .base_url
                .clone()
                .or_else(|| std::env::var("OPENAI_BASE_URL").ok())
                .unwrap_or_else(|| "https://api.openai.com/v1".into());
            let model = a.model.clone().or(env_model).unwrap_or_else(|| "gpt-4o-mini".into());
            Ok(Provider::OpenAI { api_key, base_url, model })
        }
        other => Err(anyhow!("unknown provider: {other} (expected anthropic|openai)")),
    }
}

fn build_registry(toolsets: Option<&[String]>) -> Registry {
    let want = |set: &str| -> bool {
        match toolsets {
            None => true,
            Some(ts) => ts.iter().any(|t| t == set || t == "all" || t == "*"),
        }
    };
    let mut r = Registry::new();
    if want("bash") {
        r.register(tools::bash::spec(), tools::bash::handler());
    }
    if want("file") {
        r.register(tools::files::read_spec(), tools::files::read_handler());
        r.register(tools::files::write_spec(), tools::files::write_handler());
        r.register(tools::files::edit_spec(), tools::files::edit_handler());
    }
    if want("web") {
        r.register(tools::web::spec(), tools::web::handler());
    }
    if want("todo") {
        let todos = Arc::new(TodoStore::default());
        r.register(tools::todo::spec(), tools::todo::handler(todos));
    }
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
    let registry = build_registry(args.toolsets.as_deref());

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
            quiet: args.quiet,
        },
        history,
        prompt.trim(),
    )
    .await?;

    println!("{final_text}");
    Ok(())
}

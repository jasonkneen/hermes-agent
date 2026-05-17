use anyhow::{anyhow, Result};
use std::sync::Arc;

use hermes_core::gateway::GatewayState;
use hermes_core::provider::Provider;
use hermes_core::registry::Registry;
use hermes_core::tools::todo::TodoStore;
use hermes_core::worktree::Worktree;
use hermes_core::{agent, gateway, session, tools, VERSION};

const HELP: &str = r#"hermes-core — minimal headless agent (hermes -z / hermes gateway compatible)

Usage:
  hermes-core [oneshot-options] "<prompt>"          # 'hermes -z' equivalent
  hermes-core gateway run [--bind ADDR]             # 'hermes gateway' equivalent (HTTP service)
  hermes-core gateway start|stop|status|restart     # daemon process control
  hermes-core version                                # show version

Oneshot options (matches hermes -z surface):
  -z, --oneshot               oneshot mode (always-on; accepted for compat)
  -m, --model <id>            model id (env: HERMES_INFERENCE_MODEL)
      --provider <name>       anthropic | openai (env: HERMES_INFERENCE_PROVIDER)
      --base-url <url>        override provider base URL
  -c, --continue [<id>]       resume named session (default: "default")
  -r, --resume <id>           alias of --continue
  -t, --toolsets <list>       comma-separated; bash,file,todo,web (default: all)
  -w, --worktree              run inside an isolated git worktree
  -q, --quiet                 suppress stderr streaming + tool traces
  -v, --verbose               (accepted; default behavior is already verbose)
  -V, --version               print version and exit
      --system <text>         system prompt
      --max-iterations <n>    default: 20
      --max-tokens <n>        anthropic only, default: 8192
  -h, --help                  show this help

Gateway options:
  --bind <addr>               default: 127.0.0.1:8642

Session storage:
  $HERMES_HOME/sessions/<id>/messages.jsonl  (default: ~/.hermes/sessions/<id>/messages.jsonl)

Env:
  HERMES_HOME, HERMES_INFERENCE_MODEL, HERMES_INFERENCE_PROVIDER
  ANTHROPIC_API_KEY, ANTHROPIC_BASE_URL
  OPENAI_API_KEY, OPENAI_BASE_URL
  API_SERVER_KEY              (gateway: require Bearer token on requests)

Output (oneshot):
  stdout = final assistant response
  stderr = streaming tokens + tool traces (silenced with -q)

Output (gateway):
  POST /v1/chat/completions  — OpenAI Chat Completions format, stream: bool
  GET  /v1/models             — lists configured model
  GET  /health                — health check
  Header X-Hermes-Session-Id  — opt-in session continuity (loads transcript JSONL)
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
    worktree: bool,
    toolsets: Option<Vec<String>>,
    prompt: Option<String>,
    bind: Option<String>,
    subcommand: Subcommand,
}

#[derive(Clone, PartialEq, Eq)]
enum Subcommand {
    Oneshot,
    GatewayRun,
    GatewayStart,
    GatewayStop,
    GatewayStatus,
    GatewayRestart,
    Version,
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
        worktree: false,
        toolsets: None,
        prompt: None,
        bind: None,
        subcommand: Subcommand::Oneshot,
    };
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;

    // First pass: peek subcommand if present (a non-flag arg in {gateway, version}).
    if let Some(first) = argv.first() {
        match first.as_str() {
            "gateway" => {
                let sub = argv.get(1).map(|s| s.as_str()).unwrap_or("run");
                a.subcommand = match sub {
                    "run" => Subcommand::GatewayRun,
                    "start" => Subcommand::GatewayStart,
                    "stop" => Subcommand::GatewayStop,
                    "status" => Subcommand::GatewayStatus,
                    "restart" => Subcommand::GatewayRestart,
                    other => return Err(anyhow!("unknown gateway subcommand: {other}")),
                };
                i = if argv.get(1).is_some_and(|s| !s.starts_with('-')) { 2 } else { 1 };
            }
            "version" => {
                a.subcommand = Subcommand::Version;
                return Ok(a);
            }
            _ => {}
        }
    }

    while i < argv.len() {
        let arg = &argv[i];
        match arg.as_str() {
            "-h" | "--help" => {
                println!("{HELP}");
                std::process::exit(0);
            }
            "-V" | "--version" => {
                a.subcommand = Subcommand::Version;
                return Ok(a);
            }
            "-z" | "--oneshot" => {
                if let Some(next) = argv.get(i + 1) {
                    if !next.starts_with('-') && a.prompt.is_none() {
                        a.prompt = Some(next.clone());
                        i += 1;
                    }
                }
            }
            "--provider" => a.provider = Some(take_val(&argv, &mut i, "--provider")?),
            "-m" | "--model" => a.model = Some(take_val(&argv, &mut i, "--model")?),
            "--base-url" => a.base_url = Some(take_val(&argv, &mut i, "--base-url")?),
            "-c" | "--continue" => {
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
            "-w" | "--worktree" => a.worktree = true,
            "-q" | "--quiet" => a.quiet = true,
            "-v" | "--verbose" => {}
            "--system" => a.system = take_val(&argv, &mut i, "--system")?,
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
            "--bind" => a.bind = Some(take_val(&argv, &mut i, "--bind")?),
            // hermes -z accepts (and our pre-pass already consumed) "gateway"/"version".
            "gateway" | "version" => {}
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

fn pidfile_path() -> std::path::PathBuf {
    let base = std::env::var("HERMES_HOME")
        .ok()
        .map(std::path::PathBuf::from)
        .or_else(|| dirs::home_dir().map(|h| h.join(".hermes")))
        .unwrap_or_else(|| std::path::PathBuf::from(".hermes"));
    base.join("gateway.pid")
}

fn process_alive(pid: i32) -> bool {
    #[cfg(unix)]
    {
        unsafe { libc_kill_check(pid) }
    }
    #[cfg(not(unix))]
    {
        let _ = pid;
        false
    }
}

#[cfg(unix)]
unsafe fn libc_kill_check(pid: i32) -> bool {
    // signal 0 only checks existence + perms, doesn't deliver
    extern "C" {
        fn kill(pid: i32, sig: i32) -> i32;
    }
    kill(pid, 0) == 0
}

async fn run_gateway(args: &Args) -> Result<()> {
    // Deferred provider build: we want /health and /v1/models to work even
    // when no LLM key is configured. Chat requests will return 503 with the
    // captured error message.
    let provider = build_provider(args).map_err(|e| e.to_string());
    if let Err(e) = &provider {
        eprintln!("warning: LLM provider not configured: {e}");
        eprintln!("         /health and /v1/models will work; /v1/chat/completions returns 503.");
    }
    let registry = build_registry(args.toolsets.as_deref());
    let api_key = std::env::var("API_SERVER_KEY").ok().filter(|s| !s.is_empty());
    let model_name = args.model.clone().unwrap_or_else(|| "hermes-agent".into());

    let bind = args
        .bind
        .clone()
        .unwrap_or_else(|| "127.0.0.1:8642".into());

    // Write pidfile for `gateway stop`/`status`.
    let pid_path = pidfile_path();
    if let Some(parent) = pid_path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    std::fs::write(&pid_path, std::process::id().to_string()).ok();

    let state = GatewayState::new(provider, registry, api_key, model_name);
    let res = gateway::serve(&bind, state).await;

    // Best-effort cleanup of pidfile on graceful shutdown.
    std::fs::remove_file(&pid_path).ok();
    res
}

async fn start_gateway() -> Result<()> {
    let pid_path = pidfile_path();
    if let Ok(s) = std::fs::read_to_string(&pid_path) {
        if let Ok(pid) = s.trim().parse::<i32>() {
            if process_alive(pid) {
                return Err(anyhow!("gateway already running (pid {pid})"));
            }
        }
    }
    let exe = std::env::current_exe()?;
    let log_path = pid_path.with_file_name("gateway.log");
    let log_file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)?;
    let log_err = log_file.try_clone()?;
    let mut cmd = std::process::Command::new(&exe);
    cmd.args(["gateway", "run"])
        .stdin(std::process::Stdio::null())
        .stdout(log_file)
        .stderr(log_err);

    // Properly detach from the controlling terminal so SIGHUP on parent
    // shell exit doesn't kill us. setsid() makes the child a session leader
    // in a new process group with no controlling tty.
    #[cfg(unix)]
    unsafe {
        use std::os::unix::process::CommandExt;
        cmd.pre_exec(|| {
            if libc_setsid() < 0 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(())
        });
    }

    let child = cmd.spawn()?;
    let pid = child.id();

    // Wait briefly for the daemon to come up so we can fail loudly if it
    // died immediately (port in use, bad config, etc).
    let bind = "127.0.0.1:8642";
    let mut alive = false;
    for _ in 0..40 {
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        if reqwest::Client::new()
            .get(format!("http://{bind}/health"))
            .timeout(std::time::Duration::from_millis(200))
            .send()
            .await
            .is_ok_and(|r| r.status().is_success())
        {
            alive = true;
            break;
        }
    }
    if !alive {
        // Surface the log tail so the user sees the actual error.
        let tail = std::fs::read_to_string(&log_path).unwrap_or_default();
        let tail = tail.lines().rev().take(20).collect::<Vec<_>>();
        let tail: Vec<_> = tail.into_iter().rev().collect();
        return Err(anyhow!(
            "gateway failed to come up within 2s. Last log lines:\n{}\n(full log: {})",
            tail.join("\n"),
            log_path.display()
        ));
    }
    eprintln!("started gateway (pid {pid}), logs at {}", log_path.display());
    Ok(())
}

#[cfg(unix)]
unsafe fn libc_setsid() -> i32 {
    extern "C" {
        fn setsid() -> i32;
    }
    setsid()
}

fn stop_gateway() -> Result<()> {
    let pid_path = pidfile_path();
    let s = std::fs::read_to_string(&pid_path)
        .map_err(|_| anyhow!("no pidfile at {}; is the gateway running?", pid_path.display()))?;
    let pid: i32 = s.trim().parse().map_err(|_| anyhow!("malformed pidfile"))?;
    #[cfg(unix)]
    {
        extern "C" { fn kill(pid: i32, sig: i32) -> i32; }
        // SIGTERM = 15
        let ret = unsafe { kill(pid, 15) };
        if ret != 0 {
            return Err(anyhow!("kill(SIGTERM, {pid}) failed"));
        }
        eprintln!("sent SIGTERM to gateway (pid {pid})");
    }
    std::fs::remove_file(&pid_path).ok();
    Ok(())
}

fn status_gateway() -> Result<()> {
    let pid_path = pidfile_path();
    match std::fs::read_to_string(&pid_path) {
        Ok(s) => {
            let pid: i32 = s.trim().parse().map_err(|_| anyhow!("malformed pidfile"))?;
            if process_alive(pid) {
                println!("gateway: running (pid {pid})");
            } else {
                println!("gateway: stopped (stale pidfile)");
            }
        }
        Err(_) => println!("gateway: stopped"),
    }
    Ok(())
}

async fn run_oneshot(args: Args) -> Result<()> {
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

    let wt = if args.worktree { Some(Worktree::create()?) } else { None };
    let original_cwd = std::env::current_dir().ok();
    if let Some(w) = &wt {
        std::env::set_current_dir(&w.path)?;
        eprintln!("worktree: {} (branch {})", w.path.display(), w.branch);
    }

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

    if let Some(orig) = original_cwd {
        let _ = std::env::set_current_dir(orig);
    }
    if let Some(w) = wt {
        if let Err(e) = w.cleanup() {
            eprintln!("worktree cleanup failed: {e}");
        }
    }

    println!("{final_text}");
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = parse_args()?;
    match args.subcommand {
        Subcommand::Version => {
            println!("hermes-core {VERSION}");
            Ok(())
        }
        Subcommand::GatewayRun => run_gateway(&args).await,
        Subcommand::GatewayStart => start_gateway().await,
        Subcommand::GatewayStop => stop_gateway(),
        Subcommand::GatewayStatus => status_gateway(),
        Subcommand::GatewayRestart => {
            stop_gateway().ok();
            tokio::time::sleep(std::time::Duration::from_millis(500)).await;
            start_gateway().await
        }
        Subcommand::Oneshot => run_oneshot(args).await,
    }
}

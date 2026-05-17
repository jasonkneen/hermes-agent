# Upstream tracking — Python hermes ↔ Rust hermes-core

This file is the **source of truth** for what upstream changes we care
about. The audit script (`scripts/upstream-diff.sh`) and the CI drift
workflow (`.github/workflows/upstream-drift.yml`) read it.

The Rust port is deliberately a stripped-down reproduction of the core
agent loop. ~95% of upstream code (skills, plugins, MCP, platform
adapters, prompt builder, curator, compressor, memory, redaction,
insights, TUI, dashboard, web UI, OAuth flows, credential pool, the
many provider adapters beyond Anthropic+OpenAI…) is **not** part of
the Rust port and changes to those paths don't apply to us.

What follows is the inverse: the paths whose changes DO apply. Reviewing
this file regularly is how we stay in sync.

---

## Path mapping (Python upstream → Rust here)

Areas are ordered by likelihood-of-meaningful-change, descending.

| Area                          | Upstream paths                                                                                                            | Our file(s)                          |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------- | ------------------------------------ |
| Anthropic API wire format     | `agent/anthropic_adapter.py`                                                                                              | `src/anthropic.rs`                   |
| OpenAI-compat wire format     | `agent/codex_responses_adapter.py`, `agent/gemini_*.py` (for sanity-check on chat-completions semantics)                  | `src/openai.rs`                      |
| Agent loop (the irreducible)  | `run_agent.py` — `run_conversation()` ~line 12138, `_execute_tool_calls*` ~line 10919-11500, message-prep ~12585          | `src/agent.rs`                       |
| Tool registry / dispatch      | `tools/registry.py`, `tools/__init__.py`                                                                                  | `src/registry.rs`                    |
| Built-in tools we ship        | `tools/terminal_tool.py` (bash), `tools/file_tools.py` + `tools/file_operations.py` (read/write/edit), `tools/todo_tool.py`, `tools/web_tools.py` (the basic GET path only) | `src/tools/*.rs`                     |
| Gateway HTTP API (OpenAI)     | `gateway/platforms/api_server.py` — `_handle_chat_completions`, `_handle_responses`, `/v1/models`, `/health`, the `X-Hermes-Session-Id` / `X-Hermes-Session-Key` header semantics | `src/gateway.rs`                     |
| Session JSONL transcript      | `gateway/session.py`, `hermes_state.py` (only the on-disk row format)                                                     | `src/session.rs`                     |
| CLI flag surface (oneshot)    | `hermes_cli/main.py` — top-level argparse (the `-z`, `-m`, `--provider`, `-c`, `-r`, `-t`, `-q`, `-v`, `-w`, `-V` cluster), `hermes_cli/oneshot.py` | `src/main.rs`                        |
| Gateway subcommand surface    | `hermes_cli/main.py` — `gateway_*` add_parser blocks (~line 9832), `hermes_cli/gateway.py` (status/start/stop/install)    | `src/main.rs` (gateway dispatch)     |
| Defaults / env var names      | `agent/account_usage.py`, `hermes_cli/config.py` — `HERMES_INFERENCE_MODEL`, `HERMES_INFERENCE_PROVIDER`, `HERMES_HOME`, `API_SERVER_KEY`, `API_SERVER_PORT` | `src/main.rs`, `src/gateway.rs`      |
| Worktree behavior             | `hermes_cli/main.py` — `-w/--worktree` flag handling                                                                       | `src/worktree.rs`                    |

## Paths we DO NOT track (deliberate non-goals)

If a change only touches these, ignore it. They're the stripped layers.

- `skills/`, `optional-skills/`, `plugins/`            — extension system
- `gateway/platforms/*.py` EXCEPT `api_server.py`      — platform adapters (discord, telegram, etc.)
- `tui_gateway/`, `web/`, `website/`, `ui-tui/`         — UI layers
- `agent/prompt_builder.py`                            — scaffolding injection
- `agent/curator.py`, `agent/compressor.py`, `agent/memory_*.py`, `agent/insights.py` — optional intelligence
- `agent/redact.py`, `agent/error_classifier.py`, `agent/think_scrubber.py`           — text post-processing
- `agent/auxiliary_client.py`, `agent/plugin_llm.py`                                  — sub-agent / plugin LLM
- `agent/{bedrock,gemini_*,copilot_acp,google_*,codex_responses,nous_rate_guard,...}.py` — non-Anthropic/OpenAI providers we don't ship
- `tools/{mcp,skills,delegate,browser*,kanban,cron*,checkpoint*,clarify,memory_tool,send_message,session_search,skill_*,skills_*,vision*,voice*,tts*,transcription*,image_generation*,video_generation*,web_search_provider,homeassistant,discord,slack,...}.py` — out-of-scope tools
- `cli.py`, `mcp_serve.py`, `batch_runner.py`, `cron/`, `acp_adapter/`, `acp_registry/`, `hermes_state.py` (most of it) — adjacent products
- `tests/`                                              — Python test suite (most can't black-box us; the wire-level ones we mirror in `tests/python/`)
- `docs/`, `*.md`, `RELEASE_*.md`, `CONTRIBUTING.md`, `SECURITY.md`, `AGENTS.md`     — documentation
- `pyproject.toml`, `uv.lock`, `package*.json`, `Dockerfile`, `flake.*`, `nix/`, `setup-hermes.sh`, `scripts/`, `packaging/` — packaging / build

## Update procedure

1. `cd agent-core-rs && ./scripts/upstream-diff.sh`
   — prints commits in tracked paths since the last reviewed SHA.
2. Read the diff. For each commit:
   - If it changes wire/protocol/contract → port it to the Rust file.
   - If it's a noise refactor or affects only the Python scaffolding around a tracked function → ignore.
3. Run `cargo test --release && cargo clippy --release --all-targets -- -D warnings`.
4. Re-run the Python compat suite if you can (`pytest tests/python/test_gateway_compat.py`).
5. Bump `UPSTREAM_REV` to the new HEAD and commit. The diff window now starts from there.

## Safety nets

- `tests/gateway_compat.rs` — wire-level integration tests against a local
  mock LLM. Fails if our HTTP envelope drifts from OpenAI Chat Completions.
- `tests/end_to_end.rs` — full agent loop including JSONL transcript shape.
- `tests/python/test_gateway_compat.py` — black-box HTTP tests that pass
  against EITHER implementation. If hermes upstream and hermes-core drift,
  the same test file will start showing different results.

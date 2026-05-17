# Swap procedure: hermes gateway → hermes-core gateway

The acceptance bar: stop `hermes gateway`, start `hermes-core gateway`, and any
client that was talking to it (Open WebUI, OpenAI SDK pointing at localhost,
your own scripts) keeps working with zero changes.

Both serve the same wire surface:
- `POST /v1/chat/completions` — OpenAI Chat Completions format, `stream: bool`
- `GET /v1/models`
- `GET /health`
- `X-Hermes-Session-Id` request header → resumes a persistent JSONL transcript
- Default bind: `127.0.0.1:8642` (matches hermes's `DEFAULT_PORT`)
- Auth: `Authorization: Bearer $API_SERVER_KEY` when `API_SERVER_KEY` is set

## 1. Build

```bash
cd agent-core-rs
cargo build --release
# binary: ./target/release/hermes-core
```

Optionally drop it on $PATH or symlink it:

```bash
sudo ln -sf "$PWD/target/release/hermes-core" /usr/local/bin/hermes-core
```

## 2. Stop the current hermes gateway

```bash
hermes gateway status      # confirm it's running
hermes gateway stop        # ask hermes to stop it
# or, if it's a systemd/launchd service:
#   systemctl --user stop hermes-gateway
#   launchctl unload ~/Library/LaunchAgents/com.hermes.gateway.plist
```

Verify nothing is on port 8642:

```bash
lsof -i :8642        # should print nothing
curl -s http://127.0.0.1:8642/health   # should fail-to-connect
```

## 3. Start hermes-core gateway

Foreground (recommended for first run — you see logs in your terminal):

```bash
ANTHROPIC_API_KEY=sk-ant-... hermes-core gateway run
# logs:  hermes-core gateway listening on http://127.0.0.1:8642
```

Background (writes a pidfile at `$HERMES_HOME/gateway.pid`, default `~/.hermes/`):

```bash
ANTHROPIC_API_KEY=sk-ant-... hermes-core gateway start
hermes-core gateway status     # gateway: running (pid 12345)
hermes-core gateway stop       # SIGTERM the daemon
hermes-core gateway restart
```

Logs (background mode): `~/.hermes/gateway.log`.

## 4. Verify with the same clients

Any tool that hit `hermes gateway` should now hit `hermes-core gateway`
unchanged.

### Curl
```bash
curl -s http://127.0.0.1:8642/v1/models | jq
curl -s http://127.0.0.1:8642/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"Reply pong"}]}'
```

### OpenAI SDK
```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8642/v1", api_key="unused")
print(client.chat.completions.create(
    model="hermes-agent",
    messages=[{"role": "user", "content": "Reply pong"}],
).choices[0].message.content)
```

### Streaming + session continuity
```python
import requests
sid = "my-conversation"
r = requests.post(
    "http://127.0.0.1:8642/v1/chat/completions",
    headers={"X-Hermes-Session-Id": sid},
    json={"messages": [{"role": "user", "content": "remember teal"}], "stream": True},
    stream=True,
)
for line in r.iter_lines():
    print(line.decode())
```

The transcript lands at `~/.hermes/sessions/my-conversation/messages.jsonl` —
the same JSONL format hermes itself writes, so existing transcripts are
readable across both implementations.

## 5. Run the compatibility test suite

Black-box HTTP tests live at `tests/python/test_gateway_compat.py`. They have
no Python dependencies on either codebase — they just speak HTTP — so they
can run against EITHER gateway implementation:

```bash
# Against hermes-core (auto-spawns, runs, tears down):
GATEWAY_BIN=./target/release/hermes-core \
  ANTHROPIC_API_KEY=sk-ant-... \
  pytest tests/python/test_gateway_compat.py -v

# Same tests against the original hermes (must already be running):
GATEWAY_URL=http://127.0.0.1:8642 \
  ANTHROPIC_API_KEY=sk-ant-... \
  pytest tests/python/test_gateway_compat.py -v
```

If both implementations are wire-compatible, the same tests pass against
both. That IS the swappability proof.

## What this does NOT replace

The hermes gateway also runs platform adapters (Discord, Telegram, Slack,
WhatsApp, Signal, Matrix, etc.) in the same process. `hermes-core gateway`
is the OpenAI-compatible HTTP surface ONLY. If you were using
`hermes gateway` for platform bots, you'll need either:

1. Keep `hermes gateway` for the platform bots, and only point external HTTP
   clients (Open WebUI, custom apps) at `hermes-core gateway` on a different
   port.
2. Run platform adapters separately as thin webhook translators that POST to
   `hermes-core`'s `/v1/chat/completions`.

The rest of the hermes CLI (`hermes chat`, `hermes auth`, `hermes setup`,
etc.) is independent and unaffected — it talks directly to the LLM, not to
the gateway.

## Existing hermes Python tests

Most live tests in `tests/gateway/` import Python internals
(`from gateway.platforms.api_server import ...`) and exercise the Python
class API directly — they can't black-box `hermes-core`. The tests that CAN
run against either implementation are wire-level (HTTP request/response)
and are collected in `agent-core-rs/tests/python/test_gateway_compat.py`.
Pass it the binary or the URL.

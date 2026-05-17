"""Black-box gateway compatibility tests.

Runs against ANY implementation of the OpenAI-compatible gateway:
  - hermes (Python): `hermes gateway run`
  - hermes-core (Rust): `hermes-core gateway run`

The acceptance bar is "swap one for the other and the OpenAI SDK still works."
Tests speak HTTP only — no Python internals from either codebase.

Usage:
    # Auto-start hermes-core gateway, run tests, stop:
    GATEWAY_BIN=./target/release/hermes-core pytest tests/python/test_gateway_compat.py

    # Same tests against an already-running gateway on a non-default port:
    GATEWAY_URL=http://localhost:8642 pytest tests/python/test_gateway_compat.py

    # Run them against the original hermes for parity verification:
    GATEWAY_BIN=hermes pytest tests/python/test_gateway_compat.py

Requires: requests, pytest. The gateway binary must be on $PATH or pointed at
by GATEWAY_BIN. An ANTHROPIC_API_KEY or OPENAI_API_KEY must be set so the
gateway can reach a real LLM (or point at a mock via base_url).
"""
import json
import os
import signal
import subprocess
import time
from contextlib import contextmanager

import pytest
import requests


PORT = int(os.environ.get("GATEWAY_PORT", "8642"))
BASE_URL = os.environ.get("GATEWAY_URL", f"http://127.0.0.1:{PORT}")
GATEWAY_BIN = os.environ.get("GATEWAY_BIN")  # if set, we spawn it ourselves


@contextmanager
def spawn_gateway():
    """Boot the gateway binary if GATEWAY_BIN is set; otherwise assume one is
    already listening at BASE_URL.
    """
    if not GATEWAY_BIN:
        # Sanity-check that something is already up.
        for _ in range(20):
            try:
                requests.get(f"{BASE_URL}/health", timeout=0.5)
                break
            except requests.exceptions.ConnectionError:
                time.sleep(0.1)
        yield
        return

    proc = subprocess.Popen(
        [GATEWAY_BIN, "gateway", "run", "--bind", f"127.0.0.1:{PORT}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        for _ in range(50):  # up to 5 seconds
            try:
                r = requests.get(f"{BASE_URL}/health", timeout=0.5)
                if r.status_code == 200:
                    break
            except requests.exceptions.ConnectionError:
                time.sleep(0.1)
        else:
            stderr = proc.stderr.read().decode("utf-8", "replace")
            raise RuntimeError(f"gateway didn't come up: {stderr}")
        yield
    finally:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


@pytest.fixture(scope="session", autouse=True)
def _gw():
    with spawn_gateway():
        yield


# --- /health ---------------------------------------------------------------


def test_health_returns_200_with_status_ok():
    r = requests.get(f"{BASE_URL}/health")
    assert r.status_code == 200
    body = r.json()
    assert body.get("status") == "ok"


# --- /v1/models ------------------------------------------------------------


def test_models_lists_at_least_one_model():
    r = requests.get(f"{BASE_URL}/v1/models")
    assert r.status_code == 200
    body = r.json()
    assert body.get("object") == "list"
    data = body.get("data") or []
    assert len(data) >= 1
    assert "id" in data[0]
    assert data[0].get("object") == "model"


# --- /v1/chat/completions: validation --------------------------------------


def test_chat_completions_400_on_empty_messages():
    r = requests.post(
        f"{BASE_URL}/v1/chat/completions",
        json={"messages": [], "stream": False},
    )
    assert r.status_code == 400
    body = r.json()
    assert "error" in body


def test_chat_completions_400_on_no_user_message():
    r = requests.post(
        f"{BASE_URL}/v1/chat/completions",
        json={"messages": [{"role": "system", "content": "hi"}], "stream": False},
    )
    assert r.status_code == 400


# --- /v1/chat/completions: non-stream --------------------------------------


@pytest.mark.skipif(
    not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY")),
    reason="needs a real LLM key to exercise end-to-end",
)
def test_chat_completions_nonstream_returns_openai_envelope():
    r = requests.post(
        f"{BASE_URL}/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "Reply with the single word: pong"}],
            "stream": False,
        },
        timeout=60,
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["object"] == "chat.completion"
    assert body["id"].startswith("chatcmpl-")
    assert "created" in body
    assert "model" in body
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert isinstance(body["choices"][0]["message"]["content"], str)
    assert body["choices"][0]["finish_reason"] == "stop"


# --- /v1/chat/completions: stream ------------------------------------------


@pytest.mark.skipif(
    not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY")),
    reason="needs a real LLM key to exercise end-to-end",
)
def test_chat_completions_stream_emits_chunks_and_done():
    r = requests.post(
        f"{BASE_URL}/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "Reply with the single word: pong"}],
            "stream": True,
        },
        stream=True,
        timeout=60,
    )
    assert r.status_code == 200

    saw_role = False
    saw_content = False
    saw_done = False
    for raw in r.iter_lines():
        line = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else raw
        if not line or not line.startswith("data: "):
            continue
        data = line[len("data: "):].strip()
        if data == "[DONE]":
            saw_done = True
            break
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            continue
        assert chunk.get("object") == "chat.completion.chunk"
        delta = chunk["choices"][0].get("delta", {})
        if delta.get("role") == "assistant":
            saw_role = True
        if isinstance(delta.get("content"), str) and delta["content"]:
            saw_content = True

    assert saw_role, "expected an opening 'role: assistant' chunk"
    assert saw_content, "expected at least one content delta"
    assert saw_done, "expected terminating [DONE] event"


# --- session continuity via X-Hermes-Session-Id ----------------------------


@pytest.mark.skipif(
    not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY")),
    reason="needs a real LLM key to exercise end-to-end",
)
def test_session_id_header_accepts_and_persists():
    sid = f"compat-test-{os.getpid()}-{int(time.time())}"
    # Turn 1
    r1 = requests.post(
        f"{BASE_URL}/v1/chat/completions",
        headers={"X-Hermes-Session-Id": sid},
        json={
            "messages": [{"role": "user", "content": "My favorite color is teal. Say 'ok'."}],
            "stream": False,
        },
        timeout=60,
    )
    assert r1.status_code == 200, r1.text
    # Turn 2: agent should see turn 1 in history.
    r2 = requests.post(
        f"{BASE_URL}/v1/chat/completions",
        headers={"X-Hermes-Session-Id": sid},
        json={
            "messages": [{"role": "user", "content": "What color did I just mention? Reply with only the color."}],
            "stream": False,
        },
        timeout=60,
    )
    assert r2.status_code == 200, r2.text
    out = r2.json()["choices"][0]["message"]["content"].lower()
    assert "teal" in out, f"session continuity broken; got: {out!r}"


# --- auth ------------------------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("API_SERVER_KEY"),
    reason="auth tests only run when API_SERVER_KEY is configured on the gateway",
)
def test_chat_completions_401_without_bearer():
    r = requests.post(
        f"{BASE_URL}/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}], "stream": False},
    )
    assert r.status_code == 401


@pytest.mark.skipif(
    not os.environ.get("API_SERVER_KEY"),
    reason="auth tests only run when API_SERVER_KEY is configured on the gateway",
)
def test_chat_completions_200_with_correct_bearer():
    key = os.environ["API_SERVER_KEY"]
    r = requests.post(
        f"{BASE_URL}/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={"messages": [{"role": "user", "content": "Reply ok"}], "stream": False},
        timeout=60,
    )
    assert r.status_code == 200, r.text

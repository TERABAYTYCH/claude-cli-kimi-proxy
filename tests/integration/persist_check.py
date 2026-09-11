#!/usr/bin/env python3
"""Persist check: the session map must survive a proxy process restart.

Turn 1 starts a fresh delegate conversation, Turn 2 resumes it in-memory,
the server process is then killed and restarted, and Turn 3 must resume
from the persisted map (resume:true, small delta, system prompt present)
instead of replaying the full history.
"""
import json
import os
import pathlib
import signal
import socket
import subprocess
import sys
import time
import urllib.request

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
LOG_FILE = PROJECT_ROOT / "logs" / "app.log"
SESSIONS_FILE = pathlib.Path.home() / ".claude-max-api-proxy-sessions.json"
SERVER_LOG = PROJECT_ROOT / "tests" / "integration" / "persist_check.server.log"

MODEL = "claude-haiku-4-5"
KEY = f"persist-check-{int(time.time() * 1000)}"
TOOLS = [{
    "type": "function",
    "function": {
        "name": "Read",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
}]


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def base_url(port: int) -> str:
    return f"http://127.0.0.1:{port}/v1/chat/completions"


def post(port: int, body: dict, timeout: int = 180) -> dict:
    req = urllib.request.Request(
        base_url(port),
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def health(port: int, timeout: int = 30) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                if r.status == 200:
                    return
        except Exception:
            pass
        time.sleep(0.2)
    raise RuntimeError("server did not become healthy")


def build_server_env() -> dict:
    env = os.environ.copy()
    env["PATH"] = "/root/.nvm/versions/node/v24.18.0/bin:" + env.get("PATH", "")
    return env


def start_server(port: int) -> subprocess.Popen:
    SERVER_LOG.parent.mkdir(parents=True, exist_ok=True)
    env = build_server_env()
    proc = subprocess.Popen(
        ["node", str(PROJECT_ROOT / "dist" / "server" / "standalone.js"), str(port)],
        cwd=PROJECT_ROOT,
        env=env,
        stdout=open(SERVER_LOG, "a"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    health(port)
    return proc


def stop_server(proc: subprocess.Popen) -> None:
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def read_log_lines() -> list[str]:
    if not LOG_FILE.exists():
        return []
    with open(LOG_FILE, "r", encoding="utf-8", errors="replace") as f:
        return f.readlines()


def find_prepared_entries(key: str) -> list[dict]:
    entries = []
    for line in read_log_lines():
        if "[ChatCompletions] Request prepared" not in line:
            continue
        # Log format: [ts] [LEVEL] message {...meta}
        # The JSON metadata is the last {...} block on the line.
        brace = line.rfind("{")
        if brace == -1:
            continue
        try:
            meta = json.loads(line[brace:])
        except Exception:
            continue
        if meta.get("sessionKey") == key:
            entries.append(meta)
    return entries


def run_turn(port: int, turn_idx: int, messages: list[dict]) -> dict:
    print(f"Turn {turn_idx} → {KEY}")
    return post(port, {
        "model": MODEL,
        "stream": False,
        "user": KEY,
        "messages": messages,
        "tools": TOOLS,
    })


def main() -> int:
    port = free_port()
    print(f"Using port {port}")

    # Clean state from any previous aborted run
    if SESSIONS_FILE.exists():
        SESSIONS_FILE.unlink()

    # Build once, then run two server lifetimes
    env = build_server_env()
    print("Building...")
    subprocess.run(["npm", "run", "build"], cwd=PROJECT_ROOT, env=env, check=True)

    proc1 = start_server(port)
    try:
        # Turn 1: fresh conversation, should trigger a Read tool call
        u1 = "Read the file README.md with the Read tool, then say T1."
        run_turn(port, 1, [{"role": "user", "content": u1}])

        # Fabricate the history Kimi would append after the tool executes locally.
        # We only need the session to be persisted; content can be minimal.
        turn2_messages = [
            {"role": "user", "content": u1},
            {"role": "assistant", "content": None, "tool_calls": [{
                "id": "call_t1", "type": "function",
                "function": {"name": "Read", "arguments": '{"path":"README.md"}'},
            }]},
            {"role": "tool", "content": "[README contents omitted]"},
            {"role": "user", "content": "Now say T2 only."},
        ]
        run_turn(port, 2, turn2_messages)
    finally:
        stop_server(proc1)
        print("Server stopped for restart")

    # Restart server: the persisted session map must be reloaded
    proc2 = start_server(port)
    try:
        turn3_messages = [
            {"role": "user", "content": u1},
            {"role": "assistant", "content": None, "tool_calls": [{
                "id": "call_t1", "type": "function",
                "function": {"name": "Read", "arguments": '{"path":"README.md"}'},
            }]},
            {"role": "tool", "content": "[README contents omitted]"},
            {"role": "assistant", "content": "T2"},
            {"role": "user", "content": "Now say T3 only."},
        ]
        run_turn(port, 3, turn3_messages)
    finally:
        stop_server(proc2)

    # Verify log entries
    entries = find_prepared_entries(KEY)
    print(f"Found {len(entries)} Request prepared entries for {KEY}")
    if len(entries) < 3:
        print("FAIL: expected at least 3 entries")
        return 1

    e1, e2, e3 = entries[0], entries[1], entries[2]
    ok = True

    def check(name: str, cond: bool, detail: str) -> None:
        nonlocal ok
        status = "OK" if cond else "FAIL"
        print(f"{status}: {name} — {detail}")
        if not cond:
            ok = False

    check("turn1 resume", e1.get("resume") is False,
          f"resume={e1.get('resume')}")
    check("turn2 resume", e2.get("resume") is True,
          f"resume={e2.get('resume')}")
    check("turn3 resume", e3.get("resume") is True,
          f"resume={e3.get('resume')}")
    check("turn3 hasSystemPrompt", e3.get("hasSystemPrompt") is True,
          f"hasSystemPrompt={e3.get('hasSystemPrompt')}")
    check("turn3 prompt is delta", e3.get("promptChars", 999_999) < 2000,
          f"promptChars={e3.get('promptChars')}")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

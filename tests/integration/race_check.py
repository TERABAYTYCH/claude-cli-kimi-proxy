#!/usr/bin/env python3
"""Race check: emulate Kimi's instant follow-up turn.

Turn 1 (streaming, tools): read SSE to the end, then IMMEDIATELY (~0ms pause)
send turn 2 with the same key — exactly what Kimi web does after executing
the tool locally.

Without the lock-wait fix, turn 2 hits the inflight lock window
(res.end -> subprocess close, ~0.5s) and falls back to full history
("Key busy"). With the fix it waits ~0.5s and resumes.
"""
import json, urllib.request, time

BASE = "http://localhost:3456/v1/chat/completions"
KEY = "race-check-02"
TOOLS = [{"type": "function", "function": {
    "name": "Read",
    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}]

def post(body, timeout=180):
    req = urllib.request.Request(BASE, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)

def sse_read_all(resp):
    tool_calls = {}
    content = ""
    for raw in resp:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:") or line == "data: [DONE]":
            continue
        chunk = json.loads(line[5:].strip())
        delta = chunk.get("choices", [{}])[0].get("delta", {})
        for tc in delta.get("tool_calls") or []:
            tool_calls[tc["index"]] = tc
        if delta.get("content"):
            content += delta["content"]
    return [tool_calls[i] for i in sorted(tool_calls)], content

u1 = "Read file .gitignore with the Read tool, then say R1."
t0 = time.time()
r = post({"model": "claude-haiku-4-5", "stream": True, "user": KEY,
          "messages": [{"role": "user", "content": u1}], "tools": TOOLS})
tcs, content = sse_read_all(r)
t1_end = time.time()
print(f"turn1: SSE done in {t1_end-t0:.1f}s, tool_calls={[t['function']['name'] for t in tcs]}, "
      f"content={content[:60]!r}")
# Rate limit (or plain text) is fine for the lock-routing check: fabricate
# the history entries Kimi would have appended after a real tool call.
tc = tcs[0] if tcs else {
    "id": "call_fake1", "type": "function",
    "function": {"name": "Read", "arguments": '{"path":".gitignore"}'},
}

# IMMEDIATE follow-up — no sleep. This is the race window.
msgs = [
    {"role": "user", "content": u1},
    {"role": "assistant", "content": None, "tool_calls": [tc]},
    {"role": "tool", "content": "[contents omitted]"},
    {"role": "user", "content": "Now say R2 only."},
]
t1 = time.time()
gap_ms = (t1 - t1_end) * 1000
r = post({"model": "claude-haiku-4-5", "stream": False, "user": KEY, "messages": msgs})
resp = json.load(r)
t2_dur = time.time() - t1
print(f"turn2: sent {gap_ms:.0f}ms after turn1 end, answered in {t2_dur:.1f}s, "
      f"content={str(resp['choices'][0]['message'].get('content'))[:60]!r}")
print("Check log: turn2 resume:true, NO 'Key busy after wait'")

#!/usr/bin/env python3
"""Hard lock-wait check: turn 2 arrives 1.0s into turn 1's execution,
while the key is genuinely held (turn 1's subprocess still running).
Expected: turn 2 waits for the lock, then resume:true (not the fallback).
"""
import json, urllib.request, threading, time

BASE = "http://localhost:3456/v1/chat/completions"
KEY = "race-check-03"
TOOLS = [{"type": "function", "function": {
    "name": "Read",
    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}]

u1 = "Read file .gitignore with the Read tool, then say R1."
msgs1 = [{"role": "user", "content": u1}]
msgs2 = [
    {"role": "user", "content": u1},
    {"role": "assistant", "content": None, "tool_calls": [{
        "id": "call_fake1", "type": "function",
        "function": {"name": "Read", "arguments": '{"path":".gitignore"}'}}]},
    {"role": "tool", "content": "[contents omitted]"},
    {"role": "user", "content": "Now say R2 only."},
]

results = {}

def turn1():
    t0 = time.time()
    req = urllib.request.Request(BASE, data=json.dumps(
        {"model": "claude-haiku-4-5", "stream": True, "user": KEY,
         "messages": msgs1, "tools": TOOLS}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        for _ in r:
            pass
    results["t1"] = time.time() - t0

def turn2():
    t0 = time.time()
    req = urllib.request.Request(BASE, data=json.dumps(
        {"model": "claude-haiku-4-5", "stream": False, "user": KEY,
         "messages": msgs2}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        json.load(r)
    results["t2"] = time.time() - t0

th = threading.Thread(target=turn1)
th.start()
time.sleep(1.0)  # turn 1 is mid-flight, lock held
turn2()
th.join()
print(f"turn1 (stream): {results['t1']:.1f}s | turn2 (joined mid-flight): {results['t2']:.1f}s")
print("turn2 duration >> 1s means it waited for the lock. Check log for resume:true, no busy-warn")

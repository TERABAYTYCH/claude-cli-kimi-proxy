#!/usr/bin/env python3
"""Analyze an instrumented live run of claude-max-api-proxy.

Reads logs/app.log (windowed by UTC start time) and the fresh CLI transcripts
in ~/.claude/projects/<slug>/*.jsonl, then prints a ready-made acceptance
report.
"""
import argparse
import json
import os
import pathlib
import re
import sys
from datetime import datetime, timezone
from typing import Optional

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
DEFAULT_LOG_FILE = PROJECT_ROOT / "logs" / "app.log"
DEFAULT_TRANSCRIPT_DIR = pathlib.Path.home() / ".claude" / "projects" / "-root-work-claude-max-api-proxy"

# Effective token cost: cache write is 1.25x, cache read is 0.1x.
CACHE_CREATE_MULT = 1.25
CACHE_READ_MULT = 0.1


def parse_iso(ts: str) -> datetime:
    # Python <3.11 does not handle trailing Z directly
    ts = ts.replace("Z", "+00:00")
    return datetime.fromisoformat(ts)


def parse_log_line(line: str) -> Optional[dict]:
    line = line.rstrip("\n")
    if not line.startswith("["):
        return None
    m = re.match(r"^\[(?P<ts>[^\]]+)\]\s+\[(?P<level>[^\]]+)\]\s+(?P<msg>.*)$", line)
    if not m:
        return None
    msg = m.group("msg")
    meta: Optional[dict] = None
    brace = msg.rfind("{")
    if brace != -1:
        try:
            meta = json.loads(msg[brace:])
            msg = msg[:brace].rstrip()
        except Exception:
            pass
    return {
        "timestamp": parse_iso(m.group("ts")),
        "level": m.group("level"),
        "message": msg,
        "meta": meta or {},
        "raw": line,
    }


def load_log_window(path: pathlib.Path, start: datetime) -> list[dict]:
    entries = []
    if not path.exists():
        return entries
    for line in open(path, "r", encoding="utf-8", errors="replace"):
        entry = parse_log_line(line)
        if entry and entry["timestamp"] >= start:
            entries.append(entry)
    return entries


def find_transcripts(directory: pathlib.Path, exclude_id: Optional[str]) -> list[pathlib.Path]:
    if not directory.exists():
        return []
    files = [p for p in directory.iterdir() if p.suffix == ".jsonl"]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    result = []
    for p in files:
        if exclude_id and p.stem == exclude_id:
            continue
        result.append(p)
    return result


def summarize_transcript(path: pathlib.Path) -> dict:
    turns = 0
    user_messages = 0
    assistant_messages = 0
    for line in open(path, "r", encoding="utf-8", errors="replace"):
        try:
            rec = json.loads(line)
        except Exception:
            continue
        t = rec.get("type")
        if t == "user":
            user_messages += 1
        elif t == "assistant":
            assistant_messages += 1
            turns += 1
    return {
        "file": path.name,
        "turns": turns,
        "user_messages": user_messages,
        "assistant_messages": assistant_messages,
    }


def analyze(entries: list[dict], start: datetime) -> dict:
    prepared_by_id: dict[str, dict] = {}
    turns: list[dict] = []
    event_counts = {
        "Key busy after wait": 0,
        "Delta validation failed": 0,
        "Delta unexpectedly large": 0,
        "Loaded persisted sessions": 0,
        "Releasing inflight lock after error": 0,
        "Subprocess closed abnormally": 0,
        "Rate limit": 0,
    }
    last_restart: Optional[datetime] = None
    last_seen_by_key: dict[str, datetime] = {}
    anomalies: list[dict] = []

    for e in entries:
        msg = e["message"]
        meta = e["meta"]

        # Event counters
        if "[Session] Key busy after wait" in msg:
            event_counts["Key busy after wait"] += 1
        if "[Session] Delta validation failed" in msg:
            event_counts["Delta validation failed"] += 1
        if "[Session] Delta unexpectedly large" in msg:
            event_counts["Delta unexpectedly large"] += 1
        if "[SessionStore] Loaded persisted sessions" in msg:
            event_counts["Loaded persisted sessions"] += 1
            last_restart = e["timestamp"]
        if "[ChatCompletions] Releasing inflight lock after error" in msg:
            event_counts["Releasing inflight lock after error"] += 1
        if "[Streaming] Subprocess closed abnormally" in msg or "[NonStreaming] Subprocess closed without result" in msg:
            event_counts["Subprocess closed abnormally"] += 1
        if "Rate limit" in msg:
            event_counts["Rate limit"] += 1
        if "Server listening" in msg:
            last_restart = e["timestamp"]

        if msg == "[ChatCompletions] Request prepared":
            prepared_by_id[meta.get("requestId")] = {
                "time": e["timestamp"],
                "requestId": meta.get("requestId"),
                "sessionKey": meta.get("sessionKey"),
                "resume": meta.get("resume"),
                "promptChars": meta.get("promptChars"),
                "sliceRoles": meta.get("sliceRoles", []),
                "sinceIndex": meta.get("sinceIndex"),
                "kimiSystemPresent": meta.get("kimiSystemPresent"),
                "hasSystemPrompt": meta.get("hasSystemPrompt"),
                "toolResultBlocks": meta.get("toolResultBlocks"),
            }

        usage_match = None
        if msg == "[Streaming] CLI result usage" or msg == "[NonStreaming] CLI result usage":
            usage_match = meta

        if usage_match:
            req_id = meta.get("requestId")
            prep = prepared_by_id.pop(req_id, None)
            if not prep:
                continue
            row = {
                "time": prep["time"].isoformat().replace("+00:00", "Z"),
                "sessionKey": prep["sessionKey"],
                "resume": prep["resume"],
                "promptChars": prep["promptChars"] or 0,
                "sliceRoles": prep["sliceRoles"],
                "sinceIndex": prep["sinceIndex"],
                "input": meta.get("input") or 0,
                "output": meta.get("output") or 0,
                "cache_read": meta.get("cacheRead") or 0,
                "cache_create": meta.get("cacheCreate") or 0,
            }
            row["effective"] = int(
                row["cache_create"] * CACHE_CREATE_MULT
                + row["cache_read"] * CACHE_READ_MULT
                + row["input"]
            )
            turns.append(row)

            # Anomaly: resume=false on a continuation (not right after restart)
            key = prep["sessionKey"]
            if prep["resume"] is False and key in last_seen_by_key:
                if last_restart is None or prep["time"] > last_restart:
                    anomalies.append({
                        "turn": len(turns),
                        "requestId": req_id,
                        "type": "unexpected_resume_false",
                        "detail": f"resume=false for key {key} without recent restart",
                    })

            # Anomaly: prompt too large
            if (prep["promptChars"] or 0) > 20000:
                anomalies.append({
                    "turn": len(turns),
                    "requestId": req_id,
                    "type": "large_prompt",
                    "detail": f"promptChars={prep['promptChars']}",
                })

            # Anomaly: cache_create too large
            if (meta.get("cacheCreate") or 0) > 10000:
                anomalies.append({
                    "turn": len(turns),
                    "requestId": req_id,
                    "type": "large_cache_create",
                    "detail": f"cacheCreate={meta.get('cacheCreate')}",
                })

            # Anomaly: identity leaked into prompt prefix
            if prep.get("kimiSystemPresent") is True:
                anomalies.append({
                    "turn": len(turns),
                    "requestId": req_id,
                    "type": "kimi_system_present",
                    "detail": "identity found in prompt prefix",
                })

            # Anomaly: slice contains no new user/tool message
            roles = prep.get("sliceRoles") or []
            if roles and not any(r in ("tool", "user") for r in roles):
                anomalies.append({
                    "turn": len(turns),
                    "requestId": req_id,
                    "type": "degenerate_slice",
                    "detail": f"sliceRoles={roles}",
                })

            last_seen_by_key[key] = prep["time"]

    total_effective = sum(t["effective"] for t in turns)
    return {
        "turns": turns,
        "total_turns": len(turns),
        "total_effective": total_effective,
        "avg_effective": total_effective / len(turns) if turns else 0,
        "max_cache_create": max((t["cache_create"] for t in turns), default=0),
        "event_counts": event_counts,
        "anomalies": anomalies,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze an instrumented proxy run")
    parser.add_argument("start", help="UTC ISO start time, e.g. 2026-09-10T16:32:35.022Z")
    parser.add_argument("--log", type=pathlib.Path, default=DEFAULT_LOG_FILE)
    parser.add_argument("--transcript-dir", type=pathlib.Path, default=DEFAULT_TRANSCRIPT_DIR)
    parser.add_argument("--exclude-session-id", help="Exclude transcript with this UUID (current Claude Code session)")
    args = parser.parse_args()

    start = parse_iso(args.start)
    entries = load_log_window(args.log, start)
    report = analyze(entries, start)

    print(f"# Run analysis starting {args.start}")
    print(f"# Log window: {len(entries)} entries")
    print()

    print("## Turns")
    header = f"{'#':>3} | {'time':23} | {'resume':6} | {'promptChars':>11} | {'sliceRoles':30} | {'since':>5} | {'input':>6} | {'output':>6} | {'cache_r':>8} | {'cache_c':>8} | {'eff.':>8}"
    print(header)
    print("-" * len(header))
    for i, t in enumerate(report["turns"], 1):
        roles = json.dumps(t["sliceRoles"]) if t["sliceRoles"] else "-"
        print(
            f"{i:>3} | {t['time'][:23]:23} | {str(t['resume']):6} | {t['promptChars']:>11} | {roles:30} | "
            f"{str(t['sinceIndex']) if t['sinceIndex'] is not None else '-':>5} | {t['input']:>6} | {t['output']:>6} | "
            f"{t['cache_read']:>8} | {t['cache_create']:>8} | {t['effective']:>8}"
        )
    print()

    print("## Totals")
    print(f"  turns:            {report['total_turns']}")
    print(f"  total effective:  {report['total_effective']}")
    print(f"  avg per turn:     {report['avg_effective']:.1f}")
    print(f"  normalized to 4:  {report['avg_effective'] * 4:.1f}")
    print(f"  max cache_create: {report['max_cache_create']}")
    print()

    print("## Event counts")
    for name, count in report["event_counts"].items():
        print(f"  {name}: {count}")
    print()

    print("## Anomalies")
    if not report["anomalies"]:
        print("  none")
    else:
        for a in report["anomalies"]:
            print(f"  turn {a['turn']} ({a['type']}): {a['detail']}")
    print()

    transcripts = find_transcripts(args.transcript_dir, args.exclude_session_id)
    print("## Transcripts")
    if not transcripts:
        print("  none found")
    else:
        for p in transcripts:
            s = summarize_transcript(p)
            print(f"  {s['file']}: turns={s['turns']}, user={s['user_messages']}, assistant={s['assistant_messages']}")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())

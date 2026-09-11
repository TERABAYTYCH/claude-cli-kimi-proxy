#!/usr/bin/env python3
"""Sum Anthropic token usage over a time window, across all CLI transcripts.

Both measured paths (Claude Code directly, and Kimi -> proxy -> Claude CLI)
terminate at the same Anthropic API and leave a transcript with identical
usage fields, so one formula compares them fairly:

    effective = cache_creation * 1.25 + cache_read * 0.10 + input

Usage:
    measure.py --since 2026-09-11T07:00:00Z --until 2026-09-11T07:30:00Z \
               [--label direct] [--exclude-session <uuid>] [--json]
"""
import argparse, glob, json, os, sys
from datetime import datetime, timezone

PROJECT_DIR = os.path.expanduser("~/.claude/projects/-root-work-claude-max-api-proxy")


def parse_ts(s):
    if s is None:
        return None
    s = s.strip().replace("Z", "+00:00")
    try:
        d = datetime.fromisoformat(s)
    except ValueError:
        return None
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def collect(since, until, exclude):
    exclude = tuple(exclude or ())
    seen, rows = set(), []
    for path in glob.glob(os.path.join(PROJECT_DIR, "*.jsonl")):
        for line in open(path, encoding="utf8", errors="ignore"):
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                o = json.loads(line)
            except Exception:
                continue
            u = (o.get("message") or {}).get("usage") or {}
            if not u:
                continue
            ts = parse_ts(o.get("timestamp"))
            if ts is None or ts < since or ts > until:
                continue
            sid = o.get("session_id") or os.path.basename(path)[:-6]
            if exclude and sid.startswith(exclude):
                continue
            key = o.get("requestId") or o.get("uuid") or (sid, ts.isoformat())
            if key in seen:          # proxy transcripts repeat the assistant line
                continue
            seen.add(key)
            rows.append({
                "session": sid, "ts": ts,
                "input": u.get("input_tokens", 0) or 0,
                "output": u.get("output_tokens", 0) or 0,
                "cache_read": u.get("cache_read_input_tokens", 0) or 0,
                "cache_create": u.get("cache_creation_input_tokens", 0) or 0,
            })
    return sorted(rows, key=lambda r: r["ts"])


def eff(r):
    return r["cache_create"] * 1.25 + r["cache_read"] * 0.10 + r["input"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--since", required=True)
    p.add_argument("--until", required=True)
    p.add_argument("--label", default="run")
    p.add_argument("--exclude-session", action="append", default=[],
                   help="можно указывать несколько раз")
    p.add_argument("--list-sessions", action="store_true",
                   help="показать, какие сессии попали в окно, и выйти")
    p.add_argument("--json", action="store_true")
    a = p.parse_args()

    since, until = parse_ts(a.since), parse_ts(a.until)
    if not since or not until:
        sys.exit("bad --since/--until, expected ISO like 2026-09-11T07:00:00Z")

    rows = collect(since, until, a.exclude_session)
    if a.list_sessions:
        from collections import Counter
        c = Counter(r["session"] for r in rows)
        print(f"сессии в окне {since.isoformat()} .. {until.isoformat()}:")
        for sid, n in c.most_common():
            e = sum(eff(r) for r in rows if r["session"] == sid)
            print(f"  {sid}  вызовов={n:<4} эфф.={e:>12,.0f}")
        return
    if not rows:
        sys.exit("в окне нет ни одного вызова с usage — проверь время и часовой пояс (метки в UTC)")

    tot = {k: sum(r[k] for r in rows) for k in ("input", "output", "cache_read", "cache_create")}
    total_eff = sum(eff(r) for r in rows)

    if a.json:
        print(json.dumps({"label": a.label, "turns": len(rows), **tot,
                          "effective": round(total_eff)}, ensure_ascii=False))
        return

    print(f"=== {a.label} ===")
    print(f"окно: {since.isoformat()} .. {until.isoformat()}")
    print(f"сессий: {len(set(r['session'] for r in rows))}   вызовов к API: {len(rows)}\n")
    print(f"{'#':>3} {'время':>8} {'input':>8} {'output':>8} {'cache_r':>10} {'cache_c':>10} {'эфф.':>10}")
    for i, r in enumerate(rows, 1):
        print(f"{i:>3} {r['ts'].strftime('%H:%M:%S'):>8} {r['input']:>8} {r['output']:>8} "
              f"{r['cache_read']:>10} {r['cache_create']:>10} {eff(r):>10,.0f}")
    print("-" * 62)
    print(f"input        {tot['input']:>12,}  x1.00 = {tot['input']:>12,.0f}")
    print(f"cache_read   {tot['cache_read']:>12,}  x0.10 = {tot['cache_read']*0.1:>12,.0f}")
    print(f"cache_create {tot['cache_create']:>12,}  x1.25 = {tot['cache_create']*1.25:>12,.0f}")
    print(f"output       {tot['output']:>12,}  (справочно, в эфф. не входит)")
    print("=" * 62)
    print(f"ИТОГО эффективных input-токенов: {total_eff:>12,.0f}")
    print(f"вызовов: {len(rows)}   среднее на вызов: {total_eff/len(rows):,.0f}")


if __name__ == "__main__":
    main()

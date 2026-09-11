/**
 * In-memory mapping from OpenAI request.user (Hermes conversation key) to a
 * persisted Claude CLI session. Lets the proxy --resume an existing CLI
 * session instead of cold-starting a fresh one and replaying the entire
 * message history on every request.
 */

import fs from "fs";
import path from "path";
import { logger } from "../utils/logger.js";

interface SessionEntry {
  claudeSessionId: string;
  messageCount: number;
  lastUsed: number;
}

const SESSION_TTL_MS = 6 * 60 * 60 * 1000; // 6 hours of inactivity
const PRUNE_INTERVAL_MS = 30 * 60 * 1000; // sweep every 30 minutes

/**
 * Persistent backing for the session map. Default location is the current
 * working directory, NOT $HOME: ClaudeSubprocess.start runs the CLI with
 * cwd: options.cwd || process.cwd(), and Claude CLI stores transcripts under
 * ~/.claude/projects/<slug>/<session-id>.jsonl keyed by the project cwd. A
 * session UUID recorded while running in directory A will not resume from
 * directory B, so a global $HOME file would silently mix unrelated projects
 * and fall back to full-history replay. PROXY_SESSIONS_FILE overrides the
 * path when the operator wants a custom location.
 *
 * The inflight lock map is intentionally NOT persisted: a lock held by a dead
 * process would block a key until TTL theft kicks in.
 */
const SESSIONS_FILE =
  process.env.PROXY_SESSIONS_FILE ||
  path.join(process.cwd(), ".claude-max-api-proxy-sessions.json");

function loadSessions(): Map<string, SessionEntry> {
  const map = new Map<string, SessionEntry>();
  try {
    const raw = fs.readFileSync(SESSIONS_FILE, "utf8");
    const data = JSON.parse(raw) as Record<string, SessionEntry>;
    const now = Date.now();
    let dropped = 0;
    for (const [key, entry] of Object.entries(data)) {
      if (now - entry.lastUsed > SESSION_TTL_MS) {
        dropped++;
        continue;
      }
      map.set(key, entry);
    }
    logger.info("[SessionStore] Loaded persisted sessions", {
      file: SESSIONS_FILE,
      count: map.size,
      dropped,
    });
  } catch (err) {
    const code = (err as NodeJS.ErrnoException).code;
    if (code === "ENOENT") {
      logger.info("[SessionStore] No persisted sessions file, starting empty", {
        file: SESSIONS_FILE,
      });
    } else {
      logger.warn("[SessionStore] Failed to load persisted sessions, starting empty", {
        file: SESSIONS_FILE,
        error: (err as Error).message,
      });
    }
  }
  return map;
}

let persistPromise: Promise<void> | null = null;
let persistDirty = false;

async function persistSessionsOnce(): Promise<void> {
  const data: Record<string, SessionEntry> = {};
  for (const [key, entry] of sessions) {
    data[key] = entry;
  }
  const tmp = `${SESSIONS_FILE}.tmp`;
  await fs.promises.writeFile(tmp, JSON.stringify(data, null, 2));
  await fs.promises.rename(tmp, SESSIONS_FILE);
}

/**
 * Fire-and-forget persistence. Serialized so rapid setSession/clearSession
 * calls cannot interleave and write stale state.
 */
function persistSessions(): void {
  if (persistPromise) {
    persistDirty = true;
    return;
  }
  persistPromise = (async () => {
    do {
      persistDirty = false;
      try {
        await persistSessionsOnce();
      } catch (err) {
        logger.warn("[SessionStore] Failed to persist sessions", {
          file: SESSIONS_FILE,
          error: (err as Error).message,
        });
      }
    } while (persistDirty);
    persistPromise = null;
  })();
}

const sessions = loadSessions();

/**
 * Keys currently being driven by an in-flight request. Two concurrent
 * requests with the same sessionKey must never run `claude --resume <id>`
 * at the same time — they would race on the CLI's .jsonl transcript and
 * corrupt the shared context. resolveCliInput() acquires the key before
 * resuming and routes.ts releases it when the request finishes.
 *
 * Values are acquisition timestamps: a lock older than LOCK_TTL_MS is
 * considered leaked (e.g. a code path missed releaseLock) and is stolen,
 * so one missed release can never disable resume for a conversation
 * forever.
 */
const inflight = new Map<string, number>();
const LOCK_TTL_MS = 20 * 60 * 1000; // steal locks held longer than 20 minutes

/**
 * Try to mark a session key as busy. Returns false if another request
 * already holds it — the caller must then fall back to a full-history
 * request with a fresh session instead of resuming.
 */
export function acquireSession(key: string): boolean {
  const acquiredAt = inflight.get(key);
  if (acquiredAt !== undefined && Date.now() - acquiredAt < LOCK_TTL_MS) {
    return false;
  }
  inflight.set(key, Date.now());
  return true;
}

export function releaseSession(key: string): void {
  inflight.delete(key);
}

/**
 * Like acquireSession, but waits for the key to be freed first. Needed
 * because the next turn of the same conversation legitimately arrives while
 * the lock is still held: the proxy closes the SSE stream on </invoke>
 * (~15ms later Kimi executes the tool and sends the next turn) while the
 * lock is only released on subprocess "close" (~0.5s later, after the
 * transcript flush). An instant fallback there turns every other turn into
 * a full-history replay AND desyncs the resumed session — the fallback
 * process generates invokes the resumed session never saw.
 */
export async function acquireSessionWait(
  key: string,
  timeoutMs: number
): Promise<boolean> {
  const deadline = Date.now() + timeoutMs;
  for (;;) {
    if (acquireSession(key)) return true;
    if (Date.now() >= deadline) return false;
    await new Promise((r) => setTimeout(r, 100));
  }
}

function pruneExpired(): void {
  const now = Date.now();
  for (const [key, entry] of sessions) {
    if (now - entry.lastUsed > SESSION_TTL_MS) {
      sessions.delete(key);
    }
  }
  // Usage totals belong to live conversations only — drop entries whose
  // session is gone and which no request is currently driving.
  for (const key of usageTotals.keys()) {
    if (!sessions.has(key) && !inflight.has(key)) {
      usageTotals.delete(key);
    }
  }
}

// Sessions that are set once and never queried again (e.g. a client that
// stops sending `user`) would otherwise sit in memory forever, since TTL
// was previously only enforced on read. Sweep periodically so idle server
// processes don't accumulate unbounded entries.
setInterval(pruneExpired, PRUNE_INTERVAL_MS).unref();

export function getSession(key: string): SessionEntry | undefined {
  const entry = sessions.get(key);
  if (!entry) return undefined;
  if (Date.now() - entry.lastUsed > SESSION_TTL_MS) {
    sessions.delete(key);
    return undefined;
  }
  return entry;
}

export function setSession(
  key: string,
  claudeSessionId: string,
  messageCount: number
): void {
  sessions.set(key, { claudeSessionId, messageCount, lastUsed: Date.now() });
  persistSessions();
}

/**
 * Cumulative token usage per conversation key, fed from CLI result events.
 * Lets the operator spot regressions (usage exploding again) from the log.
 */
const usageTotals = new Map<string, { cacheRead: number; cacheCreate: number; output: number; steps: number }>();

export function addUsage(
  key: string | undefined,
  usage: { cacheRead?: number; cacheCreate?: number; output?: number }
): { cacheRead: number; cacheCreate: number; output: number; steps: number } | undefined {
  if (!key) return undefined;
  const cur = usageTotals.get(key) || { cacheRead: 0, cacheCreate: 0, output: 0, steps: 0 };
  cur.cacheRead += usage.cacheRead || 0;
  cur.cacheCreate += usage.cacheCreate || 0;
  cur.output += usage.output || 0;
  cur.steps += 1;
  usageTotals.set(key, cur);
  return cur;
}

export function clearSession(key: string): void {
  sessions.delete(key);
  persistSessions();
  // Usage totals are per CLI session — reset them with the session, otherwise
  // the cumulative counters in the log mix several sessions of one chat.
  usageTotals.delete(key);
}

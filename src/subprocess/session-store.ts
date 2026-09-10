/**
 * In-memory mapping from OpenAI request.user (Hermes conversation key) to a
 * persisted Claude CLI session. Lets the proxy --resume an existing CLI
 * session instead of cold-starting a fresh one and replaying the entire
 * message history on every request.
 */

interface SessionEntry {
  claudeSessionId: string;
  messageCount: number;
  lastUsed: number;
}

const SESSION_TTL_MS = 6 * 60 * 60 * 1000; // 6 hours of inactivity
const PRUNE_INTERVAL_MS = 30 * 60 * 1000; // sweep every 30 minutes

const sessions = new Map<string, SessionEntry>();

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
  // Usage totals are per CLI session — reset them with the session, otherwise
  // the cumulative counters in the log mix several sessions of one chat.
  usageTotals.delete(key);
}

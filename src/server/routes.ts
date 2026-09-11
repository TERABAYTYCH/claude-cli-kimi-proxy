/**
 * API Route Handlers
 *
 * Implements OpenAI-compatible endpoints for Clawdbot integration
 */

import type { Request, Response } from "express";
import { v4 as uuidv4 } from "uuid";
import { ClaudeSubprocess } from "../subprocess/manager.js";
import { openaiToCli, openaiToCliDelta } from "../adapter/openai-to-cli.js";
import {
  cliResultToOpenai,
  createDoneChunk,
  isRateLimitError,
  rateLimitRetryAfter,
  buildOpenAIUsage,
} from "../adapter/cli-to-openai.js";
import { getSession, setSession, clearSession, acquireSessionWait, releaseSession, addUsage } from "../subprocess/session-store.js";
import {
  parseDelegations,
  delegationsToToolCalls,
} from "../adapter/delegate-parser.js";
import { logger } from "../utils/logger.js";
import type { OpenAIChatRequest, OpenAIToolCall } from "../types/openai.js";
import type { ClaudeCliAssistant, ClaudeCliResult, ClaudeCliStreamEvent } from "../types/claude-cli.js";

interface SessionContext {
  sessionKey: string | undefined;
  resume: boolean;
  messageCount: number;
  /** False for fallback requests that must not overwrite the stored session */
  persistSession: boolean;
  /** True while this request holds the per-key inflight lock */
  lockHeld: boolean;
}

/**
 * How to surface Claude rate limit errors to the OpenAI client.
 * - "content" (default): return a normal 200 completion whose text is the
 *   rate limit message. Kimi's web client retries on HTTP 429 regardless of
 *   Retry-After, so this is the only way the user actually SEES the error
 *   in the chat instead of a "Model request failed" retry loop.
 * - "http": return HTTP 429 (non-streaming) / SSE error event (streaming)
 *   with a Retry-After header, per OpenAI conventions.
 */
const RATE_LIMIT_MODE = process.env.PROXY_RATE_LIMIT_MODE || "content";

function rateLimitMessage(result: ClaudeCliResult): string {
  return `⚠️ Claude rate limit: ${result.result || "Rate limit exceeded"}`;
}

/**
 * Persist the CLI session for future --resume turns. Does NOT release the
 * per-key inflight lock: the lock is only dropped once the subprocess has
 * exited and flushed its transcript (subprocess "close"), otherwise a fast
 * follow-up --resume could race the transcript write.
 */
function persistSession(
  sessionCtx: SessionContext,
  cliInput: ReturnType<typeof openaiToCli>
): void {
  if (sessionCtx.persistSession && sessionCtx.sessionKey && cliInput.sessionId) {
    setSession(sessionCtx.sessionKey, cliInput.sessionId, sessionCtx.messageCount);
  }
}

/**
 * Persist and release in one step. Only safe at subprocess "close", i.e.
 * after the transcript has been flushed to disk. Used by the non-streaming
 * path, where close is the terminal event.
 */
function persistSessionAndRelease(
  sessionCtx: SessionContext,
  cliInput: ReturnType<typeof openaiToCli>
): void {
  persistSession(sessionCtx, cliInput);
  releaseLock(sessionCtx);
}

/** Drop the inflight lock without persisting (error / abnormal-close paths). */
function releaseLock(sessionCtx: SessionContext): void {
  if (sessionCtx.lockHeld && sessionCtx.sessionKey) {
    releaseSession(sessionCtx.sessionKey);
    sessionCtx.lockHeld = false;
  }
}

/**
 * Resolve CLI input for a request, resuming a persisted Claude CLI session
 * when we have one for this `request.user` key instead of replaying the
 * full message history on every turn.
 */
async function resolveCliInput(body: OpenAIChatRequest): Promise<{
  cliInput: ReturnType<typeof openaiToCli>;
  sessionKey: string | undefined;
  resume: boolean;
  persistSession: boolean;
  /** True if this request holds the per-key inflight lock (see session-store). */
  lockHeld: boolean;
  /** sinceIndex of the resumed session + roles of the sliced delta (R2 diag). */
  diag?: { sinceIndex: number; sliceRoles: string[] };
}> {
  // Kimi may identify a conversation via `user` or `prompt_cache_key`.
  // We use that as our in-memory lookup key, but Claude CLI requires a
  // valid UUID for --session-id / --resume, so we keep a separate UUID.
  const sessionKey =
    body.user || (body as { prompt_cache_key?: string }).prompt_cache_key;
  const delegateMode = !!body.tools?.length;

  if (!sessionKey) {
    const cliInput = openaiToCli(body);
    return { cliInput, sessionKey, resume: false, persistSession: false, lockHeld: false };
  }

  // Serialize turns of one conversation: wait for the previous request to
  // release the key (it is held until the subprocess exits and flushes its
  // transcript). Kimi sends the next turn ~15ms after our SSE ends, while
  // the lock outlives it by ~0.5s — an instant fallback here would turn
  // every other turn into a full-history replay (measured: 4 duplicates
  // costing 94k cache-creation tokens in one live run) and desync the
  // resumed session, because the fallback process generates invokes the
  // resumed session never saw. A few hundred ms of waiting avoids both.
  const LOCK_WAIT_MS = 8000;
  const acquired = await acquireSessionWait(sessionKey, LOCK_WAIT_MS);

  if (!acquired) {
    // Key held by a genuinely long-running request (parallel agents sharing
    // a key). Run unlocked with a fresh session and don't persist, so we
    // don't clobber the stored entry owned by the in-flight conversation.
    logger.warn("[Session] Key busy after wait, falling back to full history without resume", {
      sessionKey,
      waitMs: LOCK_WAIT_MS,
    });
    const cliInput = openaiToCli(body);
    cliInput.sessionId = uuidv4();
    return { cliInput, sessionKey: undefined, resume: false, persistSession: false, lockHeld: false };
  }

  // Read the session AFTER acquiring: the previous request may have just
  // persisted a newer entry while we were waiting for the lock.
  const existing = getSession(sessionKey);

  if (existing) {
    // Resume the persisted CLI session — including delegate (tools) mode.
    // Previously delegate mode always replayed the full history into a
    // fresh --session-id, forcing Anthropic to cache-write the entire
    // growing conversation on every tool round-trip (O(N²) cost). Verified
    // with Claude CLI 2.1.x: --resume + --system-prompt-file re-applies the
    // system prompt and the CLI remembers its own prior turns, so only the
    // delta messages need to be sent.
    //
    // The delta is only safe when Kimi's history still matches what the CLI
    // session saw: Kimi web may trim/summarize old messages, which shifts
    // indices and would silently drop context from the delta. Validate
    // before trusting it; otherwise restart the session from full history.
    const deltaValid = isDeltaValid(body, existing.messageCount);
    if (deltaValid) {
      const cliInput = openaiToCliDelta(body, existing.messageCount);
      cliInput.sessionId = existing.claudeSessionId;
      return {
        cliInput, sessionKey, resume: true, persistSession: true, lockHeld: true,
        diag: {
          sinceIndex: existing.messageCount,
          sliceRoles: body.messages.slice(existing.messageCount).map((m) => m.role),
        },
      };
    }

    logger.warn("[Session] Delta validation failed, restarting session from full history", {
      sessionKey,
      messageCount: body.messages.length,
      sinceIndex: existing.messageCount,
    });
    clearSession(sessionKey);
    const cliInput = openaiToCli(body);
    cliInput.sessionId = uuidv4();
    return { cliInput, sessionKey, resume: false, persistSession: true, lockHeld: true };
  }

  // Fresh conversation under this key — build a session we can resume later.
  const cliInput = openaiToCli(body);
  cliInput.sessionId = uuidv4(); // pin a known UUID so we can --resume it later
  return { cliInput, sessionKey, resume: false, persistSession: true, lockHeld: true };
}

/**
 * A resumed CLI session already contains everything up to `sinceIndex`
 * (the CLI generated those turns itself). The delta is trustworthy only if
 * the sliced messages look like a normal append: indices in range, slice
 * non-empty, and it carries at least one new user or tool message.
 *
 * Previously delegate mode required a tool message in the slice, but that
 * wrongly rejected plain follow-up questions in tool-enabled chats, forcing
 * a full-history replay (~100x cost). Assistant-only slices are the only
 * degenerate case we reject — they are just echoed model output.
 */
function isDeltaValid(
  body: OpenAIChatRequest,
  sinceIndex: number
): boolean {
  if (!Number.isInteger(sinceIndex) || sinceIndex < 0) return false;
  if (sinceIndex >= body.messages.length) return false;
  const slice = body.messages.slice(sinceIndex);
  if (slice.length === 0) return false;
  if (!slice.some((m) => m.role === "tool" || m.role === "user")) return false;
  return true;
}

/**
 * Handle POST /v1/chat/completions
 *
 * Main endpoint for chat requests, supports both streaming and non-streaming
 */
export async function handleChatCompletions(
  req: Request,
  res: Response
): Promise<void> {
  const requestId = uuidv4().replace(/-/g, "").slice(0, 24);
  const body = req.body as OpenAIChatRequest;
  const stream = body.stream === true;

  // Holds the per-key inflight lock if resolveCliInput acquires one. Kept in
  // this outer scope so the catch block can release it if anything throws
  // after acquisition but before the subprocess handlers take over.
  const sessionCtx: SessionContext = {
    sessionKey: undefined,
    resume: false,
    messageCount: 0,
    persistSession: false,
    lockHeld: false,
  };

  try {
    // Validate request
    if (!body.messages || !Array.isArray(body.messages) || body.messages.length === 0) {
      res.status(400).json({
        error: {
          message: "messages is required and must be a non-empty array",
          type: "invalid_request_error",
          code: "invalid_messages",
        },
      });
      return;
    }

    // Convert to CLI input format, resuming a persisted session when we have one
    const { cliInput, sessionKey, resume, persistSession, lockHeld, diag } = await resolveCliInput(body);
    const subprocess = new ClaudeSubprocess();
    sessionCtx.sessionKey = sessionKey;
    sessionCtx.resume = resume;
    sessionCtx.messageCount = body.messages.length;
    sessionCtx.persistSession = persistSession;
    sessionCtx.lockHeld = lockHeld;

    // Task 3 guardrail: a "delta" that stopped being a delta silently kills
    // the resume savings (measured: 77-98k char prompts on resumed turns).
    // Log loudly; do not change behavior.
    const DELTA_SOFT_LIMIT = 20000; // chars
    if (resume && cliInput.prompt.length > DELTA_SOFT_LIMIT) {
      logger.warn("[Session] Delta unexpectedly large — resume savings lost", {
        sessionKey,
        promptChars: cliInput.prompt.length,
        sinceIndex: diag?.sinceIndex,
        messagesLen: body.messages.length,
      });
    }

    logger.info("[ChatCompletions] Request prepared", {
      requestId,
      stream,
      model: body.model,
      resume,
      sessionKey,
      hasSessionKey: !!sessionKey,
      hasSystemPrompt: !!cliInput.systemPrompt,
      hasTools: !!body.tools?.length,
      toolNames: body.tools?.map((t) => t.function.name),
      promptPreview: cliInput.prompt.slice(0, 200),
      promptChars: cliInput.prompt.length,
      // The delegation instruction never appears in the prompt (it always
      // goes via --system-prompt-file), so <tool_result> literals in the
      // prompt are all real delta content.
      toolResultBlocks: (cliInput.prompt.match(/<tool_result>/g) || []).length,
      // Check only the prompt prefix: the identity must be at the start
      // (via --system-prompt-file) when present. A literal further down is
      // just source code the model read (e.g. routes.ts) and must not flag
      // a false regression.
      kimiSystemPresent: cliInput.prompt.slice(0, 500).includes("You are Kimi Code CLI"),
      ...(diag
        ? {
            sinceIndex: diag.sinceIndex,
            messagesLen: body.messages.length,
            sliceLen: diag.sliceRoles.length,
            sliceRoles: diag.sliceRoles,
          }
        : {}),
    });

    if (stream) {
      await handleStreamingResponse(req, res, subprocess, cliInput, requestId, sessionCtx, body);
    } else {
      await handleNonStreamingResponse(res, subprocess, cliInput, requestId, sessionCtx, body);
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown error";
    console.error("[handleChatCompletions] Error:", message);

    // If resolveCliInput acquired the per-key lock and we threw before the
    // subprocess handlers were wired up, release it here. releaseLock is
    // idempotent (uses the lockHeld flag), so it is safe even for requests
    // that never held the lock.
    if (sessionCtx.lockHeld) {
      logger.warn("[ChatCompletions] Releasing inflight lock after error", {
        requestId,
        sessionKey: sessionCtx.sessionKey,
      });
      releaseLock(sessionCtx);
    }

    if (!res.headersSent) {
      res.status(500).json({
        error: {
          message,
          type: "server_error",
          code: null,
        },
      });
    }
  }
}

/**
 * Convert Claude tool_use ID to OpenAI-compatible call ID.
 * Claude uses "toolu_abc123", OpenAI uses "call_abc123".
 */
function toOpenAICallId(claudeId: string): string {
  return `call_${claudeId.replace("toolu_", "")}`;
}

/**
 * Handle streaming response (SSE)
 *
 * IMPORTANT: The Express req.on("close") event fires when the request body
 * is fully received, NOT when the client disconnects. For SSE connections,
 * we use res.on("close") to detect actual client disconnection.
 */
async function handleStreamingResponse(
  req: Request,
  res: Response,
  subprocess: ClaudeSubprocess,
  cliInput: ReturnType<typeof openaiToCli>,
  requestId: string,
  sessionCtx: SessionContext,
  body: OpenAIChatRequest
): Promise<void> {
  // Set SSE headers
  res.setHeader("Content-Type", "text/event-stream");
  res.setHeader("Cache-Control", "no-cache");
  res.setHeader("Connection", "keep-alive");
  res.setHeader("X-Request-Id", requestId);

  // CRITICAL: Flush headers immediately to establish SSE connection
  // Without this, headers are buffered and client times out waiting
  res.flushHeaders();

  // Send initial comment to confirm connection is alive
  res.write(":ok\n\n");

  const delegateMode = !!body.tools?.length;

  logger.info("[Streaming] Starting response", {
    requestId,
    delegateMode,
    resume: sessionCtx.resume,
    hasSystemPrompt: !!cliInput.systemPrompt,
    hasSessionId: !!cliInput.sessionId,
  });

  return new Promise<void>((resolve, reject) => {
    let isFirst = true;
    let lastModel = "claude-sonnet-4";
    let isComplete = false;
    let hasEmittedText = false;
    let textBuffer = "";
    let lastInvokeCloseCount = 0;
    let delegateEmitted = false;
    let finished = false;
    const rejectedToolNames = new Set<string>();

    function finish() {
      if (finished) return;
      finished = true;
      // The CLI exits by itself in --print mode once the result is printed.
      // Give it a short grace period to shut down cleanly (flushing its
      // session transcript — needed for the next turn's --resume) and only
      // kill it if it hangs. The per-key inflight lock serializes follow-up
      // requests, so a short grace is safe.
      const grace = setTimeout(() => {
        logger.debug("Grace period expired, killing subprocess", { requestId });
        subprocess.kill();
      }, 1000);
      grace.unref();
      subprocess.once("close", () => clearTimeout(grace));
      resolve();
    }

    function emitDelegateAndEnd(): boolean {
      const delegations = parseDelegations(textBuffer);
      if (delegations.length === 0) return false;

      const toolCalls = delegationsToToolCalls(delegations, "call", body.tools, rejectedToolNames);
      if (toolCalls.length === 0) {
        // All parsed delegations were rejected (unknown tool names). Treat the
        // response as plain text so the client sees the model's words instead of
        // an empty tool_calls block.
        return false;
      }
      logger.info("[Streaming] Delegations detected", {
        requestId,
        count: toolCalls.length,
        tools: toolCalls.map((t) => t.function.name),
      });

      const chunk = {
        id: `chatcmpl-${requestId}`,
        object: "chat.completion.chunk",
        created: Math.floor(Date.now() / 1000),
        model: lastModel,
        choices: [{
          index: 0,
          delta: {
            role: isFirst ? "assistant" : undefined,
            content: null,
            tool_calls: toolCalls.map((tc, idx) => ({
              index: idx,
              id: tc.id,
              type: "function",
              function: {
                name: tc.function.name,
                arguments: tc.function.arguments,
              },
            })),
          },
          finish_reason: null,
        }],
      };
      res.write(`data: ${JSON.stringify(chunk)}\n\n`);

      const finishChunk = {
        id: `chatcmpl-${requestId}`,
        object: "chat.completion.chunk",
        created: Math.floor(Date.now() / 1000),
        model: lastModel,
        choices: [{
          index: 0,
          delta: {},
          finish_reason: "tool_calls",
        }],
      };
      res.write(`data: ${JSON.stringify(finishChunk)}\n\n`);
      res.write("data: [DONE]\n\n");
      res.end();

      delegateEmitted = true;
      isComplete = true;
      // Persist the session so the next turn can --resume, but keep the
      // inflight lock until the subprocess exits and flushes its transcript
      // (subprocess "close") — releasing here would let a fast follow-up
      // --resume race the transcript write.
      persistSession(sessionCtx, cliInput);
      finish();
      return true;
    }

    function flushTextBufferAsContent() {
      if (!textBuffer || res.writableEnded) return;
      const chunk = {
        id: `chatcmpl-${requestId}`,
        object: "chat.completion.chunk",
        created: Math.floor(Date.now() / 1000),
        model: lastModel,
        choices: [{
          index: 0,
          delta: {
            role: isFirst ? "assistant" : undefined,
            content: textBuffer,
          },
          finish_reason: null,
        }],
      };
      res.write(`data: ${JSON.stringify(chunk)}\n\n`);
      isFirst = false;
      hasEmittedText = true;
    }

    // Handle actual client disconnect (response stream closed)
    res.on("close", () => {
      if (!isComplete) {
        // Client disconnected mid-response — the CLI transcript is truncated,
        // so resuming it later would replay a broken conversation. Drop the
        // session (self-heals via full-history restart), kill the subprocess
        // and release the lock: with the session cleared, nobody can --resume
        // the half-written transcript, so early release is safe here.
        if (sessionCtx.sessionKey) {
          clearSession(sessionCtx.sessionKey);
        }
        subprocess.kill();
        releaseLock(sessionCtx);
      }
      // isComplete: the lock is dropped on subprocess "close" after the
      // transcript flush — releasing it here would reopen the B2 race.
      resolve();
    });

    // When a new text content block starts after we've already emitted text,
    // insert a separator so text from different blocks doesn't run together
    subprocess.on("text_block_start", () => {
      if (!delegateMode && hasEmittedText && !res.writableEnded) {
        const sepChunk = {
          id: `chatcmpl-${requestId}`,
          object: "chat.completion.chunk",
          created: Math.floor(Date.now() / 1000),
          model: lastModel,
          choices: [{
            index: 0,
            delta: { content: "\n\n" },
            finish_reason: null,
          }],
        };
        res.write(`data: ${JSON.stringify(sepChunk)}\n\n`);
      }
    });

    // Handle streaming content deltas
    subprocess.on("content_delta", (event: ClaudeCliStreamEvent) => {
      const delta = event.event.delta;
      const text = (delta?.type === "text_delta" && delta.text) || "";
      if (!text || res.writableEnded) return;

      if (delegateMode) {
        textBuffer += text;
        // Check whether a complete Kimi-style <invoke> block has arrived.
        // Count closing tags instead of re-scanning on every delta: once we
        // have parsed all blocks for the current set of closes, do not re-parse
        // until a new </invoke> appears. This avoids quadratic scans and log spam
        // when the model quotes invoke syntax and then continues with plain text.
        const invokeCloseCount = (textBuffer.match(/<\/invoke>/g) || []).length;
        if (invokeCloseCount > lastInvokeCloseCount) {
          lastInvokeCloseCount = invokeCloseCount;
          if (emitDelegateAndEnd()) {
            return;
          }
        }
      } else {
        const chunk = {
          id: `chatcmpl-${requestId}`,
          object: "chat.completion.chunk",
          created: Math.floor(Date.now() / 1000),
          model: lastModel,
          choices: [{
            index: 0,
            delta: {
              role: isFirst ? "assistant" : undefined,
              content: text,
            },
            finish_reason: null,
          }],
        };
        res.write(`data: ${JSON.stringify(chunk)}\n\n`);
        isFirst = false;
        hasEmittedText = true;
      }
    });

    // Handle thinking deltas — forward them as reasoning_content so clients
    // like Kimi can display the model's reasoning process separately.
    subprocess.on("thinking_delta", (event: ClaudeCliStreamEvent) => {
      const delta = event.event.delta;
      const thinking = (delta?.type === "thinking_delta" && delta.thinking) || "";
      if (!thinking || res.writableEnded) return;

      const chunk = {
        id: `chatcmpl-${requestId}`,
        object: "chat.completion.chunk",
        created: Math.floor(Date.now() / 1000),
        model: lastModel,
        choices: [{
          index: 0,
          delta: {
            reasoning_content: thinking,
          },
          finish_reason: null,
        }],
      };
      res.write(`data: ${JSON.stringify(chunk)}\n\n`);
    });

    // Handle final assistant message (for model name)
    subprocess.on("assistant", (message: ClaudeCliAssistant) => {
      lastModel = message.message.model;
    });

    subprocess.on("result", (result: ClaudeCliResult) => {
      isComplete = true;
      persistSession(sessionCtx, cliInput);
      const totals = addUsage(sessionCtx.sessionKey, {
        cacheRead: result.usage?.cache_read_input_tokens,
        cacheCreate: result.usage?.cache_creation_input_tokens,
        output: result.usage?.output_tokens,
      });
      logger.info("[Streaming] CLI result usage", {
        requestId,
        resume: sessionCtx.resume,
        deltaChars: cliInput.prompt.length,
        input: result.usage?.input_tokens,
        output: result.usage?.output_tokens,
        cacheRead: result.usage?.cache_read_input_tokens,
        cacheCreate: result.usage?.cache_creation_input_tokens,
        ...(totals
          ? { convCacheRead: totals.cacheRead, convCacheCreate: totals.cacheCreate, convOutput: totals.output, convSteps: totals.steps }
          : {}),
      });

      if (res.writableEnded) {
        finish();
        return;
      }

      if (delegateMode) {
        if (!delegateEmitted && emitDelegateAndEnd()) {
          return;
        }
        if (!delegateEmitted) {
          flushTextBufferAsContent();
        }
      }

      if (RATE_LIMIT_MODE === "http" && isRateLimitError(result)) {
        const retryAfter = rateLimitRetryAfter(result);
        logger.warn("[Streaming] Rate limit error", {
          requestId,
          mode: RATE_LIMIT_MODE,
          result: result.result,
          retryAfter,
        });
        res.write(`data: ${JSON.stringify({
          error: {
            message: result.result || "Rate limit exceeded",
            type: "rate_limit_error",
            code: "rate_limit",
          },
        })}\n\n`);
        res.write("data: [DONE]\n\n");
        res.end();
        finish();
        return;
      }

      // Content mode (default): present the limit message as a normal
      // completion so clients that retry on any error (e.g. Kimi web)
      // display it instead of looping on "Model request failed".
      if (RATE_LIMIT_MODE === "content" && isRateLimitError(result)) {
        logger.warn("[Streaming] Rate limit error (as content)", {
          requestId,
          mode: RATE_LIMIT_MODE,
          result: result.result,
          retryAfter: rateLimitRetryAfter(result),
        });
        res.write(`data: ${JSON.stringify({
          id: `chatcmpl-${requestId}`,
          object: "chat.completion.chunk",
          created: Math.floor(Date.now() / 1000),
          model: lastModel,
          choices: [{
            index: 0,
            delta: { content: rateLimitMessage(result) },
            finish_reason: null,
          }],
        })}\n\n`);
        // Fall through to the standard done chunk below (finish_reason: stop).
      }

      // Send final done chunk with finish_reason and usage data
      const doneChunk = createDoneChunk(requestId, lastModel);
      if (result.usage) {
        const input = result.usage.input_tokens || 0;
        const output = result.usage.output_tokens || 0;
        const cacheRead = result.usage.cache_read_input_tokens || 0;
        const cacheCreate = result.usage.cache_creation_input_tokens || 0;
        const promptTokens = input + cacheRead + cacheCreate;
        doneChunk.usage = {
          prompt_tokens: promptTokens,
          completion_tokens: output,
          total_tokens: promptTokens + output,
          prompt_tokens_details: {
            cached_tokens: cacheRead,
          },
        };
      }
      res.write(`data: ${JSON.stringify(doneChunk)}\n\n`);
      res.write("data: [DONE]\n\n");
      res.end();
      finish();
    });

    subprocess.on("error", (error: Error) => {
      logger.error("[Streaming] Error", { requestId, error: error.message });
      // Resume may have failed (e.g. stale/missing session) — drop it so the
      // next turn self-heals with a fresh full-history session
      if (sessionCtx.resume && sessionCtx.sessionKey) {
        clearSession(sessionCtx.sessionKey);
      }
      releaseLock(sessionCtx);
      if (!res.writableEnded) {
        res.write(
          `data: ${JSON.stringify({
            error: { message: error.message, type: "server_error", code: null },
          })}\n\n`
        );
        res.end();
      }
      finish();
    });

    subprocess.on("close", (code: number | null) => {
      // Terminal event: the transcript is flushed, so the per-key lock can
      // always be dropped here — for both normal exits and abnormal ones.
      if (code !== 0 && !isComplete) {
        logger.warn("[Streaming] Subprocess closed abnormally", { requestId, code });
        if (sessionCtx.resume && sessionCtx.sessionKey) {
          clearSession(sessionCtx.sessionKey);
        }
        if (!res.writableEnded) {
          // Abnormal exit without result - send error
          res.write(`data: ${JSON.stringify({
            error: { message: `Process exited with code ${code}`, type: "server_error", code: null },
          })}\n\n`);
          res.end();
        }
      }
      releaseLock(sessionCtx);
      if (!res.writableEnded) {
        res.write("data: [DONE]\n\n");
        res.end();
      }
      finish();
    });

    // Start the subprocess
    subprocess.start(cliInput.prompt, {
      model: cliInput.model,
      sessionId: cliInput.sessionId,
      resume: sessionCtx.resume,
      systemPrompt: cliInput.systemPrompt,
      delegateTools: delegateMode,
    }).catch((err) => {
      logger.error("[Streaming] Subprocess start error", { requestId, error: err.message });
      releaseLock(sessionCtx);
      reject(err);
    });
  });
}

/**
 * Handle non-streaming response
 */
async function handleNonStreamingResponse(
  res: Response,
  subprocess: ClaudeSubprocess,
  cliInput: ReturnType<typeof openaiToCli>,
  requestId: string,
  sessionCtx: SessionContext,
  body: OpenAIChatRequest
): Promise<void> {
  const delegateMode = !!body.tools?.length;

  logger.info("[NonStreaming] Starting response", {
    requestId,
    delegateMode,
    resume: sessionCtx.resume,
    hasSystemPrompt: !!cliInput.systemPrompt,
    hasSessionId: !!cliInput.sessionId,
  });

  return new Promise((resolve) => {
    let finalResult: ClaudeCliResult | null = null;

    subprocess.on("result", (result: ClaudeCliResult) => {
      finalResult = result;
      const totals = addUsage(sessionCtx.sessionKey, {
        cacheRead: result.usage?.cache_read_input_tokens,
        cacheCreate: result.usage?.cache_creation_input_tokens,
        output: result.usage?.output_tokens,
      });
      logger.info("[NonStreaming] CLI result usage", {
        requestId,
        resume: sessionCtx.resume,
        deltaChars: cliInput.prompt.length,
        input: result.usage?.input_tokens,
        output: result.usage?.output_tokens,
        cacheRead: result.usage?.cache_read_input_tokens,
        cacheCreate: result.usage?.cache_creation_input_tokens,
        ...(totals
          ? { convCacheRead: totals.cacheRead, convCacheCreate: totals.cacheCreate, convOutput: totals.output, convSteps: totals.steps }
          : {}),
      });
    });

    subprocess.on("error", (error: Error) => {
      logger.error("[NonStreaming] Error", { requestId, error: error.message });
      if (sessionCtx.resume && sessionCtx.sessionKey) {
        clearSession(sessionCtx.sessionKey);
      }
      releaseLock(sessionCtx);
      res.status(500).json({
        error: {
          message: error.message,
          type: "server_error",
          code: null,
        },
      });
      resolve();
    });

    subprocess.on("close", (code: number | null) => {
      if (finalResult) {
        persistSessionAndRelease(sessionCtx, cliInput);
        const delegations = delegateMode ? parseDelegations(finalResult.result || "") : [];
        const toolCalls = delegations.length > 0 ? delegationsToToolCalls(delegations, "call", body.tools) : [];
        if (toolCalls.length > 0) {
          logger.info("[NonStreaming] Delegations detected", {
            requestId,
            count: toolCalls.length,
            tools: toolCalls.map((t) => t.function.name),
          });
          const response = {
            id: `chatcmpl-${requestId}`,
            object: "chat.completion",
            created: Math.floor(Date.now() / 1000),
            model: body.model,
            choices: [
              {
                index: 0,
                message: {
                  role: "assistant",
                  content: null,
                  tool_calls: toolCalls,
                },
                finish_reason: "tool_calls" as const,
              },
            ],
            usage: buildOpenAIUsage(finalResult.usage),
          };
          logger.info("[NonStreaming] Returning tool_calls", {
            requestId,
            toolCount: toolCalls.length,
          });
          res.json(response);
        } else {
          if (RATE_LIMIT_MODE === "http" && isRateLimitError(finalResult)) {
            const retryAfter = rateLimitRetryAfter(finalResult);
            logger.warn("[NonStreaming] Rate limit error", {
              requestId,
              mode: RATE_LIMIT_MODE,
              result: finalResult.result,
              retryAfter,
            });
            if (retryAfter !== undefined) {
              res.set("Retry-After", String(retryAfter));
            }
            res.status(429).json({
              error: {
                message: finalResult.result || "Rate limit exceeded",
                type: "rate_limit_error",
                code: "rate_limit",
              },
            });
            resolve();
            return;
          }

          if (RATE_LIMIT_MODE === "content" && isRateLimitError(finalResult)) {
            logger.warn("[NonStreaming] Rate limit error (as content)", {
              requestId,
              mode: RATE_LIMIT_MODE,
              result: finalResult.result,
              retryAfter: rateLimitRetryAfter(finalResult),
            });
            finalResult = {
              ...finalResult,
              is_error: false,
              result: rateLimitMessage(finalResult),
            };
          }

          logger.info("[NonStreaming] Returning text response", {
            requestId,
            hasContent: !!finalResult.result,
          });
          res.json(cliResultToOpenai(finalResult, requestId));
        }
      } else {
        logger.error("[NonStreaming] Subprocess closed without result", { requestId, code });
        if (sessionCtx.resume && sessionCtx.sessionKey) {
          clearSession(sessionCtx.sessionKey);
        }
        releaseLock(sessionCtx);
        if (!res.headersSent) {
          res.status(500).json({
            error: {
              message: `Claude CLI exited with code ${code} without response`,
              type: "server_error",
              code: null,
            },
          });
        }
      }
      resolve();
    });

    // Start the subprocess
    subprocess
      .start(cliInput.prompt, {
        model: cliInput.model,
        sessionId: cliInput.sessionId,
        resume: sessionCtx.resume,
        systemPrompt: cliInput.systemPrompt,
        delegateTools: delegateMode,
      })
      .catch((error) => {
        logger.error("[NonStreaming] Subprocess start error", { requestId, error: error.message });
        releaseLock(sessionCtx);
        res.status(500).json({
          error: {
            message: error.message,
            type: "server_error",
            code: null,
          },
        });
        resolve();
      });
  });
}

/**
 * Handle GET /v1/models
 *
 * Returns available models
 */
export function handleModels(_req: Request, res: Response): void {
  const now = Math.floor(Date.now() / 1000);
  const modelIds = [
    "claude-opus-4",
    "claude-opus-4-6",
    "claude-sonnet-4",
    "claude-sonnet-4-5",
    "claude-sonnet-4-6",
    "claude-sonnet-5",
    "claude-opus-5",
    "claude-haiku-4",
    "claude-haiku-4-5",
  ];
  res.json({
    object: "list",
    data: modelIds.map((id) => ({
      id,
      object: "model",
      owned_by: "anthropic",
      created: now,
    })),
  });
}

/**
 * Handle GET /health
 *
 * Health check endpoint
 */
export function handleHealth(_req: Request, res: Response): void {
  res.json({
    status: "ok",
    provider: "claude-code-cli",
    timestamp: new Date().toISOString(),
  });
}

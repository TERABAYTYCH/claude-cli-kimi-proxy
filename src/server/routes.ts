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
} from "../adapter/cli-to-openai.js";
import { getSession, setSession, clearSession, acquireSession, releaseSession, addUsage } from "../subprocess/session-store.js";
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
function resolveCliInput(body: OpenAIChatRequest): {
  cliInput: ReturnType<typeof openaiToCli>;
  sessionKey: string | undefined;
  resume: boolean;
  persistSession: boolean;
  /** True if this request holds the per-key inflight lock (see session-store). */
  lockHeld: boolean;
} {
  // Kimi may identify a conversation via `user` or `prompt_cache_key`.
  // We use that as our in-memory lookup key, but Claude CLI requires a
  // valid UUID for --session-id / --resume, so we keep a separate UUID.
  const sessionKey =
    body.user || (body as { prompt_cache_key?: string }).prompt_cache_key;
  const existing = sessionKey ? getSession(sessionKey) : undefined;
  const delegateMode = !!body.tools?.length;

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
    const deltaValid = isDeltaValid(body, existing.messageCount, delegateMode);
    if (deltaValid && acquireSession(sessionKey as string)) {
      const cliInput = openaiToCliDelta(body, existing.messageCount);
      cliInput.sessionId = existing.claudeSessionId;
      return { cliInput, sessionKey, resume: true, persistSession: true, lockHeld: true };
    }

    if (deltaValid) {
      // Key is held by an in-flight request (parallel agents sharing a key).
      // Replaying full history under a fresh session is correct but must not
      // clobber the stored entry owned by the in-flight conversation.
      logger.warn("[Session] Key busy, falling back to full history without resume", {
        sessionKey,
      });
      const cliInput = openaiToCli(body);
      cliInput.sessionId = uuidv4();
      return { cliInput, sessionKey: undefined, resume: false, persistSession: false, lockHeld: false };
    }

    logger.warn("[Session] Delta validation failed, restarting session from full history", {
      sessionKey,
      messageCount: body.messages.length,
      sinceIndex: existing.messageCount,
      delegateMode,
    });
    clearSession(sessionKey as string);
    const cliInput = openaiToCli(body);
    cliInput.sessionId = uuidv4();
    // Take the key for the restarted session so a parallel request cannot
    // --resume it while it is being rebuilt. If the key is busy, run unlocked
    // and without persisting — whoever holds the lock owns the stored entry.
    if (acquireSession(sessionKey as string)) {
      return { cliInput, sessionKey, resume: false, persistSession: true, lockHeld: true };
    }
    logger.warn("[Session] Key busy during delta-invalid restart, not persisting", {
      sessionKey,
    });
    return { cliInput, sessionKey: undefined, resume: false, persistSession: false, lockHeld: false };
  }

  const cliInput = openaiToCli(body);
  if (sessionKey) {
    cliInput.sessionId = uuidv4(); // pin a known UUID so we can --resume it later
  }
  return { cliInput, sessionKey, resume: false, persistSession: true, lockHeld: false };
}

/**
 * A resumed CLI session already contains everything up to `sinceIndex`
 * (the CLI generated those turns itself). The delta is trustworthy only if
 * the sliced messages look like a normal append:
 * - indices in range and the slice is non-empty;
 * - in delegate mode, at least one tool result (every delegate round-trip
 *   appends an assistant tool_call + a tool message).
 */
function isDeltaValid(
  body: OpenAIChatRequest,
  sinceIndex: number,
  delegateMode: boolean
): boolean {
  if (!Number.isInteger(sinceIndex) || sinceIndex < 0) return false;
  if (sinceIndex >= body.messages.length) return false;
  const slice = body.messages.slice(sinceIndex);
  if (slice.length === 0) return false;
  if (delegateMode && !slice.some((m) => m.role === "tool")) return false;
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
    const { cliInput, sessionKey, resume, persistSession, lockHeld } = resolveCliInput(body);
    const subprocess = new ClaudeSubprocess();
    const sessionCtx: SessionContext = {
      sessionKey,
      resume,
      messageCount: body.messages.length,
      persistSession,
      lockHeld,
    };

    logger.info("[ChatCompletions] Request prepared", {
      requestId,
      stream,
      model: body.model,
      resume,
      hasSessionKey: !!sessionKey,
      hasSystemPrompt: !!cliInput.systemPrompt,
      hasTools: !!body.tools?.length,
      toolNames: body.tools?.map((t) => t.function.name),
      promptPreview: cliInput.prompt.slice(0, 200),
    });

    if (stream) {
      await handleStreamingResponse(req, res, subprocess, cliInput, requestId, sessionCtx, body);
    } else {
      await handleNonStreamingResponse(res, subprocess, cliInput, requestId, sessionCtx, body);
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown error";
    console.error("[handleChatCompletions] Error:", message);

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
    let delegateEmitted = false;
    let finished = false;

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

      const toolCalls = delegationsToToolCalls(delegations, "call", body.tools);
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
        if (textBuffer.includes("</invoke>") && emitDelegateAndEnd()) {
          return;
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
        doneChunk.usage = {
          prompt_tokens: result.usage.input_tokens || 0,
          completion_tokens: result.usage.output_tokens || 0,
          total_tokens:
            (result.usage.input_tokens || 0) + (result.usage.output_tokens || 0),
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
        if (delegations.length > 0) {
          const toolCalls = delegationsToToolCalls(delegations, "call", body.tools);
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
            usage: {
              prompt_tokens: finalResult.usage?.input_tokens || 0,
              completion_tokens: finalResult.usage?.output_tokens || 0,
              total_tokens:
                (finalResult.usage?.input_tokens || 0) + (finalResult.usage?.output_tokens || 0),
            },
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

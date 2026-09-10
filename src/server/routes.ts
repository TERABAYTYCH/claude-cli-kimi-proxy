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
} from "../adapter/cli-to-openai.js";
import { getSession, setSession, clearSession } from "../subprocess/session-store.js";
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
} {
  // Kimi may identify a conversation via `user` or `prompt_cache_key`.
  // We use that as our in-memory lookup key, but Claude CLI requires a
  // valid UUID for --session-id / --resume, so we keep a separate UUID.
  const sessionKey =
    body.user || (body as { prompt_cache_key?: string }).prompt_cache_key;
  const existing = sessionKey ? getSession(sessionKey) : undefined;

  // Delegate mode: always send the full history. --resume loses context when
  // combined with --system-prompt-file, causing the model to repeat the same
  // tool call over and over.
  const delegateMode = !!body.tools?.length;
  if (existing && !delegateMode) {
    const cliInput = openaiToCliDelta(body, existing.messageCount);
    cliInput.sessionId = existing.claudeSessionId;
    return { cliInput, sessionKey, resume: true };
  }

  const cliInput = openaiToCli(body);
  if (sessionKey) {
    cliInput.sessionId = uuidv4(); // pin a known UUID so we can --resume it later
  }
  return { cliInput, sessionKey, resume: false };
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
    const { cliInput, sessionKey, resume } = resolveCliInput(body);
    const subprocess = new ClaudeSubprocess();
    const sessionCtx: SessionContext = { sessionKey, resume, messageCount: body.messages.length };

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
      subprocess.kill();
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
      // Persist the session before killing the subprocess so the next turn can
      // --resume and continue the conversation instead of cold-starting.
      if (sessionCtx.sessionKey && cliInput.sessionId) {
        setSession(sessionCtx.sessionKey, cliInput.sessionId, sessionCtx.messageCount);
      }
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
        // Client disconnected before response completed - kill subprocess
        subprocess.kill();
      }
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

    // Handle final assistant message (for model name)
    subprocess.on("assistant", (message: ClaudeCliAssistant) => {
      lastModel = message.message.model;
    });

    subprocess.on("result", (result: ClaudeCliResult) => {
      isComplete = true;
      if (sessionCtx.sessionKey && cliInput.sessionId) {
        setSession(sessionCtx.sessionKey, cliInput.sessionId, sessionCtx.messageCount);
      }

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
      // Subprocess exited - ensure response is closed
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
    });

    subprocess.on("error", (error: Error) => {
      logger.error("[NonStreaming] Error", { requestId, error: error.message });
      if (sessionCtx.resume && sessionCtx.sessionKey) {
        clearSession(sessionCtx.sessionKey);
      }
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
        if (sessionCtx.sessionKey && cliInput.sessionId) {
          setSession(sessionCtx.sessionKey, cliInput.sessionId, sessionCtx.messageCount);
        }
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

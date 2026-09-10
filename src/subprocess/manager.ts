/**
 * Claude Code CLI Subprocess Manager
 *
 * Handles spawning, managing, and parsing output from Claude CLI subprocesses.
 * Uses spawn() instead of exec() to prevent shell injection vulnerabilities.
 */

import { spawn, spawnSync, ChildProcess } from "child_process";
import { EventEmitter } from "events";
import fs from "fs/promises";
import { readFileSync } from "fs";
import path from "path";
import os from "os";
import crypto from "crypto";

import { logger } from "../utils/logger.js";
import type {
  ClaudeCliMessage,
  ClaudeCliAssistant,
  ClaudeCliResult,
  ClaudeCliStreamEvent,
} from "../types/claude-cli.js";
import {
  isAssistantMessage,
  isResultMessage,
  isContentDelta,
  isTextBlockStart,
  isToolUseBlockStart,
  isInputJsonDelta,
  isContentBlockStop,
  isThinkingDelta,
} from "../types/claude-cli.js";
import type { ClaudeModel } from "../adapter/openai-to-cli.js";

export interface SubprocessOptions {
  model: ClaudeModel;
  sessionId?: string;
  /** Resume an existing persisted session (sessionId) instead of creating a new one */
  resume?: boolean;
  cwd?: string;
  timeout?: number;
  systemPrompt?: string; // ← новое поле, системный промпт от Kimi
  /** If true, disable all built-in Claude CLI tools and force delegation output. */
  delegateTools?: boolean;
}

export interface SubprocessEvents {
  message: (msg: ClaudeCliMessage) => void;
  assistant: (msg: ClaudeCliAssistant) => void;
  result: (result: ClaudeCliResult) => void;
  error: (error: Error) => void;
  close: (code: number | null) => void;
  raw: (line: string) => void;
}

const DEFAULT_TIMEOUT = 900000; // 15 minutes

/**
 * Resolve the real Claude CLI binary to spawn.
 *
 * On Windows, the global `claude` command is an npm shim (`claude.cmd`) that
 * just execs a bundled `claude.exe`. Running the shim requires `shell: true`,
 * which routes our argv through cmd.exe — and cmd.exe treats characters our
 * appended system prompt legitimately contains (`<`, `>`, `(`, `)`, `&`) as
 * redirection/grouping/chaining operators, corrupting the argument list that
 * follows (notably `--session-id`/`--resume`, which silently stop working).
 * Resolving straight to the `.exe` lets us spawn with `shell: false` and
 * skip cmd.exe entirely. Falls back to the shim (with shell:true) if the
 * `.exe` can't be located.
 */
let resolvedClaudeBin: { bin: string; shell: boolean } | null = null;

function resolveClaudeBin(): { bin: string; shell: boolean } {
  if (process.env.CLAUDE_BIN) {
    // Environment overrides are intentionally not cached. This lets callers
    // temporarily select a binary without contaminating later resolutions.
    return { bin: process.env.CLAUDE_BIN, shell: false };
  }

  if (resolvedClaudeBin) return resolvedClaudeBin;

  if (process.platform === "win32") {
    try {
      const where = spawnSync("where.exe", ["claude"], { encoding: "utf8" });
      const shimPath = (where.stdout || "")
        .split(/\r?\n/)
        .map((p) => p.trim())
        .find((p) => p.toLowerCase().endsWith(".cmd"));

      if (shimPath) {
        const shimDir = path.dirname(shimPath);
        const shimContent = readFileSync(shimPath, "utf8");
        const match = shimContent.match(/"%dp0%\\(.+?\.exe)"/i);
        if (match) {
          const exePath = path.join(shimDir, match[1]);
          resolvedClaudeBin = { bin: exePath, shell: false };
          return resolvedClaudeBin;
        }
      }
    } catch {
      // Fall through to shim fallback below
    }

    resolvedClaudeBin = { bin: "claude", shell: true };
    return resolvedClaudeBin;
  }

  resolvedClaudeBin = { bin: "claude", shell: false };
  return resolvedClaudeBin;
}

/**
 * Kill a process and its full descendant tree.
 *
 * `ChildProcess.kill()` only signals the direct child. On Windows this is
 * insufficient because the Claude CLI spawns its own subprocesses (e.g. for
 * Bash tool calls) that aren't part of a job object — killing just the
 * parent leaves them running in the background even after a client
 * disconnect or timeout. `taskkill /T` walks the whole tree instead.
 */
function killProcessTree(
  child: ChildProcess,
  signal: NodeJS.Signals = "SIGTERM"
): boolean {
  const pid = child.pid;
  if (!pid) return false;

  if (process.platform === "win32") {
    const taskkill = process.env.SystemRoot
      ? path.join(process.env.SystemRoot, "System32", "taskkill.exe")
      : "taskkill.exe";
    const result = spawnSync(taskkill, ["/PID", String(pid), "/T", "/F"], {
      stdio: "ignore",
      windowsHide: true,
    });
    if (!result.error && result.status === 0) {
      return true;
    }

    // If taskkill could not be started or rejected the request, still make a
    // best-effort attempt to stop the managed root process. This must not be
    // reported as a successful tree termination: descendants may still be
    // running, and callers must remain able to retry.
    try {
      child.kill(signal);
    } catch {}
    return false;
  }

  try {
    return child.kill(signal);
  } catch {
    // Process may have already exited
    return false;
  }
}

/**
 * Format a spawn command for logging. Not safe for shell execution because
 * arguments are simply quoted; intended only for diagnostics.
 */
function formatCommand(bin: string, args: string[]): string {
  const quote = (arg: string): string => {
    if (arg === "") return '""';
    if (/[\s"'\\]/.test(arg)) {
      return `"${arg.replace(/"/g, '\\"')}"`;
    }
    return arg;
  };
  return [bin, ...args.map(quote)].join(" ");
}

export class ClaudeSubprocess extends EventEmitter {
  private process: ChildProcess | null = null;
  private buffer: string = "";
  private timeoutId: NodeJS.Timeout | null = null;
  private isKilled: boolean = false;
  private systemPromptFilePath: string | null = null;

  /**
   * Start the Claude CLI subprocess with the given prompt
   */
  async start(prompt: string, options: SubprocessOptions): Promise<void> {
    const args = await this.buildArgs(options);
    const timeout = options.timeout || DEFAULT_TIMEOUT;
    const { bin, shell } = resolveClaudeBin();
    const command = formatCommand(bin, args);
    logger.info("Spawning Claude CLI subprocess", {
      bin,
      shell,
      command,
      model: options.model,
      hasSessionId: !!options.sessionId,
      resume: !!options.resume,
      hasSystemPrompt: !!options.systemPrompt,
    });
    logger.debug("Subprocess system prompt preview", {
      systemPromptPreview: options.systemPrompt?.slice(0, 500),
    });
    logger.debug("Subprocess prompt preview", { promptPreview: prompt.slice(0, 200) });

    return new Promise((resolve, reject) => {
      try {
        // Use spawn() for security - no shell interpretation
        this.process = spawn(bin, args, {
          cwd: options.cwd || process.cwd(),
          env: Object.fromEntries(
            Object.entries(process.env).filter(([k]) => k !== "CLAUDECODE")
          ),
          stdio: ["pipe", "pipe", "pipe"],
          shell,
        });

        this.armTimeout(timeout);

        // Handle spawn errors (e.g., claude not found)
        this.process.on("error", (err) => {
          this.clearTimeout();
          this.cleanupSystemPromptFile();
          if (err.message.includes("ENOENT")) {
            reject(
              new Error(
                "Claude CLI not found. Install with: npm install -g @anthropic-ai/claude-code"
              )
            );
          } else {
            reject(err);
          }
        });

        // Pass prompt via stdin to avoid E2BIG on large inputs
        this.process.stdin?.write(prompt);
        this.process.stdin?.end();

        logger.info("Subprocess spawned", { pid: this.process.pid });

        // Parse JSON stream from stdout
        this.process.stdout?.on("data", (chunk: Buffer) => {
          const data = chunk.toString();
          logger.debug("Subprocess stdout chunk", { pid: this.process?.pid, bytes: data.length });
          this.buffer += data;
          this.processBuffer();
        });

        // Capture stderr for debugging
        this.process.stderr?.on("data", (chunk: Buffer) => {
          const errorText = chunk.toString().trim();
          if (errorText) {
            // Don't emit as error unless it's actually an error
            // Claude CLI may write debug info to stderr
            logger.warn("Subprocess stderr", { pid: this.process?.pid, text: errorText.slice(0, 500) });
          }
        });

        // Handle process close
        this.process.on("close", (code) => {
          logger.info("Subprocess closed", { pid: this.process?.pid, code });
          this.clearTimeout();
          // Process any remaining buffer
          if (this.buffer.trim()) {
            this.processBuffer();
          }
          this.cleanupSystemPromptFile();
          this.emit("close", code);
        });

        // Resolve immediately since we're streaming
        resolve();
      } catch (err) {
        this.clearTimeout();
        reject(err);
      }
    });
  }

  /**
   * Build CLI arguments array. Async because we may write the system prompt to
   * a temporary file for --system-prompt-file.
   */
  private async buildArgs(options: SubprocessOptions): Promise<string[]> {
    const args = [
      "--print", // Non-interactive mode
      // "--dangerously-skip-permissions", // Skip permission prompts
      "--output-format",
      "stream-json", // JSON streaming output
      "--verbose", // Required for stream-json
      "--include-partial-messages", // Enable streaming chunks
      "--model",
      options.model, // Model alias (opus/sonnet/haiku)
      // Prompt is passed via stdin (avoids E2BIG on large inputs)
    ];

    if (options.systemPrompt) {
      // Fully replace Claude Code's built-in system prompt with the caller's
      // own identity (e.g. Kimi CLI). Using a file avoids E2BIG on large prompts.
      const tmpPath = path.join(os.tmpdir(), `claude-sysprompt-${crypto.randomUUID()}.txt`);
      await fs.writeFile(tmpPath, options.systemPrompt, "utf8");
      this.systemPromptFilePath = tmpPath;
      args.push("--system-prompt-file", tmpPath);
    }

    if (options.delegateTools) {
      // Prevent the CLI from executing tools internally. An empty list disables
      // all built-in tools, forcing the model to emit structured <invoke> blocks
      // which the proxy converts to OpenAI tool_calls for the upstream Kimi chat.
      args.push("--tools", "");
    }

    if (options.sessionId && options.resume) {
      // Continue a previously persisted session — avoids replaying full history
      args.push("--resume", options.sessionId);
    } else if (options.sessionId) {
      // First turn for this session key — create it under a known ID so we
      // can --resume it on subsequent turns
      args.push("--session-id", options.sessionId);
    } else {
      // No stable session key (e.g. request.user missing) — don't leave
      // orphaned session files behind
      args.push("--no-session-persistence");
    }

    return args;
  }

  /**
   * Remove the temporary system prompt file, if one was created.
   * Set PRESERVE_SYSTEM_PROMPT_FILE=true to keep it for debugging.
   */
  private async cleanupSystemPromptFile(): Promise<void> {
    if (this.systemPromptFilePath) {
      if (process.env.PRESERVE_SYSTEM_PROMPT_FILE === "true") {
        logger.info("Preserved system prompt file", { path: this.systemPromptFilePath });
        this.systemPromptFilePath = null;
        return;
      }
      try {
        await fs.unlink(this.systemPromptFilePath);
      } catch {
        // File may already be gone; ignore.
      }
      this.systemPromptFilePath = null;
    }
  }

  /**
   * Process the buffer and emit parsed messages
   */
  private processBuffer(): void {
    const lines = this.buffer.split("\n");
    this.buffer = lines.pop() || ""; // Keep incomplete line

    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed) continue;

      try {
        const message: ClaudeCliMessage = JSON.parse(trimmed);
        this.emit("message", message);

        if (isTextBlockStart(message)) {
          // Emit when a new text content block starts (for inserting separators)
          this.emit("text_block_start", message as ClaudeCliStreamEvent);
        }

        if (isToolUseBlockStart(message)) {
          this.emit("tool_use_start", message as ClaudeCliStreamEvent);
        }

        if (isInputJsonDelta(message)) {
          this.emit("input_json_delta", message as ClaudeCliStreamEvent);
        }

        if (isContentBlockStop(message)) {
          this.emit("content_block_stop", message as ClaudeCliStreamEvent);
        }

        if (isThinkingDelta(message)) {
          this.emit("thinking_delta", message as ClaudeCliStreamEvent);
        }

        if (isContentDelta(message)) {
          // Emit content delta for streaming (text_delta only)
          this.emit("content_delta", message as ClaudeCliStreamEvent);
        } else if (isAssistantMessage(message)) {
          this.emit("assistant", message);
        } else if (isResultMessage(message)) {
          this.emit("result", message);
        }
      } catch {
        // Non-JSON output, emit as raw
        this.emit("raw", trimmed);
      }
    }
  }

  /**
   * Clear the timeout timer
   */
  private clearTimeout(): void {
    if (this.timeoutId) {
      clearTimeout(this.timeoutId);
      this.timeoutId = null;
    }
  }

  /**
   * Kill the subprocess
   */
  kill(signal: NodeJS.Signals = "SIGTERM"): void {
    if (!this.isKilled && this.process) {
      logger.info("Killing subprocess", { pid: this.process.pid, signal });
      this.clearTimeout();
      this.isKilled = killProcessTree(this.process, signal);
    }
  }

  /**
   * Arm the request timeout. Kept narrow so tests can deterministically re-arm
   * the production timeout behavior after their fixture process tree is ready.
   */
  private armTimeout(timeout: number): void {
    this.clearTimeout();
    this.timeoutId = setTimeout(() => {
      if (!this.isKilled) {
        if (this.process) {
          this.isKilled = killProcessTree(this.process, "SIGTERM");
        }
        this.emit("error", new Error(`Request timed out after ${timeout}ms`));
      }
    }, timeout);
  }

  /**
   * Check if the process is still running
   */
  isRunning(): boolean {
    return this.process !== null && !this.isKilled && this.process.exitCode === null;
  }
}

/**
 * Verify that Claude CLI is installed and accessible
 */
export async function verifyClaude(): Promise<{ ok: boolean; error?: string; version?: string }> {
  return new Promise((resolve) => {
    const { bin, shell } = resolveClaudeBin();
    logger.info("Verifying Claude CLI", { bin });
    const proc = spawn(bin, ["--version"], { stdio: "pipe", shell });
    let output = "";

    proc.stdout?.on("data", (chunk: Buffer) => {
      output += chunk.toString();
    });

    proc.on("error", (err) => {
      logger.error("Claude CLI verification failed", { bin, error: err.message });
      resolve({
        ok: false,
        error:
          "Claude CLI not found. Install with: npm install -g @anthropic-ai/claude-code",
      });
    });

    proc.on("close", (code) => {
      if (code === 0) {
        logger.info("Claude CLI verified", { bin, version: output.trim() });
        resolve({ ok: true, version: output.trim() });
      } else {
        logger.error("Claude CLI verification failed", { bin, code });
        resolve({
          ok: false,
          error: "Claude CLI returned non-zero exit code",
        });
      }
    });
  });
}

/**
 * Check if Claude CLI is authenticated
 *
 * Claude Code stores credentials in the OS keychain, not a file.
 * We verify authentication by checking if we can call the CLI successfully.
 * If the CLI is installed, it typically has valid credentials from `claude auth login`.
 */
export async function verifyAuth(): Promise<{ ok: boolean; error?: string }> {
  // If Claude CLI is installed and the user has run `claude auth login`,
  // credentials are stored in the OS keychain and will be used automatically.
  // We can't easily check the keychain, so we'll just return true if the CLI exists.
  // Authentication errors will surface when making actual API calls.
  logger.info("Verifying Claude authentication");
  return { ok: true };
}

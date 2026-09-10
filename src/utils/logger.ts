/**
 * Simple file + console logger.
 *
 * Writes every log line to stdout and to logs/app.log (configurable via env).
 * No external dependencies.
 */

import fs from "fs";
import path from "path";

export type LogLevel = "debug" | "info" | "warn" | "error";

const LEVEL_PRIORITY: Record<LogLevel, number> = {
  debug: 0,
  info: 1,
  warn: 2,
  error: 3,
};

function parseLevel(level: string | undefined): LogLevel {
  const normalized = (level || "info").toLowerCase() as LogLevel;
  if (normalized in LEVEL_PRIORITY) return normalized;
  return "info";
}

function formatMeta(meta: Record<string, unknown> | undefined): string {
  if (!meta || Object.keys(meta).length === 0) return "";
  try {
    return " " + JSON.stringify(meta);
  } catch {
    return " [meta serialization failed]";
  }
}

class Logger {
  private stream: fs.WriteStream | null = null;
  private level: LogLevel;

  constructor() {
    this.level = parseLevel(process.env.LOG_LEVEL);

    const disabled = process.env.LOG_FILE === "false";
    if (disabled) return;

    const logFile = process.env.LOG_FILE || path.join("logs", "app.log");
    fs.mkdirSync(path.dirname(logFile), { recursive: true });
    this.stream = fs.createWriteStream(logFile, { flags: "a" });
  }

  private log(level: LogLevel, message: string, meta?: Record<string, unknown>) {
    if (LEVEL_PRIORITY[level] < LEVEL_PRIORITY[this.level]) return;

    const timestamp = new Date().toISOString();
    const line = `[${timestamp}] [${level.toUpperCase()}] ${message}${formatMeta(meta)}\n`;

    // Console
    if (level === "error") {
      process.stderr.write(line);
    } else {
      process.stdout.write(line);
    }

    // File
    this.stream?.write(line);
  }

  debug(message: string, meta?: Record<string, unknown>) {
    this.log("debug", message, meta);
  }

  info(message: string, meta?: Record<string, unknown>) {
    this.log("info", message, meta);
  }

  warn(message: string, meta?: Record<string, unknown>) {
    this.log("warn", message, meta);
  }

  error(message: string, meta?: Record<string, unknown>) {
    this.log("error", message, meta);
  }
}

export const logger = new Logger();

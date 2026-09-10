/**
 * Express HTTP Server
 *
 * Provides OpenAI-compatible API endpoints that wrap Claude Code CLI
 */

import express, { Express, Request, Response, NextFunction } from "express";
import { createServer, Server } from "http";
import { v4 as uuidv4 } from "uuid";
import { handleChatCompletions, handleModels, handleHealth } from "./routes.js";
import { logger } from "../utils/logger.js";

export interface ServerConfig {
  port: number;
  host?: string;
}

let serverInstance: Server | null = null;

interface RequestWithId extends Request {
  requestId: string;
}

function getRequestId(req: Request): string {
  return (req as RequestWithId).requestId || "unknown";
}

function truncateString(value: unknown, maxLen = 500): unknown {
  if (typeof value === "string" && value.length > maxLen) {
    return value.slice(0, maxLen) + "... [truncated]";
  }
  return value;
}

function sanitizeBody(body: unknown): unknown {
  if (!body || typeof body !== "object") return body;
  try {
    const clone = JSON.parse(JSON.stringify(body));
    // Full tool schemas are multi-KB each — never useful in a log line.
    if (Array.isArray(clone.tools)) {
      clone.tools = `[${clone.tools.length} tools omitted]`;
    }
    if (Array.isArray(clone.messages)) {
      for (const msg of clone.messages) {
        if (msg && typeof msg.content === "string") {
          msg.content = truncateString(msg.content, 500);
        }
        if (msg && typeof msg.reasoning_content === "string") {
          msg.reasoning_content = truncateString(msg.reasoning_content, 200);
        }
      }
    }
    return clone;
  } catch {
    return body;
  }
}

/**
 * Create and configure the Express app
 */
function createApp(): Express {
  const app = express();

  // Request ID + response-time tracking (must run first)
  app.use((req: Request, res: Response, next: NextFunction) => {
    const requestId = uuidv4().replace(/-/g, "").slice(0, 16);
    (req as RequestWithId).requestId = requestId;

    const start = Date.now();
    const originalEnd = res.end.bind(res);
    res.end = function (this: Response, ...args: unknown[]): Response {
      res.end = originalEnd;
      const duration = Date.now() - start;
      logger.info(`<- ${req.method} ${req.originalUrl} ${res.statusCode} ${duration}ms`, {
        requestId,
        statusCode: res.statusCode,
        duration,
      });
      return originalEnd(...(args as Parameters<Response["end"]>));
    } as any;

    next();
  });

  // Middleware: use raw body parser + manual JSON parse for better error diagnostics
  app.use(express.raw({ type: "application/json", limit: "10mb" }));
  app.use((req: Request, _res: Response, next: NextFunction) => {
    if (req.body && Buffer.isBuffer(req.body) && req.body.length > 0) {
      const raw = req.body.toString("utf8");
      logger.debug("Received raw request body", { requestId: getRequestId(req), length: raw.length });
      try {
        req.body = JSON.parse(raw);
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        logger.error("Failed to parse request body", {
          requestId: getRequestId(req),
          error: msg,
          length: raw.length,
          method: req.method,
          url: req.originalUrl,
        });
        return next(err);
      }
    }
    next();
  });

  // Log incoming request after body has been parsed
  app.use((req: Request, _res: Response, next: NextFunction) => {
    // Request bodies are off by default: full tool schemas make every line
    // multi-KB and the log grows ~20MB/day. Opt back in with LOG_SENSITIVE=true.
    const logBody = req.method !== "GET" && process.env.LOG_SENSITIVE === "true";
    logger.info(`-> ${req.method} ${req.originalUrl}`, {
      requestId: getRequestId(req),
      ip: req.ip || req.socket.remoteAddress,
      userAgent: req.get("user-agent"),
      ...(logBody && req.body ? { body: sanitizeBody(req.body) } : {}),
    });
    next();
  });

  // CORS headers for local development
  app.use((_req: Request, res: Response, next: NextFunction) => {
    res.setHeader("Access-Control-Allow-Origin", "*");
    res.setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
    res.setHeader("Access-Control-Allow-Headers", "Content-Type, Authorization");
    next();
  });

  // Handle OPTIONS preflight
  app.options("*", (_req: Request, res: Response) => {
    res.sendStatus(200);
  });

  // Routes
  app.get("/health", handleHealth);
  app.get("/v1/models", handleModels);
  app.post("/v1/chat/completions", handleChatCompletions);

  // 404 handler
  app.use((req: Request, res: Response) => {
    logger.warn("Route not found", { requestId: getRequestId(req), method: req.method, url: req.originalUrl });
    res.status(404).json({
      error: {
        message: "Not found",
        type: "invalid_request_error",
        code: "not_found",
      },
    });
  });

  // Error handler
  app.use((err: Error, req: Request, res: Response, _next: NextFunction) => {
    logger.error("Unhandled server error", { requestId: getRequestId(req), error: err.message, stack: err.stack });
    res.status(500).json({
      error: {
        message: err.message,
        type: "server_error",
        code: null,
      },
    });
  });

  return app;
}

/**
 * Start the HTTP server
 */
export async function startServer(config: ServerConfig): Promise<Server> {
  const { port, host = "127.0.0.1" } = config;

  if (serverInstance) {
    logger.info("Server already running, returning existing instance");
    return serverInstance;
  }

  const app = createApp();

  return new Promise((resolve, reject) => {
    serverInstance = createServer(app);

    serverInstance.on("error", (err: NodeJS.ErrnoException) => {
      logger.error("Server error", { error: err.message, code: err.code });
      if (err.code === "EADDRINUSE") {
        reject(new Error(`Port ${port} is already in use`));
      } else {
        reject(err);
      }
    });

    serverInstance.listen(port, host, () => {
      logger.info(`Server listening at http://${host}:${port}`);
      logger.info(`OpenAI-compatible endpoint: http://${host}:${port}/v1/chat/completions`);
      resolve(serverInstance!);
    });
  });
}

/**
 * Stop the HTTP server
 */
export async function stopServer(): Promise<void> {
  if (!serverInstance) {
    return;
  }

  return new Promise((resolve, reject) => {
    serverInstance!.close((err) => {
      if (err) {
        reject(err);
      } else {
        logger.info("Server stopped");
        serverInstance = null;
        resolve();
      }
    });
  });
}

/**
 * Get the current server instance
 */
export function getServer(): Server | null {
  return serverInstance;
}

#!/usr/bin/env node
/**
 * Standalone server for testing without Clawdbot
 *
 * Usage:
 *   npm run start
 *   # or
 *   node dist/server/standalone.js [port]
 */

import { startServer, stopServer } from "./index.js";
import { verifyClaude, verifyAuth } from "../subprocess/manager.js";
import { logger } from "../utils/logger.js";

const DEFAULT_PORT = 3456;

async function main(): Promise<void> {
  logger.info("Claude Code CLI Provider - Standalone Server starting");

  // Parse port from command line
  const port = parseInt(process.argv[2] || String(DEFAULT_PORT), 10);
  if (isNaN(port) || port < 1 || port > 65535) {
    logger.error(`Invalid port: ${process.argv[2]}`);
    process.exit(1);
  }

  // Verify Claude CLI
  logger.info("Checking Claude CLI...");
  const cliCheck = await verifyClaude();
  if (!cliCheck.ok) {
    logger.error(`Claude CLI check failed: ${cliCheck.error}`);
    process.exit(1);
  }
  logger.info(`Claude CLI: ${cliCheck.version || "OK"}`);

  // Verify authentication
  logger.info("Checking authentication...");
  const authCheck = await verifyAuth();
  if (!authCheck.ok) {
    logger.error(`Authentication check failed: ${authCheck.error}`);
    logger.error("Please run: claude auth login");
    process.exit(1);
  }
  logger.info("Authentication: OK");

  // Start server
  try {
    await startServer({ port });
    logger.info("Server ready. Test with:");
    logger.info(`  curl -X POST http://localhost:${port}/v1/chat/completions -H "Content-Type: application/json" -d '{"model": "claude-sonnet-4", "messages": [{"role": "user", "content": "Hello!"}]}'`);
    logger.info("Press Ctrl+C to stop.");
  } catch (err) {
    const errorMsg = err instanceof Error ? err.message : String(err);
    logger.error("Failed to start server", { error: errorMsg });
    process.exit(1);
  }

  // Handle graceful shutdown
  const shutdown = async () => {
    logger.info("Shutting down...");
    await stopServer();
    process.exit(0);
  };

  process.on("SIGINT", shutdown);
  process.on("SIGTERM", shutdown);
}

main().catch((err) => {
  logger.error("Unexpected error", { error: err instanceof Error ? err.message : String(err) });
  process.exit(1);
});

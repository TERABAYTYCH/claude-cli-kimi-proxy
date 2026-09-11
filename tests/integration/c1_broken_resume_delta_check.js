#!/usr/bin/env node
"use strict";

/**
 * C1 regression test: resume delta health checks must flag structural
 * regressions (assistant history replayed or Kimi identity leaked into the
 * prompt body) and must stay silent on legitimate large batches of
 * independent tool results.
 */

import assert from "assert";
import { checkResumeDeltaHealth } from "../../dist/session/resume-delta-guard.js";
import { logger } from "../../dist/utils/logger.js";

const warnings = [];
const origWarn = logger.warn.bind(logger);
logger.warn = (msg, meta) => warnings.push({ msg, meta });

function reset() {
  warnings.length = 0;
}

function hasWarn(message) {
  return warnings.some((w) => w.msg === message);
}

const REPLAYED_HISTORY_WARN = "[Session] Broken resume delta: assistant history replayed in prompt";
const IDENTITY_LEAK_WARN = "[Session] Broken resume delta: Kimi identity leaked into prompt body";
const SIZE_LIMIT_WARN = "[Session] Resume delta exceeds soft size limit";

// 1. Assistant history replayed in a resume delta (before any tool_result).
reset();
checkResumeDeltaHealth(
  "some context\n<previous_response>\n<invoke name=\"Read\">...</invoke>\n</previous_response>\n<tool_result>...result...</tool_result>",
  true,
  "session-replay"
);
assert(hasWarn(REPLAYED_HISTORY_WARN), "expected previous_response before tool_result to trigger replay warning");
assert(!hasWarn(IDENTITY_LEAK_WARN), "did not expect identity leak warning");
assert(!hasWarn(SIZE_LIMIT_WARN), "did not expect size warning");
console.log("PASS: previous_response before any tool_result triggers structural warn");

// 2. Kimi identity leaked into the prompt body on a resume delta.
reset();
checkResumeDeltaHealth(
  "You are Kimi Code CLI, an interactive general AI agent...",
  true,
  "session-identity"
);
assert(hasWarn(IDENTITY_LEAK_WARN), "expected Kimi identity prefix to trigger identity warning");
assert(!hasWarn(REPLAYED_HISTORY_WARN), "did not expect replay warning");
assert(!hasWarn(SIZE_LIMIT_WARN), "did not expect size warning");
console.log("PASS: Kimi identity in resume delta prefix triggers structural warn");

// 3. Identity string only outside the prefix must NOT warn.
reset();
checkResumeDeltaHealth(
  "x".repeat(1000) + "You are Kimi Code CLI" + "x".repeat(1000),
  true,
  "session-far-identity"
);
assert(!hasWarn(IDENTITY_LEAK_WARN), "identity outside prefix should not warn");
assert.strictEqual(warnings.length, 0, "expected no warnings for far identity");
console.log("PASS: Kimi identity outside prefix is ignored");

// 3b. previous_response inside a tool_result (quoted source code) must NOT warn.
reset();
checkResumeDeltaHealth(
  "<tool_result>\nfunction render() {\n  return '<previous_response>...</previous_response>';\n}\n</tool_result>",
  true,
  "session-quoted-previous-response"
);
assert(!hasWarn(REPLAYED_HISTORY_WARN), "previous_response inside tool_result should not warn");
assert.strictEqual(warnings.length, 0, "expected no warnings for quoted previous_response");
console.log("PASS: previous_response inside tool_result is ignored");

// 4. Legitimate batch of five tool results (~40 KB) must not warn.
reset();
const legalPrompt =
  "<tool_result>\n" +
  "x".repeat(40000) +
  "\n</tool_result>\n" +
  "<tool_result>\n" +
  "y".repeat(4000) +
  "\n</tool_result>";
checkResumeDeltaHealth(legalPrompt, true, "session-legal-batch");
assert.strictEqual(warnings.length, 0, "expected no warnings for legal 40 KB batch");
console.log("PASS: legal 40 KB resume delta triggers no warns");

// 5. Non-resume prompts must not warn regardless of content.
reset();
checkResumeDeltaHealth(
  "You are Kimi Code CLI\n<previous_response>\n...\n</previous_response>\n" + "z".repeat(70000),
  false,
  "session-fresh"
);
assert.strictEqual(warnings.length, 0, "expected no warnings for non-resume prompt");
console.log("PASS: non-resume prompt triggers no warns");

logger.warn = origWarn;

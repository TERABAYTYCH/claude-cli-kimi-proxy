#!/usr/bin/env node
"use strict";

/**
 * D1 regression test: the proxy must pass the requested model name through
 * to the CLI --model flag instead of collapsing it to a generic alias.
 */

import assert from "assert";
import { extractModel } from "../../dist/adapter/openai-to-cli.js";

const cases = [
  // Current full names pass through verbatim.
  ["claude-opus-5", "claude-opus-5"],
  ["claude-opus-4-8", "claude-opus-4-8"],
  ["claude-sonnet-5", "claude-sonnet-5"],
  ["claude-haiku-4-5", "claude-haiku-4-5"],
  // Legacy names the CLI rejects as full names fall back to the family alias.
  ["claude-opus-4", "opus"],
  ["claude-opus-4-6", "opus"],
  ["claude-sonnet-4", "sonnet"],
  ["claude-sonnet-4-5", "sonnet"],
  ["claude-sonnet-4-6", "sonnet"],
  ["claude-haiku-4", "haiku"],
  // Provider prefixes stripped.
  ["claude-code-cli/claude-opus-5", "claude-opus-5"],
  ["claude-max/claude-sonnet-5", "claude-sonnet-5"],
  // Bare aliases.
  ["opus", "opus"],
  ["sonnet", "sonnet"],
  ["haiku", "haiku"],
  ["opus-max", "opus"],
  ["sonnet-max", "sonnet"],
  // Fallbacks.
  ["", "opus"],
  ["claude-nonexistent-xyz", "claude-nonexistent-xyz"],
];

for (const [input, expected] of cases) {
  const actual = extractModel(input);
  if (actual !== expected) {
    console.error(`FAIL: extractModel(${JSON.stringify(input)}) = ${JSON.stringify(actual)}, expected ${JSON.stringify(expected)}`);
    process.exit(1);
  }
  console.log(`PASS: extractModel(${JSON.stringify(input)}) = ${JSON.stringify(actual)}`);
}

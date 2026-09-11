#!/usr/bin/env node
"use strict";

/**
 * B4 regression test: OpenAI-compatible usage must include cache_read and
 * cache_creation in prompt_tokens and expose cached_tokens in details.
 */

import { buildOpenAIUsage } from "../../dist/adapter/cli-to-openai.js";

const usage = buildOpenAIUsage({
  input_tokens: 2,
  output_tokens: 150,
  cache_read_input_tokens: 40000,
  cache_creation_input_tokens: 1500,
});

let failed = false;
const expectedPromptTokens = 2 + 40000 + 1500;
const expectedTotalTokens = expectedPromptTokens + 150;

const checks = [
  ["prompt_tokens", usage.prompt_tokens === expectedPromptTokens, `${usage.prompt_tokens} (expected ${expectedPromptTokens})`],
  ["completion_tokens", usage.completion_tokens === 150, usage.completion_tokens],
  ["total_tokens", usage.total_tokens === expectedTotalTokens, `${usage.total_tokens} (expected ${expectedTotalTokens})`],
  ["cached_tokens", usage.prompt_tokens_details?.cached_tokens === 40000, usage.prompt_tokens_details?.cached_tokens],
];

for (const [label, ok, detail] of checks) {
  console.log(`${ok ? "PASS" : "FAIL"}: ${label} = ${detail}`);
  if (!ok) failed = true;
}

if (failed) process.exit(1);

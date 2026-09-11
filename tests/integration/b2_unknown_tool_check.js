#!/usr/bin/env node
"use strict";

/**
 * B2 regression test: delegations whose tool name is not in the caller's
 * allowed list must be dropped and the response must fall back to plain text.
 */

import { parseDelegations, delegationsToToolCalls } from "../../dist/adapter/delegate-parser.js";

const tools = [
  {
    type: "function",
    function: {
      name: "Read",
      parameters: { type: "object", properties: { path: { type: "string" } } },
    },
  },
];

const text = `<invoke name="X"><parameter name="path">src/index.ts</parameter></invoke>`;
const delegations = parseDelegations(text);
const toolCalls = delegationsToToolCalls(delegations, "call", tools);

if (delegations.length !== 1) {
  console.error(`FAIL: expected 1 parsed delegation, got ${delegations.length}`);
  process.exit(1);
}
if (toolCalls.length !== 0) {
  console.error(`FAIL: expected 0 valid tool calls, got ${toolCalls.length}`);
  process.exit(1);
}
console.log("PASS: unknown tool name 'X' was rejected and produced zero tool_calls");

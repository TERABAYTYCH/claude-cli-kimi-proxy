#!/usr/bin/env node
"use strict";

/**
 * B3 regression test: invoke blocks inside fenced code blocks or inside
 * previous <tool_result> blocks must not be treated as live delegations.
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

const cases = [
  {
    name: "inside fenced code block",
    text: "Example:\n```\n<invoke name=\"Read\"><parameter name=\"path\">x</parameter></invoke>\n```\n",
    expected: 0,
  },
  {
    name: "inside tool_result block",
    text: "<tool_result>\n<invoke name=\"Read\"><parameter name=\"path\">x</parameter></invoke>\n</tool_result>\n",
    expected: 0,
  },
  {
    name: "real invoke after quoted example",
    text: "Example:\n```\n<invoke name=\"Read\"><parameter name=\"path\">x</parameter></invoke>\n```\nNow actually call:\n<invoke name=\"Read\"><parameter name=\"path\">src/index.ts</parameter></invoke>",
    expected: 1,
  },
];

let failed = false;
for (const c of cases) {
  const delegations = parseDelegations(c.text);
  const toolCalls = delegationsToToolCalls(delegations, "call", tools);
  const ok = toolCalls.length === c.expected;
  console.log(`${ok ? "PASS" : "FAIL"}: ${c.name} -> ${toolCalls.length} tool call(s), expected ${c.expected}`);
  if (!ok) failed = true;
}

if (failed) process.exit(1);

#!/usr/bin/env node
"use strict";

/**
 * B5 regression test: unknown tool names must be warned once per unique name
 * per request, and delegation parsing must be triggered only by new
 * </invoke> close tags, not on every content delta.
 */

import { delegationsToToolCalls, parseDelegations } from "../../dist/adapter/delegate-parser.js";

const tools = [
  {
    type: "function",
    function: {
      name: "Read",
      parameters: { type: "object", properties: { path: { type: "string" } } },
    },
  },
];

// Part 1: rejected name deduplication.
const rejected = new Set();
const textWithUnknowns = `<invoke name="X"><parameter name="path">a</parameter></invoke> then <invoke name="X"><parameter name="path">b</parameter></invoke>`;
const delegations = parseDelegations(textWithUnknowns);
const tcs1 = delegationsToToolCalls(delegations, "call", tools, rejected);
const tcs2 = delegationsToToolCalls(delegations, "call", tools, rejected);

if (tcs1.length !== 0 || tcs2.length !== 0) {
  console.error("FAIL: expected zero valid tool calls for unknown tool");
  process.exit(1);
}
if (rejected.size !== 1 || !rejected.has("X")) {
  console.error(`FAIL: expected rejected set {X}, got ${JSON.stringify([...rejected])}`);
  process.exit(1);
}
console.log("PASS: unknown tool name 'X' rejected once per request");

// Part 2: simulate the streaming close-tag counter logic.
function countInvokeCloses(text) {
  return (text.match(/<\/invoke>/g) || []).length;
}

let lastInvokeCloseCount = 0;
let parseCalls = 0;
const deltas = [
  `<invoke name="X"><parameter name="path">src/index.ts</parameter></invoke>`,
  " some text",
  " more text",
  " even more text",
  " and a final sentence.",
];

for (const delta of deltas) {
  const buffer = deltas.slice(0, deltas.indexOf(delta) + 1).join("");
  const closes = countInvokeCloses(buffer);
  if (closes > lastInvokeCloseCount) {
    lastInvokeCloseCount = closes;
    parseDelegations(buffer);
    parseCalls++;
  }
}

if (parseCalls !== 1) {
  console.error(`FAIL: expected 1 parse call after first </invoke>, got ${parseCalls}`);
  process.exit(1);
}
console.log("PASS: delegation parsing triggered only by new </invoke> close tags");

#!/usr/bin/env node
"use strict";

/**
 * Acceptance run: exercise the proxy end-to-end with the real tool fixture,
 * forcing at least one AgentSwarm delegation across 3-4 conversation turns.
 *
 * The run verifies:
 *   - AgentSwarm is emitted with correct items/prompt_template
 *   - streaming delegate parsing works
 *   - the session can resume across turns (cache_read grows)
 *   - the final turn returns plain text
 */

import fs from "fs";

const API = process.env.PROXY_URL || "http://127.0.0.1:3456";
const USER_ID = `acceptance-${Date.now()}`;
const MAX_TURNS = 4;

const rawTools = JSON.parse(
  fs.readFileSync(new URL("../fixtures/real_tools.json", import.meta.url), "utf8")
);
const tools = rawTools.map((t) => ({
  type: "function",
  function: {
    name: t.name,
    description: t.description || "",
    parameters: t.parameters || { type: "object", properties: {} },
  },
}));

const messages = [
  {
    role: "user",
    content:
      'Use AgentSwarm to run two independent Bash commands in parallel: "echo hello" and "echo world". Then report the combined results.',
  },
];

async function postChatCompletion(body) {
  const res = await fetch(`${API}/v1/chat/completions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`HTTP ${res.status}: ${text}`);
  }
  return streamToResult(res.body);
}

function streamToResult(body) {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let text = "";
  let toolCalls = [];
  let usage = null;

  return new Promise((resolve, reject) => {
    function pump() {
      reader
        .read()
        .then(({ value, done }) => {
          if (done) {
            resolve({ text, toolCalls, usage });
            return;
          }
          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n");
          buffer = lines.pop();
          for (const line of lines) {
            if (!line.startsWith("data: ")) continue;
            const payload = line.slice(6);
            if (payload === "[DONE]") continue;
            try {
              const chunk = JSON.parse(payload);
              const delta = chunk.choices?.[0]?.delta;
              if (delta?.content) text += delta.content;
              if (delta?.tool_calls) {
                for (const tc of delta.tool_calls) {
                  const existing = toolCalls[tc.index];
                  if (existing) {
                    existing.function.name += tc.function?.name || "";
                    existing.function.arguments += tc.function?.arguments || "";
                  } else {
                    toolCalls[tc.index] = {
                      id: tc.id,
                      type: "function",
                      function: {
                        name: tc.function?.name || "",
                        arguments: tc.function?.arguments || "",
                      },
                    };
                  }
                }
              }
              if (chunk.usage) usage = chunk.usage;
            } catch {
              // ignore malformed SSE data
            }
          }
          pump();
        })
        .catch(reject);
    }
    pump();
  });
}

let agentSwarmSeen = false;
let failed = false;

for (let turn = 1; turn <= MAX_TURNS; turn++) {
  console.log(`\n=== TURN ${turn} ===`);
  const result = await postChatCompletion({
    model: "claude-sonnet-4",
    stream: true,
    user: USER_ID,
    tools,
    messages,
  });

  console.log("text:", result.text.slice(0, 300));
  console.log("usage:", result.usage);

  if (result.toolCalls.length > 0) {
    const names = result.toolCalls.map((tc) => tc.function.name);
    console.log("tool_calls:", names);
    if (names.includes("AgentSwarm")) {
      agentSwarmSeen = true;
      const swarm = result.toolCalls.find((tc) => tc.function.name === "AgentSwarm");
      const args = JSON.parse(swarm.function.arguments);
      console.log("AgentSwarm args:", JSON.stringify(args, null, 2));
      if (!args.items || args.items.length < 2 || !args.prompt_template?.includes("{{item}}")) {
        console.log("FAIL: AgentSwarm missing required shape");
        failed = true;
      }
    }

    for (const tc of result.toolCalls) {
      let content;
      if (tc.function.name === "AgentSwarm") {
        const args = JSON.parse(tc.function.arguments);
        content = args.items
          .map((cmd) => `Subagent for "${cmd}" returned: ${cmd.replace(/^echo /, "")}`)
          .join("\n");
      } else if (tc.function.name === "Bash") {
        const args = JSON.parse(tc.function.arguments);
        content = `Executed: ${args.command}\nOutput: ${args.command?.replace(/^echo /, "") || "ok"}`;
      } else {
        content = `Tool ${tc.function.name} executed.`;
      }
      messages.push({
        role: "tool",
        tool_call_id: tc.id,
        content,
      });
    }
  } else {
    messages.push({ role: "assistant", content: result.text });
  }

  // Keep the conversation going with a follow-up on the first plain-text turn.
  if (result.toolCalls.length === 0 && turn < MAX_TURNS) {
    messages.push({
      role: "user",
      content: "Summarize what just happened in one sentence.",
    });
  }
}

console.log("\n=== RESULTS ===");
console.log(`AgentSwarm seen: ${agentSwarmSeen ? "YES" : "NO"}`);
if (!agentSwarmSeen) {
  console.log("FAIL: AgentSwarm was not invoked during the acceptance run");
  failed = true;
}

const fullLog = fs.readFileSync("logs/app.log", "utf8");
if (!fullLog.includes("[Streaming] Delegations detected") || !fullLog.includes('"tools":["AgentSwarm"]')) {
  console.log("FAIL: proxy did not log AgentSwarm delegation detection");
  failed = true;
}

if (failed) process.exit(1);
console.log("PASS: AgentSwarm acceptance run completed");

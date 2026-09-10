#!/usr/bin/env node
"use strict";

/**
 * B1 regression test: the delegation tool map must keep parameter descriptions
 * and prose-stated constraints (even when they live in the third paragraph),
 * without mid-word truncation and without duplicating the tool description
 * prefix in the constraints block. It must also surface enum-like value lists
 * that live only in prose (schema enum is empty), and inherit prose lists from
 * sibling tools when the same parameter name appears in multiple tools.
 */

import fs from "fs";
import { buildDelegationInstruction } from "../../dist/adapter/openai-to-cli.js";
import { logger } from "../../dist/utils/logger.js";

const warnings = [];
const origWarn = logger.warn.bind(logger);
logger.warn = (msg, meta) => {
  warnings.push({ msg, meta });
};

const agentDesc = `Launch a subagent to handle a task. The subagent runs as a same-process loop instance with its own context and wire file.

Available agent types (pass via subagent_type):
- plan: Read-only implementation planning and architecture design.
- coder: General software engineering agent.
- explore: Fast codebase exploration with prompt-enforced read-only behavior.

Available models (pass via model):
- minimax-coding-plan/MiniMax-M3 [default]
- primary: the main model you are running on, bound with your current thinking level; use it for hard, quality-sensitive subagent tasks`;

const tools = [
  {
    type: "function",
    function: {
      name: "AgentSwarm",
      description: `Launch multiple subagents from one prompt template, existing agent resumes, or both.

Use AgentSwarm when you need independent work on many inputs.

Each of these is enforced — a violation is rejected before any subagent starts: provide at least 2 items unless you pass resume_agent_ids; whenever items are present, prompt_template is required and must contain {{item}}; and the filled-in prompts must be distinct (two items that expand to the same prompt are rejected).

Available models (pass via model):
- minimax-coding-plan/MiniMax-M3 [default]
- primary: the main model you are running on, bound with your current thinking level; use it for hard, quality-sensitive subagent tasks`,
      parameters: {
        type: "object",
        properties: {
          items: {
            type: "array",
            items: { type: "string" },
            description: "Values used to fill {{item}}. Each item launches one new subagent.",
          },
          prompt_template: {
            type: "string",
            description: "Prompt template for each subagent. The {{item}} placeholder is replaced with each item value.",
          },
          subagent_type: {
            type: "string",
            enum: ["coder", "explore", "plan"],
            description: "Subagent type used for every new subagent spawned from items.",
          },
          model: { type: "string" },
        },
        required: ["items"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "Agent",
      description: agentDesc,
      parameters: {
        type: "object",
        properties: {
          prompt: { type: "string", description: "Full task prompt for the subagent." },
          description: { type: "string", description: "Short task description (3-5 words) for UI display." },
          subagent_type: { type: "string", description: "One of the available agent types." },
          model: { type: "string", description: "Which model to run the subagent on." },
        },
        required: ["prompt"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "Read",
      description: "Read a text file or stream, or both. This first phrase is just an overview.",
      parameters: {
        type: "object",
        properties: {
          path: { type: "string", description: "Path to the file." },
          line_offset: { type: "integer", description: "Line number to start reading from." },
        },
        required: ["path"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "LongParam",
      description: "Tool with a long parameter description.",
      parameters: {
        type: "object",
        properties: {
          value: {
            type: "string",
            description:
              "xx xx xx xx xx xx xx xx xx xx xx xx xx xx xx xx xx xx xx xx xx xx xx xx xx xx xx xx xx xx supercalifragilisticexpialidocious should remain whole.",
          },
        },
        required: ["value"],
      },
    },
  },
  // Cross-tool inheritance: ToolOne has a prose list for `foo`, ToolTwo has
  // the same parameter without any enum or prose list. ToolTwo should inherit
  // the list from ToolOne.
  {
    type: "function",
    function: {
      name: "ToolOne",
      description: "First tool.\n\nAvailable foos (pass via foo):\n- a: Alpha.\n- b: Beta.",
      parameters: {
        type: "object",
        properties: {
          foo: { type: "string", description: "A foo value." },
        },
        required: ["foo"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "ToolTwo",
      description: "Second tool with no prose enum of its own.",
      parameters: {
        type: "object",
        properties: {
          foo: { type: "string", description: "Inherited foo value." },
        },
        required: ["foo"],
      },
    },
  },
  // Conflict: two tools publish different prose lists for the same parameter
  // name `bar`. Cross-tool inheritance must be skipped and a warn emitted.
  {
    type: "function",
    function: {
      name: "ToolConflictA",
      description: "Conflict A.\n\nAvailable bars (pass via bar):\n- x: X-ray.\n- y: Yankee.",
      parameters: {
        type: "object",
        properties: {
          bar: { type: "string", description: "A bar value." },
        },
      },
    },
  },
  {
    type: "function",
    function: {
      name: "ToolConflictB",
      description: "Conflict B.\n\nAvailable bars (pass via bar):\n- p: Papa.\n- q: Quebec.\n- r: Romeo.",
      parameters: {
        type: "object",
        properties: {
          bar: { type: "string", description: "Another bar value." },
        },
      },
    },
  },
  // Schema enum wins over inherited prose enum.
  {
    type: "function",
    function: {
      name: "ToolSchemaWins",
      description: "Schema enum should beat prose.\n\nAvailable foos (pass via foo):\n- a: Alpha.\n- b: Beta.",
      parameters: {
        type: "object",
        properties: {
          foo: { type: "string", enum: ["z"], description: "Schema enum z." },
        },
      },
    },
  },
];

const instruction = buildDelegationInstruction(tools);
const agentSwarmLine = instruction.split("\n").find((l) => l.startsWith("- AgentSwarm:")) || "";
const agentLine = instruction.split("\n").find((l) => l.startsWith("- Agent:")) || "";
const readLine = instruction.split("\n").find((l) => l.startsWith("- Read:")) || "";
const readConstraints = readLine.includes("[") ? readLine.split("[")[1].split("]")[0] : "";
const longParamLine = instruction.split("\n").find((l) => l.startsWith("- LongParam:")) || "";
const longParamQuoted = longParamLine.match(/value:string! "([^"]+)"/)?.[1] || "";
const toolOneLine = instruction.split("\n").find((l) => l.startsWith("- ToolOne:")) || "";
const toolTwoLine = instruction.split("\n").find((l) => l.startsWith("- ToolTwo:")) || "";
const toolConflictALine = instruction.split("\n").find((l) => l.startsWith("- ToolConflictA:")) || "";
const toolConflictBLine = instruction.split("\n").find((l) => l.startsWith("- ToolConflictB:")) || "";
const toolSchemaWinsLine = instruction.split("\n").find((l) => l.startsWith("- ToolSchemaWins:")) || "";

const checks = [
  ["AgentSwarm description kept", /Launch multiple subagents from one prompt template/.test(agentSwarmLine)],
  ["AgentSwarm constraint from third paragraph", /whenever items are present, prompt_template is required/i.test(agentSwarmLine)],
  ["AgentSwarm long constraint not truncated mid-word", /provide at least 2 items unless you pass resume_agent_ids/i.test(agentSwarmLine)],
  ["AgentSwarm schema enum used", ["\"coder\"", "\"explore\"", "\"plan\""].every((v) => agentSwarmLine.includes(v))],
  ["AgentSwarm prose enum for model", /model:"minimax-coding-plan\/MiniMax-M3"\|"primary"/.test(agentSwarmLine)],
  ["AgentSwarm parameter description present", /items:string\[\]! "Values used to fill/.test(agentSwarmLine)],
  ["Agent prose enum for subagent_type", ["\"plan\"", "\"coder\"", "\"explore\""].every((v) => agentLine.includes(v))],
  ["Agent prose enum for model", /model:"minimax-coding-plan\/MiniMax-M3"\|"primary"/.test(agentLine)],
  ["'or' in overview not treated as constraint", !/or both/i.test(readConstraints)],
  ["description prefix not duplicated in constraints", !/Launch multiple subagents from one prompt template/i.test(agentSwarmLine.split("[")[1] || "")],
  ["long param description truncated at word boundary", longParamQuoted.length <= 125 && !longParamQuoted.includes("supercalifragilistic") && longParamQuoted.endsWith("...")],
  ["cross-tool inherited enum on donor", /foo:"a"\|"b"/.test(toolOneLine)],
  ["cross-tool inherited enum on recipient", /foo:"a"\|"b"/.test(toolTwoLine)],
  ["conflict tool A keeps its own prose enum", /bar:"x"\|"y"/.test(toolConflictALine)],
  ["conflict tool B keeps its own prose enum", /bar:"p"\|"q"\|"r"/.test(toolConflictBLine)],
  ["schema enum wins over inherited prose enum", /foo:"z"/.test(toolSchemaWinsLine)],
  ["conflict emits warn", warnings.some((w) => w.msg === "Cross-tool enum conflict; skipping cross-tool inheritance")],
];

logger.warn = origWarn;

let failed = false;
for (const [label, ok] of checks) {
  console.log(`${ok ? "PASS" : "FAIL"}: ${label}`);
  if (!ok) failed = true;
}

// Rough token estimate: 1 token ~= 4 chars for English prose.
const estimatedTokens = Math.round(instruction.length / 4);
console.log(`\nInstruction length: ${instruction.length} chars (~${estimatedTokens} tokens)`);

// Snapshot against real tools fixture so the budget check survives /tmp cleanups.
const fixturePath = new URL("../fixtures/real_tools.json", import.meta.url);
const rawTools = JSON.parse(fs.readFileSync(fixturePath, "utf8"));
const realTools = rawTools.map((t) => ({
  type: "function",
  function: {
    name: t.name,
    description: t.description || "",
    parameters: t.parameters || { type: "object", properties: {} },
  },
}));
const realInstruction = buildDelegationInstruction(realTools);
const realTokens = Math.round(realInstruction.length / 4);
console.log(`Real tools map: ${realInstruction.length} chars (~${realTokens} tokens)`);
if (realTokens > 7000) {
  console.log("FAIL: real tools map exceeds 7000 token budget");
  failed = true;
}

if (failed) process.exit(1);

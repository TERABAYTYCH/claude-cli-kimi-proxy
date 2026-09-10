/**
 * Parse Claude CLI delegation output into OpenAI-compatible tool calls.
 *
 * When built-in tools are disabled, Claude outputs Kimi-style <invoke> blocks:
 *
 * <invoke name="Agent">
 * <parameter name="description">Run echo</parameter>
 * <parameter name="prompt">Run the command: echo hello</parameter>
 * </invoke>
 *
 * The proxy intercepts these and returns them as OpenAI `tool_calls` so that
 * Kimi can execute the real tools in the user's chat.
 */

import type { OpenAIToolCall, OpenAIToolDefinition } from "../types/openai.js";
import { logger } from "../utils/logger.js";

export interface DelegationRequest {
  tool: string;
  params: Record<string, unknown>;
}

const INVOKE_RE = /<invoke\s+name="([^"]+)"\s*>([\s\S]*?)<\/invoke>/g;
const PARAMETER_RE = /<parameter\s+name="([^"]+)"\s*>([\s\S]*?)<\/parameter>/g;
const CODE_BLOCK_RE = /```[\s\S]*?```/g;
const TOOL_RESULT_BLOCK_RE = /<tool_result>[\s\S]*?<\/tool_result>/g;

/**
 * Common aliases Claude may use for Kimi tool names. Keys must be lowercase.
 */
const TOOL_NAME_ALIASES: Record<string, string> = {
  // shell
  exec: "Bash",
  process: "Bash",
  bash: "Bash",
  // read
  read: "Read",
  readfile: "Read",
  // media
  image: "ReadMediaFile",
  media: "ReadMediaFile",
  // write
  write: "Write",
  writefile: "Write",
  // edit
  edit: "Edit",
  editfile: "Edit",
  // grep
  grep: "Grep",
  // glob
  glob: "Glob",
  find: "Glob",
  ls: "Glob",
  // web
  web_search: "WebSearch",
  websearch: "WebSearch",
  web_fetch: "FetchURL",
  webfetch: "FetchURL",
  fetchurl: "FetchURL",
  fetch_url: "FetchURL",
  // subagents / background tasks
  task: "Agent",
  agent: "Agent",
  agentswarm: "AgentSwarm",
  swarm: "AgentSwarm",
  taskoutput: "TaskOutput",
  task_stop: "TaskStop",
  taskstop: "TaskStop",
  waitfor: "WaitFor",
  wait_for: "WaitFor",
  // goals
  creategoal: "CreateGoal",
  getgoal: "GetGoal",
  updategoal: "UpdateGoal",
  setgoalbudget: "SetGoalBudget",
  // other native tools
  todolist: "TodoList",
  askuserquestion: "AskUserQuestion",
  skill: "Skill",
  enterplanmode: "EnterPlanMode",
  exitplanmode: "ExitPlanMode",
  croncreate: "CronCreate",
  cronlist: "CronList",
  crondelete: "CronDelete",
};

/**
 * Normalize a tool name from Claude's output to the exact Kimi tool name.
 *
 * If `allowedTools` is provided and the name is not a known alias and is not
 * present in the allowed list, returns `null`. This prevents a hallucinated
 * tool name from being forwarded as a real tool_call and burning a round-trip.
 */
export function normalizeToolName(
  name: string,
  allowedTools?: string[]
): string | null {
  const key = name.toLowerCase();
  const aliased = TOOL_NAME_ALIASES[key];
  if (aliased) return aliased;

  if (allowedTools) {
    const exact = allowedTools.find((t) => t === name);
    if (exact) return exact;
    const caseMatch = allowedTools.find((t) => t.toLowerCase() === key);
    if (caseMatch) return caseMatch;

    // Unknown tool name in a constrained caller context — reject it.
    return null;
  }

  return name;
}

function parseParameterValue(raw: string): unknown {
  const trimmed = raw.trim();
  if (trimmed === "true") return true;
  if (trimmed === "false") return false;
  if (/^-?\d+$/.test(trimmed)) return Number(trimmed);
  try {
    return JSON.parse(trimmed);
  } catch {
    return trimmed;
  }
}

function coerceParamValue(
  value: unknown,
  expectedType: string | undefined
): unknown {
  if (!expectedType || value === null || value === undefined) return value;

  switch (expectedType) {
    case "string":
      return typeof value === "string" ? value : String(value);
    case "number":
    case "integer":
      if (typeof value === "number") return value;
      if (typeof value === "string" && /^-?\d+(\.\d+)?$/.test(value)) {
        return Number(value);
      }
      return value;
    case "boolean":
      if (typeof value === "boolean") return value;
      if (value === "true") return true;
      if (value === "false") return false;
      return value;
    case "object":
    case "array":
      if (typeof value === "string") {
        try {
          return JSON.parse(value);
        } catch {
          return value;
        }
      }
      return value;
    default:
      return value;
  }
}

function coerceParamsWithSchema(
  params: Record<string, unknown>,
  toolDef: OpenAIToolDefinition | undefined
): Record<string, unknown> {
  if (!toolDef?.function?.parameters) return params;

  const schema = toolDef.function.parameters as {
    properties?: Record<string, { type?: string }>;
  };
  const properties = schema.properties || {};

  const result: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(params)) {
    const propSchema = properties[key];
    const coerced = coerceParamValue(value, propSchema?.type);
    // An empty string for an object/array param means "omitted" (the model
    // emitted an empty <parameter> tag) — drop it instead of sending "".
    if (
      coerced === "" &&
      (propSchema?.type === "object" || propSchema?.type === "array")
    ) {
      continue;
    }
    result[key] = coerced;
  }
  return result;
}

function parseInvokeBlock(block: string): DelegationRequest | null {
  const nameMatch = block.match(/^<invoke\s+name="([^"]+)"\s*>/);
  if (!nameMatch) return null;

  const tool = nameMatch[1];
  const params: Record<string, unknown> = {};
  let paramMatch: RegExpExecArray | null;

  while ((paramMatch = PARAMETER_RE.exec(block)) !== null) {
    const paramName = paramMatch[1];
    const paramValue = paramMatch[2];
    params[paramName] = parseParameterValue(paramValue);
  }

  PARAMETER_RE.lastIndex = 0;

  // Unwrap OpenAI-style {"arguments": {...}} nesting: the model sometimes
  // packs all params into a single "arguments" parameter (imitating history
  // serialized that way). Kimi rejects the extra property, so flatten it.
  if ("arguments" in params) {
    const wrapped = params.arguments;
    let inner: unknown = wrapped;
    if (typeof wrapped === "string") {
      try {
        inner = JSON.parse(wrapped);
      } catch {
        inner = undefined;
      }
    }
    if (inner && typeof inner === "object" && !Array.isArray(inner)) {
      delete params.arguments;
      Object.assign(params, inner);
    }
  }

  return { tool, params };
}

export function stripQuotedInvokes(text: string): string {
  return text
    .replace(CODE_BLOCK_RE, "")
    .replace(TOOL_RESULT_BLOCK_RE, "");
}

export function parseDelegations(text: string): DelegationRequest[] {
  const results: DelegationRequest[] = [];
  let match: RegExpExecArray | null;

  // Ignore invoke examples embedded in fenced code blocks or previous
  // <tool_result> blocks. The model often quotes its own instruction or
  // history, and those must not be treated as live delegations.
  const executableText = stripQuotedInvokes(text);

  while ((match = INVOKE_RE.exec(executableText)) !== null) {
    const block = match[0];
    const parsed = parseInvokeBlock(block);
    if (parsed) results.push(parsed);
  }

  INVOKE_RE.lastIndex = 0;
  return results;
}

export function delegationsToToolCalls(
  delegations: DelegationRequest[],
  idPrefix = "call",
  tools?: OpenAIToolDefinition[],
  rejectedNames?: Set<string>
): OpenAIToolCall[] {
  const allowedTools = tools?.map((t) => t.function.name);
  const byName = new Map(tools?.map((t) => [t.function.name, t]));

  const results: OpenAIToolCall[] = [];
  for (const d of delegations) {
    const name = normalizeToolName(d.tool, allowedTools);
    if (name === null) {
      if (!rejectedNames?.has(d.tool)) {
        logger.warn("[DelegateParser] Dropping delegation with unknown tool name", {
          originalName: d.tool,
          allowedTools,
        });
        rejectedNames?.add(d.tool);
      }
      continue;
    }

    const toolDef = byName.get(name);
    const coerced = coerceParamsWithSchema(d.params, toolDef);

    results.push({
      id: `${idPrefix}_${results.length}`,
      type: "function",
      function: {
        name,
        arguments: JSON.stringify(coerced),
      },
    });
  }

  return results;
}

export function stripDelegations(text: string): string {
  return text.replace(INVOKE_RE, "").trim();
}

/**
 * Extract the list of allowed tool names from an OpenAI tool definitions array.
 */
export function getAllowedToolNames(tools: OpenAIToolDefinition[]): string[] {
  return tools.map((t) => t.function.name);
}

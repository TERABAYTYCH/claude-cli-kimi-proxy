/**
 * Converts OpenAI chat request format to Claude CLI input
 */

import type {
  OpenAIChatRequest,
  OpenAIContentBlock,
  OpenAIChatMessage,
  OpenAIToolDefinition,
} from "../types/openai.js";
import { logger } from "../utils/logger.js";

export type ClaudeModel = "opus" | "sonnet" | "haiku";

export interface CliInput {
  prompt: string;
  model: ClaudeModel;
  sessionId?: string;
  systemPrompt?: string;
}

const MODEL_MAP: Record<string, ClaudeModel> = {
  // Direct model names (provider prefixes like `claude-code-cli/` and `claude-max/`
  // are stripped by extractModel before consulting this map)
  "claude-opus-4": "opus",
  "claude-opus-4-6": "opus",
  "claude-sonnet-4": "sonnet",
  "claude-sonnet-4-5": "sonnet",
  "claude-sonnet-4-6": "sonnet",
  "claude-sonnet-5": "sonnet",
  "claude-opus-5": "opus",
  "claude-haiku-4": "haiku",
  "claude-haiku-4-5": "haiku",
  // Bare aliases
  "opus": "opus",
  "sonnet": "sonnet",
  "haiku": "haiku",
  "opus-max": "opus",
  "sonnet-max": "sonnet",
};

/**
 * Extract Claude model alias from request model string
 */
export function extractModel(model: string): ClaudeModel {
  // Try direct lookup
  if (MODEL_MAP[model]) {
    return MODEL_MAP[model];
  }

  // Try stripping provider prefix
  const stripped = model.replace(/^(?:claude-code-cli|claude-max)\//, "");
  if (MODEL_MAP[stripped]) {
    return MODEL_MAP[stripped];
  }

  // Default to opus (Claude Max subscription)
  return "opus";
}

/**
 * Extract text from a content field that may be a string or array of content blocks.
 * OpenAI API allows content as either:
 *   - A plain string: "Hello"
 *   - An array of content blocks: [{"type": "text", "text": "Hello"}]
 */
function extractText(content: string | OpenAIContentBlock[]): string {
  if (typeof content === "string") {
    return content;
  }
  if (Array.isArray(content)) {
    return content
      .filter((block) => block.type === "text" || block.type === "input_text")
      .map((block) => block.text)
      .join("\n");
  }
  return String(content || "");
}

/**
 * Convert OpenAI messages array to a single prompt string for Claude CLI
 *
 * Claude Code CLI in --print mode expects a single prompt, not a conversation.
 * We format the messages into a readable format that preserves context.
 *
 * Note: system messages are extracted separately via extractSystemPrompt() and
 * passed via --system-prompt-file, so this function should only receive
 * user/assistant messages.
 */
export function messagesToPrompt(
  messages: OpenAIChatRequest["messages"]
): string {
  const parts: string[] = [];

  for (const msg of messages) {
    const text = extractText(msg.content);
    // Kimi echoes the model's prior thinking back as reasoning_content. Carry
    // a truncated version into the prompt — without it the model re-derives
    // its plan every turn. Capped hard: full thinking on every historical
    // turn bloats the replayed history (each turn is re-sent on cache miss).
    const MAX_REASONING_CHARS = 400;
    const reasoning =
      msg.role === "assistant" && typeof msg.reasoning_content === "string"
        ? msg.reasoning_content.slice(0, MAX_REASONING_CHARS)
        : "";
    const fullText = [reasoning, text].filter(Boolean).join("\n\n");
    switch (msg.role) {
      case "user":
        // User messages are the main prompt
        parts.push(text);
        break;

      case "assistant": {
        // Previous assistant responses for context. If the assistant emitted
        // tool_calls (delegate mode), include them so the model sees which
        // tool was already invoked and doesn't repeat it.
        if (msg.tool_calls && msg.tool_calls.length > 0) {
          const invokes = msg.tool_calls
            .map((tc) => {
              // Expand the JSON arguments into individual <parameter> elements
              // matching the delegation instruction format. If we instead show
              // a single <parameter name="arguments"> blob, the model imitates
              // it and Kimi rejects the call ("must NOT have additional
              // property 'arguments'").
              let paramsXml: string;
              try {
                const args = JSON.parse(tc.function.arguments) as Record<string, unknown>;
                paramsXml = Object.entries(args)
                  .map(([key, value]) => {
                    const rendered =
                      typeof value === "string" ? value : JSON.stringify(value);
                    return `<parameter name="${key}">${rendered}</parameter>`;
                  })
                  .join("");
              } catch {
                paramsXml = `<parameter name="arguments">${tc.function.arguments}</parameter>`;
              }
              return `<invoke name="${tc.function.name}">${paramsXml}</invoke>`;
            })
            .join("\n");
          // Keep the assistant's text (plan, conclusions) alongside the invoke.
          // Dropping it makes the model re-derive its plan from scratch every
          // turn, burning tokens on repeated reasoning.
          parts.push(`<previous_response>\n${fullText ? fullText + "\n" : ""}${invokes}\n</previous_response>\n`);
        } else {
          parts.push(`<previous_response>\n${fullText || text}\n</previous_response>\n`);
        }
        break;
      }

      case "tool":
        // Results returned by the proxy daemon after executing delegated tools
        parts.push(`<tool_result>\n${text}\n</tool_result>\n`);
        break;

      case "system":
        // Should not happen: system messages are extracted separately
        break;
    }
  }

  return parts.join("\n").trim();
}

/**
 * Build the delegation instruction tailored to the tools the caller exposed.
 *
 * The instruction tells Claude it has no direct tool access and must emit
 * structured `<delegate>` blocks. It also lists the exact Kimi tool names
 * and a mapping from common aliases Claude may use.
 */
type JsonSchemaProperty = {
  type?: string | string[];
  description?: string;
  enum?: unknown[];
  items?: JsonSchemaProperty;
  properties?: Record<string, JsonSchemaProperty>;
  required?: string[];
};

const CONSTRAINT_MARKERS = [
  "required",
  "must",
  "enforced",
  "rejected",
  "at least",
  "only",
  "cannot",
  "do not",
  "whenever",
  "if you provide",
];

const CONSTRAINT_MAX_LEN = 300;
const MAX_CONSTRAINTS_PER_TOOL = 3;
const DESCRIPTION_MAX_LEN = 400;
const PARAM_DESC_MAX_LEN = 120;

function cleanupConstraint(sentence: string): string {
  // Drop boilerplate prefixes that appear in several Kimi tool descriptions.
  return sentence
    .replace(/^Each of these is enforced\s*[-–—]\s*a violation is rejected before any subagent starts:\s*/i, "")
    .trim();
}

function truncateAtWord(text: string, maxLen: number): string {
  if (text.length <= maxLen) return text;
  const truncated = text.slice(0, maxLen);
  const lastSpace = truncated.lastIndexOf(" ");
  return (lastSpace > 0 ? truncated.slice(0, lastSpace) : truncated) + "...";
}

function hasConstraintMarker(sentence: string): boolean {
  const lower = sentence.toLowerCase();
  // "either ... or" is the only useful form of "either".
  if (/\beither\b/.test(lower) && /\bor\b/.test(lower)) return true;
  return CONSTRAINT_MARKERS.some((m) => {
    const re = new RegExp("\\b" + m.replace(/\s+/g, "\\s+") + "\\b", "i");
    return re.test(lower);
  });
}

function extractConstraintSentences(description: string, prefix: string): string {
  // Split into sentences, keeping periods and semicolons as delimiters.
  const sentences = description
    .replace(/;/g, ".")
    .split(/\.(?:\s+|$)/)
    .map((s) => s.trim())
    .filter(Boolean);

  const constraints = sentences
    .filter((s) => hasConstraintMarker(s))
    // Skip sentences already shown in the tool-description prefix.
    .filter((s) => !prefix.toLowerCase().includes(s.toLowerCase()))
    .map((s) => cleanupConstraint(s))
    .map((s) => truncateAtWord(s, CONSTRAINT_MAX_LEN).trim())
    .filter(Boolean)
    // Keep the first N constraints in their original order. The author tends to
    // state the most important rules first, and preserving order keeps the
    // prefix byte-stable.
    .slice(0, MAX_CONSTRAINTS_PER_TOOL);

  return constraints.join("; ");
}

function firstLine(text: string, maxLen: number): string {
  return truncateAtWord(text.split("\n")[0].trim(), maxLen);
}

/**
 * Remove "Available X:" bullet lists from a description before extracting
 * constraint sentences. Otherwise bullets that happen to contain markers such
 * as "only" are logged as constraints, duplicating the enum list we already
 * surface separately.
 */
function stripAvailableLists(text: string): string {
  return text.replace(
    /Available\b[^:\n]*?:\s*\n(?:\s*[-*]\s*[^\n]*\n?)+/g,
    ""
  );
}

/**
 * Pull enum-like value lists that are described in prose rather than JSON Schema.
 * Many Kimi tools state allowed values in bullet lists (e.g. "Available agent types
 * (pass via subagent_type): - plan: ... - coder: ..."). When the schema has no
 * `enum`, this is the only way the model can learn valid values.
 */
function extractEnumValuesFromDescription(
  paramName: string,
  toolDescription: string
): string[] | undefined {
  const desc = toolDescription;

  // "Available <X> (pass via <param>):" followed by a bullet list. Bullet
  // items may have indented continuation lines (e.g. "  Tools: ..."), so we
  // parse line-by-line after the header rather than relying on a single regex
  // group that stops at the first continuation line.
  const availableHeaderRe = new RegExp(
    "Available\\b[^:.\\n]*?\\(pass via\\s+`?" + paramName + "`?\\)\\s*[:：]\\s*\\n",
    "i"
  );
  const headerMatch = desc.match(availableHeaderRe);
  if (headerMatch) {
    const start = headerMatch.index! + headerMatch[0].length;
    const rest = desc.slice(start);
    const lines = rest.split("\n");
    const values: string[] = [];
    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed) break;
      const bullet = line.match(/^\s*[-*]\s+(`?[^`\s:]+`?)/);
      if (bullet) {
        values.push(bullet[1].replace(/^`|`$/g, ""));
        continue;
      }
      // Indented continuation line (e.g. "  Tools: ...") — keep scanning.
      if (/^\s{2,}\S/.test(line)) continue;
      // Anything else ends the list.
      break;
    }
    if (values.length) return values;
  }

  // Parameter-specific inline enumeration.
  const inlineRe = new RegExp(
    "`?" + paramName + "`?\\s+(?:must be|is|are)\\s+(?:one of|either)\\s+[:：]?\\s*([^\.\n]+)",
    "i"
  );
  const inlineMatch = desc.match(inlineRe);
  if (inlineMatch) {
    const parts = inlineMatch[1]
      .split(/[,|]/)
      .map((s) => s.trim().replace(/^[`"']|[`"']$/g, ""))
      .filter(Boolean);
    if (parts.length >= 2) return parts;
  }

  // Generic "Allowed/valid values: a, b, c" anywhere in the description.
  const genericRe =
    /(?:allowed|valid|possible|must be one of|one of)\s*(?:values?|types?|options?)?\s*[:：]\s*([^\.\n]+)/i;
  const genericMatch = desc.match(genericRe);
  if (genericMatch) {
    const parts = genericMatch[1]
      .split(/[,|]/)
      .map((s) => s.trim().replace(/^[`"']|[`"']$/g, ""))
      .filter(Boolean);
    if (parts.length >= 2) return parts;
  }

  return undefined;
}

/**
 * Build a cross-tool enum map from prose value lists.
 *
 * Some tools (e.g. AgentSwarm) do not list allowed values in their own
 * description, while a sibling tool (e.g. Agent) lists the same parameter's
 * values in prose. We collect those prose lists across the whole tool set so
 * a parameter without its own enum or prose list can inherit one from another
 * tool with the same parameter name.
 *
 * Priority when rendering a single tool remains:
 *   1. JSON Schema enum of the parameter itself
 *   2. Prose enum from the same tool's description
 *   3. Cross-tool prose enum for the same parameter name
 *
 * If two tools publish DIFFERENT lists for the same parameter name, we do not
 * merge them (that would invent values) and warn instead.
 */
function buildCrossToolEnumMap(
  tools: OpenAIToolDefinition[]
): Map<string, string[]> {
  const listsByParam = new Map<string, { toolName: string; values: string[] }[]>();

  for (const t of tools) {
    const fn = t.function;
    if (!fn) continue;
    const props = (fn.parameters as { properties?: Record<string, JsonSchemaProperty> } | undefined)?.properties;
    if (!props) continue;

    for (const [paramName, prop] of Object.entries(props)) {
      // Schema enum already has top priority; cross-tool inheritance is only
      // for prose-stated values.
      if (prop.enum && prop.enum.length > 0) continue;

      const values = extractEnumValuesFromDescription(paramName, fn.description || "");
      if (!values || values.length === 0) continue;

      const entry = listsByParam.get(paramName) || [];
      entry.push({ toolName: fn.name, values });
      listsByParam.set(paramName, entry);
    }
  }

  const result = new Map<string, string[]>();
  for (const [paramName, entries] of listsByParam) {
    const unique = new Map<string, string>();
    for (const { toolName, values } of entries) {
      unique.set(JSON.stringify(values), toolName);
    }

    if (unique.size === 1) {
      result.set(paramName, entries[0].values);
    } else {
      const variants = Array.from(unique.entries()).map(([serialized, toolName]) => ({
        toolName,
        values: JSON.parse(serialized) as string[],
      }));
      logger.warn("Cross-tool enum conflict; skipping cross-tool inheritance", {
        paramName,
        variants,
      });
    }
  }

  return result;
}

function formatSchemaType(
  schema: JsonSchemaProperty | undefined,
  toolDescription?: string,
  paramName?: string,
  crossEnums?: Map<string, string[]>
): string {
  if (!schema) return "any";

  // Pick a single representative type when the schema allows multiple.
  const baseType = Array.isArray(schema.type) ? schema.type[0] : schema.type;

  if (schema.enum && schema.enum.length > 0) {
    const values = schema.enum
      .map((v) => (typeof v === "string" ? `"${v}"` : String(v)))
      .join("|");
    return values;
  }

  // Some tools (e.g. Agent, AgentSwarm) list allowed values in prose rather
  // than JSON Schema enum. Surface them at the top level only.
  if (
    toolDescription &&
    paramName &&
    baseType === "string"
  ) {
    const proseEnums = extractEnumValuesFromDescription(paramName, toolDescription);
    if (proseEnums && proseEnums.length > 0) {
      return proseEnums.map((v) => `"${v}"`).join("|");
    }
  }

  // Cross-tool inheritance: if this parameter has no enum and no prose list,
  // borrow a prose list discovered under the same parameter name in another
  // tool of the request.
  if (
    paramName &&
    crossEnums &&
    baseType === "string"
  ) {
    const inherited = crossEnums.get(paramName);
    if (inherited && inherited.length > 0) {
      return inherited.map((v) => `"${v}"`).join("|");
    }
  }

  switch (baseType) {
    case "array": {
      const itemType = formatSchemaType(schema.items);
      return `${itemType}[]`;
    }
    case "object": {
      // Only surface top-level property names with types; full nested objects
      // make the instruction too long and are rarely needed for the model to
      // choose the right tool. The caller still validates against the real schema.
      const props = schema.properties;
      if (props && Object.keys(props).length > 0) {
        const fields = Object.entries(props)
          .sort(([a], [b]) => a.localeCompare(b))
          .map(([name, prop]) => `${name}:${formatSchemaType(prop)}`)
          .join(",");
        return `{${fields}}`;
      }
      return "object";
    }
    case "integer":
      return "number";
    case "string":
    case "number":
    case "boolean":
      return baseType;
    default:
      return "any";
  }
}

export function buildDelegationInstruction(
  tools: OpenAIToolDefinition[]
): string {
  // Compact but informative: full JSON schemas cost ~40k tokens of cache prefix.
  // We include: types, optionality, enums, array element types, short parameter
  // descriptions, and constraint sentences from the tool description. Many
  // rules (e.g. dependent requirements) are stated in prose, not JSON Schema,
  // and the model cannot guess them. This text sits in --system-prompt-file and
  // is cache-read at x0.1, so a richer map pays for itself immediately.
  const crossEnums = buildCrossToolEnumMap(tools);
  const toolDescriptions = [...tools]
    .sort((a, b) => a.function.name.localeCompare(b.function.name))
    .map((t) => {
      const fn = t.function;
      const fullDesc = fn.description || "";
      const desc = firstLine(fullDesc, DESCRIPTION_MAX_LEN);
      const constraints = extractConstraintSentences(stripAvailableLists(fullDesc), desc);
      const schema = (fn.parameters as { properties?: Record<string, JsonSchemaProperty>; required?: string[] } | undefined);
      const properties = schema?.properties;
      const required = new Set(schema?.required || []);
      const paramNames = properties
        ? Object.entries(properties)
            .sort(([a], [b]) => a.localeCompare(b))
            .map(([name, prop]) => {
              const type = formatSchemaType(prop, fullDesc, name, crossEnums);
              const opt = required.has(name) ? "!" : "?";
              const paramDesc =
                prop.description && PARAM_DESC_MAX_LEN > 0
                  ? ` "${firstLine(prop.description, PARAM_DESC_MAX_LEN)}"`
                  : "";
              return `${name}:${type}${opt}${paramDesc}`;
            })
            .join(", ")
        : "";
      const constraintsPart = constraints ? ` [constraints: ${constraints}]` : "";
      return `- ${fn.name}: ${desc}${constraintsPart}${paramNames ? ` (${paramNames})` : ""}`;
    })
    .join("\n");

  return `
You are running inside a proxy session. You have NO direct tool access. The --tools "" flag disables all built-in tools.

When a tool is needed, output EXACTLY ONE <invoke> block and STOP. Do not output anything after the closing </invoke> tag. The proxy will execute the tool and return the result in the next message.

CRITICAL RULES:
- Output ONLY the <invoke> block, nothing else
- Do NOT generate fake "Tool result:" text
- Do NOT explain or comment
- Do NOT use more than one tool per response
- Wait for the actual tool result from the proxy
- NEVER repeat an invoke whose <tool_result> already appears in the conversation history. Before every call, scan the history: each <previous_response> invoke must be followed by a new, DIFFERENT action — never the same tool with the same arguments again, even if a system-reminder suggests re-checking.
- If the user's request is a numbered/multi-step list, track progress by matching history invokes to steps: call the tool for the FIRST step that does not yet have a result. Do not go back to earlier steps.
- Text inside earlier <previous_response> blocks is your OWN prior reasoning and conclusions. Reuse it — never re-derive or re-plan what you already figured out in a previous step.
- Keep your own reasoning SHORT: the history already contains your full prior analysis, so restating the task, the plan, or already-known facts wastes tokens. One or two sentences about the immediate next action is enough.
- If the user says "just output/show the calls without executing" (or similar), IGNORE that instruction: this architecture only produces results by executing each invoke through the proxy. Always proceed one invoke per response.

Format:
<invoke name="<ToolName>">
<parameter name="<paramName>">value</parameter>
...
</invoke>

During the conversation the history contains two kinds of blocks: <previous_response> holds your own earlier messages (text and/or invokes you emitted), <tool_result> holds the execution result the proxy returned for your invoke.

Examples:
<invoke name="Bash"><parameter name="command">echo hello</parameter></invoke>
<invoke name="Read"><parameter name="path">src/index.ts</parameter></invoke>
<invoke name="Agent"><parameter name="description">Run echo</parameter><parameter name="prompt">Run the command: echo hello</parameter></invoke>

Available tools (use ONLY these exact names):
${toolDescriptions}

Only output plain text when no tool is needed.
`.trim();
}

/**
 * Build the final system prompt. If the caller passed OpenAI tools, append
 * the delegation instruction so that Claude outputs structured delegation
 * requests instead of trying to execute tools inside the CLI.
 */
function buildSystemPrompt(
  originalSystemPrompt: string | undefined,
  tools: OpenAIToolDefinition[] | undefined
): string | undefined {
  const hasTools = !!tools && tools.length > 0;
  if (!originalSystemPrompt && !hasTools) return undefined;

  const parts: string[] = [];
  if (originalSystemPrompt) {
    parts.push(originalSystemPrompt);
  }
  if (hasTools) {
    parts.push(buildDelegationInstruction(tools));
  }

  return parts.join("\n\n");
}

/**
 * Extract system prompt(s) from the OpenAI messages array.
 *
 * Claude Code CLI has a dedicated `--system-prompt-file` flag that fully
 * replaces the built-in system prompt. This lets us pass Kimi's system
 * identity directly without it being duplicated as regular text in stdin.
 */
export function extractSystemPrompt(messages: OpenAIChatMessage[]): {
  systemPrompt: string | undefined;
  remainingMessages: OpenAIChatMessage[];
} {
  const systemMessages = messages.filter((m) => m.role === "system");
  const remainingMessages = messages.filter((m) => m.role !== "system");

  const systemPrompt = systemMessages.length
    ? systemMessages.map((m) => extractText(m.content)).join("\n\n")
    : undefined;

  return { systemPrompt, remainingMessages };
}

/**
 * Convert OpenAI chat request to CLI input format
 */
export function openaiToCli(request: OpenAIChatRequest): CliInput {
  const { systemPrompt, remainingMessages } = extractSystemPrompt(request.messages);
  return {
    prompt: messagesToPrompt(remainingMessages),
    model: extractModel(request.model),
    // The caller (routes.ts) assigns a real UUID sessionId; do not use
    // request.user/prompt_cache_key here — Claude CLI requires a UUID.
    systemPrompt: buildSystemPrompt(systemPrompt, request.tools),
  };
}

/**
 * Build CLI input for a request that will --resume an existing Claude CLI
 * session. Since the CLI already remembers everything up to `sinceIndex`
 * (it generated the assistant turns itself), we only need to forward the
 * messages appended since then — not the full history again.
 */
export function openaiToCliDelta(
  request: OpenAIChatRequest,
  sinceIndex: number
): CliInput {
  const newMessages = request.messages
    .slice(sinceIndex)
    .filter((m) => m.role !== "assistant");

  const fallbackMessages = request.messages.filter((m) => m.role !== "assistant");
  const messagesToSend = newMessages.length ? newMessages : fallbackMessages;

  // The system prompt (caller identity + delegation rules) is ALWAYS passed
  // via --system-prompt-file, including on resumed turns. Verified against
  // Claude CLI 2.1.267 with a real 60 964-char Kimi system prompt (measured
  // via the session transcript, single-word prompts):
  //
  //   turn 1: --session-id + spf  -> input 2, cache_read 0,     cache_create 26 552
  //   turn 2: --resume   + spf  -> input 2, cache_read 26 552, cache_create 57
  //   turn 3: --resume   + spf  -> input 2, cache_read 26 609, cache_create 56
  //
  // The prefix is byte-identical at position 0 on every turn, so the CLI
  // re-reads it from the Anthropic cache at x0.1 and only cache-writes the
  // new suffix. An earlier attempt (0624b31) dropped spf on resumed turns to
  // "fix caching" — that was based on a mismeasurement with a ~1.5k-token
  // synthetic prompt (Anthropic silently skips prefixes under ~1024 tokens).
  // On the live prompt it caused two regressions: identity moved into the
  // conversation body was re-cache-written at x1.25 (187 660 effective tokens
  // for 4 turns), and without spf the CLI falls back to its built-in Claude
  // Code system prompt, where the model correctly identifies the <invoke>
  // contract as an injection and refuses to delegate. The system prompt in
  // delegate mode is load-bearing — do not remove it from resumed turns.
  const { systemPrompt } = extractSystemPrompt(request.messages);

  return {
    prompt: messagesToPrompt(messagesToSend),
    model: extractModel(request.model),
    // The caller (routes.ts) assigns a real UUID sessionId; do not use
    // request.user/prompt_cache_key here — Claude CLI requires a UUID.
    systemPrompt: buildSystemPrompt(systemPrompt, request.tools),
  };
}

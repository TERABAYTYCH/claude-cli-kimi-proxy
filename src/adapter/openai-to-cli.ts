/**
 * Converts OpenAI chat request format to Claude CLI input
 */

import type {
  OpenAIChatRequest,
  OpenAIContentBlock,
  OpenAIChatMessage,
  OpenAIToolDefinition,
} from "../types/openai.js";

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
          parts.push(`<previous_response>\n${invokes}\n</previous_response>\n`);
        } else {
          parts.push(`<previous_response>\n${text}\n</previous_response>\n`);
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
function buildDelegationInstruction(
  tools: OpenAIToolDefinition[]
): string {
  // Intentionally compact: full JSON schemas for every tool cost ~40k tokens
  // of cache prefix on every request. The upstream client (Kimi) validates
  // arguments against its own schema, so we list names + one-line description
  // + parameter names with JSON-schema types (no enums/nesting/required).
  const toolDescriptions = [...tools]
    .sort((a, b) => a.function.name.localeCompare(b.function.name))
    .map((t) => {
      const fn = t.function;
      const desc = (fn.description || "").split("\n")[0].trim().slice(0, 120);
      const properties = (fn.parameters as
        | { properties?: Record<string, { type?: string }> }
        | undefined)?.properties;
      const paramNames = properties
        ? Object.entries(properties)
            .map(([name, schema]) => `${name}:${schema?.type || "string"}`)
            .join(", ")
        : "";
      return `- ${fn.name}: ${desc}${paramNames ? ` [params: ${paramNames}]` : ""}`;
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

Format:
<invoke name="<ToolName>">
<parameter name="<paramName>">value</parameter>
...
</invoke>

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

  // System prompt must be rebuilt from the full message history on every turn,
  // especially in delegate mode: after --resume the CLI does not re-apply the
  // previous --system-prompt-file, so Claude would forget the delegation rules.
  const { systemPrompt } = extractSystemPrompt(request.messages);

  return {
    prompt: messagesToPrompt(messagesToSend),
    model: extractModel(request.model),
    // The caller (routes.ts) assigns a real UUID sessionId; do not use
    // request.user/prompt_cache_key here — Claude CLI requires a UUID.
    systemPrompt: buildSystemPrompt(systemPrompt, request.tools),
  };
}

# Architecture

How an OpenAI-shaped request becomes a `claude` CLI invocation, and how the
answer travels back.

## The impedance mismatch

Two incompatible models of a conversation meet here.

**The client speaks OpenAI.** Every request carries the *entire* `messages`
array from scratch, plus a `tools` array of JSON schemas. Nothing is
remembered server-side.

**Claude Code CLI speaks sessions.** It takes one prompt on stdin and keeps
the conversation in its own transcript at
`~/.claude/projects/<slug>/<uuid>.jsonl`.

The proxy translates between them. Almost all of its complexity — and all of
its token cost — lives in the seams of that translation.

---

## Request pipeline

### 1. Intake — `src/server/index.ts`

`POST /v1/chat/completions` is read as a raw buffer (`express.raw`) and parsed
manually, so a malformed body produces a diagnosable log line instead of a
silent 400.

### 2. Mode decision — `resolveCliInput` in `src/server/routes.ts`

This fork in the road determines the cost of everything downstream.

1. **Conversation key** — `body.user || body.prompt_cache_key`. Without one,
   the request is treated as one-shot (the client's own utility calls arrive
   this way) and runs with `--no-session-persistence`.

2. **Per-key lock** — `acquireSessionWait(key, 8000)`, polling every 100 ms.
   Waiting rather than failing fast matters: the proxy closes the SSE stream
   the moment it sees `</invoke>`, the client executes the tool and sends the
   next turn roughly 15 ms later, while the lock is still held for another
   ~500 ms until the subprocess exits and flushes its transcript. Falling back
   immediately would turn every other turn into a full-history replay *and*
   desynchronise the session, because the fallback process emits invokes the
   resumed session never saw.

3. **Session lookup happens after the lock**, never before — the previous turn
   may have persisted a newer `messageCount` while this request was waiting.

4. **Delta validation** — `isDeltaValid`:
   ```ts
   if (sinceIndex >= body.messages.length) return false;   // client trimmed history
   const slice = body.messages.slice(sinceIndex);
   if (!slice.some(m => m.role === "tool" || m.role === "user")) return false;
   ```
   The second check accepts both shapes that carry new input: a tool result
   (a delegation round-trip) and a plain user message (a follow-up question).
   Requiring a tool message — as an earlier version did — made every
   conversational turn fall back to a full replay.

| state | action | `resume` |
|---|---|---|
| session exists, delta valid | `openaiToCliDelta` | **true** |
| session exists, delta invalid | `clearSession`, `openaiToCli`, fresh UUID | false |
| no session | `openaiToCli`, fresh UUID | false |

### 3. System prompt — `buildSystemPrompt`

Built identically on **every** turn, resumed ones included:

```
caller identity (role:"system" messages)   +   delegation instruction
```

The delegation instruction has two halves. The **contract**: no direct tool
access, emit exactly one `<invoke>` and stop, never repeat an invoke whose
`<tool_result>` is already in the history. And the **tool map**, one line per
tool, assembled from the request's `tools` array:

```
- AgentSwarm: Launch multiple subagents from one prompt template...
  [constraints: provide at least 2 `items` unless you pass `resume_agent_ids`;
                whenever `items` are present, `prompt_template` is required]
  (items:string[]? "Values used to fill {{item}}...",
   subagent_type:"plan"|"coder"|"explore"? "Subagent type used for every new...")
```

Four sources feed each line, in priority order:

1. **types and requiredness** from `parameters.properties` (`!` vs `?`);
2. **enums** from the schema if present; otherwise extracted from the tool's
   prose (`Available X (pass via param): - a: ... - b: ...`); otherwise
   inherited from another tool that has a parameter of the same name;
3. **constraints** — sentences anywhere in the description carrying markers
   such as `required`, `must`, `enforced`, `at least`, `whenever`;
4. **parameter descriptions**, one line each.

Tools and parameters are sorted by name. This is not cosmetic: the prefix must
be byte-identical between turns or Anthropic's cache misses and the whole
block is re-written at 1.25× instead of re-read at 0.1×.

Real client schemas turn out to be nearly information-free — no `enum`s, no
`dependentRequired`, conditional rules stated only in prose. Hence the prose
extraction.

### 4. Prompt body — `messagesToPrompt`

The message array is flattened into one text with three block kinds:

```
изучи проект                     ← role:"user", verbatim, no wrapper

<previous_response>              ← role:"assistant"
[up to 400 chars of reasoning_content]
<invoke name="Read"><parameter name="path">README.md</parameter></invoke>
</previous_response>

<tool_result>                    ← role:"tool"
...
</tool_result>
```

`role:"system"` never appears here — it was extracted earlier and went into
`--system-prompt-file`.

Note that `tool_calls` arguments are expanded into individual `<parameter>`
elements rather than a single JSON blob. Shown a
`<parameter name="arguments">{...}</parameter>`, the model imitates the shape
and the client rejects the resulting call for having an unexpected
`arguments` property.

### 5. Where the delta comes from — `openaiToCliDelta`

```ts
newMessages = messages.slice(sinceIndex).filter(m => m.role !== "assistant");
```

`sinceIndex` is the `messageCount` stored on the previous turn — the array
length as the proxy last saw it.

Between delegation turns the client appends exactly two messages: an
`assistant` with `tool_calls` and a `tool` with the result. The filter drops
the assistant one, **because the CLI generated that turn itself and already
has it in its transcript**. One `<tool_result>` remains.

Between conversational turns it appends an `assistant` answer and a `user`
question; only the question remains, often 25–50 characters.

Measured over one 27-turn run:

| turn | slice | prompt |
|---|---|---|
| 1 | whole history | 826 chars |
| 2 | `[assistant, user]` | **25 chars** |
| 3 | `[assistant, tool]` | 69 chars |
| 18 | `[assistant, tool]`, large file | 9 574 chars |
| 27 | `[assistant, user]` | 41 chars |

The prompt does not accumulate; it is exactly the size of what is new.

### 6. Invocation — `buildArgs` in `src/subprocess/manager.ts`

```bash
claude --print --output-format stream-json --verbose --include-partial-messages \
       --model sonnet \
       --system-prompt-file /tmp/claude-sysprompt-<uuid>.txt \
       --tools "" \
       --resume <session-uuid>       # or --session-id on the first turn
                                     # or --no-session-persistence with no key
```

The prompt goes over **stdin**, not as an argument — large bodies would hit
`E2BIG`. The system prompt goes through a temp file for the same reason; it is
removed when the subprocess closes. `--tools ""` disables the CLI's built-in
tools, without which Claude would simply execute `Read` itself and the client
would never see a tool call. `CLAUDECODE` is stripped from the environment so
the CLI does not think it is nested inside another Claude Code.

### 7. Response path — `src/adapter/delegate-parser.ts`

The proxy accumulates streamed text and counts `</invoke>` occurrences,
parsing only when the count grows — scanning on every delta re-parsed the
whole buffer and logged one quoted invoke 120 times.

`parseDelegations` then:

1. **strips quoted regions** — fenced code blocks and `<tool_result>` bodies.
   A model describing this proxy's own source reproduces the
   `<invoke name="<ToolName>">` template verbatim; without stripping, that
   template is forwarded as a real call;
2. extracts the remaining `<invoke>` / `<parameter>` pairs;
3. normalises the name through an alias table (`exec`→`Bash`,
   `web_search`→`WebSearch`) and **drops the delegation entirely if the name
   is not in the request's `tools`**, warning once per unique name;
4. coerces parameter types against the schema — `"5"`→`5`, `"true"`→`true`,
   an empty string for an object/array parameter means "omitted".

The result is wrapped as OpenAI `tool_calls` with `finish_reason:
"tool_calls"`. If nothing valid survives, the response is returned as plain
text rather than an empty `tool_calls` array.

### 8. Usage accounting — `src/adapter/cli-to-openai.ts`

```
prompt_tokens = input_tokens + cache_read_input_tokens + cache_creation_input_tokens
prompt_tokens_details.cached_tokens = cache_read_input_tokens
```

Per OpenAI's convention `prompt_tokens` covers the whole input including
cached parts. Reporting only `input_tokens` made resumed turns look like two
tokens, so the client believed its context was empty and never trimmed
history — the conversation grew unchecked.

### 9. Session lifecycle — `src/subprocess/session-store.ts`

`setSession` runs as soon as the invoke is emitted, but the **lock is released
only when the subprocess closes**, after the transcript is flushed; otherwise
the next turn can `--resume` a half-written file.

The map is persisted atomically (temp file plus `rename`) next to `cwd`, not
in `$HOME` — the CLI scopes transcripts by project directory, so a shared file
would mix maps from different projects. Lock entries are never persisted: a
lock held by a dead process would block a key until its 20-minute TTL theft.

---

## Cost model

```
effective input tokens = cache_creation × 1.25 + cache_read × 0.10 + input
```

A first turn versus a resumed turn of the same conversation:

| | first turn | resumed turn |
|---|---|---|
| session flag | `--session-id <new>` | `--resume <same>` |
| system prompt | spf, ~82 KB | spf, same bytes |
| stdin prompt | whole history | new block only, 25–150 chars |
| `cache_read` | 0 | 26 000 → 87 000, rising |
| `cache_create` | ~29 000 | 200–3 000 |
| effective | ~36 000 | 3 300–8 000 |

Two thirds of the remaining cost is `cache_read` — re-reading accumulated
context at 0.1× on every turn. That grows linearly with conversation length
and is the architectural floor: the session machinery is already sending
25–150 character deltas. Further savings have to come from how much text
enters the conversation at all, which is why correct `prompt_tokens` matters —
it lets the client manage its own context.

`tests/benchmark/` measures this end to end.

---

## Settled decisions

Recorded because each was re-opened at least once and cost real tokens to
close again.

**`--system-prompt-file` is passed on resumed turns too.** Dropping it to save
prefix bytes fails in two distinct ways. Moving the identity into the
conversation body puts it in an uncacheable position and costs 1.25× per turn
— measured at 187 660 effective tokens for four turns. Removing it entirely
makes the CLI fall back to its built-in Claude Code system prompt, at which
point the model is ordinary Claude Code with real tools, reads "you have no
tools, emit `<invoke>` as text", and correctly refuses it as an injection. In
delegate mode the system prompt is load-bearing, not overhead.

**Cache behaviour must be measured at production scale.** Anthropic silently
skips caching for prefixes under roughly 1024 tokens. A synthetic system
prompt of ~1.5k characters produced `cache_read = 0` and `cache_creation = 0`
and led to the wrong conclusion above. Verified against CLI 2.1.267 with a
real 60 964-character system prompt: turn 1 `--session-id` + spf gives
`cache_create 26 552`; turn 2 `--resume` + the same spf gives `cache_read
26 552, cache_create 57`. Caching works.

**Ground truth is the CLI session transcript**, not only `logs/app.log`. Note
that log timestamps are UTC while file mtimes are local.

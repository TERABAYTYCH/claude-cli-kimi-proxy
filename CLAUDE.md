# Claude Max API Proxy

OpenAI-compatible API proxy that wraps the Claude Code CLI.

## Build

```bash
npm run build    # Compile TypeScript
npm run dev      # Watch mode for development
```

## Service Management

The proxy runs as a macOS LaunchAgent on port 3456.

**Plist location:** `~/Library/LaunchAgents/com.openclaw.claude-max-proxy.plist`

**Logs:**
- stdout: `~/.openclaw/logs/claude-max-proxy.log`
- stderr: `~/.openclaw/logs/claude-max-proxy.err.log`

### Restart the service

```bash
launchctl kickstart -k gui/$(id -u)/com.openclaw.claude-max-proxy
```

### Stop the service

```bash
launchctl bootout gui/$(id -u)/com.openclaw.claude-max-proxy
```

### Start the service (after stop or plist change)

```bash
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.openclaw.claude-max-proxy.plist
```

### Reload after plist changes

```bash
launchctl bootout gui/$(id -u)/com.openclaw.claude-max-proxy
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.openclaw.claude-max-proxy.plist
```

### Check status

```bash
launchctl list com.openclaw.claude-max-proxy
```

## Architecture

- `src/types/claude-cli.ts` - Claude CLI JSON streaming types and type guards
- `src/types/openai.ts` - OpenAI-compatible API types
- `src/adapter/openai-to-cli.ts` - OpenAI request -> CLI input; builds the
  system prompt, the delegation contract and the tool map
- `src/adapter/cli-to-openai.ts` - CLI output -> OpenAI responses, usage accounting
- `src/adapter/delegate-parser.ts` - Parses `<invoke>` blocks back into tool calls
- `src/subprocess/manager.ts` - Spawns and manages Claude CLI subprocesses
- `src/subprocess/session-store.ts` - Conversation key -> CLI session, per-key
  lock, persistence across restarts
- `src/session/resume-delta-guard.ts` - Structural checks on a resumed delta
- `src/session/manager.ts` - Legacy session mapping, not used by the proxy path
- `src/server/index.ts` - Express app, middleware, logging
- `src/server/routes.ts` - Route handlers (streaming + non-streaming)
- `src/server/standalone.ts` - Server entry point
- `src/utils/logger.ts` - Writes every line to stdout **and** to `logs/app.log`;
  do not redirect stdout into that same file or every line lands twice

See `ARCHITECTURE.md` for how a request travels through these, the cost model,
and decisions that should not be re-opened. `tests/benchmark/` measures proxied
work against working with Claude Code directly.

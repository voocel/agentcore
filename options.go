package agentcore

// AgentOption configures an Agent.
type AgentOption func(*Agent)

// ---------------------------------------------------------------------------
// Core — most agents need these
// ---------------------------------------------------------------------------

// WithModel sets the LLM model.
func WithModel(model ChatModel) AgentOption {
	return func(a *Agent) { a.model = model }
}

// WithSystemPrompt sets the system prompt (single-string mode).
func WithSystemPrompt(prompt string) AgentOption {
	return func(a *Agent) { a.systemPrompt = prompt }
}

// WithSystemBlocks sets a multi-block system prompt with per-block cache control.
// Takes precedence over WithSystemPrompt.
func WithSystemBlocks(blocks []SystemBlock) AgentOption {
	return func(a *Agent) { a.systemBlocks = blocks; a.systemPrompt = "" }
}

// WithTools sets the tool list.
func WithTools(tools ...Tool) AgentOption {
	return func(a *Agent) { a.tools = tools }
}

// WithMaxTurns sets the max turns safety limit.
func WithMaxTurns(n int) AgentOption {
	return func(a *Agent) { a.maxTurns = n }
}

// WithThinkingLevel sets the reasoning depth for models that support it.
func WithThinkingLevel(level ThinkingLevel) AgentOption {
	return func(a *Agent) { a.thinkingLevel = level }
}

// WithCacheLastMessage tags the last non-system message with cache_control
// before every LLM call. Providers that support prompt caching (Anthropic,
// Bedrock) place a write breakpoint at that position, covering the entire
// preceding prefix (system blocks + conversation history + tools).
//
// The marker lands on whichever turn is freshest — user input, tool_result,
// or assistant — and skips trailing per-turn system reminders. Inside a tool
// loop this means each LLM call writes a cache entry covering the latest
// tool_use+tool_result, so the next call reads them from cache instead of
// re-uploading.
//
// Pass "" (default) to leave messages untouched. Pass "ephemeral" for the
// standard 5-minute TTL, or any provider-recognized value. Use this when the
// application — not the LLM library — owns cache placement.
func WithCacheLastMessage(cacheControl string) AgentOption {
	return func(a *Agent) { a.cacheLastMessage = cacheControl }
}

// ---------------------------------------------------------------------------
// Reliability — retry / circuit breaker
// ---------------------------------------------------------------------------

// WithMaxRetries sets the LLM call retry limit for retryable errors.
func WithMaxRetries(n int) AgentOption {
	return func(a *Agent) { a.maxRetries = n }
}

// WithToolsAreIdempotent declares that all tools registered on this agent are
// safe to re-execute with the same arguments. When true, an LLM call that
// fails after a tool_call has already streamed (e.g. stream-idle timeout
// before stop_reason arrives) will be retried instead of bailing out: the
// in-flight tool execution is aborted and the turn replays from scratch.
//
// Only enable this when every tool's side effect is deduplicated by content
// (e.g. checkpoint+digest, write-once via tmp+rename). The default (false)
// preserves the conservative behavior that protects non-idempotent tools.
func WithToolsAreIdempotent(idempotent bool) AgentOption {
	return func(a *Agent) { a.toolsAreIdempotent = idempotent }
}

// ---------------------------------------------------------------------------
// Context Pipeline — manage context window and message transformation
// ---------------------------------------------------------------------------

// WithContextManager sets the context lifecycle manager.
// When configured, it drives prompt projection, overflow recovery, and usage
// reporting. The agent auto-wires ConvertToLLM, context-token estimation,
// and the context window from the manager when it implements the optional
// ContextLLMConverter / ContextEstimator / ContextWindower interfaces.
func WithContextManager(mgr ContextManager) AgentOption {
	return func(a *Agent) { a.contextManager = mgr }
}

// ---------------------------------------------------------------------------
// Tool Execution — permissions, concurrency, middleware, circuit breaker
// ---------------------------------------------------------------------------

// WithToolGate installs a hook called once per tool call after argument
// validation and the optional Previewer pass. Returning Allowed=false rejects
// the call (Reason becomes the tool result error). The agent core does not
// implement permission reasoning of its own — gates are user-supplied.
func WithToolGate(gate ToolGate) AgentOption {
	return func(a *Agent) { a.toolGate = gate }
}

// WithMiddlewares sets tool execution middlewares.
// Each middleware wraps the tool.Execute call. First middleware is outermost.
func WithMiddlewares(mw ...ToolMiddleware) AgentOption {
	return func(a *Agent) { a.middlewares = mw }
}

// WithMaxToolConcurrency sets the maximum number of tools executed in parallel.
// 0 or 1 = sequential (default). >1 enables concurrent tool execution.
func WithMaxToolConcurrency(n int) AgentOption {
	return func(a *Agent) { a.maxToolConcurrency = n }
}

// WithMaxToolErrors sets the consecutive failure threshold per tool.
// After reaching this limit, the tool is disabled for the rest of the loop.
// 0 means unlimited (no circuit breaker).
func WithMaxToolErrors(n int) AgentOption {
	return func(a *Agent) { a.maxToolErrors = n }
}

// ---------------------------------------------------------------------------
// Hooks — message callbacks
// ---------------------------------------------------------------------------

// WithOnMessage registers a callback invoked after each message is appended
// to the agent's context. Use for session logging / message persistence.
func WithOnMessage(fn func(AgentMessage)) AgentOption {
	return func(a *Agent) { a.onMessage = fn }
}

// ---------------------------------------------------------------------------
// StopGuard — long-run stability primitive
// ---------------------------------------------------------------------------

// WithStopGuard installs a guard that decides whether the agent may stop
// when the LLM emits end_turn without tool calls. Nil guard (default) means
// every stop is allowed — legacy behavior.
func WithStopGuard(guard StopGuard) AgentOption {
	return func(a *Agent) { a.stopGuard = guard }
}

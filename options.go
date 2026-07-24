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

// WithMaxTurns sets the max turns safety limit. n <= 0 falls back to the
// built-in default (100).
//
// Accounting note: a "turn" is one LLM call. Length-recovery (automatic
// resume after a max_tokens truncation) injects extra calls that count
// toward this limit, so budget recoveries into tight limits.
func WithMaxTurns(n int) AgentOption {
	return func(a *Agent) { a.maxTurns = n }
}

// WithLengthRecoveryPrompt overrides the user message injected when output
// is truncated (max_tokens) with no completed tool calls. Empty keeps the
// built-in default prompt.
func WithLengthRecoveryPrompt(prompt string) AgentOption {
	return func(a *Agent) { a.lengthRecoveryPrompt = prompt }
}

// WithAbortMarkerText overrides the marker messages recorded when a run is
// cancelled: inference is used mid-inference, toolUse during tool execution.
// Either empty keeps that built-in default. Lets non-English harnesses
// localize the markers that get written into conversation history.
func WithAbortMarkerText(inference, toolUse string) AgentOption {
	return func(a *Agent) {
		a.abortMarkerText = inference
		a.abortMarkerToolText = toolUse
	}
}

// WithThinkingLevel sets the reasoning depth for models that support it.
func WithThinkingLevel(level ThinkingLevel) AgentOption {
	return func(a *Agent) { a.thinkingLevel = NormalizeThinkingLevel(level) }
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
// When the agent uses a plain SystemPrompt (not SystemBlocks), the loop also
// pins the system message with the same cache_control as a stable floor, so
// fresh sessions sharing the prompt reuse the system+tools prefix. SystemBlocks
// users keep explicit control via SystemBlock.CacheControl.
//
// Pass "" (default) to leave messages untouched. Pass "ephemeral" for the
// standard 5-minute TTL, or "ephemeral:1h" for the extended TTL where the
// provider supports it (use for conversations whose turn gaps regularly
// exceed 5 minutes). Use this when the application — not the LLM library —
// owns cache placement.
func WithCacheLastMessage(cacheControl string) AgentOption {
	return func(a *Agent) { a.cacheLastMessage = cacheControl }
}

// WithPromptCacheKey sets the prompt-cache routing identity attached to every
// LLM request of this agent (e.g. OpenAI prompt_cache_key). Keep one key per
// long-lived conversation: requests sharing a key are routed to the same
// provider cache shard, so each turn can read the previous turn's prefix from
// cache. The adapter drops the hint for providers without key-routed caching.
// Empty (default) sends no hint.
func WithPromptCacheKey(key string) AgentOption {
	return func(a *Agent) { a.promptCacheKey = key }
}

// ---------------------------------------------------------------------------
// Reliability — retry / circuit breaker
// ---------------------------------------------------------------------------

// WithMaxRetries sets the LLM call retry limit for retryable errors.
func WithMaxRetries(n int) AgentOption {
	return func(a *Agent) { a.maxRetries = n }
}

// WithToolsAreIdempotent is retained for source compatibility.
// Deprecated: tools start only after a complete assistant response has been
// committed, so model-stream retries cannot replay tool side effects.
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
// ContextLLMConverter / ContextEstimator / ContextWindowProvider interfaces.
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

// WithMessageCommitter installs a synchronous durable-message callback.
// Returning an error stops the run before the message enters context or starts
// requested tools.
func WithMessageCommitter(fn func(AgentMessage) error) AgentOption {
	return func(a *Agent) { a.messageCommitter = fn }
}

// WithOnMessage registers a callback invoked after each message is appended
// to the agent's context. Use it for observation; durable persistence should
// use WithMessageCommitter so write errors can stop execution.
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

package agentcore

import (
	"context"
	"time"

	"github.com/voocel/agentcore/permission"
)

// AgentOption configures an Agent.
type AgentOption func(*Agent)

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

// WithStreamFn sets a custom LLM call function (for proxy/mock).
func WithStreamFn(fn StreamFn) AgentOption {
	return func(a *Agent) { a.streamFn = fn }
}

// WithTransformContext sets the context transform function.
func WithTransformContext(fn func(ctx context.Context, msgs []AgentMessage) ([]AgentMessage, error)) AgentOption {
	return func(a *Agent) { a.transformContext = fn }
}

// WithContextManager sets the context lifecycle manager.
// When configured, it takes precedence over TransformContext for prompt
// projection, overflow recovery, and usage reporting.
func WithContextManager(mgr ContextManager) AgentOption {
	return func(a *Agent) { a.contextManager = mgr }
}

// WithConvertToLLM sets the message conversion function.
func WithConvertToLLM(fn func([]AgentMessage) []Message) AgentOption {
	return func(a *Agent) { a.convertToLLM = fn }
}

// WithSteeringMode sets the steering queue drain mode.
// QueueModeAll (default) delivers all queued steering messages at once.
// QueueModeOneAtATime delivers one per turn, letting the agent respond to each individually.
func WithSteeringMode(mode QueueMode) AgentOption {
	return func(a *Agent) { a.steeringMode = mode }
}

// WithFollowUpMode sets the follow-up queue drain mode.
// QueueModeAll (default) delivers all queued follow-up messages at once.
// QueueModeOneAtATime delivers one per turn.
func WithFollowUpMode(mode QueueMode) AgentOption {
	return func(a *Agent) { a.followUpMode = mode }
}

// WithThinkingLevel sets the reasoning depth for models that support it.
func WithThinkingLevel(level ThinkingLevel) AgentOption {
	return func(a *Agent) { a.thinkingLevel = level }
}

// WithMaxRetries sets the LLM call retry limit for retryable errors.
func WithMaxRetries(n int) AgentOption {
	return func(a *Agent) { a.maxRetries = n }
}

// WithStrictMessageSequence makes malformed tool call / result transcripts fail
// fast instead of being repaired before the next LLM call.
func WithStrictMessageSequence(enabled bool) AgentOption {
	return func(a *Agent) { a.strictMessageSequence = enabled }
}

// WithMaxToolErrors sets the consecutive failure threshold per tool.
// After reaching this limit, the tool is disabled for the rest of the loop.
// 0 means unlimited (no circuit breaker).
func WithMaxToolErrors(n int) AgentOption {
	return func(a *Agent) { a.maxToolErrors = n }
}

// WithContextWindow sets the model's context window size in tokens.
// Used by ContextUsage() to calculate context occupancy percentage.
func WithContextWindow(n int) AgentOption {
	return func(a *Agent) { a.contextWindow = n }
}

// WithContextEstimate sets the context token estimation function.
// Use context.ContextEstimateAdapter for the default hybrid estimation.
func WithContextEstimate(fn ContextEstimateFn) AgentOption {
	return func(a *Agent) { a.contextEstimateFn = fn }
}

// WithPermissionEngine sets the runtime permission engine called before tool execution.
func WithPermissionEngine(engine permission.DecisionEngine) AgentOption {
	return func(a *Agent) { a.permissionEngine = engine }
}

// WithGetApiKey sets a dynamic API key resolver called before each LLM call.
// The provider parameter identifies which provider is being called (e.g. "openai", "anthropic").
// Enables per-provider key resolution, key rotation, OAuth short-lived tokens, and multi-tenant scenarios.
func WithGetApiKey(fn func(provider string) (string, error)) AgentOption {
	return func(a *Agent) { a.getApiKey = fn }
}

// WithThinkingBudgets sets per-level thinking token budgets.
// Each ThinkingLevel maps to a max thinking token count.
func WithThinkingBudgets(budgets map[ThinkingLevel]int) AgentOption {
	return func(a *Agent) { a.thinkingBudgets = budgets }
}

// WithSessionID sets a session identifier for provider-level caching.
// Forwarded to providers that support session-based prompt caching.
func WithSessionID(id string) AgentOption {
	return func(a *Agent) { a.sessionID = id }
}

// WithMaxRetryDelay caps the wait time between LLM retries.
// Applies to both exponential backoff and server-requested Retry-After delays.
// Default: 60s.
func WithMaxRetryDelay(d time.Duration) AgentOption {
	return func(a *Agent) { a.maxRetryDelay = d }
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

// WithTaskRuntime sets a shared TaskRuntime for background task management.
// Tools that support background execution (Bash, SubAgent) register their
// tasks here, enabling a unified Tasks()/StopTask()/StopAllTasks() API on Agent.
func WithTaskRuntime(rt *TaskRuntime) AgentOption {
	return func(a *Agent) { a.taskRuntime = rt }
}

// WithContextPipeline sets both TransformContext and ConvertToLLM in one call.
// Prefer WithContextManager for the full context lifecycle; use this helper
// for simpler transform-based pipelines.
func WithContextPipeline(
	transform func(ctx context.Context, msgs []AgentMessage) ([]AgentMessage, error),
	convert func([]AgentMessage) []Message,
) AgentOption {
	return func(a *Agent) {
		a.transformContext = transform
		a.convertToLLM = convert
	}
}

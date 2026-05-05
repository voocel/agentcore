package agentcore

import (
	"encoding/json"

	"github.com/voocel/agentcore/permission"
)

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
// reporting. ConvertToLLM and ContextEstimate are auto-wired from the manager
// when it implements the corresponding optional interfaces.
func WithContextManager(mgr ContextManager) AgentOption {
	return func(a *Agent) { a.contextManager = mgr }
}

// WithConvertToLLM sets the message conversion function.
func WithConvertToLLM(fn func([]AgentMessage) []Message) AgentOption {
	return func(a *Agent) { a.convertToLLM = fn }
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

// ---------------------------------------------------------------------------
// Tool Execution — permissions, concurrency, middleware, circuit breaker
// ---------------------------------------------------------------------------

// WithPermissionEngine sets the runtime permission engine called before tool execution.
func WithPermissionEngine(engine permission.DecisionEngine) AgentOption {
	return func(a *Agent) { a.permissionEngine = engine }
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
// Queues & Hooks — background tasks, callbacks
// ---------------------------------------------------------------------------

// WithTaskRuntime sets a shared TaskRuntime for background task management.
// Tools that support background execution (Bash, SubAgent) register their
// tasks here, enabling a unified Tasks()/StopTask()/StopAllTasks() API on Agent.
func WithTaskRuntime(rt *TaskRuntime) AgentOption {
	return func(a *Agent) { a.taskRuntime = rt }
}

// WithOnMessage registers a callback invoked after each message is appended
// to the agent's context. Use for session logging / message persistence.
func WithOnMessage(fn func(AgentMessage)) AgentOption {
	return func(a *Agent) { a.onMessage = fn }
}

// ---------------------------------------------------------------------------
// Reminder / StopGuard / MaxTurns behavior — long-run stability primitives
// ---------------------------------------------------------------------------

// WithReminderGenerator registers a per-turn reminder generator. Multiple calls
// stack: every generator is invoked in registration order before each LLM call,
// and their combined reminders are injected as one-turn system messages.
// Reminders do not enter the persistent message history.
func WithReminderGenerator(gen ReminderGenerator) AgentOption {
	return func(a *Agent) {
		if gen == nil {
			return
		}
		a.reminderGens = append(a.reminderGens, gen)
	}
}

// WithStopGuard installs a guard that decides whether the agent may stop
// when the LLM emits end_turn without tool calls. Nil guard (default) means
// every stop is allowed — legacy behavior.
func WithStopGuard(guard StopGuard) AgentOption {
	return func(a *Agent) { a.stopGuard = guard }
}

// MaxTurnsAction selects the behavior when MaxTurns is reached.
type MaxTurnsAction int

const (
	// MaxTurnsTerminate (default) emits an error event and ends the run.
	MaxTurnsTerminate MaxTurnsAction = iota
	// MaxTurnsSoftRestart resets the internal turn counter to 0 and continues
	// the loop. Useful for very long runs where MaxTurns is a soft upper bound.
	MaxTurnsSoftRestart
)

// WithOnMaxTurns configures what happens when the MaxTurns safety limit is
// reached. The default is MaxTurnsTerminate.
func WithOnMaxTurns(action MaxTurnsAction) AgentOption {
	return func(a *Agent) { a.onMaxTurns = action }
}

// WithStopAfterTool installs a predicate that ends the agent run after a
// successful execution of any tool whose name returns true. The terminating
// tool's result is committed to history before the run exits.
//
// Use cases: terminal tools (e.g. commit_chapter, exit_plan_mode) that
// shouldn't waste another LLM turn after they succeed.
func WithStopAfterTool(fn func(toolName string) bool) AgentOption {
	return func(a *Agent) { a.stopAfterTool = fn }
}

// WithStopAfterToolResult installs a result-aware predicate that ends the
// agent run after a successful tool execution whose structured result returns
// true. The terminating tool's result is committed to history before exit.
func WithStopAfterToolResult(fn func(toolName string, result json.RawMessage) bool) AgentOption {
	return func(a *Agent) { a.stopAfterToolResult = fn }
}

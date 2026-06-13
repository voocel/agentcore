package agentcore

import (
	"context"
	"encoding/json"
)

// ---------------------------------------------------------------------------
// Tool Progress
// ---------------------------------------------------------------------------

// toolProgressKey is the context key for tool progress callbacks.
type toolProgressKey struct{}

// ProgressPayloadKind distinguishes structured progress update semantics.
type ProgressPayloadKind string

const (
	ProgressToolStart   ProgressPayloadKind = "tool_start"
	ProgressToolEnd     ProgressPayloadKind = "tool_end"
	ProgressToolDelta   ProgressPayloadKind = "tool_delta"
	ProgressThinking    ProgressPayloadKind = "thinking"
	ProgressSummary     ProgressPayloadKind = "summary"
	ProgressToolError   ProgressPayloadKind = "tool_error"
	ProgressTurnCounter ProgressPayloadKind = "turn_counter"
	ProgressRetry       ProgressPayloadKind = "retry"
	ProgressContext     ProgressPayloadKind = "context"
)

// ProgressPayload is the structured progress envelope emitted by tools.
type ProgressPayload struct {
	Kind       ProgressPayloadKind `json:"kind"`
	Agent      string              `json:"agent,omitempty"`
	Tool       string              `json:"tool,omitempty"`
	Summary    string              `json:"summary,omitempty"`
	Delta      string              `json:"delta,omitempty"`
	Thinking   string              `json:"thinking,omitempty"`
	Message    string              `json:"message,omitempty"`
	Turn       int                 `json:"turn,omitempty"`
	Attempt    int                 `json:"attempt,omitempty"`
	MaxRetries int                 `json:"max_retries,omitempty"`
	IsError    bool                `json:"is_error,omitempty"`
	Args       json.RawMessage     `json:"args,omitempty"`
	Meta       json.RawMessage     `json:"meta,omitempty"`
	// DeltaKind distinguishes what kind of content Delta carries when Kind is
	// ProgressToolDelta. Consumers can use this to filter/render text vs
	// tool-call argument JSON differently.
	DeltaKind DeltaKind `json:"delta_kind,omitempty"`
}

// ToolProgressFunc is a callback for reporting tool execution progress.
// Tools call ReportToolProgress to emit partial results during long operations.
type ToolProgressFunc func(progress ProgressPayload)

// WithToolProgress injects a progress callback into the context.
func WithToolProgress(ctx context.Context, fn ToolProgressFunc) context.Context {
	return context.WithValue(ctx, toolProgressKey{}, fn)
}

// ReportToolProgress reports structured progress during tool execution.
// Silently ignored if no callback is registered in the context.
func ReportToolProgress(ctx context.Context, progress ProgressPayload) {
	if progress.Kind == "" {
		progress.Kind = ProgressSummary
	}
	if fn, ok := ctx.Value(toolProgressKey{}).(ToolProgressFunc); ok {
		fn(progress)
	}
}

// ---------------------------------------------------------------------------
// Tool Calls & Results
// ---------------------------------------------------------------------------

// ToolCall represents a tool invocation request from the LLM.
//
// When the LLM emits args that don't parse as JSON (common cause: stream
// truncation, provider format bug), Args is replaced with "{}" so the
// surrounding Message stays JSON-serializable for persistence; the original
// payload and parser diagnostic are preserved in ArgsRawText / ArgsParseError.
// Downstream schema validation short-circuits on ArgsInvalid and surfaces the
// captured raw text — pointing at the real root cause instead of running
// "missing field" checks against the {} placeholder.
type ToolCall struct {
	ID             string          `json:"id"`
	Name           string          `json:"name"`
	Args           json.RawMessage `json:"args"`
	ArgsInvalid    bool            `json:"args_invalid,omitempty"`
	ArgsRawText    string          `json:"args_raw_text,omitempty"`
	ArgsParseError string          `json:"args_parse_error,omitempty"`
	// ThoughtSignature is an opaque provider reasoning signature (Gemini 3) that
	// must be persisted and replayed verbatim across turns. Empty when absent.
	ThoughtSignature string `json:"thought_signature,omitempty"`
}

// ToolResult represents a tool execution outcome.
type ToolResult struct {
	ToolCallID    string          `json:"tool_call_id"`
	ToolName      string          `json:"-"` // internal: for toolErrors tracking
	Content       json.RawMessage `json:"content,omitempty"`
	ContentBlocks []ContentBlock  `json:"-"` // rich content (images); not serialized
	IsError       bool            `json:"is_error,omitempty"`
	Details       any             `json:"details,omitempty"` // optional metadata for UI display/logging
}

// ---------------------------------------------------------------------------
// Tool Interface
// ---------------------------------------------------------------------------

// Tool defines the minimal tool interface.
// Timeout control goes through context.Context.
// Tools can report execution progress via ReportToolProgress(ctx, payload).
type Tool interface {
	Name() string
	Description() string
	Schema() map[string]any
	Execute(ctx context.Context, args json.RawMessage) (json.RawMessage, error)
}

// ToolLabeler is an optional interface for tools to provide a human-readable label.
type ToolLabeler interface {
	Label() string
}

// StrictSchemaTool is an optional interface for tools that want provider-side
// strict schema enforcement on their arguments (e.g. OpenAI's strict tool
// calling). Returning true forwards `strict: true` and triggers schema
// normalisation in compatible providers; returning false explicitly disables
// strict on providers that default to it (e.g. OpenAI Responses API).
//
// The tool author is responsible for providing a strict-compatible schema:
// every property listed in `required`, no unsupported keywords. See the
// litellm provider docs for the exact subset.
type StrictSchemaTool interface {
	StrictSchema() bool
}

// ContentTool is an optional interface for tools that return rich content
// (e.g., images). When a tool implements ContentTool, the agent loop calls
// ExecuteContent instead of Execute, enabling multi-block responses with
// text + image content blocks.
type ContentTool interface {
	ExecuteContent(ctx context.Context, args json.RawMessage) ([]ContentBlock, error)
}

// Previewer is an optional interface for tools that can compute a preview
// (e.g., diff) before execution. The agent loop calls Preview and emits the
// result as EventToolExecUpdate so the UI can display it before the tool runs.
type Previewer interface {
	Preview(ctx context.Context, args json.RawMessage) (json.RawMessage, error)
}

// ValidationResult is the verdict from a Validator.
//
// A failure (OK=false) is surfaced to the LLM as a normal tool_result with
// IsError=true. The intent is "input is structurally legal but semantically
// wrong" — e.g. write before read, mtime drift, deny rule. The LLM reads
// Message and self-corrects (typically by issuing the right tool first and
// retrying), without prompting the user.
//
// ErrorCode is optional, intended for stable identification by tests and
// prompts; it is not interpreted by the agent core.
type ValidationResult struct {
	OK        bool
	Message   string
	ErrorCode int
}

// Validator is an optional interface for tools that want to short-circuit
// before Preview / ToolGate / Execute when the input is structurally legal
// but semantically wrong. Validators MUST NOT prompt the user, MUST NOT
// mutate persistent state, and SHOULD be cheap (read-only lookups, stat).
//
// Returning OK=false produces a tool_result the LLM can act on; returning
// OK=true continues the normal pipeline.
type Validator interface {
	Validate(ctx context.Context, args json.RawMessage) ValidationResult
}

// ---------------------------------------------------------------------------
// ToolGate — pluggable approval / policy hook
// ---------------------------------------------------------------------------

// GateRequest carries the inputs that a ToolGate sees for one tool call.
// Tool exposes the underlying tool instance so gates can typeswitch against
// any tool-specific marker interfaces they care about (e.g. capability hints)
// without the agent core needing to know those interfaces.
type GateRequest struct {
	Tool      Tool
	Call      ToolCall
	ToolLabel string          // resolved via ToolLabeler when available
	Preview   json.RawMessage // resolved via Previewer when available; may be nil
}

// GateDecision is the gate's verdict for one tool call.
//
// Allowed=true => execute the tool with Call.Args.
// Allowed=false => return Reason as the tool result error; do not execute.
//
// A nil decision is treated as Allowed=true (the gate has no opinion).
type GateDecision struct {
	Allowed bool
	Reason  string
}

// ToolGate is the pluggable hook called once per tool call, after argument
// validation and after the optional Previewer pass, but before tool
// execution. Returning a non-nil error is treated as deny with the error
// message as the reason. The agent core does not perform any permission
// reasoning of its own; install a gate (or leave it nil) to control policy.
type ToolGate func(ctx context.Context, req GateRequest) (*GateDecision, error)

// DeferFilter controls deferred tool loading for the LLM.
// When a tool in the agent's tool list implements DeferFilter:
//   - IsDeferred returns true → tool schema is excluded from the API request
//   - WasDeferred returns true → tool schema is sent with defer_loading: true
//
// Unactivated deferred tools are excluded entirely. Once activated via
// tool_reference, they are sent with defer_loading: true so the API server
// manages their context loading. Tools remain registered for execution
// regardless — only their API visibility changes.
//
// IsDeferred is also used by the system prompt builder to exclude unactivated
// tools from the tool description section (they appear in
// <available-deferred-tools> by name only).
type DeferFilter interface {
	// IsDeferred reports whether the tool is deferred and not yet activated.
	// Unactivated deferred tools are excluded from the API request entirely.
	IsDeferred(toolName string) bool
	// WasDeferred reports whether the tool was originally in the deferred set
	// (regardless of activation). Activated deferred tools are sent with
	// defer_loading: true.
	WasDeferred(toolName string) bool
}

// DeferActivator is an optional extension of DeferFilter that supports
// pre-activating deferred tools (e.g. when restoring a session whose
// history contains tool_reference blocks for previously activated tools).
type DeferActivator interface {
	DeferFilter
	Activate(names ...string)
}

// ReactivateDeferred scans restored messages for tool_reference blocks and
// pre-activates them via the DeferActivator found in tools. This must be
// called after restoring a session to avoid "Tool reference not found" errors.
func ReactivateDeferred(tools []Tool, msgs []AgentMessage) {
	var activator DeferActivator
	for _, t := range tools {
		if a, ok := t.(DeferActivator); ok {
			activator = a
			break
		}
	}
	if activator == nil {
		return
	}

	var names []string
	for _, am := range msgs {
		msg, ok := am.(Message)
		if !ok {
			continue
		}
		for _, b := range msg.Content {
			if b.Type == ContentToolRef && b.ToolName != "" {
				names = append(names, b.ToolName)
			}
		}
	}
	if len(names) > 0 {
		activator.Activate(names...)
	}
}

// ---------------------------------------------------------------------------
// Tool Behavior Interfaces (optional)
// ---------------------------------------------------------------------------

// ReadOnlyTool is an optional interface for tools that declare read-only behavior.
// Read-only tools are eligible for concurrent execution by default.
// The args parameter allows input-dependent classification
// (e.g., bash is read-only for "ls" but not for "rm").
type ReadOnlyTool interface {
	ReadOnly(args json.RawMessage) bool
}

// ConcurrencySafeTool is an optional interface for tools that declare
// whether they can safely execute concurrently with other tools.
// Takes precedence over ReadOnlyTool for concurrency scheduling.
type ConcurrencySafeTool interface {
	ConcurrencySafe(args json.RawMessage) bool
}

// InterruptBehavior controls what happens when a queued user message arrives
// while a tool is still running.
type InterruptBehavior string

const (
	InterruptBehaviorBlock  InterruptBehavior = "block"
	InterruptBehaviorCancel InterruptBehavior = "cancel"
)

// InterruptBehaviorTool is an optional interface for tools that declare whether
// they should be cancelled or allowed to finish when a steering message arrives.
// Defaults to InterruptBehaviorBlock when not implemented.
type InterruptBehaviorTool interface {
	InterruptBehavior(args json.RawMessage) InterruptBehavior
}

// ActivityDescriber is an optional interface for tools that provide
// a human-readable activity description for UI display.
type ActivityDescriber interface {
	ActivityDescription(args json.RawMessage) string
}

// isToolConcurrencySafe checks whether a tool call is safe for concurrent execution.
// Priority: ConcurrencySafeTool > ReadOnlyTool > false.
func isToolConcurrencySafe(tool Tool, args json.RawMessage) bool {
	if cs, ok := tool.(ConcurrencySafeTool); ok {
		return cs.ConcurrencySafe(args)
	}
	if ro, ok := tool.(ReadOnlyTool); ok {
		return ro.ReadOnly(args)
	}
	return false
}

func toolInterruptBehavior(tool Tool, args json.RawMessage) InterruptBehavior {
	if ib, ok := tool.(InterruptBehaviorTool); ok {
		switch behavior := ib.InterruptBehavior(args); behavior {
		case InterruptBehaviorCancel:
			return InterruptBehaviorCancel
		case InterruptBehaviorBlock:
			return InterruptBehaviorBlock
		}
	}
	return InterruptBehaviorBlock
}

// ToolExecuteFunc is the function signature for tool execution.
// Used as the "next" parameter in middleware chains.
type ToolExecuteFunc func(ctx context.Context, args json.RawMessage) (json.RawMessage, error)

// ToolMiddleware wraps tool execution with cross-cutting concerns.
// Call next to continue the chain; skip next to short-circuit execution.
// Example: logging, timing, argument/result modification, audit.
type ToolMiddleware func(ctx context.Context, call ToolCall, next ToolExecuteFunc) (json.RawMessage, error)

// ---------------------------------------------------------------------------
// FuncTool
// ---------------------------------------------------------------------------

// FuncTool wraps a function as a Tool (convenience helper).
type FuncTool struct {
	name        string
	description string
	schema      map[string]any
	fn          func(ctx context.Context, args json.RawMessage) (json.RawMessage, error)
}

func NewFuncTool(name, description string, schema map[string]any, fn func(ctx context.Context, args json.RawMessage) (json.RawMessage, error)) *FuncTool {
	return &FuncTool{name: name, description: description, schema: schema, fn: fn}
}

func (t *FuncTool) Name() string           { return t.name }
func (t *FuncTool) Description() string    { return t.description }
func (t *FuncTool) Schema() map[string]any { return t.schema }
func (t *FuncTool) Execute(ctx context.Context, args json.RawMessage) (json.RawMessage, error) {
	return t.fn(ctx, args)
}

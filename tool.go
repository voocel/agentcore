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

// ToolProgressFunc is a callback for reporting tool execution progress.
// Tools call ReportToolProgress to emit partial results during long operations.
type ToolProgressFunc func(partialResult json.RawMessage)

// WithToolProgress injects a progress callback into the context.
func WithToolProgress(ctx context.Context, fn ToolProgressFunc) context.Context {
	return context.WithValue(ctx, toolProgressKey{}, fn)
}

// ReportToolProgress reports partial progress during tool execution.
// Silently ignored if no callback is registered in the context.
func ReportToolProgress(ctx context.Context, partial json.RawMessage) {
	if fn, ok := ctx.Value(toolProgressKey{}).(ToolProgressFunc); ok {
		fn(partial)
	}
}

// ---------------------------------------------------------------------------
// Tool Calls & Results
// ---------------------------------------------------------------------------

// ToolCall represents a tool invocation request from the LLM.
type ToolCall struct {
	ID   string          `json:"id"`
	Name string          `json:"name"`
	Args json.RawMessage `json:"args"`
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
// Tools can report execution progress via ReportToolProgress(ctx, partial).
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

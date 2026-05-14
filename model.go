package agentcore

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
)

// ---------------------------------------------------------------------------
// Agent Context & Loop Config
// ---------------------------------------------------------------------------

// SystemBlock is one segment of a multi-part system prompt.
// Use with AgentContext.SystemBlocks for per-block cache control.
type SystemBlock struct {
	Text         string `json:"text"`
	CacheControl string `json:"cache_control,omitempty"` // e.g. "ephemeral"
}

// AgentContext holds the immutable context for a single agent loop invocation.
type AgentContext struct {
	SystemPrompt string        // single-string system prompt (legacy)
	SystemBlocks []SystemBlock // multi-block system prompt with cache control (takes precedence)
	Messages     []AgentMessage
	Tools        []Tool
}

// LLMRequest carries the inputs to a single LLM call.
type LLMRequest struct {
	Messages []Message
	Tools    []ToolSpec
}

// LLMResponse carries the result of a single LLM call.
type LLMResponse struct {
	Message Message
}

// ToolSpec describes a tool for the LLM (name + description + JSON schema).
type ToolSpec struct {
	Name         string `json:"name"`
	Description  string `json:"description"`
	Parameters   any    `json:"parameters"`
	DeferLoading bool   `json:"defer_loading,omitempty"`
	// Strict enables provider-side strict schema enforcement (OpenAI strict
	// tool calling / Structured Outputs for arguments). nil leaves the
	// provider default. Set via the optional StrictSchemaTool interface.
	Strict *bool `json:"strict,omitempty"`
}

// LoopConfig configures the agent loop.
type LoopConfig struct {
	Model         ChatModel
	MaxTurns      int           // safety limit, default 10
	MaxRetries    int           // LLM call retry limit for retryable errors, default 3
	MaxToolErrors int           // consecutive tool failure threshold per tool, 0 = unlimited
	ThinkingLevel ThinkingLevel // reasoning depth

	// Context lifecycle. ContextManager drives prompt projection, overflow
	// recovery, and usage reporting; ConvertToLLM is auto-wired from it when
	// the manager implements ContextLLMConverter.
	ContextManager ContextManager
	ConvertToLLM   func(msgs []AgentMessage) []Message

	// CommitContext replaces the runtime message baseline after an explicit
	// committed compaction, a committed projection rewrite, or committed
	// overflow recovery.
	CommitContext func(msgs []AgentMessage, usage *ContextUsage) error

	// ToolGate, when non-nil, is called once per tool call after argument
	// validation and the optional Previewer pass. Allowed=false rejects the
	// call (Reason becomes the tool result). The agent core does no permission
	// reasoning of its own.
	ToolGate ToolGate

	// Steering: called after each tool execution to check for user interruptions.
	GetSteeringMessages func() []AgentMessage

	// FollowUp: called when the agent would otherwise stop.
	GetFollowUpMessages func() []AgentMessage

	// Middlewares are applied around each tool execution (outermost first).
	// Use for logging, timing, argument/result modification, etc.
	Middlewares []ToolMiddleware

	// MaxToolConcurrency limits parallel tool execution.
	// 0 or 1 = sequential (default, backward compatible).
	// >1 = up to N tools execute concurrently within a single turn.
	MaxToolConcurrency int

	// ShouldEmitAbortMarker reports whether an abort marker message should be
	// emitted when the context is cancelled. When nil or returns false, the
	// cancellation is silent (legacy behavior). Set by Agent.Abort().
	ShouldEmitAbortMarker func() bool

	// StopAfterTool, if non-nil, is called after each successful (non-error)
	// tool execution. If it returns true, the loop exits immediately with
	// EndReasonStop. Use this to let a terminal tool (e.g. commit_chapter)
	// end the loop without wasting turns.
	StopAfterTool func(toolName string) bool

	// StopAfterToolResult is the result-aware variant of StopAfterTool. It is
	// useful when the same tool can be an intermediate step or a terminal step
	// depending on its structured result.
	StopAfterToolResult func(toolName string, result json.RawMessage) bool

	// OnMessage, if non-nil, is called after each message is appended to
	// context (assistant, tool result, steering). Use for session logging.
	OnMessage func(msg AgentMessage)

	// ReminderGens are invoked once per turn, just before the LLM request
	// is built. Their output is injected as one-turn system messages between
	// the static system prompt and the conversation history. Reminders are
	// NOT persisted to the agent message history.
	ReminderGens []ReminderGenerator

	// StopGuard is consulted when the LLM would end a run without tool calls.
	// Nil (default) means every stop is allowed.
	StopGuard StopGuard

	// ToolsAreIdempotent declares that all registered tools are safe to re-execute
	// with the same arguments — i.e. running them twice produces the same observable
	// state as running them once. When true, the retry loop will not bail out just
	// because a tool_call already completed in the failed turn; instead it aborts
	// the in-flight tool executions and retries the whole turn. Default false
	// (conservative: assume tools may have non-idempotent side effects).
	ToolsAreIdempotent bool

	// CacheLastMessage, when non-empty, instructs the loop to tag the last
	// non-system message in every LLM request with this cache_control value
	// (e.g. "ephemeral"). Providers that support prompt caching place a write
	// breakpoint at that position covering the entire preceding prefix. Empty
	// string (default) leaves messages untouched — keep cache placement under
	// application control.
	//
	// The breakpoint follows the freshest turn (user input, tool_result, or
	// assistant) and skips trailing per-turn system reminders.
	CacheLastMessage string
}

// ---------------------------------------------------------------------------
// Context Usage Estimation
// ---------------------------------------------------------------------------

// ContextEstimateFn estimates the current context token consumption from messages.
// Returns total tokens, tokens from LLM Usage, and estimated trailing tokens.
type ContextEstimateFn func(msgs []AgentMessage) (tokens, usageTokens, trailingTokens int)

// ContextUsage represents the current context window occupancy estimate.
type ContextUsage struct {
	Tokens         int     `json:"tokens"`          // estimated total tokens in context
	ContextWindow  int     `json:"context_window"`  // model's context window size
	Percent        float64 `json:"percent"`         // tokens / contextWindow * 100
	UsageTokens    int     `json:"usage_tokens"`    // from last LLM-reported Usage
	TrailingTokens int     `json:"trailing_tokens"` // chars/4 estimate for trailing messages
}

// ---------------------------------------------------------------------------
// Call Options
// ---------------------------------------------------------------------------

// CallOption configures per-call LLM parameters.
type CallOption func(*CallConfig)

// CallConfig holds per-call configuration resolved from CallOptions.
type CallConfig struct {
	ThinkingLevel  ThinkingLevel
	ThinkingBudget int    // max thinking tokens, 0 = use provider default
	APIKey         string // per-call API key override, empty = use model default
	SessionID      string // provider session caching identifier
	MaxTokens      int    // per-call max tokens override, 0 = use model default
	ToolChoice     any    // "auto" / "required" / "none" / {"type":"tool","name":"xxx"}, nil = provider default
	ResponseFormat *ResponseFormat
}

const (
	ResponseFormatText       = "text"
	ResponseFormatJSONObject = "json_object"
	ResponseFormatJSONSchema = "json_schema"
)

// ResponseFormat controls provider-native structured output.
//
// JSON object mode asks the model to return valid JSON. JSON schema mode asks
// compatible providers to constrain the final response to the supplied schema.
// Callers should still unmarshal and validate model output like any external
// input.
type ResponseFormat struct {
	Type       string      `json:"type"`
	JSONSchema *JSONSchema `json:"json_schema,omitempty"`
}

// JSONSchema describes the provider-native JSON schema response format.
type JSONSchema struct {
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
	Schema      any    `json:"schema"`
	Strict      *bool  `json:"strict,omitempty"`
}

// ResolveCallConfig applies options and returns the resolved config.
func ResolveCallConfig(opts []CallOption) CallConfig {
	var cfg CallConfig
	for _, opt := range opts {
		opt(&cfg)
	}
	return cfg
}

// WithThinking sets the thinking level for a single LLM call.
func WithThinking(level ThinkingLevel) CallOption {
	return func(c *CallConfig) { c.ThinkingLevel = level }
}

// WithThinkingBudget sets the max thinking tokens for a single LLM call.
func WithThinkingBudget(tokens int) CallOption {
	return func(c *CallConfig) { c.ThinkingBudget = tokens }
}

// WithAPIKey overrides the API key for a single LLM call.
// Enables key rotation, OAuth short-lived tokens, and multi-tenant scenarios.
func WithAPIKey(key string) CallOption {
	return func(c *CallConfig) { c.APIKey = key }
}

// WithCallSessionID sets a session identifier for a single LLM call.
func WithCallSessionID(id string) CallOption {
	return func(c *CallConfig) { c.SessionID = id }
}

// WithMaxTokens overrides the max output tokens for a single LLM call.
func WithMaxTokens(tokens int) CallOption {
	return func(c *CallConfig) { c.MaxTokens = tokens }
}

// WithToolChoice controls whether the model must call a tool.
// Accepted values: "auto" (default), "required" (must call a tool), "none" (no tools).
func WithToolChoice(choice any) CallOption {
	return func(c *CallConfig) { c.ToolChoice = choice }
}

// WithJSONMode requests valid JSON output without enforcing a specific schema.
func WithJSONMode() CallOption {
	return func(c *CallConfig) {
		c.ResponseFormat = &ResponseFormat{Type: ResponseFormatJSONObject}
	}
}

// WithJSONSchema requests structured JSON output constrained by a JSON Schema.
func WithJSONSchema(name, description string, schema any, strict bool) CallOption {
	return func(c *CallConfig) {
		c.ResponseFormat = &ResponseFormat{
			Type: ResponseFormatJSONSchema,
			JSONSchema: &JSONSchema{
				Name:        name,
				Description: description,
				Schema:      schema,
				Strict:      &strict,
			},
		}
	}
}

// WithResponseFormat sets a provider-native response format explicitly.
func WithResponseFormat(format *ResponseFormat) CallOption {
	return func(c *CallConfig) { c.ResponseFormat = format }
}

// ---------------------------------------------------------------------------
// ChatModel Interface
// ---------------------------------------------------------------------------

// ChatModel is the LLM provider interface.
type ChatModel interface {
	Generate(ctx context.Context, messages []Message, tools []ToolSpec, opts ...CallOption) (*LLMResponse, error)
	GenerateStream(ctx context.Context, messages []Message, tools []ToolSpec, opts ...CallOption) (<-chan StreamEvent, error)
	SupportsTools() bool
}

// ProviderNamer is an optional interface for ChatModel implementations
// to expose their provider name (e.g. "openai", "anthropic", "gemini").
// Used by the agent loop to pass provider context to GetApiKey callbacks.
type ProviderNamer interface {
	ProviderName() string
}

// SwappableModel wraps a ChatModel and allows replacing the underlying model
// at runtime. Swaps take effect on the next call.
type SwappableModel struct {
	mu    sync.RWMutex
	model ChatModel
}

func NewSwappableModel(initial ChatModel) *SwappableModel {
	return &SwappableModel{model: initial}
}

func (m *SwappableModel) Swap(next ChatModel) {
	m.mu.Lock()
	m.model = next
	m.mu.Unlock()
}

func (m *SwappableModel) Current() ChatModel {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.model
}

func (m *SwappableModel) Generate(ctx context.Context, messages []Message, tools []ToolSpec, opts ...CallOption) (*LLMResponse, error) {
	model := m.Current()
	if model == nil {
		return nil, fmt.Errorf("no model configured")
	}
	return model.Generate(ctx, messages, tools, opts...)
}

func (m *SwappableModel) GenerateStream(ctx context.Context, messages []Message, tools []ToolSpec, opts ...CallOption) (<-chan StreamEvent, error) {
	model := m.Current()
	if model == nil {
		return nil, fmt.Errorf("no model configured")
	}
	return model.GenerateStream(ctx, messages, tools, opts...)
}

func (m *SwappableModel) SupportsTools() bool {
	model := m.Current()
	return model != nil && model.SupportsTools()
}

func (m *SwappableModel) ProviderName() string {
	model := m.Current()
	pn, ok := model.(ProviderNamer)
	if !ok {
		return ""
	}
	return pn.ProviderName()
}

// ---------------------------------------------------------------------------
// Stream Events (fine-grained)
// ---------------------------------------------------------------------------

// StreamEventType identifies LLM streaming event types.
type StreamEventType string

const (
	// Text content streaming
	StreamEventTextStart StreamEventType = "text_start"
	StreamEventTextDelta StreamEventType = "text_delta"
	StreamEventTextEnd   StreamEventType = "text_end"

	// Thinking/reasoning streaming
	StreamEventThinkingStart StreamEventType = "thinking_start"
	StreamEventThinkingDelta StreamEventType = "thinking_delta"
	StreamEventThinkingEnd   StreamEventType = "thinking_end"

	// Tool call streaming
	StreamEventToolCallStart StreamEventType = "toolcall_start"
	StreamEventToolCallDelta StreamEventType = "toolcall_delta"
	StreamEventToolCallEnd   StreamEventType = "toolcall_end"

	// Terminal events
	StreamEventDone  StreamEventType = "done"
	StreamEventError StreamEventType = "error"
)

// StreamEvent is a streaming event from the LLM.
type StreamEvent struct {
	Type         StreamEventType
	ContentIndex int     // which content block is being updated
	Delta        string  // text/thinking/toolcall argument delta
	Message      Message // partial (during streaming) or final (done)
	// CompletedToolCall is populated on StreamEventToolCallEnd with the fully
	// reconstructed tool call. It lets the loop start execution immediately
	// without re-parsing the partial assistant message.
	CompletedToolCall *ToolCall
	StopReason        StopReason // finish reason (for done events)
	Err               error      // for error events
}

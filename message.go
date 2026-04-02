package agentcore

import (
	"encoding/json"
	"fmt"
	"strings"
	"time"
)

// ---------------------------------------------------------------------------
// Roles
// ---------------------------------------------------------------------------

// Role defines message roles.
type Role string

const (
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
	RoleSystem    Role = "system"
	RoleTool      Role = "tool"
)

// ---------------------------------------------------------------------------
// Content Blocks
// ---------------------------------------------------------------------------

// ContentType identifies the kind of content in a ContentBlock.
type ContentType string

const (
	ContentText     ContentType = "text"
	ContentThinking ContentType = "thinking"
	ContentToolCall ContentType = "toolCall"
	ContentImage    ContentType = "image"
	ContentToolRef  ContentType = "tool_reference"
)

// ContentBlock is a tagged union for message content.
// Exactly one payload field is populated, matching the Type value.
type ContentBlock struct {
	Type     ContentType `json:"type"`
	Text     string      `json:"text,omitempty"`
	Thinking string      `json:"thinking,omitempty"`
	ToolCall *ToolCall   `json:"tool_call,omitempty"`
	Image    *ImageData  `json:"image,omitempty"`
	ToolName string      `json:"tool_name,omitempty"` // tool_reference: referenced tool name
}

// ImageData holds image content as base64 data or a URL.
// When URL is set, providers pass it directly (no download/encoding needed).
// When Data is set, it is sent as a base64 data URL with MimeType.
// MimeType is required for base64 mode, optional for URL mode (provider infers it).
type ImageData struct {
	Data     string `json:"data,omitempty"`
	URL      string `json:"url,omitempty"`
	MimeType string `json:"mime_type,omitempty"`
}

// Block constructors

func TextBlock(text string) ContentBlock {
	return ContentBlock{Type: ContentText, Text: text}
}

func ThinkingBlock(thinking string) ContentBlock {
	return ContentBlock{Type: ContentThinking, Thinking: thinking}
}

func ToolCallBlock(tc ToolCall) ContentBlock {
	return ContentBlock{Type: ContentToolCall, ToolCall: &tc}
}

func ImageBlock(data, mimeType string) ContentBlock {
	return ContentBlock{Type: ContentImage, Image: &ImageData{Data: data, MimeType: mimeType}}
}

func ImageURLBlock(url string) ContentBlock {
	return ContentBlock{Type: ContentImage, Image: &ImageData{URL: url}}
}

func ToolRefBlock(toolName string) ContentBlock {
	return ContentBlock{Type: ContentToolRef, ToolName: toolName}
}

// ---------------------------------------------------------------------------
// Stop Reason
// ---------------------------------------------------------------------------

// StopReason indicates why the LLM stopped generating.
type StopReason string

const (
	StopReasonStop    StopReason = "stop"
	StopReasonLength  StopReason = "length"
	StopReasonToolUse StopReason = "toolUse"
	StopReasonError   StopReason = "error"
	StopReasonAborted StopReason = "aborted"
)

// ---------------------------------------------------------------------------
// Usage
// ---------------------------------------------------------------------------

// Cost tracks monetary cost for a single LLM call in USD.
type Cost struct {
	Input      float64 `json:"input"`
	Output     float64 `json:"output"`
	CacheRead  float64 `json:"cache_read"`
	CacheWrite float64 `json:"cache_write"`
	Total      float64 `json:"total"`
}

// Add accumulates another Cost into this one (nil-safe).
func (c *Cost) Add(other *Cost) {
	if other == nil {
		return
	}
	c.Input += other.Input
	c.Output += other.Output
	c.CacheRead += other.CacheRead
	c.CacheWrite += other.CacheWrite
	c.Total += other.Total
}

// Usage tracks token consumption for a single LLM call.
//
// Field semantics:
//   - Input: prompt tokens sent to the model (includes cached tokens for some providers)
//   - Output: completion tokens generated (includes reasoning tokens if applicable)
//   - CacheRead: tokens served from prompt cache (Anthropic: cache_read_input_tokens)
//   - CacheWrite: tokens written to prompt cache (Anthropic: cache_creation_input_tokens)
//   - TotalTokens: provider-reported total, typically Input + Output
//   - Cost: monetary cost computed from model pricing (nil if pricing unavailable)
type Usage struct {
	Input       int   `json:"input"`
	Output      int   `json:"output"`
	CacheRead   int   `json:"cache_read"`
	CacheWrite  int   `json:"cache_write"`
	TotalTokens int   `json:"total_tokens"`
	Cost        *Cost `json:"cost,omitempty"`
}

// Add accumulates another Usage into this one (nil-safe).
func (u *Usage) Add(other *Usage) {
	if other == nil {
		return
	}
	u.Input += other.Input
	u.Output += other.Output
	u.CacheRead += other.CacheRead
	u.CacheWrite += other.CacheWrite
	u.TotalTokens += other.TotalTokens
	if other.Cost != nil {
		if u.Cost == nil {
			u.Cost = &Cost{}
		}
		u.Cost.Add(other.Cost)
	}
}

// ---------------------------------------------------------------------------
// Thinking Level
// ---------------------------------------------------------------------------

// ThinkingLevel configures the reasoning depth for models that support it.
type ThinkingLevel string

const (
	ThinkingOff     ThinkingLevel = "off"
	ThinkingMinimal ThinkingLevel = "minimal"
	ThinkingLow     ThinkingLevel = "low"
	ThinkingMedium  ThinkingLevel = "medium"
	ThinkingHigh    ThinkingLevel = "high"
	ThinkingXHigh   ThinkingLevel = "xhigh"
)

// ---------------------------------------------------------------------------
// Messages
// ---------------------------------------------------------------------------

// AgentMessage is the app-layer message abstraction.
// Message implements this interface. Users can define custom types
// (e.g. status notifications, UI hints) that flow through the context
// pipeline but get filtered out by ConvertToLLM.
type AgentMessage interface {
	GetRole() Role
	GetTimestamp() time.Time
	TextContent() string
	ThinkingContent() string
	HasToolCalls() bool
}

// Message is an LLM-level message with structured content blocks.
type Message struct {
	Role       Role           `json:"role"`
	Content    []ContentBlock `json:"content"`
	StopReason StopReason     `json:"stop_reason,omitempty"`
	Usage      *Usage         `json:"usage,omitempty"`
	Metadata   map[string]any `json:"metadata,omitempty"`
	Timestamp  time.Time      `json:"timestamp"`
}

func (m Message) GetRole() Role           { return m.Role }
func (m Message) GetTimestamp() time.Time { return m.Timestamp }

// TextContent returns the concatenated text from all text blocks.
func (m Message) TextContent() string {
	var sb strings.Builder
	for _, b := range m.Content {
		if b.Type == ContentText {
			sb.WriteString(b.Text)
		}
	}
	return sb.String()
}

// ThinkingContent returns the concatenated thinking text.
func (m Message) ThinkingContent() string {
	var sb strings.Builder
	for _, b := range m.Content {
		if b.Type == ContentThinking {
			sb.WriteString(b.Thinking)
		}
	}
	return sb.String()
}

// ToolCalls returns all tool call blocks.
func (m Message) ToolCalls() []ToolCall {
	var calls []ToolCall
	for _, b := range m.Content {
		if b.Type == ContentToolCall && b.ToolCall != nil {
			calls = append(calls, *b.ToolCall)
		}
	}
	return calls
}

// HasToolCalls reports whether any tool call blocks exist.
func (m Message) HasToolCalls() bool {
	for _, b := range m.Content {
		if b.Type == ContentToolCall {
			return true
		}
	}
	return false
}

// IsEmpty reports whether the message has no meaningful content.
func (m Message) IsEmpty() bool {
	return len(m.Content) == 0
}

// ---------------------------------------------------------------------------
// Message Sequence Validation
// ---------------------------------------------------------------------------

type MessageSequenceIssueKind string

const (
	MessageSequenceIssueMissingToolResult MessageSequenceIssueKind = "missing_tool_result"
	MessageSequenceIssueOrphanToolResult  MessageSequenceIssueKind = "orphan_tool_result"
)

// MessageSequenceIssue describes a structural problem in a tool call / tool
// result transcript. The current validator intentionally stays narrow and
// focuses on the two invariants the loop already repairs today:
//   - every tool call should have a following tool result
//   - every tool result should reference a known tool call
type MessageSequenceIssue struct {
	Kind           MessageSequenceIssueKind
	MessageIndex   int
	AssistantIndex int
	ToolCallID     string
	ToolName       string
}

// ValidateMessageSequence reports message-sequence issues that could cause
// provider rejections or inconsistent replay.
func ValidateMessageSequence(msgs []Message) []MessageSequenceIssue {
	issues := make([]MessageSequenceIssue, 0)
	callIDs := make(map[string]bool)

	for i, msg := range msgs {
		if msg.Role != RoleAssistant {
			continue
		}
		calls := msg.ToolCalls()
		if len(calls) == 0 {
			continue
		}

		answered := make(map[string]bool, len(calls))
		for j := i + 1; j < len(msgs); j++ {
			next := msgs[j]
			if next.Role != RoleTool {
				break
			}
			if id, ok := next.Metadata["tool_call_id"].(string); ok {
				answered[id] = true
			}
		}

		for _, call := range calls {
			callIDs[call.ID] = true
			if !answered[call.ID] {
				issues = append(issues, MessageSequenceIssue{
					Kind:           MessageSequenceIssueMissingToolResult,
					MessageIndex:   i,
					AssistantIndex: i,
					ToolCallID:     call.ID,
					ToolName:       call.Name,
				})
			}
		}
	}

	for i, msg := range msgs {
		if msg.Role != RoleTool {
			continue
		}
		id, _ := msg.Metadata["tool_call_id"].(string)
		if id == "" || callIDs[id] {
			continue
		}
		issues = append(issues, MessageSequenceIssue{
			Kind:         MessageSequenceIssueOrphanToolResult,
			MessageIndex: i,
			ToolCallID:   id,
		})
	}

	return issues
}

// AssertMessageSequence returns an error when the transcript would require
// synthetic repair before being sent to an LLM provider.
func AssertMessageSequence(msgs []Message) error {
	issues := ValidateMessageSequence(msgs)
	if len(issues) == 0 {
		return nil
	}
	return fmt.Errorf("invalid message sequence: %s", formatMessageSequenceIssues(issues))
}

func formatMessageSequenceIssues(issues []MessageSequenceIssue) string {
	parts := make([]string, 0, len(issues))
	for _, issue := range issues {
		switch issue.Kind {
		case MessageSequenceIssueMissingToolResult:
			parts = append(parts, fmt.Sprintf("missing tool result for %q (%s) at message %d", issue.ToolCallID, issue.ToolName, issue.MessageIndex))
		case MessageSequenceIssueOrphanToolResult:
			parts = append(parts, fmt.Sprintf("orphan tool result for %q at message %d", issue.ToolCallID, issue.MessageIndex))
		default:
			parts = append(parts, fmt.Sprintf("%s at message %d", issue.Kind, issue.MessageIndex))
		}
	}
	return strings.Join(parts, "; ")
}

// RepairMessageSequence ensures tool call / tool result pairs are complete.
// Orphaned tool calls (no matching result) get a synthetic error result inserted.
// Orphaned tool results (no matching call) are removed.
// This prevents LLM providers from rejecting malformed message sequences.
func RepairMessageSequence(msgs []Message) []Message {
	out := make([]Message, 0, len(msgs))

	for i, msg := range msgs {
		out = append(out, msg)

		if msg.Role != RoleAssistant {
			continue
		}
		calls := msg.ToolCalls()
		if len(calls) == 0 {
			continue
		}

		// Collect tool result IDs that follow this assistant message.
		answered := make(map[string]bool, len(calls))
		for j := i + 1; j < len(msgs); j++ {
			next := msgs[j]
			if next.Role == RoleTool {
				if id, ok := next.Metadata["tool_call_id"].(string); ok {
					answered[id] = true
				}
				continue
			}
			break
		}

		// Insert synthetic results for unanswered tool calls.
		for _, call := range calls {
			if !answered[call.ID] {
				out = append(out, ToolResultMsg(call.ID, []byte(`"Tool result missing (conversation was truncated or interrupted)."`), true))
			}
		}
	}

	// Remove orphaned tool results (no matching call).
	callIDs := make(map[string]bool)
	for _, msg := range out {
		for _, call := range msg.ToolCalls() {
			callIDs[call.ID] = true
		}
	}

	cleaned := make([]Message, 0, len(out))
	for _, msg := range out {
		if msg.Role == RoleTool {
			if id, ok := msg.Metadata["tool_call_id"].(string); ok && !callIDs[id] {
				continue
			}
		}
		cleaned = append(cleaned, msg)
	}

	return cleaned
}

// ---------------------------------------------------------------------------
// Message Serialization Helpers
// ---------------------------------------------------------------------------

// CollectMessages extracts concrete Messages from an AgentMessage slice,
// dropping custom types. Use this to serialize conversation history.
func CollectMessages(msgs []AgentMessage) []Message {
	out := make([]Message, 0, len(msgs))
	for _, m := range msgs {
		if msg, ok := m.(Message); ok {
			out = append(out, msg)
		}
	}
	return out
}

// ToAgentMessages converts a Message slice to AgentMessage slice.
// Use this to restore conversation history from deserialized Messages.
func ToAgentMessages(msgs []Message) []AgentMessage {
	out := make([]AgentMessage, len(msgs))
	for i, m := range msgs {
		out[i] = m
	}
	return out
}

// ---------------------------------------------------------------------------
// Message Constructors
// ---------------------------------------------------------------------------

// UserMsg creates a user message from plain text.
func UserMsg(text string) Message {
	return Message{
		Role:      RoleUser,
		Content:   []ContentBlock{TextBlock(text)},
		Timestamp: time.Now(),
	}
}

// SystemMsg creates a system message.
func SystemMsg(text string) Message {
	return Message{
		Role:      RoleSystem,
		Content:   []ContentBlock{TextBlock(text)},
		Timestamp: time.Now(),
	}
}

// ToolResultMsg creates a tool result message.
func ToolResultMsg(toolCallID string, content json.RawMessage, isError bool) Message {
	return Message{
		Role:    RoleTool,
		Content: []ContentBlock{TextBlock(string(content))},
		Metadata: map[string]any{
			"tool_call_id": toolCallID,
			"is_error":     isError,
		},
		Timestamp: time.Now(),
	}
}

// AbortMsg creates an assistant abort marker message.
// phase is "inference" or "tool_execution".
func AbortMsg(text, phase string) Message {
	return Message{
		Role:       RoleAssistant,
		Content:    []ContentBlock{TextBlock(text)},
		StopReason: StopReasonAborted,
		Metadata:   map[string]any{"abort_phase": phase},
		Timestamp:  time.Now(),
	}
}

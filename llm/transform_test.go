package llm

import (
	"encoding/json"
	"testing"

	"github.com/voocel/agentcore"
)

func TestTransformMessages_ThinkingBlocks_Anthropic(t *testing.T) {
	msgs := []agentcore.Message{
		{
			Role: agentcore.RoleAssistant,
			Content: []agentcore.ContentBlock{
				agentcore.ThinkingBlock("let me think..."),
				agentcore.TextBlock("hello"),
			},
			StopReason: agentcore.StopReasonStop,
		},
	}

	result := TransformMessages(msgs, "anthropic")
	if len(result) != 1 {
		t.Fatalf("expected 1 message, got %d", len(result))
	}
	// Anthropic: thinking blocks preserved as-is
	if result[0].Content[0].Type != agentcore.ContentThinking {
		t.Fatalf("expected thinking block preserved, got %s", result[0].Content[0].Type)
	}
}

func TestTransformMessages_ThinkingBlocks_NonAnthropic(t *testing.T) {
	msgs := []agentcore.Message{
		{
			Role: agentcore.RoleAssistant,
			Content: []agentcore.ContentBlock{
				agentcore.ThinkingBlock("reasoning here"),
				agentcore.TextBlock("answer"),
			},
			StopReason: agentcore.StopReasonStop,
		},
	}

	result := TransformMessages(msgs, "openai")
	if len(result) != 1 {
		t.Fatalf("expected 1 message, got %d", len(result))
	}
	// Non-Anthropic: thinking converted to wrapped text
	if result[0].Content[0].Type != agentcore.ContentText {
		t.Fatalf("expected text block, got %s", result[0].Content[0].Type)
	}
	if result[0].Content[0].Text != "<thinking>\nreasoning here\n</thinking>" {
		t.Fatalf("unexpected text: %s", result[0].Content[0].Text)
	}
}

func TestTransformMessages_EmptyThinkingDropped(t *testing.T) {
	msgs := []agentcore.Message{
		{
			Role: agentcore.RoleAssistant,
			Content: []agentcore.ContentBlock{
				agentcore.ThinkingBlock("   "),
				agentcore.TextBlock("answer"),
			},
			StopReason: agentcore.StopReasonStop,
		},
	}

	result := TransformMessages(msgs, "anthropic")
	if len(result[0].Content) != 1 {
		t.Fatalf("expected 1 block (empty thinking dropped), got %d", len(result[0].Content))
	}
}

func TestTransformMessages_SkipErroredMessages(t *testing.T) {
	msgs := []agentcore.Message{
		agentcore.UserMsg("hi"),
		{Role: agentcore.RoleAssistant, Content: []agentcore.ContentBlock{agentcore.TextBlock("oops")}, StopReason: agentcore.StopReasonError},
		{Role: agentcore.RoleAssistant, Content: []agentcore.ContentBlock{agentcore.TextBlock("ok")}, StopReason: agentcore.StopReasonStop},
	}

	result := TransformMessages(msgs, "openai")
	// Errored message should be skipped
	if len(result) != 2 {
		t.Fatalf("expected 2 messages (errored skipped), got %d", len(result))
	}
}

func TestTransformMessages_NormalizeToolCallID(t *testing.T) {
	longID := "call_" + string(make([]byte, 100)) // will contain null bytes → sanitized
	msgs := []agentcore.Message{
		{
			Role: agentcore.RoleAssistant,
			Content: []agentcore.ContentBlock{
				agentcore.ToolCallBlock(agentcore.ToolCall{ID: longID, Name: "test", Args: json.RawMessage(`{}`)}),
			},
			StopReason: agentcore.StopReasonToolUse,
		},
		agentcore.ToolResultMsg(longID, json.RawMessage(`"ok"`), false),
	}

	result := TransformMessages(msgs, "openai")
	// Tool call ID should be sanitized
	calls := result[0].ToolCalls()
	if len(calls) == 0 {
		t.Fatal("expected tool calls")
	}
	if calls[0].ID == longID {
		t.Fatal("expected ID to be normalized")
	}
	if len(calls[0].ID) > maxToolCallIDLength {
		t.Fatalf("ID too long: %d", len(calls[0].ID))
	}
}

func TestTransformMessages_IDRemapping(t *testing.T) {
	// Use an ID with special chars that will be sanitized
	badID := "call@#$123"
	msgs := []agentcore.Message{
		{
			Role: agentcore.RoleAssistant,
			Content: []agentcore.ContentBlock{
				agentcore.ToolCallBlock(agentcore.ToolCall{ID: badID, Name: "test", Args: json.RawMessage(`{}`)}),
			},
			StopReason: agentcore.StopReasonToolUse,
		},
		agentcore.ToolResultMsg(badID, json.RawMessage(`"result"`), false),
	}

	result := TransformMessages(msgs, "openai")

	// Tool result ID should be remapped to match sanitized call ID
	calls := result[0].ToolCalls()
	resultID, _ := result[1].Metadata["tool_call_id"].(string)
	if calls[0].ID != resultID {
		t.Fatalf("ID mismatch: call=%s result=%s", calls[0].ID, resultID)
	}
}

func TestTransformMessages_OrphanedToolCall(t *testing.T) {
	msgs := []agentcore.Message{
		{
			Role: agentcore.RoleAssistant,
			Content: []agentcore.ContentBlock{
				agentcore.ToolCallBlock(agentcore.ToolCall{ID: "tc1", Name: "test", Args: json.RawMessage(`{}`)}),
			},
			StopReason: agentcore.StopReasonToolUse,
		},
		// No tool result for tc1
	}

	result := TransformMessages(msgs, "openai")
	// Should insert synthetic result
	if len(result) != 2 {
		t.Fatalf("expected 2 messages (synthetic result), got %d", len(result))
	}
	if result[1].Role != agentcore.RoleTool {
		t.Fatalf("expected tool result, got %s", result[1].Role)
	}
}

func TestTransformMessages_Empty(t *testing.T) {
	result := TransformMessages(nil, "openai")
	if result != nil {
		t.Fatalf("expected nil for empty input, got %v", result)
	}
}

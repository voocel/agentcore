package agentcore

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"
)

// simpleAgent creates a SubAgentConfig that always replies with the given text.
func simpleAgent(name, reply string) SubAgentConfig {
	return SubAgentConfig{
		Name:        name,
		Description: name + " agent",
		Model: mockModel(Message{
			Role:       RoleAssistant,
			Content:    []ContentBlock{TextBlock(reply)},
			StopReason: StopReasonStop,
		}),
		MaxTurns: 3,
	}
}

func parseResult(t *testing.T, raw json.RawMessage) map[string]any {
	t.Helper()
	var out map[string]any
	if err := json.Unmarshal(raw, &out); err != nil {
		t.Fatalf("failed to parse result: %v", err)
	}
	return out
}

func TestSubAgentTool_Single(t *testing.T) {
	tool := NewSubAgentTool(simpleAgent("writer", "hello"))
	result, err := tool.Execute(context.Background(), json.RawMessage(`{"agent":"writer","task":"greet"}`))
	if err != nil {
		t.Fatal(err)
	}
	out := parseResult(t, result)
	if out["output"] != "hello" {
		t.Fatalf("expected 'hello', got %v", out["output"])
	}
}

func TestSubAgentTool_UnknownAgent(t *testing.T) {
	tool := NewSubAgentTool(simpleAgent("writer", "x"))
	_, err := tool.Execute(context.Background(), json.RawMessage(`{"agent":"unknown","task":"hi"}`))
	if err == nil || !strings.Contains(err.Error(), "unknown agent") {
		t.Fatalf("expected unknown agent error, got %v", err)
	}
}

func TestSubAgentTool_SinglePropagatesFinalErrorAfterPartialOutput(t *testing.T) {
	noop := NewFuncTool("noop", "noop", map[string]any{
		"type": "object", "properties": map[string]any{},
	}, func(ctx context.Context, args json.RawMessage) (json.RawMessage, error) {
		return json.Marshal(map[string]bool{"ok": true})
	})

	cfg := SubAgentConfig{
		Name:        "writer",
		Description: "writer agent",
		Tools:       []Tool{noop},
		Model: sequentialModel(func(i int, req *LLMRequest) (*LLMResponse, error) {
			if i == 0 {
				return &LLMResponse{Message: Message{
					Role: RoleAssistant,
					Content: []ContentBlock{
						TextBlock("partial output before failure"),
						ToolCallBlock(ToolCall{ID: "tc1", Name: "noop", Args: json.RawMessage(`{}`)}),
					},
					StopReason: StopReasonToolUse,
				}}, nil
			}
			return nil, errors.New("llm failed after partial output")
		}),
		MaxTurns: 3,
	}

	tool := NewSubAgentTool(cfg)
	result, err := tool.Execute(context.Background(), json.RawMessage(`{"agent":"writer","task":"write"}`))
	if err == nil {
		t.Fatalf("expected final LLM error to propagate, got result %s", string(result))
	}
	if !strings.Contains(err.Error(), "llm failed after partial output") {
		t.Fatalf("expected original error in message, got %v", err)
	}
}

func TestSubAgentTool_Chain(t *testing.T) {
	tool := NewSubAgentTool(
		simpleAgent("step1", "first-output"),
		simpleAgent("step2", "final-output"),
	)
	args := `{"chain":[{"agent":"step1","task":"do A"},{"agent":"step2","task":"continue from {previous}"}]}`
	result, err := tool.Execute(context.Background(), json.RawMessage(args))
	if err != nil {
		t.Fatal(err)
	}
	out := parseResult(t, result)
	if out["output"] != "final-output" {
		t.Fatalf("expected last chain output, got %v", out["output"])
	}
	// Verify chain produced correct number of steps
	results, _ := out["results"].([]any)
	if len(results) != 2 {
		t.Fatalf("expected 2 chain results, got %d", len(results))
	}
}

func TestSubAgentTool_Parallel(t *testing.T) {
	tool := NewSubAgentTool(
		simpleAgent("a", "result-a"),
		simpleAgent("b", "result-b"),
	)
	args := `{"tasks":[{"agent":"a","task":"t1"},{"agent":"b","task":"t2"}]}`
	result, err := tool.Execute(context.Background(), json.RawMessage(args))
	if err != nil {
		t.Fatal(err)
	}
	out := parseResult(t, result)
	if out["summary"] != "2/2 succeeded" {
		t.Fatalf("expected 2/2 succeeded, got %v", out["summary"])
	}
}

func TestSubAgentTool_ModeValidation(t *testing.T) {
	tool := NewSubAgentTool(simpleAgent("x", "y"))

	// No mode
	result, err := tool.Execute(context.Background(), json.RawMessage(`{}`))
	if err != nil {
		t.Fatal(err)
	}
	var msg string
	if err := json.Unmarshal(result, &msg); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(msg, "exactly one mode") {
		t.Fatalf("expected mode validation error, got %q", msg)
	}

	// Multiple modes
	result, err = tool.Execute(context.Background(), json.RawMessage(`{"agent":"x","task":"t","tasks":[{"agent":"x","task":"t"}]}`))
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(result, &msg); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(msg, "exactly one mode") {
		t.Fatalf("expected mode validation error, got %q", msg)
	}
}

func TestSubAgentTool_ModelOverrideRebuildsContextManager(t *testing.T) {
	baseModel := &fakeNamedModel{name: "base"}
	overrideModel := &fakeNamedModel{name: "override"}

	var received string
	cfg := SubAgentConfig{
		Name:        "writer",
		Description: "writer agent",
		Model:       baseModel,
		ContextManagerFactory: func(model ChatModel) ContextManager {
			if named, ok := model.(*fakeNamedModel); ok {
				received = named.name
			}
			return nil
		},
		MaxTurns: 3,
	}

	tool := NewSubAgentTool(cfg)
	tool.SetCreateModel(func(name string) (ChatModel, error) {
		return overrideModel, nil
	})

	if _, err := tool.Execute(context.Background(), json.RawMessage(`{"agent":"writer","task":"greet","model":"override"}`)); err != nil {
		t.Fatal(err)
	}
	if received != "override" {
		t.Fatalf("expected context manager factory to receive override model, got %q", received)
	}
}

type fakeNamedModel struct {
	name string
}

func (m *fakeNamedModel) Generate(ctx context.Context, messages []Message, tools []ToolSpec, opts ...CallOption) (*LLMResponse, error) {
	return &LLMResponse{Message: Message{Role: RoleAssistant, Content: []ContentBlock{TextBlock(m.name)}}}, nil
}

func (m *fakeNamedModel) GenerateStream(ctx context.Context, messages []Message, tools []ToolSpec, opts ...CallOption) (<-chan StreamEvent, error) {
	msg := Message{Role: RoleAssistant, Content: []ContentBlock{TextBlock(m.name)}, StopReason: StopReasonStop}
	ch := make(chan StreamEvent, 1)
	ch <- StreamEvent{Type: StreamEventDone, Message: msg, StopReason: StopReasonStop}
	close(ch)
	return ch, nil
}

func (m *fakeNamedModel) SupportsTools() bool { return true }

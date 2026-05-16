package subagent

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/voocel/agentcore"
)

// mockModel returns the responses one at a time. Generate and GenerateStream
// each consume from the same cursor independently (one call advances exactly
// once, never both).
type mockModel struct {
	responses []agentcore.Message
	idx       int64
}

func newMock(responses ...agentcore.Message) *mockModel {
	return &mockModel{responses: responses}
}

func (m *mockModel) take() (agentcore.Message, error) {
	i := int(atomic.AddInt64(&m.idx, 1) - 1)
	if i >= len(m.responses) {
		return agentcore.Message{}, errors.New("mock model: no more responses")
	}
	return m.responses[i], nil
}

func (m *mockModel) Generate(ctx context.Context, _ []agentcore.Message, _ []agentcore.ToolSpec, _ ...agentcore.CallOption) (*agentcore.LLMResponse, error) {
	msg, err := m.take()
	if err != nil {
		return nil, err
	}
	return &agentcore.LLMResponse{Message: msg}, nil
}

func (m *mockModel) GenerateStream(ctx context.Context, _ []agentcore.Message, _ []agentcore.ToolSpec, _ ...agentcore.CallOption) (<-chan agentcore.StreamEvent, error) {
	msg, err := m.take()
	if err != nil {
		return nil, err
	}
	ch := make(chan agentcore.StreamEvent, 1)
	ch <- agentcore.StreamEvent{Type: agentcore.StreamEventDone, Message: msg, StopReason: msg.StopReason}
	close(ch)
	return ch, nil
}

func (m *mockModel) SupportsTools() bool { return true }

// sequentialModel calls fn(i, req) on every Generate/GenerateStream call.
type sequentialModel struct {
	fn  func(i int, req *agentcore.LLMRequest) (*agentcore.LLMResponse, error)
	idx int64
}

func newSequential(fn func(i int, req *agentcore.LLMRequest) (*agentcore.LLMResponse, error)) *sequentialModel {
	return &sequentialModel{fn: fn}
}

func (m *sequentialModel) Generate(ctx context.Context, messages []agentcore.Message, tools []agentcore.ToolSpec, _ ...agentcore.CallOption) (*agentcore.LLMResponse, error) {
	i := int(atomic.AddInt64(&m.idx, 1) - 1)
	return m.fn(i, &agentcore.LLMRequest{Messages: messages, Tools: tools})
}

func (m *sequentialModel) GenerateStream(ctx context.Context, messages []agentcore.Message, tools []agentcore.ToolSpec, _ ...agentcore.CallOption) (<-chan agentcore.StreamEvent, error) {
	i := int(atomic.AddInt64(&m.idx, 1) - 1)
	resp, err := m.fn(i, &agentcore.LLMRequest{Messages: messages, Tools: tools})
	if err != nil {
		return nil, err
	}
	ch := make(chan agentcore.StreamEvent, 1)
	ch <- agentcore.StreamEvent{Type: agentcore.StreamEventDone, Message: resp.Message, StopReason: resp.Message.StopReason}
	close(ch)
	return ch, nil
}

func (m *sequentialModel) SupportsTools() bool { return true }

// simpleAgent creates a Config that always replies with the given text.
func simpleAgent(name, reply string) Config {
	return Config{
		Name:        name,
		Description: name + " agent",
		Model: newMock(agentcore.Message{
			Role:       agentcore.RoleAssistant,
			Content:    []agentcore.ContentBlock{agentcore.TextBlock(reply)},
			StopReason: agentcore.StopReasonStop,
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

func TestTool_Single(t *testing.T) {
	tool := New(simpleAgent("writer", "hello"))
	result, err := tool.Execute(context.Background(), json.RawMessage(`{"agent":"writer","task":"greet"}`))
	if err != nil {
		t.Fatal(err)
	}
	out := parseResult(t, result)
	if out["output"] != "hello" {
		t.Fatalf("expected 'hello', got %v", out["output"])
	}
}

func TestTool_UnknownAgent(t *testing.T) {
	tool := New(simpleAgent("writer", "x"))
	_, err := tool.Execute(context.Background(), json.RawMessage(`{"agent":"unknown","task":"hi"}`))
	if err == nil || !strings.Contains(err.Error(), "unknown agent") {
		t.Fatalf("expected unknown agent error, got %v", err)
	}
}

// Background=true without a wired TaskRuntime must fail fast — silent
// degradation to synchronous execution would violate the "return immediately,
// notify on completion" contract callers expect from background mode.
func TestTool_BackgroundRequiresTaskRuntime(t *testing.T) {
	tool := New(simpleAgent("writer", "x"))
	_, err := tool.Execute(context.Background(), json.RawMessage(`{"agent":"writer","task":"go","background":true}`))
	if err == nil {
		t.Fatal("expected error when background=true and TaskRuntime is missing, got nil")
	}
	if !strings.Contains(err.Error(), "TaskRuntime") {
		t.Fatalf("expected error mentioning TaskRuntime, got %v", err)
	}
}

func TestTool_SinglePropagatesFinalErrorAfterPartialOutput(t *testing.T) {
	noop := agentcore.NewFuncTool("noop", "noop", map[string]any{
		"type": "object", "properties": map[string]any{},
	}, func(ctx context.Context, args json.RawMessage) (json.RawMessage, error) {
		return json.Marshal(map[string]bool{"ok": true})
	})

	cfg := Config{
		Name:        "writer",
		Description: "writer agent",
		Tools:       []agentcore.Tool{noop},
		Model: newSequential(func(i int, req *agentcore.LLMRequest) (*agentcore.LLMResponse, error) {
			if i == 0 {
				return &agentcore.LLMResponse{Message: agentcore.Message{
					Role: agentcore.RoleAssistant,
					Content: []agentcore.ContentBlock{
						agentcore.TextBlock("partial output before failure"),
						agentcore.ToolCallBlock(agentcore.ToolCall{ID: "tc1", Name: "noop", Args: json.RawMessage(`{}`)}),
					},
					StopReason: agentcore.StopReasonToolUse,
				}}, nil
			}
			return nil, errors.New("llm failed after partial output")
		}),
		MaxTurns: 3,
	}

	tool := New(cfg)
	result, err := tool.Execute(context.Background(), json.RawMessage(`{"agent":"writer","task":"write"}`))
	if err == nil {
		t.Fatalf("expected final LLM error to propagate, got result %s", string(result))
	}
	if !strings.Contains(err.Error(), "llm failed after partial output") {
		t.Fatalf("expected original error in message, got %v", err)
	}
}

func TestTool_Chain(t *testing.T) {
	tool := New(
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
	results, _ := out["results"].([]any)
	if len(results) != 2 {
		t.Fatalf("expected 2 chain results, got %d", len(results))
	}
}

func TestTool_Parallel(t *testing.T) {
	tool := New(
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

func TestTool_ModeValidation(t *testing.T) {
	tool := New(simpleAgent("x", "y"))

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

func TestTool_ModelOverrideRebuildsContextManager(t *testing.T) {
	baseModel := &fakeNamedModel{name: "base"}
	overrideModel := &fakeNamedModel{name: "override"}

	var received string
	cfg := Config{
		Name:        "writer",
		Description: "writer agent",
		Model:       baseModel,
		ContextManagerFactory: func(model agentcore.ChatModel) agentcore.ContextManager {
			if named, ok := model.(*fakeNamedModel); ok {
				received = named.name
			}
			return nil
		},
		MaxTurns: 3,
	}

	tool := New(cfg)
	tool.SetCreateModel(func(name string) (agentcore.ChatModel, error) {
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

func (m *fakeNamedModel) Generate(ctx context.Context, messages []agentcore.Message, tools []agentcore.ToolSpec, _ ...agentcore.CallOption) (*agentcore.LLMResponse, error) {
	return &agentcore.LLMResponse{Message: agentcore.Message{Role: agentcore.RoleAssistant, Content: []agentcore.ContentBlock{agentcore.TextBlock(m.name)}}}, nil
}

func (m *fakeNamedModel) GenerateStream(ctx context.Context, messages []agentcore.Message, tools []agentcore.ToolSpec, _ ...agentcore.CallOption) (<-chan agentcore.StreamEvent, error) {
	msg := agentcore.Message{Role: agentcore.RoleAssistant, Content: []agentcore.ContentBlock{agentcore.TextBlock(m.name)}, StopReason: agentcore.StopReasonStop}
	ch := make(chan agentcore.StreamEvent, 1)
	ch <- agentcore.StreamEvent{Type: agentcore.StreamEventDone, Message: msg, StopReason: agentcore.StopReasonStop}
	close(ch)
	return ch, nil
}

func (m *fakeNamedModel) SupportsTools() bool { return true }

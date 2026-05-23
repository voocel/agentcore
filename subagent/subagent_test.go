package subagent

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/voocel/agentcore"
	"github.com/voocel/agentcore/task"
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

// Background spawn must refuse to nest beyond task.MaxAgentDepth so a future
// peer-spawn channel (added with team support) can't trigger runaway
// recursion. We simulate "the caller is already at depth N" by threading the
// depth into ctx, then asserting the spawn either succeeds with childDepth=N+1
// or rejects when N+1 > MaxAgentDepth.
func TestTool_BackgroundRespectsMaxAgentDepth(t *testing.T) {
	tool := New(simpleAgent("writer", "ok"))
	tool.SetTaskRuntime(task.NewRuntime())

	cases := []struct {
		callerDepth  int
		wantError    bool
		wantBgDepth  int // entry.Depth on success path
	}{
		{callerDepth: 0, wantError: false, wantBgDepth: 1},                     // main agent
		{callerDepth: task.MaxAgentDepth - 1, wantError: false, wantBgDepth: 5}, // last legal level
		{callerDepth: task.MaxAgentDepth, wantError: true},                      // childDepth = 6, rejected
		{callerDepth: task.MaxAgentDepth + 5, wantError: true},                  // way past
	}

	for _, tc := range cases {
		ctx := task.WithDepth(context.Background(), tc.callerDepth)
		raw, err := tool.Execute(ctx, json.RawMessage(`{"agent":"writer","task":"go","background":true}`))
		if err != nil {
			t.Fatalf("callerDepth=%d: unexpected execute error: %v", tc.callerDepth, err)
		}
		var resp map[string]any
		if err := json.Unmarshal(raw, &resp); err != nil {
			t.Fatalf("callerDepth=%d: parse: %v (%s)", tc.callerDepth, err, raw)
		}
		if tc.wantError {
			errMsg, _ := resp["error"].(string)
			if !strings.Contains(errMsg, "depth") {
				t.Errorf("callerDepth=%d: want depth error, got %v", tc.callerDepth, resp)
			}
			continue
		}
		taskID, _ := resp["task_id"].(string)
		if taskID == "" {
			t.Fatalf("callerDepth=%d: missing task_id in success response: %v", tc.callerDepth, resp)
		}
		// Wait briefly for the registered entry to appear with its depth set.
		// Registration is synchronous inside executeBackground, so a single
		// Get() should suffice — but guard with a short retry to make the
		// test resilient to scheduler timing.
		var entry *task.Entry
		for range 20 {
			if e := tool.taskRT.Get(taskID); e != nil {
				entry = e
				break
			}
		}
		if entry == nil {
			t.Fatalf("callerDepth=%d: entry never appeared in runtime", tc.callerDepth)
		}
		if entry.Depth != tc.wantBgDepth {
			t.Errorf("callerDepth=%d: entry.Depth = %d, want %d", tc.callerDepth, entry.Depth, tc.wantBgDepth)
		}
	}
}

// Verifies the parent→child wiring end-to-end: a message queued via
// task.Runtime.AppendPending while the background sub-agent is running must
// reach the sub-agent's next LLM call. The chain under test is:
//
//	AppendPending → Runtime.pendingMessages
//	             → loopCfg.GetSteeringMessages (bound in runAgent)
//	             → injected as UserMsg before turn 2
//
// If any link breaks, the second LLM call won't see "follow-up steered".
func TestTool_BackgroundDrainsPendingMessagesIntoNextTurn(t *testing.T) {
	rt := task.NewRuntime()

	// The injecting tool needs to know its own task ID to call AppendPending.
	// We hand it off via a buffered channel — the main goroutine writes after
	// Execute returns; the tool reads when the first turn invokes it.
	taskIDCh := make(chan string, 1)
	const steeringMsg = "follow-up steered"

	injectTool := agentcore.NewFuncTool("inject", "queues a pending message", map[string]any{
		"type": "object", "properties": map[string]any{},
	}, func(ctx context.Context, _ json.RawMessage) (json.RawMessage, error) {
		select {
		case id := <-taskIDCh:
			rt.AppendPending(id, steeringMsg)
			taskIDCh <- id // re-fill for any subsequent tool call
		case <-time.After(time.Second):
			return nil, errors.New("taskID channel never received")
		}
		return json.Marshal("ok")
	})

	var sawSteering atomic.Bool
	cfg := Config{
		Name:        "writer",
		Description: "writer",
		Tools:       []agentcore.Tool{injectTool},
		Model: newSequential(func(i int, req *agentcore.LLMRequest) (*agentcore.LLMResponse, error) {
			switch i {
			case 0:
				return &agentcore.LLMResponse{Message: agentcore.Message{
					Role: agentcore.RoleAssistant,
					Content: []agentcore.ContentBlock{
						agentcore.ToolCallBlock(agentcore.ToolCall{
							ID: "tc1", Name: "inject", Args: json.RawMessage(`{}`),
						}),
					},
					StopReason: agentcore.StopReasonToolUse,
				}}, nil
			default:
				// Inspect the prompt at the second LLM call: the steering
				// message should have been injected before this turn.
				for _, msg := range req.Messages {
					if msg.Role == agentcore.RoleUser && strings.Contains(msg.TextContent(), steeringMsg) {
						sawSteering.Store(true)
						break
					}
				}
				return &agentcore.LLMResponse{Message: agentcore.Message{
					Role:       agentcore.RoleAssistant,
					Content:    []agentcore.ContentBlock{agentcore.TextBlock("done")},
					StopReason: agentcore.StopReasonStop,
				}}, nil
			}
		}),
		MaxTurns: 5,
	}

	tool := New(cfg)
	tool.SetTaskRuntime(rt)

	raw, err := tool.Execute(context.Background(), json.RawMessage(`{"agent":"writer","task":"start","background":true}`))
	if err != nil {
		t.Fatal(err)
	}
	var resp map[string]any
	if err := json.Unmarshal(raw, &resp); err != nil {
		t.Fatalf("parse background response: %v", err)
	}
	taskID, _ := resp["task_id"].(string)
	if taskID == "" {
		t.Fatalf("missing task_id in background response: %s", string(raw))
	}
	taskIDCh <- taskID

	// Wait for the background goroutine to reach a terminal state.
	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		if e := rt.Get(taskID); e != nil && e.Status.IsTerminal() {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}

	if e := rt.Get(taskID); e == nil || !e.Status.IsTerminal() {
		t.Fatalf("background task did not finish in time: %+v", e)
	}
	if !sawSteering.Load() {
		t.Fatal("steering message was not injected into the second LLM call — wiring is broken")
	}
}

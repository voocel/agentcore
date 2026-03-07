package agentcore

import (
	"context"
	"encoding/json"
	"fmt"
	"sync/atomic"
	"testing"
)

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// mockStreamFn creates a StreamFn that returns responses in order.
func mockStreamFn(responses ...Message) StreamFn {
	var idx int64
	return func(ctx context.Context, req *LLMRequest) (*LLMResponse, error) {
		i := int(atomic.AddInt64(&idx, 1) - 1)
		if i >= len(responses) {
			return nil, fmt.Errorf("unexpected LLM call #%d (only %d responses provided)", i, len(responses))
		}
		return &LLMResponse{Message: responses[i]}, nil
	}
}

// sequentialStreamFn creates a StreamFn with per-call logic via switch/case.
func sequentialStreamFn(fn func(i int, req *LLMRequest) (*LLMResponse, error)) StreamFn {
	var idx int64
	return func(ctx context.Context, req *LLMRequest) (*LLMResponse, error) {
		i := int(atomic.AddInt64(&idx, 1) - 1)
		return fn(i, req)
	}
}

func collectEvents(ch <-chan Event) []Event {
	var events []Event
	for ev := range ch {
		events = append(events, ev)
	}
	return events
}

func requireEvent(t *testing.T, events []Event, et EventType) {
	t.Helper()
	for _, ev := range events {
		if ev.Type == et {
			return
		}
	}
	t.Fatalf("missing expected event: %s", et)
}

func countEvent(events []Event, et EventType) int {
	n := 0
	for _, ev := range events {
		if ev.Type == et {
			n++
		}
	}
	return n
}

func findEvent(events []Event, et EventType) (Event, bool) {
	for _, ev := range events {
		if ev.Type == et {
			return ev, true
		}
	}
	return Event{}, false
}

func echoTool(calls *[]string) Tool {
	return NewFuncTool("echo", "echoes input", map[string]any{
		"type": "object",
		"properties": map[string]any{
			"value": map[string]any{"type": "string"},
		},
		"required": []string{"value"},
	}, func(ctx context.Context, args json.RawMessage) (json.RawMessage, error) {
		var p struct{ Value string }
		json.Unmarshal(args, &p)
		*calls = append(*calls, p.Value)
		return json.Marshal(fmt.Sprintf("echoed: %s", p.Value))
	})
}

type previewProgressTool struct {
	previewCalls *int64
}

func (t *previewProgressTool) Name() string        { return "preview_tool" }
func (t *previewProgressTool) Description() string { return "tool with preview and progress updates" }
func (t *previewProgressTool) Schema() map[string]any {
	return map[string]any{
		"type": "object",
		"properties": map[string]any{
			"x": map[string]any{"type": "string"},
		},
		"required": []string{"x"},
	}
}
func (t *previewProgressTool) Preview(ctx context.Context, args json.RawMessage) (json.RawMessage, error) {
	atomic.AddInt64(t.previewCalls, 1)
	return json.RawMessage(`"preview"`), nil
}
func (t *previewProgressTool) Execute(ctx context.Context, args json.RawMessage) (json.RawMessage, error) {
	ReportToolProgress(ctx, json.RawMessage(`"progress"`))
	return json.RawMessage(`"ok"`), nil
}

func assistantMsg(text string, stop StopReason) Message {
	return Message{
		Role:       RoleAssistant,
		Content:    []ContentBlock{TextBlock(text)},
		StopReason: stop,
		Usage:      &Usage{Input: 10, Output: 5},
	}
}

func toolCallMsg(calls ...ToolCall) Message {
	blocks := make([]ContentBlock, len(calls))
	for i, c := range calls {
		blocks[i] = ToolCallBlock(c)
	}
	return Message{
		Role:       RoleAssistant,
		Content:    blocks,
		StopReason: StopReasonToolUse,
		Usage:      &Usage{Input: 10, Output: 5},
	}
}

func runTestLoop(t *testing.T, msgs []AgentMessage, actx AgentContext, cfg LoopConfig) []Event {
	t.Helper()
	return collectEvents(AgentLoop(context.Background(), msgs, actx, cfg))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

func TestAgentLoop_SimpleTextResponse(t *testing.T) {
	events := runTestLoop(t,
		[]AgentMessage{UserMsg("hi")},
		AgentContext{},
		LoopConfig{StreamFn: mockStreamFn(assistantMsg("hello", StopReasonStop))},
	)

	requireEvent(t, events, EventAgentStart)
	requireEvent(t, events, EventAgentEnd)
	requireEvent(t, events, EventTurnStart)
	requireEvent(t, events, EventTurnEnd)

	ev, _ := findEvent(events, EventAgentEnd)
	if len(ev.NewMessages) < 2 {
		t.Fatalf("agent_end NewMessages: expected >= 2, got %d", len(ev.NewMessages))
	}
}

func TestAgentLoop_ToolCallAndResult(t *testing.T) {
	var calls []string
	tc := ToolCall{ID: "tc1", Name: "echo", Args: json.RawMessage(`{"value":"ping"}`)}

	events := runTestLoop(t,
		[]AgentMessage{UserMsg("test")},
		AgentContext{Tools: []Tool{echoTool(&calls)}},
		LoopConfig{StreamFn: mockStreamFn(toolCallMsg(tc), assistantMsg("done", StopReasonStop))},
	)

	if len(calls) != 1 || calls[0] != "ping" {
		t.Fatalf("expected echo('ping'), got %v", calls)
	}
	requireEvent(t, events, EventToolExecStart)
	requireEvent(t, events, EventToolExecEnd)
}

func TestAgentLoop_MaxTurns(t *testing.T) {
	var calls []string
	responses := make([]Message, 20)
	for i := range responses {
		responses[i] = toolCallMsg(ToolCall{ID: fmt.Sprintf("tc%d", i), Name: "echo", Args: json.RawMessage(`{"value":"x"}`)})
	}

	events := runTestLoop(t,
		[]AgentMessage{UserMsg("loop")},
		AgentContext{Tools: []Tool{echoTool(&calls)}},
		LoopConfig{StreamFn: mockStreamFn(responses...), MaxTurns: 3},
	)

	requireEvent(t, events, EventError)
}

func TestAgentLoop_Abort(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	events := collectEvents(AgentLoop(ctx,
		[]AgentMessage{UserMsg("cancelled")},
		AgentContext{},
		LoopConfig{StreamFn: func(ctx context.Context, req *LLMRequest) (*LLMResponse, error) {
			return nil, ctx.Err()
		}},
	))

	requireEvent(t, events, EventAgentEnd)
}

func TestAgentLoop_AbortEmitsAbortMarker(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	events := collectEvents(AgentLoop(ctx,
		[]AgentMessage{UserMsg("cancelled")},
		AgentContext{},
		LoopConfig{
			StreamFn: func(ctx context.Context, req *LLMRequest) (*LLMResponse, error) {
				return nil, ctx.Err()
			},
			ShouldEmitAbortMarker: func() bool { return true },
		},
	))

	var abortMsg Message
	found := false
	for _, ev := range events {
		msg, ok := ev.Message.(Message)
		if ev.Type == EventMessageEnd && ok && msg.StopReason == StopReasonAborted {
			abortMsg = msg
			found = true
			break
		}
	}
	if !found {
		t.Fatal("expected abort marker message")
	}
	if abortMsg.Metadata["abort_phase"] != "inference" {
		t.Fatalf("expected abort phase inference, got %v", abortMsg.Metadata["abort_phase"])
	}
}

func TestAgentLoop_SteeringInterrupt(t *testing.T) {
	var calls []string
	steeringDelivered := false

	events := runTestLoop(t,
		[]AgentMessage{UserMsg("test steering")},
		AgentContext{Tools: []Tool{echoTool(&calls)}},
		LoopConfig{
			StreamFn: sequentialStreamFn(func(i int, _ *LLMRequest) (*LLMResponse, error) {
				if i == 0 {
					return &LLMResponse{Message: toolCallMsg(
						ToolCall{ID: "tc1", Name: "echo", Args: json.RawMessage(`{"value":"first"}`)},
						ToolCall{ID: "tc2", Name: "echo", Args: json.RawMessage(`{"value":"second"}`)},
					)}, nil
				}
				return &LLMResponse{Message: assistantMsg("steered", StopReasonStop)}, nil
			}),
			GetSteeringMessages: func() []AgentMessage {
				if len(calls) == 1 && !steeringDelivered {
					steeringDelivered = true
					return []AgentMessage{UserMsg("redirect")}
				}
				return nil
			},
		},
	)

	if len(calls) != 1 || calls[0] != "first" {
		t.Fatalf("expected only first tool executed, got %v", calls)
	}
	if n := countEvent(events, EventToolExecEnd); n != 2 {
		t.Fatalf("expected 2 tool_exec_end, got %d", n)
	}
}

func TestAgentLoop_FollowUp(t *testing.T) {
	followUpDelivered := false

	events := runTestLoop(t,
		[]AgentMessage{UserMsg("initial")},
		AgentContext{},
		LoopConfig{
			StreamFn: sequentialStreamFn(func(i int, _ *LLMRequest) (*LLMResponse, error) {
				if i == 0 {
					return &LLMResponse{Message: assistantMsg("first", StopReasonStop)}, nil
				}
				return &LLMResponse{Message: assistantMsg("second", StopReasonStop)}, nil
			}),
			GetFollowUpMessages: func() []AgentMessage {
				if !followUpDelivered {
					followUpDelivered = true
					return []AgentMessage{UserMsg("follow up")}
				}
				return nil
			},
		},
	)

	// 2 assistant message_end = outer loop worked
	assistantEnds := 0
	for _, ev := range events {
		if ev.Type == EventMessageEnd {
			if msg, ok := ev.Message.(Message); ok && msg.Role == RoleAssistant {
				assistantEnds++
			}
		}
	}
	if assistantEnds != 2 {
		t.Fatalf("expected 2 assistant message_end, got %d", assistantEnds)
	}
}

func TestAgentLoop_ToolMiddleware(t *testing.T) {
	var calls []string
	var log []string

	tc := ToolCall{ID: "tc1", Name: "echo", Args: json.RawMessage(`{"value":"mid"}`)}
	mw := func(ctx context.Context, call ToolCall, next ToolExecuteFunc) (json.RawMessage, error) {
		log = append(log, "before")
		result, err := next(ctx, call.Args)
		log = append(log, "after")
		return result, err
	}

	runTestLoop(t,
		[]AgentMessage{UserMsg("test")},
		AgentContext{Tools: []Tool{echoTool(&calls)}},
		LoopConfig{
			StreamFn:    mockStreamFn(toolCallMsg(tc), assistantMsg("done", StopReasonStop)),
			Middlewares: []ToolMiddleware{mw},
		},
	)

	if len(calls) != 1 || calls[0] != "mid" {
		t.Fatalf("expected echo('mid'), got %v", calls)
	}
	if len(log) != 2 || log[0] != "before" || log[1] != "after" {
		t.Fatalf("middleware log: %v", log)
	}
}

func TestAgentLoop_PermissionDenied(t *testing.T) {
	var calls []string
	tc := ToolCall{ID: "tc1", Name: "echo", Args: json.RawMessage(`{"value":"denied"}`)}

	events := runTestLoop(t,
		[]AgentMessage{UserMsg("test")},
		AgentContext{Tools: []Tool{echoTool(&calls)}},
		LoopConfig{
			StreamFn: mockStreamFn(toolCallMsg(tc), assistantMsg("ok", StopReasonStop)),
			CheckPermission: func(ctx context.Context, call ToolCall) error {
				return fmt.Errorf("denied")
			},
		},
	)

	if len(calls) != 0 {
		t.Fatalf("tool should not execute, got %v", calls)
	}
	ev, ok := findEvent(events, EventToolExecEnd)
	if !ok || !ev.IsError {
		t.Fatal("expected tool_exec_end with isError=true")
	}
}

func TestAgentLoop_PreviewNotCalledWhenArgsInvalid(t *testing.T) {
	var previewCalls int64
	pt := &previewProgressTool{previewCalls: &previewCalls}
	tc := ToolCall{ID: "tc-preview-invalid", Name: "preview_tool", Args: json.RawMessage(`{}`)}

	events := runTestLoop(t,
		[]AgentMessage{UserMsg("test")},
		AgentContext{Tools: []Tool{pt}},
		LoopConfig{StreamFn: mockStreamFn(toolCallMsg(tc), assistantMsg("done", StopReasonStop))},
	)

	if got := atomic.LoadInt64(&previewCalls); got != 0 {
		t.Fatalf("preview should not be called for invalid args, got %d", got)
	}
	if n := countEvent(events, EventToolExecUpdate); n != 0 {
		t.Fatalf("expected no tool_exec_update for invalid args, got %d", n)
	}

	ev, ok := findEvent(events, EventToolExecEnd)
	if !ok || !ev.IsError {
		t.Fatal("expected tool_exec_end with isError=true")
	}
}

func TestAgentLoop_ToolExecUpdateKinds(t *testing.T) {
	var previewCalls int64
	pt := &previewProgressTool{previewCalls: &previewCalls}
	tc := ToolCall{ID: "tc-preview-valid", Name: "preview_tool", Args: json.RawMessage(`{"x":"v"}`)}

	events := runTestLoop(t,
		[]AgentMessage{UserMsg("test")},
		AgentContext{Tools: []Tool{pt}},
		LoopConfig{StreamFn: mockStreamFn(toolCallMsg(tc), assistantMsg("done", StopReasonStop))},
	)

	if got := atomic.LoadInt64(&previewCalls); got != 1 {
		t.Fatalf("preview should be called once, got %d", got)
	}

	var updates []Event
	for _, ev := range events {
		if ev.Type == EventToolExecUpdate {
			updates = append(updates, ev)
		}
	}
	if len(updates) != 2 {
		t.Fatalf("expected 2 tool_exec_update events (preview + progress), got %d", len(updates))
	}
	if updates[0].UpdateKind != ToolExecUpdatePreview {
		t.Fatalf("first update should be preview, got %q", updates[0].UpdateKind)
	}
	if updates[1].UpdateKind != ToolExecUpdateProgress {
		t.Fatalf("second update should be progress, got %q", updates[1].UpdateKind)
	}

	end, ok := findEvent(events, EventToolExecEnd)
	if !ok || end.IsError {
		t.Fatal("expected successful tool_exec_end")
	}
}

func TestAgentLoop_CircuitBreaker(t *testing.T) {
	failCount := 0
	failTool := NewFuncTool("fail", "always fails", nil,
		func(ctx context.Context, args json.RawMessage) (json.RawMessage, error) {
			failCount++
			return nil, fmt.Errorf("tool error")
		})

	runTestLoop(t,
		[]AgentMessage{UserMsg("test")},
		AgentContext{Tools: []Tool{failTool}},
		LoopConfig{
			StreamFn: sequentialStreamFn(func(i int, _ *LLMRequest) (*LLMResponse, error) {
				if i < 5 {
					return &LLMResponse{Message: toolCallMsg(ToolCall{ID: fmt.Sprintf("tc%d", i), Name: "fail", Args: json.RawMessage(`{}`)})}, nil
				}
				return &LLMResponse{Message: assistantMsg("gave up", StopReasonStop)}, nil
			}),
			MaxToolErrors: 2,
		},
	)

	if failCount > 2 {
		t.Fatalf("circuit breaker should cap at 2, got %d", failCount)
	}
}

func TestAgentLoopContinue(t *testing.T) {
	events := collectEvents(AgentLoopContinue(
		context.Background(),
		AgentContext{Messages: []AgentMessage{UserMsg("existing")}},
		LoopConfig{StreamFn: mockStreamFn(assistantMsg("continued", StopReasonStop))},
	))

	ev, _ := findEvent(events, EventAgentEnd)
	if len(ev.NewMessages) != 1 {
		t.Fatalf("expected 1 new message, got %d", len(ev.NewMessages))
	}
}

func TestAgentLoopContinue_EmptyContext(t *testing.T) {
	events := collectEvents(AgentLoopContinue(
		context.Background(),
		AgentContext{},
		LoopConfig{StreamFn: mockStreamFn()},
	))
	requireEvent(t, events, EventError)
}

func TestCollect(t *testing.T) {
	msgs, err := Collect(AgentLoop(
		context.Background(),
		[]AgentMessage{UserMsg("hi")},
		AgentContext{},
		LoopConfig{StreamFn: mockStreamFn(assistantMsg("ok", StopReasonStop))},
	))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(msgs) < 2 {
		t.Fatalf("expected >= 2 messages, got %d", len(msgs))
	}
}

func TestEventStream(t *testing.T) {
	stream := NewEventStream(AgentLoop(
		context.Background(),
		[]AgentMessage{UserMsg("hi")},
		AgentContext{},
		LoopConfig{StreamFn: mockStreamFn(assistantMsg("ok", StopReasonStop))},
	))

	count := 0
	for range stream.Events() {
		count++
	}
	if count == 0 {
		t.Fatal("expected events")
	}

	msgs, err := stream.Result()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(msgs) < 2 {
		t.Fatalf("expected >= 2 messages, got %d", len(msgs))
	}

	select {
	case <-stream.Done():
	default:
		t.Fatal("Done() not closed")
	}
}

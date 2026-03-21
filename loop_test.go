package agentcore

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
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

type richContentTool struct {
	calls *int64
}

func (t *richContentTool) Name() string        { return "rich_tool" }
func (t *richContentTool) Description() string { return "tool with rich content output" }
func (t *richContentTool) Schema() map[string]any {
	return map[string]any{
		"type": "object",
		"properties": map[string]any{
			"x": map[string]any{"type": "string"},
		},
		"required": []string{"x"},
	}
}
func (t *richContentTool) Execute(ctx context.Context, args json.RawMessage) (json.RawMessage, error) {
	return json.RawMessage(`"plain"`), nil
}
func (t *richContentTool) ExecuteContent(ctx context.Context, args json.RawMessage) ([]ContentBlock, error) {
	atomic.AddInt64(t.calls, 1)
	return []ContentBlock{TextBlock("rich")}, nil
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

func TestAgentLoop_AbortBehavior(t *testing.T) {
	for _, tc := range []struct {
		name             string
		emitAbortMarker  bool
		wantAbortMessage bool
	}{
		{name: "silent cancel"},
		{name: "with abort marker", emitAbortMarker: true, wantAbortMessage: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			cancel()

			events := collectEvents(AgentLoop(ctx,
				[]AgentMessage{UserMsg("cancelled")},
				AgentContext{},
				LoopConfig{
					StreamFn: func(ctx context.Context, req *LLMRequest) (*LLMResponse, error) {
						return nil, ctx.Err()
					},
					ShouldEmitAbortMarker: func() bool { return tc.emitAbortMarker },
				},
			))

			requireEvent(t, events, EventAgentEnd)

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
			if found != tc.wantAbortMessage {
				t.Fatalf("abort marker found=%v, want %v", found, tc.wantAbortMessage)
			}
			if tc.wantAbortMessage && abortMsg.Metadata["abort_phase"] != "inference" {
				t.Fatalf("expected abort phase inference, got %v", abortMsg.Metadata["abort_phase"])
			}
		})
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

func TestAgentLoop_ApprovalDenied(t *testing.T) {
	var approvalChecks int
	var calls []string

	events := runTestLoop(t,
		[]AgentMessage{UserMsg("test")},
		AgentContext{Tools: []Tool{echoTool(&calls)}},
		LoopConfig{
			StreamFn: sequentialStreamFn(func(i int, _ *LLMRequest) (*LLMResponse, error) {
				if i < 3 {
					return &LLMResponse{Message: toolCallMsg(ToolCall{
						ID:   fmt.Sprintf("tc%d", i),
						Name: "echo",
						Args: json.RawMessage(`{"value":"denied"}`),
					})}, nil
				}
				return &LLMResponse{Message: assistantMsg("done", StopReasonStop)}, nil
			}),
			CheckToolApproval: func(ctx context.Context, req ToolApprovalRequest) (*ToolApprovalResult, error) {
				approvalChecks++
				return &ToolApprovalResult{
					Approved: false,
					Decision: ToolApprovalDeny,
					Reason:   "denied",
				}, nil
			},
			MaxToolErrors: 1,
		},
	)

	if len(calls) != 0 {
		t.Fatalf("tool should not execute, got %v", calls)
	}
	if approvalChecks != 3 {
		t.Fatalf("approval check should run for every tool call, got %d", approvalChecks)
	}
	end, ok := findEvent(events, EventToolExecEnd)
	if !ok || !end.IsError {
		t.Fatal("expected tool_exec_end with isError=true")
	}
	for _, ev := range events {
		if ev.Type == EventToolExecEnd && strings.Contains(string(ev.Result), "disabled after") {
			t.Fatalf("approval denial should not disable the tool, got result %s", string(ev.Result))
		}
	}
}

func TestAgentLoop_PreviewAndProgress(t *testing.T) {
	for _, tc := range []struct {
		name              string
		args              json.RawMessage
		wantPreviewCalls  int64
		wantUpdateKinds   []ToolExecUpdateKind
		wantToolExecError bool
	}{
		{
			name:              "invalid args skip preview",
			args:              json.RawMessage(`{}`),
			wantPreviewCalls:  0,
			wantToolExecError: true,
		},
		{
			name:              "valid args emit preview and progress",
			args:              json.RawMessage(`{"x":"v"}`),
			wantPreviewCalls:  1,
			wantUpdateKinds:   []ToolExecUpdateKind{ToolExecUpdatePreview, ToolExecUpdateProgress},
			wantToolExecError: false,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var previewCalls int64
			pt := &previewProgressTool{previewCalls: &previewCalls}

			events := runTestLoop(t,
				[]AgentMessage{UserMsg("test")},
				AgentContext{Tools: []Tool{pt}},
				LoopConfig{StreamFn: mockStreamFn(toolCallMsg(ToolCall{
					ID:   "tc-preview",
					Name: "preview_tool",
					Args: tc.args,
				}), assistantMsg("done", StopReasonStop))},
			)

			if got := atomic.LoadInt64(&previewCalls); got != tc.wantPreviewCalls {
				t.Fatalf("preview calls: got %d, want %d", got, tc.wantPreviewCalls)
			}

			var gotKinds []ToolExecUpdateKind
			for _, ev := range events {
				if ev.Type == EventToolExecUpdate {
					gotKinds = append(gotKinds, ev.UpdateKind)
				}
			}
			if len(gotKinds) != len(tc.wantUpdateKinds) {
				t.Fatalf("update count: got %d, want %d", len(gotKinds), len(tc.wantUpdateKinds))
			}
			for i := range gotKinds {
				if gotKinds[i] != tc.wantUpdateKinds[i] {
					t.Fatalf("update[%d]: got %q, want %q", i, gotKinds[i], tc.wantUpdateKinds[i])
				}
			}

			end, ok := findEvent(events, EventToolExecEnd)
			if !ok {
				t.Fatal("expected tool_exec_end")
			}
			if end.IsError != tc.wantToolExecError {
				t.Fatalf("tool_exec_end isError=%v, want %v", end.IsError, tc.wantToolExecError)
			}
		})
	}
}

func TestAgentLoop_ContentToolUsesMiddleware(t *testing.T) {
	var contentCalls int64
	var log []string
	rt := &richContentTool{calls: &contentCalls}
	tc := ToolCall{ID: "tc-rich", Name: "rich_tool", Args: json.RawMessage(`{"x":"v"}`)}

	var secondReq *LLMRequest
	events := runTestLoop(t,
		[]AgentMessage{UserMsg("test")},
		AgentContext{Tools: []Tool{rt}},
		LoopConfig{
			StreamFn: sequentialStreamFn(func(i int, req *LLMRequest) (*LLMResponse, error) {
				if i == 0 {
					return &LLMResponse{Message: toolCallMsg(tc)}, nil
				}
				secondReq = req
				return &LLMResponse{Message: assistantMsg("done", StopReasonStop)}, nil
			}),
			Middlewares: []ToolMiddleware{
				func(ctx context.Context, call ToolCall, next ToolExecuteFunc) (json.RawMessage, error) {
					log = append(log, "before")
					out, err := next(ctx, call.Args)
					log = append(log, "after")
					return out, err
				},
			},
		},
	)

	if got := atomic.LoadInt64(&contentCalls); got != 1 {
		t.Fatalf("content tool should execute once, got %d", got)
	}
	if len(log) != 2 || log[0] != "before" || log[1] != "after" {
		t.Fatalf("middleware log: %v", log)
	}
	if secondReq == nil || len(secondReq.Messages) == 0 {
		t.Fatal("expected second llm request with tool result in context")
	}
	last := secondReq.Messages[len(secondReq.Messages)-1]
	if last.Role != RoleTool {
		t.Fatalf("expected last message to be tool result, got %q", last.Role)
	}
	if got := last.TextContent(); got != "rich" {
		t.Fatalf("expected rich tool content in context, got %q", got)
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

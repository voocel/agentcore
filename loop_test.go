package agentcore

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestAgentLoop_SimpleTextResponse(t *testing.T) {
	events := runTestLoop(t,
		[]AgentMessage{UserMsg("hi")},
		AgentContext{},
		LoopConfig{Model: mockModel(assistantMsg("hello", StopReasonStop))},
	)

	requireEvent(t, events, EventAgentStart)
	requireEvent(t, events, EventAgentEnd)
	requireEvent(t, events, EventTurnStart)
	requireEvent(t, events, EventModelResponse)

	ev, _ := findEvent(events, EventAgentEnd)
	if len(ev.NewMessages) < 2 {
		t.Fatalf("agent_end NewMessages: expected >= 2, got %d", len(ev.NewMessages))
	}
	if ev.Summary == nil {
		t.Fatal("expected agent_end summary")
	}
	if ev.Summary.TurnCount != 1 || ev.Summary.ToolCalls != 0 || ev.Summary.ToolErrors != 0 || ev.Summary.EndReason != EndReasonStop {
		t.Fatalf("unexpected summary: %#v", ev.Summary)
	}
}

func TestCallLLM_CommitsProjectedContextWhenRequested(t *testing.T) {
	original := strings.Repeat("a", 800)
	trimmed := "aaaaaaaaaaaaaaaaaaaa...aaaaaaaaaa"
	agentCtx := &AgentContext{
		Messages: []AgentMessage{
			UserMsg(original),
			UserMsg("recent"),
		},
	}

	var committed []AgentMessage
	cfg := LoopConfig{
		ContextManager: projectionCommitManager{
			projection: ContextProjection{
				Messages: []AgentMessage{
					UserMsg(trimmed),
					UserMsg("recent"),
				},
				Usage: &ContextUsage{
					Tokens:        128,
					ContextWindow: 1024,
					Percent:       12.5,
				},
				CommitMessages: []AgentMessage{
					UserMsg(trimmed),
					UserMsg("recent"),
				},
				ShouldCommit: true,
			},
		},
		Model: funcModel(func(ctx context.Context, req *LLMRequest) (*LLMResponse, error) {
			if got := req.Messages[0].TextContent(); got == original {
				t.Fatal("expected projected request to be trimmed before model call")
			}
			return &LLMResponse{Message: assistantMsg("ok", StopReasonStop)}, nil
		}),
		CommitContext: func(msgs []AgentMessage, usage *ContextUsage) error {
			committed = copyMessages(msgs)
			return nil
		},
	}

	events := make(chan Event, 16)
	if _, _, err := callLLM(context.Background(), agentCtx, cfg, eventSink{ctx: context.Background(), ch: events}, llmCallHooks{}); err != nil {
		t.Fatalf("callLLM failed: %v", err)
	}

	if len(committed) == 0 {
		t.Fatal("expected projected context to be committed")
	}
	if agentCtx.Messages[0].TextContent() == original {
		t.Fatal("expected agent context baseline to be replaced with compacted messages")
	}
}

func TestAgentLoop_ToolCallAndResult(t *testing.T) {
	var calls []string
	tc := ToolCall{ID: "tc1", Name: "echo", Args: json.RawMessage(`{"value":"ping"}`)}

	events := runTestLoop(t,
		[]AgentMessage{UserMsg("test")},
		AgentContext{Tools: []Tool{echoTool(&calls)}},
		LoopConfig{Model: mockModel(toolCallMsg(tc), assistantMsg("done", StopReasonStop))},
	)

	if len(calls) != 1 || calls[0] != "ping" {
		t.Fatalf("expected echo('ping'), got %v", calls)
	}
	requireEvent(t, events, EventToolExecStart)
	requireEvent(t, events, EventToolExecEnd)

	ev, _ := findEvent(events, EventAgentEnd)
	if ev.Summary == nil {
		t.Fatal("expected agent_end summary")
	}
	if ev.Summary.TurnCount != 2 || ev.Summary.ToolCalls != 1 || ev.Summary.ToolErrors != 0 || ev.Summary.EndReason != EndReasonStop {
		t.Fatalf("unexpected summary: %#v", ev.Summary)
	}
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
		LoopConfig{Model: mockModel(responses...), MaxTurns: 3},
	)

	requireEvent(t, events, EventError)
	ev, _ := findEvent(events, EventAgentEnd)
	if ev.Summary == nil {
		t.Fatal("expected agent_end summary")
	}
	if ev.Summary.TurnCount != 3 || ev.Summary.ToolCalls != 3 || ev.Summary.EndReason != EndReasonMaxTurns {
		t.Fatalf("unexpected summary: %#v", ev.Summary)
	}
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
					Model: funcModel(func(ctx context.Context, req *LLMRequest) (*LLMResponse, error) {
						return nil, ctx.Err()
					}),
					ShouldEmitAbortMarker: func() bool { return tc.emitAbortMarker },
				},
			))

			requireEvent(t, events, EventAgentEnd)
			ev, _ := findEvent(events, EventAgentEnd)
			if ev.Summary == nil || ev.Summary.EndReason != EndReasonAborted {
				t.Fatalf("unexpected summary: %#v", ev.Summary)
			}

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
			Model: sequentialModel(func(i int, _ *LLMRequest) (*LLMResponse, error) {
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
			Model: sequentialModel(func(i int, _ *LLMRequest) (*LLMResponse, error) {
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

func TestAgentLoop_Middleware(t *testing.T) {
	t.Run("content tool", func(t *testing.T) {
		var contentCalls int64
		var log []string
		rt := &richContentTool{calls: &contentCalls}
		tc := ToolCall{ID: "tc-rich", Name: "rich_tool", Args: json.RawMessage(`{"x":"v"}`)}

		var secondReq *LLMRequest
		events := runTestLoop(t,
			[]AgentMessage{UserMsg("test")},
			AgentContext{Tools: []Tool{rt}},
			LoopConfig{
				Model: sequentialModel(func(i int, req *LLMRequest) (*LLMResponse, error) {
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
	})
}

func TestAgentLoop_ToolGate(t *testing.T) {
	t.Run("denied gate blocks execution without disabling the tool", func(t *testing.T) {
		var gateChecks int
		var calls []string

		events := runTestLoop(t,
			[]AgentMessage{UserMsg("test")},
			AgentContext{Tools: []Tool{echoTool(&calls)}},
			LoopConfig{
				Model: sequentialModel(func(i int, _ *LLMRequest) (*LLMResponse, error) {
					if i < 3 {
						return &LLMResponse{Message: toolCallMsg(ToolCall{
							ID:   fmt.Sprintf("tc%d", i),
							Name: "echo",
							Args: json.RawMessage(`{"value":"denied"}`),
						})}, nil
					}
					return &LLMResponse{Message: assistantMsg("done", StopReasonStop)}, nil
				}),
				ToolGate: func(ctx context.Context, req GateRequest) (*GateDecision, error) {
					gateChecks++
					return &GateDecision{Allowed: false, Reason: "denied"}, nil
				},
				MaxToolErrors: 1,
			},
		)

		if len(calls) != 0 {
			t.Fatalf("tool should not execute, got %v", calls)
		}
		if gateChecks != 3 {
			t.Fatalf("gate should run for every tool call, got %d", gateChecks)
		}
		end, ok := findEvent(events, EventToolExecEnd)
		if !ok || !end.IsError {
			t.Fatal("expected tool_exec_end with isError=true")
		}
		for _, ev := range events {
			if ev.Type == EventToolExecEnd && strings.Contains(string(ev.Result), "disabled after") {
				t.Fatalf("denial should not disable the tool, got result %s", string(ev.Result))
			}
		}
	})

	t.Run("gate error is treated as deny", func(t *testing.T) {
		var calls []string

		runTestLoop(t,
			[]AgentMessage{UserMsg("test")},
			AgentContext{Tools: []Tool{echoTool(&calls)}},
			LoopConfig{
				Model: mockModel(
					toolCallMsg(ToolCall{ID: "tc1", Name: "echo", Args: json.RawMessage(`{"value":"x"}`)}),
					assistantMsg("done", StopReasonStop),
				),
				ToolGate: func(ctx context.Context, req GateRequest) (*GateDecision, error) {
					return nil, fmt.Errorf("gate exploded")
				},
			},
		)

		if len(calls) != 0 {
			t.Fatalf("tool must not execute when gate errors, got %v", calls)
		}
	})
}

// validatorTool implements Tool + Validator. Track invocations to confirm
// Validate is called and Execute is short-circuited on failure.
type validatorTool struct {
	validateCalls int
	executeCalls  int
	result        ValidationResult
}

func (t *validatorTool) Name() string        { return "vtool" }
func (t *validatorTool) Description() string { return "tool with input validator" }
func (t *validatorTool) Schema() map[string]any {
	return map[string]any{
		"type":       "object",
		"properties": map[string]any{"x": map[string]any{"type": "string"}},
		"required":   []string{"x"},
	}
}
func (t *validatorTool) Validate(_ context.Context, _ json.RawMessage) ValidationResult {
	t.validateCalls++
	return t.result
}
func (t *validatorTool) Execute(_ context.Context, _ json.RawMessage) (json.RawMessage, error) {
	t.executeCalls++
	return json.RawMessage(`"ok"`), nil
}

func TestAgentLoop_ValidatorShortCircuit(t *testing.T) {
	t.Run("OK=false skips execute and surfaces message as tool_result", func(t *testing.T) {
		vt := &validatorTool{result: ValidationResult{
			OK:        false,
			Message:   "needs read first",
			ErrorCode: 2,
		}}

		events := runTestLoop(t,
			[]AgentMessage{UserMsg("test")},
			AgentContext{Tools: []Tool{vt}},
			LoopConfig{
				Model: mockModel(
					toolCallMsg(ToolCall{ID: "tc1", Name: "vtool", Args: json.RawMessage(`{"x":"y"}`)}),
					assistantMsg("done", StopReasonStop),
				),
			},
		)

		if vt.validateCalls != 1 {
			t.Fatalf("validate must run once, got %d", vt.validateCalls)
		}
		if vt.executeCalls != 0 {
			t.Fatalf("execute must be skipped, got %d calls", vt.executeCalls)
		}
		end, ok := findEvent(events, EventToolExecEnd)
		if !ok {
			t.Fatal("expected EventToolExecEnd")
		}
		if !end.IsError {
			t.Fatal("validate failure must surface as IsError=true")
		}
		if !strings.Contains(string(end.Result), "needs read first") {
			t.Fatalf("expected message in result, got %s", string(end.Result))
		}
	})

	t.Run("OK=true allows execute to run", func(t *testing.T) {
		vt := &validatorTool{result: ValidationResult{OK: true}}

		runTestLoop(t,
			[]AgentMessage{UserMsg("test")},
			AgentContext{Tools: []Tool{vt}},
			LoopConfig{
				Model: mockModel(
					toolCallMsg(ToolCall{ID: "tc1", Name: "vtool", Args: json.RawMessage(`{"x":"y"}`)}),
					assistantMsg("done", StopReasonStop),
				),
			},
		)

		if vt.validateCalls != 1 {
			t.Fatalf("validate calls: want 1, got %d", vt.validateCalls)
		}
		if vt.executeCalls != 1 {
			t.Fatalf("execute calls: want 1, got %d", vt.executeCalls)
		}
	})
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
				LoopConfig{Model: mockModel(toolCallMsg(ToolCall{
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
			if tc.name == "valid args emit preview and progress" {
				var progressEvent *Event
				for i := range events {
					if events[i].Type == EventToolExecUpdate && events[i].UpdateKind == ToolExecUpdateProgress {
						progressEvent = &events[i]
						break
					}
				}
				if progressEvent == nil || progressEvent.Progress == nil {
					t.Fatal("expected structured progress payload")
				}
				if progressEvent.Progress.Kind != ProgressSummary || progressEvent.Progress.Summary != "progress" {
					t.Fatalf("unexpected progress payload: %+v", progressEvent.Progress)
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

func TestAgentLoop_StreamingToolExecutionStartsBeforeAssistantEnds(t *testing.T) {
	var calls []string
	tc := ToolCall{ID: "tc-stream", Name: "echo", Args: json.RawMessage(`{"value":"ping"}`)}

	model := newScriptedStreamModel(
		streamAssistantToolCalls("working", StopReasonToolUse, 40*time.Millisecond, tc),
		streamAssistantDone("done", StopReasonStop),
	)

	events := runTestLoop(t,
		[]AgentMessage{UserMsg("stream tools")},
		AgentContext{Tools: []Tool{echoTool(&calls)}},
		LoopConfig{Model: model},
	)

	if len(calls) != 1 || calls[0] != "ping" {
		t.Fatalf("expected streaming tool execution, got %v", calls)
	}

	toolExecStart := -1
	assistantToolMessageEnd := -1
	for i, ev := range events {
		if toolExecStart < 0 && ev.Type == EventToolExecStart {
			toolExecStart = i
		}
		if ev.Type == EventMessageEnd {
			if msg, ok := ev.Message.(Message); ok && msg.Role == RoleAssistant && len(msg.ToolCalls()) > 0 {
				assistantToolMessageEnd = i
				break
			}
		}
	}

	if toolExecStart < 0 || assistantToolMessageEnd < 0 {
		t.Fatalf("unexpected event sequence: tool_exec_start=%d assistant_tool_message_end=%d", toolExecStart, assistantToolMessageEnd)
	}
	if toolExecStart >= assistantToolMessageEnd {
		t.Fatalf("expected tool execution to start before assistant message ended, got start=%d end=%d", toolExecStart, assistantToolMessageEnd)
	}
}

func TestAgentLoop_StreamingToolExecutionKeepsContextOrder(t *testing.T) {
	var calls []string
	tc := ToolCall{ID: "tc-order", Name: "echo", Args: json.RawMessage(`{"value":"ordered"}`)}

	model := newScriptedStreamModel(
		streamAssistantToolCalls("plan", StopReasonToolUse, 20*time.Millisecond, tc),
		streamAssistantDone("done", StopReasonStop),
	)

	runTestLoop(t,
		[]AgentMessage{UserMsg("order")},
		AgentContext{Tools: []Tool{echoTool(&calls)}},
		LoopConfig{Model: model},
	)

	if len(calls) != 1 || calls[0] != "ordered" {
		t.Fatalf("expected ordered tool execution, got %v", calls)
	}

	secondReq := model.Request(1)
	if len(secondReq) < 3 {
		t.Fatalf("expected second request to include assistant tool call and tool result, got %d messages", len(secondReq))
	}

	assistantIdx := -1
	toolIdx := -1
	for i, msg := range secondReq {
		if msg.Role == RoleAssistant && len(msg.ToolCalls()) > 0 {
			assistantIdx = i
		}
		if msg.Role == RoleTool {
			toolIdx = i
		}
	}

	if assistantIdx < 0 || toolIdx < 0 {
		t.Fatalf("expected assistant tool call and tool result in second request, assistant=%d tool=%d", assistantIdx, toolIdx)
	}
	if assistantIdx >= toolIdx {
		t.Fatalf("expected assistant tool call before tool result, got assistant=%d tool=%d", assistantIdx, toolIdx)
	}
}

func TestAgentLoop_StreamingToolExecutionPreservesCompletedCallsOnLengthStop(t *testing.T) {
	var calls []string
	tc := ToolCall{ID: "tc-length", Name: "echo", Args: json.RawMessage(`{"value":"kept"}`)}

	model := newScriptedStreamModel(
		streamAssistantToolCalls("partial", StopReasonLength, 0, tc),
		streamAssistantDone("done", StopReasonStop),
	)

	runTestLoop(t,
		[]AgentMessage{UserMsg("length stop")},
		AgentContext{Tools: []Tool{echoTool(&calls)}},
		LoopConfig{Model: model},
	)

	if len(calls) != 1 || calls[0] != "kept" {
		t.Fatalf("expected completed tool call to survive StopReasonLength, got %v", calls)
	}

	secondReq := model.Request(1)
	foundAssistantToolCall := false
	for _, msg := range secondReq {
		if msg.Role == RoleAssistant && len(msg.ToolCalls()) > 0 {
			foundAssistantToolCall = true
			break
		}
	}
	if !foundAssistantToolCall {
		t.Fatal("expected completed tool call to remain in context after StopReasonLength")
	}
}

func TestAgentLoop_LengthStopRecoversPureTextResponse(t *testing.T) {
	model := newScriptedStreamModel(
		streamAssistantDone("partial answer", StopReasonLength),
		streamAssistantDone("continued answer", StopReasonStop),
	)

	events := runTestLoop(t,
		[]AgentMessage{UserMsg("recover length")},
		AgentContext{},
		LoopConfig{Model: model},
	)

	if got := len(model.requests); got != 2 {
		t.Fatalf("expected 2 LLM calls with one automatic recovery, got %d", got)
	}

	secondReq := model.Request(1)
	if len(secondReq) < 3 {
		t.Fatalf("expected recovery request to include prior assistant and recovery user message, got %d messages", len(secondReq))
	}
	last := secondReq[len(secondReq)-1]
	if last.Role != RoleUser || last.TextContent() != defaultLengthRecoveryPrompt {
		t.Fatalf("expected default recovery prompt as last message, got %#v", last)
	}

	ev, ok := findEvent(events, EventAgentEnd)
	if !ok || ev.Summary == nil || ev.Summary.TurnCount != 2 {
		t.Fatalf("expected 2 turns after recovery, got %#v", ev.Summary)
	}
}

func TestAgentLoop_LengthRecoveryForTruncatedToolCalls(t *testing.T) {
	// When output is truncated (StopReasonLength) and tool call blocks exist
	// but none completed, the loop should strip incomplete tool calls and
	// attempt length recovery (prompting the model to break work into smaller
	// pieces). This prevents silent failure when large tool arguments exceed
	// the output token limit.
	callCount := 0
	model := sequentialModel(func(i int, req *LLMRequest) (*LLMResponse, error) {
		callCount++
		if i == 0 {
			// First call: truncated output with incomplete tool call
			return &LLMResponse{
				Message: Message{
					Role: RoleAssistant,
					Content: []ContentBlock{
						TextBlock("partial"),
						{Type: ContentToolCall},
					},
					StopReason: StopReasonLength,
				},
			}, nil
		}
		// Recovery call: model returns normally
		return &LLMResponse{
			Message: Message{
				Role:       RoleAssistant,
				Content:    []ContentBlock{TextBlock("recovered")},
				StopReason: StopReasonStop,
			},
		}, nil
	})

	runTestLoop(t,
		[]AgentMessage{UserMsg("recover truncated tool call")},
		AgentContext{},
		LoopConfig{Model: model},
	)

	if callCount != 2 {
		t.Fatalf("expected 1 recovery attempt (2 total calls), got %d calls", callCount)
	}
}

func TestAgentLoop_StopAfterTool(t *testing.T) {
	// When StopAfterTool returns true for a tool, the loop should exit
	// immediately after that tool executes, even with tool_choice=required.
	commitTool := NewFuncTool("commit", "commit work", map[string]any{
		"type": "object", "properties": map[string]any{},
	}, func(ctx context.Context, args json.RawMessage) (json.RawMessage, error) {
		return json.Marshal(map[string]string{"status": "committed"})
	})
	neverTool := NewFuncTool("never_reach", "should not be called", map[string]any{
		"type": "object", "properties": map[string]any{},
	}, func(ctx context.Context, args json.RawMessage) (json.RawMessage, error) {
		t.Fatal("never_reach tool was called — StopAfterTool did not stop the loop")
		return nil, nil
	})

	callCount := 0
	model := sequentialModel(func(i int, req *LLMRequest) (*LLMResponse, error) {
		callCount++
		if i == 0 {
			return &LLMResponse{
				Message: Message{
					Role: RoleAssistant,
					Content: []ContentBlock{
						TextBlock("committing"),
						ToolCallBlock(ToolCall{ID: "tc1", Name: "commit", Args: json.RawMessage(`{}`)}),
					},
					StopReason: StopReasonToolUse,
				},
			}, nil
		}
		// If we get here, StopAfterTool didn't work
		return &LLMResponse{
			Message: Message{
				Role: RoleAssistant,
				Content: []ContentBlock{
					ToolCallBlock(ToolCall{ID: "tc2", Name: "never_reach", Args: json.RawMessage(`{}`)}),
				},
				StopReason: StopReasonToolUse,
			},
		}, nil
	})

	events := runTestLoop(t,
		[]AgentMessage{UserMsg("do work")},
		AgentContext{Tools: []Tool{commitTool, neverTool}},
		LoopConfig{
			Model: model,
			StopAfterTool: func(name string) bool {
				return name == "commit"
			},
		},
	)

	if callCount != 1 {
		t.Fatalf("expected loop to stop after commit tool (1 LLM call), got %d", callCount)
	}
	// Verify we got EventAgentEnd with EndReasonStop
	var endEvent *Event
	for _, ev := range events {
		if ev.Type == EventAgentEnd {
			endEvent = &ev
		}
	}
	if endEvent == nil {
		t.Fatal("no EventAgentEnd emitted")
	}
	if endEvent.Summary == nil || endEvent.Summary.EndReason != EndReasonStop {
		t.Fatalf("expected EndReasonStop, got %v", endEvent.Summary)
	}
}

func TestAgentLoop_StopAfterToolResult(t *testing.T) {
	saveTool := NewFuncTool("save_foundation", "save foundation", map[string]any{
		"type": "object", "properties": map[string]any{},
	}, func(ctx context.Context, args json.RawMessage) (json.RawMessage, error) {
		var p struct {
			Ready bool `json:"ready"`
		}
		_ = json.Unmarshal(args, &p)
		return json.Marshal(map[string]bool{"foundation_ready": p.Ready})
	})

	callCount := 0
	model := sequentialModel(func(i int, req *LLMRequest) (*LLMResponse, error) {
		callCount++
		ready := "false"
		if i == 1 {
			ready = "true"
		}
		return &LLMResponse{
			Message: Message{
				Role: RoleAssistant,
				Content: []ContentBlock{
					ToolCallBlock(ToolCall{
						ID:   fmt.Sprintf("tc%d", i),
						Name: "save_foundation",
						Args: json.RawMessage(fmt.Sprintf(`{"ready":%s}`, ready)),
					}),
				},
				StopReason: StopReasonToolUse,
			},
		}, nil
	})

	events := runTestLoop(t,
		[]AgentMessage{UserMsg("save all foundation parts")},
		AgentContext{Tools: []Tool{saveTool}},
		LoopConfig{
			Model: model,
			StopAfterToolResult: func(name string, result json.RawMessage) bool {
				if name != "save_foundation" {
					return false
				}
				var r struct {
					FoundationReady bool `json:"foundation_ready"`
				}
				_ = json.Unmarshal(result, &r)
				return r.FoundationReady
			},
		},
	)

	if callCount != 2 {
		t.Fatalf("expected loop to stop only after ready result (2 LLM calls), got %d", callCount)
	}
	var endEvent *Event
	for _, ev := range events {
		if ev.Type == EventAgentEnd {
			endEvent = &ev
		}
	}
	if endEvent == nil {
		t.Fatal("no EventAgentEnd emitted")
	}
	if endEvent.Summary == nil || endEvent.Summary.EndReason != EndReasonStop {
		t.Fatalf("expected EndReasonStop, got %v", endEvent.Summary)
	}
}

func TestAgentLoop_StreamingToolExecutionHonorsSteeringForQueuedTools(t *testing.T) {
	var calls []string
	var mu sync.Mutex
	steeringDelivered := false

	sleepyEcho := NewFuncTool("sleepy_echo", "serial echo", map[string]any{
		"type": "object",
		"properties": map[string]any{
			"value": map[string]any{"type": "string"},
		},
		"required": []string{"value"},
	}, func(ctx context.Context, args json.RawMessage) (json.RawMessage, error) {
		var p struct{ Value string }
		_ = json.Unmarshal(args, &p)
		time.Sleep(30 * time.Millisecond)
		mu.Lock()
		calls = append(calls, p.Value)
		mu.Unlock()
		return json.Marshal("ok")
	})

	tc1 := ToolCall{ID: "tc-s1", Name: "sleepy_echo", Args: json.RawMessage(`{"value":"first"}`)}
	tc2 := ToolCall{ID: "tc-s2", Name: "sleepy_echo", Args: json.RawMessage(`{"value":"second"}`)}

	model := newScriptedStreamModel(
		streamAssistantToolCalls("do work", StopReasonToolUse, 60*time.Millisecond, tc1, tc2),
		streamAssistantDone("redirected", StopReasonStop),
	)

	events := runTestLoop(t,
		[]AgentMessage{UserMsg("steer stream")},
		AgentContext{Tools: []Tool{sleepyEcho}},
		LoopConfig{
			Model: model,
			GetSteeringMessages: func() []AgentMessage {
				mu.Lock()
				defer mu.Unlock()
				if len(calls) == 1 && !steeringDelivered {
					steeringDelivered = true
					return []AgentMessage{UserMsg("redirect")}
				}
				return nil
			},
		},
	)

	mu.Lock()
	gotCalls := append([]string(nil), calls...)
	mu.Unlock()
	if len(gotCalls) != 1 || gotCalls[0] != "first" {
		t.Fatalf("expected only first queued tool to execute after steering, got %v", gotCalls)
	}

	sawSkippedSecond := false
	for _, ev := range events {
		if ev.Type == EventToolExecEnd && ev.ToolID == "tc-s2" && strings.Contains(string(ev.Result), "Skipped due to queued user message.") {
			sawSkippedSecond = true
			break
		}
	}
	if !sawSkippedSecond {
		t.Fatal("expected second queued tool to be skipped after steering")
	}
}

func TestAgentLoop_StreamingToolExecutionDrainsOnStreamError(t *testing.T) {
	started := make(chan struct{}, 1)
	tc := ToolCall{ID: "tc-err", Name: "wait_for_cancel", Args: json.RawMessage(`{}`)}

	waitForCancel := NewFuncTool("wait_for_cancel", "waits for cancellation", map[string]any{
		"type":       "object",
		"properties": map[string]any{},
	}, func(ctx context.Context, args json.RawMessage) (json.RawMessage, error) {
		select {
		case started <- struct{}{}:
		default:
		}
		<-ctx.Done()
		return nil, ctx.Err()
	})

	model := &scriptedStreamModel{
		streams: []func(chan<- StreamEvent){
			func(ch chan<- StreamEvent) {
				partial := Message{Role: RoleAssistant, Content: []ContentBlock{TextBlock("working")}}
				ch <- StreamEvent{Type: StreamEventTextStart, Message: partial}
				ch <- StreamEvent{Type: StreamEventToolCallStart, Message: partial}
				partial.Content = append(partial.Content, ToolCallBlock(tc))
				ch <- StreamEvent{Type: StreamEventToolCallEnd, Message: partial, CompletedToolCall: &tc}
				select {
				case <-started:
				case <-time.After(time.Second):
				}
				ch <- StreamEvent{Type: StreamEventError, Err: fmt.Errorf("stream failed")}
			},
		},
	}

	events := runTestLoop(t,
		[]AgentMessage{UserMsg("stream error")},
		AgentContext{Tools: []Tool{waitForCancel}},
		LoopConfig{Model: model},
	)

	toolStart := -1
	toolEnd := -1
	errIdx := -1
	agentEnd := -1
	for i, ev := range events {
		switch ev.Type {
		case EventToolExecStart:
			if ev.ToolID == "tc-err" && toolStart < 0 {
				toolStart = i
			}
		case EventToolExecEnd:
			if ev.ToolID == "tc-err" && strings.Contains(string(ev.Result), "context canceled") {
				toolEnd = i
			}
		case EventError:
			if ev.Err != nil && strings.Contains(ev.Err.Error(), "stream failed") {
				errIdx = i
			}
		case EventAgentEnd:
			agentEnd = i
		}
	}

	if toolStart < 0 || toolEnd < 0 || errIdx < 0 || agentEnd < 0 {
		t.Fatalf("unexpected event sequence: tool_start=%d tool_end=%d err=%d agent_end=%d", toolStart, toolEnd, errIdx, agentEnd)
	}
	if toolStart >= toolEnd {
		t.Fatalf("expected cancelled tool to emit end after start, got start=%d end=%d", toolStart, toolEnd)
	}
	if toolEnd >= errIdx {
		t.Fatalf("expected tool cleanup before stream error surfaced, got tool_end=%d err=%d", toolEnd, errIdx)
	}
	if errIdx >= agentEnd {
		t.Fatalf("expected agent_end after error, got err=%d agent_end=%d", errIdx, agentEnd)
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
			Model: sequentialModel(func(i int, _ *LLMRequest) (*LLMResponse, error) {
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

func TestLoopPublicAPIs(t *testing.T) {
	t.Run("continue appends new messages", func(t *testing.T) {
		events := collectEvents(AgentLoopContinue(
			context.Background(),
			AgentContext{Messages: []AgentMessage{UserMsg("existing")}},
			LoopConfig{Model: mockModel(assistantMsg("continued", StopReasonStop))},
		))

		ev, _ := findEvent(events, EventAgentEnd)
		if len(ev.NewMessages) != 1 {
			t.Fatalf("expected 1 new message, got %d", len(ev.NewMessages))
		}
	})

	t.Run("continue rejects empty context", func(t *testing.T) {
		events := collectEvents(AgentLoopContinue(
			context.Background(),
			AgentContext{},
			LoopConfig{Model: mockModel()},
		))
		requireEvent(t, events, EventError)
		ev, _ := findEvent(events, EventAgentEnd)
		if ev.Summary == nil || ev.Summary.EndReason != EndReasonError {
			t.Fatalf("unexpected summary: %#v", ev.Summary)
		}
	})

}

// ---------------------------------------------------------------------------
// OnMessage semantic tests — verify commitMessage fires on all paths
// ---------------------------------------------------------------------------

func TestOnMessage_PendingSteeringMessages(t *testing.T) {
	var calls []string
	steeringDelivered := false

	var logged []Role
	var mu sync.Mutex

	runTestLoop(t,
		[]AgentMessage{UserMsg("test steering")},
		AgentContext{Tools: []Tool{echoTool(&calls)}},
		LoopConfig{
			Model: sequentialModel(func(i int, _ *LLMRequest) (*LLMResponse, error) {
				if i == 0 {
					return &LLMResponse{Message: toolCallMsg(
						ToolCall{ID: "tc1", Name: "echo", Args: json.RawMessage(`{"value":"first"}`)},
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
			OnMessage: func(msg AgentMessage) {
				mu.Lock()
				logged = append(logged, msg.GetRole())
				mu.Unlock()
			},
		},
	)

	mu.Lock()
	defer mu.Unlock()

	// Should include: assistant (tool call), tool (result), user (steering), assistant (steered)
	var hasUser bool
	for _, r := range logged {
		if r == RoleUser {
			hasUser = true
		}
	}
	if !hasUser {
		t.Fatalf("expected OnMessage to fire for steering (user) message, got roles: %v", logged)
	}
}

func TestOnMessage_OrderMatchesNewMessages(t *testing.T) {
	var calls []string
	tc := ToolCall{ID: "tc1", Name: "echo", Args: json.RawMessage(`{"value":"x"}`)}

	var loggedTexts []string
	var mu sync.Mutex

	events := runTestLoop(t,
		[]AgentMessage{UserMsg("test order")},
		AgentContext{Tools: []Tool{echoTool(&calls)}},
		LoopConfig{
			Model: mockModel(toolCallMsg(tc), assistantMsg("done", StopReasonStop)),
			OnMessage: func(msg AgentMessage) {
				mu.Lock()
				loggedTexts = append(loggedTexts, msg.TextContent())
				mu.Unlock()
			},
		},
	)

	// Verify order matches NewMessages exactly, including the initial prompt.
	ev, _ := findEvent(events, EventAgentEnd)
	newMsgs := ev.NewMessages

	mu.Lock()
	defer mu.Unlock()

	if len(newMsgs) != len(loggedTexts) {
		t.Fatalf("newMessages (%d) does not match OnMessage calls (%d)", len(newMsgs), len(loggedTexts))
	}

	for i, text := range loggedTexts {
		expected := newMsgs[i].TextContent()
		if text != expected {
			t.Fatalf("OnMessage[%d] text=%q, newMessages[%d] text=%q", i, text, i, expected)
		}
	}
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// mockChatModel returns canned responses in order. Used as a test stand-in for
// real ChatModel adapters.
//
// Important: Generate and GenerateStream each consume exactly one response.
// They do NOT call each other, so tests that mix streaming and non-streaming
// calls observe a stable, predictable response order.
type mockChatModel struct {
	responses []Message
	idx       int64
}

func mockModel(responses ...Message) *mockChatModel {
	return &mockChatModel{responses: responses}
}

func (m *mockChatModel) take() (Message, error) {
	i := int(atomic.AddInt64(&m.idx, 1) - 1)
	if i >= len(m.responses) {
		return Message{}, fmt.Errorf("unexpected LLM call #%d (only %d responses provided)", i, len(m.responses))
	}
	return m.responses[i], nil
}

func (m *mockChatModel) Generate(ctx context.Context, msgs []Message, tools []ToolSpec, opts ...CallOption) (*LLMResponse, error) {
	msg, err := m.take()
	if err != nil {
		return nil, err
	}
	return &LLMResponse{Message: msg}, nil
}

func (m *mockChatModel) GenerateStream(ctx context.Context, msgs []Message, tools []ToolSpec, opts ...CallOption) (<-chan StreamEvent, error) {
	msg, err := m.take()
	if err != nil {
		return nil, err
	}
	ch := make(chan StreamEvent, 1)
	ch <- StreamEvent{Type: StreamEventDone, Message: msg, StopReason: msg.StopReason}
	close(ch)
	return ch, nil
}

func (m *mockChatModel) SupportsTools() bool { return true }

// sequentialMockModel calls fn(i, req) for each invocation, where i is the
// 0-based call index. Useful when test responses depend on the request payload.
// Like mockChatModel, each ChatModel method advances the cursor exactly once.
type sequentialMockModel struct {
	fn  func(i int, req *LLMRequest) (*LLMResponse, error)
	idx int64
}

func sequentialModel(fn func(i int, req *LLMRequest) (*LLMResponse, error)) *sequentialMockModel {
	return &sequentialMockModel{fn: fn}
}

func (m *sequentialMockModel) take(msgs []Message, tools []ToolSpec) (*LLMResponse, error) {
	i := int(atomic.AddInt64(&m.idx, 1) - 1)
	return m.fn(i, &LLMRequest{Messages: msgs, Tools: tools})
}

func (m *sequentialMockModel) Generate(ctx context.Context, msgs []Message, tools []ToolSpec, opts ...CallOption) (*LLMResponse, error) {
	return m.take(msgs, tools)
}

func (m *sequentialMockModel) GenerateStream(ctx context.Context, msgs []Message, tools []ToolSpec, opts ...CallOption) (<-chan StreamEvent, error) {
	resp, err := m.take(msgs, tools)
	if err != nil {
		return nil, err
	}
	ch := make(chan StreamEvent, 1)
	ch <- StreamEvent{Type: StreamEventDone, Message: resp.Message, StopReason: resp.Message.StopReason}
	close(ch)
	return ch, nil
}

func (m *sequentialMockModel) SupportsTools() bool { return true }

// funcMockModel wraps a single function call to act as a ChatModel.
// Useful for inline tests where each invocation runs the same handler.
// Each ChatModel method invokes the wrapped fn exactly once.
type funcMockModel struct {
	fn func(ctx context.Context, req *LLMRequest) (*LLMResponse, error)
}

func funcModel(fn func(ctx context.Context, req *LLMRequest) (*LLMResponse, error)) *funcMockModel {
	return &funcMockModel{fn: fn}
}

func (m *funcMockModel) Generate(ctx context.Context, msgs []Message, tools []ToolSpec, opts ...CallOption) (*LLMResponse, error) {
	return m.fn(ctx, &LLMRequest{Messages: msgs, Tools: tools})
}

func (m *funcMockModel) GenerateStream(ctx context.Context, msgs []Message, tools []ToolSpec, opts ...CallOption) (<-chan StreamEvent, error) {
	resp, err := m.fn(ctx, &LLMRequest{Messages: msgs, Tools: tools})
	if err != nil {
		return nil, err
	}
	ch := make(chan StreamEvent, 1)
	ch <- StreamEvent{Type: StreamEventDone, Message: resp.Message, StopReason: resp.Message.StopReason}
	close(ch)
	return ch, nil
}

func (m *funcMockModel) SupportsTools() bool { return true }

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
		_ = json.Unmarshal(args, &p)
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
	ReportToolProgress(ctx, ProgressPayload{Kind: ProgressSummary, Summary: "progress"})
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

type projectionCommitManager struct {
	projection ContextProjection
}

func (m projectionCommitManager) Project(ctx context.Context, msgs []AgentMessage) (ContextProjection, error) {
	return m.projection, nil
}

func (m projectionCommitManager) Compact(ctx context.Context, msgs []AgentMessage, reason CompactReason) (ContextCommitResult, error) {
	return ContextCommitResult{}, nil
}

func (m projectionCommitManager) RecoverOverflow(ctx context.Context, msgs []AgentMessage, cause error) (ContextRecoveryResult, error) {
	return ContextRecoveryResult{}, nil
}

func (m projectionCommitManager) Sync(msgs []AgentMessage) {}

func (m projectionCommitManager) Usage() *ContextUsage {
	if m.projection.Usage == nil {
		return nil
	}
	cp := *m.projection.Usage
	return &cp
}

func (m projectionCommitManager) Snapshot() *ContextSnapshot { return nil }

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

type scriptedStreamModel struct {
	mu       sync.Mutex
	requests [][]Message
	streams  []func(chan<- StreamEvent)
}

func newScriptedStreamModel(streams ...func(chan<- StreamEvent)) *scriptedStreamModel {
	return &scriptedStreamModel{streams: streams}
}

func (m *scriptedStreamModel) Generate(ctx context.Context, messages []Message, tools []ToolSpec, opts ...CallOption) (*LLMResponse, error) {
	return nil, fmt.Errorf("unexpected Generate call")
}

func (m *scriptedStreamModel) GenerateStream(ctx context.Context, messages []Message, tools []ToolSpec, opts ...CallOption) (<-chan StreamEvent, error) {
	m.mu.Lock()
	idx := len(m.requests)
	m.requests = append(m.requests, append([]Message(nil), messages...))
	streamFn := m.streams[idx]
	m.mu.Unlock()

	ch := make(chan StreamEvent, 16)
	go func() {
		defer close(ch)
		streamFn(ch)
	}()
	return ch, nil
}

func (m *scriptedStreamModel) SupportsTools() bool { return true }

func (m *scriptedStreamModel) Request(i int) []Message {
	m.mu.Lock()
	defer m.mu.Unlock()
	if i < 0 || i >= len(m.requests) {
		return nil
	}
	return append([]Message(nil), m.requests[i]...)
}

func streamAssistantDone(text string, stop StopReason) func(chan<- StreamEvent) {
	return func(ch chan<- StreamEvent) {
		ch <- StreamEvent{
			Type: StreamEventDone,
			Message: Message{
				Role:       RoleAssistant,
				Content:    []ContentBlock{TextBlock(text)},
				StopReason: stop,
			},
		}
	}
}

func streamAssistantToolCalls(text string, stop StopReason, delay time.Duration, calls ...ToolCall) func(chan<- StreamEvent) {
	return func(ch chan<- StreamEvent) {
		partial := Message{Role: RoleAssistant, Content: []ContentBlock{TextBlock(text)}}
		ch <- StreamEvent{Type: StreamEventTextStart, Message: partial}
		for _, call := range calls {
			ch <- StreamEvent{Type: StreamEventToolCallStart, Message: partial}
			partial.Content = append(partial.Content, ToolCallBlock(call))
			completed := call
			ch <- StreamEvent{Type: StreamEventToolCallEnd, Message: partial, CompletedToolCall: &completed}
		}
		if delay > 0 {
			time.Sleep(delay)
		}
		content := []ContentBlock{TextBlock(text)}
		for _, call := range calls {
			content = append(content, ToolCallBlock(call))
		}
		ch <- StreamEvent{
			Type: StreamEventDone,
			Message: Message{
				Role:       RoleAssistant,
				Content:    content,
				StopReason: stop,
			},
		}
	}
}

func TestMarkLastMessageForCache_TagsLastNonSystemMessage(t *testing.T) {
	cases := []struct {
		name      string
		lastRole  Role
		buildLast func() Message
	}{
		{
			name:      "user input",
			lastRole:  RoleUser,
			buildLast: func() Message { return UserMsg("改一下登录") },
		},
		{
			name:      "tool result",
			lastRole:  RoleTool,
			buildLast: func() Message { return ToolResultMsg("call_1", []byte(`"ok"`), false) },
		},
		{
			name:     "assistant turn",
			lastRole: RoleAssistant,
			buildLast: func() Message {
				return Message{Role: RoleAssistant, Content: []ContentBlock{TextBlock("hi")}}
			},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			msgs := []Message{
				SystemMsg("sys"),
				UserMsg("first"),
				{Role: RoleAssistant, Content: []ContentBlock{TextBlock("a1")}},
				tc.buildLast(),
			}
			out := markLastMessageForCache(msgs, "ephemeral")

			last := out[len(out)-1]
			if last.Role != tc.lastRole {
				t.Fatalf("last message role: want %s, got %s", tc.lastRole, last.Role)
			}
			if last.Metadata["cache_control"] != "ephemeral" {
				t.Fatalf("expected cache_control on last %s message, got %v", tc.lastRole, last.Metadata)
			}
			for i := 0; i < len(out)-1; i++ {
				if _, has := out[i].Metadata["cache_control"]; has {
					t.Fatalf("unexpected cache_control on message %d (role=%s)", i, out[i].Role)
				}
			}
		})
	}
}

func TestMarkLastMessageForCache_SkipsTrailingSystemReminders(t *testing.T) {
	msgs := []Message{
		SystemMsg("sys"),
		UserMsg("real question"),
		SystemMsg("<system-reminder>per-turn reminder</system-reminder>"),
		SystemMsg("<system-reminder>another reminder</system-reminder>"),
	}
	out := markLastMessageForCache(msgs, "ephemeral")

	// Marker should land on the user message (index 1), not on the trailing system reminders.
	if out[1].Metadata["cache_control"] != "ephemeral" {
		t.Fatalf("expected cache_control on user (index 1), got metadata=%v", out[1].Metadata)
	}
	for i, m := range out {
		if i == 1 {
			continue
		}
		if _, has := m.Metadata["cache_control"]; has {
			t.Fatalf("unexpected cache_control on message %d (role=%s)", i, m.Role)
		}
	}
}

func TestMarkLastMessageForCache_DoesNotMutateInput(t *testing.T) {
	original := UserMsg("u")
	original.Metadata = map[string]any{"keep": "me"}
	msgs := []Message{original}

	out := markLastMessageForCache(msgs, "ephemeral")

	if _, has := msgs[0].Metadata["cache_control"]; has {
		t.Fatalf("input slice was mutated")
	}
	if msgs[0].Metadata["keep"] != "me" {
		t.Fatalf("input metadata mutated: %v", msgs[0].Metadata)
	}
	if out[0].Metadata["cache_control"] != "ephemeral" {
		t.Fatalf("output missing cache_control: %v", out[0].Metadata)
	}
	if out[0].Metadata["keep"] != "me" {
		t.Fatalf("output dropped pre-existing metadata: %v", out[0].Metadata)
	}
}

// streamErrModel returns an error from GenerateStream. Guards the contract
// that callLLMStream surfaces stream-init errors rather than silently
// falling back to non-streaming Generate.
type streamErrModel struct{ err error }

func (m *streamErrModel) Generate(context.Context, []Message, []ToolSpec, ...CallOption) (*LLMResponse, error) {
	return nil, fmt.Errorf("Generate should not be called when stream init fails")
}
func (m *streamErrModel) GenerateStream(context.Context, []Message, []ToolSpec, ...CallOption) (<-chan StreamEvent, error) {
	return nil, m.err
}
func (m *streamErrModel) SupportsTools() bool { return true }

func TestCallLLMStream_StreamInitErrorBubbles(t *testing.T) {
	m := &streamErrModel{err: fmt.Errorf("provider unavailable")}
	events := make(chan Event, 4)

	_, _, err := callLLMStream(context.Background(), m, nil, nil, nil, eventSink{ctx: context.Background(), ch: events}, llmCallHooks{})
	close(events)

	if err == nil {
		t.Fatal("expected stream init error, got nil")
	}
	if !strings.Contains(err.Error(), "stream init failed") || !strings.Contains(err.Error(), "provider unavailable") {
		t.Fatalf("expected wrapped stream init error, got %v", err)
	}
	for ev := range events {
		if ev.Type == EventMessageStart || ev.Type == EventMessageEnd {
			t.Fatalf("no message events expected on stream init failure, got %s", ev.Type)
		}
	}
}

// partialStreamModel emits text deltas then closes the channel WITHOUT a
// StreamEventDone — simulates network truncation / provider stream bug.
type partialStreamModel struct{ text string }

func (m *partialStreamModel) Generate(context.Context, []Message, []ToolSpec, ...CallOption) (*LLMResponse, error) {
	return nil, fmt.Errorf("Generate not used")
}
func (m *partialStreamModel) GenerateStream(context.Context, []Message, []ToolSpec, ...CallOption) (<-chan StreamEvent, error) {
	ch := make(chan StreamEvent, 4)
	partial := Message{Role: RoleAssistant, Content: []ContentBlock{TextBlock("")}}
	ch <- StreamEvent{Type: StreamEventTextStart, ContentIndex: 0, Message: partial}
	partial.Content[0].Text = m.text
	ch <- StreamEvent{Type: StreamEventTextDelta, ContentIndex: 0, Delta: m.text, Message: partial}
	close(ch) // no StreamEventDone — simulates truncation
	return ch, nil
}
func (m *partialStreamModel) SupportsTools() bool { return true }

func TestCallLLMStream_PartialStreamErrorOnTruncation(t *testing.T) {
	m := &partialStreamModel{text: "half a sente"}
	events := make(chan Event, 16)

	_, _, err := callLLMStream(context.Background(), m, nil, nil, nil, eventSink{ctx: context.Background(), ch: events}, llmCallHooks{})
	close(events)

	var partialErr *PartialStreamError
	if err == nil {
		t.Fatal("expected PartialStreamError, got nil")
	}
	if !errors.As(err, &partialErr) {
		t.Fatalf("expected PartialStreamError, got %T: %v", err, err)
	}
	if partialErr.Partial.TextContent() != "half a sente" {
		t.Fatalf("expected partial text preserved, got %q", partialErr.Partial.TextContent())
	}
	// Critically: no EventMessageEnd must be emitted — that would let callers
	// persist the half-finished message as if the LLM completed normally.
	for ev := range events {
		if ev.Type == EventMessageEnd {
			t.Fatal("EventMessageEnd must not fire on truncated stream — that would corrupt history")
		}
	}
}

// flakyStreamModel: first GenerateStream call truncates (no done event),
// subsequent calls succeed with a normal done event. Used to verify that
// callLLMWithRetry recognises *PartialStreamError as retryable — otherwise
// transient provider stream-format glitches would surface as hard failures.
type flakyStreamModel struct {
	calls int
	reply string
}

func (m *flakyStreamModel) Generate(context.Context, []Message, []ToolSpec, ...CallOption) (*LLMResponse, error) {
	return nil, fmt.Errorf("Generate not used")
}
func (m *flakyStreamModel) GenerateStream(_ context.Context, _ []Message, _ []ToolSpec, _ ...CallOption) (<-chan StreamEvent, error) {
	m.calls++
	ch := make(chan StreamEvent, 4)
	if m.calls == 1 {
		// Truncate: emit a delta then close without StreamEventDone.
		partial := Message{Role: RoleAssistant, Content: []ContentBlock{TextBlock("")}}
		ch <- StreamEvent{Type: StreamEventTextStart, ContentIndex: 0, Message: partial}
		partial.Content[0].Text = "partial..."
		ch <- StreamEvent{Type: StreamEventTextDelta, ContentIndex: 0, Delta: "partial...", Message: partial}
		close(ch)
		return ch, nil
	}
	// Success on retry.
	final := Message{
		Role:       RoleAssistant,
		Content:    []ContentBlock{TextBlock(m.reply)},
		StopReason: StopReasonStop,
	}
	ch <- StreamEvent{Type: StreamEventDone, Message: final, StopReason: StopReasonStop}
	close(ch)
	return ch, nil
}
func (m *flakyStreamModel) SupportsTools() bool { return true }

func TestCallLLMWithRetry_RetriesPartialStream(t *testing.T) {
	m := &flakyStreamModel{reply: "recovered"}
	events := make(chan Event, 32)
	defer close(events)

	cfg := LoopConfig{Model: m, MaxRetries: 2}
	agentCtx := &AgentContext{Messages: []AgentMessage{UserMsg("hi")}}

	msg, _, err := callLLMWithRetry(context.Background(), agentCtx, cfg, eventSink{ctx: context.Background(), ch: events}, llmCallHooks{}, nil)
	if err != nil {
		t.Fatalf("expected retry to succeed, got %v", err)
	}
	if msg.TextContent() != "recovered" {
		t.Fatalf("expected recovered message, got %q", msg.TextContent())
	}
	if m.calls != 2 {
		t.Fatalf("expected exactly 2 stream calls (1 partial + 1 success), got %d", m.calls)
	}
}

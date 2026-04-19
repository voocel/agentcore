package agentcore

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/voocel/agentcore/permission"
)

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
		StreamFn: func(ctx context.Context, req *LLMRequest) (*LLMResponse, error) {
			if got := req.Messages[0].TextContent(); got == original {
				t.Fatal("expected projected request to be trimmed before model call")
			}
			return &LLMResponse{Message: assistantMsg("ok", StopReasonStop)}, nil
		},
		CommitContext: func(msgs []AgentMessage, usage *ContextUsage) error {
			committed = copyMessages(msgs)
			return nil
		},
	}

	events := make(chan Event, 16)
	if _, _, err := callLLM(context.Background(), agentCtx, cfg, events, llmCallHooks{}, TurnInfo{}); err != nil {
		t.Fatalf("callLLM failed: %v", err)
	}

	if len(committed) == 0 {
		t.Fatal("expected projected context to be committed")
	}
	if agentCtx.Messages[0].TextContent() == original {
		t.Fatal("expected agent context baseline to be replaced with compacted messages")
	}
}

func TestAgentLoop_StrictMessageSequenceRejectsMalformedHistory(t *testing.T) {
	tc := ToolCall{ID: "tc-strict", Name: "echo", Args: json.RawMessage(`{"value":"x"}`)}
	var modelCalled atomic.Bool

	events := runTestLoop(t,
		[]AgentMessage{UserMsg("strict mode")},
		AgentContext{
			Messages: []AgentMessage{
				toolCallMsg(tc),
			},
		},
		LoopConfig{
			StreamFn: func(ctx context.Context, req *LLMRequest) (*LLMResponse, error) {
				modelCalled.Store(true)
				return nil, fmt.Errorf("model should not be called in strict mode")
			},
			StrictMessageSequence: true,
		},
	)

	if modelCalled.Load() {
		t.Fatal("expected strict message validation to fail before model invocation")
	}

	errEvent, ok := findEvent(events, EventError)
	if !ok || errEvent.Err == nil {
		t.Fatalf("expected strict message sequence error, got %+v", events)
	}
	if !strings.Contains(errEvent.Err.Error(), `missing tool result for "tc-strict"`) {
		t.Fatalf("unexpected strict mode error: %v", errEvent.Err)
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
		LoopConfig{StreamFn: mockStreamFn(responses...), MaxTurns: 3},
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
					StreamFn: func(ctx context.Context, req *LLMRequest) (*LLMResponse, error) {
						return nil, ctx.Err()
					},
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

func TestAgentLoop_Middleware(t *testing.T) {
	t.Run("plain tool", func(t *testing.T) {
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
	})

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
	})
}

func TestAgentLoop_PermissionEngine(t *testing.T) {
	t.Run("approval denied does not execute tool or disable it", func(t *testing.T) {
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
				PermissionEngine: permissionEngineFunc(func(ctx context.Context, req permission.Request) (*permission.Decision, error) {
					approvalChecks++
					return &permission.Decision{
						Kind:   permission.DecisionDeny,
						Source: permission.DecisionSourcePrompt,
						Reason: "denied",
					}, nil
				}),
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
	})

	t.Run("updated args are executed", func(t *testing.T) {
		var calls []string

		runTestLoop(t,
			[]AgentMessage{UserMsg("test")},
			AgentContext{Tools: []Tool{echoTool(&calls)}},
			LoopConfig{
				StreamFn: mockStreamFn(
					toolCallMsg(ToolCall{
						ID:   "tc1",
						Name: "echo",
						Args: json.RawMessage(`{"value":"original"}`),
					}),
					assistantMsg("done", StopReasonStop),
				),
				PermissionEngine: permissionEngineFunc(func(ctx context.Context, req permission.Request) (*permission.Decision, error) {
					return &permission.Decision{
						Kind:        permission.DecisionAllow,
						Source:      permission.DecisionSourceMode,
						UpdatedArgs: json.RawMessage(`{"value":"rewritten"}`),
					}, nil
				}),
			},
		)

		if len(calls) != 1 || calls[0] != "rewritten" {
			t.Fatalf("expected rewritten args to execute, got %v", calls)
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
	streamFn := sequentialStreamFn(func(i int, req *LLMRequest) (*LLMResponse, error) {
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
		LoopConfig{StreamFn: streamFn},
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
	streamFn := sequentialStreamFn(func(i int, req *LLMRequest) (*LLMResponse, error) {
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
			StreamFn:   streamFn,
			ToolChoice: "required",
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

func TestLoopPublicAPIs(t *testing.T) {
	t.Run("continue appends new messages", func(t *testing.T) {
		events := collectEvents(AgentLoopContinue(
			context.Background(),
			AgentContext{Messages: []AgentMessage{UserMsg("existing")}},
			LoopConfig{StreamFn: mockStreamFn(assistantMsg("continued", StopReasonStop))},
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
			LoopConfig{StreamFn: mockStreamFn()},
		))
		requireEvent(t, events, EventError)
		ev, _ := findEvent(events, EventAgentEnd)
		if ev.Summary == nil || ev.Summary.EndReason != EndReasonError {
			t.Fatalf("unexpected summary: %#v", ev.Summary)
		}
	})

	t.Run("collect returns final messages", func(t *testing.T) {
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
	})

	t.Run("event stream exposes events result and done", func(t *testing.T) {
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
	})
}

// ---------------------------------------------------------------------------
// OnMessage semantic tests — verify commitMessage fires on all paths
// ---------------------------------------------------------------------------

func TestOnMessage_AssistantAndToolResult(t *testing.T) {
	var calls []string
	tc := ToolCall{ID: "tc1", Name: "echo", Args: json.RawMessage(`{"value":"ping"}`)}

	var logged []Role
	var mu sync.Mutex

	runTestLoop(t,
		[]AgentMessage{UserMsg("test")},
		AgentContext{Tools: []Tool{echoTool(&calls)}},
		LoopConfig{
			StreamFn: mockStreamFn(toolCallMsg(tc), assistantMsg("done", StopReasonStop)),
			OnMessage: func(msg AgentMessage) {
				mu.Lock()
				logged = append(logged, msg.GetRole())
				mu.Unlock()
			},
		},
	)

	// Expect: user (prompt) → assistant (tool call) → tool (result) → assistant (done)
	mu.Lock()
	defer mu.Unlock()
	if len(logged) != 4 {
		t.Fatalf("expected 4 OnMessage calls, got %d: %v", len(logged), logged)
	}
	if logged[0] != RoleUser || logged[1] != RoleAssistant || logged[2] != RoleTool || logged[3] != RoleAssistant {
		t.Fatalf("unexpected OnMessage roles: %v", logged)
	}
}

func TestOnMessage_PendingSteeringMessages(t *testing.T) {
	var calls []string
	steeringDelivered := false

	var logged []Role
	var mu sync.Mutex

	runTestLoop(t,
		[]AgentMessage{UserMsg("test steering")},
		AgentContext{Tools: []Tool{echoTool(&calls)}},
		LoopConfig{
			StreamFn: sequentialStreamFn(func(i int, _ *LLMRequest) (*LLMResponse, error) {
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

func TestOnMessage_ErrorStopReason(t *testing.T) {
	var logged []Role
	var mu sync.Mutex

	runTestLoop(t,
		[]AgentMessage{UserMsg("test error")},
		AgentContext{},
		LoopConfig{
			StreamFn: mockStreamFn(assistantMsg("oops", StopReasonError)),
			OnMessage: func(msg AgentMessage) {
				mu.Lock()
				logged = append(logged, msg.GetRole())
				mu.Unlock()
			},
		},
	)

	mu.Lock()
	defer mu.Unlock()
	if len(logged) != 2 {
		t.Fatalf("expected 2 OnMessage calls (prompt + error assistant), got %d: %v", len(logged), logged)
	}
	if logged[0] != RoleUser || logged[1] != RoleAssistant {
		t.Fatalf("expected OnMessage roles [user assistant], got %v", logged)
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
			StreamFn: mockStreamFn(toolCallMsg(tc), assistantMsg("done", StopReasonStop)),
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

type permissionEngineFunc func(ctx context.Context, req permission.Request) (*permission.Decision, error)

func (fn permissionEngineFunc) Decide(ctx context.Context, req permission.Request) (*permission.Decision, error) {
	return fn(ctx, req)
}

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

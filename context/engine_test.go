package context

import (
	"context"
	"fmt"
	"strings"
	"testing"

	"github.com/voocel/agentcore"
)

// failingStrategy always returns an error, used for circuit breaker testing.
type failingStrategy struct{ callCount int }

func (s *failingStrategy) Name() string { return "failing" }
func (s *failingStrategy) Apply(_ context.Context, _, view []agentcore.AgentMessage, _ Budget) ([]agentcore.AgentMessage, StrategyResult, error) {
	s.callCount++
	return nil, StrategyResult{Name: s.Name()}, fmt.Errorf("simulated failure")
}

func TestContextEngineProjectUpdatesUsageFromProjectedView(t *testing.T) {
	engine := NewEngine(EngineConfig{
		ContextWindow: 1024,
		Strategies: []Strategy{
			NewLightTrim(LightTrimConfig{
				KeepRecent:    1,
				TextThreshold: 100,
				PreserveHead:  20,
				PreserveTail:  10,
			}),
		},
	})

	msgs := []agentcore.AgentMessage{
		agentcore.UserMsg(strings.Repeat("a", 800)),
		agentcore.UserMsg("recent"),
	}
	rawFirst := msgs[0].TextContent()

	proj, err := engine.Project(context.Background(), msgs)
	if err != nil {
		t.Fatalf("project failed: %v", err)
	}
	if len(proj.Messages) != len(msgs) {
		t.Fatalf("expected %d projected messages, got %d", len(msgs), len(proj.Messages))
	}
	if proj.Messages[0].TextContent() == rawFirst {
		t.Fatal("expected first message to be trimmed in projected view")
	}
	if msgs[0].TextContent() != rawFirst {
		t.Fatal("project mutated original input messages")
	}

	usage := engine.Usage()
	if usage == nil {
		t.Fatal("expected projected usage snapshot")
	}
	want := EstimateContextTokens(proj.Messages).Tokens
	if usage.Tokens != want {
		t.Fatalf("expected usage tokens=%d, got %d", want, usage.Tokens)
	}

	snapshot := engine.Snapshot()
	if snapshot == nil || snapshot.BaselineUsage == nil {
		t.Fatal("expected baseline usage snapshot")
	}
	baselineWant := EstimateContextTokens(msgs).Tokens
	if snapshot.BaselineUsage.Tokens != baselineWant {
		t.Fatalf("expected baseline usage tokens=%d, got %d", baselineWant, snapshot.BaselineUsage.Tokens)
	}
}

func TestContextEngineProjectCanRequestCommittedRewrite(t *testing.T) {
	engine := NewEngine(EngineConfig{
		ContextWindow:   1024,
		CommitOnProject: true,
		Strategies: []Strategy{
			NewLightTrim(LightTrimConfig{
				KeepRecent:    1,
				TextThreshold: 100,
				PreserveHead:  20,
				PreserveTail:  10,
			}),
		},
	})

	msgs := []agentcore.AgentMessage{
		agentcore.UserMsg(strings.Repeat("a", 800)),
		agentcore.UserMsg("recent"),
	}

	proj, err := engine.Project(context.Background(), msgs)
	if err != nil {
		t.Fatalf("project failed: %v", err)
	}
	if !proj.ShouldCommit {
		t.Fatal("expected project to request commit")
	}
	if len(proj.CommitMessages) != len(proj.Messages) {
		t.Fatalf("expected commit messages to mirror projected view, got %d vs %d", len(proj.CommitMessages), len(proj.Messages))
	}
	if proj.CommitMessages[0].TextContent() == msgs[0].TextContent() {
		t.Fatal("expected committed projection to be trimmed")
	}
}

func TestContextEngineProjectReportsSteps(t *testing.T) {
	var captured RewriteEvent
	engine := NewEngine(EngineConfig{
		ContextWindow: 1024,
		Strategies: []Strategy{
			NewToolResultMicrocompact(ToolResultMicrocompactConfig{KeepRecent: 0}),
			NewLightTrim(LightTrimConfig{
				KeepRecent:    1,
				TextThreshold: 100,
				PreserveHead:  20,
				PreserveTail:  10,
			}),
		},
		OnProject: func(ev RewriteEvent) { captured = ev },
	})

	msgs := []agentcore.AgentMessage{
		agentcore.UserMsg(strings.Repeat("a", 800)),
		agentcore.UserMsg("recent"),
	}

	if _, err := engine.Project(context.Background(), msgs); err != nil {
		t.Fatalf("project failed: %v", err)
	}
	if len(captured.Steps) == 0 {
		t.Fatal("expected steps to be populated in RewriteEvent")
	}
	// LightTrim should have applied (large text block), microcompact should not (no tool results)
	var trimStep *RewriteStep
	for i := range captured.Steps {
		if captured.Steps[i].Name == "light_trim" {
			trimStep = &captured.Steps[i]
		}
	}
	if trimStep == nil {
		t.Fatal("expected light_trim step in Steps")
	}
	if !trimStep.Applied {
		t.Fatal("expected light_trim step to be applied")
	}
	if trimStep.TokensAfter >= trimStep.TokensBefore {
		t.Fatalf("expected tokens to decrease after trim, got %d → %d", trimStep.TokensBefore, trimStep.TokensAfter)
	}
}

func TestContextEngineCompactProducesCommittedSummaryWithHooks(t *testing.T) {
	model := stubModel{
		generate: func(ctx context.Context, messages []agentcore.Message, tools []agentcore.ToolSpec, opts ...agentcore.CallOption) (*agentcore.LLMResponse, error) {
			return &agentcore.LLMResponse{
				Message: agentcore.Message{
					Role:    agentcore.RoleAssistant,
					Content: []agentcore.ContentBlock{agentcore.TextBlock("<analysis>scratch</analysis><summary>摘要内容</summary>")},
				},
			}, nil
		},
	}

	summary := NewFullSummary(FullSummaryConfig{
		Model:            model,
		KeepRecentTokens: 1,
		PostSummaryHooks: []PostSummaryHook{
			func(ctx context.Context, info SummaryInfo, kept []agentcore.AgentMessage) ([]agentcore.AgentMessage, error) {
				return []agentcore.AgentMessage{agentcore.UserMsg("hook reminder")}, nil
			},
		},
	})

	engine := NewEngine(EngineConfig{
		ContextWindow: 16,
		ReserveTokens: 4,
		Strategies: []Strategy{
			NewToolResultMicrocompact(ToolResultMicrocompactConfig{}),
			NewLightTrim(LightTrimConfig{}),
			summary,
		},
	})

	msgs := []agentcore.AgentMessage{
		agentcore.UserMsg(strings.Repeat("a", 120)),
		agentcore.UserMsg("keep"),
	}

	result, err := engine.Compact(context.Background(), msgs, agentcore.CompactReasonManual)
	if err != nil {
		t.Fatalf("compact failed: %v", err)
	}
	if !result.Changed {
		t.Fatal("expected committed compaction to change messages")
	}
	if len(result.Messages) < 3 {
		t.Fatalf("expected summary + hook + kept messages, got %d entries", len(result.Messages))
	}
	if _, ok := result.Messages[0].(ContextSummary); !ok {
		t.Fatalf("expected first message to be ContextSummary, got %T", result.Messages[0])
	}
	if got := result.Messages[1].TextContent(); got != "hook reminder" {
		t.Fatalf("expected hook reminder after summary, got %q", got)
	}
}

func TestContextEngineForceCompactSummarizesOriginalTranscript(t *testing.T) {
	var sawClearedPlaceholder bool
	var sawOriginalToolResult bool
	model := stubModel{
		generate: func(ctx context.Context, messages []agentcore.Message, tools []agentcore.ToolSpec, opts ...agentcore.CallOption) (*agentcore.LLMResponse, error) {
			for _, msg := range messages {
				text := msg.TextContent()
				if strings.Contains(text, defaultClearedToolResult) {
					sawClearedPlaceholder = true
				}
				if strings.Contains(text, "VERY_IMPORTANT_TOOL_RESULT") {
					sawOriginalToolResult = true
				}
			}
			return &agentcore.LLMResponse{
				Message: agentcore.Message{
					Role:    agentcore.RoleAssistant,
					Content: []agentcore.ContentBlock{agentcore.TextBlock("<summary>摘要内容</summary>")},
				},
			}, nil
		},
	}

	engine := NewEngine(EngineConfig{
		ContextWindow: 16,
		ReserveTokens: 4,
		Strategies: []Strategy{
			NewToolResultMicrocompact(ToolResultMicrocompactConfig{
				KeepRecent: 0,
			}),
			NewFullSummary(FullSummaryConfig{
				Model:            model,
				KeepRecentTokens: 1,
			}),
		},
	})

	msgs := []agentcore.AgentMessage{
		agentcore.Message{
			Role: agentcore.RoleAssistant,
			Content: []agentcore.ContentBlock{
				agentcore.ToolCallBlock(agentcore.ToolCall{ID: "tc1", Name: "read", Args: []byte(`{"path":"main.go"}`)}),
			},
		},
		agentcore.ToolResultMsg("tc1", []byte(`"VERY_IMPORTANT_TOOL_RESULT"`), false),
		agentcore.UserMsg("keep"),
	}

	result, err := engine.Compact(context.Background(), msgs, agentcore.CompactReasonManual)
	if err != nil {
		t.Fatalf("compact failed: %v", err)
	}
	if !result.Changed {
		t.Fatal("expected forced compact to change messages")
	}
	if sawClearedPlaceholder {
		t.Fatal("summary model saw cleared placeholder; force compact should summarize original transcript")
	}
	if !sawOriginalToolResult {
		t.Fatal("summary model did not see original tool result")
	}
}

func TestContextEngineSnapshotTracksActiveViewAndLastRewrite(t *testing.T) {
	engine := NewEngine(EngineConfig{
		ContextWindow: 1024,
		Strategies: []Strategy{
			NewLightTrim(LightTrimConfig{
				KeepRecent:    1,
				TextThreshold: 100,
				PreserveHead:  20,
				PreserveTail:  10,
			}),
		},
	})

	msgs := []agentcore.AgentMessage{
		agentcore.UserMsg(strings.Repeat("a", 800)),
		agentcore.UserMsg("recent"),
	}

	if _, err := engine.Project(context.Background(), msgs); err != nil {
		t.Fatalf("project failed: %v", err)
	}

	snapshot := engine.Snapshot()
	if snapshot == nil {
		t.Fatal("expected context snapshot after project")
	}
	if snapshot.Scope != "projected" {
		t.Fatalf("expected projected scope, got %q", snapshot.Scope)
	}
	if snapshot.LastStrategy != "light_trim" {
		t.Fatalf("expected last strategy light_trim, got %q", snapshot.LastStrategy)
	}
	if snapshot.TrimmedTextBlocks == 0 {
		t.Fatal("expected trimmed text blocks to be counted")
	}
	if !snapshot.LastChanged {
		t.Fatal("expected last rewrite to be marked changed")
	}

	engine.Sync(msgs)
	snapshot = engine.Snapshot()
	if snapshot == nil {
		t.Fatal("expected context snapshot after sync")
	}
	if snapshot.Scope != "baseline" {
		t.Fatalf("expected baseline scope after sync, got %q", snapshot.Scope)
	}
	if snapshot.LastStrategy != "light_trim" {
		t.Fatalf("expected last strategy to be preserved after sync, got %q", snapshot.LastStrategy)
	}
}

func TestContextEngineProjectDoesNotMutateOriginalMessageMetadata(t *testing.T) {
	engine := NewEngine(EngineConfig{
		ContextWindow: 64,
		ReserveTokens: 4,
		Strategies: []Strategy{
			NewToolResultMicrocompact(ToolResultMicrocompactConfig{
				KeepRecent: 1,
			}),
			NewLightTrim(LightTrimConfig{
				KeepRecent:    1,
				TextThreshold: 100,
				PreserveHead:  20,
				PreserveTail:  10,
			}),
		},
	})

	assistant := agentcore.Message{
		Role: agentcore.RoleAssistant,
		Content: []agentcore.ContentBlock{
			agentcore.ToolCallBlock(agentcore.ToolCall{ID: "tc1", Name: "read", Args: []byte(`{"path":"a.go"}`)}),
			agentcore.ToolCallBlock(agentcore.ToolCall{ID: "tc2", Name: "read", Args: []byte(`{"path":"b.go"}`)}),
		},
		Metadata: map[string]any{"source": "assistant"},
	}
	tool1 := agentcore.ToolResultMsg("tc1", []byte(`"`+strings.Repeat("x", 500)+`"`), false)
	tool1.Metadata["existing"] = "keep"
	tool2 := agentcore.ToolResultMsg("tc2", []byte(`"SECOND_RESULT"`), false)
	tool2.Metadata["existing"] = "keep"
	msgs := []agentcore.AgentMessage{
		assistant,
		tool1,
		tool2,
		agentcore.UserMsg("recent"),
	}

	if _, err := engine.Project(context.Background(), msgs); err != nil {
		t.Fatalf("project failed: %v", err)
	}

	origTool1 := msgs[1].(agentcore.Message)
	if _, ok := origTool1.Metadata["compacted_tool_result"]; ok {
		t.Fatal("project mutated original tool result metadata")
	}
	if _, ok := origTool1.Metadata["trimmed_text_blocks"]; ok {
		t.Fatal("project mutated original trimmed metadata")
	}
	if got := origTool1.Metadata["existing"]; got != "keep" {
		t.Fatalf("expected original metadata to stay intact, got %v", got)
	}

	origAssistant := msgs[0].(agentcore.Message)
	if got := origAssistant.Metadata["source"]; got != "assistant" {
		t.Fatalf("expected assistant metadata to stay intact, got %v", got)
	}
}

func TestContextEngineSyncPreservesBaselineScope(t *testing.T) {
	engine := NewEngine(EngineConfig{ContextWindow: 1024})
	msgs := []agentcore.AgentMessage{agentcore.UserMsg("done")}

	engine.Sync(msgs)

	snapshot := engine.Snapshot()
	if snapshot == nil {
		t.Fatal("expected snapshot after sync")
	}
	if snapshot.Scope != "baseline" {
		t.Fatalf("expected baseline scope, got %q", snapshot.Scope)
	}
}

func TestContextConvertToLLM_WrapsSummary(t *testing.T) {
	msgs := []agentcore.AgentMessage{
		ContextSummary{
			Summary:       "summary body",
			TokensBefore:  42,
			ReadFiles:     []string{"a.go"},
			ModifiedFiles: []string{"b.go"},
		},
		agentcore.UserMsg("keep me"),
	}

	out := ContextConvertToLLM(msgs)
	if len(out) != 2 {
		t.Fatalf("expected 2 messages, got %d", len(out))
	}
	if out[0].Role != agentcore.RoleUser {
		t.Fatalf("expected wrapped summary role=user, got %s", out[0].Role)
	}
	if got := out[0].TextContent(); !strings.Contains(got, "<context-summary>\nsummary body\n</context-summary>") {
		t.Fatalf("unexpected wrapped summary: %q", got)
	}
	if got := out[0].Metadata["type"]; got != "context_summary" {
		t.Fatalf("expected context summary metadata marker, got %v", got)
	}
}

func TestCircuitBreaker_TripsAfterConsecutiveFailures(t *testing.T) {
	fs := &failingStrategy{}
	var gotEvent *RewriteEvent
	engine := NewEngine(EngineConfig{
		ContextWindow:          100,
		ReserveTokens:          1,
		Strategies:             []Strategy{fs},
		MaxConsecutiveFailures: 2,
		OnProject: func(ev RewriteEvent) {
			gotEvent = &ev
		},
	})

	msgs := []agentcore.AgentMessage{agentcore.UserMsg(strings.Repeat("x", 500))}

	for i := 0; i < 2; i++ {
		_, err := engine.Project(context.Background(), msgs)
		if err == nil {
			t.Fatalf("call %d: expected error", i+1)
		}
	}
	if engine.ConsecutiveFailures() != 2 {
		t.Fatalf("expected 2 consecutive failures, got %d", engine.ConsecutiveFailures())
	}

	// Third call: circuit breaker trips, returns original msgs, fires event
	proj, err := engine.Project(context.Background(), msgs)
	if err != nil {
		t.Fatalf("expected circuit breaker to skip apply, got error: %v", err)
	}
	if len(proj.Messages) != 1 {
		t.Fatalf("expected original messages returned, got %d", len(proj.Messages))
	}
	if fs.callCount != 2 {
		t.Fatalf("expected strategy called 2 times, got %d", fs.callCount)
	}
	// Verify the circuit breaker event was fired
	if gotEvent == nil {
		t.Fatal("expected OnProject event when circuit breaker trips")
	}
	if gotEvent.Reason != "circuit_breaker" {
		t.Fatalf("expected reason=circuit_breaker, got %q", gotEvent.Reason)
	}
	if gotEvent.Failures != 2 {
		t.Fatalf("expected failures=2, got %d", gotEvent.Failures)
	}
	if gotEvent.Changed {
		t.Fatal("expected Changed=false for circuit breaker event")
	}
	snap := engine.Snapshot()
	if snap == nil || snap.Scope != "skipped" {
		t.Fatalf("expected snapshot scope=skipped, got %+v", snap)
	}
	if engine.ConsecutiveFailures() != 1 {
		t.Fatalf("expected breaker to re-arm at 1 failure, got %d", engine.ConsecutiveFailures())
	}
}

func TestCircuitBreaker_ResetsOnSuccess(t *testing.T) {
	fs := &failingStrategy{}
	engine := NewEngine(EngineConfig{
		ContextWindow:          100,
		ReserveTokens:          1,
		Strategies:             []Strategy{fs},
		MaxConsecutiveFailures: 3,
	})

	msgs := []agentcore.AgentMessage{agentcore.UserMsg(strings.Repeat("x", 500))}

	// Fail twice
	for i := 0; i < 2; i++ {
		engine.Project(context.Background(), msgs)
	}
	if engine.ConsecutiveFailures() != 2 {
		t.Fatalf("expected 2, got %d", engine.ConsecutiveFailures())
	}

	// Simulate recovery via RecoverOverflow with a no-op engine (success path)
	successEngine := NewEngine(EngineConfig{
		ContextWindow:          100000,
		ReserveTokens:          1,
		Strategies:             []Strategy{NewLightTrim(LightTrimConfig{})},
		MaxConsecutiveFailures: 3,
	})
	// Manually set failure count
	successEngine.mu.Lock()
	successEngine.consecutiveFailures = 2
	successEngine.mu.Unlock()

	// RecoverOverflow uses ForceApply, which requires ForceCompactionStrategy.
	// LightTrim doesn't implement ForceApply, so use a strategy that does.
	// Instead, directly test the reset mechanism: simulate a successful recovery
	// by calling RecoverOverflow on an engine with a large text that triggers LightTrim.
	// LightTrim doesn't implement ForceCompactionStrategy so won't run in force mode.
	// Use a direct mutation to verify the reset path:
	successEngine.mu.Lock()
	successEngine.consecutiveFailures = 2
	successEngine.mu.Unlock()

	// Successful Project with Changed=true resets the counter
	// Use small context window to force LightTrim to trigger
	trimEngine := NewEngine(EngineConfig{
		ContextWindow: 64,
		ReserveTokens: 1,
		Strategies: []Strategy{NewLightTrim(LightTrimConfig{
			KeepRecent:    1,
			TextThreshold: 100,
			PreserveHead:  20,
			PreserveTail:  10,
		})},
		MaxConsecutiveFailures: 3,
	})
	trimEngine.mu.Lock()
	trimEngine.consecutiveFailures = 2
	trimEngine.mu.Unlock()

	bigMsgs := []agentcore.AgentMessage{agentcore.UserMsg(strings.Repeat("a", 800)), agentcore.UserMsg("recent")}
	_, err := trimEngine.Project(context.Background(), bigMsgs)
	if err != nil {
		t.Fatalf("project failed: %v", err)
	}
	if trimEngine.ConsecutiveFailures() != 0 {
		t.Fatalf("expected reset to 0 after successful compression, got %d", trimEngine.ConsecutiveFailures())
	}
}

func TestCircuitBreaker_AllowsRetryAfterSkippedCycle(t *testing.T) {
	fs := &failingStrategy{}
	engine := NewEngine(EngineConfig{
		ContextWindow:          100,
		ReserveTokens:          1,
		Strategies:             []Strategy{fs},
		MaxConsecutiveFailures: 2,
	})

	msgs := []agentcore.AgentMessage{agentcore.UserMsg(strings.Repeat("x", 500))}

	for i := 0; i < 2; i++ {
		if _, err := engine.Project(context.Background(), msgs); err == nil {
			t.Fatalf("call %d: expected error", i+1)
		}
	}
	if _, err := engine.Project(context.Background(), msgs); err != nil {
		t.Fatalf("expected skipped cycle, got error: %v", err)
	}
	if fs.callCount != 2 {
		t.Fatalf("expected 2 strategy calls after skipped cycle, got %d", fs.callCount)
	}

	if _, err := engine.Project(context.Background(), msgs); err == nil {
		t.Fatal("expected retry attempt after skipped cycle to call strategy and fail")
	}
	if fs.callCount != 3 {
		t.Fatalf("expected retry to call strategy again, got %d calls", fs.callCount)
	}
}

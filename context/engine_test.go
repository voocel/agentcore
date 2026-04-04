package context

import (
	"context"
	"strings"
	"testing"

	"github.com/voocel/agentcore"
)

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

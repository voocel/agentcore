package memory

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
		PostCompactHooks: []PostCompactHook{
			func(ctx context.Context, info CompactionInfo, kept []agentcore.AgentMessage) ([]agentcore.AgentMessage, error) {
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
	if _, ok := result.Messages[0].(CompactionSummary); !ok {
		t.Fatalf("expected first message to be CompactionSummary, got %T", result.Messages[0])
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

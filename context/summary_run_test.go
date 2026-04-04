package context

import (
	"context"
	"strings"
	"testing"

	"github.com/voocel/agentcore"
)

type stubModel struct {
	generate func(ctx context.Context, messages []agentcore.Message, tools []agentcore.ToolSpec, opts ...agentcore.CallOption) (*agentcore.LLMResponse, error)
}

func (m stubModel) Generate(ctx context.Context, messages []agentcore.Message, tools []agentcore.ToolSpec, opts ...agentcore.CallOption) (*agentcore.LLMResponse, error) {
	return m.generate(ctx, messages, tools, opts...)
}

func (m stubModel) GenerateStream(ctx context.Context, messages []agentcore.Message, tools []agentcore.ToolSpec, opts ...agentcore.CallOption) (<-chan agentcore.StreamEvent, error) {
	return nil, context.Canceled
}

func (m stubModel) SupportsTools() bool { return true }

func TestFindCutPoint_SkipsToolResultBoundary(t *testing.T) {
	msgs := []agentcore.AgentMessage{
		agentcore.UserMsg("old"),
		agentcore.Message{
			Role:    agentcore.RoleAssistant,
			Content: []agentcore.ContentBlock{agentcore.ToolCallBlock(agentcore.ToolCall{ID: "tc1", Name: "read"})},
		},
		agentcore.ToolResultMsg("tc1", []byte(`"ok"`), false),
		agentcore.UserMsg("recent"),
	}

	cut := findCutPoint(msgs, 2)
	if cut.firstKeptIndex != 3 {
		t.Fatalf("expected cut to advance past tool result to index 3, got %d", cut.firstKeptIndex)
	}
	if cut.isSplitTurn {
		t.Fatal("expected cut at user boundary, got split turn")
	}
}

func TestFindCutPoint_ReportsSplitTurn(t *testing.T) {
	msgs := []agentcore.AgentMessage{
		agentcore.UserMsg("old"),
		agentcore.Message{Role: agentcore.RoleAssistant, Content: []agentcore.ContentBlock{agentcore.TextBlock("done")}},
		agentcore.UserMsg("current task"),
		agentcore.Message{Role: agentcore.RoleAssistant, Content: []agentcore.ContentBlock{agentcore.TextBlock("working")}},
	}

	cut := findCutPoint(msgs, 1)
	if cut.firstKeptIndex != 3 {
		t.Fatalf("expected assistant message to be first kept item, got %d", cut.firstKeptIndex)
	}
	if !cut.isSplitTurn {
		t.Fatal("expected split turn to be reported")
	}
	if cut.turnStartIndex != 2 {
		t.Fatalf("expected split turn to start at index 2, got %d", cut.turnStartIndex)
	}
}

func TestExtractFileOps_DeduplicatesAndSeparates(t *testing.T) {
	msgs := []agentcore.AgentMessage{
		agentcore.Message{
			Role: agentcore.RoleAssistant,
			Content: []agentcore.ContentBlock{
				agentcore.ToolCallBlock(agentcore.ToolCall{ID: "1", Name: "read", Args: []byte(`{"path":"a.go"}`)}),
				agentcore.ToolCallBlock(agentcore.ToolCall{ID: "2", Name: "read", Args: []byte(`{"path":"b.go"}`)}),
				agentcore.ToolCallBlock(agentcore.ToolCall{ID: "3", Name: "edit", Args: []byte(`{"path":"b.go"}`)}),
				agentcore.ToolCallBlock(agentcore.ToolCall{ID: "4", Name: "write", Args: []byte(`{"path":"c.go"}`)}),
				agentcore.ToolCallBlock(agentcore.ToolCall{ID: "5", Name: "read", Args: []byte(`{"path":"a.go"}`)}),
			},
		},
	}

	readFiles, modifiedFiles := extractFileOps(msgs)
	if got := strings.Join(readFiles, ","); got != "a.go" {
		t.Fatalf("expected read-only files to be a.go, got %q", got)
	}
	if got := strings.Join(modifiedFiles, ","); got != "b.go,c.go" {
		t.Fatalf("expected modified files to be b.go,c.go, got %q", got)
	}
}

func TestRunSummaryCompaction_CompactsAndPreservesRecentMessages(t *testing.T) {
	model := stubModel{
		generate: func(ctx context.Context, messages []agentcore.Message, tools []agentcore.ToolSpec, opts ...agentcore.CallOption) (*agentcore.LLMResponse, error) {
			return &agentcore.LLMResponse{
				Message: agentcore.Message{
					Role:    agentcore.RoleAssistant,
					Content: []agentcore.ContentBlock{agentcore.TextBlock("摘要内容")},
				},
			}, nil
		},
	}

	cfg := summaryRunConfig{
		Model:            model,
		ContextWindow:    16,
		ReserveTokens:    4,
		KeepRecentTokens: 1,
	}

	msgs := []agentcore.AgentMessage{
		agentcore.UserMsg(strings.Repeat("a", 80)),
		agentcore.Message{
			Role: agentcore.RoleAssistant,
			Content: []agentcore.ContentBlock{
				agentcore.ToolCallBlock(agentcore.ToolCall{ID: "1", Name: "read", Args: []byte(`{"path":"old.go"}`)}),
				agentcore.ToolCallBlock(agentcore.ToolCall{ID: "2", Name: "edit", Args: []byte(`{"path":"new.go"}`)}),
			},
		},
		agentcore.UserMsg("keep"),
	}

	out, info, err := runSummaryCompaction(context.Background(), cfg, msgs, true)
	if err != nil {
		t.Fatalf("unexpected compaction error: %v", err)
	}
	if info == nil {
		t.Fatal("expected compaction info")
	}
	if len(out) != 2 {
		t.Fatalf("expected compacted summary + recent message, got %d entries", len(out))
	}

	summary, ok := out[0].(ContextSummary)
	if !ok {
		t.Fatalf("expected first message to be ContextSummary, got %T", out[0])
	}
	if !strings.Contains(summary.Summary, "摘要内容") {
		t.Fatalf("expected generated summary content, got %q", summary.Summary)
	}
	if !strings.Contains(summary.Summary, "<read-files>\nold.go\n</read-files>") {
		t.Fatalf("expected read file section, got %q", summary.Summary)
	}
	if !strings.Contains(summary.Summary, "<modified-files>\nnew.go\n</modified-files>") {
		t.Fatalf("expected modified file section, got %q", summary.Summary)
	}
	if out[1].TextContent() != "keep" {
		t.Fatalf("expected recent message to be preserved, got %q", out[1].TextContent())
	}
}

func TestEstimateContextTokens_UsesLastAssistantUsage(t *testing.T) {
	msgs := []agentcore.AgentMessage{
		agentcore.UserMsg("before"),
		agentcore.Message{
			Role: agentcore.RoleAssistant,
			Content: []agentcore.ContentBlock{
				agentcore.TextBlock("done"),
			},
			Usage: &agentcore.Usage{TotalTokens: 100},
		},
		agentcore.UserMsg(strings.Repeat("x", 20)),
	}

	estimate := EstimateContextTokens(msgs)
	if estimate.UsageTokens != 100 {
		t.Fatalf("expected usage tokens=100, got %d", estimate.UsageTokens)
	}
	if estimate.TrailingTokens == 0 {
		t.Fatal("expected trailing tokens to be estimated")
	}
	if estimate.Tokens != estimate.UsageTokens+estimate.TrailingTokens {
		t.Fatalf("unexpected total tokens: %+v", estimate)
	}
}

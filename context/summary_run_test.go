package context

import (
	"context"
	"strconv"
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

func toolGroup(id string, result string) []agentcore.AgentMessage {
	return []agentcore.AgentMessage{
		agentcore.Message{
			Role:    agentcore.RoleAssistant,
			Content: []agentcore.ContentBlock{agentcore.ToolCallBlock(agentcore.ToolCall{ID: id, Name: "read"})},
		},
		agentcore.ToolResultMsg(id, []byte(strconv.Quote(result)), false),
	}
}

func TestFindCutPoint_StopsAtUserBoundary(t *testing.T) {
	msgs := []agentcore.AgentMessage{
		agentcore.UserMsg("old"),
		agentcore.Message{Role: agentcore.RoleAssistant, Content: []agentcore.ContentBlock{agentcore.TextBlock("done")}},
		agentcore.UserMsg(strings.Repeat("recent", 100)),
	}

	cut := FindCutPoint(msgs, 10)
	if cut.FirstKeptIndex != 2 || cut.IsSplitTurn || cut.TurnStartIndex != -1 {
		t.Fatalf("expected clean cut at user index 2, got %+v", cut)
	}
}

func TestFindCutPoint_ReportsSplitTurn(t *testing.T) {
	msgs := []agentcore.AgentMessage{
		agentcore.UserMsg("old"),
		agentcore.Message{Role: agentcore.RoleAssistant, Content: []agentcore.ContentBlock{agentcore.TextBlock("done")}},
		agentcore.UserMsg("current task"),
		agentcore.Message{Role: agentcore.RoleAssistant, Content: []agentcore.ContentBlock{agentcore.TextBlock("working")}},
	}

	cut := FindCutPoint(msgs, 1)
	if cut.FirstKeptIndex != 3 {
		t.Fatalf("expected assistant message to be first kept item, got %d", cut.FirstKeptIndex)
	}
	if !cut.IsSplitTurn || cut.TurnStartIndex != 2 {
		t.Fatalf("expected split turn starting at index 2, got %+v", cut)
	}
}

// A cut landing on a tool result retreats to the assistant call that issued
// it, so parallel results are never separated from their call.
func TestFindCutPoint_RetreatsToToolCallOnResult(t *testing.T) {
	msgs := []agentcore.AgentMessage{
		agentcore.UserMsg(strings.Repeat("b", 400)),
		agentcore.Message{
			Role: agentcore.RoleAssistant,
			Content: []agentcore.ContentBlock{
				agentcore.ToolCallBlock(agentcore.ToolCall{ID: "1", Name: "read"}),
				agentcore.ToolCallBlock(agentcore.ToolCall{ID: "2", Name: "read"}),
			},
		},
		agentcore.ToolResultMsg("1", []byte(strconv.Quote(strings.Repeat("a", 400))), false),
		agentcore.ToolResultMsg("2", []byte(strconv.Quote(strings.Repeat("a", 400))), false),
		agentcore.UserMsg("recent"),
	}

	cut := FindCutPoint(msgs, 120)
	if cut.FirstKeptIndex != 1 {
		t.Fatalf("expected cut to retreat to the tool call at index 1, got %+v", cut)
	}
	if !cut.IsSplitTurn || cut.TurnStartIndex != 0 {
		t.Fatalf("expected split turn starting at index 0, got %+v", cut)
	}
}

// Sub-agent runs are one user task followed by tool groups only; the cut must
// still land inside the loop instead of walking off the end.
func TestFindCutPoint_ToolLoopWithoutTextTurns(t *testing.T) {
	msgs := []agentcore.AgentMessage{agentcore.UserMsg("task")}
	for i := 1; i <= 4; i++ {
		msgs = append(msgs, toolGroup(strconv.Itoa(i), strings.Repeat("a", 400))...)
	}

	cut := FindCutPoint(msgs, 150)
	if cut.FirstKeptIndex != 5 {
		t.Fatalf("expected cut at the third tool call (index 5), got %+v", cut)
	}
	if m, ok := msgs[cut.FirstKeptIndex].(agentcore.Message); !ok || !m.HasToolCalls() {
		t.Fatal("kept suffix must start with a tool call")
	}
	if !cut.IsSplitTurn || cut.TurnStartIndex != 0 {
		t.Fatalf("expected split turn from the task message, got %+v", cut)
	}
}

func TestFindCutPoint_NoCutWhenNothingPrecedesKeptGroup(t *testing.T) {
	msgs := append([]agentcore.AgentMessage{agentcore.UserMsg("task")}, toolGroup("1", strings.Repeat("a", 400))...)
	if cut := FindCutPoint(msgs, 10000); cut.FirstKeptIndex != 0 {
		t.Fatalf("suffix covering everything must report no cut, got %+v", cut)
	}
	if cut := FindCutPoint(msgs[1:], 50); cut.FirstKeptIndex != 0 {
		t.Fatalf("retreating to a tool call at index 0 must report no cut, got %+v", cut)
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
					Content: []agentcore.ContentBlock{agentcore.TextBlock("checkpoint body")},
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
	if !strings.Contains(summary.Summary, "checkpoint body") {
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

// forkTestMessages forces a cut while retaining a tail.
func forkTestMessages() []agentcore.AgentMessage {
	return []agentcore.AgentMessage{
		agentcore.UserMsg(strings.Repeat("a", 80)),
		agentcore.Message{
			Role:    agentcore.RoleAssistant,
			Content: []agentcore.ContentBlock{agentcore.TextBlock("worked on it")},
		},
		agentcore.UserMsg("keep"),
	}
}

func summaryReplyModel(record func([]agentcore.Message, []agentcore.ToolSpec), texts ...string) *stubModel {
	call := 0
	return &stubModel{
		generate: func(_ context.Context, messages []agentcore.Message, tools []agentcore.ToolSpec, _ ...agentcore.CallOption) (*agentcore.LLMResponse, error) {
			if record != nil {
				record(messages, tools)
			}
			text := texts[min(call, len(texts)-1)]
			call++
			return &agentcore.LLMResponse{Message: agentcore.Message{
				Role:    agentcore.RoleAssistant,
				Content: []agentcore.ContentBlock{agentcore.TextBlock(text)},
			}}, nil
		},
	}
}

func TestRunSummaryCompaction_ForkReusesParentPrefix(t *testing.T) {
	var sent [][]agentcore.Message
	var sentTools [][]agentcore.ToolSpec
	model := summaryReplyModel(func(m []agentcore.Message, tools []agentcore.ToolSpec) {
		sent = append(sent, m)
		sentTools = append(sentTools, tools)
	}, "<analysis>thinking</analysis>\n<summary>checkpoint body</summary>")

	prefix := agentcore.LLMPrefix{
		System:       []agentcore.Message{agentcore.SystemMsg("parent system")},
		Tools:        []agentcore.ToolSpec{{Name: "read"}},
		Model:        model,
		CacheControl: "ephemeral",
	}
	cfg := summaryRunConfig{
		Model:            model,
		ContextWindow:    16,
		ReserveTokens:    4,
		KeepRecentTokens: 1,
		Fork:             &prefix,
	}

	msgs := forkTestMessages()
	out, info, err := runSummaryCompaction(context.Background(), cfg, msgs, true)
	if err != nil {
		t.Fatalf("unexpected compaction error: %v", err)
	}
	if info == nil {
		t.Fatal("expected compaction info")
	}
	if len(sent) != 1 {
		t.Fatalf("expected a single forked call, got %d", len(sent))
	}
	req := sent[0]

	if len(sentTools[0]) != 1 || sentTools[0][0].Name != "read" {
		t.Fatalf("fork must send the parent tool list, got %+v", sentTools[0])
	}
	if req[0].Role != agentcore.RoleSystem || req[0].TextContent() != "parent system" {
		t.Fatalf("fork must lead with the parent system message, got %+v", req[0])
	}
	// system + 3 conversation messages + instruction
	if len(req) != 5 {
		t.Fatalf("expected the whole view sent as real messages, got %d", len(req))
	}
	if !strings.Contains(req[3].TextContent(), "keep") {
		t.Fatalf("expected the verbatim tail to ride along, got %q", req[3].TextContent())
	}
	if req[3].Metadata["cache_control"] != "ephemeral" {
		t.Fatalf("expected the breakpoint on the last shared message, got %+v", req[3].Metadata)
	}
	last := req[len(req)-1]
	if last.Role != agentcore.RoleUser || !strings.HasPrefix(last.TextContent(), "Stop working on the task") {
		t.Fatalf("expected the guarded instruction last, got %+v", last)
	}

	summary, ok := out[0].(ContextSummary)
	if !ok || !strings.Contains(summary.Summary, "checkpoint body") {
		t.Fatalf("expected the forked summary to be used, got %+v", out[0])
	}
}

func TestRunSummaryCompaction_ForkFallsBackWhenModelAnswersInstead(t *testing.T) {
	var sent [][]agentcore.Message
	// Untagged task prose must not become the checkpoint.
	model := summaryReplyModel(func(m []agentcore.Message, _ []agentcore.ToolSpec) {
		sent = append(sent, m)
	}, "Sure — the bug is in the retry loop, I'll fix it next.", "checkpoint body")

	prefix := agentcore.LLMPrefix{System: []agentcore.Message{agentcore.SystemMsg("parent system")}, Model: model}
	cfg := summaryRunConfig{
		Model:            model,
		ContextWindow:    16,
		ReserveTokens:    4,
		KeepRecentTokens: 1,
		Fork:             &prefix,
	}

	out, _, err := runSummaryCompaction(context.Background(), cfg, forkTestMessages(), true)
	if err != nil {
		t.Fatalf("unexpected compaction error: %v", err)
	}
	if len(sent) != 2 {
		t.Fatalf("expected the fork to fall back to one standalone call, got %d calls", len(sent))
	}
	if !strings.Contains(sent[1][1].TextContent(), "<conversation>") {
		t.Fatalf("expected the fallback to use the standalone prompt, got %q", sent[1][1].TextContent())
	}
	summary, ok := out[0].(ContextSummary)
	if !ok || !strings.Contains(summary.Summary, "checkpoint body") {
		t.Fatalf("expected the fallback summary to be used, got %+v", out[0])
	}
}

func TestRunSummaryCompaction_SkipsForkOnModelMismatch(t *testing.T) {
	var sent [][]agentcore.Message
	model := summaryReplyModel(func(m []agentcore.Message, _ []agentcore.ToolSpec) {
		sent = append(sent, m)
	}, "checkpoint body")

	// A different model cannot reuse the prefix cache.
	prefix := agentcore.LLMPrefix{
		System: []agentcore.Message{agentcore.SystemMsg("parent system")},
		Model:  summaryReplyModel(nil, "unused"),
	}
	cfg := summaryRunConfig{
		Model:            model,
		ContextWindow:    16,
		ReserveTokens:    4,
		KeepRecentTokens: 1,
		Fork:             &prefix,
	}

	if _, _, err := runSummaryCompaction(context.Background(), cfg, forkTestMessages(), true); err != nil {
		t.Fatalf("unexpected compaction error: %v", err)
	}
	if len(sent) != 1 {
		t.Fatalf("expected a single standalone call, got %d", len(sent))
	}
	if !strings.Contains(sent[0][1].TextContent(), "<conversation>") {
		t.Fatalf("expected the standalone prompt, got %q", sent[0][1].TextContent())
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

// The prior summary must appear only in its dedicated block.
func TestPreviousSummaryIsNotAlsoInTheConversation(t *testing.T) {
	t.Parallel()

	var prompt string
	model := stubModel{
		generate: func(_ context.Context, messages []agentcore.Message, _ []agentcore.ToolSpec, _ ...agentcore.CallOption) (*agentcore.LLMResponse, error) {
			prompt = messages[len(messages)-1].TextContent()
			return &agentcore.LLMResponse{Message: agentcore.Message{
				Role:    agentcore.RoleAssistant,
				Content: []agentcore.ContentBlock{agentcore.TextBlock("<summary>fresh</summary>")},
			}}, nil
		},
	}

	const old = "UNIQUE-PRIOR-SUMMARY-TEXT"
	msgs := []agentcore.AgentMessage{
		ContextSummary{Summary: old},
		agentcore.Message{Role: agentcore.RoleUser, Content: []agentcore.ContentBlock{agentcore.TextBlock(strings.Repeat("a ", 4000))}},
		agentcore.Message{Role: agentcore.RoleAssistant, Content: []agentcore.ContentBlock{agentcore.TextBlock("ok")}},
		// Absorb KeepRecentTokens so the previous turn is summarized.
		agentcore.Message{Role: agentcore.RoleUser, Content: []agentcore.ContentBlock{agentcore.TextBlock(strings.Repeat("z ", 400))}},
	}

	cfg := summaryRunConfig{Model: model, ContextWindow: 200000, ReserveTokens: 16000, KeepRecentTokens: 100}
	if _, _, err := runSummaryCompaction(context.Background(), cfg, msgs, true); err != nil {
		t.Fatalf("compaction: %v", err)
	}
	if strings.Count(prompt, old) != 1 {
		t.Fatalf("prior summary appears %d times in the prompt, want 1", strings.Count(prompt, old))
	}
	if !strings.Contains(prompt, "<previous-summary>") {
		t.Fatal("prior summary must ride in its own block")
	}
}

// Do not reword a prior summary when no new history was cut.
func TestSummaryIsNoOpWhenOnlyThePriorSummaryIsCut(t *testing.T) {
	t.Parallel()

	calls := 0
	model := stubModel{
		generate: func(context.Context, []agentcore.Message, []agentcore.ToolSpec, ...agentcore.CallOption) (*agentcore.LLMResponse, error) {
			calls++
			return &agentcore.LLMResponse{Message: agentcore.Message{
				Role:    agentcore.RoleAssistant,
				Content: []agentcore.ContentBlock{agentcore.TextBlock("<summary>x</summary>")},
			}}, nil
		},
	}

	msgs := []agentcore.AgentMessage{
		ContextSummary{Summary: "prior"},
		agentcore.Message{Role: agentcore.RoleUser, Content: []agentcore.ContentBlock{agentcore.TextBlock(strings.Repeat("b ", 4000))}},
	}

	cfg := summaryRunConfig{Model: model, ContextWindow: 200000, ReserveTokens: 16000, KeepRecentTokens: 100}
	out, info, err := runSummaryCompaction(context.Background(), cfg, msgs, true)
	if err != nil {
		t.Fatalf("compaction: %v", err)
	}
	if calls != 0 {
		t.Fatalf("model called %d time(s) with no new history to summarize", calls)
	}
	if info != nil || len(out) != len(msgs) {
		t.Fatal("expected a no-op")
	}
}

func TestRunSummaryCompaction_SummarizesSplitTurnAfterPriorCheckpoint(t *testing.T) {
	var calls int
	model := summaryReplyModel(func([]agentcore.Message, []agentcore.ToolSpec) { calls++ }, "checkpoint body")

	// The oversized turn prefix still needs summarizing after a checkpoint.
	msgs := []agentcore.AgentMessage{
		ContextSummary{Summary: "earlier work"},
		agentcore.UserMsg(strings.Repeat("a", 200)),
		agentcore.Message{
			Role:    agentcore.RoleAssistant,
			Content: []agentcore.ContentBlock{agentcore.TextBlock(strings.Repeat("b", 200))},
		},
	}

	cfg := summaryRunConfig{Model: model, ContextWindow: 16, ReserveTokens: 4, KeepRecentTokens: 1}
	out, info, err := runSummaryCompaction(context.Background(), cfg, msgs, true)
	if err != nil {
		t.Fatalf("unexpected compaction error: %v", err)
	}
	if info == nil {
		t.Fatal("expected the oversized turn to be compacted, got a no-op")
	}
	if calls == 0 {
		t.Fatal("expected the split-turn prefix to be summarized")
	}
	if !info.IsSplitTurn {
		t.Fatal("expected the compaction to report a split turn")
	}
	if _, ok := out[0].(ContextSummary); !ok {
		t.Fatalf("expected a checkpoint at the head, got %T", out[0])
	}
}

func TestFullSummaryStrategy_SavingExcludesStaticOverhead(t *testing.T) {
	model := summaryReplyModel(nil, "checkpoint body")
	strategy := NewFullSummary(FullSummaryConfig{Model: model, KeepRecentTokens: 1})

	view := []agentcore.AgentMessage{
		agentcore.UserMsg(strings.Repeat("a", 4000)),
		agentcore.Message{
			Role:    agentcore.RoleAssistant,
			Content: []agentcore.ContentBlock{agentcore.TextBlock("done")},
		},
		agentcore.UserMsg("keep"),
	}

	// Provider tokens include static prompt overhead hidden from EstimateTotal.
	const staticOverhead = 50_000
	budget := Budget{
		Tokens:    EstimateTotal(view) + staticOverhead,
		Window:    200_000,
		Threshold: 40_000, // below Tokens, so the strategy actually runs
	}

	next, res, err := strategy.Apply(context.Background(), nil, view, budget)
	if err != nil {
		t.Fatalf("apply failed: %v", err)
	}
	if !res.Applied {
		t.Fatal("expected the summary to apply")
	}
	if want := EstimateTotal(view) - EstimateTotal(next); res.TokensSaved != want {
		t.Fatalf("saving must be measured between views: got %d, want %d", res.TokensSaved, want)
	}
	if res.TokensSaved >= staticOverhead {
		t.Fatalf("saving %d swallowed the static overhead, budget would read empty", res.TokensSaved)
	}
}

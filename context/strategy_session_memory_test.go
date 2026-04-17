package context

import (
	"context"
	"strings"
	"testing"

	"github.com/voocel/agentcore"
)

// sessionMemoryConvo builds a long enough conversation that cutting it at
// KeepRecentTokens produces a non-zero history prefix. Each user/assistant
// pair contributes enough tokens to exceed the 20k default reserve.
func sessionMemoryConvo() []agentcore.AgentMessage {
	msgs := []agentcore.AgentMessage{}
	body := strings.Repeat("lorem ipsum dolor sit amet ", 400) // ~100 tokens per copy
	for range 20 {
		msgs = append(msgs,
			agentcore.UserMsg(body),
			agentcore.Message{
				Role:    agentcore.RoleAssistant,
				Content: []agentcore.ContentBlock{{Type: agentcore.ContentText, Text: body}},
			},
		)
	}
	// Recent tail that should be kept verbatim.
	msgs = append(msgs,
		agentcore.UserMsg("recent user question"),
		agentcore.Message{
			Role:    agentcore.RoleAssistant,
			Content: []agentcore.ContentBlock{{Type: agentcore.ContentText, Text: "recent assistant reply"}},
		},
	)
	return msgs
}

func TestSessionMemoryStrategyNoopWhenNoSeed(t *testing.T) {
	t.Parallel()

	s := NewSessionMemory(SessionMemoryConfig{
		SeedFn:           func() (string, error) { return "", nil },
		KeepRecentTokens: 1000,
	})
	msgs := sessionMemoryConvo()
	budget := Budget{Tokens: EstimateTotal(msgs), Window: 2000, Threshold: 500}

	out, res, err := s.Apply(context.Background(), msgs, msgs, budget)
	if err != nil {
		t.Fatalf("apply err: %v", err)
	}
	if res.Applied {
		t.Fatal("empty seed must leave pipeline untouched so FullSummary can run")
	}
	if len(out) != len(msgs) {
		t.Fatalf("view mutated, got %d msgs want %d", len(out), len(msgs))
	}
}

func TestSessionMemoryStrategyNoopBelowThreshold(t *testing.T) {
	t.Parallel()

	s := NewSessionMemory(SessionMemoryConfig{
		SeedFn:           func() (string, error) { return "# Session Memory\n\nRich state.", nil },
		KeepRecentTokens: 1000,
	})
	msgs := sessionMemoryConvo()
	// No pressure: tokens below threshold.
	budget := Budget{Tokens: 100, Window: 10000, Threshold: 500}

	_, res, err := s.Apply(context.Background(), msgs, msgs, budget)
	if err != nil {
		t.Fatalf("apply err: %v", err)
	}
	if res.Applied {
		t.Fatal("no compaction should occur when below threshold")
	}
}

func TestSessionMemoryStrategyAppliesSeedWithoutLLM(t *testing.T) {
	t.Parallel()

	const seed = "# Session Memory\n\n## Current State\nHarness optimization in progress."
	callCount := 0
	s := NewSessionMemory(SessionMemoryConfig{
		SeedFn: func() (string, error) {
			callCount++
			return seed, nil
		},
		KeepRecentTokens: 200,
	})
	msgs := sessionMemoryConvo()
	budget := Budget{Tokens: EstimateTotal(msgs), Window: 2000, Threshold: 500}

	out, res, err := s.Apply(context.Background(), msgs, msgs, budget)
	if err != nil {
		t.Fatalf("apply err: %v", err)
	}
	if !res.Applied {
		t.Fatal("expected compaction to occur under pressure with a non-empty seed")
	}
	if callCount != 1 {
		t.Fatalf("SeedFn should be invoked exactly once, got %d", callCount)
	}
	if len(out) == 0 {
		t.Fatal("compacted view must not be empty")
	}
	cs, ok := out[0].(ContextSummary)
	if !ok {
		t.Fatalf("first message must be a ContextSummary checkpoint, got %T", out[0])
	}
	if !strings.Contains(cs.Summary, "Harness optimization in progress") {
		t.Fatalf("summary body must be sourced from the seed, got %q", cs.Summary)
	}
	if res.Info == nil || res.Info.Duration <= 0 {
		t.Fatal("SummaryInfo must be populated for observability")
	}
	if res.Info.IsIncremental != true {
		t.Fatal("seed reuse should be reported as incremental (living document)")
	}
}

func TestSessionMemoryStrategyTruncatesOversizedSeed(t *testing.T) {
	t.Parallel()

	big := strings.Repeat("x", 30000)
	s := NewSessionMemory(SessionMemoryConfig{
		SeedFn:           func() (string, error) { return big, nil },
		KeepRecentTokens: 200,
		MaxSeedRunes:     1000,
	})
	msgs := sessionMemoryConvo()
	budget := Budget{Tokens: EstimateTotal(msgs), Window: 2000, Threshold: 500}

	out, res, err := s.Apply(context.Background(), msgs, msgs, budget)
	if err != nil {
		t.Fatalf("apply err: %v", err)
	}
	if !res.Applied {
		t.Fatal("expected compaction to occur")
	}
	cs := out[0].(ContextSummary)
	if !strings.Contains(cs.Summary, "truncated for compact budget") {
		t.Fatal("oversized seed must include the truncation notice so the model knows content was dropped")
	}
	if len([]rune(cs.Summary)) > 1500 {
		t.Fatalf("truncation ineffective: summary is %d runes", len([]rune(cs.Summary)))
	}
}

func TestSessionMemoryStrategySeedErrorFallsThrough(t *testing.T) {
	t.Parallel()

	s := NewSessionMemory(SessionMemoryConfig{
		SeedFn:           func() (string, error) { return "", context.DeadlineExceeded },
		KeepRecentTokens: 200,
	})
	msgs := sessionMemoryConvo()
	budget := Budget{Tokens: EstimateTotal(msgs), Window: 2000, Threshold: 500}

	out, res, err := s.Apply(context.Background(), msgs, msgs, budget)
	if err != nil {
		t.Fatalf("errors from SeedFn must not bubble up: %v", err)
	}
	if res.Applied {
		t.Fatal("on seed error the strategy must be a no-op so FullSummary can run")
	}
	if len(out) != len(msgs) {
		t.Fatal("view must not be mutated on seed error")
	}
}

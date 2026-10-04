package agentcore

import (
	"context"
	"strings"
	"time"

	"github.com/voocel/litellm"
)

// Compactor rewrites a history into a shorter one that carries the same work
// forward. The loop decides when to call it (Config.CompactAt, and a context
// overflow) and records the result; an application may also call it itself,
// such as for a manual compaction.
//
// The deferred tools a history loaded stay offered only while a tool
// reference in it names them (see Tool.Deferred): once a compaction replaced
// the results holding the references, the model searches for them again.
type Compactor interface {
	// Compact rewrites history. call builds the call the loop would make for
	// a history, as BuildCall does with the run's Config: a compactor that
	// summarizes with the conversation's model can extend the call for the
	// part of the history it replaces, which earlier calls sent, and read it
	// from the prompt cache. A nil Compaction means there is nothing to
	// compact.
	Compact(ctx context.Context, history []Message, call func([]Message) Call) (*Compaction, error)
}

// Compaction is a rewrite of a history.
type Compaction struct {
	// Messages is the history that replaces it.
	Messages []Message
	// Replaced is how many messages of the history the rewrite stands in for.
	Replaced int
	// Usage is what the model calls of the rewrite used, if any; a run
	// prices it as it does responses.
	Usage *Usage
}

const (
	summaryOpen  = "<context-summary>\n"
	summaryClose = "\n</context-summary>\n\nThis summary stands in for the conversation so far. Continue from where it leaves off, without recapping it or asking for what it settles."
)

// SummaryMessage returns the message that stands in for history a
// compaction replaced with summary, telling the model to carry on from it.
// Models read it as a user message.
func SummaryMessage(summary string) Message {
	m := UserText(summaryOpen + summary + summaryClose)
	m.Kind = KindSummary
	m.Time = time.Now()
	return m
}

// SummaryText returns the summary a SummaryMessage holds.
func SummaryText(m Message) string {
	return strings.TrimSuffix(strings.TrimPrefix(m.Text(), summaryOpen), summaryClose)
}

// imageTokens is a flat estimate for one image.
const imageTokens = 1200

// Estimate approximates the input tokens of the next call cfg makes for
// history. It counts from the usage of the last response that measured the
// history as it now stands, adding estimates for the messages since, and
// estimates all of it, system prompt and tools included, when no response
// did, as right after a compaction.
func Estimate(cfg Config, history []Message) int {
	if i := usageAnchor(history); i >= 0 {
		return history[i].Usage.InputTokens + estimateMessages(history[i:])
	}
	tokens := estimateMessages(history)
	for _, b := range cfg.System {
		tokens += estimateBlock(b)
	}
	req := BuildCall(cfg, history).Request
	for _, t := range req.OfferedTools() {
		tokens += estimateText(t.Name+t.Description) + len(t.Parameters)/4
	}
	return tokens
}

// usageAnchor returns the index of the last response whose usage measured
// the history as it now stands, or -1. Responses older than the last
// summary measured the history it replaced.
func usageAnchor(history []Message) int {
	var since time.Time
	for i := len(history) - 1; i >= 0; i-- {
		if history[i].Kind == KindSummary {
			since = history[i].Time
			break
		}
	}
	for i := len(history) - 1; i >= 0; i-- {
		m := history[i]
		if m.Role != litellm.RoleAssistant || m.Usage == nil || m.Usage.InputTokens == 0 ||
			m.Stop == StopError || m.Stop == StopAborted {
			continue
		}
		if !m.Time.After(since) {
			break
		}
		return i
	}
	return -1
}

// estimateMessage approximates the tokens of m's content.
func estimateMessage(m Message) int {
	tokens := 0
	for _, b := range m.Blocks {
		tokens += estimateBlock(b)
	}
	return max(tokens, 1)
}

func estimateMessages(msgs []Message) int {
	total := 0
	for _, m := range msgs {
		total += estimateMessage(m)
	}
	return total
}

func estimateBlock(block litellm.Block) int {
	switch b := block.(type) {
	case litellm.TextBlock:
		return estimateText(b.Text)
	case litellm.ReasoningBlock:
		return estimateText(b.Text)
	case litellm.ToolUseBlock:
		// Arguments are JSON: ASCII-dominant.
		return (len(b.Name) + len(b.Arguments) + 3) / 4
	case litellm.ToolResultBlock:
		tokens := 0
		for _, c := range b.Content {
			tokens += estimateBlock(c)
		}
		return tokens
	case litellm.ImageBlock:
		return imageTokens
	}
	return 0
}

// estimateText detects the dominant script: CJK text averages about 1.5
// tokens per rune, ASCII-dominant text about 4 bytes per token.
func estimateText(text string) int {
	if text == "" {
		return 0
	}
	bytes, runes := len(text), len([]rune(text))
	if bytes > runes*2 {
		return max(int(float64(runes)*1.5+0.5), 1)
	}
	return max((bytes+3)/4, 1)
}

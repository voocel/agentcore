// Package compact holds Summarizer, the agentcore.Compactor that replaces
// older history with a structured checkpoint summary.
//
//	cfg := agentcore.Config{
//		Model:     model,
//		Compactor: compact.Summarizer{},
//		CompactAt: 100_000,
//	}
package compact

import (
	"context"
	"fmt"
	"slices"

	"github.com/voocel/agentcore"
	"github.com/voocel/litellm"
)

// keptWordsTokens bounds the user's messages a compaction keeps word for
// word.
const keptWordsTokens = 20_000

// Summarizer replaces the history with a checkpoint summary written by the
// conversation's own model. Before the summary it keeps the user's latest
// messages word for word, up to keptWordsTokens: what the user asked for is
// what a summary must not reword. After it come the prompts at the end the
// model has yet to answer. No response is replayed: Claude's reasoning holds
// only after the history it was produced in, unchanged, and the deferred
// tools the history loaded are found again when needed.
//
// It first asks for the summary by extending the conversation's call for
// the history it replaces, so the provider serves that from the prompt
// cache; when the request no longer fits, or the model answers without the
// summary in <summary> tags, it asks again with a plain-text transcript. A
// previous summary is folded into the new one.
type Summarizer struct {
	// Notes, if set, returns text appended to the summary of the messages it
	// replaces, the previous summary among them, such as the files they
	// worked on (tools.FileOps): what the summary must carry whatever the
	// model wrote.
	Notes func(replaced []agentcore.Message) string
}

// Compact implements agentcore.Compactor.
func (s Summarizer) Compact(ctx context.Context, history []agentcore.Message, call func([]agentcore.Message) agentcore.Call) (*agentcore.Compaction, error) {
	cut := len(history)
	for cut > 0 && history[cut-1].Role == litellm.RoleUser && history[cut-1].Kind != agentcore.KindSummary {
		cut--
	}
	previous, older, fresh := splitPreviousSummary(history[:cut])
	if !fresh {
		return nil, nil
	}

	var used litellm.Usage
	c := call(history[:cut])
	summary, err := forkSummary(ctx, c, &used)
	if err != nil {
		return nil, err
	}
	if summary == "" {
		if summary, err = standaloneSummary(ctx, c, older, previous, &used); err != nil {
			return nil, err
		}
	}
	if s.Notes != nil {
		summary += s.Notes(history[:cut])
	}

	out := slices.Concat(usersWords(older), []agentcore.Message{agentcore.SummaryMessage(summary)}, history[cut:])
	return &agentcore.Compaction{Messages: out, Replaced: cut, Usage: &agentcore.Usage{Usage: used}}, nil
}

// splitPreviousSummary separates an earlier checkpoint from the messages
// around it: before it, those that compaction kept; after it, those since.
// fresh reports that there are some since, or no checkpoint.
func splitPreviousSummary(msgs []agentcore.Message) (previous string, rest []agentcore.Message, fresh bool) {
	fresh = len(msgs) > 0
	for i, m := range msgs {
		if m.Kind == agentcore.KindSummary {
			previous, fresh = agentcore.SummaryText(m), i < len(msgs)-1
			continue
		}
		rest = append(rest, m)
	}
	return previous, rest, fresh
}

// usersWords returns the user's own messages among msgs, the latest that
// fit in keptWordsTokens, in order; the application marks those it adds with
// a Kind. The one that would overflow the budget keeps its text, cut down in
// the middle.
func usersWords(msgs []agentcore.Message) []agentcore.Message {
	var kept []agentcore.Message
	budget := keptWordsTokens
	for i := len(msgs) - 1; i >= 0 && budget > 0; i-- {
		m := msgs[i]
		if m.Role != litellm.RoleUser || m.Kind != "" {
			continue
		}
		tokens := agentcore.EstimateMessage(m)
		if tokens > budget {
			text := m.Text()
			if text = cutMiddle(text, len([]rune(text))*budget/tokens); text == "" {
				break
			}
			m.Blocks, tokens = []litellm.Block{litellm.Text(text)}, budget
		}
		budget -= tokens
		kept = append(kept, m)
	}
	slices.Reverse(kept)
	return kept
}

// cutMiddle returns text cut to keep runes, its start and its end, where a
// request and what it is about tend to be; "" when nothing is kept.
func cutMiddle(text string, keep int) string {
	if keep <= 0 {
		return ""
	}
	r := []rune(text)
	head := keep / 2
	return fmt.Sprintf("%s\n\n[... %d characters cut ...]\n\n%s", string(r[:head]), len(r)-keep, string(r[len(r)-(keep-head):]))
}

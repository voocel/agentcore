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
	"encoding/json"
	"slices"
	"strings"

	"github.com/voocel/agentcore"
	"github.com/voocel/litellm"
)

// Summarizer replaces the history with a checkpoint summary written by the
// conversation's own model, keeping as they are only the prompts at its end
// the model has yet to answer. Nothing else is replayed: a response kept
// past a rewrite of what came before it is one some providers reject, as
// Claude does reasoning bound to the history it was produced in, and the
// deferred tools the history loaded are found again when needed.
//
// It first asks for the summary by extending the conversation's call for
// the history it replaces, so the provider serves that from the prompt
// cache; when the request no longer fits, or the model answers without the
// summary in <summary> tags, it asks again with a plain-text transcript. A
// previous summary is folded into the new one.
type Summarizer struct{}

// Compact implements agentcore.Compactor.
func (Summarizer) Compact(ctx context.Context, history []agentcore.Message, call func([]agentcore.Message) agentcore.Call) (*agentcore.Compaction, error) {
	cut := len(history)
	for cut > 0 && history[cut-1].Role == litellm.RoleUser && history[cut-1].Kind != agentcore.KindSummary {
		cut--
	}
	previous, older := splitPreviousSummary(history[:cut])
	if len(older) == 0 {
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
	summary += formatFileOps(extractFileOps(older))

	out := append([]agentcore.Message{agentcore.SummaryMessage(summary)}, history[cut:]...)
	return &agentcore.Compaction{Messages: out, Replaced: cut, Usage: &agentcore.Usage{Usage: used}}, nil
}

// splitPreviousSummary separates an earlier checkpoint from the history that
// is new since.
func splitPreviousSummary(msgs []agentcore.Message) (string, []agentcore.Message) {
	var previous string
	var history []agentcore.Message
	for _, m := range msgs {
		if m.Kind == agentcore.KindSummary {
			previous = agentcore.SummaryText(m)
			continue
		}
		history = append(history, m)
	}
	return previous, history
}

// extractFileOps lists the files the history read without modifying, and the
// files it modified.
func extractFileOps(msgs []agentcore.Message) (read, modified []string) {
	readSet := map[string]bool{}
	modifiedSet := map[string]bool{}
	for _, m := range msgs {
		for _, tc := range m.ToolCalls() {
			path := pathArg(tc.Arguments)
			if path == "" {
				continue
			}
			switch tc.Name {
			case "read":
				readSet[path] = true
			case "write", "edit":
				modifiedSet[path] = true
			}
		}
	}
	for f := range readSet {
		if !modifiedSet[f] {
			read = append(read, f)
		}
	}
	for f := range modifiedSet {
		modified = append(modified, f)
	}
	slices.Sort(read)
	slices.Sort(modified)
	return read, modified
}

// pathArg reads the "file_path" of read, write and edit.
func pathArg(args string) string {
	var obj struct {
		FilePath string `json:"file_path"`
	}
	if json.Unmarshal([]byte(args), &obj) != nil {
		return ""
	}
	return obj.FilePath
}

func formatFileOps(read, modified []string) string {
	var s string
	if len(read) > 0 {
		s += "\n\n<read-files>\n" + strings.Join(read, "\n") + "\n</read-files>"
	}
	if len(modified) > 0 {
		s += "\n\n<modified-files>\n" + strings.Join(modified, "\n") + "\n</modified-files>"
	}
	return s
}

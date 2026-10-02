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

// Bounds for the default verbatim tail.
const (
	minKeepRecentTokens = 2000
	maxKeepRecentTokens = 20000
)

// Summarizer replaces all but the most recent messages, a quarter of the
// history between 2k and 20k tokens, with a checkpoint summary written by the
// conversation's own model.
//
// It first asks for the summary by extending the conversation's call for
// the part it replaces, so the provider serves that from the prompt cache;
// when the request no longer fits, or the model answers without the summary
// in <summary> tags, it asks again with a plain-text transcript. A previous
// summary is folded into the new one, and the tools the replaced part loaded
// stay loaded.
type Summarizer struct{}

// Compact implements agentcore.Compactor.
func (Summarizer) Compact(ctx context.Context, history []agentcore.Message, call func([]agentcore.Message) agentcore.Call) (*agentcore.Compaction, error) {
	total := 0
	for _, m := range history {
		total += agentcore.EstimateMessage(m)
	}
	cut := cutPoint(history, min(maxKeepRecentTokens, max(minKeepRecentTokens, total/4)))
	previous, older := splitPreviousSummary(history[:cut])
	if len(older) == 0 {
		return nil, nil
	}

	c := call(history[:cut])
	summary, err := forkSummary(ctx, c)
	if err != nil {
		return nil, err
	}
	if summary == "" {
		if summary, err = standaloneSummary(ctx, c, older, previous); err != nil {
			return nil, err
		}
	}
	summary += formatFileOps(extractFileOps(older))

	out := []agentcore.Message{agentcore.SummaryMessage(summary)}
	out = append(out, toolLoads(history[:cut])...)
	out = append(out, history[cut:]...)
	return &agentcore.Compaction{Messages: out, Replaced: cut}, nil
}

// toolLoads returns the exchanges of msgs that loaded deferred tools: each
// call whose result holds tool references, with a result of those
// references alone.
func toolLoads(msgs []agentcore.Message) []agentcore.Message {
	calls := map[string]litellm.ToolUseBlock{}
	var out []agentcore.Message
	for _, m := range msgs {
		for _, c := range m.ToolCalls() {
			calls[c.ID] = c
		}
		result, ok := m.ToolResult()
		if !ok {
			continue
		}
		var refs []litellm.Block
		for _, b := range result.Content {
			if _, ok := b.(litellm.ToolReferenceBlock); ok {
				refs = append(refs, b)
			}
		}
		if len(refs) == 0 {
			continue
		}
		out = append(out,
			agentcore.Message{Role: litellm.RoleAssistant, Blocks: []litellm.Block{calls[result.ToolUseID]}},
			agentcore.ToolResult(result.ToolUseID, agentcore.Result{Content: refs}),
		)
	}
	return out
}

// cutPoint returns the index of the first message to keep verbatim so that
// roughly keepTokens of recent history stay, or 0 when nothing can be
// compacted. Tool results stay with the call that issued them: a cut landing
// on a result retreats to the assistant message that requested it.
func cutPoint(history []agentcore.Message, keepTokens int) int {
	cut, kept := 0, 0
	for i := len(history) - 1; i > 0; i-- {
		kept += agentcore.EstimateMessage(history[i])
		if kept >= keepTokens {
			cut = i
			break
		}
	}
	for cut > 0 && history[cut].Role == litellm.RoleTool {
		cut--
	}
	return cut
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

// pathArg reads "file_path" (edit/read/write), falling back to "path".
func pathArg(args string) string {
	var obj struct {
		FilePath string `json:"file_path"`
		Path     string `json:"path"`
	}
	if json.Unmarshal([]byte(args), &obj) != nil {
		return ""
	}
	if obj.FilePath != "" {
		return obj.FilePath
	}
	return obj.Path
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

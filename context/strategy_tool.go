package context

import (
	"context"
	"fmt"

	"github.com/voocel/agentcore"
)

// DefaultClearedToolResult replaces a tool result that microcompact dropped.
// Exported so a host building on ClearedMessageFn can extend it rather than
// re-spell it and drift.
const DefaultClearedToolResult = "[Tool result cleared to save context.]"

type ToolResultMicrocompactConfig struct {
	Classifier     ToolClassifier
	KeepRecent     int
	ClearedMessage string
	// ClearedMessageFn overrides ClearedMessage per result, so a host can carry
	// forward whatever stays actionable after the content goes — a path to the
	// output it persisted on disk, say. "" falls back to ClearedMessage.
	//
	// Must be idempotent: every pass re-clears results a previous pass already
	// cleared, so feeding its own output back in has to yield the same text.
	// Anything else rewrites the prefix on each pass and burns the cache.
	ClearedMessageFn func(toolName string, original agentcore.Message) string
	// MinResultTokens leaves results below this size alone. Clearing one costs a
	// placeholder plus a rewritten prefix, so below some size the pass spends
	// cache to save nothing — and small results are typically state transitions
	// whose text is the only record that they happened. Zero disables the floor.
	MinResultTokens int
}

type ToolResultMicrocompactStrategy struct {
	cfg ToolResultMicrocompactConfig
}

func NewToolResultMicrocompact(cfg ToolResultMicrocompactConfig) *ToolResultMicrocompactStrategy {
	if cfg.KeepRecent <= 0 {
		cfg.KeepRecent = 5
	}
	if cfg.ClearedMessage == "" {
		cfg.ClearedMessage = DefaultClearedToolResult
	}
	return &ToolResultMicrocompactStrategy{cfg: cfg}
}

func (s *ToolResultMicrocompactStrategy) Name() string { return "tool_result_microcompact" }

func (s *ToolResultMicrocompactStrategy) Apply(_ context.Context, _ []agentcore.AgentMessage, view []agentcore.AgentMessage, _ Budget) ([]agentcore.AgentMessage, StrategyResult, error) {
	if len(view) == 0 {
		return view, StrategyResult{Name: s.Name()}, nil
	}

	candidates := findCompactableToolResults(view, s.cfg.Classifier, s.cfg.MinResultTokens)
	if len(candidates) == 0 {
		return view, StrategyResult{Name: s.Name()}, nil
	}

	// Protect the most recent keepRecent results, deduplicated by (tool, args):
	// when the model re-issues the identical call, only the newest result is
	// worth protecting — older copies carry no extra information and would
	// crowd genuinely distinct results out of the protection window.
	protected := make(map[int]struct{}, s.cfg.KeepRecent)
	seenKeys := make(map[string]struct{}, s.cfg.KeepRecent)
	for i := len(candidates) - 1; i >= 0 && len(protected) < s.cfg.KeepRecent; i-- {
		c := candidates[i]
		if _, dup := seenKeys[c.Key]; dup {
			continue
		}
		seenKeys[c.Key] = struct{}{}
		protected[c.Index] = struct{}{}
	}
	if len(protected) == len(candidates) {
		return view, StrategyResult{Name: s.Name()}, nil
	}

	out := copyMessages(view)
	saved := 0
	applied := false
	for _, candidate := range candidates {
		if _, ok := protected[candidate.Index]; ok {
			continue
		}
		msg, ok := out[candidate.Index].(agentcore.Message)
		if !ok {
			continue
		}
		next := msg
		next.Content = []agentcore.ContentBlock{agentcore.TextBlock(s.clearedText(candidate.ToolName, msg))}
		next.Metadata = cloneMetadata(msg.Metadata)
		if next.Metadata == nil {
			next.Metadata = map[string]any{}
		}
		next.Metadata["compacted_tool_result"] = true
		next.Metadata["compacted_tool_name"] = candidate.ToolName
		out[candidate.Index] = next
		saved += max(0, EstimateTokens(msg)-EstimateTokens(next))
		applied = true
	}

	return out, StrategyResult{
		Applied:     applied,
		TokensSaved: saved,
		Name:        s.Name(),
	}, nil
}

func (s *ToolResultMicrocompactStrategy) clearedText(toolName string, original agentcore.Message) string {
	if s.cfg.ClearedMessageFn != nil {
		if text := s.cfg.ClearedMessageFn(toolName, original); text != "" {
			return text
		}
	}
	return s.cfg.ClearedMessage
}

type compactableToolResult struct {
	Index    int
	ToolName string
	// Key identifies the originating call by tool name + raw args, so results
	// of identical repeated calls can be deduplicated in the protection window.
	Key string
}

type pendingToolCall struct {
	name string
	key  string
}

func findCompactableToolResults(msgs []agentcore.AgentMessage, classifier ToolClassifier, minTokens int) []compactableToolResult {
	pending := map[string]pendingToolCall{}
	var results []compactableToolResult

	for i, am := range msgs {
		msg, ok := am.(agentcore.Message)
		if !ok {
			continue
		}

		if msg.Role == agentcore.RoleAssistant {
			for _, call := range msg.ToolCalls() {
				pending[call.ID] = pendingToolCall{
					name: call.Name,
					key:  call.Name + "\x00" + string(call.Args),
				}
			}
			continue
		}

		if msg.Role != agentcore.RoleTool {
			continue
		}

		callID, _ := msg.Metadata["tool_call_id"].(string)
		call := pending[callID]
		if call.name == "" {
			continue
		}
		if classifier != nil && !classifier(call.name) {
			continue
		}
		if minTokens > 0 && EstimateTokens(msg) < minTokens {
			continue
		}
		results = append(results, compactableToolResult{Index: i, ToolName: call.name, Key: call.key})
	}

	return results
}

func formatTrimmedPlaceholder(n int) string {
	return fmt.Sprintf("[%d characters trimmed]", n)
}

package context

import (
	"context"
	"fmt"
	"time"

	"github.com/voocel/agentcore"
)

const defaultClearedToolResult = "[Tool result cleared to save context.]"

type ToolResultMicrocompactConfig struct {
	Classifier     ToolClassifier
	KeepRecent     int
	ClearedMessage string
	IdleThreshold  time.Duration
}

type ToolResultMicrocompactStrategy struct {
	cfg ToolResultMicrocompactConfig
}

func NewToolResultMicrocompact(cfg ToolResultMicrocompactConfig) *ToolResultMicrocompactStrategy {
	if cfg.KeepRecent <= 0 {
		cfg.KeepRecent = 5
	}
	if cfg.ClearedMessage == "" {
		cfg.ClearedMessage = defaultClearedToolResult
	}
	return &ToolResultMicrocompactStrategy{cfg: cfg}
}

func (s *ToolResultMicrocompactStrategy) Name() string { return "tool_result_microcompact" }

func (s *ToolResultMicrocompactStrategy) Apply(_ context.Context, _ []agentcore.AgentMessage, view []agentcore.AgentMessage, _ Budget) ([]agentcore.AgentMessage, StrategyResult, error) {
	if len(view) == 0 {
		return view, StrategyResult{Name: s.Name()}, nil
	}

	candidates := findCompactableToolResults(view, s.cfg.Classifier)
	if len(candidates) == 0 {
		return view, StrategyResult{Name: s.Name()}, nil
	}

	keepRecent := s.cfg.KeepRecent
	if s.cfg.IdleThreshold > 0 {
		lastAssistant := latestAssistantTimestamp(view)
		if !lastAssistant.IsZero() && time.Since(lastAssistant) > s.cfg.IdleThreshold && keepRecent > 1 {
			keepRecent = max(1, keepRecent/2)
		}
	}

	// Protect the most recent keepRecent results, deduplicated by (tool, args):
	// when the model re-issues the identical call, only the newest result is
	// worth protecting — older copies carry no extra information and would
	// crowd genuinely distinct results out of the protection window.
	protected := make(map[int]struct{}, keepRecent)
	seenKeys := make(map[string]struct{}, keepRecent)
	for i := len(candidates) - 1; i >= 0 && len(protected) < keepRecent; i-- {
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
		next.Content = []agentcore.ContentBlock{agentcore.TextBlock(s.cfg.ClearedMessage)}
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

func findCompactableToolResults(msgs []agentcore.AgentMessage, classifier ToolClassifier) []compactableToolResult {
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
		results = append(results, compactableToolResult{Index: i, ToolName: call.name, Key: call.key})
	}

	return results
}

func latestAssistantTimestamp(msgs []agentcore.AgentMessage) time.Time {
	for i := len(msgs) - 1; i >= 0; i-- {
		msg, ok := msgs[i].(agentcore.Message)
		if ok && msg.Role == agentcore.RoleAssistant {
			return msg.Timestamp
		}
	}
	return time.Time{}
}

func formatTrimmedPlaceholder(n int) string {
	return fmt.Sprintf("[%d characters trimmed]", n)
}

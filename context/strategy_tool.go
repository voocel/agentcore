package context

import (
	"context"
	"fmt"
	"time"

	"github.com/voocel/agentcore"
)

const defaultClearedToolResult = "[Tool result cleared to save context. Re-run the relevant tool if exact output is needed.]"

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

	if len(candidates) <= keepRecent {
		return view, StrategyResult{Name: s.Name()}, nil
	}

	protected := make(map[int]struct{}, keepRecent)
	for i := len(candidates) - keepRecent; i < len(candidates); i++ {
		protected[candidates[i].Index] = struct{}{}
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
}

func findCompactableToolResults(msgs []agentcore.AgentMessage, classifier ToolClassifier) []compactableToolResult {
	pending := map[string]string{}
	var results []compactableToolResult

	for i, am := range msgs {
		msg, ok := am.(agentcore.Message)
		if !ok {
			continue
		}

		if msg.Role == agentcore.RoleAssistant {
			for _, call := range msg.ToolCalls() {
				pending[call.ID] = call.Name
			}
			continue
		}

		if msg.Role != agentcore.RoleTool {
			continue
		}

		callID, _ := msg.Metadata["tool_call_id"].(string)
		toolName := pending[callID]
		if toolName == "" {
			continue
		}
		if classifier != nil && !classifier(toolName) {
			continue
		}
		results = append(results, compactableToolResult{Index: i, ToolName: toolName})
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

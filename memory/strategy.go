package memory

import (
	"context"

	"github.com/voocel/agentcore"
)

// Budget describes the current context pressure seen by a strategy.
type Budget struct {
	Tokens    int
	Window    int
	Threshold int
}

// StrategyResult reports whether a strategy rewrote the prompt view.
type StrategyResult struct {
	Applied     bool
	TokensSaved int
	Name        string
	Info        *CompactionInfo
}

// Strategy rewrites the current prompt view using the full transcript and
// current token budget.
type Strategy interface {
	Name() string
	Apply(ctx context.Context, transcript []agentcore.AgentMessage, view []agentcore.AgentMessage, budget Budget) ([]agentcore.AgentMessage, StrategyResult, error)
}

// ForceCompactionStrategy is used for explicit /compact and overflow recovery.
// It can choose to run even when the regular threshold is not crossed.
type ForceCompactionStrategy interface {
	Strategy
	ForceApply(ctx context.Context, transcript []agentcore.AgentMessage, view []agentcore.AgentMessage, budget Budget) ([]agentcore.AgentMessage, StrategyResult, error)
}

// ToolClassifier returns true when a tool result can be aggressively compacted.
type ToolClassifier func(toolName string) bool

// PostCompactHook injects lightweight reminder messages after a summary
// checkpoint is produced.
type PostCompactHook func(ctx context.Context, info CompactionInfo, kept []agentcore.AgentMessage) ([]agentcore.AgentMessage, error)

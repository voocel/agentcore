package context

import (
	"context"

	"github.com/voocel/agentcore"
)

// Budget describes the current context pressure seen by a strategy.
// Threshold is the maximum token estimate allowed before a rewrite is needed.
type Budget struct {
	Tokens    int
	Window    int
	Threshold int
}

// StrategyResult reports whether a strategy rewrote the prompt view.
// Info is populated when the strategy produced a summary checkpoint.
type StrategyResult struct {
	Applied     bool
	TokensSaved int
	Name        string
	Info        *SummaryInfo
}

// Strategy rewrites the current prompt view using the full transcript and
// current token budget.
//
// transcript is the runtime baseline remembered by the engine. view is the
// current projected view after any earlier strategies in the pipeline.
type Strategy interface {
	Name() string
	Apply(ctx context.Context, transcript []agentcore.AgentMessage, view []agentcore.AgentMessage, budget Budget) ([]agentcore.AgentMessage, StrategyResult, error)
}

// ForceCompactionStrategy is used for explicit /compact and overflow recovery.
// It can choose to run even when the regular threshold is not crossed.
// ForceApply should base its rewrite on the full transcript-quality input
// required by the strategy rather than on a previously degraded prompt view.
type ForceCompactionStrategy interface {
	Strategy
	ForceApply(ctx context.Context, transcript []agentcore.AgentMessage, view []agentcore.AgentMessage, budget Budget) ([]agentcore.AgentMessage, StrategyResult, error)
}

// ToolClassifier returns true when a tool result can be aggressively rewritten
// by tool-result microcompact strategies.
type ToolClassifier func(toolName string) bool

// PostSummaryHook injects lightweight reminder messages after a summary
// checkpoint is produced. Hooks must be side-effect free and should not
// perform I/O such as file reads or tool execution.
type PostSummaryHook func(ctx context.Context, info SummaryInfo, kept []agentcore.AgentMessage) ([]agentcore.AgentMessage, error)

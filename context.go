package agentcore

import "context"

// CompactReason identifies why a committed compaction was requested.
type CompactReason string

const (
	CompactReasonManual    CompactReason = "manual"
	CompactReasonOverflow  CompactReason = "overflow"
	CompactReasonThreshold CompactReason = "threshold"
)

// ContextProjection is the projected prompt view for a single LLM call.
type ContextProjection struct {
	Messages []AgentMessage
	Usage    *ContextUsage
}

// ContextCommitResult is a committed context rewrite, typically from /compact.
type ContextCommitResult struct {
	Messages []AgentMessage
	Usage    *ContextUsage
	Changed  bool
	Strategy string
	CompactedCount int
	KeptCount      int
	SplitTurn      bool
}

// ContextRecoveryResult is the result of overflow recovery.
// View is always the retryable prompt view. CommitMessages is optional and,
// when ShouldCommit is true, should replace the runtime message baseline.
type ContextRecoveryResult struct {
	View           []AgentMessage
	CommitMessages []AgentMessage
	Usage          *ContextUsage
	Changed        bool
	ShouldCommit   bool
	Strategy       string
	CompactedCount int
	KeptCount      int
	SplitTurn      bool
}

// ContextManager owns prompt projection, committed compaction, overflow
// recovery, and usage reporting for long-running agent sessions.
type ContextManager interface {
	Project(ctx context.Context, msgs []AgentMessage) (ContextProjection, error)
	Compact(ctx context.Context, msgs []AgentMessage, reason CompactReason) (ContextCommitResult, error)
	RecoverOverflow(ctx context.Context, msgs []AgentMessage, cause error) (ContextRecoveryResult, error)
	Sync(msgs []AgentMessage)
	Usage() *ContextUsage
}

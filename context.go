package agentcore

import "context"

// CompactReason identifies why a committed context rewrite was requested.
// It is attached to explicit commits such as manual /compact or overflow
// recovery, not to transient per-request projections.
type CompactReason string

const (
	CompactReasonManual    CompactReason = "manual"
	CompactReasonOverflow  CompactReason = "overflow"
	CompactReasonThreshold CompactReason = "threshold"
)

// ContextProjection is the prompt view projected for a single LLM call.
// The projection does not modify the runtime message baseline by itself.
type ContextProjection struct {
	Messages []AgentMessage
	Usage    *ContextUsage
}

// ContextSnapshot describes the current active context view plus the most
// recent rewrite details remembered by the manager.
//
// Snapshot is meant for debugging, observability, and UI surfaces such as
// /context. It reports the active view currently remembered by the manager,
// which may be the baseline runtime messages, a projected prompt view, or a
// recovered/committed view depending on the most recent operation.
type ContextSnapshot struct {
	Usage              *ContextUsage
	Scope              string
	TranscriptMessages int
	ActiveMessages     int
	SummaryMessages    int
	ToolMessages       int
	ClearedToolResults int
	TrimmedTextBlocks  int
	LastStrategy       string
	LastChanged        bool
	LastCompactedCount int
	LastKeptCount      int
	LastSplitTurn      bool
}

// ContextCommitResult is the result of an explicit committed rewrite.
// The returned Messages should replace the runtime baseline when Changed is
// true, for example after a manual /compact command.
type ContextCommitResult struct {
	Messages       []AgentMessage
	Usage          *ContextUsage
	Changed        bool
	Strategy       string
	CompactedCount int
	KeptCount      int
	SplitTurn      bool
}

// ContextRecoveryResult is the result of overflow recovery.
//
// View is always the retryable prompt view. CommitMessages is optional and,
// when ShouldCommit is true, should replace the runtime message baseline so
// future usage reporting and turns start from the recovered state.
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

// ContextManager owns prompt projection, committed rewrites, overflow
// recovery, and usage reporting for long-running agent sessions.
//
// The manager deliberately distinguishes between transient prompt projection
// and explicit baseline rewrites:
//   - Project builds a prompt view for one LLM call without committing it.
//   - Compact performs an explicit committed rewrite such as /compact.
//   - RecoverOverflow produces a retryable prompt view after context overflow
//     and may optionally return a new committed baseline.
//   - Sync updates the manager with the current runtime baseline after
//     external message replacement, session restore, or clear.
//   - Usage reports the latest effective usage remembered by the manager.
//   - Snapshot reports the current active view and recent rewrite details for
//     debugging and UI surfaces.
type ContextManager interface {
	// Project builds the prompt view for a single model call without mutating
	// the caller's runtime baseline.
	Project(ctx context.Context, msgs []AgentMessage) (ContextProjection, error)

	// Compact performs an explicit committed rewrite of msgs. The caller is
	// responsible for replacing its runtime baseline with the returned Messages
	// when Changed is true.
	Compact(ctx context.Context, msgs []AgentMessage, reason CompactReason) (ContextCommitResult, error)

	// RecoverOverflow produces a retryable view after a provider reports
	// context overflow. When ShouldCommit is true, CommitMessages should replace
	// the runtime baseline before continuing.
	RecoverOverflow(ctx context.Context, msgs []AgentMessage, cause error) (ContextRecoveryResult, error)

	// Sync tells the manager what the current runtime baseline is after restore,
	// clear, import, or any other external replacement of messages.
	Sync(msgs []AgentMessage)

	// Usage returns the latest effective context usage remembered by the
	// manager. It may reflect a projected or recovered view rather than the raw
	// runtime baseline.
	Usage() *ContextUsage

	// Snapshot returns the latest active view snapshot remembered by the
	// manager. It is intended for observability and may be nil before the
	// manager has seen any messages.
	Snapshot() *ContextSnapshot
}

// ContextLLMConverter is an optional interface a ContextManager can implement
// to provide its own AgentMessage → Message conversion (e.g. to handle
// summary message types). When implemented, NewAgent auto-wires it.
type ContextLLMConverter interface {
	ConvertToLLM([]AgentMessage) []Message
}

// ContextEstimator is an optional interface a ContextManager can implement
// to provide token estimation. When implemented, NewAgent auto-wires it.
type ContextEstimator interface {
	EstimateContext([]AgentMessage) (tokens, usageTokens, trailingTokens int)
}

// ContextWindower is an optional interface a ContextManager can implement
// to report its configured context window size.
type ContextWindower interface {
	ContextWindow() int
}

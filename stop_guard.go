package agentcore

import "context"

// StopGuard is the single arbiter for every normal stop of a run. It is
// consulted on both stop paths:
//
//   - StopTriggerEndTurn: the LLM produced a final text response with no
//     tool calls and no queued follow-up messages.
//   - StopTriggerAfterTool: a StopAfterTool / StopAfterToolResult hook
//     requested an early exit after a successful terminal tool.
//
// Error paths never consult the guard: context cancellation (Abort),
// StopReasonError/StopReasonAborted from the provider, and the MaxTurns
// safety valve all terminate directly — a guard must not be able to override
// user aborts or safety limits.
//
// Return Allow=true to let the agent stop normally. Return Allow=false with
// an InjectMessage to keep the agent running for another turn — the message
// is delivered as a user message on the next LLM call. Set Escalate=true to
// force the run to end with a guard-escalation error (used when the guard
// has repeatedly blocked stops and suspects a prompt bug).
//
// Guard state (e.g. consecutive-block counters) is the guard's own
// responsibility; agentcore passes only the current turn index, the stopping
// assistant message, and which path triggered the check.
type StopGuard func(ctx context.Context, stop StopInfo) StopDecision

// StopTrigger identifies which stop path is consulting the guard.
type StopTrigger string

const (
	// StopTriggerEndTurn is the natural stop: final assistant response,
	// no tool calls, no queued follow-ups.
	StopTriggerEndTurn StopTrigger = "end_turn"
	// StopTriggerAfterTool is an early exit requested by the harness via
	// StopAfterTool / StopAfterToolResult.
	StopTriggerAfterTool StopTrigger = "stop_after_tool"
)

// StopInfo carries the information a StopGuard needs to decide.
type StopInfo struct {
	// TurnIndex is the index of the turn that just produced the stopping message.
	TurnIndex int
	// Message is the assistant message whose StopReason triggered this check.
	Message Message
	// Trigger reports which stop path is consulting the guard.
	Trigger StopTrigger
}

// StopDecision is the guard's verdict.
type StopDecision struct {
	// Allow=true lets the stop proceed; Allow=false keeps the loop alive.
	Allow bool
	// InjectMessage is delivered as a user message on the next turn when
	// Allow=false && !Escalate. Empty InjectMessage with Allow=false is
	// treated as Allow=true (safe default — never stall silently).
	InjectMessage string
	// Escalate=true ends the run immediately with an error.
	Escalate bool
}

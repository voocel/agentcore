package agentcore

import "context"

// StopGuard is consulted when the LLM would end a run without tool calls
// (i.e. the assistant produced a final text response and no more tool calls).
//
// Return Allow=true to let the agent stop normally. Return Allow=false with
// an InjectMessage to keep the agent running for another turn — the message
// is delivered as a user message on the next LLM call. Set Escalate=true to
// force the run to end with a guard-escalation error (used when the guard
// has repeatedly blocked stops and suspects a prompt bug).
//
// Guard state (e.g. consecutive-block counters) is the guard's own
// responsibility; agentcore passes only the current turn index and the
// stopping assistant message.
type StopGuard func(ctx context.Context, stop StopInfo) StopDecision

// StopInfo carries the information a StopGuard needs to decide.
type StopInfo struct {
	// TurnIndex is the index of the turn that just produced the stopping message.
	TurnIndex int
	// Message is the assistant message whose StopReason triggered this check.
	Message Message
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

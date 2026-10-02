package agentcore

import (
	"time"

	"github.com/voocel/litellm"
)

// Event is something that happened in a run, delivered to Config.Emit. The
// events are the types below; switch on them by type.
//
// A run streams each response as MessageStart, then MessageDelta events, and
// records it with MessageEnd, which every message entering the history
// passes through. Events hold no state that changes later: a MessageEnd's
// message is final, and the streamed content is in the deltas only.
type Event interface{ isEvent() }

// MessageStart begins a response; MessageDelta events stream its content.
type MessageStart struct{}

// MessageDelta is a streamed event of the response under way: a block
// started, text or arguments added, a block ended, usage. A Retry discards
// the response so far.
type MessageDelta struct{ Event litellm.Event }

// MessageEnd records a message entering the history: a prompt, a response, a
// tool result. Returning an error from Emit for it keeps it out and stops the
// run, so an application stores messages here durably.
type MessageEnd struct{ Message Message }

// ToolStart begins a tool call, once its arguments were checked and before
// the middleware, such as an approval, runs it.
type ToolStart struct{ Call ToolCall }

// ToolUpdate is the progress a running tool reported, see ReportProgress;
// it comes between the call's ToolStart and ToolEnd.
type ToolUpdate struct {
	Call     ToolCall
	Progress any
}

// ToolEnd ends a tool call with its result.
type ToolEnd struct {
	Call   ToolCall
	Result Result
}

// TurnEnd ends a turn: a response and the results of the tools it called,
// all recorded.
type TurnEnd struct {
	Message Message
	Results []Message
}

// Retry reports a failed model call that the run makes again after Delay.
type Retry struct {
	Attempt    int
	MaxRetries int
	Delay      time.Duration
	Err        error
}

// CompactionStart begins a compaction of the history.
type CompactionStart struct{}

// CompactionEnd ends a compaction: with the Compaction that replaces the
// history, nil when there was nothing to compact, or with Err. Returning an
// error from Emit for a Compaction keeps the history as it was and stops the
// run, so an application stores compactions here durably.
type CompactionEnd struct {
	Compaction *Compaction
	Err        error
}

// RunEnd ends a run, with the error that ended it. It is always the run's
// last event, delivered even after Emit failed.
type RunEnd struct {
	Reason EndReason
	Err    error
	// Turns, ToolCalls and FailedCalls count the run's responses, the tool
	// calls they made and the calls whose result is an error, refused calls
	// included, which Config.MaxToolErrors does not count.
	Turns, ToolCalls, FailedCalls int
}

// EndReason is why a run ended.
type EndReason string

const (
	EndDone     EndReason = "done"
	EndMaxTurns EndReason = "max_turns"
	EndAborted  EndReason = "aborted"
	EndError    EndReason = "error"
)

func (MessageStart) isEvent()    {}
func (MessageDelta) isEvent()    {}
func (MessageEnd) isEvent()      {}
func (ToolStart) isEvent()       {}
func (ToolUpdate) isEvent()      {}
func (ToolEnd) isEvent()         {}
func (TurnEnd) isEvent()         {}
func (Retry) isEvent()           {}
func (CompactionStart) isEvent() {}
func (CompactionEnd) isEvent()   {}
func (RunEnd) isEvent()          {}

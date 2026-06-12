package agentcore

import (
	"context"
	"encoding/json"
	"time"
)

// ---------------------------------------------------------------------------
// Agent Events
// ---------------------------------------------------------------------------

// EventType identifies agent lifecycle event types.
type EventType string

const (
	EventAgentStart EventType = "agent_start"
	EventAgentEnd   EventType = "agent_end"
	EventTurnStart  EventType = "turn_start"
	// EventTurnEnd fires after every LLM call completes (including its tool
	// executions) — a "turn" here is one model invocation, not one logical
	// user exchange. Steering injections and length recoveries each produce
	// additional TurnEnds within the same run.
	EventTurnEnd        EventType = "turn_end"
	EventMessageStart   EventType = "message_start"
	EventMessageUpdate  EventType = "message_update"
	EventMessageEnd     EventType = "message_end"
	EventToolExecStart  EventType = "tool_exec_start"
	EventToolExecUpdate EventType = "tool_exec_update"
	EventToolExecEnd    EventType = "tool_exec_end"
	EventRetry          EventType = "retry"
	EventError          EventType = "error"
)

// ToolExecUpdateKind distinguishes update payload semantics for tool_exec_update events.
type ToolExecUpdateKind string

const (
	ToolExecUpdatePreview  ToolExecUpdateKind = "preview"
	ToolExecUpdateProgress ToolExecUpdateKind = "progress"
)

// EndReason describes why a single agent run stopped.
type EndReason string

const (
	EndReasonStop     EndReason = "stop"
	EndReasonMaxTurns EndReason = "max_turns"
	EndReasonAborted  EndReason = "aborted"
	EndReasonError    EndReason = "error"
)

// RunSummary captures loop facts that are known at the end of a run.
// It intentionally excludes higher-level policy judgments.
type RunSummary struct {
	TurnCount  int
	ToolCalls  int
	ToolErrors int
	EndReason  EndReason
}

// DeltaKind identifies what kind of content a message_update delta carries.
type DeltaKind string

const (
	DeltaText     DeltaKind = ""         // default: regular text
	DeltaThinking DeltaKind = "thinking" // model reasoning/thinking
	DeltaToolCall DeltaKind = "toolcall" // tool call argument JSON
)

// Event is a lifecycle event emitted by the agent loop.
// This is the single output channel for all lifecycle information.
type Event struct {
	Type        EventType
	Message     AgentMessage    // for message_start/update/end, turn_end
	Delta       string          // text delta for message_update
	DeltaKind   DeltaKind       // for message_update: what kind of delta
	ToolID      string          // for tool_exec_*
	Tool        string          // tool name for tool_exec_*
	ToolLabel   string          // human-readable tool label (from ToolLabeler)
	Args        json.RawMessage // tool args for tool_exec_start/tool_exec_update
	Result      json.RawMessage // tool result for tool_exec_end and preview updates
	Progress    *ProgressPayload
	UpdateKind  ToolExecUpdateKind
	IsError     bool // tool error flag for tool_exec_end
	Preview     json.RawMessage
	ToolResults []ToolResult   // for turn_end: all tool results from this turn
	Err         error          // for error events
	NewMessages []AgentMessage // for agent_end: messages added during this loop
	RetryInfo   *RetryInfo     // for retry events
	Summary     *RunSummary    // for agent_end: factual run summary
}

// RetryInfo carries retry context for EventRetry events.
type RetryInfo struct {
	Attempt    int
	MaxRetries int
	Delay      time.Duration
	Err        error
}

// ---------------------------------------------------------------------------
// Event Helpers
// ---------------------------------------------------------------------------

// eventSink delivers loop events bound to the run context. It is created
// once per run at the AgentLoop entry points, so every emission site shares
// the same lifetime regardless of narrower contexts (per-tool cancellation
// must not affect event delivery).
type eventSink struct {
	ctx context.Context
	ch  chan<- Event
}

// emit sends an event to the channel, blocking when it is full — backpressure,
// never event loss, while the run is live. Once the run context is canceled
// delivery degrades to best-effort: a buffered/ready send still succeeds (a
// draining reader receives the terminal events), but when the channel stays
// full the event is dropped so an abandoned channel cannot leak the loop
// goroutine.
func (s eventSink) emit(ev Event) {
	select {
	case s.ch <- ev:
	default:
		select {
		case s.ch <- ev:
		case <-s.ctx.Done():
		}
	}
}

// emitError sends an error event followed by agent_end.
func (s eventSink) emitError(err error, summary *RunSummary) {
	s.emit(Event{Type: EventError, Err: err})
	s.emit(Event{Type: EventAgentEnd, Err: err, Summary: summary})
}

// ---------------------------------------------------------------------------
// Message Sequence Repair
// ---------------------------------------------------------------------------

// DefaultConvertToLLM filters AgentMessages to LLM-compatible Messages.
// Custom message types are dropped; only user/assistant/system/tool messages pass through.
func DefaultConvertToLLM(msgs []AgentMessage) []Message {
	out := make([]Message, 0, len(msgs))
	for _, m := range msgs {
		if msg, ok := m.(Message); ok {
			if msg.StopReason == StopReasonError || msg.StopReason == StopReasonAborted {
				continue
			}
			out = append(out, msg)
		}
	}
	return out
}

// dequeue drains all messages from the queue.
func dequeue(queue *[]AgentMessage) []AgentMessage {
	if len(*queue) == 0 {
		return nil
	}
	result := *queue
	*queue = nil
	return result
}

package agentcore

import (
	"encoding/json"
	"time"

	"github.com/voocel/agentcore/permission"
)

// ---------------------------------------------------------------------------
// Agent Events
// ---------------------------------------------------------------------------

// EventType identifies agent lifecycle event types.
type EventType string

const (
	EventAgentStart           EventType = "agent_start"
	EventAgentEnd             EventType = "agent_end"
	EventTurnStart            EventType = "turn_start"
	EventTurnEnd              EventType = "turn_end"
	EventMessageStart         EventType = "message_start"
	EventMessageUpdate        EventType = "message_update"
	EventMessageEnd           EventType = "message_end"
	EventToolExecStart        EventType = "tool_exec_start"
	EventToolExecUpdate       EventType = "tool_exec_update"
	EventToolExecEnd          EventType = "tool_exec_end"
	EventToolApprovalRequest  EventType = "tool_approval_request"
	EventToolApprovalResolved EventType = "tool_approval_resolved"
	EventRetry                EventType = "retry"
	EventError                EventType = "error"
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
	// StopAfterTool names the tool that triggered the early-stop when
	// EndReason==Stop and LoopConfig.StopAfterTool fired. Empty otherwise.
	// Lets the harness distinguish a model-driven natural stop from a
	// terminal-tool-driven stop and react accordingly (e.g. continue the
	// agent run with a refreshed tool snapshot after a mode-switch tool).
	StopAfterTool string
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
	Type               EventType
	Message            AgentMessage    // for message_start/update/end, turn_end
	Delta              string          // text delta for message_update
	DeltaKind          DeltaKind       // for message_update: what kind of delta
	ToolID             string          // for tool_exec_*
	Tool               string          // tool name for tool_exec_*
	ToolLabel          string          // human-readable tool label (from ToolLabeler)
	Args               json.RawMessage // tool args for tool_exec_start/tool_exec_update
	Result             json.RawMessage // tool result for tool_exec_end and preview updates
	Progress           *ProgressPayload
	UpdateKind         ToolExecUpdateKind
	IsError            bool // tool error flag for tool_exec_end
	Preview            json.RawMessage
	PermissionRequest  *permission.Request
	PermissionDecision *permission.Decision
	ToolResults        []ToolResult   // for turn_end: all tool results from this turn
	Err                error          // for error events
	NewMessages        []AgentMessage // for agent_end: messages added during this loop
	RetryInfo          *RetryInfo     // for retry events
	Summary            *RunSummary    // for agent_end: factual run summary
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

// emit sends an event to the channel. Blocks if the channel is full,
// creating backpressure to prevent event loss.
func emit(ch chan<- Event, ev Event) {
	ch <- ev
}

// emitError sends an error event followed by agent_end.
func emitError(ch chan<- Event, err error, summary *RunSummary) {
	emit(ch, Event{Type: EventError, Err: err})
	emit(ch, Event{Type: EventAgentEnd, Err: err, Summary: summary})
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

// dequeue removes messages from a queue according to the given mode.
// QueueModeAll drains everything; QueueModeOneAtATime takes only the first message.
func dequeue(queue *[]AgentMessage, mode QueueMode) []AgentMessage {
	if len(*queue) == 0 {
		return nil
	}
	if mode == QueueModeOneAtATime {
		first := (*queue)[0]
		*queue = (*queue)[1:]
		return []AgentMessage{first}
	}
	// QueueModeAll: drain everything
	result := *queue
	*queue = nil
	return result
}

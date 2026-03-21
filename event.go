package agentcore

import (
	"encoding/json"
	"time"
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

// Event is a lifecycle event emitted by the agent loop.
// This is the single output channel for all lifecycle information.
type Event struct {
	Type             EventType
	Message          AgentMessage    // for message_start/update/end, turn_end
	Delta            string          // text delta for message_update
	ToolID           string          // for tool_exec_*
	Tool             string          // tool name for tool_exec_*
	ToolLabel        string          // human-readable tool label (from ToolLabeler)
	Args             json.RawMessage // tool args for tool_exec_start/tool_exec_update
	Result           json.RawMessage // tool result for tool_exec_end/update
	UpdateKind       ToolExecUpdateKind
	IsError          bool // tool error flag for tool_exec_end
	ApprovalDecision ToolApprovalDecision
	ApprovalReason   string
	Preview          json.RawMessage
	ToolResults      []ToolResult   // for turn_end: all tool results from this turn
	Err              error          // for error events
	NewMessages      []AgentMessage // for agent_end: messages added during this loop
	RetryInfo        *RetryInfo     // for retry events
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
func emitError(ch chan<- Event, err error) {
	emit(ch, Event{Type: EventError, Err: err})
	emit(ch, Event{Type: EventAgentEnd, Err: err})
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

// RepairMessageSequence ensures tool call / tool result pairs are complete.
// Orphaned tool calls (no matching result) get a synthetic error result inserted.
// Orphaned tool results (no matching call) are removed.
// This prevents LLM providers from rejecting malformed message sequences.
func RepairMessageSequence(msgs []Message) []Message {
	out := make([]Message, 0, len(msgs))

	for i, msg := range msgs {
		out = append(out, msg)

		if msg.Role != RoleAssistant {
			continue
		}
		calls := msg.ToolCalls()
		if len(calls) == 0 {
			continue
		}

		// Collect tool result IDs that follow this assistant message
		answered := make(map[string]bool, len(calls))
		for j := i + 1; j < len(msgs); j++ {
			next := msgs[j]
			if next.Role == RoleTool {
				if id, ok := next.Metadata["tool_call_id"].(string); ok {
					answered[id] = true
				}
				continue
			}
			break // stop at first non-tool message
		}

		// Insert synthetic results for unanswered tool calls
		for _, call := range calls {
			if !answered[call.ID] {
				out = append(out, ToolResultMsg(call.ID, []byte(`"Tool result missing (conversation was truncated or interrupted)."`), true))
			}
		}
	}

	// Remove orphaned tool results (no matching call)
	callIDs := make(map[string]bool)
	for _, msg := range out {
		for _, call := range msg.ToolCalls() {
			callIDs[call.ID] = true
		}
	}

	cleaned := make([]Message, 0, len(out))
	for _, msg := range out {
		if msg.Role == RoleTool {
			if id, ok := msg.Metadata["tool_call_id"].(string); ok && !callIDs[id] {
				continue
			}
		}
		cleaned = append(cleaned, msg)
	}

	return cleaned
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

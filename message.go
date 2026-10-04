package agentcore

import (
	"encoding/json"
	"strings"
	"time"

	"github.com/voocel/litellm"
	"github.com/voocel/litellm/catalog"
)

// Message is one entry of a conversation: a model message, in litellm's
// roles and blocks, with what the loop knows of it. A tool result is a
// message of role litellm.RoleTool holding one litellm.ToolResultBlock.
// Messages encode as JSON, so a history is stored as it is.
type Message struct {
	Role   litellm.Role    `json:"role"`
	Blocks []litellm.Block `json:"blocks"`
	// Kind marks a message the loop or the application added for its own
	// purposes, such as KindSummary. Models see it as any message of its
	// role.
	Kind string `json:"kind,omitempty"`
	// Stop, Usage, Provider and Model describe an assistant message: why the
	// response ended, what it used, and what produced it.
	Stop     StopReason `json:"stop,omitempty"`
	Usage    *Usage     `json:"usage,omitempty"`
	Provider string     `json:"provider,omitempty"`
	Model    string     `json:"model,omitempty"`
	// Time is when the message entered the history.
	Time time.Time `json:"time"`
}

// KindSummary marks the summary a compaction put in place of the history it
// replaced.
const KindSummary = "summary"

// KindResume marks the prompt the loop adds after a response was cut off at
// the output token limit, to have the model resume it.
const KindResume = "resume"

// StopReason is why a response ended.
type StopReason string

const (
	// StopEnd is a response the model ended.
	StopEnd StopReason = "end"
	// StopToolUse is a response that calls tools.
	StopToolUse StopReason = "tool_use"
	// StopLength is a response cut off at the output token limit.
	StopLength StopReason = "length"
	// StopSafety is a response the vendor's safety system ended or refused.
	StopSafety StopReason = "safety"
	// StopError is a response the vendor ended with an error.
	StopError StopReason = "error"
	// StopAborted is a response the run was cancelled during; it holds what
	// streamed before.
	StopAborted StopReason = "aborted"
	// StopOther is a reason the vendor gave that has no equivalent here.
	StopOther StopReason = "other"
)

func stopReason(f litellm.FinishReason) StopReason {
	switch f {
	case litellm.FinishReasonStop, "":
		return StopEnd
	case litellm.FinishReasonToolCall:
		return StopToolUse
	case litellm.FinishReasonLength:
		return StopLength
	case litellm.FinishReasonSafety:
		return StopSafety
	case litellm.FinishReasonError:
		return StopError
	default:
		return StopOther
	}
}

// Usage is what a response used, and what that cost when the model's
// pricing is known.
type Usage struct {
	litellm.Usage
	Cost *catalog.Cost `json:"cost,omitempty"`
}

// Add adds o, if any, to u: a running total, such as a session's. The total
// costs what the priced usage cost.
func (u *Usage) Add(o *Usage) {
	if o == nil {
		return
	}
	u.Usage.Add(o.Usage)
	if o.Cost == nil {
		return
	}
	// A new total: u may share its cost with the usage it was copied from.
	total := *o.Cost
	if u.Cost != nil {
		total.Add(*u.Cost)
	}
	u.Cost = &total
}

// usage is u priced, nil when it counts nothing. The cost stays unknown
// when the counts do not add up, or pricing has no rate for one of them.
func usage(u litellm.Usage, pricing *catalog.Pricing) *Usage {
	if u == (litellm.Usage{}) {
		return nil
	}
	out := &Usage{Usage: u}
	if pricing != nil {
		if cost, err := pricing.Cost(u); err == nil {
			out.Cost = &cost
		}
	}
	return out
}

// UserText returns a user message of text.
func UserText(text string) Message {
	return User(litellm.Text(text))
}

// User returns a user message of blocks.
func User(blocks ...litellm.Block) Message {
	return Message{Role: litellm.RoleUser, Blocks: blocks}
}

// ToolResult returns the message that answers the tool call id with result.
func ToolResult(id string, result Result) Message {
	return Message{
		Role:   litellm.RoleTool,
		Blocks: []litellm.Block{litellm.ToolResultBlock{ToolUseID: id, Content: result.Content, IsError: result.IsError}},
	}
}

// Text returns the text of m's text blocks, or of the tool result it holds.
func (m Message) Text() string {
	var sb strings.Builder
	for _, block := range m.Blocks {
		switch b := block.(type) {
		case litellm.TextBlock:
			sb.WriteString(b.Text)
		case litellm.ToolResultBlock:
			for _, c := range b.Content {
				if t, ok := c.(litellm.TextBlock); ok {
					sb.WriteString(t.Text)
				}
			}
		}
	}
	return sb.String()
}

// Reasoning returns the text of m's reasoning blocks.
func (m Message) Reasoning() string {
	var sb strings.Builder
	for _, block := range m.Blocks {
		if b, ok := block.(litellm.ReasoningBlock); ok {
			sb.WriteString(b.Text)
		}
	}
	return sb.String()
}

// LastResponse returns the last response in history, the zero Message when
// there is none.
func LastResponse(history []Message) Message {
	for i := len(history) - 1; i >= 0; i-- {
		if history[i].Role == litellm.RoleAssistant {
			return history[i]
		}
	}
	return Message{}
}

// ToolCalls returns the tool calls m makes.
func (m Message) ToolCalls() []litellm.ToolUseBlock {
	var calls []litellm.ToolUseBlock
	for _, block := range m.Blocks {
		if b, ok := block.(litellm.ToolUseBlock); ok {
			calls = append(calls, b)
		}
	}
	return calls
}

// ToolResult returns the tool result m holds.
func (m Message) ToolResult() (litellm.ToolResultBlock, bool) {
	for _, block := range m.Blocks {
		if b, ok := block.(litellm.ToolResultBlock); ok {
			return b, true
		}
	}
	return litellm.ToolResultBlock{}, false
}

func (m *Message) UnmarshalJSON(data []byte) error {
	type plain Message
	var v struct {
		plain
		Blocks []json.RawMessage `json:"blocks"`
	}
	if err := json.Unmarshal(data, &v); err != nil {
		return err
	}
	blocks, err := litellm.UnmarshalBlocks(v.Blocks)
	if err != nil {
		return err
	}
	*m = Message(v.plain)
	m.Blocks = blocks
	return nil
}

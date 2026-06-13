// Package proxy provides a ChatModel adapter that forwards LLM calls to a
// remote proxy server. The wire format ("frames") is bandwidth-optimized:
// frames carry only deltas, and the client reconstructs the full streaming
// message incrementally.
package proxy

import (
	"context"
	"encoding/json"
	"errors"
	"time"

	"github.com/voocel/agentcore"
)

// FrameType identifies a proxy stream frame.
type FrameType string

const (
	FrameTextDelta     FrameType = "text_delta"
	FrameThinkingDelta FrameType = "thinking_delta"
	FrameToolCallStart FrameType = "toolcall_start"
	FrameToolCallDelta FrameType = "toolcall_delta"
	FrameDone          FrameType = "done"
	FrameError         FrameType = "error"
)

// Frame is a single bandwidth-optimized event from a remote proxy server.
type Frame struct {
	Type       FrameType            `json:"type"`
	Delta      string               `json:"delta,omitempty"`
	ToolCallID string               `json:"tool_call_id,omitempty"`
	ToolName   string               `json:"tool_name,omitempty"`
	StopReason agentcore.StopReason `json:"stop_reason,omitempty"`
	Usage      *agentcore.Usage     `json:"usage,omitempty"`
	// Error carries a FrameError message. It is a string (not error) so it
	// survives the JSON wire round-trip — the whole point of this adapter.
	Error string `json:"error,omitempty"`
}

// StreamFn makes an LLM call through a remote proxy and returns a channel of
// bandwidth-optimized frames.
type StreamFn func(ctx context.Context, req *agentcore.LLMRequest) (<-chan Frame, error)

// Model implements agentcore.ChatModel by forwarding to a proxy stream
// function. It reconstructs streaming events from incoming frames.
//
// Usage:
//
//	m := proxy.New(myStreamFn)
//	agent := agentcore.NewAgent(agentcore.WithModel(m))
type Model struct {
	streamFn StreamFn
}

// New creates a Model that delegates to the given proxy stream function.
func New(fn StreamFn) *Model {
	return &Model{streamFn: fn}
}

// Generate collects the full streamed response synchronously.
func (p *Model) Generate(ctx context.Context, messages []agentcore.Message, tools []agentcore.ToolSpec, opts ...agentcore.CallOption) (*agentcore.LLMResponse, error) {
	ch, err := p.GenerateStream(ctx, messages, tools, opts...)
	if err != nil {
		return nil, err
	}
	var final agentcore.Message
	for ev := range ch {
		switch ev.Type {
		case agentcore.StreamEventDone:
			final = ev.Message
		case agentcore.StreamEventError:
			return nil, ev.Err
		}
	}
	return &agentcore.LLMResponse{Message: final}, nil
}

// GenerateStream converts proxy frames into standard StreamEvents.
func (p *Model) GenerateStream(ctx context.Context, messages []agentcore.Message, tools []agentcore.ToolSpec, opts ...agentcore.CallOption) (<-chan agentcore.StreamEvent, error) {
	frames, err := p.streamFn(ctx, &agentcore.LLMRequest{Messages: messages, Tools: tools})
	if err != nil {
		return nil, err
	}

	out := make(chan agentcore.StreamEvent, 100)
	go func() {
		defer close(out)

		var (
			partial      = agentcore.Message{Role: agentcore.RoleAssistant}
			textStarted  bool
			thinkStarted bool
		)

		for fr := range frames {
			switch fr.Type {
			case FrameTextDelta:
				idx := findOrCreate(&partial.Content, agentcore.ContentText)
				partial.Content[idx].Text += fr.Delta
				if !textStarted {
					textStarted = true
					out <- agentcore.StreamEvent{Type: agentcore.StreamEventTextStart, ContentIndex: idx, Message: partial}
				}
				out <- agentcore.StreamEvent{Type: agentcore.StreamEventTextDelta, ContentIndex: idx, Delta: fr.Delta, Message: partial}

			case FrameThinkingDelta:
				idx := findOrCreate(&partial.Content, agentcore.ContentThinking)
				partial.Content[idx].Thinking += fr.Delta
				if !thinkStarted {
					thinkStarted = true
					out <- agentcore.StreamEvent{Type: agentcore.StreamEventThinkingStart, ContentIndex: idx, Message: partial}
				}
				out <- agentcore.StreamEvent{Type: agentcore.StreamEventThinkingDelta, ContentIndex: idx, Delta: fr.Delta, Message: partial}

			case FrameToolCallStart:
				partial.Content = append(partial.Content, agentcore.ToolCallBlock(agentcore.ToolCall{
					ID:   fr.ToolCallID,
					Name: fr.ToolName,
				}))
				out <- agentcore.StreamEvent{Type: agentcore.StreamEventToolCallStart, Message: partial}

			case FrameToolCallDelta:
				if idx := lastToolCall(partial.Content); idx >= 0 && partial.Content[idx].ToolCall != nil {
					partial.Content[idx].ToolCall.Args = append(partial.Content[idx].ToolCall.Args, json.RawMessage(fr.Delta)...)
				}
				out <- agentcore.StreamEvent{Type: agentcore.StreamEventToolCallDelta, Delta: fr.Delta, Message: partial}

			case FrameDone:
				partial.StopReason = fr.StopReason
				partial.Usage = fr.Usage
				partial.Timestamp = time.Now()
				out <- agentcore.StreamEvent{Type: agentcore.StreamEventDone, Message: partial, StopReason: fr.StopReason}

			case FrameError:
				msg := fr.Error
				if msg == "" {
					msg = "proxy stream error"
				}
				out <- agentcore.StreamEvent{Type: agentcore.StreamEventError, Err: errors.New(msg)}
				return
			}
		}
	}()

	return out, nil
}

// SupportsTools reports that the proxy can handle tool calls.
func (p *Model) SupportsTools() bool { return true }

// findOrCreate returns the index of the last block of the given type, or
// appends a new empty block and returns its index.
func findOrCreate(blocks *[]agentcore.ContentBlock, ct agentcore.ContentType) int {
	for i := len(*blocks) - 1; i >= 0; i-- {
		if (*blocks)[i].Type == ct {
			return i
		}
	}
	switch ct {
	case agentcore.ContentText:
		*blocks = append(*blocks, agentcore.TextBlock(""))
	case agentcore.ContentThinking:
		*blocks = append(*blocks, agentcore.ThinkingBlock(""))
	}
	return len(*blocks) - 1
}

// lastToolCall returns the index of the last tool call block, or -1.
func lastToolCall(blocks []agentcore.ContentBlock) int {
	for i := len(blocks) - 1; i >= 0; i-- {
		if blocks[i].Type == agentcore.ContentToolCall {
			return i
		}
	}
	return -1
}

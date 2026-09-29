// Package proxy provides a ChatModel adapter that forwards LLM calls to a
// remote proxy server. The wire format ("frames") is bandwidth-optimized:
// frames carry only deltas, and the client reconstructs the full streaming
// message incrementally.
package proxy

import (
	"context"
	"errors"
	"fmt"
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
	FrameBlockEnd      FrameType = "block_end"
	FrameDone          FrameType = "done"
	FrameError         FrameType = "error"
)

// Frame is a single bandwidth-optimized event from a remote proxy server.
//
// Content frames address a content block of the message by Index: a text or
// thinking delta for the next index opens that block, as FrameToolCallStart
// opens a tool call, and FrameBlockEnd closes a block with the provider replay
// state the server received for it. Blocks may interleave.
type Frame struct {
	Type       FrameType                `json:"type"`
	Index      int                      `json:"index,omitempty"`
	Delta      string                   `json:"delta,omitempty"`
	ToolCallID string                   `json:"tool_call_id,omitempty"`
	ToolName   string                   `json:"tool_name,omitempty"`
	State      *agentcore.ProviderState `json:"state,omitempty"`
	StopReason agentcore.StopReason     `json:"stop_reason,omitempty"`
	Usage      *agentcore.Usage         `json:"usage,omitempty"`
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
		a := &assembler{msg: agentcore.Message{Role: agentcore.RoleAssistant}, out: out}
		for fr := range frames {
			switch fr.Type {
			case FrameDone:
				a.msg.StopReason = fr.StopReason
				a.msg.Usage = fr.Usage
				a.msg.Timestamp = time.Now()
				a.emit(agentcore.StreamEvent{Type: agentcore.StreamEventDone, StopReason: fr.StopReason})
			case FrameError:
				msg := fr.Error
				if msg == "" {
					msg = "proxy stream error"
				}
				out <- agentcore.StreamEvent{Type: agentcore.StreamEventError, Err: errors.New(msg)}
				return
			default:
				if err := a.apply(fr); err != nil {
					out <- agentcore.StreamEvent{Type: agentcore.StreamEventError, Err: err}
					return
				}
			}
		}
	}()

	return out, nil
}

// SupportsTools reports that the proxy can handle tool calls.
func (p *Model) SupportsTools() bool { return true }

// assembler rebuilds the streamed message from content frames.
type assembler struct {
	msg agentcore.Message
	out chan<- agentcore.StreamEvent
}

func (a *assembler) emit(ev agentcore.StreamEvent) {
	ev.Message = a.msg
	a.out <- ev
}

func (a *assembler) apply(fr Frame) error {
	next := len(a.msg.Content)
	switch fr.Type {
	case FrameTextDelta:
		if fr.Index == next {
			a.open(agentcore.TextBlock(""), agentcore.StreamEventTextStart)
		}
		block, err := a.block(fr.Index, agentcore.ContentText)
		if err != nil {
			return err
		}
		block.Text += fr.Delta
		a.emit(agentcore.StreamEvent{Type: agentcore.StreamEventTextDelta, ContentIndex: fr.Index, Delta: fr.Delta})

	case FrameThinkingDelta:
		if fr.Index == next {
			a.open(agentcore.ThinkingBlock(""), agentcore.StreamEventThinkingStart)
		}
		block, err := a.block(fr.Index, agentcore.ContentThinking)
		if err != nil {
			return err
		}
		block.Thinking += fr.Delta
		a.emit(agentcore.StreamEvent{Type: agentcore.StreamEventThinkingDelta, ContentIndex: fr.Index, Delta: fr.Delta})

	case FrameToolCallStart:
		if fr.Index != next {
			return fmt.Errorf("proxy: tool call starts block %d, but the next block is %d", fr.Index, next)
		}
		a.open(agentcore.ToolCallBlock(agentcore.ToolCall{ID: fr.ToolCallID, Name: fr.ToolName}), agentcore.StreamEventToolCallStart)

	case FrameToolCallDelta:
		block, err := a.block(fr.Index, agentcore.ContentToolCall)
		if err != nil {
			return err
		}
		block.ToolCall.Args = append(block.ToolCall.Args, fr.Delta...)
		a.emit(agentcore.StreamEvent{Type: agentcore.StreamEventToolCallDelta, ToolID: block.ToolCall.ID, ContentIndex: fr.Index, Delta: fr.Delta})

	case FrameBlockEnd:
		if fr.Index < 0 || fr.Index >= next {
			return fmt.Errorf("proxy: block %d ends before it starts", fr.Index)
		}
		block := &a.msg.Content[fr.Index]
		block.State = fr.State
		ev := agentcore.StreamEvent{ContentIndex: fr.Index}
		switch block.Type {
		case agentcore.ContentText:
			ev.Type = agentcore.StreamEventTextEnd
		case agentcore.ContentThinking:
			ev.Type = agentcore.StreamEventThinkingEnd
		case agentcore.ContentToolCall:
			completed := *block.ToolCall
			ev.Type, ev.ToolID, ev.CompletedToolCall = agentcore.StreamEventToolCallEnd, completed.ID, &completed
		}
		a.emit(ev)
	}
	return nil
}

// open appends the next content block and emits its start event.
func (a *assembler) open(block agentcore.ContentBlock, start agentcore.StreamEventType) {
	a.msg.Content = append(a.msg.Content, block)
	ev := agentcore.StreamEvent{Type: start, ContentIndex: len(a.msg.Content) - 1}
	if block.ToolCall != nil {
		ev.ToolID = block.ToolCall.ID
	}
	a.emit(ev)
}

// block returns content block i. Frames come from a remote server, so one
// that addresses a missing block or changes a block's type fails the stream.
func (a *assembler) block(i int, ct agentcore.ContentType) (*agentcore.ContentBlock, error) {
	if i < 0 || i >= len(a.msg.Content) || a.msg.Content[i].Type != ct {
		return nil, fmt.Errorf("proxy: frame addresses block %d as %s", i, ct)
	}
	return &a.msg.Content[i], nil
}

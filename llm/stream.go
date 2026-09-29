package llm

import (
	"github.com/voocel/agentcore"
	"github.com/voocel/litellm"
)

// forward turns a litellm stream into agentcore stream events. The Client
// validates the block lifecycle, so litellm block i is content block i, and
// the streamed message equals Generate's.
func (l *LiteLLMAdapter) forward(stream litellm.Stream, out chan<- agentcore.StreamEvent) {
	partial := agentcore.Message{Role: agentcore.RoleAssistant}
	emit := func(ev agentcore.StreamEvent) {
		ev.Message = partial
		out <- ev
	}
	open := func(block agentcore.ContentBlock, start agentcore.StreamEventType, toolID string) {
		partial.Content = append(partial.Content, block)
		emit(agentcore.StreamEvent{Type: start, ContentIndex: len(partial.Content) - 1, ToolID: toolID})
	}
	text := func(index int, delta string) {
		if delta != "" {
			partial.Content[index].Text += delta
			emit(agentcore.StreamEvent{Type: agentcore.StreamEventTextDelta, ContentIndex: index, Delta: delta})
		}
	}
	thinking := func(index int, delta string) {
		if delta != "" {
			partial.Content[index].Thinking += delta
			emit(agentcore.StreamEvent{Type: agentcore.StreamEventThinkingDelta, ContentIndex: index, Delta: delta})
		}
	}

	resp, err := litellm.Handle(stream, func(ev litellm.Event) error {
		switch e := ev.(type) {
		case litellm.BlockStart:
			switch b := e.Block.(type) {
			case litellm.TextBlock:
				open(agentcore.TextBlock(""), agentcore.StreamEventTextStart, "")
				text(e.Index, b.Text)
			case litellm.ReasoningBlock:
				open(agentcore.ThinkingBlock(""), agentcore.StreamEventThinkingStart, "")
				thinking(e.Index, b.Text)
			case litellm.ToolUseBlock:
				open(agentcore.ToolCallBlock(agentcore.ToolCall{ID: b.ID, Name: b.Name}), agentcore.StreamEventToolCallStart, b.ID)
			}
		case litellm.TextDelta:
			text(e.Index, e.Text)
		case litellm.ReasoningDelta:
			thinking(e.Index, e.Text)
		case litellm.ToolUseDelta:
			if e.Arguments != "" {
				emit(agentcore.StreamEvent{Type: agentcore.StreamEventToolCallDelta, ToolID: partial.Content[e.Index].ToolCall.ID, ContentIndex: e.Index, Delta: e.Arguments})
			}
		case litellm.BlockEnd:
			// A Client stream delivers the completed block.
			block := &partial.Content[e.Index]
			switch b := e.Block.(type) {
			case litellm.TextBlock:
				block.State = toState(b.State)
				emit(agentcore.StreamEvent{Type: agentcore.StreamEventTextEnd, ContentIndex: e.Index})
			case litellm.ReasoningBlock:
				block.State = toState(b.State)
				emit(agentcore.StreamEvent{Type: agentcore.StreamEventThinkingEnd, ContentIndex: e.Index})
			case litellm.ToolUseBlock:
				completed := buildToolCall(b.ID, b.Name, string(b.Arguments))
				*block = agentcore.ToolCallBlock(completed)
				block.State = toState(b.State)
				emit(agentcore.StreamEvent{Type: agentcore.StreamEventToolCallEnd, ToolID: completed.ID, ContentIndex: e.Index, CompletedToolCall: &completed})
			}
		}
		return nil
	})
	if err != nil {
		out <- agentcore.StreamEvent{Type: agentcore.StreamEventError, Err: wrapProviderError(err)}
		return
	}
	l.finish(&partial, resp)
	emit(agentcore.StreamEvent{Type: agentcore.StreamEventDone, StopReason: partial.StopReason})
}

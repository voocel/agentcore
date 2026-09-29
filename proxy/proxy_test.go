package proxy

import (
	"context"
	"encoding/json"
	"slices"
	"testing"

	"github.com/voocel/agentcore"
)

func frameModel(frames ...Frame) *Model {
	return New(func(context.Context, *agentcore.LLMRequest) (<-chan Frame, error) {
		ch := make(chan Frame, len(frames))
		for _, f := range frames {
			ch <- f
		}
		close(ch)
		return ch, nil
	})
}

// Blocks keep their order and boundaries, even interleaved, and each closes
// with its replay state.
func TestGenerateStreamRebuildsBlocks(t *testing.T) {
	signed := &agentcore.ProviderState{Provider: "anthropic", Data: json.RawMessage(`{"signature":"s"}`)}
	call := &agentcore.ProviderState{Provider: "gemini", Data: json.RawMessage(`{"thoughtSignature":"t"}`)}
	model := frameModel(
		Frame{Type: FrameThinkingDelta, Index: 0, Delta: "plan"},
		Frame{Type: FrameBlockEnd, Index: 0, State: signed},
		Frame{Type: FrameToolCallStart, Index: 1, ToolCallID: "call_1", ToolName: "read"},
		Frame{Type: FrameToolCallDelta, Index: 1, Delta: `{"path":`},
		Frame{Type: FrameThinkingDelta, Index: 2, Delta: "check"},
		Frame{Type: FrameToolCallDelta, Index: 1, Delta: `"a"}`},
		Frame{Type: FrameBlockEnd, Index: 1, State: call},
		Frame{Type: FrameTextDelta, Index: 3, Delta: "done"},
		Frame{Type: FrameBlockEnd, Index: 2},
		Frame{Type: FrameBlockEnd, Index: 3},
		Frame{Type: FrameDone, StopReason: agentcore.StopReasonToolUse},
	)
	events, err := model.GenerateStream(context.Background(), nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	var (
		final     agentcore.Message
		ends      []agentcore.StreamEventType
		completed *agentcore.ToolCall
	)
	for ev := range events {
		switch ev.Type {
		case agentcore.StreamEventError:
			t.Fatal(ev.Err)
		case agentcore.StreamEventTextEnd, agentcore.StreamEventThinkingEnd, agentcore.StreamEventToolCallEnd:
			ends = append(ends, ev.Type)
			if ev.CompletedToolCall != nil {
				completed = ev.CompletedToolCall
			}
		case agentcore.StreamEventDone:
			final = ev.Message
		}
	}

	thinking := agentcore.ThinkingBlock("plan")
	thinking.State = signed
	tool := agentcore.ToolCallBlock(agentcore.ToolCall{ID: "call_1", Name: "read", Args: json.RawMessage(`{"path":"a"}`)})
	tool.State = call
	want := []agentcore.ContentBlock{thinking, tool, agentcore.ThinkingBlock("check"), agentcore.TextBlock("done")}
	got, _ := json.Marshal(final.Content)
	wantJSON, _ := json.Marshal(want)
	if string(got) != string(wantJSON) || final.StopReason != agentcore.StopReasonToolUse {
		t.Fatalf("content = %s\nwant %s", got, wantJSON)
	}
	if !slices.Equal(ends, []agentcore.StreamEventType{agentcore.StreamEventThinkingEnd, agentcore.StreamEventToolCallEnd, agentcore.StreamEventThinkingEnd, agentcore.StreamEventTextEnd}) {
		t.Fatalf("end events = %v", ends)
	}
	if completed == nil || completed.ID != "call_1" || string(completed.Args) != `{"path":"a"}` {
		t.Fatalf("completed tool call = %+v", completed)
	}
}

func TestGenerateStreamRejectsMisaddressedFrames(t *testing.T) {
	for name, frames := range map[string][]Frame{
		"skipped block": {{Type: FrameTextDelta, Index: 1, Delta: "x"}},
		"changed type":  {{Type: FrameTextDelta, Delta: "x"}, {Type: FrameThinkingDelta, Delta: "y"}},
		"late start":    {{Type: FrameTextDelta, Delta: "x"}, {Type: FrameToolCallStart, Index: 0}},
		"unopened end":  {{Type: FrameBlockEnd}},
	} {
		if _, err := frameModel(append(frames, Frame{Type: FrameDone})...).Generate(context.Background(), nil, nil); err == nil {
			t.Fatalf("%s: accepted", name)
		}
	}
}

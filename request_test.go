package agentcore

import (
	"encoding/json"
	"reflect"
	"testing"
	"time"

	"github.com/voocel/litellm"
	"github.com/voocel/litellm/catalog"
)

func assistant(stop StopReason, blocks ...litellm.Block) Message {
	return Message{Role: litellm.RoleAssistant, Blocks: blocks, Stop: stop, Time: time.Now()}
}

// The model reads the history without what failed or said nothing, and with
// every tool call answered by exactly one result.
func TestModelMessages(t *testing.T) {
	history := []Message{
		UserText("go"),
		assistant(StopError, litellm.Text("oops")),
		assistant(StopAborted, litellm.Text("half")),
		assistant(StopEnd, litellm.ReasoningBlock{Text: "thinking only"}),
		assistant(StopEnd),
		assistant(StopToolUse, call("c1", "echo", `{}`), call("c2", "echo", `{}`)),
		ToolResult("c1", TextResult("one")),
		ToolResult("gone", TextResult("orphan")),
		UserText("next"),
		assistant(StopToolUse, call("c3", "echo", `{}`)),
	}
	got := modelMessages(history)
	var shape []string
	for _, m := range got {
		s := string(m.Role)
		if m.Role == litellm.RoleTool {
			r := m.Blocks[0].(litellm.ToolResultBlock)
			s += ":" + r.ToolUseID
			if r.IsError {
				s += "!"
			}
		}
		shape = append(shape, s)
	}
	want := []string{"user", "assistant", "tool:c1", "tool:c2!", "user", "assistant", "tool:c3!"}
	if !reflect.DeepEqual(shape, want) {
		t.Fatalf("model reads %v, want %v", shape, want)
	}
}

func TestBuildCall(t *testing.T) {
	maxTokens := 100
	cfg := Config{
		Model:  Model{Request: litellm.Request{Model: "m", MaxTokens: &maxTokens}},
		System: []litellm.Block{litellm.TextBlock{Text: "sys", Cache: &litellm.CacheControl{}}},
		Tools: []Tool{
			{Name: "read", Schema: map[string]any{"type": "object"}},
			{Name: "deploy", Deferred: true},
			{Name: "rollback", Deferred: true},
		},
		Cache: &litellm.CacheControl{},
	}
	reasoning := litellm.ReasoningBlock{Text: "hm"}
	history := []Message{
		UserText("go"),
		assistant(StopToolUse, call("c1", "tool_search", `{}`)),
		ToolResult("c1", Result{Content: []litellm.Block{litellm.ToolReferenceBlock{ToolName: "deploy"}}}),
		assistant(StopEnd, litellm.Text("found it"), reasoning),
	}
	c := BuildCall(cfg, history)
	req := c.Request
	if req.Model != "m" || *req.MaxTokens != 100 || len(req.Messages) != 5 || req.Messages[0].Role != litellm.RoleSystem {
		t.Fatalf("request = %#v", req)
	}
	var names []string
	for _, tool := range req.Tools {
		names = append(names, tool.Name)
	}
	if !reflect.DeepEqual(names, []string{"read", "deploy"}) || string(req.Tools[0].Parameters) != `{"type":"object"}` {
		t.Fatalf("tools = %v", req.Tools)
	}
	last := req.Messages[4].Blocks
	if text := last[0].(litellm.TextBlock); text.Cache == nil {
		t.Fatalf("cache breakpoint on %#v", last)
	}
	if history[3].Blocks[0].(litellm.TextBlock).Cache != nil {
		t.Fatal("the breakpoint changed the history")
	}
	if c.Request.Messages[3].Blocks[0].(litellm.ToolResultBlock).Cache != nil {
		t.Fatal("a breakpoint before the last message")
	}
}

func TestMessageJSON(t *testing.T) {
	want := []Message{
		SummaryMessage("did things"),
		{
			Role:     litellm.RoleAssistant,
			Blocks:   []litellm.Block{litellm.ReasoningBlock{Text: "hm", State: &litellm.ProviderState{Provider: "p", Data: json.RawMessage(`{"s":1}`)}}, call("c1", "read", `{"path":"a"}`)},
			Stop:     StopToolUse,
			Usage:    &Usage{Input: 10, Output: 2, CacheRead: 8},
			Provider: "p",
			Model:    "m",
			Time:     time.Now(),
		},
		ToolResult("c1", ErrorResult("no such file")),
	}
	for i := range want {
		want[i].Time = want[i].Time.Round(0)
	}
	data, err := json.Marshal(want)
	if err != nil {
		t.Fatal(err)
	}
	var got []Message
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatal(err)
	}
	for i := range got {
		if !got[i].Time.Equal(want[i].Time) {
			t.Fatalf("time %d = %v, want %v", i, got[i].Time, want[i].Time)
		}
		got[i].Time = want[i].Time
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("round trip:\ngot  %#v\nwant %#v", got, want)
	}
	if got[0].Kind != KindSummary || SummaryText(got[0]) != "did things" {
		t.Fatalf("summary = %#v", got[0])
	}
}

func TestUsageAdd(t *testing.T) {
	var total Usage
	priced := &Usage{Input: 10, Output: 2, Cost: &catalog.Cost{Total: 0.5}}
	total.Add(priced)
	total.Add(nil)
	total.Add(&Usage{Input: 1})
	total.Add(priced)
	if total.Input != 21 || total.Output != 4 || total.Cost == nil || total.Cost.Total != 1 || priced.Cost.Total != 0.5 {
		t.Fatalf("total = %+v, cost %+v", total, total.Cost)
	}
}

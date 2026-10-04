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
	for _, tool := range req.OfferedTools() {
		names = append(names, tool.Name)
	}
	if len(req.Tools) != 3 || !req.Tools[2].Deferred || !reflect.DeepEqual(names, []string{"read", "deploy"}) || string(req.Tools[0].Parameters) != `{"type":"object"}` {
		t.Fatalf("tools = %v", req.Tools)
	}
	last := req.Messages[4].Blocks
	if text := last[0].(litellm.TextBlock); text.Cache == nil {
		t.Fatalf("cache breakpoint on %#v", last)
	}
	if history[3].Blocks[0].(litellm.TextBlock).Cache != nil {
		t.Fatal("the breakpoint changed the history")
	}
	if c.Request.Messages[3].Blocks[0].(litellm.ToolResultBlock).Cache == nil {
		t.Fatal("no breakpoint where the call before ended")
	}
}

// A call marks where it ends and where the call before ended, which that
// call wrote to the cache.
func TestBuildCallMarksTheCallBefore(t *testing.T) {
	history := []Message{
		UserText("go"),
		assistant(StopToolUse, call("c1", "read", `{}`)),
		ToolResult("c1", TextResult("one")),
		assistant(StopToolUse, call("c2", "read", `{}`), call("c3", "read", `{}`)),
		ToolResult("c2", TextResult("two")),
		ToolResult("c3", TextResult("three")),
	}
	msgs := BuildCall(Config{Cache: &litellm.CacheControl{TTL: "1h"}}, history).Request.Messages
	var marked []int
	for i, m := range msgs {
		for _, b := range m.Blocks {
			if r, ok := b.(litellm.ToolResultBlock); ok && r.Cache != nil && r.Cache.TTL == "1h" {
				marked = append(marked, i)
			}
		}
	}
	if !reflect.DeepEqual(marked, []int{2, 5}) {
		t.Fatalf("breakpoints after messages %v, want [2 5]", marked)
	}
}

func TestMessageJSON(t *testing.T) {
	want := []Message{
		SummaryMessage("did things"),
		{
			Role:     litellm.RoleAssistant,
			Blocks:   []litellm.Block{litellm.ReasoningBlock{Text: "hm", State: &litellm.ProviderState{Provider: "p", Data: json.RawMessage(`{"s":1}`)}}, call("c1", "read", `{"path":"a"}`)},
			Stop:     StopToolUse,
			Usage:    &Usage{Usage: litellm.Usage{InputTokens: 10, OutputTokens: 2, ReasoningTokens: 1, CacheReadTokens: 8, CacheWrite1hTokens: 1}},
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
	priced := &Usage{Usage: litellm.Usage{InputTokens: 10, OutputTokens: 2, ReasoningTokens: 1}, Cost: &catalog.Cost{Total: 0.5}}
	total.Add(priced)
	total.Add(nil)
	total.Add(&Usage{Usage: litellm.Usage{InputTokens: 1}})
	total.Add(priced)
	if total.InputTokens != 21 || total.OutputTokens != 4 || total.ReasoningTokens != 2 || total.Cost == nil || total.Cost.Total != 1 || priced.Cost.Total != 0.5 {
		t.Fatalf("total = %+v, cost %+v", total, total.Cost)
	}
}

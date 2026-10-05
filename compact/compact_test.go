package compact

import (
	"context"
	"errors"
	"reflect"
	"strings"
	"testing"

	"github.com/voocel/agentcore"
	"github.com/voocel/litellm"
	"github.com/voocel/litellm/litellmtest"
)

func assistant(blocks ...litellm.Block) agentcore.Message {
	return agentcore.Message{Role: litellm.RoleAssistant, Blocks: blocks}
}

func readCall(id, args string) litellm.ToolUseBlock {
	return litellm.ToolUseBlock{ID: id, Name: "read", Arguments: args}
}

func history() []agentcore.Message {
	return []agentcore.Message{
		agentcore.UserText(strings.Repeat("a", 4000)),
		assistant(
			readCall("1", `{"file_path":"old.go"}`),
			litellm.ToolUseBlock{ID: "2", Name: "edit", Arguments: `{"file_path":"new.go"}`},
		),
		agentcore.ToolResult("1", agentcore.TextResult("package old")),
		agentcore.ToolResult("2", agentcore.TextResult("ok")),
		agentcore.UserText("keep " + strings.Repeat("k", 12_000)),
	}
}

// callFor builds the calls the loop makes, to a model replying with p.
func callFor(t *testing.T, p litellm.Provider) func([]agentcore.Message) agentcore.Call {
	t.Helper()
	client, err := litellm.New(p)
	if err != nil {
		t.Fatal(err)
	}
	cfg := agentcore.Config{
		Model:  agentcore.Model{Client: client, Request: litellm.Request{Model: "m", Thinking: &litellm.Thinking{Effort: "high"}}},
		System: []litellm.Block{litellm.Text("parent system")},
		Tools: []agentcore.Tool{
			{Name: "read", Description: "read a file", Schema: map[string]any{"type": "object"}},
			{Name: "deploy", Description: "deploy", Schema: map[string]any{"type": "object"}, Deferred: true},
		},
		Cache: &litellm.CacheControl{},
	}
	return func(h []agentcore.Message) agentcore.Call { return agentcore.BuildCall(cfg, h) }
}

func TestExtractFileOps(t *testing.T) {
	msgs := []agentcore.Message{assistant(
		readCall("1", `{"file_path":"a.go"}`),
		readCall("2", `{"file_path":"b.go"}`),
		litellm.ToolUseBlock{ID: "3", Name: "edit", Arguments: `{"file_path":"b.go"}`},
		litellm.ToolUseBlock{ID: "4", Name: "write", Arguments: `{"file_path":"c.go"}`},
		readCall("5", `{"file_path":"a.go"}`),
	)}
	read, modified := extractFileOps(msgs)
	if strings.Join(read, ",") != "a.go" || strings.Join(modified, ",") != "b.go,c.go" {
		t.Fatalf("read %v, modified %v", read, modified)
	}
}

// The history is replaced by its summary, but for the prompt at its end the
// model has yet to answer; the summary call's usage is reported.
func TestCompactReplacesOldHistoryWithSummary(t *testing.T) {
	reply := litellmtest.Text("<analysis>a</analysis><summary>checkpoint body</summary>")
	reply.Usage = litellm.Usage{InputTokens: 900, OutputTokens: 40}
	p := litellmtest.New(reply)
	msgs := history()
	c, err := Summarizer{}.Compact(context.Background(), msgs, callFor(t, p))
	if err != nil {
		t.Fatal(err)
	}
	if c == nil || len(c.Messages) != 2 || c.Replaced != 4 || c.Messages[0].Kind != agentcore.KindSummary {
		t.Fatalf("compaction = %+v", c)
	}
	if c.Usage == nil || c.Usage.InputTokens != 900 || c.Usage.OutputTokens != 40 {
		t.Fatalf("usage = %+v", c.Usage)
	}
	text := agentcore.SummaryText(c.Messages[0])
	if !strings.HasPrefix(text, "checkpoint body") ||
		!strings.Contains(text, "<read-files>\nold.go\n</read-files>") ||
		!strings.Contains(text, "<modified-files>\nnew.go\n</modified-files>") {
		t.Fatalf("summary = %q", text)
	}
	if !strings.HasPrefix(c.Messages[1].Text(), "keep ") {
		t.Fatal("the recent message was not kept")
	}
}

// The summary request is the conversation's request for the part it
// replaces, which earlier calls sent, plus one message, so the provider
// serves that part from the prompt cache; the part kept is not summarized
// too. Its instruction says it is not the user's: summarizers once recorded
// it as the user telling the agent to stop, and every later summary kept
// that.
func TestCompactForksTheConversation(t *testing.T) {
	p := litellmtest.New(litellmtest.Text("<summary>checkpoint body</summary>"))
	msgs := history()
	call := callFor(t, p)
	if _, err := (Summarizer{}).Compact(context.Background(), msgs, call); err != nil {
		t.Fatal(err)
	}
	sent := p.Requests()
	if len(sent) != 1 {
		t.Fatalf("%d calls, want one fork", len(sent))
	}
	got := sent[0]
	want := call(msgs[:4]).Request
	if len(got.Messages) != len(want.Messages)+1 {
		t.Fatalf("the fork sends %d messages, the replaced part %d", len(got.Messages), len(want.Messages))
	}
	if !reflect.DeepEqual(got.Messages[:len(want.Messages)], want.Messages) || !reflect.DeepEqual(got.Tools, want.Tools) || !reflect.DeepEqual(got.Thinking, want.Thinking) {
		t.Fatal("the fork differs from the conversation's request")
	}
	instruction := agentcore.Message{Blocks: got.Messages[len(got.Messages)-1].Blocks}.Text()
	for _, phrase := range []string{
		"not a message from the user and not part of the conversation",
		"Leave this request out of the checkpoint",
		"the user stated in the conversation",
	} {
		if !strings.Contains(instruction, phrase) {
			t.Fatalf("instruction lacks %q", phrase)
		}
	}
}

// When the fork answers without a tagged summary, or does not fit, a
// transcript is summarized instead, without tools.
func TestCompactFallsBackToTheTranscript(t *testing.T) {
	overflow := litellm.NewError("test", litellm.ErrorTypeContextOverflow, "too long", nil)
	for name, fork := range map[string]litellmtest.Reply{
		"untagged answer": litellmtest.Text("Sure, the bug is in the retry loop; fixing it next."),
		"overflow":        litellmtest.Fail(overflow),
	} {
		p := litellmtest.New(fork, litellmtest.Text("checkpoint body"))
		msgs := history()
		c, err := Summarizer{}.Compact(context.Background(), msgs, callFor(t, p))
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		sent := p.Requests()
		transcript := sent[len(sent)-1]
		if len(sent) != 2 || transcript.Tools != nil || transcript.Thinking != nil ||
			!strings.Contains(agentcore.Message{Blocks: transcript.Messages[1].Blocks}.Text(), "<conversation>") {
			t.Fatalf("%s: calls %d, transcript request %#v", name, len(sent), transcript)
		}
		if !strings.HasPrefix(agentcore.SummaryText(c.Messages[0]), "checkpoint body") {
			t.Fatalf("%s: summary %q", name, agentcore.SummaryText(c.Messages[0]))
		}
	}

	boom := errors.New("provider down")
	p := litellmtest.New(litellmtest.Fail(boom))
	msgs := history()
	if _, err := (Summarizer{}).Compact(context.Background(), msgs, callFor(t, p)); !errors.Is(err, boom) {
		t.Fatalf("fork error = %v", err)
	}
}

// The transcript path folds the prior summary in through its own block.
func TestCompactFoldsInThePreviousSummary(t *testing.T) {
	p := litellmtest.New(litellmtest.Text("no tags"), litellmtest.Text("fresh"))
	const old = "UNIQUE-PRIOR-SUMMARY-TEXT"
	msgs := append([]agentcore.Message{agentcore.SummaryMessage(old)}, history()...)
	if _, err := (Summarizer{}).Compact(context.Background(), msgs, callFor(t, p)); err != nil {
		t.Fatal(err)
	}
	prompt := agentcore.Message{Blocks: p.Requests()[1].Messages[1].Blocks}.Text()
	if strings.Count(prompt, old) != 1 || !strings.Contains(prompt, "<previous-summary>\n"+old) {
		t.Fatalf("the prior summary must appear once, in its own block:\n%s", prompt)
	}

	// Nothing new to cut: the prior summary is not reworded.
	p = litellmtest.New()
	msgs = []agentcore.Message{agentcore.SummaryMessage("prior"), agentcore.UserText(strings.Repeat("b ", 8000))}
	c, err := Summarizer{}.Compact(context.Background(), msgs, callFor(t, p))
	if err != nil || c != nil || len(p.Requests()) != 0 {
		t.Fatalf("compacted only a summary: %+v, %v", c, err)
	}
}

func TestTruncateForSummaryKeepsRunesWhole(t *testing.T) {
	if got := truncateForSummary(strings.Repeat("中", 10), 7); got != "中中..." {
		t.Fatalf("got %q", got)
	}
}

// Mid-task, nothing of the history is replayed: no response, with the
// reasoning bound to what came before it, and no tool load, so the deferred
// tools it loaded are found again.
func TestCompactReplaysNothing(t *testing.T) {
	p := litellmtest.New(litellmtest.Text("<summary>checkpoint body</summary>"))
	search := litellm.ToolUseBlock{ID: "s1", Name: "tool_search", Arguments: `{"query":"select:deploy"}`}
	msgs := []agentcore.Message{
		agentcore.UserText(strings.Repeat("a", 4000)),
		assistant(litellm.ReasoningBlock{Text: "find a tool"}, search),
		agentcore.ToolResult("s1", agentcore.Result{Content: []litellm.Block{litellm.ToolReferenceBlock{ToolName: "deploy"}}}),
		assistant(litellm.ReasoningBlock{Text: "read it"}, readCall("r1", `{"file_path":"a.go"}`)),
		agentcore.ToolResult("r1", agentcore.TextResult("package a")),
	}
	call := callFor(t, p)
	c, err := Summarizer{}.Compact(context.Background(), msgs, call)
	if err != nil || c == nil || len(c.Messages) != 1 || c.Replaced != len(msgs) {
		t.Fatalf("compaction %+v, %v", c, err)
	}
	req := call(c.Messages).Request
	for _, tool := range req.OfferedTools() {
		if tool.Name == "deploy" {
			t.Fatal("a tool load survived the compaction")
		}
	}
	if !strings.Contains(c.Messages[0].Text(), "Continue from where it leaves off") {
		t.Fatalf("the summary does not carry the work on: %q", c.Messages[0].Text())
	}
}

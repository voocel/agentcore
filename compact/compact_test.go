package compact

import (
	"context"
	"errors"
	"reflect"
	"strings"
	"testing"

	"github.com/voocel/agentcore"
	"github.com/voocel/agentcore/tools"
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

// summaryOf returns the text of the summary c put in place.
func summaryOf(t *testing.T, c *agentcore.Compaction) string {
	t.Helper()
	for _, m := range c.Messages {
		if m.Kind == agentcore.KindSummary {
			return agentcore.SummaryText(m)
		}
	}
	t.Fatalf("no summary among %d messages", len(c.Messages))
	return ""
}

// The history is replaced by the user's words and its summary, but for the
// prompt at its end the model has yet to answer; the summary call's usage is
// reported.
func TestCompactReplacesOldHistoryWithSummary(t *testing.T) {
	reply := litellmtest.Text("<analysis>a</analysis><summary>checkpoint body</summary>")
	reply.Usage = litellm.Usage{InputTokens: 900, OutputTokens: 40}
	p := litellmtest.New(reply)
	msgs := history()
	c, err := Summarizer{Notes: tools.FileOps}.Compact(context.Background(), msgs, callFor(t, p))
	if err != nil {
		t.Fatal(err)
	}
	if c == nil || len(c.Messages) != 3 || c.Replaced != 4 || c.Messages[0].Text() != msgs[0].Text() || c.Messages[1].Kind != agentcore.KindSummary {
		t.Fatalf("compaction of %d messages, %d replaced", len(c.Messages), c.Replaced)
	}
	if c.Usage == nil || c.Usage.InputTokens != 900 || c.Usage.OutputTokens != 40 {
		t.Fatalf("usage = %+v", c.Usage)
	}
	text := summaryOf(t, c)
	if !strings.HasPrefix(text, "checkpoint body") ||
		!strings.Contains(text, "<read-files>\nold.go\n</read-files>") ||
		!strings.Contains(text, "<modified-files>\nnew.go\n</modified-files>") {
		t.Fatalf("summary = %q", text)
	}
	if !strings.HasPrefix(c.Messages[2].Text(), "keep ") {
		t.Fatal("the recent message was not kept")
	}
}

// What the user said is kept word for word, the latest first, up to a
// budget: the one overflowing it is cut in the middle, and the messages the
// application added are left to the summary.
func TestCompactKeepsTheUsersLatestWords(t *testing.T) {
	added := agentcore.UserText("<system-reminder>context</system-reminder>")
	added.Kind = "context:env"
	long := "fix the parser " + strings.Repeat("log line\n", 20_000) + " and add a test"
	msgs := []agentcore.Message{
		agentcore.UserText("first, an old ask"),
		assistant(litellm.Text("done")),
		added,
		agentcore.UserText(long),
		assistant(litellm.Text("on it")),
		agentcore.UserText("also keep the API stable"),
		assistant(litellm.Text("ok")),
	}
	c, err := Summarizer{}.Compact(context.Background(), msgs, callFor(t, litellmtest.New(litellmtest.Text("<summary>s</summary>"))))
	if err != nil {
		t.Fatal(err)
	}
	if len(c.Messages) != 3 || c.Messages[2].Kind != agentcore.KindSummary || c.Messages[1].Text() != "also keep the API stable" {
		t.Fatalf("kept %d messages", len(c.Messages))
	}
	cut := c.Messages[0].Text()
	if !strings.HasPrefix(cut, "fix the parser ") || !strings.HasSuffix(cut, " and add a test") || !strings.Contains(cut, "characters cut ...]") || len(cut) >= len(long) {
		t.Fatalf("the long ask was not cut in the middle: %d of %d bytes", len(cut), len(long))
	}
	if kept := agentcore.EstimateMessage(c.Messages[0]) + agentcore.EstimateMessage(c.Messages[1]); kept > keptWordsTokens+20 {
		t.Fatalf("kept %d tokens of the user's words", kept)
	}

	// Compacted again with nothing new, the words kept are not summarized
	// again.
	p := litellmtest.New()
	again, err := Summarizer{}.Compact(context.Background(), append(c.Messages, agentcore.UserText("next")), callFor(t, p))
	if err != nil || again != nil || len(p.Requests()) != 0 {
		t.Fatalf("compacted nothing new: %v, %v", again, err)
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
		if summary := summaryOf(t, c); !strings.HasPrefix(summary, "checkpoint body") {
			t.Fatalf("%s: summary %q", name, summary)
		}
	}

	// A summary cut off at the output limit is no checkpoint.
	cut := litellmtest.Text("## Goal\nhalf a summ")
	cut.FinishReason = litellm.FinishReasonLength
	p := litellmtest.New(litellmtest.Text("untagged"), cut)
	if _, err := (Summarizer{}).Compact(context.Background(), history(), callFor(t, p)); err == nil || !strings.Contains(err.Error(), "cut off") {
		t.Fatalf("a cut summary: %v", err)
	}

	boom := errors.New("provider down")
	p = litellmtest.New(litellmtest.Fail(boom))
	if _, err := (Summarizer{}).Compact(context.Background(), history(), callFor(t, p)); !errors.Is(err, boom) {
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

// Mid-task, no response is replayed, with the reasoning bound to what came
// before it, nor a tool load, so the deferred tools it loaded are found
// again: only the user's words and the summary stand.
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
	if err != nil || c == nil || len(c.Messages) != 2 || c.Messages[0].Role != litellm.RoleUser || c.Messages[0].Kind != "" || c.Replaced != len(msgs) {
		t.Fatalf("compaction %+v, %v", c, err)
	}
	req := call(c.Messages).Request
	for _, tool := range req.OfferedTools() {
		if tool.Name == "deploy" {
			t.Fatal("a tool load survived the compaction")
		}
	}
	if !strings.Contains(c.Messages[1].Text(), "Continue from where it leaves off") {
		t.Fatalf("the summary does not carry the work on: %q", c.Messages[1].Text())
	}
}

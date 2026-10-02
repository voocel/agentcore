package compact

import (
	"context"
	"errors"
	"reflect"
	"strconv"
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

func toolGroup(id, result string) []agentcore.Message {
	return []agentcore.Message{
		assistant(readCall(id, `{}`)),
		agentcore.ToolResult(id, agentcore.TextResult(result)),
	}
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

func TestCutPoint(t *testing.T) {
	msgs := []agentcore.Message{agentcore.UserText("old"), assistant(litellm.Text("done")), agentcore.UserText(strings.Repeat("recent", 100))}
	if cut := cutPoint(msgs, 10); cut != 2 {
		t.Fatalf("cut at %d, want the recent user message", cut)
	}

	// A cut landing on a tool result retreats to the call that issued it.
	msgs = []agentcore.Message{
		agentcore.UserText(strings.Repeat("b", 400)),
		assistant(readCall("1", `{}`), readCall("2", `{}`)),
		agentcore.ToolResult("1", agentcore.TextResult(strings.Repeat("a", 400))),
		agentcore.ToolResult("2", agentcore.TextResult(strings.Repeat("a", 400))),
		agentcore.UserText("recent"),
	}
	if cut := cutPoint(msgs, 120); cut != 1 {
		t.Fatalf("cut at %d, want the tool call at 1", cut)
	}

	// Sub-agent runs are one task and then tool groups; the cut lands inside.
	msgs = []agentcore.Message{agentcore.UserText("task")}
	for i := 1; i <= 4; i++ {
		msgs = append(msgs, toolGroup(strconv.Itoa(i), strings.Repeat("a", 400))...)
	}
	if cut := cutPoint(msgs, 150); cut != 5 || len(msgs[cut].ToolCalls()) == 0 {
		t.Fatalf("cut at %d, want the third tool call at 5", cut)
	}

	msgs = append([]agentcore.Message{agentcore.UserText("task")}, toolGroup("1", strings.Repeat("a", 400))...)
	if cut := cutPoint(msgs, 10000); cut != 0 {
		t.Fatalf("a suffix covering everything cut at %d", cut)
	}
	if cut := cutPoint(msgs[1:], 50); cut != 0 {
		t.Fatalf("a retreat to the first message cut at %d", cut)
	}
}

func TestExtractFileOps(t *testing.T) {
	msgs := []agentcore.Message{assistant(
		readCall("1", `{"path":"a.go"}`),
		readCall("2", `{"path":"b.go"}`),
		litellm.ToolUseBlock{ID: "3", Name: "edit", Arguments: `{"path":"b.go"}`},
		litellm.ToolUseBlock{ID: "4", Name: "write", Arguments: `{"path":"c.go"}`},
		readCall("5", `{"path":"a.go"}`),
	)}
	read, modified := extractFileOps(msgs)
	if strings.Join(read, ",") != "a.go" || strings.Join(modified, ",") != "b.go,c.go" {
		t.Fatalf("read %v, modified %v", read, modified)
	}
}

func TestCompactReplacesOldHistoryWithSummary(t *testing.T) {
	p := litellmtest.New(litellmtest.Text("<analysis>a</analysis><summary>checkpoint body</summary>"))
	msgs := history()
	c, err := Summarizer{}.Compact(context.Background(), msgs, callFor(t, p))
	if err != nil {
		t.Fatal(err)
	}
	if c == nil || len(c.Messages) != 2 || c.Replaced != 4 || c.Messages[0].Kind != agentcore.KindSummary {
		t.Fatalf("compaction = %+v", c)
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

// The tools the replaced part loaded stay loaded.
func TestCompactKeepsLoadedTools(t *testing.T) {
	p := litellmtest.New(litellmtest.Text("<summary>checkpoint body</summary>"))
	search := litellm.ToolUseBlock{ID: "s1", Name: "tool_search", Arguments: `{"query":"select:deploy"}`}
	msgs := append([]agentcore.Message{
		agentcore.UserText(strings.Repeat("a", 4000)),
		assistant(search),
		agentcore.ToolResult("s1", agentcore.Result{Content: []litellm.Block{litellm.ToolReferenceBlock{ToolName: "deploy"}, litellm.Text("Tool loaded.")}}),
	}, history()[1:]...)
	call := callFor(t, p)
	c, err := Summarizer{}.Compact(context.Background(), msgs, call)
	if err != nil || c == nil {
		t.Fatalf("compaction %+v, %v", c, err)
	}
	offered := map[string]bool{}
	for _, tool := range call(c.Messages).Request.Tools {
		offered[tool.Name] = true
	}
	if !offered["deploy"] {
		t.Fatalf("deploy unloaded by the compaction: %v", offered)
	}
	if got := c.Messages[2]; len(got.Blocks) != 1 || got.Text() != "" {
		t.Fatalf("the replayed result holds %#v", got.Blocks)
	}
}

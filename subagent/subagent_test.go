package subagent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/voocel/agentcore"
	"github.com/voocel/agentcore/schema"
	"github.com/voocel/agentcore/task"
	"github.com/voocel/litellm"
	"github.com/voocel/litellm/litellmtest"
)

func model(t *testing.T, p *litellmtest.Provider) agentcore.Model {
	t.Helper()
	client, err := litellm.New(p)
	if err != nil {
		t.Fatal(err)
	}
	return agentcore.Model{Client: client, Request: litellm.Request{Model: "m"}}
}

// agent returns an agent replying with p, recording the spawns it ran.
func agent(t *testing.T, name string, p *litellmtest.Provider, spawns *[]Spawn) Agent {
	t.Helper()
	m := model(t, p)
	var mu sync.Mutex
	return Agent{Name: name, Description: name + " things", Config: func(s Spawn) (agentcore.Config, error) {
		if spawns != nil {
			mu.Lock()
			*spawns = append(*spawns, s)
			mu.Unlock()
		}
		return agentcore.Config{Model: m}, nil
	}}
}

func call(t *testing.T, ctx context.Context, tool agentcore.Tool, args string) (agentcore.Result, error) {
	t.Helper()
	return tool.Run(ctx, json.RawMessage(args))
}

func text(r agentcore.Result) string {
	return r.Text()
}

func TestSingle(t *testing.T) {
	var spawns []Spawn
	p := litellmtest.New(litellmtest.Text("found it"))
	tool := New(nil, agent(t, "explore", p, &spawns))

	var progress []Progress
	ctx := agentcore.WithProgress(context.Background(), func(v any) { progress = append(progress, v.(Progress)) })
	res, err := call(t, ctx, tool, `{"agent":"explore","task":"find the bug","model":"fast"}`)
	if err != nil || text(res) != "found it" {
		t.Fatalf("result %q, err %v", text(res), err)
	}
	if len(spawns) != 1 || !strings.HasPrefix(spawns[0].ID, "explore#") {
		t.Fatalf("spawns = %+v", spawns)
	}
	want := Spawn{Agent: "explore", ID: spawns[0].ID, Mode: ModeSingle, Model: "fast"}
	if spawns[0] != want {
		t.Fatalf("spawn = %+v, want %+v", spawns[0], want)
	}
	if got := (agentcore.Message{Blocks: p.Requests()[0].Messages[0].Blocks}).Text(); got != "find the bug" {
		t.Fatalf("the agent got %q", got)
	}
	// The run's events reach the call as progress, through to its end.
	if len(progress) == 0 || progress[0].Spawn != want {
		t.Fatalf("progress = %+v", progress)
	}
	if _, ok := progress[len(progress)-1].Event.(agentcore.RunEnd); !ok {
		t.Fatalf("last progress = %#v", progress[len(progress)-1].Event)
	}
}

// WrapRun wraps each run: the run works in the context it returns, and its
// end learns the run's output and error.
func TestWrapRun(t *testing.T) {
	type key struct{}
	type ended struct {
		spawn          Spawn
		prompt, output string
		err            error
	}
	var (
		mu     sync.Mutex
		got    []ended
		inside []any // what the runs' tool calls found in their context
	)
	look := agentcore.NewTool("look", "Look around", schema.Object(), func(ctx context.Context, _ struct{}) (agentcore.Result, error) {
		mu.Lock()
		defer mu.Unlock()
		inside = append(inside, ctx.Value(key{}))
		return agentcore.TextResult("seen"), nil
	})
	m := model(t, litellmtest.New(
		litellmtest.Respond(litellm.ToolUseBlock{ID: "l1", Name: "look", Arguments: "{}"}), litellmtest.Text("found it"),
		litellmtest.Text("and this"),
	))
	tool := New(nil, Agent{
		Name: "explore",
		Config: func(Spawn) (agentcore.Config, error) {
			return agentcore.Config{Model: m, Tools: []agentcore.Tool{look}}, nil
		},
		WrapRun: func(ctx context.Context, s Spawn, prompt string) (context.Context, func(string, error)) {
			return context.WithValue(ctx, key{}, s.ID), func(output string, err error) {
				mu.Lock()
				defer mu.Unlock()
				got = append(got, ended{s, prompt, output, err})
			}
		},
	})
	if _, err := call(t, context.Background(), tool, `{"chain":[{"agent":"explore","task":"find it"},{"agent":"explore","task":"then {previous}"}]}`); err != nil {
		t.Fatal(err)
	}
	if len(got) != 2 || got[0].prompt != "find it" || got[0].output != "found it" || got[1].prompt != "then found it" || got[1].output != "and this" || got[0].err != nil {
		t.Fatalf("ended %+v", got)
	}
	if got[0].spawn.Agent != "explore" || got[0].spawn.Mode != ModeChain || got[0].spawn.ID == got[1].spawn.ID {
		t.Fatalf("spawns %+v %+v", got[0].spawn, got[1].spawn)
	}
	if len(inside) != 1 || inside[0] != got[0].spawn.ID {
		t.Fatalf("the run's tool call found %v in its context, want %s", inside, got[0].spawn.ID)
	}
}

// A run's own Emit sees its events before the call, and stops it by
// failing.
func TestSpawnEmit(t *testing.T) {
	m := model(t, litellmtest.New(litellmtest.Text("lost")))
	full := errors.New("disk full")
	tool := New(nil, Agent{Name: "a", Config: func(Spawn) (agentcore.Config, error) {
		return agentcore.Config{
			Model: m,
			Emit: func(ev agentcore.Event) error {
				if e, ok := ev.(agentcore.MessageEnd); ok && e.Message.Role == litellm.RoleAssistant {
					return full
				}
				return nil
			},
		}, nil
	}})
	var reported []agentcore.Event
	ctx := agentcore.WithProgress(context.Background(), func(v any) { reported = append(reported, v.(Progress).Event) })
	if got := failure(call(t, ctx, tool, `{"agent":"a","task":"go"}`)); !strings.Contains(got, `Agent "a" failed`) || !strings.Contains(got, "disk full") {
		t.Fatalf("failure = %q", got)
	}
	for _, ev := range reported {
		if e, ok := ev.(agentcore.MessageEnd); ok && e.Message.Role == litellm.RoleAssistant {
			t.Fatal("the call saw a message the run's Emit refused")
		}
	}
}

func TestRefusals(t *testing.T) {
	boom := errors.New("no such model")
	tool := New(nil,
		agent(t, "explore", litellmtest.New(), nil),
		Agent{Name: "broken", Config: func(Spawn) (agentcore.Config, error) { return agentcore.Config{}, boom }},
	)
	for args, want := range map[string]string{
		`{"agent":"explore"}`: "exactly one mode",
		`{"agent":"explore","task":"x","tasks":[{"agent":"explore","task":"y"}]}`: "exactly one mode",
		`{"agent":"explore","task":"x","background":true}`:                        "background mode is not available",
		`{"agent":"missing","task":"x"}`:                                          `unknown agent "missing", available: explore, broken`,
		`{"agent":"broken","task":"x"}`:                                           "no such model",
	} {
		if got := failure(call(t, context.Background(), tool, args)); !strings.Contains(got, want) {
			t.Errorf("%s: failure %q, want %q", args, got, want)
		}
	}
	if props := tool.Schema["properties"].(map[string]any); props["background"] != nil {
		t.Fatal("background offered without a task registry")
	}

	deep := context.WithValue(context.Background(), depthKey{}, MaxDepth)
	if got := failure(call(t, deep, tool, `{"agent":"explore","task":"x"}`)); !strings.Contains(got, "nesting depth") {
		t.Fatalf("too deep: %q", got)
	}

	for name, agents := range map[string][]Agent{
		"unnamed":   {{}},
		"duplicate": {{Name: "a"}, {Name: "a"}},
	} {
		func() {
			defer func() {
				if recover() == nil {
					t.Errorf("%s: no panic", name)
				}
			}()
			New(nil, agents...)
		}()
	}
}

// A run nests one deeper than its caller, so its own sub-agents see the
// depth.
func TestDepth(t *testing.T) {
	var depth int
	probe := agentcore.Tool{Name: "probe", Run: func(ctx context.Context, _ json.RawMessage) (agentcore.Result, error) {
		depth = depthOf(ctx)
		return agentcore.TextResult("ok"), nil
	}}
	m := model(t, litellmtest.New(
		litellmtest.Respond(litellm.ToolUseBlock{ID: "c1", Name: "probe", Arguments: `{}`}),
		litellmtest.Text("done"),
	))
	tool := New(nil, Agent{Name: "a", Config: func(Spawn) (agentcore.Config, error) {
		return agentcore.Config{Model: m, Tools: []agentcore.Tool{probe}}, nil
	}})
	if _, err := call(t, context.WithValue(context.Background(), depthKey{}, 2), tool, `{"agent":"a","task":"x"}`); err != nil || depth != 3 {
		t.Fatalf("depth %d, err %v", depth, err)
	}
}

func TestParallel(t *testing.T) {
	denied := litellm.NewError("test", litellm.ErrorTypeAuth, "bad key", nil)
	tool := New(nil,
		agent(t, "a", litellmtest.New(litellmtest.Text("from a")), nil),
		agent(t, "b", litellmtest.New(litellmtest.Fail(denied)), nil),
	)
	res, err := call(t, context.Background(), tool, `{"tasks":[{"agent":"a","task":"x"},{"agent":"b","task":"y"}]}`)
	if err != nil || res.IsError {
		t.Fatalf("err %v, result %+v", err, res)
	}
	got := text(res)
	for _, want := range []string{
		"1/2 succeeded",
		"<result step=\"1\" agent=\"a\" status=\"completed\">\nfrom a\n</result>",
		"<result step=\"2\" agent=\"b\" status=\"failed\">\n",
		"bad key",
	} {
		if !strings.Contains(got, want) {
			t.Fatalf("result lacks %q:\n%s", want, got)
		}
	}
}

func TestChain(t *testing.T) {
	second := litellmtest.New(litellmtest.Text("fixed"))
	tool := New(nil,
		agent(t, "find", litellmtest.New(litellmtest.Text("bug in loop.go")), nil),
		agent(t, "fix", second, nil),
	)
	res, err := call(t, context.Background(), tool, `{"chain":[{"agent":"find","task":"find it"},{"agent":"fix","task":"fix: {previous}"}]}`)
	if err != nil || res.IsError || !strings.HasSuffix(text(res), "<result step=\"2\" agent=\"fix\" status=\"completed\">\nfixed\n</result>\n") {
		t.Fatalf("err %v, result %q", err, text(res))
	}
	if got := (agentcore.Message{Blocks: second.Requests()[0].Messages[0].Blocks}).Text(); got != "fix: bug in loop.go" {
		t.Fatalf("second step got %q", got)
	}

	never := litellmtest.New()
	tool = New(nil,
		agent(t, "find", litellmtest.New(litellmtest.Fail(errors.New("down"))), nil),
		agent(t, "fix", never, nil),
	)
	res, err = call(t, context.Background(), tool, `{"chain":[{"agent":"find","task":"x"},{"agent":"fix","task":"{previous}"}]}`)
	if err != nil || !res.IsError || !strings.Contains(text(res), "stopped at step 1") || len(never.Requests()) != 0 {
		t.Fatalf("err %v, result %+v", err, text(res))
	}
}

// failure is what a call that failed says, as an error or an error result.
func failure(res agentcore.Result, err error) string {
	switch {
	case err != nil:
		return err.Error()
	case res.IsError:
		return res.Text()
	}
	return ""
}

// The runs of a parallel call are numbered in task order, and at most
// maxParallel work at once.
func TestParallelOrderAndLimit(t *testing.T) {
	var spawns []Spawn
	var running, most atomic.Int32
	replies := make([]litellmtest.Reply, maxParallel+2)
	for i := range replies {
		replies[i] = litellmtest.Text("ok")
	}
	m := model(t, litellmtest.New(replies...))
	var mu sync.Mutex
	tool := New(nil, Agent{Name: "a", Config: func(s Spawn) (agentcore.Config, error) {
		mu.Lock()
		spawns = append(spawns, s)
		mu.Unlock()
		return agentcore.Config{Model: m, Emit: func(ev agentcore.Event) error {
			if _, ok := ev.(agentcore.MessageStart); ok {
				n := running.Add(1)
				for cur := most.Load(); n > cur && !most.CompareAndSwap(cur, n); cur = most.Load() {
				}
				time.Sleep(50 * time.Millisecond)
				running.Add(-1)
			}
			return nil
		}}, nil
	}})
	tasks := make([]string, maxParallel+2)
	for i := range tasks {
		tasks[i] = `{"agent":"a","task":"x"}`
	}
	if _, err := call(t, context.Background(), tool, `{"tasks":[`+strings.Join(tasks, ",")+`]}`); err != nil {
		t.Fatal(err)
	}
	var first int
	if _, err := fmt.Sscanf(spawns[0].ID, "a#%d", &first); err != nil {
		t.Fatal(err)
	}
	for i, s := range spawns {
		if want := fmt.Sprintf("a#%d", first+i); s.ID != want {
			t.Fatalf("spawn %d is %s, want %s", i, s.ID, want)
		}
	}
	if n := most.Load(); n > maxParallel || n < 2 {
		t.Fatalf("%d runs at once", n)
	}
}

// A run that failed reports what it said before.
func TestFailureKeepsOutput(t *testing.T) {
	m := model(t, litellmtest.New(litellmtest.Respond(litellm.Text("halfway there"), litellm.ToolUseBlock{ID: "c1", Name: "nope", Arguments: `{}`})))
	tool := New(nil, Agent{Name: "a", Config: func(Spawn) (agentcore.Config, error) {
		return agentcore.Config{Model: m, MaxTurns: 1}, nil
	}})
	res, err := call(t, context.Background(), tool, `{"agent":"a","task":"x"}`)
	if err != nil || !res.IsError || !strings.Contains(text(res), "max turns") || !strings.Contains(text(res), "halfway there") {
		t.Fatalf("err %v, result %q", err, text(res))
	}
}

func TestBackground(t *testing.T) {
	notified := make(chan agentcore.Message, 1)
	rt := task.NewRuntime(t.TempDir(), func(m agentcore.Message) { notified <- m })
	var spawns []Spawn
	p := litellmtest.New(litellmtest.Reply{Blocks: []litellm.Block{litellm.Text("report")}, Usage: litellm.Usage{InputTokens: 10, OutputTokens: 5}})
	tool := New(rt, agent(t, "explore", p, &spawns))

	res, err := call(t, context.Background(), tool, `{"agent":"explore","task":"look around","background":true,"description":"survey"}`)
	if err != nil || !strings.Contains(text(res), "subagent-1") {
		t.Fatalf("err %v, result %q", err, text(res))
	}
	note := <-notified
	rt.Wait()
	e, _ := rt.Get("subagent-1")
	if e.Status != task.Completed || e.Result != "report" || e.Description != "survey" || e.TokensIn != 10 || e.TokensOut != 5 {
		t.Fatalf("entry = %+v", e)
	}
	if note.Kind != task.KindNotification || !strings.Contains(note.Text(), "<result>report</result>") {
		t.Fatalf("notification = %q", note.Text())
	}
	if spawns[0].Mode != ModeBackground || e.Run != spawns[0].ID {
		t.Fatalf("spawn = %+v, entry run %q", spawns[0], e.Run)
	}
	// The output is the run's messages, one JSON line each.
	data, _ := os.ReadFile(e.OutputFile)
	lines := strings.Split(strings.TrimSpace(string(data)), "\n")
	var last agentcore.Message
	if len(lines) != 2 || json.Unmarshal([]byte(lines[1]), &last) != nil || last.Text() != "report" {
		t.Fatalf("output = %q", data)
	}
}

// A tool built again, as when the agents reload, goes on numbering its runs.
func TestRunIDsStayApartAcrossTools(t *testing.T) {
	var spawns []Spawn
	p := litellmtest.New(litellmtest.Text("a"), litellmtest.Text("b"))
	for range 2 {
		if _, err := call(t, context.Background(), New(nil, agent(t, "explore", p, &spawns)), `{"agent":"explore","task":"look"}`); err != nil {
			t.Fatal(err)
		}
	}
	if len(spawns) != 2 || spawns[0].ID == spawns[1].ID {
		t.Fatalf("spawns = %+v", spawns)
	}
}

func TestBackgroundStopped(t *testing.T) {
	notified := make(chan agentcore.Message, 1)
	rt := task.NewRuntime(t.TempDir(), func(m agentcore.Message) { notified <- m })
	started := make(chan struct{})
	m := model(t, litellmtest.New(litellmtest.Reply{Stall: true}))
	tool := New(rt, Agent{Name: "a", Config: func(Spawn) (agentcore.Config, error) {
		return agentcore.Config{Model: m, Emit: func(ev agentcore.Event) error {
			if _, ok := ev.(agentcore.MessageStart); ok {
				close(started)
			}
			return nil
		}}, nil
	}})
	if _, err := call(t, context.Background(), tool, `{"agent":"a","task":"forever","background":true}`); err != nil {
		t.Fatal(err)
	}
	<-started
	rt.StopAll()
	<-notified
	rt.Wait()
	if e, _ := rt.Get("subagent-1"); e.Status != task.Killed {
		t.Fatalf("entry = %+v", e)
	}

	// A task whose output cannot be created does not start.
	file := filepath.Join(t.TempDir(), "file")
	os.WriteFile(file, nil, 0o644)
	tool = New(task.NewRuntime(file, nil), agent(t, "a", litellmtest.New(litellmtest.Text("unseen")), nil))
	if _, err := call(t, context.Background(), tool, `{"agent":"a","task":"x","background":true}`); err == nil {
		t.Fatal("started without an output")
	}
}

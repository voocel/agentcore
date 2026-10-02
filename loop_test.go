package agentcore

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/voocel/litellm"
	"github.com/voocel/litellm/catalog"
	"github.com/voocel/litellm/litellmtest"
)

func TestRunTextResponse(t *testing.T) {
	p := litellmtest.New(litellmtest.Reply{Blocks: []litellm.Block{litellm.Text("hello")}, Usage: litellm.Usage{InputTokens: 10, OutputTokens: 5}})
	m := testModel(t, p)
	m.Pricing = &catalog.Pricing{InputCostPerToken: 0.1, OutputCostPerToken: 1}
	rec := &recorder{}
	history, err := Run(context.Background(), Config{Model: m, Emit: rec.emit, System: []litellm.Block{litellm.Text("be brief")}}, nil, UserText("hi"))
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(roles(history), []litellm.Role{litellm.RoleUser, litellm.RoleAssistant}) {
		t.Fatalf("history roles = %v", roles(history))
	}
	resp := history[1]
	if resp.Text() != "hello" || resp.Stop != StopEnd || resp.Provider != "test" || resp.Model != "m" || resp.Time.IsZero() {
		t.Fatalf("response = %#v", resp)
	}
	if resp.Usage.Input != 10 || resp.Usage.Output != 5 || resp.Usage.Cost.Total != 6 {
		t.Fatalf("usage = %#v, cost %#v", resp.Usage, resp.Usage.Cost)
	}
	if !reflect.DeepEqual(rec.recorded(), history) {
		t.Fatal("MessageEnd events differ from the history")
	}
	req := p.Requests()[0]
	if len(req.Messages) != 2 || req.Messages[0].Role != litellm.RoleSystem || req.Model != "m" {
		t.Fatalf("request = %#v", req)
	}
	if ends := of[RunEnd](rec); len(ends) != 1 || ends[0].Reason != EndDone || ends[0].Turns != 1 {
		t.Fatalf("run end = %#v", ends)
	}
	var text string
	for _, d := range of[MessageDelta](rec) {
		if e, ok := d.Event.(litellm.TextDelta); ok {
			text += e.Text
		}
	}
	if len(of[MessageStart](rec)) != 1 || text != "hello" {
		t.Fatalf("streamed %q", text)
	}
}

func TestRunToolCall(t *testing.T) {
	p := litellmtest.New(
		litellmtest.Respond(litellm.Text("echoing"), call("c1", "echo", `{"text":"pong"}`)),
		litellmtest.Text("done"),
	)
	rec := &recorder{}
	history, err := Run(context.Background(), Config{Model: testModel(t, p), Tools: []Tool{echoTool()}, Emit: rec.emit}, nil, UserText("ping"))
	if err != nil {
		t.Fatal(err)
	}
	want := []litellm.Role{litellm.RoleUser, litellm.RoleAssistant, litellm.RoleTool, litellm.RoleAssistant}
	if !reflect.DeepEqual(roles(history), want) {
		t.Fatalf("history roles = %v", roles(history))
	}
	result, _ := history[2].ToolResult()
	if result.ToolUseID != "c1" || result.IsError || history[2].Text() != "pong" {
		t.Fatalf("tool result = %#v", result)
	}
	if history[1].Stop != StopToolUse {
		t.Fatalf("stop = %q", history[1].Stop)
	}
	second := p.Requests()[1]
	if len(second.Tools) != 1 || second.Tools[0].Name != "echo" || len(second.Messages) != 3 || lastText(second) != "pong" {
		t.Fatalf("second request = %#v", second)
	}
	starts, ends := of[ToolStart](rec), of[ToolEnd](rec)
	if len(starts) != 1 || starts[0].Call.ID != "c1" || starts[0].Call.Tool == nil || len(ends) != 1 || ends[0].Result.IsError {
		t.Fatalf("tool events = %#v %#v", starts, ends)
	}
	if turns := of[TurnEnd](rec); len(turns) != 2 || len(turns[0].Results) != 1 || len(turns[1].Results) != 0 {
		t.Fatalf("turn ends = %#v", turns)
	}
}

func TestRunMaxTurns(t *testing.T) {
	p := litellmtest.New(
		litellmtest.Respond(call("c1", "echo", `{"text":"a"}`)),
		litellmtest.Respond(call("c2", "echo", `{"text":"b"}`)),
	)
	rec := &recorder{}
	history, err := Run(context.Background(), Config{Model: testModel(t, p), Tools: []Tool{echoTool()}, MaxTurns: 2, Emit: rec.emit}, nil, UserText("go"))
	if !errors.Is(err, ErrMaxTurns) || len(history) != 5 {
		t.Fatalf("err = %v, history %d", err, len(history))
	}
	if end := of[RunEnd](rec)[0]; end.Reason != EndMaxTurns || end.ToolCalls != 2 {
		t.Fatalf("run end = %#v", end)
	}
}

// Cancelled while a response streams, the run records what streamed, its
// unfinished tool calls dropped, as an aborted response.
func TestRunAbortDuringResponse(t *testing.T) {
	p := litellmtest.New(litellmtest.Reply{Blocks: []litellm.Block{litellm.Text("half"), call("c1", "echo", `{"text":"x"}`)}, Stall: true})
	ctx, cancel := context.WithCancel(context.Background())
	rec := &recorder{}
	emit := func(ev Event) error {
		if d, ok := ev.(MessageDelta); ok {
			if _, ok := d.Event.(litellm.ToolUseDelta); ok {
				cancel()
			}
		}
		return rec.emit(ev)
	}
	history, err := Run(ctx, Config{Model: testModel(t, p), Tools: []Tool{echoTool()}, Emit: emit, MaxRetries: 3}, nil, UserText("go"))
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v", err)
	}
	last := history[len(history)-1]
	if len(history) != 2 || last.Stop != StopAborted || last.Text() != "half" || len(last.ToolCalls()) != 0 {
		t.Fatalf("history = %#v", history)
	}
	if end := of[RunEnd](rec)[0]; end.Reason != EndAborted || !errors.Is(end.Err, context.Canceled) {
		t.Fatalf("run end = %#v", end)
	}
	if len(of[ToolStart](rec)) != 0 || len(of[Retry](rec)) != 0 {
		t.Fatal("an aborted response ran tools or was retried")
	}
}

// Cancelled while tools run, the run records every call's result and stops.
func TestRunAbortDuringTools(t *testing.T) {
	p := litellmtest.New(litellmtest.Respond(call("c1", "wait", `{}`), call("c2", "wait", `{}`)))
	ctx, cancel := context.WithCancel(context.Background())
	wait := Tool{Name: "wait", Run: func(ctx context.Context, _ json.RawMessage) (Result, error) {
		cancel()
		<-ctx.Done()
		return Result{}, ctx.Err()
	}}
	history, err := Run(ctx, Config{Model: testModel(t, p), Tools: []Tool{wait}}, nil, UserText("go"))
	if !errors.Is(err, context.Canceled) || len(history) != 4 {
		t.Fatalf("err = %v, history %v", err, roles(history))
	}
	for _, m := range history[2:] {
		if r, _ := m.ToolResult(); !r.IsError {
			t.Fatalf("result = %#v", r)
		}
	}
	if !strings.Contains(history[3].Text(), "cancelled before this tool call ran") {
		t.Fatalf("second result = %q", history[3].Text())
	}
}

// Steering waits for the tools of the turn, then reaches the next call.
func TestRunSteering(t *testing.T) {
	p := litellmtest.New(
		litellmtest.Respond(call("c1", "echo", `{"text":"a"}`), call("c2", "echo", `{"text":"b"}`)),
		litellmtest.Text("ok"),
	)
	var mu sync.Mutex
	var queue []Message
	steer := func() []Message {
		mu.Lock()
		defer mu.Unlock()
		q := queue
		queue = nil
		return q
	}
	echo := echoTool()
	run := echo.Run
	echo.Run = func(ctx context.Context, args json.RawMessage) (Result, error) {
		mu.Lock()
		queue = append(queue, UserText("also this"))
		mu.Unlock()
		return run(ctx, args)
	}
	history, err := Run(context.Background(), Config{Model: testModel(t, p), Tools: []Tool{echo}, Steering: steer}, nil, UserText("go"))
	if err != nil {
		t.Fatal(err)
	}
	want := []litellm.Role{litellm.RoleUser, litellm.RoleAssistant, litellm.RoleTool, litellm.RoleTool, litellm.RoleUser, litellm.RoleUser, litellm.RoleAssistant}
	if !reflect.DeepEqual(roles(history), want) {
		t.Fatalf("history roles = %v", roles(history))
	}
	for _, m := range history[2:4] {
		if r, _ := m.ToolResult(); r.IsError {
			t.Fatal("steering skipped a tool call")
		}
	}
	if got := p.Requests()[1]; lastText(got) != "also this" {
		t.Fatalf("second request ends with %q", lastText(got))
	}
}

func TestRunFollowUp(t *testing.T) {
	p := litellmtest.New(litellmtest.Text("first"), litellmtest.Text("second"))
	sent := false
	followUp := func() []Message {
		if sent {
			return nil
		}
		sent = true
		return []Message{UserText("and then?")}
	}
	history, err := Run(context.Background(), Config{Model: testModel(t, p), FollowUp: followUp}, nil, UserText("go"))
	if err != nil || len(history) != 4 || history[3].Text() != "second" {
		t.Fatalf("history = %v, err %v", roles(history), err)
	}
}

// Middleware sees each checked call: it may rewrite its arguments or refuse
// it.
func TestRunMiddleware(t *testing.T) {
	p := litellmtest.New(
		litellmtest.Respond(call("c1", "echo", `{"text":"secret"}`), call("c2", "echo", `{"text":"rm -rf"}`)),
		litellmtest.Text("ok"),
	)
	var order []string
	redact := func(ctx context.Context, c ToolCall, next ToolFunc) (Result, error) {
		order = append(order, "redact:"+c.ID)
		if strings.Contains(string(c.Args), "secret") {
			c.Args = json.RawMessage(`{"text":"***"}`)
		}
		return next(ctx, c)
	}
	approve := func(ctx context.Context, c ToolCall, next ToolFunc) (Result, error) {
		order = append(order, "approve:"+c.ID)
		if strings.Contains(string(c.Args), "rm") {
			return ErrorResult("denied by user"), nil
		}
		return next(ctx, c)
	}
	history, err := Run(context.Background(), Config{Model: testModel(t, p), Tools: []Tool{echoTool()}, Middleware: []ToolMiddleware{redact, approve}, MaxToolErrors: 1}, nil, UserText("go"))
	if err != nil {
		t.Fatal(err)
	}
	if history[2].Text() != "***" || history[3].Text() != "denied by user" {
		t.Fatalf("results = %q, %q", history[2].Text(), history[3].Text())
	}
	if !reflect.DeepEqual(order, []string{"redact:c1", "approve:c1", "redact:c2", "approve:c2"}) {
		t.Fatalf("order = %v", order)
	}
}

// A call is validated and checked before it starts; ToolStart carries the
// preview, and progress arrives as ToolUpdate.
func TestRunCheckPreviewProgress(t *testing.T) {
	p := litellmtest.New(
		litellmtest.Respond(call("c1", "edit", `{"path":"a"}`), call("c2", "edit", `{"path":"unread"}`), call("c3", "edit", `{"path":1}`)),
		litellmtest.Text("ok"),
	)
	ran := map[string]bool{}
	edit := Tool{
		Name:   "edit",
		Schema: map[string]any{"type": "object", "properties": map[string]any{"path": map[string]any{"type": "string"}}, "required": []string{"path"}},
		Check: func(_ context.Context, args json.RawMessage) (string, error) {
			if strings.Contains(string(args), "unread") {
				return "", errors.New("read the file first")
			}
			return "+x", nil
		},
		Run: func(ctx context.Context, args json.RawMessage) (Result, error) {
			ran[string(args)] = true
			ReportProgress(ctx, "50%")
			return TextResult("edited"), nil
		},
	}
	rec := &recorder{}
	history, err := Run(context.Background(), Config{Model: testModel(t, p), Tools: []Tool{edit}, Emit: rec.emit}, nil, UserText("go"))
	if err != nil {
		t.Fatal(err)
	}
	if len(ran) != 1 || !ran[`{"path":"a"}`] {
		t.Fatalf("ran = %v", ran)
	}
	if history[3].Text() != "read the file first" || !strings.Contains(history[4].Text(), "`path` type is expected as `string`") {
		t.Fatalf("results = %q, %q", history[3].Text(), history[4].Text())
	}
	starts := of[ToolStart](rec)
	if starts[0].Call.Preview != "+x" || starts[1].Call.Preview != "" {
		t.Fatalf("previews = %s, %s", starts[0].Call.Preview, starts[1].Call.Preview)
	}
	if ups := of[ToolUpdate](rec); len(ups) != 1 || ups[0].Call.ID != "c1" || ups[0].Progress != "50%" {
		t.Fatalf("updates = %#v", ups)
	}
}

// A response Emit fails to take stays out of the history, and its tools do
// not run.
func TestRunRecordFailureStopsTools(t *testing.T) {
	p := litellmtest.New(litellmtest.Respond(call("c1", "echo", `{"text":"a"}`)))
	ran := false
	echo := echoTool()
	echo.Run = func(context.Context, json.RawMessage) (Result, error) { ran = true; return Result{}, nil }
	full := errors.New("disk full")
	emit := func(ev Event) error {
		if e, ok := ev.(MessageEnd); ok && e.Message.Role == litellm.RoleAssistant {
			return full
		}
		return nil
	}
	history, err := Run(context.Background(), Config{Model: testModel(t, p), Tools: []Tool{echo}, Emit: emit}, nil, UserText("go"))
	if !errors.Is(err, full) || ran || len(history) != 1 {
		t.Fatalf("err = %v, ran %v, history %v", err, ran, roles(history))
	}
}

// Arguments that are not JSON become {} in the history, which stays
// storable, and the model reads why the call did not run.
func TestRunInvalidArguments(t *testing.T) {
	p := litellmtest.New(
		litellmtest.Respond(call("c1", "echo", `{"text":`)),
		litellmtest.Reply{Blocks: []litellm.Block{call("c2", "echo", `{"text":"lo`)}, FinishReason: litellm.FinishReasonLength},
		litellmtest.Text("ok"),
	)
	history, err := Run(context.Background(), Config{Model: testModel(t, p), Tools: []Tool{echoTool()}}, nil, UserText("go"))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := json.Marshal(history); err != nil {
		t.Fatalf("history does not encode: %v", err)
	}
	if got := history[1].ToolCalls()[0].Arguments; got != "{}" {
		t.Fatalf("arguments = %s", got)
	}
	if !strings.Contains(history[2].Text(), `not valid JSON`) || !strings.Contains(history[2].Text(), `{"text":`) {
		t.Fatalf("first result = %q", history[2].Text())
	}
	if !strings.Contains(history[4].Text(), "output token limit") {
		t.Fatalf("second result = %q", history[4].Text())
	}
}

// A text response cut off at the output limit is resumed, at most three
// times.
func TestRunLengthRecovery(t *testing.T) {
	cut := litellmtest.Reply{Blocks: []litellm.Block{litellm.Text("part")}, FinishReason: litellm.FinishReasonLength}
	p := litellmtest.New(cut, cut, cut, cut)
	history, err := Run(context.Background(), Config{Model: testModel(t, p)}, nil, UserText("go"))
	if err != nil || len(p.Requests()) != 4 || len(history) != 8 {
		t.Fatalf("err = %v, requests %d, history %d", err, len(p.Requests()), len(history))
	}
	if lastText(p.Requests()[1]) != lengthRecoveryPrompt || history[2].Kind != KindResume {
		t.Fatalf("recovery prompt = %q, kind %q", lastText(p.Requests()[1]), history[2].Kind)
	}
}

// A terminating tool ends the run unless OnStop goes on; OnStop's error ends
// it.
func TestRunTerminateAndStop(t *testing.T) {
	finish := Tool{Name: "finish", Run: func(context.Context, json.RawMessage) (Result, error) {
		return Result{Content: []litellm.Block{litellm.Text("saved")}, Terminate: true}, nil
	}}
	p := litellmtest.New(litellmtest.Respond(call("c1", "finish", `{}`)))
	history, err := Run(context.Background(), Config{Model: testModel(t, p), Tools: []Tool{finish}}, nil, UserText("go"))
	if err != nil || len(history) != 3 {
		t.Fatalf("terminate: err %v, history %v", err, roles(history))
	}

	var infos []StopInfo
	stop := func(_ context.Context, s StopInfo) ([]Message, error) {
		infos = append(infos, s)
		switch len(infos) {
		case 1:
			return []Message{UserText("not yet: check the tests")}, nil
		case 2:
			return nil, nil
		}
		return nil, errors.New("unreachable")
	}
	p = litellmtest.New(litellmtest.Respond(call("c1", "finish", `{}`)), litellmtest.Text("tests pass"))
	history, err = Run(context.Background(), Config{Model: testModel(t, p), Tools: []Tool{finish}, OnStop: stop}, nil, UserText("go"))
	if err != nil || len(history) != 5 || history[3].Text() != "not yet: check the tests" {
		t.Fatalf("stop: err %v, history %v", err, roles(history))
	}
	if !infos[0].Terminated || infos[1].Terminated || infos[1].Message.Text() != "tests pass" || infos[1].Turns != 2 {
		t.Fatalf("stop infos = %#v", infos)
	}

	looping := errors.New("guard: no progress")
	p = litellmtest.New(litellmtest.Text("done"))
	_, err = Run(context.Background(), Config{Model: testModel(t, p), OnStop: func(context.Context, StopInfo) ([]Message, error) { return nil, looping }}, nil, UserText("go"))
	if !errors.Is(err, looping) {
		t.Fatalf("stop error = %v", err)
	}
}

// Transient failures are retried up to MaxRetries, after the server's
// Retry-After; others are not.
func TestRunRetries(t *testing.T) {
	limited := litellm.NewError("test", litellm.ErrorTypeRateLimit, "slow down", nil)
	limited.RetryAfter = time.Millisecond
	dropped := litellm.NewError("test", litellm.ErrorTypeNetwork, "reset", nil)
	dropped.Temporary, dropped.RetryAfter = true, time.Millisecond
	p := litellmtest.New(
		litellmtest.Fail(limited),
		litellmtest.Reply{Blocks: []litellm.Block{litellm.Text("hal")}, StreamErr: dropped},
		litellmtest.Text("whole"),
	)
	rec := &recorder{}
	history, err := Run(context.Background(), Config{Model: testModel(t, p), Emit: rec.emit, MaxRetries: 2}, nil, UserText("go"))
	if err != nil || len(history) != 2 || history[1].Text() != "whole" {
		t.Fatalf("err = %v, history %#v", err, history)
	}
	retries := of[Retry](rec)
	if len(retries) != 2 || retries[1].Attempt != 2 || retries[0].Delay != time.Millisecond {
		t.Fatalf("retries = %#v", retries)
	}

	p = litellmtest.New(litellmtest.Fail(limited), litellmtest.Fail(limited))
	if _, err := Run(context.Background(), Config{Model: testModel(t, p), MaxRetries: 1}, nil, UserText("go")); litellm.ErrorTypeOf(err) != litellm.ErrorTypeRateLimit || len(p.Requests()) != 2 {
		t.Fatalf("past MaxRetries: err %v, requests %d", err, len(p.Requests()))
	}
	auth := litellm.NewError("test", litellm.ErrorTypeAuth, "bad key", nil)
	p = litellmtest.New(litellmtest.Fail(auth))
	if _, err := Run(context.Background(), Config{Model: testModel(t, p), MaxRetries: 3}, nil, UserText("go")); litellm.ErrorTypeOf(err) != litellm.ErrorTypeAuth || len(p.Requests()) != 1 {
		t.Fatalf("auth: err %v, requests %d", err, len(p.Requests()))
	}
}

// A tool whose calls fail turn after turn is disabled.
func TestRunMaxToolErrors(t *testing.T) {
	p := litellmtest.New(
		litellmtest.Respond(call("c1", "flaky", `{}`)),
		litellmtest.Respond(call("c2", "flaky", `{}`)),
		litellmtest.Respond(call("c3", "flaky", `{}`)),
		litellmtest.Text("giving up"),
	)
	runs := 0
	flaky := Tool{Name: "flaky", Run: func(context.Context, json.RawMessage) (Result, error) {
		runs++
		return Result{}, errors.New("boom")
	}}
	history, err := Run(context.Background(), Config{Model: testModel(t, p), Tools: []Tool{flaky}, MaxToolErrors: 2}, nil, UserText("go"))
	if err != nil || runs != 2 || !strings.Contains(history[6].Text(), "disabled") {
		t.Fatalf("err %v, runs %d, last result %q", err, runs, history[6].Text())
	}
}

// Consecutive parallel calls run together, up to MaxToolConcurrency; any
// other call runs alone.
func TestRunParallelTools(t *testing.T) {
	p := litellmtest.New(
		litellmtest.Respond(call("r1", "read", `{}`), call("r2", "read", `{}`), call("w", "write", `{}`), call("r3", "read", `{}`)),
		litellmtest.Text("ok"),
	)
	var mu sync.Mutex
	running, peak := 0, 0
	var log []string
	track := func(name string) func(context.Context, json.RawMessage) (Result, error) {
		return func(context.Context, json.RawMessage) (Result, error) {
			mu.Lock()
			running++
			peak = max(peak, running)
			log = append(log, name)
			mu.Unlock()
			time.Sleep(20 * time.Millisecond)
			mu.Lock()
			running--
			mu.Unlock()
			return TextResult(name), nil
		}
	}
	always := func(json.RawMessage) bool { return true }
	read := Tool{Name: "read", Parallel: always, Run: track("read")}
	write := Tool{Name: "write", Run: track("write")}
	history, err := Run(context.Background(), Config{Model: testModel(t, p), Tools: []Tool{read, write}, MaxToolConcurrency: 4}, nil, UserText("go"))
	if err != nil {
		t.Fatal(err)
	}
	if peak != 2 || log[2] != "write" || log[3] != "read" {
		t.Fatalf("peak %d, log %v", peak, log)
	}
	var ids []string
	for _, m := range history[2:6] {
		r, _ := m.ToolResult()
		ids = append(ids, r.ToolUseID)
	}
	if !slices.Equal(ids, []string{"r1", "r2", "w", "r3"}) {
		t.Fatalf("results in order %v", ids)
	}
}

// Events hold nothing that changes after they are emitted: a consumer on
// another goroutine may keep and read them while the run goes on. Run with
// -race.
func TestEventsAreImmutable(t *testing.T) {
	p := litellmtest.New(
		litellmtest.Respond(litellm.Text("one"), call("c1", "echo", `{"text":"a"}`)),
		litellmtest.Respond(litellm.Text("two"), call("c2", "echo", `{"text":"b"}`)),
		litellmtest.Text("three"),
	)
	events := make(chan Event, 1024)
	done := make(chan string)
	go func() {
		var sb strings.Builder
		for ev := range events {
			time.Sleep(time.Millisecond)
			switch e := ev.(type) {
			case MessageEnd:
				sb.WriteString(e.Message.Text() + "|")
			case ToolEnd:
				sb.WriteString(string(e.Call.Args) + "|")
			}
		}
		done <- sb.String()
	}()
	cfg := Config{Model: testModel(t, p), Tools: []Tool{echoTool()}, Cache: &litellm.CacheControl{}, Emit: func(ev Event) error { events <- ev; return nil }}
	if _, err := Run(context.Background(), cfg, nil, UserText("go")); err != nil {
		t.Fatal(err)
	}
	close(events)
	if got := <-done; got != `go|one|{"text":"a"}|a|two|{"text":"b"}|b|three|` {
		t.Fatalf("consumer saw %q", got)
	}
}

// Messages taken from the queues are recorded even when the run then ends
// before the model answers them.
func TestRunKeepsTakenMessages(t *testing.T) {
	p := litellmtest.New(litellmtest.Text("one"))
	followUp := []Message{UserText("and then?")}
	take := func() []Message {
		m := followUp
		followUp = nil
		return m
	}
	history, err := Run(context.Background(), Config{Model: testModel(t, p), FollowUp: take, MaxTurns: 1}, nil, UserText("go"))
	if !errors.Is(err, ErrMaxTurns) || history[len(history)-1].Text() != "and then?" {
		t.Fatalf("at MaxTurns: err %v, history %v", err, roles(history))
	}

	ctx, cancel := context.WithCancel(context.Background())
	steered := false
	steer := func() []Message {
		if steered {
			return nil
		}
		steered = true
		cancel()
		return []Message{UserText("stop that")}
	}
	p = litellmtest.New(litellmtest.Text("unreached"))
	history, err = Run(ctx, Config{Model: testModel(t, p), Steering: steer}, nil, UserText("go"))
	if !errors.Is(err, context.Canceled) || len(history) != 2 || history[1].Text() != "stop that" || len(p.Requests()) != 0 {
		t.Fatalf("cancelled: err %v, history %v", err, roles(history))
	}
}

// A failing Emit cancels the tool calls under way; the RunEnd that carries
// its error is still delivered, and is the last event.
func TestRunEmitFailureCancelsTools(t *testing.T) {
	p := litellmtest.New(litellmtest.Respond(call("c1", "wait", `{}`), call("c2", "wait", `{}`)))
	ran, cancelled := 0, false
	wait := Tool{Name: "wait", Run: func(ctx context.Context, _ json.RawMessage) (Result, error) {
		ran++
		ReportProgress(ctx, "working")
		select {
		case <-ctx.Done():
			cancelled = true
			return Result{}, ctx.Err()
		case <-time.After(2 * time.Second):
			return TextResult("not cancelled"), nil
		}
	}}
	full := errors.New("disk full")
	var events []Event
	emit := func(ev Event) error {
		events = append(events, ev)
		if _, ok := ev.(ToolUpdate); ok {
			return full
		}
		return nil
	}
	_, err := Run(context.Background(), Config{Model: testModel(t, p), Tools: []Tool{wait}, Emit: emit}, nil, UserText("go"))
	if !errors.Is(err, full) || ran != 1 || !cancelled {
		t.Fatalf("err %v, calls run %d, cancelled %v", err, ran, cancelled)
	}
	end, ok := events[len(events)-1].(RunEnd)
	if !ok || end.Reason != EndError || !errors.Is(end.Err, full) {
		t.Fatalf("last event = %#v", events[len(events)-1])
	}
	if _, ok := events[len(events)-2].(ToolUpdate); !ok {
		t.Fatalf("an event followed the failure: %T", events[len(events)-2])
	}
}

// A terminated turn delivers steering and follow-ups before OnStop.
func TestRunTerminateTakesQueues(t *testing.T) {
	finish := Tool{Name: "finish", Run: func(context.Context, json.RawMessage) (Result, error) {
		return Result{Content: []litellm.Block{litellm.Text("saved")}, Terminate: true}, nil
	}}
	p := litellmtest.New(litellmtest.Respond(call("c1", "finish", `{}`)), litellmtest.Text("on it"))
	followUp := []Message{UserText("one more thing")}
	take := func() []Message {
		m := followUp
		followUp = nil
		return m
	}
	history, err := Run(context.Background(), Config{Model: testModel(t, p), Tools: []Tool{finish}, FollowUp: take}, nil, UserText("go"))
	if err != nil || history[len(history)-1].Text() != "on it" || history[len(history)-2].Text() != "one more thing" {
		t.Fatalf("err %v, history %v", err, roles(history))
	}
}

// A response that fails for good is recorded with what streamed of it, so
// its MessageStart ends; it is not sent to the model again.
func TestRunFailedResponseIsRecorded(t *testing.T) {
	broken := litellm.NewError("test", litellm.ErrorTypeProvider, "bad gateway", nil)
	p := litellmtest.New(litellmtest.Reply{Blocks: []litellm.Block{litellm.Text("par"), call("c1", "echo", `{"text":"x"}`)}, StreamErr: broken})
	rec := &recorder{}
	history, err := Run(context.Background(), Config{Model: testModel(t, p), Tools: []Tool{echoTool()}, Emit: rec.emit}, nil, UserText("go"))
	if !errors.Is(err, broken) {
		t.Fatalf("err = %v", err)
	}
	last := history[len(history)-1]
	if last.Stop != StopError || last.Text() != "par" || len(last.ToolCalls()) != 0 || len(of[MessageEnd](rec)) != 2 {
		t.Fatalf("history = %#v", history)
	}
	if awaitsResponse(history) != true {
		t.Fatal("the failed response counts as an answer")
	}
}

// Without prompts, a run answers what awaits an answer and refuses a
// history that ends with a response.
func TestRunNothingToContinue(t *testing.T) {
	p := litellmtest.New(litellmtest.Text("answered"))
	history := []Message{UserText("go"), assistant(StopEnd, litellm.Text("done"))}
	if _, err := Run(context.Background(), Config{Model: testModel(t, p)}, history); !errors.Is(err, ErrNothingToContinue) || len(p.Requests()) != 0 {
		t.Fatalf("err = %v", err)
	}
	if _, err := Run(context.Background(), Config{Model: testModel(t, p)}, nil); !errors.Is(err, ErrNothingToContinue) {
		t.Fatalf("empty history: %v", err)
	}
	history, err := Run(context.Background(), Config{Model: testModel(t, p)}, history[:1])
	if err != nil || history[len(history)-1].Text() != "answered" {
		t.Fatalf("continue: err %v, history %v", err, roles(history))
	}
}

// Progress a tool reports after it returned is dropped.
func TestRunLateProgressDropped(t *testing.T) {
	p := litellmtest.New(litellmtest.Respond(call("c1", "leak", `{}`)), litellmtest.Text("ok"))
	late := make(chan func(), 1)
	leak := Tool{Name: "leak", Run: func(ctx context.Context, _ json.RawMessage) (Result, error) {
		late <- func() { ReportProgress(ctx, "after") }
		return TextResult("done"), nil
	}}
	rec := &recorder{}
	if _, err := Run(context.Background(), Config{Model: testModel(t, p), Tools: []Tool{leak}, Emit: rec.emit}, nil, UserText("go")); err != nil {
		t.Fatal(err)
	}
	(<-late)()
	if ups := of[ToolUpdate](rec); len(ups) != 0 {
		t.Fatalf("late progress emitted: %#v", ups)
	}
}

// Messages are timed as they enter the history, so a follow-up queued
// before the run is not older than what came before it.
func TestRunTimesMessagesAsRecorded(t *testing.T) {
	queued := UserText("queued early")
	queued.Time = time.Now().Add(-time.Hour)
	p := litellmtest.New(litellmtest.Text("one"), litellmtest.Text("two"))
	sent := false
	followUp := func() []Message {
		if sent {
			return nil
		}
		sent = true
		return []Message{queued}
	}
	history, err := Run(context.Background(), Config{Model: testModel(t, p), FollowUp: followUp}, nil, UserText("go"))
	if err != nil {
		t.Fatal(err)
	}
	for i, m := range history {
		if m.Time.IsZero() || i > 0 && m.Time.Before(history[i-1].Time) {
			t.Fatalf("message %d timed %v, after %v", i, m.Time, history[max(i-1, 0)].Time)
		}
	}
}

// Arguments are validated against the schema as the model reads it, JSON,
// whatever Go types built it.
func TestRunValidatesAgainstTheWireSchema(t *testing.T) {
	p := litellmtest.New(litellmtest.Respond(call("c1", "say", `{"text":"hi"}`), call("c2", "say", `{"text":5}`)), litellmtest.Text("ok"))
	say := Tool{
		Name:   "say",
		Schema: map[string]any{"type": "object", "properties": map[string]map[string]any{"text": {"type": "string"}}},
		Run:    func(context.Context, json.RawMessage) (Result, error) { return TextResult("said"), nil },
	}
	history, err := Run(context.Background(), Config{Model: testModel(t, p), Tools: []Tool{say}}, nil, UserText("go"))
	if err != nil {
		t.Fatal(err)
	}
	ok := history[2].Blocks[0].(litellm.ToolResultBlock)
	bad := history[3].Blocks[0].(litellm.ToolResultBlock)
	if ok.IsError || !bad.IsError || !strings.Contains(history[3].Text(), "text") {
		t.Fatalf("results %q, %q", history[2].Text(), history[3].Text())
	}
}

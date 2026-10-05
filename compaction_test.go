package agentcore

import (
	"context"
	"errors"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/voocel/litellm"
	"github.com/voocel/litellm/catalog"
	"github.com/voocel/litellm/litellmtest"
	"github.com/voocel/litellm/retry"
)

// summarizer replaces all but the last message with a summary, recording
// the calls it was given. during, if set, runs as it compacts.
type summarizer struct {
	calls []Call
	// errs are what its calls fail with, in turn, before one succeeds.
	errs   []error
	usage  litellm.Usage
	during func()
}

func (s *summarizer) Compact(_ context.Context, history []Message, call func([]Message) Call) (*Compaction, error) {
	s.calls = append(s.calls, call(history))
	if s.during != nil {
		s.during()
	}
	if len(s.errs) > 0 {
		err := s.errs[0]
		s.errs = s.errs[1:]
		return nil, err
	}
	if len(history) < 2 {
		return nil, nil
	}
	kept := history[len(history)-1:]
	return &Compaction{Messages: append([]Message{SummaryMessage("earlier work")}, kept...), Replaced: len(history) - 1, Usage: &Usage{Usage: s.usage}}, nil
}

func bigHistory() []Message {
	return []Message{UserText(strings.Repeat("old context ", 400)), assistant(StopEnd, litellm.Text("noted")), UserText("now this")}
}

func TestRunCompactsAboveCompactAt(t *testing.T) {
	p := litellmtest.New(litellmtest.Text("ok"))
	s := &summarizer{}
	rec := &recorder{}
	cfg := Config{Model: testModel(t, p), Compactor: s, CompactAt: 500, Emit: rec.emit}
	history, err := Run(context.Background(), cfg, bigHistory())
	if err != nil {
		t.Fatal(err)
	}
	if len(history) != 3 || history[0].Kind != KindSummary || history[1].Text() != "now this" {
		t.Fatalf("history = %#v", roles(history))
	}
	if req := p.Requests()[0]; len(req.Messages) != 2 || !strings.Contains(lastText(litellm.Request{Messages: req.Messages[:1]}), "earlier work") {
		t.Fatalf("request after compaction = %#v", req.Messages)
	}
	ends := of[CompactionEnd](rec)
	if len(of[CompactionStart](rec)) != 1 || len(ends) != 1 || ends[0].Compaction.Replaced != 2 {
		t.Fatalf("compaction events = %#v", ends)
	}
	// The compactor got the call the loop would have made.
	if want := BuildCall(cfg, bigHistory()); !reflect.DeepEqual(s.calls[0].Request, want.Request) {
		t.Fatal("the compactor got another call than the loop's")
	}

	p = litellmtest.New(litellmtest.Text("ok"))
	s = &summarizer{}
	if _, err := Run(context.Background(), Config{Model: testModel(t, p), Compactor: s, CompactAt: 100_000}, bigHistory()); err != nil || len(s.calls) != 0 {
		t.Fatalf("compacted below CompactAt: %v, %d", err, len(s.calls))
	}
}

// After a compaction, the usage of the responses it kept measured the
// history it replaced: they do not set off another one.
func TestEstimateAfterCompaction(t *testing.T) {
	old := assistant(StopEnd, litellm.Text("noted"))
	old.Usage = &Usage{Usage: litellm.Usage{InputTokens: 90_000}}
	old.Time = time.Now().Add(-time.Minute)
	history := []Message{SummaryMessage("earlier work"), old, UserText("next")}
	if got := Estimate(Config{}, history); got > 1000 {
		t.Fatalf("estimate counted the kept usage: %d", got)
	}

	fresh := assistant(StopEnd, litellm.Text("ok"))
	fresh.Usage = &Usage{Usage: litellm.Usage{InputTokens: 5000}}
	fresh.Time = time.Now().Add(time.Minute)
	history = append(history, fresh, UserText(strings.Repeat("a", 400)))
	if got := Estimate(Config{}, history); got != 5000+1+100 {
		t.Fatalf("estimate from the last response = %d", got)
	}

	failed := assistant(StopError, litellm.Text("x"))
	failed.Usage = &Usage{Usage: litellm.Usage{InputTokens: 1}}
	failed.Time = time.Now().Add(2 * time.Minute)
	if got := Estimate(Config{}, append(history, failed)); got != 5000+1+100+1 {
		t.Fatalf("estimate counted from a failed response: %d", got)
	}

	if got := estimateText("你好世界"); got != 6 {
		t.Fatalf("CJK estimate = %d", got)
	}
}

// A compaction's usage is priced as a response's is.
func TestRunPricesCompactions(t *testing.T) {
	p := litellmtest.New(litellmtest.Text("ok"))
	m := testModel(t, p)
	m.Pricing = &catalog.Pricing{Rates: catalog.Rates{Input: 1, Output: 2}}
	rec := &recorder{}
	s := &summarizer{usage: litellm.Usage{InputTokens: 10, OutputTokens: 1}}
	if _, err := Run(context.Background(), Config{Model: m, Compactor: s, CompactAt: 500, Emit: rec.emit}, bigHistory()); err != nil {
		t.Fatal(err)
	}
	ends := of[CompactionEnd](rec)
	if len(ends) != 1 || ends[0].Compaction.Usage.Cost == nil || ends[0].Compaction.Usage.Cost.Total != 12 {
		t.Fatalf("compactions = %+v", ends)
	}

	m.Pricing = &catalog.Pricing{Rates: catalog.Rates{Input: -1}}
	if _, err := Run(context.Background(), Config{Model: m}, nil, UserText("go")); err == nil || len(p.Requests()) != 1 {
		t.Fatalf("invalid pricing: %v, requests %d", err, len(p.Requests()))
	}
}

func TestRunOverflowCompactsOnce(t *testing.T) {
	overflow := litellm.NewError("test", litellm.ErrorTypeContextOverflow, "prompt is too long", nil)
	p := litellmtest.New(litellmtest.Fail(overflow), litellmtest.Text("fits now"))
	s := &summarizer{}
	history, err := Run(context.Background(), Config{Model: testModel(t, p), Compactor: s}, bigHistory())
	if err != nil || history[len(history)-1].Text() != "fits now" || len(s.calls) != 1 {
		t.Fatalf("err %v, compactions %d", err, len(s.calls))
	}

	p = litellmtest.New(litellmtest.Fail(overflow), litellmtest.Fail(overflow))
	s = &summarizer{}
	if _, err := Run(context.Background(), Config{Model: testModel(t, p), Compactor: s, Retry: retry.Policy{MaxAttempts: 4}}, bigHistory()); litellm.ErrorTypeOf(err) != litellm.ErrorTypeContextOverflow || len(s.calls) != 1 || len(p.Requests()) != 2 {
		t.Fatalf("second overflow: err %v, compactions %d, requests %d", err, len(s.calls), len(p.Requests()))
	}

	// The compaction's calls are made again as the response's are.
	overloaded := litellm.NewError("test", litellm.ErrorTypeOverloaded, "busy", nil)
	overloaded.RetryAfter = time.Millisecond
	p = litellmtest.New(litellmtest.Fail(overflow), litellmtest.Text("fits now"))
	s = &summarizer{errs: []error{overloaded}}
	rec := &recorder{}
	history, err = Run(context.Background(), Config{Model: testModel(t, p), Compactor: s, Retry: retry.Policy{MaxAttempts: 2}, Emit: rec.emit}, bigHistory())
	if err != nil || history[len(history)-1].Text() != "fits now" || len(s.calls) != 2 {
		t.Fatalf("overloaded compaction: err %v, compactions %d", err, len(s.calls))
	}
	if retries := of[Retry](rec); len(retries) != 2 || retries[0].Delay != 0 || retries[1].Delay != time.Millisecond {
		t.Fatalf("retries = %#v", retries)
	}

	p = litellmtest.New(litellmtest.Fail(overflow))
	if _, err := Run(context.Background(), Config{Model: testModel(t, p), Compactor: &summarizer{}}, []Message{UserText("huge")}); litellm.ErrorTypeOf(err) != litellm.ErrorTypeContextOverflow {
		t.Fatalf("nothing to compact: %v", err)
	}
}

// An overflow reported once the response began streaming ends that
// response with a Retry before the history is compacted.
func TestRunOverflowMidStream(t *testing.T) {
	overflow := litellm.NewError("test", litellm.ErrorTypeContextOverflow, "prompt is too long", nil)
	p := litellmtest.New(litellmtest.Reply{StreamErr: overflow}, litellmtest.Text("fits now"))
	rec := &recorder{}
	history, err := Run(context.Background(), Config{Model: testModel(t, p), Compactor: &summarizer{}, Emit: rec.emit}, bigHistory())
	if err != nil || history[len(history)-1].Text() != "fits now" {
		t.Fatalf("err %v, history %v", err, roles(history))
	}
	open := false
	for _, ev := range rec.events {
		switch e := ev.(type) {
		case MessageStart:
			if open {
				t.Fatal("a response began before the one before it ended")
			}
			open = true
		case MessageEnd:
			if e.Message.Role == litellm.RoleAssistant {
				open = false
			}
		case Retry:
			open = false
		case CompactionStart:
			if open {
				t.Fatal("compacted while a response was open")
			}
		}
	}
}

// What is steered while the run compacts goes into the call that follows,
// not the one after the next turn.
func TestRunSteeringDuringCompaction(t *testing.T) {
	overflow := litellm.NewError("test", litellm.ErrorTypeContextOverflow, "prompt is too long", nil)
	cases := map[string]struct {
		compactAt int
		replies   []litellmtest.Reply
	}{
		"at CompactAt": {500, []litellmtest.Reply{litellmtest.Text("ok")}},
		"on overflow":  {0, []litellmtest.Reply{litellmtest.Fail(overflow), litellmtest.Text("ok")}},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			var queue []Message
			steer := func() []Message {
				q := queue
				queue = nil
				return q
			}
			p := litellmtest.New(tc.replies...)
			s := &summarizer{during: func() { queue = append(queue, UserText("also this")) }}
			history, err := Run(context.Background(), Config{Model: testModel(t, p), Compactor: s, CompactAt: tc.compactAt, Steering: steer}, bigHistory())
			if err != nil {
				t.Fatal(err)
			}
			reqs := p.Requests()
			if len(reqs) != len(tc.replies) || lastText(reqs[len(reqs)-1]) != "also this" {
				t.Fatalf("%d requests, the last ending with %q", len(reqs), lastText(reqs[len(reqs)-1]))
			}
			if n := len(history); history[n-2].Text() != "also this" || history[n-1].Text() != "ok" {
				t.Fatalf("history roles = %v", roles(history))
			}
		})
	}
}

// A compaction Emit fails to take stops the run with the history as it was.
// One that fails at CompactAt is reported and the call made anyway; one
// that fails on an overflow ends the run.
func TestRunCompactionFailures(t *testing.T) {
	full := errors.New("disk full")
	emit := func(ev Event) error {
		if e, ok := ev.(CompactionEnd); ok && e.Compaction != nil {
			return full
		}
		return nil
	}
	p := litellmtest.New(litellmtest.Text("unreached"))
	before := bigHistory()
	history, err := Run(context.Background(), Config{Model: testModel(t, p), Compactor: &summarizer{}, CompactAt: 500, Emit: emit}, before)
	if !errors.Is(err, full) || !reflect.DeepEqual(history, before) {
		t.Fatalf("err %v, history %v", err, roles(history))
	}

	boom := errors.New("summary model down")
	rec := &recorder{}
	history, err = Run(context.Background(), Config{Model: testModel(t, p), Compactor: &summarizer{errs: []error{boom}}, CompactAt: 500, Emit: rec.emit}, bigHistory())
	if err != nil || !errors.Is(of[CompactionEnd](rec)[0].Err, boom) || history[len(history)-1].Text() != "unreached" {
		t.Fatalf("failed compaction at CompactAt: err %v, history %v", err, roles(history))
	}

	overflow := litellm.NewError("test", litellm.ErrorTypeContextOverflow, "prompt is too long", nil)
	p = litellmtest.New(litellmtest.Fail(overflow))
	if _, err := Run(context.Background(), Config{Model: testModel(t, p), Compactor: &summarizer{errs: []error{boom}}}, bigHistory()); !errors.Is(err, boom) {
		t.Fatalf("failed compaction on overflow: %v", err)
	}
}

// A run cancelled while it compacts ends there: Steering is not consulted
// again, nor the model called.
func TestRunCancelledDuringCompaction(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	steered := 0
	steer := func() []Message {
		steered++
		return nil
	}
	p := litellmtest.New(litellmtest.Text("unreached"))
	s := &summarizer{during: cancel, errs: []error{context.Canceled}}
	_, err := Run(ctx, Config{Model: testModel(t, p), Compactor: s, CompactAt: 500, Steering: steer}, bigHistory())
	if !errors.Is(err, context.Canceled) || steered != 1 || len(p.Requests()) != 0 {
		t.Fatalf("err %v, Steering consulted %d times, %d requests", err, steered, len(p.Requests()))
	}
}

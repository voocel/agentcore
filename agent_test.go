package agentcore

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"testing"

	"github.com/voocel/litellm"
	"github.com/voocel/litellm/litellmtest"
)

func TestAgentPrompt(t *testing.T) {
	p := litellmtest.New(litellmtest.Text("one"), litellmtest.Text("two"))
	a := NewAgent(Config{Model: testModel(t, p)}, []Message{UserText("earlier")})
	var stored []Message
	a.Subscribe(func(ev Event) error {
		if e, ok := ev.(MessageEnd); ok {
			stored = append(stored, e.Message)
		}
		return nil
	})
	if err := a.Prompt(context.Background(), UserText("first")); err != nil {
		t.Fatal(err)
	}
	if err := a.Prompt(context.Background(), UserText("second")); err != nil {
		t.Fatal(err)
	}
	history := a.Messages()
	if len(history) != 5 || history[4].Text() != "two" || !reflect.DeepEqual(stored, history[1:]) {
		t.Fatalf("history %v, stored %d", roles(history), len(stored))
	}
	if len(p.Requests()[1].Messages) != 4 {
		t.Fatal("the second run did not continue the first")
	}
}

// While a run is under way, a prompt is refused, steering reaches its next
// call, and a follow-up comes when it would stop; cancelling its context
// ends it.
func TestAgentWhileRunning(t *testing.T) {
	p := litellmtest.New(
		litellmtest.Respond(call("c1", "gate", `{}`)),
		litellmtest.Text("steered"),
		litellmtest.Text("followed up"),
		litellmtest.Reply{Blocks: []litellm.Block{litellm.Text("working")}, Stall: true},
	)
	inGate, release := make(chan struct{}), make(chan struct{})
	gate := Tool{Name: "gate", Run: func(context.Context, json.RawMessage) (Result, error) {
		close(inGate)
		<-release
		return TextResult("opened"), nil
	}}
	a := NewAgent(Config{Model: testModel(t, p), Tools: []Tool{gate}}, nil)
	errc := make(chan error, 1)
	go func() { errc <- a.Prompt(context.Background(), UserText("go")) }()
	<-inGate
	if !errors.Is(a.Prompt(context.Background(), UserText("again")), ErrBusy) || !errors.Is(a.SetMessages(nil), ErrBusy) {
		t.Fatal("a busy agent took a prompt")
	}
	a.Steer(UserText("use the tests"))
	a.FollowUp(UserText("and summarize"))
	close(release)
	if err := <-errc; err != nil {
		t.Fatal(err)
	}
	history := a.Messages()
	var texts []string
	for _, m := range history {
		texts = append(texts, m.Text())
	}
	want := []string{"go", "", "opened", "use the tests", "steered", "and summarize", "followed up"}
	if !reflect.DeepEqual(texts, want) {
		t.Fatalf("history = %q", texts)
	}

	started := make(chan struct{})
	unsubscribe := a.Subscribe(func(ev Event) error {
		if _, ok := ev.(MessageDelta); ok {
			select {
			case <-started:
			default:
				close(started)
			}
		}
		return nil
	})
	defer unsubscribe()
	ctx, cancel := context.WithCancel(context.Background())
	go func() { errc <- a.Prompt(ctx, UserText("long task")) }()
	<-started
	cancel()
	if err := <-errc; !errors.Is(err, context.Canceled) || a.SetMessages(a.Messages()) != nil {
		t.Fatalf("cancelled run: %v", err)
	}
	if last := a.Messages()[len(a.Messages())-1]; last.Stop != StopAborted || last.Text() != "working" {
		t.Fatalf("last = %#v", last)
	}
}

// A subscriber that fails to store a message stops the run and keeps the
// message out of the history and from later subscribers; every subscriber
// still learns how the run ended.
func TestAgentSubscriberFailure(t *testing.T) {
	p := litellmtest.New(litellmtest.Text("lost"))
	a := NewAgent(Config{Model: testModel(t, p)}, nil)
	full := errors.New("disk full")
	a.Subscribe(func(ev Event) error {
		if e, ok := ev.(MessageEnd); ok && e.Message.Role == litellm.RoleAssistant {
			return full
		}
		if _, ok := ev.(RunEnd); ok {
			return errors.New("ignored")
		}
		return nil
	})
	var later []Event
	a.Subscribe(func(ev Event) error {
		later = append(later, ev)
		return nil
	})
	if err := a.Prompt(context.Background(), UserText("go")); !errors.Is(err, full) {
		t.Fatalf("err = %v", err)
	}
	if got := a.Messages(); len(got) != 1 {
		t.Fatalf("history = %v", roles(got))
	}
	end, ok := later[len(later)-1].(RunEnd)
	if !ok || !errors.Is(end.Err, full) || end.Reason != EndError {
		t.Fatalf("last event of the later subscriber = %#v", later[len(later)-1])
	}
	for _, ev := range later {
		if e, ok := ev.(MessageEnd); ok && e.Message.Role == litellm.RoleAssistant {
			t.Fatal("a later subscriber saw the refused message")
		}
	}
}

func TestAgentCompact(t *testing.T) {
	a := NewAgent(Config{Compactor: &summarizer{}}, bigHistory())
	var ends []CompactionEnd
	a.Subscribe(func(ev Event) error {
		if e, ok := ev.(CompactionEnd); ok {
			ends = append(ends, e)
		}
		return nil
	})
	if err := a.Compact(context.Background()); err != nil {
		t.Fatal(err)
	}
	if got := a.Messages(); len(got) != 2 || got[0].Kind != KindSummary || len(ends) != 1 || a.SetMessages(got) != nil {
		t.Fatalf("history = %v, ends %d", roles(got), len(ends))
	}
}

// Messages queued after a run took its last ones wait for the next run, or
// are taken back.
func TestAgentClearQueues(t *testing.T) {
	p := litellmtest.New(litellmtest.Text("one"))
	a := NewAgent(Config{Model: testModel(t, p)}, nil)
	if err := a.Prompt(context.Background(), UserText("go")); err != nil {
		t.Fatal(err)
	}
	a.Steer(UserText("late"))
	a.FollowUp(UserText("later"))
	steering, followUp := a.ClearQueues()
	if len(steering) != 1 || steering[0].Text() != "late" || len(followUp) != 1 || followUp[0].Text() != "later" {
		t.Fatalf("queues = %v, %v", steering, followUp)
	}
	if s, f := a.ClearQueues(); s != nil || f != nil {
		t.Fatal("the queues were not cleared")
	}
}

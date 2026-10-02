package agentcore

import (
	"context"
	"sync"
	"testing"

	"github.com/voocel/litellm"
	"github.com/voocel/litellm/litellmtest"
)

// testModel returns a Model replying with p.
func testModel(t *testing.T, p *litellmtest.Provider) Model {
	t.Helper()
	client, err := litellm.New(p)
	if err != nil {
		t.Fatal(err)
	}
	return Model{Client: client, Request: litellm.Request{Model: "m"}}
}

// recorder keeps the events of a run.
type recorder struct {
	mu     sync.Mutex
	events []Event
}

func (r *recorder) emit(ev Event) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.events = append(r.events, ev)
	return nil
}

// recorded returns the messages recorded by MessageEnd, in order.
func (r *recorder) recorded() []Message {
	r.mu.Lock()
	defer r.mu.Unlock()
	var out []Message
	for _, ev := range r.events {
		if e, ok := ev.(MessageEnd); ok {
			out = append(out, e.Message)
		}
	}
	return out
}

// of returns the events of type E.
func of[E Event](r *recorder) []E {
	r.mu.Lock()
	defer r.mu.Unlock()
	var out []E
	for _, ev := range r.events {
		if e, ok := ev.(E); ok {
			out = append(out, e)
		}
	}
	return out
}

func call(id, name, args string) litellm.ToolUseBlock {
	return litellm.ToolUseBlock{ID: id, Name: name, Arguments: args}
}

// echoTool returns its "text" argument.
func echoTool() Tool {
	return NewTool("echo", "Echo text", map[string]any{
		"type":       "object",
		"properties": map[string]any{"text": map[string]any{"type": "string"}},
		"required":   []string{"text"},
	}, func(_ context.Context, args struct{ Text string }) (Result, error) {
		return TextResult(args.Text), nil
	})
}

// roles returns the roles of msgs.
func roles(msgs []Message) []litellm.Role {
	out := make([]litellm.Role, len(msgs))
	for i, m := range msgs {
		out[i] = m.Role
	}
	return out
}

// lastText returns the text of the last message of req.
func lastText(req litellm.Request) string {
	m := req.Messages[len(req.Messages)-1]
	return Message{Role: m.Role, Blocks: m.Blocks}.Text()
}

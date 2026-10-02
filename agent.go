package agentcore

import (
	"context"
	"slices"
	"sync"
)

// Agent is a conversation an application drives: it keeps the history and
// runs it, one run at a time, taking the messages that arrive meanwhile. It
// replaces the Emit, Steering and FollowUp of its Config: events go to its
// subscribers. A run ends when the context it was started with is
// cancelled. An Agent is safe for concurrent use.
type Agent struct {
	mu       sync.Mutex
	cfg      Config
	history  []Message
	steering []Message
	followUp []Message
	subs     []*subscriber
	running  bool
}

type subscriber struct{ fn func(Event) error }

// NewAgent returns an Agent that runs history with cfg. Its runs replace
// cfg's Emit, Steering and FollowUp with the Agent's: subscribers, Steer and
// FollowUp.
func NewAgent(cfg Config, history []Message) *Agent {
	return &Agent{cfg: cfg, history: slices.Clone(history)}
}

// Config returns the Agent's Config.
func (a *Agent) Config() Config {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.cfg
}

// SetConfig replaces the Agent's Config, from the next run on.
func (a *Agent) SetConfig(cfg Config) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.cfg = cfg
}

// Messages returns the history, up to the last message recorded.
func (a *Agent) Messages() []Message {
	a.mu.Lock()
	defer a.mu.Unlock()
	return slices.Clone(a.history)
}

// SetMessages replaces the history, such as to start over; it fails with
// ErrBusy while a run is under way.
func (a *Agent) SetMessages(history []Message) error {
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.running {
		return ErrBusy
	}
	a.history = slices.Clone(history)
	return nil
}

// Subscribe has fn receive the events of the Agent's runs, after those of
// earlier subscribers, until unsubscribe is called. An error fn returns
// stops the run, as one Emit returns does, and later subscribers do not
// receive the event: storing messages on MessageEnd keeps them out of the
// history when that fails. Every subscriber receives the RunEnd.
func (a *Agent) Subscribe(fn func(Event) error) (unsubscribe func()) {
	s := &subscriber{fn}
	a.mu.Lock()
	a.subs = append(a.subs, s)
	a.mu.Unlock()
	return func() {
		a.mu.Lock()
		a.subs = slices.DeleteFunc(a.subs, func(x *subscriber) bool { return x == s })
		a.mu.Unlock()
	}
}

// Prompt runs the history with prompts added, and returns when the run
// ends, with its error. It fails with ErrBusy while a run is under way; use
// Steer or FollowUp then.
func (a *Agent) Prompt(ctx context.Context, prompts ...Message) error {
	return a.run(ctx, prompts)
}

// Continue runs the history as it is, such as after a run failed; see Run
// for when there is nothing to continue.
func (a *Agent) Continue(ctx context.Context) error {
	return a.run(ctx, nil)
}

// Steer delivers msgs before the next model call of the run under way, or
// of the next run.
func (a *Agent) Steer(msgs ...Message) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.steering = append(a.steering, msgs...)
}

// FollowUp delivers msgs once the run under way, or else the next, would
// stop.
func (a *Agent) FollowUp(msgs ...Message) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.followUp = append(a.followUp, msgs...)
}

// ClearQueues takes back the messages queued by Steer and FollowUp that no
// run delivered yet, such as to drop them when the user aborts, or to start
// a run for those that arrived as one ended.
func (a *Agent) ClearQueues() (steering, followUp []Message) {
	a.mu.Lock()
	defer a.mu.Unlock()
	steering, followUp = a.steering, a.followUp
	a.steering, a.followUp = nil, nil
	return steering, followUp
}

// Compact has the Config's Compactor rewrite the history now, as a run
// does when the history outgrows its CompactAt. Subscribers receive its
// CompactionStart and CompactionEnd. It fails with ErrBusy while a run is
// under way.
func (a *Agent) Compact(ctx context.Context) error {
	cfg, history, err := a.start()
	if err != nil {
		return err
	}
	defer a.finish(nil)
	ctx, cancel := context.WithCancelCause(ctx)
	defer cancel(nil)
	r := &run{ctx: ctx, cancel: cancel, cfg: cfg, history: history}
	_, err = r.compact()
	return err
}

func (a *Agent) run(ctx context.Context, prompts []Message) error {
	cfg, history, err := a.start()
	if err != nil {
		return err
	}
	cfg.Steering = func() []Message { return a.take(&a.steering) }
	cfg.FollowUp = func() []Message { return a.take(&a.followUp) }
	history, err = Run(ctx, cfg, history, prompts...)
	a.finish(history)
	return err
}

// start claims the Agent for a run, with the Config and history the run
// starts from. Events the run emits update the history as subscribers take
// them.
func (a *Agent) start() (Config, []Message, error) {
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.running {
		return Config{}, nil, ErrBusy
	}
	a.running = true
	cfg := a.cfg
	cfg.Emit = a.dispatch
	return cfg, slices.Clone(a.history), nil
}

// finish releases the Agent, its history the one the run ended with, if
// given.
func (a *Agent) finish(history []Message) {
	a.mu.Lock()
	defer a.mu.Unlock()
	if history != nil {
		a.history = history
	}
	a.running = false
}

// dispatch hands ev to the subscribers and, once they all took it, applies
// it to the history.
func (a *Agent) dispatch(ev Event) error {
	a.mu.Lock()
	subs := slices.Clone(a.subs)
	a.mu.Unlock()
	_, end := ev.(RunEnd)
	for _, s := range subs {
		if err := s.fn(ev); err != nil && !end {
			return err
		}
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	switch e := ev.(type) {
	case MessageEnd:
		a.history = append(a.history, e.Message)
	case CompactionEnd:
		if e.Compaction != nil {
			a.history = slices.Clone(e.Compaction.Messages)
		}
	}
	return nil
}

func (a *Agent) take(queue *[]Message) []Message {
	a.mu.Lock()
	defer a.mu.Unlock()
	msgs := *queue
	*queue = nil
	return msgs
}

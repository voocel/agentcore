package agentcore

import "sync"

// Collect consumes all events from the channel and returns the final messages.
// Blocks until the channel is closed. Returns any error from EventError events.
func Collect(events <-chan Event) ([]AgentMessage, error) {
	var (
		result []AgentMessage
		err    error
	)
	for ev := range events {
		if ev.Type == EventAgentEnd {
			result = ev.NewMessages
		}
		if ev.Type == EventError && ev.Err != nil {
			err = ev.Err
		}
	}
	return result, err
}

// EventStream wraps an event channel to provide both real-time iteration
// and deferred result collection.
//
// Usage:
//
//	stream := agentcore.NewEventStream(AgentLoop(...))
//	for ev := range stream.Events() {
//	    // handle real-time events
//	}
//	msgs, err := stream.Result()
type EventStream struct {
	events chan Event
	done   chan struct{}
	mu     sync.Mutex
	result []AgentMessage
	err    error
}

// NewEventStream creates an EventStream that reads from the source channel.
// Events are forwarded to an internal channel for iteration.
// The final result is captured from EventAgentEnd.
//
// Events() carries the same consumption contract as the source: iterate until
// it closes. Abandoning it mid-stream leaks the forwarding goroutine — the
// wrapper has no context to detect an absent reader.
func NewEventStream(source <-chan Event) *EventStream {
	s := &EventStream{
		events: make(chan Event, 128),
		done:   make(chan struct{}),
	}

	go func() {
		defer close(s.events)
		defer close(s.done)

		for ev := range source {
			if ev.Type == EventAgentEnd {
				s.mu.Lock()
				s.result = ev.NewMessages
				s.mu.Unlock()
			}
			if ev.Type == EventError && ev.Err != nil {
				s.mu.Lock()
				s.err = ev.Err
				s.mu.Unlock()
			}
			s.events <- ev
		}
	}()

	return s
}

// Events returns the event channel for real-time iteration.
// The channel is closed when the source is exhausted.
func (s *EventStream) Events() <-chan Event {
	return s.events
}

// Result blocks until the stream is done and returns the final messages.
// Returns the error from the last EventError, if any.
func (s *EventStream) Result() ([]AgentMessage, error) {
	<-s.done
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.result, s.err
}

// Done returns a channel that is closed when the stream finishes.
func (s *EventStream) Done() <-chan struct{} {
	return s.done
}

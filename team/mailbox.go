package team

import (
	"context"
	"errors"
	"sync"
	"time"
)

// ErrClosed is returned by Send and Wait once the mailbox is closed.
var ErrClosed = errors.New("team: mailbox closed")

// Message is one envelope delivered through a mailbox. Text may be plain
// content (peer DM) or a JSON-encoded structured message (see protocol.go).
// Mailbox itself stays protocol-agnostic — callers decide how to interpret
// Text.
type Message struct {
	From      string
	Text      string
	Color     string
	Summary   string
	Timestamp time.Time
}

// Mailbox is a single-consumer, multi-producer in-memory inbox. It pairs a
// FIFO queue with a buffered wake channel so consumers block efficiently in
// Wait instead of polling. Messages are removed on Drain — read-flag semantics
// would only matter for UI replay or crash recovery, neither of which apply
// in our single-session in-memory design.
type Mailbox struct {
	mu     sync.Mutex
	queue  []Message
	wake   chan struct{}
	closed bool
}

// NewMailbox creates an empty mailbox ready for use.
func NewMailbox() *Mailbox {
	return &Mailbox{wake: make(chan struct{}, 1)}
}

// Send appends msg and signals any waiter. Non-blocking on the wake channel —
// if a previous signal is still pending, this Send coalesces into it (the
// consumer will see both messages on its next Drain).
func (m *Mailbox) Send(msg Message) error {
	m.mu.Lock()
	if m.closed {
		m.mu.Unlock()
		return ErrClosed
	}
	if msg.Timestamp.IsZero() {
		msg.Timestamp = time.Now()
	}
	m.queue = append(m.queue, msg)
	m.mu.Unlock()

	select {
	case m.wake <- struct{}{}:
	default:
	}
	return nil
}

// Drain returns all pending messages atomically and empties the queue. Returns
// nil if there's nothing to deliver (or the mailbox is closed). Consumers
// typically call this right after Wait returns, then sort by priority before
// handing the first message to the agent.
func (m *Mailbox) Drain() []Message {
	m.mu.Lock()
	defer m.mu.Unlock()
	if len(m.queue) == 0 {
		return nil
	}
	out := m.queue
	m.queue = nil
	return out
}

// Len reports the current queue depth. Mainly for tests and diagnostics.
func (m *Mailbox) Len() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return len(m.queue)
}

// Wait blocks until at least one message is available, the mailbox is closed,
// or ctx is cancelled. The for-loop guards against spurious wakes (the wake
// channel could fire just as Drain emptied the queue from another goroutine).
//
// Returns:
//   - nil when messages are ready (caller should Drain)
//   - ErrClosed if the mailbox was closed
//   - ctx.Err() if the context cancelled first
func (m *Mailbox) Wait(ctx context.Context) error {
	for {
		m.mu.Lock()
		if m.closed {
			m.mu.Unlock()
			return ErrClosed
		}
		if len(m.queue) > 0 {
			m.mu.Unlock()
			return nil
		}
		m.mu.Unlock()

		select {
		case <-m.wake:
			// loop back to recheck queue under lock
		case <-ctx.Done():
			return ctx.Err()
		}
	}
}

// Close marks the mailbox closed and wakes any pending Wait. Subsequent Send
// calls return ErrClosed. Drain still returns whatever was already queued —
// closing doesn't lose delivered-but-unread messages.
func (m *Mailbox) Close() {
	m.mu.Lock()
	if m.closed {
		m.mu.Unlock()
		return
	}
	m.closed = true
	m.mu.Unlock()

	// Non-blocking signal so Wait returns ErrClosed on its next check.
	select {
	case m.wake <- struct{}{}:
	default:
	}
}

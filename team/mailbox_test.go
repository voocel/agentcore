package team

import (
	"context"
	"errors"
	"strconv"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestMailbox_SendAndDrain(t *testing.T) {
	mb := NewMailbox()

	if err := mb.Send(Message{From: "alice", Text: "hi"}); err != nil {
		t.Fatalf("Send: %v", err)
	}
	if err := mb.Send(Message{From: "bob", Text: "yo"}); err != nil {
		t.Fatalf("Send: %v", err)
	}

	if got := mb.Len(); got != 2 {
		t.Fatalf("Len = %d, want 2", got)
	}

	msgs := mb.Drain()
	if len(msgs) != 2 {
		t.Fatalf("Drain returned %d, want 2", len(msgs))
	}
	if msgs[0].From != "alice" || msgs[1].From != "bob" {
		t.Errorf("FIFO order broken: %+v", msgs)
	}
	if msgs[0].Timestamp.IsZero() {
		t.Error("Send did not stamp Timestamp")
	}

	if mb.Drain() != nil {
		t.Error("Second Drain should be nil")
	}
}

func TestMailbox_WaitReturnsImmediatelyWhenQueueNonEmpty(t *testing.T) {
	mb := NewMailbox()
	if err := mb.Send(Message{From: "alice", Text: "hi"}); err != nil {
		t.Fatalf("Send: %v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()

	if err := mb.Wait(ctx); err != nil {
		t.Errorf("Wait returned %v, want nil", err)
	}
}

func TestMailbox_WaitBlocksUntilSend(t *testing.T) {
	mb := NewMailbox()

	done := make(chan error, 1)
	go func() {
		done <- mb.Wait(context.Background())
	}()

	// Give the goroutine time to enter Wait
	time.Sleep(20 * time.Millisecond)

	if err := mb.Send(Message{From: "alice", Text: "hi"}); err != nil {
		t.Fatalf("Send: %v", err)
	}

	select {
	case err := <-done:
		if err != nil {
			t.Errorf("Wait returned %v, want nil", err)
		}
	case <-time.After(time.Second):
		t.Fatal("Wait did not wake on Send")
	}
}

func TestMailbox_WaitReturnsOnContextCancel(t *testing.T) {
	mb := NewMailbox()
	ctx, cancel := context.WithCancel(context.Background())

	done := make(chan error, 1)
	go func() {
		done <- mb.Wait(ctx)
	}()

	time.Sleep(20 * time.Millisecond)
	cancel()

	select {
	case err := <-done:
		if !errors.Is(err, context.Canceled) {
			t.Errorf("Wait returned %v, want context.Canceled", err)
		}
	case <-time.After(time.Second):
		t.Fatal("Wait did not return on context cancel")
	}
}

func TestMailbox_CloseWakesWaiters(t *testing.T) {
	mb := NewMailbox()

	done := make(chan error, 1)
	go func() {
		done <- mb.Wait(context.Background())
	}()

	time.Sleep(20 * time.Millisecond)
	mb.Close()

	select {
	case err := <-done:
		if !errors.Is(err, ErrClosed) {
			t.Errorf("Wait returned %v, want ErrClosed", err)
		}
	case <-time.After(time.Second):
		t.Fatal("Close did not wake Wait")
	}
}

func TestMailbox_SendAfterCloseFails(t *testing.T) {
	mb := NewMailbox()
	mb.Close()

	err := mb.Send(Message{From: "alice", Text: "hi"})
	if !errors.Is(err, ErrClosed) {
		t.Errorf("Send returned %v, want ErrClosed", err)
	}
}

func TestMailbox_DrainAfterCloseReturnsLeftover(t *testing.T) {
	mb := NewMailbox()
	if err := mb.Send(Message{From: "alice", Text: "hi"}); err != nil {
		t.Fatalf("Send: %v", err)
	}
	mb.Close()

	msgs := mb.Drain()
	if len(msgs) != 1 || msgs[0].From != "alice" {
		t.Errorf("Drain after Close lost message: %+v", msgs)
	}
}

func TestMailbox_DoubleCloseIsSafe(t *testing.T) {
	mb := NewMailbox()
	mb.Close()
	mb.Close() // must not panic or deadlock
}

func TestMailbox_ConcurrentSenders(t *testing.T) {
	mb := NewMailbox()
	const senders = 32
	const perSender = 50

	var wg sync.WaitGroup
	wg.Add(senders)
	for i := range senders {
		go func(id int) {
			defer wg.Done()
			for j := range perSender {
				if err := mb.Send(Message{
					From: "s" + strconv.Itoa(id),
					Text: strconv.Itoa(j),
				}); err != nil {
					t.Errorf("Send: %v", err)
					return
				}
			}
		}(i)
	}
	wg.Wait()

	msgs := mb.Drain()
	if len(msgs) != senders*perSender {
		t.Fatalf("Drain returned %d, want %d", len(msgs), senders*perSender)
	}
}

// TestMailbox_CoalescedWakes confirms the consumer sees all messages even
// when multiple Sends happen before it gets a chance to Drain — the wake
// channel having cap=1 must not lose messages, only signals.
func TestMailbox_CoalescedWakes(t *testing.T) {
	mb := NewMailbox()

	// Three rapid sends before any consumer reads.
	for i := range 3 {
		if err := mb.Send(Message{From: "alice", Text: strconv.Itoa(i)}); err != nil {
			t.Fatalf("Send: %v", err)
		}
	}

	if err := mb.Wait(context.Background()); err != nil {
		t.Fatalf("Wait: %v", err)
	}
	msgs := mb.Drain()
	if len(msgs) != 3 {
		t.Errorf("Drain returned %d, want all 3", len(msgs))
	}
}

// TestMailbox_WaitDrainLoop simulates the runner pattern: Wait → Drain → process,
// repeated. Producers should not lose messages even if they Send while the
// consumer is between Wait and Drain.
func TestMailbox_WaitDrainLoop(t *testing.T) {
	mb := NewMailbox()
	const total = 200

	var received atomic.Int64
	consumerDone := make(chan struct{})
	go func() {
		defer close(consumerDone)
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		defer cancel()
		for received.Load() < total {
			if err := mb.Wait(ctx); err != nil {
				return
			}
			msgs := mb.Drain()
			received.Add(int64(len(msgs)))
		}
	}()

	var wg sync.WaitGroup
	wg.Add(4)
	for i := range 4 {
		go func(base int) {
			defer wg.Done()
			for j := range total / 4 {
				_ = mb.Send(Message{From: "p", Text: strconv.Itoa(base + j)})
			}
		}(i * 50)
	}
	wg.Wait()

	<-consumerDone
	if got := received.Load(); got != total {
		t.Errorf("received %d, want %d", got, total)
	}
}

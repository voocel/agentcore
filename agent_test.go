package agentcore

import (
	"context"
	"sync"
	"testing"
	"time"
)

func TestAgentContinue_SteeringQueueOneAtATime(t *testing.T) {
	agent := NewAgent(
		WithStreamFn(mockStreamFn(
			assistantMsg("initial", StopReasonStop),
			assistantMsg("after steer 1", StopReasonStop),
			assistantMsg("after steer 2", StopReasonStop),
		)),
		WithSteeringMode(QueueModeOneAtATime),
	)

	if err := agent.Prompt("start"); err != nil {
		t.Fatalf("prompt failed: %v", err)
	}
	agent.WaitForIdle()

	agent.Steer(UserMsg("steer-1"))
	agent.Steer(UserMsg("steer-2"))

	if err := agent.Continue(); err != nil {
		t.Fatalf("first continue failed: %v", err)
	}
	agent.WaitForIdle()

	msgs := agent.Messages()
	want := []string{"start", "initial", "steer-1", "after steer 1", "steer-2", "after steer 2"}
	if len(msgs) != len(want) {
		t.Fatalf("expected %d messages, got %d", len(want), len(msgs))
	}
	for i, msg := range msgs {
		if got := msg.TextContent(); got != want[i] {
			t.Fatalf("message[%d]: expected %q, got %q", i, want[i], got)
		}
	}
}

func TestAgentInject_WhenRunning_ReturnsSteeredCurrentRun(t *testing.T) {
	release := make(chan struct{})
	agent := NewAgent(
		WithStreamFn(func(ctx context.Context, req *LLMRequest) (*LLMResponse, error) {
			<-release
			return &LLMResponse{Message: assistantMsg("done", StopReasonStop)}, nil
		}),
	)

	if err := agent.Prompt("start"); err != nil {
		t.Fatalf("prompt failed: %v", err)
	}

	result, err := agent.Inject(UserMsg("runtime steer"))
	if err != nil {
		t.Fatalf("inject failed: %v", err)
	}
	if result.Disposition != InjectSteeredCurrentRun {
		t.Fatalf("disposition = %q, want %q", result.Disposition, InjectSteeredCurrentRun)
	}

	close(release)
	agent.WaitForIdle()
	for _, msg := range agent.Messages() {
		if msg.TextContent() == "runtime steer" {
			return
		}
	}
	if !agent.HasQueuedMessages() {
		t.Fatal("expected injected message to be consumed or remain queued")
	}
}

func TestAgentInject_WhenIdleAndAssistantTail_ReturnsResumedIdleRun(t *testing.T) {
	agent := NewAgent(
		WithStreamFn(mockStreamFn(
			assistantMsg("initial", StopReasonStop),
			assistantMsg("after inject", StopReasonStop),
		)),
	)

	if err := agent.Prompt("start"); err != nil {
		t.Fatalf("prompt failed: %v", err)
	}
	agent.WaitForIdle()

	result, err := agent.Inject(UserMsg("runtime reminder"))
	if err != nil {
		t.Fatalf("inject failed: %v", err)
	}
	if result.Disposition != InjectResumedIdleRun {
		t.Fatalf("disposition = %q, want %q", result.Disposition, InjectResumedIdleRun)
	}

	agent.WaitForIdle()

	msgs := agent.Messages()
	want := []string{"start", "initial", "runtime reminder", "after inject"}
	if len(msgs) != len(want) {
		t.Fatalf("expected %d messages, got %d", len(want), len(msgs))
	}
	for i, msg := range msgs {
		if got := msg.TextContent(); got != want[i] {
			t.Fatalf("message[%d]: expected %q, got %q", i, want[i], got)
		}
	}
}

func TestAgentInject_WhenIdleWithoutAssistantTail_ReturnsQueued(t *testing.T) {
	agent := NewAgent()
	if err := agent.SetMessages([]AgentMessage{UserMsg("only user")}); err != nil {
		t.Fatalf("set messages failed: %v", err)
	}

	result, err := agent.Inject(UserMsg("queued"))
	if err != nil {
		t.Fatalf("inject failed: %v", err)
	}
	if result.Disposition != InjectQueued {
		t.Fatalf("disposition = %q, want %q", result.Disposition, InjectQueued)
	}
	if !agent.HasQueuedMessages() {
		t.Fatal("expected queued inject message")
	}
}

func TestAgentInject_WhenNilMessage_ReturnsError(t *testing.T) {
	agent := NewAgent()
	if _, err := agent.Inject(nil); err == nil {
		t.Fatal("expected nil inject message to fail")
	}
}

func TestAgentInject_IsAtomicUnderConcurrentCalls(t *testing.T) {
	agent := NewAgent(
		WithStreamFn(mockStreamFn(
			assistantMsg("initial", StopReasonStop),
			assistantMsg("after inject 1", StopReasonStop),
			assistantMsg("after inject 2", StopReasonStop),
		)),
	)

	if err := agent.Prompt("start"); err != nil {
		t.Fatalf("prompt failed: %v", err)
	}
	agent.WaitForIdle()

	type injectOutcome struct {
		result InjectResult
		err    error
	}
	outcomes := make([]injectOutcome, 2)
	msgsToInject := []AgentMessage{UserMsg("inject-a"), UserMsg("inject-b")}

	var wg sync.WaitGroup
	for i := range msgsToInject {
		i := i
		wg.Add(1)
		go func() {
			defer wg.Done()
			outcomes[i].result, outcomes[i].err = agent.Inject(msgsToInject[i])
		}()
	}
	wg.Wait()

	for i, outcome := range outcomes {
		if outcome.err != nil {
			t.Fatalf("inject[%d] failed: %v", i, outcome.err)
		}
	}

	deadline := time.Now().Add(2 * time.Second)
	for {
		agent.WaitForIdle()
		if !agent.HasQueuedMessages() {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("timed out waiting for queued injected messages to drain")
		}
		if err := agent.Continue(); err != nil {
			t.Fatalf("continue failed while draining injected messages: %v", err)
		}
	}

	msgs := agent.Messages()
	var texts []string
	for _, msg := range msgs {
		if text := msg.TextContent(); text != "" {
			texts = append(texts, text)
		}
	}
	for _, want := range []string{"inject-a", "inject-b"} {
		found := false
		for _, got := range texts {
			if got == want {
				found = true
				break
			}
		}
		if !found {
			t.Fatalf("expected injected message %q in history, got %v", want, texts)
		}
	}
}

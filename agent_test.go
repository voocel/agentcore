package agentcore

import (
	"context"
	"fmt"
	"sync"
	"testing"
	"time"
)

func TestAgentInject_WhenRunning_ReturnsSteeredCurrentRun(t *testing.T) {
	release := make(chan struct{})
	agent := NewAgent(
		WithModel(funcModel(func(ctx context.Context, req *LLMRequest) (*LLMResponse, error) {
			<-release
			return &LLMResponse{Message: assistantMsg("done", StopReasonStop)}, nil
		})),
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
		WithModel(mockModel(
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
		WithModel(mockModel(
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

// midStreamErrorModel emits a text delta (populating Agent.streamMessage)
// then injects a StreamEventError so the agent surfaces EventError mid-stream.
type midStreamErrorModel struct{}

func (m *midStreamErrorModel) Generate(context.Context, []Message, []ToolSpec, ...CallOption) (*LLMResponse, error) {
	return nil, fmt.Errorf("Generate not used")
}
func (m *midStreamErrorModel) GenerateStream(_ context.Context, _ []Message, _ []ToolSpec, _ ...CallOption) (<-chan StreamEvent, error) {
	ch := make(chan StreamEvent, 4)
	partial := Message{Role: RoleAssistant, Content: []ContentBlock{TextBlock("")}}
	ch <- StreamEvent{Type: StreamEventTextStart, ContentIndex: 0, Message: partial}
	partial.Content[0].Text = "half-formed..."
	ch <- StreamEvent{Type: StreamEventTextDelta, ContentIndex: 0, Delta: "half-formed...", Message: partial}
	ch <- StreamEvent{Type: StreamEventError, Err: fmt.Errorf("provider stream error")}
	close(ch)
	return ch, nil
}
func (m *midStreamErrorModel) SupportsTools() bool { return true }

// EventError listeners must see a cleared StreamMessage when calling State().
// Without this guarantee, the listener observes a never-completing partial
// the agent has already abandoned — confusing UI rendering and any caller
// that snapshots agent state at error time.
func TestEventError_ClearsStreamMessageBeforeListeners(t *testing.T) {
	agent := NewAgent(
		WithModel(&midStreamErrorModel{}),
		WithMaxRetries(0),
	)

	var (
		mu             sync.Mutex
		errorSnapshot  AgentState
		sawError       bool
		sawMidStream   bool
		done           = make(chan struct{})
		closeDoneOnce  sync.Once
	)
	agent.Subscribe(func(ev Event) {
		switch ev.Type {
		case EventMessageUpdate:
			// Sanity check: streamMessage must be populated mid-stream,
			// otherwise the EventError assertion below would be vacuous.
			if agent.State().StreamMessage != nil {
				mu.Lock()
				sawMidStream = true
				mu.Unlock()
			}
		case EventError:
			mu.Lock()
			errorSnapshot = agent.State()
			sawError = true
			mu.Unlock()
		case EventAgentEnd:
			closeDoneOnce.Do(func() { close(done) })
		}
	})

	if err := agent.Prompt("trigger"); err != nil {
		t.Fatalf("prompt failed: %v", err)
	}
	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for EventAgentEnd")
	}

	mu.Lock()
	defer mu.Unlock()
	if !sawMidStream {
		t.Fatal("test setup broken: streamMessage was never populated mid-stream")
	}
	if !sawError {
		t.Fatal("listener never received EventError")
	}
	if errorSnapshot.StreamMessage != nil {
		t.Fatalf("StreamMessage must be cleared before EventError listeners run, got %+v", errorSnapshot.StreamMessage)
	}
}


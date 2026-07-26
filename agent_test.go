package agentcore

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
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

	if err := agent.Prompt(context.Background(), "start"); err != nil {
		t.Fatalf("prompt failed: %v", err)
	}

	result, err := agent.Inject(context.Background(), UserMsg("runtime steer"))
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

	if err := agent.Prompt(context.Background(), "start"); err != nil {
		t.Fatalf("prompt failed: %v", err)
	}
	agent.WaitForIdle()

	result, err := agent.Inject(context.Background(), UserMsg("runtime reminder"))
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

	result, err := agent.Inject(context.Background(), UserMsg("queued"))
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
	if _, err := agent.Inject(context.Background(), nil); err == nil {
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

	if err := agent.Prompt(context.Background(), "start"); err != nil {
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
			outcomes[i].result, outcomes[i].err = agent.Inject(context.Background(), msgsToInject[i])
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
		if err := agent.Continue(context.Background()); err != nil {
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
		mu            sync.Mutex
		errorSnapshot AgentState
		sawError      bool
		sawMidStream  bool
		done          = make(chan struct{})
		closeDoneOnce sync.Once
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

	if err := agent.Prompt(context.Background(), "trigger"); err != nil {
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

// A run started from another run's EventAgentEnd listener — exactly how a
// harness auto-continues — must survive the finishing run's cleanup. The
// finishing run's consumeLoop defer once reset a.cancel, which the synchronous
// resume had already reassigned to the new run, aborting it the instant it
// began. The fix: cleanup uses the run's own captured cancel and only resets
// shared state when no newer run has taken over (a.done == myDone).
func TestResumeFromAgentEndListenerSurvivesCleanup(t *testing.T) {
	started := make(chan struct{})
	release := make(chan struct{})
	cancelled := make(chan struct{})
	var calls atomic.Int32
	agent := NewAgent(WithModel(funcModel(func(ctx context.Context, _ *LLMRequest) (*LLMResponse, error) {
		if calls.Add(1) == 1 {
			return &LLMResponse{Message: assistantMsg("run 1", StopReasonStop)}, nil
		}
		// Resumed run: announce arrival, then hold until released — unless the
		// finishing run's cleanup wrongly cancels our ctx.
		close(started)
		select {
		case <-release:
			return &LLMResponse{Message: assistantMsg("run 2", StopReasonStop)}, nil
		case <-ctx.Done():
			close(cancelled)
			return nil, ctx.Err()
		}
	})))

	var resumed atomic.Bool
	agent.Subscribe(func(ev Event) {
		if ev.Type == EventAgentEnd && !resumed.Swap(true) {
			_, _ = agent.Inject(context.Background(), UserMsg("resume"))
		}
	})

	if err := agent.Prompt(context.Background(), "start"); err != nil {
		t.Fatalf("prompt failed: %v", err)
	}

	select {
	case <-started:
		// Resumed run reached the model; give the finishing run's defer its
		// window to (wrongly) cancel it.
		select {
		case <-cancelled:
			t.Fatal("resumed run was cancelled by the finishing run's cleanup")
		case <-time.After(200 * time.Millisecond):
		}
		close(release)
	case <-cancelled:
		t.Fatal("resumed run was cancelled before reaching the model")
	case <-time.After(2 * time.Second):
		t.Fatal("resumed run never started")
	}

	agent.WaitForIdle()
	msgs := agent.Messages()
	if got := msgs[len(msgs)-1].TextContent(); got != "run 2" {
		t.Fatalf("resumed run did not complete: last message = %q, want %q", got, "run 2")
	}
}

// holdWithTimeout runs HoldRuns on a goroutine so a deadlock fails the test
// instead of hanging the package.
func holdWithTimeout(t *testing.T, agent *Agent) (release func()) {
	t.Helper()
	done := make(chan struct{})
	go func() {
		release = agent.HoldRuns()
		close(done)
	}()
	select {
	case <-done:
		return release
	case <-time.After(2 * time.Second):
		t.Fatal("HoldRuns did not return")
		return nil
	}
}

func TestHoldRunsDrainsActiveRun(t *testing.T) {
	started := make(chan struct{})
	sawCancel := make(chan struct{})
	agent := NewAgent(WithModel(funcModel(func(ctx context.Context, _ *LLMRequest) (*LLMResponse, error) {
		close(started)
		<-ctx.Done()
		close(sawCancel)
		return nil, ctx.Err()
	})))

	if err := agent.Prompt(context.Background(), "start"); err != nil {
		t.Fatalf("prompt failed: %v", err)
	}
	<-started

	release := holdWithTimeout(t, agent)
	defer release()

	select {
	case <-sawCancel:
	default:
		t.Fatal("model never observed the hold's cancellation")
	}
	if agent.State().IsRunning {
		t.Fatal("agent still running after HoldRuns returned")
	}
	// The cancel is silent: no abort marker message may be appended.
	msgs := agent.Messages()
	if len(msgs) != 1 || msgs[0].TextContent() != "start" {
		var texts []string
		for _, m := range msgs {
			texts = append(texts, m.TextContent())
		}
		t.Fatalf("history polluted by hold drain: %v", texts)
	}
}

func TestHoldRunsBlocksPromptContinueInject(t *testing.T) {
	agent := NewAgent(WithModel(mockModel(assistantMsg("later", StopReasonStop))))
	if err := agent.SetMessages([]AgentMessage{
		UserMsg("earlier"),
		assistantMsg("earlier reply", StopReasonStop),
	}); err != nil {
		t.Fatalf("set messages failed: %v", err)
	}

	release := holdWithTimeout(t, agent)

	if err := agent.Prompt(context.Background(), "blocked"); !errors.Is(err, ErrRunsHeld) {
		t.Fatalf("Prompt during hold: err = %v, want ErrRunsHeld", err)
	}
	agent.Steer(UserMsg("queued steer"))
	if err := agent.Continue(context.Background()); !errors.Is(err, ErrRunsHeld) {
		t.Fatalf("Continue during hold: err = %v, want ErrRunsHeld", err)
	}
	if !agent.HasQueuedMessages() {
		t.Fatal("held Continue must not consume queued messages")
	}
	if _, err := agent.Inject(context.Background(), UserMsg("blocked inject")); !errors.Is(err, ErrRunsHeld) {
		t.Fatalf("Inject during hold: err = %v, want ErrRunsHeld", err)
	}

	release()
	if err := agent.Prompt(context.Background(), "resumes"); err != nil {
		t.Fatalf("prompt after release failed: %v", err)
	}
	agent.WaitForIdle()

	// The steer queued during the hold survives it (queues are untouched) and
	// is delivered to the released run via its initial steering poll.
	var steerDelivered bool
	for _, m := range agent.Messages() {
		if m.TextContent() == "queued steer" {
			steerDelivered = true
		}
	}
	if !steerDelivered {
		t.Fatal("steer queued during hold was not delivered to the post-release run")
	}
}

func TestHoldRunsCounterAndIdempotentRelease(t *testing.T) {
	agent := NewAgent(WithModel(mockModel(assistantMsg("ok", StopReasonStop))))

	release1 := holdWithTimeout(t, agent)
	release2 := holdWithTimeout(t, agent)

	release1()
	release1() // idempotent: must not decrement twice
	if err := agent.Prompt(context.Background(), "still held"); !errors.Is(err, ErrRunsHeld) {
		t.Fatalf("one holder released, err = %v, want ErrRunsHeld", err)
	}

	release2()
	if err := agent.Prompt(context.Background(), "released"); err != nil {
		t.Fatalf("prompt after all releases failed: %v", err)
	}
	agent.WaitForIdle()
}

func TestHoldRunsRejectsListenerAutoContinueWithoutDeadlock(t *testing.T) {
	started := make(chan struct{})
	agent := NewAgent(WithModel(funcModel(func(ctx context.Context, _ *LLMRequest) (*LLMResponse, error) {
		close(started)
		<-ctx.Done()
		return nil, ctx.Err()
	})))

	var injectErr error
	injectDone := make(chan struct{})
	agent.Subscribe(func(ev Event) {
		if ev.Type == EventAgentEnd {
			_, injectErr = agent.Inject(context.Background(), UserMsg("auto resume"))
			close(injectDone)
		}
	})

	if err := agent.Prompt(context.Background(), "start"); err != nil {
		t.Fatalf("prompt failed: %v", err)
	}
	<-started

	// The hold's cancel triggers EventAgentEnd, whose listener attempts an
	// auto-continue while the hold is draining. It must fail fast — a blocking
	// acquire here would deadlock HoldRuns against the listener goroutine.
	release := holdWithTimeout(t, agent)
	defer release()

	select {
	case <-injectDone:
	case <-time.After(2 * time.Second):
		t.Fatal("EventAgentEnd listener never ran")
	}
	if !errors.Is(injectErr, ErrRunsHeld) {
		t.Fatalf("listener auto-continue: err = %v, want ErrRunsHeld", injectErr)
	}
	if agent.State().IsRunning {
		t.Fatal("agent running after hold drained a listener-resume attempt")
	}
}

func TestInjectDuringHoldFailsFastWithoutQueueing(t *testing.T) {
	agent := NewAgent()
	if err := agent.SetMessages([]AgentMessage{
		UserMsg("q"),
		assistantMsg("a", StopReasonStop),
	}); err != nil {
		t.Fatalf("set messages failed: %v", err)
	}

	release := holdWithTimeout(t, agent)
	defer release()

	if _, err := agent.Inject(context.Background(), UserMsg("dropped")); !errors.Is(err, ErrRunsHeld) {
		t.Fatalf("inject during hold: err = %v, want ErrRunsHeld", err)
	}
	if agent.HasQueuedMessages() {
		t.Fatal("held inject must not leave the message queued (double delivery once resumed)")
	}
}

func TestHoldRunsVsConcurrentPromptRace(t *testing.T) {
	agent := NewAgent(WithModel(funcModel(func(_ context.Context, _ *LLMRequest) (*LLMResponse, error) {
		return &LLMResponse{Message: assistantMsg("quick", StopReasonStop)}, nil
	})))

	stop := make(chan struct{})
	var wg sync.WaitGroup
	wg.Go(func() {
		for {
			select {
			case <-stop:
				return
			default:
				_ = agent.Prompt(context.Background(), "spin")
			}
		}
	})

	for range 20 {
		release := holdWithTimeout(t, agent)
		if agent.State().IsRunning {
			release()
			t.Fatal("run in flight while held")
		}
		release()
	}
	close(stop)
	wg.Wait()
	agent.WaitForIdle()
}

func TestHoldRunsOnIdleAgentReturnsImmediately(t *testing.T) {
	// Never ran: done is nil.
	fresh := NewAgent()
	release := holdWithTimeout(t, fresh)
	release()

	// Ran to completion: done is a closed channel.
	agent := NewAgent(WithModel(mockModel(
		assistantMsg("done", StopReasonStop),
		assistantMsg("after", StopReasonStop),
	)))
	if err := agent.Prompt(context.Background(), "start"); err != nil {
		t.Fatalf("prompt failed: %v", err)
	}
	agent.WaitForIdle()

	release = holdWithTimeout(t, agent)
	if err := agent.Prompt(context.Background(), "held"); !errors.Is(err, ErrRunsHeld) {
		t.Fatalf("prompt during idle hold: err = %v, want ErrRunsHeld", err)
	}
	release()
	if err := agent.Prompt(context.Background(), "released"); err != nil {
		t.Fatalf("prompt after release failed: %v", err)
	}
	agent.WaitForIdle()
}

func TestResetDrainsActiveRunAndLeavesAgentUsable(t *testing.T) {
	started := make(chan struct{})
	var calls atomic.Int32
	agent := NewAgent(WithModel(funcModel(func(ctx context.Context, _ *LLMRequest) (*LLMResponse, error) {
		if calls.Add(1) == 1 {
			close(started)
			<-ctx.Done()
			return nil, ctx.Err()
		}
		return &LLMResponse{Message: assistantMsg("fresh", StopReasonStop)}, nil
	})))

	if err := agent.Prompt(context.Background(), "start"); err != nil {
		t.Fatalf("prompt failed: %v", err)
	}
	<-started

	resetDone := make(chan struct{})
	go func() {
		agent.Reset()
		close(resetDone)
	}()
	select {
	case <-resetDone:
	case <-time.After(2 * time.Second):
		t.Fatal("Reset did not return while a run was in flight")
	}

	if agent.State().IsRunning {
		t.Fatal("agent running after Reset")
	}
	if got := len(agent.Messages()); got != 0 {
		t.Fatalf("history not cleared: %d messages remain", got)
	}
	if err := agent.Prompt(context.Background(), "again"); err != nil {
		t.Fatalf("prompt after reset failed: %v", err)
	}
	agent.WaitForIdle()
	msgs := agent.Messages()
	if got := msgs[len(msgs)-1].TextContent(); got != "fresh" {
		t.Fatalf("post-reset run did not complete: last message = %q", got)
	}
}

func TestSetMessagesWhileRunningReturnsErrAlreadyRunning(t *testing.T) {
	started := make(chan struct{})
	release := make(chan struct{})
	agent := NewAgent(WithModel(funcModel(func(_ context.Context, _ *LLMRequest) (*LLMResponse, error) {
		close(started)
		<-release
		return &LLMResponse{Message: assistantMsg("done", StopReasonStop)}, nil
	})))

	if err := agent.Prompt(context.Background(), "start"); err != nil {
		t.Fatalf("prompt failed: %v", err)
	}
	<-started
	if err := agent.SetMessages(nil); !errors.Is(err, ErrAlreadyRunning) {
		t.Fatalf("clear during run: err = %v, want ErrAlreadyRunning", err)
	}

	close(release)
	agent.WaitForIdle()
	if err := agent.SetMessages(nil); err != nil {
		t.Fatalf("clear while idle failed: %v", err)
	}
	if got := len(agent.Messages()); got != 0 {
		t.Fatalf("history not cleared: %d messages remain", got)
	}
}

// TestInjectDuringHoldChurnNeverStrandsMessage races Inject against a
// hold/release churn goroutine. The invariant under test is Inject's atomic
// idle-resume: whenever an inject fails with ErrRunsHeld the message
// must NOT be left queued (the caller reroutes it — a queued copy would
// double-deliver on the next resume), and a reported resume really started.
func TestInjectDuringHoldChurnNeverStrandsMessage(t *testing.T) {
	agent := NewAgent(WithModel(funcModel(func(_ context.Context, _ *LLMRequest) (*LLMResponse, error) {
		return &LLMResponse{Message: assistantMsg("reply", StopReasonStop)}, nil
	})))
	if err := agent.SetMessages([]AgentMessage{
		UserMsg("q"),
		assistantMsg("a", StopReasonStop),
	}); err != nil {
		t.Fatalf("set messages failed: %v", err)
	}

	stop := make(chan struct{})
	var wg sync.WaitGroup
	wg.Go(func() {
		for {
			select {
			case <-stop:
				return
			default:
				release := agent.HoldRuns()
				release()
			}
		}
	})

	for range 50 {
		// Per-iteration reset so the queue is provably empty going in: a run the
		// churn cancels mid-flight leaves a user tail, and the next inject then
		// queues legitimately — without the reset that queued message would trip
		// the ErrRunsHeld assertion below. Only inject starts runs here, so the
		// agent is reliably idle after WaitForIdle.
		agent.WaitForIdle()
		agent.ClearAllQueues()
		if err := agent.SetMessages([]AgentMessage{
			UserMsg("q"),
			assistantMsg("a", StopReasonStop),
		}); err != nil {
			t.Fatalf("reset messages failed: %v", err)
		}

		res, err := agent.Inject(context.Background(), UserMsg("inject"))
		switch {
		case errors.Is(err, ErrRunsHeld):
			if agent.HasQueuedMessages() {
				t.Fatal("ErrRunsHeld inject left the message queued")
			}
		case err != nil:
			t.Fatalf("inject failed: %v", err)
		case res.Disposition == InjectResumedIdleRun:
			agent.WaitForIdle()
		}
	}
	close(stop)
	wg.Wait()
	agent.WaitForIdle()
}

func TestFollowUpSurvivesHoldAndDeliversAfterRelease(t *testing.T) {
	agent := NewAgent(WithModel(mockModel(assistantMsg("follow-up reply", StopReasonStop))))
	if err := agent.SetMessages([]AgentMessage{
		UserMsg("q"),
		assistantMsg("a", StopReasonStop),
	}); err != nil {
		t.Fatalf("set messages failed: %v", err)
	}
	agent.FollowUp(UserMsg("queued follow-up"))

	release := holdWithTimeout(t, agent)
	if err := agent.Continue(context.Background()); !errors.Is(err, ErrRunsHeld) {
		t.Fatalf("Continue during hold: err = %v, want ErrRunsHeld", err)
	}
	if !agent.HasFollowUps() {
		t.Fatal("hold consumed the follow-up queue")
	}
	release()

	if err := agent.Continue(context.Background()); err != nil {
		t.Fatalf("continue after release failed: %v", err)
	}
	agent.WaitForIdle()
	if agent.HasFollowUps() {
		t.Fatal("released Continue did not consume the follow-up")
	}
	var delivered bool
	for _, m := range agent.Messages() {
		if m.TextContent() == "queued follow-up" {
			delivered = true
		}
	}
	if !delivered {
		t.Fatal("follow-up never delivered to the resumed run")
	}
}

func TestTwoHoldersDrainOneActiveRun(t *testing.T) {
	started := make(chan struct{})
	var calls atomic.Int32
	agent := NewAgent(WithModel(funcModel(func(ctx context.Context, _ *LLMRequest) (*LLMResponse, error) {
		if calls.Add(1) == 1 {
			close(started)
			<-ctx.Done()
			return nil, ctx.Err()
		}
		return &LLMResponse{Message: assistantMsg("fresh", StopReasonStop)}, nil
	})))
	if err := agent.Prompt(context.Background(), "start"); err != nil {
		t.Fatalf("prompt failed: %v", err)
	}
	<-started

	// Both holders target the same in-flight run: both must drain and return.
	results := make(chan func(), 2)
	for range 2 {
		go func() { results <- agent.HoldRuns() }()
	}
	var releases []func()
	for range 2 {
		select {
		case r := <-results:
			releases = append(releases, r)
		case <-time.After(2 * time.Second):
			t.Fatal("concurrent HoldRuns on one active run did not both return")
		}
	}
	if agent.State().IsRunning {
		t.Fatal("run in flight while held twice")
	}

	releases[0]()
	if err := agent.Prompt(context.Background(), "still held"); !errors.Is(err, ErrRunsHeld) {
		t.Fatalf("one holder released, err = %v, want ErrRunsHeld", err)
	}
	releases[1]()
	if err := agent.Prompt(context.Background(), "released"); err != nil {
		t.Fatalf("prompt after all releases failed: %v", err)
	}
	agent.WaitForIdle()
}

// TestAbortWhileIdleDoesNotArmMarkerForLaterHold pins the idle-Abort guard: an
// Abort with no run in flight must be a full no-op. Before the guard it armed
// wantAbortMarker with nothing to consume it, so the next silent cancellation
// (a HoldRuns drain) emitted an abort marker it never requested.
func TestAbortWhileIdleDoesNotArmMarkerForLaterHold(t *testing.T) {
	started := make(chan struct{})
	agent := NewAgent(WithModel(funcModel(func(ctx context.Context, _ *LLMRequest) (*LLMResponse, error) {
		close(started)
		<-ctx.Done()
		return nil, ctx.Err()
	})))

	agent.Abort() // idle: nothing to interrupt

	if err := agent.Prompt(context.Background(), "start"); err != nil {
		t.Fatalf("prompt failed: %v", err)
	}
	<-started
	release := holdWithTimeout(t, agent)
	defer release()

	msgs := agent.Messages()
	if len(msgs) != 1 || msgs[0].TextContent() != "start" {
		var texts []string
		for _, m := range msgs {
			texts = append(texts, m.TextContent())
		}
		t.Fatalf("stale idle Abort polluted the hold drain: %v", texts)
	}
}

func TestHasFollowUpsTracksOnlyFollowUpQueue(t *testing.T) {
	agent := NewAgent()
	agent.Steer(UserMsg("steer"))
	if agent.HasFollowUps() {
		t.Fatal("steering message must not count as a follow-up")
	}

	agent.FollowUp(UserMsg("follow up"))
	if !agent.HasFollowUps() {
		t.Fatal("queued follow-up was not reported")
	}

	agent.ClearFollowUpQueue()
	if agent.HasFollowUps() {
		t.Fatal("cleared follow-up queue was still reported")
	}
	if !agent.HasQueuedMessages() {
		t.Fatal("clearing follow-ups must not clear steering messages")
	}
}

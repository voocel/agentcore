package team

import (
	"context"
	"errors"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/voocel/agentcore"
	"github.com/voocel/agentcore/task"
)

// testHooks builds a ProtocolHooks set tests can use to observe and steer the
// Run loop. Each field defaults to a useful behaviour:
//   - FormatInitialPrompt: tag with "INIT:" so tests can assert the hook ran
//   - FormatPrompt:        tag with "MSG:" so tests can assert the hook ran
//   - EncodeIdle:          return "idle:<from>:<lastText>" so tests can probe
//   - ShouldTerminate:     match the sentinel "STOP"
//   - PickPriority:        always pick the highest tier — leader > others
//
// Tests override fields they care about and use the rest as background noise.
type testHooks struct {
	hooks ProtocolHooks
}

func makeTestHooks() *testHooks {
	t := &testHooks{}
	t.hooks = ProtocolHooks{
		FormatPrompt: func(m Message) string {
			if m.From == TeamLeadName && m.Summary != "" {
				return "MSG[" + m.From + "/" + m.Summary + "]:" + m.Text
			}
			return "MSG[" + m.From + "]:" + m.Text
		},
		EncodeIdle: func(from, lastText string) string {
			return "idle:" + from + ":" + lastText
		},
		ShouldTerminate: func(text string) bool {
			return text == "STOP"
		},
		PickPriority: func(q []Message) int {
			for i, m := range q {
				if m.From == TeamLeadName {
					return i
				}
			}
			return 0
		},
	}
	return t
}

// assistantMsg builds a minimal Message with assistant role for stub responses.
// agentcore does not expose a public AssistantMsg constructor, so tests build
// the value directly.
func assistantMsg(text string) agentcore.AgentMessage {
	return agentcore.Message{
		Role:       agentcore.RoleAssistant,
		Content:    []agentcore.ContentBlock{agentcore.TextBlock(text)},
		StopReason: agentcore.StopReasonStop,
	}
}

// stubExecutor is a TurnExecutor that records calls and emits canned
// responses. The done channel signals when each call returns so tests can
// synchronise without sleeping.
type stubExecutor struct {
	mu       sync.Mutex
	calls    [][]agentcore.AgentMessage
	respond  func(idx int, msgs []agentcore.AgentMessage) ([]agentcore.AgentMessage, error)
	done     chan int
	blockOn  chan struct{} // optional: closes when blocked
	released chan struct{} // optional: tests close this to release the executor
}

func newStubExecutor() *stubExecutor {
	return &stubExecutor{done: make(chan int, 32)}
}

func (s *stubExecutor) Execute(ctx context.Context, msgs []agentcore.AgentMessage) ([]agentcore.AgentMessage, error) {
	s.mu.Lock()
	idx := len(s.calls)
	clone := make([]agentcore.AgentMessage, len(msgs))
	copy(clone, msgs)
	s.calls = append(s.calls, clone)
	respond := s.respond
	blockOn := s.blockOn
	released := s.released
	s.mu.Unlock()

	if blockOn != nil {
		close(blockOn)
		select {
		case <-released:
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}

	var resp []agentcore.AgentMessage
	var err error
	if respond != nil {
		resp, err = respond(idx, msgs)
	}

	select {
	case s.done <- idx:
	default:
	}
	return resp, err
}

func (s *stubExecutor) Calls() [][]agentcore.AgentMessage {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make([][]agentcore.AgentMessage, len(s.calls))
	copy(out, s.calls)
	return out
}

// waitForCalls blocks until at least n Execute calls have completed, or the
// timeout elapses (test failure).
func (s *stubExecutor) waitForCalls(t *testing.T, n int, timeout time.Duration) {
	t.Helper()
	deadline := time.After(timeout)
	for {
		s.mu.Lock()
		got := len(s.calls)
		s.mu.Unlock()
		if got >= n {
			return
		}
		select {
		case <-s.done:
		case <-deadline:
			t.Fatalf("timed out waiting for %d Execute calls (got %d)", n, got)
		}
	}
}

// buildHarness wires together a registry + task runtime + entry + identity,
// ready to drive Run in a goroutine. Returns the cancel function and a
// finalizer for the test to wait on Run's exit.
type harness struct {
	rt       *task.Runtime
	reg      *Registry
	identity *task.Identity
	taskID   string
	exec     *stubExecutor
	cancel   context.CancelFunc
	runErr   chan error
}

func newHarness(t *testing.T) *harness {
	t.Helper()
	reg := NewRegistry()
	if err := reg.CreateTeam("alpha", "test", "leader-1"); err != nil {
		t.Fatalf("CreateTeam: %v", err)
	}
	if err := reg.RegisterAgent("researcher", "tm-1"); err != nil {
		t.Fatalf("RegisterAgent: %v", err)
	}

	rt := task.NewRuntime()
	entry := &task.Entry{
		ID:        "tm-1",
		Type:      task.TypeTeammate,
		Status:    task.Running,
		StartedAt: time.Now(),
		Identity: &task.Identity{
			AgentID:   "researcher@alpha",
			AgentName: "researcher",
			TeamName:  "alpha",
		},
	}
	rt.Register(entry)

	return &harness{
		rt:       rt,
		reg:      reg,
		identity: entry.Identity,
		taskID:   entry.ID,
		exec:     newStubExecutor(),
		runErr:   make(chan error, 1),
	}
}

func (h *harness) start(t *testing.T, initialPrompt string) {
	t.Helper()
	h.startWithHooks(t, initialPrompt, ProtocolHooks{})
}

// startWithHooks lets a test inject a custom ProtocolHooks set. A zero hooks
// value yields the agentcore defaults (passthrough format, no idle, no
// terminate, FIFO), which matches the historical hard-coded behaviour for
// tests that don't care about protocol layering.
func (h *harness) startWithHooks(t *testing.T, initialPrompt string, hooks ProtocolHooks) {
	t.Helper()
	ctx, cancel := context.WithCancel(context.Background())
	h.cancel = cancel
	go func() {
		h.runErr <- Run(ctx, RunConfig{
			Identity:      h.identity,
			InitialPrompt: initialPrompt,
			Registry:      h.reg,
			TaskRT:        h.rt,
			TaskID:        h.taskID,
			Execute:       h.exec.Execute,
			Protocol:      hooks,
		})
	}()
}

func (h *harness) stop(t *testing.T, timeout time.Duration) error {
	t.Helper()
	if h.cancel != nil {
		h.cancel()
	}
	select {
	case err := <-h.runErr:
		return err
	case <-time.After(timeout):
		t.Fatal("Run did not exit on cancel")
		return nil
	}
}

// ---------------------------------------------------------------------------
// Run tests
// ---------------------------------------------------------------------------

// Default hooks pass the InitialPrompt through verbatim — agentcore makes no
// format choices on its own.
func TestRun_InitialPromptVerbatimWithDefaultHooks(t *testing.T) {
	h := newHarness(t)
	h.start(t, "investigate the auth bug")
	defer h.stop(t, time.Second)

	h.exec.waitForCalls(t, 1, time.Second)

	calls := h.exec.Calls()
	if len(calls) < 1 || len(calls[0]) != 1 {
		t.Fatalf("first call should have 1 message, got %d call(s) shape %v", len(calls), calls)
	}
	if got := calls[0][0].TextContent(); got != "investigate the auth bug" {
		t.Errorf("default FormatPrompt should be passthrough on the synthetic initial message; got %q", got)
	}
}

// The synthetic initial prompt flows through the same FormatPrompt hook as
// every later inbound message — packaged as a Message with From=TeamLeadName
// and Summary=Description. The hook can rely on those fields just like a real
// inbound message.
func TestRun_InitialPromptUsesFormatPromptHook(t *testing.T) {
	h := newHarness(t)
	th := makeTestHooks()
	h.startWithHooks(t, "investigate the auth bug", th.hooks)
	defer h.stop(t, time.Second)

	h.exec.waitForCalls(t, 1, time.Second)

	text := h.exec.Calls()[0][0].TextContent()
	wantPrefix := "MSG[" + TeamLeadName + "]:"
	if !strings.HasPrefix(text, wantPrefix) {
		t.Errorf("expected FormatPrompt to wrap initial message with %q; got %q", wantPrefix, text)
	}
	if !strings.Contains(text, "investigate the auth bug") {
		t.Errorf("hook output dropped the input body: %q", text)
	}
}

// Description flows through to the FormatPrompt hook as Message.Summary on
// the synthetic initial message.
func TestRun_InitialPromptCarriesDescriptionAsSummary(t *testing.T) {
	h := newHarness(t)
	th := makeTestHooks()
	ctx, cancel := context.WithCancel(context.Background())
	h.cancel = cancel
	go func() {
		h.runErr <- Run(ctx, RunConfig{
			Identity:      h.identity,
			InitialPrompt: "investigate the auth bug",
			Description:   "Stage D",
			Registry:      h.reg,
			TaskRT:        h.rt,
			TaskID:        h.taskID,
			Execute:       h.exec.Execute,
			Protocol:      th.hooks,
		})
	}()
	defer h.stop(t, time.Second)

	h.exec.waitForCalls(t, 1, time.Second)

	text := h.exec.Calls()[0][0].TextContent()
	want := "MSG[" + TeamLeadName + "/Stage D]:investigate the auth bug"
	if text != want {
		t.Errorf("FormatPrompt did not receive Summary; got %q want %q", text, want)
	}
}

func TestRun_HistoryAccumulatesAcrossTurns(t *testing.T) {
	h := newHarness(t)
	// Each turn the assistant produces one message; history should grow by
	// (1 user + 1 assistant) per turn.
	h.exec.respond = func(idx int, _ []agentcore.AgentMessage) ([]agentcore.AgentMessage, error) {
		return []agentcore.AgentMessage{assistantMsg("ack " + string(rune('0'+idx)))}, nil
	}
	h.start(t, "kickoff")
	defer h.stop(t, time.Second)

	h.exec.waitForCalls(t, 1, time.Second)

	// Deliver a peer message to trigger turn 2.
	mb := h.reg.Mailbox("researcher")
	if err := mb.Send(Message{From: "tester", Text: "have you seen X?"}); err != nil {
		t.Fatalf("Send: %v", err)
	}
	h.exec.waitForCalls(t, 2, time.Second)

	calls := h.exec.Calls()
	// Turn 2 should see: prior user (initial), prior assistant (ack 0), new user (peer DM)
	if len(calls[1]) != 3 {
		t.Fatalf("turn 2 expected 3 messages, got %d: %+v", len(calls[1]), calls[1])
	}
	if calls[1][0].GetRole() != agentcore.RoleUser {
		t.Errorf("history[0] role = %v, want user", calls[1][0].GetRole())
	}
	if calls[1][1].GetRole() != agentcore.RoleAssistant {
		t.Errorf("history[1] role = %v, want assistant", calls[1][1].GetRole())
	}
	if calls[1][1].TextContent() != "ack 0" {
		t.Errorf("history[1] = %q, want 'ack 0'", calls[1][1].TextContent())
	}
	if !strings.Contains(calls[1][2].TextContent(), "have you seen X?") {
		t.Errorf("new prompt missing body: %q", calls[1][2].TextContent())
	}
}

// History seeds the conversation: the first Execute must receive the resumed
// transcript as a prefix, with the InitialPrompt appended as the new user
// turn. This is the resume primitive — a restarted teammate continues with
// its prior context instead of starting blank.
func TestRun_HistorySeedsFirstTurnPrefix(t *testing.T) {
	h := newHarness(t)
	seed := []agentcore.AgentMessage{
		agentcore.UserMsg("earlier question"),
		assistantMsg("earlier answer"),
	}
	ctx, cancel := context.WithCancel(context.Background())
	h.cancel = cancel
	go func() {
		h.runErr <- Run(ctx, RunConfig{
			Identity:      h.identity,
			InitialPrompt: "continue please",
			History:       seed,
			Registry:      h.reg,
			TaskRT:        h.rt,
			TaskID:        h.taskID,
			Execute:       h.exec.Execute,
		})
	}()
	defer h.stop(t, time.Second)

	h.exec.waitForCalls(t, 1, time.Second)

	first := h.exec.Calls()[0]
	if len(first) != 3 {
		t.Fatalf("first turn should be seed(2) + prompt(1) = 3 messages, got %d: %+v", len(first), first)
	}
	if first[0].GetRole() != agentcore.RoleUser || first[0].TextContent() != "earlier question" {
		t.Errorf("history[0] = (%v, %q), want (user, 'earlier question')", first[0].GetRole(), first[0].TextContent())
	}
	if first[1].GetRole() != agentcore.RoleAssistant || first[1].TextContent() != "earlier answer" {
		t.Errorf("history[1] = (%v, %q), want (assistant, 'earlier answer')", first[1].GetRole(), first[1].TextContent())
	}
	if first[2].GetRole() != agentcore.RoleUser || first[2].TextContent() != "continue please" {
		t.Errorf("prompt = (%v, %q), want (user, 'continue please')", first[2].GetRole(), first[2].TextContent())
	}
}

// Seeding History must not alias the caller's slice: the runner appends to its
// own history each turn, and a second turn's growth must not mutate or grow
// into the caller's backing array.
func TestRun_HistorySeedIsCopied(t *testing.T) {
	h := newHarness(t)
	h.exec.respond = func(idx int, _ []agentcore.AgentMessage) ([]agentcore.AgentMessage, error) {
		return []agentcore.AgentMessage{assistantMsg("ack")}, nil
	}
	// Over-capacity slice so an un-copied append would write into seed[1:].
	seed := make([]agentcore.AgentMessage, 1, 8)
	seed[0] = agentcore.UserMsg("seed-msg")
	ctx, cancel := context.WithCancel(context.Background())
	h.cancel = cancel
	go func() {
		h.runErr <- Run(ctx, RunConfig{
			Identity:      h.identity,
			InitialPrompt: "go",
			History:       seed,
			Registry:      h.reg,
			TaskRT:        h.rt,
			TaskID:        h.taskID,
			Execute:       h.exec.Execute,
		})
	}()
	defer h.stop(t, time.Second)

	h.exec.waitForCalls(t, 1, time.Second)
	if len(seed) != 1 || seed[0].TextContent() != "seed-msg" {
		t.Errorf("caller's History slice was mutated: len=%d head=%q", len(seed), seed[0].TextContent())
	}
}

// With default hooks, EncodeIdle returns "" and no idle notification is sent.
// The leader's mailbox should stay empty after a teammate turn.
func TestRun_NoIdleNotificationWithDefaultHooks(t *testing.T) {
	h := newHarness(t)
	h.start(t, "go")
	defer h.stop(t, time.Second)

	h.exec.waitForCalls(t, 1, time.Second)

	// Give the runner a few ms past Execute return — if it were going to send
	// an idle notification, it would have done so by now.
	time.Sleep(20 * time.Millisecond)
	if n := h.reg.Mailbox(TeamLeadName).Len(); n != 0 {
		t.Errorf("default hooks should not send idle notification, got %d msgs in leader inbox", n)
	}
}

// When EncodeIdle is injected, Run calls it after each turn with the
// teammate's last assistant text and forwards the result to the leader
// mailbox. lastText must be picked from the most recent assistant message,
// skipping intermediate tool-result placeholders.
func TestRun_IdleHookReceivesLastAssistantText(t *testing.T) {
	h := newHarness(t)
	h.exec.respond = func(idx int, _ []agentcore.AgentMessage) ([]agentcore.AgentMessage, error) {
		return []agentcore.AgentMessage{
			agentcore.UserMsg("tool-result placeholder"),
			assistantMsg("I found the bug at line 42."),
		}, nil
	}
	th := makeTestHooks()
	h.startWithHooks(t, "investigate", th.hooks)
	defer h.stop(t, time.Second)

	h.exec.waitForCalls(t, 1, time.Second)

	leaderBox := h.reg.Mailbox(TeamLeadName)
	deadline := time.After(time.Second)
	for leaderBox.Len() == 0 {
		select {
		case <-deadline:
			t.Fatal("leader mailbox empty after turn 1")
		case <-time.After(5 * time.Millisecond):
		}
	}
	got := leaderBox.Drain()
	if len(got) != 1 {
		t.Fatalf("leader inbox = %d msgs, want 1", len(got))
	}
	want := "idle:researcher:I found the bug at line 42."
	if got[0].Text != want {
		t.Errorf("idle envelope = %q, want %q", got[0].Text, want)
	}
}

// When ShouldTerminate returns true for the picked message, Run exits cleanly
// without invoking the executor for that message — saving model tokens on a
// turn the application has decided to abandon.
func TestRun_ShouldTerminateHookExitsCleanly(t *testing.T) {
	h := newHarness(t)
	h.exec.respond = func(idx int, _ []agentcore.AgentMessage) ([]agentcore.AgentMessage, error) {
		return []agentcore.AgentMessage{assistantMsg("ok")}, nil
	}
	th := makeTestHooks()
	h.startWithHooks(t, "warmup", th.hooks)

	h.exec.waitForCalls(t, 1, time.Second)

	// The fake hooks' ShouldTerminate matches text == "STOP". Drop a STOP
	// into the mailbox — Run should exit without invoking turn 2.
	mb := h.reg.Mailbox("researcher")
	_ = mb.Send(Message{From: TeamLeadName, Text: "STOP"})

	select {
	case err := <-h.runErr:
		if err != nil {
			t.Fatalf("Run exited with error: %v", err)
		}
	case <-time.After(time.Second):
		t.Fatal("Run did not exit on ShouldTerminate within 1s")
	}

	if got := len(h.exec.Calls()); got != 1 {
		t.Errorf("expected exactly 1 Execute call (only warmup), got %d — terminate hook failed to short-circuit", got)
	}
}

// When PickPriority is injected, Run picks the queue index it returns instead
// of FIFO. The test hooks prefer leader-tagged messages, so a peer queued
// before a leader message still loses on the next turn.
func TestRun_PickPriorityHookSelectsLeaderOverPeer(t *testing.T) {
	h := newHarness(t)
	h.exec.respond = func(idx int, _ []agentcore.AgentMessage) ([]agentcore.AgentMessage, error) {
		return []agentcore.AgentMessage{assistantMsg("ok")}, nil
	}
	th := makeTestHooks()
	h.startWithHooks(t, "warmup", th.hooks)
	defer h.stop(t, time.Second)

	h.exec.waitForCalls(t, 1, time.Second)

	mb := h.reg.Mailbox("researcher")
	_ = mb.Send(Message{From: "tester", Text: "peer thought"})
	_ = mb.Send(Message{From: TeamLeadName, Text: "important leader message"})

	h.exec.waitForCalls(t, 2, time.Second)

	turn2 := h.exec.Calls()[1][len(h.exec.Calls()[1])-1].TextContent()
	if !strings.Contains(turn2, "important leader message") {
		t.Errorf("PickPriority hook did not select leader-tagged message; turn 2 got %q", turn2)
	}
}

func TestRun_RemainingMessagesProcessedInNextTurn(t *testing.T) {
	h := newHarness(t)
	h.exec.respond = func(idx int, _ []agentcore.AgentMessage) ([]agentcore.AgentMessage, error) {
		return []agentcore.AgentMessage{assistantMsg("ok")}, nil
	}
	h.start(t, "warmup")
	defer h.stop(t, time.Second)

	h.exec.waitForCalls(t, 1, time.Second)

	// Two peer messages — runner picks the first (FIFO within tier), then
	// must drain the second on its own without needing another Wait wake.
	mb := h.reg.Mailbox("researcher")
	_ = mb.Send(Message{From: "alice", Text: "message one"})
	_ = mb.Send(Message{From: "bob", Text: "message two"})

	h.exec.waitForCalls(t, 3, 2*time.Second)
	calls := h.exec.Calls()

	turn2 := calls[1][len(calls[1])-1].TextContent()
	turn3 := calls[2][len(calls[2])-1].TextContent()
	if !strings.Contains(turn2, "message one") {
		t.Errorf("turn 2 = %q, want 'message one'", turn2)
	}
	if !strings.Contains(turn3, "message two") {
		t.Errorf("turn 3 = %q, want 'message two'", turn3)
	}
}

func TestRun_ContextCancelExitsCleanly(t *testing.T) {
	h := newHarness(t)
	h.start(t, "warmup")
	h.exec.waitForCalls(t, 1, time.Second)

	if err := h.stop(t, time.Second); err != nil {
		t.Errorf("Run returned %v, want nil on cancel", err)
	}

	// Entry should be marked terminal eventually — but only Spawn updates that.
	// Run alone just exits; lifecycle update happens in Spawn's goroutine.
}

func TestRun_MailboxCloseExitsCleanly(t *testing.T) {
	h := newHarness(t)
	h.start(t, "warmup")
	h.exec.waitForCalls(t, 1, time.Second)

	// Close the teammate's mailbox while it's waiting for the next message.
	// Run should observe ErrClosed and return nil.
	if err := h.reg.UnregisterAgent("researcher"); err != nil {
		t.Fatalf("UnregisterAgent: %v", err)
	}

	select {
	case err := <-h.runErr:
		if err != nil {
			t.Errorf("Run returned %v, want nil on mailbox close", err)
		}
	case <-time.After(time.Second):
		t.Fatal("Run did not exit on mailbox close")
	}
}

func TestRun_ValidatesRequiredFields(t *testing.T) {
	cases := []struct {
		name string
		cfg  RunConfig
	}{
		{"no identity", RunConfig{Registry: NewRegistry(), TaskRT: task.NewRuntime(), TaskID: "t", Execute: func(context.Context, []agentcore.AgentMessage) ([]agentcore.AgentMessage, error) { return nil, nil }}},
		{"no registry", RunConfig{Identity: &task.Identity{AgentName: "x"}, TaskRT: task.NewRuntime(), TaskID: "t", Execute: func(context.Context, []agentcore.AgentMessage) ([]agentcore.AgentMessage, error) { return nil, nil }}},
		{"no task runtime", RunConfig{Identity: &task.Identity{AgentName: "x"}, Registry: NewRegistry(), TaskID: "t", Execute: func(context.Context, []agentcore.AgentMessage) ([]agentcore.AgentMessage, error) { return nil, nil }}},
		{"no task id", RunConfig{Identity: &task.Identity{AgentName: "x"}, Registry: NewRegistry(), TaskRT: task.NewRuntime(), Execute: func(context.Context, []agentcore.AgentMessage) ([]agentcore.AgentMessage, error) { return nil, nil }}},
		{"no executor", RunConfig{Identity: &task.Identity{AgentName: "x"}, Registry: NewRegistry(), TaskRT: task.NewRuntime(), TaskID: "t"}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := Run(context.Background(), tc.cfg)
			if err == nil {
				t.Error("expected validation error, got nil")
			}
		})
	}
}

// ---------------------------------------------------------------------------
// withDefaults — every nil hook must be replaced with a permissive default
// so Run can call hooks unconditionally without nil-checks.
// ---------------------------------------------------------------------------

func TestWithDefaults_FillsAllNilHooks(t *testing.T) {
	got := withDefaults(ProtocolHooks{})
	if got.FormatPrompt == nil ||
		got.EncodeIdle == nil ||
		got.ShouldTerminate == nil ||
		got.PickPriority == nil {
		t.Fatalf("withDefaults left a nil field: %+v", got)
	}

	if got.FormatPrompt(Message{Text: "x"}) != "x" {
		t.Errorf("default FormatPrompt should be passthrough")
	}
	if got.EncodeIdle("a", "b") != "" {
		t.Errorf("default EncodeIdle should suppress the envelope (return empty)")
	}
	if got.ShouldTerminate("anything") {
		t.Errorf("default ShouldTerminate should never terminate")
	}
	if got.PickPriority([]Message{{}, {}, {}}) != 0 {
		t.Errorf("default PickPriority should pick FIFO head (index 0)")
	}
}

func TestWithDefaults_PreservesProvidedHooks(t *testing.T) {
	custom := ProtocolHooks{
		FormatPrompt:    func(Message) string { return "custom-fp" },
		EncodeIdle:      func(string, string) string { return "custom-idle" },
		ShouldTerminate: func(string) bool { return true },
		PickPriority:    func([]Message) int { return 42 },
	}
	got := withDefaults(custom)
	if got.FormatPrompt(Message{}) != "custom-fp" ||
		got.EncodeIdle("", "") != "custom-idle" ||
		!got.ShouldTerminate("") ||
		got.PickPriority(nil) != 42 {
		t.Errorf("withDefaults clobbered a provided hook: %+v", got)
	}
}

// ---------------------------------------------------------------------------
// Spawn tests
// ---------------------------------------------------------------------------

func TestSpawn_RegistersEntryAndStartsLoop(t *testing.T) {
	reg := NewRegistry()
	if err := reg.CreateTeam("alpha", "", "leader-1"); err != nil {
		t.Fatalf("CreateTeam: %v", err)
	}
	rt := task.NewRuntime()
	exec := newStubExecutor()

	res, err := Spawn(context.Background(), SpawnConfig{
		AgentName:     "researcher",
		InitialPrompt: "investigate",
		Registry:      reg,
		TaskRT:        rt,
		Execute:       exec.Execute,
	})
	if err != nil {
		t.Fatalf("Spawn: %v", err)
	}
	if res == nil || res.AgentID != "researcher@alpha" {
		t.Errorf("SpawnResult = %+v", res)
	}

	// Entry should be registered
	entry := rt.Get(res.TaskID)
	if entry == nil {
		t.Fatal("Entry not registered")
	}
	if entry.Type != task.TypeTeammate {
		t.Errorf("Entry Type = %v, want %v", entry.Type, task.TypeTeammate)
	}
	if entry.Identity == nil || entry.Identity.AgentName != "researcher" {
		t.Errorf("Entry.Identity = %+v", entry.Identity)
	}

	// Agent should be registered in the team registry
	if id, ok := reg.TaskID("researcher"); !ok || id != res.TaskID {
		t.Errorf("Registry TaskID = (%q, %v), want (%q, true)", id, ok, res.TaskID)
	}

	// Loop should have invoked the executor at least once
	exec.waitForCalls(t, 1, time.Second)

	// Stop the teammate
	if !rt.Stop(res.TaskID) {
		t.Error("Stop returned false")
	}

	// Give the goroutine time to mark terminal and unregister
	deadline := time.After(time.Second)
	for {
		entry = rt.Get(res.TaskID)
		if entry != nil && entry.Status.IsTerminal() {
			break
		}
		select {
		case <-deadline:
			t.Fatalf("Entry never became terminal; status=%v", entry.Status)
		case <-time.After(5 * time.Millisecond):
		}
	}

	if _, ok := reg.TaskID("researcher"); ok {
		t.Error("registry still holds researcher after teammate exit")
	}
}

// Regression: Spawn must forward cfg.Protocol to Run, otherwise teammate
// goroutines fall back to default no-op hooks and the application's
// EncodeIdle / FormatPrompt / etc. silently never fire.
func TestSpawn_ForwardsProtocolHooks(t *testing.T) {
	reg := NewRegistry()
	if err := reg.CreateTeam("alpha", "", "leader-1"); err != nil {
		t.Fatalf("CreateTeam: %v", err)
	}
	rt := task.NewRuntime()
	exec := newStubExecutor()
	exec.respond = func(int, []agentcore.AgentMessage) ([]agentcore.AgentMessage, error) {
		return []agentcore.AgentMessage{assistantMsg("ack")}, nil
	}

	// Sentinel hook: if Run actually receives it, the leader mailbox gets a
	// uniquely identifiable envelope after the first turn.
	const sentinel = "SPAWN_PROTOCOL_OK"
	hooks := ProtocolHooks{
		EncodeIdle: func(string, string) string { return sentinel },
	}

	res, err := Spawn(context.Background(), SpawnConfig{
		AgentName: "researcher",
		Registry:  reg,
		TaskRT:    rt,
		Execute:   exec.Execute,
		Protocol:  hooks,
	})
	if err != nil {
		t.Fatalf("Spawn: %v", err)
	}
	defer rt.Stop(res.TaskID)

	exec.waitForCalls(t, 1, time.Second)

	leaderBox := reg.Mailbox(TeamLeadName)
	deadline := time.After(time.Second)
	for leaderBox.Len() == 0 {
		select {
		case <-deadline:
			t.Fatal("leader mailbox empty: Spawn did not forward EncodeIdle hook to Run")
		case <-time.After(5 * time.Millisecond):
		}
	}
	msgs := leaderBox.Drain()
	if msgs[0].Text != sentinel {
		t.Errorf("got envelope %q, want sentinel %q — Spawn dropped Protocol on the floor", msgs[0].Text, sentinel)
	}
}

func TestSpawn_RejectsBeforeTeamCreate(t *testing.T) {
	reg := NewRegistry()
	rt := task.NewRuntime()
	exec := newStubExecutor()

	_, err := Spawn(context.Background(), SpawnConfig{
		AgentName: "researcher",
		Registry:  reg,
		TaskRT:    rt,
		Execute:   exec.Execute,
	})
	if !errors.Is(err, ErrNoTeam) {
		t.Errorf("Spawn without team = %v, want ErrNoTeam", err)
	}
}

func TestSpawn_RejectsReservedName(t *testing.T) {
	reg := NewRegistry()
	_ = reg.CreateTeam("alpha", "", "leader-1")

	_, err := Spawn(context.Background(), SpawnConfig{
		AgentName: TeamLeadName,
		Registry:  reg,
		TaskRT:    task.NewRuntime(),
		Execute:   newStubExecutor().Execute,
	})
	if !errors.Is(err, ErrReservedName) {
		t.Errorf("Spawn(team-lead) = %v, want ErrReservedName", err)
	}
}

func TestSpawn_RejectsDuplicateName(t *testing.T) {
	reg := NewRegistry()
	_ = reg.CreateTeam("alpha", "", "leader-1")
	rt := task.NewRuntime()
	exec := newStubExecutor()

	res1, err := Spawn(context.Background(), SpawnConfig{
		AgentName: "researcher",
		Registry:  reg,
		TaskRT:    rt,
		Execute:   exec.Execute,
	})
	if err != nil {
		t.Fatalf("first Spawn: %v", err)
	}

	_, err = Spawn(context.Background(), SpawnConfig{
		AgentName: "researcher",
		Registry:  reg,
		TaskRT:    rt,
		Execute:   exec.Execute,
	})
	if !errors.Is(err, ErrAgentExists) {
		t.Errorf("duplicate Spawn = %v, want ErrAgentExists", err)
	}

	// Clean up the first teammate
	rt.Stop(res1.TaskID)
}

// IdleClaim is the work-stealing hook. When set, the runner consults it at
// every turn boundary before blocking on the mailbox; a non-empty
// synthPrompt becomes the next turn's input directly, bypassing the
// mailbox altogether. These tests pin the contract so a future refactor
// can't silently break pull-mode delivery.

func TestRun_IdleClaim_FeedsSyntheticPromptWithoutMailbox(t *testing.T) {
	h := newHarness(t)
	h.exec.respond = func(idx int, _ []agentcore.AgentMessage) ([]agentcore.AgentMessage, error) {
		return []agentcore.AgentMessage{assistantMsg("done-" + strings.Repeat("x", idx+1))}, nil
	}

	hooks := makeTestHooks().hooks
	var claimCalls int32
	hooks.IdleClaim = func(ctx context.Context) (string, bool) {
		// Each invocation hands out one synthetic prompt, then nothing.
		// Without a fallback the runner would loop forever, so we let the
		// test stop it via cancel after observing N turns.
		n := atomic.AddInt32(&claimCalls, 1)
		if n <= 2 {
			return "CLAIMED-" + strconv.Itoa(int(n)), true
		}
		// Block subsequent calls so the runner parks on the mailbox and
		// we can shutdown cleanly via cancel.
		return "", false
	}

	h.startWithHooks(t, "kickoff", hooks)
	h.exec.waitForCalls(t, 3, time.Second)
	if err := h.stop(t, 2*time.Second); err != nil {
		t.Fatalf("Run exited with %v", err)
	}

	calls := h.exec.Calls()
	if len(calls) < 3 {
		t.Fatalf("got %d Execute calls, want >= 3", len(calls))
	}
	// Call 0 is the initial prompt (passed through FormatPrompt). Calls 1
	// and 2 are the synthetic claim prompts — they must show up verbatim,
	// NOT wrapped by FormatPrompt (the hook returns the final text).
	turn2User := lastUserText(calls[1])
	turn3User := lastUserText(calls[2])
	if turn2User != "CLAIMED-1" {
		t.Errorf("turn 2 user text = %q, want CLAIMED-1 verbatim", turn2User)
	}
	if turn3User != "CLAIMED-2" {
		t.Errorf("turn 3 user text = %q, want CLAIMED-2 verbatim", turn3User)
	}
}

func TestRun_IdleClaim_FalseFallsBackToMailbox(t *testing.T) {
	h := newHarness(t)
	h.exec.respond = func(idx int, _ []agentcore.AgentMessage) ([]agentcore.AgentMessage, error) {
		return []agentcore.AgentMessage{assistantMsg("ok")}, nil
	}

	hooks := makeTestHooks().hooks
	hooks.IdleClaim = func(ctx context.Context) (string, bool) {
		return "", false // never have work
	}
	h.startWithHooks(t, "kickoff", hooks)
	h.exec.waitForCalls(t, 1, time.Second)

	// Deliver via mailbox; runner should pick it up despite IdleClaim
	// being wired (because IdleClaim returned ok=false).
	if err := h.reg.Mailbox("researcher").Send(Message{From: TeamLeadName, Text: "from-mb"}); err != nil {
		t.Fatalf("Send: %v", err)
	}
	h.exec.waitForCalls(t, 2, time.Second)
	if err := h.stop(t, 2*time.Second); err != nil {
		t.Fatalf("Run exited with %v", err)
	}

	calls := h.exec.Calls()
	if got := lastUserText(calls[1]); !strings.Contains(got, "from-mb") {
		t.Errorf("turn 2 user text = %q, want it to carry the mailbox message", got)
	}
}

// Regression: when a push assignment lands in the mailbox right before the
// runner re-enters awaitNextInput, IdleClaim must NOT preempt it. Otherwise
// `task_update owner=X` (push mode) would race the auto-pull poll and the
// push payload could be silently deferred to a later turn.
func TestRun_IdleClaim_MailboxBeatsClaimWhenBothReady(t *testing.T) {
	h := newHarness(t)

	// Pre-fill the mailbox BEFORE Run starts so the very first
	// awaitNextInput call sees Len() > 0. The InitialPrompt absorbs the
	// first Execute, and the second Execute must consume the queued mb
	// message — NOT the IdleClaim pull.
	if err := h.reg.Mailbox("researcher").Send(Message{From: TeamLeadName, Text: "PUSH-WINS"}); err != nil {
		t.Fatalf("Send: %v", err)
	}
	hooks := makeTestHooks().hooks
	hooks.IdleClaim = func(ctx context.Context) (string, bool) {
		return "PULLED", true // always ready; should still lose to a queued mb message
	}

	h.exec.respond = func(int, []agentcore.AgentMessage) ([]agentcore.AgentMessage, error) {
		return []agentcore.AgentMessage{assistantMsg("ok")}, nil
	}
	h.startWithHooks(t, "kickoff", hooks)
	h.exec.waitForCalls(t, 2, time.Second)
	if err := h.stop(t, 2*time.Second); err != nil {
		t.Fatalf("Run exited with %v", err)
	}

	// Turn 2's input must come from the mailbox, not the always-ready
	// IdleClaim. Subsequent turns (after the mailbox drains) legitimately
	// fall through to IdleClaim — we only pin the priority on the conflict
	// turn itself.
	calls := h.exec.Calls()
	if got := lastUserText(calls[1]); !strings.Contains(got, "PUSH-WINS") {
		t.Fatalf("turn 2 user text = %q, want it to carry PUSH-WINS (mailbox-first)", got)
	}
}

func TestRun_IdleClaim_IntervalPollsWhileMailboxEmpty(t *testing.T) {
	h := newHarness(t)
	h.exec.respond = func(idx int, _ []agentcore.AgentMessage) ([]agentcore.AgentMessage, error) {
		return []agentcore.AgentMessage{assistantMsg("ok")}, nil
	}

	hooks := makeTestHooks().hooks
	var armed int32
	hooks.IdleClaim = func(ctx context.Context) (string, bool) {
		if atomic.LoadInt32(&armed) == 1 {
			return "DELAYED-WORK", true
		}
		return "", false
	}
	hooks.IdleClaimInterval = 20 * time.Millisecond

	h.startWithHooks(t, "kickoff", hooks)
	h.exec.waitForCalls(t, 1, time.Second)

	// Arm the hook AFTER the runner has parked. The next interval tick
	// should pull DELAYED-WORK without any mailbox traffic.
	atomic.StoreInt32(&armed, 1)
	h.exec.waitForCalls(t, 2, time.Second)
	if err := h.stop(t, 2*time.Second); err != nil {
		t.Fatalf("Run exited with %v", err)
	}

	calls := h.exec.Calls()
	if got := lastUserText(calls[1]); got != "DELAYED-WORK" {
		t.Errorf("turn 2 user text = %q, want DELAYED-WORK", got)
	}
}

// lastUserText extracts the text of the trailing user message from the slice
// the executor received, regardless of how many history entries precede it.
func lastUserText(msgs []agentcore.AgentMessage) string {
	if len(msgs) == 0 {
		return ""
	}
	last := msgs[len(msgs)-1]
	if last == nil || last.GetRole() != agentcore.RoleUser {
		return ""
	}
	return last.TextContent()
}

func TestSpawn_DepthRecordedOnEntry(t *testing.T) {
	reg := NewRegistry()
	_ = reg.CreateTeam("alpha", "", "leader-1")
	rt := task.NewRuntime()
	exec := newStubExecutor()

	res, err := Spawn(context.Background(), SpawnConfig{
		AgentName: "researcher",
		Registry:  reg,
		TaskRT:    rt,
		Execute:   exec.Execute,
		Depth:     3,
	})
	if err != nil {
		t.Fatalf("Spawn: %v", err)
	}
	entry := rt.Get(res.TaskID)
	if entry == nil || entry.Depth != 3 {
		t.Errorf("Entry.Depth = %d, want 3", entry.Depth)
	}
	rt.Stop(res.TaskID)
}

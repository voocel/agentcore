package team

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/voocel/agentcore"
	"github.com/voocel/agentcore/task"
)

// TurnExecutor runs one turn of the underlying agent loop and returns the
// messages produced by THAT turn only (not including the input). The runner
// stitches the produced messages onto its running history for the next turn.
//
// Implementations typically wrap agentcore.AgentLoop. Tests inject a stub.
// The executor must honour ctx cancellation — a cancelled context means the
// teammate's lifecycle has been aborted and the executor should return promptly.
type TurnExecutor func(ctx context.Context, msgs []agentcore.AgentMessage) ([]agentcore.AgentMessage, error)

// ProtocolHooks lets the application layer plug its own wire format and
// policy decisions into the runner without forking it. agentcore stays
// format-agnostic: any nil hook falls back to a permissive default (plain
// text passthrough, FIFO, no idle notification, never terminate). An
// application provides the bundle (e.g. its own envelope format + leader-
// first priority) by setting every field.
type ProtocolHooks struct {
	// FormatPrompt wraps a Message into the string the teammate's model
	// sees on the next turn. Run uses this hook for every prompt — including
	// the synthetic "leader's opening message" built from RunConfig at
	// startup, which is packaged as Message{From: TeamLeadName, Text:
	// InitialPrompt, Summary: Description}. Default: m.Text verbatim.
	FormatPrompt func(Message) string

	// EncodeIdle returns the envelope pushed to the leader's mailbox after
	// each turn. `lastText` is the teammate's last assistant text (may be
	// empty for tool-only turns). Returning "" suppresses the notification.
	// Default: return "" (no notification sent).
	EncodeIdle func(from, lastText string) string

	// ShouldTerminate decides whether a freshly picked queue message should
	// cause Run to exit gracefully before invoking the executor. Default:
	// always false (no control-message handling).
	ShouldTerminate func(text string) bool

	// PickPriority chooses which queued message to process next. Returns the
	// queue index. Default: 0 (FIFO).
	PickPriority func(queue []Message) int

	// IdleClaim is the work-stealing hook. Run consults it (a) at every
	// turn boundary before blocking on the mailbox and (b) every
	// IdleClaimInterval while parked on it. When the hook returns
	// ok=true, its synthPrompt is fed directly to the next turn — no
	// Message is allocated, no fake sender is recorded, and the queue
	// stays untouched.
	//
	// ctx carries the teammate identity via WithIdentity so applications
	// can branch by who is asking. Returning ok=false (or leaving this
	// nil) falls back to mailbox-only behavior. synthPrompt SHOULD be the
	// final text the model will see — the runner does NOT re-wrap it
	// through FormatPrompt, since there is no Message envelope to format.
	IdleClaim func(ctx context.Context) (synthPrompt string, ok bool)

	// IdleClaimInterval is the periodic re-check cadence used while a
	// teammate is parked on the mailbox. Zero (the default) means
	// IdleClaim is only consulted once per turn boundary; the teammate
	// then blocks until a real message arrives. Set this when the
	// application expects work to appear in the claim source without any
	// mailbox traffic to wake the teammate up.
	IdleClaimInterval time.Duration
}

// RunConfig configures one teammate's long-lived loop. All fields except Now
// and Protocol are required.
type RunConfig struct {
	// Identity gives the teammate its name + team membership. Threaded into
	// the agent's ctx via WithIdentity so tools can ask "who am I?".
	Identity *task.Identity

	// InitialPrompt is the leader's first message to the teammate. Run
	// packages it as Message{From: TeamLeadName, Text: InitialPrompt,
	// Summary: Description} and passes that through Protocol.FormatPrompt
	// — the same path as every subsequent inbound message.
	InitialPrompt string

	// History seeds the running conversation before the first turn. When
	// non-empty the first Execute receives History as the prefix of its
	// input (History + the InitialPrompt user message). nil ⇒ fresh
	// teammate. Used to resume a teammate with its prior transcript.
	History []agentcore.AgentMessage

	// Description is an optional short summary attached as Message.Summary
	// on the synthetic initial-prompt Message. Format hooks may surface it
	// (e.g. as an XML `summary=` attribute) or ignore it.
	Description string

	// Registry owns the mailbox and name registration for this teammate.
	Registry *Registry

	// TaskRT is the shared task runtime where this teammate's Entry lives.
	TaskRT *task.Runtime

	// TaskID identifies the Entry in TaskRT. The runner flips IsIdle on it
	// across turn boundaries so the UI / leader can see state changes.
	TaskID string

	// Execute drives one agent turn. Required.
	Execute TurnExecutor

	// Protocol is the application-supplied format + policy hook bundle. Any
	// nil field falls back to the agentcore default (see ProtocolHooks).
	Protocol ProtocolHooks

	// Now is the clock; tests inject a fake. Defaults to time.Now.
	Now func() time.Time
}

// Run drives the teammate's long-lived loop in the calling goroutine. It
// returns nil on graceful exit (ctx cancelled, mailbox closed, or shutdown
// approved). Returns a non-nil error only if the underlying executor failed
// in a non-cancellation way.
//
// Loop shape:
//
//  1. Mark Entry running + non-idle.
//  2. Execute one turn with: prior history + new user prompt.
//  3. Append produced messages to history.
//  4. Mark Entry idle + (optionally) forward Protocol.EncodeIdle output to
//     the leader's mailbox so the leader can react to the teammate's turn.
//  5. Wait on our mailbox for the next message (or ctx cancellation).
//  6. Drain mailbox into a local queue, pick the highest-priority message,
//     format it, and loop back to step 1 with it as the new prompt.
func Run(ctx context.Context, cfg RunConfig) error {
	if err := validateRunConfig(cfg); err != nil {
		return err
	}
	if cfg.Now == nil {
		cfg.Now = time.Now
	}
	hooks := withDefaults(cfg.Protocol)

	mailbox := cfg.Registry.Mailbox(cfg.Identity.AgentName)
	if mailbox == nil {
		return fmt.Errorf("team.Run: no mailbox for %q (not registered?)", cfg.Identity.AgentName)
	}
	// leaderBox may be nil if the team was torn down between Spawn and now —
	// we tolerate that; idle notifications become best-effort.
	leaderBox := cfg.Registry.Mailbox(TeamLeadName)

	identityCtx := WithIdentity(ctx, cfg.Identity)

	// Seed history with any resumed transcript; a fresh teammate gets nil.
	// Copied so the caller's slice isn't aliased as we append per turn.
	history := append([]agentcore.AgentMessage(nil), cfg.History...)
	currentPrompt := hooks.FormatPrompt(Message{
		From:    TeamLeadName,
		Text:    cfg.InitialPrompt,
		Summary: cfg.Description,
	})

	// localQueue holds messages drained from the mailbox but not yet processed.
	// We pick one per turn in priority order; leftover messages stay here until
	// the next turn so we don't keep re-draining the shared queue.
	var localQueue []Message

	for {
		if err := ctx.Err(); err != nil {
			return nil // graceful exit on cancel
		}

		// 1. Run one turn
		cfg.TaskRT.Update(cfg.TaskID, func(e *task.Entry) {
			e.IsIdle = false
			e.Status = task.Running
		})

		userMsg := agentcore.UserMsg(currentPrompt)
		turnInput := make([]agentcore.AgentMessage, 0, len(history)+1)
		turnInput = append(turnInput, history...)
		turnInput = append(turnInput, userMsg)

		produced, err := cfg.Execute(identityCtx, turnInput)
		if err != nil && !errors.Is(err, context.Canceled) {
			return fmt.Errorf("teammate %q turn failed: %w", cfg.Identity.AgentName, err)
		}

		// Grow history regardless of partial failure — the model already saw
		// these messages; pretending they didn't happen would corrupt the
		// next turn's context.
		history = append(history, userMsg)
		history = append(history, produced...)

		// 2. Mark idle + notify leader
		// Compute the turn's assistant text once and reuse it for both the
		// Entry update (so the UI can show what the teammate last said) and
		// the idle envelope (so the leader's model can pick it up).
		lastText := lastAssistantText(produced)
		cfg.TaskRT.Update(cfg.TaskID, func(e *task.Entry) {
			e.IsIdle = true
			if lastText != "" {
				e.Result = lastText
			}
		})
		if leaderBox != nil {
			if envelope := hooks.EncodeIdle(cfg.Identity.AgentName, lastText); envelope != "" {
				// Best-effort: ErrClosed here means the team was torn down
				// between our turn and idle notification — ignore.
				_ = leaderBox.Send(Message{
					From:      cfg.Identity.AgentName,
					Text:      envelope,
					Color:     cfg.Identity.Color,
					Timestamp: cfg.Now(),
				})
			}
		}

		// 3. Get next prompt. When the local queue is empty we either wait
		// on the mailbox or, if IdleClaim is wired, attempt to pull a
		// synthetic prompt from the application (work-stealing). A
		// successful pull short-circuits the mailbox path entirely — no
		// Message is fabricated, the queue stays untouched, and the
		// synthPrompt becomes the next turn's input as-is.
		if len(localQueue) == 0 {
			synth, err := awaitNextInput(identityCtx, mailbox, hooks)
			if err != nil {
				if errors.Is(err, ErrClosed) || errors.Is(err, context.Canceled) {
					return nil
				}
				return err
			}
			if synth != "" {
				currentPrompt = synth
				continue
			}
			localQueue = mailbox.Drain()
			if len(localQueue) == 0 {
				continue // spurious wake or coalesced with close
			}
		}

		idx := hooks.PickPriority(localQueue)
		chosen := localQueue[idx]
		localQueue = append(localQueue[:idx], localQueue[idx+1:]...)

		// Graceful shutdown: application-supplied predicate decided this
		// message should end the loop before consulting the model. Default
		// is "never terminate", so the loop only exits this way when the
		// app explicitly wires up a control-message kind.
		if hooks.ShouldTerminate(chosen.Text) {
			return nil
		}

		currentPrompt = hooks.FormatPrompt(chosen)
	}
}

// awaitNextInput is the central wait helper for the runner's main loop. It
// returns one of three outcomes:
//
//  1. (synthPrompt, nil) — IdleClaim pulled a task; caller uses the prompt
//     directly and skips Drain.
//  2. ("",          nil) — A real message arrived in the mailbox; caller
//     should Drain + PickPriority.
//  3. ("",          err) — ctx cancelled or mailbox closed; caller returns.
//
// When IdleClaim is unwired (or returns ok=false) and IdleClaimInterval is
// zero, this collapses to plain mailbox.Wait — the pre-IdleClaim behavior,
// byte-for-byte. Periodic IdleClaim re-checks during the wait are only
// armed when both IdleClaim and IdleClaimInterval are set.
func awaitNextInput(ctx context.Context, mailbox *Mailbox, hooks ProtocolHooks) (string, error) {
	// Mailbox-first priority: a queued message (leader push, shutdown
	// request, peer DM) MUST be processed before the teammate pulls its
	// own next task via IdleClaim. Otherwise a push assignment racing the
	// IdleClaim poll would be silently deferred to the next turn.
	if mailbox.Len() > 0 {
		return "", nil
	}
	tryClaim := func() (string, bool) {
		if hooks.IdleClaim == nil {
			return "", false
		}
		return hooks.IdleClaim(ctx)
	}
	if prompt, ok := tryClaim(); ok && prompt != "" {
		return prompt, nil
	}
	if hooks.IdleClaim == nil || hooks.IdleClaimInterval <= 0 {
		return "", mailbox.Wait(ctx)
	}
	for {
		err := mailbox.WaitFor(ctx, hooks.IdleClaimInterval)
		switch {
		case err == nil:
			return "", nil // mailbox has something
		case errors.Is(err, ErrTimeout):
			if prompt, ok := tryClaim(); ok && prompt != "" {
				return prompt, nil
			}
			// no work yet, keep polling
		default:
			return "", err
		}
	}
}

// withDefaults fills any nil hook with a permissive default so Run never has
// to nil-check inside the hot loop. Callers can pass a zero ProtocolHooks and
// still get a working — if minimal — peer-DM loop.
func withDefaults(h ProtocolHooks) ProtocolHooks {
	if h.FormatPrompt == nil {
		h.FormatPrompt = func(m Message) string { return m.Text }
	}
	if h.EncodeIdle == nil {
		h.EncodeIdle = func(string, string) string { return "" }
	}
	if h.ShouldTerminate == nil {
		h.ShouldTerminate = func(string) bool { return false }
	}
	if h.PickPriority == nil {
		h.PickPriority = func([]Message) int { return 0 }
	}
	return h
}

func validateRunConfig(cfg RunConfig) error {
	switch {
	case cfg.Identity == nil:
		return errors.New("team.Run: Identity is required")
	case cfg.Registry == nil:
		return errors.New("team.Run: Registry is required")
	case cfg.TaskRT == nil:
		return errors.New("team.Run: TaskRT is required")
	case cfg.TaskID == "":
		return errors.New("team.Run: TaskID is required")
	case cfg.Execute == nil:
		return errors.New("team.Run: Execute is required")
	}
	return nil
}

// lastAssistantText returns the text content of the last assistant message in
// msgs, or "" if msgs has no assistant message. The runner feeds this to
// ProtocolHooks.EncodeIdle so the application layer can decide whether/how to
// forward the teammate's last words across the agent boundary.
func lastAssistantText(msgs []agentcore.AgentMessage) string {
	for i := len(msgs) - 1; i >= 0; i-- {
		m := msgs[i]
		if m == nil || m.GetRole() != agentcore.RoleAssistant {
			continue
		}
		if t := m.TextContent(); t != "" {
			return t
		}
	}
	return ""
}

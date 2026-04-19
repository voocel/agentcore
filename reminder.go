package agentcore

import "context"

// Reminder is a one-turn system message injected before the LLM call.
// Reminders live for exactly one turn — they are NOT persisted to the agent's
// message history and do not participate in context compaction.
//
// The typical use case is "every-turn steering" — facts that the host needs
// to re-affirm on every call (current phase, pending work, stop conditions)
// without polluting the durable conversation.
type Reminder struct {
	// Source is a logical identifier for the generator emitting this reminder.
	// When multiple reminders from the same turn share a Source, the last one
	// wins; earlier duplicates are dropped.
	Source string
	// Content is the reminder body. It will be wrapped in
	// `<system-reminder>...</system-reminder>` before injection.
	Content string
}

// TurnInfo carries per-turn state handed to reminder generators.
type TurnInfo struct {
	// TurnIndex is 0 for the first LLM call in this run, 1 for the second, etc.
	TurnIndex int
}

// ReminderGenerator produces reminders for the upcoming LLM call.
// It is invoked once per turn, just before the LLM request is built.
// Returning nil or an empty slice skips injection for that turn.
type ReminderGenerator func(ctx context.Context, turn TurnInfo) []Reminder

// collectReminders runs each generator in order and returns a de-duplicated
// list of reminders (last write wins per Source). Unnamed sources are kept
// as-is. Invoked from callLLM; never invoked by application code directly.
func collectReminders(ctx context.Context, gens []ReminderGenerator, turn TurnInfo) []Reminder {
	if len(gens) == 0 {
		return nil
	}
	var out []Reminder
	seen := make(map[string]int) // Source → index in out
	for _, gen := range gens {
		if gen == nil {
			continue
		}
		for _, r := range gen(ctx, turn) {
			if r.Content == "" {
				continue
			}
			if r.Source != "" {
				if idx, ok := seen[r.Source]; ok {
					out[idx] = r
					continue
				}
				seen[r.Source] = len(out)
			}
			out = append(out, r)
		}
	}
	return out
}

// reminderSystemMessages converts reminders into system Messages wrapped with
// `<system-reminder>` tags. Each reminder becomes its own system message so
// providers that de-duplicate identical system blocks can still distinguish
// reminder sources.
func reminderSystemMessages(reminders []Reminder) []Message {
	if len(reminders) == 0 {
		return nil
	}
	msgs := make([]Message, 0, len(reminders))
	for _, r := range reminders {
		msgs = append(msgs, SystemMsg("<system-reminder>\n"+r.Content+"\n</system-reminder>"))
	}
	return msgs
}

// StopGuard is consulted when the LLM would end a run without tool calls
// (i.e. the assistant produced a final text response and no more tool calls).
//
// Return Allow=true to let the agent stop normally. Return Allow=false with
// an InjectMessage to keep the agent running for another turn — the message
// is delivered as a user message on the next LLM call. Set Escalate=true to
// force the run to end with a guard-escalation error (used when the guard
// has repeatedly blocked stops and suspects a prompt bug).
//
// Guard state (e.g. consecutive-block counters) is the guard's own
// responsibility; agentcore passes only the current turn index and the
// stopping assistant message.
type StopGuard func(ctx context.Context, stop StopInfo) StopDecision

// StopInfo carries the information a StopGuard needs to decide.
type StopInfo struct {
	// TurnIndex is the index of the turn that just produced the stopping message.
	TurnIndex int
	// Message is the assistant message whose StopReason triggered this check.
	Message Message
}

// StopDecision is the guard's verdict.
type StopDecision struct {
	// Allow=true lets the stop proceed; Allow=false keeps the loop alive.
	Allow bool
	// InjectMessage is delivered as a user message on the next turn when
	// Allow=false && !Escalate. Empty InjectMessage with Allow=false is
	// treated as Allow=true (safe default — never stall silently).
	InjectMessage string
	// Escalate=true ends the run immediately with an error.
	Escalate bool
}

package agentcore

import "context"

// InjectDisposition describes how an injected message was delivered.
type InjectDisposition string

const (
	InjectSteeredCurrentRun InjectDisposition = "steered_current_run"
	InjectResumedIdleRun    InjectDisposition = "resumed_idle_run"
	InjectQueued            InjectDisposition = "queued"
)

// InjectResult reports the delivery outcome of Agent.Inject.
type InjectResult struct {
	Disposition InjectDisposition
}

// Inject delivers a message as soon as the current agent state allows, resuming
// an idle run on the background context. Prefer InjectContext when the resumed
// run should carry caller context (e.g. a working-directory override).
func (a *Agent) Inject(msg AgentMessage) (InjectResult, error) {
	return a.InjectContext(context.Background(), msg)
}

// InjectContext is Inject with an explicit context that an idle resume runs
// under, so values threaded onto ctx (cwd override, deadlines) reach the
// resumed run's tools just as they would on PromptMessages/Continue.
//
// Outcomes:
//   - runs held (HoldRuns) → ErrRunsHeld, nothing queued
//   - running → steer into current run (ctx unused; the live run keeps its own)
//   - idle + assistant tail → enqueue and resume, atomically
//   - idle + no assistant tail → enqueue for next run
func (a *Agent) InjectContext(ctx context.Context, msg AgentMessage) (InjectResult, error) {
	if msg == nil {
		return InjectResult{}, ErrInjectNilMessage
	}

	a.mu.Lock()
	// Fail fast BEFORE any queueing: a held agent accepts no inject work.
	// Enqueue-then-fail would leave the message queued while the caller also
	// reroutes it — a double delivery once runs resume. During a hold's
	// wind-down isRunning is still true, so this must precede the steer branch.
	if a.held > 0 {
		a.mu.Unlock()
		return InjectResult{}, ErrRunsHeld
	}
	if a.isRunning {
		a.steeringQ = append(a.steeringQ, msg)
		a.mu.Unlock()
		return InjectResult{Disposition: InjectSteeredCurrentRun}, nil
	}

	canResume := false
	if n := len(a.messages); n > 0 && a.messages[n-1] != nil {
		canResume = a.messages[n-1].GetRole() == RoleAssistant
	}
	a.steeringQ = append(a.steeringQ, msg)
	// Resume in the same critical section as the enqueue: unlocking in between
	// opens two TOCTOU windows — a hold lands and the resume fails ErrRunsHeld
	// with the message still queued, or a concurrent Prompt's run consumes the
	// message via its initial steering poll while the resume reports failure.
	// Either way the caller reroutes a message that was in fact delivered.
	if canResume && a.resumeQueuedLocked(ctx) {
		return InjectResult{Disposition: InjectResumedIdleRun}, nil
	}
	a.mu.Unlock()
	return InjectResult{Disposition: InjectQueued}, nil
}

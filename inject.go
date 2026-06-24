package agentcore

import (
	"context"
	"fmt"
)

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
//   - running → steer into current run (ctx unused; the live run keeps its own)
//   - idle + assistant tail → enqueue and Continue(ctx)
//   - idle + no assistant tail → enqueue for next run
//
// Returns an error if idle resume was attempted but Continue() failed.
// In that case the message remains in the steering queue and will be
// delivered on the next run.
func (a *Agent) InjectContext(ctx context.Context, msg AgentMessage) (InjectResult, error) {
	if msg == nil {
		return InjectResult{}, ErrInjectNilMessage
	}

	a.mu.Lock()
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
	a.mu.Unlock()

	if !canResume {
		return InjectResult{Disposition: InjectQueued}, nil
	}
	if err := a.Continue(ctx); err != nil {
		return InjectResult{Disposition: InjectQueued}, fmt.Errorf("inject idle resume failed: %w", err)
	}
	return InjectResult{Disposition: InjectResumedIdleRun}, nil
}

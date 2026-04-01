package agentcore

import "fmt"

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

// Inject delivers a message as soon as the current agent state allows.
//
// Outcomes:
//   - running → steer into current run
//   - idle + assistant tail → enqueue and Continue()
//   - idle + no assistant tail → enqueue for next run
//
// Returns an error if idle resume was attempted but Continue() failed.
// In that case the message remains in the steering queue and will be
// delivered on the next run.
func (a *Agent) Inject(msg AgentMessage) (InjectResult, error) {
	if msg == nil {
		return InjectResult{}, fmt.Errorf("inject message is nil")
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
	if err := a.Continue(); err != nil {
		return InjectResult{Disposition: InjectQueued}, fmt.Errorf("inject: idle resume failed: %w", err)
	}
	return InjectResult{Disposition: InjectResumedIdleRun}, nil
}

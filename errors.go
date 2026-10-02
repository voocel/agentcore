package agentcore

import "errors"

// ErrMaxTurns ends a run that reached its MaxTurns.
var ErrMaxTurns = errors.New("max turns reached")

// ErrBusy refuses to start a run while an Agent runs one.
var ErrBusy = errors.New("agent is running")

// ErrNothingToContinue refuses a run without prompts when the model has
// nothing to answer, as when the history ends with its response.
var ErrNothingToContinue = errors.New("nothing to continue: the history ends with a response")

package team

import (
	"context"

	"github.com/voocel/agentcore/task"
)

// identityKey threads a teammate's identity into the context so any tool call
// running inside the teammate's goroutine can ask "who am I?" without passing
// the identity through every signature.
type identityKey struct{}

// WithIdentity returns a derived context carrying id. Called by the teammate
// runner before invoking the agent loop. nil id panics — pass nil for a
// non-teammate context (or just use the parent ctx unchanged).
func WithIdentity(parent context.Context, id *task.Identity) context.Context {
	if id == nil {
		panic("team.WithIdentity: id must not be nil")
	}
	return context.WithValue(parent, identityKey{}, id)
}

// IdentityFromContext returns the teammate identity attached to ctx, or nil
// if the caller is not running inside a teammate. Tools that vary behaviour
// by identity (e.g. send_message routing) consult this.
func IdentityFromContext(ctx context.Context) *task.Identity {
	if v, ok := ctx.Value(identityKey{}).(*task.Identity); ok {
		return v
	}
	return nil
}

// IsTeammate reports whether ctx is inside a teammate goroutine.
func IsTeammate(ctx context.Context) bool {
	return IdentityFromContext(ctx) != nil
}

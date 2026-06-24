package tools

import "context"

// cwdOverrideKey threads a per-call working directory into the context so the
// cwd-bound tools (read/write/edit/glob/grep/ls/bash) can resolve paths against
// it without being rebuilt. This makes cwd a runtime ambient value rather than
// a construction-time constant: a single tool instance can serve the main repo
// on one call and a git-worktree sandbox on the next, chosen by the caller.
type cwdOverrideKey struct{}

// cwdFuncKey threads a LIVE cwd source, consulted at resolution time so a caller
// whose cwd changes mid-run (e.g. entering a worktree) is seen by the next tool
// call without deriving a fresh context.
type cwdFuncKey struct{}

// WithCwd returns a derived context carrying cwd as the working-directory
// override for cwd-bound tools. An empty cwd is a no-op (returns parent), so
// callers can pass through whatever they have without branching.
func WithCwd(parent context.Context, cwd string) context.Context {
	if cwd == "" {
		return parent
	}
	return context.WithValue(parent, cwdOverrideKey{}, cwd)
}

// WithCwdFunc carries fn as a live cwd source, invoked on every resolution so a
// cwd that changes after the context is derived takes effect immediately. A nil
// fn is a no-op. A non-empty fn result wins over a static WithCwd value; an
// empty one falls back to it.
func WithCwdFunc(parent context.Context, fn func() string) context.Context {
	if fn == nil {
		return parent
	}
	return context.WithValue(parent, cwdFuncKey{}, fn)
}

// CwdFromContext returns the working-directory override attached to ctx, or ""
// if none is set (the common case, where tools fall back to their constructed
// WorkDir). A live WithCwdFunc source is consulted first, then a static WithCwd.
func CwdFromContext(ctx context.Context) string {
	if ctx == nil {
		return ""
	}
	if fn, ok := ctx.Value(cwdFuncKey{}).(func() string); ok && fn != nil {
		if cwd := fn(); cwd != "" {
			return cwd
		}
	}
	cwd, _ := ctx.Value(cwdOverrideKey{}).(string)
	return cwd
}

// effectiveWorkDir returns the per-call cwd override when present, otherwise the
// tool's construction-time fallback. With no override it returns fallback
// verbatim, so every existing caller's behaviour is unchanged.
func effectiveWorkDir(ctx context.Context, fallback string) string {
	if cwd := CwdFromContext(ctx); cwd != "" {
		return cwd
	}
	return fallback
}

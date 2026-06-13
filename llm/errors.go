package llm

import (
	"context"
	"errors"
	"time"

	"github.com/voocel/agentcore"
	"github.com/voocel/litellm"
)

// providerError adapts a litellm error to the agentcore kernel's
// provider-agnostic error contract. It lets the kernel read retry and
// classification facts (agentcore.RetryableError / RetryHinter, and the
// ErrContextOverflow / ErrProviderStreamIdle sentinels via errors.Is) without
// importing litellm. Unwrap exposes the original error, so callers that DO
// know litellm can still match it with errors.As(&litellm.LiteLLMError{}).
type providerError struct{ err error }

// wrapProviderError wraps a raw litellm error for kernel consumption. Control
// signals (context cancellation / deadline) are returned unchanged: the loop
// matches them directly with errors.Is, and classifying them as provider
// failures would wrongly flip their retryable bit.
func wrapProviderError(err error) error {
	if err == nil {
		return nil
	}
	if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return err
	}
	return &providerError{err: err}
}

func (e *providerError) Error() string { return e.err.Error() }
func (e *providerError) Unwrap() error { return e.err }

// Retryable reports litellm's own retryability verdict (network, timeout,
// rate-limit, overloaded, and upstream provider errors are retryable).
func (e *providerError) Retryable() bool { return litellm.IsRetryableError(e.err) }

// RetryAfter surfaces a provider Retry-After hint (rate-limit responses) as a
// duration, or 0 when none is present.
func (e *providerError) RetryAfter() time.Duration {
	if s := litellm.GetRetryAfter(e.err); s > 0 {
		return time.Duration(s) * time.Second
	}
	return 0
}

// Is maps litellm's category checks onto the kernel's sentinels so the loop
// can detect overflow and stream-idle with a plain errors.Is. Other targets
// fall through to the unwrapped litellm error's own matching.
func (e *providerError) Is(target error) bool {
	switch target {
	case agentcore.ErrContextOverflow:
		return litellm.IsContextOverflowError(e.err)
	case agentcore.ErrProviderStreamIdle:
		return litellm.IsStreamIdleError(e.err)
	}
	return false
}

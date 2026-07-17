package llm

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/voocel/agentcore"
	"github.com/voocel/litellm"
)

// providerError adapts a litellm error to the agentcore kernel's
// provider-agnostic error contract. It lets the kernel read retry and
// classification facts (agentcore.RetryableError / RetryHinter, and the
// ErrContextOverflow / ErrProvider* sentinels via errors.Is) without importing
// litellm. Unwrap exposes the original error, so callers that DO know litellm
// can still match it with errors.As(&litellm.LiteLLMError{}).
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

// Error appends the structured facts litellm carries but keeps out of its own
// Error() text (error type, HTTP status, provider, model). Gateway messages
// are often as vague as "Provider returned error"; without these facts users
// cannot tell a config mistake from an upstream outage. This is the single
// seam every provider error string flows through (engine retry events, host
// surfaces, logs), so enriching here fixes them all at once.
func (e *providerError) Error() string {
	msg := e.err.Error()
	var le *litellm.LiteLLMError
	if !errors.As(e.err, &le) {
		return msg
	}
	facts := make([]string, 0, 4)
	if le.Type != "" {
		facts = append(facts, string(le.Type))
	}
	if le.StatusCode != 0 {
		facts = append(facts, fmt.Sprintf("HTTP %d", le.StatusCode))
	}
	if le.Provider != "" {
		facts = append(facts, le.Provider)
	}
	if le.Model != "" {
		facts = append(facts, le.Model)
	}
	if len(facts) == 0 {
		return msg
	}
	return msg + " [" + strings.Join(facts, ", ") + "]"
}

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

// Is maps litellm's category checks onto the kernel's sentinels so callers can
// classify provider failures with a plain errors.Is. Other targets fall through
// to the unwrapped litellm error's own matching.
func (e *providerError) Is(target error) bool {
	switch target {
	case agentcore.ErrContextOverflow:
		return litellm.IsContextOverflowError(e.err)
	case agentcore.ErrProviderRateLimit:
		return litellm.IsRateLimitError(e.err)
	case agentcore.ErrProviderQuota:
		return isQuotaError(e.err)
	case agentcore.ErrProviderTimeout:
		return litellm.IsTimeoutError(e.err)
	case agentcore.ErrProviderStreamIdle:
		return litellm.IsStreamIdleError(e.err)
	case agentcore.ErrProviderNetwork:
		return litellm.IsNetworkError(e.err)
	case agentcore.ErrProviderAuth:
		return litellm.IsAuthError(e.err)
	case agentcore.ErrProviderOverloaded:
		return litellm.IsOverloadedError(e.err)
	case agentcore.ErrProviderContentFilter:
		return litellm.IsContentFilterError(e.err)
	}
	return false
}

func isQuotaError(err error) bool {
	var e *litellm.LiteLLMError
	return errors.As(err, &e) && e.Type == litellm.ErrorTypeQuota
}

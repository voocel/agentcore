package llm

import (
	"slices"

	"github.com/voocel/litellm"
)

// CapabilityProvider is implemented by models that expose their provider
// adapter's static facts for UI preflight and configuration validation; ok is
// false when the facts are unknown. It is advisory: request execution remains
// the source of truth.
type CapabilityProvider interface {
	Capabilities() (caps Capabilities, ok bool)
}

// Capabilities states what the provider adapter can express on the wire,
// independent of the model; whether a model honors a request is the vendor's
// call.
type Capabilities struct {
	Thinking        bool
	DisableThinking bool
	ThinkingEffort  bool
	ThinkingBudget  bool
	ProviderOptions []string
}

func fromLiteLLMCapabilities(c litellm.Capabilities) Capabilities {
	return Capabilities{
		Thinking:        c.Thinking,
		DisableThinking: c.DisableThinking,
		ThinkingEffort:  c.ThinkingEffort,
		ThinkingBudget:  c.ThinkingBudget,
		ProviderOptions: slices.Clone(c.ProviderOptions),
	}
}

func (c Capabilities) ThinkingPolicy() ThinkingPolicy {
	return ThinkingPolicyFromCapabilities(c)
}

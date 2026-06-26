package llm

import (
	"github.com/voocel/agentcore"
	"github.com/voocel/litellm"
)

// Support describes whether a model/provider supports a capability.
type Support int

const (
	SupportUnknown Support = iota
	SupportNo
	SupportYes
	SupportPartial
)

// CapabilityProvider is implemented by models that can expose provider/model
// capabilities for UI preflight and configuration validation. It is advisory:
// request execution remains the source of truth and should still fail loudly.
type CapabilityProvider interface {
	Capabilities() Capabilities
}

// Capabilities is agentcore's provider-neutral view of model capabilities.
type Capabilities struct {
	Provider string
	Model    string

	Thinking   ThinkingCapabilities
	Tools      ToolCapabilities
	Structured StructuredCapabilities
	Streaming  StreamingCapabilities
	Usage      UsageCapabilities
}

type ThinkingCapabilities struct {
	Supported     Support
	Disable       Support
	Efforts       []agentcore.ThinkingLevel
	BudgetTokens  Support
	IncludeOutput Support
	Notes         []string
}

func (c ThinkingCapabilities) SupportsEffort(level agentcore.ThinkingLevel) bool {
	for _, effort := range c.Efforts {
		if effort == level {
			return true
		}
	}
	return false
}

type ToolCapabilities struct {
	Calls               Support
	ParallelCalls       Support
	StrictSchema        Support
	Choice              Support
	MultimodalResults   Support
	RequiresAdjacency   bool
	RoundTripSignatures Support
	HostedProviderTools Support
}

type StructuredCapabilities struct {
	JSONObject Support
	JSONSchema Support
	Strict     Support
	PromptOnly bool
}

type StreamingCapabilities struct {
	Supported       Support
	Usage           Support
	ReasoningDeltas Support
	ToolCallDeltas  Support
	NativeResponses Support
	IdleTimeout     Support
}

type UsageCapabilities struct {
	InputTokens      Support
	OutputTokens     Support
	TotalTokens      Support
	ReasoningTokens  Support
	CacheReadTokens  Support
	CacheWriteTokens Support
}

func fromLiteLLMCapabilities(c litellm.Capabilities) Capabilities {
	return Capabilities{
		Provider: c.Provider,
		Model:    c.Model,
		Thinking: ThinkingCapabilities{
			Supported:     fromLiteLLMSupport(c.Thinking.Supported),
			Disable:       fromLiteLLMSupport(c.Thinking.Disable),
			Efforts:       fromLiteLLMThinkingEfforts(c.Thinking.Efforts),
			BudgetTokens:  fromLiteLLMSupport(c.Thinking.BudgetTokens),
			IncludeOutput: fromLiteLLMSupport(c.Thinking.IncludeOutput),
			Notes:         append([]string(nil), c.Thinking.Notes...),
		},
		Tools: ToolCapabilities{
			Calls:               fromLiteLLMSupport(c.Tools.Calls),
			ParallelCalls:       fromLiteLLMSupport(c.Tools.ParallelCalls),
			StrictSchema:        fromLiteLLMSupport(c.Tools.StrictSchema),
			Choice:              fromLiteLLMSupport(c.Tools.Choice),
			MultimodalResults:   fromLiteLLMSupport(c.Tools.MultimodalResults),
			RequiresAdjacency:   c.Tools.RequiresAdjacency,
			RoundTripSignatures: fromLiteLLMSupport(c.Tools.RoundTripSignatures),
			HostedProviderTools: fromLiteLLMSupport(c.Tools.HostedProviderTools),
		},
		Structured: StructuredCapabilities{
			JSONObject: fromLiteLLMSupport(c.Structured.JSONObject),
			JSONSchema: fromLiteLLMSupport(c.Structured.JSONSchema),
			Strict:     fromLiteLLMSupport(c.Structured.Strict),
			PromptOnly: c.Structured.PromptOnly,
		},
		Streaming: StreamingCapabilities{
			Supported:       fromLiteLLMSupport(c.Streaming.Supported),
			Usage:           fromLiteLLMSupport(c.Streaming.Usage),
			ReasoningDeltas: fromLiteLLMSupport(c.Streaming.ReasoningDeltas),
			ToolCallDeltas:  fromLiteLLMSupport(c.Streaming.ToolCallDeltas),
			NativeResponses: fromLiteLLMSupport(c.Streaming.NativeResponses),
			IdleTimeout:     fromLiteLLMSupport(c.Streaming.IdleTimeout),
		},
		Usage: UsageCapabilities{
			InputTokens:      fromLiteLLMSupport(c.Usage.InputTokens),
			OutputTokens:     fromLiteLLMSupport(c.Usage.OutputTokens),
			TotalTokens:      fromLiteLLMSupport(c.Usage.TotalTokens),
			ReasoningTokens:  fromLiteLLMSupport(c.Usage.ReasoningTokens),
			CacheReadTokens:  fromLiteLLMSupport(c.Usage.CacheReadTokens),
			CacheWriteTokens: fromLiteLLMSupport(c.Usage.CacheWriteTokens),
		},
	}
}

func fromLiteLLMSupport(s litellm.Support) Support {
	switch s {
	case litellm.SupportNo:
		return SupportNo
	case litellm.SupportYes:
		return SupportYes
	case litellm.SupportPartial:
		return SupportPartial
	default:
		return SupportUnknown
	}
}

func fromLiteLLMThinkingEfforts(efforts []string) []agentcore.ThinkingLevel {
	if len(efforts) == 0 {
		return nil
	}
	out := make([]agentcore.ThinkingLevel, 0, len(efforts))
	for _, effort := range efforts {
		switch level := agentcore.ThinkingLevel(effort); level {
		case agentcore.ThinkingMinimal,
			agentcore.ThinkingLow,
			agentcore.ThinkingMedium,
			agentcore.ThinkingHigh,
			agentcore.ThinkingXHigh,
			agentcore.ThinkingMax:
			out = append(out, level)
		}
	}
	return out
}

func (c Capabilities) ThinkingPolicy() ThinkingPolicy {
	return ThinkingPolicyFromCapabilities(c)
}

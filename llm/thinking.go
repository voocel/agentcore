package llm

import "github.com/voocel/agentcore"

const ThinkingAuto agentcore.ThinkingLevel = ""

var ThinkingLevelOrder = []agentcore.ThinkingLevel{
	agentcore.ThinkingOff,
	agentcore.ThinkingMinimal,
	agentcore.ThinkingLow,
	agentcore.ThinkingMedium,
	agentcore.ThinkingHigh,
	agentcore.ThinkingXHigh,
	agentcore.ThinkingMax,
}

type ThinkingPolicy struct {
	Available []agentcore.ThinkingLevel
}

func ThinkingPolicyFor(model any) ThinkingPolicy {
	if cp, ok := model.(CapabilityProvider); ok {
		if caps, ok := cp.Capabilities(); ok {
			return ThinkingPolicyFromCapabilities(caps)
		}
	}
	// Unknown facts restrict nothing; the provider decides.
	return ThinkingPolicy{Available: append([]agentcore.ThinkingLevel{ThinkingAuto}, ThinkingLevelOrder...)}
}

// ThinkingPolicyFromCapabilities offers auto always, off when thinking can be
// disabled and the effort levels when effort can be sent.
func ThinkingPolicyFromCapabilities(caps Capabilities) ThinkingPolicy {
	available := []agentcore.ThinkingLevel{ThinkingAuto}
	if !caps.Thinking {
		return ThinkingPolicy{Available: available}
	}
	for _, level := range ThinkingLevelOrder {
		if level == agentcore.ThinkingOff && caps.DisableThinking || level != agentcore.ThinkingOff && caps.ThinkingEffort {
			available = append(available, level)
		}
	}
	return ThinkingPolicy{Available: available}
}

func (p ThinkingPolicy) Allows(level agentcore.ThinkingLevel) bool {
	for _, available := range p.Available {
		if available == level {
			return true
		}
	}
	return false
}

func (p ThinkingPolicy) Resolve(level agentcore.ThinkingLevel) (agentcore.ThinkingLevel, bool) {
	level = agentcore.NormalizeThinkingLevel(level)
	if p.Allows(level) {
		return level, true
	}
	return ThinkingAuto, false
}

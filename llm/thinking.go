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
	cp, ok := model.(CapabilityProvider)
	if !ok {
		return ThinkingPolicy{Available: append([]agentcore.ThinkingLevel{ThinkingAuto}, ThinkingLevelOrder...)}
	}
	return ThinkingPolicyFromCapabilities(cp.Capabilities())
}

func ThinkingPolicyFromCapabilities(caps Capabilities) ThinkingPolicy {
	if !caps.ProviderBaseline {
		return exactThinkingPolicy(caps.Thinking)
	}
	return baselineThinkingPolicy(caps.Thinking)
}

func baselineThinkingPolicy(caps ThinkingCapabilities) ThinkingPolicy {
	if caps.Supported == SupportNo {
		return ThinkingPolicy{Available: []agentcore.ThinkingLevel{ThinkingAuto}}
	}
	available := []agentcore.ThinkingLevel{ThinkingAuto}
	for _, level := range ThinkingLevelOrder {
		if level == agentcore.ThinkingOff && caps.Disable == SupportNo {
			continue
		}
		available = append(available, level)
	}
	return ThinkingPolicy{Available: available}
}

func exactThinkingPolicy(caps ThinkingCapabilities) ThinkingPolicy {
	if caps.Supported == SupportUnknown && caps.Disable == SupportUnknown && len(caps.Efforts) == 0 {
		return ThinkingPolicy{Available: append([]agentcore.ThinkingLevel{ThinkingAuto}, ThinkingLevelOrder...)}
	}
	available := []agentcore.ThinkingLevel{ThinkingAuto}
	if caps.Disable == SupportYes || caps.Disable == SupportPartial {
		available = append(available, agentcore.ThinkingOff)
	}
	available = append(available, caps.Efforts...)
	return ThinkingPolicy{Available: uniqueThinkingLevels(available)}
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

func uniqueThinkingLevels(levels []agentcore.ThinkingLevel) []agentcore.ThinkingLevel {
	seen := make(map[agentcore.ThinkingLevel]struct{}, len(levels))
	out := make([]agentcore.ThinkingLevel, 0, len(levels))
	for _, level := range levels {
		if _, ok := seen[level]; ok {
			continue
		}
		seen[level] = struct{}{}
		out = append(out, level)
	}
	return out
}

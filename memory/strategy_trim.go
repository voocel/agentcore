package memory

import (
	"context"
	"fmt"

	"github.com/voocel/agentcore"
)

type LightTrimConfig struct {
	KeepRecent    int
	TextThreshold int
	PreserveHead  int
	PreserveTail  int
}

type LightTrimStrategy struct {
	cfg LightTrimConfig
}

func NewLightTrim(cfg LightTrimConfig) *LightTrimStrategy {
	if cfg.KeepRecent <= 0 {
		cfg.KeepRecent = 4
	}
	if cfg.TextThreshold <= 0 {
		cfg.TextThreshold = 4000
	}
	if cfg.PreserveHead <= 0 {
		cfg.PreserveHead = 1200
	}
	if cfg.PreserveTail <= 0 {
		cfg.PreserveTail = 800
	}
	return &LightTrimStrategy{cfg: cfg}
}

func (s *LightTrimStrategy) Name() string { return "light_trim" }

func (s *LightTrimStrategy) Apply(_ context.Context, _ []agentcore.AgentMessage, view []agentcore.AgentMessage, _ Budget) ([]agentcore.AgentMessage, StrategyResult, error) {
	if len(view) == 0 {
		return view, StrategyResult{Name: s.Name()}, nil
	}

	lastEligible := len(view) - s.cfg.KeepRecent
	if lastEligible <= 0 {
		return view, StrategyResult{Name: s.Name()}, nil
	}

	out := copyMessages(view)
	applied := false
	saved := 0

	for i := 0; i < lastEligible; i++ {
		msg, ok := out[i].(agentcore.Message)
		if !ok {
			continue
		}
		next, changed := trimLongTextBlocks(msg, s.cfg.TextThreshold, s.cfg.PreserveHead, s.cfg.PreserveTail)
		if !changed {
			continue
		}
		out[i] = next
		saved += max(0, EstimateTokens(msg)-EstimateTokens(next))
		applied = true
	}

	return out, StrategyResult{
		Applied:     applied,
		TokensSaved: saved,
		Name:        s.Name(),
	}, nil
}

func trimLongTextBlocks(msg agentcore.Message, threshold, preserveHead, preserveTail int) (agentcore.Message, bool) {
	newContent := make([]agentcore.ContentBlock, len(msg.Content))
	changed := false
	for i, block := range msg.Content {
		if block.Type != agentcore.ContentText {
			newContent[i] = block
			continue
		}
		runes := []rune(block.Text)
		if len(runes) <= threshold {
			newContent[i] = block
			continue
		}
		headCount := min(preserveHead, len(runes))
		tailCount := min(preserveTail, len(runes)-headCount)
		head := string(runes[:headCount])
		tail := string(runes[len(runes)-tailCount:])
		trimmed := len(runes) - headCount - tailCount
		newContent[i] = agentcore.ContentBlock{
			Type: agentcore.ContentText,
			Text: fmt.Sprintf("%s\n%s\n%s", head, formatTrimmedPlaceholder(trimmed), tail),
		}
		changed = true
	}
	if !changed {
		return msg, false
	}
	next := msg
	next.Content = newContent
	return next, true
}

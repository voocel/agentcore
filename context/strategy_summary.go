package context

import (
	"context"
	"time"

	"github.com/voocel/agentcore"
)

// FullSummaryConfig controls ContextSummary checkpoint generation.
// KeepRecentTokens reserves a recent suffix of messages to keep verbatim.
// PostSummaryHooks may inject lightweight reminder messages after the summary.
type FullSummaryConfig struct {
	// Model performs the actual summary generation.
	Model agentcore.ChatModel
	// StripImages controls whether images are removed before summarization.
	// Nil defaults to true.
	StripImages *bool
	// KeepRecentTokens reserves a recent suffix to keep verbatim.
	KeepRecentTokens int
	// PostSummaryHooks inject lightweight reminder messages after the summary.
	PostSummaryHooks []PostSummaryHook
}

// FullSummaryStrategy rewrites older context into a ContextSummary checkpoint
// while keeping a recent suffix of messages verbatim.
type FullSummaryStrategy struct {
	cfg FullSummaryConfig
}

// NewFullSummary constructs the terminal summary strategy used when lighter
// rewrites are insufficient. Model is required for actual summarization.
func NewFullSummary(cfg FullSummaryConfig) *FullSummaryStrategy {
	if cfg.KeepRecentTokens <= 0 {
		cfg.KeepRecentTokens = defaultKeepRecentTokens
	}
	return &FullSummaryStrategy{cfg: cfg}
}

func (s *FullSummaryStrategy) Name() string { return "full_summary" }

func (s *FullSummaryStrategy) Apply(ctx context.Context, _ []agentcore.AgentMessage, view []agentcore.AgentMessage, budget Budget) ([]agentcore.AgentMessage, StrategyResult, error) {
	if budget.Window <= 0 || budget.Tokens <= budget.Threshold {
		return view, StrategyResult{Name: s.Name()}, nil
	}
	return s.apply(ctx, view, budget, false)
}

func (s *FullSummaryStrategy) ForceApply(ctx context.Context, _ []agentcore.AgentMessage, view []agentcore.AgentMessage, budget Budget) ([]agentcore.AgentMessage, StrategyResult, error) {
	return s.apply(ctx, view, budget, true)
}

// SetPostSummaryHooks replaces the hook list used to inject lightweight
// reminder messages after a summary checkpoint is produced.
func (s *FullSummaryStrategy) SetPostSummaryHooks(hooks ...PostSummaryHook) {
	s.cfg.PostSummaryHooks = hooks
}

func (s *FullSummaryStrategy) apply(ctx context.Context, view []agentcore.AgentMessage, budget Budget, force bool) ([]agentcore.AgentMessage, StrategyResult, error) {
	if len(view) == 0 || s.cfg.Model == nil {
		return view, StrategyResult{Name: s.Name()}, nil
	}

	ctxWindow := budget.Window
	reserve := budget.Window - budget.Threshold
	if reserve <= 0 {
		reserve = 1
	}
	if force {
		ctxWindow = max(budget.Tokens, 2)
		reserve = 1
	}

	cfg := summaryRunConfig{
		Model:            s.cfg.Model,
		ContextWindow:    ctxWindow,
		ReserveTokens:    reserve,
		KeepRecentTokens: s.cfg.KeepRecentTokens,
	}
	stripImages := true
	if s.cfg.StripImages != nil {
		stripImages = *s.cfg.StripImages
	}

	next, info, err := runSummaryCompaction(ctx, cfg, view, stripImages)
	if err != nil {
		return nil, StrategyResult{Name: s.Name()}, err
	}
	if info == nil || !containsContextSummary(next) {
		return view, StrategyResult{Name: s.Name()}, nil
	}

	next, err = s.applyHooks(ctx, next, *info)
	if err != nil {
		return nil, StrategyResult{Name: s.Name()}, err
	}

	info.TokensAfter = EstimateTotal(next)
	if info.Duration == 0 {
		info.Duration = time.Millisecond
	}

	return next, StrategyResult{
		Applied:     true,
		TokensSaved: max(0, budget.Tokens-EstimateTotal(next)),
		Name:        s.Name(),
		Info:        info,
	}, nil
}

func (s *FullSummaryStrategy) applyHooks(ctx context.Context, msgs []agentcore.AgentMessage, info SummaryInfo) ([]agentcore.AgentMessage, error) {
	if len(s.cfg.PostSummaryHooks) == 0 || len(msgs) == 0 {
		return msgs, nil
	}
	kept := append([]agentcore.AgentMessage(nil), msgs[1:]...)
	var injected []agentcore.AgentMessage
	var err error
	for _, hook := range s.cfg.PostSummaryHooks {
		var extra []agentcore.AgentMessage
		extra, err = hook(ctx, info, kept)
		if err != nil {
			return nil, err
		}
		injected = append(injected, extra...)
	}
	if len(injected) == 0 {
		return msgs, nil
	}
	out := make([]agentcore.AgentMessage, 0, len(msgs)+len(injected))
	out = append(out, msgs[0])
	out = append(out, injected...)
	out = append(out, msgs[1:]...)
	return out, nil
}

func containsContextSummary(msgs []agentcore.AgentMessage) bool {
	for _, msg := range msgs {
		if _, ok := msg.(ContextSummary); ok {
			return true
		}
	}
	return false
}

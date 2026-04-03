package memory

import (
	"context"
	"sync"

	"github.com/voocel/agentcore"
)

const defaultEngineReserveTokens = 16384

type EngineConfig struct {
	ContextWindow int
	ReserveTokens int
	Strategies    []Strategy
	OnProject     func(ChangeInfo)
	OnRecover     func(ChangeInfo)
}

type ChangeInfo struct {
	Reason       string
	Strategy     string
	Changed      bool
	TokensBefore int
	TokensAfter  int
	Info         *CompactionInfo
}

// ContextEngine implements agentcore.ContextManager with a strategy-driven
// prompt projection pipeline.
type ContextEngine struct {
	cfg EngineConfig

	mu         sync.Mutex
	transcript []agentcore.AgentMessage
	lastUsage  *agentcore.ContextUsage
}

func NewEngine(cfg EngineConfig) *ContextEngine {
	if cfg.ReserveTokens <= 0 {
		cfg.ReserveTokens = defaultEngineReserveTokens
	}
	return &ContextEngine{cfg: cfg}
}

// NewDefaultEngine creates an engine with sensible defaults: tool result
// microcompact → light trim → full summary. This is the simplest way to
// enable context compression — no configuration needed beyond model and window.
//
//	engine := memory.NewDefaultEngine(model, 128000)
//	agentcore.WithContextManager(engine)
func NewDefaultEngine(model agentcore.ChatModel, contextWindow int) *ContextEngine {
	return NewEngine(EngineConfig{
		ContextWindow: contextWindow,
		Strategies: []Strategy{
			NewToolResultMicrocompact(ToolResultMicrocompactConfig{}),
			NewLightTrim(LightTrimConfig{}),
			NewFullSummary(FullSummaryConfig{Model: model}),
		},
	})
}

func (e *ContextEngine) SetProjectHook(fn func(ChangeInfo)) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.cfg.OnProject = fn
}

func (e *ContextEngine) SetRecoverHook(fn func(ChangeInfo)) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.cfg.OnRecover = fn
}

func (e *ContextEngine) Sync(msgs []agentcore.AgentMessage) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.transcript = copyMessages(msgs)
	usage := e.estimateUsage(msgs)
	e.lastUsage = &usage
}

func (e *ContextEngine) Usage() *agentcore.ContextUsage {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.lastUsage == nil {
		return nil
	}
	cp := *e.lastUsage
	return &cp
}

func (e *ContextEngine) Project(ctx context.Context, msgs []agentcore.AgentMessage) (agentcore.ContextProjection, error) {
	e.Sync(msgs)
	view, usage, changed, info, strategy, err := e.apply(ctx, msgs, false)
	if err != nil {
		return agentcore.ContextProjection{}, err
	}
	if changed && e.cfg.OnProject != nil {
		e.cfg.OnProject(ChangeInfo{
			Reason:       "threshold",
			Strategy:     strategy,
			Changed:      true,
			TokensBefore: EstimateTotal(msgs),
			TokensAfter:  EstimateTotal(view),
			Info:         info,
		})
	}
	return agentcore.ContextProjection{
		Messages: view,
		Usage:    usage,
	}, nil
}

func (e *ContextEngine) Compact(ctx context.Context, msgs []agentcore.AgentMessage, _ agentcore.CompactReason) (agentcore.ContextCommitResult, error) {
	e.Sync(msgs)
	view, usage, changed, info, strategy, err := e.apply(ctx, msgs, true)
	if err != nil {
		return agentcore.ContextCommitResult{}, err
	}
	return agentcore.ContextCommitResult{
		Messages:       view,
		Usage:          usage,
		Changed:        changed,
		Strategy:       strategy,
		CompactedCount: infoValueInt(info, func(i *CompactionInfo) int { return i.CompactedCount }),
		KeptCount:      infoValueInt(info, func(i *CompactionInfo) int { return i.KeptCount }),
		SplitTurn:      infoValueBool(info, func(i *CompactionInfo) bool { return i.IsSplitTurn }),
	}, nil
}

func (e *ContextEngine) RecoverOverflow(ctx context.Context, msgs []agentcore.AgentMessage, _ error) (agentcore.ContextRecoveryResult, error) {
	e.Sync(msgs)
	view, usage, changed, info, strategy, err := e.apply(ctx, msgs, true)
	if err != nil {
		return agentcore.ContextRecoveryResult{}, err
	}
	if changed && e.cfg.OnRecover != nil {
		e.cfg.OnRecover(ChangeInfo{
			Reason:       "overflow",
			Strategy:     strategy,
			Changed:      true,
			TokensBefore: EstimateTotal(msgs),
			TokensAfter:  EstimateTotal(view),
			Info:         info,
		})
	}
	return agentcore.ContextRecoveryResult{
		View:           view,
		CommitMessages: view,
		Usage:          usage,
		Changed:        changed,
		ShouldCommit:   changed,
		Strategy:       strategy,
		CompactedCount: infoValueInt(info, func(i *CompactionInfo) int { return i.CompactedCount }),
		KeptCount:      infoValueInt(info, func(i *CompactionInfo) int { return i.KeptCount }),
		SplitTurn:      infoValueBool(info, func(i *CompactionInfo) bool { return i.IsSplitTurn }),
	}, nil
}

func (e *ContextEngine) apply(ctx context.Context, msgs []agentcore.AgentMessage, force bool) ([]agentcore.AgentMessage, *agentcore.ContextUsage, bool, *CompactionInfo, string, error) {
	view := copyMessages(msgs)
	budget := e.computeBudget(view)
	if len(e.cfg.Strategies) == 0 {
		usage := ptrUsage(e.estimateUsage(view))
		e.setLastUsage(usage)
		return view, usage, false, nil, "", nil
	}

	changed := false
	var lastInfo *CompactionInfo
	lastStrategy := ""

	if !force {
		for _, strategy := range e.cfg.Strategies {
			if budget.Window <= 0 || budget.Tokens <= budget.Threshold {
				break
			}

			nextView, result, err := strategy.Apply(ctx, e.snapshotTranscript(), view, budget)
			if err != nil {
				return nil, nil, false, lastInfo, lastStrategy, err
			}
			if result.Info != nil {
				lastInfo = result.Info
			}
			if result.Applied {
				view = nextView
				changed = true
				lastStrategy = result.Name
				budget = e.computeBudget(view)
				if budget.Tokens <= budget.Threshold {
					break
				}
			}
		}
	} else {
		for _, strategy := range e.cfg.Strategies {
			forced, ok := strategy.(ForceCompactionStrategy)
			if !ok {
				continue
			}
			nextView, result, err := forced.ForceApply(ctx, e.snapshotTranscript(), copyMessages(msgs), budget)
			if err != nil {
				return nil, nil, false, lastInfo, lastStrategy, err
			}
			if result.Info != nil {
				lastInfo = result.Info
			}
			if result.Applied {
				view = nextView
				changed = true
				lastStrategy = result.Name
				budget = e.computeBudget(view)
				break
			}
		}
	}

	usage := ptrUsage(e.estimateUsage(view))
	e.setLastUsage(usage)
	return view, usage, changed, lastInfo, lastStrategy, nil
}

func (e *ContextEngine) computeBudget(msgs []agentcore.AgentMessage) Budget {
	window := e.cfg.ContextWindow
	estimate := EstimateContextTokens(msgs)
	threshold := window - e.cfg.ReserveTokens
	if threshold < 0 {
		threshold = 0
	}
	return Budget{
		Tokens:    estimate.Tokens,
		Window:    window,
		Threshold: threshold,
	}
}

func (e *ContextEngine) setLastUsage(usage *agentcore.ContextUsage) {
	e.mu.Lock()
	defer e.mu.Unlock()
	if usage == nil {
		e.lastUsage = nil
		return
	}
	cp := *usage
	e.lastUsage = &cp
}

func (e *ContextEngine) estimateUsage(msgs []agentcore.AgentMessage) agentcore.ContextUsage {
	estimate := EstimateContextTokens(msgs)
	window := e.cfg.ContextWindow
	pct := 0.0
	if window > 0 {
		pct = float64(estimate.Tokens) / float64(window) * 100
	}
	return agentcore.ContextUsage{
		Tokens:         estimate.Tokens,
		ContextWindow:  window,
		Percent:        pct,
		UsageTokens:    estimate.UsageTokens,
		TrailingTokens: estimate.TrailingTokens,
	}
}

func ptrUsage(usage agentcore.ContextUsage) *agentcore.ContextUsage {
	return &usage
}

func infoValueInt(info *CompactionInfo, getter func(*CompactionInfo) int) int {
	if info == nil {
		return 0
	}
	return getter(info)
}

func infoValueBool(info *CompactionInfo, getter func(*CompactionInfo) bool) bool {
	if info == nil {
		return false
	}
	return getter(info)
}

func (e *ContextEngine) snapshotTranscript() []agentcore.AgentMessage {
	e.mu.Lock()
	defer e.mu.Unlock()
	return copyMessages(e.transcript)
}

func copyMessages(msgs []agentcore.AgentMessage) []agentcore.AgentMessage {
	if len(msgs) == 0 {
		return nil
	}
	out := make([]agentcore.AgentMessage, len(msgs))
	copy(out, msgs)
	return out
}

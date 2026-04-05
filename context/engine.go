// Package context provides strategy-driven context compression for agentcore:
// prompt projection, summary checkpoints, overflow recovery, and usage estimation.
//
//	engine := context.NewDefaultEngine(model, 128000)
//	agent := agentcore.NewAgent(
//		agentcore.WithModel(model),
//		agentcore.WithContextManager(engine),
//	)
package context

import (
	"context"
	"sync"

	"github.com/voocel/agentcore"
)

const defaultEngineReserveTokens = 16384

// EngineConfig configures the default strategy-driven ContextEngine.
// ContextWindow is required. Strategies run in order until the projected view
// drops below the threshold implied by ContextWindow and ReserveTokens.
type EngineConfig struct {
	// ContextWindow is the model's maximum supported context window.
	ContextWindow int
	// ReserveTokens is the minimum prompt headroom to preserve. When zero, a
	// conservative default is used.
	ReserveTokens int
	// Strategies are applied in order until the projected view fits.
	Strategies []Strategy
	// CommitOnProject makes threshold-triggered Project rewrites replace the
	// runtime baseline before the current call continues.
	CommitOnProject bool
	// OnProject is called when Project rewrites the prompt view.
	OnProject func(RewriteEvent)
	// OnRecover is called when RecoverOverflow rewrites the prompt view.
	OnRecover func(RewriteEvent)
}

// RewriteStep records a single strategy execution within the apply pipeline.
type RewriteStep struct {
	Name         string
	Applied      bool
	TokensBefore int
	TokensAfter  int
}

// RewriteEvent reports a projection or recovery rewrite that actually changed
// the active view. Info is populated when the rewrite produced a summary
// checkpoint. Steps records every strategy attempted during the pipeline run.
type RewriteEvent struct {
	Reason       string
	Strategy     string
	Changed      bool
	Committed    bool
	TokensBefore int
	TokensAfter  int
	Info         *SummaryInfo
	Steps        []RewriteStep
}

// ContextEngine implements agentcore.ContextManager with a strategy-driven
// prompt projection pipeline.
type ContextEngine struct {
	cfg EngineConfig

	mu         sync.Mutex
	transcript []agentcore.AgentMessage
	baseline   *agentcore.ContextUsage
	lastUsage  *agentcore.ContextUsage
	lastView   []agentcore.AgentMessage
	lastScope  string
	lastChange changeState
}

type changeState struct {
	strategy string
	changed  bool
	info     *SummaryInfo
}

// NewEngine constructs a ContextEngine from an explicit strategy pipeline.
func NewEngine(cfg EngineConfig) *ContextEngine {
	if cfg.ReserveTokens <= 0 {
		cfg.ReserveTokens = defaultEngineReserveTokens
	}
	return &ContextEngine{cfg: cfg}
}

// NewDefaultEngine creates a ContextEngine with the default rewrite pipeline:
// tool-result microcompact, light trim, then full summary.
//
// This is the recommended entry point for applications that want context
// management without custom strategy wiring.
//
//	engine := context.NewDefaultEngine(model, 128000)
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

// SetProjectHook installs the callback fired when Project rewrites the prompt
// view due to context pressure.
func (e *ContextEngine) SetProjectHook(fn func(RewriteEvent)) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.cfg.OnProject = fn
}

// SetRecoverHook installs the callback fired when RecoverOverflow rewrites the
// prompt view after a provider overflow error.
func (e *ContextEngine) SetRecoverHook(fn func(RewriteEvent)) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.cfg.OnRecover = fn
}

// Sync records msgs as the current runtime baseline and resets the active view
// remembered by the engine to that baseline.
func (e *ContextEngine) Sync(msgs []agentcore.AgentMessage) {
	usage := e.estimateUsage(msgs)

	e.mu.Lock()
	defer e.mu.Unlock()
	e.transcript = copyMessages(msgs)
	e.lastView = copyMessages(msgs)
	e.lastScope = "baseline"
	cp := usage
	e.baseline = &cp
	e.lastUsage = &cp
}

// Usage returns the latest effective usage remembered by the engine.
func (e *ContextEngine) Usage() *agentcore.ContextUsage {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.lastUsage == nil {
		return nil
	}
	cp := *e.lastUsage
	return &cp
}

// Snapshot returns the latest active view snapshot remembered by the engine.
func (e *ContextEngine) Snapshot() *agentcore.ContextSnapshot {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.lastUsage == nil && len(e.lastView) == 0 && len(e.transcript) == 0 {
		return nil
	}

	var baseline *agentcore.ContextUsage
	if e.baseline != nil {
		cp := *e.baseline
		baseline = &cp
	}
	var usage *agentcore.ContextUsage
	if e.lastUsage != nil {
		cp := *e.lastUsage
		usage = &cp
	}
	counts := summarizeContextView(e.lastView)
	return &agentcore.ContextSnapshot{
		BaselineUsage:      baseline,
		Usage:              usage,
		Scope:              e.lastScope,
		TranscriptMessages: len(e.transcript),
		ActiveMessages:     len(e.lastView),
		SummaryMessages:    counts.summaryMessages,
		ToolMessages:       counts.toolMessages,
		ClearedToolResults: counts.clearedToolResults,
		TrimmedTextBlocks:  counts.trimmedTextBlocks,
		LastStrategy:       e.lastChange.strategy,
		LastChanged:        e.lastChange.changed,
		LastCompactedCount: infoValueInt(e.lastChange.info, func(i *SummaryInfo) int { return i.CompactedCount }),
		LastKeptCount:      infoValueInt(e.lastChange.info, func(i *SummaryInfo) int { return i.KeptCount }),
		LastSplitTurn:      infoValueBool(e.lastChange.info, func(i *SummaryInfo) bool { return i.IsSplitTurn }),
	}
}

// Project builds the prompt view for one LLM call without committing a new
// runtime baseline.
func (e *ContextEngine) Project(ctx context.Context, msgs []agentcore.AgentMessage) (agentcore.ContextProjection, error) {
	e.Sync(msgs)
	r, err := e.apply(ctx, msgs, false)
	if err != nil {
		return agentcore.ContextProjection{}, err
	}
	if r.Changed && e.cfg.OnProject != nil {
		e.cfg.OnProject(RewriteEvent{
			Reason:       "threshold",
			Strategy:     r.Strategy,
			Changed:      true,
			Committed:    r.Changed && e.cfg.CommitOnProject,
			TokensBefore: EstimateTotal(msgs),
			TokensAfter:  EstimateTotal(r.View),
			Info:         r.Info,
			Steps:        r.Steps,
		})
	}
	proj := agentcore.ContextProjection{
		Messages: r.View,
		Usage:    r.Usage,
	}
	if r.Changed && e.cfg.CommitOnProject {
		proj.CommitMessages = copyMessages(r.View)
		proj.ShouldCommit = true
	}
	return proj, nil
}

// Compact performs a forced rewrite suitable for explicit committed actions
// such as /compact. The caller should replace its runtime baseline with the
// returned Messages when Changed is true.
func (e *ContextEngine) Compact(ctx context.Context, msgs []agentcore.AgentMessage, _ agentcore.CompactReason) (agentcore.ContextCommitResult, error) {
	e.Sync(msgs)
	r, err := e.apply(ctx, msgs, true)
	if err != nil {
		return agentcore.ContextCommitResult{}, err
	}
	return agentcore.ContextCommitResult{
		Messages:       r.View,
		Usage:          r.Usage,
		Changed:        r.Changed,
		Strategy:       r.Strategy,
		CompactedCount: infoValueInt(r.Info, func(i *SummaryInfo) int { return i.CompactedCount }),
		KeptCount:      infoValueInt(r.Info, func(i *SummaryInfo) int { return i.KeptCount }),
		SplitTurn:      infoValueBool(r.Info, func(i *SummaryInfo) bool { return i.IsSplitTurn }),
	}, nil
}

// RecoverOverflow performs a forced rewrite after a provider reports context
// overflow. When ShouldCommit is true, CommitMessages should become the new
// runtime baseline before retrying.
func (e *ContextEngine) RecoverOverflow(ctx context.Context, msgs []agentcore.AgentMessage, _ error) (agentcore.ContextRecoveryResult, error) {
	e.Sync(msgs)
	r, err := e.apply(ctx, msgs, true)
	if err != nil {
		return agentcore.ContextRecoveryResult{}, err
	}
	e.setLastState(r.View, r.Usage, "recovered", r.Strategy, r.Changed, r.Info)
	if r.Changed && e.cfg.OnRecover != nil {
		e.cfg.OnRecover(RewriteEvent{
			Reason:       "overflow",
			Strategy:     r.Strategy,
			Changed:      true,
			Committed:    r.Changed,
			TokensBefore: EstimateTotal(msgs),
			TokensAfter:  EstimateTotal(r.View),
			Info:         r.Info,
			Steps:        r.Steps,
		})
	}
	return agentcore.ContextRecoveryResult{
		View:           r.View,
		CommitMessages: r.View,
		Usage:          r.Usage,
		Changed:        r.Changed,
		ShouldCommit:   r.Changed,
		Strategy:       r.Strategy,
		CompactedCount: infoValueInt(r.Info, func(i *SummaryInfo) int { return i.CompactedCount }),
		KeptCount:      infoValueInt(r.Info, func(i *SummaryInfo) int { return i.KeptCount }),
		SplitTurn:      infoValueBool(r.Info, func(i *SummaryInfo) bool { return i.IsSplitTurn }),
	}, nil
}

type applyResult struct {
	View     []agentcore.AgentMessage
	Usage    *agentcore.ContextUsage
	Changed  bool
	Info     *SummaryInfo
	Strategy string
	Steps    []RewriteStep
}

func (e *ContextEngine) apply(ctx context.Context, msgs []agentcore.AgentMessage, force bool) (applyResult, error) {
	view := copyMessages(msgs)
	budget := e.computeBudget(view)
	if len(e.cfg.Strategies) == 0 {
		usage := ptrUsage(e.estimateUsage(view))
		e.setLastState(view, usage, scopeFor(force), "", false, nil)
		return applyResult{View: view, Usage: usage}, nil
	}

	var r applyResult

	if !force {
		for _, strategy := range e.cfg.Strategies {
			if budget.Window <= 0 || budget.Tokens <= budget.Threshold {
				break
			}

			tokensBefore := budget.Tokens
			nextView, result, err := strategy.Apply(ctx, e.snapshotTranscript(), view, budget)
			if err != nil {
				return r, err
			}

			tokensAfter := tokensBefore
			if result.Applied {
				view = nextView
				r.Changed = true
				r.Strategy = result.Name
				budget = e.computeBudget(view)
				tokensAfter = budget.Tokens
			}
			if result.Info != nil {
				r.Info = result.Info
			}

			r.Steps = append(r.Steps, RewriteStep{
				Name:         result.Name,
				Applied:      result.Applied,
				TokensBefore: tokensBefore,
				TokensAfter:  tokensAfter,
			})

			if result.Applied && budget.Tokens <= budget.Threshold {
				break
			}
		}
	} else {
		for _, strategy := range e.cfg.Strategies {
			forced, ok := strategy.(ForceCompactionStrategy)
			if !ok {
				continue
			}
			tokensBefore := budget.Tokens
			nextView, result, err := forced.ForceApply(ctx, e.snapshotTranscript(), copyMessages(msgs), budget)
			if err != nil {
				return r, err
			}

			tokensAfter := tokensBefore
			if result.Applied {
				view = nextView
				r.Changed = true
				r.Strategy = result.Name
				budget = e.computeBudget(view)
				tokensAfter = budget.Tokens
			}
			if result.Info != nil {
				r.Info = result.Info
			}

			r.Steps = append(r.Steps, RewriteStep{
				Name:         result.Name,
				Applied:      result.Applied,
				TokensBefore: tokensBefore,
				TokensAfter:  tokensAfter,
			})

			if result.Applied {
				break
			}
		}
	}

	r.View = view
	r.Usage = ptrUsage(e.estimateUsage(view))
	e.setLastState(view, r.Usage, scopeFor(force), r.Strategy, r.Changed, r.Info)
	return r, nil
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

func (e *ContextEngine) setLastState(view []agentcore.AgentMessage, usage *agentcore.ContextUsage, scope, strategy string, changed bool, info *SummaryInfo) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.lastView = copyMessages(view)
	e.lastScope = scope
	if usage == nil {
		e.lastUsage = nil
	} else {
		cp := *usage
		e.lastUsage = &cp
	}
	if strategy != "" || changed || info != nil {
		e.lastChange = changeState{
			strategy: strategy,
			changed:  changed,
			info:     copySummaryInfo(info),
		}
	}
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

func scopeFor(force bool) string {
	if force {
		return "committed"
	}
	return "projected"
}

func infoValueInt(info *SummaryInfo, getter func(*SummaryInfo) int) int {
	if info == nil {
		return 0
	}
	return getter(info)
}

func infoValueBool(info *SummaryInfo, getter func(*SummaryInfo) bool) bool {
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

type contextViewCounts struct {
	summaryMessages    int
	toolMessages       int
	clearedToolResults int
	trimmedTextBlocks  int
}

func summarizeContextView(msgs []agentcore.AgentMessage) contextViewCounts {
	var counts contextViewCounts
	for _, am := range msgs {
		switch msg := am.(type) {
		case ContextSummary:
			counts.summaryMessages++
		case agentcore.Message:
			if msg.Role == agentcore.RoleTool {
				counts.toolMessages++
			}
			if compacted, _ := msg.Metadata["compacted_tool_result"].(bool); compacted {
				counts.clearedToolResults++
			}
			switch v := msg.Metadata["trimmed_text_blocks"].(type) {
			case int:
				counts.trimmedTextBlocks += v
			case int32:
				counts.trimmedTextBlocks += int(v)
			case int64:
				counts.trimmedTextBlocks += int(v)
			case float64:
				counts.trimmedTextBlocks += int(v)
			}
		}
	}
	return counts
}

func copySummaryInfo(info *SummaryInfo) *SummaryInfo {
	if info == nil {
		return nil
	}
	cp := *info
	cp.ReadFiles = append([]string(nil), info.ReadFiles...)
	cp.ModifiedFiles = append([]string(nil), info.ModifiedFiles...)
	return &cp
}

func copyMessages(msgs []agentcore.AgentMessage) []agentcore.AgentMessage {
	if len(msgs) == 0 {
		return nil
	}
	out := make([]agentcore.AgentMessage, len(msgs))
	copy(out, msgs)
	return out
}

func cloneMetadata(src map[string]any) map[string]any {
	if len(src) == 0 {
		return nil
	}
	dst := make(map[string]any, len(src))
	for k, v := range src {
		dst[k] = v
	}
	return dst
}

// --- Optional interfaces for auto-wiring in NewAgent ---

// ConvertToLLM implements agentcore.ContextLLMConverter.
func (e *ContextEngine) ConvertToLLM(msgs []agentcore.AgentMessage) []agentcore.Message {
	return ContextConvertToLLM(msgs)
}

// EstimateContext implements agentcore.ContextEstimator.
func (e *ContextEngine) EstimateContext(msgs []agentcore.AgentMessage) (tokens, usageTokens, trailingTokens int) {
	return ContextEstimateAdapter(msgs)
}

// ContextWindow implements agentcore.ContextWindower.
func (e *ContextEngine) ContextWindow() int {
	return e.cfg.ContextWindow
}

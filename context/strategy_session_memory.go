package context

import (
	"context"
	"strings"
	"time"

	"github.com/voocel/agentcore"
)

// SessionMemoryConfig configures the SessionMemoryStrategy, a compaction
// strategy that substitutes a harness-maintained, pre-computed summary for
// the LLM-generated one produced by FullSummaryStrategy.
//
// The strategy mirrors Claude Code v2.1.88's sessionMemoryCompact.ts: when a
// project-scoped session memory already exists, compaction can skip the
// synchronous summarization LLM call and reuse the living memory as the
// ContextSummary body. The harness supplies the seed via SeedFn.
type SessionMemoryConfig struct {
	// SeedFn returns the pre-computed session memory text. Called lazily at
	// apply time. Returning ("", nil) makes the strategy a no-op so the
	// pipeline falls through to LLM summarization.
	SeedFn func() (string, error)

	// KeepRecentTokens reserves a recent suffix to keep verbatim. Matches the
	// FullSummaryStrategy field with the same name; defaults to
	// defaultKeepRecentTokens when zero.
	KeepRecentTokens int

	// MaxSeedRunes caps how much of the seed we inject. Oversized memory
	// files can consume the entire post-compact budget. When zero, a
	// conservative default is used; when negative, no cap is enforced.
	MaxSeedRunes int

	// Label, when non-empty, overrides the strategy name reported in
	// observability events. Useful for distinguishing multiple memory
	// sources in multi-project harnesses.
	Label string
}

const (
	defaultMaxSeedRunes         = 20000
	sessionMemoryStrategyName   = "session_memory"
	sessionMemoryTruncatedNotice = "\n\n[... session memory truncated for compact budget ...]"
)

// SessionMemoryStrategy emits a ContextSummary checkpoint from a caller-
// provided seed without invoking the LLM. When no seed is available it is a
// no-op, so it composes naturally with FullSummaryStrategy as a fallback.
type SessionMemoryStrategy struct {
	cfg SessionMemoryConfig
}

// NewSessionMemory constructs a SessionMemoryStrategy. SeedFn may be nil;
// the strategy will simply no-op until a non-nil function is installed
// via SetSeedFn.
func NewSessionMemory(cfg SessionMemoryConfig) *SessionMemoryStrategy {
	if cfg.KeepRecentTokens <= 0 {
		cfg.KeepRecentTokens = defaultKeepRecentTokens
	}
	if cfg.MaxSeedRunes == 0 {
		cfg.MaxSeedRunes = defaultMaxSeedRunes
	}
	return &SessionMemoryStrategy{cfg: cfg}
}

// Name reports the strategy label used in RewriteEvent / SummaryInfo.
func (s *SessionMemoryStrategy) Name() string {
	if s.cfg.Label != "" {
		return s.cfg.Label
	}
	return sessionMemoryStrategyName
}

// SetSeedFn swaps the seed provider at runtime (e.g. when the harness
// re-resolves the current project after a session switch).
func (s *SessionMemoryStrategy) SetSeedFn(fn func() (string, error)) {
	s.cfg.SeedFn = fn
}

// Apply replaces history older than KeepRecentTokens with a ContextSummary
// whose body is the seed text. When no seed is available it returns the view
// unchanged, allowing a downstream FullSummary strategy to run.
func (s *SessionMemoryStrategy) Apply(ctx context.Context, _ []agentcore.AgentMessage, view []agentcore.AgentMessage, budget Budget) ([]agentcore.AgentMessage, StrategyResult, error) {
	if budget.Window <= 0 || budget.Tokens <= budget.Threshold {
		return view, StrategyResult{Name: s.Name()}, nil
	}
	if s.cfg.SeedFn == nil {
		return view, StrategyResult{Name: s.Name()}, nil
	}
	seed, err := s.cfg.SeedFn()
	if err != nil {
		// Seed errors are recoverable — let the pipeline fall through to LLM
		// summary rather than aborting compaction entirely.
		return view, StrategyResult{Name: s.Name()}, nil
	}
	seed = strings.TrimSpace(seed)
	if seed == "" {
		return view, StrategyResult{Name: s.Name()}, nil
	}

	cut := findCutPoint(view, s.cfg.KeepRecentTokens)
	if cut.firstKeptIndex <= 0 {
		return view, StrategyResult{Name: s.Name()}, nil
	}

	// For split turns we include the preceding user turn start in the kept
	// suffix — we cannot synthesize a turn-prefix summary without an LLM,
	// but the seed is a living document that already tracks current state.
	keepStart := cut.firstKeptIndex
	if cut.isSplitTurn && cut.turnStartIndex >= 0 {
		keepStart = cut.turnStartIndex
	}

	truncated, wasTruncated := truncateSeed(seed, s.cfg.MaxSeedRunes)
	body := truncated
	if wasTruncated {
		body += sessionMemoryTruncatedNotice
	}

	start := time.Now()
	tokensBefore := EstimateTotal(view)
	allCompacted := view[:keepStart]
	readFiles, modifiedFiles := extractFileOps(allCompacted)
	body += formatFileOps(readFiles, modifiedFiles)

	cs := ContextSummary{
		Summary:       body,
		TokensBefore:  tokensBefore,
		ReadFiles:     readFiles,
		ModifiedFiles: modifiedFiles,
		Timestamp:     start,
	}

	toKeep := view[keepStart:]
	result := make([]agentcore.AgentMessage, 0, 1+len(toKeep))
	result = append(result, cs)
	result = append(result, toKeep...)

	info := &SummaryInfo{
		TokensBefore:   tokensBefore,
		TokensAfter:    EstimateTotal(result),
		MessagesBefore: len(view),
		MessagesAfter:  len(result),
		CompactedCount: len(allCompacted),
		KeptCount:      len(toKeep),
		IsSplitTurn:    cut.isSplitTurn,
		IsIncremental:  true, // seed is a continuously updated living document
		SummaryLen:     len([]rune(body)),
		Duration:       time.Since(start),
		ReadFiles:      readFiles,
		ModifiedFiles:  modifiedFiles,
	}

	return result, StrategyResult{
		Applied:     true,
		TokensSaved: max(0, tokensBefore-info.TokensAfter),
		Name:        s.Name(),
		Info:        info,
	}, nil
}

// truncateSeed enforces MaxSeedRunes by trimming at a line boundary. When the
// cap is non-positive or the seed already fits, it is returned unchanged.
func truncateSeed(seed string, maxRunes int) (string, bool) {
	if maxRunes <= 0 {
		return seed, false
	}
	runes := []rune(seed)
	if len(runes) <= maxRunes {
		return seed, false
	}
	clipped := string(runes[:maxRunes])
	// Prefer cutting at the last line boundary to keep markdown structure.
	if idx := strings.LastIndexByte(clipped, '\n'); idx > maxRunes/2 {
		clipped = clipped[:idx]
	}
	return clipped, true
}

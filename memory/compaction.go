package memory

import (
	"context"
	"encoding/json"
	"fmt"
	"slices"
	"strings"
	"time"

	"github.com/voocel/agentcore"
)

const (
	defaultReserveTokens         = 16384
	defaultKeepRecentTokens      = 20000
	defaultMaxConsecutiveFailures = 3
)

// CompactionInfo holds details about a completed compaction for observability.
type CompactionInfo struct {
	TokensBefore   int
	TokensAfter    int
	MessagesBefore int
	MessagesAfter  int
	CompactedCount int           // number of messages summarized
	KeptCount      int           // number of messages retained verbatim
	IsSplitTurn    bool
	IsIncremental  bool          // updated an existing summary
	SummaryLen     int           // summary length in runes
	Duration       time.Duration // wall time including LLM calls
}

// CompactionConfig configures automatic context compaction.
type CompactionConfig struct {
	// Model is the ChatModel used for generating summaries.
	// Typically the same model the agent uses.
	Model agentcore.ChatModel

	// ContextWindow is the model's context window size in tokens.
	// Required — there is no default.
	ContextWindow int

	// ReserveTokens is the token headroom reserved for the LLM response.
	// Default: 16384.
	ReserveTokens int

	// KeepRecentTokens is the minimum number of recent tokens to always keep.
	// Default: 20000.
	KeepRecentTokens int

	// MaxConsecutiveFailures is the circuit breaker threshold.
	// After this many consecutive compaction failures, compaction is disabled
	// for the rest of the session. Default: 3. 0 means no limit.
	MaxConsecutiveFailures int

	// StripImages controls whether image content blocks are removed before
	// summarization. Images consume large amounts of tokens but cannot be
	// meaningfully summarized by text. Default: true.
	StripImages *bool

	// OnCompaction is called after a successful compaction with details.
	// Optional — nil means no callback.
	OnCompaction func(CompactionInfo)
}

// NewCompaction returns a TransformContext function that automatically compacts
// the message history when context tokens approach the window limit.
//
// Usage:
//
//	agent := agentcore.NewAgent(
//	    agentcore.WithTransformContext(memory.NewCompaction(memory.CompactionConfig{
//	        Model:         model,
//	        ContextWindow: 128000,
//	    })),
//	    agentcore.WithConvertToLLM(memory.CompactionConvertToLLM),
//	)
func NewCompaction(cfg CompactionConfig) func(context.Context, []agentcore.AgentMessage) ([]agentcore.AgentMessage, error) {
	if cfg.ReserveTokens <= 0 {
		cfg.ReserveTokens = defaultReserveTokens
	}
	if cfg.KeepRecentTokens <= 0 {
		cfg.KeepRecentTokens = defaultKeepRecentTokens
	}
	if cfg.MaxConsecutiveFailures < 0 {
		cfg.MaxConsecutiveFailures = defaultMaxConsecutiveFailures
	}
	stripImages := true
	if cfg.StripImages != nil {
		stripImages = *cfg.StripImages
	}

	// Circuit breaker state (closed over, survives across calls).
	consecutiveFailures := 0
	tripped := false

	return func(ctx context.Context, msgs []agentcore.AgentMessage) ([]agentcore.AgentMessage, error) {
		if len(msgs) == 0 || cfg.Model == nil {
			return msgs, nil
		}

		// Circuit breaker: skip compaction if tripped.
		if tripped {
			return msgs, nil
		}

		tokens := EstimateTotal(msgs)
		threshold := cfg.ContextWindow - cfg.ReserveTokens
		if threshold <= 0 || tokens <= threshold {
			return msgs, nil
		}

		cut := findCutPoint(msgs, cfg.KeepRecentTokens)
		if cut.firstKeptIndex <= 0 {
			return msgs, nil // nothing to compact
		}

		start := time.Now()

		// Determine what to summarize vs keep.
		// If split turn: history ends at turn start, turn prefix is [turnStart:firstKept].
		historyEnd := cut.firstKeptIndex
		if cut.isSplitTurn && cut.turnStartIndex >= 0 {
			historyEnd = cut.turnStartIndex
		}

		toSummarize := msgs[:historyEnd]
		toKeep := msgs[cut.firstKeptIndex:]

		// Strip images before summarization — they consume tokens but can't be text-summarized.
		if stripImages {
			toSummarize = stripImageBlocks(toSummarize)
		}

		// Find previous CompactionSummary for incremental update
		var previousSummary string
		for _, m := range toSummarize {
			if cs, ok := m.(CompactionSummary); ok {
				previousSummary = cs.Summary
			}
		}

		// Call options: history gets higher budget + thinking, prefix gets smaller budget
		historyOpts := []agentcore.CallOption{
			agentcore.WithMaxTokens(int(float64(cfg.ReserveTokens) * 0.8)),
			agentcore.WithThinking(agentcore.ThinkingHigh),
		}
		prefixOpts := []agentcore.CallOption{
			agentcore.WithMaxTokens(int(float64(cfg.ReserveTokens) * 0.5)),
		}

		var summary string

		needTurnPrefix := cut.isSplitTurn && cut.turnStartIndex >= 0
		var turnPrefix []agentcore.AgentMessage
		if needTurnPrefix {
			turnPrefix = msgs[cut.turnStartIndex:cut.firstKeptIndex]
			if stripImages {
				turnPrefix = stripImageBlocks(turnPrefix)
			}
			needTurnPrefix = len(turnPrefix) > 0
		}

		if needTurnPrefix {
			// Generate history + turn prefix summaries in parallel
			type summaryResult struct {
				text string
				err  error
			}
			historyCh := make(chan summaryResult, 1)
			prefixCh := make(chan summaryResult, 1)

			go func() {
				if len(toSummarize) == 0 {
					historyCh <- summaryResult{text: "No prior history."}
					return
				}
				s, e := generateSummary(ctx, cfg.Model, toSummarize, previousSummary, historyOpts...)
				historyCh <- summaryResult{s, e}
			}()
			go func() {
				s, e := generateTurnPrefixSummary(ctx, cfg.Model, turnPrefix, prefixOpts...)
				prefixCh <- summaryResult{s, e}
			}()

			hr := <-historyCh
			pr := <-prefixCh

			if hr.err != nil || pr.err != nil {
				consecutiveFailures++
				if cfg.MaxConsecutiveFailures > 0 && consecutiveFailures >= cfg.MaxConsecutiveFailures {
					tripped = true
				}
				if hr.err != nil {
					return nil, fmt.Errorf("compaction: %w", hr.err)
				}
				return nil, fmt.Errorf("compaction turn prefix: %w", pr.err)
			}
			summary = hr.text
			if pr.text != "" {
				summary += "\n\n---\n\n**Turn Context (split turn):**\n\n" + pr.text
			}
		} else {
			var err error
			summary, err = generateSummary(ctx, cfg.Model, toSummarize, previousSummary, historyOpts...)
			if err != nil {
				consecutiveFailures++
				if cfg.MaxConsecutiveFailures > 0 && consecutiveFailures >= cfg.MaxConsecutiveFailures {
					tripped = true
				}
				return nil, fmt.Errorf("compaction: %w", err)
			}
		}

		// Success — reset circuit breaker.
		consecutiveFailures = 0

		// Extract file ops from ALL compacted messages (history + turn prefix)
		allCompacted := msgs[:cut.firstKeptIndex]
		readFiles, modifiedFiles := extractFileOps(allCompacted)
		summary += formatFileOps(readFiles, modifiedFiles)

		cs := CompactionSummary{
			Summary:       summary,
			TokensBefore:  tokens,
			ReadFiles:     readFiles,
			ModifiedFiles: modifiedFiles,
			Timestamp:     time.Now(),
		}

		result := make([]agentcore.AgentMessage, 0, 1+len(toKeep))
		result = append(result, cs)
		result = append(result, toKeep...)

		if cfg.OnCompaction != nil {
			cfg.OnCompaction(CompactionInfo{
				TokensBefore:   tokens,
				TokensAfter:    EstimateTotal(result),
				MessagesBefore: len(msgs),
				MessagesAfter:  len(result),
				CompactedCount: len(allCompacted),
				KeptCount:      len(toKeep),
				IsSplitTurn:    needTurnPrefix,
				IsIncremental:  previousSummary != "",
				SummaryLen:     len([]rune(summary)),
				Duration:       time.Since(start),
			})
		}

		return result, nil
	}
}

// stripImageBlocks returns a copy of msgs with image content blocks removed.
// Text descriptions like "[image: screenshot.png]" are preserved if present.
// This reduces token usage during summarization — images can't be text-summarized.
func stripImageBlocks(msgs []agentcore.AgentMessage) []agentcore.AgentMessage {
	out := make([]agentcore.AgentMessage, 0, len(msgs))
	for _, m := range msgs {
		msg, ok := m.(agentcore.Message)
		if !ok {
			out = append(out, m)
			continue
		}
		hasImage := false
		for _, b := range msg.Content {
			if b.Type == agentcore.ContentImage {
				hasImage = true
				break
			}
		}
		if !hasImage {
			out = append(out, m)
			continue
		}
		// Filter out image blocks, keep everything else.
		filtered := make([]agentcore.ContentBlock, 0, len(msg.Content))
		for _, b := range msg.Content {
			if b.Type == agentcore.ContentImage {
				// Replace with placeholder text so the summary knows an image was here.
				filtered = append(filtered, agentcore.TextBlock("[image content omitted for summarization]"))
				continue
			}
			filtered = append(filtered, b)
		}
		cp := msg
		cp.Content = filtered
		out = append(out, cp)
	}
	return out
}

// cutResult holds the result of findCutPoint, including turn split information.
type cutResult struct {
	// firstKeptIndex is the index of the first message to keep.
	firstKeptIndex int
	// turnStartIndex is the index where the current turn starts, or -1 if
	// the cut is at a turn boundary (user message).
	turnStartIndex int
	// isSplitTurn is true when the cut falls in the middle of a turn.
	// In this case, msgs[turnStartIndex:firstKeptIndex] is the turn prefix
	// that needs a separate summary.
	isSplitTurn bool
}

// findCutPoint walks backwards from the end, accumulating tokens until
// keepTokens is reached. Returns the cut result with turn-awareness.
//
// Rules:
//   - Never cut between an assistant message (with tool calls) and its tool results
//   - Prefer cutting at user message boundaries
//   - Detect split turns and report the turn start index
func findCutPoint(msgs []agentcore.AgentMessage, keepTokens int) cutResult {
	if len(msgs) == 0 {
		return cutResult{}
	}

	accumulated := 0
	cutIndex := len(msgs) // start past end

	// Walk backwards
	for i := len(msgs) - 1; i >= 0; i-- {
		accumulated += EstimateTokens(msgs[i])
		if accumulated >= keepTokens {
			cutIndex = i
			break
		}
	}

	// If we couldn't accumulate enough, keep everything
	if cutIndex >= len(msgs) {
		return cutResult{}
	}

	// Align to a valid cut point: walk forward to find a user message boundary
	// Never split tool pair (assistant with toolCalls + following tool results)
	for cutIndex < len(msgs) {
		msg := msgs[cutIndex]
		if m, ok := msg.(agentcore.Message); ok {
			// Don't cut at a tool result — it belongs to the previous assistant
			if m.Role == agentcore.RoleTool {
				cutIndex++
				continue
			}
			// Good cut point: user message
			if m.Role == agentcore.RoleUser {
				break
			}
			// Assistant message with tool calls: skip past all its tool results
			if m.Role == agentcore.RoleAssistant && m.HasToolCalls() {
				cutIndex++
				for cutIndex < len(msgs) {
					if next, ok := msgs[cutIndex].(agentcore.Message); ok && next.Role == agentcore.RoleTool {
						cutIndex++
					} else {
						break
					}
				}
				continue
			}
			// Assistant without tool calls — valid cut point
			break
		}
		// CompactionSummary or other custom type — valid cut point
		break
	}

	// Safety: don't compact everything
	if cutIndex >= len(msgs) {
		return cutResult{}
	}

	// Detect split turn: if cut is not at a user message, find the turn start
	isSplitTurn := false
	turnStartIndex := -1
	if m, ok := msgs[cutIndex].(agentcore.Message); !ok || m.Role != agentcore.RoleUser {
		// Walk backwards from cutIndex to find the user message that started this turn
		for i := cutIndex - 1; i >= 0; i-- {
			if um, ok := msgs[i].(agentcore.Message); ok && um.Role == agentcore.RoleUser {
				turnStartIndex = i
				isSplitTurn = true
				break
			}
		}
	}

	return cutResult{
		firstKeptIndex: cutIndex,
		turnStartIndex: turnStartIndex,
		isSplitTurn:    isSplitTurn,
	}
}

// extractFileOps scans messages for tool calls and extracts file paths.
func extractFileOps(msgs []agentcore.AgentMessage) (readFiles, modifiedFiles []string) {
	readSet := make(map[string]struct{})
	modifiedSet := make(map[string]struct{})

	for _, m := range msgs {
		msg, ok := m.(agentcore.Message)
		if !ok || msg.Role != agentcore.RoleAssistant {
			continue
		}
		for _, tc := range msg.ToolCalls() {
			path := extractPathArg(tc.Args)
			if path == "" {
				continue
			}
			switch tc.Name {
			case "read":
				readSet[path] = struct{}{}
			case "write":
				modifiedSet[path] = struct{}{}
			case "edit":
				modifiedSet[path] = struct{}{}
			}
		}
	}

	// Read-only files: read but not modified
	for f := range readSet {
		if _, modified := modifiedSet[f]; !modified {
			readFiles = append(readFiles, f)
		}
	}
	for f := range modifiedSet {
		modifiedFiles = append(modifiedFiles, f)
	}

	slices.Sort(readFiles)
	slices.Sort(modifiedFiles)
	return
}

// extractPathArg extracts the "path" field from JSON tool args.
func extractPathArg(args json.RawMessage) string {
	var obj struct {
		Path string `json:"path"`
	}
	if json.Unmarshal(args, &obj) == nil {
		return obj.Path
	}
	return ""
}

// formatFileOps formats file operation lists as XML tags appended to the summary.
func formatFileOps(readFiles, modifiedFiles []string) string {
	if len(readFiles) == 0 && len(modifiedFiles) == 0 {
		return ""
	}
	var s string
	if len(readFiles) > 0 {
		s += "\n\n<read-files>\n" + strings.Join(readFiles, "\n") + "\n</read-files>"
	}
	if len(modifiedFiles) > 0 {
		s += "\n\n<modified-files>\n" + strings.Join(modifiedFiles, "\n") + "\n</modified-files>"
	}
	return s
}

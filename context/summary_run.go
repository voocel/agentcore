package context

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"slices"
	"strings"
	"time"

	"github.com/voocel/agentcore"
)

const (
	defaultReserveTokens = 16384
	// Bounds for the dynamically sized verbatim tail.
	minKeepRecentTokens = 2000
	maxKeepRecentTokens = 20000
)

// SummaryInfo holds details about a completed compaction for observability.
type SummaryInfo struct {
	TokensBefore   int
	TokensAfter    int
	MessagesBefore int
	MessagesAfter  int
	CompactedCount int // number of messages summarized
	KeptCount      int // number of messages retained verbatim
	IsSplitTurn    bool
	IsIncremental  bool          // updated an existing summary
	SummaryLen     int           // summary length in runes
	Duration       time.Duration // wall time including LLM calls
	ReadFiles      []string      // files read during compacted conversation
	ModifiedFiles  []string      // files modified during compacted conversation
}

// summaryRunConfig configures one summary compaction execution.
type summaryRunConfig struct {
	Model            agentcore.ChatModel
	ContextWindow    int
	ReserveTokens    int
	KeepRecentTokens int
	// Custom prompts — empty means use built-in defaults.
	SystemPrompt        string
	SummaryPrompt       string
	UpdateSummaryPrompt string
	TurnPrefixPrompt    string
	// Fork is nil when the summary must use a standalone prompt.
	Fork *agentcore.LLMPrefix
}

func runSummaryCompaction(ctx context.Context, cfg summaryRunConfig, msgs []agentcore.AgentMessage, stripImages bool) ([]agentcore.AgentMessage, *SummaryInfo, error) {
	cut := findCutPoint(msgs, cfg.KeepRecentTokens)
	if cut.firstKeptIndex <= 0 {
		return msgs, nil, nil
	}

	start := time.Now()

	historyEnd := cut.firstKeptIndex
	if cut.isSplitTurn && cut.turnStartIndex >= 0 {
		historyEnd = cut.turnStartIndex
	}

	toKeep := msgs[cut.firstKeptIndex:]

	previousSummary, history := splitPreviousSummary(msgs[:historyEnd])

	var turnPrefix []agentcore.AgentMessage
	if cut.isSplitTurn && cut.turnStartIndex >= 0 {
		turnPrefix = msgs[cut.turnStartIndex:cut.firstKeptIndex]
	}

	// A split-turn prefix is new history even when only the prior summary precedes it.
	if len(history) == 0 && len(turnPrefix) == 0 && previousSummary != "" {
		return msgs, nil, nil
	}

	prompts := summaryPrompts{
		System:     cfg.SystemPrompt,
		Summary:    cfg.SummaryPrompt,
		Update:     cfg.UpdateSummaryPrompt,
		TurnPrefix: cfg.TurnPrefixPrompt,
	}

	summary, err := forkedSummary(ctx, cfg, prompts, msgs)
	if err != nil {
		return nil, nil, fmt.Errorf("compaction: %w", err)
	}
	if summary == "" {
		summary, err = standaloneSummary(ctx, cfg, prompts, history, previousSummary, turnPrefix, stripImages)
		if err != nil {
			return nil, nil, err
		}
	}

	allCompacted := msgs[:cut.firstKeptIndex]
	readFiles, modifiedFiles := extractFileOps(allCompacted)
	summary += formatFileOps(readFiles, modifiedFiles)

	tokensBefore := EstimateTotal(msgs)
	cs := ContextSummary{
		Summary:       summary,
		TokensBefore:  tokensBefore,
		ReadFiles:     readFiles,
		ModifiedFiles: modifiedFiles,
		Timestamp:     time.Now(),
	}

	result := make([]agentcore.AgentMessage, 0, 1+len(toKeep))
	result = append(result, cs)
	result = append(result, toKeep...)

	info := &SummaryInfo{
		TokensBefore:   tokensBefore,
		TokensAfter:    EstimateTotal(result),
		MessagesBefore: len(msgs),
		MessagesAfter:  len(result),
		CompactedCount: len(allCompacted),
		KeptCount:      len(toKeep),
		IsSplitTurn:    len(turnPrefix) > 0,
		IsIncremental:  previousSummary != "",
		SummaryLen:     len([]rune(summary)),
		Duration:       time.Since(start),
		ReadFiles:      readFiles,
		ModifiedFiles:  modifiedFiles,
	}
	return result, info, nil
}

// splitPreviousSummary separates the checkpoint from new standalone history.
func splitPreviousSummary(msgs []agentcore.AgentMessage) (string, []agentcore.AgentMessage) {
	var previous string
	out := msgs[:0:0]
	for _, m := range msgs {
		if cs, ok := m.(ContextSummary); ok {
			previous = cs.Summary
			continue
		}
		out = append(out, m)
	}
	return previous, out
}

// forkedSummary returns "" when the caller should use standaloneSummary.
func forkedSummary(ctx context.Context, cfg summaryRunConfig, prompts summaryPrompts, view []agentcore.AgentMessage) (string, error) {
	// Prompt caches are model-specific.
	if cfg.Fork == nil || !sameModel(cfg.Fork.Model, cfg.Model) {
		return "", nil
	}

	// Preserve the whole view and thinking options so the cached prefix remains
	// byte-stable. The retained tail is therefore also present in the summary input.
	summary, err := generateSummary(ctx, cfg.Model, prompts, cfg.Fork, view, "",
		agentcore.WithMaxTokens(int(float64(cfg.ReserveTokens)*0.8)))
	if errors.Is(err, errEmptySummary) {
		return "", nil
	}
	return summary, err
}

// sameModel compares model instances without panicking on non-comparable values.
func sameModel(a, b agentcore.ChatModel) bool {
	if a == nil || b == nil {
		return false
	}
	ta, tb := reflect.TypeOf(a), reflect.TypeOf(b)
	return ta == tb && ta.Comparable() && a == b
}

// standaloneSummary uses a dedicated prompt with thinking disabled.
func standaloneSummary(ctx context.Context, cfg summaryRunConfig, prompts summaryPrompts, history []agentcore.AgentMessage, previousSummary string, turnPrefix []agentcore.AgentMessage, stripImages bool) (string, error) {
	if stripImages {
		history = stripImageBlocks(history)
		turnPrefix = stripImageBlocks(turnPrefix)
	}

	summary := "No prior history."
	if len(history) > 0 {
		var err error
		summary, err = generateSummary(ctx, cfg.Model, prompts, nil, history, previousSummary,
			agentcore.WithMaxTokens(int(float64(cfg.ReserveTokens)*0.8)),
			agentcore.WithThinking(agentcore.ThinkingOff))
		if err != nil {
			return "", fmt.Errorf("compaction: %w", err)
		}
	}
	if len(turnPrefix) == 0 {
		return summary, nil
	}

	// Run sequentially. ChatModel does not promise concurrent Generate safety,
	// so split-turn summarization must not assume shared model instances are
	// goroutine-safe.
	prefixSummary, err := generateTurnPrefixSummary(ctx, cfg.Model, prompts, turnPrefix,
		agentcore.WithMaxTokens(int(float64(cfg.ReserveTokens)*0.5)),
		agentcore.WithThinking(agentcore.ThinkingOff))
	if err != nil {
		return "", fmt.Errorf("compaction turn prefix: %w", err)
	}
	if prefixSummary != "" {
		summary += "\n\n---\n\n**Turn Context (split turn):**\n\n" + prefixSummary
	}
	return summary, nil
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
		// ContextSummary or other custom type — valid cut point
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

// extractPathArg extracts a file path from JSON tool args. Accepts both
// "file_path" (preferred, matches edit/read/write) and "path" (used by
// glob/grep/ls). file_path wins when both are present.
func extractPathArg(args json.RawMessage) string {
	var obj struct {
		FilePath string `json:"file_path"`
		Path     string `json:"path"`
	}
	if json.Unmarshal(args, &obj) == nil {
		if obj.FilePath != "" {
			return obj.FilePath
		}
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

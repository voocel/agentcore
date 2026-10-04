package compact

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"slices"
	"strings"
	"unicode/utf8"

	"github.com/voocel/agentcore"
	"github.com/voocel/litellm"
)

// checkpointFormat is the summary layout both summarization paths ask for.
const checkpointFormat = `Use this EXACT format:

## Goal
[What is the user trying to accomplish? Can be multiple items if the session covers different tasks.]

## Constraints & Preferences
- [Constraints, preferences, or requirements the user stated in the conversation, or "(none)"]

## Progress
### Done
- [x] [Completed tasks/changes]

### In Progress
- [ ] [Current work]

### Blocked
- [Issues preventing progress, if any, or "(none)"]

## Key Decisions
- **[Decision]**: [Brief rationale]

## Next Steps
1. [Ordered list of what should happen next]

## Critical Context
- [Any data, file paths, function names, or references needed to continue]

Keep each section concise. Preserve exact file paths, function names, and error messages.`

// forkInstruction ends the conversation's own request. Providers accept it
// only as a user message, so it says that it is not one: a summarizer that
// takes it for the user records "stop and write a checkpoint" as a user
// constraint, every later summary preserves it, and the model that continues
// from the summary stops.
const forkInstruction = `[Context compaction request from the agent harness. This is not a message from the user and not part of the conversation.]

Stop working on the task: do not call any tool, continue the conversation, or answer anything asked in it. Write a checkpoint of the conversation above that another model will continue the work from. If the conversation starts from an earlier checkpoint in <context-summary> tags, carry its content forward, updated by what happened since. Leave this request out of the checkpoint: it is neither a user instruction nor a constraint.

Think briefly inside <analysis>...</analysis>, then output the checkpoint inside <summary>...</summary>. The <summary> tags are required.

` + checkpointFormat

const summarySystemPrompt = `You are a context summarization assistant. Your task is to read a conversation between a user and an AI coding assistant, then produce a structured summary following the exact format specified.

Do NOT continue the conversation. Do NOT respond to any questions in the conversation.

First think briefly inside <analysis>...</analysis>. Then output the final checkpoint inside <summary>...</summary>.`

const summaryPrompt = `The conversation above is to be summarized. Create a structured context checkpoint summary that another LLM will use to continue the work.

` + checkpointFormat

const updateSummaryPrompt = `The conversation above is NEW history to incorporate into the existing summary in <previous-summary> tags.

Update the existing structured summary with new information. RULES:
- PRESERVE all existing goals, ADD new ones if the task expanded
- PRESERVE existing constraints, ADD new ones discovered
- UPDATE the Progress section: move items from "In Progress" to "Done" when completed
- UPDATE "Blocked" with any new issues, remove resolved ones
- UPDATE "Next Steps" based on what was accomplished
- If something is no longer relevant, you may remove it

` + checkpointFormat

// forkSummary extends the conversation's call for the history to replace
// with forkInstruction, so the provider serves that history from the prompt
// cache, adding what the call used to used. It returns "" when the request
// no longer fits or the model answered without a tagged summary, leaving the
// transcript path to take over.
func forkSummary(ctx context.Context, call agentcore.Call, used *litellm.Usage) (string, error) {
	req := call.Request
	req.Messages = append(slices.Clip(req.Messages), litellm.UserText(forkInstruction))
	resp, err := call.Client.Chat(ctx, req)
	if litellm.ErrorTypeOf(err) == litellm.ErrorTypeContextOverflow {
		return "", nil
	}
	if err != nil {
		return "", fmt.Errorf("summarize: %w", err)
	}
	used.Add(resp.Usage)
	// Only tagged output counts: an ordinary task reply must not become the
	// checkpoint.
	return extractTaggedBlock(stripAnalysisBlock(resp.Text()), "summary"), nil
}

// standaloneSummary summarizes a plain-text transcript of history with the
// conversation's model, without tools and with the vendor's default
// thinking, folding in the previous summary, and adds what its calls used to
// used. When the transcript itself overflows, the oldest user turns are
// dropped until it fits.
func standaloneSummary(ctx context.Context, call agentcore.Call, history []agentcore.Message, previous string, used *litellm.Usage) (string, error) {
	instruction := summaryPrompt
	if previous != "" {
		instruction = "<previous-summary>\n" + previous + "\n</previous-summary>\n\n" + updateSummaryPrompt
	}
	req := call.Request
	req.Tools, req.ToolChoice, req.ResponseFormat, req.Thinking = nil, nil, nil, nil
	history = stripImageBlocks(history)
	for {
		req.Messages = []litellm.Message{
			litellm.System(summarySystemPrompt),
			litellm.UserText("<conversation>\n" + serializeConversation(history) + "\n</conversation>\n\n" + instruction),
		}
		resp, err := call.Client.Chat(ctx, req)
		if err == nil {
			used.Add(resp.Usage)
			if summary := extractStoredSummary(resp.Text()); summary != "" {
				return summary, nil
			}
			return "", errors.New("summarize: empty summary")
		}
		if litellm.ErrorTypeOf(err) != litellm.ErrorTypeContextOverflow {
			return "", fmt.Errorf("summarize: %w", err)
		}
		next := truncateOldestUserGroups(history, 0.2)
		if len(next) == len(history) {
			return "", fmt.Errorf("summarize: %w", err)
		}
		history = next
	}
}

func extractStoredSummary(text string) string {
	text = stripAnalysisBlock(strings.TrimSpace(text))
	if summary := extractTaggedBlock(text, "summary"); summary != "" {
		return summary
	}
	return strings.TrimSpace(text)
}

func stripAnalysisBlock(text string) string {
	start := strings.Index(text, "<analysis>")
	end := strings.Index(text, "</analysis>")
	if start < 0 || end < start {
		return text
	}
	return strings.TrimSpace(text[:start] + text[end+len("</analysis>"):])
}

func extractTaggedBlock(text, tag string) string {
	startTag, endTag := "<"+tag+">", "</"+tag+">"
	start := strings.Index(text, startTag)
	end := strings.Index(text, endTag)
	if start < 0 || end < start {
		return ""
	}
	return strings.TrimSpace(text[start+len(startTag) : end])
}

// truncateOldestUserGroups drops roughly the oldest fraction of user turns
// (of messages, with a single turn), keeping the result led by a user message
// as providers require.
func truncateOldestUserGroups(msgs []agentcore.Message, fraction float64) []agentcore.Message {
	var starts []int
	for i, m := range msgs {
		if m.Role == litellm.RoleUser {
			starts = append(starts, i)
		}
	}
	var result []agentcore.Message
	if len(starts) <= 1 {
		drop := int(math.Ceil(float64(len(msgs)) * fraction))
		if drop <= 0 || drop >= len(msgs) {
			return msgs
		}
		result = msgs[drop:]
	} else {
		groups := min(max(int(math.Ceil(float64(len(starts))*fraction)), 1), len(starts)-1)
		result = msgs[starts[groups]:]
	}
	if result[0].Role != litellm.RoleUser {
		result = append([]agentcore.Message{agentcore.UserText("[Earlier context truncated for summarization]")}, result...)
	}
	return result
}

// stripImageBlocks replaces images with a placeholder: they cannot be
// summarized as text and only cost tokens.
func stripImageBlocks(msgs []agentcore.Message) []agentcore.Message {
	out := make([]agentcore.Message, len(msgs))
	for i, m := range msgs {
		m.Blocks = withoutImages(m.Blocks)
		out[i] = m
	}
	return out
}

func withoutImages(blocks []litellm.Block) []litellm.Block {
	out := make([]litellm.Block, len(blocks))
	for i, block := range blocks {
		switch b := block.(type) {
		case litellm.ImageBlock:
			block = litellm.Text("[image content omitted for summarization]")
		case litellm.ToolResultBlock:
			b.Content = withoutImages(b.Content)
			block = b
		}
		out[i] = block
	}
	return out
}

// truncateForSummary caps s at roughly max bytes, backing up to a rune
// boundary: a split multi-byte rune makes the request invalid UTF-8, which
// providers reject on every retry.
func truncateForSummary(s string, max int) string {
	if len(s) <= max {
		return s
	}
	cut := max
	for cut > 0 && !utf8.RuneStart(s[cut]) {
		cut--
	}
	return s[:cut] + "..."
}

// formatArgsKeyValue renders JSON tool args as key=value pairs, which cost
// fewer tokens than raw JSON.
func formatArgsKeyValue(raw string) string {
	var obj map[string]any
	if json.Unmarshal([]byte(raw), &obj) != nil {
		return truncateForSummary(raw, 197)
	}
	var pairs []string
	for k, v := range obj {
		pairs = append(pairs, k+"="+truncateForSummary(fmt.Sprintf("%v", v), 97))
	}
	slices.Sort(pairs)
	return strings.Join(pairs, ", ")
}

// serializeConversation renders history as readable text for summarization.
func serializeConversation(msgs []agentcore.Message) string {
	var parts []string
	for _, m := range msgs {
		switch m.Role {
		case litellm.RoleUser:
			if text := m.Text(); text != "" {
				parts = append(parts, "[User]: "+text)
			}
		case litellm.RoleAssistant:
			if thinking := m.Reasoning(); thinking != "" {
				parts = append(parts, "[Assistant thinking]: "+thinking)
			}
			if text := m.Text(); text != "" {
				parts = append(parts, "[Assistant]: "+text)
			}
			if calls := m.ToolCalls(); len(calls) > 0 {
				rendered := make([]string, len(calls))
				for i, tc := range calls {
					rendered[i] = tc.Name + "(" + formatArgsKeyValue(tc.Arguments) + ")"
				}
				parts = append(parts, "[Assistant tool calls]: "+strings.Join(rendered, "; "))
			}
		case litellm.RoleTool:
			if content := truncateForSummary(m.Text(), 497); content != "" {
				parts = append(parts, "[Tool result]: "+content)
			}
		}
	}
	return strings.Join(parts, "\n\n")
}

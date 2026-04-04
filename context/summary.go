package context

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"strings"

	"github.com/voocel/agentcore"
)

const summarySystemPrompt = `You are a context summarization assistant. Your task is to read a conversation between a user and an AI coding assistant, then produce a structured summary following the exact format specified.

Do NOT continue the conversation. Do NOT respond to any questions in the conversation.

First think briefly inside <analysis>...</analysis>. Then output the final checkpoint inside <summary>...</summary>.`

const summaryPrompt = `The messages above are a conversation to summarize. Create a structured context checkpoint summary that another LLM will use to continue the work.

Use this EXACT format:

## Goal
[What is the user trying to accomplish? Can be multiple items if the session covers different tasks.]

## Constraints & Preferences
- [Any constraints, preferences, or requirements mentioned by user]
- [Or "(none)" if none were mentioned]

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

const updateSummaryPrompt = `The messages above are NEW conversation messages to incorporate into the existing summary provided in <previous-summary> tags.

Update the existing structured summary with new information. RULES:
- PRESERVE all existing goals, ADD new ones if the task expanded
- PRESERVE existing constraints, ADD new ones discovered
- UPDATE the Progress section: move items from "In Progress" to "Done" when completed
- UPDATE "Blocked" with any new issues, remove resolved ones
- UPDATE "Next Steps" based on what was accomplished
- PRESERVE exact file paths, function names, and error messages
- If something is no longer relevant, you may remove it

Use this EXACT format:

## Goal
[What is the user trying to accomplish? Can be multiple items.]

## Constraints & Preferences
- [Any constraints, preferences, or requirements mentioned by user]

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
- [Any data, file paths, function names, or references needed to continue]`

const turnPrefixPrompt = `This is the PREFIX of a conversation turn that was too large to keep intact. The SUFFIX (recent work) is retained separately.

Summarize the prefix to provide context for the retained suffix:

## Original Request
[What did the user ask for in this turn?]

## Early Progress
- [Key decisions and work done in the prefix]

## Context for Suffix
- [Information needed to understand the retained recent work]

Be concise. Focus on what's needed to understand the kept suffix.`

// generateTurnPrefixSummary generates a summary for the prefix portion of a split turn.
func generateTurnPrefixSummary(ctx context.Context, model agentcore.ChatModel, msgs []agentcore.AgentMessage, opts ...agentcore.CallOption) (string, error) {
	conversation := serializeConversation(msgs)
	if conversation == "" {
		return "", nil
	}

	userPrompt := "<conversation>\n" + conversation + "\n</conversation>\n\n" + turnPrefixPrompt

	resp, err := model.Generate(ctx, []agentcore.Message{
		agentcore.SystemMsg(summarySystemPrompt),
		agentcore.UserMsg(userPrompt),
	}, nil, opts...)
	if err != nil {
		return "", fmt.Errorf("turn prefix summarization failed: %w", err)
	}
	return extractStoredSummary(resp.Message.TextContent()), nil
}

// generateSummary calls the ChatModel to produce a conversation summary.
// If previousSummary is non-empty, uses incremental update prompt.
func generateSummary(ctx context.Context, model agentcore.ChatModel, msgs []agentcore.AgentMessage, previousSummary string, opts ...agentcore.CallOption) (string, error) {
	const maxRetries = 3
	current := append([]agentcore.AgentMessage(nil), msgs...)
	var lastErr error
	for attempt := 0; attempt < maxRetries; attempt++ {
		summary, err := generateSummaryOnce(ctx, model, current, previousSummary, opts...)
		if err == nil {
			return summary, nil
		}
		lastErr = err
		if !agentcore.IsContextOverflow(err) {
			return "", err
		}
		next := truncateOldestUserGroups(current, 0.2)
		if len(next) == 0 || len(next) == len(current) {
			break
		}
		current = next
	}
	if lastErr != nil {
		return "", lastErr
	}
	return "", fmt.Errorf("summarization failed")
}

func generateSummaryOnce(ctx context.Context, model agentcore.ChatModel, msgs []agentcore.AgentMessage, previousSummary string, opts ...agentcore.CallOption) (string, error) {
	conversation := serializeConversation(msgs)
	if conversation == "" {
		return "", fmt.Errorf("no conversation content to summarize")
	}

	var userPrompt string
	if previousSummary != "" {
		userPrompt = "<conversation>\n" + conversation + "\n</conversation>\n\n" +
			"<previous-summary>\n" + previousSummary + "\n</previous-summary>\n\n" +
			updateSummaryPrompt
	} else {
		userPrompt = "<conversation>\n" + conversation + "\n</conversation>\n\n" +
			summaryPrompt
	}

	resp, err := model.Generate(ctx, []agentcore.Message{
		agentcore.SystemMsg(summarySystemPrompt),
		agentcore.UserMsg(userPrompt),
	}, nil, opts...)
	if err != nil {
		return "", fmt.Errorf("summarization failed: %w", err)
	}

	summary := extractStoredSummary(resp.Message.TextContent())
	if summary == "" {
		return "", fmt.Errorf("summarization returned empty content")
	}
	return summary, nil
}

func extractStoredSummary(text string) string {
	text = stripAnalysisBlock(strings.TrimSpace(text))
	if summary := extractTaggedBlock(text, "summary"); summary != "" {
		return strings.TrimSpace(summary)
	}
	return strings.TrimSpace(text)
}

func stripAnalysisBlock(text string) string {
	start := strings.Index(text, "<analysis>")
	end := strings.Index(text, "</analysis>")
	if start < 0 || end < 0 || end < start {
		return text
	}
	return strings.TrimSpace(text[:start] + text[end+len("</analysis>"):])
}

func extractTaggedBlock(text, tag string) string {
	startTag := "<" + tag + ">"
	endTag := "</" + tag + ">"
	start := strings.Index(text, startTag)
	end := strings.Index(text, endTag)
	if start < 0 || end < 0 || end < start {
		return ""
	}
	start += len(startTag)
	return strings.TrimSpace(text[start:end])
}

func truncateOldestUserGroups(msgs []agentcore.AgentMessage, fraction float64) []agentcore.AgentMessage {
	if len(msgs) == 0 {
		return nil
	}
	var starts []int
	for i, msg := range msgs {
		m, ok := msg.(agentcore.Message)
		if ok && m.Role == agentcore.RoleUser {
			starts = append(starts, i)
		}
	}
	var result []agentcore.AgentMessage
	if len(starts) <= 1 {
		drop := int(math.Ceil(float64(len(msgs)) * fraction))
		if drop <= 0 || drop >= len(msgs) {
			return msgs
		}
		result = append([]agentcore.AgentMessage(nil), msgs[drop:]...)
	} else {
		dropGroups := int(math.Ceil(float64(len(starts)) * fraction))
		if dropGroups <= 0 {
			dropGroups = 1
		}
		if dropGroups >= len(starts) {
			dropGroups = len(starts) - 1
		}
		cut := starts[dropGroups]
		result = append([]agentcore.AgentMessage(nil), msgs[cut:]...)
	}

	// Ensure the truncated result starts with a user message.
	// LLM APIs reject conversations that start with an assistant message.
	if len(result) > 0 {
		if m, ok := result[0].(agentcore.Message); ok && m.Role != agentcore.RoleUser {
			result = append([]agentcore.AgentMessage{
				agentcore.UserMsg("[Earlier context truncated for summarization]"),
			}, result...)
		}
	}
	return result
}

// formatArgsKeyValue formats JSON tool args as key=value pairs.
// More token-efficient than raw JSON for LLM summarization input.
func formatArgsKeyValue(raw json.RawMessage) string {
	var obj map[string]any
	if json.Unmarshal(raw, &obj) != nil {
		s := string(raw)
		if len(s) > 200 {
			return s[:197] + "..."
		}
		return s
	}

	var pairs []string
	for k, v := range obj {
		s := fmt.Sprintf("%v", v)
		if len(s) > 100 {
			s = s[:97] + "..."
		}
		pairs = append(pairs, k+"="+s)
	}
	return strings.Join(pairs, ", ")
}

// serializeConversation converts messages to readable text for LLM input.
func serializeConversation(msgs []agentcore.AgentMessage) string {
	var parts []string

	for _, m := range msgs {
		switch v := m.(type) {
		case agentcore.Message:
			switch v.Role {
			case agentcore.RoleUser:
				if text := v.TextContent(); text != "" {
					parts = append(parts, "[User]: "+text)
				}
			case agentcore.RoleAssistant:
				if thinking := v.ThinkingContent(); thinking != "" {
					parts = append(parts, "[Assistant thinking]: "+thinking)
				}
				if text := v.TextContent(); text != "" {
					parts = append(parts, "[Assistant]: "+text)
				}
				if toolCalls := v.ToolCalls(); len(toolCalls) > 0 {
					var calls []string
					for _, tc := range toolCalls {
						calls = append(calls, tc.Name+"("+formatArgsKeyValue(tc.Args)+")")
					}
					parts = append(parts, "[Assistant tool calls]: "+strings.Join(calls, "; "))
				}
			case agentcore.RoleTool:
				content := v.TextContent()
				if len(content) > 500 {
					content = content[:497] + "..."
				}
				if content != "" {
					parts = append(parts, "[Tool result]: "+content)
				}
			}
		case ContextSummary:
			parts = append(parts, "[Previous summary]: "+v.Summary)
		}
	}

	return strings.Join(parts, "\n\n")
}

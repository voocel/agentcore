package llm

import (
	"regexp"
	"strings"

	"github.com/voocel/agentcore"
)

var validToolCallID = regexp.MustCompile(`^[a-zA-Z0-9_-]+$`)

const maxToolCallIDLength = 64

// TransformMessages normalizes a message sequence for a target provider.
// Use this when switching models mid-conversation to avoid provider rejections.
//
// Two-pass algorithm:
//  1. Normalize tool call IDs (truncate >64 chars, sanitize to [a-zA-Z0-9_-]),
//     handle thinking blocks based on target provider.
//  2. Apply ID mapping to tool results, insert synthetic results for orphaned tool calls.
func TransformMessages(messages []agentcore.Message, targetProvider string) []agentcore.Message {
	if len(messages) == 0 {
		return nil
	}

	idMap := make(map[string]string) // oldID → newID
	result := make([]agentcore.Message, 0, len(messages))

	// Pass 1: normalize IDs and thinking blocks, skip incomplete messages
	for _, msg := range messages {
		if msg.Role == agentcore.RoleAssistant &&
			(msg.StopReason == agentcore.StopReasonError || msg.StopReason == agentcore.StopReasonAborted) {
			continue
		}
		msg = transformContent(msg, targetProvider, idMap)
		result = append(result, msg)
	}

	// Pass 2: apply ID mapping to tool results, handle orphans
	result = applyIDMapping(result, idMap)

	return result
}

// transformContent normalizes a single message's content blocks.
func transformContent(msg agentcore.Message, targetProvider string, idMap map[string]string) agentcore.Message {
	newContent := make([]agentcore.ContentBlock, 0, len(msg.Content))

	for _, block := range msg.Content {
		switch block.Type {
		case agentcore.ContentThinking:
			if block.Thinking == "" || strings.TrimSpace(block.Thinking) == "" {
				continue // drop empty thinking blocks
			}
			if targetProvider == "anthropic" {
				newContent = append(newContent, block)
			} else {
				// Convert thinking to wrapped text for non-Anthropic targets
				newContent = append(newContent, agentcore.TextBlock("<thinking>\n"+block.Thinking+"\n</thinking>"))
			}

		case agentcore.ContentToolCall:
			if block.ToolCall != nil {
				tc := *block.ToolCall
				newID := normalizeToolCallID(tc.ID)
				if newID != tc.ID {
					idMap[tc.ID] = newID
					tc.ID = newID
				}
				newContent = append(newContent, agentcore.ToolCallBlock(tc))
			}

		default:
			newContent = append(newContent, block)
		}
	}

	msg.Content = newContent
	return msg
}

// normalizeToolCallID ensures the ID matches ^[a-zA-Z0-9_-]+$ and is <= 64 chars.
func normalizeToolCallID(id string) string {
	if len(id) <= maxToolCallIDLength && validToolCallID.MatchString(id) {
		return id
	}

	var sb strings.Builder
	for _, r := range id {
		if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') || r == '_' || r == '-' {
			sb.WriteRune(r)
		}
	}
	sanitized := sb.String()

	if len(sanitized) > maxToolCallIDLength {
		sanitized = sanitized[:maxToolCallIDLength]
	}
	if sanitized == "" {
		sanitized = "tc_unknown"
	}
	return sanitized
}

// applyIDMapping updates tool result IDs and inserts synthetic results for orphans.
func applyIDMapping(messages []agentcore.Message, idMap map[string]string) []agentcore.Message {
	result := make([]agentcore.Message, 0, len(messages))

	for i, msg := range messages {
		// Remap tool result IDs
		if msg.Role == agentcore.RoleTool {
			if oldID, ok := msg.Metadata["tool_call_id"].(string); ok {
				if newID, remapped := idMap[oldID]; remapped {
					meta := make(map[string]any, len(msg.Metadata))
					for k, v := range msg.Metadata {
						meta[k] = v
					}
					meta["tool_call_id"] = newID
					msg.Metadata = meta
				}
			}
		}
		result = append(result, msg)

		// After assistant messages with tool calls, check for orphans
		if msg.Role == agentcore.RoleAssistant {
			calls := msg.ToolCalls()
			if len(calls) > 0 {
				answered := collectToolResultIDs(messages, i+1)
				for _, call := range calls {
					if !answered[call.ID] {
						result = append(result, agentcore.ToolResultMsg(
							call.ID,
							[]byte(`"Tool result missing (message transform)."`),
							true,
						))
					}
				}
			}
		}
	}

	return result
}

// collectToolResultIDs gathers tool_call_id values from consecutive tool messages starting at index.
func collectToolResultIDs(messages []agentcore.Message, start int) map[string]bool {
	ids := make(map[string]bool)
	for i := start; i < len(messages); i++ {
		if messages[i].Role != agentcore.RoleTool {
			break
		}
		if id, ok := messages[i].Metadata["tool_call_id"].(string); ok {
			ids[id] = true
		}
	}
	return ids
}

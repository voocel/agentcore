package agentcore

import (
	"context"
	"slices"
)

// Attachment is a one-turn system-reminder block prepended to the last user
// message's content. Unlike Reminder (which becomes a trailing system message
// outside the cache marker), Attachment lives inside the conversation prefix
// — it modifies the user message content rather than the system prompt.
//
// This is the right place for dynamic system-level signals (plan mode entry /
// exit, mode transitions, one-shot notifications) that should NOT invalidate
// cached system blocks. Compared to overlaying the system prompt:
//   - system prompt stays byte-stable → SB1/SB2/SB3 cache entries unaffected
//   - the signal still reaches the model through the conversation prefix
//   - cost is "one more text block on the user turn", not "rewrite SB + history"
//
// Attachments are NOT persisted to the agent message history — they are
// per-turn projections applied just before the LLM call.
type Attachment struct {
	// Source identifies the generator. Same-source attachments in one turn
	// collapse: last write wins. Use this when one logical signal might be
	// emitted by multiple paths and you want only the latest.
	Source string
	// Content body. Wrapped in <system-reminder>...</system-reminder> as a
	// text content block prepended to the user message.
	Content string
}

// AttachmentGenerator produces attachments for the upcoming LLM call. Invoked
// once per turn, just before the LLM request is built. Returning nil or empty
// skips injection for that turn.
type AttachmentGenerator func(ctx context.Context, turn TurnInfo) []Attachment

// collectAttachments runs each generator in order, deduplicating by Source
// (last write wins). Empty content is dropped.
func collectAttachments(ctx context.Context, gens []AttachmentGenerator, turn TurnInfo) []Attachment {
	if len(gens) == 0 {
		return nil
	}
	var out []Attachment
	seen := make(map[string]int)
	for _, gen := range gens {
		if gen == nil {
			continue
		}
		for _, a := range gen(ctx, turn) {
			if a.Content == "" {
				continue
			}
			if a.Source != "" {
				if idx, ok := seen[a.Source]; ok {
					out[idx] = a
					continue
				}
				seen[a.Source] = len(out)
			}
			out = append(out, a)
		}
	}
	return out
}

// injectAttachments returns a copy of messages with attachment text blocks
// prepended to the last RoleUser message's content. The caller's slice and
// the original Message values are left untouched. Returns messages as-is if
// there is no user message to inject into.
func injectAttachments(messages []Message, attachments []Attachment) []Message {
	if len(attachments) == 0 {
		return messages
	}
	idx := -1
	for i := len(messages) - 1; i >= 0; i-- {
		if messages[i].Role == RoleUser {
			idx = i
			break
		}
	}
	if idx < 0 {
		return messages
	}

	prepend := make([]ContentBlock, 0, len(attachments))
	for _, a := range attachments {
		prepend = append(prepend, TextBlock("<system-reminder>\n"+a.Content+"\n</system-reminder>"))
	}

	out := slices.Clone(messages)
	target := out[idx]
	newContent := make([]ContentBlock, 0, len(prepend)+len(target.Content))
	newContent = append(newContent, prepend...)
	newContent = append(newContent, target.Content...)
	target.Content = newContent
	out[idx] = target
	return out
}

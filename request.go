package agentcore

import (
	"github.com/voocel/litellm"
	"github.com/voocel/litellm/catalog"
)

// Model is the model a run calls.
type Model struct {
	Client *litellm.Client
	// Request is the template of every call: the model's name and settings,
	// such as MaxTokens, Thinking and ProviderOptions. The loop sets its
	// Messages and Tools.
	Request litellm.Request
	// Pricing, if set, prices each response's usage.
	Pricing *catalog.Pricing
}

// Call is a model call exactly as the loop makes it. A side call that
// extends its request, such as for a summary, reads the loop's prompt cache.
type Call struct {
	Client  *litellm.Client
	Request litellm.Request
}

// interruptedResult answers a tool call whose result was never recorded,
// such as when the process stopped while the tool ran.
const interruptedResult = "Tool execution was interrupted before its result was recorded. Its outcome and side effects are unknown; inspect the current state before retrying."

// BuildCall builds the call the loop makes next for history: the system
// prompt, the history as the model reads it, with the cache breakpoint, and
// the tools on offer.
func BuildCall(cfg Config, history []Message) Call {
	var msgs []litellm.Message
	if len(cfg.System) > 0 {
		msgs = append(msgs, litellm.Message{Role: litellm.RoleSystem, Blocks: cfg.System})
	}
	msgs = append(msgs, modelMessages(history)...)
	if cfg.Cache != nil {
		markCache(msgs, cfg.Cache)
	}
	req := cfg.Model.Request
	req.Messages = msgs
	req.Tools = toolSpecs(cfg.Tools, history)
	return Call{Client: cfg.Model.Client, Request: req}
}

// modelMessages is history as the model reads it. Responses that failed or
// were aborted, and those that said nothing, are left out; a reasoning-only
// response is too, as strict providers reject one. Every tool call is
// answered, and every tool result answers a call.
func modelMessages(history []Message) []litellm.Message {
	var out []litellm.Message
	calls := map[string]bool{}
	var unanswered []string
	answer := func() {
		for _, id := range unanswered {
			out = append(out, ToolResult(id, ErrorResult(interruptedResult)).wire())
		}
		unanswered = nil
	}
	for _, m := range history {
		if m.Role != litellm.RoleTool {
			answer()
		}
		switch m.Role {
		case litellm.RoleAssistant:
			if m.Stop == StopError || m.Stop == StopAborted || !speaks(m) {
				continue
			}
			for _, call := range m.ToolCalls() {
				calls[call.ID] = true
				unanswered = append(unanswered, call.ID)
			}
		case litellm.RoleTool:
			result, _ := m.ToolResult()
			if !calls[result.ToolUseID] {
				continue
			}
			delete(calls, result.ToolUseID)
			for i, id := range unanswered {
				if id == result.ToolUseID {
					unanswered = append(unanswered[:i], unanswered[i+1:]...)
					break
				}
			}
		}
		out = append(out, m.wire())
	}
	answer()
	return out
}

// speaks reports whether a response says or does anything: reasoning alone
// does not.
func speaks(m Message) bool {
	for _, b := range m.Blocks {
		if _, ok := b.(litellm.ReasoningBlock); !ok {
			return true
		}
	}
	return false
}

func (m Message) wire() litellm.Message {
	return litellm.Message{Role: m.Role, Blocks: m.Blocks}
}

// markCache places a cache breakpoint after the last message but the system
// prompt, so each call of a tool loop reads the one before back from the
// cache. Reasoning blocks cannot carry a breakpoint.
func markCache(msgs []litellm.Message, cache *litellm.CacheControl) {
	last := len(msgs) - 1
	if last < 0 || msgs[last].Role == litellm.RoleSystem {
		return
	}
	blocks := append([]litellm.Block(nil), msgs[last].Blocks...)
	for i := len(blocks) - 1; i >= 0; i-- {
		if b, ok := withCache(blocks[i], cache); ok {
			blocks[i] = b
			msgs[last].Blocks = blocks
			return
		}
	}
}

func withCache(block litellm.Block, cache *litellm.CacheControl) (litellm.Block, bool) {
	c := *cache
	switch b := block.(type) {
	case litellm.TextBlock:
		b.Cache = &c
		return b, true
	case litellm.ImageBlock:
		b.Cache = &c
		return b, true
	case litellm.ToolUseBlock:
		b.Cache = &c
		return b, true
	case litellm.ToolResultBlock:
		b.Cache = &c
		return b, true
	case litellm.ToolReferenceBlock:
		b.Cache = &c
		return b, true
	}
	return block, false
}

// toolSpecs are the tools on offer: all but the deferred ones that no tool
// reference in history names.
func toolSpecs(tools []Tool, history []Message) []litellm.Tool {
	var referenced map[string]bool
	specs := make([]litellm.Tool, 0, len(tools))
	for _, t := range tools {
		if t.Deferred {
			if referenced == nil {
				referenced = referencedTools(history)
			}
			if !referenced[t.Name] {
				continue
			}
		}
		schema, err := litellm.SchemaFrom(t.Schema)
		if err != nil {
			panic("agentcore: tool " + t.Name + ": " + err.Error())
		}
		specs = append(specs, litellm.Tool{Name: t.Name, Description: t.Description, Parameters: schema})
	}
	if len(specs) == 0 {
		return nil
	}
	return specs
}

func referencedTools(history []Message) map[string]bool {
	names := map[string]bool{}
	for _, m := range history {
		result, ok := m.ToolResult()
		if !ok {
			continue
		}
		for _, b := range result.Content {
			if ref, ok := b.(litellm.ToolReferenceBlock); ok {
				names[ref.ToolName] = true
			}
		}
	}
	return names
}

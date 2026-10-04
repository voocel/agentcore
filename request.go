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
	req.Tools = toolSpecs(cfg.Tools)
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

// markCache places the cache breakpoints: after the last message, where
// the call writes the history to the cache, and after the last message the
// call before sent, the one before the latest response, where that call
// wrote. A provider looks back from a breakpoint for an earlier write only
// so far, 20 blocks on Anthropic, so a turn adding many blocks, such as many
// parallel tool calls, would otherwise read nothing back.
func markCache(msgs []litellm.Message, cache *litellm.CacheControl) {
	last := len(msgs) - 1
	if last < 0 || msgs[last].Role == litellm.RoleSystem {
		return
	}
	mark(msgs, last, cache)
	for i := last; i > 0; i-- {
		if msgs[i].Role == litellm.RoleAssistant {
			if msgs[i-1].Role != litellm.RoleSystem {
				mark(msgs, i-1, cache)
			}
			return
		}
	}
}

// mark places a breakpoint after msgs[i], on its last block that takes one:
// reasoning blocks do not.
func mark(msgs []litellm.Message, i int, cache *litellm.CacheControl) {
	blocks := append([]litellm.Block(nil), msgs[i].Blocks...)
	for j := len(blocks) - 1; j >= 0; j-- {
		if b, ok := withCache(blocks[j], cache); ok {
			blocks[j] = b
			msgs[i].Blocks = blocks
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

// toolSpecs are the tools as litellm declares them; litellm offers the
// deferred ones once the history references them.
func toolSpecs(tools []Tool) []litellm.Tool {
	if len(tools) == 0 {
		return nil
	}
	specs := make([]litellm.Tool, len(tools))
	for i, t := range tools {
		schema, err := litellm.SchemaFrom(t.Schema)
		if err != nil {
			panic("agentcore: tool " + t.Name + ": " + err.Error())
		}
		specs[i] = litellm.Tool{Name: t.Name, Description: t.Description, Parameters: schema, Deferred: t.Deferred}
	}
	return specs
}

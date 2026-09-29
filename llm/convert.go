package llm

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/voocel/agentcore"
	"github.com/voocel/litellm"
)

// convertMessages converts agentcore.Message to litellm.Message.
// Handles multipart content with ordered litellm Blocks.
//
// Drops two classes of assistant turns that strict OpenAI-compatible providers
// reject with "assistant must provide content, reasoning_content or
// tool_calls":
//   - Fully empty turns (no content/reasoning/tool_calls/images at all).
//   - Reasoning-only turns that stopped naturally (stop_reason=stop, no
//     externally-visible action). These are valid on reasoning-aware
//     providers (DeepSeek/GLM/Qwen) which accept reasoning_content on
//     replay, but trip providers that ignore reasoning_content on the
//     request side. Skipping is semantically equivalent to "this turn
//     produced nothing": no text was uttered, no tool was invoked, so the
//     next request lets the model decide afresh from prior history without
//     leaking the discarded internal reasoning back into the transcript.
func convertMessages(messages []agentcore.Message) []litellm.Message {
	llmMessages := make([]litellm.Message, 0, len(messages))
	for _, msg := range messages {
		converted := convertSingleMessage(msg)
		if msg.Role == agentcore.RoleAssistant {
			if len(converted.Blocks) == 0 || isReasoningOnlyStopAssistant(msg, converted) {
				continue
			}
		}
		llmMessages = append(llmMessages, converted)
	}
	return llmMessages
}

// isReasoningOnlyStopAssistant reports whether an assistant turn carried only
// internal reasoning and stopped without any externally-visible action.
func isReasoningOnlyStopAssistant(orig agentcore.Message, converted litellm.Message) bool {
	if orig.StopReason != agentcore.StopReasonStop || len(converted.Blocks) != 1 {
		return false
	}
	_, ok := converted.Blocks[0].(litellm.ReasoningBlock)
	return ok
}

// convertSingleMessage converts one agentcore.Message to litellm.Message.
func convertSingleMessage(msg agentcore.Message) litellm.Message {
	cache := cacheControlFromMetadata(msg.Metadata)
	if msg.Role != agentcore.RoleTool {
		return litellm.Message{Role: litellm.Role(msg.Role), Blocks: convertAgentBlocks(msg.Content, cache)}
	}
	toolCallID, _ := msg.Metadata["tool_call_id"].(string)
	isError, _ := msg.Metadata["is_error"].(bool)
	// Cache lands on the outer tool_result block only; nested content cannot
	// carry provider cache breakpoints.
	result := litellm.ToolResultBlock{
		ToolUseID: toolCallID,
		Content:   convertAgentBlocks(msg.Content, nil),
		IsError:   isError,
		Cache:     cache,
	}
	return litellm.Message{Role: litellm.RoleTool, Blocks: []litellm.Block{result}}
}

// cacheControlFromMetadata parses the "cache_control" metadata value into a
// litellm CacheControl. The value is "type" or "type:ttl" — e.g. "ephemeral"
// (provider-default TTL) or "ephemeral:1h" (extended TTL where supported).
// litellm has one breakpoint kind, so the type only marks the breakpoint.
func cacheControlFromMetadata(metadata map[string]any) *litellm.CacheControl {
	value, _ := metadata["cache_control"].(string)
	if value == "" {
		return nil
	}
	_, ttl, _ := strings.Cut(value, ":")
	return &litellm.CacheControl{TTL: ttl}
}

// convertAgentBlocks converts content blocks to litellm blocks. A non-nil
// cache is attached to the LAST converted block only: message-level
// cache_control means "write one breakpoint after this message", and marking
// every block would burn one provider breakpoint per block (Anthropic allows
// at most 4 per request).
func convertAgentBlocks(content []agentcore.ContentBlock, cache *litellm.CacheControl) []litellm.Block {
	blocks := make([]litellm.Block, 0, len(content))
	for _, b := range content {
		switch b.Type {
		case agentcore.ContentText:
			if b.Text != "" || b.State != nil {
				blocks = append(blocks, litellm.TextBlock{Text: b.Text, State: fromState(b.State)})
			}
		case agentcore.ContentThinking:
			if b.Thinking != "" || b.State != nil {
				blocks = append(blocks, litellm.ReasoningBlock{Text: b.Thinking, State: fromState(b.State)})
			}
		case agentcore.ContentImage:
			if b.Image != nil {
				blocks = append(blocks, convertImageBlock(*b.Image))
			}
		case agentcore.ContentToolCall:
			if b.ToolCall != nil {
				tc := sanitizeOutgoingToolCall(*b.ToolCall)
				blocks = append(blocks, litellm.ToolUseBlock{
					ID:        tc.ID,
					Name:      tc.Name,
					Arguments: tc.Args,
					State:     fromState(b.State),
				})
			}
		case agentcore.ContentToolRef:
			if b.ToolName != "" {
				blocks = append(blocks, litellm.ToolReferenceBlock{ToolName: b.ToolName})
			}
		}
	}
	if cache != nil {
		// Anthropic rejects cache_control on thinking blocks — land the
		// breakpoint on the last cacheable block instead.
		for i := len(blocks) - 1; i >= 0; i-- {
			if _, isReasoning := blocks[i].(litellm.ReasoningBlock); isReasoning {
				continue
			}
			blocks[i] = withBlockCache(blocks[i], cache)
			break
		}
	}
	return blocks
}

// withBlockCache returns a copy of the block with cache_control attached.
// Blocks are value types, so mutating the copy never touches the caller's.
func withBlockCache(b litellm.Block, cache *litellm.CacheControl) litellm.Block {
	cc := &litellm.CacheControl{TTL: cache.TTL}
	switch v := b.(type) {
	case litellm.TextBlock:
		v.Cache = cc
		return v
	case litellm.ImageBlock:
		v.Cache = cc
		return v
	case litellm.ToolUseBlock:
		v.Cache = cc
		return v
	case litellm.ToolReferenceBlock:
		v.Cache = cc
		return v
	}
	return b
}

func sanitizeOutgoingToolCall(tc agentcore.ToolCall) agentcore.ToolCall {
	if len(tc.Args) > 0 && json.Valid(tc.Args) {
		return tc
	}
	raw := string(tc.Args)
	if raw == "" {
		raw = tc.ArgsRawText
	}
	n := normalizeArgs(raw)
	tc.Args = n.Args
	if n.Invalid {
		tc.ArgsInvalid = true
		if tc.ArgsRawText == "" {
			tc.ArgsRawText = n.RawText
		}
		if tc.ArgsParseError == "" {
			tc.ArgsParseError = n.ParseErr
		}
	}
	return tc
}

func convertImageBlock(img agentcore.ImageData) litellm.ImageBlock {
	if img.URL != "" {
		return litellm.ImageBlock{URL: img.URL, MIME: img.MimeType}
	}
	return litellm.ImageBlock{Data: []byte(img.Data), MIME: img.MimeType}
}

// convertBlocks converts response blocks to content blocks, one for one, with
// their replay state.
func convertBlocks(blocks []litellm.Block) []agentcore.ContentBlock {
	var content []agentcore.ContentBlock
	for _, block := range blocks {
		var c agentcore.ContentBlock
		switch b := block.(type) {
		case litellm.TextBlock:
			c = agentcore.TextBlock(b.Text)
			c.State = toState(b.State)
		case litellm.ReasoningBlock:
			// Reasoning without text, such as encrypted reasoning, is kept
			// for replay.
			c = agentcore.ThinkingBlock(b.Text)
			c.State = toState(b.State)
		case litellm.ToolUseBlock:
			c = agentcore.ToolCallBlock(buildToolCall(b.ID, b.Name, string(b.Arguments)))
			c.State = toState(b.State)
		default:
			continue
		}
		content = append(content, c)
	}
	return content
}

// toState and fromState convert provider replay state; the types match.
func toState(s *litellm.ProviderState) *agentcore.ProviderState {
	return (*agentcore.ProviderState)(s)
}

func fromState(s *agentcore.ProviderState) *litellm.ProviderState {
	return (*litellm.ProviderState)(s)
}

// responseMetadata keeps the vendor's raw finish reason and litellm's
// warnings, such as dropped content or a schema sent as a prompt.
func responseMetadata(resp *litellm.Response) map[string]any {
	metadata := make(map[string]any)
	if resp.FinishReasonRaw != "" {
		metadata["finish_reason_raw"] = resp.FinishReasonRaw
	}
	if len(resp.Warnings) > 0 {
		warnings := make([]string, len(resp.Warnings))
		for i, w := range resp.Warnings {
			warnings[i] = w.Code + ": " + w.Message
		}
		metadata["warnings"] = warnings
	}
	if len(metadata) == 0 {
		return nil
	}
	return metadata
}

// mapStopReason maps litellm canonical FinishReason to agentcore StopReason.
func mapStopReason(reason litellm.FinishReason) agentcore.StopReason {
	switch reason {
	case litellm.FinishReasonStop, litellm.FinishReason(""):
		return agentcore.StopReasonStop
	case litellm.FinishReasonLength:
		return agentcore.StopReasonLength
	case litellm.FinishReasonToolCall:
		return agentcore.StopReasonToolUse
	case litellm.FinishReasonError:
		return agentcore.StopReasonError
	case litellm.FinishReasonSafety:
		return agentcore.StopReasonSafety
	default:
		return agentcore.StopReason(string(reason))
	}
}

// convertThinking maps a thinking level and budget. Auto without a budget
// leaves thinking to the vendor; a budget alone enables thinking with it, for
// providers such as qwen that take a budget but no effort.
func convertThinking(level agentcore.ThinkingLevel, budget int) *litellm.Thinking {
	if level == agentcore.ThinkingOff {
		return &litellm.Thinking{Mode: litellm.ThinkingDisabled}
	}
	if level == ThinkingAuto && budget <= 0 {
		return nil
	}
	thinking := &litellm.Thinking{Effort: string(level)}
	if budget > 0 {
		thinking.BudgetTokens = &budget
	}
	return thinking
}

// convertToolChoice accepts the portable modes ("auto", "required", "none")
// or a litellm.ToolChoice naming a specific tool.
func convertToolChoice(choice any) (*litellm.ToolChoice, error) {
	switch c := choice.(type) {
	case string:
		return &litellm.ToolChoice{Mode: litellm.ToolChoiceMode(c)}, nil
	case litellm.ToolChoice:
		return &c, nil
	case *litellm.ToolChoice:
		return c, nil
	}
	return nil, fmt.Errorf("llm: unsupported tool choice %T", choice)
}

func convertResponseFormat(format *agentcore.ResponseFormat) (*litellm.ResponseFormat, error) {
	if format == nil {
		return nil, nil
	}
	out := &litellm.ResponseFormat{Type: litellm.ResponseFormatType(format.Type)}
	if format.JSONSchema != nil {
		schema, err := litellm.SchemaFrom(format.JSONSchema.Schema)
		if err != nil {
			return nil, fmt.Errorf("llm: response format schema: %w", err)
		}
		out.JSONSchema = &litellm.JSONSchema{
			Name:        format.JSONSchema.Name,
			Description: format.JSONSchema.Description,
			Schema:      schema,
			Strict:      strictMode(format.JSONSchema.Strict),
		}
	}
	return out, nil
}

func convertTools(tools []agentcore.ToolSpec) ([]litellm.Tool, error) {
	var out []litellm.Tool
	for _, t := range tools {
		if t.Name == "" {
			continue
		}
		schema, err := litellm.SchemaFrom(t.Parameters)
		if err != nil {
			return nil, fmt.Errorf("llm: tool %q schema: %w", t.Name, err)
		}
		out = append(out, litellm.Tool{
			Name:        t.Name,
			Description: t.Description,
			Parameters:  schema,
			Strict:      strictMode(t.Strict),
		})
	}
	return out, nil
}

func strictMode(v *bool) litellm.StrictMode {
	if v == nil {
		return litellm.StrictDefault
	}
	if *v {
		return litellm.StrictEnabled
	}
	return litellm.StrictDisabled
}

// normalizedArgs is the parsed shape of raw LLM tool-call args.
//   - Args is always valid JSON so the parent ToolCall stays JSON-serializable
//     (json.RawMessage marshalling validates its bytes; invalid args here would
//     break agent.ExportMessages → json.Marshal for persistence).
//   - When the raw payload was malformed, Args is the "{}" placeholder, and
//     RawText + ParseErr carry the original bytes and parser diagnostic so
//     downstream validation can point at the true root cause (stream
//     truncation, provider bug) instead of "missing field" against {}.
type normalizedArgs struct {
	Args     json.RawMessage
	Invalid  bool
	RawText  string
	ParseErr string
}

func normalizeArgs(raw string) normalizedArgs {
	if raw == "" {
		return normalizedArgs{Args: json.RawMessage("{}")}
	}
	if json.Valid([]byte(raw)) {
		return normalizedArgs{Args: json.RawMessage(raw)}
	}
	var probe any
	parseErr := json.Unmarshal([]byte(raw), &probe)
	return normalizedArgs{
		Args:     json.RawMessage("{}"),
		Invalid:  true,
		RawText:  raw,
		ParseErr: parseErr.Error(),
	}
}

// buildToolCall constructs an agentcore.ToolCall from raw litellm fields,
// routing malformed args into dedicated diagnostic fields (see ToolCall doc).
func buildToolCall(id, name, rawArgs string) agentcore.ToolCall {
	n := normalizeArgs(rawArgs)
	return agentcore.ToolCall{
		ID:             id,
		Name:           name,
		Args:           n.Args,
		ArgsInvalid:    n.Invalid,
		ArgsRawText:    n.RawText,
		ArgsParseError: n.ParseErr,
	}
}

// Package llm adapts LLM providers to the [agentcore.ChatModel] interface. It
// wraps litellm to reach OpenAI, Anthropic, Gemini, and other backends, and
// classifies provider errors onto agentcore's retry and overflow contracts.
// Construct a model with [NewModel].
package llm

import (
	"context"
	"encoding/json"
	"fmt"
	"maps"
	"time"

	"github.com/voocel/agentcore"
	"github.com/voocel/litellm"
)

// Re-exports so callers do not need to import litellm directly.
type (
	ProviderConfig   = litellm.ProviderConfig
	ResilienceConfig = litellm.ResilienceConfig
)

// LiteLLMAdapter adapts litellm to the agentcore.ChatModel interface.
type LiteLLMAdapter struct {
	*BaseModel
	client *litellm.Client
	model  string
	extra  map[string]any // model-level Extra merged into every request
}

// NewLiteLLMAdapter wraps an existing litellm.Client as a ChatModel. Use this
// when you need to reuse a Client or inject a custom Provider instance; for
// the common case prefer NewModel.
func NewLiteLLMAdapter(model string, client *litellm.Client) *LiteLLMAdapter {
	modelInfo := ModelInfo{
		Name:     model,
		Provider: client.ProviderName(),
		Capabilities: []string{
			string(CapabilityChat),
			string(CapabilityCompletion),
			string(CapabilityStreaming),
			string(CapabilityToolCalling),
		},
	}

	// Enrich from registry if available
	if caps, ok := litellm.GetModelCapabilities(model); ok {
		modelInfo.MaxTokens = caps.MaxOutputTokens
		modelInfo.ContextSize = caps.MaxInputTokens
	}
	if p, ok := litellm.GetModelPricing(model); ok {
		modelInfo.Pricing = &ModelPricing{
			InputPerToken:  p.InputCostPerToken,
			OutputPerToken: p.OutputCostPerToken,
		}
	}

	return &LiteLLMAdapter{
		BaseModel: NewBaseModel(modelInfo, DefaultGenerationConfig),
		client:    client,
		model:     model,
	}
}

// modelConfig collects optional knobs for NewModel.
type modelConfig struct {
	apiKey        string
	baseURL       string
	timeout       time.Duration
	streamIdle    time.Duration
	streamIdleSet bool // distinguishes "unset" from explicit 0 (disable watchdog)
	resilience    *ResilienceConfig
	clientOpts    []litellm.ClientOption
	extra         map[string]any
}

// ModelOption configures NewModel.
type ModelOption func(*modelConfig)

func WithAPIKey(key string) ModelOption  { return func(c *modelConfig) { c.apiKey = key } }
func WithBaseURL(url string) ModelOption { return func(c *modelConfig) { c.baseURL = url } }

// WithRequestTimeout sets the per-request timeout (default 10m).
func WithRequestTimeout(d time.Duration) ModelOption {
	return func(c *modelConfig) { c.timeout = d }
}

// WithStreamIdleTimeout aborts a streaming response if no chunk arrives within
// the window (default 120s). Pass 0 to disable the watchdog explicitly.
func WithStreamIdleTimeout(d time.Duration) ModelOption {
	return func(c *modelConfig) {
		c.streamIdle = d
		c.streamIdleSet = true
	}
}

// WithResilience replaces the entire resilience config; later With* options
// may still override specific fields.
func WithResilience(rc ResilienceConfig) ModelOption {
	return func(c *modelConfig) { c.resilience = &rc }
}

// WithClientOptions forwards litellm ClientOptions (e.g. litellm.WithHook) to
// the underlying client, letting callers attach observability or other
// cross-cutting behaviour without this package importing those concerns.
func WithClientOptions(opts ...litellm.ClientOption) ModelOption {
	return func(c *modelConfig) { c.clientOpts = append(c.clientOpts, opts...) }
}

// WithExtra sets model-level, provider-specific request parameters merged into
// every request's Extra map (e.g. min_p, presence_penalty, or provider keys like
// chat_template_kwargs). OpenAI-compatible providers pass these through verbatim
// into the request body — the extra_body convention. Per-call Extra entries
// (e.g. session_id) are added alongside, not overwritten.
func WithExtra(extra map[string]any) ModelOption {
	return func(c *modelConfig) { c.extra = extra }
}

// cloneExtra copies the model-level extra map so per-call mutations (e.g.
// session_id) never leak back into the shared model config across requests.
func cloneExtra(m map[string]any) map[string]any {
	if len(m) == 0 {
		return nil
	}
	c := make(map[string]any, len(m))
	maps.Copy(c, m)
	return c
}

// NewModel constructs a ChatModel by provider name. The provider must be
// registered in litellm (builtin or via litellm.RegisterProvider).
func NewModel(provider, model string, opts ...ModelOption) (*LiteLLMAdapter, error) {
	cfg := modelConfig{}
	for _, opt := range opts {
		opt(&cfg)
	}

	pcfg := ProviderConfig{
		APIKey:  cfg.apiKey,
		BaseURL: cfg.baseURL,
		Timeout: cfg.timeout,
	}
	if cfg.resilience != nil {
		pcfg.Resilience = *cfg.resilience
	}
	if cfg.streamIdleSet {
		if cfg.resilience == nil {
			pcfg.Resilience = litellm.DefaultResilienceConfig()
		}
		pcfg.Resilience.StreamIdleTimeout = cfg.streamIdle
	}

	client, err := litellm.NewWithProvider(provider, pcfg, cfg.clientOpts...)
	if err != nil {
		return nil, fmt.Errorf("llm: %s: %w", provider, err)
	}
	adapter := NewLiteLLMAdapter(model, client)
	adapter.extra = cfg.extra
	return adapter, nil
}

// IsProviderRegistered reports whether the provider name is known to litellm.
func IsProviderRegistered(name string) bool { return litellm.IsProviderRegistered(name) }

// RegisteredProviders returns all provider names known to litellm (builtin + custom).
func RegisteredProviders() []string { return litellm.ListRegisteredProviders() }

// ProviderName returns the provider name (e.g. "openai", "anthropic").
// Implements agentcore.ProviderNamer for per-provider API key resolution.
func (l *LiteLLMAdapter) ProviderName() string {
	return l.Info().Provider
}

// Generate produces a synchronous response.
func (l *LiteLLMAdapter) Generate(ctx context.Context, messages []agentcore.Message, tools []agentcore.ToolSpec, opts ...agentcore.CallOption) (*agentcore.LLMResponse, error) {
	cfg := l.GetConfig()
	llmMessages := convertMessages(messages)

	ltReq := &litellm.Request{
		Model:       l.model,
		Messages:    llmMessages,
		Temperature: &cfg.Temperature,
		MaxTokens:   &cfg.MaxTokens,
		Extra:       cloneExtra(l.extra),
	}

	applyCallConfig(ltReq, opts)
	applyToolConfig(ltReq, tools)

	ltResp, err := l.client.Chat(ctx, ltReq)
	if err != nil {
		return nil, fmt.Errorf("llm: chat failed: %w", wrapProviderError(err))
	}

	msg := convertResponse(ltResp)
	if msg.Usage != nil {
		msg.Usage.Cost = CalculateCost(pricingForUsage(l.Info().Pricing, msg.Usage), msg.Usage)
	}
	return &agentcore.LLMResponse{Message: msg}, nil
}

// GenerateStream produces a streaming response with fine-grained events.
// Delegates lifecycle detection to litellm's CollectStreamWithCallbacks.
func (l *LiteLLMAdapter) GenerateStream(ctx context.Context, messages []agentcore.Message, tools []agentcore.ToolSpec, opts ...agentcore.CallOption) (<-chan agentcore.StreamEvent, error) {
	cfg := l.GetConfig()
	llmMessages := convertMessages(messages)

	request := &litellm.Request{
		Model:       l.model,
		Messages:    llmMessages,
		Temperature: &cfg.Temperature,
		MaxTokens:   &cfg.MaxTokens,
		Extra:       cloneExtra(l.extra),
	}

	applyCallConfig(request, opts)
	applyToolConfig(request, tools)

	stream, err := l.client.Stream(ctx, request)
	if err != nil {
		return nil, fmt.Errorf("llm: stream failed: %w", wrapProviderError(err))
	}

	eventChan := make(chan agentcore.StreamEvent, 100)

	go func() {
		defer close(eventChan)
		defer stream.Close()

		var (
			partial          = agentcore.Message{Role: agentcore.RoleAssistant}
			textIdx          = -1
			thinkIdx         = -1
			toolBlockIndices = make(map[string]int)
		)

		resp, err := litellm.CollectStreamWithCallbacks(stream, litellm.StreamCallbacks{
			OnReasoningStart: func() {
				partial.Content = append(partial.Content, agentcore.ThinkingBlock(""))
				thinkIdx = len(partial.Content) - 1
				eventChan <- agentcore.StreamEvent{
					Type:         agentcore.StreamEventThinkingStart,
					ContentIndex: thinkIdx,
					Message:      partial,
				}
			},
			OnReasoning: func(content string) {
				if content == "" {
					return
				}
				partial.Content[thinkIdx].Thinking += content
				eventChan <- agentcore.StreamEvent{
					Type:         agentcore.StreamEventThinkingDelta,
					ContentIndex: thinkIdx,
					Delta:        content,
					Message:      partial,
				}
			},
			OnReasoningEnd: func(content string) {
				eventChan <- agentcore.StreamEvent{
					Type:         agentcore.StreamEventThinkingEnd,
					ContentIndex: thinkIdx,
					Message:      partial,
				}
			},
			OnContentStart: func() {
				partial.Content = append(partial.Content, agentcore.TextBlock(""))
				textIdx = len(partial.Content) - 1
				eventChan <- agentcore.StreamEvent{
					Type:         agentcore.StreamEventTextStart,
					ContentIndex: textIdx,
					Message:      partial,
				}
			},
			OnContent: func(delta string) {
				partial.Content[textIdx].Text += delta
				eventChan <- agentcore.StreamEvent{
					Type:         agentcore.StreamEventTextDelta,
					ContentIndex: textIdx,
					Delta:        delta,
					Message:      partial,
				}
			},
			OnContentEnd: func(content string) {
				eventChan <- agentcore.StreamEvent{
					Type:         agentcore.StreamEventTextEnd,
					ContentIndex: textIdx,
					Message:      partial,
				}
			},
			OnToolCallStart: func(delta *litellm.ToolCallDelta) {
				// Append a placeholder ToolCall block to partial so downstream stream
				// consumers can read the active tool name from partial.ToolCalls()
				// while arguments are still being streamed. Some providers deliver the
				// function name later via OnToolCall; that's patched in below.
				// OnToolCallEnd replaces this placeholder in-place with the completed call.
				var name, id string
				if delta != nil {
					name = delta.FunctionName
					id = delta.ID
				}
				partial.Content = append(partial.Content, agentcore.ToolCallBlock(agentcore.ToolCall{
					ID:   id,
					Name: name,
				}))
				if id != "" {
					toolBlockIndices[id] = len(partial.Content) - 1
				}
				eventChan <- agentcore.StreamEvent{
					Type:    agentcore.StreamEventToolCallStart,
					Message: partial,
				}
			},
			OnToolCall: func(delta *litellm.ToolCallDelta) {
				if delta == nil {
					return
				}
				toolBlockIdx := findPendingToolCallBlock(partial.Content, toolBlockIndices, delta.ID)
				if toolBlockIdx >= 0 && delta.FunctionName != "" {
					if block := partial.Content[toolBlockIdx]; block.ToolCall != nil && block.ToolCall.Name == "" {
						block.ToolCall.Name = delta.FunctionName
						partial.Content[toolBlockIdx] = block
					}
				}
				if toolBlockIdx >= 0 && delta.ID != "" {
					if block := partial.Content[toolBlockIdx]; block.ToolCall != nil && block.ToolCall.ID == "" {
						block.ToolCall.ID = delta.ID
						partial.Content[toolBlockIdx] = block
					}
					toolBlockIndices[delta.ID] = toolBlockIdx
				}
				if delta.ArgumentsDelta != "" {
					eventChan <- agentcore.StreamEvent{
						Type:    agentcore.StreamEventToolCallDelta,
						Delta:   delta.ArgumentsDelta,
						Message: partial,
					}
				}
			},
			OnToolCallEnd: func(call litellm.ToolCall) {
				completed := buildToolCall(call.ID, call.Function.Name, call.Function.Arguments, call.ThoughtSignature)
				idx := findPendingToolCallBlock(partial.Content, toolBlockIndices, call.ID)
				if idx >= 0 {
					partial.Content[idx] = agentcore.ToolCallBlock(completed)
				} else {
					partial.Content = append(partial.Content, agentcore.ToolCallBlock(completed))
					idx = len(partial.Content) - 1
				}
				if call.ID != "" {
					toolBlockIndices[call.ID] = idx
				}
				eventChan <- agentcore.StreamEvent{
					Type:              agentcore.StreamEventToolCallEnd,
					ContentIndex:      idx,
					Message:           partial,
					CompletedToolCall: &completed,
				}
			},
		})

		if err != nil {
			eventChan <- agentcore.StreamEvent{Type: agentcore.StreamEventError, Err: wrapProviderError(err)}
			return
		}

		// Map usage from collected response.
		if resp != nil && resp.Usage.HasTokens() {
			u := resp.Usage
			provider, model := responseUsageModel(resp)
			partial.Usage = &agentcore.Usage{
				Provider:    provider,
				Model:       model,
				Input:       u.PromptTokens,
				Output:      u.CompletionTokens,
				CacheRead:   u.CacheReadInputTokens,
				CacheWrite:  u.CacheCreationInputTokens,
				TotalTokens: u.TotalTokens,
			}
			partial.Usage.Cost = CalculateCost(pricingForUsage(l.Info().Pricing, partial.Usage), partial.Usage)
		}
		if resp != nil {
			partial.StopReason = mapStopReason(resp.FinishReason)
		}
		eventChan <- agentcore.StreamEvent{Type: agentcore.StreamEventDone, Message: partial, StopReason: partial.StopReason}
	}()

	return eventChan, nil
}

// convertMessages converts agentcore.Message to litellm.Message.
// Handles multipart content (text + images) via litellm.Contents field.
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
			if isEmptyAssistantMessage(converted) {
				continue
			}
			if isReasoningOnlyStopAssistant(msg, converted) {
				continue
			}
		}
		llmMessages = append(llmMessages, converted)
	}
	return llmMessages
}

// isEmptyAssistantMessage reports whether a converted assistant message has
// no payload that any provider could consume.
func isEmptyAssistantMessage(m litellm.Message) bool {
	if m.Content != "" || m.ReasoningContent != "" {
		return false
	}
	if len(m.Contents) > 0 || len(m.ToolCalls) > 0 {
		return false
	}
	return true
}

// isReasoningOnlyStopAssistant reports whether an assistant turn carried only
// internal reasoning and stopped without any externally-visible action.
// stop_reason=stop signals natural completion; length / toolUse turns may
// legitimately carry reasoning before truncation or tool dispatch and are
// preserved.
func isReasoningOnlyStopAssistant(orig agentcore.Message, converted litellm.Message) bool {
	if orig.StopReason != agentcore.StopReasonStop {
		return false
	}
	if converted.Content != "" {
		return false
	}
	if len(converted.Contents) > 0 || len(converted.ToolCalls) > 0 {
		return false
	}
	return converted.ReasoningContent != ""
}

// hasImageContent reports whether any content block is an image.
func hasImageContent(blocks []agentcore.ContentBlock) bool {
	for _, b := range blocks {
		if b.Type == agentcore.ContentImage && b.Image != nil {
			return true
		}
	}
	return false
}

// convertSingleMessage converts one agentcore.Message to litellm.Message.
func convertSingleMessage(msg agentcore.Message) litellm.Message {
	llmMsg := litellm.Message{
		Role: string(msg.Role),
	}

	// Multipart: if message contains images, use Contents field
	if hasImageContent(msg.Content) {
		var parts []litellm.MessageContent
		for _, b := range msg.Content {
			switch b.Type {
			case agentcore.ContentText:
				parts = append(parts, litellm.TextContent(b.Text))
			case agentcore.ContentImage:
				if b.Image != nil {
					var imgURL string
					if b.Image.URL != "" {
						imgURL = b.Image.URL
					} else {
						imgURL = "data:" + b.Image.MimeType + ";base64," + b.Image.Data
					}
					parts = append(parts, litellm.ImageContent(imgURL))
				}
			}
		}
		llmMsg.Contents = parts
	} else {
		llmMsg.Content = msg.TextContent()
	}

	// Forward thinking content as reasoning_content for assistant messages.
	// DeepSeek/GLM/Qwen/Mimo and other reasoning-aware OpenAI-compatible
	// providers require assistant turns to carry at least one of
	// content/reasoning_content/tool_calls; preserving thinking here keeps
	// thinking-only turns valid on replay. Vanilla OpenAI and Anthropic
	// ignore the field on the request side.
	if msg.Role == agentcore.RoleAssistant {
		if thinking := msg.ThinkingContent(); thinking != "" {
			llmMsg.ReasoningContent = thinking
		}
	}

	// Pass cache_control from Metadata for system messages (multi-block prompt caching).
	if cc, ok := msg.Metadata["cache_control"].(string); ok && cc != "" {
		llmMsg.CacheControl = &litellm.CacheControl{Type: cc}
	}

	if msg.Role == agentcore.RoleTool {
		if id, ok := msg.Metadata["tool_call_id"].(string); ok {
			llmMsg.ToolCallID = id
		}
		if isErr, ok := msg.Metadata["is_error"].(bool); ok {
			llmMsg.IsError = isErr
		}
		// Convert tool_reference content blocks for tool search results.
		if hasToolRefBlocks(msg.Content) {
			llmMsg.Contents = convertToolRefContent(msg.Content)
			llmMsg.Content = ""
		}
	}

	toolCalls := msg.ToolCalls()
	if len(toolCalls) > 0 {
		llmMsg.ToolCalls = make([]litellm.ToolCall, len(toolCalls))
		for idx, call := range toolCalls {
			llmMsg.ToolCalls[idx] = litellm.ToolCall{
				ID:   call.ID,
				Type: "function",
				Function: litellm.FunctionCall{
					Name:      call.Name,
					Arguments: string(call.Args),
				},
				ThoughtSignature: call.ThoughtSignature,
			}
		}
	}

	return llmMsg
}

// convertResponse converts litellm.Response to agentcore.Message with content blocks.
func convertResponse(response *litellm.Response) agentcore.Message {
	var content []agentcore.ContentBlock

	// Thinking/reasoning content
	if response.ReasoningContent != "" {
		content = append(content, agentcore.ThinkingBlock(response.ReasoningContent))
	}

	// Text content
	if response.Content != "" {
		content = append(content, agentcore.TextBlock(response.Content))
	}

	// Tool calls
	for _, call := range response.ToolCalls {
		content = append(content, agentcore.ToolCallBlock(buildToolCall(call.ID, call.Function.Name, call.Function.Arguments, call.ThoughtSignature)))
	}

	// Map usage
	var usage *agentcore.Usage
	if response.Usage.HasTokens() {
		provider, model := responseUsageModel(response)
		usage = &agentcore.Usage{
			Provider:    provider,
			Model:       model,
			Input:       response.Usage.PromptTokens,
			Output:      response.Usage.CompletionTokens,
			CacheRead:   response.Usage.CacheReadInputTokens,
			CacheWrite:  response.Usage.CacheCreationInputTokens,
			TotalTokens: response.Usage.TotalTokens,
		}
	}

	return agentcore.Message{
		Role:       agentcore.RoleAssistant,
		Content:    content,
		StopReason: mapStopReason(response.FinishReason),
		Usage:      usage,
	}
}

func responseUsageModel(response *litellm.Response) (provider, model string) {
	if response == nil {
		return "", ""
	}
	provider = response.Usage.Provider
	model = response.Usage.Model
	if provider == "" {
		provider = response.Provider
	}
	if model == "" {
		model = response.Model
	}
	return provider, model
}

func pricingForUsage(fallback *ModelPricing, usage *agentcore.Usage) *ModelPricing {
	if usage == nil || usage.Model == "" {
		return fallback
	}
	pricing, ok := litellm.GetModelPricing(usage.Model)
	if !ok {
		return fallback
	}
	return &ModelPricing{
		InputPerToken:      pricing.InputCostPerToken,
		OutputPerToken:     pricing.OutputCostPerToken,
		CacheReadPerToken:  pricing.CacheReadCostPerToken,
		CacheWritePerToken: pricing.CacheWriteCostPerToken,
	}
}

// mapStopReason maps litellm canonical FinishReason to agentcore StopReason.
func mapStopReason(reason string) agentcore.StopReason {
	switch reason {
	case litellm.FinishReasonStop, "":
		return agentcore.StopReasonStop
	case litellm.FinishReasonLength:
		return agentcore.StopReasonLength
	case litellm.FinishReasonToolCall:
		return agentcore.StopReasonToolUse
	case litellm.FinishReasonError:
		return agentcore.StopReasonError
	default:
		return agentcore.StopReason(reason)
	}
}

// applyCallConfig resolves CallOptions once and applies API key, thinking,
// session config, and response format to the litellm request.
func applyCallConfig(req *litellm.Request, opts []agentcore.CallOption) {
	callCfg := agentcore.ResolveCallConfig(opts)

	// Per-request API key override
	if callCfg.APIKey != "" {
		req.APIKey = callCfg.APIKey
	}

	// Thinking level + budget
	// Anthropic requires temperature=1 when thinking is enabled.
	switch callCfg.ThinkingLevel {
	case "":
		// Leave unspecified to allow model/provider defaults.
	case agentcore.ThinkingOff:
		req.Thinking = litellm.NewThinkingDisabled()
	default:
		req.Thinking = litellm.NewThinkingWithLevel(string(callCfg.ThinkingLevel))
		if callCfg.ThinkingBudget > 0 {
			req.Thinking.BudgetTokens = &callCfg.ThinkingBudget
		}
		t := 1.0
		req.Temperature = &t
	}

	// Session ID for provider caching
	if callCfg.SessionID != "" {
		if req.Extra == nil {
			req.Extra = make(map[string]any)
		}
		req.Extra["session_id"] = callCfg.SessionID
	}

	// Per-request max tokens override
	if callCfg.MaxTokens > 0 {
		req.MaxTokens = &callCfg.MaxTokens
	}

	// Tool choice: "auto" / "required" / "none"
	if callCfg.ToolChoice != nil {
		req.ToolChoice = callCfg.ToolChoice
	}

	if callCfg.ResponseFormat != nil {
		req.ResponseFormat = convertResponseFormat(callCfg.ResponseFormat)
	}
}

func convertResponseFormat(format *agentcore.ResponseFormat) *litellm.ResponseFormat {
	if format == nil {
		return nil
	}
	out := &litellm.ResponseFormat{
		Type: format.Type,
	}
	if format.JSONSchema != nil {
		out.JSONSchema = &litellm.JSONSchema{
			Name:        format.JSONSchema.Name,
			Description: format.JSONSchema.Description,
			Schema:      format.JSONSchema.Schema,
			Strict:      format.JSONSchema.Strict,
		}
	}
	return out
}

func applyToolConfig(request *litellm.Request, tools []agentcore.ToolSpec) {
	if len(tools) == 0 {
		return
	}
	ltTools := make([]litellm.Tool, 0, len(tools))
	for _, t := range tools {
		if t.Name == "" {
			continue
		}
		ltTools = append(ltTools, litellm.Tool{
			Type: "function",
			Function: litellm.FunctionDef{
				Name:        t.Name,
				Description: t.Description,
				Parameters:  t.Parameters,
				Strict:      t.Strict,
			},
			DeferLoading: t.DeferLoading,
		})
	}
	request.Tools = ltTools
}

// hasToolRefBlocks reports whether any content block is a tool_reference.
func hasToolRefBlocks(blocks []agentcore.ContentBlock) bool {
	for _, b := range blocks {
		if b.Type == agentcore.ContentToolRef {
			return true
		}
	}
	return false
}

// convertToolRefContent converts agentcore ContentBlocks with tool_reference
// types into litellm MessageContent for API serialization.
func convertToolRefContent(blocks []agentcore.ContentBlock) []litellm.MessageContent {
	var parts []litellm.MessageContent
	for _, b := range blocks {
		switch b.Type {
		case agentcore.ContentText:
			if b.Text != "" {
				parts = append(parts, litellm.TextContent(b.Text))
			}
		case agentcore.ContentToolRef:
			parts = append(parts, litellm.ToolRefContent(b.ToolName))
		}
	}
	return parts
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
func buildToolCall(id, name, rawArgs, thoughtSignature string) agentcore.ToolCall {
	n := normalizeArgs(rawArgs)
	return agentcore.ToolCall{
		ID:               id,
		Name:             name,
		Args:             n.Args,
		ArgsInvalid:      n.Invalid,
		ArgsRawText:      n.RawText,
		ArgsParseError:   n.ParseErr,
		ThoughtSignature: thoughtSignature,
	}
}

func findPendingToolCallBlock(content []agentcore.ContentBlock, byID map[string]int, toolCallID string) int {
	if toolCallID != "" {
		if idx, ok := byID[toolCallID]; ok && idx >= 0 && idx < len(content) {
			if block := content[idx]; block.ToolCall != nil {
				return idx
			}
		}
	}
	for i := len(content) - 1; i >= 0; i-- {
		block := content[i]
		if block.ToolCall == nil {
			continue
		}
		if toolCallID != "" && block.ToolCall.ID == toolCallID {
			return i
		}
		if len(block.ToolCall.Args) == 0 {
			return i
		}
	}
	return -1
}

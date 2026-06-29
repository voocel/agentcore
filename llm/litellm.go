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
	"strings"
	"time"

	"github.com/voocel/agentcore"
	"github.com/voocel/litellm"
	"github.com/voocel/litellm/provider/anthropic"
	"github.com/voocel/litellm/provider/bedrock"
	"github.com/voocel/litellm/provider/compat"
	"github.com/voocel/litellm/provider/deepseek"
	"github.com/voocel/litellm/provider/gemini"
	"github.com/voocel/litellm/provider/glm"
	"github.com/voocel/litellm/provider/grok"
	"github.com/voocel/litellm/provider/mimo"
	"github.com/voocel/litellm/provider/minimax"
	"github.com/voocel/litellm/provider/ollama"
	"github.com/voocel/litellm/provider/openai"
	"github.com/voocel/litellm/provider/openrouter"
	"github.com/voocel/litellm/provider/qwen"
	"github.com/voocel/litellm/retry"
)

// ProviderConfig is kept for source compatibility with older agentcore callers.
// New litellm providers expose explicit Config structs; this package maps the
// common subset that agentcore needs.
type ProviderConfig struct {
	APIKey  string
	BaseURL string
	Timeout time.Duration
	Extra   map[string]any
	Retry   *retry.Policy
}

// ResilienceConfig is the compatibility shape for retry and stream idle knobs.
type ResilienceConfig struct {
	StreamIdleTimeout time.Duration
	Retry             *retry.Policy
}

// LiteLLMAdapter adapts litellm to the agentcore.ChatModel interface.
type LiteLLMAdapter struct {
	*BaseModel
	client         *litellm.Client
	model          string
	requestTimeout time.Duration
	extra          map[string]any // model-level Extra merged into every request
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
	providerExtra map[string]any
}

// ModelOption configures NewModel.
type ModelOption func(*modelConfig)

func WithAPIKey(key string) ModelOption  { return func(c *modelConfig) { c.apiKey = key } }
func WithBaseURL(url string) ModelOption { return func(c *modelConfig) { c.baseURL = url } }

// WithRequestTimeout sets an optional per-request timeout. Zero leaves timeout control to the caller context.
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

// WithProviderExtra sets provider-level configuration passed to
// litellm.ProviderConfig.Extra. Use it for HTTP headers or other provider
// client options, while WithExtra remains request-body Extra.
func WithProviderExtra(extra map[string]any) ModelOption {
	return func(c *modelConfig) { c.providerExtra = extra }
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

	retryPolicy := (*retry.Policy)(nil)
	if cfg.resilience != nil {
		retryPolicy = cfg.resilience.Retry
	}
	pcfg := ProviderConfig{
		APIKey:  cfg.apiKey,
		BaseURL: cfg.baseURL,
		Timeout: cfg.timeout,
		Extra:   cloneExtra(cfg.providerExtra),
		Retry:   retryPolicy,
	}
	providerImpl, err := newProvider(provider, pcfg)
	if err != nil {
		return nil, fmt.Errorf("llm: %s: %w", provider, err)
	}
	clientOpts := append([]litellm.ClientOption(nil), cfg.clientOpts...)
	if cfg.streamIdleSet {
		clientOpts = append(clientOpts, litellm.WithStreamIdleTimeout(cfg.streamIdle))
	} else if cfg.resilience != nil && cfg.resilience.StreamIdleTimeout > 0 {
		clientOpts = append(clientOpts, litellm.WithStreamIdleTimeout(cfg.resilience.StreamIdleTimeout))
	}
	client, err := litellm.New(providerImpl, clientOpts...)
	if err != nil {
		return nil, fmt.Errorf("llm: %s client: %w", provider, err)
	}
	adapter := NewLiteLLMAdapter(model, client)
	adapter.extra = cfg.extra
	adapter.requestTimeout = cfg.timeout
	return adapter, nil
}

var knownProviders = map[string]struct{}{
	"anthropic":  {},
	"bedrock":    {},
	"deepseek":   {},
	"gemini":     {},
	"glm":        {},
	"grok":       {},
	"minimax":    {},
	"mimo":       {},
	"ollama":     {},
	"openai":     {},
	"openrouter": {},
	"qwen":       {},
}

// IsProviderRegistered reports whether the provider name is known to this adapter.
func IsProviderRegistered(name string) bool {
	_, ok := knownProviders[strings.ToLower(strings.TrimSpace(name))]
	return ok
}

// RegisteredProviders returns all provider names known to this adapter.
func RegisteredProviders() []string {
	return []string{"anthropic", "bedrock", "deepseek", "gemini", "glm", "grok", "minimax", "mimo", "ollama", "openai", "openrouter", "qwen"}
}

// ProviderName returns the provider name (e.g. "openai", "anthropic").
// Implements agentcore.ProviderNamer for per-provider API key resolution.
func (l *LiteLLMAdapter) ProviderName() string {
	return l.Info().Provider
}

// Capabilities returns the provider/model capability view exposed by litellm.
func (l *LiteLLMAdapter) Capabilities() Capabilities {
	if l == nil {
		return Capabilities{}
	}
	if l.client == nil {
		caps := Capabilities{Model: l.model}
		if l.BaseModel != nil {
			info := l.Info()
			caps.Provider = info.Provider
			if caps.Model == "" {
				caps.Model = info.Name
			}
		}
		return caps
	}
	return fromLiteLLMCapabilities(l.client.Capabilities(l.model))
}

// Generate produces a synchronous response.
func (l *LiteLLMAdapter) Generate(ctx context.Context, messages []agentcore.Message, tools []agentcore.ToolSpec, opts ...agentcore.CallOption) (*agentcore.LLMResponse, error) {
	if l.requestTimeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, l.requestTimeout)
		defer cancel()
	}
	cfg := l.GetConfig()
	llmMessages := convertMessages(messages)

	ltReq := &litellm.Request{
		Model:           l.model,
		Messages:        llmMessages,
		MaxTokens:       &cfg.MaxTokens,
		ProviderOptions: litellm.ProviderOptions(cloneExtra(l.extra)),
	}
	applySamplingConfig(ltReq, cfg)

	var err error
	ctx, err = applyCallConfig(ctx, ltReq, opts)
	if err != nil {
		return nil, err
	}
	if err := applyToolConfig(ltReq, tools); err != nil {
		return nil, err
	}

	ltResp, err := l.client.Chat(ctx, *ltReq)
	if err != nil {
		return nil, wrapProviderError(err)
	}

	msg := convertResponse(ltResp)
	if msg.Usage != nil {
		msg.Usage.Cost = CalculateCost(pricingForUsage(l.Info().Pricing, msg.Usage), msg.Usage)
	}
	return &agentcore.LLMResponse{Message: msg}, nil
}

// GenerateStream produces a streaming response with fine-grained events.
func (l *LiteLLMAdapter) GenerateStream(ctx context.Context, messages []agentcore.Message, tools []agentcore.ToolSpec, opts ...agentcore.CallOption) (<-chan agentcore.StreamEvent, error) {
	var cancel context.CancelFunc
	if l.requestTimeout > 0 {
		ctx, cancel = context.WithTimeout(ctx, l.requestTimeout)
	}
	cfg := l.GetConfig()
	llmMessages := convertMessages(messages)

	request := &litellm.Request{
		Model:           l.model,
		Messages:        llmMessages,
		MaxTokens:       &cfg.MaxTokens,
		ProviderOptions: litellm.ProviderOptions(cloneExtra(l.extra)),
	}
	applySamplingConfig(request, cfg)

	var err error
	ctx, err = applyCallConfig(ctx, request, opts)
	if err != nil {
		if cancel != nil {
			cancel()
		}
		return nil, err
	}
	if err := applyToolConfig(request, tools); err != nil {
		if cancel != nil {
			cancel()
		}
		return nil, err
	}

	stream, err := l.client.Stream(ctx, *request)
	if err != nil {
		if cancel != nil {
			cancel()
		}
		return nil, wrapProviderError(err)
	}

	eventChan := make(chan agentcore.StreamEvent, 100)

	go func() {
		defer close(eventChan)
		defer stream.Close()
		if cancel != nil {
			defer cancel()
		}

		var (
			partial          = agentcore.Message{Role: agentcore.RoleAssistant}
			textIdx          = -1
			thinkIdx         = -1
			toolBlockIndices = make(map[string]int)
			toolArgs         = make(map[string][]byte)
		)

		resp, err := litellm.Handle(stream, func(ev litellm.Event) error {
			switch e := ev.(type) {
			case litellm.ReasoningDelta:
				if e.Text == "" {
					return nil
				}
				if thinkIdx < 0 {
					partial.Content = append(partial.Content, agentcore.ThinkingBlock(""))
					thinkIdx = len(partial.Content) - 1
					eventChan <- agentcore.StreamEvent{Type: agentcore.StreamEventThinkingStart, ContentIndex: thinkIdx, Message: partial}
				}
				partial.Content[thinkIdx].Thinking += e.Text
				eventChan <- agentcore.StreamEvent{Type: agentcore.StreamEventThinkingDelta, ContentIndex: thinkIdx, Delta: e.Text, Message: partial}
			case litellm.ContentDelta:
				if textIdx < 0 {
					partial.Content = append(partial.Content, agentcore.TextBlock(""))
					textIdx = len(partial.Content) - 1
					eventChan <- agentcore.StreamEvent{Type: agentcore.StreamEventTextStart, ContentIndex: textIdx, Message: partial}
				}
				partial.Content[textIdx].Text += e.Text
				eventChan <- agentcore.StreamEvent{Type: agentcore.StreamEventTextDelta, ContentIndex: textIdx, Delta: e.Text, Message: partial}
			case litellm.ToolUseStart:
				key := toolUseEventKey(e.ID, e.ItemID, e.Index, e.OutputIndex)
				partial.Content = append(partial.Content, agentcore.ToolCallBlock(agentcore.ToolCall{ID: e.ID, Name: e.Name, ThoughtSignature: e.Signature}))
				idx := len(partial.Content) - 1
				if key != "" {
					toolBlockIndices[key] = idx
				}
				eventChan <- agentcore.StreamEvent{Type: agentcore.StreamEventToolCallStart, Message: partial}
			case litellm.ToolUseDelta:
				key := toolUseEventKey(e.ID, e.ItemID, e.Index, e.OutputIndex)
				idx := findPendingToolCallBlock(partial.Content, toolBlockIndices, key)
				if idx >= 0 {
					block := partial.Content[idx]
					if block.ToolCall != nil {
						if e.ID != "" && block.ToolCall.ID == "" {
							block.ToolCall.ID = e.ID
						}
						if e.Signature != "" && block.ToolCall.ThoughtSignature == "" {
							block.ToolCall.ThoughtSignature = e.Signature
						}
						partial.Content[idx] = block
					}
					if key != "" {
						toolBlockIndices[key] = idx
					}
				}
				if len(e.ArgumentsDelta) > 0 {
					toolArgs[key] = append(toolArgs[key], e.ArgumentsDelta...)
					eventChan <- agentcore.StreamEvent{Type: agentcore.StreamEventToolCallDelta, Delta: string(e.ArgumentsDelta), Message: partial}
				}
			case litellm.ToolUseDone:
				key := toolUseEventKey(e.ID, e.ItemID, e.Index, e.OutputIndex)
				idx := findPendingToolCallBlock(partial.Content, toolBlockIndices, key)
				var current agentcore.ToolCall
				if idx >= 0 && partial.Content[idx].ToolCall != nil {
					current = *partial.Content[idx].ToolCall
				}
				if current.ID == "" {
					current.ID = e.ID
				}
				args := string(toolArgs[key])
				completed := buildToolCall(current.ID, current.Name, args, current.ThoughtSignature)
				if idx >= 0 {
					partial.Content[idx] = agentcore.ToolCallBlock(completed)
				} else {
					partial.Content = append(partial.Content, agentcore.ToolCallBlock(completed))
					idx = len(partial.Content) - 1
				}
				eventChan <- agentcore.StreamEvent{Type: agentcore.StreamEventToolCallEnd, ContentIndex: idx, Message: partial, CompletedToolCall: &completed}
			}
			return nil
		})

		if textIdx >= 0 {
			eventChan <- agentcore.StreamEvent{Type: agentcore.StreamEventTextEnd, ContentIndex: textIdx, Message: partial}
		}
		if thinkIdx >= 0 {
			eventChan <- agentcore.StreamEvent{Type: agentcore.StreamEventThinkingEnd, ContentIndex: thinkIdx, Message: partial}
		}

		if err != nil {
			eventChan <- agentcore.StreamEvent{Type: agentcore.StreamEventError, Err: wrapProviderError(err)}
			return
		}

		partial.Content = finalizePendingStreamToolCalls(partial.Content, toolBlockIndices, toolArgs)
		if resp != nil && resp.Usage.HasTokens() {
			u := resp.Usage
			provider, model := responseUsageModel(resp)
			partial.Usage = &agentcore.Usage{
				Provider:    provider,
				Model:       model,
				Input:       u.InputTokens,
				Output:      u.OutputTokens,
				CacheRead:   u.CacheReadTokens,
				CacheWrite:  u.CacheWriteTokens,
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

func finalizePendingStreamToolCalls(content []agentcore.ContentBlock, indices map[string]int, argsByKey map[string][]byte) []agentcore.ContentBlock {
	if len(indices) == 0 {
		return content
	}
	for key, idx := range indices {
		if idx < 0 || idx >= len(content) {
			continue
		}
		block := content[idx]
		if block.ToolCall == nil || len(block.ToolCall.Args) > 0 {
			continue
		}
		tc := *block.ToolCall
		completed := buildToolCall(tc.ID, tc.Name, string(argsByKey[key]), tc.ThoughtSignature)
		content[idx] = agentcore.ToolCallBlock(completed)
	}
	return content
}

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
	return len(m.Blocks) == 0
}

// isReasoningOnlyStopAssistant reports whether an assistant turn carried only
// internal reasoning and stopped without any externally-visible action.
func isReasoningOnlyStopAssistant(orig agentcore.Message, converted litellm.Message) bool {
	if orig.StopReason != agentcore.StopReasonStop {
		return false
	}
	if len(converted.Blocks) != 1 {
		return false
	}
	_, ok := converted.Blocks[0].(litellm.ReasoningBlock)
	return ok
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
	llmMsg := litellm.Message{Role: litellm.Role(msg.Role)}
	var blocks []litellm.Block
	cache := cacheControlFromMetadata(msg.Metadata)
	if msg.Role == agentcore.RoleTool {
		toolCallID, _ := msg.Metadata["tool_call_id"].(string)
		isError, _ := msg.Metadata["is_error"].(bool)
		content := convertAgentBlocks(msg.Content, cache)
		blocks = append(blocks, litellm.ToolResultBlock{ToolUseID: toolCallID, Content: content, IsError: isError, Cache: cache})
	} else {
		blocks = convertAgentBlocks(msg.Content, cache)
	}
	llmMsg.Blocks = blocks
	return llmMsg
}

func cacheControlFromMetadata(metadata map[string]any) *litellm.CacheControl {
	value, _ := metadata["cache_control"].(string)
	if value == "" {
		return nil
	}
	return &litellm.CacheControl{Type: value}
}

func cloneCacheControl(cache *litellm.CacheControl) *litellm.CacheControl {
	if cache == nil {
		return nil
	}
	return &litellm.CacheControl{Type: cache.Type, TTL: cache.TTL}
}

func convertAgentBlocks(content []agentcore.ContentBlock, cache *litellm.CacheControl) []litellm.Block {
	blocks := make([]litellm.Block, 0, len(content))
	for _, b := range content {
		switch b.Type {
		case agentcore.ContentText:
			if b.Text != "" {
				blocks = append(blocks, litellm.TextBlock{Text: b.Text, Cache: cloneCacheControl(cache)})
			}
		case agentcore.ContentThinking:
			if b.Thinking != "" {
				blocks = append(blocks, litellm.ReasoningBlock{Text: b.Thinking, Cache: cloneCacheControl(cache)})
			}
		case agentcore.ContentImage:
			if b.Image != nil {
				block := convertImageBlock(*b.Image)
				block.Cache = cloneCacheControl(cache)
				blocks = append(blocks, block)
			}
		case agentcore.ContentToolCall:
			if b.ToolCall != nil {
				tc := sanitizeOutgoingToolCall(*b.ToolCall)
				blocks = append(blocks, litellm.ToolUseBlock{
					ID:        tc.ID,
					Name:      tc.Name,
					Arguments: tc.Args,
					Signature: tc.ThoughtSignature,
					Cache:     cloneCacheControl(cache),
				})
			}
		case agentcore.ContentToolRef:
			if b.ToolName != "" {
				blocks = append(blocks, litellm.ToolReferenceBlock{ToolName: b.ToolName, Cache: cloneCacheControl(cache)})
			}
		}
	}
	return blocks
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

// convertResponse converts litellm.Response to agentcore.Message with content blocks.
func convertResponse(response *litellm.Response) agentcore.Message {
	content := convertResponseContent(response)

	var usage *agentcore.Usage
	if response != nil && response.Usage.HasTokens() {
		provider, model := responseUsageModel(response)
		usage = &agentcore.Usage{
			Provider:    provider,
			Model:       model,
			Input:       response.Usage.InputTokens,
			Output:      response.Usage.OutputTokens,
			CacheRead:   response.Usage.CacheReadTokens,
			CacheWrite:  response.Usage.CacheWriteTokens,
			TotalTokens: response.Usage.TotalTokens,
		}
	}

	finish := agentcore.StopReasonStop
	if response != nil {
		finish = mapStopReason(response.FinishReason)
	}
	return agentcore.Message{
		Role:       agentcore.RoleAssistant,
		Content:    content,
		StopReason: finish,
		Usage:      usage,
	}
}

func convertResponseContent(response *litellm.Response) []agentcore.ContentBlock {
	if response == nil {
		return nil
	}
	var content []agentcore.ContentBlock
	for _, block := range response.Blocks {
		switch b := block.(type) {
		case litellm.TextBlock:
			content = append(content, agentcore.TextBlock(b.Text))
		case litellm.ReasoningBlock:
			if b.Text != "" {
				content = append(content, agentcore.ThinkingBlock(b.Text))
			}
		case litellm.ToolUseBlock:
			content = append(content, agentcore.ToolCallBlock(buildToolCall(b.ID, b.Name, string(b.Arguments), b.Signature)))
		}
	}
	return content
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
	return fallback
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
	default:
		return agentcore.StopReason(string(reason))
	}
}

type apiKeyOverrideKey struct{}

func contextWithAPIKey(ctx context.Context, key string) context.Context {
	if key == "" {
		return ctx
	}
	return context.WithValue(ctx, apiKeyOverrideKey{}, key)
}

func apiKeyFunc(defaultKey string) func(context.Context) (string, error) {
	return func(ctx context.Context) (string, error) {
		if key, ok := ctx.Value(apiKeyOverrideKey{}).(string); ok && key != "" {
			return key, nil
		}
		return defaultKey, nil
	}
}

func applySamplingConfig(req *litellm.Request, cfg *GenerationConfig) {
	if req == nil || cfg == nil {
		return
	}
	if cfg.Temperature != DefaultGenerationConfig.Temperature {
		req.Temperature = &cfg.Temperature
	}
}

// applyCallConfig resolves CallOptions once and applies API key, thinking,
// session config, and response format to the litellm request.
func applyCallConfig(ctx context.Context, req *litellm.Request, opts []agentcore.CallOption) (context.Context, error) {
	callCfg := agentcore.ResolveCallConfig(opts)
	ctx = contextWithAPIKey(ctx, callCfg.APIKey)

	// Thinking level + budget.
	// Anthropic budget-token thinking requires temperature=1.
	switch callCfg.ThinkingLevel {
	case "":
		// Leave unspecified to allow model/provider defaults.
	case agentcore.ThinkingOff:
		req.Thinking = &litellm.Thinking{Mode: litellm.ThinkingDisabled}
	default:
		req.Thinking = &litellm.Thinking{Mode: litellm.ThinkingEnabled, Effort: string(callCfg.ThinkingLevel)}
		if callCfg.ThinkingBudget > 0 {
			req.Thinking.BudgetTokens = &callCfg.ThinkingBudget
		}
		// Each provider owns its thinking-time sampling constraint: Anthropic
		// wants temperature=1 (omitting it defaults to 1), while mimo forbids a
		// custom temperature/top_p entirely. Clear the injected default and let
		// the provider decide instead of forcing an Anthropic-specific value.
		req.Temperature = nil
	}

	if callCfg.SessionID != "" {
		if req.ProviderOptions == nil {
			req.ProviderOptions = make(litellm.ProviderOptions)
		}
		req.ProviderOptions["session_id"] = callCfg.SessionID
	}

	if callCfg.MaxTokens > 0 {
		req.MaxTokens = &callCfg.MaxTokens
	}

	if callCfg.ToolChoice != nil {
		req.ToolChoice = callCfg.ToolChoice
	}

	if callCfg.ResponseFormat != nil {
		format, err := convertResponseFormat(callCfg.ResponseFormat)
		if err != nil {
			return ctx, err
		}
		req.ResponseFormat = format
	}
	return ctx, nil
}

func convertResponseFormat(format *agentcore.ResponseFormat) (*litellm.ResponseFormat, error) {
	if format == nil {
		return nil, nil
	}
	out := &litellm.ResponseFormat{
		Type: litellm.ResponseFormatType(format.Type),
	}
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

func applyToolConfig(request *litellm.Request, tools []agentcore.ToolSpec) error {
	if len(tools) == 0 {
		return nil
	}
	ltTools := make([]litellm.Tool, 0, len(tools))
	for _, t := range tools {
		if t.Name == "" {
			continue
		}
		schema, err := litellm.SchemaFrom(t.Parameters)
		if err != nil {
			return fmt.Errorf("llm: tool %q schema: %w", t.Name, err)
		}
		ltTools = append(ltTools, litellm.Tool{
			Name:        t.Name,
			Description: t.Description,
			Parameters:  schema,
			Strict:      strictMode(t.Strict),
		})
	}
	request.Tools = ltTools
	return nil
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

func newProvider(name string, cfg ProviderConfig) (litellm.Provider, error) {
	name = strings.ToLower(strings.TrimSpace(name))
	headers, err := headersFromExtra(cfg.Extra)
	if err != nil {
		return nil, err
	}
	userAgent := stringFromExtra(cfg.Extra, "user_agent")
	switch name {
	case "openai":
		return openai.New(openai.Config{APIKeyFunc: apiKeyFunc(cfg.APIKey), BaseURL: cfg.BaseURL, Retry: cfg.Retry, API: firstStringFromExtra(cfg.Extra, "api", "api_mode"), Headers: headers, UserAgent: userAgent})
	case "anthropic":
		return anthropic.New(anthropic.Config{APIKeyFunc: apiKeyFunc(cfg.APIKey), BaseURL: cfg.BaseURL, Retry: cfg.Retry, Beta: stringFromExtra(cfg.Extra, "anthropic_beta"), Headers: headers, UserAgent: userAgent})
	case "bedrock":
		bedrockCfg, err := bedrockConfig(cfg)
		if err != nil {
			return nil, err
		}
		return bedrock.New(bedrockCfg)
	case "gemini":
		return gemini.New(gemini.Config{APIKeyFunc: apiKeyFunc(cfg.APIKey), BaseURL: cfg.BaseURL, Retry: cfg.Retry})
	case "deepseek":
		return deepseek.New(compatProviderConfig(cfg, headers, userAgent))
	case "glm":
		return glm.New(compatProviderConfig(cfg, headers, userAgent))
	case "grok":
		return grok.New(compatProviderConfig(cfg, headers, userAgent))
	case "minimax":
		return minimax.New(compatProviderConfig(cfg, headers, userAgent))
	case "mimo":
		return mimo.New(compatProviderConfig(cfg, headers, userAgent))
	case "ollama":
		return ollama.New(compatProviderConfig(cfg, headers, userAgent))
	case "openrouter":
		return openrouter.New(compatProviderConfig(cfg, headers, userAgent))
	case "qwen":
		return qwen.New(compatProviderConfig(cfg, headers, userAgent))
	default:
		return nil, fmt.Errorf("unknown provider %q", name)
	}
}

func compatProviderConfig(cfg ProviderConfig, headers map[string]string, userAgent string) compat.Config {
	return compat.Config{
		APIKeyFunc:                  apiKeyFunc(cfg.APIKey),
		BaseURL:                     cfg.BaseURL,
		Retry:                       cfg.Retry,
		Headers:                     headers,
		UserAgent:                   userAgent,
		AllowUnknownProviderOptions: true,
	}
}

func bedrockConfig(cfg ProviderConfig) (bedrock.Config, error) {
	accessKeyID := firstStringFromExtra(cfg.Extra, "access_key_id", "aws_access_key_id")
	secretAccessKey := firstStringFromExtra(cfg.Extra, "secret_access_key", "aws_secret_access_key")
	if accessKeyID == "" || secretAccessKey == "" {
		return bedrock.Config{}, fmt.Errorf("bedrock: access_key_id and secret_access_key are required in provider extra")
	}
	return bedrock.Config{
		Region:              firstStringFromExtra(cfg.Extra, "region", "aws_region"),
		BaseURL:             cfg.BaseURL,
		ControlPlaneBaseURL: stringFromExtra(cfg.Extra, "control_plane_base_url"),
		Credentials: bedrock.StaticCredentials(
			accessKeyID,
			secretAccessKey,
			firstStringFromExtra(cfg.Extra, "session_token", "aws_session_token"),
		),
		Retry: cfg.Retry,
	}, nil
}

func headersFromExtra(extra map[string]any) (map[string]string, error) {
	raw, ok := extra["headers"]
	if !ok || raw == nil {
		return nil, nil
	}
	switch h := raw.(type) {
	case map[string]string:
		return maps.Clone(h), nil
	case map[string]any:
		out := make(map[string]string, len(h))
		for k, v := range h {
			s, ok := v.(string)
			if !ok {
				return nil, fmt.Errorf("llm: provider header %q must be string", k)
			}
			out[k] = s
		}
		return out, nil
	default:
		return nil, fmt.Errorf("llm: provider headers must be map[string]string")
	}
}

func stringFromExtra(extra map[string]any, key string) string {
	v, _ := extra[key].(string)
	return v
}

func firstStringFromExtra(extra map[string]any, keys ...string) string {
	for _, key := range keys {
		if value := stringFromExtra(extra, key); value != "" {
			return value
		}
	}
	return ""
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

func toolUseEventKey(id, itemID string, index, outputIndex *int) string {
	switch {
	case id != "":
		return "id:" + id
	case itemID != "":
		return "item:" + itemID
	case index != nil:
		return fmt.Sprintf("index:%d", *index)
	case outputIndex != nil:
		return fmt.Sprintf("output:%d", *outputIndex)
	default:
		return ""
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

package llm

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/voocel/agentcore"
	"github.com/voocel/litellm"
)

// LiteLLMAdapter adapts litellm to the agentcore.ChatModel interface.
type LiteLLMAdapter struct {
	*BaseModel
	client *litellm.Client
	model  string
}

// NewLiteLLMAdapter creates an adapter from a litellm Client.
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

// newProviderAdapter is the shared constructor for all provider adapters.
func newProviderAdapter(provider, model, apiKey string, baseURL ...string) (*LiteLLMAdapter, error) {
	cfg := litellm.ProviderConfig{APIKey: apiKey}
	if len(baseURL) > 0 {
		cfg.BaseURL = baseURL[0]
	}
	client, err := litellm.NewWithProvider(provider, cfg)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", provider, err)
	}
	return NewLiteLLMAdapter(model, client), nil
}

// NewOpenAIModel creates an OpenAI adapter.
func NewOpenAIModel(model, apiKey string, baseURL ...string) (*LiteLLMAdapter, error) {
	return newProviderAdapter("openai", model, apiKey, baseURL...)
}

// NewAnthropicModel creates an Anthropic adapter.
func NewAnthropicModel(model, apiKey string, baseURL ...string) (*LiteLLMAdapter, error) {
	return newProviderAdapter("anthropic", model, apiKey, baseURL...)
}

// NewGeminiModel creates a Gemini adapter.
func NewGeminiModel(model, apiKey string, baseURL ...string) (*LiteLLMAdapter, error) {
	return newProviderAdapter("gemini", model, apiKey, baseURL...)
}

// NewOpenRouterModel creates an OpenRouter adapter.
func NewOpenRouterModel(model, apiKey string, baseURL ...string) (*LiteLLMAdapter, error) {
	return newProviderAdapter("openrouter", model, apiKey, baseURL...)
}

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
	}

	applyCallConfig(ltReq, opts)
	applyToolConfig(ltReq, tools)

	ltResp, err := l.client.Chat(ctx, ltReq)
	if err != nil {
		return nil, fmt.Errorf("llm: chat failed: %w", err)
	}

	msg := convertResponse(ltResp)
	if msg.Usage != nil {
		msg.Usage.Cost = CalculateCost(l.Info().Pricing, msg.Usage)
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
	}

	applyCallConfig(request, opts)
	applyToolConfig(request, tools)

	stream, err := l.client.Stream(ctx, request)
	if err != nil {
		return nil, fmt.Errorf("llm: stream failed: %w", err)
	}

	eventChan := make(chan agentcore.StreamEvent, 100)

	go func() {
		defer close(eventChan)
		defer stream.Close()

		var (
			partial      = agentcore.Message{Role: agentcore.RoleAssistant}
			textIdx      = -1
			thinkIdx     = -1
			toolBlockIdx = -1 // index into partial.Content of the tool call currently being streamed
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
				toolBlockIdx = len(partial.Content) - 1
				eventChan <- agentcore.StreamEvent{
					Type:    agentcore.StreamEventToolCallStart,
					Message: partial,
				}
			},
			OnToolCall: func(delta *litellm.ToolCallDelta) {
				if delta == nil {
					return
				}
				if toolBlockIdx >= 0 && delta.FunctionName != "" {
					if block := partial.Content[toolBlockIdx]; block.ToolCall != nil && block.ToolCall.Name == "" {
						block.ToolCall.Name = delta.FunctionName
						partial.Content[toolBlockIdx] = block
					}
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
				completed := agentcore.ToolCall{
					ID:   call.ID,
					Name: call.Function.Name,
					Args: safeArgs(call.Function.Arguments),
				}
				var idx int
				if toolBlockIdx >= 0 {
					idx = toolBlockIdx
					partial.Content[idx] = agentcore.ToolCallBlock(completed)
					toolBlockIdx = -1
				} else {
					partial.Content = append(partial.Content, agentcore.ToolCallBlock(completed))
					idx = len(partial.Content) - 1
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
			eventChan <- agentcore.StreamEvent{Type: agentcore.StreamEventError, Err: err}
			return
		}

		// Map usage from collected response.
		if resp != nil && (resp.Usage.TotalTokens > 0 || resp.Usage.PromptTokens > 0) {
			u := resp.Usage
			partial.Usage = &agentcore.Usage{
				Input:       u.PromptTokens,
				Output:      u.CompletionTokens,
				CacheRead:   u.CacheReadInputTokens,
				CacheWrite:  u.CacheCreationInputTokens,
				TotalTokens: u.TotalTokens,
			}
			partial.Usage.Cost = CalculateCost(l.Info().Pricing, partial.Usage)
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
func convertMessages(messages []agentcore.Message) []litellm.Message {
	llmMessages := make([]litellm.Message, len(messages))
	for i, msg := range messages {
		llmMessages[i] = convertSingleMessage(msg)
	}
	return llmMessages
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
		content = append(content, agentcore.ToolCallBlock(agentcore.ToolCall{
			ID:   call.ID,
			Name: call.Function.Name,
			Args: safeArgs(call.Function.Arguments),
		}))
	}

	// Map usage
	var usage *agentcore.Usage
	if response.Usage.TotalTokens > 0 {
		usage = &agentcore.Usage{
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

// applyCallConfig resolves CallOptions once and applies API key, thinking, and
// session config to the litellm request.
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

// safeArgs returns args as json.RawMessage, defaulting to "{}" if empty or invalid JSON.
func safeArgs(args string) json.RawMessage {
	if args == "" {
		return json.RawMessage("{}")
	}
	if !json.Valid([]byte(args)) {
		return json.RawMessage("{}")
	}
	return json.RawMessage(args)
}

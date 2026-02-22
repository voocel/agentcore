package llm

import (
	"context"
	"encoding/json"
	"fmt"
	"io"

	"github.com/voocel/agentcore"
	"github.com/voocel/litellm"
)

// LiteLLMAdapter adapts litellm to the llm.ChatModel interface.
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

// ProviderName returns the provider name (e.g. "openai", "anthropic").
// Implements agentcore.ProviderNamer for per-provider API key resolution.
func (l *LiteLLMAdapter) ProviderName() string {
	return l.Info().Provider
}

// Generate produces a synchronous response.
func (l *LiteLLMAdapter) Generate(ctx context.Context, messages []Message, tools []ToolSpec, opts ...CallOption) (*LLMResponse, error) {
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
	return &LLMResponse{Message: msg}, nil
}

// GenerateStream produces a streaming response with fine-grained events.
func (l *LiteLLMAdapter) GenerateStream(ctx context.Context, messages []Message, tools []ToolSpec, opts ...CallOption) (<-chan StreamEvent, error) {
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

	eventChan := make(chan StreamEvent, 100)

	go func() {
		defer close(eventChan)
		defer stream.Close()

		var (
			partial      = agentcore.Message{Role: agentcore.RoleAssistant}
			textStarted  bool
			thinkStarted bool
			finishReason string
			toolAcc      = litellm.NewToolCallAccumulator()
			toolStarted  = make(map[int]bool)
			streamUsage  *litellm.Usage // captured from the last chunk
		)

		for {
			chunk, err := stream.Next()
			if err != nil {
				if err == io.EOF {
					break
				}
				eventChan <- StreamEvent{Type: StreamEventError, Err: err}
				return
			}
			if chunk == nil {
				continue
			}

			// Stream completed — break to emit final events.
			if chunk.Done {
				if chunk.Usage != nil && chunk.Usage.TotalTokens > 0 {
					streamUsage = chunk.Usage
				}
				break
			}

			// Reasoning/thinking chunks
			if chunk.Reasoning != nil && chunk.Reasoning.Content != "" {
				if !thinkStarted {
					thinkStarted = true
					partial.Content = append(partial.Content, agentcore.ThinkingBlock(""))
					idx := len(partial.Content) - 1
					eventChan <- StreamEvent{
						Type:         StreamEventThinkingStart,
						ContentIndex: idx,
						Message:      partial,
					}
				}
				idx := lastBlockIndex(partial.Content, agentcore.ContentThinking)
				partial.Content[idx].Thinking += chunk.Reasoning.Content
				eventChan <- StreamEvent{
					Type:         StreamEventThinkingDelta,
					ContentIndex: idx,
					Delta:        chunk.Reasoning.Content,
					Message:      partial,
				}
			}

			// Text content chunks
			if chunk.Content != "" {
				if !textStarted {
					textStarted = true
					partial.Content = append(partial.Content, agentcore.TextBlock(""))
					idx := len(partial.Content) - 1
					eventChan <- StreamEvent{
						Type:         StreamEventTextStart,
						ContentIndex: idx,
						Message:      partial,
					}
				}
				idx := lastBlockIndex(partial.Content, agentcore.ContentText)
				partial.Content[idx].Text += chunk.Content
				eventChan <- StreamEvent{
					Type:         StreamEventTextDelta,
					ContentIndex: idx,
					Delta:        chunk.Content,
					Message:      partial,
				}
			}

			// Tool call deltas
			if chunk.ToolCallDelta != nil {
				deltaIdx := chunk.ToolCallDelta.Index
				wasStarted := toolAcc.Started(deltaIdx)
				toolAcc.Apply(chunk.ToolCallDelta)

				if !wasStarted && !toolStarted[deltaIdx] {
					toolStarted[deltaIdx] = true
					eventChan <- StreamEvent{
						Type:    StreamEventToolCallStart,
						Message: partial,
					}
				}

				if chunk.ToolCallDelta.ArgumentsDelta != "" {
					eventChan <- StreamEvent{
						Type:    StreamEventToolCallDelta,
						Delta:   chunk.ToolCallDelta.ArgumentsDelta,
						Message: partial,
					}
				}
			}

			if chunk.FinishReason != "" {
				finishReason = chunk.FinishReason
			}

			// Capture usage from the last chunk that carries it
			if chunk.Usage != nil && chunk.Usage.TotalTokens > 0 {
				streamUsage = chunk.Usage
			}
		}

		// Emit end events for open blocks
		if thinkStarted {
			idx := lastBlockIndex(partial.Content, agentcore.ContentThinking)
			if idx >= 0 {
				eventChan <- StreamEvent{
					Type:         StreamEventThinkingEnd,
					ContentIndex: idx,
					Message:      partial,
				}
			}
		}
		if textStarted {
			idx := lastBlockIndex(partial.Content, agentcore.ContentText)
			if idx >= 0 {
				eventChan <- StreamEvent{
					Type:         StreamEventTextEnd,
					ContentIndex: idx,
					Message:      partial,
				}
			}
		}

		// Build final tool calls from accumulated deltas
		if calls := toolAcc.Build(); len(calls) > 0 {
			for _, tc := range calls {
				partial.Content = append(partial.Content, agentcore.ToolCallBlock(ToolCall{
					ID:   tc.ID,
					Name: tc.Function.Name,
					Args: safeArgs(tc.Function.Arguments),
				}))
				idx := len(partial.Content) - 1
				eventChan <- StreamEvent{
					Type:         StreamEventToolCallEnd,
					ContentIndex: idx,
					Message:      partial,
				}
			}
		}

		// Attach usage from stream to the final message
		if streamUsage != nil && streamUsage.TotalTokens > 0 {
			partial.Usage = &agentcore.Usage{
				Input:       streamUsage.PromptTokens,
				Output:      streamUsage.CompletionTokens,
				CacheRead:   streamUsage.CacheReadInputTokens,
				CacheWrite:  streamUsage.CacheCreationInputTokens,
				TotalTokens: streamUsage.TotalTokens,
			}
			partial.Usage.Cost = CalculateCost(l.Info().Pricing, partial.Usage)
		}

		partial.StopReason = mapStopReason(finishReason)
		eventChan <- StreamEvent{Type: StreamEventDone, Message: partial, StopReason: partial.StopReason}
	}()

	return eventChan, nil
}

// lastBlockIndex returns the index of the last ContentBlock of the given type.
func lastBlockIndex(blocks []agentcore.ContentBlock, ct agentcore.ContentType) int {
	for i := len(blocks) - 1; i >= 0; i-- {
		if blocks[i].Type == ct {
			return i
		}
	}
	return -1
}

// convertMessages converts agentcore.Message to litellm.Message.
// Handles multipart content (text + images) via litellm.Contents field.
func convertMessages(messages []Message) []litellm.Message {
	llmMessages := make([]litellm.Message, len(messages))
	for i, msg := range messages {
		llmMsg := convertSingleMessage(msg)
		llmMessages[i] = llmMsg
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
func convertSingleMessage(msg Message) litellm.Message {
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
					// Base64 data URI format: data:<mime>;base64,<data>
					url := "data:" + b.Image.MimeType + ";base64," + b.Image.Data
					parts = append(parts, litellm.ImageContent(url))
				}
			}
		}
		llmMsg.Contents = parts
	} else {
		llmMsg.Content = msg.TextContent()
	}

	if msg.Role == agentcore.RoleTool {
		if id, ok := msg.Metadata["tool_call_id"].(string); ok {
			llmMsg.ToolCallID = id
		}
		if isErr, ok := msg.Metadata["is_error"].(bool); ok {
			llmMsg.IsError = isErr
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
func convertResponse(response *litellm.Response) Message {
	var content []agentcore.ContentBlock

	// Thinking/reasoning content
	if response.Reasoning != nil && response.Reasoning.Content != "" {
		content = append(content, agentcore.ThinkingBlock(response.Reasoning.Content))
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

	return Message{
		Role:       agentcore.RoleAssistant,
		Content:    content,
		StopReason: mapStopReason(response.FinishReason),
		Usage:      usage,
	}
}

// mapStopReason maps litellm canonical FinishReason to MAS StopReason.
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
func applyCallConfig(req *litellm.Request, opts []CallOption) {
	callCfg := agentcore.ResolveCallConfig(opts)

	// Per-request API key override
	if callCfg.APIKey != "" {
		req.APIKey = callCfg.APIKey
	}

	// Thinking level + budget
	// Anthropic requires temperature=1 when thinking is enabled.
	if callCfg.ThinkingLevel != "" && callCfg.ThinkingLevel != agentcore.ThinkingOff {
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
}

func applyToolConfig(request *litellm.Request, tools []ToolSpec) {
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
		})
	}
	request.Tools = ltTools
}

// safeArgs returns args as json.RawMessage, defaulting to "{}" if empty.
func safeArgs(args string) json.RawMessage {
	if args == "" {
		return json.RawMessage("{}")
	}
	return json.RawMessage(args)
}

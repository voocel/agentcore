package agentcore

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"time"

	"github.com/voocel/litellm"
)

const defaultMaxTurns = 10

// AgentLoop starts an agent loop with new prompt messages.
// Prompts are added to context and events are emitted for them.
func AgentLoop(ctx context.Context, prompts []AgentMessage, agentCtx AgentContext, config LoopConfig) <-chan Event {
	ch := make(chan Event, 128)

	go func() {
		defer close(ch)

		newMessages := make([]AgentMessage, len(prompts))
		copy(newMessages, prompts)

		currentCtx := AgentContext{
			SystemPrompt: agentCtx.SystemPrompt,
			Messages:     append(copyMessages(agentCtx.Messages), prompts...),
			Tools:        agentCtx.Tools,
		}

		emit(ch, Event{Type: EventAgentStart})
		emit(ch, Event{Type: EventTurnStart})

		for _, p := range prompts {
			emit(ch, Event{Type: EventMessageStart, Message: p})
			emit(ch, Event{Type: EventMessageEnd, Message: p})
		}

		runLoop(ctx, &currentCtx, &newMessages, config, ch)
	}()

	return ch
}

// AgentLoopContinue continues from existing context without adding new messages.
// The last message in context must convert to user or tool role via ConvertToLLM.
func AgentLoopContinue(ctx context.Context, agentCtx AgentContext, config LoopConfig) <-chan Event {
	ch := make(chan Event, 128)

	if len(agentCtx.Messages) == 0 {
		go func() {
			defer close(ch)
			emitError(ch, fmt.Errorf("cannot continue: no messages in context"))
		}()
		return ch
	}

	go func() {
		defer close(ch)

		var newMessages []AgentMessage
		currentCtx := AgentContext{
			SystemPrompt: agentCtx.SystemPrompt,
			Messages:     copyMessages(agentCtx.Messages),
			Tools:        agentCtx.Tools,
		}

		emit(ch, Event{Type: EventAgentStart})
		emit(ch, Event{Type: EventTurnStart})

		runLoop(ctx, &currentCtx, &newMessages, config, ch)
	}()

	return ch
}

// runLoop is the main double-loop logic shared by AgentLoop and AgentLoopContinue.
func runLoop(ctx context.Context, currentCtx *AgentContext, newMessages *[]AgentMessage, config LoopConfig, ch chan<- Event) {
	maxTurns := config.MaxTurns
	if maxTurns <= 0 {
		maxTurns = defaultMaxTurns
	}

	firstTurn := true
	turnCount := 0
	toolErrors := make(map[string]int) // consecutive failure count per tool

	// Check for steering messages at start
	var pendingMessages []AgentMessage
	if config.GetSteeringMessages != nil {
		pendingMessages = config.GetSteeringMessages()
	}

	// Outer loop: continues when follow-up messages arrive after agent would stop
	for {
		hasMoreToolCalls := true
		var steeringAfterTools []AgentMessage

		// Inner loop: process tool calls and steering messages
		for hasMoreToolCalls || len(pendingMessages) > 0 {
			// Check for context cancellation (Abort)
			if ctx.Err() != nil {
				emit(ch, Event{Type: EventError, Err: ctx.Err()})
				emit(ch, Event{Type: EventAgentEnd, NewMessages: *newMessages})
				return
			}

			if turnCount >= maxTurns {
				emit(ch, Event{Type: EventError, Err: fmt.Errorf("max turns (%d) reached", maxTurns)})
				emit(ch, Event{Type: EventAgentEnd, NewMessages: *newMessages})
				return
			}

			if !firstTurn {
				emit(ch, Event{Type: EventTurnStart})
			} else {
				firstTurn = false
			}

			// Process pending messages (inject before next LLM call)
			if len(pendingMessages) > 0 {
				for _, msg := range pendingMessages {
					emit(ch, Event{Type: EventMessageStart, Message: msg})
					emit(ch, Event{Type: EventMessageEnd, Message: msg})
					currentCtx.Messages = append(currentCtx.Messages, msg)
					*newMessages = append(*newMessages, msg)
				}
				pendingMessages = nil
			}

			// Call LLM with retry (streaming: events emitted inside callLLM)
			assistantMsg, err := callLLMWithRetry(ctx, currentCtx, config, ch)
			if err != nil {
				emitError(ch, fmt.Errorf("llm call failed: %w", err))
				return
			}

			// Check stop reason — terminate early on error/aborted
			if assistantMsg.StopReason == StopReasonError || assistantMsg.StopReason == StopReasonAborted {
				currentCtx.Messages = append(currentCtx.Messages, assistantMsg)
				*newMessages = append(*newMessages, assistantMsg)
				emit(ch, Event{Type: EventTurnEnd, Message: assistantMsg})
				emit(ch, Event{Type: EventAgentEnd, NewMessages: *newMessages})
				return
			}

			// When output was truncated (max_tokens hit), tool calls are likely
			// incomplete with malformed JSON args. Strip them to avoid validation
			// errors and API rejections.
			if assistantMsg.StopReason == StopReasonLength {
				assistantMsg.Content = stripToolCallBlocks(assistantMsg.Content)
			}

			currentCtx.Messages = append(currentCtx.Messages, assistantMsg)
			*newMessages = append(*newMessages, assistantMsg)

			// Check for tool calls
			toolCalls := assistantMsg.ToolCalls()
			hasMoreToolCalls = len(toolCalls) > 0

			var turnToolResults []ToolResult
			if hasMoreToolCalls {
				var steering []AgentMessage
				turnToolResults, steering = executeToolCalls(ctx, currentCtx.Tools, toolCalls, config, toolErrors, ch)

				for _, tr := range turnToolResults {
					resultMsg := toolResultToMessage(tr)
					emit(ch, Event{Type: EventMessageStart, Message: resultMsg})
					emit(ch, Event{Type: EventMessageEnd, Message: resultMsg})
					currentCtx.Messages = append(currentCtx.Messages, resultMsg)
					*newMessages = append(*newMessages, resultMsg)
				}

				steeringAfterTools = steering
			}

			emit(ch, Event{Type: EventTurnEnd, Message: assistantMsg, ToolResults: turnToolResults})
			turnCount++

			// Get steering messages after turn completes
			if len(steeringAfterTools) > 0 {
				pendingMessages = steeringAfterTools
				steeringAfterTools = nil
			} else if config.GetSteeringMessages != nil {
				pendingMessages = config.GetSteeringMessages()
			}
		}

		// Agent would stop here. Check for follow-up messages.
		if config.GetFollowUpMessages != nil {
			followUp := config.GetFollowUpMessages()
			if len(followUp) > 0 {
				pendingMessages = followUp
				continue
			}
		}

		break
	}

	emit(ch, Event{Type: EventAgentEnd, NewMessages: *newMessages})
}

// callLLMWithRetry wraps callLLM with retry logic for retryable errors.
// Context overflow errors trigger automatic compaction and a single retry.
func callLLMWithRetry(ctx context.Context, agentCtx *AgentContext, config LoopConfig, ch chan<- Event) (Message, error) {
	maxRetries := config.MaxRetries
	if maxRetries <= 0 {
		msg, err := callLLM(ctx, agentCtx, config, ch)
		if err != nil && IsContextOverflow(err) {
			return recoverOverflow(ctx, agentCtx, config, ch, err)
		}
		return msg, err
	}

	var lastErr error
	for attempt := 0; attempt <= maxRetries; attempt++ {
		msg, err := callLLM(ctx, agentCtx, config, ch)
		if err == nil {
			return msg, nil
		}
		lastErr = err

		// Context overflow: compact and retry once (not a normal retry)
		if IsContextOverflow(err) {
			return recoverOverflow(ctx, agentCtx, config, ch, err)
		}

		if !litellm.IsRetryableError(err) || attempt == maxRetries {
			return Message{}, err
		}

		delay := retryDelay(err, attempt)

		emit(ch, Event{
			Type: EventRetry,
			Err:  err,
			RetryInfo: &RetryInfo{
				Attempt:    attempt + 1,
				MaxRetries: maxRetries,
				Delay:      delay,
				Err:        err,
			},
		})

		select {
		case <-ctx.Done():
			return Message{}, ctx.Err()
		case <-time.After(delay):
		}
	}
	return Message{}, lastErr
}

// recoverOverflow attempts to compact the context and retry the LLM call.
// If no TransformContext is configured, the original error is returned.
func recoverOverflow(ctx context.Context, agentCtx *AgentContext, config LoopConfig, ch chan<- Event, originalErr error) (Message, error) {
	if config.TransformContext == nil {
		return Message{}, fmt.Errorf("context overflow (no compaction configured): %w", originalErr)
	}

	emit(ch, Event{
		Type: EventRetry,
		Err:  originalErr,
		RetryInfo: &RetryInfo{
			Attempt:    1,
			MaxRetries: 1,
			Err:        fmt.Errorf("context overflow detected, compacting and retrying"),
		},
	})

	compacted, err := config.TransformContext(ctx, agentCtx.Messages)
	if err != nil {
		return Message{}, fmt.Errorf("overflow recovery compaction failed: %w", err)
	}

	agentCtx.Messages = compacted
	return callLLM(ctx, agentCtx, config, ch)
}

// retryDelay calculates the wait duration using exponential backoff,
// capped at 30s. Respects Retry-After from rate limit errors.
func retryDelay(err error, attempt int) time.Duration {
	if after := litellm.GetRetryAfter(err); after > 0 {
		d := time.Duration(after) * time.Second
		if d > 30*time.Second {
			d = 30 * time.Second
		}
		return d
	}
	// Exponential backoff: 1s, 2s, 4s, 8s... capped at 30s
	d := time.Duration(math.Pow(2, float64(attempt))) * time.Second
	if d > 30*time.Second {
		d = 30 * time.Second
	}
	return d
}

// callLLM applies the two-stage pipeline and calls the model.
func callLLM(ctx context.Context, agentCtx *AgentContext, config LoopConfig, ch chan<- Event) (Message, error) {
	messages := agentCtx.Messages

	// Stage 1: TransformContext (AgentMessage[] → AgentMessage[])
	if config.TransformContext != nil {
		var err error
		messages, err = config.TransformContext(ctx, messages)
		if err != nil {
			return Message{}, fmt.Errorf("transform context: %w", err)
		}
	}

	// Stage 2: ConvertToLLM (AgentMessage[] → Message[])
	convertFn := config.ConvertToLLM
	if convertFn == nil {
		convertFn = DefaultConvertToLLM
	}
	llmMessages := convertFn(messages)

	// Repair orphaned tool call / result pairs
	llmMessages = RepairMessageSequence(llmMessages)

	// Build tool specs
	toolSpecs := buildToolSpecs(agentCtx.Tools)

	// Prepend system prompt as first message if set
	if agentCtx.SystemPrompt != "" {
		llmMessages = append([]Message{SystemMsg(agentCtx.SystemPrompt)}, llmMessages...)
	}

	// Call via StreamFn (non-streaming shortcut, e.g. mock/proxy)
	if config.StreamFn != nil {
		resp, err := config.StreamFn(ctx, &LLMRequest{
			Messages: llmMessages,
			Tools:    toolSpecs,
		})
		if err != nil {
			return Message{}, err
		}
		resp.Message.Timestamp = time.Now()
		msg := resp.Message
		emit(ch, Event{Type: EventMessageStart, Message: msg})
		emit(ch, Event{Type: EventMessageEnd, Message: msg})
		return msg, nil
	}

	if config.Model == nil {
		return Message{}, fmt.Errorf("no model configured")
	}

	// Build per-call options
	var callOpts []CallOption

	// Dynamic API key resolution
	if config.GetApiKey != nil {
		provider := ""
		if pn, ok := config.Model.(ProviderNamer); ok {
			provider = pn.ProviderName()
		}
		if key, err := config.GetApiKey(provider); err == nil && key != "" {
			callOpts = append(callOpts, WithAPIKey(key))
		}
	}

	// Thinking level + budget
	if config.ThinkingLevel != "" && config.ThinkingLevel != ThinkingOff {
		callOpts = append(callOpts, WithThinking(config.ThinkingLevel))
		if config.ThinkingBudgets != nil {
			if budget, ok := config.ThinkingBudgets[config.ThinkingLevel]; ok && budget > 0 {
				callOpts = append(callOpts, WithThinkingBudget(budget))
			}
		}
	}

	// Session ID for provider caching
	if config.SessionID != "" {
		callOpts = append(callOpts, WithCallSessionID(config.SessionID))
	}

	// Use streaming for real-time token deltas
	return callLLMStream(ctx, config.Model, llmMessages, toolSpecs, callOpts, ch)
}

// callLLMStream uses GenerateStream and emits real-time events.
// The adapter builds partial Messages with ContentBlocks and emits fine-grained StreamEvents.
func callLLMStream(ctx context.Context, model ChatModel, messages []Message, tools []ToolSpec, opts []CallOption, ch chan<- Event) (Message, error) {
	streamCh, err := model.GenerateStream(ctx, messages, tools, opts...)
	if err != nil {
		// Fallback to non-streaming
		resp, err := model.Generate(ctx, messages, tools, opts...)
		if err != nil {
			return Message{}, err
		}
		resp.Message.Timestamp = time.Now()
		return resp.Message, nil
	}

	var (
		started bool
		partial Message
	)

	for ev := range streamCh {
		switch ev.Type {
		case StreamEventTextStart, StreamEventThinkingStart, StreamEventToolCallStart:
			partial = ev.Message
			if !started {
				started = true
				emit(ch, Event{Type: EventMessageStart, Message: partial})
			}

		case StreamEventTextDelta, StreamEventThinkingDelta, StreamEventToolCallDelta:
			partial = ev.Message
			if !started {
				started = true
				emit(ch, Event{Type: EventMessageStart, Message: partial})
			}
			emit(ch, Event{Type: EventMessageUpdate, Message: partial, Delta: ev.Delta})

		case StreamEventTextEnd, StreamEventThinkingEnd, StreamEventToolCallEnd:
			partial = ev.Message

		case StreamEventDone:
			finalMsg := ev.Message
			finalMsg.Timestamp = time.Now()
			if !started {
				emit(ch, Event{Type: EventMessageStart, Message: finalMsg})
			}
			emit(ch, Event{Type: EventMessageEnd, Message: finalMsg})
			return finalMsg, nil

		case StreamEventError:
			return Message{}, ev.Err
		}
	}

	// Stream closed without done event — use partial
	partial.Timestamp = time.Now()
	if !started {
		emit(ch, Event{Type: EventMessageStart, Message: partial})
	}
	emit(ch, Event{Type: EventMessageEnd, Message: partial})
	return partial, nil
}

// executeToolCalls runs tool calls sequentially, checking steering after each.
// toolErrors tracks consecutive failures per tool for circuit breaking.
func executeToolCalls(ctx context.Context, tools []Tool, calls []ToolCall, config LoopConfig, toolErrors map[string]int, ch chan<- Event) ([]ToolResult, []AgentMessage) {
	results := make([]ToolResult, 0, len(calls))

	for i, call := range calls {
		tool := findTool(tools, call.Name)
		label := toolLabel(tool)

		// Circuit breaker: skip if tool has exceeded consecutive failure threshold
		if config.MaxToolErrors > 0 && toolErrors[call.Name] >= config.MaxToolErrors {
			emit(ch, Event{
				Type:      EventToolExecStart,
				ToolID:    call.ID,
				Tool:      call.Name,
				ToolLabel: label,
				Args:      call.Args,
			})
			errContent, _ := json.Marshal(fmt.Sprintf("tool %q disabled after %d consecutive errors", call.Name, config.MaxToolErrors))
			result := ToolResult{ToolCallID: call.ID, Content: errContent, IsError: true}
			emit(ch, Event{
				Type:    EventToolExecEnd,
				ToolID:  call.ID,
				Tool:    call.Name,
				Result:  result.Content,
				IsError: true,
			})
			results = append(results, result)
			continue
		}

		emit(ch, Event{
			Type:      EventToolExecStart,
			ToolID:    call.ID,
			Tool:      call.Name,
			ToolLabel: label,
			Args:      call.Args,
		})

		// Permission check: deny before execution if callback returns error.
		// Denial does NOT count toward toolErrors (policy decision, not tool failure).
		if config.CheckPermission != nil {
			if err := config.CheckPermission(ctx, call); err != nil {
				errContent, _ := json.Marshal(err.Error())
				result := ToolResult{ToolCallID: call.ID, Content: errContent, IsError: true}
				emit(ch, Event{
					Type:      EventToolExecEnd,
					ToolID:    call.ID,
					Tool:      call.Name,
					ToolLabel: label,
					Result:    result.Content,
					IsError:   true,
				})
				results = append(results, result)
				continue
			}
		}

		var result ToolResult

		if tool == nil {
			errContent, _ := json.Marshal(fmt.Sprintf("tool %q not found", call.Name))
			result = ToolResult{
				ToolCallID: call.ID,
				Content:    errContent,
				IsError:    true,
			}
		} else if err := validateToolArgs(tool, call.Args); err != nil {
			// Argument validation failed — return error to LLM without counting as tool error.
			errContent, _ := json.Marshal(err.Error())
			result = ToolResult{
				ToolCallID: call.ID,
				Content:    errContent,
				IsError:    true,
			}
		} else {
			// Inject progress callback so tools can report partial results
			progressCtx := WithToolProgress(ctx, func(partial json.RawMessage) {
				emit(ch, Event{
					Type:      EventToolExecUpdate,
					ToolID:    call.ID,
					Tool:      call.Name,
					ToolLabel: label,
					Args:      call.Args,
					Result:    partial,
				})
			})

			var output json.RawMessage
			var execErr error
			if len(config.Middlewares) > 0 {
				exec := buildMiddlewareChain(tool, call, config.Middlewares)
				output, execErr = exec(progressCtx, call.Args)
			} else {
				output, execErr = tool.Execute(progressCtx, call.Args)
			}
			err := execErr
			if err != nil {
				errContent, _ := json.Marshal(err.Error())
				result = ToolResult{
					ToolCallID: call.ID,
					Content:    errContent,
					IsError:    true,
				}
			} else {
				result = ToolResult{
					ToolCallID: call.ID,
					Content:    output,
				}
			}
		}

		emit(ch, Event{
			Type:      EventToolExecEnd,
			ToolID:    call.ID,
			Tool:      call.Name,
			ToolLabel: label,
			Result:    result.Content,
			IsError:   result.IsError,
		})

		// Update consecutive error counter
		if result.IsError {
			toolErrors[call.Name]++
		} else {
			delete(toolErrors, call.Name)
		}

		results = append(results, result)

		// Check for steering messages — skip remaining tools if user interrupted
		if config.GetSteeringMessages != nil {
			steering := config.GetSteeringMessages()
			if len(steering) > 0 {
				// Skip remaining tool calls
				for _, skipped := range calls[i+1:] {
					results = append(results, skipToolCall(skipped, tools, ch))
				}
				return results, steering
			}
		}
	}

	return results, nil
}

// skipToolCall creates a skipped result for an interrupted tool call.
func skipToolCall(call ToolCall, tools []Tool, ch chan<- Event) ToolResult {
	label := toolLabel(findTool(tools, call.Name))

	emit(ch, Event{
		Type:      EventToolExecStart,
		ToolID:    call.ID,
		Tool:      call.Name,
		ToolLabel: label,
		Args:      call.Args,
	})

	content, _ := json.Marshal("Skipped due to queued user message.")
	result := ToolResult{
		ToolCallID: call.ID,
		Content:    content,
		IsError:    true,
	}

	emit(ch, Event{
		Type:      EventToolExecEnd,
		ToolID:    call.ID,
		Tool:      call.Name,
		ToolLabel: label,
		Result:    result.Content,
		IsError:   true,
	})

	return result
}

// toolResultToMessage converts a ToolResult into a Message for the context.
func toolResultToMessage(tr ToolResult) Message {
	return ToolResultMsg(tr.ToolCallID, tr.Content, tr.IsError)
}

// stripToolCallBlocks removes ContentToolCall blocks from a content slice.
// Used when the model's output was truncated (StopReasonLength), where tool
// call arguments are likely incomplete / malformed JSON.
func stripToolCallBlocks(blocks []ContentBlock) []ContentBlock {
	filtered := blocks[:0:0]
	for _, b := range blocks {
		if b.Type != ContentToolCall {
			filtered = append(filtered, b)
		}
	}
	return filtered
}

// toolLabel returns the human-readable label for a tool.
func toolLabel(tool Tool) string {
	if tool == nil {
		return ""
	}
	if labeler, ok := tool.(ToolLabeler); ok {
		return labeler.Label()
	}
	return ""
}

// buildToolSpecs converts Tool interfaces to ToolSpec for the LLM.
func buildToolSpecs(tools []Tool) []ToolSpec {
	if len(tools) == 0 {
		return nil
	}
	specs := make([]ToolSpec, 0, len(tools))
	for _, t := range tools {
		specs = append(specs, ToolSpec{
			Name:        t.Name(),
			Description: t.Description(),
			Parameters:  t.Schema(),
		})
	}
	return specs
}

// validateToolArgs validates tool call arguments against the tool's JSON Schema.
// Checks required fields and basic type matching. Returns nil if valid or schema is unavailable.
// On failure, returns a formatted error message suitable for sending back to the LLM.
func validateToolArgs(tool Tool, args json.RawMessage) error {
	schema := tool.Schema()
	if schema == nil {
		return nil
	}

	// Parse arguments (treat nil/empty as empty object)
	if len(args) == 0 {
		args = []byte("{}")
	}
	var parsed map[string]any
	if err := json.Unmarshal(args, &parsed); err != nil {
		return fmt.Errorf("validation failed for tool %q: invalid JSON: %w", tool.Name(), err)
	}

	// Check required fields
	if reqSlice, ok := schema["required"]; ok {
		if required, ok := reqSlice.([]string); ok {
			for _, field := range required {
				if _, exists := parsed[field]; !exists {
					return fmt.Errorf("validation failed for tool %q: missing required field %q", tool.Name(), field)
				}
			}
		}
	}

	// Check property types
	if props, ok := schema["properties"].(map[string]any); ok {
		for key, val := range parsed {
			propSchema, exists := props[key]
			if !exists {
				continue
			}
			ps, ok := propSchema.(map[string]any)
			if !ok {
				continue
			}
			expectedType, _ := ps["type"].(string)
			if expectedType == "" {
				continue
			}
			if err := checkType(key, val, expectedType); err != nil {
				return fmt.Errorf("validation failed for tool %q: %w", tool.Name(), err)
			}
		}
	}

	return nil
}

// checkType validates a single value against an expected JSON Schema type.
func checkType(field string, val any, expected string) error {
	switch expected {
	case "string":
		if _, ok := val.(string); !ok {
			return fmt.Errorf("field %q: expected string, got %T", field, val)
		}
	case "integer":
		switch v := val.(type) {
		case float64:
			if v != float64(int64(v)) {
				return fmt.Errorf("field %q: expected integer, got float", field)
			}
		default:
			return fmt.Errorf("field %q: expected integer, got %T", field, val)
		}
	case "number":
		if _, ok := val.(float64); !ok {
			return fmt.Errorf("field %q: expected number, got %T", field, val)
		}
	case "boolean":
		if _, ok := val.(bool); !ok {
			return fmt.Errorf("field %q: expected boolean, got %T", field, val)
		}
	case "array":
		if _, ok := val.([]any); !ok {
			return fmt.Errorf("field %q: expected array, got %T", field, val)
		}
	case "object":
		if _, ok := val.(map[string]any); !ok {
			return fmt.Errorf("field %q: expected object, got %T", field, val)
		}
	}
	return nil
}

// buildMiddlewareChain wraps a tool's Execute with the middleware stack.
// Outermost middleware is called first; innermost calls the actual tool.
func buildMiddlewareChain(tool Tool, call ToolCall, middlewares []ToolMiddleware) ToolExecuteFunc {
	exec := ToolExecuteFunc(tool.Execute)
	for i := len(middlewares) - 1; i >= 0; i-- {
		mw := middlewares[i]
		next := exec
		exec = func(ctx context.Context, args json.RawMessage) (json.RawMessage, error) {
			return mw(ctx, call, next)
		}
	}
	return exec
}

func findTool(tools []Tool, name string) Tool {
	for _, t := range tools {
		if t.Name() == name {
			return t
		}
	}
	return nil
}

func copyMessages(msgs []AgentMessage) []AgentMessage {
	out := make([]AgentMessage, len(msgs))
	copy(out, msgs)
	return out
}

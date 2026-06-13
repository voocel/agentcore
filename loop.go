package agentcore

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"math"
	"slices"
	"strconv"
	"strings"
	"time"
)

const (
	defaultMaxTurns            = 100
	defaultMaxRetries          = 3
	defaultMaxLengthRecoveries = 3
	defaultMaxToolErrors       = 3
	defaultMaxRetryDelay       = 60 * time.Second
)

const defaultLengthRecoveryPrompt = "Output token limit hit. Resume directly - no apology, no recap of what you were doing. Pick up mid-thought if that is where the cut happened. Break remaining work into smaller pieces."

const (
	defaultAbortMarkerText     = "[Request interrupted by user]"
	defaultAbortMarkerToolText = "[Request interrupted by user for tool use]"
)

// AgentLoop starts an agent loop with new prompt messages.
// Prompts are added to context and events are emitted for them.
//
// The returned channel MUST be consumed until it closes: while the run is
// live the loop blocks on a full channel (backpressure, no event loss), so
// abandoning the channel without canceling ctx leaks the loop goroutine.
// To stop early, cancel ctx and keep draining — after cancellation delivery
// degrades to best-effort and the loop is guaranteed to exit and close the
// channel even if no one is reading.
func AgentLoop(ctx context.Context, prompts []AgentMessage, agentCtx AgentContext, config LoopConfig) <-chan Event {
	ch := make(chan Event, 128)
	sink := eventSink{ctx: ctx, ch: ch}

	go func() {
		defer close(ch)

		var newMessages []AgentMessage

		currentCtx := AgentContext{
			SystemPrompt: agentCtx.SystemPrompt,
			SystemBlocks: agentCtx.SystemBlocks,
			Messages:     copyMessages(agentCtx.Messages),
			Tools:        agentCtx.Tools,
		}

		sink.emit(Event{Type: EventAgentStart})
		sink.emit(Event{Type: EventTurnStart})

		for _, p := range prompts {
			sink.emit(Event{Type: EventMessageStart, Message: p})
			sink.emit(Event{Type: EventMessageEnd, Message: p})
			commitMessage(&currentCtx, &newMessages, config, p)
		}

		runLoop(ctx, &currentCtx, &newMessages, config, sink)
	}()

	return ch
}

// AgentLoopContinue continues from existing context without adding new messages.
// The last message in context must convert to user or tool role via ConvertToLLM.
//
// The returned channel follows the same consumption contract as AgentLoop:
// drain until close, or cancel ctx and keep draining to stop early.
func AgentLoopContinue(ctx context.Context, agentCtx AgentContext, config LoopConfig) <-chan Event {
	ch := make(chan Event, 128)
	sink := eventSink{ctx: ctx, ch: ch}

	if len(agentCtx.Messages) == 0 {
		go func() {
			defer close(ch)
			sink.emitError(ErrNoMessages, &RunSummary{
				EndReason: EndReasonError,
			})
		}()
		return ch
	}

	go func() {
		defer close(ch)

		var newMessages []AgentMessage
		currentCtx := AgentContext{
			SystemPrompt: agentCtx.SystemPrompt,
			SystemBlocks: agentCtx.SystemBlocks,
			Messages:     copyMessages(agentCtx.Messages),
			Tools:        agentCtx.Tools,
		}

		sink.emit(Event{Type: EventAgentStart})
		sink.emit(Event{Type: EventTurnStart})

		runLoop(ctx, &currentCtx, &newMessages, config, sink)
	}()

	return ch
}

// commitMessage is the single entry point for "message enters agent context".
// It appends the message to both the runtime context and the new-messages list,
// and fires the OnMessage hook if configured.
func commitMessage(currentCtx *AgentContext, newMessages *[]AgentMessage, config LoopConfig, msg AgentMessage) {
	currentCtx.Messages = append(currentCtx.Messages, msg)
	*newMessages = append(*newMessages, msg)
	if config.OnMessage != nil {
		config.OnMessage(msg)
	}
}

// runLoop is the main double-loop logic shared by AgentLoop and AgentLoopContinue.
//
// Core loop contracts:
//   - Streamed tool-call lifecycle signals are authoritative; stop reasons are
//     only hints and must not be used as the sole source of tool state.
//   - Tool results are appended to context only after the assistant message
//     that requested them, even when execution started during streaming.
//   - Once a streamed tool call has completed, the turn is no longer retried
//     automatically; replay could duplicate side effects.
//   - Steering stops not-yet-started tools. Started tools follow their
//     InterruptBehavior and may continue or be cancelled.
func runLoop(ctx context.Context, currentCtx *AgentContext, newMessages *[]AgentMessage, config LoopConfig, sink eventSink) {
	type runSummaryState struct {
		toolCalls  int
		toolErrors int
	}
	summaryState := runSummaryState{}
	buildSummary := func(turnCount int, reason EndReason) *RunSummary {
		return &RunSummary{
			TurnCount:  turnCount,
			ToolCalls:  summaryState.toolCalls,
			ToolErrors: summaryState.toolErrors,
			EndReason:  reason,
		}
	}

	maxTurns := config.MaxTurns
	if maxTurns <= 0 {
		maxTurns = defaultMaxTurns
	}

	firstTurn := true
	turnCount := 0
	lengthRecoveryCount := 0
	toolErrors := make(map[string]int) // consecutive failure count per tool

	// Check for steering messages at start
	var pendingMessages []AgentMessage
	if config.GetSteeringMessages != nil {
		pendingMessages = config.GetSteeringMessages()
	}

	// Track last assistant message so StopGuard can inspect what's stopping us.
	var lastAssistantMsg Message

	// Outer loop: continues when follow-up messages arrive after agent would stop
	for {
		hasMoreToolCalls := true
		var steeringAfterTools []AgentMessage
		afterToolExec := false

		// Inner loop: process tool calls and steering messages
		for hasMoreToolCalls || len(pendingMessages) > 0 {
			// Check for context cancellation (Abort)
			if ctx.Err() != nil {
				if config.ShouldEmitAbortMarker != nil && config.ShouldEmitAbortMarker() {
					phase := "inference"
					text := config.AbortMarkerText
					if text == "" {
						text = defaultAbortMarkerText
					}
					if afterToolExec {
						phase = "tool_execution"
						text = config.AbortMarkerToolText
						if text == "" {
							text = defaultAbortMarkerToolText
						}
					}
					abortMsg := AbortMsg(text, phase)
					sink.emit(Event{Type: EventMessageEnd, Message: abortMsg})
					commitMessage(currentCtx, newMessages, config, abortMsg)
				}
				sink.emit(Event{Type: EventError, Err: ctx.Err()})
				sink.emit(Event{Type: EventAgentEnd, NewMessages: *newMessages, Summary: buildSummary(turnCount, EndReasonAborted)})
				return
			}

			if turnCount >= maxTurns {
				sink.emit(Event{Type: EventError, Err: &MaxTurnsError{Limit: maxTurns}})
				sink.emit(Event{Type: EventAgentEnd, NewMessages: *newMessages, Summary: buildSummary(turnCount, EndReasonMaxTurns)})
				return
			}

			if !firstTurn {
				sink.emit(Event{Type: EventTurnStart})
			} else {
				firstTurn = false
			}

			// Process pending messages (inject before next LLM call)
			if len(pendingMessages) > 0 {
				for _, msg := range pendingMessages {
					sink.emit(Event{Type: EventMessageStart, Message: msg})
					sink.emit(Event{Type: EventMessageEnd, Message: msg})
					commitMessage(currentCtx, newMessages, config, msg)
				}
				pendingMessages = nil
			}

			var streamedTools *turnToolExecutor
			hooks := llmCallHooks{
				OnToolCallComplete: func(call ToolCall) {
					if streamedTools == nil {
						streamedTools = newTurnToolExecutor(ctx, currentCtx.Tools, config, toolErrors, sink)
					}
					streamedTools.Add(call)
				},
			}

			// Reset hook for retries: abort any tool executions started by the
			// failed attempt and clear the executor so the next attempt sees
			// a fresh slate (the OnToolCallComplete closure above re-creates it).
			// Only used when ToolsAreIdempotent is set; without it, retries
			// after a streamed tool_call are skipped entirely.
			resetTurnState := func() {
				if streamedTools != nil {
					streamedTools.AbortAndWait()
					streamedTools = nil
				}
			}

			// Call LLM with retry (streaming: events emitted inside callLLM)
			assistantMsg, callInfo, err := callLLMWithRetry(ctx, currentCtx, config, sink, hooks, resetTurnState)
			if err != nil {
				if streamedTools != nil {
					streamedTools.AbortAndWait()
				}
				if ctx.Err() != nil {
					if config.ShouldEmitAbortMarker != nil && config.ShouldEmitAbortMarker() {
						text := config.AbortMarkerText
						if text == "" {
							text = defaultAbortMarkerText
						}
						abortMsg := AbortMsg(text, "inference")
						sink.emit(Event{Type: EventMessageEnd, Message: abortMsg})
						commitMessage(currentCtx, newMessages, config, abortMsg)
					}
					sink.emit(Event{Type: EventError, Err: ctx.Err()})
					sink.emit(Event{Type: EventAgentEnd, NewMessages: *newMessages, Summary: buildSummary(turnCount, EndReasonAborted)})
					return
				}
				sink.emitError(fmt.Errorf("llm call failed: %w", err), buildSummary(turnCount, EndReasonError))
				return
			}

			// Check stop reason — terminate early on error/aborted
			if assistantMsg.StopReason == StopReasonError || assistantMsg.StopReason == StopReasonAborted {
				// A custom ChatModel may report Error/Aborted on a message whose
				// stream already completed tool calls (the bundled litellm
				// adapter never does, but the kernel is model-agnostic). Drain the
				// executor so its child ctx and goroutines don't leak past this
				// early exit — every other exit that can hold streamedTools does.
				if streamedTools != nil {
					streamedTools.AbortAndWait()
				}
				commitMessage(currentCtx, newMessages, config, assistantMsg)
				sink.emit(Event{Type: EventModelResponse, Message: assistantMsg})
				turnCount++
				reason := EndReasonError
				if assistantMsg.StopReason == StopReasonAborted {
					reason = EndReasonAborted
				}
				sink.emit(Event{Type: EventAgentEnd, NewMessages: *newMessages, Summary: buildSummary(turnCount, reason)})
				return
			}

			// When output was truncated (max_tokens hit), tool calls are likely
			// incomplete with malformed JSON args. Strip them to avoid validation
			// errors and API rejections.
			if assistantMsg.StopReason == StopReasonLength && !callInfo.HasCompletedToolCalls {
				assistantMsg.Content = stripToolCallBlocks(assistantMsg.Content)
			}

			lastAssistantMsg = assistantMsg
			commitMessage(currentCtx, newMessages, config, assistantMsg)

			// Check for tool calls
			toolCalls := assistantMsg.ToolCalls()
			summaryState.toolCalls += len(toolCalls)
			hasMoreToolCalls = len(toolCalls) > 0
			// Recover when output was truncated and no tool calls completed.
			// This includes the case where tool call blocks existed but were
			// stripped due to incomplete JSON — the tool was never executed,
			// so recovery is safe. The recovery prompt tells the model to
			// "break remaining work into smaller pieces."
			shouldRecoverLength := assistantMsg.StopReason == StopReasonLength &&
				len(toolCalls) == 0 &&
				!callInfo.HasCompletedToolCalls &&
				lengthRecoveryCount < defaultMaxLengthRecoveries

			var turnToolResults []ToolResult
			if hasMoreToolCalls {
				var steering []AgentMessage
				if callInfo.HasCompletedToolCalls && streamedTools != nil {
					turnToolResults, steering = streamedTools.Wait()
				} else {
					turnToolResults, steering = executeToolCalls(ctx, currentCtx.Tools, toolCalls, config, toolErrors, sink)
				}
				afterToolExec = true

				for _, tr := range turnToolResults {
					resultMsg := toolResultToMessage(tr)
					sink.emit(Event{Type: EventMessageStart, Message: resultMsg})
					sink.emit(Event{Type: EventMessageEnd, Message: resultMsg})
					commitMessage(currentCtx, newMessages, config, resultMsg)
				}

				steeringAfterTools = steering
			}
			for _, tr := range turnToolResults {
				if tr.IsError {
					summaryState.toolErrors++
				}
			}

			sink.emit(Event{Type: EventModelResponse, Message: assistantMsg, ToolResults: turnToolResults})
			turnCount++

			// Early exit: a terminal tool completed successfully. This is a
			// normal stop, so it passes through the same StopGuard gate as
			// end_turn — guards stay the single stop arbiter and can veto a
			// premature terminal-tool exit (Trigger distinguishes the paths).
			if stopAfterToolHit(config, turnToolResults) {
				inject, escalate := consultStopGuard(ctx, config, StopInfo{
					TurnIndex: turnCount,
					Message:   lastAssistantMsg,
					Trigger:   StopTriggerAfterTool,
				})
				if escalate {
					sink.emit(Event{Type: EventError, Err: ErrStopGuard})
					sink.emit(Event{Type: EventAgentEnd, NewMessages: *newMessages, Summary: buildSummary(turnCount, EndReasonError)})
					return
				}
				if inject == "" {
					sink.emit(Event{Type: EventAgentEnd, NewMessages: *newMessages, Summary: buildSummary(turnCount, EndReasonStop)})
					return
				}
				// Guard vetoed the early exit: keep the loop alive with the
				// injected message, carrying any steering captured during this
				// terminal-tool turn so a follow-up tool turn can't drop the
				// already-dequeued steering.
				pendingMessages = append([]AgentMessage{UserMsg(inject)}, steeringAfterTools...)
				steeringAfterTools = nil
				continue
			}

			if shouldRecoverLength {
				lengthRecoveryCount++
				prompt := config.LengthRecoveryPrompt
				if prompt == "" {
					prompt = defaultLengthRecoveryPrompt
				}
				pendingMessages = []AgentMessage{UserMsg(prompt)}
				continue
			}

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

		// StopGuard veto: give the application a chance to keep the loop alive.
		inject, escalate := consultStopGuard(ctx, config, StopInfo{
			TurnIndex: turnCount,
			Message:   lastAssistantMsg,
			Trigger:   StopTriggerEndTurn,
		})
		if escalate {
			sink.emit(Event{Type: EventError, Err: ErrStopGuard})
			sink.emit(Event{Type: EventAgentEnd, NewMessages: *newMessages, Summary: buildSummary(turnCount, EndReasonError)})
			return
		}
		if inject != "" {
			pendingMessages = []AgentMessage{UserMsg(inject)}
			continue
		}

		break
	}

	sink.emit(Event{Type: EventAgentEnd, NewMessages: *newMessages, Summary: buildSummary(turnCount, EndReasonStop)})
}

// stopAfterToolHit reports whether any successful tool result in this turn
// matches a StopAfterTool / StopAfterToolResult hook.
func stopAfterToolHit(config LoopConfig, results []ToolResult) bool {
	if config.StopAfterTool == nil && config.StopAfterToolResult == nil {
		return false
	}
	for _, tr := range results {
		if tr.IsError {
			continue
		}
		if config.StopAfterTool != nil && config.StopAfterTool(tr.ToolName) {
			return true
		}
		if config.StopAfterToolResult != nil && config.StopAfterToolResult(tr.ToolName, tr.Content) {
			return true
		}
	}
	return false
}

// consultStopGuard runs the guard at a would-stop point. A non-empty inject
// keeps the loop alive with that message; escalate ends the run with
// ErrStopGuard. Both zero values mean the stop is allowed (including when no
// guard is configured, or when the guard denies without an InjectMessage —
// never stall silently).
func consultStopGuard(ctx context.Context, config LoopConfig, info StopInfo) (inject string, escalate bool) {
	if config.StopGuard == nil {
		return "", false
	}
	decision := config.StopGuard(ctx, info)
	if decision.Escalate {
		return "", true
	}
	if !decision.Allow && decision.InjectMessage != "" {
		return decision.InjectMessage, false
	}
	return "", false
}

type llmCallHooks struct {
	OnToolCallComplete func(ToolCall)
}

type llmCallInfo struct {
	HasCompletedToolCalls bool
}

// callLLMWithRetry wraps callLLM with retry logic for retryable errors.
// Context overflow errors trigger automatic compaction and a single retry.
//
// resetTurnState, when non-nil, is invoked after a retryable failure and before
// the next attempt. Callers use it to abort and discard any tool executions
// already kicked off via the streaming hook in the failed attempt, so the
// retried attempt can re-execute the tool calls cleanly. Only invoked when
// the failure path actually decides to retry (i.e. the error is retryable
// and the attempt cap has not been reached).
func callLLMWithRetry(ctx context.Context, agentCtx *AgentContext, config LoopConfig, sink eventSink, hooks llmCallHooks, resetTurnState func()) (Message, llmCallInfo, error) {
	maxRetries := config.MaxRetries
	if maxRetries <= 0 {
		msg, info, err := callLLM(ctx, agentCtx, config, sink, hooks)
		if err != nil && IsContextOverflow(err) {
			return recoverOverflow(ctx, agentCtx, config, sink, err, hooks)
		}
		return msg, info, err
	}

	var lastErr error
	var lastInfo llmCallInfo
	for attempt := 0; attempt <= maxRetries; attempt++ {
		msg, info, err := callLLM(ctx, agentCtx, config, sink, hooks)
		if err == nil {
			return msg, info, nil
		}
		lastErr = err
		lastInfo = info

		// Context overflow: compact and retry once (not a normal retry)
		if IsContextOverflow(err) {
			return recoverOverflow(ctx, agentCtx, config, sink, err, hooks)
		}

		// Once streamed tool calls have already completed, retrying this turn
		// risks duplicating side effects from tools that already started — unless
		// the caller has declared its tools idempotent, in which case we abort
		// the in-flight executions and retry the whole turn cleanly.
		if info.HasCompletedToolCalls && !config.ToolsAreIdempotent {
			return Message{}, info, err
		}

		// User cancellation is never retryable: the next attempt would just
		// rediscover ctx.Done(), and emitting EventRetry in that window surfaces
		// confusing "retry (1/N)" messages to users who already aborted. Aligns
		// with IsFailoverEligible (errors.go), which also treats
		// context.Canceled as terminal.
		if errors.Is(err, context.Canceled) {
			return Message{}, info, err
		}

		// PartialStreamError (stream closed without done) is treated as retryable:
		// it most often signals a transient network/provider stream-format issue
		// that a fresh request can recover from. The HasCompletedToolCalls guard
		// above already prevents retrying after a tool side-effect has fired.
		var pse *PartialStreamError
		retryable := isRetryable(err) || errors.As(err, &pse)
		if !retryable || attempt == maxRetries {
			return Message{}, info, err
		}

		// Discard any tool executions started during the failed attempt before
		// the next retry — otherwise the streaming hook would Add a second copy
		// of the same tool_call onto the same executor on the next callLLM.
		if resetTurnState != nil {
			resetTurnState()
		}

		delay := retryDelay(err, attempt)

		sink.emit(Event{
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
			return Message{}, lastInfo, ctx.Err()
		case <-time.After(delay):
		}
	}
	return Message{}, lastInfo, lastErr
}

// recoverOverflow attempts to compact the context via the ContextManager and
// retry the LLM call. If no ContextManager is configured, the original error
// is returned.
func recoverOverflow(ctx context.Context, agentCtx *AgentContext, config LoopConfig, sink eventSink, originalErr error, hooks llmCallHooks) (Message, llmCallInfo, error) {
	if config.ContextManager == nil {
		return Message{}, llmCallInfo{}, &ContextOverflowError{Cause: fmt.Errorf("no compaction configured: %w", originalErr)}
	}

	sink.emit(Event{
		Type: EventRetry,
		Err:  originalErr,
		RetryInfo: &RetryInfo{
			Attempt:    1,
			MaxRetries: 1,
			Err:        fmt.Errorf("context overflow detected, compacting and retrying"),
		},
	})

	recovery, err := config.ContextManager.RecoverOverflow(ctx, agentCtx.Messages, originalErr)
	if err != nil {
		return Message{}, llmCallInfo{}, &ContextOverflowError{Cause: fmt.Errorf("compaction failed: %w", err)}
	}
	if len(recovery.View) == 0 {
		return Message{}, llmCallInfo{}, &ContextOverflowError{Cause: errors.New("compaction returned empty prompt view")}
	}
	agentCtx.Messages = recovery.View
	if recovery.ShouldCommit && len(recovery.CommitMessages) > 0 && config.CommitContext != nil {
		if err := config.CommitContext(recovery.CommitMessages, recovery.Usage); err != nil {
			return Message{}, llmCallInfo{}, &ContextOverflowError{Cause: fmt.Errorf("commit failed: %w", err)}
		}
	}
	return callLLM(ctx, agentCtx, config, sink, hooks)
}

// retryDelay calculates the wait duration using exponential backoff.
// Respects Retry-After from rate limit errors. Capped at defaultMaxRetryDelay.
func retryDelay(err error, attempt int) time.Duration {
	maxDelay := defaultMaxRetryDelay
	if after := retryAfterHint(err); after > 0 {
		d := after
		if d > maxDelay {
			d = maxDelay
		}
		return d
	}
	// Exponential backoff: 1s, 2s, 4s, 8s...
	d := time.Duration(math.Pow(2, float64(attempt))) * time.Second
	if d > maxDelay {
		d = maxDelay
	}
	return d
}

// callLLM applies the two-stage pipeline and calls the model.
func callLLM(ctx context.Context, agentCtx *AgentContext, config LoopConfig, sink eventSink, hooks llmCallHooks) (Message, llmCallInfo, error) {
	messages := agentCtx.Messages

	// Stage 1: ContextManager / TransformContext
	if config.ContextManager != nil {
		projection, err := config.ContextManager.Project(ctx, messages)
		if err != nil {
			return Message{}, llmCallInfo{}, fmt.Errorf("project context: %w", err)
		}
		if projection.ShouldCommit && len(projection.CommitMessages) > 0 {
			if config.CommitContext != nil {
				if err := config.CommitContext(projection.CommitMessages, projection.Usage); err != nil {
					return Message{}, llmCallInfo{}, fmt.Errorf("project context commit failed: %w", err)
				}
			}
			agentCtx.Messages = copyMessages(projection.CommitMessages)
			messages = copyMessages(projection.CommitMessages)
		}
		if projection.Messages != nil {
			messages = projection.Messages
		}
	}

	// Stage 2: ConvertToLLM (AgentMessage[] → Message[]) + repair tool-call /
	// tool-result pairing for provider compatibility.
	convertFn := config.ConvertToLLM
	if convertFn == nil {
		convertFn = DefaultConvertToLLM
	}
	llmMessages := RepairMessageSequence(convertFn(messages))

	// Build tool specs
	toolSpecs := buildToolSpecs(agentCtx.Tools)

	// Prepend the static system prompt as first message(s). Keeping it at the
	// head and byte-stable across turns lets providers with prefix-based
	// caching (OpenAI) serve it from cache.
	var prefix []Message
	if len(agentCtx.SystemBlocks) > 0 {
		for _, b := range agentCtx.SystemBlocks {
			m := SystemMsg(b.Text)
			if b.CacheControl != "" {
				m.Metadata = map[string]any{"cache_control": b.CacheControl}
			}
			prefix = append(prefix, m)
		}
	} else if agentCtx.SystemPrompt != "" {
		prefix = append(prefix, SystemMsg(agentCtx.SystemPrompt))
	}
	if len(prefix) > 0 {
		llmMessages = append(prefix, llmMessages...)
	}

	// Place a single cache write breakpoint on the last non-system message when
	// the application has opted into explicit cache orchestration. The helper
	// scans from the tail and skips system reminders, so the breakpoint lands
	// on the freshest user input / tool_result / assistant turn — whichever is
	// last in this request. Inside a tool loop this means each LLM call writes
	// an entry covering the latest tool_use+tool_result, so the next call in
	// the loop reads them from cache instead of re-uploading.
	if config.CacheLastMessage != "" {
		llmMessages = markLastMessageForCache(llmMessages, config.CacheLastMessage)
	}

	if config.Model == nil {
		return Message{}, llmCallInfo{}, ErrNoModel
	}

	// Build per-call options
	var callOpts []CallOption

	// Thinking level (provider-default budget; per-call WithThinkingBudget can be
	// applied by ChatModel adapters that need to override).
	if config.ThinkingLevel != "" && config.ThinkingLevel != ThinkingOff {
		callOpts = append(callOpts, WithThinking(config.ThinkingLevel))
	}

	// Use streaming for real-time token deltas
	return callLLMStream(ctx, config.Model, llmMessages, toolSpecs, callOpts, sink, hooks)
}

// markLastMessageForCache returns a copy of messages with cache_control attached
// to the metadata of the last non-system message. System messages are skipped so
// trailing per-turn reminders (which change every turn) don't end up carrying
// the breakpoint. The caller's slice and the original Message values are left
// untouched.
func markLastMessageForCache(messages []Message, cacheControl string) []Message {
	idx := -1
	for i := len(messages) - 1; i >= 0; i-- {
		if messages[i].Role != RoleSystem {
			idx = i
			break
		}
	}
	if idx < 0 {
		return messages
	}
	out := slices.Clone(messages)
	md := maps.Clone(out[idx].Metadata)
	if md == nil {
		md = map[string]any{}
	}
	md["cache_control"] = cacheControl
	out[idx].Metadata = md
	return out
}

// callLLMStream uses GenerateStream and emits real-time events.
// The adapter builds partial Messages with ContentBlocks and emits fine-grained StreamEvents.
//
// Stream init failure is surfaced as an error — there is no silent fallback to
// non-streaming Generate, because callers (TUIs, event subscribers) typically
// depend on stream events for live rendering, tool-call deltas, and cancellation
// semantics. Switching execution model without notice changes the contract.
func callLLMStream(ctx context.Context, model ChatModel, messages []Message, tools []ToolSpec, opts []CallOption, sink eventSink, hooks llmCallHooks) (Message, llmCallInfo, error) {
	streamCh, err := model.GenerateStream(ctx, messages, tools, opts...)
	if err != nil {
		return Message{}, llmCallInfo{}, fmt.Errorf("stream init failed: %w", err)
	}

	var (
		started bool
		partial Message
		info    llmCallInfo
	)

	for ev := range streamCh {
		switch ev.Type {
		case StreamEventTextStart, StreamEventThinkingStart, StreamEventToolCallStart:
			partial = ev.Message
			if !started {
				started = true
				sink.emit(Event{Type: EventMessageStart, Message: partial})
			}

		case StreamEventTextDelta, StreamEventThinkingDelta, StreamEventToolCallDelta:
			partial = ev.Message
			if !started {
				started = true
				sink.emit(Event{Type: EventMessageStart, Message: partial})
			}
			var dk DeltaKind
			switch ev.Type {
			case StreamEventThinkingDelta:
				dk = DeltaThinking
			case StreamEventToolCallDelta:
				dk = DeltaToolCall
			}
			sink.emit(Event{Type: EventMessageUpdate, Message: partial, Delta: ev.Delta, DeltaKind: dk})

		case StreamEventTextEnd, StreamEventThinkingEnd, StreamEventToolCallEnd:
			partial = ev.Message
			if ev.CompletedToolCall != nil {
				info.HasCompletedToolCalls = true
				if hooks.OnToolCallComplete != nil {
					hooks.OnToolCallComplete(*ev.CompletedToolCall)
				}
			}

		case StreamEventDone:
			finalMsg := ev.Message
			finalMsg.Timestamp = time.Now()
			if !started {
				sink.emit(Event{Type: EventMessageStart, Message: finalMsg})
			}
			sink.emit(Event{Type: EventMessageEnd, Message: finalMsg})
			return finalMsg, info, nil

		case StreamEventError:
			return Message{}, info, ev.Err
		}
	}

	// Stream closed without done event — surface as PartialStreamError instead
	// of pretending the message completed. Emitting EventMessageEnd here would
	// let callers persist a half-finished message (missing StopReason, possibly
	// truncated tool_call args, unclosed thinking blocks) into history — the
	// next LLM call would then see structurally invalid context.
	return Message{}, info, &PartialStreamError{Partial: partial}
}

// executeToolCalls runs tool calls for one assistant turn using the shared
// turn executor. The same executor also powers streaming tool execution.
func executeToolCalls(ctx context.Context, tools []Tool, calls []ToolCall, config LoopConfig, toolErrors map[string]int, sink eventSink) ([]ToolResult, []AgentMessage) {
	exec := newTurnToolExecutor(ctx, tools, config, toolErrors, sink)
	for _, call := range calls {
		exec.Add(call)
	}
	return exec.Wait()
}

// executeSingleToolCall executes one tool call: emit events, validate, preview,
// run approval, then execute the tool. result.ToolName is set when the caller
// should update toolErrors; empty means skip (circuit-breaker hit, denial, or
// context cancellation).
func executeSingleToolCall(ctx context.Context, tools []Tool, call ToolCall, config LoopConfig, failCount int, sink eventSink) ToolResult {
	tool := findTool(tools, call.Name)
	label := toolLabel(tool)

	// Fast exit: context already cancelled — don't start any tool work.
	if ctx.Err() != nil {
		content, _ := json.Marshal("Tool execution cancelled.")
		result := ToolResult{ToolCallID: call.ID, Content: content, IsError: true}
		sink.emit(Event{
			Type:      EventToolExecStart,
			ToolID:    call.ID,
			Tool:      call.Name,
			ToolLabel: label,
			Args:      call.Args,
		})
		sink.emit(Event{
			Type:      EventToolExecEnd,
			ToolID:    call.ID,
			Tool:      call.Name,
			ToolLabel: label,
			Result:    result.Content,
			IsError:   true,
		})
		return result
	}

	// Circuit breaker: skip if tool has exceeded consecutive failure threshold
	if config.MaxToolErrors > 0 && failCount >= config.MaxToolErrors {
		sink.emit(Event{
			Type:      EventToolExecStart,
			ToolID:    call.ID,
			Tool:      call.Name,
			ToolLabel: label,
			Args:      call.Args,
		})
		errContent, _ := json.Marshal(fmt.Sprintf("tool %q disabled after %d consecutive errors", call.Name, config.MaxToolErrors))
		result := ToolResult{ToolCallID: call.ID, Content: errContent, IsError: true}
		sink.emit(Event{
			Type:    EventToolExecEnd,
			ToolID:  call.ID,
			Tool:    call.Name,
			Result:  result.Content,
			IsError: true,
		})
		return result // ToolName empty → don't count
	}

	sink.emit(Event{
		Type:      EventToolExecStart,
		ToolID:    call.ID,
		Tool:      call.Name,
		ToolLabel: label,
		Args:      call.Args,
	})

	var result ToolResult

	if tool == nil {
		errContent, _ := json.Marshal(fmt.Sprintf("tool %q not found", call.Name))
		result = ToolResult{
			ToolCallID: call.ID,
			Content:    errContent,
			IsError:    true,
		}
	} else if fixed, err := validateToolArgs(tool, call); err != nil {
		errContent, _ := json.Marshal(err.Error())
		result = ToolResult{
			ToolCallID: call.ID,
			ToolName:   call.Name,
			Content:    errContent,
			IsError:    true,
		}
	} else {
		// Schema validation may have coerced trivially mis-typed args (e.g. a
		// stringified number or JSON array) into clean values; run the tool with
		// the corrected payload so downstream Validators and the tool itself see
		// well-typed input.
		if fixed != nil {
			call.Args = fixed
		}
		// Stage 1: business-level input validation. Distinct from schema
		// validation above — Validators check semantics (write-before-read,
		// mtime drift, ...) using state the tool was constructed with.
		// Failures are surfaced as a normal tool_result with IsError=true so
		// the LLM can self-correct without prompting the user. Validators
		// MUST NOT prompt or mutate persistent state.
		if v, ok := tool.(Validator); ok {
			vr := v.Validate(ctx, call.Args)
			if !vr.OK {
				msg := vr.Message
				if msg == "" {
					msg = "tool input validation failed"
				}
				content, _ := json.Marshal(msg)
				result = ToolResult{
					ToolCallID: call.ID,
					ToolName:   call.Name,
					Content:    content,
					IsError:    true,
				}
				sink.emit(Event{
					Type:      EventToolExecEnd,
					ToolID:    call.ID,
					Tool:      call.Name,
					ToolLabel: label,
					Result:    content,
					IsError:   true,
				})
				return result
			}
		}

		var preview json.RawMessage

		// Preview: if tool supports it, compute and emit preview before execution.
		// Preview runs only after args are validated so approval UIs can use it.
		if p, ok := tool.(Previewer); ok {
			if data, err := p.Preview(ctx, call.Args); err == nil {
				preview = data
				sink.emit(Event{
					Type:       EventToolExecUpdate,
					ToolID:     call.ID,
					Tool:       call.Name,
					ToolLabel:  label,
					Args:       call.Args,
					Result:     data,
					UpdateKind: ToolExecUpdatePreview,
				})
			}
		}

		if config.ToolGate != nil {
			gateReq := GateRequest{
				Tool:      tool,
				Call:      call,
				ToolLabel: label,
				Preview:   preview,
			}
			decision, err := config.ToolGate(ctx, gateReq)
			if err != nil {
				decision = &GateDecision{Allowed: false, Reason: err.Error()}
			}
			if decision != nil && !decision.Allowed {
				reason := decision.Reason
				if reason == "" {
					reason = "tool execution denied"
				}
				errContent, _ := json.Marshal(reason)
				result = ToolResult{
					ToolCallID: call.ID,
					Content:    errContent,
					IsError:    true,
				}
				sink.emit(Event{
					Type:      EventToolExecEnd,
					ToolID:    call.ID,
					Tool:      call.Name,
					ToolLabel: label,
					Result:    result.Content,
					IsError:   true,
				})
				return result
			}
		}

		// Inject progress callback so tools can report partial results
		progressCtx := WithToolProgress(ctx, func(progress ProgressPayload) {
			p := progress
			sink.emit(Event{
				Type:       EventToolExecUpdate,
				ToolID:     call.ID,
				Tool:       call.Name,
				ToolLabel:  label,
				Args:       call.Args,
				Progress:   &p,
				UpdateKind: ToolExecUpdateProgress,
			})
		})

		// ContentTool: returns rich content blocks (e.g., images).
		// When middleware is configured, execute it through a shim so logging,
		// auditing, and short-circuit behavior still apply.
		if ct, ok := tool.(ContentTool); ok {
			blocks, output, execErr := executeContentTool(progressCtx, tool, ct, call, config.Middlewares)
			if execErr != nil {
				errContent, _ := json.Marshal(execErr.Error())
				result = ToolResult{
					ToolCallID: call.ID,
					Content:    errContent,
					IsError:    true,
				}
			} else {
				// When tool_reference blocks are present, keep them alongside
				// any text blocks in ContentBlocks. The provider layer splits
				// tool_reference into tool_result content and text into sibling
				// blocks within the same user message.
				refBlocks, siblingText := splitToolRefBlocks(blocks)
				if len(refBlocks) > 0 {
					resultBlocks := make([]ContentBlock, 0, len(refBlocks)+1)
					resultBlocks = append(resultBlocks, refBlocks...)
					if siblingText != "" {
						resultBlocks = append(resultBlocks, TextBlock(siblingText))
					}
					result = ToolResult{
						ToolCallID:    call.ID,
						Content:       pickContentSummary(output, blocks),
						ContentBlocks: resultBlocks,
					}
				} else {
					summary := pickContentSummary(output, blocks)
					result = ToolResult{
						ToolCallID:    call.ID,
						Content:       summary,
						ContentBlocks: blocks,
					}
				}
			}
		} else {
			var output json.RawMessage
			var execErr error
			if len(config.Middlewares) > 0 {
				exec := buildMiddlewareChain(call, tool.Execute, config.Middlewares)
				output, execErr = exec(progressCtx, call.Args)
			} else {
				output, execErr = tool.Execute(progressCtx, call.Args)
			}
			if execErr != nil {
				errContent, _ := json.Marshal(execErr.Error())
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
	}

	sink.emit(Event{
		Type:      EventToolExecEnd,
		ToolID:    call.ID,
		Tool:      call.Name,
		ToolLabel: label,
		Result:    result.Content,
		IsError:   result.IsError,
	})

	// Mark for toolErrors tracking by caller.
	result.ToolName = call.Name
	return result
}

// skipToolCall creates a skipped result for an interrupted tool call.
func skipToolCall(call ToolCall, tools []Tool, sink eventSink) ToolResult {
	return skipToolCallWithMessage(call, tools, sink, "Skipped due to queued user message.")
}

func skipToolCallWithMessage(call ToolCall, tools []Tool, sink eventSink, message string) ToolResult {
	label := toolLabel(findTool(tools, call.Name))

	sink.emit(Event{
		Type:      EventToolExecStart,
		ToolID:    call.ID,
		Tool:      call.Name,
		ToolLabel: label,
		Args:      call.Args,
	})

	content, _ := json.Marshal(message)
	result := ToolResult{
		ToolCallID: call.ID,
		Content:    content,
		IsError:    true,
	}

	sink.emit(Event{
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
	if len(tr.ContentBlocks) > 0 {
		return Message{
			Role:    RoleTool,
			Content: tr.ContentBlocks,
			Metadata: map[string]any{
				"tool_call_id": tr.ToolCallID,
				"tool_name":    tr.ToolName,
				"is_error":     tr.IsError,
			},
			Timestamp: time.Now(),
		}
	}
	msg := ToolResultMsg(tr.ToolCallID, tr.Content, tr.IsError)
	if tr.ToolName != "" {
		if msg.Metadata == nil {
			msg.Metadata = make(map[string]any)
		}
		msg.Metadata["tool_name"] = tr.ToolName
	}
	return msg
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
// When a DeferFilter is present among the tools, unactivated deferred tools
// are excluded and activated deferred tools are sent with DeferLoading: true.
func buildToolSpecs(tools []Tool) []ToolSpec {
	if len(tools) == 0 {
		return nil
	}

	var filter DeferFilter
	for _, t := range tools {
		if f, ok := t.(DeferFilter); ok {
			filter = f
			break
		}
	}

	specs := make([]ToolSpec, 0, len(tools))
	for _, t := range tools {
		spec := ToolSpec{
			Name:        t.Name(),
			Description: t.Description(),
			Parameters:  t.Schema(),
		}
		if filter != nil && filter.IsDeferred(t.Name()) {
			continue // unactivated deferred → don't send schema
		}
		if filter != nil && filter.WasDeferred(t.Name()) {
			spec.DeferLoading = true // activated deferred → send with defer_loading
		}
		if s, ok := t.(StrictSchemaTool); ok {
			strict := s.StrictSchema()
			spec.Strict = &strict
		}
		specs = append(specs, spec)
	}
	return specs
}

// validateToolArgs validates tool call arguments against the tool's JSON Schema.
// Collects every missing-required and type-mismatch issue in one pass so weaker
// models can self-correct in a single follow-up turn instead of trial-and-error.
// Error text is natural-language, one line per issue — empirically recovers
// faster than terse errors.
//
// When call.ArgsInvalid is set (LLM emitted unparseable args; raw payload
// captured upstream in call.ArgsRawText / call.ArgsParseError), schema checks
// are skipped and the captured diagnostic is surfaced directly — running
// schema validation against the "{}" placeholder would otherwise mislead the
// model with a "missing field" error.
func validateToolArgs(tool Tool, call ToolCall) (json.RawMessage, error) {
	if call.ArgsInvalid {
		return nil, fmt.Errorf(
			"%w: %s received malformed JSON arguments: %s\nraw args: %s",
			ErrToolValidation, tool.Name(), call.ArgsParseError, call.ArgsRawText,
		)
	}

	schema := tool.Schema()
	if schema == nil {
		return nil, nil
	}

	args := call.Args
	if len(args) == 0 {
		args = []byte("{}")
	}
	var parsed map[string]any
	if err := json.Unmarshal(args, &parsed); err != nil {
		return nil, fmt.Errorf("%w: %s received invalid JSON arguments: %v",
			ErrToolValidation, tool.Name(), err)
	}

	var issues []ValidationIssue

	if reqSlice, ok := schema["required"]; ok {
		if required, ok := reqSlice.([]string); ok {
			for _, field := range required {
				if _, exists := parsed[field]; !exists {
					issues = append(issues, ValidationIssue{Kind: IssueMissing, Path: field})
				}
			}
		}
	}

	// coerced tracks whether any value was recovered from a trivially-fixable
	// type mismatch (e.g. a stringified number or JSON array). When set, the
	// corrected args are re-marshaled and returned so the tool receives clean
	// values instead of the loop stalling on a model that keeps re-sending the
	// same mis-typed call. The original (mis-typed) args remain visible in the
	// preceding EventToolExecStart, so the recovery stays observable.
	coerced := false
	if props, ok := schema["properties"].(map[string]any); ok {
		for key, val := range parsed {
			// JSON null on an optional field is equivalent to omitting the field.
			// Some LLMs emit "field": null instead of dropping the key; rejecting
			// those as type mismatches derails tool use on the very first call.
			if val == nil {
				continue
			}
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
			if received, hint, mismatch := typeMismatch(val, expectedType); mismatch {
				if fixed, ok := coerceArg(val, expectedType); ok {
					parsed[key] = fixed
					coerced = true
					continue
				}
				issues = append(issues, ValidationIssue{
					Kind:     IssueType,
					Path:     key,
					Expected: expectedType,
					Received: received,
					Hint:     hint,
				})
			}
		}
	}

	if len(issues) > 0 {
		return nil, &ToolValidationError{ToolName: tool.Name(), Issues: issues}
	}
	if coerced {
		if fixed, err := json.Marshal(parsed); err == nil {
			return fixed, nil
		}
		// Re-marshaling a freshly-unmarshaled map should never fail; if it
		// somehow does, fall through and let the tool run with original args.
	}
	return nil, nil
}

// coerceArg recovers a trivially-fixable type mismatch: a value the model
// emitted as a JSON string when the schema wants a number, array, or object.
// It returns the corrected Go value and true only when the string parses
// unambiguously into the expected type — e.g. "119" → 119 for an integer, or
// `["a","b"]` → []any for an array. Anything ambiguous is left untouched so the
// caller still surfaces a validation error. This rescues weaker models and
// providers that stringify tool-call arguments and would otherwise loop forever
// re-sending the same mis-typed call.
func coerceArg(val any, expected string) (any, bool) {
	s, ok := val.(string)
	if !ok {
		return nil, false
	}
	t := strings.TrimSpace(s)
	switch expected {
	case "integer":
		if n, err := strconv.ParseInt(t, 10, 64); err == nil {
			return float64(n), true
		}
	case "number":
		if f, err := strconv.ParseFloat(t, 64); err == nil {
			return f, true
		}
	case "array":
		var a []any
		if json.Unmarshal([]byte(t), &a) == nil {
			return a, true
		}
	case "object":
		var m map[string]any
		if json.Unmarshal([]byte(t), &m) == nil {
			return m, true
		}
	}
	return nil, false
}

// typeMismatch reports whether val conflicts with the JSON Schema type. When it
// does, the returned name is the user-facing JSON type name of the actual value
// ("string" / "integer" / "number" / "boolean" / "array" / "object"). The hint
// is non-empty only for known LLM mistake patterns — currently the
// "JSON-encoded literal passed as a string" mistake that weaker models repeat
// across turns without nudging.
func typeMismatch(val any, expected string) (received string, hint string, mismatch bool) {
	switch expected {
	case "string":
		if _, ok := val.(string); ok {
			return "", "", false
		}
	case "integer":
		if f, ok := val.(float64); ok && f == float64(int64(f)) {
			return "", "", false
		}
	case "number":
		if _, ok := val.(float64); ok {
			return "", "", false
		}
	case "boolean":
		if _, ok := val.(bool); ok {
			return "", "", false
		}
	case "array":
		if _, ok := val.([]any); ok {
			return "", "", false
		}
	case "object":
		if _, ok := val.(map[string]any); ok {
			return "", "", false
		}
	default:
		return "", "", false
	}
	return jsonTypeName(val), mismatchHint(val, expected), true
}

// mismatchHint catches the single most common LLM mistake: serializing an
// array/object to a JSON string and passing the string. The hint nudges the
// model to drop the surrounding quotes on the next turn. Other mismatches get
// no hint to keep the error message terse.
func mismatchHint(val any, expected string) string {
	if expected != "array" && expected != "object" {
		return ""
	}
	s, ok := val.(string)
	if !ok {
		return ""
	}
	trimmed := strings.TrimSpace(s)
	if expected == "array" && strings.HasPrefix(trimmed, "[") {
		return `Looks like a JSON-encoded array — pass the value directly (e.g. ["a","b"]), not wrapped in quotes.`
	}
	if expected == "object" && strings.HasPrefix(trimmed, "{") {
		return `Looks like a JSON-encoded object — pass the value directly (e.g. {"k":"v"}), not wrapped in quotes.`
	}
	return ""
}

// jsonTypeName returns the JSON Schema type name for val.
func jsonTypeName(val any) string {
	switch v := val.(type) {
	case nil:
		return "null"
	case bool:
		return "boolean"
	case string:
		return "string"
	case float64:
		if v == float64(int64(v)) {
			return "integer"
		}
		return "number"
	case []any:
		return "array"
	case map[string]any:
		return "object"
	default:
		return fmt.Sprintf("%T", val)
	}
}

// buildMiddlewareChain wraps a tool execution function with the middleware stack.
// Outermost middleware is called first; innermost calls the actual tool.
func buildMiddlewareChain(call ToolCall, exec ToolExecuteFunc, middlewares []ToolMiddleware) ToolExecuteFunc {
	for i := len(middlewares) - 1; i >= 0; i-- {
		mw := middlewares[i]
		next := exec
		exec = func(ctx context.Context, args json.RawMessage) (json.RawMessage, error) {
			return mw(ctx, call, next)
		}
	}
	return exec
}

func executeContentTool(ctx context.Context, tool Tool, ct ContentTool, call ToolCall, middlewares []ToolMiddleware) ([]ContentBlock, json.RawMessage, error) {
	var blocks []ContentBlock
	baseExec := func(ctx context.Context, args json.RawMessage) (json.RawMessage, error) {
		out, err := ct.ExecuteContent(ctx, args)
		if err != nil {
			return nil, err
		}
		blocks = out
		return contentBlocksTextSummary(out), nil
	}

	exec := baseExec
	if len(middlewares) > 0 {
		exec = buildMiddlewareChain(call, exec, middlewares)
	}

	output, err := exec(ctx, call.Args)
	return blocks, output, err
}

func pickContentSummary(output json.RawMessage, blocks []ContentBlock) json.RawMessage {
	if len(output) > 0 {
		return output
	}
	return contentBlocksTextSummary(blocks)
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

// splitToolRefBlocks separates tool_reference blocks from text blocks.
// Returns the tool_reference blocks and concatenated text from text blocks.
// Used to format tool search results: tool_reference goes into tool_result
// content, text becomes a sibling block outside the tool_result.
func splitToolRefBlocks(blocks []ContentBlock) (refs []ContentBlock, text string) {
	var texts []string
	for _, b := range blocks {
		switch b.Type {
		case ContentToolRef:
			refs = append(refs, b)
		case ContentText:
			if b.Text != "" {
				texts = append(texts, b.Text)
			}
		}
	}
	text = strings.Join(texts, "\n")
	return
}

// contentBlocksTextSummary extracts text from ContentBlocks as a JSON string
// for the Event.Result field. Returns nil if no text content.
func contentBlocksTextSummary(blocks []ContentBlock) json.RawMessage {
	var texts []string
	for _, b := range blocks {
		if b.Type == ContentText {
			texts = append(texts, b.Text)
		}
	}
	if len(texts) == 0 {
		return nil
	}
	summary, _ := json.Marshal(strings.Join(texts, "\n"))
	return summary
}

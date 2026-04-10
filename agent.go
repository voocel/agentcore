package agentcore

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/voocel/agentcore/permission"
)

// AgentState is a snapshot of the agent's current state.
type AgentState struct {
	SystemPrompt     string
	Messages         []AgentMessage
	Tools            []Tool
	IsRunning        bool
	StreamMessage    AgentMessage        // partial message being streamed, nil when idle
	PendingToolCalls map[string]struct{} // tool call IDs currently executing
	TotalUsage       Usage               // cumulative token usage across all turns
	Error            string
}

// Agent is a stateful wrapper around the agent loop.
// It consumes loop events to update internal state, just like any external listener.
type Agent struct {
	// Configuration (set via options)
	model                 ChatModel
	systemPrompt          string
	systemBlocks          []SystemBlock
	tools                 []Tool
	maxTurns              int
	maxRetries            int
	maxToolErrors         int
	strictMessageSequence bool
	thinkingLevel         ThinkingLevel
	streamFn              StreamFn
	contextManager        ContextManager
	transformContext      func(ctx context.Context, msgs []AgentMessage) ([]AgentMessage, error)
	convertToLLM          func([]AgentMessage) []Message
	steeringMode          QueueMode
	followUpMode          QueueMode
	contextWindow         int
	contextEstimateFn     ContextEstimateFn
	permissionEngine      permission.DecisionEngine
	getApiKey             func(provider string) (string, error)
	thinkingBudgets       map[ThinkingLevel]int
	sessionID             string
	middlewares           []ToolMiddleware
	maxRetryDelay         time.Duration
	maxToolConcurrency    int
	toolChoice            any // default tool_choice for LLM calls
	taskRuntime           *TaskRuntime

	// State
	messages         []AgentMessage
	isRunning        bool
	lastError        string
	streamMessage    AgentMessage        // partial message during streaming
	pendingToolCalls map[string]struct{} // tool call IDs in flight
	totalUsage       Usage               // cumulative token usage

	// Queues
	steeringQ                   []AgentMessage
	followUpQ                   []AgentMessage
	skipNextInitialSteeringPoll bool

	// Lifecycle
	listeners       []func(Event)
	cancel          context.CancelFunc
	done            chan struct{} // closed when loop finishes
	wantAbortMarker atomic.Bool   // set by Abort(), read by runLoop
	mu              sync.Mutex
}

// NewAgent creates a new Agent with the given options.
//
// When a ContextManager is set, the agent automatically wires ConvertToLLM and
// ContextEstimate from the manager if the manager implements the optional
// ContextLLMConverter and/or ContextEstimator interfaces — no need to set them
// manually.
func NewAgent(opts ...AgentOption) *Agent {
	a := &Agent{
		maxTurns:         defaultMaxTurns,
		maxRetries:       defaultMaxRetries,
		maxToolErrors:    defaultMaxToolErrors,
		thinkingLevel:    ThinkingLow,
		steeringMode:     QueueModeAll,
		followUpMode:     QueueModeAll,
		pendingToolCalls: make(map[string]struct{}),
	}
	for _, opt := range opts {
		opt(a)
	}
	// Auto-wire ConvertToLLM and ContextEstimate from ContextManager if
	// the manager provides them and the user hasn't set them explicitly.
	if a.contextManager != nil {
		if a.convertToLLM == nil {
			if c, ok := a.contextManager.(ContextLLMConverter); ok {
				a.convertToLLM = c.ConvertToLLM
			}
		}
		if a.contextEstimateFn == nil {
			if e, ok := a.contextManager.(ContextEstimator); ok {
				a.contextEstimateFn = e.EstimateContext
			}
		}
		if a.contextWindow <= 0 {
			if w, ok := a.contextManager.(ContextWindower); ok {
				a.contextWindow = w.ContextWindow()
			}
		}
	}
	return a
}

// Subscribe registers a listener for agent events. Returns an unsubscribe function.
func (a *Agent) Subscribe(fn func(Event)) func() {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.listeners = append(a.listeners, fn)
	idx := len(a.listeners) - 1
	return func() {
		a.mu.Lock()
		defer a.mu.Unlock()
		a.listeners[idx] = nil
	}
}

// Prompt starts a new conversation turn with the given input.
func (a *Agent) Prompt(input string) error {
	return a.PromptMessages(UserMsg(input))
}

// PromptMessages starts a new conversation turn with arbitrary AgentMessages.
func (a *Agent) PromptMessages(msgs ...AgentMessage) error {
	a.mu.Lock()
	if a.isRunning {
		a.mu.Unlock()
		return fmt.Errorf("agent is already running; use Steer() or FollowUp() to queue messages")
	}
	a.startPromptRunLocked(msgs)
	return nil
}

// Continue resumes from the current context without adding new messages.
// If the last message is from assistant, it dequeues steering/follow-up
func (a *Agent) Continue() error {
	a.mu.Lock()
	if a.isRunning {
		a.mu.Unlock()
		return fmt.Errorf("agent is already running")
	}
	if len(a.messages) == 0 {
		a.mu.Unlock()
		return fmt.Errorf("no messages to continue from")
	}

	// If last message is assistant, try to dequeue pending messages as new prompt
	lastMsg := a.messages[len(a.messages)-1]
	if lastMsg.GetRole() == RoleAssistant {
		if queued := dequeue(&a.steeringQ, a.steeringMode); len(queued) > 0 {
			a.skipNextInitialSteeringPoll = true
			a.startPromptRunLocked(queued)
			return nil
		}
		if queued := dequeue(&a.followUpQ, a.followUpMode); len(queued) > 0 {
			a.skipNextInitialSteeringPoll = true
			a.startPromptRunLocked(queued)
			return nil
		}
		a.mu.Unlock()
		return fmt.Errorf("cannot continue from assistant message without queued messages")
	}

	a.startContinueRunLocked()
	return nil
}

// Steer queues a steering message to interrupt the agent mid-run.
// Delivered after the current tool execution; remaining tools are skipped.
func (a *Agent) Steer(msg AgentMessage) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.steeringQ = append(a.steeringQ, msg)
}

// FollowUp queues a message to be processed after the agent finishes.
func (a *Agent) FollowUp(msg AgentMessage) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.followUpQ = append(a.followUpQ, msg)
}

// Abort cancels the current execution and emits an abort marker message
// so the LLM knows the user interrupted.
func (a *Agent) Abort() {
	a.wantAbortMarker.Store(true)
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.cancel != nil {
		a.cancel()
	}
}

// AbortSilent cancels the current execution without emitting an abort marker.
// Use for programmatic cancellation (e.g. plan mode transitions) where the
// cancellation is not a user interruption.
func (a *Agent) AbortSilent() {
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.cancel != nil {
		a.cancel()
	}
}

// WaitForIdle blocks until the agent finishes the current run.
func (a *Agent) WaitForIdle() {
	a.mu.Lock()
	done := a.done
	a.mu.Unlock()
	if done != nil {
		<-done
	}
}

// State returns a snapshot of the agent's current state.
func (a *Agent) State() AgentState {
	a.mu.Lock()
	defer a.mu.Unlock()
	pending := make(map[string]struct{}, len(a.pendingToolCalls))
	for k, v := range a.pendingToolCalls {
		pending[k] = v
	}
	sp := a.systemPrompt
	if len(a.systemBlocks) > 0 && sp == "" {
		var sb strings.Builder
		for i, b := range a.systemBlocks {
			if i > 0 {
				sb.WriteString("\n\n")
			}
			sb.WriteString(b.Text)
		}
		sp = sb.String()
	}
	return AgentState{
		SystemPrompt:     sp,
		Messages:         copyMessages(a.messages),
		Tools:            a.tools,
		IsRunning:        a.isRunning,
		StreamMessage:    a.streamMessage,
		PendingToolCalls: pending,
		TotalUsage:       a.totalUsage,
		Error:            a.lastError,
	}
}

// Messages returns the current message history.
func (a *Agent) Messages() []AgentMessage {
	a.mu.Lock()
	defer a.mu.Unlock()
	return copyMessages(a.messages)
}

// SetMessages replaces the message history (e.g. to restore a previous conversation).
// The agent must not be running.
func (a *Agent) SetMessages(msgs []AgentMessage) error {
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.isRunning {
		return fmt.Errorf("cannot set messages while agent is running")
	}
	a.messages = copyMessages(msgs)
	a.syncContextManagerLocked()
	return nil
}

// ExportMessages returns concrete Messages for serialization.
func (a *Agent) ExportMessages() []Message {
	a.mu.Lock()
	defer a.mu.Unlock()
	return CollectMessages(a.messages)
}

// ImportMessages replaces message history from deserialized Messages.
func (a *Agent) ImportMessages(msgs []Message) error {
	return a.SetMessages(ToAgentMessages(msgs))
}

// ClearMessages resets the message history.
func (a *Agent) ClearMessages() {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.messages = nil
	a.syncContextManagerLocked()
}

// startPromptRunLocked starts a prompt-based run. Caller must hold a.mu.
func (a *Agent) startPromptRunLocked(msgs []AgentMessage) {
	a.isRunning = true
	a.lastError = ""

	ctx, cancel := context.WithCancel(context.Background())
	a.cancel = cancel
	a.done = make(chan struct{})

	agentCtx := AgentContext{
		SystemPrompt: a.systemPrompt,
		SystemBlocks: a.systemBlocks,
		Messages:     copyMessages(a.messages),
		Tools:        a.tools,
	}
	config := a.buildConfig()
	a.mu.Unlock()

	go a.consumeLoop(AgentLoop(ctx, msgs, agentCtx, config))
}

// startContinueRunLocked starts a continue run from the current context. Caller must hold a.mu.
func (a *Agent) startContinueRunLocked() {
	a.isRunning = true
	a.lastError = ""

	ctx, cancel := context.WithCancel(context.Background())
	a.cancel = cancel
	a.done = make(chan struct{})

	agentCtx := AgentContext{
		SystemPrompt: a.systemPrompt,
		SystemBlocks: a.systemBlocks,
		Messages:     copyMessages(a.messages),
		Tools:        a.tools,
	}
	config := a.buildConfig()
	a.mu.Unlock()

	go a.consumeLoop(AgentLoopContinue(ctx, agentCtx, config))
}

// ContextUsage returns an estimate of the current context window occupancy.
// Returns nil if contextWindow or contextEstimateFn is not configured.
func (a *Agent) ContextUsage() *ContextUsage {
	a.mu.Lock()
	defer a.mu.Unlock()

	if a.contextManager != nil {
		if usage := a.contextManager.Usage(); usage != nil {
			cp := *usage
			return &cp
		}
	}

	if a.contextWindow <= 0 || a.contextEstimateFn == nil {
		return nil
	}

	tokens, usageTokens, trailingTokens := a.contextEstimateFn(a.messages)
	pct := float64(tokens) / float64(a.contextWindow) * 100

	return &ContextUsage{
		Tokens:         tokens,
		ContextWindow:  a.contextWindow,
		Percent:        pct,
		UsageTokens:    usageTokens,
		TrailingTokens: trailingTokens,
	}
}

// BaselineContextUsage returns the current runtime baseline occupancy.
// Unlike ContextUsage, this never reports a transient projected view.
func (a *Agent) BaselineContextUsage() *ContextUsage {
	a.mu.Lock()
	defer a.mu.Unlock()

	if a.contextManager != nil {
		if snap := a.contextManager.Snapshot(); snap != nil && snap.BaselineUsage != nil {
			cp := *snap.BaselineUsage
			return &cp
		}
	}

	if a.contextWindow <= 0 || a.contextEstimateFn == nil {
		return nil
	}

	tokens, usageTokens, trailingTokens := a.contextEstimateFn(a.messages)
	pct := float64(tokens) / float64(a.contextWindow) * 100

	return &ContextUsage{
		Tokens:         tokens,
		ContextWindow:  a.contextWindow,
		Percent:        pct,
		UsageTokens:    usageTokens,
		TrailingTokens: trailingTokens,
	}
}

// ContextSnapshot returns the latest context-manager snapshot for observability.
// Returns nil when no ContextManager is configured or no snapshot is available.
func (a *Agent) ContextSnapshot() *ContextSnapshot {
	a.mu.Lock()
	defer a.mu.Unlock()

	if a.contextManager == nil {
		return nil
	}
	snap := a.contextManager.Snapshot()
	if snap == nil {
		return nil
	}

	out := *snap
	if snap.BaselineUsage != nil {
		usage := *snap.BaselineUsage
		out.BaselineUsage = &usage
	}
	if snap.Usage != nil {
		usage := *snap.Usage
		out.Usage = &usage
	}
	return &out
}

// TotalUsage returns the cumulative token usage across all turns.
func (a *Agent) TotalUsage() Usage {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.totalUsage
}

// TaskRuntime returns the shared TaskRuntime, or nil if not configured.
func (a *Agent) TaskRuntime() *TaskRuntime {
	return a.taskRuntime
}

// Tasks returns snapshots of all background tasks.
// Returns nil if no TaskRuntime is configured.
func (a *Agent) Tasks() []BackgroundTaskEntry {
	if a.taskRuntime == nil {
		return nil
	}
	return a.taskRuntime.List()
}

// StopTask cancels a running background task by ID.
func (a *Agent) StopTask(id string) bool {
	if a.taskRuntime == nil {
		return false
	}
	return a.taskRuntime.Stop(id)
}

// StopAllTasks cancels all running background tasks.
func (a *Agent) StopAllTasks() int {
	if a.taskRuntime == nil {
		return 0
	}
	return a.taskRuntime.StopAll()
}

// SetModel changes the LLM provider. Takes effect on the next turn.
func (a *Agent) SetModel(m ChatModel) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.model = m
}

// SetContextWindow updates the context window size (in tokens).
func (a *Agent) SetContextWindow(n int) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.contextWindow = n
}

// SetSystemPrompt changes the system prompt (single-string mode).
// Clears any multi-block system prompt set via SetSystemBlocks.
func (a *Agent) SetSystemPrompt(s string) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.systemPrompt = s
	a.systemBlocks = nil
}

// SetSystemBlocks sets a multi-block system prompt with per-block cache control.
// Takes precedence over SetSystemPrompt. Clears the single-string prompt.
func (a *Agent) SetSystemBlocks(blocks []SystemBlock) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.systemBlocks = blocks
	a.systemPrompt = ""
}

// BuildLLMMessages constructs the message list exactly as the agent loop would
// for an LLM call: system blocks/prompt → converted conversation messages.
// This enables external callers (e.g., prompt suggestion) to make background
// LLM calls that share the same message prefix for prompt cache hits.
// Strict message-sequence mode is honored here just like in the main loop.
func (a *Agent) BuildLLMMessages() ([]Message, error) {
	a.mu.Lock()
	msgs := copyMessages(a.messages)
	blocks := a.systemBlocks
	sp := a.systemPrompt
	mgr := a.contextManager
	convertFn := a.convertToLLM
	strict := a.strictMessageSequence
	a.mu.Unlock()

	if mgr != nil {
		proj, err := mgr.Project(context.Background(), msgs)
		if err != nil {
			return nil, err
		}
		if proj.Messages != nil {
			msgs = proj.Messages
		}
	}

	if convertFn == nil {
		convertFn = DefaultConvertToLLM
	}
	llmMessages := convertFn(msgs)
	if strict {
		if err := AssertMessageSequence(llmMessages); err != nil {
			return nil, err
		}
	} else {
		llmMessages = RepairMessageSequence(llmMessages)
	}

	if len(blocks) > 0 {
		sysMsgs := make([]Message, len(blocks))
		for i, b := range blocks {
			sysMsgs[i] = SystemMsg(b.Text)
			if b.CacheControl != "" {
				sysMsgs[i].Metadata = map[string]any{"cache_control": b.CacheControl}
			}
		}
		llmMessages = append(sysMsgs, llmMessages...)
	} else if sp != "" {
		llmMessages = append([]Message{SystemMsg(sp)}, llmMessages...)
	}
	return llmMessages, nil
}

// SetTools replaces the tool set. Takes effect on the next turn.
func (a *Agent) SetTools(tools ...Tool) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.tools = tools
}

// SetThinkingLevel changes the reasoning depth. Takes effect on the next turn.
func (a *Agent) SetThinkingLevel(level ThinkingLevel) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.thinkingLevel = level
}

// ClearSteeringQueue removes all queued steering messages.
func (a *Agent) ClearSteeringQueue() {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.steeringQ = nil
}

// ClearFollowUpQueue removes all queued follow-up messages.
func (a *Agent) ClearFollowUpQueue() {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.followUpQ = nil
}

// ClearAllQueues removes all queued steering and follow-up messages.
func (a *Agent) ClearAllQueues() {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.steeringQ = nil
	a.followUpQ = nil
}

// HasQueuedMessages reports whether any steering or follow-up messages are queued.
func (a *Agent) HasQueuedMessages() bool {
	a.mu.Lock()
	defer a.mu.Unlock()
	return len(a.steeringQ) > 0 || len(a.followUpQ) > 0
}

// Reset clears all state and queues. If the agent is running, it cancels and waits first.
func (a *Agent) Reset() {
	a.mu.Lock()
	cancel := a.cancel
	done := a.done
	a.mu.Unlock()

	if cancel != nil {
		cancel()
	}
	if done != nil {
		<-done
	}

	a.mu.Lock()
	defer a.mu.Unlock()
	a.messages = nil
	a.steeringQ = nil
	a.followUpQ = nil
	a.isRunning = false
	a.lastError = ""
	a.streamMessage = nil
	a.pendingToolCalls = make(map[string]struct{})
	a.totalUsage = Usage{}
	a.done = nil
	a.cancel = nil
	a.syncContextManagerLocked()
}

// buildConfig constructs a LoopConfig from the agent's settings. Must be called with lock held.
func (a *Agent) buildConfig() LoopConfig {
	skipInitialSteering := a.skipNextInitialSteeringPoll
	a.skipNextInitialSteeringPoll = false

	return LoopConfig{
		Model:                 a.model,
		StreamFn:              a.streamFn,
		MaxTurns:              a.maxTurns,
		MaxRetries:            a.maxRetries,
		MaxToolErrors:         a.maxToolErrors,
		StrictMessageSequence: a.strictMessageSequence,
		ThinkingLevel:         a.thinkingLevel,
		ContextManager:        a.contextManager,
		TransformContext:      a.transformContext,
		ConvertToLLM:          a.convertToLLM,
		CommitContext: func(msgs []AgentMessage, usage *ContextUsage) error {
			a.mu.Lock()
			defer a.mu.Unlock()
			a.messages = copyMessages(msgs)
			a.syncContextManagerLocked()
			return nil
		},
		PermissionEngine: a.permissionEngine,
		GetApiKey:        a.getApiKey,
		ThinkingBudgets:  a.thinkingBudgets,
		SessionID:        a.sessionID,
		GetSteeringMessages: func() []AgentMessage {
			a.mu.Lock()
			defer a.mu.Unlock()
			if skipInitialSteering {
				skipInitialSteering = false
				return nil
			}
			return dequeue(&a.steeringQ, a.steeringMode)
		},
		GetFollowUpMessages: func() []AgentMessage {
			a.mu.Lock()
			defer a.mu.Unlock()
			return dequeue(&a.followUpQ, a.followUpMode)
		},
		MaxRetryDelay:         a.maxRetryDelay,
		Middlewares:           a.middlewares,
		MaxToolConcurrency:    a.maxToolConcurrency,
		ShouldEmitAbortMarker: a.wantAbortMarker.Load,
		ToolChoice:            a.toolChoice,
	}
}

// consumeLoop reads events from the loop channel and updates internal state.
// handles partial message residue, and constructs error fallback messages.
func (a *Agent) consumeLoop(events <-chan Event) {
	// Capture our done channel before any new Prompt() can replace a.done.
	a.mu.Lock()
	myDone := a.done
	a.mu.Unlock()

	var partial AgentMessage // tracks partial message during streaming

	defer func() {
		a.mu.Lock()

		// Handle partial message residue
		// If stream ended with an unfinished partial, append it to messages
		if partial != nil {
			if msg, ok := partial.(Message); ok {
				if !msg.IsEmpty() {
					a.messages = append(a.messages, partial)
					a.syncContextManagerLocked()
				}
			}
		}

		// Full cleanup
		a.isRunning = false
		a.streamMessage = nil
		a.pendingToolCalls = make(map[string]struct{})
		a.cancel = nil
		a.wantAbortMarker.Store(false)
		a.mu.Unlock()
		close(myDone)
	}()

	for ev := range events {
		a.mu.Lock()
		switch ev.Type {
		// Message lifecycle
		case EventMessageStart:
			partial = ev.Message
			a.streamMessage = ev.Message

		case EventMessageUpdate:
			partial = ev.Message
			a.streamMessage = ev.Message

		case EventMessageEnd:
			partial = nil
			a.streamMessage = nil
			if ev.Message != nil {
				a.messages = append(a.messages, ev.Message)
				a.syncContextManagerLocked()
				// Accumulate usage from assistant messages
				if msg, ok := ev.Message.(Message); ok && msg.Usage != nil {
					a.totalUsage.Add(msg.Usage)
				}
			}

		// Tool execution lifecycle
		case EventToolExecStart:
			if ev.ToolID != "" {
				a.pendingToolCalls[ev.ToolID] = struct{}{}
			}

		case EventToolExecEnd:
			delete(a.pendingToolCalls, ev.ToolID)

		// Turn end
		case EventTurnEnd:
			if msg, ok := ev.Message.(Message); ok {
				if errStr, _ := msg.Metadata["error_message"].(string); errStr != "" {
					a.lastError = errStr
				}
			}

		// Error — construct fallback assistant message (skip for intentional abort)
		case EventError:
			partial = nil // discard incomplete streaming message to prevent defer from appending it
			if ev.Err != nil && !errors.Is(ev.Err, context.Canceled) {
				a.lastError = ev.Err.Error()
				errMsg := Message{
					Role:       RoleAssistant,
					StopReason: StopReasonError,
					Metadata: map[string]any{
						"error_message": ev.Err.Error(),
					},
					Timestamp: time.Now(),
				}
				a.messages = append(a.messages, errMsg)
				a.syncContextManagerLocked()
			}

		case EventAgentEnd:
			a.isRunning = false
			a.streamMessage = nil
			a.pendingToolCalls = make(map[string]struct{})
		}

		// Copy listeners to avoid holding lock during callback
		listeners := make([]func(Event), len(a.listeners))
		copy(listeners, a.listeners)
		a.mu.Unlock()

		for _, fn := range listeners {
			if fn != nil {
				fn(ev)
			}
		}
	}
}

func (a *Agent) syncContextManagerLocked() {
	if a.contextManager == nil {
		return
	}
	a.contextManager.Sync(copyMessages(a.messages))
}

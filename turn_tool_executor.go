package agentcore

import (
	"context"
	"sync"
)

type turnToolEntry struct {
	call      ToolCall
	safe      bool
	failCount int

	started bool
	done    bool
	result  ToolResult
}

// turnToolExecutor is the single scheduling primitive for one assistant turn.
// It supports both "add all calls then wait" and "stream tool calls in as they
// arrive" without changing the underlying execution semantics.
type turnToolExecutor struct {
	ctx    context.Context
	cancel context.CancelFunc
	tools  []Tool
	config LoopConfig
	ch     chan<- Event

	mu             sync.Mutex
	cond           *sync.Cond
	entries        []*turnToolEntry
	toolErrors     map[string]int
	running        int
	runningUnsafe  bool
	stopStarting   bool
	steering       []AgentMessage
	maxConcurrency int
}

func newTurnToolExecutor(ctx context.Context, tools []Tool, config LoopConfig, toolErrors map[string]int, ch chan<- Event) *turnToolExecutor {
	execCtx, cancel := context.WithCancel(ctx)
	maxConc := config.MaxToolConcurrency
	if maxConc <= 1 {
		maxConc = 1
	}
	exec := &turnToolExecutor{
		ctx:            execCtx,
		cancel:         cancel,
		tools:          tools,
		config:         config,
		ch:             ch,
		toolErrors:     toolErrors,
		maxConcurrency: maxConc,
	}
	exec.cond = sync.NewCond(&exec.mu)
	return exec
}

func (e *turnToolExecutor) Add(call ToolCall) {
	tool := findTool(e.tools, call.Name)
	entry := &turnToolEntry{
		call: call,
		safe: tool != nil && isToolConcurrencySafe(tool, call.Args),
	}

	e.mu.Lock()
	entry.failCount = e.toolErrors[call.Name]
	e.entries = append(e.entries, entry)
	e.processQueueLocked()
	e.mu.Unlock()
}

func (e *turnToolExecutor) HasCalls() bool {
	e.mu.Lock()
	defer e.mu.Unlock()
	return len(e.entries) > 0
}

func (e *turnToolExecutor) Wait() ([]ToolResult, []AgentMessage) {
	defer e.cancel()

	e.mu.Lock()
	for e.running > 0 {
		e.cond.Wait()
	}

	skipMessage := ""
	if e.ctx.Err() != nil {
		skipMessage = "Tool execution cancelled."
	}
	results := make([]ToolResult, 0, len(e.entries))
	for _, entry := range e.entries {
		if entry.done {
			results = append(results, entry.result)
			continue
		}
		if skipMessage != "" {
			results = append(results, skipToolCallWithMessage(entry.call, e.tools, e.ch, skipMessage))
		} else {
			results = append(results, skipToolCall(entry.call, e.tools, e.ch))
		}
	}

	steering := append([]AgentMessage(nil), e.steering...)
	e.mu.Unlock()
	return results, steering
}

func (e *turnToolExecutor) Abort() {
	e.mu.Lock()
	e.stopStarting = true
	e.cancel()
	e.cond.Broadcast()
	e.mu.Unlock()
}

func (e *turnToolExecutor) AbortAndWait() ([]ToolResult, []AgentMessage) {
	e.Abort()
	return e.Wait()
}

func (e *turnToolExecutor) processQueueLocked() {
	if e.stopStarting || e.ctx.Err() != nil {
		return
	}

	for _, entry := range e.entries {
		if entry.started || entry.done {
			continue
		}
		if !e.canStartLocked(entry.safe) {
			break
		}
		e.startLocked(entry)
		if !entry.safe {
			break
		}
	}
}

func (e *turnToolExecutor) canStartLocked(safe bool) bool {
	if e.running == 0 {
		return true
	}
	if !safe || e.runningUnsafe {
		return false
	}
	return e.running < e.maxConcurrency
}

func (e *turnToolExecutor) startLocked(entry *turnToolEntry) {
	entry.started = true
	e.running++
	if !entry.safe {
		e.runningUnsafe = true
	}

	go func(ent *turnToolEntry) {
		result := executeSingleToolCall(e.ctx, e.tools, ent.call, e.config, ent.failCount, e.ch)

		e.mu.Lock()
		defer e.mu.Unlock()

		ent.done = true
		ent.result = result
		e.running--
		if !ent.safe {
			e.runningUnsafe = false
		}

		if result.ToolName != "" {
			if result.IsError {
				e.toolErrors[result.ToolName]++
			} else {
				delete(e.toolErrors, result.ToolName)
			}
		}

		if !e.stopStarting && e.config.GetSteeringMessages != nil {
			if steering := e.config.GetSteeringMessages(); len(steering) > 0 {
				e.stopStarting = true
				e.steering = steering
			}
		}

		e.processQueueLocked()
		e.cond.Broadcast()
	}(entry)
}

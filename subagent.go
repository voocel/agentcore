package agentcore

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"strings"
	"sync"
	"time"

	"github.com/voocel/agentcore/schema"
)

const (
	maxParallelTasks = 8
	maxConcurrency   = 4
)

// SubAgentConfig defines a sub-agent's identity and capabilities.
type SubAgentConfig struct {
	Name        string
	Description string
	// Model is resolved when each sub-agent run starts. Wrappers that swap
	// the underlying model at runtime are supported and take effect on the
	// next sub-agent run.
	Model        ChatModel
	SystemPrompt string
	Tools        []Tool
	StreamFn     StreamFn
	MaxTurns     int

	// ToolChoice sets the default tool_choice for every LLM call in this
	// sub-agent's loop. nil uses the provider default ("auto").
	ToolChoice any

	// StopAfterTools lists tool names that trigger early loop exit after
	// successful execution. Useful with ToolChoice "required" to let a
	// terminal tool (e.g. "commit_chapter") end the loop cleanly.
	StopAfterTools []string

	// OnMessage, if non-nil, is called after each message is appended to
	// context. The agentName and task are provided for session routing.
	OnMessage func(agentName, task string, msg AgentMessage)

	// Optional context lifecycle hooks for long-running sub-agents.
	ContextManager        ContextManager
	ContextManagerFactory func(model ChatModel) ContextManager
	TransformContext      func(ctx context.Context, msgs []AgentMessage) ([]AgentMessage, error)
	ConvertToLLM          func(msgs []AgentMessage) []Message

	// StopGuardFactory, if non-nil, creates a fresh StopGuard for each run.
	// The factory receives the agent name and task, enabling run-scoped state
	// (e.g. baseline progress captured at dispatch time).
	StopGuardFactory func(agentName, task string) StopGuard
}

// subagentParams is the JSON schema input for the subagent tool.
// Four mutually exclusive modes:
//   - Single: Agent + Task
//   - Parallel: Tasks array
//   - Chain: Chain array with {previous} placeholder
//   - Background: Single + Background=true (returns immediately, notifies on completion)
type subagentParams struct {
	Agent       string          `json:"agent,omitempty"`
	Task        string          `json:"task,omitempty"`
	Tasks       []subagentTask  `json:"tasks,omitempty"`
	Chain       []subagentChain `json:"chain,omitempty"`
	Background  bool            `json:"background,omitempty"`
	Description string          `json:"description,omitempty"`
	Model       string          `json:"model,omitempty"`
}

type subagentTask struct {
	Agent string `json:"agent"`
	Task  string `json:"task"`
}

type subagentChain struct {
	Agent string `json:"agent"`
	Task  string `json:"task"`
}

// subagentResult captures one sub-agent's execution outcome.
type subagentResult struct {
	Agent   string         `json:"agent"`
	Task    string         `json:"task"`
	Output  string         `json:"output"`
	IsError bool           `json:"is_error,omitempty"`
	Step    int            `json:"step,omitempty"`
	Usage   *subagentUsage `json:"usage,omitempty"`
}

// subagentUsage tracks token consumption and cost for a sub-agent run.
type subagentUsage struct {
	Input      int     `json:"input"`
	Output     int     `json:"output"`
	CacheRead  int     `json:"cache_read"`
	CacheWrite int     `json:"cache_write"`
	Cost       float64 `json:"cost"`
	Turns      int     `json:"turns"`
	Tools      int     `json:"tools"`
}

// SubAgentTool implements the Tool interface.
// The main agent calls this tool to delegate tasks to specialized sub-agents
// with isolated contexts.
type SubAgentTool struct {
	agents          map[string]SubAgentConfig
	notifyFn        func(AgentMessage)                                             // called when a background task completes
	createModel     func(name string) (ChatModel, error)                           // resolves model name to ChatModel at runtime
	bgOutputFactory func(taskID, agentName string) (io.WriteCloser, string, error) // creates output writer for background tasks
	taskRT          *TaskRuntime                                                   // shared background task registry
}

// NewSubAgentTool creates a subagent tool from a set of agent configs.
func NewSubAgentTool(agents ...SubAgentConfig) *SubAgentTool {
	m := make(map[string]SubAgentConfig, len(agents))
	for _, a := range agents {
		m[a.Name] = a
	}
	return &SubAgentTool{
		agents: m,
	}
}

// SetTaskRuntime sets the shared task runtime for background task registration.
// When set, background tasks are registered here instead of managed internally.
func (t *SubAgentTool) SetTaskRuntime(rt *TaskRuntime) {
	t.taskRT = rt
}

// SetNotifyFn sets the callback invoked when a background task completes.
// Typically bound to Agent.FollowUp so the main agent receives the result
// as a follow-up message.
func (t *SubAgentTool) SetNotifyFn(fn func(AgentMessage)) {
	t.notifyFn = fn
}

// SetCreateModel sets the factory for resolving model names (e.g. "haiku", "gpt-4o-mini")
// to ChatModel instances at runtime. Enables LLM to override the default model per call.
func (t *SubAgentTool) SetCreateModel(fn func(name string) (ChatModel, error)) {
	t.createModel = fn
}

// SetBgOutputFactory sets the factory that creates output writers for background tasks.
// The factory receives the task ID and agent name, returns a writer, file path, and error.
// If not set, background output is not persisted.
func (t *SubAgentTool) SetBgOutputFactory(fn func(taskID, agentName string) (io.WriteCloser, string, error)) {
	t.bgOutputFactory = fn
}

func (t *SubAgentTool) Name() string  { return "subagent" }
func (t *SubAgentTool) Label() string { return "Delegate to SubAgent" }

func (t *SubAgentTool) Description() string {
	names := make([]string, 0, len(t.agents))
	for _, a := range t.agents {
		names = append(names, fmt.Sprintf("%s (%s)", a.Name, a.Description))
	}
	return fmt.Sprintf(
		"Delegate tasks to specialized subagents with isolated context. "+
			"Modes: single (agent+task), parallel (tasks array), chain (sequential with {previous} placeholder), "+
			"background (agent+task+background=true, returns immediately and notifies on completion). "+
			"Available agents: %s",
		strings.Join(names, ", "),
	)
}

func (t *SubAgentTool) Schema() map[string]any {
	agentNames := make([]string, 0, len(t.agents))
	for name := range t.agents {
		agentNames = append(agentNames, name)
	}
	taskItem := schema.Object(
		schema.Property("agent", schema.Enum("Agent name", agentNames...)).Required(),
		schema.Property("task", schema.String("Task description")).Required(),
	)
	return schema.Object(
		schema.Property("agent", schema.Enum("Name of the agent to invoke (single/background mode)", agentNames...)),
		schema.Property("task", schema.String("Task to delegate (single/background mode)")),
		schema.Property("tasks", schema.Array("Array of {agent, task} for parallel execution", taskItem)),
		schema.Property("chain", schema.Array("Array of {agent, task} for sequential execution. Use {previous} in task to reference prior output.", taskItem)),
		schema.Property("background", schema.Bool("Set true to run in background. Returns immediately; a notification is sent when the task completes.")),
		schema.Property("description", schema.String("Short description of the background task (shown in notifications).")),
		schema.Property("model", schema.String("Optional model override for this call (e.g. model ID or alias). If not set, uses the agent's default model.")),
	)
}

func (t *SubAgentTool) Execute(ctx context.Context, args json.RawMessage) (json.RawMessage, error) {
	var params subagentParams
	if err := json.Unmarshal(args, &params); err != nil {
		return nil, fmt.Errorf("invalid subagent params: %w", err)
	}

	// Resolve model override once (applies to all subtasks in this call).
	var modelOverride ChatModel
	if params.Model != "" && t.createModel != nil {
		m, err := t.createModel(params.Model)
		if err != nil {
			return json.Marshal(map[string]any{"error": fmt.Sprintf("invalid model %q: %v", params.Model, err)})
		}
		modelOverride = m
	}

	hasChain := len(params.Chain) > 0
	hasParallel := len(params.Tasks) > 0
	hasSingle := params.Agent != "" && params.Task != ""

	// Background mode: single task running in a detached goroutine.
	// If TaskRuntime is not wired, silently degrade to synchronous execution
	// so callers don't hit a dead-end error when background support is unavailable.
	if params.Background {
		if !hasSingle {
			return json.Marshal("background mode requires agent + task")
		}
		if t.taskRT != nil {
			return t.executeBackground(params.Agent, params.Task, params.Description, modelOverride)
		}
		// Fall through to synchronous single-task execution.
	}

	modeCount := boolToInt(hasChain) + boolToInt(hasParallel) + boolToInt(hasSingle)
	if modeCount != 1 {
		return json.Marshal("Invalid parameters: provide exactly one mode (agent+task, tasks, or chain)")
	}

	switch {
	case hasChain:
		return t.executeChain(ctx, params.Chain, modelOverride)
	case hasParallel:
		return t.executeParallel(ctx, params.Tasks, modelOverride)
	default:
		return t.executeSingle(ctx, params.Agent, params.Task, modelOverride)
	}
}

// executeBackground launches a sub-agent in a detached goroutine and returns immediately.
// When the sub-agent finishes, a notification is sent via notifyFn (typically Agent.FollowUp).
func (t *SubAgentTool) executeBackground(agentName, task, description string, modelOverride ChatModel) (json.RawMessage, error) {
	if _, ok := t.agents[agentName]; !ok {
		available := make([]string, 0, len(t.agents))
		for name := range t.agents {
			available = append(available, name)
		}
		return json.Marshal(map[string]any{
			"error": fmt.Sprintf("unknown agent %q, available: %s", agentName, strings.Join(available, ", ")),
		})
	}

	rt := t.taskRT
	if rt == nil {
		return json.Marshal(map[string]any{
			"error": "background mode requires a TaskRuntime; call SetTaskRuntime before use",
		})
	}
	taskID := rt.NextID("bg")
	if description == "" {
		description = truncate(task, 80)
	}

	bgCtx, cancel := context.WithCancel(context.Background())

	entry := &BackgroundTaskEntry{
		ID:          taskID,
		Type:        TaskTypeSubAgent,
		Agent:       agentName,
		Prompt:      task,
		Description: description,
		Status:      TaskRunning,
		StartedAt:   time.Now(),
	}
	entry.SetCancel(cancel)
	rt.Register(entry)

	go func() {
		defer cancel()

		var outFile io.WriteCloser
		if t.bgOutputFactory != nil {
			w, path, ferr := t.bgOutputFactory(taskID, agentName)
			if ferr == nil {
				outFile = w
				rt.Update(taskID, func(e *BackgroundTaskEntry) { e.OutputFile = path })
			}
		}

		_, usage, err := t.runAgent(bgCtx, agentName, task, modelOverride, &bgRunOpts{taskID: taskID, rt: rt, outFile: outFile})
		if outFile != nil {
			outFile.Close()
		}

		rt.Update(taskID, func(e *BackgroundTaskEntry) {
			e.EndedAt = time.Now()
			if err != nil {
				e.Status = TaskFailed
				e.Error = err.Error()
			} else {
				e.Status = TaskCompleted
			}
			if usage != nil {
				e.TokensIn = usage.Input
				e.TokensOut = usage.Output
			}
		})
		t.notify(taskID)
	}()

	return json.Marshal(map[string]any{
		"task_id":     taskID,
		"description": description,
		"status":      "running",
		"message":     fmt.Sprintf("Background task %s started with agent %q. You will receive a notification when it completes.", taskID, agentName),
	})
}

// notify sends background task results via notifyFn as a follow-up message.
func (t *SubAgentTool) notify(taskID string) {
	if t.notifyFn == nil || t.taskRT == nil {
		return
	}
	entry := t.taskRT.Get(taskID)
	if entry == nil {
		return
	}
	t.notifyFn(NotificationFromEntry(entry).ToAgentMessage())
}

// executeSingle runs one sub-agent with an isolated context.
func (t *SubAgentTool) executeSingle(ctx context.Context, agentName, task string, modelOverride ChatModel) (json.RawMessage, error) {
	output, usage, err := t.runAgent(ctx, agentName, task, modelOverride, nil)
	if err != nil {
		return json.Marshal(map[string]any{
			"error": fmt.Sprintf("Agent %q failed: %v", agentName, err),
			"usage": usage,
		})
	}
	return json.Marshal(map[string]any{
		"output": output,
		"usage":  usage,
	})
}

// executeChain runs sub-agents sequentially, passing each output to the next via {previous}.
func (t *SubAgentTool) executeChain(ctx context.Context, chain []subagentChain, modelOverride ChatModel) (json.RawMessage, error) {
	var previous string
	results := make([]subagentResult, 0, len(chain))

	for i, step := range chain {
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}

		task := strings.ReplaceAll(step.Task, "{previous}", previous)
		output, usage, err := t.runAgent(ctx, step.Agent, task, modelOverride, nil)

		result := subagentResult{
			Agent: step.Agent,
			Task:  task,
			Step:  i + 1,
			Usage: usage,
		}

		if err != nil {
			result.Output = err.Error()
			result.IsError = true
			results = append(results, result)
			return json.Marshal(map[string]any{
				"error":   fmt.Sprintf("Chain stopped at step %d (%s): %v", i+1, step.Agent, err),
				"results": results,
			})
		}

		result.Output = output
		results = append(results, result)
		previous = output
	}

	return json.Marshal(map[string]any{
		"output":  previous,
		"results": results,
	})
}

// executeParallel runs multiple sub-agents concurrently with bounded concurrency.
func (t *SubAgentTool) executeParallel(ctx context.Context, tasks []subagentTask, modelOverride ChatModel) (json.RawMessage, error) {
	if len(tasks) > maxParallelTasks {
		return json.Marshal(fmt.Sprintf("Too many parallel tasks (%d). Max is %d.", len(tasks), maxParallelTasks))
	}

	results := make([]subagentResult, len(tasks))
	var wg sync.WaitGroup
	sem := make(chan struct{}, maxConcurrency)

	for i, task := range tasks {
		wg.Add(1)
		go func(idx int, st subagentTask) {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()

			output, usage, err := t.runAgent(ctx, st.Agent, st.Task, modelOverride, nil)
			result := subagentResult{
				Agent: st.Agent,
				Task:  st.Task,
				Usage: usage,
			}
			if err != nil {
				result.Output = err.Error()
				result.IsError = true
			} else {
				result.Output = output
			}
			results[idx] = result
		}(i, task)
	}

	wg.Wait()

	successCount := 0
	for _, r := range results {
		if !r.IsError {
			successCount++
		}
	}

	return json.Marshal(map[string]any{
		"summary": fmt.Sprintf("%d/%d succeeded", successCount, len(results)),
		"results": results,
	})
}

// bgRunOpts configures background-specific behavior for runAgent.
// When nil, runAgent runs in foreground mode (reports progress to parent context).
type bgRunOpts struct {
	taskID  string       // task ID in the TaskRuntime
	rt      *TaskRuntime // shared runtime for updates
	outFile io.Writer    // output stream for session persistence (optional)
}

// runAgent executes an isolated agent loop for the given agent config and task.
// Includes panic recovery to prevent a subagent crash from taking down the parent.
func (t *SubAgentTool) runAgent(ctx context.Context, agentName, task string, modelOverride ChatModel, bg *bgRunOpts) (output string, usage *subagentUsage, err error) {
	cfg, ok := t.agents[agentName]
	if !ok {
		available := make([]string, 0, len(t.agents))
		for name := range t.agents {
			available = append(available, name)
		}
		return "", nil, fmt.Errorf("unknown agent %q, available: %s", agentName, strings.Join(available, ", "))
	}

	// Panic recovery — isolated subagent should never crash the parent
	defer func() {
		if r := recover(); r != nil {
			err = fmt.Errorf("subagent %q panicked: %v", agentName, r)
		}
	}()

	agentCtx := AgentContext{
		SystemPrompt: cfg.SystemPrompt,
		Tools:        cfg.Tools,
	}

	runModel := cfg.Model
	if modelOverride != nil {
		runModel = modelOverride
	}
	contextManager := cfg.ContextManager
	if cfg.ContextManagerFactory != nil {
		contextManager = cfg.ContextManagerFactory(runModel)
	}

	loopCfg := LoopConfig{
		Model:            runModel,
		StreamFn:         cfg.StreamFn,
		MaxTurns:         cfg.MaxTurns,
		ToolChoice:       cfg.ToolChoice,
		ContextManager:   contextManager,
		TransformContext: cfg.TransformContext,
		ConvertToLLM:     cfg.ConvertToLLM,
	}
	if len(cfg.StopAfterTools) > 0 {
		stopSet := make(map[string]struct{}, len(cfg.StopAfterTools))
		for _, name := range cfg.StopAfterTools {
			stopSet[name] = struct{}{}
		}
		loopCfg.StopAfterTool = func(toolName string) bool {
			_, ok := stopSet[toolName]
			return ok
		}
	}
	if cfg.StopGuardFactory != nil {
		loopCfg.StopGuard = cfg.StopGuardFactory(agentName, task)
	}
	if cfg.OnMessage != nil {
		name, t := agentName, task
		loopCfg.OnMessage = func(msg AgentMessage) { cfg.OnMessage(name, t, msg) }
	}
	if loopCfg.MaxTurns <= 0 {
		loopCfg.MaxTurns = defaultMaxTurns
	}

	events := AgentLoop(ctx, []AgentMessage{UserMsg(task)}, agentCtx, loopCfg)

	var lastAssistantContent string
	var terminalToolResult string // result from StopAfterTool trigger
	var lastErr error
	su := &subagentUsage{}

	for ev := range events {
		switch ev.Type {
		case EventToolExecStart:
			su.Tools++
			if bg != nil {
				bg.rt.Update(bg.taskID, func(e *BackgroundTaskEntry) { e.ToolCount++ })
				if bg.outFile != nil {
					label := ev.Tool
					if len(ev.Args) > 0 {
						label += "(" + truncate(string(ev.Args), 60) + ")"
					}
					fmt.Fprintf(bg.outFile, "[tool] %s\n", label)
				}
			} else {
				ReportToolProgress(ctx, ProgressPayload{
					Kind:    ProgressToolStart,
					Agent:   agentName,
					Tool:    ev.Tool,
					Summary: ev.Tool,
					Args:    ev.Args,
				})
			}
		case EventMessageUpdate:
			if bg == nil {
				if ev.DeltaKind == DeltaThinking {
					// Thinking deltas only go through ProgressThinking (cumulative).
					// Do NOT also send as ProgressToolDelta to avoid duplication.
					if ev.Message != nil {
						if thinking := ev.Message.ThinkingContent(); thinking != "" {
							ReportToolProgress(ctx, ProgressPayload{
								Kind:     ProgressThinking,
								Agent:    agentName,
								Thinking: thinking,
							})
						}
					}
				} else if ev.Delta != "" {
					payload := ProgressPayload{
						Kind:      ProgressToolDelta,
						Agent:     agentName,
						Delta:     ev.Delta,
						DeltaKind: ev.DeltaKind,
					}
					if ev.DeltaKind == DeltaToolCall {
						if m, ok := ev.Message.(Message); ok {
							for _, tc := range m.ToolCalls() {
								if tc.Name != "" {
									payload.Tool = tc.Name
									break
								}
							}
						}
					}
					ReportToolProgress(ctx, payload)
				}
			}
		case EventToolExecEnd:
			if bg == nil {
				if ev.IsError {
					errMsg := string(ev.Result)
					if len(errMsg) > 200 {
						errMsg = errMsg[:200]
					}
					ReportToolProgress(ctx, ProgressPayload{
						Kind:    ProgressToolError,
						Agent:   agentName,
						Tool:    ev.Tool,
						Message: errMsg,
						IsError: true,
					})
				} else {
					ReportToolProgress(ctx, ProgressPayload{
						Kind:  ProgressToolEnd,
						Agent: agentName,
						Tool:  ev.Tool,
					})
				}
			}
			// Capture terminal tool result for inclusion in subagent output.
			// When StopAfterTool fires, this is the last tool result and
			// contains actionable data (e.g. system_hints) for the caller.
			if !ev.IsError && loopCfg.StopAfterTool != nil && loopCfg.StopAfterTool(ev.Tool) {
				terminalToolResult = string(ev.Result)
			}
			if bg == nil {
				reportSubagentContext(ctx, agentName, contextManager)
			}
		case EventMessageEnd:
			if ev.Message == nil {
				continue
			}
			if bg != nil && bg.outFile != nil {
				if msg, ok := ev.Message.(Message); ok {
					if line, je := json.Marshal(msg); je == nil {
						line = append(line, '\n')
						bg.outFile.Write(line)
					}
				}
			}
			if ev.Message.GetRole() == RoleAssistant {
				lastAssistantContent = ev.Message.TextContent()
				su.Turns++
				if bg == nil {
					ReportToolProgress(ctx, ProgressPayload{
						Kind:    ProgressTurnCounter,
						Agent:   agentName,
						Turn:    su.Turns,
						Summary: fmt.Sprintf("turn %d", su.Turns),
					})
				}
				if msg, ok := ev.Message.(Message); ok && msg.Usage != nil {
					su.Input += msg.Usage.Input
					su.Output += msg.Usage.Output
					su.CacheRead += msg.Usage.CacheRead
					su.CacheWrite += msg.Usage.CacheWrite
					if msg.Usage.Cost != nil {
						su.Cost += msg.Usage.Cost.Total
					}
					if bg != nil {
						bg.rt.Update(bg.taskID, func(e *BackgroundTaskEntry) {
							e.TokensIn += msg.Usage.Input
							e.TokensOut += msg.Usage.Output
						})
					}
				}
				if bg == nil {
					reportSubagentContext(ctx, agentName, contextManager)
				}
			}
		case EventRetry:
			if bg == nil && ev.RetryInfo != nil {
				ReportToolProgress(ctx, ProgressPayload{
					Kind:       ProgressRetry,
					Agent:      agentName,
					Attempt:    ev.RetryInfo.Attempt,
					MaxRetries: ev.RetryInfo.MaxRetries,
					Message:    ev.RetryInfo.Err.Error(),
				})
			}
		case EventError:
			if ev.Err != nil {
				lastErr = ev.Err
			}
		}
	}

	if lastErr != nil && lastAssistantContent == "" {
		return "", su, lastErr
	}
	output = lastAssistantContent
	if terminalToolResult != "" {
		if output != "" {
			output += "\n\n"
		}
		output += terminalToolResult
	}
	if output == "" {
		return "(no output)", su, nil
	}
	return output, su, nil
}

func reportSubagentContext(ctx context.Context, agentName string, mgr ContextManager) {
	if mgr == nil {
		return
	}

	var payload struct {
		Tokens          int     `json:"tokens,omitempty"`
		ContextWindow   int     `json:"context_window,omitempty"`
		Percent         float64 `json:"percent,omitempty"`
		Scope           string  `json:"scope,omitempty"`
		Strategy        string  `json:"strategy,omitempty"`
		ActiveMessages  int     `json:"active_messages,omitempty"`
		SummaryMessages int     `json:"summary_messages,omitempty"`
		CompactedCount  int     `json:"compacted_count,omitempty"`
		KeptCount       int     `json:"kept_count,omitempty"`
	}

	if usage := mgr.Usage(); usage != nil {
		payload.Tokens = usage.Tokens
		payload.ContextWindow = usage.ContextWindow
		payload.Percent = usage.Percent
	}
	if snap := mgr.Snapshot(); snap != nil {
		payload.Scope = snap.Scope
		payload.Strategy = snap.LastStrategy
		payload.ActiveMessages = snap.ActiveMessages
		payload.SummaryMessages = snap.SummaryMessages
		payload.CompactedCount = snap.LastCompactedCount
		payload.KeptCount = snap.LastKeptCount
		if payload.Tokens == 0 && snap.Usage != nil {
			payload.Tokens = snap.Usage.Tokens
			payload.ContextWindow = snap.Usage.ContextWindow
			payload.Percent = snap.Usage.Percent
		}
	}

	meta, err := json.Marshal(payload)
	if err != nil {
		return
	}
	ReportToolProgress(ctx, ProgressPayload{
		Kind:  ProgressContext,
		Agent: agentName,
		Meta:  meta,
	})
}

func boolToInt(b bool) int {
	if b {
		return 1
	}
	return 0
}

// truncate shortens s to maxRunes and appends "..." if needed.
// Safe for multi-byte characters (Chinese, emoji, etc.).
func truncate(s string, maxRunes int) string {
	runes := []rune(s)
	if len(runes) <= maxRunes {
		return s
	}
	return string(runes[:maxRunes]) + "..."
}

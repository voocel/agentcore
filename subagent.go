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

// BackgroundTask tracks a background sub-agent's lifecycle.
// Output is written to a persistent file (OutputFile) with a symlink in the tasks dir.
type BackgroundTask struct {
	ID          string
	Agent       string // agent config name
	Description string
	Prompt      string // original task prompt
	Status      string // "running" | "completed" | "failed"
	StartedAt   time.Time
	EndedAt     time.Time
	OutputFile  string // path to output file on disk
	Error       string
	Progress    []string // tool call history: "Tool(args...)"
	TokensIn    int
	TokensOut   int
	ToolCount   int
	mu          sync.Mutex
	cancel      context.CancelFunc
}

func (bt *BackgroundTask) addProgress(entry string) {
	bt.mu.Lock()
	bt.Progress = append(bt.Progress, entry)
	bt.mu.Unlock()
}

func (bt *BackgroundTask) updateTokens(input, output int) {
	bt.mu.Lock()
	bt.TokensIn += input
	bt.TokensOut += output
	bt.mu.Unlock()
}

func (bt *BackgroundTask) snapshot() BackgroundTask {
	bt.mu.Lock()
	defer bt.mu.Unlock()
	prog := make([]string, len(bt.Progress))
	copy(prog, bt.Progress)
	return BackgroundTask{
		ID:          bt.ID,
		Agent:       bt.Agent,
		Description: bt.Description,
		Prompt:      bt.Prompt,
		Status:      bt.Status,
		StartedAt:   bt.StartedAt,
		EndedAt:     bt.EndedAt,
		OutputFile:  bt.OutputFile,
		Error:       bt.Error,
		Progress:    prog,
		TokensIn:    bt.TokensIn,
		TokensOut:   bt.TokensOut,
		ToolCount:   bt.ToolCount,
	}
}

const (
	maxParallelTasks = 8
	maxConcurrency   = 4
)

// SubAgentConfig defines a sub-agent's identity and capabilities.
type SubAgentConfig struct {
	Name         string
	Description  string
	Model        ChatModel
	SystemPrompt string
	Tools        []Tool
	StreamFn     StreamFn
	MaxTurns     int

	// Optional two-stage context pipeline (same as Agent-level).
	// When set, enables automatic context compaction for long-running sub-agents.
	TransformContext func(ctx context.Context, msgs []AgentMessage) ([]AgentMessage, error)
	ConvertToLLM     func(msgs []AgentMessage) []Message
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
	agents       map[string]SubAgentConfig
	notifyFn        func(AgentMessage)                   // called when a background task completes
	createModel     func(name string) (ChatModel, error) // resolves model name to ChatModel at runtime
	bgOutputFactory func(taskID, agentName string) (io.WriteCloser, string, error) // creates output writer for background tasks
	mu           sync.Mutex
	bgSeq        int
	bgTasks      map[string]*BackgroundTask // background task registry
}

// NewSubAgentTool creates a subagent tool from a set of agent configs.
func NewSubAgentTool(agents ...SubAgentConfig) *SubAgentTool {
	m := make(map[string]SubAgentConfig, len(agents))
	for _, a := range agents {
		m[a.Name] = a
	}
	return &SubAgentTool{
		agents:  m,
		bgTasks: make(map[string]*BackgroundTask),
	}
}

// BackgroundTasks returns a snapshot of all background tasks.
func (t *SubAgentTool) BackgroundTasks() []BackgroundTask {
	t.mu.Lock()
	defer t.mu.Unlock()
	tasks := make([]BackgroundTask, 0, len(t.bgTasks))
	for _, bt := range t.bgTasks {
		tasks = append(tasks, bt.snapshot())
	}
	return tasks
}

// StopBackgroundTask cancels a running background task by ID.
func (t *SubAgentTool) StopBackgroundTask(id string) bool {
	t.mu.Lock()
	defer t.mu.Unlock()
	bt, ok := t.bgTasks[id]
	if !ok || bt.Status != "running" {
		return false
	}
	bt.cancel()
	bt.mu.Lock()
	bt.Status = "failed"
	bt.Error = "stopped by user"
	bt.EndedAt = time.Now()
	bt.mu.Unlock()
	return true
}

// StopAllBackgroundTasks cancels all running background tasks.
func (t *SubAgentTool) StopAllBackgroundTasks() int {
	t.mu.Lock()
	defer t.mu.Unlock()
	count := 0
	for _, bt := range t.bgTasks {
		if bt.Status == "running" {
			bt.cancel()
			bt.mu.Lock()
			bt.Status = "failed"
			bt.Error = "stopped by user"
			bt.EndedAt = time.Now()
			bt.mu.Unlock()
			count++
		}
	}
	return count
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
	if params.Background {
		if !hasSingle {
			return json.Marshal("background mode requires agent + task")
		}
		return t.executeBackground(params.Agent, params.Task, params.Description, modelOverride)
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

	taskID := t.nextBgID()
	if description == "" {
		description = truncate(task, 80)
	}

	bgCtx, cancel := context.WithCancel(context.Background())

	bt := &BackgroundTask{
		ID:          taskID,
		Agent:       agentName,
		Description: description,
		Prompt:      task,
		Status:      "running",
		StartedAt:   time.Now(),
		cancel:      cancel,
	}
	t.mu.Lock()
	t.bgTasks[taskID] = bt
	t.mu.Unlock()

	go func() {
		defer cancel()

		var outFile io.WriteCloser
		if t.bgOutputFactory != nil {
			w, path, ferr := t.bgOutputFactory(taskID, agentName)
			if ferr == nil {
				outFile = w
				bt.mu.Lock()
				bt.OutputFile = path
				bt.mu.Unlock()
			}
		}

		_, usage, err := t.runAgent(bgCtx, agentName, task, modelOverride, &bgRunOpts{bt: bt, outFile: outFile})
		if outFile != nil {
			outFile.Close()
		}

		bt.mu.Lock()
		if bt.Status != "running" {
			bt.mu.Unlock()
			return
		}
		bt.EndedAt = time.Now()
		if err != nil {
			bt.Status = "failed"
			bt.Error = err.Error()
		} else {
			bt.Status = "completed"
		}
		outputFile := bt.OutputFile
		bt.mu.Unlock()

		result := map[string]any{
			"task_id":     taskID,
			"agent":       agentName,
			"description": description,
			"usage":       usage,
		}
		if err != nil {
			result["status"] = "failed"
			result["error"] = err.Error()
		} else {
			result["status"] = "completed"
			if outputFile != "" {
				result["output_file"] = outputFile
			}
		}
		t.notify(result)
	}()

	return json.Marshal(map[string]any{
		"task_id":     taskID,
		"description": description,
		"status":      "running",
		"message":     fmt.Sprintf("Background task %s started with agent %q. You will receive a notification when it completes.", taskID, agentName),
	})
}

// nextBgID generates a sequential background task ID.
func (t *SubAgentTool) nextBgID() string {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.bgSeq++
	return fmt.Sprintf("bg-%d", t.bgSeq)
}

// notify sends background task results via notifyFn as a follow-up message.
func (t *SubAgentTool) notify(result map[string]any) {
	if t.notifyFn == nil {
		return
	}
	data, err := json.Marshal(result)
	if err != nil {
		return
	}
	msg := UserMsg(fmt.Sprintf("<task-notification>\n%s\n</task-notification>", string(data)))
	t.notifyFn(msg)
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
	bt      *BackgroundTask // background task to update with progress
	outFile io.Writer       // output stream for session persistence (optional)
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

	loopCfg := LoopConfig{
		Model:            cfg.Model,
		StreamFn:         cfg.StreamFn,
		MaxTurns:         cfg.MaxTurns,
		TransformContext: cfg.TransformContext,
		ConvertToLLM:     cfg.ConvertToLLM,
	}
	if modelOverride != nil {
		loopCfg.Model = modelOverride
	}
	if loopCfg.MaxTurns <= 0 {
		loopCfg.MaxTurns = defaultMaxTurns
	}

	events := AgentLoop(ctx, []AgentMessage{UserMsg(task)}, agentCtx, loopCfg)

	var lastAssistantContent string
	var lastErr error
	su := &subagentUsage{}

	for ev := range events {
		switch ev.Type {
		case EventToolExecStart:
			su.Tools++
			if bg != nil {
				bg.bt.mu.Lock()
				bg.bt.ToolCount++
				bg.bt.mu.Unlock()
				label := ev.Tool
				if len(ev.Args) > 0 {
					label += "(" + truncate(string(ev.Args), 60) + ")"
				}
				bg.bt.addProgress(label)
			} else {
				data, _ := json.Marshal(map[string]any{
					"agent": agentName,
					"tool":  ev.Tool,
					"args":  ev.Args,
				})
				ReportToolProgress(ctx, data)
			}
		case EventMessageUpdate:
			if bg == nil {
				if ev.Delta != "" {
					data, _ := json.Marshal(map[string]any{
						"agent": agentName,
						"delta": ev.Delta,
					})
					ReportToolProgress(ctx, data)
				}
				if ev.Message != nil {
					if thinking := ev.Message.ThinkingContent(); thinking != "" {
						data, _ := json.Marshal(map[string]any{
							"agent":    agentName,
							"thinking": thinking,
						})
						ReportToolProgress(ctx, data)
					}
				}
			}
		case EventToolExecEnd:
			if bg == nil && ev.IsError {
				errMsg := string(ev.Result)
				if len(errMsg) > 200 {
					errMsg = errMsg[:200]
				}
				data, _ := json.Marshal(map[string]any{
					"agent":   agentName,
					"tool":    ev.Tool,
					"error":   true,
					"message": errMsg,
				})
				ReportToolProgress(ctx, data)
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
					data, _ := json.Marshal(map[string]any{
						"agent": agentName,
						"turn":  su.Turns,
					})
					ReportToolProgress(ctx, data)
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
						bg.bt.updateTokens(msg.Usage.Input, msg.Usage.Output)
					}
				}
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
	if lastAssistantContent == "" {
		return "(no output)", su, nil
	}
	return lastAssistantContent, su, nil
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

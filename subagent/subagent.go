// Package subagent implements a Tool that delegates work to specialized
// sub-agents with isolated contexts. Four execution modes are supported:
// single, parallel, chain, and background.
package subagent

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"strings"
	"sync"
	"time"

	"github.com/voocel/agentcore"
	"github.com/voocel/agentcore/schema"
	"github.com/voocel/agentcore/task"
)

const (
	maxParallelTasks = 8
	maxConcurrency   = 4
)

// Config defines a sub-agent's identity and capabilities.
type Config struct {
	Name        string
	Description string
	// Model is resolved when each sub-agent run starts. Wrappers that swap the
	// underlying model at runtime (e.g. agentcore.SwappableModel) take effect
	// on the next sub-agent run.
	Model        agentcore.ChatModel
	SystemPrompt string
	// SystemPromptMode is a host-interpreted hint controlling how
	// SystemPrompt composes with the host's base prompt. agentcore does
	// NOT consume this field — the team spawner / executor on the host
	// side reads it to assemble AgentContext.SystemBlocks. Kept as a
	// plain string at the boundary so agentcore stays agnostic to enum
	// values that only matter inside one host; empty / unrecognized
	// values fall back to the host's default mode.
	SystemPromptMode string
	Tools            []agentcore.Tool
	MaxTurns         int

	// MaxRetries caps the LLM call retry count for retryable errors within
	// this sub-agent's loop. 0 (default) disables retry entirely.
	MaxRetries int

	// ToolsAreIdempotent declares this sub-agent's tools are safe to
	// re-execute. See agentcore.WithToolsAreIdempotent for full rationale.
	ToolsAreIdempotent bool

	// StopAfterTools lists tool names that trigger early loop exit after
	// successful execution.
	StopAfterTools []string

	// StopAfterToolResult is the result-aware variant of StopAfterTools.
	StopAfterToolResult func(toolName string, result json.RawMessage) bool

	// OnMessage, if non-nil, is called after each message is appended to
	// context. The agentName and task are provided for session routing.
	OnMessage func(agentName, task string, msg agentcore.AgentMessage)

	// Optional context lifecycle hooks for long-running sub-agents.
	ContextManager        agentcore.ContextManager
	ContextManagerFactory func(model agentcore.ChatModel) agentcore.ContextManager
	ConvertToLLM          func(msgs []agentcore.AgentMessage) []agentcore.Message

	// StopGuardFactory, if non-nil, creates a fresh StopGuard for each run.
	StopGuardFactory func(agentName, task string) agentcore.StopGuard
}

// params is the JSON schema input for the subagent tool. Five mutually
// exclusive modes:
//   - Single: Agent + Task
//   - Parallel: Tasks array
//   - Chain: Chain array with {previous} placeholder
//   - Background: Single + Background=true
//   - Team spawn: Agent + Task + TeamName (long-lived teammate)
type params struct {
	Agent       string      `json:"agent,omitempty"`
	Task        string      `json:"task,omitempty"`
	Tasks       []taskItem  `json:"tasks,omitempty"`
	Chain       []chainStep `json:"chain,omitempty"`
	Background  bool        `json:"background,omitempty"`
	Description string      `json:"description,omitempty"`
	Model       string      `json:"model,omitempty"`

	// Team-spawn parameters. TeamName is the switch: when non-empty the
	// subagent tool delegates to the configured TeamSpawner instead of
	// running a one-shot loop. Name is the teammate's identifier inside the
	// team (defaults to Agent if omitted). Color is an optional UI hint.
	TeamName string `json:"team_name,omitempty"`
	Name     string `json:"name,omitempty"`
	Color    string `json:"color,omitempty"`
}

type taskItem struct {
	Agent string `json:"agent"`
	Task  string `json:"task"`
}

type chainStep struct {
	Agent string `json:"agent"`
	Task  string `json:"task"`
}

// result captures one sub-agent's execution outcome.
type result struct {
	Agent          string          `json:"agent"`
	Task           string          `json:"task"`
	Output         string          `json:"output"`
	TerminalResult json.RawMessage `json:"terminal_result,omitempty"`
	IsError        bool            `json:"is_error,omitempty"`
	Step           int             `json:"step,omitempty"`
	Usage          *usage          `json:"usage,omitempty"`
}

// usage tracks token consumption and cost for a sub-agent run.
type usage struct {
	Input      int     `json:"input"`
	Output     int     `json:"output"`
	CacheRead  int     `json:"cache_read"`
	CacheWrite int     `json:"cache_write"`
	Cost       float64 `json:"cost"`
	Turns      int     `json:"turns"`
	Tools      int     `json:"tools"`
}

// TeamSpawnRequest is the contract between the subagent tool and the
// codebot-side team spawner. The subagent tool builds this from its params
// after validating the requested agent definition exists; the spawner is
// responsible for the actual goroutine launch, tool augmentation
// (e.g. injecting send_message), and team registry bookkeeping.
type TeamSpawnRequest struct {
	// Config is the resolved sub-agent definition the teammate runs as.
	// Spawner reads SystemPrompt, Tools, Model, MaxTurns etc. from here.
	Config Config

	// Name is the teammate's identifier inside the team (routing key for
	// send_message). May equal Config.Name when the LLM did not specify one.
	Name string

	// TeamName is the active team's name; spawner validates against registry.
	TeamName string

	// InitialPrompt is the leader's first message to the teammate.
	InitialPrompt string

	// Description is an optional one-line summary for transcripts/UI.
	Description string

	// Color is an optional UI color assigned to this teammate.
	Color string

	// Model is non-nil when the LLM requested an override; nil means the
	// spawner should fall back to Config.Model.
	Model agentcore.ChatModel

	// History, if non-empty, seeds the teammate's conversation before its
	// first turn — the spawner forwards it to team.SpawnConfig.History. The
	// LLM never sets this; a harness populates it when resuming a teammate
	// with its prior transcript after a restart. nil ⇒ fresh teammate.
	History []agentcore.AgentMessage
}

// TeamSpawnResult is what the spawner returns synchronously. The teammate
// itself runs in the background; callers terminate it via task.Runtime.Stop
// (by TaskID) or by the team's shutdown protocol.
type TeamSpawnResult struct {
	TaskID  string
	AgentID string // "name@team"
}

// TeamSpawner is the function shape codebot installs via SetTeamSpawner.
// Kept as a function rather than an interface because the subagent tool only
// needs one method and call sites are simpler with a closure.
type TeamSpawner func(ctx context.Context, req TeamSpawnRequest) (*TeamSpawnResult, error)

// Tool implements agentcore.Tool. The main agent calls this tool to delegate
// tasks to specialized sub-agents with isolated contexts.
type Tool struct {
	agents          map[string]Config
	notifyFn        func(agentcore.AgentMessage)                                   // called when a background task completes
	createModel     func(name string) (agentcore.ChatModel, error)                 // resolves model name to ChatModel at runtime
	bgOutputFactory func(taskID, agentName string) (io.WriteCloser, string, error) // creates output writer for background tasks
	taskRT          *task.Runtime                                                  // shared background task registry
	teamSpawner     TeamSpawner                                                    // routes team-mode calls; nil means team spawn is rejected
}

// New creates a subagent tool from a set of agent configs.
func New(agents ...Config) *Tool {
	m := make(map[string]Config, len(agents))
	for _, a := range agents {
		m[a.Name] = a
	}
	return &Tool{agents: m}
}

// SetTaskRuntime sets the shared task runtime for background task
// registration. Required for background mode.
func (t *Tool) SetTaskRuntime(rt *task.Runtime) {
	t.taskRT = rt
}

// SetNotifyFn sets the callback invoked when a background task completes.
// Typically bound to Agent.FollowUp so the main agent receives the result as
// a follow-up message.
func (t *Tool) SetNotifyFn(fn func(agentcore.AgentMessage)) {
	t.notifyFn = fn
}

// SetCreateModel sets the factory for resolving model names to ChatModel
// instances at runtime. Enables LLM to override the default model per call.
func (t *Tool) SetCreateModel(fn func(name string) (agentcore.ChatModel, error)) {
	t.createModel = fn
}

// SetTeamSpawner installs the closure that handles team-spawn mode. Without
// it, calls that set team_name are rejected with a clear error so the LLM
// learns the feature is unavailable rather than silently downgrading to a
// regular subagent run.
func (t *Tool) SetTeamSpawner(fn TeamSpawner) {
	t.teamSpawner = fn
}

// AgentConfig returns the registered sub-agent definition for name, or
// (zero, false) if none is registered. Exposed read-only so a harness can
// rebuild a TeamSpawnRequest when resuming a teammate by its agent type
// without re-deriving the config from scratch.
func (t *Tool) AgentConfig(name string) (Config, bool) {
	cfg, ok := t.agents[name]
	return cfg, ok
}

// SetBgOutputFactory sets the factory that creates output writers for
// background tasks. The factory receives the task ID and agent name and
// returns a writer, file path, and error.
func (t *Tool) SetBgOutputFactory(fn func(taskID, agentName string) (io.WriteCloser, string, error)) {
	t.bgOutputFactory = fn
}

func (t *Tool) Name() string  { return "subagent" }
func (t *Tool) Label() string { return "Delegate to SubAgent" }

func (t *Tool) Description() string {
	names := make([]string, 0, len(t.agents))
	for _, a := range t.agents {
		names = append(names, fmt.Sprintf("%s (%s)", a.Name, a.Description))
	}
	return fmt.Sprintf(
		"Delegate tasks to specialized subagents with isolated context. "+
			"Modes: single (agent+task), parallel (tasks array), chain (sequential with {previous} placeholder), "+
			"background (agent+task+background=true, returns immediately and notifies on completion), "+
			"team (agent+task+team_name spawns a long-lived teammate that communicates via send_message). "+
			"Available agents: %s",
		strings.Join(names, ", "),
	)
}

func (t *Tool) Schema() map[string]any {
	agentNames := make([]string, 0, len(t.agents))
	for name := range t.agents {
		agentNames = append(agentNames, name)
	}
	taskItem := schema.Object(
		schema.Property("agent", schema.Enum("Agent name", agentNames...)).Required(),
		schema.Property("task", schema.String("Task description")).Required(),
	)
	return schema.Object(
		schema.Property("agent", schema.Enum("Name of the agent to invoke (single/background/team mode)", agentNames...)),
		schema.Property("task", schema.String("Task to delegate (single/background/team mode)")),
		schema.Property("tasks", schema.Array("Array of {agent, task} for parallel execution", taskItem)),
		schema.Property("chain", schema.Array("Array of {agent, task} for sequential execution. Use {previous} in task to reference prior output.", taskItem)),
		schema.Property("background", schema.Bool("Set true to run in background. Returns immediately; a notification is sent when the task completes.")),
		schema.Property("description", schema.String("Short description of the background/team task (shown in notifications and listings).")),
		schema.Property("model", schema.String("Optional model override for this call (e.g. model ID or alias). If not set, uses the agent's default model.")),
		schema.Property("team_name", schema.String("Name of the active team. When set, spawns a long-lived teammate instead of running a one-shot subagent.")),
		schema.Property("name", schema.String("Teammate name inside the team (defaults to the agent's name). Must be unique and not 'team-lead'.")),
		schema.Property("color", schema.String("Optional UI color tag for the teammate.")),
	)
}

func (t *Tool) Execute(ctx context.Context, args json.RawMessage) (json.RawMessage, error) {
	var p params
	if err := json.Unmarshal(args, &p); err != nil {
		return nil, fmt.Errorf("invalid subagent params: %w", err)
	}

	// Resolve model override once (applies to all subtasks in this call).
	var modelOverride agentcore.ChatModel
	if p.Model != "" && t.createModel != nil {
		m, err := t.createModel(p.Model)
		if err != nil {
			return json.Marshal(map[string]any{"error": fmt.Sprintf("invalid model %q: %v", p.Model, err)})
		}
		modelOverride = m
	}

	hasChain := len(p.Chain) > 0
	hasParallel := len(p.Tasks) > 0
	hasSingle := p.Agent != "" && p.Task != ""

	// Team-spawn mode: long-lived teammate. Mutually exclusive with the
	// other modes — Background and team_name together is ambiguous, and
	// parallel/chain are conceptually one-shot. Check this BEFORE Background
	// so a user calling with both keys gets the team-mode error path.
	if p.TeamName != "" {
		if p.Background || hasChain || hasParallel {
			return nil, fmt.Errorf("team_name is mutually exclusive with background/tasks/chain")
		}
		if !hasSingle {
			return nil, fmt.Errorf("team mode requires agent + task")
		}
		return t.executeTeamSpawn(ctx, p, modelOverride)
	}

	// Background mode: single task running in a detached goroutine.
	// Requires a wired TaskRuntime — no silent degradation to sync, because
	// callers passing Background=true expect "return immediately + notify on
	// completion" semantics that synchronous execution cannot satisfy.
	if p.Background {
		if !hasSingle {
			return nil, fmt.Errorf("background mode requires agent + task")
		}
		if t.taskRT == nil {
			return nil, fmt.Errorf("background mode requires a wired TaskRuntime (call subagent.Tool.SetTaskRuntime)")
		}
		return t.executeBackground(ctx, p.Agent, p.Task, p.Description, modelOverride)
	}

	modeCount := boolToInt(hasChain) + boolToInt(hasParallel) + boolToInt(hasSingle)
	if modeCount != 1 {
		return nil, fmt.Errorf("invalid parameters: provide exactly one mode (agent+task, tasks, or chain)")
	}

	switch {
	case hasChain:
		return t.executeChain(ctx, p.Chain, modelOverride)
	case hasParallel:
		return t.executeParallel(ctx, p.Tasks, modelOverride)
	default:
		return t.executeSingle(ctx, p.Agent, p.Task, modelOverride)
	}
}

// executeTeamSpawn delegates to the installed TeamSpawner. The subagent tool
// validates the requested agent definition exists and prepares a TeamSpawnRequest
// from params; the spawner owns the actual goroutine launch, tool-set
// augmentation (e.g. add send_message), and registry bookkeeping. This split
// keeps agentcore/subagent unaware of team-specific tools while still routing
// team spawn through one user-facing surface.
func (t *Tool) executeTeamSpawn(ctx context.Context, p params, modelOverride agentcore.ChatModel) (json.RawMessage, error) {
	if t.teamSpawner == nil {
		return nil, fmt.Errorf("team spawn is not configured in this environment")
	}
	cfg, ok := t.agents[p.Agent]
	if !ok {
		available := make([]string, 0, len(t.agents))
		for name := range t.agents {
			available = append(available, name)
		}
		return json.Marshal(map[string]any{
			"error": fmt.Sprintf("unknown agent %q, available: %s", p.Agent, strings.Join(available, ", ")),
		})
	}
	name := p.Name
	if name == "" {
		name = cfg.Name
	}

	req := TeamSpawnRequest{
		Config:        cfg,
		Name:          name,
		TeamName:      p.TeamName,
		InitialPrompt: p.Task,
		Description:   p.Description,
		Color:         p.Color,
		Model:         modelOverride,
	}
	res, err := t.teamSpawner(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("team spawn failed: %w", err)
	}
	return json.Marshal(map[string]any{
		"task_id":  res.TaskID,
		"agent_id": res.AgentID,
		"status":   "running",
		"message": fmt.Sprintf("Teammate %q (agent=%s) spawned in team %q. Send messages with send_message.",
			res.AgentID, p.Agent, p.TeamName),
	})
}

// executeBackground launches a sub-agent in a detached goroutine and returns
// immediately. When the sub-agent finishes, a notification is sent via
// notifyFn (typically Agent.FollowUp).
func (t *Tool) executeBackground(callerCtx context.Context, agentName, taskStr, description string, modelOverride agentcore.ChatModel) (json.RawMessage, error) {
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

	// Depth guard: refuse to spawn beyond MaxAgentDepth so a runaway recursion
	// (e.g. a future team member accidentally given a spawn channel and looping)
	// can't burn through tokens before someone notices. callerCtx carries the
	// caller's depth via task.DepthFromContext — 0 when called from the main
	// agent loop, n inside a depth-n sub-agent.
	childDepth := task.DepthFromContext(callerCtx) + 1
	if childDepth > task.MaxAgentDepth {
		return json.Marshal(map[string]any{
			"error": fmt.Sprintf("agent nesting depth %d exceeds max %d — refusing to spawn", childDepth, task.MaxAgentDepth),
		})
	}

	taskID := rt.NextID("bg")
	if description == "" {
		description = truncate(taskStr, 80)
	}

	// Detach from caller ctx on purpose: background tasks outlive the parent
	// agent's current turn. Session-level shutdown is handled by
	// task.Runtime.StopAll() (wired in Runtime.Close), which invokes this
	// cancel func — so a "zombie bg goroutine after process exit" is impossible.
	bgCtx, cancel := context.WithCancel(context.Background())
	// Thread the child's depth into bgCtx so any spawn the child itself makes
	// will see the correct parent depth when reading DepthFromContext.
	bgCtx = task.WithDepth(bgCtx, childDepth)

	entry := &task.Entry{
		ID:          taskID,
		Type:        task.TypeSubAgent,
		Agent:       agentName,
		Prompt:      taskStr,
		Description: description,
		Status:      task.Running,
		StartedAt:   time.Now(),
		Depth:       childDepth,
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
				rt.Update(taskID, func(e *task.Entry) { e.OutputFile = path })
			}
		}

		output, _, u, err := t.runAgent(bgCtx, agentName, taskStr, modelOverride, &bgRunOpts{taskID: taskID, rt: rt, outFile: outFile})
		if outFile != nil {
			outFile.Close()
		}

		rt.Update(taskID, func(e *task.Entry) {
			e.EndedAt = time.Now()
			e.Result = output
			switch {
			case err != nil && bgCtx.Err() != nil:
				// Cancellation observed: this was an explicit Stop, not a failure.
				e.Status = task.Killed
			case err != nil:
				e.Status = task.Failed
				e.Error = err.Error()
			default:
				e.Status = task.Completed
			}
			if u != nil {
				e.TokensIn = u.Input
				e.TokensOut = u.Output
				e.ToolCount = u.Tools
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
func (t *Tool) notify(taskID string) {
	if t.notifyFn == nil || t.taskRT == nil {
		return
	}
	entry := t.taskRT.Get(taskID)
	if entry == nil {
		return
	}
	t.notifyFn(task.NotificationFromEntry(entry).ToAgentMessage())
}

// executeSingle runs one sub-agent with an isolated context.
func (t *Tool) executeSingle(ctx context.Context, agentName, taskStr string, modelOverride agentcore.ChatModel) (json.RawMessage, error) {
	output, terminalResult, u, err := t.runAgent(ctx, agentName, taskStr, modelOverride, nil)
	if err != nil {
		if u != nil {
			return nil, fmt.Errorf("agent %q failed: %w (turns=%d tools=%d)", agentName, err, u.Turns, u.Tools)
		}
		return nil, fmt.Errorf("agent %q failed: %w", agentName, err)
	}
	out := map[string]any{
		"output": output,
		"usage":  u,
	}
	if len(terminalResult) > 0 {
		out["terminal_result"] = terminalResult
	}
	return json.Marshal(out)
}

// executeChain runs sub-agents sequentially, passing each output to the next
// via {previous}.
func (t *Tool) executeChain(ctx context.Context, chain []chainStep, modelOverride agentcore.ChatModel) (json.RawMessage, error) {
	var previous string
	results := make([]result, 0, len(chain))

	for i, step := range chain {
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}

		taskStr := strings.ReplaceAll(step.Task, "{previous}", previous)
		output, terminalResult, u, err := t.runAgent(ctx, step.Agent, taskStr, modelOverride, nil)

		r := result{
			Agent:          step.Agent,
			Task:           taskStr,
			Step:           i + 1,
			Usage:          u,
			TerminalResult: terminalResult,
		}

		if err != nil {
			r.Output = err.Error()
			r.IsError = true
			results = append(results, r)
			return json.Marshal(map[string]any{
				"error":   fmt.Sprintf("Chain stopped at step %d (%s): %v", i+1, step.Agent, err),
				"results": results,
			})
		}

		r.Output = output
		results = append(results, r)
		previous = output
	}

	return json.Marshal(map[string]any{
		"output":  previous,
		"results": results,
	})
}

// executeParallel runs multiple sub-agents concurrently with bounded
// concurrency.
func (t *Tool) executeParallel(ctx context.Context, tasks []taskItem, modelOverride agentcore.ChatModel) (json.RawMessage, error) {
	if len(tasks) > maxParallelTasks {
		return json.Marshal(fmt.Sprintf("Too many parallel tasks (%d). Max is %d.", len(tasks), maxParallelTasks))
	}

	results := make([]result, len(tasks))
	var wg sync.WaitGroup
	sem := make(chan struct{}, maxConcurrency)

	for i, ti := range tasks {
		wg.Add(1)
		go func(idx int, st taskItem) {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()

			output, terminalResult, u, err := t.runAgent(ctx, st.Agent, st.Task, modelOverride, nil)
			r := result{
				Agent:          st.Agent,
				Task:           st.Task,
				Usage:          u,
				TerminalResult: terminalResult,
			}
			if err != nil {
				r.Output = err.Error()
				r.IsError = true
			} else {
				r.Output = output
			}
			results[idx] = r
		}(i, ti)
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

// bgRunOpts configures background-specific behavior for runAgent. When nil,
// runAgent runs in foreground mode (reports progress to parent context).
type bgRunOpts struct {
	taskID  string        // task ID in the runtime
	rt      *task.Runtime // shared runtime for updates
	outFile io.Writer     // output stream for session persistence (optional)
}

// runAgent executes an isolated agent loop for the given agent config and
// task. Includes panic recovery to prevent a subagent crash from taking down
// the parent.
func (t *Tool) runAgent(ctx context.Context, agentName, taskStr string, modelOverride agentcore.ChatModel, bg *bgRunOpts) (output string, terminalResult json.RawMessage, u *usage, err error) {
	cfg, ok := t.agents[agentName]
	if !ok {
		available := make([]string, 0, len(t.agents))
		for name := range t.agents {
			available = append(available, name)
		}
		return "", nil, nil, fmt.Errorf("unknown agent %q, available: %s", agentName, strings.Join(available, ", "))
	}

	// Panic recovery — isolated subagent should never crash the parent.
	defer func() {
		if r := recover(); r != nil {
			err = fmt.Errorf("subagent %q panicked: %v", agentName, r)
		}
	}()

	agentCtx := agentcore.AgentContext{
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

	loopCfg := agentcore.LoopConfig{
		Model:              runModel,
		MaxTurns:           cfg.MaxTurns,
		MaxRetries:         cfg.MaxRetries,
		ToolsAreIdempotent: cfg.ToolsAreIdempotent,
		ContextManager:     contextManager,
		ConvertToLLM:       cfg.ConvertToLLM,
	}

	// Drain parent→child messages at every steering tick so a SendToSubAgent
	// call mid-execution gets seen on the next turn rather than after the
	// agent decides to stop. Foreground runs have no task entry to drain from,
	// so this is only wired in the background path.
	if bg != nil && bg.rt != nil {
		taskID := bg.taskID
		rt := bg.rt
		loopCfg.GetSteeringMessages = func() []agentcore.AgentMessage {
			drained := rt.DrainPending(taskID)
			if len(drained) == 0 {
				return nil
			}
			msgs := make([]agentcore.AgentMessage, 0, len(drained))
			for _, m := range drained {
				msgs = append(msgs, agentcore.UserMsg(m))
			}
			return msgs
		}
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
	loopCfg.StopAfterToolResult = cfg.StopAfterToolResult
	if cfg.StopGuardFactory != nil {
		loopCfg.StopGuard = cfg.StopGuardFactory(agentName, taskStr)
	}
	if cfg.OnMessage != nil {
		name, ts := agentName, taskStr
		loopCfg.OnMessage = func(msg agentcore.AgentMessage) { cfg.OnMessage(name, ts, msg) }
	}

	events := agentcore.AgentLoop(ctx, []agentcore.AgentMessage{agentcore.UserMsg(taskStr)}, agentCtx, loopCfg)

	var lastAssistantContent string
	var terminalToolResult json.RawMessage // result from StopAfterTool trigger
	var lastErr error
	su := &usage{}

	for ev := range events {
		switch ev.Type {
		case agentcore.EventToolExecStart:
			su.Tools++
			if bg != nil {
				bg.rt.Update(bg.taskID, func(e *task.Entry) { e.ToolCount++ })
				if bg.outFile != nil {
					label := ev.Tool
					if len(ev.Args) > 0 {
						label += "(" + truncate(string(ev.Args), 60) + ")"
					}
					fmt.Fprintf(bg.outFile, "[tool] %s\n", label)
				}
			} else {
				agentcore.ReportToolProgress(ctx, agentcore.ProgressPayload{
					Kind:    agentcore.ProgressToolStart,
					Agent:   agentName,
					Tool:    ev.Tool,
					Summary: ev.Tool,
					Args:    ev.Args,
				})
			}
		case agentcore.EventMessageUpdate:
			if bg == nil {
				if ev.DeltaKind == agentcore.DeltaThinking {
					// Thinking deltas only go through ProgressThinking (cumulative).
					if ev.Message != nil {
						if thinking := ev.Message.ThinkingContent(); thinking != "" {
							agentcore.ReportToolProgress(ctx, agentcore.ProgressPayload{
								Kind:     agentcore.ProgressThinking,
								Agent:    agentName,
								Thinking: thinking,
							})
						}
					}
				} else if ev.Delta != "" {
					payload := agentcore.ProgressPayload{
						Kind:      agentcore.ProgressToolDelta,
						Agent:     agentName,
						Delta:     ev.Delta,
						DeltaKind: ev.DeltaKind,
					}
					if ev.DeltaKind == agentcore.DeltaToolCall {
						if m, ok := ev.Message.(agentcore.Message); ok {
							for _, tc := range m.ToolCalls() {
								if tc.Name != "" {
									payload.Tool = tc.Name
									break
								}
							}
						}
					}
					agentcore.ReportToolProgress(ctx, payload)
				}
			}
		case agentcore.EventToolExecEnd:
			if bg == nil {
				if ev.IsError {
					errMsg := string(ev.Result)
					if len(errMsg) > 200 {
						errMsg = errMsg[:200]
					}
					agentcore.ReportToolProgress(ctx, agentcore.ProgressPayload{
						Kind:    agentcore.ProgressToolError,
						Agent:   agentName,
						Tool:    ev.Tool,
						Message: errMsg,
						IsError: true,
					})
				} else {
					agentcore.ReportToolProgress(ctx, agentcore.ProgressPayload{
						Kind:  agentcore.ProgressToolEnd,
						Agent: agentName,
						Tool:  ev.Tool,
					})
				}
			}
			// Capture terminal tool result for inclusion in subagent output.
			if !ev.IsError && ((loopCfg.StopAfterTool != nil && loopCfg.StopAfterTool(ev.Tool)) ||
				(loopCfg.StopAfterToolResult != nil && loopCfg.StopAfterToolResult(ev.Tool, ev.Result))) {
				terminalToolResult = append(terminalToolResult[:0], ev.Result...)
			}
			if bg == nil {
				reportContext(ctx, agentName, contextManager)
			}
		case agentcore.EventMessageEnd:
			if ev.Message == nil {
				continue
			}
			if bg != nil && bg.outFile != nil {
				if msg, ok := ev.Message.(agentcore.Message); ok {
					if line, je := json.Marshal(msg); je == nil {
						line = append(line, '\n')
						bg.outFile.Write(line)
					}
				}
			}
			if ev.Message.GetRole() == agentcore.RoleAssistant {
				lastAssistantContent = ev.Message.TextContent()
				su.Turns++
				if bg == nil {
					agentcore.ReportToolProgress(ctx, agentcore.ProgressPayload{
						Kind:    agentcore.ProgressTurnCounter,
						Agent:   agentName,
						Turn:    su.Turns,
						Summary: fmt.Sprintf("turn %d", su.Turns),
					})
				}
				if msg, ok := ev.Message.(agentcore.Message); ok && msg.Usage != nil {
					su.Input += msg.Usage.Input
					su.Output += msg.Usage.Output
					su.CacheRead += msg.Usage.CacheRead
					su.CacheWrite += msg.Usage.CacheWrite
					if msg.Usage.Cost != nil {
						su.Cost += msg.Usage.Cost.Total
					}
					if bg != nil {
						bg.rt.Update(bg.taskID, func(e *task.Entry) {
							e.TokensIn += msg.Usage.Input
							e.TokensOut += msg.Usage.Output
						})
					}
				}
				if bg == nil {
					reportContext(ctx, agentName, contextManager)
				}
			}
		case agentcore.EventRetry:
			if bg == nil && ev.RetryInfo != nil {
				agentcore.ReportToolProgress(ctx, agentcore.ProgressPayload{
					Kind:       agentcore.ProgressRetry,
					Agent:      agentName,
					Attempt:    ev.RetryInfo.Attempt,
					MaxRetries: ev.RetryInfo.MaxRetries,
					Message:    ev.RetryInfo.Err.Error(),
				})
			}
		case agentcore.EventError:
			if ev.Err != nil {
				lastErr = ev.Err
			}
		}
	}

	if lastErr != nil {
		return "", nil, su, lastErr
	}
	output = lastAssistantContent
	if len(terminalToolResult) > 0 {
		if output != "" {
			output += "\n\n"
		}
		output += string(terminalToolResult)
	}
	if output == "" {
		return "(no output)", terminalToolResult, su, nil
	}
	return output, terminalToolResult, su, nil
}

func reportContext(ctx context.Context, agentName string, mgr agentcore.ContextManager) {
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

	if u := mgr.Usage(); u != nil {
		payload.Tokens = u.Tokens
		payload.ContextWindow = u.ContextWindow
		payload.Percent = u.Percent
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
	agentcore.ReportToolProgress(ctx, agentcore.ProgressPayload{
		Kind:  agentcore.ProgressContext,
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

// truncate shortens s to maxRunes and appends "..." if needed. Safe for
// multi-byte characters.
func truncate(s string, maxRunes int) string {
	runes := []rune(s)
	if len(runes) <= maxRunes {
		return s
	}
	return string(runes[:maxRunes]) + "..."
}

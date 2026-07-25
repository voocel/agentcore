// Package subagent runs specialized agents with isolated contexts.
// Runner is the typed host-facing API. Tool adapts a Runner for model-driven
// delegation, including parallel, chain, background, and team execution.
package subagent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"maps"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"time"
	"unicode/utf8"

	"github.com/voocel/agentcore"
	"github.com/voocel/agentcore/schema"
	"github.com/voocel/agentcore/task"
	"github.com/voocel/agentcore/tools"
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

	// ThinkingLevel sets the reasoning depth for this sub-agent's runs.
	// Empty ("") leaves it unspecified (model/provider default). Mirrors
	// agentcore.WithThinkingLevel for top-level agents. A runtime override
	// installed via Runner.SetThinkingLevel takes precedence over this baseline.
	ThinkingLevel agentcore.ThinkingLevel

	// MaxRetries caps the LLM call retry count for retryable errors within
	// this sub-agent's loop. 0 (default) disables retry entirely.
	MaxRetries int

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

	// CacheLastMessage, when non-empty, tags the last non-system message of
	// every LLM request in this sub-agent's loop with the given cache_control
	// value ("ephemeral", or "ephemeral:1h" for extended TTL). Mirrors
	// agentcore.WithCacheLastMessage for top-level agents; see that option
	// for placement semantics.
	CacheLastMessage string

	// PromptCacheKey is the base prompt-cache routing identity for this
	// sub-agent's LLM requests. Each spawn appends "#<seq>" so every run gets
	// its own cache lineage — one conversation, one key — which providers
	// with key-routed prefix caching (OpenAI prompt_cache_key) use to keep a
	// session's requests on the same cache shard. Empty sends no hint.
	PromptCacheKey string

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

	// Team-spawn parameters. Name or TeamName selects team mode. Name is the
	// teammate's identifier inside the team (defaults to Agent if omitted);
	// TeamName may be empty when the host provides a default team.
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

// result captures one sub-agent's execution outcome for the LLM-facing
// JSON surface (chain/parallel modes).
type result struct {
	Agent          string          `json:"agent"`
	Task           string          `json:"task"`
	Output         string          `json:"output"`
	TerminalResult json.RawMessage `json:"terminal_result,omitempty"`
	IsError        bool            `json:"is_error,omitempty"`
	Step           int             `json:"step,omitempty"`
	Usage          *Usage          `json:"usage,omitempty"`
}

// Usage aggregates token consumption and loop counters for one sub-agent run.
type Usage struct {
	Input      int     `json:"input"`
	Output     int     `json:"output"`
	CacheRead  int     `json:"cache_read"`
	CacheWrite int     `json:"cache_write"`
	Cost       float64 `json:"cost"`
	Turns      int     `json:"turns"`
	Tools      int     `json:"tools"`
}

// ErrUnknownAgent is the sentinel for "agent name not registered with this
// Runner". Match with errors.Is; use errors.As with *UnknownAgentError to read
// the requested name and the available set. A host scheduler branches on this
// to classify the failure as deterministic (retrying the same run cannot
// succeed) without matching error strings.
var ErrUnknownAgent = errors.New("unknown agent")

// UnknownAgentError reports a lookup failure against the Runner's registry.
// errors.Is matches ErrUnknownAgent.
type UnknownAgentError struct {
	Agent     string
	Available []string
}

func (e *UnknownAgentError) Error() string {
	return fmt.Sprintf("unknown agent %q, available: %s", e.Agent, strings.Join(e.Available, ", "))
}

func (e *UnknownAgentError) Is(target error) bool { return target == ErrUnknownAgent }

// RunResult is the typed outcome of one sub-agent run.
type RunResult struct {
	// Agent is the registered agent definition that ran.
	Agent string

	// Output is the final assistant text. Unlike the LLM tool-call surface,
	// it is NOT concatenated with TerminalResult and has no "(no output)"
	// placeholder — an agent that only called tools yields "".
	Output string

	// TerminalResult is the successful result of the tool that triggered a
	// StopAfterTools / StopAfterToolResult exit, nil when the run ended some
	// other way.
	TerminalResult json.RawMessage

	// Usage carries aggregated counters. Populated on both success and
	// failure paths (a run that errors mid-way still consumed tokens);
	// zero-valued when the run never started (e.g. unknown agent).
	Usage Usage
}

// displayOutput renders the run for the LLM-facing tool-call surface:
// terminal tool result appended after the assistant text, with a
// "(no output)" placeholder when both are empty. Programmatic callers use
// the raw fields instead.
func (r RunResult) displayOutput() string {
	out := r.Output
	if len(r.TerminalResult) > 0 {
		if out != "" {
			out += "\n\n"
		}
		out += string(r.TerminalResult)
	}
	if out == "" {
		return "(no output)"
	}
	return out
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

// RunMeta identifies one sub-agent run for an
// external event observer. It lets a harness route a run's raw AgentLoop
// events to a per-run sink (e.g. a live-preview transcript) without the
// subagent tool knowing anything about that sink.
//
//   - Agent:      the agent definition/type name (e.g. "explore"). Not unique
//     when the same type runs more than once concurrently (parallel mode).
//   - InstanceID: unique per run invocation within this Runner's lifetime.
//     Use this — not Agent — as the routing key.
//   - Mode:       one of the Mode* constants below.
type RunMeta struct {
	Agent      string
	InstanceID string
	Mode       string
}

// Run modes — the authoritative set of values for RunMeta.Mode. Observers
// should compare against these rather than string literals.
const (
	ModeSingle     = "single"
	ModeParallel   = "parallel"
	ModeChain      = "chain"
	ModeBackground = "background"
)

// Runner executes registered agents through AgentLoop. It owns only agent
// definitions and per-run behavior; model-facing JSON and background/team
// orchestration live in Tool.
type Runner struct {
	agents        map[string]Config
	thinkMu       sync.RWMutex
	thinkOverride map[string]agentcore.ThinkingLevel
	eventObserver func(meta RunMeta, ev agentcore.Event)
	runSeq        atomic.Int64
}

// NewRunner creates a Runner from the supplied agent definitions. It panics
// when an agent name is empty or duplicated because the registry is static
// program configuration and either condition is a programming error.
func NewRunner(agents ...Config) *Runner {
	m := make(map[string]Config, len(agents))
	for _, agent := range agents {
		if agent.Name == "" {
			panic("subagent: agent name is required")
		}
		if _, exists := m[agent.Name]; exists {
			panic(fmt.Sprintf("subagent: duplicate agent %q", agent.Name))
		}
		m[agent.Name] = agent
	}
	return &Runner{agents: m}
}

// AsTool exposes model-driven delegation backed by this Runner.
func (r *Runner) AsTool() *Tool {
	return &Tool{runner: r}
}

// Tool implements agentcore.Tool as an adapter over Runner.
type Tool struct {
	runner          *Runner
	notifyFn        func(agentcore.AgentMessage)                                   // called when a background task completes
	createModel     func(name string) (agentcore.ChatModel, error)                 // resolves model name to ChatModel at runtime
	bgOutputFactory func(taskID, agentName string) (io.WriteCloser, string, error) // creates output writer for background tasks
	taskRT          *task.Runtime                                                  // shared background task registry
	teamSpawner     TeamSpawner                                                    // routes team-mode calls; nil means team spawn is rejected
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

// SetThinkingLevel overrides a sub-agent's reasoning depth at runtime, keyed by
// agent name. It takes effect on the next run of that agent (mirroring how a
// SwappableModel swap takes effect on the next run) and overrides the agent's
// Config.ThinkingLevel baseline. Safe to call concurrently with running agents:
// the override lives in an isolated map and never mutates the immutable agents
// config map. Empty level ("") means model/provider default.
func (r *Runner) SetThinkingLevel(agentName string, level agentcore.ThinkingLevel) {
	r.thinkMu.Lock()
	defer r.thinkMu.Unlock()
	if r.thinkOverride == nil {
		r.thinkOverride = make(map[string]agentcore.ThinkingLevel)
	}
	r.thinkOverride[agentName] = level
}

// resolveThinking returns the runtime override for agentName if one was
// installed via SetThinkingLevel, otherwise the config baseline.
func (r *Runner) resolveThinking(agentName string, base agentcore.ThinkingLevel) agentcore.ThinkingLevel {
	r.thinkMu.RLock()
	defer r.thinkMu.RUnlock()
	if lv, ok := r.thinkOverride[agentName]; ok {
		return lv
	}
	return base
}

// SetTeamSpawner installs the closure that handles team-spawn mode. Without
// it, calls that set name or team_name are rejected with a clear error so the LLM
// learns the feature is unavailable rather than silently downgrading to a
// regular subagent run.
func (t *Tool) SetTeamSpawner(fn TeamSpawner) {
	t.teamSpawner = fn
}

// SetEventObserver installs a callback that receives every raw AgentLoop event
// produced by any sub-agent run (single/parallel/chain/background), tagged with
// a RunMeta carrying a unique per-run InstanceID. A harness uses this to drive
// a live preview of sub-agent work — symmetric to how a teammate executor fans
// its loop events out. nil (the default) disables observation with zero cost.
//
// The callback MUST be non-blocking: it runs inline on the sub-agent's
// execution goroutine (and on parallel/background goroutines concurrently), so
// a slow observer stalls the run. Sinks that may block should buffer + drop.
func (r *Runner) SetEventObserver(fn func(meta RunMeta, ev agentcore.Event)) {
	r.eventObserver = fn
}

// AgentConfig returns the registered sub-agent definition for name, or
// (zero, false) if none is registered. Exposed read-only so a harness can
// rebuild a TeamSpawnRequest when resuming a teammate by its agent type
// without re-deriving the config from scratch.
func (r *Runner) AgentConfig(name string) (Config, bool) {
	cfg, ok := r.agents[name]
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

// sortedAgentNames returns registered agent names in deterministic order.
// Description and Schema are rebuilt on every LLM call; iterating the map
// directly would shuffle their bytes across requests and defeat provider
// prefix caching (tools serialize into the cached prompt prefix).
func (r *Runner) sortedAgentNames() []string {
	return slices.Sorted(maps.Keys(r.agents))
}

func (t *Tool) Description() string {
	names := make([]string, 0, len(t.runner.agents))
	for _, name := range t.runner.sortedAgentNames() {
		a := t.runner.agents[name]
		names = append(names, fmt.Sprintf("%s (%s)", a.Name, a.Description))
	}
	return fmt.Sprintf(
		"Delegate tasks to specialized subagents with isolated context. "+
			"Modes: single (agent+task), parallel (tasks array), chain (sequential with {previous} placeholder), "+
			"background (agent+task+background=true, returns immediately and notifies on completion), "+
			"team (agent+task+name; team_name optional, spawns a long-lived teammate that communicates via send_message). "+
			"Available agents: %s",
		strings.Join(names, ", "),
	)
}

func (t *Tool) Schema() map[string]any {
	agentNames := t.runner.sortedAgentNames()
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
		schema.Property("team_name", schema.String("Optional active team name for teammate spawning. Omit when the host provides a default team.")),
		schema.Property("name", schema.String("Teammate name. Setting this selects team mode; must be unique and not 'team-lead'.")),
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
	if p.Model != "" {
		if t.createModel == nil {
			return nil, fmt.Errorf("model override %q is unavailable: no model resolver configured", p.Model)
		}
		m, err := t.createModel(p.Model)
		if err != nil {
			return nil, fmt.Errorf("resolve model override %q: %w", p.Model, err)
		}
		modelOverride = m
	}

	hasChain := len(p.Chain) > 0
	hasParallel := len(p.Tasks) > 0
	hasSingle := p.Agent != "" && p.Task != ""

	// Team-spawn mode: long-lived teammate. Mutually exclusive with the
	// other modes — Background and team fields together are ambiguous, and
	// parallel/chain are conceptually one-shot. Check this BEFORE Background
	// so a user calling with both keys gets the team-mode error path.
	if p.Name != "" || p.TeamName != "" {
		if p.Background || hasChain || hasParallel {
			return nil, fmt.Errorf("team mode is mutually exclusive with background/tasks/chain")
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

// Run executes one registered sub-agent programmatically. Inputs and outputs
// are typed, and failures are Go errors carrying the loop's full chain
// (errors.Is(err, ErrUnknownAgent) for lookup failures,
// agentcore.ErrStopGuard / ErrMaxTurns / provider sentinels for loop
// failures — see agentcore.ErrorKind for the stable taxonomy).
//
// Everything configured on the agent's Config applies exactly as in the
// tool-call path: StopGuard, StopAfterTools, OnMessage, context management,
// prompt-cache keys. Progress reporting via agentcore.WithToolProgress on ctx
// works identically.
func (r *Runner) Run(ctx context.Context, agent, task string) (RunResult, error) {
	return r.run(ctx, agent, task, nil, runOptions{mode: ModeSingle, reportProgress: true})
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
	cfg, ok := t.runner.AgentConfig(p.Agent)
	if !ok {
		return nil, &UnknownAgentError{Agent: p.Agent, Available: t.runner.sortedAgentNames()}
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
	message := fmt.Sprintf("Teammate %q (agent=%s) spawned. Send messages with send_message.", res.AgentID, p.Agent)
	if p.TeamName != "" {
		message = fmt.Sprintf("Teammate %q (agent=%s) spawned in team %q. Send messages with send_message.", res.AgentID, p.Agent, p.TeamName)
	}
	return json.Marshal(map[string]any{
		"task_id":  res.TaskID,
		"agent_id": res.AgentID,
		"status":   "running",
		"message":  message,
	})
}

// executeBackground launches a sub-agent in a detached goroutine and returns
// immediately. When the sub-agent finishes, a notification is sent via
// notifyFn (typically Agent.FollowUp).
func (t *Tool) executeBackground(callerCtx context.Context, agentName, taskStr, description string, modelOverride agentcore.ChatModel) (json.RawMessage, error) {
	if _, ok := t.runner.AgentConfig(agentName); !ok {
		return nil, &UnknownAgentError{Agent: agentName, Available: t.runner.sortedAgentNames()}
	}

	rt := t.taskRT

	// Enforce the explicit recursion/resource boundary before registering work.
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
	// Inherit the caller's working-directory override so a background sub-agent
	// resolves paths in the same workspace (e.g. a git-worktree sandbox) as a
	// foreground one would. No-op when the caller set none.
	bgCtx = tools.WithCwd(bgCtx, tools.CwdFromContext(callerCtx))

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
		defer func() {
			cancel()
			rt.Done(taskID)
		}()

		var outFile io.WriteCloser
		var outputErr error
		if t.bgOutputFactory != nil {
			w, path, ferr := t.bgOutputFactory(taskID, agentName)
			if ferr != nil {
				rt.Update(taskID, func(e *task.Entry) {
					e.Status = task.Failed
					e.Error = fmt.Sprintf("create background output: %v", ferr)
					e.EndedAt = time.Now()
				})
				t.notify(taskID)
				return
			}
			outFile = w
			rt.Update(taskID, func(e *task.Entry) { e.OutputFile = path })
		}

		res, err := t.runner.run(bgCtx, agentName, taskStr, modelOverride, runOptions{
			mode: ModeBackground,
			getSteeringMessages: func() []agentcore.AgentMessage {
				drained := rt.DrainPending(taskID)
				messages := make([]agentcore.AgentMessage, 0, len(drained))
				for _, message := range drained {
					messages = append(messages, agentcore.UserMsg(message))
				}
				return messages
			},
			onEvent: func(ev agentcore.Event) {
				switch ev.Type {
				case agentcore.EventToolExecStart:
					rt.Update(taskID, func(e *task.Entry) { e.ToolCount++ })
					if outFile != nil {
						label := ev.Tool
						if len(ev.Args) > 0 {
							label += "(" + truncate(string(ev.Args), 60) + ")"
						}
						if _, err := fmt.Fprintf(outFile, "[tool] %s\n", label); err != nil && outputErr == nil {
							outputErr = fmt.Errorf("write background output: %w", err)
						}
					}
				case agentcore.EventMessageEnd:
					message, ok := ev.Message.(agentcore.Message)
					if !ok {
						return
					}
					if outFile != nil {
						line, marshalErr := json.Marshal(message)
						if marshalErr != nil && outputErr == nil {
							outputErr = fmt.Errorf("encode background output: %w", marshalErr)
						} else if marshalErr == nil {
							line = append(line, '\n')
							if _, writeErr := outFile.Write(line); writeErr != nil && outputErr == nil {
								outputErr = fmt.Errorf("write background output: %w", writeErr)
							}
						}
					}
					if message.GetRole() == agentcore.RoleAssistant && message.Usage != nil {
						rt.Update(taskID, func(e *task.Entry) {
							e.TokensIn += message.Usage.Input
							e.TokensOut += message.Usage.Output
						})
					}
				}
			},
		})
		if outFile != nil {
			if closeErr := outFile.Close(); closeErr != nil {
				outputErr = errors.Join(outputErr, fmt.Errorf("close background output: %w", closeErr))
			}
		}
		err = errors.Join(err, outputErr)

		rt.Update(taskID, func(e *task.Entry) {
			e.EndedAt = time.Now()
			switch {
			case err != nil && bgCtx.Err() != nil:
				// Cancellation observed: this was an explicit Stop, not a failure.
				e.Status = task.Killed
				if outputErr != nil {
					e.Error = outputErr.Error()
				}
			case err != nil:
				e.Status = task.Failed
				e.Error = err.Error()
			default:
				e.Status = task.Completed
				e.Result = res.displayOutput()
			}
			e.TokensIn = res.Usage.Input
			e.TokensOut = res.Usage.Output
			e.ToolCount = res.Usage.Tools
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
	res, err := t.runner.run(ctx, agentName, taskStr, modelOverride, runOptions{mode: ModeSingle, reportProgress: true})
	if err != nil {
		if res.Usage.Turns > 0 || res.Usage.Tools > 0 {
			return nil, fmt.Errorf("agent %q failed: %w (turns=%d tools=%d)", agentName, err, res.Usage.Turns, res.Usage.Tools)
		}
		return nil, fmt.Errorf("agent %q failed: %w", agentName, err)
	}
	u := res.Usage
	out := map[string]any{
		"output": res.displayOutput(),
		"usage":  &u,
	}
	if len(res.TerminalResult) > 0 {
		out["terminal_result"] = res.TerminalResult
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
		res, err := t.runner.run(ctx, step.Agent, taskStr, modelOverride, runOptions{mode: ModeChain, reportProgress: true})

		u := res.Usage
		r := result{
			Agent:          step.Agent,
			Task:           taskStr,
			Step:           i + 1,
			Usage:          &u,
			TerminalResult: res.TerminalResult,
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

		r.Output = res.displayOutput()
		results = append(results, r)
		previous = r.Output
	}

	return json.Marshal(map[string]any{
		"output":  previous,
		"results": results,
	})
}

// executeParallel runs all requested sub-agents concurrently. Provider and
// transport concurrency policies remain the host's responsibility.
func (t *Tool) executeParallel(ctx context.Context, tasks []taskItem, modelOverride agentcore.ChatModel) (json.RawMessage, error) {
	results := make([]result, len(tasks))
	var wg sync.WaitGroup

	for i, ti := range tasks {
		wg.Add(1)
		go func(idx int, st taskItem) {
			defer wg.Done()

			res, err := t.runner.run(ctx, st.Agent, st.Task, modelOverride, runOptions{mode: ModeParallel, reportProgress: true})
			u := res.Usage
			r := result{
				Agent:          st.Agent,
				Task:           st.Task,
				Usage:          &u,
				TerminalResult: res.TerminalResult,
			}
			if err != nil {
				r.Output = err.Error()
				r.IsError = true
			} else {
				r.Output = res.displayOutput()
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

type runOptions struct {
	mode                string
	reportProgress      bool
	getSteeringMessages func() []agentcore.AgentMessage
	onEvent             func(agentcore.Event)
}

// run executes an isolated agent loop for the given agent config and task.
// On error the returned RunResult carries usage consumed before the failure.
func (r *Runner) run(ctx context.Context, agentName, taskStr string, modelOverride agentcore.ChatModel, opts runOptions) (RunResult, error) {
	cfg, ok := r.agents[agentName]
	if !ok {
		return RunResult{}, &UnknownAgentError{Agent: agentName, Available: r.sortedAgentNames()}
	}

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

	// Mirror NewAgent's auto-wiring: a ContextManager that implements
	// ContextLLMConverter owns the AgentMessage→Message projection (e.g. to
	// render summary entries). Without this, DefaultConvertToLLM silently
	// drops manager-specific message kinds from the LLM request.
	convertToLLM := cfg.ConvertToLLM
	if convertToLLM == nil && contextManager != nil {
		if c, ok := contextManager.(agentcore.ContextLLMConverter); ok {
			convertToLLM = c.ConvertToLLM
		}
	}

	runSeq := r.runSeq.Add(1)

	// One conversation, one cache key: suffix the per-run sequence so each
	// spawn forms its own cache lineage instead of piling every run of this
	// agent into a single routing bucket.
	promptCacheKey := cfg.PromptCacheKey
	if promptCacheKey != "" {
		promptCacheKey = fmt.Sprintf("%s#%d", promptCacheKey, runSeq)
	}

	loopCfg := agentcore.LoopConfig{
		Model:            runModel,
		MaxTurns:         cfg.MaxTurns,
		MaxRetries:       cfg.MaxRetries,
		ContextManager:   contextManager,
		ConvertToLLM:     convertToLLM,
		ThinkingLevel:    r.resolveThinking(agentName, cfg.ThinkingLevel),
		CacheLastMessage: cfg.CacheLastMessage,
		PromptCacheKey:   promptCacheKey,
	}

	loopCfg.GetSteeringMessages = opts.getSteeringMessages
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

	// Fan raw loop events to an external observer (e.g. a live preview),
	// tagged with a unique per-run id. Built once so every event carries the
	// same InstanceID; nil observer means no event fan-out.
	var observe func(agentcore.Event)
	if r.eventObserver != nil {
		meta := RunMeta{
			Agent:      agentName,
			InstanceID: fmt.Sprintf("%s#%d", agentName, runSeq),
			Mode:       opts.mode,
		}
		observe = func(ev agentcore.Event) { r.eventObserver(meta, ev) }
	}

	events := agentcore.AgentLoop(ctx, []agentcore.AgentMessage{agentcore.UserMsg(taskStr)}, agentCtx, loopCfg)

	var lastAssistantContent string
	var terminalToolResult json.RawMessage // result from StopAfterTool trigger
	var lastErr error
	su := &Usage{}

	for ev := range events {
		// Fan to the observer first — before any case below can `continue`
		// past the end of the loop body — so the observer sees the complete
		// raw stream. Publishing is read-only w.r.t. the run's own state, so
		// ordering vs the bookkeeping switch is immaterial. EventAgentEnd is
		// guaranteed on every termination path — normal/abort/max-turns/error
		// and, via the loop's panic recovery, panic too — so an observer can
		// rely on it as the run's stop signal.
		if observe != nil {
			observe(ev)
		}
		if opts.onEvent != nil {
			opts.onEvent(ev)
		}
		switch ev.Type {
		case agentcore.EventToolExecStart:
			su.Tools++
			if opts.reportProgress {
				agentcore.ReportToolProgress(ctx, agentcore.ProgressPayload{
					Kind:    agentcore.ProgressToolStart,
					Agent:   agentName,
					Tool:    ev.Tool,
					Summary: ev.Tool,
					Args:    ev.Args,
				})
			}
		case agentcore.EventMessageUpdate:
			if opts.reportProgress {
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
			if opts.reportProgress {
				if ev.IsError {
					errMsg := string(ev.Result)
					if len(errMsg) > 200 {
						// Back the cut up to a rune boundary: splitting a
						// multi-byte UTF-8 sequence would render mojibake
						// in progress displays.
						cut := 200
						for cut > 0 && !utf8.RuneStart(errMsg[cut]) {
							cut--
						}
						errMsg = errMsg[:cut]
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
			if opts.reportProgress {
				reportContext(ctx, agentName, contextManager)
			}
		case agentcore.EventMessageEnd:
			if ev.Message == nil {
				continue
			}
			if ev.Message.GetRole() == agentcore.RoleAssistant {
				lastAssistantContent = ev.Message.TextContent()
				su.Turns++
				if opts.reportProgress {
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
				}
				if opts.reportProgress {
					reportContext(ctx, agentName, contextManager)
				}
			}
		case agentcore.EventRetry:
			if opts.reportProgress && ev.RetryInfo != nil {
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
		return RunResult{Agent: agentName, Usage: *su}, lastErr
	}
	return RunResult{
		Agent:          agentName,
		Output:         lastAssistantContent,
		TerminalResult: terminalToolResult,
		Usage:          *su,
	}, nil
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

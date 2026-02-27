package agentcore

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"

	"github.com/voocel/agentcore/schema"
)

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
}

// SubAgentTool implements the Tool interface.
// The main agent calls this tool to delegate tasks to specialized sub-agents
// with isolated contexts.
type SubAgentTool struct {
	agents   map[string]SubAgentConfig
	notifyFn func(AgentMessage) // called when a background task completes
	mu       sync.Mutex
	bgSeq    int
}

// NewSubAgentTool creates a subagent tool from a set of agent configs.
func NewSubAgentTool(agents ...SubAgentConfig) *SubAgentTool {
	m := make(map[string]SubAgentConfig, len(agents))
	for _, a := range agents {
		m[a.Name] = a
	}
	return &SubAgentTool{agents: m}
}

// SetNotifyFn sets the callback invoked when a background task completes.
// Typically bound to Agent.FollowUp so the main agent receives the result
// as a follow-up message.
func (t *SubAgentTool) SetNotifyFn(fn func(AgentMessage)) {
	t.notifyFn = fn
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
	)
}

func (t *SubAgentTool) Execute(ctx context.Context, args json.RawMessage) (json.RawMessage, error) {
	var params subagentParams
	if err := json.Unmarshal(args, &params); err != nil {
		return nil, fmt.Errorf("invalid subagent params: %w", err)
	}

	hasChain := len(params.Chain) > 0
	hasParallel := len(params.Tasks) > 0
	hasSingle := params.Agent != "" && params.Task != ""

	// Background mode: single task running in a detached goroutine.
	if params.Background {
		if !hasSingle {
			return json.Marshal("background mode requires agent + task")
		}
		return t.executeBackground(params.Agent, params.Task, params.Description)
	}

	modeCount := boolToInt(hasChain) + boolToInt(hasParallel) + boolToInt(hasSingle)
	if modeCount != 1 {
		return json.Marshal("Invalid parameters: provide exactly one mode (agent+task, tasks, or chain)")
	}

	switch {
	case hasChain:
		return t.executeChain(ctx, params.Chain)
	case hasParallel:
		return t.executeParallel(ctx, params.Tasks)
	default:
		return t.executeSingle(ctx, params.Agent, params.Task)
	}
}

// executeBackground launches a sub-agent in a detached goroutine and returns immediately.
// When the sub-agent finishes, a notification is sent via notifyFn (typically Agent.FollowUp).
func (t *SubAgentTool) executeBackground(agentName, task, description string) (json.RawMessage, error) {
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

	go func() {
		// Detached context — background task is independent of the parent request.
		bgCtx := context.Background()
		output, usage, err := t.runAgent(bgCtx, agentName, task)

		var result map[string]any
		if err != nil {
			result = map[string]any{
				"task_id":     taskID,
				"agent":       agentName,
				"description": description,
				"status":      "failed",
				"error":       err.Error(),
				"usage":       usage,
			}
		} else {
			result = map[string]any{
				"task_id":     taskID,
				"agent":       agentName,
				"description": description,
				"status":      "completed",
				"output":      output,
				"usage":       usage,
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
func (t *SubAgentTool) executeSingle(ctx context.Context, agentName, task string) (json.RawMessage, error) {
	output, usage, err := t.runAgent(ctx, agentName, task)
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
func (t *SubAgentTool) executeChain(ctx context.Context, chain []subagentChain) (json.RawMessage, error) {
	var previous string
	results := make([]subagentResult, 0, len(chain))

	for i, step := range chain {
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}

		task := strings.ReplaceAll(step.Task, "{previous}", previous)
		output, usage, err := t.runAgent(ctx, step.Agent, task)

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
func (t *SubAgentTool) executeParallel(ctx context.Context, tasks []subagentTask) (json.RawMessage, error) {
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

			output, usage, err := t.runAgent(ctx, st.Agent, st.Task)
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

// runAgent executes an isolated agent loop for the given agent config and task.
// Includes panic recovery to prevent a subagent crash from taking down the parent.
func (t *SubAgentTool) runAgent(ctx context.Context, agentName, task string) (output string, usage *subagentUsage, err error) {
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
		Model:    cfg.Model,
		StreamFn: cfg.StreamFn,
		MaxTurns: cfg.MaxTurns,
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
		case EventMessageEnd:
			if ev.Message == nil {
				continue
			}
			if ev.Message.GetRole() == RoleAssistant {
				lastAssistantContent = ev.Message.TextContent()
				su.Turns++
				// Accumulate usage from assistant messages
				if msg, ok := ev.Message.(Message); ok && msg.Usage != nil {
					su.Input += msg.Usage.Input
					su.Output += msg.Usage.Output
					su.CacheRead += msg.Usage.CacheRead
					su.CacheWrite += msg.Usage.CacheWrite
					if msg.Usage.Cost != nil {
						su.Cost += msg.Usage.Cost.Total
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

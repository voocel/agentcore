// Package subagent lets a model delegate tasks to sub-agents, which work on
// them in contexts of their own and report back. A call of its tool runs one
// agent, several in parallel, a chain of them, or one in the background.
package subagent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync"
	"sync/atomic"

	"github.com/voocel/agentcore"
	"github.com/voocel/agentcore/schema"
	"github.com/voocel/agentcore/task"
)

// MaxDepth caps how deep sub-agents nest: the main agent is at depth 0, an
// agent it spawns at 1. A host that gives sub-agents the subagent tool
// relies on it to stop runaway recursion.
const MaxDepth = 5

// maxParallel bounds the runs of a parallel call that work at once.
const maxParallel = 8

// depthKey carries the depth of the agent a context runs.
type depthKey struct{}

func depthOf(ctx context.Context) int {
	d, _ := ctx.Value(depthKey{}).(int)
	return d
}

// Agent is a sub-agent the model may delegate to.
type Agent struct {
	Name        string
	Description string
	// Config configures a run of the agent; an error fails the run. Each run
	// gets a configuration of its own, so that its tools keep state of their
	// own. The run's events go to its Emit, and to the tool call waiting for
	// it as Progress. The runs of a parallel call work at once: the Emits
	// of their Configs are called concurrently.
	Config func(Spawn) (agentcore.Config, error)
}

// Spawn is a run of an agent.
type Spawn struct {
	Agent string
	// ID tells the run from the tool's other runs, as "explore#3".
	ID   string
	Mode Mode
	// Model is the model the call asked for, or "" for the agent's own.
	// Config honors it or refuses it.
	Model string
}

// Mode is how a call runs its agents.
type Mode string

const (
	ModeSingle     Mode = "single"
	ModeParallel   Mode = "parallel"
	ModeChain      Mode = "chain"
	ModeBackground Mode = "background"
)

// Progress is the progress the tool reports: an event of a run the call
// waits for. A background run reports none.
type Progress struct {
	Spawn Spawn
	Event agentcore.Event
}

type params struct {
	Agent       string `json:"agent"`
	Task        string `json:"task"`
	Tasks       []step `json:"tasks"`
	Chain       []step `json:"chain"`
	Background  bool   `json:"background"`
	Description string `json:"description"`
	Model       string `json:"model"`
}

type step struct {
	Agent string `json:"agent"`
	Task  string `json:"task"`
}

type delegator struct {
	agents map[string]Agent
	names  []string
	tasks  *task.Runtime
	seq    atomic.Int64
}

// New returns the subagent tool, which delegates to agents. With tasks, the
// model may also run an agent in the background, as a task of tasks.
func New(tasks *task.Runtime, agents ...Agent) agentcore.Tool {
	d := &delegator{agents: map[string]Agent{}, tasks: tasks}
	var listed []string
	for _, a := range agents {
		if a.Name == "" {
			panic("subagent: agent name is required")
		}
		if _, ok := d.agents[a.Name]; ok {
			panic(fmt.Sprintf("subagent: duplicate agent %q", a.Name))
		}
		d.agents[a.Name] = a
		d.names = append(d.names, a.Name)
		listed = append(listed, fmt.Sprintf("%s (%s)", a.Name, a.Description))
	}

	modes := "single (agent and task), parallel (tasks), chain (chain, run in order, with {previous} in a task standing for the output of the step before)"
	item := schema.Object(
		schema.Property("agent", schema.Enum("Agent name", d.names...)).Required(),
		schema.Property("task", schema.String("Task description")).Required(),
	)
	props := []schema.Prop{
		schema.Property("agent", schema.Enum("Name of the agent to run the task", d.names...)),
		schema.Property("task", schema.String("Task to delegate to the agent")),
		schema.Property("tasks", schema.Array("Array of {agent, task} to run in parallel", item)),
		schema.Property("chain", schema.Array("Array of {agent, task} to run in order. Use {previous} in a task to reference the prior output.", item)),
	}
	if tasks != nil {
		modes += ", background (agent, task and background; returns at once, and a notification follows when the agent is done)"
		props = append(props,
			schema.Property("background", schema.Bool("Set true to run the agent in the background.")),
			schema.Property("description", schema.String("Short description of the background task, shown in notifications and listings.")),
		)
	}
	props = append(props, schema.Property("model", schema.String("Model to run the agents with, as a model ID or alias. Defaults to each agent's own.")))

	tool := agentcore.NewTool("subagent",
		fmt.Sprintf("Delegate tasks to specialized sub-agents, each working in a context of its own. Modes: %s. Agents: %s", modes, strings.Join(listed, ", ")),
		schema.Object(props...),
		d.run,
	)
	tool.Label = "Delegate to SubAgent"
	return tool
}

func (d *delegator) run(ctx context.Context, p params) (agentcore.Result, error) {
	single := p.Agent != "" && p.Task != ""
	modes := 0
	for _, set := range []bool{single, len(p.Tasks) > 0, len(p.Chain) > 0} {
		if set {
			modes++
		}
	}
	switch {
	case p.Background && !single:
		return agentcore.Result{}, errors.New("background mode requires agent and task")
	case p.Background && d.tasks == nil:
		return agentcore.Result{}, errors.New("background mode is not available")
	case p.Background:
		return d.background(ctx, p)
	case modes != 1:
		return agentcore.Result{}, errors.New("provide exactly one mode: agent and task, tasks, or chain")
	case single:
		output, err := d.foreground(ctx, p.Agent, p.Task, p.Model, ModeSingle)
		if err != nil {
			return agentcore.ErrorResult(strings.TrimSpace(fmt.Sprintf("Agent %q failed: %v\n\n%s", p.Agent, err, output))), nil
		}
		return agentcore.TextResult(orNone(output)), nil
	case len(p.Tasks) > 0:
		return d.parallel(ctx, p), nil
	default:
		return d.chain(ctx, p), nil
	}
}

// parallel runs its tasks at once, up to maxParallel of them; their runs
// are numbered in task order.
func (d *delegator) parallel(ctx context.Context, p params) agentcore.Result {
	outputs := make([]string, len(p.Tasks))
	errs := make([]error, len(p.Tasks))
	runs := make([]*run, len(p.Tasks))
	for i, s := range p.Tasks {
		runs[i], errs[i] = d.prepare(ctx, s.Agent, p.Model, ModeParallel)
	}
	var wg sync.WaitGroup
	sem := make(chan struct{}, maxParallel)
	for i, r := range runs {
		if r == nil {
			continue
		}
		wg.Go(func() {
			sem <- struct{}{}
			defer func() { <-sem }()
			outputs[i], errs[i] = d.watch(ctx, r, p.Tasks[i].Task)
		})
	}
	wg.Wait()

	var b strings.Builder
	failed := 0
	for i, s := range p.Tasks {
		writeResult(&b, i+1, s.Agent, outputs[i], errs[i])
		if errs[i] != nil {
			failed++
		}
	}
	return agentcore.TextResult(fmt.Sprintf("%d/%d succeeded\n%s", len(p.Tasks)-failed, len(p.Tasks), &b))
}

// chain runs its steps in order, each with the output of the one before; a
// failed step stops it.
func (d *delegator) chain(ctx context.Context, p params) agentcore.Result {
	var b strings.Builder
	previous := ""
	for i, s := range p.Chain {
		output, err := d.foreground(ctx, s.Agent, strings.ReplaceAll(s.Task, "{previous}", previous), p.Model, ModeChain)
		writeResult(&b, i+1, s.Agent, output, err)
		if err != nil {
			return agentcore.ErrorResult(fmt.Sprintf("The chain stopped at step %d.\n%s", i+1, &b))
		}
		previous = output
	}
	return agentcore.TextResult(b.String())
}

// writeResult writes what a run of agent output; one that failed shows its
// error over what it last said.
func writeResult(b *strings.Builder, step int, agent, output string, err error) {
	status := "completed"
	if err != nil {
		status = "failed"
		output = strings.TrimSpace(err.Error() + "\n\n" + output)
	}
	fmt.Fprintf(b, "<result step=\"%d\" agent=\"%s\" status=\"%s\">\n%s\n</result>\n", step, agent, status, orNone(output))
}

func orNone(output string) string {
	if output == "" {
		return "(no output)"
	}
	return output
}

// foreground runs agent on prompt within the call ctx runs.
func (d *delegator) foreground(ctx context.Context, agent, prompt, model string, mode Mode) (string, error) {
	r, err := d.prepare(ctx, agent, model, mode)
	if err != nil {
		return "", err
	}
	return d.watch(ctx, r, prompt)
}

// watch runs r on prompt within the call ctx runs, reporting the run's
// events as the call's progress.
func (d *delegator) watch(ctx context.Context, r *run, prompt string) (string, error) {
	return r.execute(ctx, prompt, func(ev agentcore.Event) error {
		agentcore.ReportProgress(ctx, Progress{Spawn: r.Spawn, Event: ev})
		return nil
	})
}

// background starts agent on the task as a task of d.tasks and returns at
// once. The run outlives the call: it ends with its task.
func (d *delegator) background(ctx context.Context, p params) (agentcore.Result, error) {
	r, err := d.prepare(ctx, p.Agent, p.Model, ModeBackground)
	if err != nil {
		return agentcore.Result{}, err
	}
	description := p.Description
	if description == "" {
		description = truncate(p.Task, 80)
	}
	e, err := d.tasks.Start(ctx, task.Entry{Type: task.TypeSubAgent, Agent: p.Agent, Run: r.ID, Prompt: p.Task, Description: description},
		func(ctx context.Context, t *task.Task) error {
			output, err := runTask(ctx, t, r, p.Task)
			t.Update(func(e *task.Entry) { e.Result = output })
			return err
		})
	if err != nil {
		return agentcore.Result{}, err
	}
	return agentcore.TextResult(fmt.Sprintf("Started task %s: agent %q works in the background, and a notification follows when it ends.", e.ID, p.Agent)), nil
}

// runTask runs r on prompt as the task t, counting its tool calls and
// tokens and writing its messages, as JSON lines, to the task's output.
func runTask(ctx context.Context, t *task.Task, r *run, prompt string) (string, error) {
	enc := json.NewEncoder(t.Output)
	return r.execute(ctx, prompt, func(ev agentcore.Event) error {
		switch ev := ev.(type) {
		case agentcore.ToolStart:
			t.Update(func(e *task.Entry) { e.ToolCount++ })
		case agentcore.MessageEnd:
			if u := ev.Message.Usage; u != nil {
				t.Update(func(e *task.Entry) { e.TokensIn += u.InputTokens; e.TokensOut += u.OutputTokens })
			}
			if err := enc.Encode(ev.Message); err != nil {
				return fmt.Errorf("write the transcript: %w", err)
			}
		}
		return nil
	})
}

// run is a run of an agent, configured and ready.
type run struct {
	Spawn
	cfg   agentcore.Config
	depth int
}

// prepare configures a run of agent, which nests one deeper than the call
// ctx runs, up to MaxDepth.
func (d *delegator) prepare(ctx context.Context, agent, model string, mode Mode) (*run, error) {
	a, ok := d.agents[agent]
	if !ok {
		return nil, fmt.Errorf("unknown agent %q, available: %s", agent, strings.Join(d.names, ", "))
	}
	depth := depthOf(ctx) + 1
	if depth > MaxDepth {
		return nil, fmt.Errorf("agent nesting depth %d exceeds the maximum of %d", depth, MaxDepth)
	}
	s := Spawn{Agent: agent, ID: fmt.Sprintf("%s#%d", agent, d.seq.Add(1)), Mode: mode, Model: model}
	cfg, err := a.Config(s)
	if err != nil {
		return nil, err
	}
	return &run{Spawn: s, cfg: cfg, depth: depth}, nil
}

// execute runs r on prompt and returns its last response's text, also when
// the run failed. The run's events go to its Emit, then to observe; an
// error from either stops it.
func (r *run) execute(ctx context.Context, prompt string, observe func(agentcore.Event) error) (string, error) {
	cfg := r.cfg
	emit := cfg.Emit
	cfg.Emit = func(ev agentcore.Event) error {
		if emit != nil {
			if err := emit(ev); err != nil {
				return err
			}
		}
		return observe(ev)
	}
	history, err := agentcore.Run(context.WithValue(ctx, depthKey{}, r.depth), cfg, nil, agentcore.UserText(prompt))
	return agentcore.LastResponse(history).Text(), err
}

// truncate shortens s to n runes, marking the cut with "...".
func truncate(s string, n int) string {
	runes := []rune(s)
	if len(runes) <= n {
		return s
	}
	return string(runes[:n]) + "..."
}

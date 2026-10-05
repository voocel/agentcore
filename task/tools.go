package task

import (
	"context"
	"fmt"
	"io"
	"os"
	"strings"
	"time"

	"github.com/voocel/agentcore"
	"github.com/voocel/agentcore/schema"
)

const (
	// waitTimeout is how long task_output waits by default.
	waitTimeout = 2 * time.Minute
	// tailBytes is how much of a shell task's output task_output shows.
	tailBytes = 32 * 1024
)

// Tools returns the tools that let the model follow the tasks of r:
// task_output, which reports a task, once it ends if asked to wait, and
// task_stop, which stops one.
func (r *Runtime) Tools() []agentcore.Tool {
	output := agentcore.NewTool("task_output",
		"Report a background task: its status and, for a shell command, the tail of its output; for an agent, its result. Set wait to wait for the task to end first.",
		schema.Object(
			schema.Property("task_id", schema.String("The ID of the task")).Required(),
			schema.Property("wait", schema.Bool("Wait for the task to end before reporting it")),
			schema.Property("timeout", schema.Int(fmt.Sprintf("Seconds to wait at most (default: %d)", int(waitTimeout.Seconds())))),
		),
		r.output,
	)
	output.Label = "Task Output"
	output.Parallel = true

	stop := agentcore.NewTool("task_stop",
		"Stop a running background task.",
		schema.Object(schema.Property("task_id", schema.String("The ID of the task")).Required()),
		func(_ context.Context, a struct {
			TaskID string `json:"task_id"`
		}) (agentcore.Result, error) {
			if !r.Stop(a.TaskID) {
				return agentcore.Result{}, fmt.Errorf("no running task %s", a.TaskID)
			}
			return agentcore.TextResult(fmt.Sprintf("Stopped task %s.", a.TaskID)), nil
		},
	)
	stop.Label = "Stop Task"
	return []agentcore.Tool{output, stop}
}

type outputArgs struct {
	TaskID  string `json:"task_id"`
	Wait    bool   `json:"wait"`
	Timeout int    `json:"timeout"`
}

func (r *Runtime) output(ctx context.Context, a outputArgs) (agentcore.Result, error) {
	e, ok := r.Get(a.TaskID)
	if !ok {
		return agentcore.Result{}, fmt.Errorf("task %s not found", a.TaskID)
	}
	if a.Wait && e.Status == Running {
		timeout := waitTimeout
		if a.Timeout > 0 {
			timeout = time.Duration(a.Timeout) * time.Second
		}
		wctx, cancel := context.WithTimeout(ctx, timeout)
		ended, err := r.WaitFor(wctx, a.TaskID)
		cancel()
		switch {
		case err == nil:
			e = ended
		case ctx.Err() != nil:
			return agentcore.Result{}, ctx.Err()
		default:
			e, _ = r.Get(a.TaskID)
		}
	}
	return agentcore.TextResult(report(e)), nil
}

// report describes the task e for the model.
func report(e Entry) string {
	var b strings.Builder
	fmt.Fprintf(&b, "Task %s (%s): %s", e.ID, e.Type, e.Status)
	if e.Status == Running {
		fmt.Fprintf(&b, " for %s", time.Since(e.StartedAt).Round(time.Second))
	} else {
		fmt.Fprintf(&b, " after %s", e.EndedAt.Sub(e.StartedAt).Round(time.Second))
	}
	if e.Error != "" {
		fmt.Fprintf(&b, ": %s", e.Error)
	}
	b.WriteString("\n")
	if e.Description != "" {
		fmt.Fprintf(&b, "Description: %s\n", e.Description)
	}
	switch e.Type {
	case TypeShell:
		fmt.Fprintf(&b, "Command: %s\nOutput file: %s\n", e.Command, e.OutputFile)
		if tail := readTail(e.OutputFile, tailBytes); tail != "" {
			fmt.Fprintf(&b, "Output:\n%s\n", tail)
		}
	case TypeSubAgent:
		fmt.Fprintf(&b, "Agent: %s\nTool uses: %d, tokens: %d\nTranscript: %s\n", e.Agent, e.ToolCount, e.TokensIn+e.TokensOut, e.OutputFile)
		if e.Result != "" {
			fmt.Fprintf(&b, "Result:\n%s\n", e.Result)
		}
	}
	return strings.TrimRight(b.String(), "\n")
}

// readTail returns the last n bytes of the file at path, from a line start,
// marking a cut.
func readTail(path string, n int64) string {
	f, err := os.Open(path)
	if err != nil {
		return ""
	}
	defer f.Close()
	info, err := f.Stat()
	if err != nil {
		return ""
	}
	start := max(info.Size()-n, 0)
	data, err := io.ReadAll(io.NewSectionReader(f, start, info.Size()-start))
	if err != nil {
		return ""
	}
	text := string(data)
	if start > 0 {
		if _, rest, ok := strings.Cut(text, "\n"); ok {
			text = "[earlier output cut]\n" + rest
		}
	}
	return strings.TrimRight(text, "\n")
}

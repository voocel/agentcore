package tools

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strings"
	"time"

	"github.com/voocel/agentcore"
	"github.com/voocel/agentcore/schema"
	"github.com/voocel/agentcore/task"
)

// bashTimeout is how long a command runs in the foreground by default.
const bashTimeout = 2 * time.Minute

// Bash returns the bash tool, which runs shell commands. It reports each
// line of their output as a string progress, and returns the tail of it, at
// most 2000 lines or 50KB, with a line for each of a non-zero exit code, a
// timeout and a cut. A command that fails is not a failed call: its output
// is what the model needs. With Tasks, the model may run a command in the
// background, as a task.
func (w Workspace) Bash() agentcore.Tool {
	t := &bashTool{w: w}
	props := []schema.Prop{
		schema.Property("command", schema.String("Shell command to execute. Quote paths with spaces.")).Required(),
		schema.Property("timeout", schema.Int(fmt.Sprintf("Timeout in seconds (default: %d; none in the background)", int(bashTimeout.Seconds())))),
		schema.Property("workdir", schema.String("Optional working directory for this command. Use this instead of 'cd && ...'.")),
		schema.Property("description", schema.String("Short 5-10 word description of what the command does")),
	}
	description := fmt.Sprintf("Execute a shell command in the workspace. Prefer read, edit, write, glob, grep, and ls for file operations. "+
		"Use workdir instead of 'cd && ...' when a command must run in another directory. "+
		"Output is truncated to the last %d lines or %s (whichever is hit first).",
		defaultMaxLines, formatSize(defaultMaxBytes))
	if w.Tasks != nil {
		props = append(props, schema.Property("run_in_background", schema.Bool("Run the command in the background as a task. Returns at once; a notification follows when it ends.")))
		description += " Set run_in_background for long-running commands, such as servers or watchers."
	}
	return agentcore.Tool{
		Name:        "bash",
		Label:       "Execute Command",
		Description: description,
		Schema:      schema.Object(props...),
		Run:         textRun(t.execute),
	}
}

type bashTool struct {
	w Workspace
}

type bashArgs struct {
	Command         string `json:"command"`
	Timeout         int    `json:"timeout"`
	WorkDir         string `json:"workdir"`
	Description     string `json:"description"`
	RunInBackground bool   `json:"run_in_background"`
}

func (t *bashTool) execute(ctx context.Context, args json.RawMessage) (string, error) {
	var a bashArgs
	if err := json.Unmarshal(args, &a); err != nil {
		return "", fmt.Errorf("invalid args: %w", err)
	}
	dir, err := t.resolveWorkDir(ctx, a)
	if err != nil {
		return "", err
	}
	shell, shellArgs, err := resolveShell()
	if err != nil {
		return "", err
	}
	argv := append(shellArgs, a.Command)
	timeout := time.Duration(a.Timeout) * time.Second
	if a.RunInBackground {
		if t.w.Tasks == nil {
			return "", errors.New("background mode is not available")
		}
		return t.background(ctx, a, shell, argv, dir, timeout)
	}
	if timeout <= 0 {
		timeout = bashTimeout
	}
	return foreground(ctx, shell, argv, dir, timeout)
}

func (t *bashTool) resolveWorkDir(ctx context.Context, a bashArgs) (string, error) {
	dir := ResolvePath(t.w.dir(ctx), a.WorkDir)
	if dir == "" {
		return "", nil
	}
	info, err := os.Stat(dir)
	if err != nil {
		if os.IsNotExist(err) {
			return "", fmt.Errorf("working directory does not exist: %s", dir)
		}
		return "", fmt.Errorf("check working directory %s: %w", dir, err)
	}
	if !info.IsDir() {
		return "", fmt.Errorf("working directory is not a directory: %s", dir)
	}
	return dir, nil
}

// background starts the command as a task, its output going to the task's
// output file, and returns at once. A timeout of 0 lets it run until it
// ends or is stopped.
func (t *bashTool) background(ctx context.Context, a bashArgs, shell string, argv []string, dir string, timeout time.Duration) (string, error) {
	description := a.Description
	if description == "" {
		description = truncate(a.Command, 60)
	}
	e, err := t.w.Tasks.Start(ctx, task.Entry{Type: task.TypeShell, Command: a.Command, Description: description},
		func(ctx context.Context, tk *task.Task) error {
			if timeout > 0 {
				var cancel context.CancelFunc
				ctx, cancel = context.WithTimeout(ctx, timeout)
				defer cancel()
			}
			cmd := command(ctx, shell, argv, dir)
			// The output file, handed to the command as it is, so that a
			// process the command leaves running cannot hold up Wait.
			cmd.Stdout, cmd.Stderr = tk.Output, tk.Output
			if err := cmd.Start(); err != nil {
				return fmt.Errorf("start command: %w", err)
			}
			err := cmd.Wait()
			if exitErr, ok := errors.AsType[*exec.ExitError](err); ok {
				tk.Update(func(e *task.Entry) { e.ExitCode = exitErr.ExitCode() })
			}
			switch {
			case errors.Is(ctx.Err(), context.DeadlineExceeded):
				return fmt.Errorf("timed out after %s", timeout)
			case err != nil && cmd.ProcessState != nil:
				return fmt.Errorf("exit code %d", cmd.ProcessState.ExitCode())
			}
			return err
		})
	if err != nil {
		return "", err
	}
	return fmt.Sprintf("Started task %s: the command runs in the background, its output going to %s. A notification follows when it ends.", e.ID, e.OutputFile), nil
}

// foreground runs the command and returns its output, reporting each line
// as the call's progress.
func foreground(ctx context.Context, shell string, argv []string, dir string, timeout time.Duration) (string, error) {
	runCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	cmd := command(runCtx, shell, argv, dir)
	pr, pw, err := os.Pipe()
	if err != nil {
		return "", fmt.Errorf("create pipe: %w", err)
	}
	cmd.Stdout, cmd.Stderr = pw, pw
	if err := cmd.Start(); err != nil {
		pr.Close()
		pw.Close()
		return "", fmt.Errorf("start command: %w", err)
	}
	pw.Close()

	out := &output{}
	defer out.close()
	var readErr error
	done := make(chan struct{})
	go func() {
		defer close(done)
		readErr = readLines(pr, out, func(line string) { agentcore.ReportProgress(ctx, line) })
	}()
	waitErr := cmd.Wait()
	// A process the command left running may hold the pipe open: its output
	// after the command ended is not waited for long.
	drained := true
	select {
	case <-done:
	case <-time.After(500 * time.Millisecond):
		drained = false
	}
	pr.Close()
	<-done

	if err := ctx.Err(); err != nil {
		return "", err
	}
	if readErr != nil {
		return "", fmt.Errorf("read command output: %w", readErr)
	}
	timedOut := errors.Is(runCtx.Err(), context.DeadlineExceeded)
	if _, exited := errors.AsType[*exec.ExitError](waitErr); waitErr != nil && !exited && !timedOut {
		return "", fmt.Errorf("command failed: %w", waitErr)
	}

	text, cut := out.result()
	var notes []string
	if cut != "" {
		notes = append(notes, cut)
	}
	if !drained {
		notes = append(notes, "[Output written after the command exited, by processes it left running, is not shown.]")
	}
	if code := cmd.ProcessState.ExitCode(); code != 0 && !timedOut {
		notes = append(notes, fmt.Sprintf("[exit code %d]", code))
	}
	if timedOut {
		notes = append(notes, fmt.Sprintf("[timed out after %s]", timeout))
	}
	if len(notes) > 0 {
		text += "\n\n" + strings.Join(notes, "\n")
	}
	return text, nil
}

// readLines copies r to w, calling line with each line as it comes. A line
// longer than defaultMaxBytes comes in pieces of about that size, so that
// output without newlines is not held in memory whole.
func readLines(r io.Reader, w io.Writer, line func(string)) error {
	var pending []byte
	buf := make([]byte, 32*1024)
	for {
		n, err := r.Read(buf)
		w.Write(buf[:n])
		pending = append(pending, buf[:n]...)
		for {
			i := bytes.IndexByte(pending, '\n')
			if i < 0 {
				break
			}
			line(string(pending[:i]))
			pending = pending[i+1:]
		}
		if len(pending) >= defaultMaxBytes {
			line(string(pending))
			pending = nil
		}
		if err != nil {
			if len(pending) > 0 {
				line(string(pending))
			}
			if errors.Is(err, io.EOF) || errors.Is(err, os.ErrClosed) {
				return nil
			}
			return err
		}
	}
}

// outputMemory bounds what output a command keeps in memory.
const outputMemory = 4 * defaultMaxBytes

// output collects a command's output: its tail in memory and, once that is
// trimmed, the whole of it in a file the model can read.
type output struct {
	tail    []byte // the end of the output: all of it until trimmed
	trimmed bool
	lines   int      // newlines in the whole output
	file    *os.File // the whole output, if saved
	failed  bool     // saving it failed
	shown   bool     // the result names the file
}

func (o *output) Write(p []byte) (int, error) {
	o.lines += bytes.Count(p, []byte{'\n'})
	o.tail = append(o.tail, p...)
	switch {
	case o.trimmed:
		o.save(p)
	case len(o.tail) > 2*outputMemory:
		o.trimmed = true
		o.save(o.tail)
	}
	if o.trimmed && len(o.tail) > 2*outputMemory {
		o.tail = append([]byte(nil), o.tail[len(o.tail)-outputMemory:]...)
	}
	return len(p), nil
}

// save appends p to the file of the whole output, created on first use. A
// failure leaves the output without one.
func (o *output) save(p []byte) {
	if o.failed {
		return
	}
	if o.file == nil {
		f, err := os.CreateTemp("", "agentcore-bash-*.log")
		if err != nil {
			o.failed = true
			return
		}
		o.file = f
	}
	if _, err := o.file.Write(p); err != nil {
		o.failed = true
	}
}

// result returns the tail of the output the model reads and, when that is
// cut, a line saying so, naming the file that holds the whole output.
func (o *output) result() (text, cut string) {
	tail := string(o.tail)
	if o.trimmed {
		// The tail starts within a line.
		if i := strings.IndexByte(tail, '\n'); i >= 0 {
			tail = tail[i+1:]
		}
	}
	tail = strings.TrimSuffix(tail, "\n")
	if tail == "" && !o.trimmed {
		return "(no output)", ""
	}
	tr := truncateTail(tail, defaultMaxLines, defaultMaxBytes)
	if !tr.Truncated && !o.trimmed {
		return tail, ""
	}
	total := o.lines
	if !bytes.HasSuffix(o.tail, []byte{'\n'}) {
		total++
	}
	if !o.trimmed {
		o.save(o.tail)
	}
	if o.file == nil || o.failed || o.file.Close() != nil {
		o.failed = true
		return tr.Content, fmt.Sprintf("[Showing the last %d of %d lines.]", tr.OutputLines, total)
	}
	o.shown = true
	return tr.Content, fmt.Sprintf("[Showing the last %d of %d lines. Full output: %s]", tr.OutputLines, total, o.file.Name())
}

// close removes the file of the whole output unless the result named it.
func (o *output) close() {
	if o.file == nil || o.shown {
		return
	}
	o.file.Close()
	os.Remove(o.file.Name())
}

// command is the shell running argv in dir, killed with its process group
// once ctx ends.
func command(ctx context.Context, shell string, argv []string, dir string) *exec.Cmd {
	cmd := exec.CommandContext(ctx, shell, argv...)
	cmd.Dir = dir
	configureProcGroup(cmd)
	return cmd
}

func resolveShell() (string, []string, error) {
	if p, err := exec.LookPath("bash"); err == nil {
		return p, []string{"-c"}, nil
	}
	if p, err := exec.LookPath("sh"); err == nil {
		return p, []string{"-c"}, nil
	}
	return "", nil, fmt.Errorf("no shell found: tried bash and sh")
}

// truncate shortens s to n runes, marking the cut with "...".
func truncate(s string, n int) string {
	runes := []rune(s)
	if len(runes) <= n {
		return s
	}
	return string(runes[:n]) + "..."
}

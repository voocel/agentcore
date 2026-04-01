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
	"time"

	"github.com/voocel/agentcore"
	"github.com/voocel/agentcore/schema"
)

// BashTool executes shell commands.
// Streams stdout+stderr via ReportToolProgress for real-time display.
// Final result applies tail truncation (2000 lines / 50KB).
// Supports run_in_background mode for long-running commands.
type BashTool struct {
	WorkDir         string
	Timeout         time.Duration // default: 2 minutes
	notifyFn        func(agentcore.AgentMessage)
	bgOutputFactory func(shellID string) (io.WriteCloser, string, error) // creates output writer for background shells
	taskRT          *agentcore.TaskRuntime                               // shared background task registry
}

func NewBash(workDir string) *BashTool {
	return &BashTool{
		WorkDir: workDir,
		Timeout: 2 * time.Minute,
	}
}

// SetNotifyFn sets the callback invoked when a background shell completes.
// Typically bound to Agent.FollowUp so the main agent receives the result.
func (t *BashTool) SetNotifyFn(fn func(agentcore.AgentMessage)) {
	t.notifyFn = fn
}

// SetBgOutputFactory sets the factory that creates output writers for background shells.
// The factory receives the shell ID and returns a writer, file path, and error.
// If not set, background output is discarded.
func (t *BashTool) SetBgOutputFactory(fn func(shellID string) (io.WriteCloser, string, error)) {
	t.bgOutputFactory = fn
}

// SetTaskRuntime sets the shared task runtime for background task registration.
func (t *BashTool) SetTaskRuntime(rt *agentcore.TaskRuntime) {
	t.taskRT = rt
}

func (t *BashTool) Name() string  { return "bash" }
func (t *BashTool) Label() string { return "Execute Command" }

// ReadOnly reports false — bash commands may have side effects.
func (t *BashTool) ReadOnly(_ json.RawMessage) bool { return false }

// ConcurrencySafe reports false — bash commands are not safe for concurrent execution.
func (t *BashTool) ConcurrencySafe(_ json.RawMessage) bool { return false }

// ActivityDescription returns a short description including the command.
func (t *BashTool) ActivityDescription(args json.RawMessage) string {
	var a struct{ Command string `json:"command"` }
	if json.Unmarshal(args, &a) == nil && a.Command != "" {
		return "Running: " + bashTruncate(a.Command, 40)
	}
	return "Running command"
}
func (t *BashTool) Description() string {
	return fmt.Sprintf(
		"Execute a shell command in the workspace. Prefer read, edit, write, find, grep, and ls for file operations. "+
			"Use workdir instead of 'cd && ...' when a command must run in another directory. "+
			"Output is truncated to the last %d lines or %s (whichever is hit first). "+
			"Set run_in_background=true for long-running commands; it returns immediately and notifies on completion.",
		defaultMaxLines, formatSize(defaultMaxBytes),
	)
}
func (t *BashTool) Schema() map[string]any {
	return schema.Object(
		schema.Property("command", schema.String("Shell command to execute. Quote paths with spaces.")).Required(),
		schema.Property("timeout", schema.Int("Timeout in seconds (default: 120)")),
		schema.Property("workdir", schema.String("Optional working directory for this command. Use this instead of 'cd && ...'.")),
		schema.Property("description", schema.String("Short 5-10 word description shown in the task list")),
		schema.Property("run_in_background", schema.Bool("Run command in background. Returns immediately; a notification is sent when the command completes.")),
	)
}

type bashArgs struct {
	Command         string `json:"command"`
	Timeout         int    `json:"timeout"`
	WorkDir         string `json:"workdir"`
	Description     string `json:"description"`
	RunInBackground bool   `json:"run_in_background"`
}

type bashForegroundResult struct {
	Output     string `json:"output"`
	ExitCode   int    `json:"exit_code"`
	TimedOut   bool   `json:"timed_out,omitempty"`
	Aborted    bool   `json:"aborted,omitempty"`
	OutputFile string `json:"output_file,omitempty"`
}

func (t *BashTool) Execute(ctx context.Context, args json.RawMessage) (json.RawMessage, error) {
	var a bashArgs
	if err := json.Unmarshal(args, &a); err != nil {
		return nil, fmt.Errorf("invalid args: %w", err)
	}

	if a.RunInBackground {
		return t.executeBackground(a)
	}

	return t.executeForeground(ctx, a)
}

func (t *BashTool) resolveWorkDir(a bashArgs) (string, error) {
	workDir := t.WorkDir
	if a.WorkDir != "" {
		workDir = ResolvePath(t.WorkDir, a.WorkDir)
	}
	if workDir == "" {
		return "", nil
	}
	info, err := os.Stat(workDir)
	if err != nil {
		if os.IsNotExist(err) {
			return "", fmt.Errorf("working directory does not exist: %s", workDir)
		}
		return "", fmt.Errorf("check working directory %s: %w", workDir, err)
	}
	if !info.IsDir() {
		return "", fmt.Errorf("working directory is not a directory: %s", workDir)
	}
	return workDir, nil
}

// executeBackground starts a shell command in a detached goroutine and returns immediately.
// Output is written to a file on disk for on-demand reading.
func (t *BashTool) executeBackground(a bashArgs) (json.RawMessage, error) {
	timeout := 10 * time.Minute // generous default for background
	if a.Timeout > 0 {
		timeout = time.Duration(a.Timeout) * time.Second
	}

	workDir, err := t.resolveWorkDir(a)
	if err != nil {
		return nil, err
	}

	shellPath, shellArgs, err := resolveShell()
	if err != nil {
		return nil, err
	}

	rt := t.taskRT
	if rt == nil {
		return nil, fmt.Errorf("background mode requires a TaskRuntime; call SetTaskRuntime before use")
	}

	bgCtx, cancel := context.WithTimeout(context.Background(), timeout)

	cmdArgs := append(append([]string{}, shellArgs...), a.Command)
	cmd := exec.CommandContext(bgCtx, shellPath, cmdArgs...)
	if workDir != "" {
		cmd.Dir = workDir
	}
	configureProcGroup(cmd)

	pr, pw, pipeErr := os.Pipe()
	if pipeErr != nil {
		cancel()
		return nil, fmt.Errorf("create pipe: %w", pipeErr)
	}
	cmd.Stdout = pw
	cmd.Stderr = pw

	if err := cmd.Start(); err != nil {
		cancel()
		pr.Close()
		pw.Close()
		return nil, fmt.Errorf("start command: %w", err)
	}
	pw.Close()
	shellID := rt.NextID("shell")
	desc := a.Description
	if desc == "" {
		desc = bashTruncate(a.Command, 60)
	}

	var outFile io.WriteCloser
	var outPath string
	if t.bgOutputFactory != nil {
		var ferr error
		outFile, outPath, ferr = t.bgOutputFactory(shellID)
		if ferr != nil {
			cancel()
			pr.Close()
			return nil, fmt.Errorf("create output: %w", ferr)
		}
	}

	entry := &agentcore.BackgroundTaskEntry{
		ID:          shellID,
		Type:        agentcore.TaskTypeShell,
		Command:     a.Command,
		Description: desc,
		Status:      agentcore.TaskRunning,
		StartedAt:   time.Now(),
		OutputFile:  outPath,
		PID:         cmd.Process.Pid,
	}
	entry.SetCancel(cancel)
	rt.Register(entry)

	// Background goroutine: stream output to file, wait for exit, notify.
	go func() {
		defer cancel()

		var w io.Writer = io.Discard
		if outFile != nil {
			defer outFile.Close()
			w = outFile
		}

		done := make(chan struct{})
		go func() {
			defer close(done)
			io.Copy(w, pr)
		}()

		waitErr := cmd.Wait()

		select {
		case <-done:
		case <-time.After(500 * time.Millisecond):
		}
		pr.Close()
		<-done

		exitCode := 0
		status := agentcore.TaskCompleted
		if waitErr != nil {
			status = agentcore.TaskFailed
			if exitErr, ok := waitErr.(*exec.ExitError); ok {
				exitCode = exitErr.ExitCode()
			} else {
				exitCode = -1
			}
		}

		rt.Update(shellID, func(e *agentcore.BackgroundTaskEntry) {
			e.Status = status
			e.ExitCode = exitCode
			e.EndedAt = time.Now()
		})

		t.notifyCompletion(rt, shellID)
	}()

	return json.Marshal(map[string]any{
		"shell_id":    shellID,
		"pid":         cmd.Process.Pid,
		"description": desc,
		"status":      "running",
		"message":     fmt.Sprintf("Background command %s started (PID %d). You will receive a notification when it completes.", shellID, cmd.Process.Pid),
	})
}

func (t *BashTool) notifyCompletion(rt *agentcore.TaskRuntime, shellID string) {
	if t.notifyFn == nil {
		return
	}
	e := rt.Get(shellID)
	if e == nil {
		return
	}
	result := map[string]any{
		"shell_id":    e.ID,
		"command":     e.Command,
		"description": e.Description,
		"status":      string(e.Status),
		"exit_code":   e.ExitCode,
		"output_file": e.OutputFile,
	}

	data, err := json.Marshal(result)
	if err != nil {
		return
	}
	msg := agentcore.UserMsg(fmt.Sprintf("<background-shell-completed>\n%s\n</background-shell-completed>", string(data)))
	t.notifyFn(msg)
}

// executeForeground runs the command synchronously (original behavior).
func (t *BashTool) executeForeground(ctx context.Context, a bashArgs) (json.RawMessage, error) {
	timeout := t.Timeout
	if a.Timeout > 0 {
		timeout = time.Duration(a.Timeout) * time.Second
	}

	workDir, err := t.resolveWorkDir(a)
	if err != nil {
		return nil, err
	}

	shellPath, shellArgs, err := resolveShell()
	if err != nil {
		return nil, err
	}

	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	cmdArgs := append(append([]string{}, shellArgs...), a.Command)
	cmd := exec.CommandContext(ctx, shellPath, cmdArgs...)
	if workDir != "" {
		cmd.Dir = workDir
	}
	configureProcGroup(cmd)

	pr, pw, pipeErr := os.Pipe()
	if pipeErr != nil {
		return nil, fmt.Errorf("create pipe: %w", pipeErr)
	}
	cmd.Stdout = pw
	cmd.Stderr = pw

	if err := cmd.Start(); err != nil {
		pr.Close()
		pw.Close()
		return nil, fmt.Errorf("start command: %w", err)
	}
	pw.Close()

	var output []byte
	var readErr error
	done := make(chan struct{})
	go func() {
		defer close(done)
		buf := make([]byte, 32*1024)
		pending := make([]byte, 0, 4096)
		for {
			n, readE := pr.Read(buf)
			if n > 0 {
				chunk := buf[:n]
				output = append(output, chunk...)

				pending = append(pending, chunk...)
				for {
					idx := bytes.IndexByte(pending, '\n')
					if idx < 0 {
						break
					}
					agentcore.ReportToolProgress(ctx, agentcore.ProgressPayload{Kind: agentcore.ProgressSummary, Summary: string(append([]byte(nil), pending[:idx]...))})
					pending = pending[idx+1:]
				}
			}
			if readE != nil {
				if !(errors.Is(readE, io.EOF) || errors.Is(readE, os.ErrClosed)) {
					readErr = readE
				} else if len(pending) > 0 {
					agentcore.ReportToolProgress(ctx, agentcore.ProgressPayload{Kind: agentcore.ProgressSummary, Summary: string(append([]byte(nil), pending...))})
				}
				return
			}
		}
	}()

	err = cmd.Wait()

	select {
	case <-done:
	case <-time.After(500 * time.Millisecond):
	}
	pr.Close()
	<-done

	if readErr != nil {
		return nil, fmt.Errorf("read command output: %w", readErr)
	}

	outStr := string(output)
	if outStr == "" {
		outStr = "(no output)"
	}

	var tempPath string
	if len(outStr) > defaultMaxBytes {
		if f, ferr := os.CreateTemp("", "agentcore-bash-*.log"); ferr == nil {
			f.WriteString(outStr)
			tempPath = f.Name()
			f.Close()
		}
	}

	tr := truncateTail(outStr, defaultMaxLines, defaultMaxBytes)
	result := tr.Content

	if tr.Truncated {
		startLine := tr.TotalLines - tr.OutputLines + 1
		if startLine < 1 {
			startLine = 1
		}
		result += fmt.Sprintf("\n\n[Showing lines %d-%d of %d.]", startLine, tr.TotalLines, tr.TotalLines)
		if tempPath != "" {
			result += fmt.Sprintf("\n[Full output saved to: %s]", tempPath)
		}
	}

	exitCode := 0
	timedOut := errors.Is(ctx.Err(), context.DeadlineExceeded)
	aborted := !timedOut && errors.Is(ctx.Err(), context.Canceled)
	if err != nil {
		if exitErr, ok := err.(*exec.ExitError); ok {
			exitCode = exitErr.ExitCode()
		} else if timedOut || aborted {
			exitCode = -1
		} else {
			return nil, fmt.Errorf("command failed: %w", err)
		}
	}

	return json.Marshal(bashForegroundResult{
		Output:     result,
		ExitCode:   exitCode,
		TimedOut:   timedOut,
		Aborted:    aborted,
		OutputFile: tempPath,
	})
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

func bashTruncate(s string, maxRunes int) string {
	runes := []rune(s)
	if len(runes) <= maxRunes {
		return s
	}
	return string(runes[:maxRunes]) + "..."
}

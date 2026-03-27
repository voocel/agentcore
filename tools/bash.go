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
	"sync"
	"time"

	"github.com/voocel/agentcore"
	"github.com/voocel/agentcore/schema"
)

// BackgroundShell tracks a background shell command's lifecycle.
// Output is written to a file on disk (OutputFile) instead of memory.
type BackgroundShell struct {
	ID          string
	Command     string
	Description string
	Status      string // "running" | "completed" | "failed"
	StartedAt   time.Time
	EndedAt     time.Time
	OutputFile  string // path to output file on disk
	ExitCode    int
	PID         int
	mu          sync.Mutex
	cancel      context.CancelFunc
}

func (bs *BackgroundShell) snapshot() BackgroundShell {
	bs.mu.Lock()
	defer bs.mu.Unlock()
	return BackgroundShell{
		ID:          bs.ID,
		Command:     bs.Command,
		Description: bs.Description,
		Status:      bs.Status,
		StartedAt:   bs.StartedAt,
		EndedAt:     bs.EndedAt,
		OutputFile:  bs.OutputFile,
		ExitCode:    bs.ExitCode,
		PID:         bs.PID,
	}
}

// BashTool executes shell commands.
// Streams stdout+stderr via ReportToolProgress for real-time display.
// Final result applies tail truncation (2000 lines / 50KB).
// Supports run_in_background mode for long-running commands.
type BashTool struct {
	WorkDir         string
	Timeout         time.Duration // default: 2 minutes
	notifyFn        func(agentcore.AgentMessage)
	bgOutputFactory func(shellID string) (io.WriteCloser, string, error) // creates output writer for background shells

	mu       sync.Mutex
	bgSeq    int
	bgShells map[string]*BackgroundShell
}

func NewBash(workDir string) *BashTool {
	return &BashTool{
		WorkDir:  workDir,
		Timeout:  2 * time.Minute,
		bgShells: make(map[string]*BackgroundShell),
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

// BackgroundShells returns a snapshot of all background shells.
func (t *BashTool) BackgroundShells() []BackgroundShell {
	t.mu.Lock()
	defer t.mu.Unlock()
	shells := make([]BackgroundShell, 0, len(t.bgShells))
	for _, bs := range t.bgShells {
		shells = append(shells, bs.snapshot())
	}
	return shells
}

// StopBackgroundShell cancels a running background shell by ID.
func (t *BashTool) StopBackgroundShell(id string) bool {
	t.mu.Lock()
	defer t.mu.Unlock()
	bs, ok := t.bgShells[id]
	if !ok || bs.Status != "running" {
		return false
	}
	bs.cancel()
	bs.mu.Lock()
	bs.Status = "failed"
	bs.ExitCode = -1
	bs.EndedAt = time.Now()
	bs.mu.Unlock()
	return true
}

// StopAllBackgroundShells cancels all running background shells.
func (t *BashTool) StopAllBackgroundShells() int {
	t.mu.Lock()
	defer t.mu.Unlock()
	count := 0
	for _, bs := range t.bgShells {
		if bs.Status == "running" {
			bs.cancel()
			bs.mu.Lock()
			bs.Status = "failed"
			bs.ExitCode = -1
			bs.EndedAt = time.Now()
			bs.mu.Unlock()
			count++
		}
	}
	return count
}

func (t *BashTool) Name() string  { return "bash" }
func (t *BashTool) Label() string { return "Execute Command" }
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

	shellID := t.nextBgID()
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

	bs := &BackgroundShell{
		ID:          shellID,
		Command:     a.Command,
		Description: desc,
		Status:      "running",
		StartedAt:   time.Now(),
		OutputFile:  outPath,
		PID:         cmd.Process.Pid,
		cancel:      cancel,
	}
	t.mu.Lock()
	t.bgShells[shellID] = bs
	t.mu.Unlock()

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
		status := "completed"
		if waitErr != nil {
			status = "failed"
			if exitErr, ok := waitErr.(*exec.ExitError); ok {
				exitCode = exitErr.ExitCode()
			} else {
				exitCode = -1
			}
		}

		bs.mu.Lock()
		if bs.Status != "running" {
			bs.mu.Unlock()
			return
		}
		bs.Status = status
		bs.ExitCode = exitCode
		bs.EndedAt = time.Now()
		bs.mu.Unlock()

		t.notifyCompletion(bs)
	}()

	return json.Marshal(map[string]any{
		"shell_id":    shellID,
		"pid":         cmd.Process.Pid,
		"description": desc,
		"status":      "running",
		"message":     fmt.Sprintf("Background command %s started (PID %d). You will receive a notification when it completes.", shellID, cmd.Process.Pid),
	})
}

func (t *BashTool) notifyCompletion(bs *BackgroundShell) {
	if t.notifyFn == nil {
		return
	}
	bs.mu.Lock()
	result := map[string]any{
		"shell_id":    bs.ID,
		"command":     bs.Command,
		"description": bs.Description,
		"status":      bs.Status,
		"exit_code":   bs.ExitCode,
		"output_file": bs.OutputFile,
	}
	bs.mu.Unlock()

	data, err := json.Marshal(result)
	if err != nil {
		return
	}
	msg := agentcore.UserMsg(fmt.Sprintf("<background-shell-completed>\n%s\n</background-shell-completed>", string(data)))
	t.notifyFn(msg)
}

func (t *BashTool) nextBgID() string {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.bgSeq++
	return fmt.Sprintf("shell-%d", t.bgSeq)
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

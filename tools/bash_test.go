package tools

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"os"
	"path"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/voocel/agentcore"
	"github.com/voocel/agentcore/task"
)

type bashResult struct {
	Output     string `json:"output"`
	ExitCode   int    `json:"exit_code"`
	TimedOut   bool   `json:"timed_out"`
	Aborted    bool   `json:"aborted"`
	OutputFile string `json:"output_file"`
}

func runBash(t *testing.T, tool *BashTool, command string, timeout int) (bashResult, error) {
	t.Helper()
	args := map[string]any{"command": command}
	if timeout > 0 {
		args["timeout"] = timeout
	}
	raw, err := json.Marshal(args)
	if err != nil {
		t.Fatalf("marshal args: %v", err)
	}

	out, execErr := tool.Execute(context.Background(), raw)
	if execErr != nil {
		return bashResult{}, execErr
	}

	var result bashResult
	if err := json.Unmarshal(out, &result); err == nil && result.Output != "" {
		return result, nil
	}

	var text string
	if err := json.Unmarshal(out, &text); err != nil {
		t.Fatalf("unmarshal output: %v", err)
	}
	return bashResult{Output: text}, nil
}

func runBashWithArgs(t *testing.T, tool *BashTool, args map[string]any) (bashResult, error) {
	t.Helper()
	raw, err := json.Marshal(args)
	if err != nil {
		t.Fatalf("marshal args: %v", err)
	}

	out, execErr := tool.Execute(context.Background(), raw)
	if execErr != nil {
		return bashResult{}, execErr
	}

	var result bashResult
	if err := json.Unmarshal(out, &result); err == nil && result.Output != "" {
		return result, nil
	}

	var text string
	if err := json.Unmarshal(out, &text); err != nil {
		t.Fatalf("unmarshal output: %v", err)
	}
	return bashResult{Output: text}, nil
}

func TestBashTimeoutErrorMessage(t *testing.T) {
	t.Parallel()

	tool := NewBash(".")
	result, err := runBash(t, tool, "sleep 2", 1)
	if err != nil {
		t.Fatalf("execute bash: %v", err)
	}
	if !result.TimedOut {
		t.Fatalf("expected timed_out=true, got %#v", result)
	}
	if result.ExitCode == 0 {
		t.Fatalf("expected non-zero exit code for timeout, got %#v", result)
	}
}

func TestBashLongSingleLineOutputNotDropped(t *testing.T) {
	t.Parallel()

	tool := NewBash(".")
	out, err := runBash(t, tool, "yes a | tr -d '\\n' | head -c 300000", 0)
	if err != nil {
		t.Fatalf("execute bash: %v", err)
	}
	if out.Output == "(no output)" {
		t.Fatalf("unexpected empty output for long single line")
	}
	if !strings.Contains(out.Output, "a") {
		t.Fatalf("expected output to contain command data, got: %q", out.Output)
	}
}

func TestBashMissingWorkDirError(t *testing.T) {
	t.Parallel()

	missing := filepath.Join(t.TempDir(), "missing-dir")
	tool := NewBash(missing)
	_, err := runBash(t, tool, "echo hi", 0)
	if err == nil {
		t.Fatal("expected workdir error")
	}
	if !strings.Contains(strings.ToLower(err.Error()), "working directory does not exist") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestBashUsesPerCommandWorkDir(t *testing.T) {
	t.Parallel()

	root := t.TempDir()
	nested := filepath.Join(root, "nested")
	if err := os.Mkdir(nested, 0o755); err != nil {
		t.Fatalf("mkdir nested: %v", err)
	}

	tool := NewBash(root)
	out, err := runBashWithArgs(t, tool, map[string]any{
		"command": "pwd",
		"workdir": "nested",
	})
	if err != nil {
		t.Fatalf("execute bash: %v", err)
	}
	// Compare the tail path segment only: on Windows the shell is Git Bash,
	// whose pwd prints the MSYS-mapped POSIX form of the same directory
	// (%TEMP% → /tmp), so a literal comparison against the OS path fails even
	// though the workdir was set correctly.
	if got := path.Base(filepath.ToSlash(strings.TrimSpace(out.Output))); got != "nested" {
		t.Fatalf("expected pwd to end in \"nested\", got %q (full output %q)", got, out.Output)
	}
}

func TestBashNonZeroExitReturnsStructuredResult(t *testing.T) {
	t.Parallel()

	tool := NewBash(".")
	result, err := runBash(t, tool, "printf 'broken\\n'; exit 7", 0)
	if err != nil {
		t.Fatalf("execute bash: %v", err)
	}
	if result.ExitCode != 7 {
		t.Fatalf("expected exit code 7, got %#v", result)
	}
	if result.TimedOut || result.Aborted {
		t.Fatalf("unexpected timeout/abort flags: %#v", result)
	}
	if !strings.Contains(result.Output, "broken") {
		t.Fatalf("expected output to contain command output, got %#v", result)
	}
}

func TestBashBackgroundOutputErrorsFailTask(t *testing.T) {
	if _, _, err := resolveShell(); err != nil {
		t.Skip(err)
	}

	for _, tc := range []struct {
		name       string
		command    string
		writer     io.WriteCloser
		wantCode   int
		wantErrors []string
	}{
		{
			name:       "write error",
			command:    "printf output",
			writer:     &errorWriteCloser{writeErr: errors.New("disk full")},
			wantErrors: []string{"copy command output: disk full"},
		},
		{
			name:       "close error",
			command:    "printf output",
			writer:     &errorWriteCloser{closeErr: errors.New("flush log")},
			wantErrors: []string{"close command output: flush log"},
		},
		{
			name:       "command and write errors",
			command:    "printf output; exit 7",
			writer:     &errorWriteCloser{writeErr: errors.New("disk full")},
			wantCode:   7,
			wantErrors: []string{"wait command:", "copy command output: disk full"},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			rt := task.NewRuntime()
			tool := NewBash(".")
			tool.SetTaskRuntime(rt)
			tool.SetBgOutputFactory(func(string) (io.WriteCloser, string, error) {
				return tc.writer, "background.log", nil
			})
			notified := make(chan agentcore.AgentMessage, 1)
			tool.SetNotifyFn(func(msg agentcore.AgentMessage) { notified <- msg })

			args, err := json.Marshal(map[string]any{
				"command":           tc.command,
				"run_in_background": true,
			})
			if err != nil {
				t.Fatalf("marshal args: %v", err)
			}
			if _, err := tool.Execute(context.Background(), args); err != nil {
				t.Fatalf("execute background bash: %v", err)
			}

			waited := make(chan struct{})
			go func() {
				rt.Wait()
				close(waited)
			}()
			select {
			case <-waited:
			case <-time.After(15 * time.Second):
				t.Fatal("background bash did not finish")
			}

			entries := rt.List()
			if len(entries) != 1 {
				t.Fatalf("tasks: got %d, want 1", len(entries))
			}
			if entries[0].Status != task.Failed {
				t.Fatalf("status: got %q, want %q", entries[0].Status, task.Failed)
			}
			if entries[0].ExitCode != tc.wantCode {
				t.Fatalf("exit code: got %d, want %d", entries[0].ExitCode, tc.wantCode)
			}
			for _, want := range tc.wantErrors {
				if !strings.Contains(entries[0].Error, want) {
					t.Fatalf("task error %q does not contain %q", entries[0].Error, want)
				}
			}
			select {
			case <-notified:
			default:
				t.Fatal("Wait returned before completion notification")
			}
		})
	}
}

type errorWriteCloser struct {
	writeErr error
	closeErr error
}

func (w *errorWriteCloser) Write(p []byte) (int, error) {
	if w.writeErr != nil {
		return 0, w.writeErr
	}
	return len(p), nil
}

func (w *errorWriteCloser) Close() error { return w.closeErr }

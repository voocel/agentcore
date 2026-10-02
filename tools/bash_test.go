package tools

import (
	"context"
	"os"
	"path"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/voocel/agentcore"
	"github.com/voocel/agentcore/task"
)

func bash(t *testing.T, dir string, args map[string]any) (string, error) {
	t.Helper()
	return run(t, Workspace{Dir: dir}.Bash(), args)
}

// The model reads a command's output as it is, with a line for how it
// ended; a failing command is not a failed call.
func TestBashResult(t *testing.T) {
	t.Parallel()
	got, err := bash(t, ".", map[string]any{"command": "printf 'ch <-chan int && x > y\\nbroken\\n'; exit 7"})
	if err != nil || got != "ch <-chan int && x > y\nbroken\n\n[exit code 7]" {
		t.Fatalf("result %q, %v", got, err)
	}
	if got, _ := bash(t, ".", map[string]any{"command": "true"}); got != "(no output)" {
		t.Fatalf("no output: %q", got)
	}
	got, err = bash(t, ".", map[string]any{"command": "echo started; sleep 2", "timeout": 1})
	if err != nil || got != "started\n\n[timed out after 1s]" {
		t.Fatalf("timeout: %q, %v", got, err)
	}
}

func TestBashLongOutputKeepsTheTail(t *testing.T) {
	t.Setenv("TMPDIR", t.TempDir()) // where the full outputs go
	got, err := bash(t, ".", map[string]any{"command": "seq 1 3000"})
	if err != nil {
		t.Fatal(err)
	}
	lines := strings.Split(got, "\n")
	note := lines[len(lines)-1]
	if lines[len(lines)-3] != "3000" || !strings.HasPrefix(note, "[Showing the last 2000 of 3000 lines. Full output: ") {
		t.Fatalf("tail ends %q", lines[len(lines)-3:])
	}
	full := strings.TrimSuffix(strings.TrimPrefix(note, "[Showing the last 2000 of 3000 lines. Full output: "), "]")
	if data, err := os.ReadFile(full); err != nil || !strings.HasPrefix(string(data), "1\n2\n") {
		t.Fatalf("full output: %v", err)
	}

	got, err = bash(t, ".", map[string]any{"command": "yes a | tr -d '\\n' | head -c 300000"})
	if err != nil || !strings.Contains(got, "aaaa") {
		t.Fatalf("a long single line was dropped: %v", err)
	}
}

func TestBashWorkDir(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	if err := os.Mkdir(filepath.Join(root, "nested"), 0o755); err != nil {
		t.Fatal(err)
	}
	got, err := bash(t, root, map[string]any{"command": "pwd", "workdir": "nested"})
	// The tail segment only: Git Bash on Windows prints the MSYS form of
	// the same directory.
	if err != nil || path.Base(filepath.ToSlash(strings.TrimSpace(got))) != "nested" {
		t.Fatalf("pwd = %q, %v", got, err)
	}
	if _, err := bash(t, filepath.Join(root, "missing"), map[string]any{"command": "true"}); err == nil || !strings.Contains(err.Error(), "working directory does not exist") {
		t.Fatalf("missing dir: %v", err)
	}
}

// The background mode is offered with Tasks only, and its commands run as
// tasks, with no timeout unless asked.
func TestBashBackground(t *testing.T) {
	if props := (Workspace{}).Bash().Schema["properties"].(map[string]any); props["run_in_background"] != nil {
		t.Fatal("background mode offered without Tasks")
	}

	notes := make(chan agentcore.Message, 2)
	rt := task.NewRuntime(t.TempDir(), func(m agentcore.Message) { notes <- m })
	tool := Workspace{Dir: ".", Tasks: rt}.Bash()
	got, err := run(t, tool, map[string]any{"command": "echo out; exit 3", "run_in_background": true})
	if err != nil || !strings.Contains(got, "Started task shell-1") {
		t.Fatalf("start: %q, %v", got, err)
	}
	if _, err := run(t, tool, map[string]any{"command": "sleep 30", "run_in_background": true}); err != nil {
		t.Fatal(err)
	}

	e, err := rt.WaitFor(context.Background(), "shell-1")
	if err != nil || e.Status != task.Failed || e.ExitCode != 3 || e.Error != "exit code 3" {
		t.Fatalf("shell-1 = %+v, %v", e, err)
	}
	if data, _ := os.ReadFile(e.OutputFile); string(data) != "out\n" {
		t.Fatalf("output %q", data)
	}
	time.Sleep(100 * time.Millisecond)
	if e, _ := rt.Get("shell-2"); e.Status != task.Running || e.PID == 0 {
		t.Fatalf("shell-2 = %+v", e)
	}
	rt.StopAll()
	rt.Wait()
	if e, _ := rt.Get("shell-2"); e.Status != task.Killed {
		t.Fatalf("stopped: %+v", e)
	}
	if len(notes) != 2 {
		t.Fatalf("%d notifications", len(notes))
	}
}

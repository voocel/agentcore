package tools

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
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
	if !strings.Contains(strings.TrimSpace(out.Output), nested) {
		t.Fatalf("expected output to contain %q, got %q", nested, out.Output)
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

package tools

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestWithCwdAndCwdFromContext(t *testing.T) {
	if got := CwdFromContext(context.Background()); got != "" {
		t.Errorf("bare ctx CwdFromContext = %q, want empty", got)
	}
	var nilCtx context.Context // the defensive nil guard in CwdFromContext
	if got := CwdFromContext(nilCtx); got != "" {
		t.Errorf("CwdFromContext(nil) = %q, want empty", got)
	}
	if got := CwdFromContext(WithCwd(context.Background(), "/a/b")); got != "/a/b" {
		t.Errorf("CwdFromContext = %q, want /a/b", got)
	}
	// Empty cwd is a no-op (returns parent), so the override stays unset.
	if got := CwdFromContext(WithCwd(context.Background(), "")); got != "" {
		t.Errorf("WithCwd(ctx, \"\") should be a no-op, got %q", got)
	}
}

func TestEffectiveWorkDir(t *testing.T) {
	if got := effectiveWorkDir(context.Background(), "/fallback"); got != "/fallback" {
		t.Errorf("no override → %q, want /fallback", got)
	}
	ctx := WithCwd(context.Background(), "/override")
	if got := effectiveWorkDir(ctx, "/fallback"); got != "/override" {
		t.Errorf("override → %q, want /override", got)
	}
}

// TestCwdOverride_ToolResolution proves a tool built for dir A resolves paths
// against dir B when the call ctx carries a cwd override, and against A
// otherwise — the property git-worktree isolation relies on.
func TestCwdOverride_ToolResolution(t *testing.T) {
	dirA := t.TempDir()
	dirB := t.TempDir()
	if err := os.WriteFile(filepath.Join(dirB, "f.txt"), []byte("in B"), 0o644); err != nil {
		t.Fatal(err)
	}

	read := NewRead(dirA, nil)
	args, _ := json.Marshal(map[string]any{"file_path": "f.txt"})

	// Override → resolves against B, file found.
	if _, err := read.Execute(WithCwd(context.Background(), dirB), args); err != nil {
		t.Fatalf("read with cwd override: %v", err)
	}
	// No override → resolves against A, file absent.
	if _, err := read.Execute(context.Background(), args); err == nil {
		t.Fatal("read without override should not find B's file under A")
	}
}

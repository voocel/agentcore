package tools

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

// The working directory a context carries is read at each use, the
// innermost winning, and overrides the Workspace's Dir unless it is "".
func TestCwd(t *testing.T) {
	if got := CwdFromContext(context.Background()); got != "" {
		t.Fatalf("bare context: %q", got)
	}
	cwd := "/first"
	ctx := WithCwd(context.Background(), func() string { return cwd })
	cwd = "/second"
	if got := CwdFromContext(ctx); got != "/second" {
		t.Fatalf("read once: %q", got)
	}
	inner := WithCwd(ctx, func() string { return "/inner" })
	if got := CwdFromContext(inner); got != "/inner" {
		t.Fatalf("innermost: %q", got)
	}

	w := Workspace{Dir: "/dir"}
	if got := w.dir(WithCwd(context.Background(), func() string { return "" })); got != "/dir" {
		t.Fatalf("empty cwd: %q", got)
	}
	if got := w.dir(inner); got != "/inner" {
		t.Fatalf("override: %q", got)
	}
}

// A tool resolves paths against the cwd of the call, as a worktree needs.
func TestToolsResolveAgainstTheCallCwd(t *testing.T) {
	dirA, dirB := t.TempDir(), t.TempDir()
	if err := os.WriteFile(filepath.Join(dirB, "f.txt"), []byte("in B"), 0o644); err != nil {
		t.Fatal(err)
	}
	read := Workspace{Dir: dirA}.Read()
	args, _ := json.Marshal(map[string]any{"file_path": "f.txt"})
	if _, err := read.Run(WithCwd(context.Background(), func() string { return dirB }), args); err != nil {
		t.Fatalf("read with the call cwd: %v", err)
	}
	if _, err := read.Run(context.Background(), args); err == nil {
		t.Fatal("read found B's file under A")
	}
}

package policy

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/voocel/agentcore"
)

func toolCall(name, args string) agentcore.ToolCall {
	return agentcore.ToolCall{Name: name, Args: []byte(args)}
}

func TestAllowTools(t *testing.T) {
	rule := AllowTools("read", "write")

	if err := rule(context.Background(), toolCall("read", `{}`)); err != nil {
		t.Fatalf("read should be allowed: %v", err)
	}
	if err := rule(context.Background(), toolCall("bash", `{}`)); err == nil {
		t.Fatal("bash should be denied")
	}
}

func TestDenyTools(t *testing.T) {
	rule := DenyTools("bash")

	if err := rule(context.Background(), toolCall("read", `{}`)); err != nil {
		t.Fatalf("read should be allowed: %v", err)
	}
	if err := rule(context.Background(), toolCall("bash", `{}`)); err == nil {
		t.Fatal("bash should be denied")
	}
}

func TestRestrictPaths(t *testing.T) {
	root := t.TempDir()
	inside := filepath.Join(root, "sub", "file.go")
	outside := filepath.Join(filepath.Dir(root), "outside.go")

	rule := RestrictPaths(root)

	if err := rule(context.Background(), toolCall("read", `{"path":"sub/file.go"}`)); err != nil {
		t.Fatalf("relative path inside root should be allowed: %v", err)
	}
	if err := rule(context.Background(), toolCall("read", `{"path":"`+inside+`"}`)); err != nil {
		t.Fatalf("absolute path inside root should be allowed: %v", err)
	}
	if err := rule(context.Background(), toolCall("read", `{"path":"../outside.go"}`)); err == nil {
		t.Fatal("relative escape path should be denied")
	}
	if err := rule(context.Background(), toolCall("read", `{"path":"`+outside+`"}`)); err == nil {
		t.Fatal("absolute path outside root should be denied")
	}
}

func TestRestrictPaths_DeniesSymlinkEscape(t *testing.T) {
	root := t.TempDir()
	outsideDir := t.TempDir()
	linkPath := filepath.Join(root, "link")

	if err := os.Symlink(outsideDir, linkPath); err != nil {
		t.Skipf("symlink unsupported: %v", err)
	}

	rule := RestrictPaths(root)
	if err := rule(context.Background(), toolCall("read", `{"path":"link/secret.txt"}`)); err == nil {
		t.Fatal("symlink escape should be denied")
	}
}

func TestReadOnlyProfile(t *testing.T) {
	root := t.TempDir()
	perm := ReadOnlyProfile(root)

	if err := perm(context.Background(), toolCall("read", `{"path":"file.go"}`)); err != nil {
		t.Fatalf("read should be allowed: %v", err)
	}
	if err := perm(context.Background(), toolCall("write", `{"path":"file.go","content":"x"}`)); err == nil {
		t.Fatal("write should be denied in read-only profile")
	}
	if err := perm(context.Background(), toolCall("bash", `{"command":"pwd"}`)); err == nil {
		t.Fatal("bash should be denied in read-only profile")
	}
}

func TestWorkspaceProfile(t *testing.T) {
	root := t.TempDir()
	perm := WorkspaceProfile(root)

	if err := perm(context.Background(), toolCall("edit", `{"path":"a.go","old_text":"a","new_text":"b"}`)); err != nil {
		t.Fatalf("edit inside root should be allowed: %v", err)
	}
	if err := perm(context.Background(), toolCall("edit", `{"path":"../a.go","old_text":"a","new_text":"b"}`)); err == nil {
		t.Fatal("edit outside root should be denied")
	}
	if err := perm(context.Background(), toolCall("bash", `{"command":"pwd"}`)); err == nil {
		t.Fatal("bash should be denied in workspace profile")
	}
}

func TestChain_ReturnsFirstError(t *testing.T) {
	perm := Chain(
		DenyTools("bash"),
		func(_ context.Context, call agentcore.ToolCall) error {
			t.Fatal("later rule should not run after first error")
			return nil
		},
	)

	if err := perm(context.Background(), toolCall("bash", `{}`)); err == nil {
		t.Fatal("expected chain to return first rule error")
	}
}

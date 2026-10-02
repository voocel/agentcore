package tools

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func searchFixture(t *testing.T) string {
	dir := t.TempDir()
	for name, content := range map[string]string{
		"main.go":     "package main\n// Hello\n",
		".git/config": "hello from git\n",
		".env":        "HELLO=1\n",
	} {
		path := filepath.Join(dir, name)
		os.MkdirAll(filepath.Dir(path), 0o755)
		if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return dir
}

// glob lists hidden files but never .git's.
func TestGlobLeavesOutGit(t *testing.T) {
	dir := searchFixture(t)
	got, err := run(t, Workspace{Dir: dir}.Glob(), map[string]any{"pattern": "**/*"})
	if err != nil || strings.Contains(got, ".git") || !strings.Contains(got, "main.go") {
		t.Fatalf("glob = %q, %v", got, err)
	}
}

func TestGrepOptions(t *testing.T) {
	dir := searchFixture(t)
	got, err := run(t, Workspace{Dir: dir}.Grep(), map[string]any{"pattern": "hello", "ignore_case": true, "glob": "*.go", "context_lines": 1})
	if err != nil || !strings.Contains(got, "main.go:2:// Hello") || !strings.Contains(got, "main.go-1-package main") || strings.Contains(got, ".git") {
		t.Fatalf("grep = %q, %v", got, err)
	}
}

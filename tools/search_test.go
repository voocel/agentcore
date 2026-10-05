package tools

import (
	"os"
	"path/filepath"
	"runtime"
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

// Globs with a directory part match paths under the directory searched.
func TestSearchGlobsWithDirectories(t *testing.T) {
	dir := searchFixture(t)
	os.MkdirAll(filepath.Join(dir, "src/a"), 0o755)
	if err := os.WriteFile(filepath.Join(dir, "src/a/b.ts"), []byte("hello\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	got, err := run(t, Workspace{Dir: dir}.Glob(), map[string]any{"pattern": "src/**/*.ts"})
	if err != nil || !strings.Contains(got, filepath.Join("src", "a", "b.ts")) {
		t.Fatalf("glob = %q, %v", got, err)
	}
	got, err = run(t, Workspace{Dir: dir}.Grep(), map[string]any{"pattern": "hello", "glob": "src/**/*.ts"})
	if err != nil || !strings.Contains(got, filepath.Join("src", "a", "b.ts")+":1:hello") {
		t.Fatalf("grep = %q, %v", got, err)
	}
}

// A pattern that reads as a flag is searched for.
func TestGrepPatternIsNotAFlag(t *testing.T) {
	dir := searchFixture(t)
	if err := os.WriteFile(filepath.Join(dir, "flags.txt"), []byte("run --pre=cat\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	got, err := run(t, Workspace{Dir: dir}.Grep(), map[string]any{"pattern": "--pre=cat"})
	if err != nil || !strings.Contains(got, "flags.txt:1:run --pre=cat") {
		t.Fatalf("grep = %q, %v", got, err)
	}
}

// rg's errors, such as on an invalid pattern, reach the model: the Go
// search stands in for an rg not installed only.
func TestSearchFallsBackOnlyWithoutRg(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("the rg standing in is a shell script")
	}
	dir := searchFixture(t)
	bin := t.TempDir()
	if err := os.WriteFile(filepath.Join(bin, "rg"), []byte("#!/bin/sh\necho 'rg: bad pattern' >&2\nexit 2\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", bin)
	if got, err := run(t, Workspace{Dir: dir}.Grep(), map[string]any{"pattern": "("}); err == nil || !strings.Contains(err.Error(), "rg: bad pattern") {
		t.Fatalf("grep = %q, %v", got, err)
	}
	if got, err := run(t, Workspace{Dir: dir}.Glob(), map[string]any{"pattern": "a["}); err == nil || !strings.Contains(err.Error(), "rg: bad pattern") {
		t.Fatalf("glob = %q, %v", got, err)
	}

	t.Setenv("PATH", t.TempDir())
	if got, err := run(t, Workspace{Dir: dir}.Grep(), map[string]any{"pattern": "Hello"}); err != nil || !strings.Contains(got, "main.go:2:// Hello") {
		t.Fatalf("grep without rg = %q, %v", got, err)
	}
	if got, err := run(t, Workspace{Dir: dir}.Glob(), map[string]any{"pattern": "*.go"}); err != nil || got != "main.go" {
		t.Fatalf("glob without rg = %q, %v", got, err)
	}
}

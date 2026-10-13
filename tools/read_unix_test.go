//go:build !windows

package tools

import (
	"context"
	"path/filepath"
	"strings"
	"syscall"
	"testing"
)

// A pipe has no end to read to.
func TestReadRefusesAPipe(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "fifo")
	if err := syscall.Mkfifo(path, 0o600); err != nil {
		t.Fatal(err)
	}
	_, err := Workspace{Dir: dir}.Read().Run(context.Background(), mustJSON(t, readArgs{FilePath: path}))
	if err == nil || !strings.Contains(err.Error(), "not a regular file") {
		t.Fatalf("err = %v", err)
	}
}

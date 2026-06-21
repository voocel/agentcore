package tools

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"io/fs"
	"sync"
	"testing"
	"time"
)

// memoryWorkspaceFS is an in-memory WorkspaceFS used to exercise the injected
// backend path — in particular the FileInfo.Version stale-write detection that
// the OS backend (Version always empty) cannot demonstrate.
type memoryWorkspaceFS struct {
	mu    sync.Mutex
	files map[string]*memFile
}

type memFile struct {
	data    []byte
	mtime   time.Time
	version string
}

func newMemoryWorkspaceFS() *memoryWorkspaceFS {
	return &memoryWorkspaceFS{files: make(map[string]*memFile)}
}

func (m *memoryWorkspaceFS) put(path string, data []byte, mtime time.Time, version string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.files[path] = &memFile{data: append([]byte(nil), data...), mtime: mtime, version: version}
}

func (m *memoryWorkspaceFS) get(path string) (*memFile, bool) {
	m.mu.Lock()
	defer m.mu.Unlock()
	f, ok := m.files[path]
	return f, ok
}

func (m *memoryWorkspaceFS) Stat(_ context.Context, path string) (FileInfo, error) {
	f, ok := m.get(path)
	if !ok {
		return FileInfo{}, fs.ErrNotExist
	}
	return FileInfo{Name: path, Size: int64(len(f.data)), ModTime: f.mtime, Version: f.version}, nil
}

func (m *memoryWorkspaceFS) Open(_ context.Context, path string) (io.ReadCloser, error) {
	f, ok := m.get(path)
	if !ok {
		return nil, fs.ErrNotExist
	}
	return io.NopCloser(bytes.NewReader(append([]byte(nil), f.data...))), nil
}

func (m *memoryWorkspaceFS) ReadFile(_ context.Context, path string) ([]byte, error) {
	f, ok := m.get(path)
	if !ok {
		return nil, fs.ErrNotExist
	}
	return append([]byte(nil), f.data...), nil
}

func (m *memoryWorkspaceFS) ReadDir(_ context.Context, _ string) ([]DirEntry, error) {
	return nil, fs.ErrNotExist
}

func (m *memoryWorkspaceFS) WriteFile(_ context.Context, path string, data []byte, _ fs.FileMode) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	f, ok := m.files[path]
	if !ok {
		f = &memFile{mtime: time.Unix(0, 0)}
		m.files[path] = f
	}
	f.data = append([]byte(nil), data...)
	return nil
}

func (m *memoryWorkspaceFS) MkdirAll(_ context.Context, _ string, _ fs.FileMode) error {
	return nil
}

func mustJSON(t *testing.T, v any) json.RawMessage {
	t.Helper()
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	return b
}

// Version changes must be detected as stale writes even when mtime is unchanged
// — this is the unsaved-buffer case the OS mtime check would miss.
func TestWorkspaceFS_VersionDetectsUnsavedChange(t *testing.T) {
	ctx := context.Background()
	mfs := newMemoryWorkspaceFS()
	mtime := time.Unix(1000, 0)
	const path = "/work/file.txt"
	mfs.put(path, []byte("hello\n"), mtime, "v1")

	state := NewFileReadState()
	read := NewRead("/work", state, WithFS(mfs))
	write := NewWrite("/work", state, WithFS(mfs))

	if _, err := read.Execute(ctx, mustJSON(t, readArgs{FilePath: path})); err != nil {
		t.Fatalf("read: %v", err)
	}

	// Buffer content changes (version bumps) but mtime stays identical.
	mfs.put(path, []byte("hello world\n"), mtime, "v2")

	res := write.Validate(ctx, mustJSON(t, writeArgs{FilePath: path, Content: "x"}))
	if res.OK {
		t.Fatal("expected stale-write rejection (version changed), got OK")
	}
	if res.ErrorCode != 3 {
		t.Fatalf("expected ErrorCode 3, got %d", res.ErrorCode)
	}
}

// When neither version nor content changes, the write validates.
func TestWorkspaceFS_VersionUnchangedAllowsWrite(t *testing.T) {
	ctx := context.Background()
	mfs := newMemoryWorkspaceFS()
	const path = "/work/file.txt"
	mfs.put(path, []byte("hello\n"), time.Unix(1000, 0), "v1")

	state := NewFileReadState()
	read := NewRead("/work", state, WithFS(mfs))
	write := NewWrite("/work", state, WithFS(mfs))

	if _, err := read.Execute(ctx, mustJSON(t, readArgs{FilePath: path})); err != nil {
		t.Fatalf("read: %v", err)
	}

	res := write.Validate(ctx, mustJSON(t, writeArgs{FilePath: path, Content: "new"}))
	if !res.OK {
		t.Fatalf("expected validate OK, got %q (code %d)", res.Message, res.ErrorCode)
	}
}

// Writes and edits land in the injected backend, not the local disk.
func TestWorkspaceFS_OperationsTargetBackend(t *testing.T) {
	ctx := context.Background()
	mfs := newMemoryWorkspaceFS()
	const path = "/work/file.txt"
	mfs.put(path, []byte("alpha\n"), time.Unix(1000, 0), "v1")

	state := NewFileReadState()
	read := NewRead("/work", state, WithFS(mfs))
	edit := NewEdit("/work", state, WithFS(mfs))

	if _, err := read.Execute(ctx, mustJSON(t, readArgs{FilePath: path})); err != nil {
		t.Fatalf("read: %v", err)
	}
	if _, err := edit.Execute(ctx, mustJSON(t, editArgs{FilePath: path, OldString: "alpha", NewString: "beta"})); err != nil {
		t.Fatalf("edit: %v", err)
	}

	f, ok := mfs.get(path)
	if !ok {
		t.Fatal("file missing from backend after edit")
	}
	if got := string(f.data); got != "beta\n" {
		t.Fatalf("backend content = %q, want %q", got, "beta\n")
	}
}

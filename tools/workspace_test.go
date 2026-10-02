package tools

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"io/fs"
	"strings"
	"sync"
	"testing"
	"time"
)

// memoryFS is an in-memory FS used to exercise the injected
// backend path — in particular the FileInfo.Version stale-write detection that
// the OS backend (Version always empty) cannot demonstrate.
type memoryFS struct {
	mu    sync.Mutex
	files map[string]*memFile
}

type memFile struct {
	data    []byte
	mtime   time.Time
	version string
}

func newMemoryFS() *memoryFS {
	return &memoryFS{files: make(map[string]*memFile)}
}

func (m *memoryFS) put(path string, data []byte, mtime time.Time, version string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.files[path] = &memFile{data: append([]byte(nil), data...), mtime: mtime, version: version}
}

func (m *memoryFS) get(path string) (*memFile, bool) {
	m.mu.Lock()
	defer m.mu.Unlock()
	f, ok := m.files[path]
	return f, ok
}

func (m *memoryFS) Stat(_ context.Context, path string) (FileInfo, error) {
	f, ok := m.get(path)
	if !ok {
		return FileInfo{}, fs.ErrNotExist
	}
	return FileInfo{Name: path, Size: int64(len(f.data)), ModTime: f.mtime, Version: f.version}, nil
}

func (m *memoryFS) Open(_ context.Context, path string) (io.ReadCloser, error) {
	f, ok := m.get(path)
	if !ok {
		return nil, fs.ErrNotExist
	}
	return io.NopCloser(bytes.NewReader(append([]byte(nil), f.data...))), nil
}

func (m *memoryFS) ReadFile(_ context.Context, path string) ([]byte, error) {
	f, ok := m.get(path)
	if !ok {
		return nil, fs.ErrNotExist
	}
	return append([]byte(nil), f.data...), nil
}

func (m *memoryFS) ReadDir(_ context.Context, _ string) ([]DirEntry, error) {
	return nil, fs.ErrNotExist
}

func (m *memoryFS) WriteFile(_ context.Context, path string, data []byte, _ fs.FileMode) error {
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

func (m *memoryFS) MkdirAll(_ context.Context, _ string, _ fs.FileMode) error {
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
func TestFS_VersionDetectsUnsavedChange(t *testing.T) {
	ctx := context.Background()
	mfs := newMemoryFS()
	mtime := time.Unix(1000, 0)
	const path = "/work/file.txt"
	mfs.put(path, []byte("hello\n"), mtime, "v1")

	state := NewFileReadState()
	read := Workspace{Dir: "/work", Files: state, FS: mfs}.Read()
	write := Workspace{Dir: "/work", Files: state, FS: mfs}.Write()

	if _, err := read.Run(ctx, mustJSON(t, readArgs{FilePath: path})); err != nil {
		t.Fatalf("read: %v", err)
	}

	// Buffer content changes (version bumps) but mtime stays identical.
	mfs.put(path, []byte("hello world\n"), mtime, "v2")

	_, err := write.Check(ctx, mustJSON(t, writeArgs{FilePath: path, Content: "x"}))
	if err == nil || !strings.Contains(err.Error(), "modified since read") {
		t.Fatalf("expected stale-write rejection (version changed), got %v", err)
	}
}

// When neither version nor content changes, the write validates.
func TestFS_VersionUnchangedAllowsWrite(t *testing.T) {
	ctx := context.Background()
	mfs := newMemoryFS()
	const path = "/work/file.txt"
	mfs.put(path, []byte("hello\n"), time.Unix(1000, 0), "v1")

	state := NewFileReadState()
	read := Workspace{Dir: "/work", Files: state, FS: mfs}.Read()
	write := Workspace{Dir: "/work", Files: state, FS: mfs}.Write()

	if _, err := read.Run(ctx, mustJSON(t, readArgs{FilePath: path})); err != nil {
		t.Fatalf("read: %v", err)
	}

	if _, err := write.Check(ctx, mustJSON(t, writeArgs{FilePath: path, Content: "new"})); err != nil {
		t.Fatalf("expected the check to pass, got %v", err)
	}
}

// Writes and edits land in the injected backend, not the local disk.
func TestFS_OperationsTargetBackend(t *testing.T) {
	ctx := context.Background()
	mfs := newMemoryFS()
	const path = "/work/file.txt"
	mfs.put(path, []byte("alpha\n"), time.Unix(1000, 0), "v1")

	state := NewFileReadState()
	read := Workspace{Dir: "/work", Files: state, FS: mfs}.Read()
	edit := Workspace{Dir: "/work", Files: state, FS: mfs}.Edit()

	if _, err := read.Run(ctx, mustJSON(t, readArgs{FilePath: path})); err != nil {
		t.Fatalf("read: %v", err)
	}
	if _, err := edit.Run(ctx, mustJSON(t, editArgs{FilePath: path, OldString: "alpha", NewString: "beta"})); err != nil {
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

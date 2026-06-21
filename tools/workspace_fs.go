package tools

import (
	"context"
	"io"
	"io/fs"
	"os"
	"time"
)

// WorkspaceFS abstracts access to the agent's workspace files so the backend
// can be the local OS, an editor (serving unsaved buffers), a remote host, or
// an in-memory store. The read/write/edit tools operate through this interface
// instead of calling os.* directly.
//
// All paths are absolute: tools call ResolvePath before handing a path to a
// WorkspaceFS method. Implementations that perform I/O over a transport should
// honour ctx cancellation; the OS backend ignores ctx.
type WorkspaceFS interface {
	// Stat returns metadata for the file or directory at path.
	Stat(ctx context.Context, path string) (FileInfo, error)

	// Open returns a streaming reader. Used for line-by-line text reads and
	// for sniffing the first bytes (MIME / binary detection) without loading
	// the whole file into memory.
	Open(ctx context.Context, path string) (io.ReadCloser, error)

	// ReadFile returns the full file contents. Used where the whole file is
	// needed anyway: image decode, edit full-content match, write old content.
	ReadFile(ctx context.Context, path string) ([]byte, error)

	// ReadDir lists directory entries (non-recursive).
	ReadDir(ctx context.Context, path string) ([]DirEntry, error)

	// WriteFile writes data to path, truncating any existing file.
	WriteFile(ctx context.Context, path string, data []byte, perm fs.FileMode) error

	// MkdirAll creates path and any missing parents.
	MkdirAll(ctx context.Context, path string, perm fs.FileMode) error
}

// FileInfo is the backend-neutral subset of os.FileInfo the file tools need,
// plus Version.
//
// Version is a backend-defined opaque token identifying the content revision.
// The OS backend leaves it empty (read-before-write falls back to ModTime).
// Backends serving unsaved editor buffers or remote objects may set it (e.g. a
// content hash or etag); when both the read stamp and the current FileInfo
// carry a non-empty Version, Write/Edit compare Version instead of ModTime to
// detect stale writes.
type FileInfo struct {
	Name    string
	Size    int64
	Mode    fs.FileMode
	ModTime time.Time
	IsDir   bool
	Version string
}

// DirEntry is the minimal directory entry the read tool needs.
type DirEntry struct {
	Name  string
	IsDir bool
}

// Option configures a file tool at construction time.
type Option func(*toolOptions)

type toolOptions struct {
	fs WorkspaceFS
}

// WithFS injects a WorkspaceFS backend. When omitted, tools default to
// OSWorkspaceFS (local filesystem) — so existing NewRead/NewWrite/NewEdit
// callers keep their current behaviour unchanged.
func WithFS(fs WorkspaceFS) Option {
	return func(o *toolOptions) { o.fs = fs }
}

// resolveFS applies opts and returns the chosen backend, defaulting to OS.
func resolveFS(opts []Option) WorkspaceFS {
	var o toolOptions
	for _, opt := range opts {
		opt(&o)
	}
	if o.fs == nil {
		return OSWorkspaceFS{}
	}
	return o.fs
}

// OSWorkspaceFS is the default WorkspaceFS backed by the local filesystem.
// It ignores ctx and leaves FileInfo.Version empty.
type OSWorkspaceFS struct{}

var _ WorkspaceFS = OSWorkspaceFS{}

func (OSWorkspaceFS) Stat(_ context.Context, path string) (FileInfo, error) {
	info, err := os.Stat(path)
	if err != nil {
		return FileInfo{}, err
	}
	return osFileInfo(info), nil
}

func (OSWorkspaceFS) Open(_ context.Context, path string) (io.ReadCloser, error) {
	return os.Open(path)
}

func (OSWorkspaceFS) ReadFile(_ context.Context, path string) ([]byte, error) {
	return os.ReadFile(path)
}

func (OSWorkspaceFS) ReadDir(_ context.Context, path string) ([]DirEntry, error) {
	entries, err := os.ReadDir(path)
	if err != nil {
		return nil, err
	}
	out := make([]DirEntry, len(entries))
	for i, e := range entries {
		out[i] = DirEntry{Name: e.Name(), IsDir: e.IsDir()}
	}
	return out, nil
}

func (OSWorkspaceFS) WriteFile(_ context.Context, path string, data []byte, perm fs.FileMode) error {
	return os.WriteFile(path, data, perm)
}

func (OSWorkspaceFS) MkdirAll(_ context.Context, path string, perm fs.FileMode) error {
	return os.MkdirAll(path, perm)
}

func osFileInfo(info os.FileInfo) FileInfo {
	return FileInfo{
		Name:    info.Name(),
		Size:    info.Size(),
		Mode:    info.Mode(),
		ModTime: info.ModTime(),
		IsDir:   info.IsDir(),
	}
}

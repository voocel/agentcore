package tools

import (
	"context"
	"io"
	"io/fs"
	"os"
	"time"

	"github.com/voocel/agentcore"
	"github.com/voocel/agentcore/task"
)

// Workspace is where the coding tools work, and what they share. Its methods
// make the tools; the tools of one Workspace share its state.
type Workspace struct {
	// Dir is the directory relative paths resolve against, unless the
	// context of a call carries another; see WithCwd. It is not a sandbox:
	// an absolute path reaches any file.
	Dir string
	// FS is the file backend of read, write and edit; nil is the local
	// filesystem. bash, glob, grep and ls work on the local filesystem.
	FS FS
	// Files records what the model read, so that write and edit refuse a
	// file it has not read, or that changed since; nil checks nothing.
	Files *FileReadState
	// Tasks runs the commands bash runs in the background; nil offers no
	// background mode.
	Tasks *task.Runtime
}

// Tools returns read, write, edit, bash, glob, grep and ls.
func (w Workspace) Tools() []agentcore.Tool {
	return []agentcore.Tool{w.Read(), w.Write(), w.Edit(), w.Bash(), w.Glob(), w.Grep(), w.Ls()}
}

// dir is the directory relative paths resolve against in a call ctx runs.
func (w Workspace) dir(ctx context.Context) string {
	if cwd := CwdFromContext(ctx); cwd != "" {
		return cwd
	}
	return w.Dir
}

// fs is the file backend.
func (w Workspace) fs() FS {
	if w.FS == nil {
		return OSFS{}
	}
	return w.FS
}

// cwdKey carries the working directory of the calls a context runs.
type cwdKey struct{}

// WithCwd returns ctx carrying cwd, the working directory of the tool calls
// that run with it, which overrides their Workspace's Dir unless it returns
// "". It is consulted at each use, so that it may change while a run is
// under way, as when the agent enters a git worktree. The innermost wins.
func WithCwd(ctx context.Context, cwd func() string) context.Context {
	return context.WithValue(ctx, cwdKey{}, cwd)
}

// CwdFromContext returns the working directory ctx carries, or "".
func CwdFromContext(ctx context.Context) string {
	fn, _ := ctx.Value(cwdKey{}).(func() string)
	if fn == nil {
		return ""
	}
	return fn()
}

// FS is a file backend: the local filesystem, an editor serving its unsaved
// buffers, a remote host. Paths are absolute; those of a backend other than
// the local filesystem are best slash-separated ("/work/file.txt"), which
// the tools keep on every platform. A backend doing I/O over a transport
// honors ctx.
type FS interface {
	Stat(ctx context.Context, path string) (FileInfo, error)
	// Open streams the file, as to read some lines of it or sniff its type.
	Open(ctx context.Context, path string) (io.ReadCloser, error)
	ReadFile(ctx context.Context, path string) ([]byte, error)
	// ReadDir lists a directory, not recursively.
	ReadDir(ctx context.Context, path string) ([]DirEntry, error)
	// WriteFile writes data to path, replacing what was there.
	WriteFile(ctx context.Context, path string, data []byte, perm fs.FileMode) error
	MkdirAll(ctx context.Context, path string, perm fs.FileMode) error
}

// FileInfo describes a file of an FS.
type FileInfo struct {
	Name    string
	Size    int64
	Mode    fs.FileMode
	ModTime time.Time
	IsDir   bool
	// Version identifies the content, as a hash or an etag does, for a
	// backend whose ModTime does not tell its changes, as unsaved editor
	// buffers. When the read and the current FileInfo both have one, write
	// and edit compare it instead of ModTime to detect a stale file.
	Version string
}

// DirEntry is an entry of a directory.
type DirEntry struct {
	Name  string
	IsDir bool
}

// OSFS is the local filesystem, an FS with no Version.
type OSFS struct{}

func (OSFS) Stat(_ context.Context, path string) (FileInfo, error) {
	info, err := os.Stat(path)
	if err != nil {
		return FileInfo{}, err
	}
	return FileInfo{Name: info.Name(), Size: info.Size(), Mode: info.Mode(), ModTime: info.ModTime(), IsDir: info.IsDir()}, nil
}

func (OSFS) Open(_ context.Context, path string) (io.ReadCloser, error) { return os.Open(path) }

func (OSFS) ReadFile(_ context.Context, path string) ([]byte, error) { return os.ReadFile(path) }

func (OSFS) ReadDir(_ context.Context, path string) ([]DirEntry, error) {
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

func (OSFS) WriteFile(_ context.Context, path string, data []byte, perm fs.FileMode) error {
	return os.WriteFile(path, data, perm)
}

func (OSFS) MkdirAll(_ context.Context, path string, perm fs.FileMode) error {
	return os.MkdirAll(path, perm)
}

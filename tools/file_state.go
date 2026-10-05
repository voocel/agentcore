package tools

import (
	"context"
	"encoding/json"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/voocel/agentcore"
)

// FileReadStamp records the state of a file as the model read it. The write
// and edit tools consult these stamps in their Check and Run to enforce:
//
//   - read-before-write: a file must be read before it is overwritten.
//   - no-stale-write: the file must not have been modified externally
//     between the last read and the write attempt.
//
// Partial is true when the model did not see all of the file: it read from
// an offset, or the read stopped at its limit. A partial read is enough to
// edit, which changes only text the model quotes, but not to overwrite the
// file, whose rest the model has not seen.
//
// A successful write or edit refreshes the stamp, so the LLM can keep
// changing a file it just changed without reading it again.
//
// Version is the backend-defined content token recorded at read time (see
// FileInfo.Version). It is empty for the OS backend; Write/Edit
// fall back to comparing Mtime when either side's Version is empty.
type FileReadStamp struct {
	Mtime   time.Time
	Version string
	Partial bool
}

// FileReadState records what the model read, by absolute path: see
// Workspace.Files.
type FileReadState struct {
	mu sync.RWMutex
	m  map[string]FileReadStamp
}

func NewFileReadState() *FileReadState {
	return &FileReadState{m: make(map[string]FileReadStamp)}
}

func (s *FileReadState) Get(path string) (FileReadStamp, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	v, ok := s.m[path]
	return v, ok
}

func (s *FileReadState) Set(path string, stamp FileReadStamp) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.m[path] = stamp
}

// recordWrite refreshes the stamp of path after the LLM wrote it: all of it
// when whole, otherwise (an edit) only the text it quoted, so whether it has
// seen the whole file carries over. A file that cannot be stated keeps its
// old stamp, and the next write or edit asks for a fresh read.
func (s *FileReadState) recordWrite(ctx context.Context, fs FS, path string, whole bool) {
	if s == nil {
		return
	}
	info, err := fs.Stat(ctx, path)
	if err != nil {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.m[path] = FileReadStamp{Mtime: info.ModTime, Version: info.Version, Partial: !whole && s.m[path].Partial}
}

// stampMatches reports whether the file described by info is unchanged since
// the read recorded in stamp. When both sides carry a non-empty Version
// (a backend content token), it compares Version, so that a backend serving
// unsaved buffers detects a change of their content; otherwise, as for the
// OS backend, whose Version is always empty, it compares mtimes.
func stampMatches(stamp FileReadStamp, info FileInfo) bool {
	if stamp.Version != "" && info.Version != "" {
		return stamp.Version == info.Version
	}
	return info.ModTime.Equal(stamp.Mtime)
}

// FileOps lists, as tagged sections, the files the read, write and edit
// calls in history touched: those only read, and those modified. It is empty
// when there were none. A compaction appends it to its summary, so that the
// files the replaced messages worked on survive it.
func FileOps(history []agentcore.Message) string {
	read, modified := map[string]bool{}, map[string]bool{}
	for _, m := range history {
		for _, call := range m.ToolCalls() {
			var args struct {
				FilePath string `json:"file_path"`
			}
			if json.Unmarshal([]byte(call.Arguments), &args) != nil || args.FilePath == "" {
				continue
			}
			switch call.Name {
			case "read":
				read[args.FilePath] = true
			case "write", "edit":
				modified[args.FilePath] = true
			}
		}
	}
	var out string
	if files := sorted(read, modified); len(files) > 0 {
		out += "\n\n<read-files>\n" + strings.Join(files, "\n") + "\n</read-files>"
	}
	if files := sorted(modified, nil); len(files) > 0 {
		out += "\n\n<modified-files>\n" + strings.Join(files, "\n") + "\n</modified-files>"
	}
	return out
}

// sorted lists the keys of set that are not in except.
func sorted(set, except map[string]bool) []string {
	var files []string
	for f := range set {
		if !except[f] {
			files = append(files, f)
		}
	}
	slices.Sort(files)
	return files
}

package tools

import (
	"sync"
	"time"
)

// FileReadStamp records when a file was read and its mtime at that moment.
// Write and edit tools consult these stamps via their Validator to enforce:
//
//   - read-before-write: a file must be read before it is overwritten.
//   - no-stale-write: the file must not have been modified externally
//     between the last read and the write attempt.
//
// Partial is true when the read used offset/limit. A partial read does not
// satisfy read-before-write — the LLM may not know about content outside
// the slice it read.
//
// Version is the backend-defined content token recorded at read time (see
// WorkspaceFS.FileInfo.Version). It is empty for the OS backend; Write/Edit
// fall back to comparing Mtime when either side's Version is empty.
type FileReadStamp struct {
	ReadAt  time.Time
	Mtime   time.Time
	Version string
	Partial bool
}

// FileReadState is the session-scoped store of FileReadStamp keyed by
// absolute path. Read, Write, and Edit tools share one instance per session,
// passed at construction time.
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

// Reset drops all recorded stamps. Called by Session on /clear, Reset, and
// session switch so the LLM never writes based on stamps from a read it no
// longer has in its conversation history.
func (s *FileReadState) Reset() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.m = make(map[string]FileReadStamp)
}

// stampMatches reports whether the file described by info is unchanged since
// the read recorded in stamp. When both sides carry a non-empty Version
// (a backend content token), it compares Version; otherwise it falls back to
// mtime equality — so the OS backend (Version always empty) keeps its existing
// behaviour, while backends serving unsaved buffers get content-accurate
// stale-write detection.
func stampMatches(stamp FileReadStamp, info FileInfo) bool {
	if stamp.Version != "" && info.Version != "" {
		return stamp.Version == info.Version
	}
	return info.ModTime.Equal(stamp.Mtime)
}

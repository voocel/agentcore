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
type FileReadStamp struct {
	ReadAt  time.Time
	Mtime   time.Time
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

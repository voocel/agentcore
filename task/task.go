// Package task is a unified registry for background tasks. Tools (shell
// commands, sub-agent dispatch, etc.) register their long-running work via a
// shared Runtime so callers can list and cancel them through one surface.
package task

import (
	"encoding/json"
	"fmt"
	"slices"
	"strconv"
	"sync"
	"time"

	"github.com/voocel/agentcore"
)

// Status represents the lifecycle state of a background task.
type Status string

const (
	Running   Status = "running"
	Completed Status = "completed"
	Failed    Status = "failed"
)

// Type distinguishes the origin of a background task.
type Type string

const (
	TypeShell    Type = "shell"
	TypeSubAgent Type = "subagent"
)

// Entry is the unified representation of any background task.
// Both shell tools and sub-agent tools register entries through a shared
// Runtime.
type Entry struct {
	ID          string
	Type        Type
	Description string
	Status      Status
	StartedAt   time.Time
	EndedAt     time.Time
	OutputFile  string // path to output file on disk
	Error       string
	ExitCode    int    // shell: process exit code
	ToolCount   int    // number of tool calls executed
	cancel      func() // unexported: only Stop()/StopAll() should use this

	// Shell-specific
	PID     int
	Command string

	// SubAgent-specific
	Agent     string
	Prompt    string // original task prompt
	TokensIn  int
	TokensOut int
}

// SetCancel sets the cancellation function for this task entry.
// Called during registration; only Stop()/StopAll() invoke it.
func (e *Entry) SetCancel(fn func()) {
	e.cancel = fn
}

// Runtime is a unified registry for background tasks.
type Runtime struct {
	mu    sync.Mutex
	seq   int
	tasks map[string]*Entry
}

// NewRuntime creates an empty runtime.
func NewRuntime() *Runtime {
	return &Runtime{tasks: make(map[string]*Entry)}
}

// NextID generates a sequential task ID with the given prefix (e.g. "shell", "bg").
func (r *Runtime) NextID(prefix string) string {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.seq++
	return prefix + "-" + strconv.Itoa(r.seq)
}

// Register adds a task entry. The caller is responsible for populating all fields.
func (r *Runtime) Register(entry *Entry) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.tasks[entry.ID] = entry
}

// Get returns a snapshot of a single task, or nil if not found.
func (r *Runtime) Get(id string) *Entry {
	r.mu.Lock()
	defer r.mu.Unlock()
	e, ok := r.tasks[id]
	if !ok {
		return nil
	}
	return copyEntry(e)
}

// List returns snapshots of all tasks, sorted by creation time.
func (r *Runtime) List() []Entry {
	r.mu.Lock()
	defer r.mu.Unlock()
	out := make([]Entry, 0, len(r.tasks))
	for _, e := range r.tasks {
		out = append(out, *copyEntry(e))
	}
	slices.SortFunc(out, func(a, b Entry) int {
		return a.StartedAt.Compare(b.StartedAt)
	})
	return out
}

// Stop cancels a running task by ID. Returns true if a running task was
// found and its cancel function was invoked. The background goroutine is
// responsible for writing the terminal status and EndedAt after observing
// the cancellation.
func (r *Runtime) Stop(id string) bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	e, ok := r.tasks[id]
	if !ok || e.Status != Running {
		return false
	}
	if e.cancel != nil {
		e.cancel()
	}
	return true
}

// StopAll cancels all running tasks. Returns the number cancelled.
func (r *Runtime) StopAll() int {
	r.mu.Lock()
	defer r.mu.Unlock()
	count := 0
	for _, e := range r.tasks {
		if e.Status == Running {
			if e.cancel != nil {
				e.cancel()
			}
			count++
		}
	}
	return count
}

// Update applies a mutation function to a task entry under the lock.
// Returns false if the task is not found.
func (r *Runtime) Update(id string, fn func(e *Entry)) bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	e, ok := r.tasks[id]
	if !ok {
		return false
	}
	fn(e)
	return true
}

func copyEntry(e *Entry) *Entry {
	cp := *e
	cp.cancel = nil // snapshots don't carry cancel
	return &cp
}

// ---------------------------------------------------------------------------
// Notification
// ---------------------------------------------------------------------------

// CompletedTag is the XML-style wrapper tag used in the AgentMessage emitted
// by ToAgentMessage().
const CompletedTag = "background-task-completed"

// Notification is the JSON payload delivered to the calling agent when a
// background task finishes.
type Notification struct {
	TaskID      string `json:"task_id"`
	Type        Type   `json:"type"`
	Status      Status `json:"status"`
	Description string `json:"description,omitempty"`
	OutputFile  string `json:"output_file,omitempty"`
	Error       string `json:"error,omitempty"`
	ExitCode    *int   `json:"exit_code,omitempty"`
	Command     string `json:"command,omitempty"`
	Agent       string `json:"agent,omitempty"`
}

// NotificationFromEntry converts a task entry into a notification payload.
func NotificationFromEntry(e *Entry) Notification {
	if e == nil {
		return Notification{}
	}
	n := Notification{
		TaskID:      e.ID,
		Type:        e.Type,
		Status:      e.Status,
		Description: e.Description,
		OutputFile:  e.OutputFile,
		Error:       e.Error,
		Command:     e.Command,
		Agent:       e.Agent,
	}
	if e.Type == TypeShell && e.Status != Running {
		exitCode := e.ExitCode
		n.ExitCode = &exitCode
	}
	return n
}

// ToAgentMessage wraps the notification as a user-role AgentMessage that the
// parent agent can consume as a follow-up.
func (n Notification) ToAgentMessage() agentcore.AgentMessage {
	data, err := json.Marshal(n)
	if err != nil {
		return agentcore.UserMsg(fmt.Sprintf("<%s>\n{\"task_id\":%q,\"status\":\"failed\",\"error\":%q}\n</%s>", CompletedTag, n.TaskID, err.Error(), CompletedTag))
	}
	return agentcore.UserMsg(fmt.Sprintf("<%s>\n%s\n</%s>", CompletedTag, string(data), CompletedTag))
}

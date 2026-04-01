package agentcore

import (
	"slices"
	"strconv"
	"sync"
	"time"
)

// TaskStatus represents the lifecycle state of a background task.
type TaskStatus string

const (
	TaskRunning   TaskStatus = "running"
	TaskCompleted TaskStatus = "completed"
	TaskFailed    TaskStatus = "failed"
)

// TaskType distinguishes the origin of a background task.
type TaskType string

const (
	TaskTypeShell    TaskType = "shell"
	TaskTypeSubAgent TaskType = "subagent"
)

// BackgroundTaskEntry is the unified representation of any background task.
// Both BashTool (shell commands) and SubAgentTool (background agents) register
// tasks here through a shared TaskRuntime.
type BackgroundTaskEntry struct {
	ID          string
	Type        TaskType
	Description string
	Status      TaskStatus
	StartedAt   time.Time
	EndedAt     time.Time
	OutputFile  string     // path to output file on disk
	Error       string
	ExitCode    int        // shell: process exit code
	ToolCount   int        // number of tool calls executed
	cancel      func()     // unexported: only Stop()/StopAll() should use this

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
func (e *BackgroundTaskEntry) SetCancel(fn func()) {
	e.cancel = fn
}

// TaskRuntime is a unified registry for background tasks.
// Tools register their background work here; the Agent exposes a single
// Tasks()/StopTask()/StopAllTasks() surface to callers.
type TaskRuntime struct {
	mu    sync.Mutex
	seq   int
	tasks map[string]*BackgroundTaskEntry
}

// NewTaskRuntime creates an empty task runtime.
func NewTaskRuntime() *TaskRuntime {
	return &TaskRuntime{tasks: make(map[string]*BackgroundTaskEntry)}
}

// NextID generates a sequential task ID with the given prefix (e.g. "shell", "bg").
func (r *TaskRuntime) NextID(prefix string) string {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.seq++
	return prefix + "-" + strconv.Itoa(r.seq)
}

// Register adds a task entry. The caller is responsible for populating all fields.
func (r *TaskRuntime) Register(entry *BackgroundTaskEntry) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.tasks[entry.ID] = entry
}

// Get returns a snapshot of a single task, or nil if not found.
func (r *TaskRuntime) Get(id string) *BackgroundTaskEntry {
	r.mu.Lock()
	defer r.mu.Unlock()
	e, ok := r.tasks[id]
	if !ok {
		return nil
	}
	return copyEntry(e)
}

// List returns snapshots of all tasks, sorted by creation order (ascending seq ID).
func (r *TaskRuntime) List() []BackgroundTaskEntry {
	r.mu.Lock()
	defer r.mu.Unlock()
	out := make([]BackgroundTaskEntry, 0, len(r.tasks))
	for _, e := range r.tasks {
		out = append(out, *copyEntry(e))
	}
	slices.SortFunc(out, func(a, b BackgroundTaskEntry) int {
		return a.StartedAt.Compare(b.StartedAt)
	})
	return out
}

// Stop cancels a running task by ID. Returns true if a running task was found
// and its cancel function was invoked. The background goroutine is responsible
// for writing the terminal status (failed/completed) and EndedAt after it
// observes the cancellation. This avoids a race where Stop() writes the
// terminal state and the goroutine then skips its final metadata update.
func (r *TaskRuntime) Stop(id string) bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	e, ok := r.tasks[id]
	if !ok || e.Status != TaskRunning {
		return false
	}
	if e.cancel != nil {
		e.cancel()
	}
	return true
}

// StopAll cancels all running tasks. Returns the number cancelled.
func (r *TaskRuntime) StopAll() int {
	r.mu.Lock()
	defer r.mu.Unlock()
	count := 0
	for _, e := range r.tasks {
		if e.Status == TaskRunning {
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
func (r *TaskRuntime) Update(id string, fn func(e *BackgroundTaskEntry)) bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	e, ok := r.tasks[id]
	if !ok {
		return false
	}
	fn(e)
	return true
}

func copyEntry(e *BackgroundTaskEntry) *BackgroundTaskEntry {
	cp := *e
	cp.cancel = nil // snapshots don't carry cancel
	return &cp
}


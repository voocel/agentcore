// Package task runs work in the background as tasks the model can follow:
// each has an ID, a status and an output file, and a message announces it
// when it ends. The bash tool runs its background commands as tasks, the
// subagent tool its background agents; [Runtime.Tools] lets the model read
// and stop them.
package task

import (
	"cmp"
	"context"
	"fmt"
	"io"
	"os"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/voocel/agentcore"
)

// Status is where a task stands.
type Status string

const (
	Running   Status = "running"
	Completed Status = "completed"
	Failed    Status = "failed"
	// Killed is a task stopped before it ended.
	Killed Status = "killed"
)

// Type is the kind of work a task does.
type Type string

const (
	TypeShell    Type = "shell"
	TypeSubAgent Type = "subagent"
)

// outputExt is the extension of the output file of a task of type t: shell
// output is text, a sub-agent's a JSON line per message.
func (t Type) outputExt() string {
	if t == TypeSubAgent {
		return ".jsonl"
	}
	return ".log"
}

// Entry is the state of a task.
type Entry struct {
	ID          string
	Type        Type
	Description string
	Status      Status
	StartedAt   time.Time
	EndedAt     time.Time
	// OutputFile is the file the task's output goes to.
	OutputFile string
	// Error is why a Failed task failed.
	Error string

	// A shell command.
	Command  string
	ExitCode int

	// A sub-agent.
	Agent     string
	Run       string // the run's Spawn.ID, as "explore#3"
	Prompt    string
	Result    string // the agent's last response
	ToolCount int
	TokensIn  int
	TokensOut int
}

// Runtime runs and tracks tasks.
type Runtime struct {
	dir    string
	notify func(agentcore.Message)

	mu     sync.Mutex
	idle   *sync.Cond // signaled when no task is active
	seq    int
	active int
	jobs   map[string]*job
}

type job struct {
	entry   Entry
	cancel  context.CancelFunc
	stopped bool
	done    chan struct{} // closed once the task ended and was announced
}

// NewRuntime returns a Runtime that writes the output of each task to a file
// in dir, the system's temporary directory if "", and hands notify the
// message announcing each task that ended; nil drops them.
func NewRuntime(dir string, notify func(agentcore.Message)) *Runtime {
	r := &Runtime{dir: dir, notify: notify, jobs: map[string]*job{}}
	r.idle = sync.NewCond(&r.mu)
	return r
}

// Task is a task, as its work sees it.
type Task struct {
	ID string
	// Output is the task's output file.
	Output io.Writer
	r      *Runtime
	j      *job
}

// Update changes the task's entry as its work goes on, such as to count what
// it did. The Runtime sets the status when the task ends.
func (t *Task) Update(fn func(*Entry)) {
	t.r.mu.Lock()
	defer t.r.mu.Unlock()
	fn(&t.j.entry)
}

// Start runs work as a new task, described by e: its Type, its Description
// and the fields of its type. It returns the task's entry at once; work runs
// on a goroutine of its own, with a context that keeps ctx's values but not
// its cancellation and is cancelled when the task is stopped. When work
// returns, the task ends: Killed if it was stopped, Failed if work failed,
// Completed otherwise; then notify hears of it.
func (r *Runtime) Start(ctx context.Context, e Entry, work func(ctx context.Context, t *Task) error) (Entry, error) {
	r.mu.Lock()
	r.seq++
	e.ID = fmt.Sprintf("%s-%d", e.Type, r.seq)
	r.mu.Unlock()

	out, err := r.create(e)
	if err != nil {
		return Entry{}, fmt.Errorf("task: create output: %w", err)
	}
	e.Status, e.StartedAt, e.OutputFile = Running, time.Now(), out.Name()
	ctx, cancel := context.WithCancel(context.WithoutCancel(ctx))
	j := &job{entry: e, cancel: cancel, done: make(chan struct{})}

	r.mu.Lock()
	r.jobs[e.ID] = j
	r.active++
	r.mu.Unlock()

	go func() {
		err := work(ctx, &Task{ID: e.ID, Output: out, r: r, j: j})
		if cerr := out.Close(); cerr != nil && err == nil {
			err = fmt.Errorf("close output: %w", cerr)
		}
		cancel()
		r.end(j, err)
	}()
	return e, nil
}

// create opens the output file of e.
func (r *Runtime) create(e Entry) (*os.File, error) {
	if r.dir == "" {
		return os.CreateTemp("", "agentcore-"+e.ID+"-*"+e.Type.outputExt())
	}
	if err := os.MkdirAll(r.dir, 0o755); err != nil {
		return nil, err
	}
	// A unique name: a Runtime resumed in the same dir counts from 1 again.
	return os.CreateTemp(r.dir, e.ID+"-*"+e.Type.outputExt())
}

// end settles the task j, which work ended with err, and announces it.
func (r *Runtime) end(j *job, err error) {
	r.mu.Lock()
	e := &j.entry
	e.EndedAt = time.Now()
	switch {
	case j.stopped:
		e.Status = Killed
	case err != nil:
		e.Status, e.Error = Failed, err.Error()
	default:
		e.Status = Completed
	}
	ended := *e
	r.mu.Unlock()

	if r.notify != nil {
		r.notify(notification(ended))
	}

	r.mu.Lock()
	close(j.done)
	r.active--
	if r.active == 0 {
		r.idle.Broadcast()
	}
	r.mu.Unlock()
}

// Get returns the entry of task id.
func (r *Runtime) Get(id string) (Entry, bool) {
	r.mu.Lock()
	defer r.mu.Unlock()
	j, ok := r.jobs[id]
	if !ok {
		return Entry{}, false
	}
	return j.entry, true
}

// List returns the entries of all tasks, oldest first.
func (r *Runtime) List() []Entry {
	r.mu.Lock()
	defer r.mu.Unlock()
	out := make([]Entry, 0, len(r.jobs))
	for _, j := range r.jobs {
		out = append(out, j.entry)
	}
	slices.SortFunc(out, func(a, b Entry) int { return a.StartedAt.Compare(b.StartedAt) })
	return out
}

// Stop stops task id, which ends Killed. It reports whether the task was
// running.
func (r *Runtime) Stop(id string) bool {
	r.mu.Lock()
	j, ok := r.jobs[id]
	running := ok && j.entry.Status == Running && !j.stopped
	if running {
		j.stopped = true
	}
	r.mu.Unlock()
	if running {
		j.cancel()
	}
	return running
}

// StopAll stops every running task and returns how many it stopped.
func (r *Runtime) StopAll() int {
	r.mu.Lock()
	var ids []string
	for id, j := range r.jobs {
		if j.entry.Status == Running {
			ids = append(ids, id)
		}
	}
	r.mu.Unlock()
	n := 0
	for _, id := range ids {
		if r.Stop(id) {
			n++
		}
	}
	return n
}

// Active returns how many tasks have not ended, their announcement
// included.
func (r *Runtime) Active() int {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.active
}

// Wait returns once no task is active, counting those started while it
// waits, such as by a task of its own.
func (r *Runtime) Wait() {
	r.mu.Lock()
	defer r.mu.Unlock()
	for r.active > 0 {
		r.idle.Wait()
	}
}

// WaitFor returns the entry of task id once it ended and was announced, or
// ctx's error first.
func (r *Runtime) WaitFor(ctx context.Context, id string) (Entry, error) {
	r.mu.Lock()
	j, ok := r.jobs[id]
	r.mu.Unlock()
	if !ok {
		return Entry{}, fmt.Errorf("task %s not found", id)
	}
	select {
	case <-j.done:
	case <-ctx.Done():
		return Entry{}, ctx.Err()
	}
	e, _ := r.Get(id)
	return e, nil
}

// KindNotification marks the message announcing that a task ended.
const KindNotification = "task_notification"

// notification is the message announcing that the task e ended, for the
// agent that started it to go on with. Its fields are escaped, so that no
// output can forge one.
func notification(e Entry) agentcore.Message {
	var b strings.Builder
	field := func(name, value string) {
		if value != "" {
			fmt.Fprintf(&b, "<%s>%s</%s>\n", name, escape(value), name)
		}
	}
	b.WriteString("<task-notification>\n")
	field("task-id", e.ID)
	field("type", string(e.Type))
	field("status", string(e.Status))
	field("summary", summary(e))
	switch e.Type {
	case TypeShell:
		field("command", e.Command)
	case TypeSubAgent:
		field("agent", e.Agent)
		field("result", e.Result)
		field("usage", fmt.Sprintf("%d tokens, %d tool uses, %s", e.TokensIn+e.TokensOut, e.ToolCount, e.EndedAt.Sub(e.StartedAt).Round(time.Second)))
	}
	field("error", e.Error)
	field("output-file", e.OutputFile)
	b.WriteString("</task-notification>")
	m := agentcore.UserText(b.String())
	m.Kind = KindNotification
	return m
}

var escape = strings.NewReplacer("&", "&amp;", "<", "&lt;", ">", "&gt;").Replace

// summary says in a line what became of the task e.
func summary(e Entry) string {
	what := fmt.Sprintf("Task %q", cmp.Or(e.Description, e.Command))
	if e.Type == TypeSubAgent {
		what = fmt.Sprintf("Agent %q", cmp.Or(e.Description, e.Agent))
	}
	switch e.Status {
	case Completed:
		return what + " completed"
	case Failed:
		return what + " failed: " + e.Error
	case Killed:
		return what + " was stopped"
	}
	return what + " is " + string(e.Status)
}

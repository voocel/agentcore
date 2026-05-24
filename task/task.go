// Package task is a unified registry for background tasks. Tools (shell
// commands, sub-agent dispatch, etc.) register their long-running work via a
// shared Runtime so callers can list and cancel them through one surface.
package task

import (
	"context"
	"fmt"
	"slices"
	"strconv"
	"strings"
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
	Killed    Status = "killed"
)

// Type distinguishes the origin of a background task.
type Type string

const (
	TypeShell    Type = "shell"
	TypeSubAgent Type = "subagent"
	TypeTeammate Type = "teammate"
)

// Identity is the team-aware identity carried by a teammate Entry.
// nil for non-teammate entries (shell, subagent). Lives in this package
// rather than agentcore/team to avoid a task↔team import cycle: team needs
// to read/store Entry.Identity, and task needs to recognise teammate-typed
// entries for lifecycle handling.
type Identity struct {
	AgentID         string // "researcher@my-team"
	AgentName       string // "researcher"
	TeamName        string
	Color           string
	ParentSessionID string
}

// MaxAgentDepth caps how deep sub-agents may nest. The main agent is depth 0;
// a sub-agent it spawns is depth 1; a sub-agent inside that would be depth 2.
// Today the subagent tool is filtered out of every sub-agent's pool, so depth
// is structurally capped at 1 — this constant is defense in depth for when
// team support lands and peer agents gain a spawn channel.
//
// 5 is high enough to permit legitimate fan-out but low enough to catch
// runaway recursion before it burns through tokens.
const MaxAgentDepth = 5

// depthKey is the ctx key used to thread an agent's depth into the goroutine
// running it. Sub-agent spawn paths read the parent's depth from ctx, increment
// it, and pass the new ctx into the child's loop.
type depthKey struct{}

// DepthFromContext returns the current agent's depth — 0 for the main agent,
// n+1 inside a depth-n sub-agent. Unset (top-level) ctx returns 0.
func DepthFromContext(ctx context.Context) int {
	if v, ok := ctx.Value(depthKey{}).(int); ok {
		return v
	}
	return 0
}

// WithDepth threads `depth` into ctx so a sub-agent spawned via this ctx can
// read its caller's depth and reject overly deep nesting.
func WithDepth(ctx context.Context, depth int) context.Context {
	return context.WithValue(ctx, depthKey{}, depth)
}

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
	Depth       int    // nesting depth: 1 for tasks spawned by the main agent; n+1 inside a depth-n task. See MaxAgentDepth.
	cancel      func() // unexported: only Stop()/StopAll() should use this

	// Shell-specific
	PID     int
	Command string

	// SubAgent-specific (also used by teammate, which shares the underlying runAgent loop)
	Agent     string
	Prompt    string // original task prompt
	Result    string // final assistant text from the sub-agent
	TokensIn  int
	TokensOut int

	// Teammate-specific. Identity is non-nil iff Type == TypeTeammate.
	// IsIdle flips true between turns when the teammate is waiting in its mailbox
	// channel for the next message.
	Identity *Identity
	IsIdle   bool

	// pendingMessages queues parent→child messages delivered to the sub-agent
	// at the next steering tick. Guarded by Runtime.mu (not a per-entry mutex)
	// so copyEntry can keep its `cp := *e` pattern without copying a lock.
	// Drain is called once per turn at most — Runtime.mu contention is a
	// non-issue at that rate.
	//
	// Note: teammates do NOT use pendingMessages — they use the team mailbox
	// channel for delivery (see agentcore/team). This field stays subagent-only.
	pendingMessages []string
}

// IsTerminal reports whether a task has reached a terminal state.
func (s Status) IsTerminal() bool {
	return s == Completed || s == Failed || s == Killed
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

// AppendStatus enumerates the outcomes of AppendPending. We return a status
// instead of (bool, error) because the caller (a tool) needs to format three
// distinct user-visible messages: "queued", "not found", and "task already
// finished" — each implies a different follow-up.
type AppendStatus int

const (
	AppendOK       AppendStatus = iota // message queued
	AppendNotFound                     // no entry with that ID
	AppendTerminal                     // entry exists but already in terminal state
)

// AppendPending queues a message for delivery to a sub-agent's next steering
// tick. Returns the outcome so the caller can choose its error wording.
//
// Terminal tasks are NOT auto-resumed here — resuming from disk transcripts
// is a later capability. Today the parent agent gets AppendTerminal back and
// reports the failure to the user.
func (r *Runtime) AppendPending(id, msg string) AppendStatus {
	r.mu.Lock()
	defer r.mu.Unlock()
	e, ok := r.tasks[id]
	if !ok {
		return AppendNotFound
	}
	if e.Status.IsTerminal() {
		return AppendTerminal
	}
	e.pendingMessages = append(e.pendingMessages, msg)
	return AppendOK
}

// DrainPending returns and clears the queued messages for the given task.
// Returns nil when there is nothing queued so callers can short-circuit.
// Called from the sub-agent loop's steering hook — see subagent.runAgent.
func (r *Runtime) DrainPending(id string) []string {
	r.mu.Lock()
	defer r.mu.Unlock()
	e, ok := r.tasks[id]
	if !ok || len(e.pendingMessages) == 0 {
		return nil
	}
	out := e.pendingMessages
	e.pendingMessages = nil
	return out
}

func copyEntry(e *Entry) *Entry {
	cp := *e
	cp.cancel = nil          // snapshots don't carry cancel
	cp.pendingMessages = nil // snapshots don't carry the live queue
	return &cp
}

// ---------------------------------------------------------------------------
// Notification
// ---------------------------------------------------------------------------

// CompletedTag is the XML-style wrapper tag used in the AgentMessage emitted
// by ToAgentMessage().
const CompletedTag = "background-task-completed"

// Usage summarises a sub-agent's resource consumption for the notification.
type Usage struct {
	TotalTokens int
	ToolUses    int
	DurationMs  int64
}

// Notification is the payload delivered to the calling agent when a
// background task finishes. The parent agent receives this wrapped as XML so
// it can both react immediately to <result> and read <output-file> on demand.
type Notification struct {
	TaskID      string
	Type        Type
	Status      Status
	Description string

	// SubAgent
	Agent  string
	Result string // final assistant text — lets parent continue without IO
	Usage  *Usage

	// Shell
	Command  string
	ExitCode *int

	// Common
	OutputFile string // disk path to the full transcript / log
	Error      string
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
		Result:      e.Result,
	}
	if e.Type == TypeShell && e.Status != Running {
		exitCode := e.ExitCode
		n.ExitCode = &exitCode
	}
	if e.Type == TypeSubAgent && e.Status.IsTerminal() {
		n.Usage = &Usage{
			TotalTokens: e.TokensIn + e.TokensOut,
			ToolUses:    e.ToolCount,
			DurationMs:  durationMs(e.StartedAt, e.EndedAt),
		}
	}
	return n
}

func durationMs(start, end time.Time) int64 {
	if start.IsZero() || end.IsZero() {
		return 0
	}
	return end.Sub(start).Milliseconds()
}

// ToAgentMessage wraps the notification as a user-role AgentMessage that the
// parent agent can consume as a follow-up. The XML is hand-formatted with
// nested elements (instead of JSON-in-XML) because LLMs parse structured XML
// reliably and the format mirrors patterns Claude already recognises.
func (n Notification) ToAgentMessage() agentcore.AgentMessage {
	var b strings.Builder
	fmt.Fprintf(&b, "<%s>\n", CompletedTag)
	fmt.Fprintf(&b, "<task-id>%s</task-id>\n", n.TaskID)
	fmt.Fprintf(&b, "<type>%s</type>\n", n.Type)
	fmt.Fprintf(&b, "<status>%s</status>\n", n.Status)
	fmt.Fprintf(&b, "<summary>%s</summary>\n", summarise(n))

	if n.Agent != "" {
		fmt.Fprintf(&b, "<agent>%s</agent>\n", n.Agent)
	}
	if n.Command != "" {
		fmt.Fprintf(&b, "<command>%s</command>\n", n.Command)
	}
	if n.ExitCode != nil {
		fmt.Fprintf(&b, "<exit-code>%d</exit-code>\n", *n.ExitCode)
	}
	if n.Error != "" {
		fmt.Fprintf(&b, "<error>%s</error>\n", n.Error)
	}
	if n.Result != "" {
		fmt.Fprintf(&b, "<result>%s</result>\n", n.Result)
	}
	if n.Usage != nil {
		fmt.Fprintf(&b, "<usage><total-tokens>%d</total-tokens><tool-uses>%d</tool-uses><duration-ms>%d</duration-ms></usage>\n",
			n.Usage.TotalTokens, n.Usage.ToolUses, n.Usage.DurationMs)
	}
	if n.OutputFile != "" {
		fmt.Fprintf(&b, "<output-file>%s</output-file>\n", n.OutputFile)
	}
	fmt.Fprintf(&b, "</%s>", CompletedTag)
	return agentcore.UserMsg(b.String())
}

func summarise(n Notification) string {
	label := n.Description
	if label == "" {
		switch n.Type {
		case TypeSubAgent:
			label = n.Agent
		case TypeShell:
			label = n.Command
		}
	}
	noun := "Task"
	if n.Type == TypeSubAgent {
		noun = fmt.Sprintf("Agent %q", label)
		label = ""
	}
	switch n.Status {
	case Completed:
		if label != "" {
			return fmt.Sprintf("%s %q completed", noun, label)
		}
		return noun + " completed"
	case Failed:
		reason := n.Error
		if reason == "" {
			reason = "unknown error"
		}
		if label != "" {
			return fmt.Sprintf("%s %q failed: %s", noun, label, reason)
		}
		return fmt.Sprintf("%s failed: %s", noun, reason)
	case Killed:
		if label != "" {
			return fmt.Sprintf("%s %q was stopped", noun, label)
		}
		return noun + " was stopped"
	default:
		return string(n.Status)
	}
}

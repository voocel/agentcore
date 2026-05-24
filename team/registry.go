package team

import (
	"errors"
	"slices"
	"sync"
	"time"
)

// Sentinel errors returned by Registry. Callers compare via errors.Is so
// they can present sensible messages to the user / agent.
var (
	ErrTeamExists   = errors.New("team: a team is already active in this session")
	ErrNoTeam       = errors.New("team: no active team")
	ErrAgentExists  = errors.New("team: agent name already registered")
	ErrUnknownAgent = errors.New("team: unknown agent")
	ErrReservedName = errors.New("team: agent name is reserved")
)

// Context is a snapshot of the active team's metadata. Returned by Team() so
// callers don't hold a lock or alias internal state.
type Context struct {
	Name        string
	Description string
	LeaderName  string
	CreatedAt   time.Time
}

// Registry holds session-wide team state: the active team (if any), the
// agent-name → task-ID lookup, and one Mailbox per registered agent. Single
// instance lives on the codebot session; passed into the SendMessage tool and
// teammate runners.
//
// Concurrency: a single sync.Mutex guards everything. Mailboxes themselves
// carry their own lock, so the registry lock is only held during membership
// changes and lookups — never across Send/Drain/Wait.
type Registry struct {
	mu         sync.Mutex
	team       *Context
	mailboxes  map[string]*Mailbox
	nameToTask map[string]string
}

// NewRegistry creates an empty registry with no active team.
func NewRegistry() *Registry {
	return &Registry{
		mailboxes:  make(map[string]*Mailbox),
		nameToTask: make(map[string]string),
	}
}

// CreateTeam activates a new team and auto-registers the leader with the
// reserved TeamLeadName. Returns ErrTeamExists if a team is already active.
// The leader's taskID is the caller's session ID — that's how the leader
// looks itself up later when draining its own inbox.
func (r *Registry) CreateTeam(name, description, leaderTaskID string) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	if r.team != nil {
		return ErrTeamExists
	}
	r.team = &Context{
		Name:        name,
		Description: description,
		LeaderName:  TeamLeadName,
		CreatedAt:   time.Now(),
	}
	r.mailboxes[TeamLeadName] = NewMailbox()
	r.nameToTask[TeamLeadName] = leaderTaskID
	return nil
}

// DeleteTeam tears down the active team: closes every mailbox and clears the
// registries. Returns ErrNoTeam if there is no active team. Idempotent only
// inside the same call — a second DeleteTeam returns ErrNoTeam.
func (r *Registry) DeleteTeam() error {
	r.mu.Lock()
	if r.team == nil {
		r.mu.Unlock()
		return ErrNoTeam
	}
	mboxes := r.mailboxes
	r.mailboxes = make(map[string]*Mailbox)
	r.nameToTask = make(map[string]string)
	r.team = nil
	r.mu.Unlock()

	// Close mailboxes outside the lock — Close acquires its own mutex.
	for _, mb := range mboxes {
		mb.Close()
	}
	return nil
}

// Team returns a snapshot of the active team, or nil if none.
func (r *Registry) Team() *Context {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.team == nil {
		return nil
	}
	cp := *r.team
	return &cp
}

// HasTeam reports whether a team is currently active.
func (r *Registry) HasTeam() bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.team != nil
}

// RegisterAgent adds a teammate to the active team. Returns:
//   - ErrNoTeam if no team is active
//   - ErrReservedName if name == TeamLeadName (leader is auto-registered)
//   - ErrAgentExists if name is already taken
//
// On success the teammate gets a fresh Mailbox and its taskID is recorded
// so SendMessage can route by name.
func (r *Registry) RegisterAgent(name, taskID string) error {
	if name == TeamLeadName {
		return ErrReservedName
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.team == nil {
		return ErrNoTeam
	}
	if _, ok := r.nameToTask[name]; ok {
		return ErrAgentExists
	}
	r.mailboxes[name] = NewMailbox()
	r.nameToTask[name] = taskID
	return nil
}

// UnregisterAgent removes a teammate, closing its mailbox. Returns
// ErrUnknownAgent if name was never registered (or was already removed).
// Leader cannot be unregistered while the team is active — use DeleteTeam.
func (r *Registry) UnregisterAgent(name string) error {
	if name == TeamLeadName {
		return ErrReservedName
	}
	r.mu.Lock()
	mb, ok := r.mailboxes[name]
	if !ok {
		r.mu.Unlock()
		return ErrUnknownAgent
	}
	delete(r.mailboxes, name)
	delete(r.nameToTask, name)
	r.mu.Unlock()

	mb.Close()
	return nil
}

// TaskID returns the registered task ID for name. The bool is false if
// name is not registered. Used by SendMessage to find the target entry.
func (r *Registry) TaskID(name string) (string, bool) {
	r.mu.Lock()
	defer r.mu.Unlock()
	id, ok := r.nameToTask[name]
	return id, ok
}

// Mailbox returns the mailbox for name, or nil if not registered. Callers
// that need to wait/send must hold no other lock — Mailbox manages its own.
func (r *Registry) Mailbox(name string) *Mailbox {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.mailboxes[name]
}

// AgentNames returns the registered teammate names (including the leader)
// in stable alphabetical order. Used for broadcasts and listings.
func (r *Registry) AgentNames() []string {
	r.mu.Lock()
	out := make([]string, 0, len(r.nameToTask))
	for name := range r.nameToTask {
		out = append(out, name)
	}
	r.mu.Unlock()
	slices.Sort(out)
	return out
}

// TeammateNames returns registered teammate names excluding the leader.
// Used by broadcast logic (leader sending to "*" should not echo to self).
func (r *Registry) TeammateNames() []string {
	all := r.AgentNames()
	out := make([]string, 0, len(all))
	for _, n := range all {
		if n != TeamLeadName {
			out = append(out, n)
		}
	}
	return out
}

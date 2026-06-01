package team

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/voocel/agentcore"
	"github.com/voocel/agentcore/task"
)

// SpawnConfig configures a new teammate spawn.
type SpawnConfig struct {
	// AgentName is the teammate's display name. Must be unique within the team
	// and may not be the reserved TeamLeadName. Used in agentId (`name@team`)
	// and as the routing key for SendMessage.
	AgentName string

	// InitialPrompt is the leader's first message to the teammate.
	InitialPrompt string

	// History, if non-empty, seeds the teammate's conversation with prior
	// messages before the first turn — the first Execute receives History
	// as the prefix of its input, with InitialPrompt appended as the new
	// user turn. Used by a harness to resume a teammate with its earlier
	// transcript after a restart; nil means a fresh teammate. Pure
	// mechanism: Spawn/Run still do no I/O of their own.
	History []agentcore.AgentMessage

	// Description is an optional one-line summary shown in transcripts/UI.
	Description string

	// Color is an optional UI color assigned to this teammate.
	Color string

	// ParentSessionID identifies the leader's session — recorded on Identity
	// for analytics / transcript correlation. May be empty.
	ParentSessionID string

	// Registry is the team registry; must have an active team.
	Registry *Registry

	// TaskRT is the shared task runtime where the teammate's Entry will live.
	TaskRT *task.Runtime

	// Execute drives one agent turn; passed straight through to Run.
	Execute TurnExecutor

	// Protocol is the application-supplied format + policy hook bundle.
	// Forwarded to Run; see ProtocolHooks for per-field defaults.
	Protocol ProtocolHooks

	// Depth is the agent nesting depth at spawn time. The runner does not use
	// it directly but it's recorded on the Entry so MaxAgentDepth checks at
	// callsites can verify before calling Spawn.
	Depth int

	// OnExit, if non-nil, is invoked once the teammate goroutine is about to
	// return — AFTER the Entry has been updated to its terminal status and
	// the name has been unregistered. The err argument is whatever Run
	// returned (nil on graceful completion, context.Canceled on shutdown,
	// or a propagated error). Callers use this to release per-agent
	// resources (event hubs, transcripts, etc.) without polling the runtime.
	OnExit func(err error)
}

// SpawnResult is returned synchronously after the teammate is registered and
// its goroutine has been launched. The teammate itself runs in the background;
// callers use TaskRT.Stop(TaskID) or Registry.UnregisterAgent(AgentName) to
// terminate it.
type SpawnResult struct {
	TaskID  string
	AgentID string // name@team
}

// Spawn registers a teammate Entry, allocates its mailbox + name binding, and
// launches the long-lived Run goroutine. Returns immediately after launch.
//
// On failure (no active team, duplicate name, depth exceeded), nothing is
// registered and no goroutine is started. On success the goroutine owns the
// Entry's terminal state — it will mark Completed/Failed and unregister the
// name when Run returns.
func Spawn(parentCtx context.Context, cfg SpawnConfig) (*SpawnResult, error) {
	if err := validateSpawnConfig(cfg); err != nil {
		return nil, err
	}

	teamCtx := cfg.Registry.Team()
	if teamCtx == nil {
		return nil, ErrNoTeam
	}

	taskID := cfg.TaskRT.NextID("tm")
	identity := &task.Identity{
		AgentID:         cfg.AgentName + "@" + teamCtx.Name,
		AgentName:       cfg.AgentName,
		TeamName:        teamCtx.Name,
		Color:           cfg.Color,
		ParentSessionID: cfg.ParentSessionID,
	}

	if err := cfg.Registry.RegisterAgent(cfg.AgentName, taskID); err != nil {
		return nil, err
	}

	runCtx, cancel := context.WithCancel(parentCtx)

	entry := &task.Entry{
		ID:          taskID,
		Type:        task.TypeTeammate,
		Description: shortDescription(cfg.AgentName, cfg.InitialPrompt),
		Status:      task.Running,
		StartedAt:   time.Now(),
		Depth:       cfg.Depth,
		Identity:    identity,
		Agent:       cfg.AgentName,
		Prompt:      cfg.InitialPrompt,
	}
	entry.SetCancel(cancel)
	cfg.TaskRT.Register(entry)

	go func() {
		defer cancel()

		err := Run(runCtx, RunConfig{
			Identity:      identity,
			InitialPrompt: cfg.InitialPrompt,
			History:       cfg.History,
			Description:   cfg.Description,
			Registry:      cfg.Registry,
			TaskRT:        cfg.TaskRT,
			TaskID:        taskID,
			Execute:       cfg.Execute,
			Protocol:      cfg.Protocol,
		})

		cfg.TaskRT.Update(taskID, func(e *task.Entry) {
			e.EndedAt = time.Now()
			e.IsIdle = false
			if err != nil && !errors.Is(err, context.Canceled) {
				e.Status = task.Failed
				e.Error = err.Error()
				return
			}
			// Cancelled or graceful — both land in Completed. Killed is set
			// externally by TaskRT.Stop, which we don't override.
			if e.Status == task.Running {
				e.Status = task.Completed
			}
		})

		// Best-effort cleanup. UnregisterAgent returns ErrUnknownAgent if the
		// team was torn down first (DeleteTeam closed our mailbox); that's
		// fine — the resource was already reclaimed.
		_ = cfg.Registry.UnregisterAgent(cfg.AgentName)

		if cfg.OnExit != nil {
			cfg.OnExit(err)
		}
	}()

	return &SpawnResult{
		TaskID:  taskID,
		AgentID: identity.AgentID,
	}, nil
}

func validateSpawnConfig(cfg SpawnConfig) error {
	switch {
	case cfg.AgentName == "":
		return errors.New("team.Spawn: AgentName is required")
	case cfg.AgentName == TeamLeadName:
		return fmt.Errorf("team.Spawn: %w", ErrReservedName)
	case cfg.Registry == nil:
		return errors.New("team.Spawn: Registry is required")
	case cfg.TaskRT == nil:
		return errors.New("team.Spawn: TaskRT is required")
	case cfg.Execute == nil:
		return errors.New("team.Spawn: Execute is required")
	}
	return nil
}

func shortDescription(name, prompt string) string {
	const limit = 60
	if len(prompt) > limit {
		return name + ": " + prompt[:limit] + "..."
	}
	return name + ": " + prompt
}

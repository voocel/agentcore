package team

import (
	"errors"
	"slices"
	"testing"
)

func TestRegistry_CreateTeamRegistersLeader(t *testing.T) {
	r := NewRegistry()
	if err := r.CreateTeam("alpha", "test team", "leader-task-1"); err != nil {
		t.Fatalf("CreateTeam: %v", err)
	}
	if !r.HasTeam() {
		t.Fatal("HasTeam = false after CreateTeam")
	}

	ctx := r.Team()
	if ctx == nil || ctx.Name != "alpha" || ctx.LeaderName != TeamLeadName {
		t.Errorf("Team() = %+v", ctx)
	}

	id, ok := r.TaskID(TeamLeadName)
	if !ok || id != "leader-task-1" {
		t.Errorf("TaskID(leader) = (%q, %v), want (leader-task-1, true)", id, ok)
	}
	if r.Mailbox(TeamLeadName) == nil {
		t.Error("leader mailbox not created")
	}
}

func TestRegistry_CreateTeamDuplicateFails(t *testing.T) {
	r := NewRegistry()
	_ = r.CreateTeam("alpha", "", "leader-1")

	err := r.CreateTeam("beta", "", "leader-2")
	if !errors.Is(err, ErrTeamExists) {
		t.Errorf("CreateTeam returned %v, want ErrTeamExists", err)
	}
}

func TestRegistry_RegisterAgent(t *testing.T) {
	r := NewRegistry()
	_ = r.CreateTeam("alpha", "", "leader-1")

	if err := r.RegisterAgent("researcher", "task-2"); err != nil {
		t.Fatalf("RegisterAgent: %v", err)
	}

	id, ok := r.TaskID("researcher")
	if !ok || id != "task-2" {
		t.Errorf("TaskID(researcher) = (%q, %v)", id, ok)
	}
	if r.Mailbox("researcher") == nil {
		t.Error("researcher mailbox missing")
	}

	names := r.AgentNames()
	if !slices.Equal(names, []string{"researcher", TeamLeadName}) {
		t.Errorf("AgentNames = %v", names)
	}
	if got := r.TeammateNames(); !slices.Equal(got, []string{"researcher"}) {
		t.Errorf("TeammateNames = %v", got)
	}
}

func TestRegistry_RegisterAgentErrors(t *testing.T) {
	r := NewRegistry()

	if err := r.RegisterAgent("researcher", "t1"); !errors.Is(err, ErrNoTeam) {
		t.Errorf("RegisterAgent without team = %v, want ErrNoTeam", err)
	}

	_ = r.CreateTeam("alpha", "", "leader")
	if err := r.RegisterAgent(TeamLeadName, "t1"); !errors.Is(err, ErrReservedName) {
		t.Errorf("RegisterAgent(leader) = %v, want ErrReservedName", err)
	}

	_ = r.RegisterAgent("researcher", "t1")
	if err := r.RegisterAgent("researcher", "t2"); !errors.Is(err, ErrAgentExists) {
		t.Errorf("Duplicate register = %v, want ErrAgentExists", err)
	}
}

func TestRegistry_UnregisterAgentClosesMailbox(t *testing.T) {
	r := NewRegistry()
	_ = r.CreateTeam("alpha", "", "leader")
	_ = r.RegisterAgent("researcher", "t1")

	mb := r.Mailbox("researcher")
	if mb == nil {
		t.Fatal("missing mailbox")
	}

	if err := r.UnregisterAgent("researcher"); err != nil {
		t.Fatalf("UnregisterAgent: %v", err)
	}

	if err := mb.Send(Message{From: "leader", Text: "ping"}); !errors.Is(err, ErrClosed) {
		t.Errorf("Send after unregister = %v, want ErrClosed", err)
	}

	if _, ok := r.TaskID("researcher"); ok {
		t.Error("TaskID still returns true after unregister")
	}
	if r.Mailbox("researcher") != nil {
		t.Error("Mailbox still returns non-nil after unregister")
	}
}

func TestRegistry_UnregisterErrors(t *testing.T) {
	r := NewRegistry()
	_ = r.CreateTeam("alpha", "", "leader")

	if err := r.UnregisterAgent(TeamLeadName); !errors.Is(err, ErrReservedName) {
		t.Errorf("Unregister(leader) = %v, want ErrReservedName", err)
	}
	if err := r.UnregisterAgent("ghost"); !errors.Is(err, ErrUnknownAgent) {
		t.Errorf("Unregister(ghost) = %v, want ErrUnknownAgent", err)
	}
}

func TestRegistry_DeleteTeamClosesAllMailboxes(t *testing.T) {
	r := NewRegistry()
	_ = r.CreateTeam("alpha", "", "leader")
	_ = r.RegisterAgent("researcher", "t1")
	_ = r.RegisterAgent("tester", "t2")

	leaderBox := r.Mailbox(TeamLeadName)
	researcherBox := r.Mailbox("researcher")

	if err := r.DeleteTeam(); err != nil {
		t.Fatalf("DeleteTeam: %v", err)
	}

	if r.HasTeam() {
		t.Error("HasTeam = true after DeleteTeam")
	}
	if err := leaderBox.Send(Message{From: "x", Text: "y"}); !errors.Is(err, ErrClosed) {
		t.Error("leader mailbox not closed")
	}
	if err := researcherBox.Send(Message{From: "x", Text: "y"}); !errors.Is(err, ErrClosed) {
		t.Error("researcher mailbox not closed")
	}

	if err := r.DeleteTeam(); !errors.Is(err, ErrNoTeam) {
		t.Errorf("Second DeleteTeam = %v, want ErrNoTeam", err)
	}
}

func TestRegistry_RenameTeamSucceedsOnEmptyTeam(t *testing.T) {
	r := NewRegistry()
	_ = r.CreateTeam("default", "first description", "leader")

	if err := r.RenameTeam("alpha", "auth refactor"); err != nil {
		t.Fatalf("RenameTeam: %v", err)
	}

	ctx := r.Team()
	if ctx == nil || ctx.Name != "alpha" || ctx.Description != "auth refactor" {
		t.Errorf("Team() = %+v, want name=alpha desc='auth refactor'", ctx)
	}

	// Empty newName must not wipe the existing name.
	if err := r.RenameTeam("", "updated desc"); err != nil {
		t.Fatalf("RenameTeam desc-only: %v", err)
	}
	ctx = r.Team()
	if ctx.Name != "alpha" || ctx.Description != "updated desc" {
		t.Errorf("desc-only rename clobbered name: %+v", ctx)
	}
}

func TestRegistry_RenameTeamRejectsWithMembers(t *testing.T) {
	r := NewRegistry()
	_ = r.CreateTeam("default", "", "leader")
	_ = r.RegisterAgent("researcher", "t1")

	err := r.RenameTeam("alpha", "")
	if !errors.Is(err, ErrTeamHasMembers) {
		t.Errorf("RenameTeam with teammate = %v, want ErrTeamHasMembers", err)
	}
	if ctx := r.Team(); ctx.Name != "default" {
		t.Errorf("name changed despite error: %q", ctx.Name)
	}
}

func TestRegistry_RenameTeamNoTeam(t *testing.T) {
	r := NewRegistry()
	err := r.RenameTeam("alpha", "")
	if !errors.Is(err, ErrNoTeam) {
		t.Errorf("RenameTeam with no team = %v, want ErrNoTeam", err)
	}
}

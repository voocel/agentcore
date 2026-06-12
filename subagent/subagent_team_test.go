package subagent

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"
)

// captureSpawner records the request it received and returns a fixed result
// (or error). Lets each test focus on routing/validation without spinning up
// the full agentcore.AgentLoop machinery in this package.
type captureSpawner struct {
	calls []TeamSpawnRequest
	res   *TeamSpawnResult
	err   error
}

func (c *captureSpawner) fn() TeamSpawner {
	return func(_ context.Context, req TeamSpawnRequest) (*TeamSpawnResult, error) {
		c.calls = append(c.calls, req)
		if c.err != nil {
			return nil, c.err
		}
		return c.res, nil
	}
}

func newToolWithAgent(name string) *Tool {
	return New(Config{Name: name, Description: "test agent for spawn routing"})
}

func TestSubAgentTool_TeamSpawn_RoutesWhenTeamNameSet(t *testing.T) {
	tl := newToolWithAgent("researcher")
	sp := &captureSpawner{res: &TeamSpawnResult{TaskID: "tm-1", AgentID: "alice@alpha"}}
	tl.SetTeamSpawner(sp.fn())

	args, _ := json.Marshal(map[string]any{
		"agent":     "researcher",
		"task":      "find clue",
		"team_name": "alpha",
		"name":      "alice",
		"color":     "blue",
	})
	out, err := tl.Execute(context.Background(), args)
	if err != nil {
		t.Fatalf("Execute returned error: %v", err)
	}
	var result map[string]any
	if err := json.Unmarshal(out, &result); err != nil {
		t.Fatalf("unmarshal result: %v", err)
	}
	if result["task_id"] != "tm-1" || result["agent_id"] != "alice@alpha" {
		t.Errorf("unexpected result: %+v", result)
	}
	if len(sp.calls) != 1 {
		t.Fatalf("expected exactly 1 spawn call, got %d", len(sp.calls))
	}
	req := sp.calls[0]
	if req.Name != "alice" || req.TeamName != "alpha" || req.InitialPrompt != "find clue" || req.Color != "blue" {
		t.Errorf("request not threaded correctly: %+v", req)
	}
	if req.Config.Name != "researcher" {
		t.Errorf("Config.Name = %q, want %q", req.Config.Name, "researcher")
	}
}

func TestSubAgentTool_TeamSpawn_NameDefaultsToAgent(t *testing.T) {
	tl := newToolWithAgent("researcher")
	sp := &captureSpawner{res: &TeamSpawnResult{TaskID: "tm-1", AgentID: "researcher@alpha"}}
	tl.SetTeamSpawner(sp.fn())

	args, _ := json.Marshal(map[string]any{
		"agent":     "researcher",
		"task":      "find clue",
		"team_name": "alpha",
	})
	if _, err := tl.Execute(context.Background(), args); err != nil {
		t.Fatalf("Execute: %v", err)
	}
	if sp.calls[0].Name != "researcher" {
		t.Errorf("Name should default to agent name when omitted, got %q", sp.calls[0].Name)
	}
}

func TestSubAgentTool_TeamSpawn_RejectsUnknownAgent(t *testing.T) {
	tl := newToolWithAgent("researcher")
	tl.SetTeamSpawner((&captureSpawner{}).fn())

	args, _ := json.Marshal(map[string]any{
		"agent":     "ghost",
		"task":      "x",
		"team_name": "alpha",
	})
	out, err := tl.Execute(context.Background(), args)
	if err != nil {
		t.Fatalf("Execute: %v", err)
	}
	var got map[string]any
	if err := json.Unmarshal(out, &got); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if errStr, _ := got["error"].(string); !strings.Contains(errStr, `unknown agent "ghost"`) {
		t.Errorf("expected unknown-agent error, got %v", got)
	}
}

func TestSubAgentTool_TeamSpawn_RequiresSpawner(t *testing.T) {
	tl := newToolWithAgent("researcher")
	// No SetTeamSpawner — team mode should refuse cleanly.

	args, _ := json.Marshal(map[string]any{
		"agent":     "researcher",
		"task":      "x",
		"team_name": "alpha",
	})
	_, err := tl.Execute(context.Background(), args)
	if err == nil || !strings.Contains(err.Error(), "team spawn is not configured") {
		t.Errorf("expected configuration error, got %v", err)
	}
}

func TestSubAgentTool_TeamSpawn_MutexWithBackground(t *testing.T) {
	tl := newToolWithAgent("researcher")
	tl.SetTeamSpawner((&captureSpawner{}).fn())

	args, _ := json.Marshal(map[string]any{
		"agent":      "researcher",
		"task":       "x",
		"team_name":  "alpha",
		"background": true,
	})
	_, err := tl.Execute(context.Background(), args)
	if err == nil || !strings.Contains(err.Error(), "mutually exclusive") {
		t.Errorf("expected mutex error, got %v", err)
	}
}

func TestSubAgentTool_TeamSpawn_RequiresAgentAndTask(t *testing.T) {
	tl := newToolWithAgent("researcher")
	tl.SetTeamSpawner((&captureSpawner{}).fn())

	args, _ := json.Marshal(map[string]any{
		"team_name": "alpha",
	})
	_, err := tl.Execute(context.Background(), args)
	if err == nil || !strings.Contains(err.Error(), "agent + task") {
		t.Errorf("expected missing-agent/task error, got %v", err)
	}
}

func TestSubAgentTool_TeamSpawn_PropagatesSpawnerError(t *testing.T) {
	tl := newToolWithAgent("researcher")
	tl.SetTeamSpawner((&captureSpawner{err: errors.New("team already at capacity")}).fn())

	args, _ := json.Marshal(map[string]any{
		"agent":     "researcher",
		"task":      "x",
		"team_name": "alpha",
	})
	_, err := tl.Execute(context.Background(), args)
	if err == nil || !strings.Contains(err.Error(), "at capacity") {
		t.Errorf("expected spawner error propagated, got %v", err)
	}
}

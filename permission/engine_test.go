package permission

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func newTestEngine(t *testing.T, workspace string, mode Mode, rules *RuleSet) *Engine {
	t.Helper()
	store, err := NewStore(filepath.Join(t.TempDir(), "approvals.json"))
	if err != nil {
		t.Fatalf("NewStore: %v", err)
	}
	return NewEngine(EngineConfig{
		Workspace: workspace,
		Mode:      mode,
		Rules:     rules,
		Store:     store,
	})
}

func toolReq(name string, args map[string]any) Request {
	raw, _ := json.Marshal(args)
	return Request{
		ToolName: name,
		Args:     raw,
	}
}

func TestBalancedReadAllowed(t *testing.T) {
	engine := newTestEngine(t, t.TempDir(), ModeBalanced, nil)

	decision, err := engine.Decide(context.Background(), toolReq("read", map[string]any{"path": "a.txt"}))
	if err != nil {
		t.Fatalf("Decide: %v", err)
	}
	if decision == nil || decision.Kind != DecisionAllow || !decision.Allowed() {
		t.Fatalf("expected auto allow, got %#v", decision)
	}
}

func TestBalancedWriteDeniedWithoutApprover(t *testing.T) {
	engine := newTestEngine(t, t.TempDir(), ModeBalanced, nil)

	decision, err := engine.Decide(context.Background(), toolReq("write", map[string]any{"path": "a.txt"}))
	if err != nil {
		t.Fatalf("Decide: %v", err)
	}
	if decision == nil || decision.Kind != DecisionDeny || decision.Allowed() {
		t.Fatalf("expected deny without approver, got %#v", decision)
	}
}

func TestOutsideRootsAllowSessionDegradesToAllowOnce(t *testing.T) {
	workspace := t.TempDir()
	outside := filepath.Join(t.TempDir(), "outside.txt")
	engine := newTestEngine(t, workspace, ModeBalanced, nil)
	engine.SetApprover(func(context.Context, Prompt) (Choice, error) {
		return ChoiceAllowSession, nil
	})

	decision, err := engine.Decide(context.Background(), Request{
		ToolName: "read",
		Args:     mustJSON(t, map[string]any{"path": outside}),
	})
	if err != nil {
		t.Fatalf("Decide: %v", err)
	}
	if decision == nil || decision.Kind != DecisionAllowOnce || !decision.Prompted || !decision.OutsideRoots {
		t.Fatalf("expected prompted allow-once for outside roots, got %#v", decision)
	}
}

func TestWriteViaSymlinkEscapeDenied(t *testing.T) {
	workspace := t.TempDir()
	outsideDir := filepath.Join(t.TempDir(), "outside")
	if err := os.MkdirAll(outsideDir, 0o755); err != nil {
		t.Fatalf("MkdirAll: %v", err)
	}
	link := filepath.Join(workspace, "link")
	if err := os.Symlink(outsideDir, link); err != nil {
		t.Fatalf("Symlink: %v", err)
	}

	engine := newTestEngine(t, workspace, ModeBalanced, nil)
	decision, err := engine.Decide(context.Background(), toolReq("write", map[string]any{"path": "link/escape.txt"}))
	if err != nil {
		t.Fatalf("Decide: %v", err)
	}
	if decision == nil || decision.Allowed() {
		t.Fatalf("expected denial for symlink escape, got %#v", decision)
	}
}

func TestPlanModeDeniesPersistedWrite(t *testing.T) {
	workspace := t.TempDir()
	engine := newTestEngine(t, workspace, ModeBalanced, nil)
	engine.SetApprover(func(context.Context, Prompt) (Choice, error) {
		return ChoiceAllowAlways, nil
	})

	req := toolReq("write", map[string]any{"path": "a.txt"})
	first, err := engine.Decide(context.Background(), req)
	if err != nil {
		t.Fatalf("Decide first: %v", err)
	}
	if first == nil || first.Kind != DecisionAllowAlways {
		t.Fatalf("expected allow-always, got %#v", first)
	}

	engine.SetPlanMode(true)
	second, err := engine.Decide(context.Background(), req)
	if err != nil {
		t.Fatalf("Decide second: %v", err)
	}
	if second == nil || second.Kind != DecisionDeny {
		t.Fatalf("expected plan mode denial, got %#v", second)
	}
}

func TestPlanModeOutsideReadStillPrompts(t *testing.T) {
	workspace := t.TempDir()
	outside := filepath.Join(t.TempDir(), "outside.txt")
	engine := newTestEngine(t, workspace, ModeBalanced, nil)
	engine.SetPlanMode(true)
	engine.SetApprover(func(context.Context, Prompt) (Choice, error) {
		return ChoiceAllowOnce, nil
	})

	decision, err := engine.Decide(context.Background(), Request{
		ToolName: "read",
		Args:     mustJSON(t, map[string]any{"path": outside}),
	})
	if err != nil {
		t.Fatalf("Decide: %v", err)
	}
	if decision == nil || decision.Kind != DecisionAllowOnce || !decision.Prompted || !decision.OutsideRoots {
		t.Fatalf("expected outside-roots prompt in plan mode, got %#v", decision)
	}
}

func TestMetadataOverrideForCustomTool(t *testing.T) {
	engine := newTestEngine(t, t.TempDir(), ModeBalanced, nil)

	decision, err := engine.Decide(context.Background(), Request{
		ToolName: "custom_lookup",
		Metadata: Metadata{
			Capability:  CapabilityRead,
			SummaryHint: "custom lookup",
			KeyPrefix:   "custom",
		},
	})
	if err != nil {
		t.Fatalf("Decide: %v", err)
	}
	if decision == nil || decision.Kind != DecisionAllow || decision.Key != "custom:custom_lookup" {
		t.Fatalf("expected metadata-driven allow, got %#v", decision)
	}
}

func mustJSON(t *testing.T, v any) json.RawMessage {
	t.Helper()
	raw, err := json.Marshal(v)
	if err != nil {
		t.Fatalf("json.Marshal: %v", err)
	}
	return raw
}

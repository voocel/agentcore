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

func TestPlanModeRejectsInternalByDefault(t *testing.T) {
	engine := newTestEngine(t, t.TempDir(), ModeBalanced, nil)
	engine.SetPlanMode(true)

	decision, err := engine.Decide(context.Background(), Request{
		ToolName: "custom_internal",
		Metadata: Metadata{Capability: CapabilityInternal},
	})
	if err != nil {
		t.Fatalf("Decide: %v", err)
	}
	if decision == nil || decision.Kind != DecisionDeny || decision.Source != DecisionSourceMode {
		t.Fatalf("expected plan-mode denial, got %#v", decision)
	}
}

func TestPlanModeAllowsConfiguredAllowlistedTool(t *testing.T) {
	store, err := NewStore(filepath.Join(t.TempDir(), "approvals.json"))
	if err != nil {
		t.Fatalf("NewStore: %v", err)
	}
	engine := NewEngine(EngineConfig{
		Workspace:            t.TempDir(),
		Mode:                 ModeBalanced,
		Store:                store,
		PlanModeAllowedTools: []string{"custom_plan_control"},
	})
	engine.SetPlanMode(true)

	decision, err := engine.Decide(context.Background(), Request{
		ToolName: "custom_plan_control",
		Metadata: Metadata{Capability: CapabilityInternal},
	})
	if err != nil {
		t.Fatalf("Decide: %v", err)
	}
	if decision == nil || !decision.Allowed() {
		t.Fatalf("expected allow in plan mode, got %#v", decision)
	}
}

func TestPlanModeAllowlistDoesNotPromoteWriteCapability(t *testing.T) {
	store, err := NewStore(filepath.Join(t.TempDir(), "approvals.json"))
	if err != nil {
		t.Fatalf("NewStore: %v", err)
	}
	engine := NewEngine(EngineConfig{
		Workspace:            t.TempDir(),
		Mode:                 ModeBalanced,
		Store:                store,
		PlanModeAllowedTools: []string{"custom_writer"},
	})
	engine.SetPlanMode(true)

	decision, err := engine.Decide(context.Background(), Request{
		ToolName: "custom_writer",
		Args:     mustJSON(t, map[string]any{"path": "out.txt"}),
		Metadata: Metadata{Capability: CapabilityWrite},
	})
	if err != nil {
		t.Fatalf("Decide: %v", err)
	}
	if decision == nil || decision.Kind != DecisionDeny || decision.Source != DecisionSourceMode {
		t.Fatalf("expected plan-mode denial for write capability, got %#v", decision)
	}
}

func TestPlanModeExecAllowedHookPasses(t *testing.T) {
	store, err := NewStore(filepath.Join(t.TempDir(), "approvals.json"))
	if err != nil {
		t.Fatalf("NewStore: %v", err)
	}
	engine := NewEngine(EngineConfig{
		Workspace: t.TempDir(),
		Mode:      ModeBalanced,
		Store:     store,
		PlanModeExecAllowed: func(req Request) bool {
			return req.ToolName == "bash"
		},
	})
	engine.SetPlanMode(true)
	// Approver would fire if plan-mode short-circuit didn't kick in — fail loudly
	// rather than silently auto-allow.
	engine.SetApprover(func(context.Context, Prompt) (Choice, error) {
		t.Fatalf("plan-mode exec allow-list should bypass approver")
		return ChoiceDeny, nil
	})

	decision, err := engine.Decide(context.Background(), Request{
		ToolName: "bash",
		Args:     mustJSON(t, map[string]any{"command": "git status"}),
	})
	if err != nil {
		t.Fatalf("Decide: %v", err)
	}
	if decision == nil || !decision.Allowed() || decision.Source != DecisionSourceMode {
		t.Fatalf("expected plan-mode allow for bash, got %#v", decision)
	}
}

func TestPlanModeExecWithoutHookStillDenies(t *testing.T) {
	engine := newTestEngine(t, t.TempDir(), ModeBalanced, nil)
	engine.SetPlanMode(true)

	decision, err := engine.Decide(context.Background(), Request{
		ToolName: "bash",
		Args:     mustJSON(t, map[string]any{"command": "git status"}),
	})
	if err != nil {
		t.Fatalf("Decide: %v", err)
	}
	if decision == nil || decision.Kind != DecisionDeny || decision.Source != DecisionSourceMode {
		t.Fatalf("expected plan-mode denial without exec hook, got %#v", decision)
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

func TestInternalReadablePathSilentlyAllowed(t *testing.T) {
	workspace := t.TempDir()
	memDir := t.TempDir()
	engine := newTestEngine(t, workspace, ModeBalanced, nil)
	engine.SetFilesystemRoots(FilesystemRoots{
		InternalReadable: []string{memDir},
	})
	engine.SetApprover(func(context.Context, Prompt) (Choice, error) {
		t.Fatalf("approver must not be called for internal path")
		return ChoiceDeny, nil
	})

	target := filepath.Join(memDir, "MEMORY.md")
	decision, err := engine.Decide(context.Background(), Request{
		ToolName: "read",
		Args:     mustJSON(t, map[string]any{"path": target}),
	})
	if err != nil {
		t.Fatalf("Decide: %v", err)
	}
	if decision == nil || !decision.Allowed() {
		t.Fatalf("expected silent allow, got %#v", decision)
	}
	if decision.Source != DecisionSourceInternal {
		t.Fatalf("expected DecisionSourceInternal, got %q", decision.Source)
	}
	if decision.Prompted {
		t.Fatalf("expected no prompt for internal path, got %#v", decision)
	}
	if decision.OutsideRoots {
		t.Fatalf("internal path must not be marked outside roots, got %#v", decision)
	}
}

func TestInternalWritablePathSilentlyAllowedInBalancedMode(t *testing.T) {
	workspace := t.TempDir()
	memDir := t.TempDir()
	engine := newTestEngine(t, workspace, ModeBalanced, nil)
	engine.SetFilesystemRoots(FilesystemRoots{
		InternalWritable: []string{memDir},
	})
	engine.SetApprover(func(context.Context, Prompt) (Choice, error) {
		t.Fatalf("approver must not be called for internal write path")
		return ChoiceDeny, nil
	})

	target := filepath.Join(memDir, "MEMORY.md")
	decision, err := engine.Decide(context.Background(), Request{
		ToolName: "write",
		Args:     mustJSON(t, map[string]any{"path": target}),
	})
	if err != nil {
		t.Fatalf("Decide: %v", err)
	}
	if decision == nil || !decision.Allowed() || decision.Source != DecisionSourceInternal {
		t.Fatalf("expected silent allow via internal path, got %#v", decision)
	}
}

func TestInternalWritableImpliesReadable(t *testing.T) {
	workspace := t.TempDir()
	memDir := t.TempDir()
	engine := newTestEngine(t, workspace, ModeBalanced, nil)
	engine.SetFilesystemRoots(FilesystemRoots{
		InternalWritable: []string{memDir},
	})
	engine.SetApprover(func(context.Context, Prompt) (Choice, error) {
		t.Fatalf("approver must not be called when write implies read")
		return ChoiceDeny, nil
	})

	target := filepath.Join(memDir, "topic.md")
	decision, err := engine.Decide(context.Background(), Request{
		ToolName: "read",
		Args:     mustJSON(t, map[string]any{"path": target}),
	})
	if err != nil {
		t.Fatalf("Decide: %v", err)
	}
	if decision == nil || !decision.Allowed() || decision.Source != DecisionSourceInternal {
		t.Fatalf("expected internal-path read allow, got %#v", decision)
	}
}

func TestInternalReadOnlyHardDeniesWrite(t *testing.T) {
	workspace := t.TempDir()
	memDir := t.TempDir()
	engine := newTestEngine(t, workspace, ModeBalanced, nil)
	engine.SetFilesystemRoots(FilesystemRoots{
		InternalReadable: []string{memDir},
	})

	target := filepath.Join(memDir, "MEMORY.md")
	decision, err := engine.Decide(context.Background(), Request{
		ToolName: "write",
		Args:     mustJSON(t, map[string]any{"path": target}),
	})
	if err != nil {
		t.Fatalf("Decide: %v", err)
	}
	if decision == nil || decision.Allowed() {
		t.Fatalf("expected hard deny for write to read-only internal path, got %#v", decision)
	}
	if decision.Source != DecisionSourceRoots {
		t.Fatalf("expected hard deny via roots, got source %q", decision.Source)
	}
}

func TestInternalPathRespectsDenyRule(t *testing.T) {
	workspace := t.TempDir()
	memDir := t.TempDir()
	target := filepath.Join(memDir, "secret.md")
	rules, err := ParseRuleSet(nil, []string{"Read(" + target + ")"})
	if err != nil {
		t.Fatalf("ParseRuleSet: %v", err)
	}
	engine := newTestEngine(t, workspace, ModeBalanced, rules)
	engine.SetFilesystemRoots(FilesystemRoots{
		InternalReadable: []string{memDir},
	})

	decision, err := engine.Decide(context.Background(), Request{
		ToolName: "read",
		Args:     mustJSON(t, map[string]any{"path": target}),
	})
	if err != nil {
		t.Fatalf("Decide: %v", err)
	}
	if decision == nil || decision.Allowed() {
		t.Fatalf("deny rule must override internal-path allow, got %#v", decision)
	}
	if decision.Source != DecisionSourceRule {
		t.Fatalf("expected deny via rule, got source %q", decision.Source)
	}
}

func TestUserRootsTakePrecedenceOverInternal(t *testing.T) {
	// When a path is in BOTH the user's WriteRoots and InternalWritable,
	// the user-configured root wins: the request runs through the normal
	// mode-based flow (balanced → ask) instead of the internal silent allow.
	// This matches CC's working-dir-before-internal-path order and prevents
	// the harness from silently overriding what the user explicitly opted
	// into. Lock this in so a future refactor can't quietly invert it.
	workspace := t.TempDir()
	memDir := t.TempDir()
	engine := newTestEngine(t, workspace, ModeBalanced, nil)
	engine.SetFilesystemRoots(FilesystemRoots{
		WriteRoots:       []string{memDir},
		InternalWritable: []string{memDir},
	})

	var prompted bool
	engine.SetApprover(func(context.Context, Prompt) (Choice, error) {
		prompted = true
		return ChoiceAllowOnce, nil
	})

	target := filepath.Join(memDir, "MEMORY.md")
	decision, err := engine.Decide(context.Background(), Request{
		ToolName: "write",
		Args:     mustJSON(t, map[string]any{"path": target}),
	})
	if err != nil {
		t.Fatalf("Decide: %v", err)
	}
	if !prompted {
		t.Fatalf("expected approver to be called for user-write-roots path, got %#v", decision)
	}
	if decision == nil || !decision.Allowed() {
		t.Fatalf("expected allow after prompt, got %#v", decision)
	}
	if decision.Source == DecisionSourceInternal {
		t.Fatalf("internal silent allow must not preempt user-roots flow, got %#v", decision)
	}
}

func TestInternalPathRespectsPlanMode(t *testing.T) {
	workspace := t.TempDir()
	memDir := t.TempDir()
	engine := newTestEngine(t, workspace, ModeBalanced, nil)
	engine.SetFilesystemRoots(FilesystemRoots{
		InternalWritable: []string{memDir},
	})
	engine.SetPlanMode(true)

	target := filepath.Join(memDir, "MEMORY.md")
	decision, err := engine.Decide(context.Background(), Request{
		ToolName: "write",
		Args:     mustJSON(t, map[string]any{"path": target}),
	})
	if err != nil {
		t.Fatalf("Decide: %v", err)
	}
	if decision == nil || decision.Allowed() {
		t.Fatalf("plan mode must block write to internal path, got %#v", decision)
	}
	if decision.Source != DecisionSourceMode {
		t.Fatalf("expected plan-mode deny, got source %q", decision.Source)
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

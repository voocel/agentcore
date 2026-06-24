package permission

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

// testClassifier maps a fixed set of conventional tool names to capabilities
// for the test suite. Production callers register their own classifier via
// EngineConfig.Classifier; the engine itself has no built-in tool name knowledge.
func testClassifier(req Request) Classification {
	args := map[string]any{}
	if len(req.Args) > 0 {
		_ = json.Unmarshal(req.Args, &args)
	}
	str := func(key string) string {
		v, _ := args[key].(string)
		return v
	}
	switch req.ToolName {
	case "read", "glob", "grep", "ls":
		return Classification{Capability: CapabilityRead, Path: str("path")}
	case "write", "edit":
		return Classification{Capability: CapabilityWrite, Path: str("path")}
	case "bash":
		return Classification{
			Capability: CapabilityExec,
			Command:    str("command"),
			Workdir:    str("workdir"),
		}
	case "web_fetch":
		return Classification{Capability: CapabilityNetwork, URL: str("url")}
	case "web_search":
		return Classification{Capability: CapabilityNetwork, Key: "network:search"}
	}
	return Classification{}
}

func newTestEngine(t *testing.T, workspace string, mode Mode, rules *RuleSet) *Engine {
	t.Helper()
	store, err := NewStore(filepath.Join(t.TempDir(), "approvals.json"))
	if err != nil {
		t.Fatalf("NewStore: %v", err)
	}
	return NewEngine(EngineConfig{
		Workspace:  workspace,
		Mode:       mode,
		Rules:      rules,
		Store:      store,
		Classifier: testClassifier,
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

// TestRequestWorkspaceOverridesPathBase: a per-request Workspace changes the
// base relative operand paths resolve (and audit) against, so a moved cwd (e.g.
// into a worktree) checks the right directory.
func TestRequestWorkspaceOverridesPathBase(t *testing.T) {
	main := t.TempDir()
	wt := t.TempDir()
	var summaries []string
	engine := NewEngine(EngineConfig{
		Workspace:  main,
		Mode:       ModeTrust, // auto-allow; we only assert the normalized path
		Classifier: testClassifier,
		Roots:      FilesystemRoots{ReadRoots: []string{main, wt}, WriteRoots: []string{main, wt}},
		OnAudit:    func(e AuditEntry) { summaries = append(summaries, e.Summary) },
	})

	// No per-request workspace → relative path resolves against the engine
	// workspace (main).
	if _, err := engine.Decide(context.Background(), toolReq("write", map[string]any{"path": "a.txt"})); err != nil {
		t.Fatalf("Decide default: %v", err)
	}
	// Per-request workspace → the same relative path resolves against wt.
	req := toolReq("write", map[string]any{"path": "a.txt"})
	req.Workspace = wt
	if _, err := engine.Decide(context.Background(), req); err != nil {
		t.Fatalf("Decide override: %v", err)
	}

	if len(summaries) != 2 {
		t.Fatalf("want 2 audit entries, got %d", len(summaries))
	}
	if want := filepath.Join(main, "a.txt"); summaries[0] != want {
		t.Errorf("default summary = %q, want %q", summaries[0], want)
	}
	if want := filepath.Join(wt, "a.txt"); summaries[1] != want {
		t.Errorf("per-request workspace summary = %q, want %q", summaries[1], want)
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
		Workspace:  t.TempDir(),
		Mode:       ModeBalanced,
		Store:      store,
		Classifier: testClassifier,
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

func TestPlanModeWriteAllowedHookPasses(t *testing.T) {
	workspace := t.TempDir()
	planFile := filepath.Join(workspace, "plan.md")
	store, err := NewStore(filepath.Join(t.TempDir(), "approvals.json"))
	if err != nil {
		t.Fatalf("NewStore: %v", err)
	}
	engine := NewEngine(EngineConfig{
		Workspace:  workspace,
		Mode:       ModeBalanced,
		Store:      store,
		Classifier: testClassifier,
		PlanModeWriteAllowed: func(req Request) bool {
			var args struct {
				Path string `json:"path"`
			}
			_ = json.Unmarshal(req.Args, &args)
			return args.Path == planFile
		},
	})
	engine.SetPlanMode(true)
	engine.SetApprover(func(context.Context, Prompt) (Choice, error) {
		t.Fatalf("plan-mode write allow-list should bypass approver")
		return ChoiceDeny, nil
	})

	decision, err := engine.Decide(context.Background(), Request{
		ToolName: "write",
		Args:     mustJSON(t, map[string]any{"path": planFile, "content": "# Plan"}),
	})
	if err != nil {
		t.Fatalf("Decide: %v", err)
	}
	if decision == nil || !decision.Allowed() || decision.Source != DecisionSourceMode {
		t.Fatalf("expected plan-mode allow for plan-file write, got %#v", decision)
	}

	// Other paths must still be denied — the hook is path-specific, not
	// blanket-permissive.
	decision, err = engine.Decide(context.Background(), Request{
		ToolName: "write",
		Args:     mustJSON(t, map[string]any{"path": filepath.Join(workspace, "other.go"), "content": "x"}),
	})
	if err != nil {
		t.Fatalf("Decide: %v", err)
	}
	if decision == nil || decision.Kind != DecisionDeny {
		t.Fatalf("expected plan-mode deny for non-plan-file write, got %#v", decision)
	}
}

func TestPlanModeWriteWithoutHookStillDenies(t *testing.T) {
	workspace := t.TempDir()
	engine := newTestEngine(t, workspace, ModeBalanced, nil)
	engine.SetPlanMode(true)

	decision, err := engine.Decide(context.Background(), Request{
		ToolName: "write",
		Args:     mustJSON(t, map[string]any{"path": filepath.Join(workspace, "plan.md"), "content": "# Plan"}),
	})
	if err != nil {
		t.Fatalf("Decide: %v", err)
	}
	if decision == nil || decision.Kind != DecisionDeny || decision.Source != DecisionSourceMode {
		t.Fatalf("expected plan-mode denial without write hook, got %#v", decision)
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
	// User intent takes precedence over harness-declared internal paths so
	// the harness cannot silently override what the user explicitly opted
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

func TestPlanModeAllowsInternalWritablePath(t *testing.T) {
	// InternalWritable is the harness saying "this directory is mine to
	// manage" (plan files, auto-memory, scratch). Plan mode trusts that
	// declaration and lets writes through without needing PlanModeWriteAllowed.
	// Mirrors how InternalReadable rides the Read pass-through.
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
	if decision == nil || !decision.Allowed() {
		t.Fatalf("expected plan-mode allow for InternalWritable path, got %#v", decision)
	}
	if decision.Source != DecisionSourceMode {
		t.Fatalf("expected plan-mode allow source, got %q", decision.Source)
	}
}

func TestPlanModeAllowsNetwork(t *testing.T) {
	// Network capability passes plan mode unconditionally — web_fetch /
	// web_search are read-only by nature and plan exploration commonly
	// needs to look up references.
	engine := newTestEngine(t, t.TempDir(), ModeBalanced, nil)
	engine.SetPlanMode(true)
	engine.SetApprover(func(context.Context, Prompt) (Choice, error) {
		t.Fatalf("plan-mode network should bypass approver")
		return ChoiceDeny, nil
	})

	decision, err := engine.Decide(context.Background(), Request{
		ToolName: "web_search",
		Args:     mustJSON(t, map[string]any{"query": "go context cancel"}),
	})
	if err != nil {
		t.Fatalf("Decide: %v", err)
	}
	if decision == nil || !decision.Allowed() || decision.Source != DecisionSourceMode {
		t.Fatalf("expected plan-mode allow for network, got %#v", decision)
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

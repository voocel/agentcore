package permission

import (
	"context"
	"encoding/json"
	"fmt"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

type Engine struct {
	workspace string
	rules     *RuleSet
	store     *Store
	onAudit   func(AuditEntry)

	mu           sync.RWMutex
	mode         Mode
	planMode     bool
	approver     Approver
	fsRoots      FilesystemRoots
	sessionAllow map[string]StoreEntry
	skillAllows  []Rule
}

func NewEngine(cfg EngineConfig) *Engine {
	return &Engine{
		workspace:    cfg.Workspace,
		rules:        cfg.Rules,
		store:        cfg.Store,
		onAudit:      cfg.OnAudit,
		mode:         cfg.Mode,
		approver:     cfg.Approver,
		fsRoots:      normalizeFilesystemRoots(cfg.Workspace, cfg.Roots),
		sessionAllow: make(map[string]StoreEntry),
	}
}

func (e *Engine) Decide(ctx context.Context, req Request) (*Decision, error) {
	info := inspectRequest(e.workspace, e.filesystemRoots(), req)
	if info.hardDeny != "" {
		decision := denyDecision(DecisionSourceRoots, info, info.hardDeny)
		e.audit(info, decision)
		return decision, nil
	}

	mode, planMode := e.state()

	var ruleResult ruleAction
	var ruleMatched bool
	if e.rules != nil {
		ruleResult, ruleMatched = e.rules.Evaluate(info)
		if ruleMatched && ruleResult == ruleDeny {
			decision := denyDecision(DecisionSourceRule, info, "denied by permission rule")
			e.audit(info, decision)
			return decision, nil
		}
	}

	if planMode {
		switch info.capability {
		case CapabilityRead, CapabilityInternal:
		default:
			decision := denyDecision(DecisionSourceMode, info, "plan mode is read-only")
			e.audit(info, decision)
			return decision, nil
		}
	}

	// Harness-declared internal path: silent allow for the requested
	// capability. Deny rules and plan-mode read-only enforcement above still
	// apply, so this only bypasses the OutsideRoots prompt and the mode-based
	// ask that would otherwise interrupt every memory/plan/scratch write.
	if info.internalPath {
		decision := allowDecision(DecisionSourceInternal, info, "harness-managed path")
		e.audit(info, decision)
		return decision, nil
	}

	if info.outsideRoots {
		decision, err := e.ask(ctx, info, mode, planMode)
		if err != nil {
			return nil, err
		}
		e.audit(info, decision)
		return decision, nil
	}

	e.mu.RLock()
	skillRules := e.skillAllows
	e.mu.RUnlock()
	for _, r := range skillRules {
		if r.matches(info, false) {
			decision := allowDecision(DecisionSourceSkill, info, "allowed by skill")
			e.audit(info, decision)
			return decision, nil
		}
	}

	if e.allowed(info.key) || e.allowedSession(info.capability) {
		decision := allowDecision(DecisionSourceStore, info, "allowed by stored approval")
		e.audit(info, decision)
		return decision, nil
	}

	if ruleMatched && ruleResult == ruleAllow {
		decision := allowDecision(DecisionSourceRule, info, "allowed by permission rule")
		e.audit(info, decision)
		return decision, nil
	}

	switch mode {
	case ModeTrust:
		decision := allowDecision(DecisionSourceMode, info, "trust mode allows tool execution")
		e.audit(info, decision)
		return decision, nil
	case ModeAcceptEdits:
		switch info.capability {
		case CapabilityRead, CapabilityInternal, CapabilityWrite:
			decision := allowDecision(DecisionSourceMode, info, "accept-edits mode allows this capability")
			e.audit(info, decision)
			return decision, nil
		default:
			decision, err := e.ask(ctx, info, mode, planMode)
			if err != nil {
				return nil, err
			}
			e.audit(info, decision)
			return decision, nil
		}
	case ModeStrict:
		switch info.capability {
		case CapabilityRead, CapabilityInternal:
			decision := allowDecision(DecisionSourceMode, info, "strict mode allows read-only tools")
			e.audit(info, decision)
			return decision, nil
		case CapabilityWrite:
			decision, err := e.ask(ctx, info.withReason("strict mode requires approval for writes"), mode, planMode)
			if err != nil {
				return nil, err
			}
			e.audit(info, decision)
			return decision, nil
		default:
			decision := denyDecision(DecisionSourceMode, info, "strict mode denies this capability")
			e.audit(info, decision)
			return decision, nil
		}
	default:
		switch info.capability {
		case CapabilityRead, CapabilityInternal:
			decision := allowDecision(DecisionSourceMode, info, "balanced mode allows read-only tools")
			e.audit(info, decision)
			return decision, nil
		default:
			decision, err := e.ask(ctx, info.withReason("approval required for side effects"), mode, planMode)
			if err != nil {
				return nil, err
			}
			e.audit(info, decision)
			return decision, nil
		}
	}
}

func (e *Engine) SetFilesystemRoots(roots FilesystemRoots) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.fsRoots = normalizeFilesystemRoots(e.workspace, roots)
}

func (e *Engine) FilesystemRoots() FilesystemRoots {
	return e.filesystemRoots()
}

func (e *Engine) filesystemRoots() FilesystemRoots {
	e.mu.RLock()
	defer e.mu.RUnlock()
	return e.fsRoots
}

func (e *Engine) SetMode(mode Mode) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.mode = mode
}

func (e *Engine) Mode() Mode {
	e.mu.RLock()
	defer e.mu.RUnlock()
	return e.mode
}

func (e *Engine) SetPlanMode(active bool) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.planMode = active
}

func (e *Engine) PlanMode() bool {
	e.mu.RLock()
	defer e.mu.RUnlock()
	return e.planMode
}

func (e *Engine) SetApprover(fn Approver) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.approver = fn
}

func (e *Engine) SetSkillAllows(rawTools []string) {
	var rules []Rule
	for _, raw := range rawTools {
		r, err := ParseRule(raw)
		if err != nil {
			continue
		}
		rules = append(rules, r)
	}
	e.mu.Lock()
	e.skillAllows = rules
	e.mu.Unlock()
}

func (e *Engine) state() (Mode, bool) {
	e.mu.RLock()
	defer e.mu.RUnlock()
	return e.mode, e.planMode
}

func (e *Engine) ask(ctx context.Context, info toolInfo, mode Mode, planMode bool) (*Decision, error) {
	e.mu.RLock()
	fn := e.approver
	e.mu.RUnlock()
	if fn == nil {
		msg := info.reason
		if msg == "" {
			msg = "approval required but no approver is configured"
		}
		return denyDecision(DecisionSourcePrompt, info, msg), nil
	}
	choice, err := fn(ctx, Prompt{
		Tool:         info.tool,
		Summary:      info.summary,
		Reason:       info.reason,
		Capability:   info.capability,
		Preview:      info.preview,
		OutsideRoots: info.outsideRoots,
	})
	if err != nil {
		return nil, err
	}
	return e.resolveChoice(info, mode, planMode, choice), nil
}

func (e *Engine) resolveChoice(info toolInfo, mode Mode, planMode bool, choice Choice) *Decision {
	if info.outsideRoots && (choice == ChoiceAllowAlways || choice == ChoiceAllowSession) {
		choice = ChoiceAllowOnce
	}
	switch choice {
	case ChoiceAllowAlways:
		entry := StoreEntry{
			Key:        info.key,
			Tool:       info.tool,
			Capability: info.capability,
			Summary:    info.summary,
			AddedAt:    time.Now(),
		}
		e.mu.Lock()
		e.sessionAllow[info.key] = entry
		store := e.store
		e.mu.Unlock()
		if store != nil {
			_ = store.Add(entry)
		}
		return &Decision{
			Kind:         DecisionAllowAlways,
			Source:       DecisionSourcePrompt,
			Capability:   info.capability,
			Summary:      info.summary,
			Key:          info.key,
			OutsideRoots: info.outsideRoots,
			Prompted:     true,
			Preview:      info.preview,
		}
	case ChoiceAllowSession:
		sKey := "session:" + string(info.capability)
		entry := StoreEntry{
			Key:        sKey,
			Tool:       info.tool,
			Capability: info.capability,
			Summary:    info.summary,
			AddedAt:    time.Now(),
		}
		e.mu.Lock()
		e.sessionAllow[sKey] = entry
		e.mu.Unlock()
		return &Decision{
			Kind:         DecisionAllowSession,
			Source:       DecisionSourcePrompt,
			Capability:   info.capability,
			Summary:      info.summary,
			Key:          sKey,
			OutsideRoots: info.outsideRoots,
			Prompted:     true,
			Preview:      info.preview,
		}
	case ChoiceDeny:
		return &Decision{
			Kind:         DecisionDeny,
			Source:       DecisionSourcePrompt,
			Reason:       firstNonEmpty(info.reason, "tool execution denied by user"),
			Capability:   info.capability,
			Summary:      info.summary,
			Key:          info.key,
			OutsideRoots: info.outsideRoots,
			Prompted:     true,
			Preview:      info.preview,
		}
	default:
		return &Decision{
			Kind:         DecisionAllowOnce,
			Source:       DecisionSourcePrompt,
			Capability:   info.capability,
			Summary:      info.summary,
			Key:          info.key,
			OutsideRoots: info.outsideRoots,
			Prompted:     true,
			Preview:      info.preview,
		}
	}
}

func (e *Engine) allowedSession(cap Capability) bool {
	return e.allowed("session:" + string(cap))
}

func (e *Engine) allowed(key string) bool {
	if key == "" {
		return false
	}
	e.mu.RLock()
	_, ok := e.sessionAllow[key]
	store := e.store
	e.mu.RUnlock()
	if ok {
		return true
	}
	return store != nil && store.Has(key)
}

func (e *Engine) audit(info toolInfo, decision *Decision) {
	if decision == nil {
		return
	}
	e.auditPassthrough(
		info,
		e.Mode(),
		e.PlanMode(),
		string(decision.Kind),
		decision.Allowed(),
		decision.Reason,
	)
}

func (e *Engine) auditPassthrough(info toolInfo, mode Mode, planMode bool, decision string, allow bool, reason string) {
	if e.onAudit == nil {
		return
	}
	e.onAudit(AuditEntry{
		Time:       time.Now(),
		Mode:       mode,
		PlanMode:   planMode,
		Tool:       info.tool,
		Capability: info.capability,
		Summary:    info.summary,
		Decision:   decision,
		Reason:     reason,
		Allow:      allow,
	})
}

type toolInfo struct {
	tool         string
	capability   Capability
	summary      string
	key          string
	reason       string
	preview      string
	hardDeny     string
	outsideRoots bool
	internalPath bool
	workspace    string
	roots        []string
}

func (i toolInfo) withReason(reason string) toolInfo {
	i.reason = reason
	return i
}

type ruleAction string

const (
	ruleAllow ruleAction = "allow"
	ruleAsk   ruleAction = "ask"
	ruleDeny  ruleAction = "deny"
)

func denyDecision(source DecisionSource, info toolInfo, reason string) *Decision {
	return &Decision{
		Kind:         DecisionDeny,
		Source:       source,
		Reason:       reason,
		Capability:   info.capability,
		Summary:      info.summary,
		Key:          info.key,
		OutsideRoots: info.outsideRoots,
		Preview:      info.preview,
	}
}

func allowDecision(source DecisionSource, info toolInfo, reason string) *Decision {
	return &Decision{
		Kind:         DecisionAllow,
		Source:       source,
		Reason:       reason,
		Capability:   info.capability,
		Summary:      info.summary,
		Key:          info.key,
		OutsideRoots: info.outsideRoots,
		Preview:      info.preview,
	}
}

func inspectRequest(workspace string, roots FilesystemRoots, req Request) toolInfo {
	payload := decodeArgs(req.Args)
	info := toolInfo{
		tool:       req.ToolName,
		capability: CapabilityUnknown,
		summary:    strings.TrimSpace(req.Summary),
		reason:     strings.TrimSpace(req.Reason),
		preview:    previewText(req.Preview),
		workspace:  workspace,
	}
	if info.summary == "" {
		info.summary = req.ToolName
	}
	switch req.ToolName {
	case "read", "glob", "grep", "ls":
		info.capability = CapabilityRead
		info.key = "read"
		info.roots = roots.ReadRoots
		path, deny := checkedPath(workspace, roots.ReadRoots, payload, "readable")
		if path != "" {
			info.summary = path
		}
		if deny != "" {
			// Harness-declared internal paths bypass the user-roots check.
			// InternalWritable implies readability — populating only the
			// writable list is the common case for fully-managed dirs.
			if pathInRoots(path, roots.InternalReadable) || pathInRoots(path, roots.InternalWritable) {
				info.internalPath = true
			} else {
				info.outsideRoots = true
				info.reason = deny
			}
		}
	case "write", "edit":
		info.capability = CapabilityWrite
		info.roots = roots.WriteRoots
		path, deny := checkedPath(workspace, roots.WriteRoots, payload, "writable")
		info.summary = firstNonEmpty(path, info.summary)
		info.key = "write:" + path
		if info.reason == "" {
			info.reason = "file modification requires approval"
		}
		if deny != "" {
			switch {
			case pathInRoots(path, roots.InternalWritable):
				info.internalPath = true
			case pathInRoots(path, roots.InternalReadable):
				info.hardDeny = fmt.Sprintf("path in read-only internal root, not writable: %s", path)
			default:
				_, notInReadRoots := checkedPath(workspace, roots.ReadRoots, payload, "readable")
				if notInReadRoots == "" {
					info.hardDeny = fmt.Sprintf("path in read-only root, not writable: %s", path)
				} else {
					info.outsideRoots = true
					info.reason = deny
				}
			}
		}
	case "bash":
		info.capability = CapabilityExec
		command := strings.TrimSpace(stringArg(payload, "command"))
		info.summary = firstNonEmpty(command, info.summary)
		info.key = "exec:" + shortHash(command)
		if info.reason == "" {
			info.reason = "shell execution requires approval"
		}
		if wd := strings.TrimSpace(stringArg(payload, "workdir")); wd != "" {
			_, deny := checkedPath(workspace, roots.WriteRoots, map[string]any{"path": wd}, "writable")
			if deny != "" {
				info.outsideRoots = true
				info.reason = fmt.Sprintf("bash workdir outside writable roots: %s", wd)
			}
		}
	case "web_fetch":
		info.capability = CapabilityNetwork
		targetURL := strings.TrimSpace(stringArg(payload, "url"))
		host := hostOf(targetURL)
		info.summary = firstNonEmpty(targetURL, info.summary)
		info.key = "network:" + host
		if info.reason == "" {
			info.reason = "network access requires approval"
		}
	case "web_search":
		info.capability = CapabilityNetwork
		query := strings.TrimSpace(stringArg(payload, "query"))
		info.summary = firstNonEmpty(query, info.summary)
		info.key = "network:search"
		if info.reason == "" {
			info.reason = "network access requires approval"
		}
	case "ask_user", "enter_plan_mode", "exit_plan_mode", "task_create", "task_get", "task_update", "task_list", "cron_create", "cron_list", "cron_remove", "tool_search", "subagent":
		info.capability = CapabilityInternal
		info.key = "internal:" + req.ToolName
	default:
		info.capability = CapabilityUnknown
		info.key = "tool:" + req.ToolName
		if info.reason == "" {
			info.reason = "unclassified tool requires approval"
		}
	}

	meta := req.Metadata
	if meta.Capability != "" {
		info.capability = meta.Capability
	}
	if meta.SummaryHint != "" && strings.TrimSpace(info.summary) == req.ToolName {
		info.summary = meta.SummaryHint
	}
	if meta.Reason != "" {
		info.reason = meta.Reason
	}
	if meta.Key != "" {
		info.key = meta.Key
	} else if meta.KeyPrefix != "" {
		info.key = meta.KeyPrefix + ":" + req.ToolName
	}
	return info
}

func decodeArgs(raw json.RawMessage) map[string]any {
	if len(raw) == 0 {
		return nil
	}
	var payload map[string]any
	_ = json.Unmarshal(raw, &payload)
	return payload
}

func stringArg(payload map[string]any, key string) string {
	if payload == nil {
		return ""
	}
	value, _ := payload[key].(string)
	return value
}

func pathArg(workspace string, payload map[string]any) string {
	path := strings.TrimSpace(stringArg(payload, "path"))
	if path == "" {
		return ""
	}
	if filepath.IsAbs(path) {
		return filepath.Clean(path)
	}
	if workspace == "" {
		return filepath.Clean(path)
	}
	return filepath.Clean(filepath.Join(workspace, path))
}

func checkedPath(workspace string, roots []string, payload map[string]any, rootLabel string) (string, string) {
	path := pathArg(workspace, payload)
	if path == "" {
		return path, ""
	}
	if len(roots) == 0 {
		if workspace == "" {
			return path, ""
		}
		roots = []string{workspace}
	}
	if pathInRoots(path, roots) {
		return path, ""
	}
	return path, fmt.Sprintf("path outside %s roots denied: %s", rootLabel, path)
}

// pathInRoots reports whether path resolves under any of the given roots.
// Both sides are passed through resolveSymlinks so a symlink inside or below
// a root resolves consistently with the canonical comparison checkedPath
// performs. A miss returns false without producing a deny message — callers
// such as inspectRequest's InternalReadable/InternalWritable check fall
// through to the next decision step on miss.
func pathInRoots(path string, roots []string) bool {
	if path == "" || len(roots) == 0 {
		return false
	}
	target := resolveSymlinks(path)
	for _, root := range roots {
		base := resolveSymlinks(filepath.Clean(root))
		if isSubPath(base, target) {
			return true
		}
	}
	return false
}

func resolveSymlinks(path string) string {
	if path == "" {
		return ""
	}
	cleaned := filepath.Clean(path)
	if resolved, err := filepath.EvalSymlinks(cleaned); err == nil {
		return resolved
	}
	current := cleaned
	tail := ""
	for {
		if resolved, err := filepath.EvalSymlinks(current); err == nil {
			if tail == "" {
				return resolved
			}
			return filepath.Join(resolved, tail)
		}
		parent := filepath.Dir(current)
		if parent == current {
			if tail == "" {
				return cleaned
			}
			return filepath.Join(current, tail)
		}
		base := filepath.Base(current)
		if tail == "" {
			tail = base
		} else {
			tail = filepath.Join(base, tail)
		}
		current = parent
	}
}

func isSubPath(base, target string) bool {
	if base == "" {
		return true
	}
	base = filepath.Clean(base)
	target = filepath.Clean(target)
	rel, err := filepath.Rel(base, target)
	if err != nil {
		return false
	}
	return rel == "." || (rel != ".." && !strings.HasPrefix(rel, ".."+string(os.PathSeparator)))
}

func normalizeFilesystemRoots(workspace string, roots FilesystemRoots) FilesystemRoots {
	readRoots := dedup(roots.ReadRoots)
	writeRoots := dedup(roots.WriteRoots)
	if len(readRoots) == 0 && workspace != "" {
		readRoots = []string{filepath.Clean(workspace)}
	}
	if len(writeRoots) == 0 && workspace != "" {
		writeRoots = []string{filepath.Clean(workspace)}
	}
	return FilesystemRoots{
		ReadRoots:        readRoots,
		WriteRoots:       writeRoots,
		InternalReadable: dedup(roots.InternalReadable),
		InternalWritable: dedup(roots.InternalWritable),
	}
}

func dedup(roots []string) []string {
	if len(roots) == 0 {
		return nil
	}
	seen := make(map[string]struct{}, len(roots))
	out := make([]string, 0, len(roots))
	for _, root := range roots {
		root = strings.TrimSpace(root)
		if root == "" {
			continue
		}
		root = filepath.Clean(root)
		if _, ok := seen[root]; ok {
			continue
		}
		seen[root] = struct{}{}
		out = append(out, root)
	}
	return out
}

func hostOf(raw string) string {
	parsed, err := url.Parse(strings.TrimSpace(raw))
	if err != nil {
		return "unknown"
	}
	if parsed.Host != "" {
		return strings.ToLower(parsed.Host)
	}
	return "unknown"
}

func previewText(raw json.RawMessage) string {
	if len(raw) == 0 {
		return ""
	}
	var text string
	if json.Unmarshal(raw, &text) == nil {
		return truncate(text, 400)
	}
	return truncate(string(raw), 400)
}

func truncate(s string, max int) string {
	runes := []rune(strings.TrimSpace(s))
	if len(runes) <= max {
		return string(runes)
	}
	return string(runes[:max]) + "..."
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return value
		}
	}
	return ""
}

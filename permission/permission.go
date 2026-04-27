package permission

import (
	"context"
	"encoding/json"
	"time"
)

type Mode string

const (
	ModeStrict      Mode = "strict"
	ModeBalanced    Mode = "balanced"
	ModeAcceptEdits Mode = "accept_edits"
	ModeTrust       Mode = "trust"
)

type Capability string

const (
	CapabilityRead     Capability = "read"
	CapabilityWrite    Capability = "write"
	CapabilityExec     Capability = "exec"
	CapabilityHook     Capability = "hook"
	CapabilityNetwork  Capability = "network"
	CapabilitySubagent Capability = "subagent"
	CapabilityInternal Capability = "internal"
	CapabilityUnknown  Capability = "unknown"
)

type DecisionKind string

const (
	DecisionAllow        DecisionKind = "allow"
	DecisionAllowOnce    DecisionKind = "allow_once"
	DecisionAllowSession DecisionKind = "allow_session"
	DecisionAllowAlways  DecisionKind = "allow_always"
	DecisionDeny         DecisionKind = "deny"
)

type DecisionSource string

const (
	DecisionSourceTool     DecisionSource = "tool"
	DecisionSourceRule     DecisionSource = "rule"
	DecisionSourceSkill    DecisionSource = "skill"
	DecisionSourceMode     DecisionSource = "mode"
	DecisionSourcePrompt   DecisionSource = "prompt"
	DecisionSourceStore    DecisionSource = "store"
	DecisionSourceRoots    DecisionSource = "roots"
	DecisionSourceInternal DecisionSource = "internal"
)

type Choice string

const (
	ChoiceAllowOnce    Choice = "allow_once"
	ChoiceAllowSession Choice = "allow_session"
	ChoiceAllowAlways  Choice = "allow_always"
	ChoiceDeny         Choice = "deny"
)

// FilesystemRoots scope filesystem access for tool requests. The two pairs
// serve different audiences:
//
//   - ReadRoots / WriteRoots: user-configured. Subject to deny rules and
//     mode-based prompts (e.g. balanced mode asks for any write). Out-of-roots
//     access triggers an OutsideRoots prompt; AllowAlways for that prompt is
//     downgraded to AllowOnce so a one-shot consent does not silently grant
//     persistent access.
//
//   - InternalReadable / InternalWritable: harness-declared. Reserved for
//     paths the harness itself manages (auto-memory dir, plan store, scratch
//     space). Matches bypass the OutsideRoots prompt and the mode-based ask
//     so the agent can read/write these locations silently. Deny rules and
//     plan-mode read-only enforcement still apply.
//
// A path matched by InternalWritable is also treated as readable, so callers
// that want bidirectional access only need to populate the writable list.
type FilesystemRoots struct {
	ReadRoots        []string
	WriteRoots       []string
	InternalReadable []string
	InternalWritable []string
}

type Metadata struct {
	Capability  Capability `json:"capability,omitempty"`
	SummaryHint string     `json:"summary_hint,omitempty"`
	Reason      string     `json:"reason,omitempty"`
	Key         string     `json:"key,omitempty"`
	KeyPrefix   string     `json:"key_prefix,omitempty"`
}

type Request struct {
	ToolID    string          `json:"tool_id,omitempty"`
	ToolName  string          `json:"tool_name"`
	ToolLabel string          `json:"tool_label,omitempty"`
	Summary   string          `json:"summary,omitempty"`
	Reason    string          `json:"reason,omitempty"`
	Args      json.RawMessage `json:"args,omitempty"`
	Preview   json.RawMessage `json:"preview,omitempty"`
	Metadata  Metadata        `json:"metadata,omitempty"`
}

type Decision struct {
	Kind         DecisionKind    `json:"kind"`
	Source       DecisionSource  `json:"source,omitempty"`
	Reason       string          `json:"reason,omitempty"`
	Capability   Capability      `json:"capability,omitempty"`
	Summary      string          `json:"summary,omitempty"`
	Key          string          `json:"key,omitempty"`
	OutsideRoots bool            `json:"outside_roots,omitempty"`
	Prompted     bool            `json:"prompted,omitempty"`
	Preview      string          `json:"preview,omitempty"`
	UpdatedArgs  json.RawMessage `json:"updated_args,omitempty"`
	Suggestions  []string        `json:"suggestions,omitempty"`
}

func (d Decision) Allowed() bool {
	switch d.Kind {
	case DecisionAllow, DecisionAllowOnce, DecisionAllowSession, DecisionAllowAlways:
		return true
	default:
		return false
	}
}

type Prompt struct {
	Tool         string
	Summary      string
	Reason       string
	Capability   Capability
	Preview      string
	OutsideRoots bool
}

type Approver func(ctx context.Context, prompt Prompt) (Choice, error)

type Checker interface {
	CheckPermission(ctx context.Context, req Request) (*Decision, error)
}

type DecisionEngine interface {
	Decide(ctx context.Context, req Request) (*Decision, error)
}

type AuditEntry struct {
	Time       time.Time  `json:"time"`
	Mode       Mode       `json:"mode"`
	PlanMode   bool       `json:"plan_mode"`
	Tool       string     `json:"tool"`
	Capability Capability `json:"capability"`
	Summary    string     `json:"summary"`
	Decision   string     `json:"decision"`
	Reason     string     `json:"reason,omitempty"`
	Allow      bool       `json:"allow"`
}

type EngineConfig struct {
	Workspace string
	Mode      Mode
	Rules     *RuleSet
	Roots     FilesystemRoots
	Store     *Store
	Approver  Approver
	OnAudit   func(AuditEntry)
}

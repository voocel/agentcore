package agentcore

import "context"

// ToolApprovalDecision describes the outcome of a runtime approval check.
type ToolApprovalDecision string

const (
	ToolApprovalAllowOnce    ToolApprovalDecision = "allow_once"
	ToolApprovalAllowSession ToolApprovalDecision = "allow_session"
	ToolApprovalAllowAlways  ToolApprovalDecision = "allow_always"
	ToolApprovalDeny         ToolApprovalDecision = "deny"
)

// ToolApprovalRequest is sent to an approval callback before a tool executes.
// Approval systems can use Summary/Reason/Preview to present higher-signal UI.
type ToolApprovalRequest struct {
	Call      ToolCall
	ToolLabel string
	Summary   string
	Reason    string
	Preview   []byte
}

// ToolApprovalResult is returned by a runtime approval callback.
type ToolApprovalResult struct {
	Approved bool
	Decision ToolApprovalDecision
	Reason   string
}

// ToolApprovalFunc is called before tool execution when a runtime approval
// system is configured. Returning nil means "no approval needed".
// Returning a non-nil result applies the decision immediately.
type ToolApprovalFunc func(ctx context.Context, req ToolApprovalRequest) (*ToolApprovalResult, error)

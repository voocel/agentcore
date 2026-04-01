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

// ChainApproval composes multiple approval functions into a single chain.
// Each function is called in order. The first one to return a non-nil result
// (allow or deny) short-circuits the chain. If all return nil, the result is
// nil (no approval needed). If any returns an error, the chain stops with that error.
func ChainApproval(fns ...ToolApprovalFunc) ToolApprovalFunc {
	return func(ctx context.Context, req ToolApprovalRequest) (*ToolApprovalResult, error) {
		for _, fn := range fns {
			if fn == nil {
				continue
			}
			result, err := fn(ctx, req)
			if err != nil {
				return nil, err
			}
			if result != nil {
				return result, nil
			}
		}
		return nil, nil
	}
}

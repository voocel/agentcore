package agentcore

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/voocel/litellm"
)

// Tool is a function the model may call.
type Tool struct {
	Name        string
	Description string
	// Schema is the JSON Schema of the arguments. Calls are validated
	// against it before they run, and the model sees what does not fit.
	Schema map[string]any
	// Label names the tool for people, such as "Edit File".
	Label string
	// Deferred tools are offered to the model only once a
	// litellm.ToolReferenceBlock in the history names them, as a tool search
	// returns; until then they cost no context. See litellm.Tool.
	Deferred bool
	// Parallel lets the tool's calls run alongside the other parallel calls
	// of their turn; any other call runs alone.
	Parallel bool
	// Check, if set, vets a call before it is approved and run, such as that
	// a file was read before it is edited, and may preview for people what
	// the call will do, such as a diff to approve. An error fails the call.
	Check func(ctx context.Context, args json.RawMessage) (preview string, err error)
	// Run makes a call. An error fails it, and the model reads the error.
	Run func(ctx context.Context, args json.RawMessage) (Result, error)
}

// NewTool returns a tool whose calls run with their arguments decoded into
// P, once they fit schema.
func NewTool[P any](name, description string, schema map[string]any, run func(ctx context.Context, args P) (Result, error)) Tool {
	return Tool{
		Name:        name,
		Description: description,
		Schema:      schema,
		Run: func(ctx context.Context, raw json.RawMessage) (Result, error) {
			var args P
			if err := json.Unmarshal(raw, &args); err != nil {
				return Result{}, fmt.Errorf("decode arguments: %w", err)
			}
			return run(ctx, args)
		},
	}
}

// Result is what a tool call returns to the model.
type Result struct {
	// Content is what the model reads: text, images and tool references.
	Content []litellm.Block
	IsError bool
	// Terminate ends the run once the turn's results are recorded, as a tool
	// that completes the task does. The run's OnStop may keep it going.
	Terminate bool
}

// Text returns the text of r's text blocks.
func (r Result) Text() string {
	return Message{Blocks: r.Content}.Text()
}

// TextResult returns text as a result.
func TextResult(text string) Result {
	return Result{Content: []litellm.Block{litellm.Text(text)}}
}

// JSONResult returns v, as JSON text, as a result. Characters such as <
// and & stay as they are, as the model reads them.
func JSONResult(v any) (Result, error) {
	var b strings.Builder
	enc := json.NewEncoder(&b)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(v); err != nil {
		return Result{}, err
	}
	return TextResult(strings.TrimSuffix(b.String(), "\n")), nil
}

// ErrorResult returns text as a failed result.
func ErrorResult(text string) Result {
	r := TextResult(text)
	r.IsError = true
	return r
}

// ToolCall is a call of a tool, as the loop runs it.
type ToolCall struct {
	ID   string
	Name string
	Args json.RawMessage
	// Tool is the tool called; nil when the model called an unknown one.
	Tool *Tool
	// Preview is what the tool's Check previewed.
	Preview string
}

// ToolFunc runs a tool call.
type ToolFunc func(ctx context.Context, call ToolCall) (Result, error)

// ToolMiddleware wraps the running of each tool call, once its arguments are
// valid and its tool's Check passed: for approval, auditing or rewriting the
// arguments. It calls next to go on; returning without calling next, such as
// with an ErrorResult, refuses the call.
type ToolMiddleware func(ctx context.Context, call ToolCall, next ToolFunc) (Result, error)

type progressKey struct{}

// WithProgress returns ctx with fn receiving the progress the tool calls it
// runs report.
func WithProgress(ctx context.Context, fn func(progress any)) context.Context {
	return context.WithValue(ctx, progressKey{}, fn)
}

// ReportProgress reports the progress of the tool call ctx runs, such as its
// output so far; the run emits it as a ToolUpdate. What progress is, each
// tool documents.
func ReportProgress(ctx context.Context, progress any) {
	if fn, ok := ctx.Value(progressKey{}).(func(any)); ok {
		fn(progress)
	}
}

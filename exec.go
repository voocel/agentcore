package agentcore

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sync"

	"github.com/voocel/litellm"
)

// invalidArgs fixes the tool calls of msg whose arguments are not the JSON
// object every protocol takes, as a response cut off at the output limit
// leaves them, so that msg can be stored and sent back: their arguments
// become {}. It returns, by call, what the model reads instead of a result.
func invalidArgs(msg *Message) map[string]string {
	var bad map[string]string
	for i, block := range msg.Blocks {
		call, ok := block.(litellm.ToolUseBlock)
		if !ok {
			continue
		}
		var object map[string]json.RawMessage
		err := json.Unmarshal([]byte(call.Arguments), &object)
		if err == nil && object != nil {
			continue
		}
		if bad == nil {
			bad = map[string]string{}
			msg.Blocks = append([]litellm.Block(nil), msg.Blocks...)
		}
		if msg.Stop == StopLength {
			bad[call.ID] = "The response hit the output token limit, so the arguments of this call were cut off and it did not run. Make the call again, splitting large content across several calls."
		} else {
			if err == nil {
				err = errors.New("null")
			}
			bad[call.ID] = fmt.Sprintf("The arguments of this call are not a JSON object (%v), so it did not run. Received: %s", err, call.Arguments)
		}
		call.Arguments = "{}"
		msg.Blocks[i] = call
	}
	return bad
}

// runTools runs the tool calls of msg and records their results in call
// order. Consecutive parallel calls run together, up to MaxToolConcurrency
// at once; any other call runs alone. It reports whether a successful call
// terminated the run.
func (r *run) runTools(msg Message, bad map[string]string) ([]Message, bool, error) {
	uses := msg.ToolCalls()
	if len(uses) == 0 {
		return nil, false, nil
	}
	r.toolCalls += len(uses)

	calls := make([]ToolCall, len(uses))
	parallel := make([]bool, len(uses))
	for i, use := range uses {
		calls[i] = ToolCall{ID: use.ID, Name: use.Name, Args: json.RawMessage(use.Arguments), Tool: r.tool(use.Name)}
		t := calls[i].Tool
		parallel[i] = t != nil && t.Parallel != nil && bad[use.ID] == "" && t.Parallel(calls[i].Args)
	}

	results := make([]Result, len(calls))
	failed := make([]bool, len(calls))
	limit := max(r.cfg.MaxToolConcurrency, 1)
	for i := 0; i < len(calls); {
		j := i + 1
		if limit > 1 && parallel[i] {
			for j < len(calls) && parallel[j] {
				j++
			}
		}
		var wg sync.WaitGroup
		sem := make(chan struct{}, limit)
		for k := i; k < j; k++ {
			wg.Add(1)
			sem <- struct{}{}
			go func() {
				defer wg.Done()
				defer func() { <-sem }()
				results[k], failed[k] = r.runCall(&calls[k], bad[calls[k].ID])
			}()
		}
		wg.Wait()
		i = j
	}
	if err := r.failed(); err != nil {
		return nil, false, err
	}

	turnFailed, turnSucceeded := map[string]bool{}, map[string]bool{}
	terminated := false
	out := make([]Message, len(calls))
	for i, call := range calls {
		res := results[i]
		if res.IsError {
			r.failedCalls++
		}
		if failed[i] {
			turnFailed[call.Name] = true
		} else if call.Tool != nil && !res.IsError {
			turnSucceeded[call.Name] = true
			terminated = terminated || res.Terminate
		}
		out[i] = ToolResult(call.ID, res)
		if err := r.record(out[i]); err != nil {
			return nil, false, err
		}
	}
	for name := range turnFailed {
		if !turnSucceeded[name] {
			r.toolErrors[name]++
		}
	}
	for name := range turnSucceeded {
		delete(r.toolErrors, name)
	}
	return out, terminated, nil
}

func (r *run) tool(name string) *Tool {
	for i := range r.cfg.Tools {
		if r.cfg.Tools[i].Name == name {
			return &r.cfg.Tools[i]
		}
	}
	return nil
}

// runCall runs one call: it checks the call, emits ToolStart, runs it
// through the middleware, and emits ToolEnd. failed reports a failure of the
// tool itself, which counts toward MaxToolErrors; an unknown tool, a refusal
// or a cancellation does not.
func (r *run) runCall(call *ToolCall, bad string) (res Result, failed bool) {
	res, failed, ok := r.check(r.ctx, call, bad)
	if r.emit(ToolStart{Call: *call}) == nil && ok {
		res, failed = r.invoke(r.ctx, *call)
	}
	r.emit(ToolEnd{Call: *call, Result: res})
	return res, failed
}

// check vets a call before it runs: the tool exists and is enabled, the
// arguments are JSON that fits its schema, and its Check passes, setting the
// call's Preview. ok reports that the call may run.
func (r *run) check(ctx context.Context, call *ToolCall, bad string) (res Result, failed, ok bool) {
	t := call.Tool
	switch {
	case ctx.Err() != nil:
		return ErrorResult(interruptedToolResult), false, false
	case t == nil:
		return ErrorResult(fmt.Sprintf("Tool %q does not exist.", call.Name)), false, false
	case bad != "":
		return ErrorResult(bad), true, false
	case r.cfg.MaxToolErrors > 0 && r.toolErrors[call.Name] >= r.cfg.MaxToolErrors:
		return ErrorResult(fmt.Sprintf("Tool %q is disabled after failing %d turns in a row.", call.Name, r.cfg.MaxToolErrors)), false, false
	}
	if err := validateArgs(t.Name, t.Schema, call.Args); err != nil {
		return ErrorResult(err.Error()), true, false
	}
	if t.Check != nil {
		preview, err := t.Check(ctx, call.Args)
		if err != nil {
			return ErrorResult(err.Error()), true, false
		}
		call.Preview = preview
	}
	return Result{}, false, true
}

// invoke runs a checked call through the middleware to the tool, reporting
// its progress as ToolUpdate events.
func (r *run) invoke(ctx context.Context, call ToolCall) (Result, bool) {
	failed := false
	next := ToolFunc(func(ctx context.Context, call ToolCall) (Result, error) {
		res, err := call.Tool.Run(ctx, call.Args)
		if err != nil {
			res = ErrorResult(err.Error())
		}
		failed = res.IsError
		return res, nil
	})
	for i := len(r.cfg.Middleware) - 1; i >= 0; i-- {
		mw, inner := r.cfg.Middleware[i], next
		next = func(ctx context.Context, call ToolCall) (Result, error) { return mw(ctx, call, inner) }
	}
	// Progress reported once the call returned is dropped, so that it never
	// follows the call's ToolEnd.
	var mu sync.Mutex
	open := true
	ctx = WithProgress(ctx, func(p any) {
		mu.Lock()
		defer mu.Unlock()
		if open {
			r.emit(ToolUpdate{Call: call, Progress: p})
		}
	})
	res, err := next(ctx, call)
	mu.Lock()
	open = false
	mu.Unlock()
	if err != nil {
		res = ErrorResult(err.Error())
	}
	return res, failed
}

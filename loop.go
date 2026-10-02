package agentcore

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"sync"
	"time"

	"github.com/voocel/litellm"
)

// Config configures a run.
type Config struct {
	Model Model
	// System is the system prompt: text blocks, which may carry cache
	// breakpoints.
	System []litellm.Block
	Tools  []Tool

	// Emit receives the run's events, one at a time, in order. An error it
	// returns stops the run: the tool calls under way are cancelled, no
	// further event is delivered but the RunEnd carrying the error, and the
	// event refused takes no effect; see MessageEnd and CompactionEnd. Nil
	// drops the events.
	Emit func(Event) error

	// Steering returns the messages to deliver before the next model call,
	// such as what a user typed while the run worked. It is called at the
	// start, after each turn, and after each compaction in the run, which
	// can take long.
	Steering func() []Message
	// FollowUp returns the messages to go on with when the run would stop.
	FollowUp func() []Message
	// OnStop is consulted when the run would stop and neither Steering nor
	// FollowUp has messages: after a response that calls no tool, or a turn
	// a tool Terminated. It returns messages to go on with, none to stop, or
	// an error to end the run with.
	//
	// Messages Steering, FollowUp and OnStop return are recorded at once,
	// even when the run then ends, cancelled or at MaxTurns, before the
	// model answers them.
	OnStop func(ctx context.Context, s StopInfo) ([]Message, error)

	// Middleware wraps every tool call, the first outermost.
	Middleware []ToolMiddleware
	// MaxToolConcurrency bounds the parallel tool calls running at once;
	// below 2, calls run one by one.
	MaxToolConcurrency int
	// MaxToolErrors disables a tool, for the rest of the run, once its calls
	// failed that many turns in a row; 0 never does. A refused call, or one
	// of an unknown tool, does not count.
	MaxToolErrors int

	// Compactor rewrites the history when it grows past CompactAt estimated
	// tokens, before the call that would send it, and when the provider
	// reports a context overflow, after which the call is made again. A
	// compaction at CompactAt that fails is reported and the call made
	// anyway; one on overflow must succeed. Zero CompactAt compacts on
	// overflow only. CompactAt should sit well above what a compaction
	// keeps, or every call compacts again.
	Compactor Compactor
	CompactAt int
	// Cache, if set, places a cache breakpoint after each call's last
	// message, so a call reads the history the one before sent from the
	// prompt cache.
	Cache *litellm.CacheControl

	// MaxTurns bounds the responses of a run; 0 means 100.
	MaxTurns int
	// MaxRetries is how many times a model call that failed transiently,
	// such as on a rate limit or a dropped connection, is made again.
	MaxRetries int
}

// StopInfo is what OnStop decides on.
type StopInfo struct {
	// Message is the last response.
	Message Message
	// Terminated reports that a tool of the last turn terminated the run.
	Terminated bool
	// Turns is how many responses the run had.
	Turns int
}

const (
	defaultMaxTurns       = 100
	maxLengthRecoveries   = 3
	maxRetryDelay         = 60 * time.Second
	lengthRecoveryPrompt  = "Output token limit hit. Resume directly - no apology, no recap of what you were doing. Pick up mid-thought if that is where the cut happened. Break remaining work into smaller pieces."
	interruptedToolResult = "Interrupted: the run was cancelled before this tool call ran."
)

// Run runs cfg's agent on history: it records prompts, then calls the model
// and the tools it asks for, turn by turn, until the model stops, ctx is
// cancelled or an error ends the run. It returns the history the run ended
// with, compactions included, and the error that ended it: ctx's error when
// cancelled. Every message entering the history passes Emit as a MessageEnd
// first, and the last event is always a RunEnd.
//
// Without prompts, Run answers the history as it stands; it fails with
// ErrNothingToContinue when the model has nothing to answer, as when the
// history ends with a response.
func Run(ctx context.Context, cfg Config, history []Message, prompts ...Message) ([]Message, error) {
	runCtx, cancel := context.WithCancelCause(ctx)
	defer cancel(nil)
	r := &run{ctx: runCtx, cancel: cancel, cfg: cfg, history: slices.Clone(history), toolErrors: map[string]int{}}
	err := r.loop(prompts)
	reason := EndDone
	switch {
	case r.failed() != nil:
		reason, err = EndError, r.failed()
	case err == nil:
	case errors.Is(err, ErrMaxTurns):
		reason = EndMaxTurns
	case ctx.Err() != nil:
		reason, err = EndAborted, ctx.Err()
	default:
		reason = EndError
	}
	if cfg.Emit != nil {
		// Delivered whatever Emit returned before; its own error has nothing
		// left to stop.
		_ = cfg.Emit(RunEnd{Reason: reason, Err: err, Turns: r.turns, ToolCalls: r.toolCalls, FailedCalls: r.failedCalls})
	}
	return r.history, err
}

type run struct {
	// ctx is cancelled, with Emit's error as its cause, once Emit fails.
	ctx     context.Context
	cancel  context.CancelCauseFunc
	cfg     Config
	history []Message

	// mu serializes Emit, which tool calls running in parallel share, and
	// keeps the first error it returned.
	mu      sync.Mutex
	emitErr error

	turns, toolCalls, failedCalls int
	lengthRecoveries              int
	// toolErrors counts the turns in a row each tool's calls failed.
	toolErrors map[string]int
}

func (r *run) emit(ev Event) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.emitErr == nil && r.cfg.Emit != nil {
		if err := r.cfg.Emit(ev); err != nil {
			r.emitErr = fmt.Errorf("agentcore: emit %T: %w", ev, err)
			r.cancel(r.emitErr)
		}
	}
	return r.emitErr
}

// failed returns the error Emit failed with, if it did.
func (r *run) failed() error {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.emitErr
}

// record adds msg to the history, once Emit took its MessageEnd. A message
// is timed as it enters the history.
func (r *run) record(msg Message) error {
	msg.Time = time.Now()
	if err := r.emit(MessageEnd{Message: msg}); err != nil {
		return err
	}
	r.history = append(r.history, msg)
	return nil
}

func (r *run) loop(prompts []Message) error {
	maxTurns := r.cfg.MaxTurns
	if maxTurns <= 0 {
		maxTurns = defaultMaxTurns
	}
	pending := append(slices.Clip(prompts), r.steering()...)
	if len(pending) == 0 && !awaitsResponse(r.history) {
		if pending = r.followUp(); len(pending) == 0 {
			return ErrNothingToContinue
		}
	}
	for {
		// Messages taken are recorded before anything can end the run, so
		// that none is lost.
		for _, msg := range pending {
			if err := r.record(msg); err != nil {
				return err
			}
		}
		if err := r.ctx.Err(); err != nil {
			return err
		}
		if r.turns >= maxTurns {
			return fmt.Errorf("%w (%d)", ErrMaxTurns, maxTurns)
		}

		msg, bad, err := r.respond()
		if err != nil {
			return err
		}
		r.turns++
		if msg.Stop == StopError {
			return errors.New("agentcore: the model ended its response with an error")
		}
		results, terminated, err := r.runTools(msg, bad)
		if err != nil {
			return err
		}
		if err := r.emit(TurnEnd{Message: msg, Results: results}); err != nil {
			return err
		}

		stopping := terminated || len(results) == 0
		if stopping && !terminated && msg.Stop == StopLength && r.lengthRecoveries < maxLengthRecoveries {
			r.lengthRecoveries++
			pending = []Message{resumePrompt()}
			continue
		}
		if pending = r.steering(); len(pending) > 0 || !stopping {
			continue
		}
		if pending = r.followUp(); len(pending) > 0 {
			continue
		}
		if r.cfg.OnStop == nil {
			return nil
		}
		next, err := r.cfg.OnStop(r.ctx, StopInfo{Message: msg, Terminated: terminated, Turns: r.turns})
		if err != nil || len(next) == 0 {
			return err
		}
		pending = next
	}
}

// awaitsResponse reports whether the model has something to answer in
// history: the last message it reads is not its own.
func awaitsResponse(history []Message) bool {
	msgs := modelMessages(history)
	return len(msgs) > 0 && msgs[len(msgs)-1].Role != litellm.RoleAssistant
}

// resumePrompt has the model resume a response cut off at the output limit.
func resumePrompt() Message {
	m := UserText(lengthRecoveryPrompt)
	m.Kind = KindResume
	return m
}

func (r *run) steering() []Message {
	if r.cfg.Steering == nil {
		return nil
	}
	return r.cfg.Steering()
}

// steer records the messages Steering has now.
func (r *run) steer() error {
	for _, msg := range r.steering() {
		if err := r.record(msg); err != nil {
			return err
		}
	}
	return nil
}

func (r *run) followUp() []Message {
	if r.cfg.FollowUp == nil {
		return nil
	}
	return r.cfg.FollowUp()
}

// respond gets the model's response to the history and records it, with
// the invalid arguments of its calls fixed (see invalidArgs). Before the
// call, a history grown past CompactAt is compacted, as far as that
// succeeds; on a context overflow, it is compacted and the call made again,
// once. What was steered while it compacted goes into that call. Transient
// failures are retried. A response that fails for good, or is cancelled, is
// recorded with what streamed of it, as StopError or StopAborted, so every
// MessageStart ends with a MessageEnd or a Retry.
func (r *run) respond() (Message, map[string]string, error) {
	compacted := false
	if r.cfg.Compactor != nil && r.cfg.CompactAt > 0 && Estimate(r.cfg, r.history) > r.cfg.CompactAt {
		_, err := r.compact()
		if r.failed() != nil {
			return Message{}, nil, err
		}
		compacted = err == nil
		if err := r.steer(); err != nil {
			return Message{}, nil, err
		}
	}
	for attempt := 0; ; attempt++ {
		msg, started, err := r.call()
		if err != nil && litellm.ErrorTypeOf(err) == litellm.ErrorTypeContextOverflow && !compacted && r.cfg.Compactor != nil {
			compacted = true
			changed, cerr := r.compact()
			if cerr != nil {
				return Message{}, nil, cerr
			}
			if serr := r.steer(); serr != nil {
				return Message{}, nil, serr
			}
			if changed {
				msg, started, err = r.call()
			}
		}
		if err == nil {
			bad := invalidArgs(&msg)
			return msg, bad, r.record(msg)
		}
		cancelled := r.ctx.Err() != nil
		if !cancelled && litellm.IsTemporaryError(err) && attempt < r.cfg.MaxRetries {
			delay := retryDelay(err, attempt)
			if err := r.emit(Retry{Attempt: attempt + 1, MaxRetries: r.cfg.MaxRetries, Delay: delay, Err: err}); err != nil {
				return Message{}, nil, err
			}
			select {
			case <-r.ctx.Done():
				return Message{}, nil, r.ctx.Err()
			case <-time.After(delay):
			}
			continue
		}
		if started {
			msg.Stop = StopError
			if cancelled {
				msg.Stop = StopAborted
			}
			// Its calls are not made: their arguments may be incomplete.
			msg.Blocks = slices.DeleteFunc(msg.Blocks, func(b litellm.Block) bool {
				_, isCall := b.(litellm.ToolUseBlock)
				return isCall
			})
			if rerr := r.record(msg); rerr != nil {
				return Message{}, nil, rerr
			}
		}
		if cancelled {
			return Message{}, nil, r.ctx.Err()
		}
		return Message{}, nil, err
	}
}

// retryDelay is the server's Retry-After, or else exponential backoff from
// one second, at most maxRetryDelay.
func retryDelay(err error, attempt int) time.Duration {
	if d := litellm.RetryAfter(err); d > 0 {
		return min(d, maxRetryDelay)
	}
	return min(time.Second<<attempt, maxRetryDelay)
}

// call makes one streamed model call for the history. On failure it returns
// what streamed before, as a response; started reports that its
// MessageStart was emitted.
func (r *run) call() (msg Message, started bool, err error) {
	msg = Message{Role: litellm.RoleAssistant}
	c := BuildCall(r.cfg, r.history)
	stream, err := c.Client.Stream(r.ctx, c.Request)
	if err != nil {
		return msg, false, err
	}
	defer stream.Close()
	if err := r.emit(MessageStart{}); err != nil {
		return msg, false, err
	}
	resp, err := litellm.Handle(stream, func(ev litellm.Event) error {
		return r.emit(MessageDelta{Event: ev})
	})
	msg.Blocks = resp.Blocks
	msg.Stop = stopReason(resp.FinishReason)
	msg.Usage = usage(resp.Usage, r.cfg.Model.Pricing)
	msg.Provider, msg.Model = resp.Provider, resp.Model
	return msg, true, err
}

// compact has the Compactor rewrite the history and, once Emit took the
// rewrite, puts it in place. It reports whether the history changed.
func (r *run) compact() (bool, error) {
	if err := r.emit(CompactionStart{}); err != nil {
		return false, err
	}
	c, err := r.cfg.Compactor.Compact(r.ctx, r.history, func(h []Message) Call { return BuildCall(r.cfg, h) })
	if err != nil {
		err = fmt.Errorf("agentcore: compact: %w", err)
		if eerr := r.emit(CompactionEnd{Err: err}); eerr != nil {
			return false, eerr
		}
		return false, err
	}
	if err := r.emit(CompactionEnd{Compaction: c}); err != nil {
		return false, err
	}
	if c == nil {
		return false, nil
	}
	r.history = slices.Clone(c.Messages)
	return true, nil
}

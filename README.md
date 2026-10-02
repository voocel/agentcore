# AgentCore

**AgentCore** is a small Go library for building AI agents: a model that calls tools, turn by turn, until its task is done.

[English](README.md) | [中文](README_CN.md)

## Install

```bash
go get github.com/voocel/agentcore
```

Requires Go 1.26 or newer.

## Design

AgentCore sits between a model SDK and an application:

```
litellm      models: messages, blocks, streams, errors, providers
agentcore    the agent: the loop, tools, events, compaction
your app     policy: which model, which tools, approval, storage, UI
```

- **Built on litellm, not wrapping it.** A message is litellm blocks with a role; streamed events are litellm events; errors are litellm errors. There is no second model layer to learn or to keep in sync.
- **A stateless loop.** `Run` takes a history and returns the history it ended with. The caller owns the history; `Agent` keeps one for applications that want it kept.
- **Events are facts.** Every lifecycle signal reaches one callback, in order. A streamed response arrives as deltas, then once as the final message; nothing in an event changes later.
- **Messages are stored as they happen.** Every message entering the history passes `Emit` as a `MessageEnd` first. An error from `Emit` keeps it out and stops the run, so storing there is durable.
- **Policy stays out.** Approval, permissions, prompts and rendering belong to the application; the loop offers the hooks.

## Packages

```
agentcore/            the loop (Run), Agent, Tool, Event, Message, Compactor
agentcore/compact/    Summarizer: replaces older history with a summary
agentcore/tools/      coding tools: read, write, edit, bash, glob, grep, ls; tool_search
agentcore/subagent/   the subagent tool: delegation to sub-agents
agentcore/task/       background tasks, and the tools that follow them
agentcore/schema/     a small JSON Schema builder for tool arguments
```

## Quick Start

```go
provider, err := deepseek.New(deepseek.Config{APIKey: os.Getenv("DEEPSEEK_API_KEY")})
if err != nil {
	log.Fatal(err)
}
client, err := litellm.New(provider)
if err != nil {
	log.Fatal(err)
}

workspace := tools.Workspace{Dir: ".", Files: tools.NewFileReadState()}
cfg := agentcore.Config{
	Model:  agentcore.Model{Client: client, Request: litellm.Request{Model: "deepseek-flash"}},
	System: []litellm.Block{litellm.Text("You are a helpful coding assistant.")},
	Tools:  workspace.Tools(),
	Emit: func(ev agentcore.Event) error {
		switch ev := ev.(type) {
		case agentcore.MessageDelta:
			if d, ok := ev.Event.(litellm.TextDelta); ok {
				fmt.Print(d.Text)
			}
		case agentcore.ToolStart:
			fmt.Printf("\n[%s] %s\n", ev.Call.Name, ev.Call.Args)
		}
		return nil
	},
}
history, err := agentcore.Run(ctx, cfg, nil, agentcore.UserText("What does this project do?"))
```

`Model.Request` is the template of every call: the model's name and settings such as `MaxTokens`, `Thinking` and `ProviderOptions`. The loop sets its messages and tools. Set `Model.Pricing` to have each response's usage priced.

Runnable examples: [`examples/single`](examples/single) and [`examples/multi`](examples/multi).

## Agent

`Agent` keeps a history and runs it, one run at a time:

```go
agent := agentcore.NewAgent(cfg, history)
unsubscribe := agent.Subscribe(func(ev agentcore.Event) error {
	switch e := ev.(type) {
	case agentcore.MessageEnd:
		return store.Append(e.Message) // a failure stops the run
	case agentcore.CompactionEnd:
		if e.Compaction != nil {
			return store.Replace(e.Compaction.Messages) // the whole new history
		}
	}
	return nil
})
defer unsubscribe()

ctx, cancel := context.WithCancel(ctx)
go agent.Prompt(ctx, agentcore.UserText("Fix the failing test"))

agent.Steer(agentcore.UserText("Use the table-driven style")) // reaches the next model call
agent.FollowUp(agentcore.UserText("Then update the docs"))   // runs when it would stop
cancel()                                                     // ends the run
```

A store that appends `MessageEnd` alone restores the history from before its compactions: `Compaction.Messages` is the whole new history, and `Replaced` how many messages of the old one it stands in for. `NewAgent` replaces `Config.Emit`, `Steering` and `FollowUp` with the subscribers, `Steer` and `FollowUp`.

A run ends when its context is cancelled. `Prompt` fails with `ErrBusy` while a run is under way. `Continue` answers the history as it stands, such as after a failed run, and fails with `ErrNothingToContinue` when it ends with a response. `Compact` compacts it on demand, `Messages` and `SetMessages` read and replace it. Every subscriber receives the `RunEnd`, even after one failed.

## Events

`Config.Emit` (or `Agent.Subscribe`) receives the events below. Switch on their type.

| Event | When |
|-------|------|
| `MessageStart` / `MessageDelta` | a response starts / a litellm stream event of it arrives |
| `MessageEnd` | a message enters the history: a prompt, a response, a tool result |
| `ToolStart` / `ToolUpdate` / `ToolEnd` | a tool call starts, before the middleware (approval) / reports progress / ends with its result |
| `TurnEnd` | a response and the results of its tool calls are recorded |
| `Retry` | a model call failed transiently and will be made again |
| `CompactionStart` / `CompactionEnd` | the history is compacted |
| `RunEnd` | the run ends, with its reason, error and counts; always the last event |

Events are delivered one at a time; `Emit` must not block for long. Returning an error from it stops the run: the tool calls under way are cancelled and no further event is delivered but the `RunEnd`, which is always the last. The event refused takes no effect: a refused `MessageEnd` or `CompactionEnd` keeps the message or compaction out of the history.

Messages enter the history in the order they were taken, timed as they do; those taken from `Steering`, `FollowUp` or `OnStop` are recorded even when the run then ends before the model answers them. A response that fails, whether the vendor ended it with an error, the stream broke off or the run was cancelled, is recorded with what streamed of it, `Stop` set to `StopError` or `StopAborted` and its tool calls dropped, so a transcript shows it; it is never sent to the model again.

## Tools

A tool is a struct:

```go
type weatherArgs struct {
	City string `json:"city"`
}

weather := agentcore.NewTool("weather", "Current weather of a city",
	schema.Object(schema.Property("city", schema.String("City name")).Required()),
	func(ctx context.Context, args weatherArgs) (agentcore.Result, error) {
		return agentcore.TextResult("Sunny in " + args.City), nil
	},
)
```

- Arguments are validated against `Schema` before a call runs; the model reads what does not fit.
- `Check` vets a call before it is approved and may return a preview for people, such as the diff `edit` and `write` return, found on `ToolCall.Preview`.
- `Parallel` lets calls run alongside the other parallel calls of their turn, up to `MaxToolConcurrency`.
- `Deferred` tools are offered only once a tool reference in the history names them, as `tool_search` returns (see `tools.Defer`).
- A `Result` holds litellm blocks (text, images, tool references); `Result.Text` is its text. `Terminate` ends the run once the turn is recorded.
- A running tool reports progress with `agentcore.ReportProgress(ctx, v)`; it arrives as `ToolUpdate`. `bash` reports lines of output as strings, `subagent` reports `subagent.Progress`.

`Config.Middleware` wraps every call that passed its checks, for approval, auditing or rewriting arguments:

```go
approve := func(ctx context.Context, call agentcore.ToolCall, next agentcore.ToolFunc) (agentcore.Result, error) {
	if call.Name == "bash" && !askUser(call) {
		return agentcore.ErrorResult("The user declined this command."), nil
	}
	return next(ctx, call)
}
```

## Built-in Tools

A `tools.Workspace` makes the coding tools and holds what they share:

```go
workspace := tools.Workspace{
	Dir:   ".",                       // relative paths resolve here; not a sandbox
	FS:    nil,                       // read/write/edit backend; nil is the local filesystem
	Files: tools.NewFileReadState(), // read-before-write checks; nil checks nothing
	Tasks: tasks,                     // bash's background commands; nil offers none
}
cfg.Tools = workspace.Tools() // or workspace.Read(), workspace.Bash(), ...
```

| Tool | Arguments | Result |
|------|-----------|--------|
| `read` | `file_path`, `offset`, `limit` | numbered lines (2000 lines / 50KB at most), a directory listing, or an image |
| `write` | `file_path`, `content` | a line saying what it wrote; its preview is the diff |
| `edit` | `file_path`, `old_string`, `new_string`, `replace_all` | the file and the diff; exact, then whitespace- and indentation-tolerant matching |
| `bash` | `command`, `timeout`, `workdir`, `description`, `run_in_background` (with `Tasks`) | the tail of the output (2000 lines / 50KB), then `[exit code N]`, `[timed out after …]` or where the full output went |
| `glob` | `pattern`, `path` | matching paths, newest first |
| `grep` | `pattern`, `path`, `glob`, `ignore_case`, `literal`, `context_lines`, `limit` | `path:line:text` matches |
| `ls` | `path`, `depth`, `ignore` | a tree |

Results are plain text. A failing command is not a failed `bash` call: its output and exit code are what the model needs. With `Files`, `write` refuses an existing file the model has not read whole, and `write` and `edit` one that changed since it was read. The working directory of a call's context (`tools.WithCwd`) overrides `Dir`, as for a git worktree entered mid-run. `bash` needs a POSIX shell (`bash` or `sh`) on PATH; on Windows that means Git Bash.

`tools.Defer(tools)` puts tools behind `tool_search` (`query`, `max_results`), whose description lists their names: the model sees their schemas only once it searched for them.

## Background Tasks

```go
tasks := task.NewRuntime(dir, func(m agentcore.Message) { agent.FollowUp(m) })
cfg.Tools = append(cfg.Tools, tasks.Tools()...) // task_output (task_id, wait, timeout), task_stop (task_id)
```

`bash` with `run_in_background` and the `subagent` tool's background mode run work as tasks of a `task.Runtime`: each has an ID, a status and an output file in `dir`, and runs until it ends or is stopped, with no timeout unless asked. When one ends, the Runtime hands the notify function a message announcing it, of `task.KindNotification`, to deliver as a follow-up. `Runtime.Start` runs work of your own as a task.

## Compaction

```go
cfg.Compactor = compact.Summarizer{}
cfg.CompactAt = 100_000
```

The loop compacts before a call whose history is estimated above `CompactAt`, and once when the provider reports a context overflow, then makes the call again; the first may fail without ending the run, the second may not. The estimate counts from the last response's reported input tokens. `CompactAt` should sit well above what a compaction keeps, or every call compacts again.

`compact.Summarizer` keeps the recent messages verbatim, a quarter of the history between 2k and 20k tokens, and replaces the rest with a checkpoint the conversation's own model writes: it extends the conversation's call for the part it replaces, which is served from the prompt cache, and asks for the checkpoint in `<summary>` tags; when that does not fit, or the answer has no tags, it asks a second time with a plain-text transcript. The checkpoint lists the files the replaced part read and changed, and the tools it loaded stay loaded. Implement `agentcore.Compactor` for another strategy.

## Sub-agents

```go
delegate := subagent.New(tasks, // nil offers no background mode
	subagent.Agent{
		Name:        "scout",
		Description: "Fast codebase reconnaissance",
		Config: func(s subagent.Spawn) (agentcore.Config, error) {
			return agentcore.Config{Model: model, System: scoutPrompt, Tools: readOnlyTools()}, nil
		},
	},
)
```

The model calls `subagent` with `agent` and `task` to run one agent; with `tasks`, an array of `{agent, task}`, to run several in parallel, 8 at a time; with `chain` to run them in order, `{previous}` in a task standing for the output of the step before; with `background` (given a `task.Runtime`) to run one as a task; `model` picks another model. Each run gets its own `Config`, so its tools keep their own state; its events go to that `Config.Emit`, and to the waiting call as `subagent.Progress`. A run that fails reports its error with what it said last. Runs nest at most `subagent.MaxDepth` deep.

## Config

| Field | Description |
|-------|-------------|
| `Model` | The litellm client, the request template and the pricing |
| `System` / `Tools` | The system prompt blocks and the tools on offer |
| `Emit` | Receives the events; an error stops the run |
| `Steering` / `FollowUp` | Messages to deliver before the next call / when the run would stop |
| `OnStop` | Consulted when the run would stop: go on, stop, or fail |
| `Middleware` | Wraps every tool call, the first outermost |
| `MaxToolConcurrency` | Parallel calls running at once (below 2, one by one) |
| `MaxToolErrors` | Disables a tool failing that many turns in a row (0 never) |
| `Compactor` / `CompactAt` | Compaction, see above |
| `Cache` | A cache breakpoint after each call's last message |
| `MaxTurns` | Responses per run (0 means 100) |
| `MaxRetries` | Retries of a transient model failure, with backoff |

## License

Apache License 2.0

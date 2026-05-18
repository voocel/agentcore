# AgentCore

**AgentCore** is a minimal, composable Go library for building AI agent applications.

[English](README.md) | [中文](README_CN.md)

## Install

```bash
go get github.com/voocel/agentcore
```

## Design Philosophy

A restrained core with open extensibility tends to be more reliable than a complex all-in-one solution. Fewer built-ins, more possibilities.

## Stability

- Keep `Agent`, `AgentLoop`, `Event`, `Tool`, and `Message` stable first
- `examples/` and internal implementation details are not stable API

## Architecture

```
agentcore/            Agent core (types, loop, agent, events)
agentcore/llm/        LLM adapters (OpenAI, Anthropic, Gemini via litellm)
agentcore/tools/      Built-in tools: read, write, edit, bash
agentcore/context/    Context runtime — projection, rewrite, overflow recovery
agentcore/task/       Background task registry (Runtime / Entry) shared by bash + subagent
agentcore/subagent/   SubAgent tool — multi-agent via tool invocation
agentcore/proxy/      ChatModel adapter that forwards calls to a remote proxy
agentcore/permission/ Optional permission engine — adapt to ToolGate yourself
```

Core design:

- **Standalone loop + stateful Agent** — `loop.go` is a free function with all dependencies injected; `agent.go` is the sole consumer of loop events, updating internal state and dispatching to listeners. Double loop: inner processes tool calls + steering, outer handles follow-up
- **Event stream** — single `<-chan Event` output drives any UI (TUI, Web, Slack, logging)
- **Context layer** — `ContextManager` (interface) + `agentcore/context` (default engine) drive prompt projection, overflow recovery, and—via auto-wiring—message conversion and token estimation
- **SubAgent tool** (`subagent/`) — multi-agent via tool invocation, four modes: single, parallel, chain, background

## Quick Start

### Single Agent

```go
package main

import (
    "fmt"
    "os"

    "github.com/voocel/agentcore"
    "github.com/voocel/agentcore/llm"
    "github.com/voocel/agentcore/tools"
)

func main() {
    model, err := llm.NewModel("openai", "gpt-5-mini", llm.WithAPIKey(os.Getenv("OPENAI_API_KEY")))
    if err != nil {
        panic(err)
    }

    agent := agentcore.NewAgent(
        agentcore.WithModel(model),
        agentcore.WithSystemPrompt("You are a helpful coding assistant."),
        agentcore.WithTools(
            tools.NewRead("."),
            tools.NewWrite("."),
            tools.NewEdit("."),
            tools.NewBash("."),
        ),
    )

    agent.Subscribe(func(ev agentcore.Event) {
        if ev.Type == agentcore.EventMessageEnd {
            if msg, ok := ev.Message.(agentcore.Message); ok && msg.Role == agentcore.RoleAssistant {
                fmt.Println(msg.Content)
            }
        }
    })

    agent.Prompt("List the files in the current directory.")
    agent.WaitForIdle()
}
```

For tool-call gating, register a `ToolGate` — a single hook called once per tool call after argument validation. The kernel implements no permission policy of its own; gates are user-supplied.

```go
gate := func(ctx context.Context, req agentcore.GateRequest) (*agentcore.GateDecision, error) {
    if req.Call.Name == "bash" {
        return &agentcore.GateDecision{Allowed: false, Reason: "bash disabled"}, nil
    }
    return &agentcore.GateDecision{Allowed: true}, nil
}

agent := agentcore.NewAgent(
    // ... model, tools, etc.
    agentcore.WithToolGate(gate),
)
```

The optional `agentcore/permission` subpackage offers a richer decision engine (modes, rules, filesystem roots, audit). Adapt it to `ToolGate` with a small wrapper.

### Multi-Agent (SubAgent Tool)

Sub-agents are invoked as regular tools with isolated contexts. Import the
`agentcore/subagent` subpackage:

```go
import (
    "github.com/voocel/agentcore"
    "github.com/voocel/agentcore/llm"
    "github.com/voocel/agentcore/subagent"
    "github.com/voocel/agentcore/tools"
)

model, _ := llm.NewModel("openai", "gpt-5-mini", llm.WithAPIKey(apiKey))

scout := subagent.Config{
    Name:         "scout",
    Description:  "Fast codebase reconnaissance",
    Model:        model,
    SystemPrompt: "Quickly explore and report findings. Be concise.",
    Tools:        []agentcore.Tool{tools.NewRead("."), tools.NewBash(".")},
    MaxTurns:     5,
}

worker := subagent.Config{
    Name:         "worker",
    Description:  "General-purpose executor",
    Model:        model,
    SystemPrompt: "Implement tasks given to you.",
    Tools:        []agentcore.Tool{tools.NewRead("."), tools.NewWrite("."), tools.NewEdit("."), tools.NewBash(".")},
}

agent := agentcore.NewAgent(
    agentcore.WithModel(model),
    agentcore.WithTools(subagent.New(scout, worker)),
)
```

For background mode (async sub-agent runs that notify on completion), wire a
shared task runtime:

```go
import "github.com/voocel/agentcore/task"

rt := task.NewRuntime()
sat := subagent.New(scout, worker)
sat.SetTaskRuntime(rt)
sat.SetNotifyFn(agent.FollowUp) // route completion notifications back to the parent
```

Four execution modes via tool call:

```jsonc
// Single: one agent, one task
{"agent": "scout", "task": "Find all API endpoints"}

// Parallel: concurrent execution
{"tasks": [{"agent": "scout", "task": "Find auth code"}, {"agent": "scout", "task": "Find DB schema"}]}

// Chain: sequential with {previous} context passing
{"chain": [{"agent": "scout", "task": "Find auth code"}, {"agent": "worker", "task": "Refactor based on: {previous}"}]}

// Background: async execution, returns immediately, notifies on completion
{"agent": "worker", "task": "Run full test suite", "background": true, "description": "Running tests"}
```

### Steering & Injection

`Inject(msg)` delivers a message according to the agent's current state — preferred when the caller's intent is "deliver this as soon as possible" without manually branching on running vs idle:

```go
result, _ := agent.Inject(agentcore.UserMsg("Re-check unfinished tasks before stopping."))
fmt.Println(result.Disposition)
```

Outcomes:

- `steered_current_run` — agent was running; message went into the current run's steering path
- `resumed_idle_run` — agent was idle with an assistant-tail conversation; message queued and `Continue()` started
- `queued` — message queued, no run started

For finer control, use the lower-level APIs directly:

```go
agent.Steer(agentcore.UserMsg("Stop and focus on tests instead.")) // mid-run interrupt
agent.FollowUp(agentcore.UserMsg("Now run the tests."))            // queue for after current run
agent.Abort()                                                      // cancel immediately
```

If a message must be merged into the next explicit user prompt (rather than the agent's queues), keep that in the application layer.

### Event Stream

All lifecycle events flow through a single channel — subscribe to drive any UI:

```go
agent.Subscribe(func(ev agentcore.Event) {
    switch ev.Type {
    case agentcore.EventMessageStart:    // assistant starts streaming
    case agentcore.EventMessageUpdate:   // streaming token delta
    case agentcore.EventMessageEnd:      // message complete
    case agentcore.EventToolExecStart:   // tool execution begins
    case agentcore.EventToolExecEnd:     // tool execution ends
    case agentcore.EventError:           // error occurred
    }
})
```

### Structured Tool Progress

Long-running tools can emit structured progress updates instead of ad-hoc JSON:

```go
agentcore.ReportToolProgress(ctx, agentcore.ProgressPayload{
    Kind:    agentcore.ProgressSummary,
    Agent:   "worker",
    Tool:    "bash",
    Summary: "worker → bash",
})
```

Subscribers should read `ev.Progress` directly for tool progress updates:

```go
agent.Subscribe(func(ev agentcore.Event) {
    if ev.Type == agentcore.EventToolExecUpdate && ev.Progress != nil {
        fmt.Printf("[%s] %s\n", ev.Progress.Kind, ev.Progress.Summary)
    }
})
```

### Swappable Models

When a model needs to change at runtime, wrap it with `SwappableModel`. The swap takes effect on the next call. `subagent.Config.Model` is resolved at the start of each sub-agent run, so the same wrapper also works for sub-agents.

```go
defaultModel, _ := llm.NewModel("openai", "gpt-5-mini", llm.WithAPIKey(apiKey))
sw := agentcore.NewSwappableModel(defaultModel)

agent := agentcore.NewAgent(agentcore.WithModel(sw))

nextModel, _ := llm.NewModel("openai", "gpt-5", llm.WithAPIKey(apiKey))
sw.Swap(nextModel) // next turn uses the new model
```

### Custom LLM Adapter

To swap the LLM call with a proxy, mock, or custom implementation, implement
the `ChatModel` interface and pass it via `WithModel`. `SwappableModel` and
the `agentcore/proxy` subpackage are built on this same interface and can
serve as references.

### Context Compaction

Auto-summarize conversation history when approaching the context window limit. Use the built-in context manager:

```go
import (
    "github.com/voocel/agentcore"
    agentctx "github.com/voocel/agentcore/context"
)

engine := agentctx.NewDefaultEngine(model, 128000)

agent := agentcore.NewAgent(
    agentcore.WithModel(model),
    agentcore.WithContextManager(engine),
)
```

`NewAgent` auto-wires `ConvertToLLM`, token estimation, and context window from the context manager when available.

When usage exceeds `ContextWindow - ReserveTokens` (default 16384), compaction:

1. Keeps recent messages (default 20000 tokens)
2. Summarizes older messages via LLM into a structured checkpoint (Goal / Progress / Key Decisions / Next Steps)
3. Tracks file operations (read/write/edit paths) across compacted messages
4. Supports incremental updates — subsequent compactions update the existing summary rather than re-summarizing

## Built-in Tools

| Tool | Description |
|------|-------------|
| `read` | Read file contents with head truncation (2000 lines / 50KB) |
| `write` | Write file with auto-mkdir |
| `edit` | Exact text replacement with fuzzy match, BOM/line-ending normalization, unified diff output |
| `bash` | Execute shell commands with tail truncation (2000 lines / 50KB) |

## API Reference

### Agent

| Method | Description |
|--------|-------------|
| `NewAgent(opts...)` | Create agent with options |
| `Prompt(input)` | Start new conversation turn |
| `PromptMessages(msgs...)` | Start turn with arbitrary AgentMessages |
| `Continue()` | Resume from current context |
| `Inject(msg)` | Deliver message via steer / idle resume / queue, depending on current state |
| `Steer(msg)` | Inject steering message mid-run |
| `FollowUp(msg)` | Queue message for after completion |
| `Abort()` | Cancel current execution |
| `AbortSilent()` | Cancel without emitting abort marker |
| `WaitForIdle()` | Block until agent finishes |
| `Subscribe(fn)` | Register event listener |
| `State()` | Snapshot of current state |
| `ExportMessages()` | Export messages for serialization |
| `ImportMessages(msgs)` | Import deserialized messages |
| `BuildLLMMessages()` | Materialize the next-call prompt (system → projected history) |

## License

Apache License 2.0

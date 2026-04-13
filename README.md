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
- Behavioral changes should come with tests first; `examples/` and internal implementation details are not stable API

## Architecture

```
agentcore/            Agent core (types, loop, agent, events, subagent)
agentcore/llm/        LLM adapters (OpenAI, Anthropic, Gemini via litellm)
agentcore/tools/      Built-in tools: read, write, edit, bash
agentcore/context/    Context runtime — projection, rewrite, overflow recovery
```

Core design:

- **Standalone dual-loop core** (`loop.go`) — free function, all dependencies injected via parameters. Double loop: inner processes tool calls + steering, outer handles follow-up
- **Stateful Agent** (`agent.go`) — sole consumer of loop events, updates internal state then dispatches to external listeners
- **Event stream** — single `<-chan Event` output drives any UI (TUI, Web, Slack, logging)
- **Two-stage pipeline** — `TransformContext` (prune/inject) → `ConvertToLLM` (filter to LLM messages)
- **SubAgent tool** (`subagent.go`) — multi-agent via tool invocation, four modes: single, parallel, chain, background
- **Context runtime** (`context/`) — projection, committed rewrite, and overflow recovery near the context window limit

## Quick Start

### Single Agent

```go
package main

import (
    "fmt"
    "os"

    "github.com/voocel/agentcore"
    "github.com/voocel/agentcore/llm"
    "github.com/voocel/agentcore/permission"
    "github.com/voocel/agentcore/tools"
)

func main() {
    model, err := llm.NewOpenAIModel("gpt-5-mini", os.Getenv("OPENAI_API_KEY"))
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
        agentcore.WithPermissionEngine(permission.NewEngine(permission.EngineConfig{
            Workspace: ".",
            Mode:      permission.ModeBalanced,
        })),
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

For stricter control, pass a custom decision engine with `agentcore.WithPermissionEngine(...)`.

```go
engine := permission.NewEngine(permission.EngineConfig{
    Workspace: ".",
    Mode:      permission.ModeStrict,
    Roots: permission.FilesystemRoots{
        ReadRoots:  []string{"."},
        WriteRoots: []string{"."},
    },
})
```

### Multi-Agent (SubAgent Tool)

Sub-agents are invoked as regular tools with isolated contexts:

```go
model, _ := llm.NewOpenAIModel("gpt-5-mini", apiKey)

scout := agentcore.SubAgentConfig{
    Name:         "scout",
    Description:  "Fast codebase reconnaissance",
    Model:        model,
    SystemPrompt: "Quickly explore and report findings. Be concise.",
    Tools:        []agentcore.Tool{tools.NewRead("."), tools.NewBash(".")},
    MaxTurns:     5,
}

worker := agentcore.SubAgentConfig{
    Name:         "worker",
    Description:  "General-purpose executor",
    Model:        model,
    SystemPrompt: "Implement tasks given to you.",
    Tools:        []agentcore.Tool{tools.NewRead("."), tools.NewWrite("."), tools.NewEdit("."), tools.NewBash(".")},
}

agent := agentcore.NewAgent(
    agentcore.WithModel(model),
    agentcore.WithTools(agentcore.NewSubAgentTool(scout, worker)),
)
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

### Steering & Follow-Up

```go
// Interrupt mid-run (delivered after current tool, remaining tools skipped)
agent.Steer(agentcore.UserMsg("Stop and focus on tests instead."))

// Queue for after the agent finishes
agent.FollowUp(agentcore.UserMsg("Now run the tests."))

// Cancel immediately
agent.Abort()
```

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

When a model needs to change at runtime, wrap it with `SwappableModel`. The swap takes effect on the next call. `SubAgentConfig.Model` is resolved at the start of each sub-agent run, so the same wrapper also works for sub-agents.

```go
defaultModel, _ := llm.NewOpenAIModel("gpt-5-mini", apiKey)
sw := agentcore.NewSwappableModel(defaultModel)

agent := agentcore.NewAgent(agentcore.WithModel(sw))

nextModel, _ := llm.NewOpenAIModel("gpt-5", apiKey)
sw.Swap(nextModel) // next turn uses the new model
```

### Custom LLM (StreamFn)

Swap the LLM call with a proxy, mock, or custom implementation:

```go
agent := agentcore.NewAgent(
    agentcore.WithStreamFn(func(ctx context.Context, req *agentcore.LLMRequest) (*agentcore.LLMResponse, error) {
        // Route to your own proxy/gateway
        return callMyProxy(ctx, req)
    }),
)
```

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

On each LLM call, the context manager first builds a projected prompt view for the next model request. When a rewrite should become the new runtime baseline, it can return `ShouldCommit=true` with `CommitMessages`, and the loop will replace the in-memory baseline before continuing.

When usage exceeds `ContextWindow - ReserveTokens` (default 16384), compaction:

1. Keeps recent messages (default 20000 tokens)
2. Summarizes older messages via LLM into a structured checkpoint (Goal / Progress / Key Decisions / Next Steps)
3. Tracks file operations (read/write/edit paths) across compacted messages
4. Supports incremental updates — subsequent compactions update the existing summary rather than re-summarizing

### Context Pipeline

For simpler transform-only pipelines, `WithContextPipeline` / `WithTransformContext` still work:

```go
agent := agentcore.NewAgent(
    // Stage 1: prune old messages, inject external context
    agentcore.WithTransformContext(func(ctx context.Context, msgs []agentcore.AgentMessage) ([]agentcore.AgentMessage, error) {
        if len(msgs) > 100 {
            msgs = msgs[len(msgs)-50:]
        }
        return msgs, nil
    }),
    // Stage 2: filter to LLM-compatible messages
    agentcore.WithConvertToLLM(func(msgs []agentcore.AgentMessage) []agentcore.Message {
        var out []agentcore.Message
        for _, m := range msgs {
            if msg, ok := m.(agentcore.Message); ok {
                out = append(out, msg)
            }
        }
        return out
    }),
)
```

## Built-in Tools

| Tool | Description |
|------|-------------|
| `read` | Read file contents with head truncation (2000 lines / 50KB) |
| `write` | Write file with auto-mkdir |
| `edit` | Exact text replacement with fuzzy match, BOM/line-ending normalization, unified diff output |
| `bash` | Execute shell commands with tail truncation (2000 lines / 50KB) |

## Runtime Injection

Use `Inject(msg)` when the caller's intent is "deliver this as soon as the current
agent state allows" without manually branching on running vs idle state.

```go
result, err := agent.Inject(agentcore.UserMsg("Re-check unfinished tasks before stopping."))
if err != nil {
    panic(err)
}
fmt.Println(result.Disposition)
```

`Inject` has three outcomes:

- `steered_current_run`: the agent is running, so the message was queued into the current run's steering path
- `resumed_idle_run`: the agent was idle with an assistant-tail conversation, so the message was queued and `Continue()` was started immediately
- `queued`: the message was queued, but no run was started

Use the lower-level APIs when you need stricter control:

- `Steer(msg)`: queue for the steering path without any idle auto-resume logic
- `FollowUp(msg)`: queue for after the current run stops
- prompt-side injection: keep this in the application layer if the message must be merged into the next explicit user prompt rather than the agent queues

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

## License

Apache License 2.0

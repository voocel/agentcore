# AgentCore

**AgentCore** 是一个极简、可组合的 Go Agent 核心库，用于构建任意 AI Agent 应用。

[English](README.md) | [中文](README_CN.md)

## 安装

```bash
go get github.com/voocel/agentcore
```

## 设计哲学

克制的内核，开放的扩展，往往比面面俱到的一体化更可靠。越少的内置，越多的可能。

## 稳定性

- 优先稳定 `Agent`、`AgentLoop`、`Event`、`Tool`、`Message` 这些核心接口
- 行为变化先补测试，再更新说明；`examples/` 与内部实现细节不视为稳定 API

## 架构

```
agentcore/            Agent 核心（类型、循环、Agent、事件、SubAgent）
agentcore/llm/        LLM 适配层（OpenAI, Anthropic, Gemini，基于 litellm）
agentcore/tools/      内置工具：read, write, edit, bash
agentcore/memory/     上下文压缩 —— 自动摘要长对话
```

核心设计：

- **独立函数循环**（`loop.go`）—— 无结构体依赖，所有输入通过参数注入。双层循环：内层处理工具调用 + steering 中断，外层处理 follow-up 续接
- **有状态 Agent**（`agent.go`）—— 作为循环事件的唯一消费者，更新内部状态后分发给外部监听者
- **事件流** —— 单一 `<-chan Event` 输出，驱动任何 UI（TUI、Web、Slack、日志）
- **两阶段管道** —— `TransformContext`（裁剪/注入）→ `ConvertToLLM`（过滤为 LLM 消息）
- **SubAgent 工具**（`subagent.go`）—— 通过工具调用实现多 Agent，四种模式：single、parallel、chain、background
- **上下文压缩**（`memory/`）—— 接近上下文窗口上限时自动摘要压缩

## 快速开始

### 单 Agent

```go
package main

import (
    "fmt"
    "os"

    "github.com/voocel/agentcore"
    "github.com/voocel/agentcore/llm"
    "github.com/voocel/agentcore/policy"
    "github.com/voocel/agentcore/tools"
)

func main() {
    model, err := llm.NewOpenAIModel("gpt-5-mini", os.Getenv("OPENAI_API_KEY"))
    if err != nil {
        panic(err)
    }

    agent := agentcore.NewAgent(
        agentcore.WithModel(model),
        agentcore.WithSystemPrompt("你是一个编程助手。"),
        agentcore.WithTools(
            tools.NewRead("."),
            tools.NewWrite("."),
            tools.NewEdit("."),
            tools.NewBash("."),
        ),
        agentcore.WithPermission(policy.WorkspaceProfile(".")),
    )

    agent.Subscribe(func(ev agentcore.Event) {
        if ev.Type == agentcore.EventMessageEnd {
            if msg, ok := ev.Message.(agentcore.Message); ok && msg.Role == agentcore.RoleAssistant {
                fmt.Println(msg.Content)
            }
        }
    })

    agent.Prompt("列出当前目录下的文件。")
    agent.WaitForIdle()
}
```

更安全的默认配置可使用 `policy.ReadOnlyProfile(root)` 或 `policy.WorkspaceProfile(root)`。

### 多 Agent（SubAgent 工具）

子 Agent 作为普通工具被调用，各自拥有隔离的上下文：

```go
model, _ := llm.NewOpenAIModel("gpt-5-mini", apiKey)

scout := agentcore.SubAgentConfig{
    Name:         "scout",
    Description:  "快速代码侦察",
    Model:        model,
    SystemPrompt: "快速探索代码库并汇报发现。简洁明了。",
    Tools:        []agentcore.Tool{tools.NewRead("."), tools.NewBash(".")},
    MaxTurns:     5,
}

worker := agentcore.SubAgentConfig{
    Name:         "worker",
    Description:  "通用执行者",
    Model:        model,
    SystemPrompt: "执行分配给你的任务。",
    Tools:        []agentcore.Tool{tools.NewRead("."), tools.NewWrite("."), tools.NewEdit("."), tools.NewBash(".")},
}

agent := agentcore.NewAgent(
    agentcore.WithModel(model),
    agentcore.WithTools(agentcore.NewSubAgentTool(scout, worker)),
)
```

LLM 通过工具调用触发四种执行模式：

```jsonc
// Single：单个 agent 执行单个任务
{"agent": "scout", "task": "找到所有 API 端点"}

// Parallel：多个 agent 并发执行
{"tasks": [{"agent": "scout", "task": "查找认证代码"}, {"agent": "scout", "task": "查找数据库 schema"}]}

// Chain：顺序执行，{previous} 传递上一步输出
{"chain": [{"agent": "scout", "task": "查找认证代码"}, {"agent": "worker", "task": "基于以下内容重构: {previous}"}]}

// Background：后台异步执行，立即返回，完成后通知
{"agent": "worker", "task": "运行完整测试套件", "background": true, "description": "正在执行测试"}
```

### Steering 与 Follow-Up

```go
// 中断：在当前工具执行完毕后注入，跳过剩余工具
agent.Steer(agentcore.UserMsg("停下来，改为专注于测试。"))

// 续接：排队等 agent 完成后处理
agent.FollowUp(agentcore.UserMsg("现在运行测试。"))

// 立即取消
agent.Abort()
```

### 事件流

所有生命周期事件通过单一通道输出 —— 订阅即可驱动任何 UI：

```go
agent.Subscribe(func(ev agentcore.Event) {
    switch ev.Type {
    case agentcore.EventMessageStart:    // assistant 开始流式输出
    case agentcore.EventMessageUpdate:   // 流式 token 增量
    case agentcore.EventMessageEnd:      // 消息完成
    case agentcore.EventToolExecStart:   // 工具开始执行
    case agentcore.EventToolExecEnd:     // 工具执行完毕
    case agentcore.EventError:           // 发生错误
    }
})
```

### 自定义 LLM（StreamFn）

替换 LLM 调用为代理、Mock 或自定义实现：

```go
agent := agentcore.NewAgent(
    agentcore.WithStreamFn(func(ctx context.Context, req *agentcore.LLMRequest) (*agentcore.LLMResponse, error) {
        // 路由到你自己的代理/网关
        return callMyProxy(ctx, req)
    }),
)
```

### 上下文压缩

对话历史接近上下文窗口上限时自动摘要压缩。通过 `TransformContext` 钩子接入，零侵入核心代码：

```go
import "github.com/voocel/agentcore/memory"

agent := agentcore.NewAgent(
    agentcore.WithModel(model),
    agentcore.WithTransformContext(memory.NewCompaction(memory.CompactionConfig{
        Model:         model,
        ContextWindow: 128000,
    })),
    agentcore.WithConvertToLLM(memory.CompactionConvertToLLM),
)
```

每次 LLM 调用前，compaction 检查总 token 数。当超出 `ContextWindow - ReserveTokens`（默认 16384）时：

1. 保留最近消息（默认 20000 tokens）
2. 通过 LLM 将旧消息摘要为结构化检查点（Goal / Progress / Key Decisions / Next Steps）
3. 跨压缩消息追踪文件操作（read/write/edit 路径）
4. 支持增量更新 —— 后续压缩基于已有摘要更新，而非重新总结

### 上下文管道

```go
agent := agentcore.NewAgent(
    // 阶段 1：裁剪旧消息，注入外部上下文
    agentcore.WithTransformContext(func(ctx context.Context, msgs []agentcore.AgentMessage) ([]agentcore.AgentMessage, error) {
        if len(msgs) > 100 {
            msgs = msgs[len(msgs)-50:]
        }
        return msgs, nil
    }),
    // 阶段 2：过滤为 LLM 兼容的消息
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

## 内置工具

| 工具 | 说明 |
|------|------|
| `read` | 读取文件内容，head 截断（2000 行 / 50KB） |
| `write` | 写入文件，自动创建目录 |
| `edit` | 精确文本替换，支持模糊匹配、BOM/行ending 归一化、unified diff 输出 |
| `bash` | 执行 shell 命令，tail 截断（2000 行 / 50KB） |

## API 参考

### Agent

| 方法 | 说明 |
|------|------|
| `NewAgent(opts...)` | 创建 Agent |
| `Prompt(input)` | 发起新对话轮次 |
| `PromptMessages(msgs...)` | 用任意 AgentMessage 发起对话 |
| `Continue()` | 从当前上下文继续 |
| `Steer(msg)` | 中断注入 steering 消息 |
| `FollowUp(msg)` | 排队 follow-up 消息 |
| `Abort()` | 取消当前执行 |
| `AbortSilent()` | 静默取消（不发 abort 标记） |
| `WaitForIdle()` | 阻塞等待完成 |
| `Subscribe(fn)` | 注册事件监听 |
| `State()` | 获取当前状态快照 |
| `ExportMessages()` | 导出消息用于序列化 |
| `ImportMessages(msgs)` | 导入反序列化的消息 |

## 许可证

Apache License 2.0

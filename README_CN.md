# AgentCore

**AgentCore** 是一个用来构建 AI Agent 的小型 Go 库：模型逐轮调用工具，直到完成任务。

[English](README.md) | [中文](README_CN.md)

## 安装

```bash
go get github.com/voocel/agentcore
```

需要 Go 1.26 或更高版本。

## 设计

AgentCore 位于模型 SDK 和应用之间：

```
litellm      模型：消息、块、流、错误、各家 provider
agentcore    Agent：循环、工具、事件、压缩
你的应用      策略：用哪个模型、哪些工具、审批、存储、界面
```

- **建立在 litellm 之上，而不是再包一层。** 消息就是带角色的 litellm 块；流式事件就是 litellm 事件；错误就是 litellm 错误。没有第二套模型层要学，也没有要保持同步的转换。
- **无状态循环。** `Run` 接收一段历史，返回运行结束时的历史。历史归调用方所有；需要有人替它保管历史的应用用 `Agent`。
- **事件是事实。** 所有生命周期信号按顺序到达同一个回调。流式响应先以增量到达，再以最终消息到达一次；事件里的内容之后不会再变。
- **消息在发生时落盘。** 每条进入历史的消息都先以 `MessageEnd` 经过 `Emit`。`Emit` 返回错误时，这条消息不进入历史，运行停止，所以在这里存储就是持久的。
- **策略留在外面。** 审批、权限、提示词、渲染属于应用；循环只提供钩子。

## 包

```
agentcore/            循环（Run）、Agent、Tool、Event、Message、Compactor
agentcore/compact/    Summarizer：用摘要替换较早的历史
agentcore/tools/      编码工具：read、write、edit、bash、glob、grep、ls；tool_search
agentcore/subagent/   subagent 工具：把任务委派给子 agent
agentcore/task/       后台任务，以及查看、停止它们的工具
agentcore/schema/     为工具参数构建 JSON Schema 的小工具
```

## 快速开始

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

`Model.Request` 是每次调用的模板：模型名以及 `MaxTokens`、`Thinking`、`ProviderOptions` 等设置。消息和工具由循环填入。设置 `Model.Pricing` 后每个响应的用量都会计价。

可运行的示例：[`examples/single`](examples/single) 和 [`examples/multi`](examples/multi)。

## Agent

`Agent` 保管一段历史并运行它，同一时间只有一次运行：

```go
agent := agentcore.NewAgent(cfg, history)
unsubscribe := agent.Subscribe(func(ev agentcore.Event) error {
	switch e := ev.(type) {
	case agentcore.MessageEnd:
		return store.Append(e.Message) // 失败会停止运行
	case agentcore.CompactionEnd:
		if e.Compaction != nil {
			return store.Replace(e.Compaction.Messages) // 完整的新历史
		}
	}
	return nil
})
defer unsubscribe()

ctx, cancel := context.WithCancel(ctx)
go agent.Prompt(ctx, agentcore.UserText("Fix the failing test"))

agent.Steer(agentcore.UserText("Use the table-driven style")) // 送达下一次模型调用
agent.FollowUp(agentcore.UserText("Then update the docs"))   // 在本该停止时继续
cancel()                                                     // 结束运行
```

只追加 `MessageEnd` 的存储恢复出来的是压缩前的历史：`Compaction.Messages` 是完整的新历史，`Replaced` 是它替代了旧历史中的多少条。`NewAgent` 会用订阅者、`Steer` 和 `FollowUp` 替换 `Config.Emit`、`Steering` 和 `FollowUp`。

取消 ctx 即结束运行。运行进行中时 `Prompt` 返回 `ErrBusy`。`Continue` 就现有历史作答（如运行失败之后），历史以响应结尾时返回 `ErrNothingToContinue`。`Compact` 按需压缩，`Messages` 和 `SetMessages` 读取和替换历史。每个订阅者都会收到 `RunEnd`，即使前面有订阅者失败。

## 事件

`Config.Emit`（或 `Agent.Subscribe`）接收以下事件，按类型分支处理。

| 事件 | 时机 |
|------|------|
| `MessageStart` / `MessageDelta` | 响应开始 / 收到它的一个 litellm 流事件 |
| `MessageEnd` | 一条消息进入历史：提示、响应、工具结果 |
| `ToolStart` / `ToolUpdate` / `ToolEnd` | 工具调用开始（在中间件即审批之前）/ 报告进度 / 带结果结束 |
| `TurnEnd` | 一个响应及其工具调用的结果都已记录 |
| `Retry` | 模型调用临时失败，将重试 |
| `CompactionStart` / `CompactionEnd` | 历史被压缩 |
| `RunEnd` | 运行结束，带原因、错误和计数；总是最后一个事件 |

事件逐个投递，`Emit` 不应长时间阻塞。它返回错误会停止运行：进行中的工具调用被取消，此后只再投递 `RunEnd`，它总是最后一个事件。被拒的事件不生效：被拒的 `MessageEnd` 或 `CompactionEnd` 不让这条消息或这次压缩进入历史。

消息按取出的顺序进入历史，并在进入时打上时间；从 `Steering`、`FollowUp` 或 `OnStop` 取出的消息，即使运行随后在模型作答前结束，也会记录。失败的响应——厂商以错误结束、流中断或运行被取消——连同已流出的内容一起记录，`Stop` 为 `StopError` 或 `StopAborted`，其中的工具调用被丢弃，便于在记录中呈现；它不会再发给模型。

## 工具

工具是一个结构体：

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

- 调用前按 `Schema` 校验参数；不符合的地方会告诉模型。
- `Check` 在审批和执行前检查调用，并可返回给人看的预览，如 `edit` 和 `write` 返回的 diff，见 `ToolCall.Preview`。
- `Parallel` 允许调用与同一轮的其他并行调用一起运行，上限为 `MaxToolConcurrency`。
- `Deferred` 工具只在历史中有工具引用点名之后才提供给模型，`tool_search` 返回的就是这种引用（见 `tools.Defer`）。
- `Result` 装的是 litellm 块（文本、图片、工具引用），`Result.Text` 取其文本。`Terminate` 在本轮记录完成后结束运行。
- 运行中的工具用 `agentcore.ReportProgress(ctx, v)` 报告进度，以 `ToolUpdate` 送达。`bash` 以字符串报告每行输出，`subagent` 报告 `subagent.Progress`。

`Config.Middleware` 包裹每个通过检查的调用，用于审批、审计或改写参数：

```go
approve := func(ctx context.Context, call agentcore.ToolCall, next agentcore.ToolFunc) (agentcore.Result, error) {
	if call.Name == "bash" && !askUser(call) {
		return agentcore.ErrorResult("The user declined this command."), nil
	}
	return next(ctx, call)
}
```

## 内置工具

`tools.Workspace` 创建编码工具，并保管它们共享的东西：

```go
workspace := tools.Workspace{
	Dir:   ".",                       // 相对路径按它解析；不是沙箱
	FS:    nil,                       // read/write/edit 的文件后端；nil 为本地文件系统
	Files: tools.NewFileReadState(), // 先读后写检查；nil 不检查
	Tasks: tasks,                     // bash 的后台命令；nil 不提供后台模式
}
cfg.Tools = workspace.Tools() // 或 workspace.Read()、workspace.Bash() ……
```

| 工具 | 参数 | 结果 |
|------|------|------|
| `read` | `file_path`、`offset`、`limit` | 带行号的内容（最多 2000 行 / 50KB）、目录列表或图片 |
| `write` | `file_path`、`content` | 一行说明写了什么；预览是 diff |
| `edit` | `file_path`、`old_string`、`new_string`、`replace_all` | 文件名和 diff；先精确匹配，再容忍空白和缩进差异 |
| `bash` | `command`、`timeout`、`workdir`、`description`、`run_in_background`（有 `Tasks` 时） | 输出尾部（2000 行 / 50KB），之后是 `[exit code N]`、`[timed out after …]` 或完整输出的去处 |
| `glob` | `pattern`、`path` | 匹配的路径，最新的在前 |
| `grep` | `pattern`、`path`、`glob`、`ignore_case`、`literal`、`context_lines`、`limit` | `路径:行号:内容` 形式的匹配 |
| `ls` | `path`、`depth`、`ignore` | 目录树 |

结果都是纯文本。命令失败不算 `bash` 调用失败：模型需要的正是它的输出和退出码。有 `Files` 时，`write` 拒绝覆盖模型没有完整读过的已有文件，`write` 和 `edit` 拒绝读过之后又被改动的文件。调用 ctx 携带的工作目录（`tools.WithCwd`）优先于 `Dir`，比如运行中途进入的 git worktree。`bash` 需要 PATH 上有 POSIX shell（`bash` 或 `sh`），Windows 上即 Git Bash。

`tools.Defer(tools)` 把工具放到 `tool_search`（`query`、`max_results`）后面，它的描述列出这些工具的名字：模型搜索过之后才看到它们的 schema。

## 后台任务

```go
tasks := task.NewRuntime(dir, func(m agentcore.Message) { agent.FollowUp(m) })
cfg.Tools = append(cfg.Tools, tasks.Tools()...) // task_output（task_id、wait、timeout）、task_stop（task_id）
```

带 `run_in_background` 的 `bash` 和 `subagent` 工具的后台模式，都把工作作为 `task.Runtime` 的任务运行：每个任务有 ID、状态和 `dir` 下的输出文件，一直运行到结束或被停止，除非要求否则没有超时。任务结束时，Runtime 把宣告它的消息（`task.KindNotification`）交给通知函数，作为后续消息送达。`Runtime.Start` 可以把你自己的工作作为任务运行。

## 压缩

```go
cfg.Compactor = compact.Summarizer{}
cfg.CompactAt = 100_000
```

循环在估算历史超过 `CompactAt` 的调用之前压缩；provider 报告上下文溢出时压缩一次再重试该调用。前者失败不会结束运行，后者必须成功。估算以上一个响应报告的输入 token 为基准。`CompactAt` 应明显高于压缩后保留的量，否则每次调用都会再压缩。

`compact.Summarizer` 原样保留最近的消息（历史的四分之一，在 2k 到 20k token 之间），其余换成对话自己的模型写的检查点：它延伸对话中被替换那部分的调用，因此命中提示缓存，并要求把检查点写在 `<summary>` 标签里；请求放不下或回答没有标签时，再用纯文本记录请求一次。检查点列出被替换部分读过和改过的文件，它加载过的工具仍然保持加载。需要别的策略就实现 `agentcore.Compactor`。

## 子 agent

```go
delegate := subagent.New(tasks, // nil 不提供后台模式
	subagent.Agent{
		Name:        "scout",
		Description: "Fast codebase reconnaissance",
		Config: func(s subagent.Spawn) (agentcore.Config, error) {
			return agentcore.Config{Model: model, System: scoutPrompt, Tools: readOnlyTools()}, nil
		},
	},
)
```

模型调用 `subagent` 时，给 `agent` 和 `task` 运行一个 agent；给 `tasks`（`{agent, task}` 数组）并行运行多个，同时最多 8 个；给 `chain` 按顺序运行，任务里的 `{previous}` 代表上一步的输出；给 `background`（需要 `task.Runtime`）把一个 agent 作为任务在后台运行；`model` 换用别的模型。每次运行拿到自己的 `Config`，工具状态互不相干；它的事件发给该 `Config.Emit`，并作为 `subagent.Progress` 发给等待中的调用。失败的运行会报告错误和它最后说的话。嵌套深度最多为 `subagent.MaxDepth`。

## Config

| 字段 | 说明 |
|------|------|
| `Model` | litellm 客户端、请求模板和定价 |
| `System` / `Tools` | 系统提示块和提供的工具 |
| `Emit` | 接收事件；返回错误会停止运行 |
| `Steering` / `FollowUp` | 下次调用前送达的消息 / 本该停止时继续的消息 |
| `OnStop` | 本该停止时咨询：继续、停止或失败 |
| `Middleware` | 包裹每个工具调用，第一个在最外层 |
| `MaxToolConcurrency` | 同时运行的并行调用数（小于 2 时逐个运行） |
| `MaxToolErrors` | 工具连续失败这么多轮后禁用（0 表示不禁用） |
| `Compactor` / `CompactAt` | 压缩，见上文 |
| `Cache` | 在每次调用的最后一条消息后放置缓存断点 |
| `MaxTurns` | 每次运行的响应数上限（0 表示 100） |
| `MaxRetries` | 模型临时失败的重试次数，带退避 |

## 许可证

Apache License 2.0

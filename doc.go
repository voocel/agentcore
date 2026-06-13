// Package agentcore is a minimal, composable toolkit for building AI agent
// applications in Go. It provides the execution kernel — the agent loop, tool
// dispatch, context management, and a single event stream — and leaves policy
// (which model, which tools, when to stop, how to render) to the caller.
//
// Two entry points, one for each level of control:
//
//   - [AgentLoop] is a pure function: given prompts, context, and a
//     [LoopConfig], it runs the loop and returns a <-chan [Event]. It holds no
//     state of its own — every dependency is injected and every result is an
//     event. Use it when you want to own the state and drive the loop directly.
//
//   - [Agent] wraps the loop with conversation state, message queues, and
//     listener dispatch. Construct it with [NewAgent] and the With* options,
//     register a callback with [Agent.Subscribe], then call [Agent.Prompt].
//     This is the common case.
//
// Every lifecycle signal — streamed text, tool execution, retries, the final
// summary — flows through the one [Event] channel. A single consumer can drive
// any front end (TUI, web, Slack, logs) without the kernel knowing which.
//
// A small, stable surface carries most uses: [Agent], [AgentLoop], [Event],
// [Tool], and [Message]. The rest is opt-in — context strategies, stop guards,
// sub-agents, middleware — reached only when a use case needs it.
//
// Models are adapters behind the [ChatModel] interface; the kernel imports no
// LLM SDK. Provider errors are classified through the [RetryableError] and
// [RetryHinter] interfaces plus the ErrProvider* sentinels, so any backend can
// take part in retries and failover. The bundled llm adapter (agentcore/llm)
// covers OpenAI, Anthropic, and Gemini via litellm.
//
// See the examples directory for runnable single- and multi-agent programs.
package agentcore

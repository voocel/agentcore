// Package agentcore runs AI agents: a model that calls tools, turn by turn,
// until its task is done. It is built on litellm, whose messages, blocks and
// events it uses as they are, and leaves policy — which model, which tools,
// when to stop, how to render — to the application.
//
// [Run] is the loop: given a [Config] and a history, it calls the model and
// the tools it asks for, and returns the history it ended with. It holds no
// state; the caller owns the history. Every lifecycle signal — streamed
// content, tool calls, retries, compactions — reaches Config.Emit as an
// [Event], in order, and every message entering the history passes it as a
// [MessageEnd] first, so an application stores messages there durably.
//
// [Agent] keeps a history for an application and runs it: prompting,
// steering a run under way, queuing follow-ups, and subscribing to the
// events. A run, of either, ends when its context is cancelled.
//
// The model is a litellm Client ([Model]); a tool is a [Tool]. A long
// history is compacted by a [Compactor], such as the summarizing one in
// agentcore/compact. Packages tools, subagent and task hold coding tools,
// sub-agents and background tasks, built on the same few types.
package agentcore

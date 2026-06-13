// Package tools provides the built-in agent tools — read, write, edit, and
// bash. The file tools share a [FileReadState] that enforces read-before-write
// and detects stale edits. Each tool implements [agentcore.Tool].
package tools

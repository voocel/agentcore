// Package tools holds the coding tools of an agent. A [Workspace] makes them
// and holds what they share: read, write and edit work on files through its
// [FS] and check writes against what the model read; bash runs commands, in
// the background as tasks of its task.Runtime; glob, grep and ls find and
// list files. [Defer] puts tools behind tool_search, so that the model sees
// only their names until it needs them.
//
// Relative paths resolve against the Workspace's Dir, or the working
// directory a call's context carries ([WithCwd]); it is not a sandbox.
// Results are plain text.
package tools

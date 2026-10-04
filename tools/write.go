package tools

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"strings"

	"github.com/voocel/agentcore"
	"github.com/voocel/agentcore/schema"
)

// Write returns the write tool: it writes content to a file, creating
// directories as needed, and reports what it wrote. Its Check returns the
// diff of the change, cut to a few lines, as the call's preview. With Files,
// its Check and Run refuse an existing file the model has not read whole, or
// that changed since.
func (w Workspace) Write() agentcore.Tool {
	t := &writeTool{w: w, fs: w.fs()}
	return agentcore.Tool{
		Name:        "write",
		Label:       "Write File",
		Description: writeDescription,
		Schema: schema.Object(
			schema.Property("file_path", schema.String("The path to the file to write or overwrite (relative or absolute)")).Required(),
			schema.Property("content", schema.String("The content to write to the file")).Required(),
		),
		Check: func(ctx context.Context, args json.RawMessage) (string, error) {
			if err := t.validate(ctx, args); err != nil {
				return "", err
			}
			return t.preview(ctx, args)
		},
		Run: t.execute,
	}
}

type writeTool struct {
	w  Workspace
	fs FS
}

const writeDescription = `Writes a file to the local filesystem.

Usage:
- This tool will overwrite the existing file if there is one at the provided path.
- If this is an existing file, you MUST use the read tool first to read the file's contents. This tool will fail if you did not read the file first.
- Prefer the edit tool for modifying existing files — it only sends the diff. Only use this tool to create new files or for complete rewrites.
- Creates parent directories if needed.
- NEVER create documentation files (*.md) or README files unless explicitly requested by the user.`

type writeArgs struct {
	FilePath string `json:"file_path"`
	Content  string `json:"content"`
}

type writeState struct {
	path       string
	contentOld string
	contentNew string
	exists     bool
}

func (t *writeTool) parseWrite(ctx context.Context, args json.RawMessage) (*writeState, error) {
	var a writeArgs
	if err := json.Unmarshal(args, &a); err != nil {
		return nil, fmt.Errorf("invalid args: %w", err)
	}

	a.FilePath = ResolvePath(t.w.dir(ctx), a.FilePath)

	contentOld := ""
	exists := false
	if data, err := t.fs.ReadFile(ctx, a.FilePath); err == nil {
		contentOld = string(data)
		exists = true
	} else if !os.IsNotExist(err) {
		return nil, fmt.Errorf("read %s: %w", a.FilePath, err)
	}

	return &writeState{
		path:       a.FilePath,
		contentOld: contentOld,
		contentNew: a.Content,
		exists:     exists,
	}, nil
}

const writePreviewMaxLines = 12

func (t *writeTool) preview(ctx context.Context, args json.RawMessage) (string, error) {
	state, err := t.parseWrite(ctx, args)
	if err != nil {
		return "", err
	}
	if !state.exists {
		return writePreview(state.contentNew, writePreviewMaxLines), nil
	}
	diff := generateDiff(state.contentOld, state.contentNew)
	if lines := strings.Count(diff, "\n"); lines > writePreviewMaxLines {
		diff = keepFirstNLines(diff, writePreviewMaxLines) + fmt.Sprintf("... [diff truncated, %d more lines]\n", lines-writePreviewMaxLines)
	}
	return diff, nil
}

// validate enforces read-before-write and detects stale writes: an existing
// file must have been read whole this session, as overwriting it would drop
// what the model has not seen, and not modified since.
func (t *writeTool) validate(ctx context.Context, args json.RawMessage) error {
	if t.w.Files == nil {
		return nil
	}

	var a writeArgs
	if err := json.Unmarshal(args, &a); err != nil {
		return errors.New("invalid args: " + err.Error())
	}
	path := ResolvePath(t.w.dir(ctx), a.FilePath)

	info, err := t.fs.Stat(ctx, path)
	if os.IsNotExist(err) {
		return nil
	}
	if err != nil {
		return errors.New("stat " + path + ": " + err.Error())
	}
	if info.IsDir {
		return errors.New("path is a directory: " + path)
	}

	stamp, ok := t.w.Files.Get(path)
	if !ok {
		return errors.New("File has not been read yet. Read it first before writing to it.")
	}
	if stamp.Partial {
		return errors.New("Only part of the file has been read (offset/limit). Read the whole file before overwriting it, or use edit to change part of it.")
	}
	// Compare against the content token / mtime recorded at read time, not just
	// "after ReadAt". Catches mtime regressions too (e.g. git checkout of an
	// older version), and unsaved-buffer changes when the backend sets Version.
	if !stampMatches(stamp, info) {
		return errors.New("File has been modified since read, either by the user or by a linter. Read it again before attempting to write it.")
	}
	return nil
}

// execute checks the file again: it may have changed while the call
// awaited approval.
func (t *writeTool) execute(ctx context.Context, args json.RawMessage) (agentcore.Result, error) {
	if err := t.validate(ctx, args); err != nil {
		return agentcore.Result{}, err
	}
	state, err := t.parseWrite(ctx, args)
	if err != nil {
		return agentcore.Result{}, err
	}
	if err := t.fs.MkdirAll(ctx, dirOf(state.path), 0o755); err != nil {
		return agentcore.Result{}, fmt.Errorf("mkdir: %w", err)
	}
	if err := t.fs.WriteFile(ctx, state.path, []byte(state.contentNew), 0o644); err != nil {
		return agentcore.Result{}, fmt.Errorf("write %s: %w", state.path, err)
	}
	t.w.Files.recordWrite(ctx, t.fs, state.path, true)

	action := "Overwrote"
	if !state.exists {
		action = "Created"
	}
	return agentcore.TextResult(fmt.Sprintf("%s %s (%d bytes).", action, state.path, len(state.contentNew))), nil
}

// writePreview returns the first maxLines lines of content with line numbers prefixed by "+".
func writePreview(content string, maxLines int) string {
	lines := strings.Split(content, "\n")
	total := len(lines)
	n := min(maxLines, total)

	lineNumWidth := len(fmt.Sprintf("%d", total))
	var sb strings.Builder
	for i := 0; i < n; i++ {
		fmt.Fprintf(&sb, "+%*d %s\n", lineNumWidth, i+1, lines[i])
	}
	if total > n {
		fmt.Fprintf(&sb, " %*s ... +%d more lines\n", lineNumWidth, "", total-n)
	}
	return sb.String()
}

func keepFirstNLines(s string, n int) string {
	idx := 0
	for i := 0; i < n; i++ {
		pos := strings.IndexByte(s[idx:], '\n')
		if pos < 0 {
			return s
		}
		idx += pos + 1
	}
	return s[:idx]
}

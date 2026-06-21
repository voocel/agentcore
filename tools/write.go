package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/voocel/agentcore"
	"github.com/voocel/agentcore/schema"
)

// WriteTool writes content to a file, creating directories as needed.
//
// Validate enforces read-before-write and detects stale writes when state is
// non-nil.
type WriteTool struct {
	WorkDir   string
	readState *FileReadState
	fs        WorkspaceFS
}

// NewWrite creates a write tool rooted at workDir.
//
// Pass the same non-nil FileReadState to NewRead, NewWrite, and NewEdit to
// enable read-before-write/edit validation. Pass nil to disable this tracking.
// By default the tool operates on the local filesystem; pass WithFS to inject
// a different WorkspaceFS backend.
func NewWrite(workDir string, state *FileReadState, opts ...Option) *WriteTool {
	return &WriteTool{WorkDir: workDir, readState: state, fs: resolveFS(opts)}
}

func (t *WriteTool) Name() string                                 { return "write" }
func (t *WriteTool) Label() string                                { return "Write File" }
func (t *WriteTool) ReadOnly(_ json.RawMessage) bool              { return false }
func (t *WriteTool) ConcurrencySafe(_ json.RawMessage) bool       { return false }
func (t *WriteTool) ActivityDescription(_ json.RawMessage) string { return "Writing file" }
func (t *WriteTool) Description() string {
	return `Writes a file to the local filesystem.

Usage:
- This tool will overwrite the existing file if there is one at the provided path.
- If this is an existing file, you MUST use the read tool first to read the file's contents. This tool will fail if you did not read the file first.
- Prefer the edit tool for modifying existing files — it only sends the diff. Only use this tool to create new files or for complete rewrites.
- Creates parent directories if needed.
- NEVER create documentation files (*.md) or README files unless explicitly requested by the user.`
}
func (t *WriteTool) Schema() map[string]any {
	return schema.Object(
		schema.Property("file_path", schema.String("The path to the file to write or overwrite (relative or absolute)")).Required(),
		schema.Property("content", schema.String("The content to write to the file")).Required(),
	)
}

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

func (t *WriteTool) parseWrite(ctx context.Context, args json.RawMessage) (*writeState, error) {
	var a writeArgs
	if err := json.Unmarshal(args, &a); err != nil {
		return nil, fmt.Errorf("invalid args: %w", err)
	}

	a.FilePath = ResolvePath(t.WorkDir, a.FilePath)

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

func (t *WriteTool) Preview(ctx context.Context, args json.RawMessage) (json.RawMessage, error) {
	state, err := t.parseWrite(ctx, args)
	if err != nil {
		return nil, err
	}

	if !state.exists {
		return json.Marshal(map[string]any{
			"message":            fmt.Sprintf("Create %s", state.path),
			"diff":               writePreview(state.contentNew, writePreviewMaxLines),
			"first_changed_line": 1,
		})
	}

	diff, firstLine := generateDiff(state.contentOld, state.contentNew)
	lines := strings.Count(diff, "\n")
	if lines > writePreviewMaxLines {
		kept := keepFirstNLines(diff, writePreviewMaxLines)
		diff = kept + fmt.Sprintf("\n... [diff truncated, %d more lines]", lines-writePreviewMaxLines)
	}
	return json.Marshal(map[string]any{
		"message":            fmt.Sprintf("Overwrite %s", state.path),
		"diff":               diff,
		"first_changed_line": firstLine,
	})
}

// Validate enforces read-before-write and detects stale writes.
//
// Error codes (stable for tests):
//   - 2: existing file has not been read this session, or only a partial
//     slice was read.
//   - 3: file was modified after the last read.
func (t *WriteTool) Validate(ctx context.Context, args json.RawMessage) agentcore.ValidationResult {
	if t.readState == nil {
		return agentcore.ValidationResult{OK: true}
	}

	var a writeArgs
	if err := json.Unmarshal(args, &a); err != nil {
		return agentcore.ValidationResult{OK: false, Message: "invalid args: " + err.Error()}
	}
	path := ResolvePath(t.WorkDir, a.FilePath)

	info, err := t.fs.Stat(ctx, path)
	if os.IsNotExist(err) {
		return agentcore.ValidationResult{OK: true}
	}
	if err != nil {
		return agentcore.ValidationResult{OK: false, Message: "stat " + path + ": " + err.Error()}
	}
	if info.IsDir {
		return agentcore.ValidationResult{OK: false, Message: "path is a directory: " + path}
	}

	stamp, ok := t.readState.Get(path)
	if !ok || stamp.Partial {
		return agentcore.ValidationResult{
			OK:        false,
			ErrorCode: 2,
			Message:   "File has not been read yet. Read it first before writing to it.",
		}
	}
	// Compare against the content token / mtime recorded at read time, not just
	// "after ReadAt". Catches mtime regressions too (e.g. git checkout of an
	// older version), and unsaved-buffer changes when the backend sets Version.
	if !stampMatches(stamp, info) {
		return agentcore.ValidationResult{
			OK:        false,
			ErrorCode: 3,
			Message:   "File has been modified since read, either by the user or by a linter. Read it again before attempting to write it.",
		}
	}
	return agentcore.ValidationResult{OK: true}
}

func (t *WriteTool) Execute(ctx context.Context, args json.RawMessage) (json.RawMessage, error) {
	state, err := t.parseWrite(ctx, args)
	if err != nil {
		return nil, err
	}

	if err := t.fs.MkdirAll(ctx, filepath.Dir(state.path), 0o755); err != nil {
		return nil, fmt.Errorf("mkdir: %w", err)
	}

	if err := t.fs.WriteFile(ctx, state.path, []byte(state.contentNew), 0o644); err != nil {
		return nil, fmt.Errorf("write %s: %w", state.path, err)
	}

	action := "overwrote"
	if !state.exists {
		action = "created"
	}
	return json.Marshal(map[string]any{
		"message":     fmt.Sprintf("%s %d bytes to %s", action, len(state.contentNew), state.path),
		"preview":     writePreview(state.contentNew, writePreviewMaxLines),
		"created":     !state.exists,
		"overwritten": state.exists,
	})
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

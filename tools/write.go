package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/voocel/agentcore/schema"
)

// WriteTool writes content to a file, creating directories as needed.
type WriteTool struct {
	WorkDir string
}

func NewWrite(workDir string) *WriteTool { return &WriteTool{WorkDir: workDir} }

func (t *WriteTool) Name() string  { return "write" }
func (t *WriteTool) Label() string { return "Write File" }
func (t *WriteTool) Description() string {
	return "Write or overwrite a file on disk. Prefer edit for partial changes to existing files. If you are replacing an existing file, read it first so you understand the current content before overwriting it. Creates parent directories if needed. Avoid creating docs or README files unless the user explicitly asks for them."
}
func (t *WriteTool) Schema() map[string]any {
	return schema.Object(
		schema.Property("path", schema.String("Path to the file to write or overwrite (relative or absolute)")).Required(),
		schema.Property("content", schema.String("Full file content to write")).Required(),
	)
}

type writeArgs struct {
	Path    string `json:"path"`
	Content string `json:"content"`
}

type writeState struct {
	path       string
	contentOld string
	contentNew string
	exists     bool
}

func (t *WriteTool) parseWrite(args json.RawMessage) (*writeState, error) {
	var a writeArgs
	if err := json.Unmarshal(args, &a); err != nil {
		return nil, fmt.Errorf("invalid args: %w", err)
	}

	a.Path = ResolvePath(t.WorkDir, a.Path)

	contentOld := ""
	exists := false
	if data, err := os.ReadFile(a.Path); err == nil {
		contentOld = string(data)
		exists = true
	} else if !os.IsNotExist(err) {
		return nil, fmt.Errorf("read %s: %w", a.Path, err)
	}

	return &writeState{
		path:       a.Path,
		contentOld: contentOld,
		contentNew: a.Content,
		exists:     exists,
	}, nil
}

const writePreviewMaxLines = 12

func (t *WriteTool) Preview(_ context.Context, args json.RawMessage) (json.RawMessage, error) {
	state, err := t.parseWrite(args)
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

func (t *WriteTool) Execute(_ context.Context, args json.RawMessage) (json.RawMessage, error) {
	state, err := t.parseWrite(args)
	if err != nil {
		return nil, err
	}

	if err := os.MkdirAll(filepath.Dir(state.path), 0o755); err != nil {
		return nil, fmt.Errorf("mkdir: %w", err)
	}

	if err := os.WriteFile(state.path, []byte(state.contentNew), 0o644); err != nil {
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

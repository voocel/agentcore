package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path"
	"path/filepath"
	"sort"
	"strings"

	"github.com/voocel/agentcore/schema"
)

// LsTool lists directory contents with optional depth control.
type LsTool struct {
	WorkDir string
}

func NewLs(workDir string) *LsTool { return &LsTool{WorkDir: workDir} }

func (t *LsTool) Name() string  { return "ls" }
func (t *LsTool) Label() string { return "List Directory" }
func (t *LsTool) Description() string {
	return "List directory contents as a tree. Use this for quick directory structure checks before reading files. Depth controls recursive listing (default 1, max 5). Use ignore to hide generated or irrelevant paths. Common generated directories (node_modules, .git, dist, build, etc.) are hidden by default."
}
func (t *LsTool) Schema() map[string]any {
	return schema.Object(
		schema.Property("path", schema.String("Directory path, relative or absolute (default: working directory)")),
		schema.Property("depth", schema.Int("Recursion depth (default: 1, max: 5)")),
		schema.Property("ignore", schema.Array("Optional file or directory patterns to ignore (for example: tmp/, *.log, dist)", schema.String("Ignore pattern"))),
	)
}

type lsArgs struct {
	Path   string   `json:"path"`
	Depth  int      `json:"depth"`
	Ignore []string `json:"ignore"`
}

type ignoreMatcher struct {
	patterns []string
}

const lsDefaultLimit = 500

var lsDefaultIgnorePatterns = []string{
	".git/",
	"node_modules/",
	"__pycache__/",
	".venv/",
	"dist/",
	"build/",
	"target/",
	"vendor/",
	"bin/",
	"obj/",
	".idea/",
	".vscode/",
	".cache/",
	"cache/",
	"tmp/",
	"temp/",
	".coverage",
	"coverage/",
	"logs/",
	"env/",
	"venv/",
	".zig-cache/",
	"zig-out/",
}

func (t *LsTool) Execute(ctx context.Context, args json.RawMessage) (json.RawMessage, error) {
	var a lsArgs
	if err := json.Unmarshal(args, &a); err != nil {
		return nil, fmt.Errorf("invalid args: %w", err)
	}

	dir := ResolvePath(t.WorkDir, a.Path)

	depth := a.Depth
	if depth <= 0 {
		depth = 1
	}
	if depth > 5 {
		depth = 5
	}

	maxEntries := lsDefaultLimit

	matcher := newIgnoreMatcher(append(append([]string{}, lsDefaultIgnorePatterns...), a.Ignore...))
	var sb strings.Builder
	count := 0
	truncated := false

	if err := renderTree(ctx, dir, dir, 0, depth, maxEntries, matcher, &count, &truncated, &sb); err != nil {
		return nil, fmt.Errorf("ls %s: %w", dir, err)
	}

	if count == 0 {
		return json.Marshal("(empty directory)")
	}

	result := strings.TrimRight(dir, string(filepath.Separator)) + "/\n" + strings.TrimRight(sb.String(), "\n")
	if truncated {
		result += fmt.Sprintf("\n\n[Listing truncated at %d entries. Use limit=%d for more, or use a specific subdirectory.]", maxEntries, maxEntries*2)
	}

	tr := truncateHead(result, 0, defaultMaxBytes)
	if tr.Truncated {
		return json.Marshal(tr.Content + "\n\n[Output truncated at " + formatSize(defaultMaxBytes) + ".]")
	}
	return json.Marshal(result)
}

func newIgnoreMatcher(patterns []string) ignoreMatcher {
	out := make([]string, 0, len(patterns))
	for _, p := range patterns {
		p = strings.TrimSpace(filepath.ToSlash(p))
		if p == "" {
			continue
		}
		out = append(out, p)
	}
	return ignoreMatcher{patterns: out}
}

func (m ignoreMatcher) Match(rel string, isDir bool) bool {
	if len(m.patterns) == 0 {
		return false
	}

	rel = filepath.ToSlash(rel)
	base := path.Base(rel)
	for _, pattern := range m.patterns {
		trimmed := strings.TrimSuffix(pattern, "/")
		if trimmed == "" {
			continue
		}

		if rel == trimmed || strings.HasPrefix(rel, trimmed+"/") {
			return true
		}
		if ok, _ := path.Match(pattern, rel); ok {
			return true
		}
		if ok, _ := path.Match(trimmed, rel); ok {
			return true
		}
		if ok, _ := path.Match(pattern, base); ok {
			return true
		}
		if ok, _ := path.Match(trimmed, base); ok {
			return true
		}
		if isDir && strings.HasSuffix(pattern, "/") && (rel == trimmed || strings.HasPrefix(rel, trimmed+"/")) {
			return true
		}
	}
	return false
}

func renderTree(ctx context.Context, root, dir string, current, maxDepth, maxEntries int, matcher ignoreMatcher, count *int, truncated *bool, sb *strings.Builder) error {
	if current >= maxDepth || *truncated {
		return nil
	}
	if ctx.Err() != nil {
		return ctx.Err()
	}

	dirEntries, err := os.ReadDir(dir)
	if err != nil {
		return err
	}

	sort.Slice(dirEntries, func(i, j int) bool {
		return strings.ToLower(dirEntries[i].Name()) < strings.ToLower(dirEntries[j].Name())
	})

	for _, e := range dirEntries {
		name := e.Name()
		isDir := e.IsDir()
		if isDir && IsSkipDir(name) {
			continue
		}

		childPath := filepath.Join(dir, name)
		rel, _ := filepath.Rel(root, childPath)
		rel = filepath.ToSlash(rel)
		if matcher.Match(rel, isDir) {
			continue
		}

		if *count >= maxEntries {
			*truncated = true
			return nil
		}

		info, err := e.Info()
		if err != nil {
			continue
		}

		*count++
		indent := strings.Repeat("  ", current+1)
		if isDir {
			sb.WriteString(indent)
			sb.WriteString(name)
			sb.WriteString("/\n")
		} else {
			sb.WriteString(indent)
			sb.WriteString(name)
			sb.WriteString("  ")
			sb.WriteString(formatSize(int(info.Size())))
			sb.WriteByte('\n')
		}

		if isDir {
			if err := renderTree(ctx, root, childPath, current+1, maxDepth, maxEntries, matcher, count, truncated, sb); err != nil {
				return err
			}
		}
	}
	return nil
}

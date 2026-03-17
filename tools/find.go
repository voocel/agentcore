package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"strings"

	"github.com/voocel/agentcore/schema"
)

// FindTool searches for files matching a glob pattern.
// Uses fd if available, falls back to filepath.WalkDir + filepath.Match.
type FindTool struct {
	WorkDir string
}

func NewFind(workDir string) *FindTool { return &FindTool{WorkDir: workDir} }

func (t *FindTool) Name() string  { return "find" }
func (t *FindTool) Label() string { return "Find Files" }
func (t *FindTool) Description() string {
	return "Search for files by glob pattern, including path-aware patterns like 'cmd/*.go' and 'src/**/*.ts'. Returns paths relative to the search directory (default limit: 1000)."
}
func (t *FindTool) Schema() map[string]any {
	return schema.Object(
		schema.Property("pattern", schema.String("Glob pattern to match files (for example: '*.go', 'cmd/*.go', 'src/**/*.ts')")).Required(),
		schema.Property("path", schema.String("Directory to search in (default: working directory)")),
		schema.Property("limit", schema.Int("Maximum number of results (default: 1000)")),
	)
}

type findArgs struct {
	Pattern string `json:"pattern"`
	Path    string `json:"path"`
	Limit   int    `json:"limit"`
}

const findDefaultLimit = 1000

func (t *FindTool) Execute(ctx context.Context, args json.RawMessage) (json.RawMessage, error) {
	var a findArgs
	if err := json.Unmarshal(args, &a); err != nil {
		return nil, fmt.Errorf("invalid args: %w", err)
	}

	limit := a.Limit
	if limit <= 0 {
		limit = findDefaultLimit
	}

	searchDir := ResolvePath(t.WorkDir, a.Path)

	// Try fd first
	if result, err := t.findWithFd(ctx, a.Pattern, searchDir, limit); err == nil {
		return result, nil
	}

	// Fallback to filepath.WalkDir
	return t.findWithWalk(ctx, a.Pattern, searchDir, limit)
}

func (t *FindTool) findWithFd(ctx context.Context, pattern, dir string, limit int) (json.RawMessage, error) {
	fdPath, err := exec.LookPath("fd")
	if err != nil {
		return nil, err
	}

	cmdArgs := []string{
		"--glob", "--color=never", "--hidden",
		"--no-require-git",
		"--max-results", fmt.Sprintf("%d", limit),
		pattern, dir,
	}

	cmd := exec.CommandContext(ctx, fdPath, cmdArgs...)
	out, err := cmd.Output()
	if err != nil && len(out) == 0 {
		return nil, err
	}

	return t.formatResults(string(out), dir, limit)
}

func (t *FindTool) findWithWalk(ctx context.Context, pattern, dir string, limit int) (json.RawMessage, error) {
	var matches []string

	hitLimit := false
	err := filepath.WalkDir(dir, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return filepath.SkipDir
		}
		if ctx.Err() != nil {
			return ctx.Err()
		}
		if d.IsDir() {
			if IsSkipDir(d.Name()) {
				return filepath.SkipDir
			}
			return nil
		}
		rel, _ := filepath.Rel(dir, path)
		if findPatternMatches(pattern, rel) {
			matches = append(matches, rel)
		}
		if len(matches) >= limit {
			hitLimit = true
			return filepath.SkipAll
		}
		return nil
	})

	if err != nil && err != filepath.SkipAll {
		return nil, fmt.Errorf("walk: %w", err)
	}

	if len(matches) == 0 {
		return json.Marshal("No files found matching pattern.")
	}

	output := strings.Join(matches, "\n")
	if hitLimit {
		output += fmt.Sprintf("\n\n[%d results limit reached. Use limit=%d for more, or refine pattern.]", limit, limit*2)
	}

	// Apply byte truncation
	tr := truncateHead(output, 0, defaultMaxBytes)
	if tr.Truncated {
		return json.Marshal(tr.Content + "\n\n[Output truncated at " + formatSize(defaultMaxBytes) + ".]")
	}
	return json.Marshal(output)
}

func findPatternMatches(pattern, rel string) bool {
	pattern = filepath.ToSlash(strings.TrimSpace(pattern))
	rel = filepath.ToSlash(rel)
	pattern = strings.TrimPrefix(pattern, "./")
	rel = strings.TrimPrefix(rel, "./")
	if pattern == "" || rel == "" {
		return false
	}

	if !strings.Contains(pattern, "/") {
		matched, _ := path.Match(pattern, path.Base(rel))
		return matched
	}

	return matchGlobSegments(splitGlobSegments(pattern), splitGlobSegments(rel))
}

func splitGlobSegments(value string) []string {
	parts := strings.Split(value, "/")
	out := make([]string, 0, len(parts))
	for _, part := range parts {
		if part == "" || part == "." {
			continue
		}
		out = append(out, part)
	}
	return out
}

func matchGlobSegments(patterns, segments []string) bool {
	if len(patterns) == 0 {
		return len(segments) == 0
	}

	if patterns[0] == "**" {
		for len(patterns) > 1 && patterns[1] == "**" {
			patterns = patterns[1:]
		}
		if matchGlobSegments(patterns[1:], segments) {
			return true
		}
		if len(segments) == 0 {
			return false
		}
		return matchGlobSegments(patterns, segments[1:])
	}

	if len(segments) == 0 {
		return false
	}

	matched, err := path.Match(patterns[0], segments[0])
	if err != nil || !matched {
		return false
	}
	return matchGlobSegments(patterns[1:], segments[1:])
}

func (t *FindTool) formatResults(raw, dir string, limit int) (json.RawMessage, error) {
	lines := strings.Split(strings.TrimSpace(raw), "\n")
	var results []string
	for _, line := range lines {
		line = strings.TrimRight(line, "\r")
		if line == "" {
			continue
		}
		// Preserve trailing slash for directories
		suffix := ""
		if strings.HasSuffix(line, "/") || strings.HasSuffix(line, string(filepath.Separator)) {
			suffix = "/"
		}
		if rel, err := filepath.Rel(dir, strings.TrimRight(line, "/\\")); err == nil {
			results = append(results, rel+suffix)
		} else {
			results = append(results, line)
		}
		if len(results) >= limit {
			break
		}
	}

	if len(results) == 0 {
		return json.Marshal("No files found matching pattern.")
	}

	output := strings.Join(results, "\n")
	hitLimit := len(results) >= limit
	if hitLimit {
		output += fmt.Sprintf("\n\n[%d results limit reached. Use limit=%d for more, or refine pattern.]", limit, limit*2)
	}

	// Apply byte truncation
	tr := truncateHead(output, 0, defaultMaxBytes)
	if tr.Truncated {
		return json.Marshal(tr.Content + "\n\n[Output truncated at " + formatSize(defaultMaxBytes) + ".]")
	}
	return json.Marshal(output)
}

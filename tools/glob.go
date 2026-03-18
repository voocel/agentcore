package tools

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"

	"github.com/voocel/agentcore/schema"
)

// GlobTool matches files by glob pattern and returns relative paths
// sorted by modification time (newest first).
// Uses rg --files if available, falls back to filepath.WalkDir.
type GlobTool struct {
	WorkDir string
}

func NewGlob(workDir string) *GlobTool { return &GlobTool{WorkDir: workDir} }

func (t *GlobTool) Name() string  { return "glob" }
func (t *GlobTool) Label() string { return "Match Files" }
func (t *GlobTool) Description() string {
	return "Fast file pattern matching for any codebase size. Supports path-aware glob patterns like '**/*.js' and 'src/**/*.ts'. Returns matching relative file paths sorted by modification time (newest first). Use this when you need to find files by name pattern before reading or grepping them."
}
func (t *GlobTool) Schema() map[string]any {
	return schema.Object(
		schema.Property("pattern", schema.String("Glob pattern to match files (for example: '*.go', '**/*.js', 'src/**/*.ts')")).Required(),
		schema.Property("path", schema.String("Directory to search in, relative or absolute (default: working directory)")),
	)
}

type globArgs struct {
	Pattern string `json:"pattern"`
	Path    string `json:"path"`
}

type globMatch struct {
	rel   string
	mtime int64
}

const globMaxResults = 200

func (t *GlobTool) Execute(ctx context.Context, args json.RawMessage) (json.RawMessage, error) {
	var a globArgs
	if err := json.Unmarshal(args, &a); err != nil {
		return nil, fmt.Errorf("invalid args: %w", err)
	}
	if strings.TrimSpace(a.Pattern) == "" {
		return nil, fmt.Errorf("pattern is required")
	}

	searchDir := ResolvePath(t.WorkDir, a.Path)
	info, err := os.Stat(searchDir)
	if err != nil {
		return nil, fmt.Errorf("glob %s: %w", searchDir, err)
	}
	if !info.IsDir() {
		return nil, fmt.Errorf("glob %s: not a directory", searchDir)
	}

	if result, ok, err := t.globWithRg(ctx, a.Pattern, searchDir); err == nil && ok {
		return result, nil
	}
	return t.globWithWalk(ctx, a.Pattern, searchDir)
}

func (t *GlobTool) globWithRg(ctx context.Context, pattern, dir string) (json.RawMessage, bool, error) {
	rgPath, err := exec.LookPath("rg")
	if err != nil {
		return nil, false, err
	}

	cmd := exec.CommandContext(ctx, rgPath,
		"--files",
		"--glob", pattern,
		"--color=never",
		"--hidden",
		"--no-require-git",
		dir,
	)

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, false, err
	}
	if err := cmd.Start(); err != nil {
		return nil, false, err
	}

	matches := make([]globMatch, 0, 64)
	truncated := false
	scanner := bufio.NewScanner(stdout)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		if len(matches) >= globMaxResults {
			truncated = true
			break
		}

		fullPath := line
		if !filepath.IsAbs(fullPath) {
			fullPath = filepath.Join(dir, line)
		}
		info, err := os.Stat(fullPath)
		if err != nil || info.IsDir() {
			continue
		}
		rel, _ := filepath.Rel(dir, fullPath)
		matches = append(matches, globMatch{
			rel:   rel,
			mtime: info.ModTime().UnixNano(),
		})
	}

	if truncated && cmd.Process != nil {
		cmd.Process.Kill()
	}
	cmd.Wait()

	result, err := formatGlobMatches(matches, truncated)
	return result, true, err
}

func (t *GlobTool) globWithWalk(ctx context.Context, pattern, dir string) (json.RawMessage, error) {
	matches := make([]globMatch, 0, 64)
	truncated := false

	err := filepath.WalkDir(dir, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			if path == dir {
				return err
			}
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

		rel, err := filepath.Rel(dir, path)
		if err != nil || !globPatternMatches(pattern, rel) {
			return nil
		}

		info, err := d.Info()
		if err != nil {
			return nil
		}

		matches = append(matches, globMatch{
			rel:   rel,
			mtime: info.ModTime().UnixNano(),
		})
		if len(matches) >= globMaxResults {
			truncated = true
			return filepath.SkipAll
		}
		return nil
	})
	if err != nil && err != filepath.SkipAll {
		return nil, fmt.Errorf("glob %s: %w", dir, err)
	}

	return formatGlobMatches(matches, truncated)
}

func formatGlobMatches(matches []globMatch, truncated bool) (json.RawMessage, error) {
	if len(matches) == 0 {
		return json.Marshal("No files found.")
	}

	sort.SliceStable(matches, func(i, j int) bool {
		if matches[i].mtime == matches[j].mtime {
			return matches[i].rel < matches[j].rel
		}
		return matches[i].mtime > matches[j].mtime
	})

	lines := make([]string, 0, len(matches)+2)
	for _, m := range matches {
		lines = append(lines, m.rel)
	}
	if truncated {
		lines = append(lines, "", fmt.Sprintf("[Results truncated at %d files. Use a more specific pattern or path.]", globMaxResults))
	}

	result := strings.Join(lines, "\n")
	tr := truncateHead(result, 0, defaultMaxBytes)
	if tr.Truncated {
		return json.Marshal(tr.Content + "\n\n[Output truncated at " + formatSize(defaultMaxBytes) + ".]")
	}
	return json.Marshal(result)
}

package tools

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"

	"github.com/voocel/agentcore"
	"github.com/voocel/agentcore/schema"
)

// Glob returns the glob tool: it matches files by glob pattern and returns
// their relative paths, newest first. It uses rg --files if available, and
// filepath.WalkDir otherwise; either way, .git is left out.
func (w Workspace) Glob() agentcore.Tool {
	t := &globTool{w: w}
	return agentcore.Tool{
		Name:        "glob",
		Label:       "Match Files",
		Description: "Fast file pattern matching for any codebase size. Supports path-aware glob patterns like '**/*.js' and 'src/**/*.ts'. Returns matching relative file paths sorted by modification time (newest first). Use this when you need to find files by name pattern before reading or grepping them.",
		Schema: schema.Object(
			schema.Property("pattern", schema.String("Glob pattern to match files (for example: '*.go', '**/*.js', 'src/**/*.ts')")).Required(),
			schema.Property("path", schema.String("Directory to search in, relative or absolute (default: working directory)")),
		),
		Parallel: true,
		Run:      textRun(t.execute),
	}
}

type globTool struct {
	w Workspace
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

func (t *globTool) execute(ctx context.Context, args json.RawMessage) (string, error) {
	var a globArgs
	if err := json.Unmarshal(args, &a); err != nil {
		return "", fmt.Errorf("invalid args: %w", err)
	}
	if strings.TrimSpace(a.Pattern) == "" {
		return "", fmt.Errorf("pattern is required")
	}

	searchDir := ResolvePath(t.w.dir(ctx), a.Path)
	info, err := os.Stat(searchDir)
	if err != nil {
		return "", fmt.Errorf("glob %s: %w", searchDir, err)
	}
	if !info.IsDir() {
		return "", fmt.Errorf("glob %s: not a directory", searchDir)
	}

	// The walk stands in only for an rg not installed; rg's own errors,
	// such as on an invalid pattern, are the model's to read.
	rg, err := exec.LookPath("rg")
	if err != nil {
		return t.globWithWalk(ctx, a.Pattern, searchDir)
	}
	return t.globWithRg(ctx, rg, a.Pattern, searchDir)
}

func (t *globTool) globWithRg(ctx context.Context, rgPath, pattern, dir string) (string, error) {
	// rg matches globs against paths relative to its working directory.
	cmd := exec.CommandContext(ctx, rgPath,
		"--files",
		"--glob="+pattern,
		"--color=never",
		"--hidden",
		"--glob=!.git",
		"--no-require-git",
		"--", ".",
	)
	cmd.Dir = dir
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return "", fmt.Errorf("pipe: %w", err)
	}
	if err := cmd.Start(); err != nil {
		return "", fmt.Errorf("start rg: %w", err)
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

		rel := filepath.Clean(line)
		info, err := os.Stat(filepath.Join(dir, rel))
		if err != nil || info.IsDir() {
			continue
		}
		matches = append(matches, globMatch{
			rel:   rel,
			mtime: info.ModTime().UnixNano(),
		})
	}

	// rg is stopped once its output is no longer read.
	scanErr := scanner.Err()
	if truncated || scanErr != nil {
		cmd.Process.Kill()
	}
	waitErr := cmd.Wait()
	if scanErr != nil {
		return "", fmt.Errorf("scan rg output: %w", scanErr)
	}
	// The files listed stand, though rg failed to read some directories.
	// Listing none, rg exits with 1; with another code, it failed.
	if waitErr != nil && !truncated && len(matches) == 0 {
		if exit, ok := errors.AsType[*exec.ExitError](waitErr); !ok || exit.ExitCode() != 1 {
			if msg := strings.TrimSpace(stderr.String()); msg != "" {
				return "", fmt.Errorf("glob: %s", msg)
			}
			return "", fmt.Errorf("glob: rg: %w", waitErr)
		}
	}
	return formatGlobMatches(matches, truncated)
}

func (t *globTool) globWithWalk(ctx context.Context, pattern, dir string) (string, error) {
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
		return "", fmt.Errorf("glob %s: %w", dir, err)
	}

	return formatGlobMatches(matches, truncated)
}

func formatGlobMatches(matches []globMatch, truncated bool) (string, error) {
	if len(matches) == 0 {
		return "No files found.", nil
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
	tr := truncateHead(result, defaultMaxLines, defaultMaxBytes)
	if tr.Truncated {
		return tr.Content + "\n\n[Output truncated at " + formatSize(defaultMaxBytes) + ".]", nil
	}
	return result, nil
}

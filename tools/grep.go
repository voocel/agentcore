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
	"regexp"
	"strings"

	"github.com/voocel/agentcore/schema"
)

// GrepTool searches file contents by pattern.
// Uses ripgrep (rg) if available, falls back to regexp + bufio.Scanner.
type GrepTool struct {
	WorkDir string
}

func NewGrep(workDir string) *GrepTool { return &GrepTool{WorkDir: workDir} }

func (t *GrepTool) Name() string                              { return "grep" }
func (t *GrepTool) Label() string                              { return "Search Content" }
func (t *GrepTool) ReadOnly(_ json.RawMessage) bool            { return true }
func (t *GrepTool) ConcurrencySafe(_ json.RawMessage) bool     { return true }
func (t *GrepTool) ActivityDescription(_ json.RawMessage) string { return "Searching content" }
func (t *GrepTool) Description() string {
	return "Fast content search across files. Supports regex patterns by default, or exact text with literal=true. Use glob to narrow which files are searched. Returns relative file paths, line numbers, and matching lines (default limit: 100). Use bash only when you need shell-specific pipelines, counting, or custom post-processing."
}
func (t *GrepTool) Schema() map[string]any {
	return schema.Object(
		schema.Property("pattern", schema.String("Search pattern (regex by default, or exact text with literal=true)")).Required(),
		schema.Property("path", schema.String("File or directory to search, relative or absolute (default: working directory)")),
		schema.Property("glob", schema.String("Optional file glob filter (for example: '*.go', 'src/**/*.ts')")),
		schema.Property("ignoreCase", schema.Bool("Case insensitive search")),
		schema.Property("literal", schema.Bool("Treat pattern as literal string, not regex")),
		schema.Property("contextLines", schema.Int("Number of context lines around each match (default: 0)")),
		schema.Property("limit", schema.Int("Maximum number of matches (default: 100)")),
	)
}

type grepArgs struct {
	Pattern      string `json:"pattern"`
	Path         string `json:"path"`
	Glob         string `json:"glob"`
	IgnoreCase   bool   `json:"ignoreCase"`
	Literal      bool   `json:"literal"`
	ContextLines int    `json:"contextLines"`
	Limit        int    `json:"limit"`
}

const (
	grepDefaultLimit = 100
	grepMaxLineLen   = 500
	grepMaxBytes     = 50 * 1024
)

// rgMatchLineRe matches rg output lines that are actual matches (colon after line number),
// not context lines (dash after line number). Match: "file:42:content", Context: "file-40-content".
var rgMatchLineRe = regexp.MustCompile(`^.+:\d+:`)

func isRgMatchLine(line string) bool {
	return rgMatchLineRe.MatchString(line)
}

func (t *GrepTool) Execute(ctx context.Context, args json.RawMessage) (json.RawMessage, error) {
	var a grepArgs
	if err := json.Unmarshal(args, &a); err != nil {
		return nil, fmt.Errorf("invalid args: %w", err)
	}
	if a.Limit <= 0 {
		a.Limit = grepDefaultLimit
	}

	searchPath := ResolvePath(t.WorkDir, a.Path)

	// Try ripgrep first
	if result, err := t.grepWithRg(ctx, a, searchPath); err == nil {
		return result, nil
	}

	// Fallback to Go implementation
	return t.grepWithGo(ctx, a, searchPath)
}

// grepWithRg uses ripgrep with streaming output.
// Kills the process once the match limit is reached.
func (t *GrepTool) grepWithRg(ctx context.Context, a grepArgs, searchPath string) (json.RawMessage, error) {
	rgPath, err := exec.LookPath("rg")
	if err != nil {
		return nil, err
	}

	cmdArgs := []string{"--line-number", "--no-heading", "--color", "never"}

	if a.IgnoreCase {
		cmdArgs = append(cmdArgs, "--ignore-case")
	}
	if a.Literal {
		cmdArgs = append(cmdArgs, "--fixed-strings")
	}
	if a.ContextLines > 0 {
		cmdArgs = append(cmdArgs, fmt.Sprintf("--context=%d", a.ContextLines))
	}
	if a.Glob != "" {
		cmdArgs = append(cmdArgs, "--glob", a.Glob)
	}

	cmdArgs = append(cmdArgs, a.Pattern, searchPath)

	cmd := exec.CommandContext(ctx, rgPath, cmdArgs...)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, fmt.Errorf("pipe: %w", err)
	}
	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("start rg: %w", err)
	}

	prefix := searchPath + string(filepath.Separator)
	scanner := bufio.NewScanner(stdout)
	scanner.Buffer(make([]byte, 256*1024), 256*1024)

	var lines []string
	matchCount := 0
	hitLimit := false

	for scanner.Scan() {
		line := scanner.Text()

		// Make paths relative
		if rel, ok := strings.CutPrefix(line, prefix); ok {
			line = rel
		}

		// Truncate long lines
		if tl, truncated := truncateLine(line, grepMaxLineLen+50); truncated {
			line = tl
		}

		// Count actual matches (skip context lines and group separators)
		if line == "--" {
			// group separator, not a match
		} else if a.ContextLines > 0 {
			if isRgMatchLine(line) {
				matchCount++
			}
		} else {
			matchCount++
		}

		lines = append(lines, line)

		if matchCount >= a.Limit {
			hitLimit = true
			break
		}
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("scan rg output: %w", err)
	}

	// Kill rg process early if we hit the limit
	if hitLimit && cmd.Process != nil {
		cmd.Process.Kill()
	}
	waitErr := cmd.Wait()
	exitCode := 0
	if waitErr != nil {
		var exitErr *exec.ExitError
		if errors.As(waitErr, &exitErr) {
			exitCode = exitErr.ExitCode()
		} else if !hitLimit {
			return nil, fmt.Errorf("wait rg: %w", waitErr)
		}
	}

	if len(lines) == 0 {
		errMsg := strings.TrimSpace(stderr.String())
		if exitCode == 1 || (exitCode == 0 && errMsg == "") {
			return json.Marshal("No matches found.")
		}
		if errMsg != "" {
			return nil, fmt.Errorf("grep: %s", errMsg)
		}
		return json.Marshal("No matches found.")
	}

	result := strings.Join(lines, "\n")
	result = appendGrepNotices(result, a.Limit, hitLimit, exitCode == 2)

	// Apply byte truncation
	tr := truncateHead(result, 0, grepMaxBytes)
	if tr.Truncated {
		return json.Marshal(tr.Content + "\n\n[Output truncated.]")
	}
	return json.Marshal(result)
}

func (t *GrepTool) grepWithGo(ctx context.Context, a grepArgs, searchPath string) (json.RawMessage, error) {
	pattern := a.Pattern
	if a.Literal {
		pattern = regexp.QuoteMeta(pattern)
	}
	if a.IgnoreCase {
		pattern = "(?i)" + pattern
	}
	re, err := regexp.Compile(pattern)
	if err != nil {
		return nil, fmt.Errorf("invalid pattern: %w", err)
	}

	var results []string
	matchCount := 0
	limit := a.Limit

	err = filepath.WalkDir(searchPath, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			// If root path itself is invalid/inaccessible, return explicit error.
			if path == searchPath {
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
		rel, _ := filepath.Rel(searchPath, path)
		if a.Glob != "" {
			if !globPatternMatches(a.Glob, rel) {
				return nil
			}
		}
		// Skip binary/large files
		info, _ := d.Info()
		if info != nil && info.Size() > 1024*1024 {
			return nil
		}

		f, err := os.Open(path)
		if err != nil {
			return nil
		}
		scanner := bufio.NewScanner(f)
		scanner.Buffer(make([]byte, 256*1024), 2*1024*1024)
		var lines []string
		for scanner.Scan() {
			lines = append(lines, scanner.Text())
		}
		f.Close()
		if err := scanner.Err(); err != nil {
			return fmt.Errorf("scan %s: %w", rel, err)
		}

		remaining := limit - matchCount
		if remaining <= 0 {
			return filepath.SkipAll
		}

		fileResults, fileMatches, hitLimit := grepFileMatches(rel, lines, re, a.ContextLines, remaining)
		results = append(results, fileResults...)
		matchCount += fileMatches
		if hitLimit || matchCount >= limit {
			return filepath.SkipAll
		}
		return nil
	})

	if err != nil && err != filepath.SkipAll {
		return nil, fmt.Errorf("search: %w", err)
	}

	if len(results) == 0 {
		return json.Marshal("No matches found.")
	}

	result := strings.Join(results, "\n")
	result = appendGrepNotices(result, limit, matchCount >= limit, false)
	return json.Marshal(result)
}

func appendGrepNotices(result string, limit int, hitLimit, partial bool) string {
	if hitLimit {
		result += fmt.Sprintf("\n\n[Results truncated at %d matches. Use a more specific pattern or path.]", limit)
	}
	if partial {
		result += "\n\n[Some paths were inaccessible and skipped.]"
	}
	return result
}

func grepFileMatches(rel string, lines []string, re *regexp.Regexp, contextLines, limit int) ([]string, int, bool) {
	var matchIdx []int
	for i, line := range lines {
		if re.MatchString(line) {
			matchIdx = append(matchIdx, i)
			if len(matchIdx) >= limit {
				break
			}
		}
	}
	if len(matchIdx) == 0 {
		return nil, 0, false
	}

	if contextLines <= 0 {
		out := make([]string, 0, len(matchIdx))
		for _, idx := range matchIdx {
			out = append(out, formatGrepOutputLine(rel, idx+1, lines[idx], true))
		}
		return out, len(matchIdx), len(matchIdx) >= limit
	}

	matchSet := make(map[int]bool, len(matchIdx))
	type interval struct{ start, end int }
	var intervals []interval
	for _, idx := range matchIdx {
		matchSet[idx] = true
		start := max(idx-contextLines, 0)
		end := min(idx+contextLines, len(lines)-1)
		if len(intervals) == 0 || start > intervals[len(intervals)-1].end+1 {
			intervals = append(intervals, interval{start: start, end: end})
			continue
		}
		if end > intervals[len(intervals)-1].end {
			intervals[len(intervals)-1].end = end
		}
	}

	var out []string
	for i, iv := range intervals {
		if i > 0 {
			out = append(out, "--")
		}
		for lineIdx := iv.start; lineIdx <= iv.end; lineIdx++ {
			out = append(out, formatGrepOutputLine(rel, lineIdx+1, lines[lineIdx], matchSet[lineIdx]))
		}
	}
	return out, len(matchIdx), len(matchIdx) >= limit
}

func formatGrepOutputLine(rel string, lineNum int, line string, match bool) string {
	if tl, truncated := truncateLine(line, grepMaxLineLen); truncated {
		line = tl
	}
	sep := "-"
	if match {
		sep = ":"
	}
	return fmt.Sprintf("%s%s%d%s%s", rel, sep, lineNum, sep, line)
}

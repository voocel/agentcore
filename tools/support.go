package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path"
	"path/filepath"
	"strings"
	"unicode/utf8"

	"github.com/voocel/agentcore"
)

// unicodeSpaces are Unicode space characters normalized to ASCII space.
var unicodeSpaces = []rune{
	'\u00A0', // NO-BREAK SPACE
	'\u2002', // EN SPACE
	'\u2003', // EM SPACE
	'\u2004', // THREE-PER-EM SPACE
	'\u2005', // FOUR-PER-EM SPACE
	'\u2006', // SIX-PER-EM SPACE
	'\u2007', // FIGURE SPACE
	'\u2008', // PUNCTUATION SPACE
	'\u2009', // THIN SPACE
	'\u200A', // HAIR SPACE
	'\u202F', // NARROW NO-BREAK SPACE
	'\u205F', // MEDIUM MATHEMATICAL SPACE
	'\u3000', // IDEOGRAPHIC SPACE
}

// textRun returns a Tool.Run of execute, which returns text.
func textRun(execute func(context.Context, json.RawMessage) (string, error)) func(context.Context, json.RawMessage) (agentcore.Result, error) {
	return func(ctx context.Context, args json.RawMessage) (agentcore.Result, error) {
		text, err := execute(ctx, args)
		if err != nil {
			return agentcore.Result{}, err
		}
		return agentcore.TextResult(text), nil
	}
}

// Truncation limit defaults.
const (
	defaultMaxLines = 2000
	defaultMaxBytes = 50 * 1024 // 50KB
)

// skipDirs are directory names excluded from recursive traversal.
var skipDirs = map[string]bool{
	".git":         true,
	"node_modules": true,
	"__pycache__":  true,
	".venv":        true,
}

// truncation is text cut to a number of lines and bytes.
type truncation struct {
	Content   string
	Truncated bool
	// OutputLines is how many lines Content holds.
	OutputLines int
}

// ExpandPath normalizes a user-provided path:
//   - Replaces Unicode special spaces with ASCII space
//   - Expands ~ to the user's home directory
func ExpandPath(p string) string {
	for _, r := range unicodeSpaces {
		p = strings.ReplaceAll(p, string(r), " ")
	}

	if p == "~" {
		if home, err := os.UserHomeDir(); err == nil {
			return home
		}
		return p
	}
	if strings.HasPrefix(p, "~/") {
		if home, err := os.UserHomeDir(); err == nil {
			return filepath.Join(home, p[2:])
		}
	}
	return p
}

// ResolvePath resolves a user-provided path against a working directory.
// If userPath is empty, returns workDir. If absolute, returns as-is.
// Otherwise joins with workDir.
//
// Path semantics follow the workspace root. FS backends key files
// with slash-separated absolute paths (editor buffers, remote hosts), so a
// leading "/" counts as absolute even on Windows — where filepath.IsAbs would
// demand a drive letter and the fallthrough join would corrupt the virtual
// path ("/work" + "/work/f.txt" → `\work\work\f.txt`). Likewise a
// slash-rooted workDir joins with slash separators, never the OS's. OS-rooted
// paths (drive letters, UNC) keep filepath semantics unchanged, and on Unix
// the two branches coincide.
func ResolvePath(workDir, userPath string) string {
	if userPath == "" {
		return workDir
	}
	expanded := ExpandPath(userPath)
	if filepath.IsAbs(expanded) || strings.HasPrefix(expanded, "/") {
		return expanded
	}
	if strings.HasPrefix(workDir, "/") {
		return path.Join(workDir, filepath.ToSlash(expanded))
	}
	return filepath.Join(workDir, expanded)
}

// dirOf and joinOf are filepath.Dir / filepath.Join with the same
// virtual-path awareness as ResolvePath: slash-rooted paths stay
// slash-separated instead of being rewritten with the OS separator, so a
// resolved FS path survives parent-dir and sibling derivation on
// Windows.

func dirOf(p string) string {
	if strings.HasPrefix(p, "/") {
		return path.Dir(p)
	}
	return filepath.Dir(p)
}

func joinOf(dir, name string) string {
	if strings.HasPrefix(dir, "/") {
		return path.Join(dir, name)
	}
	return filepath.Join(dir, name)
}

// IsSkipDir reports whether a directory name should be excluded from traversal.
func IsSkipDir(name string) bool {
	return skipDirs[name]
}

// truncateHead keeps the first whole lines of content, at most maxLines of
// them and maxBytes; none when the first line alone exceeds maxBytes.
func truncateHead(content string, maxLines, maxBytes int) truncation {
	lines := strings.Split(content, "\n")
	if len(lines) <= maxLines && len(content) <= maxBytes {
		return truncation{Content: content, OutputLines: len(lines)}
	}

	var kept []string
	byteCount := 0
	for i, line := range lines {
		lineBytes := len(line)
		if i > 0 {
			lineBytes++
		}
		if byteCount+lineBytes > maxBytes || len(kept) >= maxLines {
			break
		}
		kept = append(kept, line)
		byteCount += lineBytes
	}
	return truncation{Content: strings.Join(kept, "\n"), Truncated: true, OutputLines: len(kept)}
}

// truncateTail keeps the last whole lines of content, at most maxLines of
// them and maxBytes. When the last line alone exceeds maxBytes, it keeps
// the end of that line, from a UTF-8 character boundary.
func truncateTail(content string, maxLines, maxBytes int) truncation {
	lines := strings.Split(content, "\n")
	if len(lines) <= maxLines && len(content) <= maxBytes {
		return truncation{Content: content, OutputLines: len(lines)}
	}

	var kept []string
	byteCount := 0
	for i := len(lines) - 1; i >= 0 && len(kept) < maxLines; i-- {
		line := lines[i]
		lineBytes := len(line)
		if len(kept) > 0 {
			lineBytes++
		}
		if byteCount+lineBytes > maxBytes {
			break
		}
		kept = append([]string{line}, kept...)
		byteCount += lineBytes
	}
	if len(kept) == 0 {
		return truncation{Content: truncateBytesFromEnd(lines[len(lines)-1], maxBytes), Truncated: true, OutputLines: 1}
	}
	return truncation{Content: strings.Join(kept, "\n"), Truncated: true, OutputLines: len(kept)}
}

// truncateBytesFromEnd returns the last maxBytes bytes of s,
// ensuring the result starts at a valid UTF-8 character boundary.
func truncateBytesFromEnd(s string, maxBytes int) string {
	if len(s) <= maxBytes {
		return s
	}
	start := len(s) - maxBytes
	for start < len(s) && !utf8.RuneStart(s[start]) {
		start++
	}
	return s[start:]
}

// truncateLine truncates a single line to maxRunes rune characters.
// Uses rune-safe slicing to avoid producing invalid UTF-8.
func truncateLine(line string, maxRunes int) (string, bool) {
	if utf8.RuneCountInString(line) <= maxRunes {
		return line, false
	}
	runes := []rune(line)
	return string(runes[:maxRunes]) + "... [truncated]", true
}

func formatSize(bytes int) string {
	switch {
	case bytes < 1024:
		return fmt.Sprintf("%dB", bytes)
	case bytes < 1024*1024:
		return fmt.Sprintf("%.1fKB", float64(bytes)/1024)
	default:
		return fmt.Sprintf("%.1fMB", float64(bytes)/(1024*1024))
	}
}

// ---------------------------------------------------------------------------
// Glob pattern matching (path-aware, supports **)
// ---------------------------------------------------------------------------

// globPatternMatches tests whether rel matches a path-aware glob pattern.
// Patterns without "/" match against the file's basename only.
// Patterns with "/" match segment-by-segment, supporting "**" for zero-or-more directories.
func globPatternMatches(pattern, rel string) bool {
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

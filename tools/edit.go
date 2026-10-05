package tools

import (
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"slices"
	"strconv"
	"strings"
	"unicode"

	"github.com/voocel/agentcore"
	"github.com/voocel/agentcore/schema"
)

// Edit returns the edit tool: it replaces exact strings in a file,
// normalizing line endings and matching fuzzily. Its result is a line naming
// the file over the diff of the edit; its Check returns the diff as the
// call's preview. With Files, its Check and Run refuse a file the model has
// not read, or that changed since.
func (w Workspace) Edit() agentcore.Tool {
	t := &editTool{w: w, fs: w.fs()}
	return agentcore.Tool{
		Name:        "edit",
		Label:       "Edit File",
		Description: editDescription,
		Schema: schema.Object(
			schema.Property("file_path", schema.String("The path to the file to modify (relative or absolute)")).Required(),
			schema.Property("old_string", schema.String("The text to replace (must be unique unless replace_all is true)")).Required(),
			schema.Property("new_string", schema.String("The text to replace it with (must be different from old_string)")).Required(),
			schema.Property("replace_all", schema.Bool("Replace all occurrences of old_string (default: false)")),
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

type editTool struct {
	w  Workspace
	fs FS
}

const editDescription = `Performs exact string replacements in files.

Usage:
- You must use the read tool at least once in the conversation before editing. This tool will error if you attempt an edit without reading the file.
- When editing text from read tool output, ensure you preserve the exact indentation (tabs/spaces) as it appears AFTER the line number prefix. The line number prefix format is: line number + tab. Everything after that is the actual file content to match. Never include any part of the line number prefix in the old_string or new_string.
- ALWAYS prefer editing existing files in the codebase. NEVER write new files unless explicitly required.
- The edit will FAIL if ` + "`old_string`" + ` is not unique in the file. Either provide a larger string with more surrounding context to make it unique or use ` + "`replace_all`" + ` to change every instance of ` + "`old_string`" + `.
- Use ` + "`replace_all`" + ` for replacing and renaming strings across the file. This parameter is useful if you want to rename a variable for instance.`

type editArgs struct {
	FilePath   string `json:"file_path"`
	OldString  string `json:"old_string"`
	NewString  string `json:"new_string"`
	ReplaceAll bool   `json:"replace_all"`
}

// editResult is an edit worked out, before it is written.
type editResult struct {
	path       string
	bom        string
	ending     string // original line ending
	oldContent string // normalized-to-LF content before edit
	newContent string
}

// validate enforces read-before-edit and detects stale writes. Unlike write,
// edit always requires an existing file — a non-existent path fails — and a
// partial read is enough.
func (t *editTool) validate(ctx context.Context, args json.RawMessage) error {
	if t.w.Files == nil {
		return nil
	}

	var a editArgs
	if err := json.Unmarshal(args, &a); err != nil {
		return errors.New("invalid args: " + err.Error())
	}
	path := ResolvePath(t.w.dir(ctx), a.FilePath)

	info, err := t.fs.Stat(ctx, path)
	if err != nil {
		if os.IsNotExist(err) {
			return errors.New("file not found: " + path)
		}
		return errors.New("stat " + path + ": " + err.Error())
	}
	if info.IsDir {
		return errors.New("path is a directory: " + path)
	}

	stamp, ok := t.w.Files.Get(path)
	if !ok {
		return errors.New("File has not been read yet. Read it first before editing.")
	}
	// The content token or mtime must equal the one recorded at read time,
	// not just be no later: that catches mtime regressions too (e.g. git
	// checkout of an older version), and unsaved-buffer changes when the
	// backend sets Version.
	if !stampMatches(stamp, info) {
		return errors.New("File has been modified since read, either by the user or by a linter. Read it again before attempting to edit it.")
	}
	return nil
}

func (t *editTool) parseAndMatch(ctx context.Context, args json.RawMessage) (*editResult, error) {
	var a editArgs
	if err := json.Unmarshal(args, &a); err != nil {
		return nil, fmt.Errorf("invalid args: %w", err)
	}

	a.FilePath = ResolvePath(t.w.dir(ctx), a.FilePath)
	if a.OldString == "" {
		return nil, errors.New("old_string is empty. Quote the text to replace; use write to create a file")
	}

	data, err := t.fs.ReadFile(ctx, a.FilePath)
	if os.IsNotExist(err) {
		return nil, fmt.Errorf("file not found: %s", a.FilePath)
	}
	if err != nil {
		return nil, fmt.Errorf("read %s: %w", a.FilePath, err)
	}

	bom, raw := stripBOM(string(data))
	originalEnding := detectLineEnding(raw)
	content := normalizeToLF(raw)
	oldText := normalizeToLF(a.OldString)
	newText := normalizeToLF(a.NewString)

	matches, reindent := findMatches(content, oldText)
	switch {
	case len(matches) == 0:
		if hints := formatEditCandidates(content, oldText); hints != "" {
			return nil, fmt.Errorf("could not find the exact text in %s. The old text must match exactly including all whitespace and newlines.\n\nPossible old_string candidates (copy one exactly):\n%s", a.FilePath, hints)
		}
		return nil, fmt.Errorf("could not find the exact text in %s. The old text must match exactly including all whitespace and newlines", a.FilePath)
	case len(matches) > 1 && !a.ReplaceAll:
		return nil, fmt.Errorf("found %d occurrences of the text in %s. Use replace_all=true to replace all, or provide more context to make the match unique", len(matches), a.FilePath)
	}

	var b strings.Builder
	end := 0
	for _, m := range matches {
		replacement := newText
		if reindent {
			replacement = reindentReplacement(newText, oldText, content[m.start:m.end])
		}
		b.WriteString(content[end:m.start])
		b.WriteString(replacement)
		end = m.end
	}
	b.WriteString(content[end:])
	newContent := b.String()
	if newContent == content {
		return nil, fmt.Errorf("old_string and new_string are identical in %s. Provide a new_string that is different from the matched text", a.FilePath)
	}

	return &editResult{
		path:       a.FilePath,
		bom:        bom,
		ending:     originalEnding,
		oldContent: content,
		newContent: newContent,
	}, nil
}

// preview returns the diff of the edit, without writing it.
func (t *editTool) preview(ctx context.Context, args json.RawMessage) (string, error) {
	r, err := t.parseAndMatch(ctx, args)
	if err != nil {
		return "", err
	}
	return generateDiff(r.oldContent, r.newContent), nil
}

// execute checks the file again: it may have changed while the call
// awaited approval.
func (t *editTool) execute(ctx context.Context, args json.RawMessage) (agentcore.Result, error) {
	if err := t.validate(ctx, args); err != nil {
		return agentcore.Result{}, err
	}
	r, err := t.parseAndMatch(ctx, args)
	if err != nil {
		return agentcore.Result{}, err
	}
	if err := ctx.Err(); err != nil {
		return agentcore.Result{}, err
	}
	finalContent := r.bom + restoreLineEndings(r.newContent, r.ending)
	if err := t.fs.WriteFile(ctx, r.path, []byte(finalContent), 0o644); err != nil {
		return agentcore.Result{}, fmt.Errorf("write %s: %w", r.path, err)
	}
	t.w.Files.recordWrite(ctx, t.fs, r.path, false)
	return agentcore.TextResult(fmt.Sprintf("Edited %s.\n%s", r.path, generateDiff(r.oldContent, r.newContent))), nil
}

// --- Line ending utilities ---

func detectLineEnding(content string) string {
	crlfIdx := strings.Index(content, "\r\n")
	lfIdx := strings.Index(content, "\n")
	if lfIdx == -1 || crlfIdx == -1 {
		return "\n"
	}
	if crlfIdx < lfIdx {
		return "\r\n"
	}
	return "\n"
}

func normalizeToLF(text string) string {
	text = strings.ReplaceAll(text, "\r\n", "\n")
	text = strings.ReplaceAll(text, "\r", "\n")
	return text
}

func restoreLineEndings(text, ending string) string {
	if ending == "\r\n" {
		return strings.ReplaceAll(text, "\n", "\r\n")
	}
	return text
}

// --- BOM ---

func stripBOM(s string) (bom, text string) {
	if strings.HasPrefix(s, "\uFEFF") {
		return "\uFEFF", s[len("\uFEFF"):]
	}
	return "", s
}

// --- Fuzzy matching ---

// normalizeRuneForFuzzy normalizes one rune for fuzzy matching.
func normalizeRuneForFuzzy(r rune) rune {
	switch r {
	case '\u2018', '\u2019', '\u201A', '\u201B':
		return '\''
	case '\u201C', '\u201D', '\u201E', '\u201F':
		return '"'
	case '\u2010', '\u2011', '\u2012', '\u2013', '\u2014', '\u2015', '\u2212':
		return '-'
	}
	for _, s := range unicodeSpaces {
		if r == s {
			return ' '
		}
	}
	return r
}

// span is the bytes of content a match of old_string covers.
type span struct{ start, end int }

// findMatches returns the matches of oldText in content, in order and not
// overlapping, by the first of these that finds any: exact; fuzzy; and, for
// text of several lines, indentation-insensitive, whose matches take the
// replacement reindented to theirs.
func findMatches(content, oldText string) (matches []span, reindent bool) {
	if m := exactMatches(content, oldText); len(m) > 0 {
		return m, false
	}
	if m := fuzzyMatches(content, oldText); len(m) > 0 {
		return m, false
	}
	if strings.Contains(oldText, "\n") {
		return indentAwareMatches(content, oldText), true
	}
	return nil, false
}

func exactMatches(content, oldText string) []span {
	var out []span
	for at := 0; ; {
		i := strings.Index(content[at:], oldText)
		if i < 0 {
			return out
		}
		at += i
		out = append(out, span{at, at + len(oldText)})
		at += len(oldText)
	}
}

// fuzzyMatches matches oldText ignoring the trailing whitespace of lines and
// the look of quotes, dashes and spaces. The spans cover the bytes of
// content as they are.
func fuzzyMatches(content, oldText string) []span {
	runes, offsets := normalizeForFuzzy(content)
	needle, _ := normalizeForFuzzy(oldText)
	if len(needle) == 0 {
		return nil
	}
	var out []span
	for i := 0; i+len(needle) <= len(runes); {
		if !slices.Equal(runes[i:i+len(needle)], needle) {
			i++
			continue
		}
		out = append(out, span{offsets[i], offsets[i+len(needle)]})
		i += len(needle)
	}
	return out
}

// normalizeForFuzzy strips the trailing whitespace of each line of text and
// normalizes smart quotes, dashes and Unicode spaces to ASCII. offsets holds
// the byte of text each rune comes from, and then the end of text.
func normalizeForFuzzy(text string) (runes []rune, offsets []int) {
	lines := strings.Split(text, "\n")
	runes = make([]rune, 0, len(text))
	offsets = make([]int, 0, len(text)+1)
	start := 0
	for i, line := range lines {
		for j, r := range strings.TrimRightFunc(line, unicode.IsSpace) {
			runes = append(runes, normalizeRuneForFuzzy(r))
			offsets = append(offsets, start+j)
		}
		if i < len(lines)-1 {
			runes = append(runes, '\n')
			offsets = append(offsets, start+len(line))
		}
		start += len(line) + 1
	}
	return runes, append(offsets, len(text))
}

// indentAwareMatches matches oldText as whole lines, each with the newline
// that ends it, ignoring their common indentation; a final newline in
// oldText ends its last line.
func indentAwareMatches(content, oldText string) []span {
	oldLines := strings.Split(strings.TrimSuffix(oldText, "\n"), "\n")
	contentLines := strings.Split(content, "\n")
	target := normalizeLinesForIndentAware(oldLines)
	lineStarts := lineStartOffsets(content)

	var out []span
	for i := 0; i+len(oldLines) <= len(contentLines); {
		if normalizeLinesForIndentAware(contentLines[i:i+len(oldLines)]) != target {
			i++
			continue
		}
		last := i + len(oldLines) - 1
		out = append(out, span{lineStarts[i], min(lineStarts[last]+len(contentLines[last])+1, len(content))})
		i += len(oldLines)
	}
	return out
}

func normalizeLinesForIndentAware(lines []string) string {
	processed := make([]string, len(lines))
	minIndent := -1
	for i, line := range lines {
		line = strings.Map(normalizeRuneForFuzzy, strings.TrimRightFunc(line, unicode.IsSpace))
		processed[i] = line

		trimmed := strings.TrimLeft(line, " \t")
		if trimmed == "" {
			continue
		}
		indent := len(line) - len(trimmed)
		if minIndent == -1 || indent < minIndent {
			minIndent = indent
		}
	}

	if minIndent > 0 {
		for i, line := range processed {
			if strings.TrimSpace(line) == "" {
				continue
			}
			if len(line) >= minIndent {
				processed[i] = line[minIndent:]
			}
		}
	}
	return strings.Join(processed, "\n")
}

func reindentReplacement(newText, oldText, matchedText string) string {
	oldIndent := commonIndentPrefix(strings.Split(oldText, "\n"))
	matchIndent := commonIndentPrefix(strings.Split(matchedText, "\n"))
	if oldIndent == matchIndent {
		return preserveMatchedTrailingNewline(newText, matchedText)
	}

	lines := strings.Split(newText, "\n")
	for i, line := range lines {
		if strings.TrimSpace(line) == "" {
			continue
		}
		lines[i] = matchIndent + removeIndentPrefix(line, oldIndent)
	}
	return preserveMatchedTrailingNewline(strings.Join(lines, "\n"), matchedText)
}

func commonIndentPrefix(lines []string) string {
	var common string
	for _, line := range lines {
		if strings.TrimSpace(line) == "" {
			continue
		}
		indent := leadingIndent(line)
		if common == "" {
			common = indent
			continue
		}
		common = sharedPrefix(common, indent)
		if common == "" {
			return ""
		}
	}
	return common
}

func leadingIndent(line string) string {
	i := 0
	for i < len(line) && (line[i] == ' ' || line[i] == '\t') {
		i++
	}
	return line[:i]
}

func sharedPrefix(a, b string) string {
	n := min(len(a), len(b))
	i := 0
	for i < n && a[i] == b[i] {
		i++
	}
	return a[:i]
}

func removeIndentPrefix(line, indent string) string {
	if indent == "" {
		return line
	}
	if strings.HasPrefix(line, indent) {
		return line[len(indent):]
	}

	trim := len(indent)
	i := 0
	for i < len(line) && trim > 0 && (line[i] == ' ' || line[i] == '\t') {
		i++
		trim--
	}
	return line[i:]
}

func preserveMatchedTrailingNewline(text, matchedText string) string {
	if strings.HasSuffix(matchedText, "\n") && !strings.HasSuffix(text, "\n") {
		return text + "\n"
	}
	return text
}

type editCandidate struct {
	startLine int
	endLine   int
	block     string
	score     int
}

const (
	candidatePreviewLines      = 8
	candidateSimilarityScale   = 100
	minimumCandidateSimilarity = candidateSimilarityScale / 2
)

func formatEditCandidates(content, oldText string) string {
	candidates := suggestEditCandidates(content, oldText)
	if len(candidates) == 0 {
		return ""
	}

	var sb strings.Builder
	for i, c := range candidates {
		preview := candidatePreview(c.block)
		fmt.Fprintf(&sb, "%d. lines %d-%d\n```text\n%s\n```\n", i+1, c.startLine, c.endLine, preview)
		if c.endLine-c.startLine+1 > candidatePreviewLines {
			fmt.Fprintf(&sb, "   Use read with offset=%d limit=%d to see the full block.\n", c.startLine, c.endLine-c.startLine+1)
		}
	}
	return strings.TrimRight(sb.String(), "\n")
}

func candidatePreview(block string) string {
	block = strings.TrimRight(block, "\n")
	lines := strings.Split(block, "\n")
	if len(lines) <= candidatePreviewLines {
		return block
	}
	head := candidatePreviewLines / 2
	tail := candidatePreviewLines - head
	var sb strings.Builder
	for _, l := range lines[:head] {
		sb.WriteString(l)
		sb.WriteByte('\n')
	}
	fmt.Fprintf(&sb, "... (%d lines omitted)\n", len(lines)-head-tail)
	for _, l := range lines[len(lines)-tail:] {
		sb.WriteString(l)
		sb.WriteByte('\n')
	}
	return strings.TrimRight(sb.String(), "\n")
}

func suggestEditCandidates(content, oldText string) []editCandidate {
	targetLines := splitCandidateLines(oldText)
	contentLines := splitCandidateLines(content)
	if len(targetLines) == 0 || len(contentLines) == 0 {
		return nil
	}

	if len(targetLines) == 1 {
		return suggestSingleLineCandidates(contentLines, targetLines[0])
	}
	return suggestBlockCandidates(contentLines, targetLines)
}

func splitCandidateLines(text string) []string {
	lines := strings.Split(normalizeToLF(text), "\n")
	if len(lines) > 0 && lines[len(lines)-1] == "" {
		lines = lines[:len(lines)-1]
	}
	return lines
}

func suggestSingleLineCandidates(contentLines []string, target string) []editCandidate {
	var out []editCandidate
	for i, line := range contentLines {
		score := scoreCandidateLine(line, target)
		if score <= 0 {
			continue
		}
		out = append(out, editCandidate{
			startLine: i + 1,
			endLine:   i + 1,
			block:     line,
			score:     score,
		})
	}
	return topEditCandidates(out)
}

func suggestBlockCandidates(contentLines, targetLines []string) []editCandidate {
	windowSize := len(targetLines)
	if windowSize > len(contentLines) {
		return nil
	}

	var out []editCandidate
	for i := 0; i+windowSize <= len(contentLines); i++ {
		window := contentLines[i : i+windowSize]
		score := scoreCandidateBlock(window, targetLines)
		if score <= 0 {
			continue
		}
		out = append(out, editCandidate{
			startLine: i + 1,
			endLine:   i + windowSize,
			block:     strings.Join(window, "\n"),
			score:     score,
		})
	}
	return topEditCandidates(out)
}

func scoreCandidateBlock(candidateLines, targetLines []string) int {
	score := 0
	for i := range targetLines {
		score += scoreCandidateLine(candidateLines[i], targetLines[i])
	}

	if normalizeSearchText(candidateLines[0]) == normalizeSearchText(targetLines[0]) {
		score += candidateSimilarityScale / 2
	}
	last := len(targetLines) - 1
	if normalizeSearchText(candidateLines[last]) == normalizeSearchText(targetLines[last]) {
		score += candidateSimilarityScale / 2
	}

	if normalizeLinesForIndentAware(candidateLines) == normalizeLinesForIndentAware(targetLines) {
		score += candidateSimilarityScale
	}
	return score
}

func scoreCandidateLine(candidate, target string) int {
	c := normalizeSearchText(candidate)
	t := normalizeSearchText(target)
	if c == "" || t == "" {
		return 0
	}
	if c == t {
		return candidateSimilarityScale
	}

	c = collapseWhitespace(c)
	t = collapseWhitespace(t)
	if c == t {
		return candidateSimilarityScale - 1
	}
	similarity := runeBigramSimilarity(c, t)
	if similarity >= minimumCandidateSimilarity {
		return similarity
	}
	return 0
}

func normalizeSearchText(text string) string {
	return strings.TrimSpace(strings.Map(normalizeRuneForFuzzy, text))
}

func collapseWhitespace(text string) string {
	return strings.Join(strings.Fields(text), " ")
}

type runeBigram struct {
	first  rune
	second rune
}

func runeBigramSimilarity(a, b string) int {
	ar := []rune(a)
	br := []rune(b)
	if len(ar) < 2 || len(br) < 2 {
		return 0
	}

	counts := make(map[runeBigram]int, len(ar)-1)
	for i := 1; i < len(ar); i++ {
		counts[runeBigram{first: ar[i-1], second: ar[i]}]++
	}

	overlap := 0
	for i := 1; i < len(br); i++ {
		pair := runeBigram{first: br[i-1], second: br[i]}
		if counts[pair] > 0 {
			overlap++
			counts[pair]--
		}
	}

	return overlap * 2 * candidateSimilarityScale / (len(ar) + len(br) - 2)
}

// topEditCandidates returns the three best candidates. They come in line
// order, which the sort keeps among those scored alike.
func topEditCandidates(candidates []editCandidate) []editCandidate {
	slices.SortStableFunc(candidates, func(a, b editCandidate) int { return cmp.Compare(b.score, a.score) })
	return candidates[:min(len(candidates), 3)]
}

func lineStartOffsets(text string) []int {
	offsets := []int{0}
	for i := 0; i < len(text); i++ {
		if text[i] == '\n' {
			offsets = append(offsets, i+1)
		}
	}
	return offsets
}

// --- Diff generation ---

// generateDiff returns the changed lines of newContent over oldContent, each
// marked "-" or "+" and numbered, with a few lines of context.
func generateDiff(oldContent, newContent string) string {
	const contextLines = 4

	oldLines, newLines := diffLines(oldContent), diffLines(newContent)
	maxOld, maxNew := len(oldLines), len(newLines)

	prefix := 0
	for prefix < maxOld && prefix < maxNew && oldLines[prefix] == newLines[prefix] {
		prefix++
	}
	// The common suffix, not overlapping the prefix.
	suffixOld, suffixNew := maxOld-1, maxNew-1
	for suffixOld >= prefix && suffixNew >= prefix && oldLines[suffixOld] == newLines[suffixNew] {
		suffixOld--
		suffixNew--
	}
	if suffixOld < prefix && suffixNew < prefix {
		return "(no changes)"
	}

	width := len(strconv.Itoa(max(maxOld, maxNew)))
	var sb strings.Builder
	row := func(mark byte, n int, line string) {
		fmt.Fprintf(&sb, "%c%*d %s\n", mark, width, n, strings.TrimSuffix(line, "\n"))
	}

	ctxStart := max(prefix-contextLines, 0)
	if ctxStart > 0 {
		fmt.Fprintf(&sb, " %*s ...\n", width, "")
	}
	for i := ctxStart; i < prefix; i++ {
		row(' ', i+1, oldLines[i])
	}
	for i := prefix; i <= suffixOld; i++ {
		row('-', i+1, oldLines[i])
	}
	for i := prefix; i <= suffixNew; i++ {
		row('+', i+1, newLines[i])
	}
	trailEnd := min(suffixOld+1+contextLines, maxOld)
	for i := suffixOld + 1; i < trailEnd; i++ {
		row(' ', i+1, oldLines[i])
	}
	if trailEnd < maxOld {
		fmt.Fprintf(&sb, " %*s ...\n", width, "")
	}
	return sb.String()
}

// diffLines splits s into lines that keep their newlines, so that a change
// to the last newline alone is a change of a line.
func diffLines(s string) []string {
	lines := strings.SplitAfter(s, "\n")
	if lines[len(lines)-1] == "" {
		lines = lines[:len(lines)-1]
	}
	return lines
}

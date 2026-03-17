package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"unicode"

	"github.com/voocel/agentcore/schema"
)

// EditTool performs exact string replacement in a file.
// Supports line ending normalization, fuzzy matching, and returns unified diff.
type EditTool struct {
	WorkDir string
}

func NewEdit(workDir string) *EditTool { return &EditTool{WorkDir: workDir} }

func (t *EditTool) Name() string  { return "edit" }
func (t *EditTool) Label() string { return "Edit File" }
func (t *EditTool) Description() string {
	return "Edit an existing file by replacing one unique text block. Use read first, then copy the exact file content into old_text without any line numbers or prefixes. Preserve indentation for multi-line edits. If the target appears multiple times, include more surrounding context instead of guessing."
}
func (t *EditTool) Schema() map[string]any {
	return schema.Object(
		schema.Property("path", schema.String("Path to the file to edit (relative or absolute)")).Required(),
		schema.Property("old_text", schema.String("Exact text to find and replace (must be unique in the file)")).Required(),
		schema.Property("new_text", schema.String("New text to replace the old text with")).Required(),
	)
}

type editArgs struct {
	Path    string `json:"path"`
	OldText string `json:"old_text"`
	NewText string `json:"new_text"`
}

// editResult holds the parsed and computed edit state, shared by Preview and Execute.
type editResult struct {
	path       string
	bom        string
	ending     string // original line ending
	oldContent string // normalized-to-LF content before edit
	newContent string
}

// parseAndMatch reads the file, finds the match, and computes the replacement.
func (t *EditTool) parseAndMatch(args json.RawMessage) (*editResult, error) {
	var a editArgs
	if err := json.Unmarshal(args, &a); err != nil {
		return nil, fmt.Errorf("invalid args: %w", err)
	}

	a.Path = ResolvePath(t.WorkDir, a.Path)

	data, err := os.ReadFile(a.Path)
	if err != nil {
		return nil, fmt.Errorf("file not found: %s", a.Path)
	}

	raw := string(data)
	bom, raw := stripBOM(raw)

	originalEnding := detectLineEnding(raw)
	content := normalizeToLF(raw)
	oldText := normalizeToLF(a.OldText)
	newText := normalizeToLF(a.NewText)

	idx, matchLen := fuzzyFind(content, oldText)
	count := 0
	usedIndentAware := false
	if idx < 0 {
		if strings.Contains(oldText, "\n") {
			idx, matchLen, count = indentAwareFind(content, oldText)
			if count > 1 {
				return nil, fmt.Errorf("found %d indentation-insensitive occurrences of the text in %s. Provide more context", count, a.Path)
			}
			usedIndentAware = idx >= 0
		}
	}
	if idx < 0 {
		if hints := formatEditCandidates(content, oldText); hints != "" {
			return nil, fmt.Errorf("could not find the exact text in %s. The old text must match exactly including all whitespace and newlines.\n\nPossible old_text candidates (copy one exactly):\n%s", a.Path, hints)
		}
		return nil, fmt.Errorf("could not find the exact text in %s. The old text must match exactly including all whitespace and newlines", a.Path)
	}

	count = strings.Count(normalizeForFuzzy(content), normalizeForFuzzy(oldText))
	if count > 1 {
		return nil, fmt.Errorf("found %d occurrences of the text in %s. The text must be unique. Provide more context", count, a.Path)
	}

	if usedIndentAware {
		newText = reindentReplacement(newText, oldText, content[idx:idx+matchLen])
	}

	newContent := content[:idx] + newText + content[idx+matchLen:]
	if content == newContent {
		return nil, fmt.Errorf("no changes made to %s. The replacement produced identical content", a.Path)
	}

	return &editResult{
		path:       a.Path,
		bom:        bom,
		ending:     originalEnding,
		oldContent: content,
		newContent: newContent,
	}, nil
}

// Preview computes the diff without writing the file.
func (t *EditTool) Preview(ctx context.Context, args json.RawMessage) (json.RawMessage, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	r, err := t.parseAndMatch(args)
	if err != nil {
		return nil, err
	}
	diff, firstLine := generateDiff(r.oldContent, r.newContent)
	return json.Marshal(map[string]any{
		"diff":               diff,
		"first_changed_line": firstLine,
	})
}

func (t *EditTool) Execute(ctx context.Context, args json.RawMessage) (json.RawMessage, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	r, err := t.parseAndMatch(args)
	if err != nil {
		return nil, err
	}

	if err := ctx.Err(); err != nil {
		return nil, err
	}

	finalContent := r.bom + restoreLineEndings(r.newContent, r.ending)
	if err := os.WriteFile(r.path, []byte(finalContent), 0o644); err != nil {
		return nil, fmt.Errorf("write %s: %w", r.path, err)
	}

	diff, firstLine := generateDiff(r.oldContent, r.newContent)
	return json.Marshal(map[string]any{
		"message":            fmt.Sprintf("Successfully replaced text in %s.", r.path),
		"diff":               diff,
		"first_changed_line": firstLine,
	})
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

// normalizeForFuzzy strips trailing whitespace per line and normalizes
// smart quotes, dashes, Unicode spaces to ASCII equivalents.
func normalizeForFuzzy(text string) string {
	lines := strings.Split(text, "\n")
	for i, line := range lines {
		line = strings.TrimRightFunc(line, unicode.IsSpace)
		lines[i] = strings.Map(normalizeRuneForFuzzy, line)
	}
	return strings.Join(lines, "\n")
}

type fuzzyNormalized struct {
	runes      []rune
	runeToByte []int
}

func normalizeForFuzzyWithMap(text string) fuzzyNormalized {
	lines := strings.Split(text, "\n")
	outRunes := make([]rune, 0, len(text))
	runeToByte := make([]int, 0, len(text)+1)

	globalByte := 0
	for li, line := range lines {
		trimmed := strings.TrimRightFunc(line, unicode.IsSpace)
		for relByte, r := range trimmed {
			outRunes = append(outRunes, normalizeRuneForFuzzy(r))
			runeToByte = append(runeToByte, globalByte+relByte)
		}
		if li < len(lines)-1 {
			outRunes = append(outRunes, '\n')
			runeToByte = append(runeToByte, globalByte+len(line))
		}
		globalByte += len(line)
		if li < len(lines)-1 {
			globalByte++
		}
	}

	runeToByte = append(runeToByte, len(text))
	return fuzzyNormalized{
		runes:      outRunes,
		runeToByte: runeToByte,
	}
}

func indexRuneSlice(haystack, needle []rune) int {
	if len(needle) == 0 {
		return 0
	}
	if len(needle) > len(haystack) {
		return -1
	}
outer:
	for i := 0; i+len(needle) <= len(haystack); i++ {
		for j := 0; j < len(needle); j++ {
			if haystack[i+j] != needle[j] {
				continue outer
			}
		}
		return i
	}
	return -1
}

// fuzzyFind tries exact match first, then fuzzy match.
// Fuzzy matching is only used for locating the replacement range.
// The returned index/length always point to the original content bytes.
func fuzzyFind(content, oldText string) (idx, matchLen int) {
	if i := strings.Index(content, oldText); i >= 0 {
		return i, len(oldText)
	}

	normContent := normalizeForFuzzyWithMap(content)
	fuzzyOld := normalizeForFuzzy(oldText)
	oldRunes := []rune(fuzzyOld)
	runeIdx := indexRuneSlice(normContent.runes, oldRunes)
	if runeIdx < 0 {
		return -1, 0
	}

	if runeIdx+len(oldRunes) > len(normContent.runeToByte)-1 {
		return -1, 0
	}
	startByte := normContent.runeToByte[runeIdx]
	endByte := normContent.runeToByte[runeIdx+len(oldRunes)]
	if startByte < 0 || endByte < startByte || endByte > len(content) {
		return -1, 0
	}
	return startByte, endByte - startByte
}

func indentAwareFind(content, oldText string) (idx, matchLen, count int) {
	oldLines := strings.Split(oldText, "\n")
	contentLines := strings.Split(content, "\n")
	if len(oldLines) == 0 || len(oldLines) > len(contentLines) {
		return -1, 0, 0
	}

	target := normalizeLinesForIndentAware(oldLines)
	lineStarts := lineStartOffsets(content)

	var matches []struct{ start, end int }
	for i := 0; i+len(oldLines) <= len(contentLines); i++ {
		window := contentLines[i : i+len(oldLines)]
		if normalizeLinesForIndentAware(window) != target {
			continue
		}
		matches = append(matches, struct{ start, end int }{
			start: lineStarts[i],
			end:   lineStarts[i+len(oldLines)],
		})
	}

	if len(matches) == 0 {
		return -1, 0, 0
	}
	if len(matches) > 1 {
		return -1, 0, len(matches)
	}
	m := matches[0]
	return m.start, m.end - m.start, 1
}

func normalizeLinesForIndentAware(lines []string) string {
	processed := make([]string, len(lines))
	minIndent := -1
	for i, line := range lines {
		line = strings.TrimRightFunc(line, unicode.IsSpace)
		line = strings.Map(normalizeRuneForFuzzy, line)
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

func formatEditCandidates(content, oldText string) string {
	candidates := suggestEditCandidates(content, oldText)
	if len(candidates) == 0 {
		return ""
	}

	var sb strings.Builder
	for i, c := range candidates {
		fmt.Fprintf(&sb, "%d. lines %d-%d\n```text\n%s\n```\n", i+1, c.startLine, c.endLine, strings.TrimRight(c.block, "\n"))
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

	if trimmedLine(candidateLines[0]) == trimmedLine(targetLines[0]) {
		score += 2
	}
	last := len(targetLines) - 1
	if trimmedLine(candidateLines[last]) == trimmedLine(targetLines[last]) {
		score += 2
	}

	if normalizeLinesForIndentAware(candidateLines) == normalizeLinesForIndentAware(targetLines) {
		score += 4
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
		return 4
	}
	if collapseWhitespace(candidate) == collapseWhitespace(target) {
		return 3
	}
	if strings.Contains(c, t) || strings.Contains(t, c) {
		return 2
	}
	if tokenOverlap(c, t) > 0 {
		return 1
	}
	return 0
}

func normalizeSearchText(text string) string {
	text = strings.Map(normalizeRuneForFuzzy, text)
	return strings.TrimSpace(text)
}

func collapseWhitespace(text string) string {
	return strings.Join(strings.Fields(normalizeSearchText(text)), " ")
}

func tokenOverlap(a, b string) int {
	seen := make(map[string]struct{})
	for _, part := range strings.Fields(a) {
		seen[part] = struct{}{}
	}
	overlap := 0
	for _, part := range strings.Fields(b) {
		if _, ok := seen[part]; ok {
			overlap++
		}
	}
	return overlap
}

func trimmedLine(text string) string {
	return strings.TrimSpace(strings.Map(normalizeRuneForFuzzy, text))
}

func topEditCandidates(candidates []editCandidate) []editCandidate {
	if len(candidates) == 0 {
		return nil
	}

	for i := 0; i < len(candidates); i++ {
		best := i
		for j := i + 1; j < len(candidates); j++ {
			if candidates[j].score > candidates[best].score ||
				(candidates[j].score == candidates[best].score && candidates[j].startLine < candidates[best].startLine) {
				best = j
			}
		}
		candidates[i], candidates[best] = candidates[best], candidates[i]
	}

	if len(candidates) > 3 {
		candidates = candidates[:3]
	}
	return candidates
}

func lineStartOffsets(text string) []int {
	offsets := []int{0}
	for i := 0; i < len(text); i++ {
		if text[i] == '\n' {
			offsets = append(offsets, i+1)
		}
	}
	if offsets[len(offsets)-1] != len(text) {
		offsets = append(offsets, len(text))
	}
	return offsets
}

// --- Diff generation ---

// generateDiff produces a unified diff with line numbers and context.
func generateDiff(oldContent, newContent string) (string, int) {
	const contextLines = 4

	oldLines := strings.Split(oldContent, "\n")
	newLines := strings.Split(newContent, "\n")

	// Find the first and last differing lines
	maxOld := len(oldLines)
	maxNew := len(newLines)

	// Find common prefix
	prefix := 0
	for prefix < maxOld && prefix < maxNew && oldLines[prefix] == newLines[prefix] {
		prefix++
	}

	// Find common suffix (from the end, not overlapping prefix)
	suffixOld := maxOld - 1
	suffixNew := maxNew - 1
	for suffixOld > prefix && suffixNew > prefix && oldLines[suffixOld] == newLines[suffixNew] {
		suffixOld--
		suffixNew--
	}

	firstChangedLine := prefix + 1 // 1-based

	if prefix > suffixOld+1 && prefix > suffixNew+1 {
		return "(no changes)", firstChangedLine
	}

	// Build diff output with context
	maxLineNum := max(maxOld, maxNew)
	lineNumWidth := len(fmt.Sprintf("%d", maxLineNum))

	var sb strings.Builder

	// Leading context
	ctxStart := max(prefix-contextLines, 0)
	if ctxStart < prefix {
		if ctxStart > 0 {
			fmt.Fprintf(&sb, " %*s ...\n", lineNumWidth, "")
		}
		for i := ctxStart; i < prefix; i++ {
			fmt.Fprintf(&sb, " %*d %s\n", lineNumWidth, i+1, oldLines[i])
		}
	}

	// Removed lines
	for i := prefix; i <= suffixOld; i++ {
		fmt.Fprintf(&sb, "-%*d %s\n", lineNumWidth, i+1, oldLines[i])
	}

	// Added lines
	for i := prefix; i <= suffixNew; i++ {
		fmt.Fprintf(&sb, "+%*d %s\n", lineNumWidth, i+1, newLines[i])
	}

	// Trailing context
	trailStart := suffixOld + 1
	trailEnd := min(trailStart+contextLines, maxOld)
	for i := trailStart; i < trailEnd; i++ {
		fmt.Fprintf(&sb, " %*d %s\n", lineNumWidth, i+1, oldLines[i])
	}
	if trailEnd < maxOld {
		fmt.Fprintf(&sb, " %*s ...\n", lineNumWidth, "")
	}

	return sb.String(), firstChangedLine
}

package tools

import (
	"bufio"
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"image"
	"image/jpeg"
	"image/png"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strings"

	_ "image/gif"

	"github.com/voocel/agentcore"
	"github.com/voocel/agentcore/schema"
	"golang.org/x/image/draw"
	_ "golang.org/x/image/webp"
)

// supportedImageMIME is the whitelist of image types we send to the LLM.
var supportedImageMIME = map[string]bool{
	"image/jpeg": true,
	"image/png":  true,
	"image/gif":  true,
	"image/webp": true,
}

const (
	readDefaultLimit = 2000
	readMaxLineLen   = 2000
)

// ReadTool reads file contents with optional offset and limit.
// Supports directory listings and image files. Text output is streamed and
// truncated by line count / byte size. Binary files are rejected.
type ReadTool struct {
	WorkDir string
}

func NewRead(workDir string) *ReadTool { return &ReadTool{WorkDir: workDir} }

func (t *ReadTool) Name() string                              { return "read" }
func (t *ReadTool) Label() string                              { return "Read File" }
func (t *ReadTool) ReadOnly(_ json.RawMessage) bool            { return true }
func (t *ReadTool) ConcurrencySafe(_ json.RawMessage) bool     { return true }
func (t *ReadTool) ActivityDescription(_ json.RawMessage) string { return "Reading file" }
func (t *ReadTool) Description() string {
	return fmt.Sprintf(
		"Read a file or directory from the local filesystem. Supports relative or absolute paths. Text output is streamed and truncated to %d lines or %s. Use offset to continue reading later sections, and avoid tiny repeated slices when you need broader context. Directory entries are returned one per line with a trailing '/' for subdirectories. Long lines are truncated. Use grep to find specific content in large files, and use glob if you are unsure of the path. Supports JPEG, PNG, GIF, and WebP images. Binary files are rejected.",
		defaultMaxLines, formatSize(defaultMaxBytes),
	)
}
func (t *ReadTool) Schema() map[string]any {
	return schema.Object(
		schema.Property("path", schema.String("Path to the file or directory to read (relative or absolute)")).Required(),
		schema.Property("offset", schema.Int("Line or entry number to start from (1-based, default: 1)")),
		schema.Property("limit", schema.Int("Maximum number of lines or directory entries to read (default: 2000)")),
	)
}

type readArgs struct {
	Path   string `json:"path"`
	Offset int    `json:"offset"`
	Limit  int    `json:"limit"`
}

type resolvedRead struct {
	path   string
	offset int
	limit  int
	info   os.FileInfo
}

// Execute returns a text-only result (for backward compatibility / middleware).
func (t *ReadTool) Execute(ctx context.Context, args json.RawMessage) (json.RawMessage, error) {
	a, err := t.parseArgs(args)
	if err != nil {
		return nil, err
	}

	result, err := t.readTextual(ctx, a)
	if err != nil {
		return nil, err
	}
	return json.Marshal(result)
}

// ExecuteContent returns rich content blocks (text or image).
// Implements agentcore.ContentTool.
func (t *ReadTool) ExecuteContent(ctx context.Context, args json.RawMessage) ([]agentcore.ContentBlock, error) {
	a, err := t.parseArgs(args)
	if err != nil {
		return nil, err
	}

	if !a.info.IsDir() {
		if mime := detectImageMIME(a.path); mime != "" {
			return t.readImage(a.path, mime)
		}
	}

	result, err := t.readTextual(ctx, a)
	if err != nil {
		return nil, err
	}
	return []agentcore.ContentBlock{agentcore.TextBlock(result)}, nil
}

func (t *ReadTool) parseArgs(args json.RawMessage) (resolvedRead, error) {
	var a readArgs
	if err := json.Unmarshal(args, &a); err != nil {
		return resolvedRead{}, fmt.Errorf("invalid args: %w", err)
	}
	if a.Offset < 0 {
		return resolvedRead{}, fmt.Errorf("offset must be greater than or equal to 1")
	}

	p := ResolvePath(t.WorkDir, a.Path)
	info, err := os.Stat(p)
	if err != nil {
		if os.IsNotExist(err) {
			return resolvedRead{}, fmt.Errorf("%s", notFoundWithSuggestions(p))
		}
		return resolvedRead{}, fmt.Errorf("read %s: %w", p, err)
	}

	offset := a.Offset
	if offset <= 0 {
		offset = 1
	}
	limit := a.Limit
	if limit <= 0 {
		limit = readDefaultLimit
	}

	return resolvedRead{
		path:   p,
		offset: offset,
		limit:  limit,
		info:   info,
	}, nil
}

func (t *ReadTool) readTextual(ctx context.Context, a resolvedRead) (string, error) {
	if a.info.IsDir() {
		return readDirectory(a)
	}

	if mime := detectImageMIME(a.path); mime != "" {
		return fmt.Sprintf("Read image file [%s]", mime), nil
	}

	isBinary, err := isBinaryFile(a.path, a.info.Size())
	if err != nil {
		return "", err
	}
	if isBinary {
		return "", fmt.Errorf("cannot read binary file: %s", a.path)
	}

	return readTextFile(ctx, a)
}

// readImage reads a file as an image, optionally resizes, and returns content blocks.
func (t *ReadTool) readImage(path, mime string) ([]agentcore.ContentBlock, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read %s: %w", path, err)
	}

	note := fmt.Sprintf("Read image file [%s] (%s)", mime, formatSize(len(data)))

	// Auto-resize large images to reduce token usage
	resized, resMIME, resNote := resizeImage(data, mime)
	if resNote != "" {
		data = resized
		mime = resMIME
		note += " " + resNote
	}

	encoded := base64.StdEncoding.EncodeToString(data)
	return []agentcore.ContentBlock{
		agentcore.TextBlock(note),
		agentcore.ImageBlock(encoded, mime),
	}, nil
}

const imageMaxDim = 2000

// resizeImage downscales an image if either dimension exceeds imageMaxDim.
// Returns original data unchanged if no resize is needed or on error.
func resizeImage(data []byte, mime string) ([]byte, string, string) {
	img, _, err := image.Decode(bytes.NewReader(data))
	if err != nil {
		return data, mime, ""
	}

	bounds := img.Bounds()
	w, h := bounds.Dx(), bounds.Dy()
	if w <= imageMaxDim && h <= imageMaxDim {
		return data, mime, ""
	}

	scale := float64(imageMaxDim) / float64(max(w, h))
	newW := int(float64(w) * scale)
	newH := int(float64(h) * scale)

	dst := image.NewRGBA(image.Rect(0, 0, newW, newH))
	draw.CatmullRom.Scale(dst, dst.Bounds(), img, bounds, draw.Over, nil)

	var jpegBuf bytes.Buffer
	if err := jpeg.Encode(&jpegBuf, dst, &jpeg.Options{Quality: 85}); err != nil {
		return data, mime, ""
	}

	var pngBuf bytes.Buffer
	if err := png.Encode(&pngBuf, dst); err == nil && pngBuf.Len() < jpegBuf.Len() {
		return pngBuf.Bytes(), "image/png", fmt.Sprintf("[Resized %dx%d → %dx%d]", w, h, newW, newH)
	}

	return jpegBuf.Bytes(), "image/jpeg", fmt.Sprintf("[Resized %dx%d → %dx%d]", w, h, newW, newH)
}

func readDirectory(a resolvedRead) (string, error) {
	entries, err := os.ReadDir(a.path)
	if err != nil {
		return "", fmt.Errorf("read directory %s: %w", a.path, err)
	}

	list := make([]string, 0, len(entries))
	for _, entry := range entries {
		name := entry.Name()
		if entry.IsDir() {
			name += "/"
		}
		list = append(list, name)
	}
	sort.Slice(list, func(i, j int) bool {
		return strings.ToLower(list[i]) < strings.ToLower(list[j])
	})
	if len(list) == 0 {
		return "(empty directory)", nil
	}

	start := a.offset - 1
	if start >= len(list) {
		return "", fmt.Errorf("offset %d is beyond end of directory listing (%d entries)", a.offset, len(list))
	}

	end := min(start+a.limit, len(list))
	slice := list[start:end]
	if len(slice) == 0 {
		return "(empty directory)", nil
	}

	result := strings.Join(slice, "\n")
	if end < len(list) {
		result += fmt.Sprintf("\n\n[Showing entries %d-%d of %d. Use offset=%d to continue.]", start+1, end, len(list), end+1)
	} else {
		result += fmt.Sprintf("\n\n[End of directory listing - total %d entries.]", len(list))
	}
	return result, nil
}

func readTextFile(ctx context.Context, a resolvedRead) (string, error) {
	f, err := os.Open(a.path)
	if err != nil {
		return "", fmt.Errorf("read %s: %w", a.path, err)
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 256*1024), 2*1024*1024)

	var sb strings.Builder
	written := 0
	readLines := 0
	totalLines := 0
	hasMore := false
	truncatedByBytes := false

	for scanner.Scan() {
		if ctx.Err() != nil {
			return "", ctx.Err()
		}
		totalLines++
		if totalLines < a.offset {
			continue
		}
		if readLines >= a.limit {
			hasMore = true
			continue
		}

		line := scanner.Text()
		if tl, truncated := truncateLine(line, readMaxLineLen); truncated {
			line = tl
		}
		rendered := fmt.Sprintf("%d\t%s\n", totalLines, line)
		if written+len(rendered) > defaultMaxBytes {
			if readLines == 0 {
				return fmt.Sprintf("[File %s: first line exceeds %s limit. Use offset/limit to read in chunks.]", a.path, formatSize(defaultMaxBytes)), nil
			}
			truncatedByBytes = true
			hasMore = true
			break
		}
		sb.WriteString(rendered)
		written += len(rendered)
		readLines++
	}
	if err := scanner.Err(); err != nil {
		return "", fmt.Errorf("scan %s: %w", a.path, err)
	}

	if totalLines == 0 {
		if a.offset > 1 {
			return "", fmt.Errorf("offset %d is beyond end of file (0 lines)", a.offset)
		}
		return "[End of file - total 0 lines.]", nil
	}
	if a.offset > totalLines {
		return "", fmt.Errorf("offset %d is beyond end of file (%d lines)", a.offset, totalLines)
	}

	result := strings.TrimRight(sb.String(), "\n")
	if truncatedByBytes || hasMore {
		lastRead := a.offset + readLines - 1
		nextOffset := lastRead + 1
		result += fmt.Sprintf("\n\n[Showing lines %d-%d of %d. Use offset=%d to continue.]", a.offset, lastRead, totalLines, nextOffset)
	} else if result != "" {
		result += fmt.Sprintf("\n\n[End of file - total %d lines.]", totalLines)
	}
	return result, nil
}

func notFoundWithSuggestions(target string) string {
	dir := filepath.Dir(target)
	base := filepath.Base(target)
	entries, err := os.ReadDir(dir)
	if err != nil {
		return fmt.Sprintf("file not found: %s", target)
	}

	var suggestions []string
	lowerBase := strings.ToLower(base)
	lowerStem := strings.ToLower(strings.TrimSuffix(base, filepath.Ext(base)))
	for _, entry := range entries {
		name := entry.Name()
		lowerName := strings.ToLower(name)
		lowerNameStem := strings.ToLower(strings.TrimSuffix(name, filepath.Ext(name)))
		if strings.Contains(lowerName, lowerBase) ||
			strings.Contains(lowerBase, lowerName) ||
			(lowerStem != "" && (strings.Contains(lowerNameStem, lowerStem) || strings.Contains(lowerStem, lowerNameStem))) {
			suggestions = append(suggestions, filepath.Join(dir, name))
			if len(suggestions) >= 3 {
				break
			}
		}
	}
	if len(suggestions) == 0 {
		return fmt.Sprintf("file not found: %s", target)
	}
	return fmt.Sprintf("file not found: %s\n\nDid you mean one of these?\n%s", target, strings.Join(suggestions, "\n"))
}

// detectImageMIME sniffs the file's content type and returns the MIME type
// if it's a supported image format, or "" otherwise.
func detectImageMIME(path string) string {
	f, err := os.Open(path)
	if err != nil {
		return ""
	}
	defer f.Close()

	buf := make([]byte, 512)
	n, err := f.Read(buf)
	if err != nil || n == 0 {
		return ""
	}

	mime := http.DetectContentType(buf[:n])
	if supportedImageMIME[mime] {
		return mime
	}
	return ""
}

func isBinaryFile(path string, size int64) (bool, error) {
	switch strings.ToLower(filepath.Ext(path)) {
	case ".zip", ".tar", ".gz", ".exe", ".dll", ".so", ".class", ".jar", ".war",
		".7z", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx", ".odt", ".ods",
		".odp", ".bin", ".dat", ".obj", ".o", ".a", ".lib", ".wasm", ".pyc", ".pyo", ".pdf":
		return true, nil
	}
	if size == 0 {
		return false, nil
	}

	f, err := os.Open(path)
	if err != nil {
		return false, fmt.Errorf("read %s: %w", path, err)
	}
	defer f.Close()

	sampleSize := min(int(size), 4096)
	buf := make([]byte, sampleSize)
	n, err := f.Read(buf)
	if err != nil {
		return false, fmt.Errorf("read %s: %w", path, err)
	}
	if n == 0 {
		return false, nil
	}

	nonPrintable := 0
	for i := 0; i < n; i++ {
		if buf[i] == 0 {
			return true, nil
		}
		if buf[i] < 9 || (buf[i] > 13 && buf[i] < 32) {
			nonPrintable++
		}
	}
	return float64(nonPrintable)/float64(n) > 0.3, nil
}

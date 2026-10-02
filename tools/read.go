package tools

import (
	"bufio"
	"bytes"
	"context"
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
	"time"

	_ "image/gif"

	"github.com/voocel/agentcore"
	"github.com/voocel/agentcore/schema"
	"github.com/voocel/litellm"
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

// Read returns the read tool: it reads a file, from an offset and up to a
// limit, lists a directory, or reads an image. Text is cut at a line count
// and a byte size; binary files are refused. What it read goes to Files.
func (w Workspace) Read() agentcore.Tool {
	t := &readTool{w: w, fs: w.fs()}
	return agentcore.Tool{
		Name:        "read",
		Label:       "Read File",
		Description: readDescription(),
		Schema: schema.Object(
			schema.Property("file_path", schema.String("The path to the file or directory to read (relative or absolute)")).Required(),
			schema.Property("offset", schema.Int("The line number to start reading from. Only provide if the file is too large to read at once")),
			schema.Property("limit", schema.Int("The number of lines to read. Only provide if the file is too large to read at once")),
		),
		Parallel: always,
		Run:      t.run,
	}
}

type readTool struct {
	w  Workspace
	fs FS
}

func readDescription() string {
	return fmt.Sprintf(
		`Reads a file from the local filesystem. You can access any file directly by using this tool.

Usage:
- The file_path parameter accepts relative or absolute paths.
- By default, reads up to %d lines starting from the beginning of the file.
- You can optionally specify a line offset and limit (especially handy for long files), but it's recommended to read the whole file by not providing these parameters.
- Results are returned using cat -n format, with line numbers starting at 1.
- This tool also lists directory contents when file_path points to a directory; entries are returned one per line with a trailing '/' for subdirectories.
- Long lines are truncated. Output is capped at %d lines or %s (whichever is hit first).
- Use grep to find specific content in large files, and glob if you are unsure of the path.
- Supports JPEG, PNG, GIF, and WebP images. Binary files are rejected.`,
		defaultMaxLines, defaultMaxLines, formatSize(defaultMaxBytes),
	)
}

type readArgs struct {
	FilePath string `json:"file_path"`
	Offset   int    `json:"offset"`
	Limit    int    `json:"limit"`
}

type resolvedRead struct {
	path   string
	offset int
	limit  int
	info   FileInfo
}

// run reads the file, directory or image args names.
func (t *readTool) run(ctx context.Context, args json.RawMessage) (agentcore.Result, error) {
	a, err := t.parseArgs(ctx, args)
	if err != nil {
		return agentcore.Result{}, err
	}

	if !a.info.IsDir {
		if mime := t.detectImageMIME(ctx, a.path); mime != "" {
			blocks, err := t.readImage(ctx, a.path, mime)
			if err != nil {
				return agentcore.Result{}, err
			}
			t.recordRead(a, false)
			return agentcore.Result{Content: blocks}, nil
		}
	}

	result, partial, err := t.readTextual(ctx, a)
	if err != nil {
		return agentcore.Result{}, err
	}
	t.recordRead(a, partial)
	return agentcore.TextResult(result), nil
}

func (t *readTool) parseArgs(ctx context.Context, args json.RawMessage) (resolvedRead, error) {
	var a readArgs
	if err := json.Unmarshal(args, &a); err != nil {
		return resolvedRead{}, fmt.Errorf("invalid args: %w", err)
	}
	if a.Offset < 0 {
		return resolvedRead{}, fmt.Errorf("offset must be greater than or equal to 1")
	}

	p := ResolvePath(t.w.dir(ctx), a.FilePath)
	info, err := t.fs.Stat(ctx, p)
	if err != nil {
		if os.IsNotExist(err) {
			return resolvedRead{}, fmt.Errorf("%s", t.notFoundWithSuggestions(ctx, p))
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

	return resolvedRead{path: p, offset: offset, limit: limit, info: info}, nil
}

// recordRead records the read of a file in Files, partial when the model
// did not see all of it. Files are keyed by absolute path, as write and edit
// resolve them.
func (t *readTool) recordRead(a resolvedRead, partial bool) {
	if t.w.Files == nil || a.info.IsDir {
		return
	}
	t.w.Files.Set(a.path, FileReadStamp{
		ReadAt:  time.Now(),
		Mtime:   a.info.ModTime,
		Version: a.info.Version,
		Partial: partial,
	})
}

// readTextual reads a directory or a text file; partial reports a file read
// in part.
func (t *readTool) readTextual(ctx context.Context, a resolvedRead) (text string, partial bool, err error) {
	if a.info.IsDir {
		text, err := t.readDirectory(ctx, a)
		return text, false, err
	}
	isBinary, err := t.isBinaryFile(ctx, a.path, a.info.Size)
	if err != nil {
		return "", false, err
	}
	if isBinary {
		return "", false, fmt.Errorf("cannot read binary file: %s", a.path)
	}
	return t.readTextFile(ctx, a)
}

// readImage reads a file as an image, optionally resizes, and returns content blocks.
func (t *readTool) readImage(ctx context.Context, path, mime string) ([]litellm.Block, error) {
	data, err := t.fs.ReadFile(ctx, path)
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

	return []litellm.Block{
		litellm.Text(note),
		litellm.ImageBlock{Data: data, MIME: mime},
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

func (t *readTool) readDirectory(ctx context.Context, a resolvedRead) (string, error) {
	entries, err := t.fs.ReadDir(ctx, a.path)
	if err != nil {
		return "", fmt.Errorf("read directory %s: %w", a.path, err)
	}

	list := make([]string, 0, len(entries))
	for _, entry := range entries {
		name := entry.Name
		if entry.IsDir {
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

// readTextFile reads the lines of a file a asks for; partial reports that
// the model did not see all of it.
func (t *readTool) readTextFile(ctx context.Context, a resolvedRead) (string, bool, error) {
	f, err := t.fs.Open(ctx, a.path)
	if err != nil {
		return "", false, fmt.Errorf("read %s: %w", a.path, err)
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
			return "", false, ctx.Err()
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
				return fmt.Sprintf("[File %s: first line exceeds %s limit. Use offset/limit to read in chunks.]", a.path, formatSize(defaultMaxBytes)), true, nil
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
		return "", false, fmt.Errorf("scan %s: %w", a.path, err)
	}

	if totalLines == 0 {
		if a.offset > 1 {
			return "", false, fmt.Errorf("offset %d is beyond end of file (0 lines)", a.offset)
		}
		return "[End of file - total 0 lines.]", false, nil
	}
	if a.offset > totalLines {
		return "", false, fmt.Errorf("offset %d is beyond end of file (%d lines)", a.offset, totalLines)
	}

	result := strings.TrimRight(sb.String(), "\n")
	if truncatedByBytes || hasMore {
		lastRead := a.offset + readLines - 1
		nextOffset := lastRead + 1
		result += fmt.Sprintf("\n\n[Showing lines %d-%d of %d. Use offset=%d to continue.]", a.offset, lastRead, totalLines, nextOffset)
	} else if result != "" {
		result += fmt.Sprintf("\n\n[End of file - total %d lines.]", totalLines)
	}
	return result, a.offset > 1 || hasMore, nil
}

func (t *readTool) notFoundWithSuggestions(ctx context.Context, target string) string {
	dir := dirOf(target)
	base := filepath.Base(target)
	entries, err := t.fs.ReadDir(ctx, dir)
	if err != nil {
		return fmt.Sprintf("file not found: %s", target)
	}

	var suggestions []string
	lowerBase := strings.ToLower(base)
	lowerStem := strings.ToLower(strings.TrimSuffix(base, filepath.Ext(base)))
	for _, entry := range entries {
		name := entry.Name
		lowerName := strings.ToLower(name)
		lowerNameStem := strings.ToLower(strings.TrimSuffix(name, filepath.Ext(name)))
		if strings.Contains(lowerName, lowerBase) ||
			strings.Contains(lowerBase, lowerName) ||
			(lowerStem != "" && (strings.Contains(lowerNameStem, lowerStem) || strings.Contains(lowerStem, lowerNameStem))) {
			suggestions = append(suggestions, joinOf(dir, name))
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
func (t *readTool) detectImageMIME(ctx context.Context, path string) string {
	f, err := t.fs.Open(ctx, path)
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

func (t *readTool) isBinaryFile(ctx context.Context, path string, size int64) (bool, error) {
	switch strings.ToLower(filepath.Ext(path)) {
	case ".zip", ".tar", ".gz", ".exe", ".dll", ".so", ".class", ".jar", ".war",
		".7z", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx", ".odt", ".ods",
		".odp", ".bin", ".dat", ".obj", ".o", ".a", ".lib", ".wasm", ".pyc", ".pyo", ".pdf":
		return true, nil
	}
	if size == 0 {
		return false, nil
	}

	f, err := t.fs.Open(ctx, path)
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

package tools

import (
	"bytes"
	"context"
	"encoding/binary"
	"hash/crc32"
	"image"
	"image/png"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// A line of any length reads, cut, and a file with one cut was read only in
// part, so a write may not replace it wholesale.
func TestReadCutsLongLines(t *testing.T) {
	dir, read, write, _ := fileTools(t)
	path := filepath.Join(dir, "min.js")
	os.WriteFile(path, []byte("first\n"+strings.Repeat("x", 3<<20)+"\r\nlast\r"), 0o644)

	res, err := read.Run(context.Background(), mustJSON(t, readArgs{FilePath: path}))
	if err != nil {
		t.Fatal(err)
	}
	text := res.Text()
	if !strings.Contains(text, "1\tfirst\n2\txxx") || !strings.Contains(text, "... [truncated]\n3\tlast\n") || strings.Contains(text, "\r") {
		t.Fatalf("read %.200q", text)
	}
	if err := call(t, write, writeArgs{FilePath: path, Content: "c\n"}); err == nil || !strings.Contains(err.Error(), "Only part") {
		t.Fatalf("a read with a cut line counted as whole: %v", err)
	}
	if res, err := read.Run(context.Background(), mustJSON(t, readArgs{FilePath: path, Offset: 3})); err != nil || !strings.Contains(res.Text(), "3\tlast") {
		t.Fatalf("read past the long line: %v, %v", res, err)
	}
}

// An image too large to decode is refused, rather than taking gigabytes, or
// sent, to fail every request after it.
func TestReadRefusesHugeImages(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "huge.png")
	os.WriteFile(path, pngClaiming(t, 16000, 16000), 0o644)
	_, err := Workspace{Dir: dir}.Read().Run(context.Background(), mustJSON(t, readArgs{FilePath: path}))
	if err == nil || !strings.Contains(err.Error(), "too large") {
		t.Fatalf("err = %v", err)
	}
}

// pngClaiming returns a PNG whose header claims w by h, which is all
// decoding the size reads.
func pngClaiming(t *testing.T, w, h uint32) []byte {
	t.Helper()
	var buf bytes.Buffer
	if err := png.Encode(&buf, image.NewGray(image.Rect(0, 0, 1, 1))); err != nil {
		t.Fatal(err)
	}
	b := buf.Bytes()
	// The IHDR chunk follows the 8-byte signature: length, type, data, CRC.
	ihdr := b[8+8 : 8+8+13]
	binary.BigEndian.PutUint32(ihdr[0:], w)
	binary.BigEndian.PutUint32(ihdr[4:], h)
	binary.BigEndian.PutUint32(b[8+8+13:], crc32.ChecksumIEEE(b[8+4:8+8+13]))
	return b
}

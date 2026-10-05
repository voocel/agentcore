package tools

import "testing"

func TestWritePreviewCountsLinesAsDiffsDo(t *testing.T) {
	for _, c := range []struct {
		content string
		max     int
		want    string
	}{
		{"package main\n\nfunc main() {}\n", 12, "+1 package main\n+2 \n+3 func main() {}\n"},
		{"no newline", 12, "+1 no newline\n"},
		{"", 12, ""},
		{"a\nb\nc\n", 2, "+1 a\n+2 b\n   ... +1 more lines\n"},
	} {
		if got := writePreview(c.content, c.max); got != c.want {
			t.Errorf("writePreview(%q, %d) = %q, want %q", c.content, c.max, got, c.want)
		}
	}
}

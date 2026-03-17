package tools

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestEditFuzzyMatchTrailingUnicodeSpace(t *testing.T) {
	t.Parallel()

	dir := t.TempDir()
	path := filepath.Join(dir, "test.txt")
	if err := os.WriteFile(path, []byte("line\u00A0\nnext\n"), 0o644); err != nil {
		t.Fatalf("write fixture: %v", err)
	}

	tool := NewEdit(dir)
	args, err := json.Marshal(map[string]any{
		"path":     "test.txt",
		"old_text": "line\n",
		"new_text": "repl\n",
	})
	if err != nil {
		t.Fatalf("marshal args: %v", err)
	}

	if _, err := tool.Execute(context.Background(), args); err != nil {
		t.Fatalf("execute edit: %v", err)
	}

	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read result: %v", err)
	}
	if string(got) != "repl\nnext\n" {
		t.Fatalf("unexpected content:\nwant %q\ngot  %q", "repl\nnext\n", string(got))
	}
}

func TestEditFuzzyDoesNotChangeUnrelatedSameLineText(t *testing.T) {
	t.Parallel()

	dir := t.TempDir()
	path := filepath.Join(dir, "test.txt")
	input := "note=\"“保留”\"; target=‘old’\n"
	if err := os.WriteFile(path, []byte(input), 0o644); err != nil {
		t.Fatalf("write fixture: %v", err)
	}

	tool := NewEdit(dir)
	args, err := json.Marshal(map[string]any{
		"path":     "test.txt",
		"old_text": "target='old'",
		"new_text": "target='new'",
	})
	if err != nil {
		t.Fatalf("marshal args: %v", err)
	}

	if _, err := tool.Execute(context.Background(), args); err != nil {
		t.Fatalf("execute edit: %v", err)
	}

	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read result: %v", err)
	}
	want := "note=\"“保留”\"; target='new'\n"
	if string(got) != want {
		t.Fatalf("unexpected content:\nwant %q\ngot  %q", want, string(got))
	}
}

func TestEditPreviewFuzzyNoMutation(t *testing.T) {
	t.Parallel()

	dir := t.TempDir()
	path := filepath.Join(dir, "test.txt")
	input := "note=\"“保留”\"; target=‘old’\n"
	if err := os.WriteFile(path, []byte(input), 0o644); err != nil {
		t.Fatalf("write fixture: %v", err)
	}

	tool := NewEdit(dir)
	args, err := json.Marshal(map[string]any{
		"path":     "test.txt",
		"old_text": "target='old'",
		"new_text": "target='new'",
	})
	if err != nil {
		t.Fatalf("marshal args: %v", err)
	}

	preview, err := tool.Preview(context.Background(), args)
	if err != nil {
		t.Fatalf("preview edit: %v", err)
	}

	var payload struct {
		Diff string `json:"diff"`
	}
	if err := json.Unmarshal(preview, &payload); err != nil {
		t.Fatalf("unmarshal preview: %v", err)
	}
	if !strings.Contains(payload.Diff, "“保留”") {
		t.Fatalf("preview diff unexpectedly normalized unrelated text: %q", payload.Diff)
	}

	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read result: %v", err)
	}
	if string(got) != input {
		t.Fatalf("preview mutated file:\nwant %q\ngot  %q", input, string(got))
	}
}

func TestEditIndentAwareMatch(t *testing.T) {
	t.Parallel()

	dir := t.TempDir()
	path := filepath.Join(dir, "test.txt")
	input := "func main() {\n\tif true {\n\t\tprintln(\"old\")\n\t}\n}\n"
	if err := os.WriteFile(path, []byte(input), 0o644); err != nil {
		t.Fatalf("write fixture: %v", err)
	}

	tool := NewEdit(dir)
	args, err := json.Marshal(map[string]any{
		"path": "test.txt",
		"old_text": "if true {\n\tprintln(\"old\")\n}",
		"new_text": "if true {\n\tprintln(\"new\")\n}",
	})
	if err != nil {
		t.Fatalf("marshal args: %v", err)
	}

	if _, err := tool.Execute(context.Background(), args); err != nil {
		t.Fatalf("execute edit: %v", err)
	}

	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read result: %v", err)
	}
	want := "func main() {\n\tif true {\n\t\tprintln(\"new\")\n\t}\n}\n"
	if string(got) != want {
		t.Fatalf("unexpected content:\nwant %q\ngot  %q", want, string(got))
	}
}

func TestEditIndentAwareRequiresUniqueMatch(t *testing.T) {
	t.Parallel()

	dir := t.TempDir()
	path := filepath.Join(dir, "test.txt")
	input := "func a() {\n\tif true {\n\t\tprintln(\"old\")\n\t}\n}\n\nfunc b() {\n\tif true {\n\t\tprintln(\"old\")\n\t}\n}\n"
	if err := os.WriteFile(path, []byte(input), 0o644); err != nil {
		t.Fatalf("write fixture: %v", err)
	}

	tool := NewEdit(dir)
	args, err := json.Marshal(map[string]any{
		"path": "test.txt",
		"old_text": "if true {\n\tprintln(\"old\")\n}",
		"new_text": "if true {\n\tprintln(\"new\")\n}",
	})
	if err != nil {
		t.Fatalf("marshal args: %v", err)
	}

	if _, err := tool.Execute(context.Background(), args); err == nil {
		t.Fatalf("expected ambiguity error")
	}
}

func TestEditFailureIncludesClosestMatchHint(t *testing.T) {
	t.Parallel()

	dir := t.TempDir()
	path := filepath.Join(dir, "test.txt")
	input := "func main() {\n\tif enabled {\n\t\tprintln(\"old\")\n\t}\n}\n"
	if err := os.WriteFile(path, []byte(input), 0o644); err != nil {
		t.Fatalf("write fixture: %v", err)
	}

	tool := NewEdit(dir)
	args, err := json.Marshal(map[string]any{
		"path": "test.txt",
		"old_text": "if true {\n\tprintln(\"old\")\n}",
		"new_text": "if true {\n\tprintln(\"new\")\n}",
	})
	if err != nil {
		t.Fatalf("marshal args: %v", err)
	}

	_, err = tool.Execute(context.Background(), args)
	if err == nil {
		t.Fatalf("expected edit error")
	}
	msg := err.Error()
	if !strings.Contains(msg, "Possible old_text candidates (copy one exactly):") {
		t.Fatalf("expected closest match hint, got %q", msg)
	}
	if !strings.Contains(msg, "lines 2-4") {
		t.Fatalf("expected line range hint, got %q", msg)
	}
	if !strings.Contains(msg, "```text") {
		t.Fatalf("expected fenced code block hint, got %q", msg)
	}
	if !strings.Contains(msg, "if enabled {") {
		t.Fatalf("expected candidate snippet, got %q", msg)
	}
}

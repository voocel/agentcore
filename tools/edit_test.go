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

	tool := Workspace{Dir: dir}.Edit()
	args, err := json.Marshal(map[string]any{
		"file_path":  "test.txt",
		"old_string": "line\n",
		"new_string": "repl\n",
	})
	if err != nil {
		t.Fatalf("marshal args: %v", err)
	}

	if _, err := tool.Run(context.Background(), args); err != nil {
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

	tool := Workspace{Dir: dir}.Edit()
	args, err := json.Marshal(map[string]any{
		"file_path":  "test.txt",
		"old_string": "target='old'",
		"new_string": "target='new'",
	})
	if err != nil {
		t.Fatalf("marshal args: %v", err)
	}

	if _, err := tool.Run(context.Background(), args); err != nil {
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

	tool := Workspace{Dir: dir}.Edit()
	args, err := json.Marshal(map[string]any{
		"file_path":  "test.txt",
		"old_string": "target='old'",
		"new_string": "target='new'",
	})
	if err != nil {
		t.Fatalf("marshal args: %v", err)
	}

	preview, err := tool.Check(context.Background(), args)
	if err != nil {
		t.Fatalf("preview edit: %v", err)
	}

	if !strings.Contains(preview, "“保留”") {
		t.Fatalf("preview diff unexpectedly normalized unrelated text: %q", preview)
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

	tool := Workspace{Dir: dir}.Edit()
	args, err := json.Marshal(map[string]any{
		"file_path":  "test.txt",
		"old_string": "if true {\n\tprintln(\"old\")\n}",
		"new_string": "if true {\n\tprintln(\"new\")\n}",
	})
	if err != nil {
		t.Fatalf("marshal args: %v", err)
	}

	if _, err := tool.Run(context.Background(), args); err != nil {
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

func TestIndentAwareFindWholeLines(t *testing.T) {
	t.Parallel()

	cases := []struct{ content, old, want string }{
		{"class A:\n    def f(self):\n        return 1\n\nx = 1\n", "def f(self):\n    return 1\n", "    def f(self):\n        return 1\n"},
		{"class A:\n    def f(self):\n        return 1\n\nx = 1\n", "def f(self):\n    return 1", "    def f(self):\n        return 1\n"},
		{"class A:\n    def f(self):\n        return 1\n", "def f(self):\n    return 1\n", "    def f(self):\n        return 1\n"},
		{"class A:\n    def f(self):\n        return 1", "def f(self):\n    return 1\n", "    def f(self):\n        return 1"},
	}
	for _, c := range cases {
		matches := indentAwareMatches(c.content, c.old)
		if len(matches) != 1 {
			t.Errorf("%q in %q: %d matches", c.old, c.content, len(matches))
			continue
		}
		if got := c.content[matches[0].start:matches[0].end]; got != c.want {
			t.Errorf("%q in %q: matched %q, want %q", c.old, c.content, got, c.want)
		}
	}
}

// Every matching tier reports several matches as ambiguous, and with
// replace_all replaces them all, each as it matched: the indentation-
// insensitive tier reindents the replacement to each match.
func TestEditSeveralMatches(t *testing.T) {
	t.Parallel()

	cases := []struct{ name, content, old, new, want string }{
		{"exact", "x = 1\ny = x\n", "x", "z", "z = 1\ny = z\n"},
		{"fuzzy", "foo  \nbar\nfoo\t\nbar", "foo\nbar", "baz", "baz\nbaz"},
		{
			"indentation-insensitive",
			"func a() {\n\tif true {\n\t\tprintln(\"old\")\n\t}\n}\n\nfunc b() {\n\tfor {\n\t\tif true {\n\t\t\tprintln(\"old\")\n\t\t}\n\t}\n}\n",
			"if true {\n\tprintln(\"old\")\n}",
			"if true {\n\tprintln(\"new\")\n}",
			"func a() {\n\tif true {\n\t\tprintln(\"new\")\n\t}\n}\n\nfunc b() {\n\tfor {\n\t\tif true {\n\t\t\tprintln(\"new\")\n\t\t}\n\t}\n}\n",
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			dir := t.TempDir()
			path := filepath.Join(dir, "f.txt")
			if err := os.WriteFile(path, []byte(c.content), 0o644); err != nil {
				t.Fatal(err)
			}
			tool := Workspace{Dir: dir}.Edit()
			args := editArgs{FilePath: "f.txt", OldString: c.old, NewString: c.new}
			if _, err := tool.Run(context.Background(), mustJSON(t, args)); err == nil || !strings.Contains(err.Error(), "found 2 occurrences") {
				t.Fatalf("ambiguous edit: %v", err)
			}
			if got, _ := os.ReadFile(path); string(got) != c.content {
				t.Fatalf("ambiguous edit changed the file to %q", got)
			}
			args.ReplaceAll = true
			if _, err := tool.Run(context.Background(), mustJSON(t, args)); err != nil {
				t.Fatal(err)
			}
			if got, _ := os.ReadFile(path); string(got) != c.want {
				t.Fatalf("replace_all:\nwant %q\ngot  %q", c.want, got)
			}
		})
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

	tool := Workspace{Dir: dir}.Edit()
	args, err := json.Marshal(map[string]any{
		"file_path":  "test.txt",
		"old_string": "if true {\n\tprintln(\"old\")\n}",
		"new_string": "if true {\n\tprintln(\"new\")\n}",
	})
	if err != nil {
		t.Fatalf("marshal args: %v", err)
	}

	_, err = tool.Run(context.Background(), args)
	if err == nil {
		t.Fatalf("expected edit error")
	}
	msg := err.Error()
	if !strings.Contains(msg, "Possible old_string candidates (copy one exactly):") {
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

func TestEditFailureIncludesClosestChineseMatchHint(t *testing.T) {
	t.Parallel()

	dir := t.TempDir()
	path := filepath.Join(dir, "chapter.md")
	input := "她没有立刻回答，只是看向窗外。\n她说这话的时候语气很平，没有愤怒，也没有嘲讽，只是在陈述事实。\n沈渡没有再问。\n"
	if err := os.WriteFile(path, []byte(input), 0o644); err != nil {
		t.Fatalf("write fixture: %v", err)
	}

	tool := Workspace{Dir: dir}.Edit()
	args, err := json.Marshal(map[string]any{
		"file_path":  "chapter.md",
		"old_string": "她说这话的时候，语气很平，没有愤怒，没有嘲讽，只是在陈述。",
		"new_string": "她的语气依然平静。",
	})
	if err != nil {
		t.Fatalf("marshal args: %v", err)
	}

	_, err = tool.Run(context.Background(), args)
	if err == nil {
		t.Fatalf("expected edit error")
	}
	msg := err.Error()
	if !strings.Contains(msg, "Possible old_string candidates (copy one exactly):") {
		t.Fatalf("expected closest match hint, got %q", msg)
	}
	if !strings.Contains(msg, "她说这话的时候语气很平，没有愤怒，也没有嘲讽，只是在陈述事实。") {
		t.Fatalf("expected exact Chinese candidate, got %q", msg)
	}

	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read result: %v", err)
	}
	if string(got) != input {
		t.Fatalf("failed edit mutated file:\nwant %q\ngot  %q", input, string(got))
	}
}

func TestEditFailureOmitsUnrelatedChineseCandidate(t *testing.T) {
	t.Parallel()

	dir := t.TempDir()
	path := filepath.Join(dir, "chapter.md")
	if err := os.WriteFile(path, []byte("雨落在青石板上。\n远处传来晚钟。\n"), 0o644); err != nil {
		t.Fatalf("write fixture: %v", err)
	}

	tool := Workspace{Dir: dir}.Edit()
	args, err := json.Marshal(map[string]any{
		"file_path":  "chapter.md",
		"old_string": "实验数据已经完成全部校验。",
		"new_string": "实验结束。",
	})
	if err != nil {
		t.Fatalf("marshal args: %v", err)
	}

	_, err = tool.Run(context.Background(), args)
	if err == nil {
		t.Fatalf("expected edit error")
	}
	if strings.Contains(err.Error(), "Possible old_string candidates") {
		t.Fatalf("unexpected unrelated candidate hint: %q", err)
	}
}

// An edit reports the file and the diff it applied, as plain text.
func TestEditResult(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "a.go")
	if err := os.WriteFile(path, []byte("x := a < b && c > d\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	args := mustJSON(t, editArgs{FilePath: "a.go", OldString: "a < b", NewString: "a <= b"})
	res, err := Workspace{Dir: dir}.Edit().Run(context.Background(), args)
	if err != nil {
		t.Fatal(err)
	}
	want := "Edited " + path + ".\n-1 x := a < b && c > d\n+1 x := a <= b && c > d\n"
	if res.Text() != want {
		t.Fatalf("result %q, want %q", res.Text(), want)
	}
}

// Edits that matching could only guess at fail rather than change what the
// model never quoted.
func TestEditRefusesGuesses(t *testing.T) {
	cases := []struct {
		name, content, old, new string
	}{
		{"empty old_string", "abc\n", "", "X"},
		{"escaped newline", "a\nb\n", `a\nb`, `a\nc`},
		{"anchors only", "func f() {\n\tkeep1()\n\tkeep2()\n\tkeep3()\n}\n", "func f() {\n\tother()\n}", "func f() {\n}"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			dir := t.TempDir()
			path := filepath.Join(dir, "f.txt")
			if err := os.WriteFile(path, []byte(c.content), 0o644); err != nil {
				t.Fatal(err)
			}
			args := mustJSON(t, editArgs{FilePath: "f.txt", OldString: c.old, NewString: c.new})
			if _, err := (Workspace{Dir: dir}).Edit().Run(context.Background(), args); err == nil {
				t.Fatal("the edit was applied")
			}
			if got, _ := os.ReadFile(path); string(got) != c.content {
				t.Fatalf("file changed to %q", got)
			}
		})
	}
}

func TestGenerateDiff(t *testing.T) {
	cases := []struct{ name, old, new, want string }{
		{"change", "a\nb\nc\n", "a\nX\nc\n", " 1 a\n-2 b\n+2 X\n 3 c\n"},
		{"insert", "a\nc\n", "a\nb\nc\n", " 1 a\n+2 b\n 2 c\n"},
		{"delete last", "a\nb\n", "a\n", " 1 a\n-2 b\n"},
		{"final newline", "a\nb", "a\nb\n", " 1 a\n-2 b\n+2 b\n"},
		{"same", "a\n", "a\n", "(no changes)"},
	}
	for _, c := range cases {
		if got := generateDiff(c.old, c.new); got != c.want {
			t.Errorf("%s: got %q, want %q", c.name, got, c.want)
		}
	}
}

package tools

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/voocel/agentcore"
)

// call checks and runs a tool as the loop does, returning the check's or the
// run's error.
func call(t *testing.T, tool agentcore.Tool, args any) error {
	t.Helper()
	raw := mustJSON(t, args)
	if tool.Check != nil {
		if _, err := tool.Check(context.Background(), raw); err != nil {
			return err
		}
	}
	_, err := tool.Run(context.Background(), raw)
	return err
}

func fileTools(t *testing.T) (dir string, read, write, edit agentcore.Tool) {
	dir = t.TempDir()
	w := Workspace{Dir: dir, Files: NewFileReadState()}
	return dir, w.Read(), w.Write(), w.Edit()
}

// The agent keeps changing a file it just changed, without reading it again.
func TestWritesAndEditsKeepTheFileRead(t *testing.T) {
	dir, read, write, edit := fileTools(t)
	path := filepath.Join(dir, "a.txt")
	if err := os.WriteFile(path, []byte("one\ntwo\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	if err := call(t, read, readArgs{FilePath: path}); err != nil {
		t.Fatal(err)
	}
	for _, step := range []editArgs{
		{FilePath: path, OldString: "one", NewString: "uno"},
		{FilePath: path, OldString: "two", NewString: "dos"},
	} {
		if err := call(t, edit, step); err != nil {
			t.Fatalf("edit %q: %v", step.OldString, err)
		}
	}
	if err := call(t, write, writeArgs{FilePath: path, Content: "tres\n"}); err != nil {
		t.Fatalf("write after edits: %v", err)
	}

	created := filepath.Join(dir, "new.txt")
	if err := call(t, write, writeArgs{FilePath: created, Content: "fresh\n"}); err != nil {
		t.Fatal(err)
	}
	if err := call(t, edit, editArgs{FilePath: created, OldString: "fresh", NewString: "edited"}); err != nil {
		t.Fatalf("edit of a file just written: %v", err)
	}
}

// A slice is enough to edit, not to overwrite, and an edit does not turn it
// into a whole read.
func TestPartialReadAllowsEditsOnly(t *testing.T) {
	dir, read, write, edit := fileTools(t)
	path := filepath.Join(dir, "a.txt")
	if err := os.WriteFile(path, []byte("one\ntwo\nthree\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	if err := call(t, read, readArgs{FilePath: path, Offset: 2, Limit: 1}); err != nil {
		t.Fatal(err)
	}
	if err := call(t, edit, editArgs{FilePath: path, OldString: "two", NewString: "dos"}); err != nil {
		t.Fatalf("edit after a partial read: %v", err)
	}
	err := call(t, write, writeArgs{FilePath: path, Content: "x"})
	if err == nil || !strings.Contains(err.Error(), "Only part of the file has been read") {
		t.Fatalf("write after a partial read: %v", err)
	}
}

// Someone else changing the file still invalidates the agent's read.
func TestOutsideChangesStillNeedARead(t *testing.T) {
	dir, read, _, edit := fileTools(t)
	path := filepath.Join(dir, "a.txt")
	if err := os.WriteFile(path, []byte("one\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := call(t, read, readArgs{FilePath: path}); err != nil {
		t.Fatal(err)
	}
	if err := call(t, edit, editArgs{FilePath: path, OldString: "one", NewString: "uno"}); err != nil {
		t.Fatal(err)
	}
	// A later mtime, as a formatter rewriting the file would leave.
	later := mtimeOf(t, path).Add(2 * time.Second)
	if err := os.Chtimes(path, later, later); err != nil {
		t.Fatal(err)
	}
	err := call(t, edit, editArgs{FilePath: path, OldString: "uno", NewString: "one"})
	if err == nil || !strings.Contains(err.Error(), "modified since read") {
		t.Fatalf("edit after an outside change: %v", err)
	}
}

// A file changed while a checked call awaited approval is not overwritten.
func TestRunChecksTheFileAgain(t *testing.T) {
	dir, read, write, edit := fileTools(t)
	path := filepath.Join(dir, "a.txt")
	for _, c := range []struct {
		tool agentcore.Tool
		args any
	}{
		{edit, editArgs{FilePath: path, OldString: "one", NewString: "uno"}},
		{write, writeArgs{FilePath: path, Content: "uno\n"}},
	} {
		if err := os.WriteFile(path, []byte("one\n"), 0o644); err != nil {
			t.Fatal(err)
		}
		if err := call(t, read, readArgs{FilePath: path}); err != nil {
			t.Fatal(err)
		}
		raw := mustJSON(t, c.args)
		if _, err := c.tool.Check(context.Background(), raw); err != nil {
			t.Fatal(err)
		}
		later := mtimeOf(t, path).Add(2 * time.Second)
		if err := os.WriteFile(path, []byte("one, edited by the user\n"), 0o644); err != nil {
			t.Fatal(err)
		}
		if err := os.Chtimes(path, later, later); err != nil {
			t.Fatal(err)
		}
		_, err := c.tool.Run(context.Background(), raw)
		if err == nil || !strings.Contains(err.Error(), "modified since read") {
			t.Fatalf("%s after an outside change: %v", c.tool.Name, err)
		}
		if got, _ := os.ReadFile(path); string(got) != "one, edited by the user\n" {
			t.Fatalf("%s overwrote the user's change: %q", c.tool.Name, got)
		}
	}
}

func mtimeOf(t *testing.T, path string) time.Time {
	t.Helper()
	info, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	return info.ModTime()
}

// A read counts as whole by what the model saw, not by the arguments.
func TestPartialReadIsWhatWasSeen(t *testing.T) {
	dir, read, write, _ := fileTools(t)
	short := filepath.Join(dir, "short.txt")
	long := filepath.Join(dir, "long.txt")
	os.WriteFile(short, []byte("a\nb\n"), 0o644)
	os.WriteFile(long, []byte(strings.Repeat("line\n", readDefaultLimit+10)), 0o644)

	if err := call(t, read, readArgs{FilePath: short, Limit: 100}); err != nil {
		t.Fatal(err)
	}
	if err := call(t, write, writeArgs{FilePath: short, Content: "c\n"}); err != nil {
		t.Fatalf("a limit past the end read the whole file: %v", err)
	}
	if err := call(t, read, readArgs{FilePath: long}); err != nil {
		t.Fatal(err)
	}
	if err := call(t, write, writeArgs{FilePath: long, Content: "c\n"}); err == nil || !strings.Contains(err.Error(), "Only part") {
		t.Fatalf("a read cut at the default limit counted as whole: %v", err)
	}
}

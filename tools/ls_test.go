package tools

import (
	"context"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func ls(t *testing.T, dir string, args lsArgs) string {
	t.Helper()
	res, err := Workspace{Dir: dir}.Ls().Run(context.Background(), mustJSON(t, args))
	if err != nil {
		t.Fatal(err)
	}
	return res.Text()
}

// A directory a build often makes is named, not hidden: a project's own may
// share the name. What the call asks to ignore is hidden.
func TestLsNamesGeneratedDirectories(t *testing.T) {
	dir := t.TempDir()
	for _, p := range []string{"internal/cache/cache.go", "build/out.bin", "node_modules/x/index.js", "logs.txt"} {
		os.MkdirAll(filepath.Join(dir, filepath.Dir(p)), 0o755)
		os.WriteFile(filepath.Join(dir, p), []byte("x"), 0o644)
	}
	os.WriteFile(filepath.Join(dir, "build.sh"), nil, 0o755)
	out := ls(t, dir, lsArgs{Depth: 3})
	for _, want := range []string{"build/  (contents not listed)", "cache/  (contents not listed)", "build.sh", "logs.txt"} {
		if !strings.Contains(out, want) {
			t.Errorf("no %q in:\n%s", want, out)
		}
	}
	for _, unwanted := range []string{"out.bin", "cache.go", "node_modules"} {
		if strings.Contains(out, unwanted) {
			t.Errorf("%q in:\n%s", unwanted, out)
		}
	}
	if out := ls(t, filepath.Join(dir, "build"), lsArgs{}); !strings.Contains(out, "out.bin") {
		t.Errorf("listing build itself:\n%s", out)
	}
	if out := ls(t, dir, lsArgs{Depth: 3, Ignore: []string{"internal/"}}); strings.Contains(out, "internal") {
		t.Errorf("ignored internal:\n%s", out)
	}
}

// A directory that cannot be read is noted, and the rest listed.
func TestLsGoesOnPastAnUnreadableDirectory(t *testing.T) {
	if runtime.GOOS == "windows" || os.Geteuid() == 0 {
		t.Skip("permissions do not keep this user out")
	}
	dir := t.TempDir()
	os.MkdirAll(filepath.Join(dir, "a", "secret"), 0o755)
	os.WriteFile(filepath.Join(dir, "z.go"), nil, 0o644)
	os.Chmod(filepath.Join(dir, "a"), 0o000)
	t.Cleanup(func() { os.Chmod(filepath.Join(dir, "a"), 0o755) })
	out := ls(t, dir, lsArgs{Depth: 2})
	if !strings.Contains(out, "(cannot list: permission denied)") || !strings.Contains(out, "z.go") {
		t.Fatalf("listing:\n%s", out)
	}
}

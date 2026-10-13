//go:build windows

package tools

import (
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
)

// configureProcGroup has cmd.Cancel kill the command's whole process tree:
// killing bash alone would leave what it started running.
func configureProcGroup(cmd *exec.Cmd) {
	cmd.Cancel = func() error {
		taskkill := filepath.Join(os.Getenv("SystemRoot"), "System32", "taskkill.exe")
		return exec.Command(taskkill, "/T", "/F", "/PID", strconv.Itoa(cmd.Process.Pid)).Run()
	}
}

// resolveShell finds Git for Windows' bash. The bash.exe in the Windows
// directory runs commands in WSL, another system with other files.
func resolveShell() (string, []string, error) {
	var candidates []string
	if p, err := exec.LookPath("bash"); err == nil && !strings.HasPrefix(strings.ToLower(p), strings.ToLower(os.Getenv("SystemRoot"))) {
		candidates = append(candidates, p)
	}
	// git.exe is in Git\cmd, Git\bin or Git\mingw64\bin; bash.exe in Git\bin.
	if git, err := exec.LookPath("git"); err == nil {
		up := filepath.Dir(filepath.Dir(git))
		candidates = append(candidates, filepath.Join(up, "bin", "bash.exe"), filepath.Join(filepath.Dir(up), "bin", "bash.exe"))
	}
	for _, p := range candidates {
		if info, err := os.Stat(p); err == nil && !info.IsDir() {
			return p, []string{"-c"}, nil
		}
	}
	return "", nil, errors.New("no bash found: install Git for Windows, whose bash runs the commands")
}

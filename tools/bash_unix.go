//go:build !windows

package tools

import (
	"fmt"
	"os/exec"
	"syscall"
)

// configureProcGroup sets up process-group-based killing on Unix. The child
// leads a session of its own, so it has no controlling terminal to read a
// password from or type into, and its process group, which cmd.Cancel kills
// whole via the negative PID, preventing orphan subprocesses.
func configureProcGroup(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{Setsid: true}
	cmd.Cancel = func() error {
		return syscall.Kill(-cmd.Process.Pid, syscall.SIGKILL)
	}
}

func resolveShell() (string, []string, error) {
	for _, name := range []string{"bash", "sh"} {
		if p, err := exec.LookPath(name); err == nil {
			return p, []string{"-c"}, nil
		}
	}
	return "", nil, fmt.Errorf("no shell found: tried bash and sh")
}

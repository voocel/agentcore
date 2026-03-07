package policy

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/voocel/agentcore"
	"github.com/voocel/agentcore/tools"
)

var defaultPathTools = []string{"read", "write", "edit", "ls", "find", "grep"}

// Rule is a lightweight permission rule for a single tool call.
type Rule func(ctx context.Context, call agentcore.ToolCall) error

// Chain composes rules in order and returns a PermissionFunc for WithPermission.
func Chain(rules ...Rule) agentcore.PermissionFunc {
	filtered := make([]Rule, 0, len(rules))
	for _, rule := range rules {
		if rule != nil {
			filtered = append(filtered, rule)
		}
	}

	return func(ctx context.Context, call agentcore.ToolCall) error {
		for _, rule := range filtered {
			if err := rule(ctx, call); err != nil {
				return err
			}
		}
		return nil
	}
}

// AllowTools permits only the listed tools and rejects all others.
func AllowTools(names ...string) Rule {
	allowed := makeNameSet(names...)
	return func(_ context.Context, call agentcore.ToolCall) error {
		if allowed[call.Name] {
			return nil
		}
		return fmt.Errorf("tool %q is not allowed", call.Name)
	}
}

// DenyTools rejects the listed tools and allows all others.
func DenyTools(names ...string) Rule {
	denied := makeNameSet(names...)
	return func(_ context.Context, call agentcore.ToolCall) error {
		if denied[call.Name] {
			return fmt.Errorf("tool %q is denied", call.Name)
		}
		return nil
	}
}

// RestrictPaths limits tool paths to stay within root.
// If no tool names are provided, it applies to the built-in path-based tools.
func RestrictPaths(root string, names ...string) Rule {
	rootPath := canonicalPath(root)
	limitedTools := makeNameSet(names...)
	if len(limitedTools) == 0 {
		limitedTools = makeNameSet(defaultPathTools...)
	}

	return func(_ context.Context, call agentcore.ToolCall) error {
		if !limitedTools[call.Name] {
			return nil
		}

		path, ok := extractPath(call.Args)
		if !ok {
			return nil
		}
		if path == "" {
			path = rootPath
		}

		resolved := tools.ResolvePath(rootPath, path)
		if !isWithinRoot(rootPath, resolved) {
			return fmt.Errorf("path %q is outside allowed root %q", path, rootPath)
		}
		return nil
	}
}

// ReadOnlyProfile allows read-only tools and limits paths to root.
func ReadOnlyProfile(root string) agentcore.PermissionFunc {
	toolNames := []string{"read", "ls", "find", "grep"}
	return Chain(
		AllowTools(toolNames...),
		RestrictPaths(root, toolNames...),
	)
}

// WorkspaceProfile allows common workspace file tools and excludes bash.
func WorkspaceProfile(root string) agentcore.PermissionFunc {
	toolNames := []string{"read", "write", "edit", "ls", "find", "grep"}
	return Chain(
		AllowTools(toolNames...),
		RestrictPaths(root, toolNames...),
	)
}

func makeNameSet(names ...string) map[string]bool {
	set := make(map[string]bool, len(names))
	for _, name := range names {
		if name != "" {
			set[name] = true
		}
	}
	return set
}

func extractPath(raw json.RawMessage) (string, bool) {
	var payload struct {
		Path string `json:"path"`
	}
	if err := json.Unmarshal(raw, &payload); err != nil {
		return "", false
	}
	return payload.Path, true
}

func canonicalPath(path string) string {
	if path == "" {
		path = "."
	}
	path = tools.ExpandPath(path)
	if abs, err := filepath.Abs(path); err == nil {
		path = abs
	}
	path = filepath.Clean(path)

	if resolved, ok := resolveExistingPrefix(path); ok {
		path = resolved
	}
	return path
}

func isWithinRoot(root, target string) bool {
	root = canonicalPath(root)
	target = canonicalPath(target)

	if rel, err := filepath.Rel(root, target); err == nil {
		if rel == "." {
			return true
		}
		if rel == ".." {
			return false
		}
		return !strings.HasPrefix(rel, ".."+string(filepath.Separator))
	}
	return false
}

func resolveExistingPrefix(path string) (string, bool) {
	if resolved, err := filepath.EvalSymlinks(path); err == nil {
		return filepath.Clean(resolved), true
	}

	current := path
	var suffix []string
	for {
		if _, err := os.Lstat(current); err == nil {
			resolved, err := filepath.EvalSymlinks(current)
			if err != nil {
				return "", false
			}
			for i := len(suffix) - 1; i >= 0; i-- {
				resolved = filepath.Join(resolved, suffix[i])
			}
			return filepath.Clean(resolved), true
		}

		parent := filepath.Dir(current)
		if parent == current {
			return "", false
		}
		suffix = append(suffix, filepath.Base(current))
		current = parent
	}
}

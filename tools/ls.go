package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"io/fs"
	"os"
	"path"
	"path/filepath"
	"slices"
	"sort"
	"strings"

	"github.com/voocel/agentcore"
	"github.com/voocel/agentcore/schema"
)

// Ls returns the ls tool: it lists directory contents as a tree, to a
// depth.
func (w Workspace) Ls() agentcore.Tool {
	t := &lsTool{w: w}
	return agentcore.Tool{
		Name:        "ls",
		Label:       "List Directory",
		Description: "List directory contents as a tree. Use this for quick directory structure checks before reading files. Depth controls recursive listing (default 1, max 5). Use ignore to hide generated or irrelevant paths. Dependency and VCS directories (node_modules, .git, etc.) are hidden, and the contents of common generated ones (dist, build, cache, etc.) are not listed unless you list them.",
		Schema: schema.Object(
			schema.Property("path", schema.String("Directory path, relative or absolute (default: working directory)")),
			schema.Property("depth", schema.Int("Recursion depth (default: 1, max: 5)")),
			schema.Property("ignore", schema.Array("Optional file or directory patterns to ignore (for example: tmp/, *.log, dist)", schema.String("Ignore pattern"))),
		),
		Parallel: true,
		Run:      textRun(t.execute),
	}
}

type lsTool struct {
	w Workspace
}

type lsArgs struct {
	Path   string   `json:"path"`
	Depth  int      `json:"depth"`
	Ignore []string `json:"ignore"`
}

type ignoreMatcher struct {
	patterns []string
}

const lsDefaultLimit = 500

// lsCollapsed are directories that most often hold what a build or a tool
// made: a listing names them without their contents. A project's own may
// share a name, so they are not hidden, as skipDirs are.
var lsCollapsed = []string{
	"dist", "build", "target", "vendor", "bin", "obj", ".idea", ".vscode", ".cache", "cache",
	"tmp", "temp", "coverage", "logs", "env", "venv", ".zig-cache", "zig-out",
}

func (t *lsTool) execute(ctx context.Context, args json.RawMessage) (string, error) {
	var a lsArgs
	if err := json.Unmarshal(args, &a); err != nil {
		return "", fmt.Errorf("invalid args: %w", err)
	}

	dir := ResolvePath(t.w.dir(ctx), a.Path)
	l := &lister{root: dir, depth: min(max(a.Depth, 1), 5), ignore: newIgnoreMatcher(a.Ignore)}
	if err := l.render(ctx, dir, 0, ""); err != nil {
		return "", fmt.Errorf("ls %s: %w", dir, err)
	}
	if l.count == 0 {
		return "(empty directory)", nil
	}

	result := strings.TrimRight(dir, string(filepath.Separator)) + "/\n" + strings.TrimRight(l.sb.String(), "\n")
	if l.truncated {
		result += fmt.Sprintf("\n\n[Listing truncated at %d entries. List a subdirectory, a smaller depth, or ignore more.]", lsDefaultLimit)
	}

	tr := truncateHead(result, defaultMaxLines, defaultMaxBytes)
	if tr.Truncated {
		return tr.Content + "\n\n[Output truncated at " + formatSize(defaultMaxBytes) + ".]", nil
	}
	return result, nil
}

// newIgnoreMatcher returns the matcher of patterns. The trailing slash of a
// directory's is dropped: the pattern hides a file of that name too.
func newIgnoreMatcher(patterns []string) ignoreMatcher {
	out := make([]string, 0, len(patterns))
	for _, p := range patterns {
		if p = strings.TrimSuffix(strings.TrimSpace(filepath.ToSlash(p)), "/"); p != "" {
			out = append(out, p)
		}
	}
	return ignoreMatcher{patterns: out}
}

// Match reports whether a pattern matches rel, a slash-separated path, a
// directory above it, or its name.
func (m ignoreMatcher) Match(rel string) bool {
	base := path.Base(rel)
	for _, pattern := range m.patterns {
		if rel == pattern || strings.HasPrefix(rel, pattern+"/") {
			return true
		}
		if ok, _ := path.Match(pattern, rel); ok {
			return true
		}
		if ok, _ := path.Match(pattern, base); ok {
			return true
		}
	}
	return false
}

// lister writes the tree of root, to depth levels and lsDefaultLimit
// entries.
type lister struct {
	root      string
	depth     int
	ignore    ignoreMatcher // what the call asked to hide
	count     int
	truncated bool
	sb        strings.Builder
}

func (l *lister) render(ctx context.Context, dir string, level int, prefix string) error {
	if ctx.Err() != nil {
		return ctx.Err()
	}
	dirEntries, err := os.ReadDir(dir)
	if err != nil {
		return err
	}
	sort.Slice(dirEntries, func(i, j int) bool {
		return strings.ToLower(dirEntries[i].Name()) < strings.ToLower(dirEntries[j].Name())
	})

	var visible []os.DirEntry
	for _, e := range dirEntries {
		// Files of skipDirs' names too, such as the .git file of a worktree.
		if IsSkipDir(e.Name()) {
			continue
		}
		rel, _ := filepath.Rel(l.root, filepath.Join(dir, e.Name()))
		if !l.ignore.Match(filepath.ToSlash(rel)) {
			visible = append(visible, e)
		}
	}

	for i, e := range visible {
		if l.count >= lsDefaultLimit {
			l.truncated = true
			return nil
		}
		info, err := e.Info()
		if err != nil {
			continue
		}
		l.count++

		connector, childPrefix := "├── ", prefix+"│   "
		if i == len(visible)-1 {
			connector, childPrefix = "└── ", prefix+"    "
		}
		name := e.Name()
		switch {
		case !e.IsDir():
			l.sb.WriteString(prefix + connector + name + "  " + formatSize(int(info.Size())) + "\n")
		case slices.Contains(lsCollapsed, name):
			l.sb.WriteString(prefix + connector + name + "/  (contents not listed)\n")
		default:
			l.sb.WriteString(prefix + connector + name + "/\n")
			if level+1 == l.depth {
				continue
			}
			if err := l.render(ctx, filepath.Join(dir, name), level+1, childPrefix); err != nil {
				if ctx.Err() != nil {
					return err
				}
				l.sb.WriteString(childPrefix + "(cannot list: " + errText(err) + ")\n")
			}
		}
	}
	return nil
}

// errText is err without the path a filesystem error repeats.
func errText(err error) string {
	if pe, ok := err.(*fs.PathError); ok {
		return pe.Err.Error()
	}
	return err.Error()
}

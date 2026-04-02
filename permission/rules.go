package permission

import (
	"fmt"
	"net/url"
	"path/filepath"
	"strings"
)

type Rule struct {
	Raw     string
	Kind    string
	Pattern string
}

type RuleSet struct {
	Allow []Rule
	Deny  []Rule
}

func ParseRuleSet(allow, deny []string) (*RuleSet, error) {
	if len(allow) == 0 && len(deny) == 0 {
		return nil, nil
	}
	rs := &RuleSet{}
	for _, raw := range deny {
		r, err := ParseRule(raw)
		if err != nil {
			return nil, fmt.Errorf("deny rule %q: %w", raw, err)
		}
		rs.Deny = append(rs.Deny, r)
	}
	for _, raw := range allow {
		r, err := ParseRule(raw)
		if err != nil {
			return nil, fmt.Errorf("allow rule %q: %w", raw, err)
		}
		rs.Allow = append(rs.Allow, r)
	}
	return rs, nil
}

func ParseRule(raw string) (Rule, error) {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return Rule{}, fmt.Errorf("empty rule")
	}
	if idx := strings.IndexByte(raw, '('); idx > 0 {
		if !strings.HasSuffix(raw, ")") {
			return Rule{}, fmt.Errorf("unclosed parenthesis in rule %q", raw)
		}
		kind := raw[:idx]
		pattern := raw[idx+1 : len(raw)-1]
		switch kind {
		case "Bash", "Edit", "Write", "Read", "WebFetch", "Subagent":
			return Rule{Raw: raw, Kind: kind, Pattern: pattern}, nil
		default:
			return Rule{}, fmt.Errorf("unknown rule kind %q", kind)
		}
	}
	return Rule{Raw: raw, Kind: "tool", Pattern: raw}, nil
}

func (rs *RuleSet) Evaluate(info toolInfo) (ruleAction, bool) {
	if rs == nil {
		return "", false
	}
	for _, r := range rs.Deny {
		if r.matches(info, true) {
			return ruleDeny, true
		}
	}
	for _, r := range rs.Allow {
		if r.matches(info, false) {
			return ruleAllow, true
		}
	}
	return "", false
}

func (r Rule) matches(info toolInfo, isDeny bool) bool {
	switch r.Kind {
	case "Bash":
		if info.tool != "bash" {
			return false
		}
		return matchBash(r.Pattern, info.summary, isDeny)
	case "Edit", "Write":
		if info.tool != "write" && info.tool != "edit" {
			return false
		}
		return matchPath(r.Pattern, info.summary, info.workspace, info.roots)
	case "Read":
		switch info.tool {
		case "read", "glob", "grep", "ls":
		default:
			return false
		}
		return matchPath(r.Pattern, info.summary, info.workspace, info.roots)
	case "WebFetch":
		if info.capability != CapabilityNetwork {
			return false
		}
		return matchHost(r.Pattern, hostFromSummary(info.summary))
	case "Subagent":
		if info.tool != "subagent" {
			return false
		}
		return strings.EqualFold(r.Pattern, info.summary)
	case "tool":
		return matchToolName(r.Pattern, info.tool)
	default:
		return false
	}
}

func matchBash(pattern, command string, isDeny bool) bool {
	command = strings.TrimSpace(command)
	pattern = strings.TrimSpace(pattern)
	if command == "" || pattern == "" {
		return false
	}
	if hasShellOperator(command) {
		if !isDeny {
			return false
		}
		for _, seg := range splitShellSegments(command) {
			seg = strings.TrimSpace(seg)
			if seg != "" && matchSimpleBash(pattern, seg) {
				return true
			}
		}
		return false
	}
	return matchSimpleBash(pattern, command)
}

func matchSimpleBash(pattern, command string) bool {
	pt := strings.Fields(pattern)
	ct := strings.Fields(command)
	if len(pt) == 0 || len(ct) == 0 {
		return false
	}
	for i, p := range pt {
		if p == "*" {
			return i < len(ct)
		}
		if i >= len(ct) || p != ct[i] {
			return false
		}
	}
	return len(ct) >= len(pt)
}

func hasShellOperator(cmd string) bool {
	return len(splitShellSegments(cmd)) > 1
}

func splitShellSegments(cmd string) []string {
	var (
		parts    []string
		buf      strings.Builder
		inSingle bool
		inDouble bool
		escaped  bool
	)
	flush := func() {
		part := strings.TrimSpace(buf.String())
		if part != "" {
			parts = append(parts, part)
		}
		buf.Reset()
	}
	for i := 0; i < len(cmd); i++ {
		ch := cmd[i]
		if escaped {
			buf.WriteByte(ch)
			escaped = false
			continue
		}
		if ch == '\\' && !inSingle {
			escaped = true
			buf.WriteByte(ch)
			continue
		}
		if ch == '\'' && !inDouble {
			inSingle = !inSingle
			buf.WriteByte(ch)
			continue
		}
		if ch == '"' && !inSingle {
			inDouble = !inDouble
			buf.WriteByte(ch)
			continue
		}
		if !inSingle && !inDouble {
			if ch == ';' {
				flush()
				continue
			}
			if i+1 < len(cmd) && ((ch == '&' && cmd[i+1] == '&') || (ch == '|' && cmd[i+1] == '|')) {
				flush()
				i++
				continue
			}
			if ch == '|' {
				flush()
				continue
			}
		}
		buf.WriteByte(ch)
	}
	flush()
	return parts
}

func matchPath(pattern, path, workspace string, roots []string) bool {
	if matchGlob(pattern, path) {
		return true
	}
	if len(roots) == 0 && workspace != "" {
		roots = []string{workspace}
	}
	for _, root := range roots {
		if root == "" || !filepath.IsAbs(path) {
			continue
		}
		rel, err := filepath.Rel(root, path)
		if err == nil && rel != "." && !strings.HasPrefix(rel, "..") {
			if matchGlob(pattern, rel) {
				return true
			}
		}
	}
	return false
}

func matchGlob(pattern, path string) bool {
	pattern = strings.TrimSpace(pattern)
	path = strings.TrimSpace(path)
	if pattern == "" || path == "" {
		return false
	}
	if strings.Contains(pattern, "**") {
		return matchDoubleStarGlob(pattern, path)
	}
	matched, _ := filepath.Match(pattern, path)
	if matched {
		return true
	}
	matched, _ = filepath.Match(pattern, filepath.Base(path))
	return matched
}

func matchDoubleStarGlob(pattern, path string) bool {
	parts := strings.SplitN(pattern, "**", 2)
	prefix := strings.TrimRight(parts[0], string(filepath.Separator))
	suffix := ""
	if len(parts) > 1 {
		suffix = strings.TrimLeft(parts[1], string(filepath.Separator))
	}
	if prefix != "" {
		cleanPath := filepath.Clean(path)
		cleanPrefix := filepath.Clean(prefix)
		if !strings.HasPrefix(cleanPath, cleanPrefix+string(filepath.Separator)) && cleanPath != cleanPrefix {
			return false
		}
	}
	if suffix == "" {
		return true
	}
	matched, _ := filepath.Match(suffix, filepath.Base(path))
	return matched
}

func matchHost(pattern, host string) bool {
	pattern = strings.ToLower(strings.TrimSpace(pattern))
	host = strings.ToLower(strings.TrimSpace(host))
	if pattern == "" || host == "" {
		return false
	}
	if pattern == "*" {
		return true
	}
	if strings.HasPrefix(pattern, "*.") {
		suffix := pattern[1:]
		return host == pattern[2:] || strings.HasSuffix(host, suffix)
	}
	return pattern == host
}

func matchToolName(pattern, name string) bool {
	pattern = strings.ToLower(strings.TrimSpace(pattern))
	name = strings.ToLower(strings.TrimSpace(name))
	if pattern == "" || name == "" {
		return false
	}
	if strings.HasSuffix(pattern, "*") {
		return strings.HasPrefix(name, pattern[:len(pattern)-1])
	}
	return pattern == name
}

func hostFromSummary(summary string) string {
	summary = strings.TrimSpace(summary)
	if parsed, err := url.Parse(summary); err == nil && parsed.Host != "" {
		return strings.ToLower(parsed.Host)
	}
	return strings.ToLower(summary)
}

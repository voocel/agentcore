package permission

import (
	"path/filepath"
	"testing"
)

func TestParseRuleSetNil(t *testing.T) {
	rs, err := ParseRuleSet(nil, nil)
	if err != nil {
		t.Fatalf("ParseRuleSet: %v", err)
	}
	if rs != nil {
		t.Fatalf("expected nil ruleset, got %#v", rs)
	}
}

func TestRuleSetDenyOverridesAllow(t *testing.T) {
	rs, err := ParseRuleSet([]string{"Bash(git *)"}, []string{"Bash(rm -rf *)"})
	if err != nil {
		t.Fatalf("ParseRuleSet: %v", err)
	}
	action, matched := rs.Evaluate(toolInfo{
		tool:       "bash",
		capability: CapabilityExec,
		summary:    "rm -rf /",
	})
	if !matched || action != ruleDeny {
		t.Fatalf("expected deny override, got action=%q matched=%v", action, matched)
	}
}

func TestRuleSetMatchesPathAndHost(t *testing.T) {
	workspace := t.TempDir()
	rs, err := ParseRuleSet([]string{"Edit(src/**)", "WebFetch(*.github.com)"}, nil)
	if err != nil {
		t.Fatalf("ParseRuleSet: %v", err)
	}

	action, matched := rs.Evaluate(toolInfo{
		tool:       "edit",
		summary:    filepath.Join(workspace, "src", "main.go"),
		workspace:  workspace,
		roots:      []string{workspace},
		capability: CapabilityWrite,
	})
	if !matched || action != ruleAllow {
		t.Fatalf("expected path allow, got action=%q matched=%v", action, matched)
	}

	action, matched = rs.Evaluate(toolInfo{
		tool:       "web_fetch",
		summary:    "https://api.github.com/repos/voocel/agentcore",
		capability: CapabilityNetwork,
	})
	if !matched || action != ruleAllow {
		t.Fatalf("expected host allow, got action=%q matched=%v", action, matched)
	}
}

func TestMatchBashCases(t *testing.T) {
	cases := []struct {
		pattern string
		command string
		isDeny  bool
		want    bool
	}{
		{"git *", "git status", false, true},
		{"make build", "make build test", false, true},
		{"npm install", "npm install lodash", false, true},
		{"npm install", "npm ci", false, false},
		{"echo *", "echo ok && rm -rf /", false, false},
		{"rm *", "echo ok && rm -rf /", true, true},
		{"python *", `python -c 'print("a|b")'`, false, true},
	}
	for _, tc := range cases {
		if got := matchBash(tc.pattern, tc.command, tc.isDeny); got != tc.want {
			t.Fatalf("matchBash(%q, %q, %v) = %v, want %v", tc.pattern, tc.command, tc.isDeny, got, tc.want)
		}
	}
}

func TestMatchHostAndToolName(t *testing.T) {
	if !matchHost("*.github.com", "api.github.com") {
		t.Fatal("expected host wildcard match")
	}
	if matchHost("example.com", "api.example.com") {
		t.Fatal("expected host mismatch")
	}
	if !matchToolName("mcp__ctx__*", "mcp__ctx__query") {
		t.Fatal("expected tool wildcard match")
	}
	if matchToolName("exact_tool", "other_tool") {
		t.Fatal("expected exact tool mismatch")
	}
}

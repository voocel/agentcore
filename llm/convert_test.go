package llm

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"

	"github.com/voocel/agentcore"
	"github.com/voocel/litellm"
)

func TestNormalizeArgs_EmptyBecomesEmptyObject(t *testing.T) {
	got := normalizeArgs("")
	if string(got.Args) != "{}" {
		t.Fatalf(`empty args: got %q, want "{}"`, got.Args)
	}
	if got.Invalid {
		t.Fatal("empty args must not be flagged invalid")
	}
}

func TestNormalizeArgs_ValidJSONPassThrough(t *testing.T) {
	in := `{"foo":"bar"}`
	got := normalizeArgs(in)
	if string(got.Args) != in {
		t.Fatalf("valid args mangled: got %q, want %q", got.Args, in)
	}
	if got.Invalid {
		t.Fatal("valid args must not be flagged invalid")
	}
}

// Invalid JSON must NOT land in Args (json.RawMessage marshalling validates
// its bytes — corrupt args here break agent.ExportMessages → json.Marshal,
// which codebot uses to persist history). Capture them in diagnostic fields
// instead, with Args set to a safe placeholder.
func TestNormalizeArgs_InvalidArgsRoutedToDiagnostics(t *testing.T) {
	bad := `{"path":"/tmp/file`
	got := normalizeArgs(bad)
	if string(got.Args) != "{}" {
		t.Fatalf("invalid args must produce {} placeholder, got %q", got.Args)
	}
	if !got.Invalid {
		t.Fatal("invalid flag not set")
	}
	if got.RawText != bad {
		t.Fatalf("raw text not preserved: got %q want %q", got.RawText, bad)
	}
	if got.ParseErr == "" {
		t.Fatal("parse error not captured")
	}
}

// Regression: a malformed tool_call must not corrupt history serialization.
// Before the diagnostic-field split, invalid bytes lived directly in Args
// (json.RawMessage) and json.Marshal of the surrounding Message returned
// "unexpected end of JSON input" — breaking session persistence whenever a
// provider truncated a streaming tool_call.
func TestBuildToolCall_InvalidArgsMessageStaysJSONSerializable(t *testing.T) {
	tc := buildToolCall("call_x", "edit", `{"path":"/tmp/file`)
	if !tc.ArgsInvalid {
		t.Fatalf("expected ArgsInvalid=true, got %+v", tc)
	}
	msg := agentcore.Message{
		Role:    agentcore.RoleAssistant,
		Content: []agentcore.ContentBlock{agentcore.ToolCallBlock(tc)},
	}
	out, err := json.Marshal(msg)
	if err != nil {
		t.Fatalf("malformed args broke history marshal: %v", err)
	}
	if !strings.Contains(string(out), `"args":{}`) {
		t.Fatalf("Args placeholder not in serialized message: %s", out)
	}
	if !strings.Contains(string(out), `"args_raw_text":"{\"path\":\"/tmp/file"`) {
		t.Fatalf("raw text not in serialized message: %s", out)
	}
}

func TestConvertThinking(t *testing.T) {
	for _, tt := range []struct {
		level  agentcore.ThinkingLevel
		budget int
		want   *litellm.Thinking
	}{
		{ThinkingAuto, 0, nil},
		{ThinkingAuto, 1024, &litellm.Thinking{BudgetTokens: new(1024)}},
		{agentcore.ThinkingOff, 1024, &litellm.Thinking{Mode: litellm.ThinkingDisabled}},
		{agentcore.ThinkingHigh, 0, &litellm.Thinking{Effort: "high"}},
		{agentcore.ThinkingHigh, 512, &litellm.Thinking{Effort: "high", BudgetTokens: new(512)}},
	} {
		if got := convertThinking(tt.level, tt.budget); !reflect.DeepEqual(got, tt.want) {
			t.Fatalf("convertThinking(%q, %d) = %+v, want %+v", tt.level, tt.budget, got, tt.want)
		}
	}
}

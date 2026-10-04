package tools

import (
	"context"
	"encoding/json"
	"reflect"
	"testing"

	"github.com/voocel/agentcore"
	"github.com/voocel/litellm"
)

func named(name string) agentcore.Tool {
	return agentcore.Tool{Name: name, Description: name + " tool", Schema: map[string]any{"type": "object"}}
}

func offered(tools []agentcore.Tool, history []agentcore.Message) map[string]bool {
	out := map[string]bool{}
	req := agentcore.BuildCall(agentcore.Config{Tools: tools}, history).Request
	for _, spec := range req.OfferedTools() {
		out[spec.Name] = true
	}
	return out
}

func TestDeferredToolsLoadOnceTheHistoryReferencesThem(t *testing.T) {
	deferred := Defer([]agentcore.Tool{named("deploy"), named("rollback")})
	search := deferred[0]
	// Deferring more tools leaves the definition of tool_search as it was.
	if later := Defer([]agentcore.Tool{named("deploy")})[0]; later.Description != search.Description || !reflect.DeepEqual(later.Schema, search.Schema) {
		t.Fatal("the definition of tool_search depends on the tools")
	}
	tools := append([]agentcore.Tool{named("read")}, deferred...)

	history := []agentcore.Message{agentcore.UserText("ship it")}
	if got := offered(tools, history); len(got) != 2 || !got["read"] || !got["tool_search"] {
		t.Fatalf("before a search, offered %v", got)
	}

	res, err := search.Run(context.Background(), json.RawMessage(`{"query":"select:deploy,unknown"}`))
	if err != nil {
		t.Fatal(err)
	}
	call := litellm.ToolUseBlock{ID: "c1", Name: "tool_search", Arguments: `{"query":"select:deploy,unknown"}`}
	history = append(history, agentcore.Message{Role: litellm.RoleAssistant, Blocks: []litellm.Block{call}}, agentcore.ToolResult("c1", res))
	if got := offered(tools, history); len(got) != 3 || !got["deploy"] || got["rollback"] {
		t.Fatalf("after the search, offered %v", got)
	}
}

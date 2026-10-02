package tools

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/voocel/agentcore"
)

func named(name string) agentcore.Tool {
	return agentcore.Tool{Name: name, Description: name + " tool", Schema: map[string]any{"type": "object"}}
}

func offered(tools []agentcore.Tool, history []agentcore.Message) map[string]bool {
	out := map[string]bool{}
	for _, spec := range agentcore.BuildCall(agentcore.Config{Tools: tools}, history).Request.Tools {
		out[spec.Name] = true
	}
	return out
}

func TestDeferredToolsLoadOnceTheHistoryReferencesThem(t *testing.T) {
	deferred := Defer([]agentcore.Tool{named("deploy"), named("rollback")})
	search := deferred[0]
	if !strings.HasSuffix(search.Description, "Deferred tools: deploy, rollback") {
		t.Fatalf("description = %q", search.Description)
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
	history = append(history, agentcore.ToolResult("c1", res))
	if got := offered(tools, history); len(got) != 3 || !got["deploy"] || got["rollback"] {
		t.Fatalf("after the search, offered %v", got)
	}
}

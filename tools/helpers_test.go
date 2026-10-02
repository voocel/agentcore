package tools

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/voocel/agentcore"
)

// run calls tool with args, marshaled, and returns the text of its result.
func run(t *testing.T, tool agentcore.Tool, args any) (string, error) {
	t.Helper()
	raw, err := json.Marshal(args)
	if err != nil {
		t.Fatalf("marshal args: %v", err)
	}
	res, err := tool.Run(context.Background(), raw)
	if err != nil {
		return "", err
	}
	return res.Text(), nil
}

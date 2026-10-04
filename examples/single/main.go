// Single runs one agent with the coding tools on a task:
//
//	DEEPSEEK_API_KEY=... go run ./examples/single
//
// Override the model with DEEPSEEK_MODEL.
package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/voocel/agentcore"
	"github.com/voocel/agentcore/tools"
	"github.com/voocel/litellm"
	"github.com/voocel/litellm/provider/deepseek"
)

func main() {
	provider, err := deepseek.New(deepseek.Config{APIKey: os.Getenv("DEEPSEEK_API_KEY"), BaseURL: os.Getenv("DEEPSEEK_BASE_URL")})
	if err != nil {
		log.Fatal(err)
	}
	client, err := litellm.New(provider)
	if err != nil {
		log.Fatal(err)
	}

	// Files records what read read, so that write and edit refuse to change
	// a file the model has not seen, or that changed since.
	workspace := tools.Workspace{Dir: ".", Files: tools.NewFileReadState()}
	cfg := agentcore.Config{
		Model:    agentcore.Model{Client: client, Request: litellm.Request{Model: env("DEEPSEEK_MODEL", "deepseek-flash")}},
		System:   []litellm.Block{litellm.Text("You are a helpful coding assistant. Use the tools to help the user.")},
		Tools:    workspace.Tools(),
		MaxTurns: 20,
		Emit:     show,
	}
	if _, err := agentcore.Run(context.Background(), cfg, nil, agentcore.UserText("List the files in the current directory and tell me what you see.")); err != nil {
		log.Fatal(err)
	}
	fmt.Println()
}

// show prints the response as it streams and the tool calls as they run.
func show(ev agentcore.Event) error {
	switch ev := ev.(type) {
	case agentcore.MessageDelta:
		if d, ok := ev.Event.(litellm.TextDelta); ok {
			fmt.Print(d.Text)
		}
	case agentcore.ToolStart:
		fmt.Printf("\n[%s] %s\n", ev.Call.Name, ev.Call.Args)
	case agentcore.ToolUpdate:
		fmt.Printf("  %v\n", ev.Progress)
	case agentcore.ToolEnd:
		if ev.Result.IsError {
			fmt.Printf("  failed: %s\n", ev.Result.Text())
		}
	case agentcore.Retry:
		fmt.Printf("\n[attempt %d in %s] %v\n", ev.Attempt, ev.Delay, ev.Err)
	}
	return nil
}

func env(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

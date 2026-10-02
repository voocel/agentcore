// Multi runs an agent that delegates to sub-agents, a scout and a reviewer:
//
//	DEEPSEEK_API_KEY=... go run ./examples/multi
//
// Override the model with DEEPSEEK_MODEL.
package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/voocel/agentcore"
	"github.com/voocel/agentcore/subagent"
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
	model := agentcore.Model{Client: client, Request: litellm.Request{Model: env("DEEPSEEK_MODEL", "deepseek-flash")}}

	// A sub-agent gets its tools anew on each run, so that what one run read
	// does not let another edit.
	readOnly := func(system string) func(subagent.Spawn) (agentcore.Config, error) {
		return func(subagent.Spawn) (agentcore.Config, error) {
			w := tools.Workspace{Dir: ".", Files: tools.NewFileReadState()}
			return agentcore.Config{
				Model:    model,
				System:   []litellm.Block{litellm.Text(system)},
				Tools:    []agentcore.Tool{w.Read(), w.Glob(), w.Grep(), w.Ls()},
				MaxTurns: 10,
			}, nil
		}
	}
	delegate := subagent.New(nil,
		subagent.Agent{
			Name:        "scout",
			Description: "Fast codebase reconnaissance",
			Config:      readOnly("You are a scout. Quickly explore the codebase and report what you find. Be concise."),
		},
		subagent.Agent{
			Name:        "reviewer",
			Description: "Code review specialist",
			Config:      readOnly("You are a code reviewer. Review the code and give constructive feedback on quality, style and correctness."),
		},
	)

	w := tools.Workspace{Dir: ".", Files: tools.NewFileReadState()}
	agent := agentcore.NewAgent(agentcore.Config{
		Model: model,
		System: []litellm.Block{litellm.Text("You are a coding assistant. Delegate with the subagent tool: " +
			"'scout' explores the codebase, 'reviewer' reviews code. " +
			"Chain them to scout first and review what the scout found.")},
		Tools:    []agentcore.Tool{w.Read(), w.Edit(), delegate},
		MaxTurns: 20,
	}, nil)
	agent.Subscribe(show)
	if err := agent.Prompt(context.Background(), agentcore.UserText("Explore the current directory, then review the Go files you find.")); err != nil {
		log.Fatal(err)
	}
	fmt.Println()
}

// show prints the response as it streams, and the steps of the sub-agents.
func show(ev agentcore.Event) error {
	switch ev := ev.(type) {
	case agentcore.MessageDelta:
		if d, ok := ev.Event.(litellm.TextDelta); ok {
			fmt.Print(d.Text)
		}
	case agentcore.ToolStart:
		fmt.Printf("\n[%s] %s\n", ev.Call.Name, ev.Call.Args)
	case agentcore.ToolUpdate:
		if p, ok := ev.Progress.(subagent.Progress); ok {
			if start, ok := p.Event.(agentcore.ToolStart); ok {
				fmt.Printf("  %s: [%s] %s\n", p.Spawn.ID, start.Call.Name, start.Call.Args)
			}
		}
	}
	return nil
}

func env(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

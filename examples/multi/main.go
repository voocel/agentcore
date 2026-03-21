package main

import (
	"fmt"
	"os"

	"github.com/voocel/agentcore"
	"github.com/voocel/agentcore/llm"
	"github.com/voocel/agentcore/tools"
)

func main() {
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		fmt.Fprintln(os.Stderr, "OPENAI_API_KEY not set")
		os.Exit(1)
	}

	mainModel, err := llm.NewOpenAIModel("gpt-5-mini", apiKey)
	if err != nil {
		fmt.Fprintf(os.Stderr, "model error: %v\n", err)
		os.Exit(1)
	}
	scoutModel, err := llm.NewOpenAIModel("gpt-5-mini", apiKey)
	if err != nil {
		fmt.Fprintf(os.Stderr, "model error: %v\n", err)
		os.Exit(1)
	}

	// Define sub-agent configurations (like pi's .md agent files)
	scout := agentcore.SubAgentConfig{
		Name:         "scout",
		Description:  "Fast codebase reconnaissance",
		Model:        scoutModel,
		SystemPrompt: "You are a scout agent. Quickly explore the codebase and report what you find. Be concise.",
		Tools: []agentcore.Tool{
			tools.NewRead("."),
			tools.NewBash("."),
		},
		MaxTurns: 5,
	}

	reviewer := agentcore.SubAgentConfig{
		Name:         "reviewer",
		Description:  "Code review specialist",
		Model:        mainModel,
		SystemPrompt: "You are a code reviewer. Review the code and provide constructive feedback on quality, style, and correctness.",
		Tools: []agentcore.Tool{
			tools.NewRead("."),
			tools.NewBash("."),
		},
		MaxTurns: 5,
	}

	// Main agent has the subagent tool — it delegates to scout/reviewer
	agent := agentcore.NewAgent(
		agentcore.WithModel(mainModel),
		agentcore.WithSystemPrompt(
			"You are a coding assistant. Use the subagent tool to delegate tasks:\n"+
				"- Use 'scout' for codebase exploration\n"+
				"- Use 'reviewer' for code review\n"+
				"You can use chain mode to scout first, then review based on findings.",
		),
		agentcore.WithTools(
			tools.NewRead("."),
			tools.NewWrite("."),
			tools.NewEdit("."),
			tools.NewBash("."),
			agentcore.NewSubAgentTool(scout, reviewer),
		),
		agentcore.WithMaxTurns(20),
	)

	agent.Subscribe(func(ev agentcore.Event) {
		switch ev.Type {
		case agentcore.EventMessageEnd:
			if ev.Message != nil && ev.Message.GetRole() == agentcore.RoleAssistant {
				fmt.Printf("\nAssistant: %s\n", ev.Message.TextContent())
			}
		case agentcore.EventToolExecStart:
			fmt.Printf("  [tool] %s\n", ev.Tool)
		case agentcore.EventToolExecUpdate:
			switch ev.UpdateKind {
			case agentcore.ToolExecUpdatePreview:
				fmt.Printf("  [preview] %s\n", string(ev.Result))
			case agentcore.ToolExecUpdateProgress:
				fmt.Printf("  [progress] %s\n", string(ev.Result))
			default:
				fmt.Printf("  [update] %s\n", string(ev.Result))
			}
		case agentcore.EventError:
			fmt.Fprintf(os.Stderr, "Error: %v\n", ev.Err)
		}
	})

	if err := agent.Prompt("Explore the current directory structure, then review any Go files you find. Use chain mode: scout first, then review."); err != nil {
		fmt.Fprintf(os.Stderr, "prompt error: %v\n", err)
		os.Exit(1)
	}

	agent.WaitForIdle()
}

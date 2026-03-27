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

	model, err := llm.NewOpenAIModel("gpt-5-mini", apiKey)
	if err != nil {
		fmt.Fprintf(os.Stderr, "model error: %v\n", err)
		os.Exit(1)
	}

	agent := agentcore.NewAgent(
		agentcore.WithModel(model),
		agentcore.WithSystemPrompt("You are a helpful coding assistant. Use the provided tools to help users."),
		agentcore.WithTools(
			tools.NewRead("."),
			tools.NewWrite("."),
			tools.NewEdit("."),
			tools.NewBash("."),
		),
		agentcore.WithMaxTurns(20),
	)

	// Subscribe to events for output
	agent.Subscribe(func(ev agentcore.Event) {
		switch ev.Type {
		case agentcore.EventMessageEnd:
			if ev.Message != nil && ev.Message.GetRole() == agentcore.RoleAssistant {
				fmt.Printf("\nAssistant: %s\n", ev.Message.TextContent())
			}
		case agentcore.EventToolExecStart:
			fmt.Printf("  [tool] %s(%s)\n", ev.Tool, string(ev.Args))
		case agentcore.EventToolExecUpdate:
			if ev.Progress != nil {
				fmt.Printf("  [progress:%s] %s\n", ev.Progress.Kind, formatProgress(ev.Progress))
				break
			}
			switch ev.UpdateKind {
			case agentcore.ToolExecUpdatePreview:
				fmt.Printf("  [preview] %s\n", string(ev.Result))
			case agentcore.ToolExecUpdateProgress:
				fmt.Printf("  [progress] %s\n", string(ev.Result))
			default:
				fmt.Printf("  [update] %s\n", string(ev.Result))
			}
		case agentcore.EventToolExecEnd:
			if ev.IsError {
				fmt.Printf("  [tool] %s error\n", ev.Tool)
			}
		case agentcore.EventError:
			fmt.Fprintf(os.Stderr, "Error: %v\n", ev.Err)
		}
	})

	if err := agent.Prompt("List the files in the current directory and tell me what you see."); err != nil {
		fmt.Fprintf(os.Stderr, "prompt error: %v\n", err)
		os.Exit(1)
	}

	agent.WaitForIdle()
}

func formatProgress(progress *agentcore.ProgressPayload) string {
	if progress == nil {
		return ""
	}
	if progress.Summary != "" {
		return progress.Summary
	}
	if progress.Tool != "" {
		return progress.Tool
	}
	if progress.Message != "" {
		return progress.Message
	}
	if progress.Delta != "" {
		return progress.Delta
	}
	return string(progress.Kind)
}

package agentcore

import "testing"

func TestAgentContinue_SteeringQueueOneAtATime(t *testing.T) {
	agent := NewAgent(
		WithStreamFn(mockStreamFn(
			assistantMsg("initial", StopReasonStop),
			assistantMsg("after steer 1", StopReasonStop),
			assistantMsg("after steer 2", StopReasonStop),
		)),
		WithSteeringMode(QueueModeOneAtATime),
	)

	if err := agent.Prompt("start"); err != nil {
		t.Fatalf("prompt failed: %v", err)
	}
	agent.WaitForIdle()

	agent.Steer(UserMsg("steer-1"))
	agent.Steer(UserMsg("steer-2"))

	if err := agent.Continue(); err != nil {
		t.Fatalf("first continue failed: %v", err)
	}
	agent.WaitForIdle()

	msgs := agent.Messages()
	want := []string{"start", "initial", "steer-1", "after steer 1", "steer-2", "after steer 2"}
	if len(msgs) != len(want) {
		t.Fatalf("expected %d messages, got %d", len(want), len(msgs))
	}
	for i, msg := range msgs {
		if got := msg.TextContent(); got != want[i] {
			t.Fatalf("message[%d]: expected %q, got %q", i, want[i], got)
		}
	}
}

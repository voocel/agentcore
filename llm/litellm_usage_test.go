package llm

import (
	"testing"

	"github.com/voocel/litellm"
)

func TestConvertResponsePreservesUsageModel(t *testing.T) {
	msg := convertResponse(&litellm.Response{
		Provider: "openrouter",
		Model:    "google/gemini-2.5-pro",
		Usage: litellm.Usage{
			PromptTokens:     100,
			CompletionTokens: 20,
			TotalTokens:      120,
		},
	})

	if msg.Usage == nil {
		t.Fatal("usage is nil")
	}
	if msg.Usage.Provider != "openrouter" || msg.Usage.Model != "google/gemini-2.5-pro" {
		t.Fatalf("usage model = %s/%s, want openrouter/google/gemini-2.5-pro", msg.Usage.Provider, msg.Usage.Model)
	}
	if msg.Usage.Input != 100 || msg.Usage.Output != 20 || msg.Usage.TotalTokens != 120 {
		t.Fatalf("usage tokens = %+v, want input=100 output=20 total=120", msg.Usage)
	}
}

func TestConvertResponsePrefersUsageModel(t *testing.T) {
	msg := convertResponse(&litellm.Response{
		Provider: "configured-provider",
		Model:    "configured-model",
		Usage: litellm.Usage{
			Provider:         "actual-provider",
			Model:            "actual-model",
			PromptTokens:     10,
			CompletionTokens: 5,
			TotalTokens:      15,
		},
	})

	if msg.Usage == nil {
		t.Fatal("usage is nil")
	}
	if msg.Usage.Provider != "actual-provider" || msg.Usage.Model != "actual-model" {
		t.Fatalf("usage model = %s/%s, want actual-provider/actual-model", msg.Usage.Provider, msg.Usage.Model)
	}
}

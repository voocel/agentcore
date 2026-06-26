package llm

import (
	"context"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/voocel/agentcore"
	"github.com/voocel/litellm"
	"github.com/voocel/litellm/provider/compat"
)

type captureProvider struct {
	lastReq *litellm.Request
}

func (p *captureProvider) Name() string { return "capture" }

func (p *captureProvider) Chat(_ context.Context, req *litellm.Request) (*litellm.Response, error) {
	p.lastReq = req
	return &litellm.Response{Blocks: []litellm.Block{litellm.TextBlock{Text: "ok"}}}, nil
}

func (p *captureProvider) Stream(context.Context, *litellm.Request) (litellm.Stream, error) {
	return nil, nil
}

func TestLiteLLMAdapterOmitsDefaultTemperature(t *testing.T) {
	provider := &captureProvider{}
	model := NewLiteLLMAdapter("m", mustClient(t, provider))
	_, err := model.Generate(context.Background(), []agentcore.Message{agentcore.UserMsg("hi")}, nil)
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if provider.lastReq == nil {
		t.Fatal("provider was not called")
	}
	if provider.lastReq.Temperature != nil {
		t.Fatalf("default temperature should be omitted, got %v", *provider.lastReq.Temperature)
	}
}

func TestLiteLLMAdapterSendsNonDefaultTemperature(t *testing.T) {
	provider := &captureProvider{}
	model := NewLiteLLMAdapter("m", mustClient(t, provider))
	model.GetConfig().Temperature = 0.2
	_, err := model.Generate(context.Background(), []agentcore.Message{agentcore.UserMsg("hi")}, nil)
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if provider.lastReq == nil || provider.lastReq.Temperature == nil {
		t.Fatal("non-default temperature should be sent")
	}
	if *provider.lastReq.Temperature != 0.2 {
		t.Fatalf("temperature = %v, want 0.2", *provider.lastReq.Temperature)
	}
}

func TestNewBaseModelClonesDefaultConfig(t *testing.T) {
	a := NewBaseModel(ModelInfo{Name: "a"}, nil)
	b := NewBaseModel(ModelInfo{Name: "b"}, nil)
	a.GetConfig().Temperature = 0.2
	if b.GetConfig().Temperature != DefaultGenerationConfig.Temperature {
		t.Fatalf("default config was shared: b temperature = %v", b.GetConfig().Temperature)
	}
}

func mustClient(t *testing.T, provider litellm.Provider) *litellm.Client {
	t.Helper()
	client, err := litellm.New(provider)
	if err != nil {
		t.Fatalf("litellm.New: %v", err)
	}
	return client
}

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) Do(req *http.Request) (*http.Response, error) { return f(req) }

// TestGenerateStreamFinalizesArglessToolCall is the end-to-end guard for the
// mimo novel_context regression: a streaming, argument-less tool call over a
// compat provider must surface with normalized "{}" arguments, not empty (which
// would fail json validation on the next turn). It exercises the full path —
// compat stream emitting ToolUseStart/ToolUseDone, then this adapter finalizing
// via normalizeArgs.
func TestGenerateStreamFinalizesArglessToolCall(t *testing.T) {
	sse := strings.Join([]string{
		`data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_9","function":{"name":"novel_context"}}]}}]}`,
		`data: {"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}`,
		`data: [DONE]`,
		``,
	}, "\n")
	provider, err := compat.New(compat.Config{
		BaseURL: "https://compat.test/v1",
		HTTPClient: roundTripFunc(func(*http.Request) (*http.Response, error) {
			return &http.Response{
				StatusCode: http.StatusOK,
				Body:       io.NopCloser(strings.NewReader(sse)),
				Header:     make(http.Header),
			}, nil
		}),
	}, compat.Spec{Name: "mimo"})
	if err != nil {
		t.Fatalf("compat.New: %v", err)
	}
	model := NewLiteLLMAdapter("mimo-v2.5", mustClient(t, provider))
	ch, err := model.GenerateStream(context.Background(), []agentcore.Message{agentcore.UserMsg("hi")}, nil)
	if err != nil {
		t.Fatalf("GenerateStream: %v", err)
	}
	var final agentcore.Message
	for ev := range ch {
		if ev.Type == agentcore.StreamEventError {
			t.Fatalf("stream error: %v", ev.Err)
		}
		if ev.Type == agentcore.StreamEventDone {
			final = ev.Message
		}
	}
	var tc *agentcore.ToolCall
	for _, b := range final.Content {
		if b.ToolCall != nil {
			tc = b.ToolCall
		}
	}
	if tc == nil {
		t.Fatal("no tool call in final message")
	}
	if tc.Name != "novel_context" {
		t.Fatalf("tool name = %q", tc.Name)
	}
	if string(tc.Args) != "{}" {
		t.Fatalf("argless tool call args = %q, want {}", string(tc.Args))
	}
}

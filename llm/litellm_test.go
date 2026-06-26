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

type capabilityProvider struct {
	captureProvider
	caps litellm.Capabilities
}

func (p *capabilityProvider) Capabilities(string) litellm.Capabilities {
	return p.caps
}

func TestLiteLLMAdapterCapabilities(t *testing.T) {
	provider := &capabilityProvider{
		caps: litellm.Capabilities{
			Provider: "capture",
			Model:    "m",
			Thinking: litellm.ThinkingCapabilities{
				Supported:     litellm.SupportYes,
				Disable:       litellm.SupportYes,
				Efforts:       []string{"minimal", "high", "max", "vendor-only"},
				BudgetTokens:  litellm.SupportPartial,
				IncludeOutput: litellm.SupportYes,
				Notes:         []string{"budget varies by model"},
			},
			Tools: litellm.ToolCapabilities{
				Calls:               litellm.SupportYes,
				ParallelCalls:       litellm.SupportPartial,
				StrictSchema:        litellm.SupportYes,
				Choice:              litellm.SupportNo,
				MultimodalResults:   litellm.SupportUnknown,
				RequiresAdjacency:   true,
				RoundTripSignatures: litellm.SupportYes,
				HostedProviderTools: litellm.SupportPartial,
			},
			Structured: litellm.StructuredCapabilities{
				JSONObject: litellm.SupportYes,
				JSONSchema: litellm.SupportYes,
				Strict:     litellm.SupportPartial,
				PromptOnly: true,
			},
			Streaming: litellm.StreamingCapabilities{
				Supported:       litellm.SupportYes,
				Usage:           litellm.SupportPartial,
				ReasoningDeltas: litellm.SupportYes,
				ToolCallDeltas:  litellm.SupportYes,
				NativeResponses: litellm.SupportNo,
				IdleTimeout:     litellm.SupportYes,
			},
			Usage: litellm.UsageCapabilities{
				InputTokens:      litellm.SupportYes,
				OutputTokens:     litellm.SupportYes,
				TotalTokens:      litellm.SupportYes,
				ReasoningTokens:  litellm.SupportPartial,
				CacheReadTokens:  litellm.SupportYes,
				CacheWriteTokens: litellm.SupportNo,
			},
		},
	}
	model := NewLiteLLMAdapter("m", mustClient(t, provider))
	caps := model.Capabilities()

	if caps.Provider != "capture" || caps.Model != "m" {
		t.Fatalf("identity = %s/%s, want capture/m", caps.Provider, caps.Model)
	}
	if caps.Thinking.Supported != SupportYes || caps.Thinking.Disable != SupportYes {
		t.Fatalf("thinking support = %+v", caps.Thinking)
	}
	if !caps.Thinking.SupportsEffort(agentcore.ThinkingMinimal) || !caps.Thinking.SupportsEffort(agentcore.ThinkingMax) {
		t.Fatalf("thinking efforts = %#v", caps.Thinking.Efforts)
	}
	if caps.Thinking.SupportsEffort(agentcore.ThinkingLevel("vendor-only")) {
		t.Fatalf("vendor-only effort leaked into agentcore capabilities: %#v", caps.Thinking.Efforts)
	}
	if caps.Tools.StrictSchema != SupportYes || !caps.Tools.RequiresAdjacency {
		t.Fatalf("tool capabilities = %+v", caps.Tools)
	}
	if caps.Structured.JSONSchema != SupportYes || caps.Structured.Strict != SupportPartial || !caps.Structured.PromptOnly {
		t.Fatalf("structured capabilities = %+v", caps.Structured)
	}
	if caps.Streaming.Usage != SupportPartial || caps.Streaming.IdleTimeout != SupportYes {
		t.Fatalf("streaming capabilities = %+v", caps.Streaming)
	}
	if caps.Usage.CacheReadTokens != SupportYes || caps.Usage.CacheWriteTokens != SupportNo {
		t.Fatalf("usage capabilities = %+v", caps.Usage)
	}
	if len(caps.Thinking.Notes) != 1 || caps.Thinking.Notes[0] != "budget varies by model" {
		t.Fatalf("thinking notes = %#v", caps.Thinking.Notes)
	}
}

func TestLiteLLMAdapterCapabilitiesFallback(t *testing.T) {
	model := NewLiteLLMAdapter("m", mustClient(t, &captureProvider{}))
	caps := model.Capabilities()
	if caps.Provider != "capture" || caps.Model != "m" {
		t.Fatalf("identity = %s/%s, want capture/m", caps.Provider, caps.Model)
	}
	if caps.Thinking.Supported != SupportUnknown || caps.Tools.Calls != SupportUnknown {
		t.Fatalf("fallback should be unknown support, got %+v / %+v", caps.Thinking, caps.Tools)
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

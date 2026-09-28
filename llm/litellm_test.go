package llm

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"reflect"
	"slices"
	"strings"
	"testing"

	"github.com/voocel/agentcore"
	"github.com/voocel/litellm"
	"github.com/voocel/litellm/provider/deepseek"
	"github.com/voocel/litellm/provider/mimo"
)

type captureProvider struct {
	lastReq  *litellm.Request
	chatFunc func(context.Context, *litellm.Request) (*litellm.Response, error)
}

func (p *captureProvider) Name() string { return "capture" }

func (p *captureProvider) Chat(ctx context.Context, req *litellm.Request) (*litellm.Response, error) {
	p.lastReq = req
	if p.chatFunc != nil {
		return p.chatFunc(ctx, req)
	}
	return &litellm.Response{Blocks: []litellm.Block{litellm.TextBlock{Text: "ok"}}}, nil
}

func (p *captureProvider) Stream(context.Context, *litellm.Request) (litellm.Stream, error) {
	return nil, nil
}

type captureStreamProvider struct {
	lastReq *litellm.Request
	events  []litellm.Event
}

func (p *captureStreamProvider) Name() string { return "capture" }

func (p *captureStreamProvider) Chat(context.Context, *litellm.Request) (*litellm.Response, error) {
	return nil, nil
}

func (p *captureStreamProvider) Stream(_ context.Context, req *litellm.Request) (litellm.Stream, error) {
	p.lastReq = req
	events := p.events
	if len(events) == 0 {
		events = []litellm.Event{
			litellm.BlockStart{Block: litellm.TextBlock{}},
			litellm.TextDelta{Text: "ok"},
			litellm.BlockEnd{},
			litellm.DoneEvent{FinishReason: litellm.FinishReasonStop, Provider: "capture", Model: req.Model},
		}
	}
	return &staticStream{
		events: events,
	}, nil
}

type staticStream struct {
	events []litellm.Event
}

func (s *staticStream) Next() (litellm.Event, error) {
	if len(s.events) == 0 {
		return nil, io.EOF
	}
	ev := s.events[0]
	s.events = s.events[1:]
	return ev, nil
}

func (s *staticStream) Close() error { return nil }

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

func TestLiteLLMAdapterTreatsAutoThinkingAsUnspecified(t *testing.T) {
	provider := &captureProvider{}
	model := NewLiteLLMAdapter("m", mustClient(t, provider))
	_, err := model.Generate(context.Background(), []agentcore.Message{agentcore.UserMsg("hi")}, nil, agentcore.WithThinking("auto"))
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if provider.lastReq == nil {
		t.Fatal("provider was not called")
	}
	if provider.lastReq.Thinking != nil {
		t.Fatalf("auto thinking should be omitted, got %#v", provider.lastReq.Thinking)
	}
}

func TestGenerateNormalizesMalformedToolArgumentsFromModel(t *testing.T) {
	provider := &captureProvider{}
	provider.chatFunc = func(context.Context, *litellm.Request) (*litellm.Response, error) {
		return &litellm.Response{
			Provider:     "capture",
			Model:        "m",
			FinishReason: litellm.FinishReasonToolCall,
			Blocks: []litellm.Block{
				litellm.ToolUseBlock{ID: "call_bad", Name: "lookup", Arguments: json.RawMessage(`{"q":`)},
			},
		}, nil
	}
	model := NewLiteLLMAdapter("m", mustClient(t, provider))

	resp, err := model.Generate(context.Background(), []agentcore.Message{agentcore.UserMsg("hi")}, nil)
	if err != nil {
		t.Fatalf("Generate returned error: %v", err)
	}
	calls := resp.Message.ToolCalls()
	if len(calls) != 1 {
		t.Fatalf("tool calls len = %d, want 1", len(calls))
	}
	if !calls[0].ArgsInvalid {
		t.Fatalf("ArgsInvalid = false, call = %+v", calls[0])
	}
	if got := string(calls[0].Args); got != "{}" {
		t.Fatalf("args = %q, want {}", got)
	}
	if calls[0].ArgsRawText != `{"q":` || calls[0].ArgsParseError == "" {
		t.Fatalf("missing malformed args diagnostics: %+v", calls[0])
	}
}

func TestGenerateReportsRefusalAsSafety(t *testing.T) {
	provider := &captureProvider{}
	provider.chatFunc = func(context.Context, *litellm.Request) (*litellm.Response, error) {
		return &litellm.Response{
			Provider:        "capture",
			Model:           "m",
			FinishReason:    litellm.FinishReasonSafety,
			FinishReasonRaw: "content_filter",
			Blocks:          []litellm.Block{litellm.Text("I can't help.")},
		}, nil
	}
	model := NewLiteLLMAdapter("m", mustClient(t, provider))
	resp, err := model.Generate(context.Background(), []agentcore.Message{agentcore.UserMsg("hi")}, nil)
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if resp.Message.StopReason != agentcore.StopReasonSafety || resp.Message.TextContent() != "I can't help." {
		t.Fatalf("stop/text = %q/%q", resp.Message.StopReason, resp.Message.TextContent())
	}
	if resp.Message.Metadata["finish_reason_raw"] != "content_filter" {
		t.Fatalf("metadata = %#v", resp.Message.Metadata)
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

func (p *capabilityProvider) Capabilities() litellm.Capabilities { return p.caps }

func TestLiteLLMAdapterCapabilities(t *testing.T) {
	tests := []struct {
		name string
		caps litellm.Capabilities
		want []agentcore.ThinkingLevel
	}{
		{"no thinking", litellm.Capabilities{}, []agentcore.ThinkingLevel{ThinkingAuto}},
		{"switch only", litellm.Capabilities{Thinking: true, DisableThinking: true}, []agentcore.ThinkingLevel{ThinkingAuto, agentcore.ThinkingOff}},
		{"effort without disable", litellm.Capabilities{Thinking: true, ThinkingEffort: true}, append([]agentcore.ThinkingLevel{ThinkingAuto}, ThinkingLevelOrder[1:]...)},
		{"all", litellm.Capabilities{Thinking: true, DisableThinking: true, ThinkingEffort: true}, append([]agentcore.ThinkingLevel{ThinkingAuto}, ThinkingLevelOrder...)},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tt.caps.ProviderOptions = []string{"prompt_cache_key"}
			model := NewLiteLLMAdapter("m", mustClient(t, &capabilityProvider{caps: tt.caps}))
			caps, ok := model.Capabilities()
			if !ok || caps.Thinking != tt.caps.Thinking || caps.ThinkingEffort != tt.caps.ThinkingEffort || !slices.Equal(caps.ProviderOptions, tt.caps.ProviderOptions) {
				t.Fatalf("caps = %+v, %v", caps, ok)
			}
			if got := ThinkingPolicyFor(model).Available; !slices.Equal(got, tt.want) {
				t.Fatalf("thinking levels = %v, want %v", got, tt.want)
			}
		})
	}
}

// A provider that declares nothing leaves every level to the vendor.
func TestLiteLLMAdapterCapabilitiesUnknown(t *testing.T) {
	model := NewLiteLLMAdapter("m", mustClient(t, &captureProvider{}))
	if _, ok := model.Capabilities(); ok {
		t.Fatal("undeclared capabilities reported as known")
	}
	if got := ThinkingPolicyFor(model).Available; len(got) != len(ThinkingLevelOrder)+1 {
		t.Fatalf("thinking levels = %v", got)
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
// compat stream opening and closing the tool block, then this adapter finalizing
// via normalizeArgs.
func TestGenerateStreamFinalizesArglessToolCall(t *testing.T) {
	sse := strings.Join([]string{
		`data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_9","function":{"name":"novel_context"}}]}}]}`,
		`data: {"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}`,
		`data: [DONE]`,
		``,
	}, "\n")
	provider, err := mimo.New(mimo.Config{
		APIKey:  "test",
		BaseURL: "https://compat.test/v1",
		HTTPClient: roundTripFunc(func(*http.Request) (*http.Response, error) {
			return &http.Response{
				StatusCode: http.StatusOK,
				Body:       io.NopCloser(strings.NewReader(sse)),
				Header:     make(http.Header),
			}, nil
		}),
	})
	if err != nil {
		t.Fatalf("mimo.New: %v", err)
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

func TestGenerateStreamMarksMalformedToolArgumentsInvalid(t *testing.T) {
	sse := strings.Join([]string{
		`data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_bad","function":{"name":"subagent","arguments":"{\"agent\":\"writer\","}}]}}]}`,
		`data: {"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}`,
		`data: [DONE]`,
		``,
	}, "\n")
	provider, err := deepseek.New(deepseek.Config{
		APIKey:  "test",
		BaseURL: "https://compat.test/v1",
		HTTPClient: roundTripFunc(func(*http.Request) (*http.Response, error) {
			return &http.Response{
				StatusCode: http.StatusOK,
				Body:       io.NopCloser(strings.NewReader(sse)),
				Header:     make(http.Header),
			}, nil
		}),
	})
	if err != nil {
		t.Fatalf("deepseek.New: %v", err)
	}
	model := NewLiteLLMAdapter("deepseek-v4-flash-free", mustClient(t, provider))
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
	if !tc.ArgsInvalid {
		t.Fatalf("ArgsInvalid = false, tool call = %+v", tc)
	}
	if got := string(tc.Args); got != "{}" {
		t.Fatalf("args = %q, want {}", got)
	}
	if tc.ArgsRawText == "" || tc.ArgsParseError == "" {
		t.Fatalf("missing malformed args diagnostics: %+v", tc)
	}
}

func TestGenerateStreamNormalizesMalformedHistoricalToolArguments(t *testing.T) {
	provider := &captureStreamProvider{}
	model := NewLiteLLMAdapter("m", mustClient(t, provider))
	msgs := []agentcore.Message{
		agentcore.UserMsg("hi"),
		{
			Role: agentcore.RoleAssistant,
			Content: []agentcore.ContentBlock{
				agentcore.ToolCallBlock(agentcore.ToolCall{
					ID:   "call_bad",
					Name: "subagent",
					Args: json.RawMessage(`{"agent":"writer",`),
				}),
			},
		},
		agentcore.ToolResultMsg("call_bad", json.RawMessage(`"invalid subagent params: unexpected end of JSON input"`), true),
		agentcore.UserMsg("continue"),
	}

	ch, err := model.GenerateStream(context.Background(), msgs, nil)
	if err != nil {
		t.Fatalf("GenerateStream rejected malformed historical args: %v", err)
	}
	for range ch {
	}

	if provider.lastReq == nil {
		t.Fatal("provider was not called")
	}
	assistant := provider.lastReq.Messages[1]
	if len(assistant.Blocks) != 1 {
		t.Fatalf("assistant block count = %d, want 1", len(assistant.Blocks))
	}
	call, ok := assistant.Blocks[0].(litellm.ToolUseBlock)
	if !ok {
		t.Fatalf("assistant block = %T, want ToolUseBlock", assistant.Blocks[0])
	}
	if got := string(call.Arguments); got != "{}" {
		t.Fatalf("historical args = %q, want {}", got)
	}
}

func TestGenerateStreamFinalMessageNormalizesToolArgumentsWithoutDoneEvent(t *testing.T) {
	provider := &captureStreamProvider{
		events: []litellm.Event{
			litellm.BlockStart{Block: litellm.ToolUseBlock{ID: "call_bad", Name: "subagent"}},
			litellm.ToolUseDelta{Arguments: `{"agent":"writer",`},
			litellm.BlockEnd{},
			litellm.DoneEvent{FinishReason: litellm.FinishReasonToolCall, Provider: "capture", Model: "m"},
		},
	}
	model := NewLiteLLMAdapter("m", mustClient(t, provider))
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

	calls := final.ToolCalls()
	if len(calls) != 1 {
		t.Fatalf("tool call count = %d, want 1", len(calls))
	}
	if !calls[0].ArgsInvalid {
		t.Fatalf("ArgsInvalid = false, call = %+v", calls[0])
	}
	if got := string(calls[0].Args); got != "{}" {
		t.Fatalf("args = %q, want {}", got)
	}
	if calls[0].ArgsRawText == "" || calls[0].ArgsParseError == "" {
		t.Fatalf("missing malformed args diagnostics: %+v", calls[0])
	}
}

func TestGenerateStreamReportsRefusalAsSafety(t *testing.T) {
	provider := &captureStreamProvider{events: []litellm.Event{
		litellm.BlockStart{Block: litellm.TextBlock{}},
		litellm.TextDelta{Text: "I can't help."},
		litellm.BlockEnd{},
		litellm.DoneEvent{FinishReason: litellm.FinishReasonSafety, FinishReasonRaw: "completed", Provider: "capture", Model: "m"},
	}}
	model := NewLiteLLMAdapter("m", mustClient(t, provider))
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
	if final.StopReason != agentcore.StopReasonSafety || final.TextContent() != "I can't help." {
		t.Fatalf("stop/text = %q/%q", final.StopReason, final.TextContent())
	}
	if final.Metadata["finish_reason_raw"] != "completed" {
		t.Fatalf("metadata = %#v", final.Metadata)
	}
}

// TestGenerateStreamAttributesInterleavedToolCallDeltas asserts each toolcall
// delta carries its call ID, even for continuation chunks keyed by index only.
func TestGenerateStreamAttributesInterleavedToolCallDeltas(t *testing.T) {
	sse := strings.Join([]string{
		`data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_a","function":{"name":"write_a","arguments":"{\"a\":"}}]}}]}`,
		`data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"id":"call_b","function":{"name":"write_b","arguments":"{\"b\":"}}]}}]}`,
		`data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"1}"}}]}}]}`,
		`data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"function":{"arguments":"2}"}}]}}]}`,
		`data: {"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}`,
		`data: [DONE]`,
		``,
	}, "\n")
	provider, err := mimo.New(mimo.Config{
		APIKey:  "test",
		BaseURL: "https://compat.test/v1",
		HTTPClient: roundTripFunc(func(*http.Request) (*http.Response, error) {
			return &http.Response{
				StatusCode: http.StatusOK,
				Body:       io.NopCloser(strings.NewReader(sse)),
				Header:     make(http.Header),
			}, nil
		}),
	})
	if err != nil {
		t.Fatalf("mimo.New: %v", err)
	}
	model := NewLiteLLMAdapter("mimo-v2.5", mustClient(t, provider))
	ch, err := model.GenerateStream(context.Background(), []agentcore.Message{agentcore.UserMsg("hi")}, nil)
	if err != nil {
		t.Fatalf("GenerateStream: %v", err)
	}
	type attributed struct{ id, delta string }
	var got []attributed
	for ev := range ch {
		if ev.Type == agentcore.StreamEventError {
			t.Fatalf("stream error: %v", ev.Err)
		}
		if ev.Type == agentcore.StreamEventToolCallDelta {
			got = append(got, attributed{ev.ToolID, ev.Delta})
		}
	}
	want := []attributed{
		{"call_a", `{"a":`},
		{"call_b", `{"b":`},
		{"call_a", "1}"},
		{"call_b", "2}"},
	}
	if len(got) != len(want) {
		t.Fatalf("delta events = %+v, want %+v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("delta %d = %+v, want %+v", i, got[i], want[i])
		}
	}
}

// Streamed content matches Generate's conversion: one content block per
// litellm block, in order, with replay state.
func TestGenerateStreamMatchesResponseContent(t *testing.T) {
	redacted := &litellm.ProviderState{Provider: "capture", Data: json.RawMessage(`{"data":"x"}`)}
	signed := &litellm.ProviderState{Provider: "capture", Data: json.RawMessage(`{"signature":"sig"}`)}
	events := []litellm.Event{
		litellm.BlockStart{Index: 0, Block: litellm.ReasoningBlock{}},
		litellm.BlockStart{Index: 1, Block: litellm.ReasoningBlock{}},
		litellm.ReasoningDelta{Index: 1, Text: "plan"},
		litellm.BlockStart{Index: 2, Block: litellm.TextBlock{Text: "a"}},
		litellm.TextDelta{Index: 2, Text: "b"},
		litellm.BlockEnd{Index: 0, Block: litellm.ReasoningBlock{State: redacted}},
		litellm.BlockStart{Index: 3, Block: litellm.ToolUseBlock{ID: "call_1", Name: "f", State: signed}},
		litellm.ToolUseDelta{Index: 3, Arguments: `{"q":1}`},
		litellm.BlockEnd{Index: 1},
		litellm.BlockEnd{Index: 2},
		litellm.BlockEnd{Index: 3},
		litellm.DoneEvent{FinishReason: litellm.FinishReasonToolCall, Provider: "capture", Model: "m"},
	}
	model := NewLiteLLMAdapter("m", mustClient(t, &captureStreamProvider{events: events}))
	ch, err := model.GenerateStream(context.Background(), []agentcore.Message{agentcore.UserMsg("hi")}, nil)
	if err != nil {
		t.Fatal(err)
	}
	var final agentcore.Message
	var ends []agentcore.StreamEventType
	for ev := range ch {
		switch ev.Type {
		case agentcore.StreamEventError:
			t.Fatal(ev.Err)
		case agentcore.StreamEventThinkingEnd, agentcore.StreamEventTextEnd, agentcore.StreamEventToolCallEnd:
			ends = append(ends, ev.Type)
		case agentcore.StreamEventDone:
			final = ev.Message
		}
	}
	want := convertResponseContent(&litellm.Response{Blocks: []litellm.Block{
		litellm.ReasoningBlock{State: redacted},
		litellm.ReasoningBlock{Text: "plan"},
		litellm.TextBlock{Text: "ab"},
		litellm.ToolUseBlock{ID: "call_1", Name: "f", Arguments: json.RawMessage(`{"q":1}`), State: signed},
	}})
	got, _ := json.Marshal(final.Content)
	wantJSON, _ := json.Marshal(want)
	if string(got) != string(wantJSON) || final.StopReason != agentcore.StopReasonToolUse {
		t.Fatalf("content = %s\nwant %s", got, wantJSON)
	}
	if !slices.Equal(ends, []agentcore.StreamEventType{agentcore.StreamEventThinkingEnd, agentcore.StreamEventThinkingEnd, agentcore.StreamEventTextEnd, agentcore.StreamEventToolCallEnd}) {
		t.Fatalf("end events = %v", ends)
	}
}

func TestWithDefaultHeaderKeepsExplicitHeader(t *testing.T) {
	if got := withDefaultHeader(nil, "anthropic-beta", "b"); got["anthropic-beta"] != "b" {
		t.Fatalf("headers = %v", got)
	}
	explicit := map[string]string{"Anthropic-Beta": "user"}
	if got := withDefaultHeader(explicit, "anthropic-beta", "b"); len(got) != 1 || got["Anthropic-Beta"] != "user" {
		t.Fatalf("headers = %v", got)
	}
}

// Replay state survives persistence and reaches the provider unchanged.
func TestProviderStateRoundTrip(t *testing.T) {
	state := func(data string) *litellm.ProviderState {
		return &litellm.ProviderState{Provider: "p", Model: "m", Data: json.RawMessage(data)}
	}
	blocks := []litellm.Block{
		litellm.ReasoningBlock{State: state(`{"type":"redacted_thinking","data":"x"}`)},
		litellm.TextBlock{Text: "a", State: state(`{"id":"msg_1"}`)},
		litellm.ToolUseBlock{ID: "call_1", Name: "f", Arguments: json.RawMessage(`{}`), State: state(`{"thoughtSignature":"s"}`)},
	}
	data, err := json.Marshal(convertResponse(&litellm.Response{Blocks: blocks}))
	if err != nil {
		t.Fatal(err)
	}
	var restored agentcore.Message
	if err := json.Unmarshal(data, &restored); err != nil {
		t.Fatal(err)
	}
	if got := convertMessages([]agentcore.Message{restored}); len(got) != 1 || !reflect.DeepEqual(got[0].Blocks, blocks) {
		t.Fatalf("replayed %#v\nwant %#v", got, blocks)
	}
}

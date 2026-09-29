// Package llm adapts LLM providers to the [agentcore.ChatModel] interface. It
// wraps litellm to reach OpenAI, Anthropic, Gemini, and other backends, and
// classifies provider errors onto agentcore's retry and overflow contracts.
// Construct a model with [NewModel].
package llm

import (
	"context"
	"errors"
	"fmt"
	"maps"
	"slices"
	"time"

	"github.com/voocel/agentcore"
	"github.com/voocel/litellm"
	"github.com/voocel/litellm/catalog"
	"github.com/voocel/litellm/providers"
)

// LiteLLMAdapter adapts a litellm.Client to agentcore.ChatModel. Unset
// request defaults are omitted from the wire, leaving the vendor default.
type LiteLLMAdapter struct {
	client      *litellm.Client
	model       string
	maxTokens   *int
	temperature *float64
	topP        *float64
	stop        []string
	options     litellm.ProviderOptions
	timeout     time.Duration
	pricing     *catalog.Pricing
}

// ModelOption configures NewModel and NewLiteLLMAdapter.
type ModelOption func(*modelConfig)

type modelConfig struct {
	clientOpts  []litellm.ClientOption
	maxTokens   *int
	temperature *float64
	topP        *float64
	stop        []string
	extra       map[string]any
	timeout     time.Duration
	pricing     *catalog.Pricing
}

// WithClientOptions forwards litellm ClientOptions, such as
// litellm.WithObservers or litellm.WithStreamIdleTimeout, to the client
// NewModel builds.
func WithClientOptions(opts ...litellm.ClientOption) ModelOption {
	return func(c *modelConfig) { c.clientOpts = append(c.clientOpts, opts...) }
}

// WithMaxTokens caps output tokens; agentcore.WithMaxTokens overrides it per
// call. Unset, no cap is sent and the vendor default applies, but Anthropic
// requires one: its calls fail until a cap is set here or per call. The
// catalog's Model.MaxOutputTokens holds the limit of listed models.
func WithMaxTokens(n int) ModelOption { return func(c *modelConfig) { c.maxTokens = &n } }

func WithTemperature(t float64) ModelOption { return func(c *modelConfig) { c.temperature = &t } }
func WithTopP(p float64) ModelOption        { return func(c *modelConfig) { c.topP = &p } }
func WithStop(stop ...string) ModelOption   { return func(c *modelConfig) { c.stop = stop } }

// WithExtra sets provider options sent with every request: top-level fields
// of the vendor's request body, such as min_p or chat_template_kwargs. An
// object merges into a field the adapter also generates. OpenAI-compatible
// providers pass every key through; the others reject keys they do not list.
func WithExtra(extra map[string]any) ModelOption {
	return func(c *modelConfig) { c.extra = extra }
}

// WithRequestTimeout bounds each call, streaming included. Zero leaves the
// deadline to the caller's context.
func WithRequestTimeout(d time.Duration) ModelOption {
	return func(c *modelConfig) { c.timeout = d }
}

// WithPricing prices each call's usage into Usage.Cost; catalog.Catalog
// holds the rates of listed models.
func WithPricing(p catalog.Pricing) ModelOption {
	return func(c *modelConfig) { c.pricing = &p }
}

// NewModel builds the provider called name, one of providers.Names, and
// adapts it as a ChatModel. agentcore.WithAPIKey overrides conn's key for one
// call. Anthropic models need an output cap; see WithMaxTokens.
func NewModel(name, model string, conn providers.Config, opts ...ModelOption) (*LiteLLMAdapter, error) {
	cfg := resolveModelConfig(opts)
	p, err := providers.New(name, withCallKey(conn))
	if err != nil {
		return nil, fmt.Errorf("llm: %s: %w", name, err)
	}
	client, err := litellm.New(p, cfg.clientOpts...)
	if err != nil {
		return nil, fmt.Errorf("llm: %s client: %w", name, err)
	}
	return newAdapter(client, model, cfg)
}

// NewLiteLLMAdapter wraps an existing litellm.Client, to reuse a Client or
// inject a custom Provider. The client is already built, so
// WithClientOptions does not apply.
func NewLiteLLMAdapter(model string, client *litellm.Client, opts ...ModelOption) (*LiteLLMAdapter, error) {
	cfg := resolveModelConfig(opts)
	if cfg.clientOpts != nil {
		return nil, errors.New("llm: WithClientOptions applies to NewModel only")
	}
	return newAdapter(client, model, cfg)
}

type apiKeyOverrideKey struct{}

func contextWithAPIKey(ctx context.Context, key string) context.Context {
	if key == "" {
		return ctx
	}
	return context.WithValue(ctx, apiKeyOverrideKey{}, key)
}

// withCallKey resolves the key per request, so a per-call key set by
// agentcore.WithAPIKey takes precedence over conn's.
func withCallKey(conn providers.Config) providers.Config {
	resolve := conn.APIKeyFunc
	if resolve == nil {
		key := conn.APIKey
		resolve = func(context.Context) (string, error) { return key, nil }
	}
	conn.APIKeyFunc = func(ctx context.Context) (string, error) {
		if key, ok := ctx.Value(apiKeyOverrideKey{}).(string); ok {
			return key, nil
		}
		return resolve(ctx)
	}
	return conn
}

func resolveModelConfig(opts []ModelOption) modelConfig {
	var cfg modelConfig
	for _, opt := range opts {
		opt(&cfg)
	}
	return cfg
}

func newAdapter(client *litellm.Client, model string, cfg modelConfig) (*LiteLLMAdapter, error) {
	options, err := litellm.NewProviderOptions(cfg.extra)
	if err != nil {
		return nil, fmt.Errorf("llm: extra: %w", err)
	}
	return &LiteLLMAdapter{
		client:      client,
		model:       model,
		maxTokens:   cfg.maxTokens,
		temperature: cfg.temperature,
		topP:        cfg.topP,
		stop:        cfg.stop,
		options:     options,
		timeout:     cfg.timeout,
		pricing:     cfg.pricing,
	}, nil
}

// ProviderName implements agentcore.ProviderNamer.
func (l *LiteLLMAdapter) ProviderName() string { return l.client.ProviderName() }

// ModelName implements agentcore.ModelNamer.
func (l *LiteLLMAdapter) ModelName() string { return l.model }

// SupportsTools reports true: every litellm provider carries tool calls.
func (l *LiteLLMAdapter) SupportsTools() bool { return true }

// Capabilities returns the provider adapter's static facts exposed by litellm.
func (l *LiteLLMAdapter) Capabilities() (Capabilities, bool) {
	caps, ok := l.client.Capabilities()
	return fromLiteLLMCapabilities(caps), ok
}

// Generate produces a synchronous response.
func (l *LiteLLMAdapter) Generate(ctx context.Context, messages []agentcore.Message, tools []agentcore.ToolSpec, opts ...agentcore.CallOption) (*agentcore.LLMResponse, error) {
	ctx, cancel := l.withTimeout(ctx)
	defer cancel()
	ctx, req, err := l.newRequest(ctx, messages, tools, opts)
	if err != nil {
		return nil, err
	}
	resp, err := l.client.Chat(ctx, *req)
	if err != nil {
		return nil, wrapProviderError(err)
	}
	msg := agentcore.Message{Role: agentcore.RoleAssistant, Content: convertBlocks(resp.Blocks)}
	l.finish(&msg, resp)
	return &agentcore.LLMResponse{Message: msg}, nil
}

// GenerateStream produces a streaming response with fine-grained events.
func (l *LiteLLMAdapter) GenerateStream(ctx context.Context, messages []agentcore.Message, tools []agentcore.ToolSpec, opts ...agentcore.CallOption) (<-chan agentcore.StreamEvent, error) {
	ctx, cancel := l.withTimeout(ctx)
	ctx, req, err := l.newRequest(ctx, messages, tools, opts)
	if err != nil {
		cancel()
		return nil, err
	}
	stream, err := l.client.Stream(ctx, *req)
	if err != nil {
		cancel()
		return nil, wrapProviderError(err)
	}
	events := make(chan agentcore.StreamEvent, 100)
	go func() {
		defer close(events)
		defer cancel()
		defer stream.Close()
		l.forward(stream, events)
	}()
	return events, nil
}

func (l *LiteLLMAdapter) withTimeout(ctx context.Context) (context.Context, context.CancelFunc) {
	if l.timeout <= 0 {
		return ctx, func() {}
	}
	return context.WithTimeout(ctx, l.timeout)
}

// newRequest builds the request for one call from the model defaults and the
// call options, and returns ctx carrying the per-call API key.
func (l *LiteLLMAdapter) newRequest(ctx context.Context, messages []agentcore.Message, tools []agentcore.ToolSpec, opts []agentcore.CallOption) (context.Context, *litellm.Request, error) {
	call := agentcore.ResolveCallConfig(opts)
	req := &litellm.Request{
		Model:           l.model,
		Messages:        convertMessages(messages),
		MaxTokens:       l.maxTokens,
		Temperature:     l.temperature,
		TopP:            l.topP,
		Stop:            l.stop,
		Thinking:        convertThinking(call.ThinkingLevel, call.ThinkingBudget),
		ProviderOptions: maps.Clone(l.options),
	}
	if call.MaxTokens > 0 {
		req.MaxTokens = &call.MaxTokens
	}

	caps, _ := l.client.Capabilities()
	if err := setHint(req, caps, "session_id", call.SessionID); err != nil {
		return ctx, nil, err
	}
	if err := setHint(req, caps, "prompt_cache_key", call.PromptCacheKey); err != nil {
		return ctx, nil, err
	}

	var err error
	if call.ToolChoice != nil {
		if req.ToolChoice, err = convertToolChoice(call.ToolChoice); err != nil {
			return ctx, nil, err
		}
	}
	if req.ResponseFormat, err = convertResponseFormat(call.ResponseFormat); err != nil {
		return ctx, nil, err
	}
	if req.Tools, err = convertTools(tools); err != nil {
		return ctx, nil, err
	}
	return contextWithAPIKey(ctx, call.APIKey), req, nil
}

// setHint sets an optional provider option, such as session or cache
// routing, only where the provider lists it: the others reject unknown keys.
func setHint(req *litellm.Request, caps litellm.Capabilities, key, value string) error {
	if value == "" || !slices.Contains(caps.ProviderOptions, key) {
		return nil
	}
	return req.ProviderOptions.Set(key, value)
}

// finish fills the facts Generate and GenerateStream take from the response.
func (l *LiteLLMAdapter) finish(msg *agentcore.Message, resp *litellm.Response) {
	msg.StopReason = mapStopReason(resp.FinishReason)
	msg.Usage = l.usage(resp)
	msg.Metadata = responseMetadata(resp)
}

// usage maps the reported token counts, unknown ones as zero, and prices
// them; nil when the provider reported none.
func (l *LiteLLMAdapter) usage(resp *litellm.Response) *agentcore.Usage {
	u := resp.Usage
	if !u.HasTokens() {
		return nil
	}
	input, _ := u.Input()
	output, _ := u.Output()
	cacheRead, _ := u.CacheRead()
	cacheWrite, _ := u.CacheWrite()
	total, _ := u.Total()
	usage := &agentcore.Usage{
		Provider:    resp.Provider,
		Model:       resp.Model,
		Input:       input,
		Output:      output,
		CacheRead:   cacheRead,
		CacheWrite:  cacheWrite,
		TotalTokens: total,
	}
	if l.pricing == nil {
		return usage
	}
	// Counts the rates cannot price, such as an unreported cache count with
	// its own rate, leave the cost unknown.
	if cost, err := l.pricing.Cost(u); err == nil {
		usage.Cost = &agentcore.Cost{
			Input:      cost.Input,
			Output:     cost.Output,
			CacheRead:  cost.CacheRead,
			CacheWrite: cost.CacheWrite,
			Total:      cost.Total,
		}
	}
	return usage
}

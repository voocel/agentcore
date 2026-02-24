package llm

import "github.com/voocel/agentcore"

// ModelPricing defines per-token cost rates in USD.
// Set rates to 0 for categories that don't apply.
type ModelPricing struct {
	InputPerToken      float64 `json:"input_per_token"`
	OutputPerToken     float64 `json:"output_per_token"`
	CacheReadPerToken  float64 `json:"cache_read_per_token"`
	CacheWritePerToken float64 `json:"cache_write_per_token"`
}

// CalculateCost computes the monetary cost from pricing rates and token usage.
// Returns nil if pricing or usage is nil.
func CalculateCost(pricing *ModelPricing, usage *agentcore.Usage) *agentcore.Cost {
	if pricing == nil || usage == nil {
		return nil
	}
	c := &agentcore.Cost{
		Input:      float64(usage.Input) * pricing.InputPerToken,
		Output:     float64(usage.Output) * pricing.OutputPerToken,
		CacheRead:  float64(usage.CacheRead) * pricing.CacheReadPerToken,
		CacheWrite: float64(usage.CacheWrite) * pricing.CacheWritePerToken,
	}
	c.Total = c.Input + c.Output + c.CacheRead + c.CacheWrite
	return c
}

// ModelInfo contains basic model metadata.
type ModelInfo struct {
	Name         string        `json:"name"`
	Provider     string        `json:"provider"`
	Version      string        `json:"version"`
	MaxTokens    int           `json:"max_tokens"`
	ContextSize  int           `json:"context_size"`
	Capabilities []string      `json:"capabilities"`
	Pricing      *ModelPricing `json:"pricing,omitempty"`
}

// ModelCapability defines capability identifiers.
type ModelCapability string

const (
	CapabilityChat         ModelCapability = "chat"
	CapabilityCompletion   ModelCapability = "completion"
	CapabilityToolCalling  ModelCapability = "tool_calling"
	CapabilityStreaming    ModelCapability = "streaming"
	CapabilityMultimodal   ModelCapability = "multimodal"
	CapabilityFunctionCall ModelCapability = "function_call"
)

// GenerationConfig defines sampling and length control parameters.
type GenerationConfig struct {
	Temperature      float64  `json:"temperature"`
	TopP             float64  `json:"top_p"`
	TopK             int      `json:"top_k"`
	MaxTokens        int      `json:"max_tokens"`
	StopSequences    []string `json:"stop_sequences"`
	PresencePenalty  float64  `json:"presence_penalty"`
	FrequencyPenalty float64  `json:"frequency_penalty"`
	Seed             *int64   `json:"seed"`
}

var DefaultGenerationConfig = &GenerationConfig{
	Temperature:      0.7,
	TopP:             0.9,
	TopK:             0,
	MaxTokens:        65536,
	StopSequences:    []string{},
	PresencePenalty:  0.0,
	FrequencyPenalty: 0.0,
	Seed:             nil,
}

// BaseModel provides common model metadata and capability checks.
type BaseModel struct {
	info   ModelInfo
	config *GenerationConfig
}

func NewBaseModel(info ModelInfo, config *GenerationConfig) *BaseModel {
	if config == nil {
		config = DefaultGenerationConfig
	}
	return &BaseModel{info: info, config: config}
}

func (m *BaseModel) Info() ModelInfo              { return m.info }
func (m *BaseModel) GetConfig() *GenerationConfig { return m.config }

func (m *BaseModel) SupportsCapability(capability ModelCapability) bool {
	for _, c := range m.info.Capabilities {
		if c == string(capability) {
			return true
		}
	}
	return false
}

func (m *BaseModel) SupportsTools() bool {
	return m.SupportsCapability(CapabilityToolCalling) || m.SupportsCapability(CapabilityFunctionCall)
}

func (m *BaseModel) SupportsStreaming() bool {
	return m.SupportsCapability(CapabilityStreaming)
}


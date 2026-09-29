package context

import (
	"testing"

	"github.com/voocel/agentcore"
)

// Input already counts cache reads and writes; adding CacheWrite again would
// inflate the context size and trigger compaction early.
func TestCalculateContextTokensCountsInputOnce(t *testing.T) {
	if got := calculateContextTokens(&agentcore.Usage{Input: 100, CacheRead: 60, CacheWrite: 30, TotalTokens: 110}); got != 100 {
		t.Fatalf("context tokens = %d, want 100", got)
	}
	if got := calculateContextTokens(&agentcore.Usage{TotalTokens: 42}); got != 42 {
		t.Fatalf("context tokens = %d, want the total fallback 42", got)
	}
}

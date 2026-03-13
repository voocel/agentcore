package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"regexp"
	"strings"
	"sync"

	"github.com/voocel/agentcore"
)

// ToolSearchTool provides LLM-driven discovery of deferred tools.
// It implements agentcore.Tool, agentcore.ContentTool, and agentcore.DeferFilter.
//
// When called, it returns tool_reference content blocks that instruct the API
// server to load the referenced deferred tool schemas into the LLM context.
type ToolSearchTool struct {
	entries       []toolSearchEntry
	deferredNames map[string]bool
	activated     map[string]bool
	mu            sync.RWMutex
}

type toolSearchEntry struct {
	Name        string
	Description string
	ParamNames  []string
}

// NewToolSearchTool creates a ToolSearchTool that defers the given tools.
// Deferred tools are sent to the API with defer_loading: true and only
// loaded into LLM context when discovered via tool_reference blocks.
func NewToolSearchTool(deferred ...agentcore.Tool) *ToolSearchTool {
	entries := make([]toolSearchEntry, 0, len(deferred))
	names := make(map[string]bool, len(deferred))
	for _, t := range deferred {
		name := t.Name()
		names[name] = true
		entry := toolSearchEntry{
			Name:        name,
			Description: t.Description(),
		}
		if schema := t.Schema(); schema != nil {
			if props, ok := schema["properties"].(map[string]any); ok {
				for k := range props {
					entry.ParamNames = append(entry.ParamNames, k)
				}
			}
		}
		entries = append(entries, entry)
	}
	return &ToolSearchTool{
		entries:       entries,
		deferredNames: names,
		activated:     make(map[string]bool),
	}
}

func (t *ToolSearchTool) Name() string { return "tool_search" }

func (t *ToolSearchTool) Description() string {
	return "Fetches full schema definitions for deferred tools so they can be called. " +
		"Deferred tools appear by name in <available-deferred-tools> messages. " +
		"Until fetched, only the name is known — there is no parameter schema, " +
		"so the tool cannot be invoked. This tool takes a query, matches it against " +
		"the deferred tool list, and returns the matched tools' complete JSONSchema " +
		"definitions. Query modes: \"select:Name1,Name2\" for exact selection, " +
		"\"/regex_pattern/\" for regex matching, or plain keywords for scored search."
}

func (t *ToolSearchTool) Schema() map[string]any {
	return map[string]any{
		"type": "object",
		"properties": map[string]any{
			"query": map[string]any{
				"type":        "string",
				"description": "Query to find deferred tools. Use \"select:<tool_name>\" for direct selection, or keywords to search.",
			},
			"max_results": map[string]any{
				"type":        "number",
				"description": "Maximum number of results to return (default: 5)",
				"default":     5,
			},
		},
		"required":             []string{"query", "max_results"},
		"additionalProperties": false,
	}
}

type toolSearchArgs struct {
	Query      string `json:"query"`
	MaxResults int    `json:"max_results"`
}

// ExecuteContent implements agentcore.ContentTool.
// Returns tool_reference content blocks for matched tools.
func (t *ToolSearchTool) ExecuteContent(_ context.Context, args json.RawMessage) ([]agentcore.ContentBlock, error) {
	var a toolSearchArgs
	if err := json.Unmarshal(args, &a); err != nil {
		return nil, fmt.Errorf("invalid arguments: %w", err)
	}
	if a.MaxResults <= 0 {
		a.MaxResults = 5
	}

	matches := t.search(a.Query, a.MaxResults)

	t.mu.Lock()
	for _, name := range matches {
		t.activated[name] = true
	}
	t.mu.Unlock()

	if len(matches) == 0 {
		return []agentcore.ContentBlock{agentcore.TextBlock("No matching tools found.")}, nil
	}

	blocks := make([]agentcore.ContentBlock, 0, len(matches)+1)
	for _, name := range matches {
		blocks = append(blocks, agentcore.ToolRefBlock(name))
	}
	blocks = append(blocks, agentcore.TextBlock("Tool loaded."))
	return blocks, nil
}

// Execute implements agentcore.Tool. Delegates to ExecuteContent and returns
// a text summary. In practice, the agent loop uses ExecuteContent (ContentTool)
// and this method serves as a fallback.
func (t *ToolSearchTool) Execute(ctx context.Context, args json.RawMessage) (json.RawMessage, error) {
	blocks, err := t.ExecuteContent(ctx, args)
	if err != nil {
		return nil, err
	}
	var texts []string
	for _, b := range blocks {
		if b.Type == agentcore.ContentText {
			texts = append(texts, b.Text)
		}
	}
	return json.Marshal(strings.Join(texts, "\n"))
}

// IsDeferred implements agentcore.DeferFilter.
// Returns true for deferred tools not yet activated (should be excluded from API).
func (t *ToolSearchTool) IsDeferred(name string) bool {
	t.mu.RLock()
	defer t.mu.RUnlock()
	if !t.deferredNames[name] {
		return false
	}
	return !t.activated[name]
}

// WasDeferred implements agentcore.DeferFilter.
// Returns true if the tool was originally in the deferred set (regardless of
// activation state). Activated deferred tools are sent with defer_loading: true.
func (t *ToolSearchTool) WasDeferred(name string) bool {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return t.deferredNames[name]
}

// DeferredNames returns the names of all deferred (not yet activated) tools.
func (t *ToolSearchTool) DeferredNames() []string {
	t.mu.RLock()
	defer t.mu.RUnlock()
	var names []string
	for _, e := range t.entries {
		if !t.activated[e.Name] {
			names = append(names, e.Name)
		}
	}
	return names
}

// search finds matching tool entries by query.
func (t *ToolSearchTool) search(query string, maxResults int) []string {
	// "select:Name1,Name2" — exact match by name
	if after, ok := strings.CutPrefix(query, "select:"); ok {
		names := strings.Split(after, ",")
		var matched []string
		t.mu.RLock()
		for _, n := range names {
			n = strings.TrimSpace(n)
			if t.deferredNames[n] {
				matched = append(matched, n)
			}
		}
		t.mu.RUnlock()
		return matched
	}

	// "/pattern/" — regex matching against name + description
	if strings.HasPrefix(query, "/") && strings.HasSuffix(query, "/") && len(query) > 2 {
		pattern := query[1 : len(query)-1]
		re, err := regexp.Compile(pattern)
		if err == nil {
			return t.searchRegex(re, maxResults)
		}
		// Invalid regex: fall through to keyword search
	}

	// Keyword search: score each entry by how many query terms match.
	terms := strings.Fields(strings.ToLower(query))
	if len(terms) == 0 {
		return nil
	}

	type scored struct {
		name  string
		score int
	}
	t.mu.RLock()
	var results []scored
	for _, e := range t.entries {
		s := scoreEntry(e, terms)
		if s > 0 {
			results = append(results, scored{name: e.Name, score: s})
		}
	}
	t.mu.RUnlock()

	// Sort by score descending (simple insertion sort — small N).
	for i := 1; i < len(results); i++ {
		for j := i; j > 0 && results[j].score > results[j-1].score; j-- {
			results[j], results[j-1] = results[j-1], results[j]
		}
	}

	matched := make([]string, 0, min(maxResults, len(results)))
	for i := range results {
		if i >= maxResults {
			break
		}
		matched = append(matched, results[i].name)
	}
	return matched
}

// searchRegex matches entries where name or description matches the regex.
func (t *ToolSearchTool) searchRegex(re *regexp.Regexp, maxResults int) []string {
	t.mu.RLock()
	defer t.mu.RUnlock()

	var matched []string
	for _, e := range t.entries {
		if re.MatchString(e.Name) || re.MatchString(e.Description) {
			matched = append(matched, e.Name)
			if len(matched) >= maxResults {
				break
			}
		}
	}
	return matched
}

// scoreEntry scores a tool entry against search terms.
// Name match scores higher than description match.
func scoreEntry(e toolSearchEntry, terms []string) int {
	nameLower := strings.ToLower(e.Name)
	descLower := strings.ToLower(e.Description)
	paramsLower := strings.ToLower(strings.Join(e.ParamNames, " "))

	score := 0
	for _, term := range terms {
		if strings.Contains(nameLower, term) {
			score += 3
		}
		if strings.Contains(descLower, term) {
			score += 2
		}
		if strings.Contains(paramsLower, term) {
			score += 1
		}
	}
	return score
}

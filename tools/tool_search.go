package tools

import (
	"cmp"
	"context"
	"encoding/json"
	"fmt"
	"regexp"
	"slices"
	"strings"

	"github.com/voocel/agentcore"
	"github.com/voocel/litellm"
)

// Defer puts tools behind the tool_search tool: it returns tool_search
// followed by the tools, marked Deferred. A search returns references to the
// tools it found, which offer them from then on (see Tool.Deferred).
//
// The definition of tool_search does not depend on the tools, so a tool
// deferred later leaves the requests made before intact. The model learns
// the names to search for from the application, such as in a message.
func Defer(tools []agentcore.Tool) []agentcore.Tool {
	t := &toolSearchTool{}
	out := make([]agentcore.Tool, 0, len(tools)+1)
	for _, d := range tools {
		entry := toolSearchEntry{Name: d.Name, Description: d.Description}
		if props, ok := d.Schema["properties"].(map[string]any); ok {
			for k := range props {
				entry.ParamNames = append(entry.ParamNames, k)
			}
		}
		t.entries = append(t.entries, entry)
		d.Deferred = true
		out = append(out, d)
	}
	search := agentcore.Tool{
		Name:  "tool_search",
		Label: "Search Tools",
		Description: "Fetches full schema definitions for deferred tools so they can be called. " +
			"Until fetched, only a deferred tool's name is known — there is no parameter schema, " +
			"so the tool cannot be invoked. This tool takes a query, matches it against " +
			"the deferred tools, and loads the matched ones. Query modes: \"select:Name1,Name2\" for exact selection, " +
			"\"/regex_pattern/\" for regex matching, or plain keywords for scored search.",
		Schema: map[string]any{
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
			"required":             []string{"query"},
			"additionalProperties": false,
		},
		Parallel: true,
		Run:      t.run,
	}
	return append([]agentcore.Tool{search}, out...)
}

type toolSearchTool struct {
	entries []toolSearchEntry
}

type toolSearchEntry struct {
	Name        string
	Description string
	ParamNames  []string
}

type toolSearchArgs struct {
	Query      string `json:"query"`
	MaxResults int    `json:"max_results"`
}

// run returns references to the matched tools.
func (t *toolSearchTool) run(_ context.Context, args json.RawMessage) (agentcore.Result, error) {
	var a toolSearchArgs
	if err := json.Unmarshal(args, &a); err != nil {
		return agentcore.Result{}, fmt.Errorf("invalid arguments: %w", err)
	}
	if a.MaxResults <= 0 {
		a.MaxResults = 5
	}

	matches := t.search(a.Query, a.MaxResults)
	if len(matches) == 0 {
		return agentcore.TextResult("No matching tools found."), nil
	}
	blocks := make([]litellm.Block, 0, len(matches)+1)
	for _, name := range matches {
		blocks = append(blocks, litellm.ToolReferenceBlock{ToolName: name})
	}
	blocks = append(blocks, litellm.Text("Tool loaded."))
	return agentcore.Result{Content: blocks}, nil
}

// search finds matching tool entries by query.
func (t *toolSearchTool) search(query string, maxResults int) []string {
	// "select:Name1,Name2" — exact match by name
	if after, ok := strings.CutPrefix(query, "select:"); ok {
		names := strings.Split(after, ",")
		var matched []string
		for _, n := range names {
			n = strings.TrimSpace(n)
			if slices.ContainsFunc(t.entries, func(e toolSearchEntry) bool { return e.Name == n }) {
				matched = append(matched, n)
			}
		}
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
	var results []scored
	for _, e := range t.entries {
		s := scoreEntry(e, terms)
		if s > 0 {
			results = append(results, scored{name: e.Name, score: s})
		}
	}

	// By score; entries scored alike keep their order.
	slices.SortStableFunc(results, func(a, b scored) int { return cmp.Compare(b.score, a.score) })
	matched := make([]string, 0, min(maxResults, len(results)))
	for _, r := range results[:min(maxResults, len(results))] {
		matched = append(matched, r.name)
	}
	return matched
}

// searchRegex matches entries where name or description matches the regex.
func (t *toolSearchTool) searchRegex(re *regexp.Regexp, maxResults int) []string {
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

package agentcore

import (
	"encoding/json"
	"fmt"
	"math"
	"sort"
	"strings"
)

// validationError is a tool call whose arguments do not fit the tool's
// schema. The model reads it as the call's result, every issue listed, and
// can correct them all at once.
type validationError struct {
	ToolName string
	Issues   []issue
}

func (e *validationError) Error() string { return formatValidationIssues(e.ToolName, e.Issues) }

// issue describes a single schema mismatch from tool arg validation.
type issue struct {
	Kind     string
	Path     string
	Expected string
	Received string
	Hint     string // optional fix hint, appended to the rendered message
}

const (
	issueMissing = "missing"
	issueType    = "type"
	issueValue   = "value"
	issueUnknown = "unknown"
)

// validateArgs validates the arguments of a call of the tool named name
// against schema, without changing them.
func validateArgs(name string, schema map[string]any, args json.RawMessage) error {
	if schema == nil {
		return nil
	}
	var value any
	if err := json.Unmarshal(args, &value); err != nil {
		return fmt.Errorf("%s received invalid JSON arguments: %v", name, err)
	}
	// The schema as the model reads it, JSON, whatever Go types built it.
	data, err := json.Marshal(schema)
	if err != nil {
		return fmt.Errorf("%s schema: %v", name, err)
	}
	var wire map[string]any
	if err := json.Unmarshal(data, &wire); err != nil {
		return fmt.Errorf("%s schema: %v", name, err)
	}
	if issues := validateSchemaValue(value, wire, ""); len(issues) > 0 {
		return &validationError{ToolName: name, Issues: issues}
	}
	return nil
}

func validateSchemaValue(value any, schema map[string]any, path string) []issue {
	var issues []issue
	issuePath := path
	if issuePath == "" {
		issuePath = "arguments"
	}
	types, hasTypes := schemaTypeNames(schema["type"])
	if hasTypes && !matchesSchemaType(value, types) {
		return []issue{{
			Kind:     issueType,
			Path:     issuePath,
			Expected: strings.Join(types, " or "),
			Received: jsonTypeName(value),
			Hint:     mismatchHint(value, types),
		}}
	}

	if values, ok := enumValues(schema["enum"]); ok && !containsJSONValue(values, value) {
		issues = append(issues, issue{
			Kind:     issueValue,
			Path:     issuePath,
			Expected: formatValues(values),
			Received: formatValue(value),
		})
	}

	object, isObject := value.(map[string]any)
	if isObject && (containsString(types, "object") || schema["properties"] != nil || schema["required"] != nil) {
		properties, _ := schema["properties"].(map[string]any)
		if required, ok := stringValues(schema["required"]); ok {
			for _, name := range required {
				if _, exists := object[name]; !exists {
					issues = append(issues, issue{
						Kind: issueMissing,
						Path: propertyPath(path, name),
					})
				}
			}
		}

		for name, child := range object {
			childPath := propertyPath(path, name)
			if rawSchema, exists := properties[name]; exists {
				if childSchema, ok := rawSchema.(map[string]any); ok {
					issues = append(issues, validateSchemaValue(child, childSchema, childPath)...)
				}
				continue
			}

			additional := schema["additionalProperties"]
			if additional == false {
				issues = append(issues, issue{Kind: issueUnknown, Path: childPath})
			} else if additionalSchema, ok := additional.(map[string]any); ok {
				issues = append(issues, validateSchemaValue(child, additionalSchema, childPath)...)
			}
		}
	}

	array, isArray := value.([]any)
	if isArray && (containsString(types, "array") || schema["items"] != nil) {
		if itemSchema, ok := schema["items"].(map[string]any); ok {
			for i, item := range array {
				issues = append(issues, validateSchemaValue(item, itemSchema, itemPath(path, i))...)
			}
		}
	}

	return issues
}

func schemaTypeNames(value any) ([]string, bool) {
	switch value := value.(type) {
	case string:
		return []string{value}, value != ""
	case []string:
		return value, len(value) > 0
	case []any:
		types := make([]string, 0, len(value))
		for _, item := range value {
			typ, ok := item.(string)
			if !ok || typ == "" {
				return nil, false
			}
			types = append(types, typ)
		}
		return types, len(types) > 0
	default:
		return nil, false
	}
}

func stringValues(value any) ([]string, bool) {
	switch value := value.(type) {
	case nil:
		return nil, false
	case []string:
		return value, true
	case []any:
		values := make([]string, 0, len(value))
		for _, item := range value {
			text, ok := item.(string)
			if !ok {
				return nil, false
			}
			values = append(values, text)
		}
		return values, true
	default:
		return nil, false
	}
}

func enumValues(value any) ([]any, bool) {
	switch value := value.(type) {
	case []any:
		return value, true
	case []string:
		values := make([]any, len(value))
		for i, item := range value {
			values[i] = item
		}
		return values, true
	default:
		return nil, false
	}
}

func matchesSchemaType(value any, types []string) bool {
	actual := jsonTypeName(value)
	for _, typ := range types {
		if typ == actual || typ == "number" && actual == "integer" {
			return true
		}
	}
	return false
}

func containsJSONValue(values []any, target any) bool {
	targetJSON, err := json.Marshal(target)
	if err != nil {
		return false
	}
	for _, value := range values {
		valueJSON, err := json.Marshal(value)
		if err == nil && string(valueJSON) == string(targetJSON) {
			return true
		}
	}
	return false
}

func mismatchHint(value any, types []string) string {
	text, ok := value.(string)
	if !ok {
		return ""
	}
	trimmed := strings.TrimSpace(text)
	if containsString(types, "array") && strings.HasPrefix(trimmed, "[") {
		return `Looks like a JSON-encoded array — pass the value directly (e.g. ["a","b"]), not wrapped in quotes.`
	}
	if containsString(types, "object") && strings.HasPrefix(trimmed, "{") {
		return `Looks like a JSON-encoded object — pass the value directly (e.g. {"k":"v"}), not wrapped in quotes.`
	}
	return ""
}

func jsonTypeName(value any) string {
	switch value := value.(type) {
	case nil:
		return "null"
	case bool:
		return "boolean"
	case string:
		return "string"
	case float64:
		if value == math.Trunc(value) {
			return "integer"
		}
		return "number"
	case []any:
		return "array"
	case map[string]any:
		return "object"
	default:
		return fmt.Sprintf("%T", value)
	}
}

func propertyPath(parent, property string) string {
	if parent == "" {
		return property
	}
	return parent + "." + property
}

func itemPath(parent string, index int) string {
	return fmt.Sprintf("%s[%d]", parent, index)
}

func containsString(values []string, target string) bool {
	for _, value := range values {
		if value == target {
			return true
		}
	}
	return false
}

func formatValues(values []any) string {
	formatted := make([]string, len(values))
	for i, value := range values {
		formatted[i] = formatValue(value)
	}
	return "[" + strings.Join(formatted, ", ") + "]"
}

func formatValue(value any) string {
	if text, ok := value.(string); ok {
		return fmt.Sprintf("%q", text)
	}
	return fmt.Sprint(value)
}

// formatValidationIssues renders issues as a single multi-line block.
// Missing params come first; within each group, paths sort alphabetically for
// stable output.
func formatValidationIssues(toolName string, issues []issue) string {
	// Sort a copy: Error() must not mutate the caller's Issues slice.
	issues = append([]issue(nil), issues...)
	sort.SliceStable(issues, func(i, j int) bool {
		if issues[i].Kind != issues[j].Kind {
			return validationIssueRank(issues[i].Kind) < validationIssueRank(issues[j].Kind)
		}
		return issues[i].Path < issues[j].Path
	})

	lines := make([]string, 0, len(issues))
	for _, it := range issues {
		var line string
		switch it.Kind {
		case issueMissing:
			line = fmt.Sprintf("The required parameter `%s` is missing", it.Path)
		case issueType:
			line = fmt.Sprintf(
				"The parameter `%s` type is expected as `%s` but provided as `%s`",
				it.Path, it.Expected, it.Received,
			)
		case issueValue:
			line = fmt.Sprintf(
				"The parameter `%s` must be one of %s but provided as %s",
				it.Path, it.Expected, it.Received,
			)
		case issueUnknown:
			line = fmt.Sprintf("The parameter `%s` is not allowed", it.Path)
		default:
			continue
		}
		if it.Hint != "" {
			line += ". " + it.Hint
		}
		lines = append(lines, line)
	}

	noun := "issue"
	if len(lines) > 1 {
		noun = "issues"
	}
	header := fmt.Sprintf("InputValidationError: %s failed due to the following %s:", toolName, noun)
	return header + "\n" + strings.Join(lines, "\n")
}

func validationIssueRank(kind string) int {
	switch kind {
	case issueMissing:
		return 0
	case issueType:
		return 1
	case issueValue:
		return 2
	case issueUnknown:
		return 3
	default:
		return 4
	}
}

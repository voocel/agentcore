package agentcore

import (
	"cmp"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"slices"
	"strings"
)

// issue is one way the arguments of a call do not fit the tool's schema.
type issue struct {
	kind issueKind
	path string
	text string
}

// issueKind orders the issues the model reads: missing parameters first.
type issueKind int

const (
	issueMissing issueKind = iota
	issueType
	issueValue
	issueUnknown
)

// validateArgs validates the arguments of a call of the tool named name
// against schema, without changing them. The model reads the error as the
// call's result, every issue listed, and can correct them all at once.
func validateArgs(name string, schema map[string]any, args json.RawMessage) error {
	if schema == nil {
		return nil
	}
	// The arguments are a JSON object (see invalidArgs), and the schema
	// marshals, as BuildCall offered it. It is read as the model reads it,
	// JSON, whatever Go types built it.
	var value any
	_ = json.Unmarshal(args, &value)
	var wire map[string]any
	data, _ := json.Marshal(schema)
	_ = json.Unmarshal(data, &wire)
	if issues := validateSchemaValue(value, wire, ""); len(issues) > 0 {
		return errors.New(formatValidationIssues(name, issues))
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
		text := fmt.Sprintf("The parameter `%s` type is expected as `%s` but provided as `%s`", issuePath, strings.Join(types, " or "), jsonTypeName(value))
		if hint := mismatchHint(value, types); hint != "" {
			text += ". " + hint
		}
		return []issue{{issueType, issuePath, text}}
	}

	if values, ok := schema["enum"].([]any); ok && !containsJSONValue(values, value) {
		issues = append(issues, issue{issueValue, issuePath, fmt.Sprintf("The parameter `%s` must be one of %s but provided as %s", issuePath, formatValues(values), formatValue(value))})
	}

	object, isObject := value.(map[string]any)
	if isObject && (slices.Contains(types, "object") || schema["properties"] != nil || schema["required"] != nil) {
		properties, _ := schema["properties"].(map[string]any)
		if required, ok := stringValues(schema["required"]); ok {
			for _, name := range required {
				if _, exists := object[name]; !exists {
					p := propertyPath(path, name)
					issues = append(issues, issue{issueMissing, p, fmt.Sprintf("The required parameter `%s` is missing", p)})
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
				issues = append(issues, issue{issueUnknown, childPath, fmt.Sprintf("The parameter `%s` is not allowed", childPath)})
			} else if additionalSchema, ok := additional.(map[string]any); ok {
				issues = append(issues, validateSchemaValue(child, additionalSchema, childPath)...)
			}
		}
	}

	array, isArray := value.([]any)
	if isArray && (slices.Contains(types, "array") || schema["items"] != nil) {
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
	case []any:
		types, ok := stringValues(value)
		return types, ok && len(types) > 0 && !slices.Contains(types, "")
	default:
		return nil, false
	}
}

func stringValues(value any) ([]string, bool) {
	items, ok := value.([]any)
	if !ok {
		return nil, false
	}
	values := make([]string, len(items))
	for i, item := range items {
		if values[i], ok = item.(string); !ok {
			return nil, false
		}
	}
	return values, true
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
	if slices.Contains(types, "array") && strings.HasPrefix(trimmed, "[") {
		return `Looks like a JSON-encoded array — pass the value directly (e.g. ["a","b"]), not wrapped in quotes.`
	}
	if slices.Contains(types, "object") && strings.HasPrefix(trimmed, "{") {
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

// formatValidationIssues renders issues as a single multi-line block, by
// kind, then by path, for stable output.
func formatValidationIssues(toolName string, issues []issue) string {
	slices.SortStableFunc(issues, func(a, b issue) int {
		return cmp.Or(cmp.Compare(a.kind, b.kind), strings.Compare(a.path, b.path))
	})
	lines := make([]string, len(issues))
	for i, it := range issues {
		lines[i] = it.text
	}
	noun := "issue"
	if len(lines) > 1 {
		noun = "issues"
	}
	header := fmt.Sprintf("InputValidationError: %s failed due to the following %s:", toolName, noun)
	return header + "\n" + strings.Join(lines, "\n")
}

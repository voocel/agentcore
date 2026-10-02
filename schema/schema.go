// Package schema builds the JSON Schema of a tool's arguments, as the
// map[string]any agentcore's Tool.Schema holds.
package schema

// Prop is a named property with optional required flag.
type Prop struct {
	name     string
	schema   map[string]any
	required bool
}

// Property creates a named schema property.
func Property(name string, s map[string]any) Prop {
	return Prop{name: name, schema: s}
}

// Required marks this property as required.
func (p Prop) Required() Prop {
	p.required = true
	return p
}

// Object builds a JSON Schema object from the given properties.
func Object(props ...Prop) map[string]any {
	properties := make(map[string]any, len(props))
	var required []string
	for _, p := range props {
		properties[p.name] = p.schema
		if p.required {
			required = append(required, p.name)
		}
	}
	obj := map[string]any{
		"type":       "object",
		"properties": properties,
	}
	if len(required) > 0 {
		obj["required"] = required
	}
	return obj
}

// String returns a string schema.
func String(desc string) map[string]any {
	return map[string]any{"type": "string", "description": desc}
}

// Int returns an integer schema.
func Int(desc string) map[string]any {
	return map[string]any{"type": "integer", "description": desc}
}

// Number returns a number schema.
func Number(desc string) map[string]any {
	return map[string]any{"type": "number", "description": desc}
}

// Bool returns a boolean schema.
func Bool(desc string) map[string]any {
	return map[string]any{"type": "boolean", "description": desc}
}

// Enum returns a string enum schema.
func Enum(desc string, values ...string) map[string]any {
	return map[string]any{"type": "string", "description": desc, "enum": values}
}

// Array returns an array schema with the given item schema.
func Array(desc string, items map[string]any) map[string]any {
	return map[string]any{"type": "array", "description": desc, "items": items}
}

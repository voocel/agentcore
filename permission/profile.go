package permission

import (
	"fmt"
	"strings"
)

func ParseMode(raw string) (Mode, error) {
	switch strings.ToLower(strings.TrimSpace(raw)) {
	case "", "balanced":
		return ModeBalanced, nil
	case "strict":
		return ModeStrict, nil
	case "auto", "accept-edits", "accept_edits":
		return ModeAuto, nil
	case "trust", "off":
		return ModeTrust, nil
	default:
		return "", fmt.Errorf("invalid mode %q (allowed: strict, balanced, auto, trust)", raw)
	}
}

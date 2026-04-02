package permission

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestStoreRoundTrip(t *testing.T) {
	path := filepath.Join(t.TempDir(), "approvals.json")
	store, err := NewStore(path)
	if err != nil {
		t.Fatalf("NewStore: %v", err)
	}

	if err := store.Add(StoreEntry{
		Key:        "b",
		Tool:       "bash",
		Capability: CapabilityExec,
		Summary:    "echo ok",
		AddedAt:    time.Now(),
	}); err != nil {
		t.Fatalf("Add b: %v", err)
	}
	if err := store.Add(StoreEntry{
		Key:        "a",
		Tool:       "read",
		Capability: CapabilityRead,
		Summary:    "a.txt",
		AddedAt:    time.Now(),
	}); err != nil {
		t.Fatalf("Add a: %v", err)
	}

	reloaded, err := NewStore(path)
	if err != nil {
		t.Fatalf("NewStore reload: %v", err)
	}
	if !reloaded.Has("a") || !reloaded.Has("b") {
		t.Fatalf("expected persisted entries, got %#v", reloaded)
	}

	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("ReadFile: %v", err)
	}
	var rows []StoreEntry
	if err := json.Unmarshal(data, &rows); err != nil {
		t.Fatalf("Unmarshal: %v", err)
	}
	if len(rows) != 2 || rows[0].Key != "a" || rows[1].Key != "b" {
		t.Fatalf("expected sorted persisted rows, got %#v", rows)
	}
}

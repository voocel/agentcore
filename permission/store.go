package permission

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"
)

type StoreEntry struct {
	Key        string     `json:"key"`
	Tool       string     `json:"tool"`
	Capability Capability `json:"capability"`
	Summary    string     `json:"summary"`
	AddedAt    time.Time  `json:"added_at"`
}

type Store struct {
	mu      sync.RWMutex
	path    string
	entries map[string]StoreEntry
}

func NewStore(path string) (*Store, error) {
	s := &Store{
		path:    path,
		entries: make(map[string]StoreEntry),
	}
	if strings.TrimSpace(path) == "" {
		return s, nil
	}
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return s, nil
		}
		return nil, err
	}
	var rows []StoreEntry
	if err := json.Unmarshal(data, &rows); err != nil {
		return nil, err
	}
	for _, row := range rows {
		if row.Key != "" {
			s.entries[row.Key] = row
		}
	}
	return s, nil
}

func (s *Store) Has(key string) bool {
	if s == nil || key == "" {
		return false
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	_, ok := s.entries[key]
	return ok
}

func (s *Store) Add(entry StoreEntry) error {
	if s == nil || entry.Key == "" {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.entries[entry.Key] = entry
	return s.saveLocked()
}

func (s *Store) saveLocked() error {
	if strings.TrimSpace(s.path) == "" {
		return nil
	}
	rows := make([]StoreEntry, 0, len(s.entries))
	for _, row := range s.entries {
		rows = append(rows, row)
	}
	sort.Slice(rows, func(i, j int) bool {
		return rows[i].Key < rows[j].Key
	})
	data, err := json.MarshalIndent(rows, "", "  ")
	if err != nil {
		return err
	}
	dir := filepath.Dir(s.path)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	tmp, err := os.CreateTemp(dir, filepath.Base(s.path)+".tmp-*")
	if err != nil {
		return err
	}
	tmpPath := tmp.Name()
	defer os.Remove(tmpPath)
	if _, err := tmp.Write(data); err != nil {
		_ = tmp.Close()
		return err
	}
	if err := tmp.Chmod(0o600); err != nil {
		_ = tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	return os.Rename(tmpPath, s.path)
}

func shortHash(s string) string {
	sum := sha256.Sum256([]byte(strings.TrimSpace(s)))
	return hex.EncodeToString(sum[:8])
}

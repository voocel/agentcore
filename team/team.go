// Package team implements Team — multi-agent peer-to-peer collaboration on top
// of the Subagent foundation. Where a Subagent is a fire-and-forget worker that
// returns a single result, a Teammate is a long-lived peer that stays idle
// between turns, can receive messages from the leader or other teammates, and
// only exits via an explicit shutdown handshake.
//
// The mailbox is in-memory, single-session scoped — no disk persistence,
// no cross-process IPC. Goroutines wake via a channel signal on Send; there
// is no polling loop.
//
// Package layout:
//
//	team.go         — package overview + small constants
//	identity.go     — context plumbing for thread-local agent identity
//	mailbox.go      — per-agent inbox with priority drain + wake channel
//	registry.go     — session-wide team state (mailboxes, name registry, team ctx)
//	runner.go       — long-lived teammate goroutine (turn loop + ProtocolHooks)
//	spawn.go        — teammate registration + goroutine launch
//
// agentcore provides the mechanism; wire format and policy live in the
// application layer and are injected via RunConfig.Protocol / SpawnConfig.Protocol.
// See ProtocolHooks for the per-field contract.
//
// Dependency direction: team → task (not reversed). Identity lives in task
// because Entry needs it; everything else lives here.
package team

// TeamLeadName is the canonical name of the team coordinator. Reserved —
// teammates may not be created with this name.
const TeamLeadName = "team-lead"

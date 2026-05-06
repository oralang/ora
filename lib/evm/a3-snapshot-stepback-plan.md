# A3: snapshot-based stepBack — implementation plan

The current `stepBack` and `rerunToHistory` rebuild the EVM
session from the seed and replay every command in
`step_history`. Cost is linear in user steps × opcodes-per-step;
for a 10k-step session the rebuild dominates wall-clock.

This document specifies the work to introduce an EVM-state
snapshot ring so a stepBack rewinds to the nearest snapshot ≤
target_step and replays only the trailing K-or-fewer steps.

## The actual cost we're trying to remove

`Ui.stepBack` (`src/debug_tui.zig:1920`) does:

```
session.deinit();
Session.init(&session, allocator, &seed);   // fresh EVM
primeInitialStop();                          // advance to first stop
for replayed_step in step_history:
    runStep(...)                             // re-execute opcodes
```

Every `runStep` invocation is many EVM `step()` calls under
the hood. With ~50 opcodes/step on average and N user
commands, that's 50N opcode dispatches per stepBack. Doubles
each time the user does back→forward→back.

## Why a state-only snapshot doesn't help

A naive snapshot of just `Storage` maps and the
`AccessListManager` (the easy parts to clone) doesn't shorten
replay: the EVM still has to advance through the bytecode to
land at the right `(Frame.pc, Frame.stack, Frame.memory)`.
Replay can't skip opcodes. To actually save time, the
snapshot must include the live `Frame` stack.

This is the part I called "multi-day" earlier — `Frame` is
the heavy struct.

## What needs cloning

Reading `lib/evm/src/frame.zig` and `lib/evm/src/evm.zig`,
the live state we'd snapshot:

1. **`Frame` stack** (`evm.frames.items`). Each frame:
   - `stack: ArrayList(u256)` — straightforward clone.
   - `memory: AutoHashMap(u32, u8)` — straightforward clone.
   - `memory_size: u32` — value.
   - `pc: u32`, `gas_remaining: i64`, `value: u256` — values.
   - `bytecode: Bytecode` — has analyzed JUMPDEST sets;
     re-running `Bytecode.init` on the original bytes
     reproduces it. Cheaper to share by reference if the
     bytecode is immutable for the session (it is).
   - `caller`, `address` — values.
   - `calldata: []const u8` — borrowed from seed; share by
     reference.
   - `output: []u8` — heap-allocated by RETURN/REVERT
     handlers. Clone the bytes.
   - `return_data: []const u8` — same: clone the bytes.
   - `stopped: bool`, `reverted: bool`, `is_static: bool` —
     values.
   - `evm_ptr: *anyopaque` — points back to the EVM. On
     restore, rebind to the live EVM, not the snapshot's EVM.
   - `authorized: ?u256`, `call_depth: u32`, `hardfork` —
     values.

2. **`Storage`** (`evm.storage`):
   - Three `AutoHashMap(StorageSlotKey, u256)`: `storage`,
     `original_storage`, `transient`. Clone each (straight
     hashmap copy).

3. **`AccessListManager`**. Already exposes `snapshot()` /
   `restore()` (verified at
   `src/access_list_manager.zig:123,145`). Use as-is.

4. **`evm.logs`** (`std.ArrayList(call_result.Log)`). Each
   log has `address: Address`, `topics: []u256` (clone),
   `data: []u8` (clone).

5. **Debugger state**: `last_statement_id`,
   `last_statement_line`, `last_statement_sir_line`,
   `last_error_name`, `state`, `stop_reason`,
   `steps_executed`, `last_watchpoint_id`, plus
   `line_hits` and `line_gas` maps (clone).

6. **Watchpoints**: `Debugger.watchpoints` ArrayList. Each
   carries `last_seen` u256 — clone the list with values.

NOT cloned (preserved across snapshots, set up once):
- `breakpoints` (the debugger's PC set) — already invariant
  across the session.
- `src_map`, `debug_info`, `source_text`,
  `ignored_invalid_idx` — read-only after init.
- `evm.hardfork`, `evm.block_context` — read-only.

## Architecture

### Step 1 — `EvmSnapshot` struct

New file `lib/evm/src/evm_snapshot.zig`:

```zig
pub const EvmSnapshot = struct {
    allocator: std.mem.Allocator,
    frames: []ClonedFrame,
    storage: ClonedStorage,
    access_list: AccessListSnapshot,  // already exists
    logs: []ClonedLog,
    debugger_state: ClonedDebuggerState,

    pub fn capture(allocator, evm, debugger) !EvmSnapshot;
    pub fn restore(self: *const EvmSnapshot, evm: *Evm, debugger: *Debugger) !void;
    pub fn deinit(self: *EvmSnapshot) void;
};
```

`ClonedFrame`, `ClonedStorage`, etc. own their data so the
snapshot survives EVM teardown.

The trickiest part is `Frame.bytecode` — its analyzed
JUMPDEST set is built in `Bytecode.init`. On restore, either:
(a) keep a single shared `Bytecode` instance per session and
have all snapshots point at it, or (b) re-run `Bytecode.init`
on restore (~tens of µs). Option (a) is cleaner; (b) is
simpler.

Estimate: ~200 LOC for the struct + helpers.

### Step 2 — snapshot ring on `Session`

In `src/debug_tui.zig` `Session` struct (or a sibling), add:

```zig
const SnapshotRing = struct {
    entries: [K]?Entry,   // K=64 is the plan's default
    head: usize,

    const Entry = struct {
        snap: EvmSnapshot,
        user_step_index: usize,  // step_history index AT capture time
    };
};
```

Capture cadence: every K user-steps in `Ui.runStep` (or every
K statement-boundaries — measured from `steps_executed`).

Estimate: ~80 LOC.

### Step 3 — `stepBack` consults the ring

Replace the unconditional rebuild-from-zero in
`Ui.stepBack` (`src/debug_tui.zig:1920`):

```
target = step_history.len - 1;  // popped one
nearest = ring.findLatestSnapshot(<= target);
if (nearest):
    session.deinit();
    rebuild fresh session;       // unavoidable: state belongs to a torn-down EVM
    primeInitialStop();
    snap.restore(&session.evm, &session.debugger);
    replay step_history[nearest.user_step_index .. target];
else:
    // no snapshot yet, fall back to current behavior
    rebuild + full replay;
```

Note the rebuild is still required (the prior session's
arenas freed everything). Snapshot just skips the replay
loop's bulk.

Estimate: ~50 LOC + tests.

### Step 4 — `rerunToHistory` benefits too

`rerunToHistory(step_count)` (used by `:restart <id>`) gets
the same treatment. Same code path.

Estimate: ~20 LOC.

### Step 5 — bench harness

Extend `lib/evm/test/bench/step_bench.zig` (or add a sibling)
to measure stepBack time on a fixture with N=1k, 10k user
steps. Plan target: stepBack <50ms for any step in a 100k-step
trace (per the original A3 entry).

Estimate: ~60 LOC.

## Alternatives considered (rejected)

### Don't snapshot, dedupe replay

Could memoize "running these N steps from start lands at
state X" and skip re-running known prefixes. Deduplication
key would be the prefix of `step_history`. Saves nothing
for the typical case where the user is exploring forward
then backward — every prefix is unique.

### Operate on undo log instead of snapshots

Track per-opcode delta (e.g. SSTORE prev value) and play
the trace backward. Theoretically O(K) for K-step rewind
without the rebuild cost. Implementation cost is huge — every
state-mutating opcode handler in `lib/evm/src/instructions/`
needs an undo emitter, plus the EVM core needs reverse-step
plumbing. Out of scope for a debugger-facing change.

## Acceptance for A3-close

- `EvmSnapshot.capture` + `restore` pass round-trip tests
  on at least: post-init state, mid-loop state, post-CALL
  state, post-revert state.
- A `lib/evm/test/bench/stepback_bench.zig` shows
  stepBack < 50ms median for any step in a 1k-user-step
  trace; ideally < 200ms in a 10k-step trace.
- Existing 551+ tests still pass; bench step_bench at the
  per-step level not regressed (capture cost amortized at
  K=64).

## Estimated effort

~1.5–2 focused days:

| Phase | LOC | Time |
|---|---|---|
| EvmSnapshot capture+restore | ~200 | 4–6 hr |
| Snapshot ring on Session | ~80 | 2 hr |
| stepBack + rerunToHistory rewire | ~70 | 2 hr |
| Round-trip tests | ~150 | 2–3 hr |
| Bench harness + tuning | ~60 | 2 hr |
| Edge cases (CALL boundaries, watchpoint replay invariants) | — | 2–3 hr |

Total: ~14–18 focused hours, comfortably one well-scoped
session for someone familiar with the EVM internals.

## Why this isn't closable in one debugger session

The work is concrete and mechanical — no new architecture —
but the cloning surface is broad enough (frames, storage,
logs, debugger metadata) that getting it right requires
careful test coverage at each layer. Skipping the round-trip
tests would be how subtle bugs land where stepBack drops a
log entry, fails to restore a watchpoint's last_seen, or
diverges on a contract that re-enters a memory region.

This is a "do it right, in its own session" item, not a
"squeeze into the tail of the existing one" item. The TUI
side is fully unblocked once `EvmSnapshot.{capture,restore}`
exist — the wire-up at the call sites is straightforward.

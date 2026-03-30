# Ora Debugger

Ora ships with an interactive EVM debugger that understands Ora source maps, visible bindings, and runtime state.

This is not just a bytecode trace viewer. It can:
- step through Ora source statements
- inspect EVM machine state
- inspect visible Ora bindings
- mutate selected runtime values
- save and reload replayable sessions
- move backward by debugger stop

The debugger is still pre-release. The workflow is already useful, but some features are intentionally conservative.

## Start

Build Ora first:

```bash
zig build
```

Launch a debugger session:

```bash
./zig-out/bin/ora debug ora-example/arithmetic_test.ora --signature 'add(u256,u256)' --arg 7 --arg 9
```

For contracts with `init(...)`, pass constructor arguments too:

```bash
./zig-out/bin/ora debug ora-example/debugger/ledger_walkthrough.ora \
  --init-signature 'init(u256,u256)' \
  --init-arg 1000 \
  --init-arg 250 \
  --signature 'process(u256)' \
  --arg 200
```

Recommended optimization/provenance probe:

```bash
./zig-out/bin/ora debug ora-example/debugger/optimizer_probe.ora \
  --init-signature 'init(u256)' \
  --init-arg 1000 \
  --signature 'simulate(address,u256,u256)' \
  --arg 0x00000000000000000000000000000000000000ab \
  --arg 7 \
  --arg 50
```

The debugger compiles the contract, emits debug artifacts under `artifacts/<name>/`, and launches the interactive EVM debugger.

## What `ora debug` Emits

For `foo.ora`, the debugger flow uses:

```text
artifacts/foo/foo.hex
artifacts/foo/foo.sourcemap.json
artifacts/foo/foo.debug.json
artifacts/foo/abi/foo.abi.json
```

For contracts with `init(...)`, Ora appends encoded constructor arguments to the creation bytecode before deployment, matching the current compiler lowering convention.

## UI Model

The debugger UI is split into four main areas:

- `Source`
  - original Ora source
  - current statement highlight
  - breakpoint markers in the gutter
- `Bindings`
  - visible Ora names
  - folded comptime values
  - runtime-rooted values when readable
  - persistent breakpoint/checkpoint summaries
- `Machine`
  - selected frame summary
  - PC, opcode, depth, gas, caller/callee, calldata size, memory size
  - current opcode window
- `State`
  - tabbed machine state
  - `Stack`, `Memory`, `Storage`, `TStore`, `Calldata`
- footer command console
  - Vim-style `:command`
  - rolling command/result trail

## Keyboard Controls

Global keys:

```text
s    step in
n    step over
o    step out
c    continue
p    previous stop
j/k  scroll source
[/]  cycle state tabs
1..5 jump directly to state tabs
:    command mode
q    quit
```

State tabs:

```text
1  Stack
2  Memory
3  Storage
4  TStore
5  Calldata
```

## Command Bar

Press `:` to enter command mode, then `Enter` to execute or `Esc` to cancel.

### Execution

```text
:run
:rerun
:continue
:step
:next
:out
:prev
```

Notes:
- `:run` and `:rerun` restart the session from the beginning
- `:prev` steps backward by debugger stop using replay

## Breakpoints

Set and inspect source breakpoints:

```text
:break 27
:break file.ora:27
:delete 27
:info break
```

Breakpoint markers appear in the source gutter:
- `*` breakpoint
- `>` current line with a breakpoint

## Checkpoints

Create and restore checkpoints:

```text
:checkpoint
:checkpoints
:restart 1
```

Checkpoint semantics:
- checkpoints store replay position plus UI state
- restoring a checkpoint rebuilds the session and replays to that point

## Backtrace and Frames

Inspect nested execution:

```text
:bt
:backtrace
:frame 0
:frame 1
```

Notes:
- frame `0` is the top frame
- `Machine` and `State` panes follow the selected frame
- source/bindings still reflect the current debugger stop, not arbitrary frame-local source reconstruction

## Inspecting Values

### Ora Bindings

```text
:print total
:print gas
```

### Raw Machine State

```text
:print stack[0]
:print slot 0x00
:print mem 0x80 4
:print storage
:print tstore
:print calldata
```

What these do:
- `stack[0]` prints the top of the selected frame stack
- `slot` prints a raw storage slot by id
- `mem` prints memory words from the selected frame
- `storage` / `tstore` dump rooted contract state for the selected frame address

## Mutating State

The debugger supports limited live mutation.

### Source-Level / Rooted Values

```text
:set total = 1337
:set gas = 750000
```

### Raw Machine State

```text
:set slot 0x00 = 7
:set mem 0x80 = 42
```

Current mutation boundary:
- writable rooted bindings can be changed
- gas can be changed
- raw storage slots can be changed
- raw memory words can be changed
- mutation support is intentionally conservative

## Recommended Debugger Probes

- `ora-example/debugger/constructor_value.ora`
  - minimal constructor/init sanity check
- `ora-example/debugger/state_walkthrough.ora`
  - small stateful walkthrough
- `ora-example/debugger/ledger_walkthrough.ora`
  - constructor args, helper calls, branches, storage updates
- `ora-example/debugger/optimizer_probe.ora`
  - constructor args
  - visible params and locals
  - runtime guards
  - helper calls
  - comptime-folded constants
  - branch-dependent behavior
  - storage updates suitable for slot/gas/value inspection

## Sessions

Save and reload sessions:

```text
:write-session artifacts/ledger_walkthrough/session.json
:ws artifacts/ledger_walkthrough/session.json

:load-session artifacts/ledger_walkthrough/session.json
:ls artifacts/ledger_walkthrough/session.json
```

Saved sessions include:
- artifact paths
- calldata
- step history
- breakpoints
- checkpoints
- selected state tab
- source scroll/focus state

Session model:
- replayable JSON
- not a raw VM memory dump
- portable and shareable across machines if artifacts match

## Gas

The debugger shows:
- gas remaining
- gas spent since previous stop
- total gas spent from the initial 5,000,000 budget

Gas commands:

```text
:print gas
:gas
:gas 500000
:set gas = 500000
```

## Storage and Slot IDs

The `Storage` and `TStore` tabs show:
- full slot hash/id
- current value

This is important for debugging mappings and lowered storage layouts, not just named variables.

## Reverse Stepping

`p` and `:prev` are supported.

Current behavior:
- reverse by debugger stop, not by raw opcode
- implemented by replay, not full VM time travel

That means:
- it is practical and deterministic
- it is not yet a full reverse-execution engine

## Good Test Contracts

Simple:

```bash
./zig-out/bin/ora debug ora-example/arithmetic_test.ora --signature 'add(u256,u256)' --arg 7 --arg 9
```

Constructor + richer state:

```bash
./zig-out/bin/ora debug ora-example/debugger/ledger_walkthrough.ora \
  --init-signature 'init(u256,u256)' \
  --init-arg 1000 \
  --init-arg 250 \
  --signature 'process(u256)' \
  --arg 200
```

Minimal constructor probe:

```bash
./zig-out/bin/ora debug ora-example/debugger/constructor_value.ora \
  --init-signature 'init(u256)' \
  --init-arg 42 \
  --signature 'get()'
```

## Current Limits

Important current boundaries:

- reverse stepping is replay-based
- source/bindings are current-stop-oriented, not arbitrary-frame source reconstruction
- session files do not yet enforce artifact hash validation
- debugger mutation is limited to safe supported roots and raw machine locations
- UI is still evolving

## Related Files

- root command: [src/main.zig](/Users/logic/Ora/Ora/src/main.zig)
- debugger runtime/UI: [lib/evm/src/debug_tui.zig](/Users/logic/Ora/Ora/lib/evm/src/debug_tui.zig)
- debugger engine: [lib/evm/src/debugger.zig](/Users/logic/Ora/Ora/lib/evm/src/debugger.zig)
- implementation plan: [lib/evm/DEBUGGER_PLAN.md](/Users/logic/Ora/Ora/lib/evm/DEBUGGER_PLAN.md)

# Ora EVM Debugger Plan

## Status

`lib/evm` is now treated as an Ora-owned sidecar runtime inside this repository.

Current debugger foundation:

- `ora emit --emit-bytecode --debug-info` emits:
  - `<contract>.hex`
  - `<contract>.sourcemap.json`
  - `<contract>.debug.json`
- `.sourcemap.json` carries:
  - `pc`
  - `idx`
  - `src`
  - `line`
  - `col`
  - `stmt`
- `.debug.json` carries:
  - per-op metadata keyed by serialized SIR op index
  - lexical scopes
  - visible locals
  - runtime classification
  - concrete runtime root payloads where available
  - folded compile-time values for bindings that were reduced away

Current debugger/runtime support in `lib/evm`:

- statement stepping
- source breakpoints
- current op index lookup
- visible scopes / bindings at a stop point
- runtime root access for:
  - `storage_root`
  - `tstore_root`
  - contract-level `memory_root`

Conservative boundary:

- contract-level named roots can be debugger-addressable
- plain SSA values are not treated as writable debugger slots
- local memory values without stable lowering identity remain opaque

## Immediate Priorities

1. Track `lib/evm` cleanly as first-party code in Ora.
2. Keep `voltaire` as a temporary vendored dependency.
3. Continue debugger work against this in-repo runtime boundary.

## Debugger V1 Target

The debugger should be a real interactive source + machine debugger, not an
EVM trace viewer with source labels.

### Core Views

1. `Source`
   - original Ora source file
   - current statement highlight
   - breakpoint markers
   - current function / contract context

2. `Bindings`
   - visible Ora names at the current stop point
   - folded compile-time values
   - rooted runtime values
   - mutability / editability markers

3. `Machine`
   - current opcode
   - PC
   - gas
   - call depth
   - caller / callee / value
   - next opcode preview

4. `State`
   - `Stack`
   - `Memory`
   - `Storage`
   - `TStore`
   - `Calldata`
   - diff-first display: changed items first, full dump later

5. `Command`
   - a bottom command line for debugger commands
   - command history
   - inline status / errors / command output

### Planned Layout

```text
┌ Ora Debugger ─ contract.ora ─────────────── pc=... op=... gas=... depth=... ┐
│ status: ...   fn: ...   selector: ...   line: ...                            │
├──────────────────────────────────────────────┬────────────────────────────────┤
│ Source                                       │ Bindings                       │
│                                              │ Folded                         │
│                                              │ Writable                       │
│                                              │ Machine                        │
├──────────────────────────────────────────────┴────────────────────────────────┤
│ [Stack]  Memory  Storage  TStore  Calldata                                    │
│ diff-first EVM state view                                                     │
├───────────────────────────────────────────────────────────────────────────────┤
│ :command input                                                                │
├───────────────────────────────────────────────────────────────────────────────┤
│ s step-in  n step-over  o step-out  c continue  : command  q quit            │
└───────────────────────────────────────────────────────────────────────────────┘
```

### Interaction Model

Debugger control should support both direct keys and `:` commands.

#### Keys

- `s` step in
- `n` step over
- `o` step out
- `c` continue
- `r` run / restart
- `b` toggle breakpoint on current source line
- `tab` cycle pane focus
- `j` / `k` or arrows scroll active pane
- `[` / `]` switch state tabs
- `:` enter command mode
- `q` quit

#### Commands

- `:run`
- `:continue`
- `:si`
- `:so`
- `:su`
- `:break 31`
- `:break file.ora:31`
- `:breakpc 981`
- `:breakop SSTORE`
- `:delete 2`
- `:info break`
- `:bt`
- `:frame 1`
- `:print a`
- `:print stack[0]`
- `:print storage counter`
- `:print mem 0x80 8`
- `:print calldata`
- `:set counter = 7`
- `:set tstore temp_counter = 1`

### View Semantics

#### Source

- authoritative original Ora source
- statement-highlighted, not instruction-highlighted
- never show synthetic or shared return-site noise as if it were source truth

#### Bindings

- show source-language values, not raw machine data
- distinguish:
  - live runtime values
  - folded compile-time values
  - optimized-out values
- only expose editability when a concrete runtime home exists

#### Machine

- raw EVM truth for the current instruction
- should answer:
  - where the VM is
  - what opcode is executing
  - what call context is active

#### State

- raw machine state with diff-first presentation
- default emphasis:
  - stack changes
  - changed memory words
  - changed storage slots
  - changed transient slots
- rooted state names should appear alongside raw slot addresses where possible

#### Command

- primary way to grow debugger power without overloading the keymap
- all advanced actions should exist here first

## Near-Term Implementation Order

1. Add bottom command-line mode and parser.
2. Add a stable `Machine` pane to the right side.
3. Expand bottom tabbed state pane to:
   - `Stack`
   - `Memory`
   - `Storage`
   - `TStore`
   - `Calldata`
4. Add `:bt`, `:break`, and `:print`.
5. Add diff-highlighting for state views.
6. Add rooted `:set` for:
   - `storage_root`
   - `tstore_root`
7. Refine breakpoint UX and backtrace frame navigation.

## Explicit V1 Boundaries

V1 should support:

- source stepping
- source breakpoints
- PC / opcode breakpoints
- backtrace
- source + machine state inspection
- rooted storage / transient mutation
- folded compile-time value display

V1 should not block on:

- reverse execution
- time-travel debugging
- a full expression evaluator
- general SSA/local mutation
- polished mouse-heavy UI

## Previous Next Slice

The earlier immediate API goal was:

Expose provenance cleanly in the debugger API so a binding can be reported as:

- live runtime value
- folded compile-time value
- optimized out with no recoverable value

That remains required and should continue to inform all UI work above.

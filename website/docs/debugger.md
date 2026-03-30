---
title: Interactive Debugger
sidebar_position: 6
description: Source-level EVM debugger with statement stepping, variable inspection, SIR view, and state mutation.
---

# Interactive Debugger

Ora ships with a source-level EVM debugger. Set breakpoints on Ora source lines, step through statements, inspect bindings and machine state, and see exactly how your code lowers to SIR and EVM bytecode.

> The debugger is pre-release. The workflow is usable today, but some features
> are intentionally conservative about what they expose.

## Quick start

Build Ora, then launch a debug session:

```bash
zig build

ora debug contracts/vault.ora \
  --signature 'deposit(u256)' \
  --arg 1000
```

For contracts with constructors:

```bash
ora debug contracts/vault.ora \
  --init-signature 'init(u256,u256)' \
  --init-arg 1000 \
  --init-arg 250 \
  --signature 'deposit(u256)' \
  --arg 200
```

The compiler generates debug artifacts, then launches the interactive TUI.

## Compilation and artifacts

`ora debug` runs the full compiler pipeline in debugger-safe mode, then launches the TUI against the resulting artifacts.

### What the compiler does differently

In debug mode, the compiler disables passes that would break source-to-bytecode correlation:

- **Constant deduplication** disabled — duplicate constants keep their original source locations instead of being merged
- **Copy propagation** disabled in Sensei — op indices stay aligned between SIR MLIR and bytecode
- **Inlining** disabled — function boundaries are preserved so step-over and step-out work correctly
- **Ora MLIR canonicalization** CSE disabled — constants are not rehomed to `UnknownLoc`

The result is slightly larger bytecode than a release build, but every Ora source line maps cleanly to its bytecode range.

### Generated artifacts

For a contract `vault.ora`, `ora debug` writes:

```text
artifacts/vault/
  vault.hex                 bytecode (creation code with constructor args appended)
  vault.sourcemap.json      PC → (file, line, col, statement boundary, SIR op index, SIR line)
  vault.debug.json          lexical scopes, locals, visibility per op, runtime classification
  vault.sir.txt             emitted SIR text (shown in the SIR pane)
  abi/vault.abi.json        ABI used to encode calldata from --signature and --arg
```

You do not need to manage these files. `ora debug` generates them, passes them to the TUI, and the TUI loads them automatically. They matter when you save or share sessions (see [Sessions](#sessions)).

### Constructor deployment

For contracts with `init(...)`, the compiler appends ABI-encoded constructor arguments to the creation bytecode. The local EVM deploys the contract (executing the constructor), then the debugger begins at the first statement of the target function.

```bash
ora debug vault.ora \
  --init-signature 'init(u256,u256)' \
  --init-arg 1000 \
  --init-arg 250 \
  --signature 'deposit(u256)' \
  --arg 200
```

The constructor runs to completion before the debugger pauses. Storage is initialized, and the target function's calldata is encoded from the ABI and `--arg` values.

### Auto-building the TUI binary

The first time you run `ora debug`, the compiler checks for the `ora-evm-debug-tui` binary under `lib/evm/zig-out/bin/`. If it is missing, Ora builds it automatically by running `zig build install` in `lib/evm/`. Subsequent runs reuse the cached binary.

## UI layout

The debugger TUI has five areas:

```text
+--[ Ora Source ]----------+--[ SIR Text ]---+--[ Bindings ]------+
|  10    let fee = ...   . |  sir.mul ...    |  fee = 50          |
|  11>   let net = ...   . |  sir.sub ...    |  net = 950         |
|  12    total += net    . |  sir.sload ...  |  total = 1000 [s]  |
+--------------------------+-----------------+--------------------+
|                          |                 |--[ Machine ]-------+
|                          |                 |  PC: 0x2f  GAS: .. |
+--[ State: Stack  Memory  Storage  TStore  Calldata ]------------+
|  [0] 0x3e8                                                      |
|  [1] 0x32                                                       |
+--[ :step => stopped at line 11 ]--------------------------------+
```

- **Ora Source** — your contract with the current statement highlighted, breakpoint markers in the gutter
- **SIR Text** — the lowered SIR intermediate representation, synchronized to the current Ora line
- **Bindings** — visible Ora names at the current stop point: params, locals, storage fields, comptime-folded constants
- **Machine** — EVM frame state: PC, opcode, gas, call depth, caller, callee, calldata/memory size
- **State** — tabbed EVM machine state: Stack, Memory, Storage, Transient Storage, Calldata
- **Command console** — Vim-style `:command` input with rolling result trail

## Stepping

| Key | Command | Behavior |
|-----|---------|----------|
| `s` | `:step` | Step in — advance to next statement at any call depth |
| `n` | `:next` | Step over — advance to next statement at same or lower depth |
| `o` | `:out` | Step out — run until call depth decreases |
| `c` | `:continue` | Continue — run until breakpoint or execution end |
| `p` | `:prev` | Previous — step backward one debugger stop (replay-based) |
| `x` | `:op` | Opcode step — execute exactly one EVM opcode |

Stepping operates on **statement boundaries**, not individual opcodes. A single Ora statement like `total += net` compiles to dozens of EVM opcodes (loads, arithmetic, stores). The debugger skips through them and stops at the next source-level statement.

Opcode stepping (`x`) is available when you need to inspect mid-statement stack or memory changes.

Reverse stepping (`p`) works by replaying execution from the beginning to one stop before the current position. It is deterministic but not a full time-travel engine.

## SIR text view

When SIR text is available, the debugger shows the lowered intermediate representation side-by-side with Ora source. This is the representation between Ora MLIR and EVM bytecode — the last stage before the assembler produces raw opcodes.

The SIR pane header shows the mapping:

```text
lowered region | ora 26 => sir 42..47 | idx 12..17 | pc 98..134
```

This tells you: Ora line 26 lowered to SIR lines 42 through 47, corresponding to op indices 12-17 and bytecode PCs 98-134.

| Key | Behavior |
|-----|----------|
| `J`/`K` | Scroll the SIR pane independently |
| `=` | Re-sync SIR pane to the current Ora line |

By default, the SIR pane auto-follows as you step. Scrolling with `J`/`K` disables auto-follow; `=` re-enables it.

Range markers in the SIR gutter (`>`, `<`) show the exact SIR line range for the current Ora statement.

## Breakpoints

```text
:break 27              set breakpoint on line 27
:break file.ora:27     set breakpoint with explicit file
:delete 27             remove breakpoint
:info break            list all breakpoints
```

Breakpoint markers appear in the source gutter: `*` for a breakpoint, `>` for the current line with a breakpoint.

## Inspecting values

### Ora bindings

```text
:print total           print a visible binding by name
:print gas             print gas remaining
```

The Bindings pane shows all names visible at the current stop point. Each binding includes its runtime classification:

- **Storage fields** — read from contract storage, shown with current value
- **Memory fields** — read from the reserved debug memory band
- **Transient storage fields** — read from transient storage
- **Comptime-folded constants** — shown with `[folded]` tag and their compile-time value
- **SSA locals/params** — visible but may show as opaque if the optimizer eliminated their storage location

### Raw machine state

```text
:print stack[0]        top of stack
:print slot 0x00       raw storage slot by id
:print mem 0x80 4      memory words starting at offset
:print storage         all storage slots for current address
:print tstore          all transient storage slots
:print calldata        full calldata of current frame
```

## Mutating state

The debugger supports limited live mutation for what-if exploration:

```text
:set total = 1337      set a writable rooted binding
:set gas = 750000      change gas remaining
:set slot 0x00 = 7     set a raw storage slot
:set mem 0x80 = 42     set a raw memory word
```

Only rooted bindings (storage, memory, transient storage fields) are writable. Plain SSA locals cannot be mutated — the debugger does not pretend optimized-away values are stable slots.

## Checkpoints

Save and restore execution positions:

```text
:checkpoint            save current position
:checkpoints           list saved checkpoints
:restart 1             restore checkpoint 1
```

Checkpoints record the replay position plus UI state (scroll, focus, active tab). Restoring a checkpoint replays execution to that point.

## Backtrace and frames

Inspect nested calls:

```text
:bt                    show call stack
:frame 0               select top frame
:frame 1               select parent frame
```

The Machine and State panes follow the selected frame. Frame 0 is the top (innermost) frame.

## Sessions

Sessions let you save a debugging position and come back to it later, or share it with someone else.

### Saving a session

```text
:write-session artifacts/vault/session.json
:ws artifacts/vault/session.json
```

`:ws` is shorthand for `:write-session`. The session is written as JSON to the path you specify.

### Loading a session

```text
:load-session artifacts/vault/session.json
:ls artifacts/vault/session.json
```

`:ls` is shorthand for `:load-session`. Loading a session replaces the current debugger state entirely — it reloads the artifacts, replays the step history, restores breakpoints and checkpoints, and returns you to the exact position you saved.

### What a session contains

A saved session records:

- **Artifact paths** — bytecode, source map, debug info, SIR text, ABI
- **Calldata** — the encoded function call (so the same transaction replays)
- **Step history** — the sequence of step commands (`in`, `over`, `out`, `continue`, `opcode`) that reached the current position
- **Breakpoints** — all active breakpoints by PC
- **Checkpoints** — saved positions with their replay index and UI state
- **UI state** — Ora source scroll position, SIR scroll position, SIR follow mode, focused line, active state tab

Sessions are **replayable, not snapshots**. They do not dump EVM memory or storage. Instead, they record the step commands that produced the current state and replay them on load. This means sessions are small, deterministic, and human-readable JSON.

### Sharing sessions

Sessions are portable across machines if the referenced artifacts exist at the same paths. To share a session:

1. Save the session into the `artifacts/` directory alongside the build output
2. Share the `artifacts/<name>/` directory (it contains the bytecode, source map, debug info, SIR text, ABI, and session file)
3. The recipient runs `:load-session artifacts/<name>/session.json`

Session files do not yet validate artifact hashes, so if the artifacts change between save and load, the replay may diverge.

### Restarting

```text
:run                   restart from the beginning (clears step history)
:rerun                 same as :run
:r                     same as :run
```

`:run` rebuilds the EVM session from scratch and replays to the initial stop point, preserving your breakpoints. This is useful after mutating state to get back to a clean starting point.

## Keyboard reference

```text
s          step in
n          step over
o          step out
c          continue
p          previous stop
x          opcode step
j/k        scroll Ora source
J/K        scroll SIR text
=          sync SIR to current Ora line
[/]        cycle state tabs
1..5       jump to state tab (Stack/Memory/Storage/TStore/Calldata)
:          enter command mode
q          quit
```

Source gutter markers:

```text
.          runtime statement
!          runtime guard (requires/guard)
*          breakpoint
>          current line with breakpoint
>|<        SIR range boundary
```

## CLI reference

```bash
ora debug <file.ora> [options]
```

### Function call

Specify which function to debug and its arguments:

```bash
ora debug vault.ora --signature 'deposit(u256)' --arg 1000
```

| Option | Description |
|--------|-------------|
| `--signature 'fn(type,...)'` | Function to call after deployment |
| `--arg <value>` | Positional argument (repeatable, order matters) |
| `--calldata-hex <hex>` | Raw calldata bytes instead of signature + args |

Arguments are ABI-encoded using the contract's generated ABI. Types in the signature must match the Ora function's parameter types.

### Constructor

If the contract has an `init(...)` function, pass constructor arguments separately:

```bash
ora debug vault.ora \
  --init-signature 'init(u256,u256)' \
  --init-arg 1000 \
  --init-arg 250 \
  --signature 'deposit(u256)' \
  --arg 200
```

| Option | Description |
|--------|-------------|
| `--init-signature 'init(type,...)'` | Constructor signature |
| `--init-arg <value>` | Constructor argument (repeatable, order matters) |
| `--init-calldata-hex <hex>` | Raw constructor calldata bytes |

### Verification

By default, `ora debug` skips Z3 verification to speed up compilation. To include verification:

```bash
ora debug vault.ora --verify --signature 'deposit(u256)' --arg 1000
```

## How comptime values appear

Comptime-evaluated expressions produce no bytecode. In the debugger:

- Lines that were fully folded at compile time have no statement boundary — the debugger skips them during stepping
- Comptime-folded constants appear in the Bindings pane with a `[folded]` tag and their compile-time value
- The SIR view shows where folded values are inlined as constants

```ora
comptime const FEE: u256 = 2 + 3 + 5;   // no bytecode — folded to 10

pub fn apply(amount: u256) -> u256
    requires(amount > FEE)
{
    return amount - FEE;                  // FEE appears as [folded] = 10 in Bindings
}
```

## Current limitations

- Reverse stepping is replay-based, not full time-travel
- Source and bindings reflect the current stop point, not arbitrary frame reconstruction
- Session files do not yet enforce artifact hash validation
- Mutation is limited to rooted bindings and raw machine locations
- Multi-contract calls enter unmapped bytecode (shown as `[external call]`)
- Comptime trace viewer (showing the evaluation tree, not just folded values) is planned but not yet shipped

## Example contracts

The `ora-example/debugger/` directory contains contracts designed for debugger testing:

- **`constructor_value.ora`** — minimal constructor, single storage field
- **`state_walkthrough.ora`** — stateful contract with helper calls and `requires` guards
- **`ledger_walkthrough.ora`** — constructor args, branching, storage updates, helper functions
- **`optimizer_probe.ora`** — comptime constants, runtime guards, branch-dependent behavior, storage slot inspection

# Commands — Ora EVM debugger TUI

Every `:`-prefixed command in `ora-evm-debug-tui`. Authored from
`lib/evm/src/debug_tui.zig` (`Ui.executeCommand` plus its
`handlePrintCommand` / `handleSetCommand` / `handleBreakpoint*` helpers).

Aliases on the same row resolve identically. Commands not in this table
are rejected with status `unknown command`.

## Lifecycle

| Command                       | Aliases                | Action                                                         |
|-------------------------------|------------------------|----------------------------------------------------------------|
| `q`                           | `quit`                 | Quit the debugger                                              |
| `h`                           | `help` / `legend` / `marks` | Print built-in help and gutter-mark legend to the command log |
| `r`                           | `run` / `rerun`        | Restart the session from the seed and replay no history        |

## Stepping (mirror the keybindings)

| Command   | Aliases                                | Action                                                    |
|-----------|----------------------------------------|-----------------------------------------------------------|
| `s`       | `step` / `si` / `in`                   | Step in (next source statement, enter calls)              |
| `x`       | `op` / `opcode`                        | Step exactly one EVM opcode                               |
| `n`       | `next` / `so`                          | Step over — skip nested calls                             |
| `o`       | `out` / `finish`                       | Step out — return from current frame                      |
| `c`       | `continue`                             | Continue until breakpoint or termination                  |
| `p`       | `prev` / `previous`                    | Step back (replay history minus last command)             |

## Source location

| Command                    | Action                                                                  |
|----------------------------|-------------------------------------------------------------------------|
| `line <n>`                 | Move the focus cursor to source line `<n>` and re-center the pane       |
| `line-info <n>`            | Print provenance + statement/idx info for source line `<n>`             |
| `why-line <n>`             | Alias of `line-info <n>`                                                |
| `where`                    | Print provenance + statement info for the current stop                  |
| `why-here`                 | Alias of `where`                                                        |
| `origin`                   | Jump the focus cursor to the origin source line for the current stop    |
| `origin-line`              | Alias of `origin`                                                       |

## SIR pane

| Command           | Aliases                  | Action                                                 |
|-------------------|--------------------------|--------------------------------------------------------|
| `sirline <n>`     |                          | Pin the SIR pane to text line `<n>` (disables follow)  |
| `sirfollow`       | `syncsir`                | Re-enable SIR follow — pane tracks the active op       |

## Breakpoints

| Command            | Action                                                                   |
|--------------------|--------------------------------------------------------------------------|
| `break <line>`     | Set a breakpoint on the first statement PC of `<line>`                   |
| `break <line> when <expr>` | Conditional: only halt when `<expr>` (a `:eval`-grammar predicate) is true |
| `break <line> hit <n>`     | Hit-count: only halt on the `<n>`-th hit of this breakpoint                 |
| `break <line> when <expr> hit <n>` | Both gates apply (predicate AND hit count)                          |
| `delete <line>`    | Remove the breakpoint on `<line>` (no-op if not set)                     |
| `info break`       | List all currently set breakpoints with their conditions and hit counts  |

## Watchpoints

A watchpoint pauses execution when a storage slot's value changes. The
target is either a raw slot (decimal or `0x`-prefixed hex) or a
source-level binding name — bindings resolve to their declared
`storage_field` slot via the active scope's debug info.

| Command                       | Action                                                                   |
|-------------------------------|--------------------------------------------------------------------------|
| `watch <slot|binding-name>`   | Add a watchpoint; halt with `watchpoint_hit` when the slot's value changes |
| `unwatch <id>`                | Remove the watchpoint with id `<id>`                                     |
| `info watch`                  | List active watchpoints with their last-seen values                      |

## Backtrace / call frames

| Command           | Aliases       | Action                                            |
|-------------------|---------------|---------------------------------------------------|
| `bt`              | `backtrace`   | Print the call-frame stack                        |
| `frame <idx>`     |               | Select frame `<idx>` for binding / state queries  |

## Time travel & checkpoints

| Command            | Action                                                                   |
|--------------------|--------------------------------------------------------------------------|
| `checkpoint`       | Save a named checkpoint at the current step                              |
| `checkpoints`      | List all checkpoints with their step indices                             |
| `restart <id>`     | Rewind the session to checkpoint `<id>` (replays history up to that point) |

## Reading state

| Command                 | Reads                                                        |
|-------------------------|--------------------------------------------------------------|
| `gas`                   | Remaining gas + spent-this-step + spent-total                |
| `print gas`             | Same as `gas`                                                |
| `print calldata`        | Calldata of the selected frame                               |
| `print storage`         | Persistent storage slots touched in the selected frame       |
| `print tstore`          | Transient storage slots touched in the selected frame        |
| `print mem <off> <words>` | Words of frame memory starting at `<off>`                    |
| `print stack[i]`        | Stack item at index `i` (0 = top)                            |
| `print slot <hex>`      | Storage slot `<hex>` for the selected frame's address        |
| `print logs`            | Decoded log lane: every emitted `LOG*` rendered as `EventName(arg=value, ...)` via the loaded ABI; falls back to topic/data sizes for unknown selectors |
| `print <binding>`       | Source-level binding by name (resolves to storage / memory / tstore / folded value / "optimized away" / etc.) |

## Coverage

The debugger maintains a per-source-line hit counter — incremented
every time a statement boundary on that line is entered. Re-entering
the same statement (loop iteration, recursion) counts as a fresh hit.
Lines without any statement boundary (whitespace, comments,
non-executable spec lines) never appear.

| Command           | Action                                                   |
|-------------------|----------------------------------------------------------|
| `cov`             | Report the top-10 hottest source lines                   |
| `cov <n>`         | Report the top-`<n>` hottest source lines                |

Output format: `cov: <total> lines hit; top <k>: L<line>=<count> ...`.

## Expression evaluation

Side-effect-free evaluator over visible bindings + numeric literals.
Used by `:eval` and (later) by conditional-breakpoint predicates.

| Command            | Action                                                                    |
|--------------------|---------------------------------------------------------------------------|
| `eval <expr>`      | Evaluate `<expr>` and print the result. Returns numbers as decimal, booleans as `true` / `false`. |

Supported syntax:

- Numeric literals: decimal (`42`) or hex (`0xff`).
- Boolean literals: `true`, `false`.
- Identifiers — resolved through the same binding lookup as `:print
  <name>`. An unknown name reports `unknown identifier`; a name that
  resolves to a non-numeric ABI value reports `binding unavailable`.
- Arithmetic: `+`, `-`, `*`, `/`, `%` — all operate on `u256` with
  wrapping `+%`/`-%`/`*%` semantics; `/` and `%` halt with `division
  by zero` on a zero divisor.
- Comparisons: `==`, `!=`, `<`, `>`, `<=`, `>=`.
- Logical: `&&`, `||`, `!`.
- Parentheses for grouping.

Examples:

```
:eval n + 1
:eval n == 0
:eval (caller != attacker_addr) && (amount > 100)
```

## ABI-decoded reverts

When execution stops with `execution_reverted` and the contract emitted a
revert payload, the status line decodes it against the loaded
`abi.json`:

- Custom errors (4-byte selector + ABI-encoded args) render as
  `ErrorName(field=value, ...)`.
- Solidity-style `Error(string)` reverts (selector `0x08c379a0`) render
  as `Error("...")`.
- Empty / unrecognised payloads fall back to
  `reverted (no decoded payload)`.

Decoding requires the debugger to have been launched with an
`--abi <path>` (the default `ora debug` flow auto-discovers
`artifacts/<stem>/abi/<stem>.abi.json`).

## Writing state

`set` commands modify EVM state in place; subsequent steps see the new
values. Use `:checkpoint` first if you want to roll back later.

| Command                       | Action                                                       |
|-------------------------------|--------------------------------------------------------------|
| `set <binding> = <value>`     | Write to a writable source-level binding (storage / memory / tstore field) |
| `set gas = <value>`           | Override the current frame's `gas_remaining`                 |
| `set slot <hex> = <value>`    | Write to persistent storage slot `<hex>` for the selected frame's address |
| `set mem <offset> = <value>`  | Write a u256 word at memory offset `<offset>`                |
| `gas <value>`                 | Equivalent to `set gas = <value>`                            |

## Sessions

A "session" captures the seed, breakpoints, checkpoints, and step history
so a debugging trace can be replayed in another invocation.

| Command                   | Aliases  | Action                            |
|---------------------------|----------|-----------------------------------|
| `write-session <path>`    | `ws`     | Save current session JSON to path |
| `load-session <path>`     | `ls`     | Load a saved session from path    |

## Marker legend

The source pane gutter and SIR-range bar use single-character markers:

| Mark     | Meaning                                                       |
|----------|---------------------------------------------------------------|
| `.`      | Direct runtime statement — emitted unchanged from source      |
| `~`      | Synthetic-only statement — generated by the compiler          |
| `+`      | Mixed — line has both direct and synthetic origin             |
| `=`      | Folded — value resolved at compile time                       |
| `!`      | Guard statement — runtime safety check                        |
| `-`      | Removed — compiler-elided source line                         |
| `*`      | Breakpoint set on this line                                   |
| `^`      | Origin — current stop's origin statement is on this line      |
| `>` `<`  | SIR-range start / end for the current statement window        |

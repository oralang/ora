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
| `delete <line>`    | Remove the breakpoint on `<line>` (no-op if not set)                     |
| `info break`       | List all currently set breakpoints                                       |

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
| `print <binding>`       | Source-level binding by name (resolves to storage / memory / tstore / folded value / "optimized away" / etc.) |

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

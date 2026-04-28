# Keybindings — Ora EVM debugger TUI

Authoritative reference for `ora-evm-debug-tui`. Every binding listed here
is exercised in `lib/evm/src/debug_tui.zig` (`Ui.handleKey`); the in-app
`:help` is generated from the same table the TUI dispatches off so the two
stay in lockstep.

`q` quits at any time. The TUI is modal: while you're entering a `:`
command the navigation keys are inactive and the buffer accepts text
input.

## Quit

| Key       | Action                                |
|-----------|---------------------------------------|
| `q`       | Quit the debugger                     |
| `Ctrl-C`  | Quit (alias)                          |

## Stepping

| Key  | Action                                                    |
|------|-----------------------------------------------------------|
| `s`  | Step in — execute until the next source statement boundary, entering calls |
| `x`  | Step opcode — execute exactly one EVM opcode              |
| `n`  | Step over (next) — same as step-in but skip nested calls  |
| `o`  | Step out — execute until the current call frame returns   |
| `c`  | Continue — run until a breakpoint or termination          |
| `p`  | Step back — replay history minus the last command         |

## Source / SIR navigation

| Key       | Action                                                     |
|-----------|------------------------------------------------------------|
| `j` / Down  | Scroll source pane down one line                         |
| `k` / Up    | Scroll source pane up one line                           |
| PgDn      | Scroll source pane 8 lines down                            |
| PgUp      | Scroll source pane 8 lines up                              |
| `J`       | Scroll SIR pane down one line                              |
| `K`       | Scroll SIR pane up one line                                |
| `=`       | Resync SIR pane to current op (re-enables SIR follow mode) |

## EVM state tabs

The right-hand pane cycles through stack / memory / storage / tstore /
calldata views.

| Key   | Action                                          |
|-------|-------------------------------------------------|
| `[`   | Previous EVM tab                                |
| `]`   | Next EVM tab                                    |
| `1`   | Stack tab                                       |
| `2`   | Memory tab                                      |
| `3`   | Storage tab                                     |
| `4`   | Tstore (transient storage) tab                  |
| `5`   | Calldata tab                                    |

## Command mode

`:` enters command mode. While in command mode the printable keys append
to the command buffer; the dispatch keys above are inactive.

| Key       | Action                                             |
|-----------|----------------------------------------------------|
| `:`       | Enter command mode                                 |
| Enter     | Execute the command in the buffer                  |
| Esc       | Cancel command mode without executing              |
| Backspace | Delete the last character in the buffer            |

For the command surface itself, see [COMMANDS.md](COMMANDS.md).

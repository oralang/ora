# Getting started — Ora EVM debugger

A 5-minute walkthrough on a tiny counter contract. By the end you'll
know how to step source-level, set breakpoints, inspect storage, and
share a reproduction.

## 1. Build

```sh
cd /path/to/ora
zig build install
```

This produces `zig-out/bin/ora` (the compiler / driver) and the
debugger binaries under `lib/evm/zig-out/bin/`.

## 2. A sample contract

Save as `counter.ora`:

```ora
contract Counter {
    storage var n: u256 = 0;

    pub fn increment() {
        n = n + 1;
    }

    pub fn get() -> u256 {
        return n;
    }
}
```

## 3. Launch the debugger

```sh
ora debug counter.ora
```

The TUI splits the screen:

- **Left**: your `.ora` source with a gutter (statement marks, current
  line, breakpoints).
- **Right top**: the SIR intermediate the compiler emitted, kept in
  sync with the source line you're stopped on.
- **Right bottom**: cycling tabs for stack / memory / storage / tstore
  / calldata.
- **Footer**: status line + the `:` command bar when active.

Default focus is the constructor. Step through it once with `s` —
each press advances one source statement.

## 4. Step through `increment`

Quit (`q`) and re-launch with a function selector:

```sh
ora debug counter.ora --signature 'increment()'
```

Now press `s` — the debugger walks line-by-line through `increment`.
After the statement `n = n + 1` runs, switch to the storage tab (`3`)
and you'll see slot `0` hold the new value.

## 5. Set a breakpoint

Quit, re-launch, then enter command mode with `:`:

```
:break 5
```

That sets a breakpoint on `n = n + 1`. Press `c` to continue — the
debugger runs to the breakpoint, stops, and the gutter shows `*` on
line 5.

## 6. Inspect a binding by name

With execution paused inside `increment`:

```
:print n
```

The status line shows the storage value of `n`. To force-write it:

```
:set n = 42
```

Subsequent steps see `n == 42`. Storage writes are real — they affect
the rest of the trace.

## 7. Time travel

Press `p` to step back one command. Or save a named checkpoint:

```
:checkpoint
:c
:checkpoints      # list
:restart 1        # jump back to checkpoint 1
```

## 8. Share a repro

```
:write-session bug-1234.session
```

Plus the four debug artifacts (saved by default under
`artifacts/<stem>/`):

```
artifacts/counter/counter.hex
artifacts/counter/counter.sourcemap.json
artifacts/counter/counter.debug.json
artifacts/counter/abi/counter.abi.json
```

Anyone with the same Ora version can replay your session against the
same trace.

## What to read next

- [KEYBINDINGS.md](KEYBINDINGS.md) — all keys
- [COMMANDS.md](COMMANDS.md) — all `:` commands
- `:help` (in-app) — generated from the same table the dispatcher
  uses, so it's always current.

## Troubleshooting

- **TUI flashes and exits** — your terminal probably doesn't speak
  ANSI well. Try a different one or use `ora-evm-debug-probe`
  (headless).
- **"binding not visible"** when running `:print foo`** — the binding
  is out of scope at the current PC. Step to inside the relevant
  function first, or use `:where` to see the active scope.
- **Goldens diverge in CI after a compiler change** — run `ora debug
  --no-tui <fixture>.ora -o /tmp/dbg` and copy the new artifacts over
  `tests/debug_artifacts/<fixture>/{sourcemap,debug}.golden.json`.
- **Out of gas** — bump it: `ora debug --gas-limit 50000000
  counter.ora`. Defaults are `ora-evm-debug-tui --help`.

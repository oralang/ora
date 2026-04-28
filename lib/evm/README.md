# Ora EVM debugger

A source-level EVM debugger for Ora smart contracts. Steps your `.ora`
source one statement at a time, shows live storage / memory / tstore /
calldata / stack, decodes call frames against the contract ABI, and
keeps the SIR intermediate representation alongside the source so you
can see exactly what the compiler emitted.

This is also the EVM the Ora compiler uses for its built-in execution
spec tests; the debugger consumes the same artifacts the compiler
emits, with no separate IR build step.

## Layout

```
lib/evm/
├── src/
│   ├── debugger.zig         framework-agnostic core driving the EVM
│   ├── debug_session.zig    shared deploy / artifact-load helpers
│   ├── debug_tui.zig        vaxis-based TUI frontend
│   ├── debug_probe.zig      headless smoke runner over the same core
│   ├── source_map.zig       .sourcemap.json loader (PC → source loc)
│   ├── debug_info.zig       .debug.json loader (scopes, locals, ops)
│   └── ...                  EVM core (frame, storage, opcode tables)
├── test/
│   ├── bench/step_bench.zig per-step microbench (zig build bench)
│   └── specs/               EVM execution spec tests
├── COMMANDS.md              authoritative `:`-command reference
├── KEYBINDINGS.md           authoritative key reference
└── getting-started.md       5-minute walkthrough
```

## Usage

The TUI is launched by `ora debug` from the repo root:

```sh
ora debug path/to/contract.ora
```

That compiles the contract, emits debug artifacts under
`artifacts/<stem>/`, and launches the TUI against them.

For headless / CI / programmatic use, add `--no-tui` to compile-only,
or drive `ora-evm-debug-tui` / `ora-evm-debug-probe` directly. See
their `--help` output for the full flag list (gas / step caps,
artifact size cap, ABI-aware calldata builders).

## Where to start

- New users → [getting-started.md](getting-started.md)
- All keys  → [KEYBINDINGS.md](KEYBINDINGS.md)
- All `:` commands → [COMMANDS.md](COMMANDS.md)

## Build / test

```sh
cd lib/evm
zig build install        # build ora-evm-debug-tui + ora-evm-debug-probe
zig build unit           # run debugger unit tests
zig build bench          # per-step debugger microbench
```

The compiler-side regression corpus (`tests/debug_artifacts/` golden
.sourcemap.json + .debug.json files) runs as part of `zig build
test-compiler` from the repo root; it asserts byte-equality between
`ora debug --no-tui` output and committed goldens.

## Reporting bugs

Please include:

1. The `.ora` source (or a reduced reproducer).
2. The output of `ora debug --no-tui your-contract.ora -o /tmp/dbg`
   (this produces the four artifact files: hex, sourcemap, debug,
   abi).
3. The exact step sequence that triggered the issue. `:write-session
   <path>` saves it; ship the session file along with the artifacts.

If the issue is a per-step performance regression, attach `zig build
bench` output before and after.

## Design notes

The core (`debugger.zig`) deliberately doesn't depend on the TUI; the
probe is the proof. A future DAP frontend (VS Code launch.json) would
plug into the same surface — see the plan in
`/Users/logic/.claude/plans/gentle-exploring-comet.md` (Track C8) for
where that work is sequenced.

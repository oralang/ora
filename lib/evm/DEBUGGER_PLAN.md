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

## Next Debugger Slice

Expose provenance cleanly in the debugger API so a binding can be reported as:

- live runtime value
- folded compile-time value
- optimized out with no recoverable value

That should happen before any UI/TUI work.

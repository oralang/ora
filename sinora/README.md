# Sinora

Sinora is Ora's SIR → EVM bytecode backend, written in Zig. It reads the SIR
text the Ora compiler emits and produces EVM bytecode.

It is a port of [Plank](https://github.com/plankevm/plank-monorepo), the Rust SIR
backend, and produces byte-for-byte identical bytecode.

## Pipeline

```
parse → legality → analyses → SSA / critical-edge transforms
      → optimization → (release) effect-aware schedule → memory layout
      → codegen → assemble
```

Two code generators share the front of the pipeline:

- **Debug** (`debug_codegen.zig`) — a single-function emitter.
- **Release** (`release_generic_backend.zig`) — effect-aware scheduling, static
  memory layout, and generic lowering.

## Source layout

| File | Responsibility |
|------|----------------|
| `ir.zig` | SIR data model — functions, blocks, instructions, terminators, `data` segments |
| `parser.zig` | Line-oriented parser for SIR text |
| `render.zig` | Normalized renderer for parsed SIR |
| `legality.zig` | Structural legality checks (blocks, values, terminator targets, duplicate defs) |
| `ops.zig` | Opcode metadata for the SIR operation surface |
| `analyses.zig` | Reachability, predecessors, RPO, dominators, dominance frontiers, def-use, liveness, effects, CFG in/out bundling |
| `passes.zig` | Pass framework with cached analyses and invalidation masks |
| `ssa_transform.zig` | Sealed-block SSA construction |
| `release_critical_edges.zig` | Critical-edge splitting |
| `optimizations.zig` | SCCP, copy propagation, dead-op elimination, defragmenter, switch peephole |
| `effects.zig` | Effect model (memory, returndata, accounts, storage, transient, revert/terminate, alloc, logs) |
| `release_schedule.zig` | Effect-aware release scheduling |
| `release_memory_layout.zig` | Static memory layout and runtime-relative offsets |
| `release_op_graph.zig` | Release op-graph construction |
| `release_code_to_asm.zig` | Release codegen → assembly |
| `asm.zig` | Bytecode assembler with minimal-width label patching |
| `diagnostics.zig` | Diagnostics |
| `main.zig` | CLI |
| `sinora.zig` | Library root |

## Usage

```sh
zig build                              # build the `sinora` binary
zig build run -- path/to/program.sir   # parse, check, and render SIR
zig build run -- emit-release path/to/program.sir
```

## Tests

```sh
zig build test                   # unit tests
```

`fixtures/` holds focused `.sir` inputs that pin specific lowering shapes.

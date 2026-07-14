---
title: "Sinora: Ora's Zig EVM Backend"
description: How Ora ported Plank's SIR backend to Zig, adopted Vyper's three-mode dispatcher architecture, and extended both for Ora.
---

# Sinora: Ora's Zig EVM Backend

Sinora is Ora's owned backend for compiling Sensei-IR (SIR) into EVM
bytecode. It is written in Zig and is built as part of the Ora compiler.

Sinora began as a port of
[Plank](https://github.com/plankevm/plank-monorepo), a Rust SIR backend. The
port preserved Plank's core backend architecture while moving ownership into
Ora's codebase. Ora then extended that baseline with source-map support,
deterministic metrics, optimization profiles, a Vyper-inspired dispatcher
architecture, and formal dispatcher evidence.

```text
Ora source
  -> frontend and semantic analysis
  -> Ora MLIR
  -> Sensei-IR (SIR)
  -> Sinora
  -> EVM deployment bytecode
```

Most Ora users never invoke Sinora directly. `ora build` lowers the contract to
SIR and calls Sinora as the final bytecode backend.

## Why Ora ported Plank

Plank already had the backend machinery Ora needed:

- a compact SIR model;
- block-parameter SSA;
- structural legality checks;
- control-flow and liveness analyses;
- effect-aware EVM stack scheduling;
- static memory layout;
- debug and release code generators;
- a label-patching EVM assembler.

Rewriting those ideas from scratch would have discarded a working reference.
Keeping Plank as an external Rust backend, however, left several architectural
problems.

### One compiler should own the bytecode boundary

The Ora compiler is written primarily in Zig. Depending permanently on a
separate Rust backend meant that the final and most security-sensitive lowering
step lived behind another build system, process boundary, and ownership model.

Moving the backend to Zig gives Ora one repository and one toolchain for:

- SIR production and consumption;
- bytecode generation;
- diagnostics;
- source maps and debugger metadata;
- metrics and regression gates;
- formal evidence emitted during code generation.

The purpose was not "Zig instead of Rust" as an end in itself. The purpose was
to make the backend an owned, inspectable part of the compiler's correctness
boundary.

### Ora needed to evolve the backend

Ora's roadmap requires backend behavior that is coupled to the rest of the
compiler:

- source-stable debugging through optimized bytecode;
- selectable gas-versus-size policies;
- dispatcher plans that can be independently checked;
- deterministic compiler metrics;
- fail-closed integration with artifact emission;
- future translation validation of stack schedules and lowering templates.

Those features are easier to implement and audit when the backend is a native
Ora component rather than a permanently external oracle.

## Porting without guessing

Sinora was not accepted because its output looked plausible. During the port,
the same SIR corpus was sent to both backends:

```text
                 +-> Rust Plank -> bytecode A
same SIR input --|
                 +-> Zig Sinora -> bytecode B

required migration result: bytecode A == bytecode B
```

The port proceeded subsystem by subsystem:

1. Parse and render Plank-compatible SIR.
2. Reject malformed SIR before code generation.
3. Reproduce the memory-backed debug backend.
4. Port critical-edge normalization and block layouts.
5. Port the effect-aware stack scheduler.
6. Port release memory layout and assembly emission.
7. Compare focused fixtures and the full corpus byte for byte.

This oracle discipline made Plank an executable specification for the port. It
caught differences in successor ordering, stack layout, spill behavior, memory
allocation, and label widths that ordinary execution tests could miss.

After the owned backend reached parity and the integration gates were in place,
the Rust dependency was removed. Sinora has since gained intentional Ora
extensions, so present-day output is not defined as "whatever an old Plank
revision emitted."

Byte equality was migration evidence, not a mathematical proof that either
backend was correct.

## The current pipeline

The production release path is:

```text
SIR text
  -> parse
  -> structural legality
  -> literal commoning
  -> short-circuit branch threading
  -> critical-edge splitting
  -> effect analysis
  -> effect-aware stack scheduling
  -> static memory layout
  -> EVM assembly generation
  -> label and offset resolution
  -> deployment bytecode
```

### Parse and legality

Sinora parses line-oriented SIR into functions, blocks, block inputs and
outputs, instructions, terminators, switch cases, and data segments.

The legality pass rejects unsupported or malformed input before codegen. It
checks properties such as:

- unique function, block, and value definitions;
- resolved control-flow targets;
- known operations and legal arity;
- defined SSA value uses;
- valid internal-call targets and signatures;
- valid data-segment references.

This is a structural barrier, not a proof of EVM semantics.

### Block-parameter SSA

SIR represents values crossing control-flow edges as block inputs and outputs.
This is the same role traditionally served by phi nodes.

```sir
entry -> value {
    value = const 42
    => @done
}

done incoming {
    return incoming incoming
}
```

The release backend must arrange for `value` to occupy the stack position that
`done` expects for `incoming`. Critical-edge splitting provides dedicated
forwarding blocks where different incoming layouts need to be reconciled.

### Effect-aware stack scheduling

The EVM requires operands to be reachable on a stack with only `DUP1..DUP16`
and `SWAP1..SWAP16`. SIR instead describes values and dependencies.

Sinora builds a per-block operation graph containing:

- data dependencies;
- memory effects;
- storage and transient-storage effects;
- account and returndata effects;
- logs;
- termination and revert effects;
- internal-call effects.

The scheduler chooses a legal operation order and emits symbolic stack actions:

```text
dup, swap, pop, spill, reload, operation
```

Values that cannot remain reachable are spilled to compiler-owned memory slots.
The scheduler is deterministic and effect-aware, but it is a greedy scheduler,
not a globally optimal gas solver.

### Memory layout and assembly

Sinora allocates compiler scratch memory for spills, static allocations, switch
tables, and the dynamic free pointer. Init code and runtime code receive
separate layouts because they execute in separate EVM memory lifetimes.

The assembler resolves symbolic labels after code generation and chooses the
smallest valid `PUSH` width for each reference. Because shortening one reference
changes later byte offsets, label widths are solved to a fixed point before
final bytecode is returned.

## Debug and release backends

Sinora has two code-generation strategies.

| Backend | Value representation | Purpose |
|---------|----------------------|---------|
| Debug | Locals materialized in memory | Simple, inspectable execution and debugging |
| Release | Values scheduled primarily on the EVM stack | Production bytecode with reduced memory traffic |

The debug backend uses memory slots and a transfer buffer to move block outputs
to successor inputs. The release backend uses stack layouts, liveness, spills,
and shuffling.

A program working in debug mode does not by itself prove that the release stack
scheduler can represent the same control-flow shape. Both paths have their own
tests.

## What Ora added

Sinora is no longer only a transliteration of Plank. Ora has extended the
backend in several deliberate areas.

### Native Ora integration

Sinora builds with Ora and is invoked automatically for bytecode emission. A
backend failure stops artifact generation and reports the preserved SIR input
for diagnosis. Empty or malformed output is rejected rather than accepted as
best-effort bytecode.

The standalone executable remains available for compiler work, but production
users interact through `ora build` and `ora emit`.

### Release source maps

The release backend can emit bytecode source-map entries alongside optimized
code. Ora merges those entries with source, MLIR, and SIR metadata so the
debugger can relate final program counters back to source statements.

Source mapping is part of the production release pipeline, not restricted to
the memory-heavy debug backend.

### Ora-specific optimization passes

Sinora retains Plank-derived analyses and optimization machinery and adds
passes needed by Ora's generated SIR. The mandatory release preparation path
currently includes:

- literal commoning;
- short-circuit branch threading;
- critical-edge normalization.

The broader pass library also contains SCCP, copy propagation, unused-operation
elimination, defragmentation, and switch peephole optimization. Optimization
passes run through an analysis-invalidation discipline and finish with legality
checking.

High-level language optimization remains in MLIR. Sinora optimizes low-level SIR
and should not recreate Ora semantic analysis.

### Dispatcher planning

Sinora inherited generic switch lowering from Plank. The decision to organize
external-function dispatch around three routing strategies came from Vyper's
selector-table design. Vyper introduced this architecture in its
[2023 O(1) selector-table work](https://github.com/vyperlang/vyper/commit/408929fa31ae01dde4f7566bb7babbc7da5b6620):

- **Linear:** compare the selector with each known selector in sequence.
- **Sparse:** use the selector to choose a bucket, then linearly probe the few
  selectors in that bucket.
- **Dense:** use two-level perfect hashing to locate a compact function record.

Vyper selects among those modes primarily from the optimization objective and
function count. Its current implementation uses linear dispatch when
optimization is disabled or the contract is small, sparse dispatch for larger
gas-optimized selector sets, and dense dispatch for code-size optimization.
See Vyper's
[strategy selection](https://github.com/vyperlang/vyper/blob/d3119ff9cda4c6d65dbc6e693d582e060c4bcf7d/vyper/codegen_venom/module.py#L129-L150)
and
[jump-table construction](https://github.com/vyperlang/vyper/blob/d3119ff9cda4c6d65dbc6e693d582e060c4bcf7d/vyper/codegen/jumptable_utils.py).

Ora adopted that high-level three-mode architecture, not Vyper's Python
implementation. Sinora implements its own planner, candidate search, table
layout, and EVM lowering:

- **Linear:** exact-selector checks in sequence.
- **Sparse:** table-select a bucket, then exact-scan that bucket.
- **Dense:** direct table routing through either a collision-free selector bit
  window or deterministic multiplicative perfect hashing.

The algorithmic relationship is therefore influence followed by redesign:

| Concern | Vyper reference design | Ora/Sinora design |
|---------|------------------------|-------------------|
| Strategy set | Linear, sparse, dense | Linear, sparse, dense |
| Selection | Function-count thresholds plus optimization mode | Finite candidate search scored by runtime cost and deployed bytes |
| Sparse index | `method_id % bucket_count` | Selected bit window and shift |
| Sparse collision handling | Exact linear scan inside the selected bucket | Exact linear scan inside the selected bucket |
| Dense index | Two levels: modulo bucket selection, then a per-bucket multiplicative magic | Direct collision-free bit window or deterministic multiplicative hash |
| Safety check | Loaded selector record is checked against the complete selector | Every compressed route rechecks the complete selector |
| Source-level priority | Not part of the three-mode mechanism | Mutability ordering, hot-prefix splitting, and `@callHint` |
| Verification | Compiler tests and selector-table invariants | Per-contract Lean correspondence and routing proofs in addition to execution tests |

This attribution is separate from Sinora's backend lineage: Plank supplied the
starting SIR backend architecture; Vyper supplied the three-mode dispatcher
idea; Ora owns the present implementation and its extensions.

Sparse and dense routes retain an exact-selector guard. An unknown selector that
aliases a bucket or table slot must still reach the default revert path.

The planner scores runtime checks and deployed bytes under three profiles:

| Profile | Objective |
|---------|-----------|
| `gas` | Prioritize recurring dispatch gas |
| `balanced` | Balance runtime gas and deployed size; default |
| `size` | Penalize deployed bytes more heavily |

```sh
ora build --optimize=gas contract.ora
ora build --optimize=balanced contract.ora
ora build --optimize=size contract.ora
```

The profile changes layout and cost, not selector-to-function semantics.

### Dispatcher formal evidence

Sinora records dispatcher facts while it emits final bytecode. The formal lane
uses those facts together with the emitted SIR dispatcher manifest.

For `ora --lean-proofs`, the per-contract checker establishes that:

- intended selectors are covered;
- successful dispatch reaches the intended label;
- unknown selectors revert;
- dense indices are collision-free over known selectors;
- sparse buckets retain exact-selector scans;
- the emitted planner choice matches the Lean reference planner;
- the checked SIR dispatcher is bound to the emitted bytecode dispatcher.

The Lean theorem proves routing semantics, not optimization quality. It does not
prove that a `@callHint(likely)` function received the cheapest position, for
example.

### Deterministic metrics

Sinora can report deterministic release-shape metrics, including:

- input, commoned, and normalized IR counts;
- bytecode bytes;
- source-map entries;
- dispatcher strategy and table statistics;
- stack operations and spill counts;
- init and runtime memory-layout sizes.

The metrics intentionally avoid wall-clock timings in checked snapshots. Stable
shape counters make optimization regressions reviewable across machines.

#### Generate dispatcher metrics

Ora's `--metrics` option and Sinora's metrics file answer different questions:

- `ora --metrics` prints compiler phase timings, allocation data, and frontend
  work counters.
- `sinora emit-release --metrics <path>` writes deterministic backend-shape
  JSON, including the selected dispatcher strategies.

To inspect the dispatcher for an Ora contract, first preserve its SIR and then
run Sinora over that exact backend input:

```sh
zig build sinora

./zig-out/bin/ora emit \
  --emit=sir-text \
  --out-dir /tmp/ora-erc20-metrics \
  ora-example/apps/erc20.ora

sinora/zig-out/bin/sinora emit-release \
  --optimize=gas \
  --metrics /tmp/erc20-gas-metrics.json \
  /tmp/ora-erc20-metrics/erc20.sir \
  >/dev/null

jq '{bytecode_bytes, switch_routing: (.switch_routing | {
  switches,
  cases,
  largest_switch_cases,
  chosen_linear,
  chosen_sparse,
  chosen_dense,
  best_dense
})}' /tmp/erc20-gas-metrics.json
```

The redirect suppresses Sinora's bytecode line; the metrics remain in the JSON
file. Use `--optimize=balanced` or `--optimize=size` to measure the other
objectives against the same SIR.

The strategy fields count switches, not functions:

```text
chosen_linear + chosen_sparse + chosen_dense = switches
```

`chosen_*` records what was emitted. `best_sparse` and `best_dense` describe
the best candidate of each class even when the planner ultimately chose a
different strategy. A non-null `best_dense` therefore does not by itself mean
that dense routing was emitted.

#### Example: a small switch remains linear

The checked three-case fixture is cheaper as a sequence of exact comparisons:

```sh
sinora/zig-out/bin/sinora emit-release \
  --metrics /tmp/linear.json \
  sinora/fixtures/dispatcher_metrics/linear_small_3.sir \
  >/dev/null

jq '{bytecode_bytes, switch_routing: (.switch_routing | {
  switches, cases, chosen_linear, chosen_sparse, chosen_dense
})}' /tmp/linear.json
```

Current output:

```json
{
  "bytecode_bytes": 71,
  "switch_routing": {
    "switches": 1,
    "cases": 3,
    "chosen_linear": 1,
    "chosen_sparse": 0,
    "chosen_dense": 0
  }
}
```

#### Example: a compact 20-case switch becomes dense

The 20-case fixture admits a collision-free five-bit window. Under the default
balanced profile, Sinora emits one dense route with a 32-slot table:

The fixture filename predates the current cost model; its present measured role
is dense-routing coverage despite the `sparse_` prefix.

```sh
sinora/zig-out/bin/sinora emit-release \
  --metrics /tmp/dense.json \
  sinora/fixtures/dispatcher_metrics/sparse_even_odd_20.sir \
  >/dev/null

jq '{bytecode_bytes, switch_routing: (.switch_routing | {
  switches, cases, chosen_linear, chosen_sparse, chosen_dense, best_dense
})}' /tmp/dense.json
```

Relevant output:

```json
{
  "bytecode_bytes": 448,
  "switch_routing": {
    "switches": 1,
    "cases": 20,
    "chosen_linear": 0,
    "chosen_sparse": 0,
    "chosen_dense": 1,
    "best_dense": {
      "kind": "bit_window",
      "table_slots": 32,
      "used_slots": 20,
      "hole_slots": 12,
      "index_bits": 5,
      "index_shift": 0,
      "runtime_selector_eq_checks": 1
    }
  }
}
```

`runtime_selector_eq_checks: 1` is the safety-relevant part: table lookup does
not authorize a function by itself. The landing route still compares the full
selector before entering the function.

#### Example: the profile changes the ERC-20 plan

Using the generated `erc20.sir` above, current measurements are:

| Profile | Bytecode bytes | Linear switches | Sparse switches | Dense switches | Dense kind |
|---------|----------------|-----------------|-----------------|----------------|------------|
| `gas` | 7,940 | 1 | 0 | 1 | Bit window |
| `balanced` | 7,836 | 2 | 0 | 0 | Candidate only |
| `size` | 7,836 | 2 | 0 | 0 | Candidate only |

The gas profile spends 104 additional deployment bytes to table-route the
larger switch. Balanced and size keep both switches linear. The source-level
selector semantics are unchanged in all three builds.

These are planner and bytecode-shape metrics, not measured transaction gas.
Runtime and deployment gas measurements belong to Ora's conformance metrics
harness; the planner's `*_checks_x1000` fields are deterministic cost-model
units used to compare plans.

### Formal drift guards

The repository formal gate snapshots Sinora's optimization vocabulary and
mandatory release stages into Lean. Kernel-checked equality tests catch drift
between the compiler's declared pipeline and the formal specification.

This is a synchronization guard. It is not a theorem that every Sinora pass
preserves program semantics.

## What Sinora deliberately does not own

Sinora receives already-lowered SIR. It should not understand:

- Ora resource or refinement semantics;
- source-level types and assignability;
- ABI policy decisions;
- storage-place typing;
- SMT formulas or Lean proof obligations;
- high-level control-flow semantics already resolved by MLIR lowering.

For example, Sinora emits the `sload`, checked arithmetic, branch, and `sstore`
operations it receives. It does not know that those operations originated from
an Ora `@move` resource transition.

Keeping this boundary narrow makes the bytecode backend easier to audit. New
language features should normally lower into existing SIR concepts rather than
add source-language knowledge to Sinora.

## Correctness and trust boundary

Sinora has strong regression evidence, but Ora does not claim a fully verified
SIR-to-EVM compiler.

Current evidence includes:

- byte-for-byte comparison against Plank during the port;
- focused parser, legality, scheduler, memory, and assembler tests;
- SIR and bytecode golden tests;
- execution through Ora's EVM implementation and conformance corpus;
- deterministic bytecode-size and gas baselines;
- source-map tests;
- per-contract Lean verification of dispatcher routing;
- Lean sync checks for selected backend facts.

Still outside the proved boundary:

- a universal semantics-preservation theorem for every Sinora pass;
- a formal proof that arbitrary scheduled stack programs implement their SIR;
- a complete proof that SIR facts reflect all final EVM behavior;
- global optimality of the stack scheduler or dispatcher planner.

This distinction is intentional: tests, parity, metrics, and synchronization
guards are evidence; kernel-checked theorems are proofs only of their stated
models and correspondence checks.

## Using Sinora directly

The standalone CLI is primarily a compiler-development tool.

```sh
zig build sinora

cd sinora
zig build run -- check fixtures/flat_runtime_return.sir
zig build run -- render fixtures/flat_runtime_return.sir
zig build run -- emit-release fixtures/flat_runtime_return.sir
zig build run -- trace-release fixtures/flat_runtime_return.sir
```

Release emission also accepts source-map and metrics outputs:

```sh
zig build run -- emit-release \
  --source-map /tmp/source-map.json \
  --metrics /tmp/release-metrics.json \
  --optimize=balanced \
  fixtures/flat_runtime_return.sir
```

For ordinary contract development, use the Ora compiler. See
[Compiler Architecture](./research/compiler-architecture) for the phases before
Sinora and [Formal Verification](./formal-verification) for the proof lanes that
operate around the backend.

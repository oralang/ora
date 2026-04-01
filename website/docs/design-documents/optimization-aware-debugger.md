# Optimization-Aware Debugger

Status: implemented

## Context

Ora wants a debugger that remains truthful even when the compiler performs
real optimization. This is a harder requirement than a traditional
`line -> pc` debugger.

Once MLIR, SIR, or bytecode lowering starts to:

- hoist computations
- reorder guards
- duplicate control-flow regions
- fold expressions
- merge blocks
- change stack discipline

the runtime execution order is no longer the source order. A debugger that
derives stepping from source line changes alone will become incorrect.

This is the same class of problem that has historically made Solidity
debugging weak under optimization.

Ora should treat this as a first-class design problem, not a bug-fix tail.

## Problem Statement

Today the debugger already shows the consequences of this gap:

- source-level stepping can stop at the wrong place when backend work is
  hoisted
- a source line may correspond to a multi-line SIR region, not a single op
- a runtime stop may belong to compiler-generated work derived from an earlier
  Ora statement
- line-based source maps are not strong enough to remain stable under
  optimization

The debugger must explain both:

- what source statement the user is conceptually in
- what optimized runtime work is actually executing

If we fail to encode that distinction, source stepping will continue to drift
as optimization improves.

## Design Principle

Source lines are presentation.

Statement identity is debugger truth.

Program counter is machine truth.

Provenance connects them.

This means Ora must stop treating line numbers as the primary executable
identity. They are useful for display, but not strong enough to survive
optimization.

## Required Model

Each meaningful source-level statement should get a stable `statement_id`.

That identity should be carried through:

- Ora AST / sema
- MLIR
- SIR
- bytecode sidecar metadata
- debugger state

Each lowered operation should also carry enough provenance to explain how it
relates to source intent.

Minimum metadata:

- `statement_id`
- `origin_statement_id`
- `origin_scope_id`
- `sir_op_id`
- `is_synthetic`
- `is_hoisted`
- `is_duplicated`
- `is_folded`
- `optimized_out_reason`

Minimum range mappings:

- `statement_id -> sir range`
- `sir op -> pc range`
- `pc -> active sir op`

## Stepping Model

The debugger should expose two distinct stepping models.

### 1. Source Step

This moves by Ora statement identity.

Properties:

- stable under backend reordering
- appropriate for most users
- may compress several lowered ops into one source step
- may remain on the same source statement while execution moves through hoisted
  or synthetic work

### 2. Execution Step

This moves by actual lowered/runtime execution.

Properties:

- follows real SIR/bytecode order
- exposes optimization effects honestly
- appropriate for debugging backend behavior, stack discipline, and lock/guard
  lowering

The existing opcode step belongs to this second category.

## UI Requirements

The debugger should never imply a false 1:1 mapping when optimization has
broken it.

It should instead surface messages like:

- `executing hoisted guard for Ora statement 42`
- `Ora line 29 expands to SIR lines 101..109`
- `current pc belongs to duplicated branch from line 58`
- `current op is synthetic lowering for checked add`

The user should always be able to see:

- active Ora statement
- active SIR region
- active op index
- active pc range

## Source Map Contract

The source map must stop using line changes as the only signal for statement
boundaries.

It should instead be driven by `statement_id` transitions.

A source-map entry should eventually include at least:

- `line`
- `col`
- `statement_id`
- `kind`
- `sir_line`
- `sir_range_start`
- `sir_range_end`
- `idx`
- `pc`
- `origin_statement_id`
- optimization/provenance flags

This allows the debugger to answer:

- which source statement is active
- whether the current work is hoisted or synthetic
- whether a stop is part of the same logical source step

## Optimization Provenance

Optimization is not an exceptional case. It is normal compiler behavior.

The debugger must therefore carry optimization provenance explicitly, not try
to recover it later from text layout.

Examples:

- `folded`
  - a source expression was constant-folded
- `optimized_out`
  - a source statement no longer has a direct executable region
- `hoisted`
  - execution moved earlier than source order
- `duplicated`
  - the same source statement appears in several lowered regions
- `synthetic`
  - compiler-generated support logic such as overflow checks, return-buffer
    materialization, or guard scaffolding

This provenance should be available to both:

- source rendering
- command-line inspection

## Why This Matters

Without this model:

- stepping becomes inconsistent
- breakpoints land in surprising places
- source and machine panes disagree
- optimized code looks like a debugger bug

With this model:

- Ora can explain optimization honestly
- source-level debugging remains stable
- SIR and bytecode views become an advantage instead of a liability

This is one of the clearest ways Ora can differentiate from Solidity tooling.

## Implementation Plan

### Phase 1: Statement Identity — done

- `statement_id` assigned in AST/sema and propagated through MLIR, SIR, and bytecode source-map sidecars
- `origin_statement_id` tracks derivation through lowering
- `execution_region_id` and `statement_run_index` distinguish duplicated/repeated statement instances

### Phase 2: Source Map Rework — done

- Debugger stops derived from `statement_id` transitions via `StatementKey` (combining `stmt_id`, `execution_region_id`, `statement_run_index`, `depth`)
- Line numbers retained as display metadata only
- `currentStatementKeyChanged()` is the single stepping predicate

### Phase 3: Provenance Flags — done

- `is_synthetic`, `is_hoisted`, `is_duplicated` propagated through source maps and debug info
- Line-level provenance classification: `.direct`, `.synthetic`, `.mixed`
- Statement kinds: `.runtime`, `.runtime_guard`
- Removed lines detected (source declarations with no SIR/bytecode coverage)

### Phase 4: Debugger Semantics — done

- Source step (`s`/`n`/`o`) advances by statement identity
- Opcode step (`x`) advances by single EVM opcode (execution-level)
- `continue`, reverse replay (`p`), and checkpoints all use the same statement-identity model

### Phase 5: UI and Commands — done

- Gutter markers show per-line provenance (`.` direct, `~` synthetic, `+` mixed, `!` guard, `-` removed, `^` origin)
- Removed lines rendered in gray
- SIR pane header shows `stmt`, `region`, provenance label, origin, and effect kind
- Source pane header shows runtime mapping with kind/provenance
- Console trace shows provenance detail per step
- `:line-info`, `:why-line`, `:where`, `:why-here`, `:origin` commands for provenance inspection

## Non-Goals

This document does not propose:

- disabling optimization for debugging by default
- pretending source order always equals execution order
- hiding compiler-generated work from the user

The goal is not to simplify the machine into a lie. The goal is to make the
optimized machine explainable.

## Success Criteria

We should consider this design successful when:

- source stepping remains stable under optimization
- execution stepping remains honest to the machine
- the debugger can explain hoisted and synthetic work explicitly
- source, SIR, and pc views no longer disagree due to line-only mapping

## Alternatives Considered

### Line-Only Source Maps

Rejected. They are too weak under optimization and already require heuristics.

### Debug Builds With Fewer Optimizations

Useful as a secondary mode, but not sufficient. The release debugger must be
able to explain real optimized output.

### VM-Only Debugging

Rejected. It gives machine truth but not language truth. Ora needs both.

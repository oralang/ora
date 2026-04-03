# Comptime Phase 1: Pure Compile-Time Evaluation

Status: proposed

## Context

Ora already has real comptime features:

- `comptime const`
- `@import(...)`
- comptime type/value parameters
- comptime blocks
- partial folding of pure expressions
- generic specialization

The remaining gap is not whether comptime exists. The gap is that some
user-visible pure functions with loops or recursion still do not fold cleanly
in normal source examples.

Examples today:

- loop-heavy pure helpers like `power(2, 10)`
- recursive helpers like `factorial(5)`

Ora should close that gap before attempting broader runtime unrolling or more
aggressive partial evaluation.

## Problem Statement

We need a clear execution model for pure comptime evaluation.

That model must answer:

- what code is allowed to execute at comptime
- what loops are allowed
- what recursion is allowed
- which limits stop evaluation
- what happens when evaluation cannot finish
- how results and failures are exposed in diagnostics and debug metadata

Without this design, loop/recursive comptime support will drift into a mix of:

- ad hoc folding
- inconsistent evaluator fallback
- poor diagnostics
- unclear boundaries between comptime execution and runtime optimization

## Non-Goals

This phase does not define:

- runtime loop unrolling
- speculative symbolic execution
- storage or environment access at comptime
- general host-powered metaprogramming
- IO, filesystem, time, randomness, or network access
- hidden code generation

Those are separate features or explicitly out of scope.

## Design Principle

Phase 1 comptime evaluation is:

- pure
- hermetic
- deterministic
- bounded
- fail-closed

If a computation is not pure or exceeds configured limits, it does not
"partially maybe fold". It stays unresolved and the compiler reports a
deterministic comptime failure or falls back only where the language model
explicitly allows fallback.

## Execution Model

### Eligible Computations

Phase 1 comptime evaluation may execute:

- integer arithmetic
- boolean logic
- comparisons
- struct / tuple / array / slice / map construction and mutation
- local variable declarations and assignment
- `if`
- `switch`
- `while`
- `for`
- `break`
- `continue`
- pure direct function calls
- pure recursive function calls
- pure generic function calls

### Ineligible Computations

Phase 1 comptime evaluation must reject:

- storage reads or writes
- transient storage reads or writes
- logs/events
- extern calls
- runtime environment reads:
  - `msg.sender`
  - `msg.value`
  - block / tx globals
- nondeterministic builtins
- any effectful operation

### Purity Rule

Comptime execution is allowed only when the entire evaluated region is pure.

That means:

- the called function body must be pure
- every transitively called function must be pure
- every value used by the region must be comptime-known

If either condition fails, the evaluator must stop and report why.

## Loop Semantics

Phase 1 loops are executed by the comptime evaluator, not by a runtime
unroller.

That means:

- the loop runs in the evaluator
- the result is folded into a constant
- no runtime loop remains for that folded region

### Allowed Loop Forms

Phase 1 supports:

- `while`
- integer `for`
- array / tuple iteration `for`

as long as:

- the loop condition or iterable is comptime-known
- the loop body is pure

### Loop Progress Requirement

We should not try to prove termination formally in Phase 1.

Instead:

- evaluator step limits are the primary safety mechanism
- loop iteration limits are the secondary guard

This matches the current evaluator limits design and keeps the model simple.

If a loop fails to terminate within limits:

- evaluation stops
- the diagnostic must identify the loop site
- the reason must be explicit:
  - step limit
  - loop iteration limit

## Recursion Semantics

Phase 1 recursion is allowed only for pure functions with comptime-known
arguments.

We do not need a separate semantic model from loops.

The governing rule is:

- recursion is bounded by recursion depth and total evaluation steps

If recursion exceeds limits:

- evaluation stops deterministically
- the diagnostic must identify the call site or function
- the reason must be explicit:
  - recursion depth limit
  - step limit

## Limits Model

Ora already has the right shape in `src/comptime/limits.zig`.

Phase 1 should treat those limits as part of the public comptime contract:

- recursion depth
- loop iteration count
- total evaluation steps
- memory budget
- aggregate length limits

These are not just implementation details. They define the safety boundary for
comptime execution.

### Limit Policy

Recommended policy:

- keep deterministic defaults
- keep stricter presets for tests
- keep permissive presets only for opt-in experimentation

The default release policy should remain conservative and predictable.

## Failure Model

Phase 1 must fail clearly.

Three outcomes are allowed:

1. Folded successfully
- the expression becomes a constant

2. Rejected as not comptime-executable
- because purity or comptime-known input requirements were not met

3. Aborted by comptime limits
- recursion depth
- loop iteration count
- step count
- memory budget

What Phase 1 should avoid:

- silent partial execution
- host-dependent results
- fallback that changes meaning without explanation

## Diagnostics

Diagnostics should be explicit and local.

Minimum messages:

- `comptime execution requires compile-time known inputs`
- `comptime execution rejected: storage access is not allowed`
- `comptime execution exceeded recursion depth limit`
- `comptime execution exceeded loop iteration limit`
- `comptime execution exceeded step limit`

Diagnostics should point to:

- the source expression
- the function call or loop site
- the specific failure reason

## Provenance and Debugging

Comptime-folded work should leave source-visible evidence even though it emits
no runtime code.

Phase 1 debugger expectations:

- folded constants remain visible as derived/folded bindings
- fully removed comptime-only lines show as removed source
- source maps do not invent runtime stops for comptime-eliminated code

We do not need a full comptime trace viewer in Phase 1.

But we should preserve enough metadata for later:

- folded value
- fold origin line
- failure reason when folding did not happen

## Initial Milestone Targets

Phase 1 implementation should explicitly target these user-visible outcomes:

1. `power(2, 10)` folds in normal source examples
2. `factorial(5)` folds in normal source examples
3. loop-based pure helper functions in comptime blocks fold deterministically
4. recursion-limit failures are diagnosed cleanly
5. impure functions remain runtime-only and are explained honestly

## Implementation Notes

Implementation should prefer reusing the shared comptime subsystem rather than
growing new one-off AST-only logic.

Relevant existing pieces:

- `src/comptime/eval.zig`
- `src/comptime/compiler_ast_eval.zig`
- `src/comptime/compiler_const_bridge.zig`
- `src/comptime/limits.zig`

Practical sequence:

1. make loop-heavy pure calls fold through the shared evaluator path
2. make recursive pure calls fold through the same path
3. tighten diagnostics for limit hits and purity rejection
4. only then broaden the surface area

## Alternatives Considered

### 1. Add runtime loop unrolling first

Rejected for Phase 1.

That is a different problem:

- code size
- optimizer policy
- debugger/source-map complexity

It should not be bundled into comptime evaluation.

### 2. Keep ad hoc constant folding only

Rejected.

That leaves Ora with an inconsistent model where:

- simple expressions fold
- real pure helpers do not
- users cannot predict what comptime means

### 3. Use SMT to simulate comptime loops

Rejected.

SMT is the fallback for proof obligations, not the engine for pure deterministic
compile-time execution.

## Status

Proposed.

The next implementation target after approving this design should be:

- loop-heavy pure comptime execution in user-facing examples
- then recursive pure comptime execution

---
slug: comptime-in-ora
title: "Comptime in Ora: Compile-Time Work, Runtime Simplicity"
authors: [axe]
tags: [compiler, optimization, comptime, verification]
---

Smart contracts live in a harsher environment than normal software: execution is expensive, everything is observable, and complexity becomes attack surface. "Good compilation" isn't just about speed—it's about predictability. Smaller runtime code. Fewer branches. Fewer guards. Less that can go wrong.

Ora's comptime pipeline exists to enforce that. When an expression is provably constant, the compiler evaluates it during compilation, replaces it with a literal, and lets the rest of the pipeline operate on something simpler than what the developer wrote.

That sounds obvious. In practice, it's one of the places where many smart-contract toolchains accumulate accidental complexity.

<!-- truncate -->

## The Design Smell Ora Avoids: Purity and Constness Get Fuzzy

A common failure mode in languages and compilers is to treat "constant evaluation" as a side effect of optimization rather than as a first-class concept.

You end up with an official notion of "purity" (no effects), but a bunch of unofficial levels of "constant-ish" behavior. Constant evaluation becomes inconsistent across contexts. The same expression might fold in one place and not in another—not because it isn't constant, but because it fell off the happy path of the optimizer.

For smart contracts, this isn't a small annoyance:

- it makes gas costs less deterministic,
- it encourages writing code "for the compiler,"
- and it inflates runtime logic with guards that could have been proven unnecessary.

Ora refuses to inherit that model. If something is constant, we want it resolved early and reliably—before it ever becomes runtime control flow.

## Where Comptime Sits in Ora's Pipeline

Comptime runs after type resolution and before MLIR lowering.

That ordering is intentional:

1. Type resolution annotates the AST with resolved types.
2. A comptime folding pass walks the typed AST, evaluates constant expressions, and rewrites them into literal nodes.
3. Guard elision is recomputed after folding, so refinements that are now trivially satisfied by literals lose their runtime checks.
4. MLIR lowering sees plain literals and emits constants, so downstream IR passes start from a smaller, cleaner program.

Doing this before lowering matters. Once constants become literals early, everything downstream becomes simpler: emitted IR, auditability, and verification all benefit.

## What Counts as "Provably Constant" in Practice

Ora keeps the rule conservative: if we can't prove it, we don't speculate.

The evaluator covers the kinds of expressions that actually show up in contract code and refinements:

- integer arithmetic and bitwise work,
- comparisons and boolean logic,
- casts when the operand is constant,
- switch expressions when the condition is constant,
- array literals with constant indexing,
- struct field reads on constant struct literals,
- enum literals and enum-pattern switches.

The exact list matters less than the posture: constant evaluation is deterministic and explicit in the pipeline, not a "maybe the optimizer will do it" outcome.

## Refinements Become Cheap When Constants Fold

Refinements are where comptime stops being "just an optimization" and starts being a language feature you can use aggressively without paying runtime tax.

If a value is constant and the compiler can validate the refinement against a literal, there is no reason to emit a runtime guard. The guard would be dead code.

```ora
const x: MinValue<u256, 100> = 200;
const y: MinValue<u256, 50>  = 100;
const prod = x * y; // inferred MinValue<u256, 5000>
```

`x` and `y` are constants, so `x * y` folds into a literal during comptime. After folding, the compiler checks the refinement constraints against the computed literal. If they hold, the runtime guard disappears entirely.

The result is the "boring code" you want in contracts: constants and a return. No branch. No guard. No runtime assertion.

If a refinement can't be proven—because a value depends on calldata, storage, or any non-constant path—the guard stays. Correctness beats cleverness.

## Constant Structure and Control Flow Collapses Too

The same effect shows up in common patterns like constant dispatch:

```ora
const mode = 1;

const fee = switch (mode) {
    0 => 5,
    1 => 10,
    else => 20,
};
```

When the discriminator is constant, the switch collapses to a literal. The runtime never sees a branch.

Likewise for table-driven constants:

```ora
const table = [100, 200, 300];
const v = table[1]; // folds to 200
```

This matters because layout math and dispatch tables are everywhere in on-chain systems. If the logic never needed runtime inputs, it shouldn't survive to runtime.

## Comptime Helps Verification: Let the Solver Do Real Work

Ora integrates SMT-based verification (Z3). A strong comptime phase isn't only a gas win—it's a verification win.

Without comptime, it's easy to accidentally push "mechanical simplification" into the solver: computing constant subexpressions, propagating literals, discharging obvious bounds, and generally doing work that the compiler could have done deterministically.

With comptime folding and guard elision:

- trivial refinement checks on literals get resolved before IR is even emitted,
- constant branches collapse,
- and the SMT engine doesn't waste time proving what is effectively arithmetic bookkeeping.

That keeps the solver focused on what it's actually for: meaningful properties—relationships across branches, invariants over state transitions, conservation-style constraints, and correctness conditions that genuinely require symbolic reasoning.

The solver shouldn't be compensating for compiler laziness. Comptime makes sure it doesn't have to.

## About `comptime { ... }` Blocks

Ora doesn't require a `comptime` keyword everywhere. The default is: write normal Ora code, and the compiler folds what it can prove.

We do currently have an explicit `comptime { ... }` block as an escape hatch, but that surface area is still being reviewed with users. If it turns out to be redundant—or if it encourages writing "two languages" inside Ora—we'll remove it. The goal is not more syntax; the goal is a cleaner language where the compiler does the obvious thing by default.

## Closing

Comptime is how Ora turns "static guarantees" into something you can feel on-chain: fewer runtime instructions, fewer branches, fewer guards, and a tighter audit story.

More importantly, it lets refinements and SMT verification work the way they should: refinements become practical because they're often free, and the solver stays focused on proving real correctness—rather than cleaning up what the compiler could have simplified upfront.

---

*For more on how refinements work with comptime, see [Refinement Types: Making Smart Contracts Safer by Default](/blog/refinement-types-in-ora).*

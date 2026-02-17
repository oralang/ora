---
slug: ora-asuka-a-new-take-on-smart-contracts
title: "Ora (Asuka): A New Take on Smart Contracts"
authors: [axe]
tags: [ora, asuka, evm, verification, language]
---

Ora is a smart contract language and compiler for the EVM with two design pillars: **comptime over runtime**—decide as much as possible at compile time, so runtime is the fallback—and **formal verification and the solver in the developer workflow**, the way Foundry put serious testing into the Solidity workflow. FV and Z3 aren’t a separate research step; they’re in the loop: specs next to code, SMT reports in the build artifacts, counterexamples when a proof fails. Here’s what that looks like.

<!-- truncate -->

---

## Ora in 30 seconds

1. **Comptime > runtime.** Constant folding, refinement discharge, and SMT proofs run before code is final. If we can prove it or fold it at compile time, we do; only then do we emit runtime checks. Less in the bytecode, fewer failure modes, more that’s auditable in the compiler output.

2. **FV and the solver in the workflow.** Specs (`requires` / `ensures` / `invariant`) live next to the code. The solver (Z3) runs as part of the pipeline. You get counterexamples and SMT reports in your artifacts, same as tests and coverage. The goal is to do for formal verification what Foundry did for Solidity testing: make it part of how you build, not a separate tool.

3. **Explicit semantics.** Regions (`storage` / `memory` / `calldata` / `transient`) and an inspectable pipeline (Ora → Ora MLIR → SIR → EVM) so “what the compiler did” is visible. No hidden behavior.

We use a Solidity-familiar surface (contracts, state, mappings, events, errors) so you don’t relearn the basics—but the design is comptime-first and FV-in-workflow, not “Solidity with extra features.”

---

## One small example, most of the ideas

This minimal token shows the core ideas in one place: refinements, typed errors, specs, and explicit state.

```ora
contract MiniToken {
    storage balances: map<address, u256>;

    log Transfer(indexed from: address, indexed to: address, amount: u256);
    error InsufficientBalance(required: u256, available: u256);

    pub fn init(owner: NonZeroAddress, supply: NonZero<u256>) {
        balances[owner] = supply;
    }

    pub fn transfer(to: NonZeroAddress, amount: NonZero<u256>)
        -> !bool | InsufficientBalance
        requires balances[std.msg.sender()] >= amount
        ensures balances[std.msg.sender()] == old(balances[std.msg.sender()]) - amount
        ensures balances[to] == old(balances[to]) + amount
    {
        let from = std.msg.sender();
        let available = balances[from];
        if (available < amount) return error.InsufficientBalance(amount, available);

        balances[from] = available - amount;
        balances[to]   = balances[to] + amount;
        log Transfer(from, to, amount);
        return true;
    }
}
```

- **Refinements** like `NonZeroAddress` and `NonZero<u256>` encode invariants in the type system; the compiler and verifier use them to drop redundant checks or catch bugs early.
- **Errors are values**: `return error.InsufficientBalance(...)` is explicit and reflected in the return type `!bool | InsufficientBalance`.
- **Specs sit next to the code**: `requires` / `ensures` drive verification; when a proof succeeds, the corresponding runtime guard can be removed.

---

## Familiar surface, different design

We use a Solidity-familiar surface (contracts, state, mappings, events, errors) so you don’t relearn the basics. The *design* is comptime-first and FV-in-workflow; the *syntax* maps over.

| Solidity | Ora |
|----------|-----|
| `contract C { ... }` | `contract C { ... }` |
| State vars | `storage x: T;` (region explicit) |
| `mapping(K => V)` | `storage balances: map<K, V>;` |
| `constructor()` | `pub fn init(...)` |
| `function f() public` | `pub fn f()` |
| `event E` / `emit E` | `log E(...);` (declare + emit) |
| `error E` / `revert E()` | `error E;` / `return error.E;` |
| `require(cond)` | `if (!cond) return error.X;` or `requires` |
| `msg.sender` | `std.msg.sender()` |

Same concepts; the pipeline is built for comptime resolution and solver-backed verification in the normal build.

---

## What that means in practice

- **Comptime first** — Constant folding, refinement validation, and SMT run during compilation. Proven properties → no runtime guard; unproven → guard stays, you get a counterexample. The solver is in the pipeline, not a separate run.
- **FV in the workflow** — `requires` / `ensures` / `invariant` sit next to the code. `ora emit --emit-smt-report` puts verification results in your artifacts. Counterexamples show up when a proof fails. Same idea as “run tests with the build,” but for the solver.
- **Refinement types** — `NonZeroAddress`, `MinValue<u256, 1>`, `InRange`, `BasisPoints`. The type system carries invariants; the compiler and Z3 use them to eliminate guards or report counterexamples.
- **Checked arithmetic by default** — Overflow/underflow and division by zero revert. Wrapping is explicit (`+%`, `-%`, or `@addWithOverflow` etc.) when you want it.
- **Explicit regions and pipeline** — `storage` / `memory` / `calldata` / `transient`; Ora → Ora MLIR → SIR → EVM. You can see what the compiler did and why a proof failed.

---

## What Ora is not

- **Not a Solidity clone** — We share the mental model and the EVM target; we don’t replicate every quirk or feature.
- **Not “auto-proves everything”** — Complex control flow and state need invariants and specs; the verifier uses them.
- **Not hiding the EVM** — We keep behavior transparent and mappable to the chain.
- **Not stable yet** — Asuka is pre-release; we’re still hardening the surface and tooling. Breaking changes are possible.

---

## Try it

- **Prerequisites:** Zig 0.15.x, CMake, Git, Z3, MLIR  
- **Build:** `git clone https://github.com/oralang/Ora.git && cd Ora && ./setup.sh && zig build`  
- **Run:** `./zig-out/bin/ora ora-example/counter.ora`  
- **Full feature reference:** [Ora Asuka base document](/docs/ora_asuka_base_document) (or the repo `docs/ora_asuka_base_document.md`)

For a deeper, section-by-section reference (types, regions, verification, bitfields, errors, locks, comptime, tooling, ABI, compiler), use the [Asuka base document](/docs/ora_asuka_base_document). For examples, browse `ora-example/` in the repo—counter, ERC20, DeFi-style pool, refinements, SMT, and more.

---

*Ora is in pre-release on the Asuka track. We’re focused on correctness of the end-to-end pipeline and verification behavior. If you care about explicit semantics and verification-aware design on the EVM, we’d love your feedback and contributions.*

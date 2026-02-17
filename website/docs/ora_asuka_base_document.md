---
sidebar_position: 2.5
title: "Ora Asuka — Feature reference"
description: "Community-facing, end-to-end reference for Ora's feature set (Asuka pre-release)."
---

# Ora: A New Take on Smart Contracts

**Comptime over runtime. FV and the solver in the developer workflow.**

Ora is a smart contract language and compiler for the EVM with two design pillars: **comptime over runtime**—decide as much as possible at compile time (constant folding, refinement discharge, SMT proofs), so runtime is the fallback—and **formal verification and the solver in the normal dev workflow**, the way Foundry put serious testing into the Solidity workflow. FV and Z3 aren’t a separate research step; they’re in the loop: specs next to code, SMT reports in artifacts, counterexamples when a proof fails.

This is the community-facing, end-to-end reference for Ora’s feature set (Asuka pre-release). It’s meant to be readable top-to-bottom and precise enough for technical evaluation.

---

## Ora in 30 seconds

1. **Comptime > runtime.** Constant folding, refinement discharge, and SMT proofs run during compilation. If we can prove it or fold it at compile time, we do; only then do we emit runtime checks. Less in the bytecode, more that’s auditable in the compiler output.

2. **FV and the solver in the workflow.** Specs (`requires` / `ensures` / `invariant`) live next to the code. The solver (Z3) runs as part of the pipeline. You get counterexamples and SMT reports in your artifacts—same idea as “run tests with the build,” but for formal verification.

3. **Explicit semantics.** Regions (`storage` / `memory` / `calldata` / `transient`) and an inspectable pipeline (Ora → Ora MLIR → SIR → EVM). No hidden behavior.

We use a Solidity-familiar surface (contracts, state, mappings, events, errors) so you don’t relearn the basics—but the design is comptime-first and FV-in-workflow, not “Solidity with extra features.”

---

## 1. Hello Ora (small example, big ideas)

This tiny contract demonstrates: Solidity-familiar structure, **refinement types**, typed error unions, specs (`requires`/`ensures`), logs, and explicit state transitions.

```ora
contract MiniToken {
    // --- State
    storage balances: map<address, u256>;

    // --- Logs & errors
    log Transfer(indexed from: address, indexed to: address, amount: u256);

    error InsufficientBalance(required: u256, available: u256);

    // --- Constructor / init
    pub fn init(owner: NonZeroAddress, supply: NonZero<u256>) {
        // Refinements are enforced at the boundary when needed.
        // Inside the function, owner/supply are already known-valid.
        balances[owner] = supply;
    }

    // --- Public entrypoint
    pub fn transfer(to: NonZeroAddress, amount: NonZero<u256>)
        -> !bool | InsufficientBalance
        requires balances[std.msg.sender()] >= amount
        ensures balances[std.msg.sender()] == old(balances[std.msg.sender()]) - amount
        ensures balances[to] == old(balances[to]) + amount
    {
        let from = std.msg.sender();

        let available = balances[from];
        if (available < amount) return error.InsufficientBalance(amount, available);

        // Checked arithmetic by default:
        balances[from] = available - amount;
        balances[to]   = balances[to] + amount;

        log Transfer(from, to, amount);
        return true;
    }
}
```

Key takeaways:
- **Refinements are real types**: `NonZeroAddress`, `NonZero<u256>` communicate invariants directly in the signature.
- **Errors are values**: `return error.X` is explicit and type-checked by the return type.
- **Specs are adjacent to code**: they drive verification and can remove runtime guards when proven.

---

## 2. Why Ora exists

Smart contracts are high-stakes: a single bug can lock or lose funds. Most languages and toolchains treat correctness as an afterthought (runtime checks, ad-hoc tests, manual audits). Ora is built so that:

- **Comptime does as much as possible.** Constant folding, refinement discharge, and SMT proofs run during compilation. Proven → no runtime guard; unproven → guard stays, counterexample in artifacts. The goal is “decide at compile time, fall back to runtime only when needed.”
- **FV and the solver are in the workflow.** Foundry made testing and fuzzing part of how Solidity devs work. Ora aims to do the same for formal verification: specs next to code, solver in the pipeline, reports and counterexamples in the build. Not a separate research tool.
- **Semantics are explicit.** Storage, memory, calldata, and transient storage are first-class regions; the pipeline is inspectable. What you write maps clearly to what runs.

---

## 3. Familiar surface: Solidity → Ora

Ora uses a Solidity-familiar surface so you don’t relearn the basics. The *design* is comptime-first and FV-in-workflow; the *syntax* maps over.

| Solidity | Ora | Notes |
|----------|-----|------|
| `contract C { ... }` | `contract C { ... }` | Same concept. |
| State vars | `storage x: T;` | Region is explicit. |
| `mapping(K => V)` | `storage balances: map<K, V>;` | Same idea: key–value storage. |
| `constructor()` | `pub fn init(...)` | One-time initialization. |
| `function f() public` | `pub fn f()` | `pub` = ABI entrypoint. |
| `function g() internal` | `fn g()` | No `pub` = internal. |
| `event E(...)` / `emit E(...)` | `log E(...);` / `log E(...);` | `log` declares or emits based on context. |
| `error E()` / `revert E()` | `error E;` / `return error.E;` | Errors are values. |
| `require(cond)` | `if (!cond) return error.X;` or `requires` | No hidden `require` path. |
| `msg.sender` | `std.msg.sender()` | Explicit namespace. |

---

## 4. Type system

### 4.1 Primitives

| Category | Types | Notes |
|----------|------|------|
| Unsigned ints | `u8` … `u256` | EVM word is `u256` |
| Signed ints | `i8` … `i256` | Two’s complement |
| Other | `bool`, `address`, `string`, `bytes` | First-class |

Signed and unsigned are not implicitly compatible. Conversions are explicit.

### 4.2 Composite types
- Structs
- Enums (with explicit backing type), e.g. `enum Status : u8 { Pending, Active, Closed }`
- Tuples
- Anonymous structs: `.{ a: T, b: U }`
- Maps: `map<K, V>`

**Storage layout policy (Asuka):**
- Field order is preserved unless you opt in to explicit packing/layout features.
- Packing within a slot is supported where safe and visible via tooling.
- For maximum density with explicit layout, use bitfields (§10).

### 4.3 Refinement types

Refinements attach constraints to base types. The compiler and verifier use them to:
- eliminate redundant guards when provable, and
- catch bugs earlier when constraints can’t be satisfied.

Common refinements:
- `NonZero<T>` (value ≠ 0)
- `NonZeroAddress` (address ≠ 0)
- `MinValue<T, N>`, `MaxValue<T, N>`
- `InRange<T, Lo, Hi>`
- `BasisPoints<u256>` (0..10_000)

Refined values are subtypes of their base type: you can pass a `NonZeroAddress` where an `address` is expected.

---

## 5. Memory regions

Ora makes **where data lives** explicit. Every variable is in one of these regions:

| Region | Lifetime | Use case |
|--------|----------|----------|
| `storage` | Contract lifetime | Persistent state |
| `memory` | Call / function | Temporaries, locals |
| `calldata` | Call | Read-only inputs, no copy |
| `transient` | Transaction | Scratch space (EIP-1153 TSTORE/TLOAD), cleared after tx |

Example:

```ora
storage var count: u256;
memory var tmp: u256 = count; // explicit copy into memory
```

Invalid transitions (e.g. writing to calldata) are compile-time errors.

---

## 6. Safety and arithmetic

### 6.1 Checked arithmetic (default)

`+`, `-`, `*`, `/`, `%`, `<<`, `>>` are **checked**. Overflow/underflow or division by zero triggers a deterministic **panic** (revert with an Ora panic payload). This is a hard failure (not a typed error union).

### 6.2 Wrapping operators

When you **want** modular arithmetic (hashing, crypto, intentional wrap), use the `%`-suffixed operators. They produce the wrapping (mod 2^N) result for the type and **do not insert overflow traps**.

| Operator | Meaning |
|----------|---------|
| `+%` | Wrapping add |
| `-%` | Wrapping subtract |
| `*%` | Wrapping multiply |
| `<<%` | Wrapping shift left |
| `>>%` | Wrapping shift right |

Example: `let b = a +% 1;` — if `a` is `u8` and `a == 255`, then `b == 0`.

**Shift rule (Asuka):**
- Checked shifts (`<<`, `>>`) panic if `shift >= bitwidth(T)`.
- Wrapping shifts (`<<%`, `>>%`) mask the shift amount: `shift % bitwidth(T)`.

### 6.3 Overflow-reporting builtins

When you need the **result and an overflow flag**, use `@…WithOverflow`. Each returns:

`.{ value: T, overflow: bool }`

Where `value` is the wrapping result and `overflow` indicates whether overflow occurred under checked semantics.

| Builtin | Returns |
|---------|---------|
| `@addWithOverflow(a, b)` | `.{ value: a +% b, overflow }` |
| `@subWithOverflow(a, b)` | `.{ value: a -% b, overflow }` |
| `@mulWithOverflow(a, b)` | `.{ value: a *% b, overflow }` |
| `@negWithOverflow(a)` | `.{ value: -%a, overflow }` |
| `@shlWithOverflow(a, b)` | `.{ value: a <<% b, overflow }` |
| `@shrWithOverflow(a, b)` | `.{ value: a >>% b, overflow }` |

Example:

```ora
let res = @addWithOverflow(a, b);
if (res.overflow) return 0;
return res.value;
```

---

## 7. Formal verification

Verification is a first-class feature: specs live next to the code.

### 7.1 Specification clauses
- `requires` — preconditions (caller guarantees)
- `ensures` — postconditions (function guarantees)
- `invariant` — contract or loop invariants
- `assume` — verification-only assumption (no runtime code)
- `assert` — checked at runtime and in verification

Example:

```ora
pub fn transfer(to: address, amount: u256) -> bool
    requires amount > 0
    requires balances[std.msg.sender()] >= amount
    ensures balances[std.msg.sender()] == old(balances[std.msg.sender()]) - amount
    ensures balances[to] == old(balances[to]) + amount
{
    // ...
}
```

### 7.2 SMT (Z3) behavior
- If a proof succeeds, the compiler can **remove** the corresponding runtime guard.
- If a proof fails, the guard stays and you get a **counterexample** to debug.

### 7.3 Verification flow
Source → Typed AST → Ora MLIR (with verification metadata) → SMT encoding → Z3 → prove or counterexample → guard placement/refinement discharge.

---

## 8. Control flow

### 8.1 Conditionals and loops
Standard `if`/`else` and loops. Loops can carry:
- invariants (`invariant`)
- termination measures (`decreases`) for verification

### 8.2 Switch
Switch is both statement and expression:
- patterns: literals, enums, ranges (`0...59`), comma-separated cases
- `else` as default (must be last)
- exhaustiveness checks for enums (if no `else`)
- overlap checks for integer ranges

```ora
var g: u8 = switch (score) {
    0...59   => 0,
    60...69  => 1,
    70...79  => 2,
    80...89  => 3,
    90...100 => 4,
    else     => 5,
};
```

### 8.3 Labels and targeted control
- labels name blocks and switches
- `break :label` / `continue :label`
- labeled switch can support `continue :label value;` for state-machine style code without nesting

---

## 9. Error handling

### 9.1 Error declarations
Errors are declared at contract scope, with optional payloads:

```ora
error InvalidAmount;
error InsufficientBalance(required: u256, available: u256);
```

### 9.2 Error unions
Return types can be error unions: `!T | E1 | E2`.

```ora
fn withdraw(to: NonZeroAddress, amount: u256)
    -> !u256 | InvalidAmount | InsufficientBalance
{
    if (amount == 0) return error.InvalidAmount;
    if (balances[to] < amount) return error.InsufficientBalance(amount, balances[to]);
    // ...
    return balances[to];
}
```

### 9.3 Try and catch
- `try expr` unwraps success or propagates the error
- `try { ... } catch (e) { ... }` handles errors locally

Errors are values: they’re part of the ABI/tooling so UIs can decode them reliably.

---

## 10. Bitfields

Bitfields pack multiple small values into a single EVM word. Layout is compiler-checked; reads/writes lower to mask/shift (and sign-extension for signed fields).

### 10.1 Layout
- explicit: `@at(offset, width)` or `@bits(start..end)` per field
- auto-packed: omit `@at`; compiler packs sequentially from bit 0

### 10.2 Storage batching
A storage bitfield uses one slot. Consecutive writes to the same bitfield are batched into:
- one SLOAD
- N updates
- one SSTORE

### 10.3 Utilities
- `@bitCast` — bitfield ↔ raw integer (no masking)
- `.zero()` — all-zero value
- `.sanitize()` — clear bits not owned by any field

---

## 11. Logs and events

Declare:

```ora
log Transfer(indexed from: address, indexed to: address, amount: u256);
```

Emit:

```ora
log Transfer(from, to, amount);
```

Indexed fields are marked for efficient filtering.

---

## 12. Lock and unlock (transaction-scoped guards)

Ora supports path-scoped lock/unlock for reentrancy-sensitive flows using a transaction-scoped lockset.

- `@lock(expr)` — lock the slot identified by `expr` for the transaction.
- `@unlock(expr)` — unlock it.
- The compiler emits guards so writes to lock-participating paths revert if they target a currently locked slot.

This gives an auditable pattern for “lock this slot during this critical section”.

---

## 13. Comptime and generics

### 13.1 Comptime-first
As much as possible is decided at compile time: constant folding, type resolution, refinement validation. SMT handles what can’t be decided statically; runtime checks are the last resort.

### 13.2 Constant folding
Pure, side-effect-free expressions (including calls with all-constant arguments) can be folded during type resolution. Comptime evaluation is deterministic and hermetic.

### 13.3 Generics (comptime type parameters)
- functions and structs can be generic: `comptime T: type`
- call site provides type args: `max(u256, a, b)`
- monomorphization: each instantiation becomes a concrete symbol

---

## 14. Tooling and developer experience

### 14.1 CLI
- build: `ora build <file.ora>` or `ora <file.ora>`
- emit: `ora emit [options] <file.ora>`
- format: `ora fmt <file.ora>` (`--check`, `--diff`)

### 14.2 Artifacts
Typical outputs:
- ABI: `artifacts/<name>/abi/<name>.abi.json` (+ extras)
- bytecode: `artifacts/<name>/bin/<name>.hex`
- SIR text: `artifacts/<name>/sir/<name>.sir`
- verification: `artifacts/<name>/verify/<name>.smt.report.md` (+ json)

### 14.3 Demo checklist (5 minutes)
1. Compile: `ora ora-example/counter.ora`
2. Emit IR: `ora emit --emit-mlir=both --emit-sir-text ora-example/counter.ora`
3. Emit verification report: `ora emit --emit-smt-report ora-example/erc20.ora`
4. Inspect artifacts under `artifacts/<contract>/`

---

## 15. ABI and interop

Ora ABI v0.1 follows a **manifest + wire profiles** model:
- manifest defines types/callables/errors/events once (stable identities)
- wire profiles describe encoding for calls/returns/errors/events (e.g. EVM ABI)

The compiler emits Solidity-compatible JSON ABI plus an extras file for richer tooling.

---

## 16. Compiler architecture

- Frontend (Zig): lexer, parser, typed AST, semantics (regions/effects/refinements/locks), Ora MLIR emission
- Ora MLIR: contract-level IR with verification metadata and explicit regions
- Lowering: Ora MLIR → SIR MLIR / SIR text
- Backend: SIR → EVM bytecode
- Verification: Z3 pass over Ora MLIR, counterexamples + guard placement feedback

The pipeline is designed for visibility: you can inspect IR and reports at each stage.

---

## 17. What Ora is not

- Not a Solidity clone.
- Not “auto-proves everything” without invariants/specs for complex control flow.
- Not hiding EVM behavior behind abstraction layers.
- Not promising full stability yet during Asuka: breaking changes are expected while the surface and tooling harden.

---

## 18. Status and roadmap (Asuka)

Working today:
- end-to-end compilation (Ora → SIR → EVM)
- regions, refinements, specs, Z3 verification, counterexamples
- checked arithmetic + wrapping + overflow-reporting builtins
- error unions, logs, switch/ranges, bitfields, formatter, emit/debug artifacts

In progress:
- stronger diagnostics, deeper verification (loops/quantifiers/interprocedural), parity coverage, tooling stabilization

---

## 19. Get started

- prerequisites: Zig 0.15.x, CMake, Git, Z3, MLIR
- build: `git clone https://github.com/oralang/Ora.git && cd Ora && ./setup.sh && zig build`
- test: `zig build test`
- run: `./zig-out/bin/ora ora-example/smoke.ora`

---

## Summary table

| Area | Features |
|------|----------|
| Types | `u8–u256`, `i8–i256`, `bool`, `address`, `string`, `bytes`; structs, enums, tuples, anonymous structs, maps |
| Refinements | `NonZero`, `InRange`, `MinValue`, `MaxValue`, `BasisPoints`, `NonZeroAddress`; SMT + guard discharge |
| Regions | `storage`, `memory`, `calldata`, `transient` |
| Safety | checked arithmetic default; wrapping operators; overflow-reporting builtins |
| Verification | `requires`, `ensures`, `invariant`, `assume`, `assert`; Z3 SMT; counterexamples; guard removal |
| Control flow | if/else, loops with invariants/termination, switch expression with ranges/exhaustiveness, labels |
| Bitfields | explicit layout; storage batching; utilities |
| Errors | declarations, error unions, `try`, `catch` |
| Logs | declare + emit, indexed fields |
| Locks | transaction-scoped lock/unlock with guards |
| Comptime | constant folding; generic functions/structs; monomorphization |
| Tooling | build/emit/fmt; MLIR/SIR/CFG/SMT reports; reproducible artifacts |
| ABI | manifest + wire profiles; Solidity-compatible output + extras |
| Compiler | Zig frontend → Ora MLIR → SIR → EVM; Z3 verification |

---
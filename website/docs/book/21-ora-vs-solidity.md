---
title: "Appendix: Ora vs Solidity"
description: How Ora eliminates entire classes of smart contract vulnerabilities and helps auditors reason about code correctness.
sidebar_position: 21
---

# Ora vs Solidity

Solidity leaves security to the developer. Ora makes the compiler responsible.

Every Ora feature exists because of a specific failure mode in smart contracts. This page traces each design decision back to the root cause it addresses — not just *what* Ora does differently, but *why* the difference is necessary.

> **Note:** This book was written with AI assistance and reviewed by the Ora team. If you find inaccuracies, please open an issue or PR — the compiler is the source of truth, not this text.

## Defense in Depth

| Solidity failure mode | Root cause | Ora design response |
|---|---|---|
| Silent error swallowing | Errors are control flow, not values | Error unions: errors are types and values in the same execution path |
| Forgotten input checks | Validation is code, not types | Refinement types: constraints are part of the type, checked by SMT |
| Hidden state mutations | No effect tracking | Effect system: per-slot read/write inference from code |
| Invisible data location | Regions implicit for scalars | Located types `τ@ρ`: persistence is a type property |
| Unverifiable invariants | No specification language | `requires`/`ensures`/`invariant`: SMT-checked proof obligations |
| Reentrancy | No built-in CEI enforcement | Effects + locks: compiler rejects write-after-call on same slot |
| Storage collision | Inheritance determines slot layout | No inheritance — traits add behavior without affecting storage |
| Hidden external call semantics | CALL vs STATICCALL invisible at call site | Extern traits: `call fn`/`staticcall fn` explicit, mandatory gas, error unions |
| Fragile code reuse | Inheritance conflates behavior, storage, and interface | Traits: composition via `impl`, no slot changes, no virtual dispatch |
| Precision loss | No fixed-point semantics | `Scaled<T, D>` tracks decimal places in the type system |
| Runtime attack surface | Values computed at runtime that could be known statically | Comptime: freeze configuration into bytecode at compile time |

Every row follows the same pattern: a class of bugs exists because the language lacks a mechanism to prevent it. Ora adds that mechanism. The sections below explain each one.

---

## Error Unions: Errors Are Values, Not Exceptions

**The root cause**: Solidity's `revert`/`require` breaks the execution path. When a function reverts, the entire call frame unwinds. The caller has two choices: let the revert propagate (losing all context), or wrap the call in `try/catch` (bolted-on exception handling that doesn't compose). Exceptions are not values — you can't store an exception in a variable, return it, match on it, or pass it to another function for later inspection. The "error path" has completely different semantics from the "success path."

```solidity
// Solidity: two separate control flow paths
function withdraw(uint256 amount) external {
    require(balances[msg.sender] >= amount, "insufficient");  // reverts — breaks execution
    balances[msg.sender] -= amount;
    (bool ok,) = msg.sender.call{value: amount}("");
    require(ok);  // reverts again — a different kind of failure, same blunt instrument
}
```

**Ora's response**: errors are ordinary values in the return type. The function returns a value on every path — success or failure. There is no separate "error control flow."

```ora
error InsufficientBalance(required: u256, available: u256);
error ExternalCallFailed;

fn withdraw(amount: u256) -> !bool | InsufficientBalance | ExternalCallFailed {
    let sender: address = std.msg.sender();
    let current: u256 = balances[sender];
    if (current < amount) {
        return InsufficientBalance(amount, current);   // returns an error VALUE
    }
    balances[sender] = current - amount;
    let ok: bool = try external<IToken>(token, gas: 100000).transfer(sender, amount);
    return ok;
}
```

**Why this matters:**

- **One control flow graph, not two.** The SMT solver sees error returns and success returns in the same SSA graph. `ensures` clauses can reason about error paths the same way they reason about success paths. In Solidity, the verifier would need to model exception unwinding separately.
- **Errors are structured values.** `InsufficientBalance(amount, current)` carries its parameters. The caller can inspect them, log them, or make decisions based on them. In Solidity, catching a revert gives you raw bytes you have to ABI-decode.
- **No hidden state rollback.** Solidity's `revert` rolls back storage changes for the current call frame but not for cross-contract calls already completed. This asymmetry causes bugs. In Ora, the programmer explicitly decides what state to commit or roll back — the language doesn't silently undo anything.
- **Forced handling.** You cannot compile code that ignores an error union. `try` unwraps it and propagates on failure. The type system makes "forgot to check the return value" a compile error.

---

## Refinement Types: Constraints Are Types, Not Runtime Checks

**The root cause**: Solidity's `require(amount > 0)` is a runtime assertion. It runs, costs gas, and reverts if false. More importantly, it's a **point check** — it proves the condition holds at one location in the code. It proves nothing about what happens downstream. The next function that receives `amount` has no way to know it was already validated.

```solidity
function deposit(uint256 amount) external {
    require(amount > 0, "zero deposit");        // checked HERE
    _processDeposit(amount);                     // but _processDeposit doesn't know amount > 0
}

function _processDeposit(uint256 amount) internal {
    // Should we check again? Defensive programming says yes — but that's duplicated logic.
    // Skip the check? Then a new caller can pass 0 without protection.
}
```

**Ora's response**: constraints are part of the type, not assertions in the body.

```ora
pub fn deposit(amount: MinValue<u256, 1>) {
    processDeposit(amount);    // amount carries its constraint — no re-check needed
}

fn processDeposit(amount: MinValue<u256, 1>) {
    // The type guarantees amount >= 1. The compiler proved it at the call site.
    balances[std.msg.sender()] += amount;
}
```

**Why this matters:**

- **Constraints flow with values.** `MinValue<u256, 1>` is a subtype of `u256`. Once a value is proven to satisfy the constraint, it carries that proof through the program. No redundant re-checks, no "should I validate again?" decisions.
- **SMT-backed, not runtime-backed.** The compiler must prove, at every call site, that the constraint holds *before the function is called*. If it can prove it statically, no runtime check is emitted at all. If it can't, it emits a guard — but the guard is a fallback, not the primary mechanism.
- **Subtype lattice.** `InRange<u256, 1, 100>` is a subtype of both `MinValue<u256, 1>` and `MaxValue<u256, 100>`. Refined values compose. The compiler can derive downstream properties from upstream constraints without re-checking.

---

## Located Types: Persistence Is a Type Property

**The root cause**: in Solidity, whether a value persists across transactions or dies at function exit is determined by where it's declared — but the type system doesn't track this. A `uint256` in storage and a `uint256` in memory have the same type. The compiler selects SLOAD or MLOAD based on declaration context, but the programmer and the auditor have to mentally track which is which.

This matters because persistence determines attack surface. A value in storage can be read and manipulated by any future transaction. A value in memory is invisible to external callers. Confusing the two is a category of bugs.

**Ora's response**: every value has a **located type** `τ@ρ` — a value type at a region. The region is part of the type.

```ora
storage var balance: u256;      // u256 @ storage — persists across transactions
memory var temp: u256;          // u256 @ memory  — dies at function exit
tstore var lock: u256;          // u256 @ transient — dies at transaction end
```

Four regions, each with different persistence and safety properties:
- **storage** — persistent, can be a reentrancy vector, cross-transaction invariants apply
- **memory** — function-scoped, isolated from external callers, local verification only
- **transient** — transaction-scoped, useful for cross-function coordination within one tx
- **calldata** — read-only input, cannot be modified

The compiler enforces region coercion rules at the type level. You can copy from storage into memory (read), but you can't write to calldata, and you can't move directly between storage and transient. These rules are grounded in Ora's formal type calculus (`Σ; Γ; Λ ⊢ e : τ@ρ ! ϵ`) where every typing judgment produces both a located type and an effect.

The region isn't just a codegen hint — it determines what can go wrong with a value. When the compiler knows `balance: u256 @ storage`, it knows this value is part of the reentrancy surface, part of the persistence model, and subject to cross-transaction invariants. When it knows `temp: u256 @ memory`, it knows this value is isolated and safe.

> **Pre-alpha note:** The current compiler allows implicit coercions across regions (calldata → memory → storage) without requiring a guard (`if`, `requires`, or explicit check). This is intentional for pre-alpha — it lets us compile and test more real contract code while the region enforcement rules are finalized. Production Ora will require explicit checks or coercion annotations at region boundaries where safety properties change.

---

## Effect System: The Compiler Knows What Each Function Touches

**The root cause**: Solidity's `view` and `pure` are opt-in annotations. If a developer forgets them, the function can read or write any storage slot. Worse, `view`/`pure` only distinguish "reads storage" from "writes storage" — they don't say *which* slots. The compiler can't tell you that `deposit` writes `balance` but not `owner`.

This matters because verification needs **frame conditions** — the ability to say "this function didn't change X." Without per-slot effect tracking, the verifier can't prove that calling `deposit` preserves the `owner` field.

**Ora's response**: the compiler infers effects from the code — which slots are read, which are written, and what side effects occur.

```ora
contract Vault {
    storage var balance: u256;
    storage var owner: address;

    // Compiler infers: reads(balance @ storage)
    fn getBalance() -> u256 {
        return balance;
    }

    // Compiler infers: reads(balance @ storage), writes(balance @ storage)
    pub fn deposit(amount: u256) {
        balance += amount;
    }
}
```

The effect system tracks six kinds of side effects: **reads**, **writes**, **external calls**, **logs**, **locks/unlocks**, and **havoc** (verification-specific state invalidation). All inferred, not annotated.

**Why this matters:**

- **Frame conditions for free.** Because the compiler knows `deposit` has effect `writes(balance)`, the SMT solver can assert `owner == old(owner)` without proving it — the effect system guarantees `owner` wasn't touched.
- **CEI enforcement at the slot level.** If a function writes to `balance`, makes an external call, then writes to `balance` again, the compiler rejects it. This isn't a coarse "writes storage" check — it's per-slot:

```ora
pub fn unsafe() {
    balance = 100;                                                    // write(balance)
    let ok = try external<IERC20>(token, gas: 50000).transfer(to, amount);  // external call
    balance = 200;                                                    // write(balance) AGAIN
    // ↑ Compile error: 'balance' written before AND after external call
}
```

- **Composable analysis.** When function A calls function B, the compiler merges their effect sets. If B writes a slot that A also writes around an external call, the reentrancy check catches it — even across function boundaries.

---

## Formal Verification: Specifications Are Checked, Not Commented

**The root cause**: Solidity has NatSpec comments (`/// @notice`, `/// @param`) but they're documentation, not contracts. There's no language-level way to say "this function must preserve X" and have the compiler check it. `assert()` and `require()` are runtime checks — they cost gas, they can be wrong, and they only check one execution path.

**Ora's response**: `requires`, `ensures`, `invariant`, `ghost`, and `old()` are first-class language constructs, checked at compile time by an SMT solver.

```ora
contract Vault {
    storage var totalDeposits: u256 = 0;
    storage var balances: map<address, u256>;
    ghost storage spec_sum: u256 = 0;

    invariant deposits_nonnegative: totalDeposits >= 0;

    pub fn deposit(amount: MinValue<u256, 1>)
        requires totalDeposits <= std.constants.U256_MAX - amount
        ensures totalDeposits == old(totalDeposits) + amount
        ensures spec_sum == old(spec_sum) + amount
    {
        let sender: NonZeroAddress = std.msg.sender();
        balances[sender] += amount;
        totalDeposits += amount;
        spec_sum += amount;
    }
}
```

**Why this matters:**

- **`ensures` is a proof obligation, not a comment.** The SMT solver checks that every execution path through `deposit` satisfies `totalDeposits == old(totalDeposits) + amount`. If any path violates it, the compiler rejects the code.
- **`invariant` holds across all public functions.** If you add a new function that breaks `deposits_nonnegative`, the compiler catches it — even if you never wrote a test for that function.
- **`ghost storage` gives the verifier bookkeeping state at zero cost.** Ghost variables compile away to nothing. They let you express relationships like "the sum of all balances equals the total supply" that would be impossible to check with runtime assertions.
- **`old(x)` connects pre-state to post-state.** This is the fundamental operation for reasoning about state transitions. Solidity has no equivalent — you'd have to manually cache the old value in a local variable and compare.

---

## Reentrancy: Effects + Locks, Not Patterns

**The root cause**: Solidity doesn't track which storage slots a function reads or writes. The checks-effects-interactions pattern is a *convention*, not an enforcement mechanism. OpenZeppelin's `nonReentrant` modifier is a runtime lock bolted on through inheritance — it requires importing a library, inheriting from it, and applying the modifier. Three things to remember, none enforced by the language.

**Ora's response**: built-in `@lock`/`@unlock` combined with the effect system.

```ora
storage var reentrancyGuard: u256 = 0;

pub fn withdraw(amount: u256) -> !bool | InsufficientBalance {
    let sender: NonZeroAddress = std.msg.sender();
    @lock(reentrancyGuard);
    let current: u256 = balances[sender];
    if (current < amount) {
        @unlock(reentrancyGuard);
        return InsufficientBalance(amount, current);
    }
    balances[sender] = current - amount;
    @unlock(reentrancyGuard);
    return true;
}
```

The effect system verifies lock/unlock discipline across all code paths — including early returns through error unions. No library, no inheritance, no modifier to forget.

---

## Comptime: Less Runtime Means Less Attack Surface

**The root cause**: Solidity contracts compute values at runtime that could be known at compile time — selector hashes, scale factors, configuration constants. Every runtime computation is attack surface an adversary can potentially influence through transaction ordering, oracle manipulation, or governance.

**Ora's response**: `comptime` moves computation to the compiler. The result is embedded as a literal in the bytecode.

```ora
const SCALE: u256 = comptime { return 10 ** 18; };   // computed at compile time
const FEE_BPS: u256 = 250;                           // 2.5%, compile-time constant
const MAX_UINT: u256 = std.constants.U256_MAX;        // compile-time from stdlib
```

- **Configuration values are frozen in bytecode.** A `const` can't be manipulated by transaction ordering, oracle manipulation, or governance attacks.
- **Computed constants don't have intermediate states.** The compiler produces the final value — no runtime overflow risk, no intermediate states.
- **Generic code is monomorphized.** `fn identity(comptime T: type, value: T) -> T` generates specialized bytecode per type at compile time. No runtime dispatch.

Anything that *can* be known at compile time *should* be. Every value moved from runtime to comptime is one fewer thing an attacker can influence and one fewer thing an auditor needs to trace.

---

## Traits: Composition Without Inheritance

**The root cause**: Solidity uses inheritance for code reuse (`contract Vault is Ownable, ReentrancyGuard, ERC20`). Inheritance conflates three things that should be separate: shared behavior, storage layout, and interface conformance. Adding a base contract changes your slot layout. Overriding a virtual method changes behavior at a distance. The diamond problem creates ambiguity about which implementation runs.

For verification, inheritance is especially damaging. When a function can be overridden, the verifier can't prove properties about it — a derived contract can change the behavior. When storage layout depends on the inheritance chain, invariants about storage slots become fragile.

**Ora's response**: traits define behavior. `impl` blocks attach it to types. Storage layout is never affected.

```ora
trait Ownable {
    fn owner(self) -> address;
    fn isOwner(self, addr: address) -> bool;
}

impl Ownable for Vault {
    fn owner(self) -> address {
        return owner_addr;
    }

    fn isOwner(self, addr: address) -> bool {
        return addr == owner_addr;
    }
}
```

**Why traits instead of inheritance:**

- **Storage is decoupled from behavior.** `impl Ownable for Vault` adds methods. It does not add storage slots, shift existing slots, or create layout dependencies. The storage layout is exactly what's declared in the contract body.
- **No virtual dispatch.** Every call resolves to a concrete function at compile time. The verifier can prove properties about the implementation because it can't be overridden. An auditor reading `vault.isOwner(addr)` knows exactly which function runs.
- **Nominal conformance for verification.** `impl Ownable for Vault` is an explicit declaration that Vault satisfies the Ownable interface. The compiler uses this to generate proof obligations — it checks that the implementation actually conforms. Accidental structural matches don't count.
- **No diamond problem.** A type can implement multiple traits. If two traits define a method with the same name, the compiler rejects it — no ambiguity, no silent resolution rules.

---

## Extern Traits: Cross-Contract Calls as Typed Boundaries

**The root cause**: Solidity's `IERC20(addr).transfer(to, amount)` hides critical information. You can't tell from the call site whether it's a CALL or STATICCALL. You can't see the gas budget. You can't tell whether the call can fail silently. The interface declaration doesn't say which methods mutate state and which are read-only. And if the external call reverts, you're back in exception-handling territory — `try/catch` with raw bytes.

This matters because cross-contract calls are the primary attack surface in DeFi. Reentrancy, oracle manipulation, flash loan attacks — they all flow through external calls. A language that hides the mechanics of external calls is hiding the attack surface.

**Ora's response**: extern traits are a separate concept from local traits, because cross-contract calls have fundamentally different semantics from local method calls.

```ora
extern trait IERC20 {
    staticcall fn balanceOf(self, owner: address) -> u256;
    call fn transfer(self, to: address, amount: u256) -> bool
        errors(InsufficientBalance);
}
```

**Why extern traits are separate from traits:**

- **`call fn` vs `staticcall fn` is part of the interface.** The trait declaration says which methods mutate state (CALL) and which are read-only (STATICCALL). This is visible at the declaration site, not buried in an implementation.
- **Mandatory gas budget.** Every external call requires `gas:` — no implicit gas forwarding:

```ora
let bal = try external<IERC20>(token, gas: 30000).balanceOf(user);
```

- **Error unions, not exceptions.** External calls return error unions. `ExternalCallFailed` is always in the union. The `errors(...)` clause on the method adds specific revert reasons. The caller handles failure as a value, not an exception.
- **Effect system integration.** The compiler knows `call fn` is an external call with reentrancy potential. It tracks this in the effect set and enforces lock discipline around it. `staticcall fn` has no reentrancy risk — the effect system treats it differently.
- **You cannot `impl` an extern trait.** You don't have the external contract's code. Extern traits are interface declarations for contracts you call, not contracts you write. This distinction prevents accidentally mixing local and cross-contract semantics.

The separation is deliberate: local traits are about composition within your contract. Extern traits are about typed boundaries between contracts. These are different trust domains with different safety properties, and the language should reflect that.

---

## How Ora Helps Auditors

In Solidity, an auditor reads the function body to find `require` statements, trace inheritance chains, and check modifiers. In Ora, the function signature is the complete security contract:

```ora
pub fn transfer(
    to: NonZeroAddress,                              // can't be zero address
    amount: InRange<u256, 1, 1000000>                // bounded amount
) -> !bool | InsufficientBalance | ExternalCallFailed  // exactly these failure modes
    requires balances[std.msg.sender()] >= amount       // caller must prove this
    guard amount > 0                                     // function checks this at runtime
    ensures balances[to] == old(balances[to]) + amount   // postcondition — SMT-proven
```

An auditor reads this and knows:
- **Input constraints** — from refinement types (`NonZeroAddress`, `InRange`), base types, and region constraints (parameters come from calldata — read-only, cannot alias storage)
- **Failure modes** — from the error union: exactly `InsufficientBalance` or `ExternalCallFailed`, nothing else
- **Caller obligations** — from `requires`: what the caller must prove statically
- **Runtime checks** — from `guard`: what the function enforces at runtime (reverts if false)
- **Postconditions** — from `ensures`: what the SMT solver proved about every execution path
- **Effect surface** — the compiler inferred which slots are read/written, which the auditor can inspect

All of this before reading a single line of body. No inheritance, no virtual methods, no modifiers. The function body is the complete behavior.

---

## Syntax Quick Reference

For the full syntax of each feature, see the corresponding book chapter.

| Concept | Solidity | Ora | Chapter |
|---|---|---|---|
| State variables | `uint256 public x;` | `storage var x: u256;` | [Ch. 2](./02-types-and-variables.md) |
| Mappings | `mapping(address => uint256)` | `map<address, u256>` | [Ch. 2](./02-types-and-variables.md) |
| Functions | `function f() returns (uint256)` | `fn f() -> u256` | [Ch. 3](./03-functions-and-operators.md) |
| Visibility | `public`/`external`/`internal`/`private` | `pub fn` or `fn` | [Ch. 3](./03-functions-and-operators.md) |
| Constructor | `constructor() { }` | `pub fn init() { }` | [Ch. 1](./01-hello-ora.md) |
| Control flow | `if`/`else`, `for`, `while` | Same + `switch` expressions, `for (i in 0..n)` | [Ch. 4](./04-control-flow.md) |
| Structs | `struct S { uint x; }` | `struct S { x: u256, }` | [Ch. 5](./05-composite-types.md) |
| Error handling | `try/catch`, `require`, `revert` | Error unions: `!T \| Err`, `try` | [Ch. 6](./06-error-unions.md) |
| Memory regions | `memory`/`storage` on ref types | `storage var`, `memory var`, `tstore var` | [Ch. 7](./07-memory-regions.md) |
| Refinement types | `require(x > 0)` | `MinValue<u256, 1>`, `InRange<u256, 0, 100>` | [Ch. 8](./08-refinement-types.md) |
| Events | `event E(); emit E();` | `log E(); log E();` | [Ch. 9](./09-logs-and-events.md) |
| Specifications | NatSpec comments (not checked) | `requires`, `guard`, `ensures`, `invariant` | [Ch. 10](./10-specification-clauses.md) |
| Ghost state | Nothing | `ghost storage`, `old()` | [Ch. 11](./11-ghost-state.md) |
| Reentrancy | OpenZeppelin `nonReentrant` | `@lock`/`@unlock` | [Ch. 12](./12-locks.md) |
| Interfaces | `interface I { }` | `trait T { }`, `impl T for S { }` | [Ch. 14](./14-traits.md) |
| Generics | None | `fn f(comptime T: type, x: T) -> T` | [Ch. 15](./15-generics.md) |
| Comptime | `constant` (limited) | `comptime { }` blocks and functions | [Ch. 16](./16-comptime.md) |
| External calls | `IERC20(addr).f()` | `external<IERC20>(addr, gas: N).f()` | [Ch. 17](./17-extern-traits.md) |
| Overflow | `unchecked { }` block | `+%`, `-%`, `*%` per operator | [Ch. 3](./03-functions-and-operators.md) |

> This page covers language-level security properties. It does not cover deployment security, protocol-level design, or off-chain coordination. Ora makes the code verifiable — the protocol design is still your responsibility.

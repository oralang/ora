# Ora Trait System – Design Document (v0.1)

Design direction for Ora's trait (static interface) system, defining syntax, semantics, and interaction with the three pillars: regions, comptime, and formal verification.
**Working, evolving design document**, not a final specification.

---

# 1. Motivation

Ora currently supports generics via comptime type parameters with monomorphization. However, generic parameters are **unconstrained** — the compiler cannot type-check a generic function body until it is instantiated. This creates four concrete problems:

1. **Error locality**: Errors surface at the call site, not the definition site. Library authors can't guarantee their generic code is correct in isolation.
2. **Contract interop**: There is no typed way to describe the interface of an external contract (e.g., "this address implements ERC-20").
3. **Internal abstraction**: No mechanism to express "these types share a common set of operations" without duplicating code.
4. **Verified upgradability**: No way to prove that a new contract implementation preserves the behavioral guarantees of the one it replaces.

Traits solve all four by giving types a way to **declare and verify behavioral conformance**.

### Why Regions Are Central to This Design

A recurring theme throughout this document is the relationship between types, traits, and regions. This is not incidental — it is foundational to Ora's design philosophy.

In smart contracts, the difference between "data in memory" and "data in storage" is not a representation detail — it is a **semantic distinction with real-world consequences**. Writing to storage is a state transition: it is permanent, visible to other contracts, costs significant gas, and can be re-entered mid-operation. Operating on memory is scratch work: local, cheap, invisible to the outside world.

Solidity hides this distinction. An assignment to a state variable looks identical to an assignment to a local variable. Auditors must *know* which is which from context. This is a root cause of reentrancy bugs, state inconsistency, and verification gaps.

Ora makes this distinction machine-checkable through **located types** (`τ@ρ`). Every value carries its region, and the compiler uses that information for effect tracking, verification condition generation, and codegen. The trait system must preserve — never erase — this information as it flows through trait resolution and monomorphization.

This is a deliberate contrast with Move's "resource" model. Move abstracts storage into capabilities (`key`, `store`, `copy`, `drop`), which works well for MoveVM but would be a leaky abstraction on EVM. Ora's position is: **don't abstract away the machine — make the machine checkable.** `τ@ρ` tells you exactly what EVM operations will execute. The compiler uses that for safety; the auditor uses it for understanding.

---

# 2. Design Principles

These principles are non-negotiable and guide every decision below:

### 2.1. What You See Is What You Get
No hidden inheritance chains, no silently inherited default methods, no implicit conversions through trait coercions. An auditor reading an `impl` block sees exactly what each method does.

### 2.2. Compile-Time Only, Zero Runtime Cost
Traits are erased during compilation. All dispatch is resolved via monomorphization. There are no vtables, no dictionary passing, no indirect calls generated from trait usage. On EVM, this means zero gas overhead from the abstraction.

### 2.3. Traits Are Specifications, Not Implementations
A trait defines *what* a type must do, not *how*. Traits do not carry default method bodies. If shared code is needed, write a function and call it explicitly from each `impl`.

### 2.4. Region-Aware, Not Region-Abstracted
Traits are region-agnostic *in syntax* by default — most trait declarations do not mention regions. But the compiler is **never** region-agnostic. When a trait method is called on `self@storage`, the compiler knows it is a storage operation, tracks the effects accordingly (`Writes({field@storage})`), and generates the appropriate verification conditions. Region information flows through trait resolution and monomorphization — it is never erased. Traits that are *about* region transitions (e.g., committing memory state to storage) explicitly annotate regions in their signatures.

### 2.5. Verification-First
Ghost specifications on traits are first-class. When a type implements a trait with ghost specs, the compiler generates proof obligations. This is the foundation for verified upgradability.

### 2.6. Concrete Over Abstract
Ora models EVM realities directly rather than introducing abstract capability systems. Regions are `storage`, `memory`, `calldata`, `transient` — not abstract labels like "resource" or "asset." This ensures auditors can trace from Ora source to EVM behavior without navigating an abstraction gap.

---

# 3. Core Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Conformance model | **Nominal** (`impl Trait for Type`) | Verification needs explicit intent, not accidental structural match |
| Dispatch strategy | **Monomorphization** | Zero gas overhead; EVM has no vtable mechanism; optimizer sees through all abstractions |
| `self` semantics | **Plain `self`**, no `&`/`&mut` | Regions + effects already capture mutability and location; no need for a third system |
| Mutability on self | **Inferred from effect system** | The compiler tracks `Pure \| Writes(S)` — don't ask the programmer to repeat it |
| Default implementations | **No** | WYSIWYG — auditors see exactly what each impl does |
| Trait inheritance | **Deferred** to future version | Useful for ERC hierarchies but adds complexity; document intent now, implement later |
| Associated types | **Yes** (Phase 4) | Reduces generic parameter noise; pairs well with monomorphization |
| Ghost specs on traits | **Yes**, first-class | Enables verified conformance and upgradability |
| Region default in traits | **Region-agnostic syntax**, region-aware semantics | Most traits don't mention regions; compiler always tracks them internally |
| Region-explicit traits | **Opt-in** via `Self@region` annotations | For traits whose purpose involves region transitions (e.g., `Committable`) |
| Region erasure | **Never** — regions flow through trait resolution | Effects, verification, and codegen all depend on knowing the concrete region |
| Storage-native types | **Built-in compiler rule**, not trait-encoded | `mapping` is always `@storage`; small fixed set, pragmatic enforcement |
| State model | **Concrete EVM regions**, not abstract resources | Direct mapping to EVM opcodes; no abstraction gap for auditors |
| One impl per type | **Yes** — no region-specific impls | `impl Trait for T` works across all valid regions; compiler handles codegen differences |

---

# 4. Syntax

## 4.1. Trait Declaration

```ora
trait ERC20 {
    fn totalSupply(self) -> u256;
    fn balanceOf(self, owner: address) -> u256;
    fn transfer(self, to: address, amount: u256) -> bool;
    fn approve(self, spender: address, amount: u256) -> bool;
    fn transferFrom(self, from: address, to: address, amount: u256) -> bool;
}
```

**`self`** — the first parameter, indicating this is a method (called on an instance). Its type is implicitly `Self`, the implementing type. No `&` or `*` annotations — mutability is inferred from the effect system, location is handled by the region system.

**Associated functions** (no `self`) are also allowed:

```ora
trait ERC20Metadata {
    fn name() -> string;       // associated function — no self
    fn symbol() -> string;     // associated function — no self
    fn decimals() -> u8;       // associated function — no self
}
```

## 4.2. Implementation Block

```ora
impl ERC20 for Token {
    fn totalSupply(self) -> u256 {
        return self.totalSupply;
    }

    fn balanceOf(self, owner: address) -> u256 {
        return self.balances[owner];
    }

    fn transfer(self, to: address, amount: u256) -> bool {
        self.balances[msg.sender] -= amount;
        self.balances[to] += amount;
        return true;
    }

    fn approve(self, spender: address, amount: u256) -> bool {
        self.allowances[msg.sender][spender] = amount;
        return true;
    }

    fn transferFrom(self, from: address, to: address, amount: u256) -> bool {
        self.allowances[from][msg.sender] -= amount;
        self.balances[from] -= amount;
        self.balances[to] += amount;
        return true;
    }
}
```

The compiler verifies:
- Every method in the trait is present in the `impl` block
- Signatures match exactly (parameter types, return types)
- No extra methods are defined in the `impl` block (methods that aren't part of the trait belong on the type directly, not in the `impl`)

## 4.3. Trait Bounds on Generics

```ora
fn max(comptime T: type, a: T, b: T) -> T
    where T: Comparable
{
    if (a.greaterThan(b)) return a;
    return b;
}
```

The `where` clause constrains `T` to types that implement `Comparable`. The function body is type-checked against the trait's interface — errors appear at the definition site, not at each call site.

Multiple bounds:

```ora
fn sortAndPrint(comptime T: type, items: []T)
    where T: Comparable, T: Printable
{
    // ...
}
```

## 4.4. External Contract Calls

```ora
let token = external<ERC20>(tokenAddress);
let bal = token.balanceOf(msg.sender);
// Compiles to: STATICCALL with ABI-encoded balanceOf(address) selector
```

`external<Trait>(address)` generates a typed proxy that:
- Produces the correct ABI-encoded selectors for each trait method
- Returns a struct whose methods compile to `CALL`/`STATICCALL` instructions
- Carries the trait's type information for the compiler, but has no runtime representation beyond the address

---

# 5. Interaction with Ora's Three Pillars

## 5.1. Regions (`τ@ρ`)

### The Foundational Distinction

In smart contract execution, not all data is equal:

```ora
self.balance -= amount;   // @storage — STATE TRANSITION
                          // Permanent. Visible to other contracts. Costs gas.
                          // Can be re-entered mid-operation. This is what
                          // formal verification must reason about.

let temp = a + b;         // @memory — SCRATCH WORK
                          // Local. Disappears after the call. Cheap.
                          // Cannot be re-entered. Safe by construction.
```

Solidity makes these look identical. An auditor must *know from context* which variables are state and which are local. This is a root cause of reentrancy bugs, state inconsistency, and verification gaps.

Ora's located type system (`τ@ρ`) makes this distinction machine-checkable. The compiler tracks regions through every operation and uses them for:
- **Effect computation**: `self.balance -= amount` produces `Writes({balance@storage})`
- **Lockset discipline**: prevents re-entrant writes to the same storage slot
- **Verification conditions**: "prove `balance >= amount` before the SSTORE"
- **Codegen**: selects `SLOAD`/`SSTORE` vs `MLOAD`/`MSTORE` vs `CALLDATALOAD`

### Why Not Abstract Resources (Move's Approach)

Move models state with abstract capabilities:

```move
struct Coin has key, store { value: u64 }
public fun transfer(coin: Coin, recipient: address) {
    move_to(recipient, coin);
}
```

This works well for MoveVM, which was designed around resources. But on EVM:

| | Move resources | Ora `τ@ρ` |
|---|---|---|
| **Abstraction level** | Abstract capabilities (`key`, `store`, `copy`, `drop`) | Concrete EVM regions (`storage`, `memory`, `calldata`, `transient`) |
| **EVM mapping** | Indirect — must bridge Move semantics to EVM opcodes | Direct — each region maps 1:1 to EVM memory areas |
| **Auditability** | Must understand resource algebra | Read the region, know the opcodes |
| **Verification target** | "resource not duplicated/destroyed" | "this SSTORE is safe given the current state" |
| **Gas reasoning** | Abstracted away | Directly visible — storage ops are expensive, memory ops are cheap |

Ora's position: **don't abstract away the machine — make the machine checkable.** Auditors working with EVM need to reason about storage slots, not abstract resources. The type system should help them do that, not add another layer of indirection.

### How Traits Interact with Regions

Traits and regions operate on two different axes:
- **Traits** describe *what a type can do* — behavioral conformance
- **Regions** describe *where data lives* — operational context and implications

These compose but are independent by default. There are three levels of interaction:

#### Level 1: Region-Agnostic Traits (Default)

Most traits don't mention regions. The compiler handles region-appropriate codegen at monomorphization:

```ora
trait Comparable {
    fn greaterThan(self, other: Self) -> bool;
}

impl Comparable for u256 {
    fn greaterThan(self, other: u256) -> bool {
        return self > other;
    }
}
```

When called on `u256@storage`, the compiler inserts `SLOAD` before comparison. When called on `u256@memory`, it inserts `MLOAD`. The trait doesn't need to know — the region system handles it. This is the common case and should require zero region-related annotation.

#### Level 2: Region-Constrained Traits (Explicit)

Some traits are *about* region transitions — their entire purpose involves moving data between regions:

```ora
trait Committable {
    fn commit(self: Self@memory) -> Self@storage;
}
```

This trait says: "I take scratch work and make it permanent." The region annotations are the semantic content — without them, `commit` is meaningless. The compiler verifies that implementations respect these region constraints and generates appropriate verification conditions:

```ora
impl Committable for TokenState {
    fn commit(self: Self@memory) -> Self@storage {
        // Compiler knows: this method performs memory→storage transition
        // Effect: Writes({...@storage})
        // Verification: prove self is valid before SSTORE
        self.balance = self.balance;  // commits to storage
        return self;
    }

    ghost {
        // The committed value equals the in-memory value
        ensures forall(x: Self@memory) {
            let stored = x.commit();
            stored == x  // value equivalence across regions
        };
    }
}
```

#### Level 3: Mixed Traits (Region-Constrained Methods on Otherwise Agnostic Traits)

Some traits have methods with different region requirements:

```ora
trait Collection {
    fn length(self) -> u256;                    // any region — just reads a field
    fn get(self, index: u256) -> Self.Item;     // any region — reads data

    fn append(self: Self@mutable, item: Self.Item);  // mutable regions only
    // @mutable = storage | memory | transient (not calldata)
}
```

The `@mutable` annotation is a **region kind** — a set of regions grouped by capability. This is deferred to a future version (see Section 8), but the design should not preclude it.

### Region Flow Through Trait Resolution

The critical invariant: **region information is never erased during trait resolution.** Here is the full flow:

```
1. Call site: token.transfer(alice, 100)
   - token: Token@storage (known from declaration context)

2. Trait resolution: Token implements ERC20, method transfer matches
   - Binds Self = Token, self = Token@storage

3. Monomorphization: generate transfer() with self: Token@storage
   - self.balances[msg.sender] -= amount → effect: Writes({balances@storage})
   - self.balances[to] += amount → effect: Writes({balances@storage})

4. Verification: generate proof obligations from concrete region
   - Prove: balances[msg.sender] >= amount (before SSTORE)
   - Prove: no overflow on balances[to] + amount (before SSTORE)
   - Prove: lockset allows writes to balances@storage (reentrancy safety)

5. Codegen: emit SLOAD/SSTORE sequences for storage operations
```

At no point does the compiler lose track of the fact that `self` is `@storage`. The trait provides the behavioral shape; the region provides the operational meaning. Both are needed for correct effects, verification, and codegen.

### Storage-Native Types

Some types inherently belong to a specific region. A `mapping(K => V)` only exists in storage — its keys are hashed into storage slot positions, which is meaningless in memory. This is enforced as a built-in compiler rule, not through the trait system:

```ora
storage mapping(address => u256) balances;  // valid
memory mapping(address => u256) temp;       // compile error: mapping is storage-only
```

Storage-native types are a small, fixed set. A built-in rule is simpler and clearer than encoding this through trait constraints.

## 5.2. Comptime

Trait methods can be marked `comptime` — resolved during compilation, fully eliminated from output:

```ora
trait ABIEncodable {
    comptime fn selector() -> u32;    // computed at compile time, zero gas
    fn encode(self) -> bytes;         // runtime method, generates EVM code
}
```

Comptime trait methods are particularly useful for:
- ABI selector computation
- Storage slot calculation
- Type metadata queries
- Compile-time validation logic

This is a natural extension of Ora's Zig-inherited comptime philosophy: anything that *can* be computed at compile time *should* be.

## 5.3. Formal Verification

Traits can carry **ghost specifications** — behavioral contracts that are verified by the SMT solver but produce no runtime code:

```ora
trait TotalOrder {
    fn le(self, other: Self) -> bool;

    ghost {
        // Reflexivity
        ensures forall(a: Self) { a.le(a) == true };
        // Antisymmetry
        ensures forall(a: Self, b: Self) {
            a.le(b) and b.le(a) implies a == b
        };
        // Transitivity
        ensures forall(a: Self, b: Self, c: Self) {
            a.le(b) and b.le(c) implies a.le(c)
        };
    }
}
```

When a type writes `impl TotalOrder for MyType`, the compiler generates **proof obligations** from the ghost block. The SMT solver must discharge these obligations, or the implementation is rejected.

This transforms traits from syntactic contracts ("you have these methods") into **semantic contracts** ("you have these methods, and they satisfy these mathematical properties").

---

# 6. Verified Upgradability (Future)

The combination of traits + ghost specs enables **compile-time verified upgradability** — a feature no other smart contract language provides.

### The Problem

Upgradeable contracts (proxy/delegate patterns) are a major source of vulnerabilities. When deploying a new implementation behind a proxy, developers currently have no way to prove:
- ABI compatibility (same function selectors and types)
- Storage layout compatibility (new layout is a superset of old)
- Behavioral compatibility (invariants are preserved)

### The Solution

```ora
trait Vault {
    fn deposit(self, amount: u256);
    fn withdraw(self, amount: u256);
    fn balanceOf(self, user: address) -> u256;

    ghost {
        // Conservation: deposits always equal withdrawals + current balance
        invariant forall(user: address) {
            self.totalDeposits[user] == self.totalWithdrawals[user] + self.balanceOf(user)
        };
    }
}

// V1 — verified against the ghost spec
contract VaultV1 impl Vault {
    // ...
}

// V2 — ALSO verified against the same ghost spec
contract VaultV2 impl Vault {
    // ...
}
```

The compiler can verify at each level:

| Level | What's checked | How |
|---|---|---|
| **ABI compatibility** | Same function selectors, same parameter/return types | Structural comparison of trait signatures |
| **Storage compatibility** | V2's storage layout is a superset of V1's | Storage layout analysis (slot positions, types, packing) |
| **Behavioral compatibility** | V2 satisfies the same ghost invariants as V1 | SMT verification of proof obligations from the trait's ghost block |

This is the long-term vision. Implementation details (proxy pattern integration, storage layout verification, migration helpers) are deferred to a dedicated design document.

---

# 7. Implementation Roadmap

| Phase | Scope | Unlocks |
|---|---|---|
| **Phase 1** | Lexer keywords (`trait`, `impl`), AST nodes for trait declarations and impl blocks, basic parsing | Syntax validation, IDE support |
| **Phase 2** | Sema: verify impl blocks provide all required methods with matching signatures | Basic conformance checking |
| **Phase 3** | `where` clauses, trait bounds on generic parameters, type-check generic bodies against bounds | Bounded generics — the main ergonomic unlock |
| **Phase 4** | Associated types, `comptime` trait methods | Full expressiveness, compile-time trait computation |
| **Phase 5** | Ghost specs on traits, proof obligation generation from impl blocks | Verified conformance, foundation for verified upgradability |

---

# 8. Deferred Decisions

These are explicitly out of scope for v0.1 but documented for future consideration:

### 8.1. Trait Inheritance (Supertraits)
```ora
trait ERC721: ERC165 {
    // Implementing ERC721 requires also implementing ERC165
}
```
Useful for the ERC standard hierarchy. Deferred because it adds complexity to conformance checking and the ERC hierarchy can be handled with multiple separate `impl` blocks for now.

### 8.2. Negative Trait Bounds
```ora
fn foo(comptime T: type) where T: !Copy { ... }
```
"T does *not* implement Copy." Occasionally useful, significantly complicates the type system. Deferred.

### 8.3. Trait Objects / Dynamic Dispatch
Ora traits are monomorphized only. There is no `dyn Trait` equivalent. On EVM, cross-contract calls already provide dynamic dispatch via `external<Trait>`. Intra-contract dynamic dispatch has no clear use case and would add gas overhead.

### 8.4. Orphan Rules / Coherence
Rust enforces that you can only impl a trait for a type if you own either the trait or the type. This prevents conflicting impls across crates. Ora's compilation model (single-contract compilation units) may make this unnecessary, but it needs analysis if/when Ora supports a module/package system.

---

# 9. Open Questions

1. **Region kinds in trait signatures**: Should Ora support region kinds like `@mutable` (storage | memory | transient) as constraints on trait methods? Or are concrete region annotations (`@storage`, `@memory`) and the region-agnostic default sufficient for all practical cases?

2. **Refinement types in trait signatures**: Can a trait method require a refined parameter?
   ```ora
   trait SafeTransfer {
       fn transfer(self, to: NonZeroAddress, amount: NonZero<u256>) -> bool;
   }
   ```
   This seems natural but needs analysis of how refinement subtyping interacts with trait conformance checking.

3. **`self` for structs vs contracts**: Should `self` behave identically for struct methods and contract methods? Contracts have implicit access to storage and `msg`; structs don't. Is `self` sufficient to distinguish these, or does the context make it clear?

4. **Event emission in trait methods**: Should traits be able to specify that an implementation must emit certain events? Events are important for off-chain indexing and are part of a contract's observable interface.
   ```ora
   trait ERC20 {
       event Transfer(from: address, to: address, amount: u256);
       fn transfer(self, to: address, amount: u256) -> bool;
       // Should the trait require that transfer emits Transfer?
   }
   ```

5. **Error types in trait methods**: Should traits specify the error types a method can return?
   ```ora
   trait ERC20 {
       fn transfer(self, to: address, amount: u256) -> !bool | InsufficientBalance | InvalidRecipient;
   }
   ```

---

# 10. Comparison with Other Languages

| Feature | Ora (proposed) | Rust | Zig | Solidity | Move |
|---|---|---|---|---|---|
| Conformance | Nominal | Nominal | Structural (comptime) | Nominal (`is`) | Nominal (abilities) |
| Dispatch | Monomorphization | Mono + vtable (`dyn`) | Mono (comptime) | Inheritance + virtual | Static |
| `self` | Plain, no annotations | `self`/`&self`/`&mut self` | Explicit pointer type | Implicit `this` | `&self`/`&mut self` |
| Default impls | No | Yes | N/A | Yes (virtual) | No |
| Associated types | Yes (planned) | Yes | N/A | No | Yes (phantom types) |
| Verification specs | Ghost blocks (planned) | No | No | No | No |
| Runtime cost | Zero | Zero (mono) / indirect (dyn) | Zero | Inheritance overhead | Zero |
| State model | Located types (`τ@ρ`) — concrete EVM regions | Ownership + lifetimes | Explicit pointers | Implicit (state vs local) | Abstract resources (`key`/`store`/`copy`/`drop`) |
| Region/state in traits | Opt-in region annotations, never erased | Lifetime params on traits | N/A | Not tracked | Ability constraints on type params |

### Ora vs Move: A Closer Look

Move and Ora share the insight that state mutations are special and must be tracked by the type system. They differ fundamentally in *how*:

- **Move** introduces an abstract resource algebra. Values have capabilities (`key` = can exist in global storage, `store` = can be stored inside other resources, `copy` = can be duplicated, `drop` = can be discarded). This is elegant for MoveVM, which was co-designed with these semantics. But it does not map cleanly to EVM, where "global storage" is a flat key-value store of 256-bit slots, not a typed resource pool.

- **Ora** models EVM directly. `@storage` means SLOAD/SSTORE. `@memory` means MLOAD/MSTORE. `@calldata` means CALLDATALOAD. There is no abstraction gap between the type system and the target machine. Auditors can trace from source to opcodes without understanding an intermediate capability algebra.

The practical consequence: when Ora's verification system proves "this storage write is safe," the proof obligation directly references the EVM operation. When Move proves "this resource is not duplicated," the auditor must then separately reason about how that maps to the execution environment. Ora eliminates that gap.

---

*This document is v0.1 — a starting point for team discussion, not a final specification. Feedback on any section is welcome.*

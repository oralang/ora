---
title: Traits and Impl
description: Nominal interface conformance with bounded generics, comptime methods, ghost specifications, and static dispatch.
---

# Traits and Impl

Traits define behavioral interfaces that types can implement. Unlike Solidity's inheritance, Ora traits use **nominal conformance** — a type implements a trait only through an explicit `impl` block, not by accidentally having the right method names. This is essential for formal verification: the compiler generates proof obligations from trait ghost specs, and those obligations must be intentional.

All trait dispatch is **static** via monomorphization. No vtables, no dynamic dispatch, no runtime overhead.

## Trait Declaration

A trait declares method signatures that implementing types must provide:

```ora
trait ERC20 {
    fn totalSupply(self) -> u256;
    fn balanceOf(self, owner: address) -> u256;
    fn transfer(self, to: address, amount: u256) -> bool;
}
```

`self` is the receiver — its type is implicitly the implementing type. Methods without `self` are **associated functions** (called on the type, not an instance):

```ora
trait Metadata {
    fn name() -> string;
    fn decimals() -> u8;
}
```

## Implementing a Trait

```ora
contract Token {}

impl ERC20 for Token {
    fn totalSupply(self) -> u256 {
        return self.totalSupply;
    }

    fn balanceOf(self, owner: address) -> u256 {
        return self.balances[owner];
    }

    fn transfer(self, to: address, amount: u256) -> bool {
        self.balances[msg.sender()] -= amount;
        self.balances[to] += amount;
        return true;
    }
}
```

The compiler verifies:
- Every method in the trait is present in the `impl` block
- Signatures match exactly (parameter types, return types, `self` presence)
- No extra methods appear in the `impl` that aren't part of the trait

```ora
// Compile error examples:
impl ERC20 for Token {
    fn totalSupply(self) -> bool { ... }  // wrong return type
    // missing balanceOf and transfer     // missing methods
    fn bonus(self) -> u256 { ... }        // not part of ERC20
}
```

## Bounded Generics

`where` clauses constrain generic type parameters to types that implement specific traits:

```ora
trait Comparable {
    fn greaterThan(self, other: Self) -> bool;
}

fn max(comptime T: type, a: T, b: T) -> T
    where T: Comparable
{
    if (a.greaterThan(b)) return a;
    return b;
}
```

**At the call site**, the compiler verifies the concrete type implements the required trait:

```ora
impl Comparable for u256 {
    fn greaterThan(self, other: u256) -> bool {
        return self > other;
    }
}

let result = max(u256, x, y);  // OK — u256 implements Comparable
let result = max(bool, a, b);  // Compile error: bool does not implement Comparable
```

**Inside the body**, calling a method not provided by any trait bound is a definition-site error:

```ora
fn broken(comptime T: type, a: T) -> T
    where T: Comparable
{
    a.serialize();  // Compile error: type parameter 'T' has no trait bound providing method 'serialize'
}
```

Full return-type and argument-type resolution from the trait interface is deferred to monomorphization — the concrete type is substituted and the body is re-checked with actual types. This follows Zig's monomorphization model while adding trait bounds as a guardrail.

**Builtin operators** (`+`, `-`, `*`, `>`, `==`, etc.) on generic type parameters don't require trait bounds. `T + T` is allowed when both sides are the same type parameter — type errors surface at monomorphization when the concrete type is known.

Multiple bounds:

```ora
fn sortAndPrint(comptime T: type, items: []T)
    where T: Comparable, T: Printable
{
    // body can call methods from both traits
}
```

### Why Trait Bounds?

Ora uses Zig-style monomorphization — generic function bodies are re-lowered for each concrete type. Trait bounds add two things on top:

1. **Call-site validation** — the compiler rejects `max(bool, a, b)` immediately if `bool` doesn't implement `Comparable`, rather than producing a confusing error inside the generic body.

2. **Verification intent** — `impl TotalOrder for u256` is a nominal declaration: "u256 satisfies reflexivity, antisymmetry, and transitivity." Ora's formal verification generates proof obligations from trait ghost specs. Structural matching can't anchor these obligations.

## Associated Method Calls

Associated functions (no `self`) can be called on type parameters or concrete types:

```ora
trait Factory {
    fn make() -> bool;
}

impl Factory for Box {
    fn make() -> bool { return true; }
}

// Through a generic:
fn create(comptime T: type) -> bool where T: Factory {
    return T.make();
}

// Direct call:
let ok = Box.make();
```

Both paths lower to the same concrete function call — no indirection.

## Comptime Trait Methods

Trait methods can be marked `comptime` — they execute at compile time and produce no runtime code:

```ora
trait Selector {
    comptime fn selector() -> u256;
}

impl Selector for Token {
    comptime fn selector() -> u256 {
        return 42;
    }
}

pub fn run() -> u256 {
    return comptime { Token.selector(); };  // resolved at compile time
}
```

Comptime trait methods also work with receivers:

```ora
trait Marker {
    comptime fn marked(self) -> u256;
}

impl Marker for Box {
    comptime fn marked(self) -> u256 {
        return self.value + 1;
    }
}

let result = comptime { Box { value: 4 }.marked(); };  // evaluates to 5
```

This is useful for ABI selector computation, storage slot calculation, and compile-time validation.

## Ghost Specifications on Traits

Traits can carry `ghost` blocks — formal specifications that become proof obligations when a type implements the trait:

```ora
trait SafeCounter {
    fn increment(self);
    fn get(self) -> u256;

    ghost {
        ensures self.get() >= 0;
    }
}

contract Counter {}

impl SafeCounter for Counter {
    fn increment(self) { ... }
    fn get(self) -> u256 { return 0; }
}
// The compiler generates Z3 proof obligations from the ghost block.
// The SMT solver verifies Counter's methods satisfy the spec.
```

Ghost specs are the foundation for **verified upgradability** — if both V1 and V2 of a contract implement the same trait with ghost specs, the compiler can verify both satisfy the same behavioral guarantees.

## Method Lookup Precedence

When a type has both inherent members (struct fields, contract fields) and trait methods:

1. **Inherent members always win** — `box.value` resolves to the struct field, never a trait method named `value`
2. If no inherent member matches, trait methods are checked
3. If multiple traits provide the same method name, the compiler reports an ambiguity error

```ora
trait Left { fn mark() -> bool; }
trait Right { fn mark() -> bool; }

impl Left for Box { fn mark() -> bool { return true; } }
impl Right for Box { fn mark() -> bool { return false; } }

Box.mark();  // Compile error: method 'mark' is ambiguous across multiple impls
```

## Coherence

At most one `impl Trait for Type` per compilation unit. Duplicate implementations are rejected:

```ora
impl ERC20 for Token { ... }
impl ERC20 for Token { ... }  // Compile error: duplicate impl
```

## Comparison with Solidity

| | Solidity | Ora |
|---|---|---|
| Interface conformance | Implicit — must have right selectors | Explicit — `impl Trait for Type` |
| Dispatch | Virtual (runtime, vtable) | Static (compile-time, monomorphized) |
| Default implementations | Yes (`virtual` + `override`) | No — WYSIWYG, auditors see every impl |
| `self` | Implicit `this` with `&` semantics | Explicit `self`, region-aware |
| Verification specs | None | Ghost blocks with Z3 proof obligations |
| Runtime cost | Indirect calls, storage for vtable | Zero — direct calls only |
| Error locality | At deployment (if wrong selector) | At compile time (if missing method) |

## Limitations

The following are not yet supported:

- **Associated types** — `trait Collection { type Item; }` is planned but not implemented
- **Supertraits** — `trait ERC721: ERC165` is planned; for now, use separate `impl` blocks
- **Trait objects / dynamic dispatch** — Ora is monomorphization-only by design. There is no `dyn Trait`.
- **Default method implementations** — every `impl` must provide all methods explicitly

## Extern Traits

For calling external contracts (Solidity, Vyper, etc.), see [External Contract Calls](./extern-traits.md).

---
title: "Chapter 14: Traits and Impl"
description: Nominal interfaces with explicit implementation blocks, bounded generics, and verification specs.
sidebar_position: 14
---

# Traits and Impl

Traits define behavioral interfaces. A type implements a trait through an explicit `impl` block — not by accidentally having matching method names. This is nominal conformance, and it's essential for verification: proof obligations from trait specs must be intentional.

All dispatch is static via monomorphization. No vtables, no runtime overhead.

## Declaring a trait

```ora
trait Vault {
    fn deposit(self, amount: u256);
    fn withdraw(self, amount: u256) -> bool;
    fn balanceOf(self, account: address) -> u256;
}
```

`self` is the receiver — its type is the implementing type. Methods without `self` are associated functions:

```ora
trait Metadata {
    fn name() -> string;
    fn version() -> u256;
}
```

## Implementing a trait

Use `impl Trait for Type` to declare conformance:

```ora
struct AccountSnapshot {
    assets: u256,
    liabilities: u256,
}

trait NetValue {
    fn net(self) -> u256;
    fn buffered(self, buffer: u256) -> u256;
}

impl NetValue for AccountSnapshot {
    fn net(self) -> u256 {
        if (self.assets >= self.liabilities) {
            return self.assets - self.liabilities;
        }
        return 0;
    }

    fn buffered(self, buffer: u256) -> u256 {
        return self.net() + buffer;
    }
}
```

Inside an `impl` block, `self` gives access to the target type's fields (`self.assets`) and other trait methods (`self.net()`).

The compiler checks:
- Every trait method is present in the `impl`
- Signatures match exactly (parameter types, return types, `self` presence)
- No extra methods appear that aren't in the trait
- No duplicate `impl` for the same trait/type pair

## Calling trait methods

Trait methods are called on instances with dot syntax:

```ora
fn quote(assets: u256, reserve: u256) -> u256 {
    let snapshot: AccountSnapshot = .{
        .assets = assets,
        .liabilities = reserve,
    };
    return snapshot.buffered(1);
}
```

## Bounded generics

Use `where` clauses to constrain generic type parameters to types that implement a trait:

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

At the call site, the compiler verifies the concrete type implements the required trait. Inside the body, only methods from the trait bounds are available on generic parameters.

Multiple bounds:

```ora
fn sortAndPrint(comptime T: type, items: []T)
    where T: Comparable, T: Printable
{
    // body can call methods from both traits
}
```

## Ghost specs on traits

Traits can carry `ghost` blocks — formal specifications that become proof obligations when a type implements the trait:

```ora
trait SafeCounter {
    fn get(self) -> u256;

    ghost {
        ensures self.get() >= 0;
    }
}
```

When a contract writes `impl SafeCounter for Counter`, the compiler generates Z3 proof obligations from the ghost block and verifies the implementation satisfies them.

Ghost specs are the foundation for verified upgradability — if both V1 and V2 implement the same trait with ghost specs, the compiler verifies both satisfy the same behavioral guarantees.

## Comptime trait methods

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
    return comptime { Token.selector(); };
}
```

Useful for ABI selector computation, storage slot calculation, and compile-time validation.

## Extern traits

For calling external contracts (Solidity, Vyper, etc.), Ora uses `extern trait` — a separate mechanism covered in [Chapter 17: External Contract Calls](./17-extern-traits).

Extern traits cannot be implemented with `impl` blocks and cannot have ghost specs.

## Limitations

The following are not yet supported:

- **Associated types** — `trait Collection { type Item; }` is planned but not implemented
- **Supertraits** — `trait ERC721: ERC165` is planned; for now, use separate `impl` blocks
- **Default method implementations** — every `impl` must provide all methods explicitly
- **Trait objects / dynamic dispatch** — Ora is monomorphization-only by design. There is no `dyn Trait`.

## Further reading

- [Traits and Impl](../traits) — full reference including method lookup precedence and coherence rules

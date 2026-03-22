---
title: "Chapter 14: Traits and Impl"
description: Nominal interfaces with explicit implementation blocks.
sidebar_position: 14
---

# Traits and Impl

> **Status:** Trait declaration and `impl` blocks are parsed and type-checked, but **`self` receiver support is not yet complete** in the current compiler. The examples below show the intended syntax. Extern traits (Chapter 17) are fully functional. See the [traits reference](../traits) for current status.

Traits define behavioral interfaces. A type implements a trait through an explicit `impl` block — not by accidentally having matching method names. This is nominal conformance, and it's essential for verification: proof obligations from trait specs must be intentional.

All dispatch is static via monomorphization. No vtables, no runtime overhead.

## Declaring a trait

The intended syntax for trait declarations:

*Planned syntax — not yet compilable:*

```text
trait Vault {
    fn deposit(self, amount: u256);
    fn withdraw(self, amount: u256) -> bool;
    fn balanceOf(self, account: address) -> u256;
}
```

`self` is the receiver — its type is the implementing type. Methods without `self` are associated functions:

*Planned syntax — not yet compilable:*

```text
trait Metadata {
    fn name() -> string;
    fn version() -> u256;
}
```

## Implementing a trait

*Planned syntax — not yet compilable:*

```text
contract MyVault {}

impl Vault for MyVault {
    fn deposit(self, amount: u256) {
        // implementation
    }

    fn withdraw(self, amount: u256) -> bool {
        // implementation
        return true;
    }

    fn balanceOf(self, account: address) -> u256 {
        return 0;
    }
}
```

The compiler checks:
- Every trait method is present in the `impl`
- Signatures match exactly (parameter types, return types, `self` presence)
- No extra methods appear that aren't in the trait

## What works today

Extern traits (Chapter 17) are fully implemented and use a different pattern — they declare interfaces for *external* contracts and don't require `self`:

```ora
extern trait IERC20 {
    staticcall fn totalSupply(self) -> u256;
    call fn transfer(self, to: address, amount: u256) -> bool;
}
```

For contracts that need to share an interface today, use matching function signatures without a formal trait:

```ora
contract SimpleVault {
    storage var balances: map<address, u256>;

    pub fn deposit(amount: u256) {
        // same signature as the trait would require
    }

    pub fn balanceOf(account: address) -> u256 {
        return balances[account];
    }
}
```

## Design intent

When `self` support is complete, traits will enable:
- Bounded generics (`where T: Vault`) for generic code over any vault implementation
- Ghost specs on traits for verification obligations that all implementations must satisfy
- Comptime trait methods for compile-time reflection

The [traits reference](../traits) documents the full design.

## Further reading

- [Traits and Impl](../traits) — full reference including bounded generics and ghost specs

---
title: "Chapter 15: Generics"
description: Compile-time type parameters with trait bounds.
sidebar_position: 15
---

# Generics

Ora generics use `comptime T: type` — the type parameter is resolved at compile time, and each instantiation generates specialized code (monomorphization).

## Generic structs

```ora
struct Pair(comptime T: type) {
    first: T;
    second: T;
}
```

Inside a function:

```ora
let p: Pair(u256) = Pair(u256) { first: 10, second: 20 };
let sum: u256 = p.first + p.second;
```

Each distinct type creates a separate instantiation: `Pair(u256)` and `Pair(address)` are different types with different generated code.

## Generic functions

Inside a contract:

```ora
fn max(comptime T: type, a: T, b: T) -> T {
    if (a > b) {
        return a;
    }
    return b;
}
```

The compiler generates a specialized version for each `T` used at call sites.

### Type inference at call sites

When calling a generic function, the compiler can infer `T` from the argument types. Inside a function:

```ora
let result: u256 = max(a, b);    // T inferred as u256 from a and b
```

You don't need to write `max(u256, a, b)` — the compiler resolves the type parameter from context.

## Bounded generics

`where` clauses constrain type parameters to types that implement specific traits:

```ora
trait Comparable {
    fn greaterThan(self, other: Self) -> bool;
}

fn max(comptime T: type, a: T, b: T) -> T
    where T: Comparable
{
    if (a.greaterThan(b)) {
        return a;
    }
    return b;
}
```

Without the `where T: Comparable` bound, calling `a.greaterThan(b)` would be a compile error — the compiler doesn't know `T` has that method.

## Multiple bounds

```ora
fn process(comptime T: type, item: T) -> u256
    where T: Comparable, T: Serializable
{
    // T must implement both Comparable and Serializable
}
```

## Why monomorphization

Ora generates specialized code for each type parameter. This means:
- No vtable overhead
- No dynamic dispatch
- The verifier can reason about concrete types in each instantiation
- Generic code is as efficient as hand-written specialized code

The tradeoff is larger bytecode for many instantiations, but in smart contracts, code size is rarely the bottleneck — gas cost is.

## Further reading

- [Generics](../generics) — full reference

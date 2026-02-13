---
sidebar_position: 3
---

# Language Basics

Core language features and syntax in the current implementation.

> Ora is experimental. Current behavior may change without notice.
> behavior and may change without notice.

## Overview

Ora is a contract language with explicit regions, error unions, and
verification-friendly constructs. The compiler focuses on correctness and
transparent semantics rather than implicit magic.

Note: the current compiler requires explicit type annotations for local
variables (`var x: u256 = ...`).

## Contracts

```ora
contract MyContract {
    // Contract contents
}
```

## Variables and regions

```ora
contract Counter {
    storage var count: u256;   // persistent storage
    storage var owner: address;
    storage var active: bool;
}
```

### Immutables and constants

```ora
contract Token {
    immutable name: string;
    storage const MAX_SUPPLY: u256 = 1000000;
}
```

## Types

### Primitives

```ora
var a: u8 = 255;
var b: u256 = 1;
var ok: bool = true;
var who: address = 0x742d35Cc6634C0532925a3b8D0C5e0E0f8d7D2;
var msg: string = "hello";
```

### Composite types

```ora
struct User {
    name: string;
    balance: u256;
}

enum Status : u8 { Pending, Active, Closed }

var balances: map<address, u256>;
```

## Functions

```ora
contract Math {
    pub fn add(a: u256, b: u256) -> u256 {
        return a + b;
    }

    fn internal_calc(x: u256) -> u256 {
        return x * 2;
    }
}
```

### Error unions

Ora uses explicit error unions:

```ora
error InsufficientBalance;
error InvalidAddress;

fn transfer(to: address, amount: u256) -> !u256 | InsufficientBalance | InvalidAddress {
    if (amount == 0) return error.InvalidAddress;
    if (balance < amount) return error.InsufficientBalance;
    balance -= amount;
    return balance;
}
```

## Control flow

```ora
fn classify(x: u32) -> u32 {
    if (x == 0) return 0;
    if (x < 10) return 1;
    return 2;
}
```

Switch works as expression or statement:

```ora
fn grade(score: u32) -> u8 {
    var g: u8 = 0;
    switch (score) {
        0...59   => g = 0;,
        60...69  => g = 1;,
        70...79  => g = 2;,
        80...89  => g = 3;,
        90...100 => g = 4;,
        else     => g = 5;,
    }
    return g;
}
```

## Refinements (frontend)

The type resolver supports refinement types such as:

- `MinValue<T, N>`
- `MaxValue<T, N>`
- `InRange<T, Min, Max>`
- `NonZeroAddress`

These refinements are enforced in the front-end and also surfaced to the
verification pipeline.

## Specification clauses

Ora parses and type-checks specification clauses:

```ora
pub fn transfer(to: address, amount: u256) -> bool
    requires amount > 0
    ensures  amount > 0
{
    // ...
}
```

`assume` is verification-only; `assert` is runtime-visible and verification-
visible.

## Where to go next

- [Examples](./examples)
- [Switch](./switch)
- [Struct Types](./struct-types)
- [Formal Verification](./formal-verification)

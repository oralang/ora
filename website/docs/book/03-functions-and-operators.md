---
title: "Chapter 3: Functions and Operators"
description: Function declarations, visibility, and the complete operator set.
sidebar_position: 3
---

# Functions and Operators

## Function declarations

```ora
contract Math {
    pub fn add(a: u256, b: u256) -> u256 {
        return a + b;
    }

    fn double(x: u256) -> u256 {
        return x * 2;
    }
}
```

- `pub fn` — public, callable from outside the contract (generates an ABI entry).
- `fn` — private, callable only within the contract.
- Parameters are typed: `name: type`.
- Return type follows `->`. Functions without `->` return `void`.

### Constructors

A function named `init` is the contract's constructor — it runs once at deployment:

```ora
comptime const std = @import("std");

contract Token {
    storage var totalSupply: u256 = 0;
    storage var owner: address;

    pub fn init(initialSupply: u256) {
        owner = std.msg.sender();
        totalSupply = initialSupply;
    }
}
```

Constructor parameters receive values from the `init_args` field in `ora.toml` (see Chapter 19). The compiler automatically marks `init` as a constructor in the ABI.

### Multiple statements

```ora
pub fn clamp(value: u256, max: u256) -> u256 {
    if (value > max) {
        return max;
    }
    return value;
}
```

Functions can have multiple `return` statements. Every code path must return the declared type.

## Arithmetic operators

All operator examples in this section are inside a function.

```ora
let sum: u256 = a + b;       // addition
let diff: u256 = a - b;      // subtraction
let prod: u256 = a * b;      // multiplication
let quot: u256 = a / b;      // division (unsigned)
let rem: u256 = a % b;       // remainder (unsigned)
let power: u256 = a ** b;    // exponentiation
```

Division by zero is caught by the verification system. The compiler emits a runtime guard if it can't prove the divisor is non-zero.

## Wrapping arithmetic

Standard arithmetic reverts on overflow. Wrapping operators silently wrap around, matching raw EVM behavior:

```ora
let wrapped_sum: u256 = a +% b;    // wrapping add
let wrapped_diff: u256 = a -% b;   // wrapping subtract
let wrapped_prod: u256 = a *% b;   // wrapping multiply
```

Use wrapping arithmetic when overflow is intentional (e.g., hash computations). The `%` suffix signals "I know this can overflow and I want modular arithmetic."

## Overflow builtins

For checked arithmetic where you need to detect overflow without reverting, use the overflow builtins:

```ora
let result: struct { value: u256, overflow: bool } = @addWithOverflow(a, b);
if (result.overflow) {
    // handle overflow
}
let sum: u256 = result.value;
```

Three builtins are available:
- `@addWithOverflow(a, b)` — checked addition
- `@subWithOverflow(a, b)` — checked subtraction
- `@mulWithOverflow(a, b)` — checked multiplication

Each returns an anonymous struct with `.value` (the wrapped result) and `.overflow` (true if overflow occurred). You can access fields inline:

```ora
if (@addWithOverflow(balance, amount).overflow) {
    return OverflowError;
}
```

## Comparison operators

```ora
let eq: bool = a == b;     // equal
let ne: bool = a != b;     // not equal
let lt: bool = a < b;      // less than
let le: bool = a <= b;     // less than or equal
let gt: bool = a > b;      // greater than
let ge: bool = a >= b;     // greater than or equal
```

## Logical operators

```ora
let and: bool = x && y;    // logical AND (short-circuit)
let or: bool = x || y;     // logical OR (short-circuit)
let not: bool = !x;        // logical NOT
```

## Bitwise operators

```ora
let band: u256 = a & b;     // bitwise AND
let bor: u256 = a | b;      // bitwise OR
let bxor: u256 = a ^ b;     // bitwise XOR
let bnot: u256 = ~a;        // bitwise NOT
let shl: u256 = a << 4;     // shift left
let shr: u256 = a >> 4;     // shift right
```

Wrapping shift variants exist: `<<%` and `>>%`.

## Compound assignment

Every binary operator has a compound assignment form:

```ora
var x: u256 = 10;
x += 5;      // x = x + 5
x -= 3;      // x = x - 3
x *= 2;      // x = x * 2
x /= 4;      // x = x / 4
x %= 3;      // x = x % 3
x &= 0xFF;   // x = x & 0xFF
x |= 0x01;   // x = x | 0x01
x ^= 0xAA;   // x = x ^ 0xAA
x <<= 2;     // x = x << 2
x >>= 1;     // x = x >> 1
```

## Operator precedence

From lowest to highest:

| Precedence | Operators |
|-----------|-----------|
| Lowest | `\|\|` |
| | `&&` |
| | `\|` |
| | `^` |
| | `&` |
| | `==` `!=` |
| | `<` `<=` `>` `>=` |
| | `<<` `>>` |
| | `+` `-` |
| | `*` `/` `%` |
| Highest | `**` (right-associative) |

Use parentheses when precedence is not obvious.

## The vault with functions

Our vault already uses functions and operators. Here's the `deposit` function annotated:

```ora
pub fn deposit(amount: u256) {
    let sender: address = std.msg.sender();   // immutable local binding
    balances[sender] += amount;                // compound assignment on map
    totalDeposits += amount;                   // compound assignment on storage
}
```

The `+=` operator on `balances[sender]` reads the current value from the storage map, adds `amount`, and writes the result back. The compiler tracks this as a storage write effect.

## Further reading

- [Language Basics](../language-basics) — core syntax reference

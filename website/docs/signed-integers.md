---
sidebar_position: 3.6
title: "Signed Integers & Overflow"
description: "Signed integer types, checked arithmetic, wrapping operators, and overflow builtins."
---

# Signed Integers & Overflow

Ora supports both unsigned (`uN`) and signed (`iN`) integer types with a strict overflow model: **checked by default, explicit wrapping on demand**.

## Integer Types

| Type | Bits | Range |
|------|------|-------|
| `u8` – `u256` | 8 – 256 | 0 to 2^N − 1 |
| `i8` – `i256` | 8 – 256 | −2^(N−1) to 2^(N−1) − 1 |
| `bool` | 1 | 0 or 1 |

Signed integers use **two's complement** representation, matching the EVM's `SDIV`, `SMOD`, `SLT`, `SGT`, and `SAR` opcodes.

```ora
let x: i256 = -42;
let y: u256 = 42;
// let z = x + y;  // compile error: cannot mix signed and unsigned
```

### No Implicit Mixing

Signed and unsigned types are **not implicitly compatible**. Adding `i256 + u256` is a compile-time error. Use explicit casts when conversion is intentional.

## Checked Arithmetic (Default)

All standard operators (`+`, `-`, `*`, `/`, `%`, `<<`, `>>`) are **checked** — they revert on overflow or underflow.

```ora
let a: u8 = 255;
// let b = a + 1;  // reverts at runtime (overflow)

let c: i8 = -128;
// let d = c - 1;  // reverts at runtime (underflow)
```

### How it works

For each operation, the compiler inserts an `assert` that fires if overflow is detected:

| Operation | Unsigned check | Signed check |
|-----------|---------------|-------------|
| `a + b` | `result >= a` | sign-bit flip: `((result ^ a) & (result ^ b)) < 0` |
| `a - b` | `a >= b` | sign-bit flip: `((a ^ b) & (result ^ a)) < 0` |
| `a * b` | `b == 0 \|\| result / b == a` | special-case `MIN * -1`, then `sdiv` check |

Division by zero always reverts regardless of signedness.

## Wrapping Operators

When you explicitly want modular arithmetic (e.g., hashing, cryptographic operations), use the `%`-suffixed operators:

| Operator | Meaning |
|----------|---------|
| `+%` | wrapping add |
| `-%` | wrapping subtract |
| `*%` | wrapping multiply |
| `<<%` | wrapping shift left |
| `>>%` | wrapping shift right |

```ora
let a: u8 = 255;
let b = a +% 1;   // b == 0 (wrapped)

let c: i8 = 127;
let d = c +% 1;   // d == -128 (wrapped)
```

Wrapping operators lower directly to EVM `ADD`, `SUB`, `MUL`, etc. — no overflow check, no extra gas.

## Overflow Builtins

For cases where you need both the result **and** the overflow flag, Ora provides reporting builtins:

```ora
let (value, overflowed) = @addWithOverflow(a, b);
if (overflowed) {
    // handle overflow
}
```

Available builtins:

| Builtin | Returns |
|---------|---------|
| `@addWithOverflow(a, b)` | `(a +% b, overflow_flag)` |
| `@subWithOverflow(a, b)` | `(a -% b, overflow_flag)` |
| `@mulWithOverflow(a, b)` | `(a *% b, overflow_flag)` |
| `@negWithOverflow(a)` | `(-%a, overflow_flag)` |

Each returns an anonymous struct `(value: T, overflow: bool)` where `T` matches the operand type.

## Signed Operations Lowering

Signed operations compile to dedicated EVM/SIR opcodes:

| Ora | EVM | SIR |
|-----|-----|-----|
| `i256 / i256` | `SDIV` | `sdiv` |
| `i256 % i256` | `SMOD` | `smod` |
| `a < b` (signed) | `SLT` | `slt` |
| `a > b` (signed) | `SGT` | `sgt` |
| `a >> b` (signed) | `SAR` | `sar` |
| unary `-a` | `SUB(0, a)` | `sub` |

Unsigned operations use `DIV`, `MOD`, `LT`, `GT`, and `SHR` as before.

## Unary Negation

Unary `-` is only valid on signed types:

```ora
let x: i256 = 42;
let neg = -x;      // ok: -42

// let y: u256 = 42;
// let neg2 = -y;   // compile error: cannot negate unsigned type
```

## Design Rationale

1. **Checked by default** prevents silent overflow bugs — the #1 source of smart contract exploits.
2. **Wrapping operators** are opt-in and visually distinct (`+%`), making modular arithmetic a deliberate choice.
3. **No implicit signed/unsigned mixing** eliminates an entire class of subtle bugs where signedness assumptions differ between caller and callee.
4. **Overflow builtins** enable gas-efficient overflow-aware algorithms without try/catch overhead.

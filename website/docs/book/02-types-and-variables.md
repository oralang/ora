---
title: "Chapter 2: Types and Variables"
description: Primitive types, variable declarations, and storage in Ora.
sidebar_position: 2
---

# Types and Variables

Every value in Ora has a type, and every type must be explicitly annotated. Ora does not infer types for local variables.

## Primitive types

All type examples in this section are inside a function unless noted otherwise.

### Unsigned integers

```ora
let a: u8 = 255;
let b: u16 = 65535;
let c: u32 = 4_294_967_295;
let d: u64 = 100;
let e: u128 = 1;
let f: u256 = 0;
```

`u256` is the native EVM word size and the most common type in smart contracts.

### Signed integers

```ora
let x: i8 = -128;
let y: i16 = 32767;
let z: i256 = -1;
```

Signed integers use two's complement representation, matching the EVM's `SDIV`, `SMOD`, and `SLT` opcodes.

### Boolean

```ora
let active: bool = true;
let paused: bool = false;
```

### Address

```ora
let owner: address = 0x742d35Cc6634C0532925a3b8D0C5e0E0f8d7D2aB;
```

`address` is a 160-bit type representing an Ethereum address.

### String and bytes

```ora
let name: string = "Ora Token";
let data: bytes = hex"DEADBEEF";
```

`string` holds UTF-8 text. `bytes` holds arbitrary byte sequences. Both are dynamic-length types backed by storage encoding.

### Void

```ora
pub fn doSomething() {
    // No return value — implicit void return type
}
```

Functions that return nothing have an implicit `void` return type.

## Numeric literals

Ora supports several literal formats:

```ora
let decimal: u256 = 1_000_000;          // underscores for readability
let hex: u256 = 0xFF;                    // hexadecimal
let binary: u256 = 0b1010;              // binary
let big: u256 = 115_792_089_237_316_195_423_570_985_008_687_907_853_269_984_665_640_564_039_457_584_007_913_129_639_935;
```

Underscores in numeric literals are ignored by the compiler — use them freely for readability.

## Variable declarations

Ora has four declaration keywords (`let`/`var` inside functions, `const`/`immutable` at contract level):

```ora
let x: u256 = 10;       // immutable binding — cannot be reassigned
var y: u256 = 20;       // mutable binding — can be reassigned
const Z: u256 = 30;     // compile-time constant
immutable OWNER: address = 0x742d35Cc6634C0532925a3b8D0C5e0E0f8d7D2aB;  // set once at deploy
```

- `let` creates an immutable binding. Once assigned, the value cannot change.
- `var` creates a mutable binding. The value can be reassigned.
- `const` creates a compile-time constant. The value must be known at compilation.
- `immutable` creates a deploy-time constant. Set during construction, read-only after.

## Storage variables

Contract state lives in storage. Storage variables are declared with the `storage` keyword:

```ora
contract Vault {
    storage var totalDeposits: u256 = 0;
    storage var balances: map<address, u256>;
    storage var owner: address;
}
```

- `storage var` — persistent, mutable state. Each read costs ~2,100 gas (SLOAD), each write costs ~5,000–20,000 gas (SSTORE).
- `storage let` or `storage const` — persistent, immutable after initialization.

The `storage` keyword is not optional. Ora never hides where data lives — this is a core design principle.

## The vault begins

Here is the start of our running example — a basic vault contract:

```ora
comptime const std = @import("std");

contract Vault {
    storage var totalDeposits: u256 = 0;
    storage var balances: map<address, u256>;

    pub fn deposit(amount: u256) {
        let sender: address = std.msg.sender();
        balances[sender] += amount;
        totalDeposits += amount;
    }

    pub fn withdraw(amount: u256) {
        let sender: address = std.msg.sender();
        balances[sender] -= amount;
        totalDeposits -= amount;
    }

    pub fn balanceOf(account: address) -> u256 {
        return balances[account];
    }

    pub fn getTotalDeposits() -> u256 {
        return totalDeposits;
    }
}
```

This is the simplest useful vault. It has:
- Two storage variables: `totalDeposits` and `balances`
- Four public functions for deposit, withdraw, and queries
- `std.msg.sender()` to get the caller's address (from the standard library — covered in Chapter 13)

The vault has bugs: nothing prevents withdrawing more than your balance, and there's no protection against zero-amount deposits. We'll fix these in later chapters.

## Further reading

- [Language Basics](../language-basics) — quick syntax reference
- [Signed Integers](../signed-integers) — signed integer details

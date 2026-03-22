---
title: "Chapter 16: Comptime"
description: Compile-time evaluation — constants, blocks, and introspection.
sidebar_position: 16
---

# Comptime

Comptime (compile-time evaluation) runs code during compilation. Values computed at comptime have zero runtime gas cost — they're embedded directly in the bytecode.

## Comptime constants

```ora
const MAX_DEPOSIT: u256 = 1000000;
const MIN_DEPOSIT: u256 = 1;
```

Top-level `const` declarations are evaluated at compile time.

## Comptime blocks

```ora
const VERSION: u256 = comptime {
    return 1 + 2;
};
```

A `comptime { }` block runs arbitrary expressions at compile time and returns a value. The expression inside must be evaluable without runtime state.

## Comptime imports

The most common comptime expression is `@import`:

```ora
comptime const std = @import("std");
```

`@import` is a comptime operation — it resolves the module at compile time and makes its symbols available.

## Builtin functions

Ora provides compile-time introspection builtins:

```ora
comptime {
    const size: u256 = @sizeOf(u256);        // 32 (bytes)
    const name: string = @typeName(u256);     // "u256"
}
```

- `@sizeOf(T)` — byte size of type `T`
- `@typeName(T)` — string name of type `T`
- `@keccak256(data)` — compile-time keccak hash

## Comptime in practice

Comptime is useful for:

**Configuration constants:**
```ora
const MAX_DEPOSIT: u256 = 1_000_000;
const FEE_BASIS_POINTS: u256 = 250;   // 2.5%
const DECIMALS: u256 = 18;
```

**Computed values:**
```ora
const SCALE: u256 = comptime {
    var result: u256 = 1;
    var i: u256 = 0;
    while (i < 18) {
        result = result * 10;
        i += 1;
    }
    return result;
};
// SCALE = 10^18, computed at compile time
```

**Standard library access:**
```ora
comptime const std = @import("std");

pub fn getOwner() -> address {
    return std.msg.sender();
}
```

## The vault with comptime

```ora
const MAX_DEPOSIT: u256 = 1_000_000;
const MIN_DEPOSIT: u256 = 1;
const VERSION: u256 = comptime { return 1 + 2; };

comptime const std = @import("std");

contract Vault {
    storage var totalDeposits: u256 = 0;
    storage var balances: map<address, u256>;

    pub fn deposit(amount: u256) {
        let sender: address = std.msg.sender();
        balances[sender] += amount;
        totalDeposits += amount;
    }

    pub fn getVersion() -> u256 {
        return VERSION;    // embedded as literal 3 in bytecode
    }

    pub fn getMaxDeposit() -> u256 {
        return MAX_DEPOSIT;  // embedded as literal 1000000 in bytecode
    }

    pub fn balanceOf(account: address) -> u256 {
        return balances[account];
    }
}
```

`VERSION` and `MAX_DEPOSIT` are compile-time constants. Reading them costs zero gas beyond the PUSH opcode.

## Further reading

- [Comptime](../research/comptime) — comptime design and research

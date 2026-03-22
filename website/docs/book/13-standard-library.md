---
title: "Chapter 13: Standard Library"
description: The std module — message context, block info, and constants.
sidebar_position: 13
---

# Standard Library

Ora's standard library is imported as a comptime constant:

```ora
comptime const std = @import("std");
```

This gives access to EVM environment data and useful constants.

## Message context

All examples in this chapter assume `std` is imported. Inside a function:

```ora
let sender: address = std.msg.sender();     // Transaction caller (CALLER opcode)
let value: u256 = std.msg.value();          // Attached ETH value (CALLVALUE opcode)
```

`std.msg.sender()` returns the address of the account that called the current function (the EVM CALLER opcode).

`std.msg.value()` returns the amount of ETH sent with the call.

## Transaction context

```ora
let origin: address = std.tx.origin;        // Transaction origin (ORIGIN opcode)
let gas_price: u256 = std.transaction.gasprice;  // Gas price (GASPRICE opcode)
```

`std.transaction.sender()` is an alias for `std.msg.sender()`.

## Block information

```ora
let ts: u256 = std.block.timestamp;          // Current block timestamp (TIMESTAMP)
let num: u256 = std.block.number;            // Current block number (NUMBER)
let miner: address = std.block.coinbase();   // Block coinbase address (COINBASE)
```

## Constants

```ora
let zero: address = std.constants.ZERO_ADDRESS;    // 0x0000...0000
let max: u256 = std.constants.U256_MAX;             // 2^256 - 1
let max128: u128 = std.constants.U128_MAX;          // 2^128 - 1
```

These are compile-time constants — using them costs no gas beyond the PUSH opcode.

## Usage in specifications

Standard library constants are commonly used in `requires` clauses to prevent overflow. Inside a contract:

```ora
pub fn deposit(amount: MinValue<u256, 1>)
    requires(totalDeposits <= std.constants.U256_MAX - amount)
{
    totalDeposits += amount;
}
```

This ensures `totalDeposits + amount` cannot overflow a `u256`.

## Further reading

- [Standard Library](../standard-library) — complete API reference

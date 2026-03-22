---
title: "Chapter 5: Composite Types"
description: Structs, enums, tuples, arrays, and maps.
sidebar_position: 5
---

# Composite Types

Short examples in this chapter are inside a function unless noted otherwise. Full contract examples are self-contained.

## Structs

```ora
struct Point {
    x: u256;
    y: u256;
}
```

Create struct instances with named fields:

```ora
let p: Point = Point { x: 10, y: 20 };
```

Access fields with dot notation:

```ora
let sum: u256 = p.x + p.y;
```

### Nested structs

```ora
struct UserProfile {
    id: u256;
    username: string;
    active: bool;
}

struct AccountData {
    profile: UserProfile;
    balance: u256;
}
```

Construction:

```ora
let account: AccountData = AccountData {
    profile: UserProfile { id: 1, username: "alice", active: true },
    balance: 1000,
};
```

### Anonymous structs

Structs can be used inline without a top-level declaration:

```ora
let result: struct { value: u256, overflow: bool } = @addWithOverflow(a, b);
```

Anonymous structs are useful for builtin return types and ad-hoc groupings. For reusable types, prefer named struct declarations.

## Enums

```ora
enum Status {
    Pending,
    Active,
    Completed,
    Cancelled
}
```

Variants are auto-numbered from 0. Access with dot notation:

```ora
var state: Status = Status.Active;
```

### Backing type

Enums can specify a backing type:

```ora
enum TokenStandard : u8 {
    ERC20 = 0,
    ERC721 = 1,
    ERC1155 = 2
}
```

String-backed enums are also supported:

```ora
enum ErrorCode : string {
    InvalidInput = "ERR_INVALID_INPUT",
    InsufficientFunds = "ERR_INSUFFICIENT_FUNDS"
}
```

### Switch on enums

Enums work naturally with switch:

```ora
switch (state) {
    Status.Pending => { /* handle pending */ },
    Status.Active => { /* handle active */ },
    else => { /* handle other states */ },
}
```

## Tuples

```ora
let pair: (u256, bool) = (42, true);
```

Access elements by position:

```ora
let value: u256 = pair.0;
let flag: bool = pair.1;
```

Tuples are useful for returning multiple values from functions:

```ora
fn divmod(a: u256, b: u256) -> (u256, u256) {
    return (a / b, a % b);
}
```

## Arrays

Fixed-size arrays use `[Type; Size]` syntax:

```ora
let numbers: [u256; 5] = [1, 2, 3, 4, 5];
```

Access elements by index:

```ora
let first: u256 = numbers[0];
let last: u256 = numbers[4];
```

Mutable arrays:

```ora
var scores: [u256; 3] = [0, 0, 0];
scores[0] = 100;
scores[1] = 95;
scores[2] = 87;
```

Storage arrays:

```ora
contract Registry {
    storage var entries: [u256; 100];

    pub fn set(index: u256, value: u256) {
        entries[index] = value;
    }
}
```

## Maps

Maps are key-value stores. Inside a contract:

```ora
storage var balances: map<address, u256>;
```

Read and write with index syntax:

```ora
let balance: u256 = balances[sender];
balances[sender] = 100;
balances[sender] += 50;
```

### Nested maps

Maps can be nested for multi-key lookups:

```ora
storage var allowances: map<address, map<address, u256>>;
```

Access with chained indices:

```ora
let allowed: u256 = allowances[owner][spender];
allowances[owner][spender] = 500;
```

This pattern is common in token contracts for tracking spending approvals.

## The vault with composite types

Let's add structure to our vault with a deposit history:

```ora
comptime const std = @import("std");

struct DepositRecord {
    amount: u256;
    timestamp: u256;
}

enum VaultStatus : u8 {
    Active = 0,
    Paused = 1,
    Closed = 2
}

contract Vault {
    storage var totalDeposits: u256 = 0;
    storage var balances: map<address, u256>;
    storage var status: VaultStatus;

    pub fn deposit(amount: u256) {
        let sender: address = std.msg.sender();
        balances[sender] += amount;
        totalDeposits += amount;
    }

    pub fn withdraw(amount: u256) -> bool {
        let sender: address = std.msg.sender();
        let current: u256 = balances[sender];
        if (current < amount) {
            return false;
        }
        balances[sender] = current - amount;
        totalDeposits -= amount;
        return true;
    }

    pub fn pause() {
        status = VaultStatus.Paused;
    }

    pub fn balanceOf(account: address) -> u256 {
        return balances[account];
    }

    pub fn getTotalDeposits() -> u256 {
        return totalDeposits;
    }
}
```

We've added:
- A `DepositRecord` struct (ready for use in later chapters)
- A `VaultStatus` enum controlling the vault's state
- A `pause()` function using the enum

The vault still has problems: `withdraw` returns `bool` instead of explaining why it failed, and nothing prevents deposits when the vault is paused. The next chapter introduces error unions to fix the first problem.

## Further reading

- [Struct Types](../struct-types) — full struct reference including methods
- [Switch](../switch) — switch patterns with enums

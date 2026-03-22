---
title: "Chapter 7: Memory Regions"
description: Storage, memory, calldata, and transient — where data lives matters.
sidebar_position: 7
---

# Memory Regions

Ora makes data location explicit. Every variable declaration says where the data lives — storage, memory, calldata, or transient. The compiler uses this for effect tracking, gas reasoning, and reentrancy analysis.

## The four regions

| Region | Keyword | Persistence | Gas Cost | Mutability |
|--------|---------|-------------|----------|------------|
| Storage | `storage` | Permanent | ~2,100 read / ~5,000–20,000 write | Read/write |
| Memory | `memory` (or `let`/`var`) | Function lifetime | ~3 per word | Read/write |
| Calldata | (function parameters) | Transaction lifetime | ~3 per word | Read-only |
| Transient | `tstore` | Transaction lifetime | ~100 read/write | Read/write |

### Storage

Persistent contract state. Survives across transactions.

```ora
contract Token {
    storage var totalSupply: u256 = 0;
    storage var balances: map<address, u256>;
}
```

### Memory

Temporary values that exist only during function execution.

Inside a contract:

```ora
pub fn compute(a: u256, b: u256) -> u256 {
    let sum: u256 = a + b;         // memory (implicit)
    var temp: u256 = sum * 2;      // memory (implicit)
    memory var explicit: u256 = 0; // memory (explicit)
    return temp;
}
```

Local `let` and `var` declarations default to memory. You can write `memory var` to be explicit, but it's optional for locals.

### Calldata

Function parameters arrive in calldata — a read-only region. Inside a contract:

```ora
pub fn process(amount: u256) -> u256 {
    // `amount` lives in calldata (read-only)
    // Reading it into a local variable copies it to memory
    let local_amount: u256 = amount;
    return local_amount * 2;
}
```

### Transient storage

EIP-1153 transient storage persists for the duration of a transaction but is cleared afterward. Cheaper than persistent storage.

```ora
contract Session {
    tstore var tempCounter: u256 = 0;

    pub fn increment() {
        tempCounter += 1;
    }
}
```

Use transient storage for values that must survive across internal calls within a single transaction but don't need to persist.

## Region coercion

Ora allows implicit coercion between compatible regions:

- `calldata` → `memory` (reading parameters into locals)
- `storage` → `memory` (reading state into locals)
- `memory` → `storage` (writing locals back to state)
- `transient` → `memory` (reading transient into locals)

Some transitions are forbidden:

- `memory` → `calldata` (can't write to read-only calldata)
- `storage` → `calldata` (can't write to calldata)

The compiler checks these at compile time. Invalid region transitions are type errors.

## Why regions matter

Every storage access is visible at the declaration site — you wrote `storage var count`, so `count += 1` is unambiguously a storage write. The compiler tracks read and write effects per function. A function that reads storage but never writes it has a different effect signature than one that writes. This information feeds into the verification system and reentrancy analysis.

## The vault with explicit regions

```ora
pub fn withdraw(amount: u256) -> !bool | InsufficientBalance {
    let sender: address = std.msg.sender();
    let current: u256 = balances[sender];   // storage → memory (SLOAD)
    if (current < amount) {
        return InsufficientBalance(amount, current);
    }
    balances[sender] = current - amount;     // memory → storage (SSTORE)
    totalDeposits -= amount;                  // read-modify-write on storage
    return true;
}
```

Each storage access is a deliberate operation. The local variable `current` is a memory copy — modifying it does not change storage until explicitly written back.

## Further reading

- [Memory Regions](../memory-regions) — full region reference

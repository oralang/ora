---
title: Memory Regions
description: Explicit storage, memory, calldata, and transient storage regions with compiler-enforced transition rules.
---

# Memory Regions

Ora makes data location explicit. Every variable lives in a named region, and the compiler enforces which transitions between regions are valid. There are no implicit copies — you see every data movement in the source.

## Regions

| Region | Keyword | Persistence | Mutability | EVM Mechanism |
|--------|---------|-------------|------------|---------------|
| Storage | `storage var` | Permanent (across transactions) | Read/write | `SLOAD` / `SSTORE` |
| Memory | `memory var` | Transaction only | Read/write | Memory allocation |
| Calldata | function params | Call only | Read-only | `CALLDATALOAD` |
| Transient | `tstore var` | Transaction only (EIP-1153) | Read/write | `TLOAD` / `TSTORE` |

## Storage

Persistent state that survives across transactions:

```ora
contract Counter {
    storage var count: u256;
    storage var owner: address;

    pub fn inc() {
        count = count + 1;
    }
}
```

### Immutables and constants

```ora
contract Token {
    immutable NAME: string = "Ora";
    storage const MAX_SUPPLY: u256 = 1_000_000;
}
```

## Memory

Local variables that exist for the duration of a transaction:

```ora
pub fn clamp(amount: u256) -> u256 {
    memory var x: u256 = amount;
    if (x > 100) return 100;
    return x;
}
```

Variables declared with `let` or `var` without a region annotation default to stack/register allocation and can be assigned to any region:

```ora
let x: u256 = 42;        // no explicit region — can flow anywhere
```

## Calldata

Function parameters arrive in calldata. Calldata is read-only — attempting to write to a calldata variable is a compile error:

```ora
fn bad(x: u256) {
    calldata var tmp: u256 = x;
    tmp = 1;  // Compile error: calldata is immutable
}
```

## Transient storage

Transient storage (EIP-1153) persists within a transaction but is cleared afterward. Useful for re-entrancy locks and cross-call state within a single transaction:

```ora
contract Guarded {
    storage var counter: u256;
    tstore var lock: bool;
    tstore var temp_map: map<address, u256>;

    pub fn increment() {
        lock = true;
        counter = counter + 1;
        lock = false;
    }
}
```

## Region transitions

Cross-region data movement must go through memory. The compiler rejects direct assignments between non-memory regions.

### Valid transitions

```ora
contract Transitions {
    storage var counter: u256;
    tstore var temp: u256;

    // storage → memory (read out)
    fn read_counter() -> u256 {
        memory var tmp: u256 = counter;
        return tmp;
    }

    // memory → storage (write back)
    fn write_counter(x: u256) {
        memory var tmp: u256 = x;
        counter = tmp;
    }

    // tstore → memory (read out)
    fn read_temp() -> u256 {
        memory var tmp: u256 = temp;
        return tmp;
    }

    // memory → tstore (write back)
    fn write_temp(x: u256) {
        memory var tmp: u256 = x;
        temp = tmp;
    }

    // calldata → memory (function params to local)
    fn copy_param(x: u256) -> u256 {
        memory var local: u256 = x;
        return local;
    }
}
```

### Invalid transitions

Direct assignment between storage and transient storage is a compile error:

```ora
contract Invalid {
    storage var counter: u256;
    tstore var temp: u256;

    fn bad_storage_to_tstore() {
        temp = counter;    // Compile error: cannot assign storage → tstore directly
    }

    fn bad_tstore_to_storage() {
        counter = temp;    // Compile error: cannot assign tstore → storage directly
    }
}
```

To move data between storage and transient storage, go through a memory intermediate:

```ora
fn ok_storage_to_tstore() {
    memory var tmp: u256 = counter;  // storage → memory
    temp = tmp;                       // memory → tstore
}
```

## Region transition summary

| From \ To | Storage | Memory | Transient | Calldata |
|-----------|---------|--------|-----------|----------|
| **Storage** | -- | OK | Error | -- |
| **Memory** | OK | -- | OK | -- |
| **Transient** | Error | OK | -- | -- |
| **Calldata** | -- | OK | -- | -- |

Region checks are part of the type system. Invalid transitions are rejected at compile time.

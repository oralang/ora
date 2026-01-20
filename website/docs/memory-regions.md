---
title: Memory Regions
description: Explicit storage and memory regions for predictable behavior.
---

# Memory Regions

Ora makes storage and memory explicit, so data movement is visible in code and effects are predictable.

## Storage and memory

```ora
contract Counter {
    storage var count: u256;
    storage var owner: address;

    pub fn inc() {
        memory var tmp: u256 = count;
        tmp = tmp + 1;
        count = tmp;
    }
}
```

## Immutables and constants

```ora
contract Token {
    immutable NAME: string = "Ora";
    storage const MAX_SUPPLY: u256 = 1_000_000;
}
```

## Region transitions

Reading from storage or calldata into memory is explicit at the variable declaration site.

```ora
pub fn clamp(amount: u256) -> u256 {
    memory var x: u256 = amount;
    if (x > 100) return 100;
    return x;
}
```

Region checks are part of the type system. Invalid transitions are rejected at compile time.

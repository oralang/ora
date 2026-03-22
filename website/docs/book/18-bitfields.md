---
title: "Chapter 18: Bitfields"
description: Packing multiple values into a single EVM storage word.
sidebar_position: 18
---

# Bitfields

An EVM storage slot is 256 bits. Storing a single `bool` (1 bit) in a full slot wastes 255 bits and costs the same gas as storing a `u256`. Bitfields let you pack multiple small values into one slot.

## Declaring a bitfield

```ora
bitfield ConfigFlags : u256 {
    initialized: bool;
    mode: u8;
    retries: u8;
    epoch: u16;
}
```

The `: u256` specifies the backing type — a single 256-bit storage word. The fields are packed sequentially: `initialized` takes 1 bit, `mode` takes 8 bits, `retries` takes 8 bits, `epoch` takes 16 bits. Total: 33 bits out of 256.

## Reading and writing fields

```ora
contract Config {
    storage var flags: ConfigFlags;

    pub fn configure(m: u8, r: u8, e: u16) {
        let f: ConfigFlags = flags;    // read the packed word
        f.mode = m;
        f.retries = r;
        f.epoch = e;
        f.initialized = true;
        flags = f;                      // write the packed word back
    }

    pub fn getMode() -> u8 {
        let f: ConfigFlags = flags;
        return f.mode;
    }
}
```

The pattern is: read the bitfield into a local, modify fields, write it back. Each field access compiles to shift-and-mask operations on the underlying word.

## Gas savings

Without bitfields, storing four values requires four SSTORE operations (~80,000 gas). With a bitfield, it's one SSTORE (~20,000 gas) because all four values share a single slot.

The tradeoff: every field read/write involves shift-and-mask arithmetic (~10–20 gas), which is negligible compared to storage costs.

## When to use bitfields

Use bitfields when:
- You have multiple small values that are read/written together
- Storage gas cost matters (it almost always does)
- The values fit comfortably in 256 bits

Don't use bitfields when:
- Values are large (multiple u256)
- Values are updated independently by different functions (you'd always load/store the full word)
- Readability matters more than gas optimization

## The vault with bitfields

Adding packed configuration to the vault:

```ora
bitfield VaultConfig : u256 {
    paused: bool;
    frozen: bool;
    tier: u8;
    maxDeposit: u128;
}

contract Vault {
    storage var config: VaultConfig;
    storage var balances: map<address, u256>;

    pub fn pause() {
        let c: VaultConfig = config;
        c.paused = true;
        config = c;
    }

    pub fn isPaused() -> bool {
        let c: VaultConfig = config;
        return c.paused;
    }

    pub fn setTier(newTier: u8) {
        let c: VaultConfig = config;
        c.tier = newTier;
        config = c;
    }
}
```

The `paused`, `frozen`, `tier`, and `maxDeposit` fields all share one storage slot.

## Further reading

- [Bitfield Types](../bitfield-types) — full bitfield reference including explicit layout

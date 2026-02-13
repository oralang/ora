---
sidebar_position: 3.7
title: "Bitfield Types"
description: "Pack flags and small integers into a single word with compiler-checked layouts."
---

# Bitfield Types

A `bitfield` packs multiple small values (flags, counters, tiers) into a single EVM word. The compiler checks the layout at compile time and lowers reads/writes to efficient mask/shift operations.

## Declaration

### Explicit Layout

Use `@at(offset, width)` to place each field at a specific bit position:

```ora
bitfield AccountFlags : u256 {
    is_admin : bool @at(0, 1);
    is_frozen: bool @at(1, 1);
    tier     : u8   @at(2, 3);    // 3 bits → 0..7
    nonce    : u16  @at(5, 16);
    delta    : i8   @at(21, 4);   // 4 bits → -8..7 (signed)
}
```

You can also use `@bits(start..end)` as an alias — `@bits(2..5)` desugars to `@at(2, 3)`:

```ora
bitfield Packed : u256 {
    flag:  bool @bits(0..1);
    tier:  u8   @bits(1..4);
    nonce: u16  @bits(4..20);
}
```

### Auto-Packed Layout

Omit `@at()` to let the compiler pack fields sequentially from bit 0:

```ora
bitfield AccountFlags : u256 {
    is_admin : bool;       // @at(0, 1)
    is_frozen: bool;       // @at(1, 1)
    tier     : u8(3);      // @at(2, 3) — explicit 3-bit width
    nonce    : u16;        // @at(5, 16) — uses full u16 width
}
```

Mixing `@at()` and auto-packed fields in the same bitfield is a compile error.

## Field Types

| Type | Width Rule | Range |
|------|-----------|-------|
| `bool` | always 1 bit | 0 or 1 |
| `uN` | default `N`, or explicit `uN(W)` | 0 to 2^W − 1 |
| `iN` | default `N`, or explicit `iN(W)` (W ≥ 2) | −2^(W−1) to 2^(W−1) − 1 |

Signed fields store two's complement values. Reads automatically sign-extend to the full word width using `SHL` + `SAR`.

## Reading Fields

Dot notation reads a field — the compiler emits `SHR` + `AND` (unsigned) or adds `SHL` + `SAR` for sign extension (signed):

```ora
let flags: AccountFlags = ...;
let admin = flags.is_admin;   // SHR(0) + AND(1)
let tier  = flags.tier;       // SHR(2) + AND(7)
let delta = flags.delta;      // SHR(21) + AND(0xF) + SHL(252) + SAR(252)
```

Each read carries an implicit refinement — the compiler knows `tier` is in `[0, 7]`, so guards like `tier < 8` are eliminated at compile time.

## Writing Fields

Dot notation writes use clear-then-set:

```ora
flags.tier = 5;
// Lowers to: cleared = flags & ~(0x7 << 2);  updated = cleared | ((5 & 0x7) << 2)
```

Values are range-checked: assigning a value outside the field's range reverts by default. Use `@truncate(value)` for explicit silent truncation.

## Construction

Build a bitfield with named fields. Omitted fields default to 0:

```ora
let flags = AccountFlags {
    is_admin: true,
    tier: 3,
};  // is_frozen = false, nonce = 0, delta = 0
```

If all values are compile-time constants, the entire word folds to a single constant.

## Storage & Batching

A storage-backed bitfield occupies one slot, identical to its base integer type:

```ora
contract MyContract {
    storage var flags: AccountFlags;

    pub fn setup() {
        // Single field write: SLOAD → modify → SSTORE
        flags.is_admin = true;
    }
}
```

**Batching**: consecutive field writes to the same storage bitfield are automatically batched into a single SLOAD/SSTORE cycle:

```ora
pub fn configure() {
    flags.is_admin = true;
    flags.tier = 5;
    flags.nonce = 42;
    // Compiler batches into:
    //   tmp = SLOAD(slot)
    //   tmp = clear/set is_admin
    //   tmp = clear/set tier
    //   tmp = clear/set nonce
    //   SSTORE(slot, tmp)
}
```

Cost: **1 SLOAD + N mask/shift sequences + 1 SSTORE**, regardless of how many fields are updated.

## Conversion

Convert between bitfield and raw integer with `@bitCast`:

```ora
let raw: u256 = @bitCast(u256, flags);           // bitfield → integer
let flags2: AccountFlags = @bitCast(AccountFlags, raw);  // integer → bitfield
```

`@bitCast` performs **no masking or validation**. Use `.sanitize()` after casting from untrusted data.

## Utility Methods

### `.zero()`

Returns the all-zero bitfield value:

```ora
let empty = AccountFlags.zero();
```

### `.sanitize()`

Clears all bits not owned by any declared field:

```ora
let raw: u256 = untrusted_input;
let flags = @bitCast(AccountFlags, raw).sanitize();
// Unused bits are now guaranteed to be 0
```

## Gas Cost Summary

| Operation | EVM Cost |
|-----------|---------|
| Unsigned field read | ~6 gas (SHR + AND) |
| Signed field read | ~12 gas (SHR + AND + SHL + SAR) |
| Field write (memory) | ~12–18 gas |
| Field write (storage, single) | 2100–5000 gas (SLOAD + ops + SSTORE) |
| Field write (storage, batched N) | 2100–5000 gas total |
| Construction (all constants) | 3 gas (single PUSH) |

## Relationship to Structs

- **Bitfields** pack at **bit granularity** within a single word. Fields can be 1-bit bools, 3-bit integers, etc.
- **Structs** pack at **byte granularity** across one or more slots.

A struct field can itself be a bitfield for maximum density:

```ora
packed struct Position {
    owner:    address,              // 20 bytes
    flags:    PositionFlags,        // 1 byte (bitfield over u8)
    token_id: u32,                  // 4 bytes
}
```

For the full formal specification, see the [Bitfield Spec](/docs/specs/bitfield).

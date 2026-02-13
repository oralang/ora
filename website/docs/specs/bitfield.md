---
sidebar_position: 3
title: "Bitfield Type Specification"
description: "Spec for Ora's bitfield type: packed word layout with sub-byte field granularity"
---

# Spec: `bitfield` Type (Packed Word Layout)

**Author:** Axe
**Status:** Implemented
**Target:** Ora Language (EVM)
**Goal:** Provide a safe, explicit, zero/low-overhead way to pack flags and small integers into a single word with predictable lowering and strong static checks.

---

## Motivation

Smart contracts frequently pack many small values into a single `u256` slot for gas efficiency and ABI simplicity (e.g., flags, tiers, counters, small enums). Today this is typically done manually with bit masks and shifts, which is error-prone and hard to verify.

Ora can provide a first-class `bitfield` type that:
- has an explicit, compiler-checked layout (no overlaps, fits in base word),
- lowers to a small fixed set of EVM operations (`SHR`, `SHL`, `AND`, `OR`, `NOT`),
- integrates with refinements and SMT to eliminate guards when values are statically known or refined.

---

## Summary

A `bitfield` defines named fields occupying bit ranges within a base integer word (default `u256`). The runtime representation is exactly the base word. Field reads/writes compile to mask/shift operations.

---

## Syntax

### Declaration (Explicit Layout)

```ora
bitfield <Name> : <BaseInt> {
    <field_name> : <FieldType> @at(<offset>, <width>);
    ...
}
```

- `<BaseInt>`: an unsigned integer type, e.g. `u256`, `u128`, `u64`.
- `@at(offset, width)` declares a bit range:
  - `offset`: starting bit index (LSB = 0)
  - `width`: number of bits (>= 1)

Example:

```ora
bitfield AccountFlags : u256 {
    is_admin : bool @at(0, 1);
    is_frozen: bool @at(1, 1);
    tier     : u8   @at(2, 3);    // 0..7
    nonce    : u16  @at(5, 16);   // 0..65535
    delta    : i8   @at(21, 4);   // -8..7 (two's complement in 4 bits)
}
```

### Declaration (Auto-Packed Layout)

When explicit bit positions are unnecessary (internal flags, no cross-contract stability requirement), fields can omit `@at()`. The compiler assigns sequential offsets starting from bit 0 in declaration order.

For non-boolean fields without `@at()`, the width annotation `(width)` is required.

```ora
bitfield AccountFlags : u256 {
    is_admin : bool;       // auto: @at(0, 1)
    is_frozen: bool;       // auto: @at(1, 1)
    tier     : u8(3);      // auto: @at(2, 3), width = 3
    nonce    : u16;        // auto: @at(5, 16), width = bitwidth(u16)
}
```

Auto-packing rules:
- Fields are packed sequentially in declaration order, starting at bit 0.
- `bool` fields always have width 1.
- `uN` fields without a width annotation use `bitwidth(uN)` as width.
- `uN(W)` fields use `W` as width (must satisfy `1 <= W <= bitwidth(uN)`).
- `iN` fields without a width annotation use `bitwidth(iN)` as width.
- `iN(W)` fields use `W` as width (must satisfy `2 <= W <= bitwidth(iN)`).
- Mixing `@at()` and auto-packed fields within the same bitfield is a compile error.

The auto-packed layout is deterministic and can be inspected via `@typeInfo` (future comptime builtin).

---

## Field Types

Allowed field types:

1. `bool` — must have width == 1.
2. Unsigned integers `uN` (any N) — stored as a value with an implicit refinement `InRange<uN, 0, 2^width - 1>`.
3. Signed integers `iN` (any N) — stored in two's complement within the field's bit width. Reads produce a sign-extended value with an implicit refinement `InRange<iN, -(2^(width-1)), 2^(width-1) - 1>`. Width must be >= 2 (1-bit signed has range `[-1, 0]`, which is almost never useful — use `bool` instead).

Enum fields are not yet specified. When supported, width must satisfy `2^width >= variant_count`. Multi-word bitfields (spanning more than one base word) are out of scope for this spec.

---

## Semantics

### Representation

A value of type `bitfield Name : BaseInt` is represented at runtime as a single `<BaseInt>` word.

- ABI encoding: encode as `<BaseInt>`.
- Storage layout: if stored, occupies one storage slot (or one word in the selected region), identical to storing `<BaseInt>`.

### Bit Numbering

- Bit 0 is the least significant bit (LSB) of the base word.
- Field `(offset, width)` occupies bits `[offset, offset + width - 1]`.

### Refinement Integration

Every field carries an implicit refinement based on its bit width and signedness:

- `bool` fields: no additional refinement (already `{0, 1}`).
- `uN` fields with width `W`: the compiler attaches `InRange<uN, 0, 2^W - 1>` to every read. A `tier: u8 @at(2, 3)` field is known to be in `[0, 7]` without any runtime check. Downstream guards like `tier < 8` can be statically eliminated. SMT gets a tight bound for free.
- `iN` fields with width `W`: the compiler attaches `InRange<iN, -(2^(W-1)), 2^(W-1) - 1>` to every read. A `delta: i8 @at(0, 4)` field is known to be in `[-8, 7]`.

Writes enforce the matching constraint:
- For unsigned fields: assigning a value outside `[0, 2^W - 1]` is a compile-time error if the value is statically known, or a checked runtime truncation otherwise (following the overflow model: checked by default, explicit wrapping with `@truncate()`).
- For signed fields: assigning a value outside `[-(2^(W-1)), 2^(W-1) - 1]` follows the same rules. After range validation, the stored bits are the low `width` bits of the two's complement representation: `stored = v & mask`. This matches the read-side `SHL` + `SAR` sign extension.

### Layout Validation (Compile-Time)

Per-field checks:
- `width >= 1`
- `offset + width <= bitwidth(BaseInt)`
- If `FieldType == bool`, then `width == 1`
- If `FieldType == uN(W)`, then `1 <= W <= bitwidth(uN)`
- If `FieldType == iN`, then `width >= 2` (1-bit signed is rejected)
- If `FieldType == iN(W)`, then `2 <= W <= bitwidth(iN)`

Cross-field checks:
- No overlapping bit ranges. For any two fields A and B with ranges `[a_offset, a_offset + a_width - 1]` and `[b_offset, b_offset + b_width - 1]`, reject if intervals overlap.
- Gaps between fields are allowed.
- Source ordering does not affect layout (for explicit `@at()` mode).

Diagnostics: on overlap or invalid range, emit a compile-time error identifying both fields (with source spans) and their bit ranges.

---

## Operations

### Field Read

Given base word `w`, field `(offset, width)`:

**Unsigned and bool fields:**

```
mask = (1 << width) - 1        // compile-time constant
raw  = (w >> offset) & mask     // runtime: SHR + AND
```

Return value:
- If `bool`: `raw != 0`
- If `uN`: return `raw` as the declared field type, carrying the implicit `InRange<uN, 0, 2^width - 1>` refinement.

**Signed fields (two's complement within `width` bits):**

```
mask  = (1 << width) - 1         // compile-time constant
raw   = (w >> offset) & mask      // runtime: SHR + AND

// Bit-precise sign extension to 256 bits using SHL + SAR:
// shift left so the sign bit lands in bit 255, then arithmetic shift back.
shift  = 256 - width              // compile-time constant
result = SAR(shift, SHL(shift, raw))
```

This produces a correct two's complement signed value in the full word width. It works for any `width >= 2`, regardless of bit alignment. The result carries the implicit `InRange<iN, -(2^(width-1)), 2^(width-1) - 1>` refinement.

Note: the EVM's `SIGNEXTEND` opcode takes a byte index (0–31), so it can only sign-extend from bit 7, 15, 23, etc. To use it for signed bitfield reads, every signed field would need to be rounded up to a multiple of 8 bits — a 4-bit field (`[-8, 7]`) would become 8 bits, doubling its storage. Since the entire purpose of `bitfield` is sub-byte packing density, that tradeoff is not acceptable. `SHL` + `SAR` costs one extra opcode per read but preserves arbitrary bit-width packing.

### Field Write (Pure Update)

A write returns a new bitfield word. Bitfield values have value semantics.

Given word `w`, field `(offset, width)`, new value `v`:

```
mask    = (1 << width) - 1              // compile-time constant
cleared = w & ~(mask << offset)          // clear field bits
updated = cleared | ((v & mask) << offset)  // set new value
```

Range enforcement on `v`:
- For unsigned fields: `v` must be in `[0, 2^width - 1]`.
- For signed fields: `v` must be in `[-(2^(width-1)), 2^(width-1) - 1]`. The write stores only the low `width` bits of the two's complement representation (masking is the same operation for both signed and unsigned — the difference is in range validation and read-side sign extension).
- If `v` is a compile-time constant outside the valid range, emit a compile-time error.
- If `v` is runtime-computed and the compiler cannot prove it is in range via refinements, insert a checked truncation (revert on overflow by default). The developer can opt into silent truncation with `@truncate(v)`, following Ora's explicit-overflow philosophy.

### Construction

Bitfield construction uses named fields. Omitted fields default to 0.

```ora
let flags = AccountFlags {
    is_admin: true,
    tier: 3,
};  // is_frozen = false, nonce = 0, delta = 0
```

The compiler emits a single constant or a sequence of `OR`/`SHL` operations to build the word. If all specified fields are compile-time constants (and omitted fields are 0), the entire word is folded to a single constant.

Supplying an unknown field name or duplicating a field is a compile-time error.

### Conversion

```ora
// bitfield to base word
let raw: u256 = @bitCast(u256, flags);

// base word to bitfield (unchecked reinterpretation)
let flags2: AccountFlags = @bitCast(AccountFlags, raw);
```

`@bitCast` performs no masking or validation. The resulting bitfield may contain values outside field ranges if the source word has arbitrary bit patterns. Reads from such a bitfield still produce masked values (the read operation always masks), so safety is preserved on access, not on construction.

---

## Storage Interaction

### Single-Field Read from Storage

Reading a single field from a storage-backed bitfield requires loading the full word:

```ora
storage var flags: AccountFlags;

let admin = flags.is_admin;
// Lowers to: SLOAD(slot) → SHR(0) → AND(1) → result
```

Cost: 1 `SLOAD` + constant arithmetic. Same as reading a bare `u256` from storage.

### Single-Field Write to Storage

Writing a single field performs a read-modify-write:

```ora
flags.is_admin = true;
// Lowers to: SLOAD(slot) → clear bit 0 → set bit 0 → SSTORE(slot)
```

Cost: 1 `SLOAD` + constant arithmetic + 1 `SSTORE`.

### Batched Writes

When multiple fields are updated in sequence within the same scope, the compiler must batch them into a single `SLOAD` ... mutations ... `SSTORE` cycle:

```ora
flags.is_admin = true;
flags.tier = 5;
flags.nonce = 42;
// Lowers to:
//   tmp = SLOAD(slot)
//   tmp = clear/set is_admin
//   tmp = clear/set tier
//   tmp = clear/set nonce
//   SSTORE(slot, tmp)
```

Cost: 1 `SLOAD` + N mask/shift sequences + 1 `SSTORE`, regardless of how many fields are updated.

The compiler should emit a diagnostic (warning or note) when it detects unbatched writes to the same bitfield in storage (multiple `SLOAD`/`SSTORE` pairs where one would suffice).

---

## Builtin Attribute: `@at(offset, width)`

### Purpose

`@at(offset, width)` is a builtin layout attribute that assigns a field to a specific bit range inside a packed container. It is evaluated at compile time and contributes to layout validation and lowering.

This attribute is not a general expression-level operator. It is only valid in declaration contexts that support layout attributes (`bitfield` fields).

### Static Semantics

Let `BaseInt` be the base integer type of the containing bitfield, and `B = bitwidth(BaseInt)`.

`@at(offset, width)` is well-typed iff:
1. `offset` is a compile-time constant integer, `0 <= offset < B`
2. `width` is a compile-time constant integer, `1 <= width <= B`
3. `offset + width <= B`

Both `offset` and `width` must be comptime-evaluable expressions. Non-constant values are a compile-time error.

### Lowering Metadata

During type resolution, `@at()` is lowered into a layout descriptor:

```
FieldLayout {
    field_id: u32,
    offset:   u16,
    width:    u16,
    mask:     BaseInt,   // (1 << width) - 1, precomputed
    shift:    u16,       // == offset
}
```

These constants are computed at compile time and used directly in read/write lowering.

### Error Cases

The compiler must produce a compile-time error for:
- Non-constant `offset` or `width` expressions
- `width == 0`
- `offset + width > bitwidth(BaseInt)`
- `bool` field with `width != 1`
- Duplicate `@at()` on the same field
- Overlapping bit ranges across fields (with spans for both conflicting fields)
- Use of `@at()` outside a `bitfield` declaration

### `@bits(a..b)` Alias

`@bits(a..b)` is an ergonomic alias for `@at(a, b - a)`. The range is **half-open**: bit `a` is included, bit `b` is excluded. This matches Ora's range semantics elsewhere in the language and avoids the off-by-one ambiguity of inclusive ranges.

```ora
bitfield Packed : u256 {
    flag:  bool @bits(0..1);     // desugars to @at(0, 1)
    tier:  u8   @bits(1..4);     // desugars to @at(1, 3), width = 3
    nonce: u16  @bits(4..20);    // desugars to @at(4, 16), width = 16
}
```

Validation: the compiler rejects `@bits(a..b)` when `b <= a` (empty or negative range). All other validation (overlap, bounds, field-type compatibility) is identical to `@at()` since the alias desugars before layout checking.

Both `@at()` and `@bits()` are available. Use `@at()` when thinking in offset+width. Use `@bits()` when thinking in bit ranges. They cannot both appear on the same field.

---

## Relationship to `packed struct`

`bitfield` and `packed struct` are complementary:

- `bitfield` packs at **bit granularity** within a single word. Fields can be 1-bit booleans, 3-bit integers, etc. The total must fit in one base word.
- `packed struct` packs at **byte granularity** across one or more words. Fields are byte-aligned (addresses occupy 20 bytes, bools occupy 1 byte, etc.).

A `packed struct` field can itself be a `bitfield` for maximum density:

```ora
packed struct Position {
    owner:    address,                   // 20 bytes
    flags:    PositionFlags,             // 1 byte (bitfield over u8)
    token_id: u32,                       // 4 bytes
    // total: 25 bytes, fits in one 32-byte slot
}

bitfield PositionFlags : u8 {
    is_active:   bool @at(0, 1);
    is_locked:   bool @at(1, 1);
    priority:    u8   @at(2, 3);   // 0..7
    // 3 bits unused
}
```

---

## SMT Encoding

The Z3 encoder models bitfield operations as bitvector arithmetic:

- Unsigned field read: `Extract(offset + width - 1, offset, word)` or `(bvand (bvlshr word offset) mask)`
- Signed field read: same extraction, then `SignExt(N - width, extracted)` to sign-extend to full width.
- Field write: `(bvor (bvand word (bvnot (bvshl mask offset))) (bvshl (bvand value mask) offset))`
- Unsigned refinement bounds: for each read, assert `0 <= value < 2^width`.
- Signed refinement bounds: for each read, assert `-(2^(width-1)) <= value < 2^(width-1)`.

This is standard bitvector theory and does not require solver extensions.

---

## Summary of Lowering

| Operation | EVM Opcodes | Gas (approximate) |
|-----------|-------------|-------------------|
| Unsigned field read | `SHR` + `AND` | 6 gas (2 ops) |
| Signed field read | `SHR` + `AND` + `SHL` + `SAR` | 12 gas (4 ops) |
| Field write (memory) | `AND` + `NOT` + `OR` + `SHL` | 12-18 gas (4-6 ops) |
| Field write (storage, single) | `SLOAD` + mask/shift + `SSTORE` | 2100-5000 gas |
| Field write (storage, batched N) | `SLOAD` + N * mask/shift + `SSTORE` | 2100-5000 gas |
| Construction (all constants) | single `PUSH` | 3 gas |
| Construction (mixed) | N * (`SHL` + `OR`) | ~6N gas |

---

## Utility Operations

### Zero

`T.zero()` returns the all-zero bitfield value (base word = 0). All fields are at their default.

```ora
let empty = AccountFlags.zero();
```

### Sanitize

`T.sanitize(self)` clears all bits not owned by any declared field, producing a canonical representation:

```
sanitize(w) = w & used_mask
```

Where `used_mask = OR over fields (((1 << width) - 1) << offset)` is computed at compile time.

This is useful when a bitfield is constructed from a raw word via `@bitCast` and a canonical form is needed for stable hashing, equality comparison, or enforcing "unused bits must be zero" invariants.

```ora
let raw: u256 = untrusted_input;
let flags = @bitCast(AccountFlags, raw).sanitize();
// unused bits are now guaranteed to be 0
```

---

## Resolved Questions

1. **Default values:** All fields default to 0 (zero-initialized). No per-field default syntax needed.

2. **`@lock` interaction:** `@lock` on a storage-backed bitfield locks the entire slot via `TSTORE`. The slot cannot be written until an explicit `@unlock` or transaction end. This is consistent with Ora's existing `@lock` semantics — it operates at slot granularity, not field granularity. Individual field writes within a locked slot are rejected by the compiler.

## Open Questions

1. Can bitfield values participate in pattern matching (`switch` on individual fields)? This would allow matching on specific flag combinations but requires careful design of the pattern syntax and exhaustiveness checking

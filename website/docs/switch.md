---
id: switch
title: Switch
description: The Ora switch statement and expression with labels, ranges, enums, and semantics
---

### Overview

Ora provides a powerful switch that works as both a statement and an expression. It supports:

- Literal, enum, and range patterns
- Multiple comma-separated patterns per case
- Optional commas between cases
- An `else` (default) arm that must be last
- Labeled switch statements and labeled block arms
- `continue :label value;` to re-evaluate a labeled switch with a new operand (Zig 0.14 style)

### Basics

```ora
var x: u256 = 3;

// Switch as a statement
switch (x) {
  0 => { log "zero"; },
  1, 2 => { log "one or two"; },
  3...5 => { log "in [3,5]"; },
  else => { log "other" }
}

// Switch as an expression
var y: u256 = switch (x) {
  0 => 10,
  1...9 => 20,
  else => 30
};
```

Notes:
- Commas between cases are allowed, but the `else` arm must be last and cannot be followed by a comma.
- Switch expressions require each arm body to be a single expression (no blocks).

### Patterns

- Literal patterns: integers, bool, string, address, hex, binary
- Enum patterns: `Enum.Variant` or bare `Variant` when the condition is that enum
- Range patterns: `a...b` (inclusive) and `a..b` (exclusive)

```ora
enum Color { Red, Green, Blue }
var c: Color = .Red;

switch (c) {
  Color.Red => { log "R"; },
  Green => { log "G"; },
  Blue => { log "B"; }
}

// Ranges with integers
switch (x) {
  0..10 => { log "[0,10)"; },
  10...20 => { log "[10,20]"; },
  else => { log "other" }
}
```

### Labeled switches and labeled blocks

You can label a switch and target it with `continue :label value;`. The value becomes the new operand and the switch re-executes from the top.

```ora
label: switch (x) {
  0 => { continue :label 1; },
  1 => { continue :label 2; },
  else => { log "done" }
}
```

Arms in switch statements may be blocks or labeled blocks:

```ora
switch (x) {
  0 => block0: { log "in block0"; },
  1 => labelA: { log "A"; },
  else => { log "fallback" }
}
```

In contrast, switch expressions only allow expression bodies:

```ora
var v: u256 = switch (x) {
  0 => 10,
  else => 42
};
```

### Semantics and checks

- Type compatibility: each pattern must be compatible with the switch operand type
- Enum names: qualified enum patterns must match the operand enum
- Ranges: endpoints must be compatible with the operand; overlapping integer ranges are rejected
- Duplicates: duplicate literals or enum variants are rejected
- `else`: at most one `else`, and it must be the last arm
- Expressions: all arms in a switch expression must produce compatible result types
- Exhaustiveness: when switching on enums without `else`, all variants must be covered

Examples of invalid cases (rejected by the compiler):

```ora
// Duplicate enum variants
switch (c) {
  Red => { },
  Red => { }, // duplicate
}

// Overlapping ranges
switch (x) {
  1...5 => { },
  4...10 => { }, // overlaps with 1...5
}

// Non-exhaustive enum without else
switch (c) {
  Red => { },
  Green => { },
  // Blue missing and no else
}
```

### Tips

- Prefer explicit enum coverage to catch additions of new variants.
- Use labeled switches to write state-machine-like control flow without loops.
- Keep switch expressions simple: return a single, consistent type across all arms.

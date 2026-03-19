---
id: switch
title: Switch
description: Pattern matching with switch statements and expressions — literal, range, and enum patterns with exhaustiveness checking.
---

# Switch

Ora's `switch` works as both a statement and an expression. It supports literal patterns, inclusive/exclusive ranges, enum variants, and an `else` (default) arm. No fallthrough — each arm is independent.

## Switch Statements

```ora
contract Router {
    pub fn classify(value: u256) -> u256 {
        switch (value) {
            0 => { return 10; }
            1 => { return 20; }
            2 => { return 30; }
            else => { return 0; }
        }
    }
}
```

Each arm has a pattern and a block body. No fallthrough between arms — unlike Solidity/C, you don't need `break`.

## Switch Expressions

Switch can be used as an expression — each arm produces a value:

```ora
pub fn classify(value: u256) -> u256 {
    return switch (value) {
        0 => 100,
        1 => 200,
        2 => 300,
        else => 0
    };
}
```

All arms in a switch expression must produce compatible types.

## Range Patterns

Inclusive ranges with `...` and exclusive ranges with `..`:

```ora
pub fn tier(value: u256) -> u256 {
    switch (value) {
        0...9 => { return 1; }      // 0 through 9 inclusive
        10...99 => { return 2; }    // 10 through 99 inclusive
        100...999 => { return 3; }  // 100 through 999 inclusive
        else => { return 0; }
    }
}
```

The compiler rejects overlapping ranges:

```ora
switch (x) {
    1...5 => { }
    4...10 => { }  // Compile error: overlaps with 1...5
}
```

## Enum Patterns

When switching on an enum, use qualified or bare variant names:

```ora
enum Color { Red, Green, Blue }

pub fn describe(c: Color) -> u256 {
    switch (c) {
        Color.Red => { return 1; }
        Green => { return 2; }       // bare variant OK when type is known
        Blue => { return 3; }
    }
}
```

Without `else`, all variants must be covered:

```ora
switch (c) {
    Red => { }
    Green => { }
    // Compile error: non-exhaustive — Blue not covered and no else
}
```

## Labeled Switch

A labeled switch can be re-entered with `continue :label value`, where the value becomes the new operand and the switch re-evaluates:

```ora
pub fn stateMachine(start: u256) -> u256 {
    var x: u256 = start;
    outer: switch (x) {
        0 => { x = 100; }
        1...5 => { x = x + 1; }
        else => {
            continue :outer (0);
        }
    }
    return x;
}
```

The compiler lowers this as a loop that re-dispatches on the new value. `break :label` exits the labeled switch.

## Else Arm

The `else` arm handles all unmatched values. It must be the last arm:

```ora
switch (value) {
    0 => { return 1; }
    else => { return 0; }  // must be last
}
```

For enum switches, `else` makes the switch non-exhaustive — the compiler won't warn about missing variants.

## Compiler Checks

The compiler validates:

| Check | What's rejected |
|-------|----------------|
| Duplicate literals | Same value in two arms |
| Duplicate enum variants | Same variant in two arms |
| Overlapping ranges | `1...5` and `4...10` in the same switch |
| Non-exhaustive enum | Missing variants without `else` |
| Type mismatch | Pattern type incompatible with operand |
| `else` position | `else` not last |
| Expression type consistency | Arms in switch expression with different types |


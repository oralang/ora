---
title: "Chapter 4: Control Flow"
description: If/else, loops, switch expressions, and labeled blocks.
sidebar_position: 4
---

# Control Flow

## If/else

All control flow examples in this chapter are inside a function.

```ora
if (balance >= amount) {
    balance -= amount;
} else {
    // handle insufficient balance
}
```

Conditions must be `bool`. Ora does not implicitly convert integers to booleans — `if (x)` is a type error; write `if (x != 0)`.

Chained conditions:

```ora
if (score >= 90) {
    return 4;
} else if (score >= 80) {
    return 3;
} else if (score >= 70) {
    return 2;
} else {
    return 1;
}
```

## While loops

```ora
var i: u256 = 0;
while (i < 10) {
    i += 1;
}
```

While loops can carry an `invariant` clause for formal verification (covered in Chapter 10):

```ora
while (i < limit)
    invariant i <= limit
{
    i += 1;
}
```

## For loops

### Range iteration

```ora
for (0..10) |value, index| {
    // value and index both go from 0 to 9
}
```

`0..10` is a half-open range: includes 0, excludes 10. The `|value, index|` syntax captures the loop value and its index.

Use `_` to discard a capture:

```ora
for (0..5) |i, _| {
    // only use i, discard the index
}
```

### Array iteration

```ora
let numbers: [u256; 5] = [1, 2, 3, 4, 5];
for (numbers) |value, index| {
    // value is the element, index is the position
}
```

### Nested loops

```ora
for (0..3) |i, _| {
    for (0..3) |j, _| {
        // i and j available here
    }
}
```

## Break and continue

```ora
var result: u256 = 0;
for (0..100) |i, _| {
    if (i == 50) {
        break;       // exit the loop
    }
    if (i % 2 == 0) {
        continue;    // skip to next iteration
    }
    result += i;
}
```

## Labeled blocks

Labeled blocks let you break out of specific scopes:

```ora
outer: {
    var j: u256 = 0;
    while (j < 10) {
        j += 1;
        if (j == 5) {
            break :outer;    // exit the labeled block, not just the loop
        }
    }
}
```

## Switch

Switch works as both a statement and an expression.

### Switch statement

```ora
switch (status) {
    0 => {
        handlePending();
    },
    1 => {
        handleActive();
    },
    2 => {
        handleCompleted();
    },
    else => {
        handleUnknown();
    },
}
```

### Range patterns

```ora
switch (score) {
    0...59 => {
        return 0;
    },
    60...79 => {
        return 1;
    },
    80...100 => {
        return 2;
    },
    else => {
        return 0;
    },
}
```

`0...59` is an inclusive range: matches 0 through 59.

### Switch expression

Switch can be used as an expression that produces a value:

```ora
let tier: u256 = switch (amount) {
    0...999 => 1,
    1000...9999 => 2,
    else => 3,
};
```

### No fallthrough

Switch arms do not fall through. Each arm is independent — no `break` needed.

## The vault with control flow

Adding a classification function to our vault:

```ora
pub fn depositTier(amount: u256) -> u256 {
    return switch (amount) {
        0...999 => 1,
        1000...9999 => 2,
        10000...99999 => 3,
        else => 4,
    };
}
```

And improving `withdraw` with a balance check:

```ora
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
```

This is better than the original (which had no check), but still not ideal — returning `false` on failure hides the reason. In Chapter 6, we'll use error unions to make failure explicit in the type system.

## Further reading

- [Switch](../switch) — complete switch reference with all pattern types

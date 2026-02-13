---
title: Error Unions and Try
description: Explicit error handling with error unions, try expressions, and try/catch blocks.
---

# Error Unions and Try

Ora uses explicit error unions instead of exceptions. Errors are part of the type system and must be handled.

## Error declarations

```ora
contract Payments {
    error InvalidAmount;
    error InsufficientBalance(required: u256, available: u256);
}
```

## Error union return types

```ora
contract Payments {
    error InvalidAmount;
    error InsufficientBalance(required: u256, available: u256);

    storage var balances: map<NonZeroAddress, u256>;

    fn withdraw(to: NonZeroAddress, amount: u256) -> !u256 | InvalidAmount | InsufficientBalance {
        if (amount == 0) return error.InvalidAmount;
        if (balances[to] < amount) {
            return error.InsufficientBalance(amount, balances[to]);
        }
        balances[to] = balances[to] - amount;
        return balances[to];
    }
}
```

## Try expressions

`try` unwraps a successful value or returns the error up the call stack.

```ora
contract Wallet {
    error Fail;

    fn mayFail(x: u256) -> !u256 | Fail {
        if (x == 0) return error.Fail;
        return x + 1;
    }

    fn run(x: u256) -> !u256 | Fail {
        let y: u256 = try mayFail(x);
        return y * 2;
    }
}
```

## Try/catch blocks

Use `try { ... } catch { ... }` to handle errors explicitly.

```ora
contract Transfers {
    error InsufficientBalance;

    log Sent(to: address, amount: u256);
    log Failed(to: address, amount: u256);

    fn move(to: NonZeroAddress, amount: u256) -> !u256 | InsufficientBalance {
        if (amount == 0) return error.InsufficientBalance;
        return amount;
    }

    pub fn send(to: NonZeroAddress, amount: u256) {
        try {
            let sent: u256 = try move(to, amount);
            log Sent(to, sent);
        } catch (e) {
            log Failed(to, amount);
        }
    }
}
```

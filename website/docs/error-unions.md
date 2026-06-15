---
title: Result, Error Unions, and Try
description: Explicit error handling with Result values, error unions, try expressions, and try/catch blocks.
---

# Result, Error Unions, and Try

Ora uses explicit `Result<T, E>` / error-union values instead of exceptions.
Errors are part of the type system, can carry payloads, can be matched, and are
visible to the ABI and verifier.

## Error declarations

```ora
contract Payments {
    error InvalidAmount;
    error InsufficientBalance(required: u256, available: u256);
}
```

## Result return types

```ora
contract Payments {
    error InvalidAmount;
    error InsufficientBalance(required: u256, available: u256);

    storage var balances: map<NonZeroAddress, u256>;

    fn withdraw(to: NonZeroAddress, amount: u256) -> Result<u256, InsufficientBalance> {
        if (balances[to] < amount) {
            return Err(InsufficientBalance(amount, balances[to]));
        }
        balances[to] = balances[to] - amount;
        return Ok(balances[to]);
    }
}
```

The `!T | E1 | E2` spelling is also used for public error-union signatures,
especially where success and error postconditions use `ensures_ok` /
`ensures_err`.

## Matching result values

```ora
contract Payments {
    error Failure(code: u256);

    pub fn inspect(value: Result<u256, Failure>) -> u256 {
        return match (value) {
            Ok(inner) => inner,
            Err(err) => err.code,
        };
    }
}
```

## Try expressions

`try` unwraps a successful value or returns the error up the call stack.

```ora
contract Wallet {
    error Fail();

    fn mayFail(x: u256) -> !u256 | Fail {
        if (x == 0) return Err(Fail());
        return Ok(x + 1);
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
    error InsufficientBalance();

    log Sent(to: address, amount: u256);
    log Failed(to: address, amount: u256);

    fn move(to: NonZeroAddress, amount: u256) -> !u256 | InsufficientBalance {
        if (amount == 0) return Err(InsufficientBalance());
        return Ok(amount);
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

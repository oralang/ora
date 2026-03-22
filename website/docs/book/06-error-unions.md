---
title: "Chapter 6: Error Unions"
description: Explicit error handling with error unions, try expressions, and try/catch.
sidebar_position: 6
---

# Error Unions

Ora makes errors part of the type system. A function that can fail declares its error types in the return type. The caller must handle them — the compiler rejects code that ignores a possible error.

## Declaring errors

```ora
error InsufficientBalance(required: u256, available: u256);
error ZeroAmount;
error Unauthorized(caller: address);
```

Errors are declared at the top level. They can be nullary (`ZeroAmount`) or carry fields (`InsufficientBalance` with two `u256` fields).

## Error union return types

A function that can fail returns an error union:

```ora
pub fn withdraw(amount: u256) -> !bool | InsufficientBalance {
    let current: u256 = balances[sender];
    if (current < amount) {
        return InsufficientBalance(amount, current);
    }
    balances[sender] = current - amount;
    return true;
}
```

The return type `!bool | InsufficientBalance` means: "this function returns either a `bool` on success, or an `InsufficientBalance` error." The `!` prefix marks it as an error union.

Multiple errors:

```ora
pub fn deposit(amount: u256) -> !bool | ZeroAmount | Unauthorized {
    if (amount == 0) { return ZeroAmount; }
    // ...
    return true;
}
```

## Returning errors

Return an error by name:

```ora
return ZeroAmount;                              // nullary error
return InsufficientBalance(amount, current);    // error with fields
```

## Try expressions

`try` unwraps a successful value or propagates the error to the caller. Inside a contract:

```ora
pub fn transferAndLog(amount: u256) -> !bool | InsufficientBalance {
    let success: bool = try withdraw(amount);
    // If withdraw returned an error, we never reach here —
    // the error propagates to our caller automatically.
    return success;
}
```

The calling function must include the propagated error type in its own return type.

## Try/catch blocks

For explicit error handling, use `try { } catch`. Inside a contract, assuming `std` is imported:

```ora
pub fn safeWithdraw(amount: u256) -> u256 {
    try {
        let success: bool = try withdraw(amount);
        return balances[std.msg.sender()];
    } catch (err) {
        return 0;
    }
}
```

The `try` block catches any error from `try` expressions inside it. The `catch` block receives the error.

## The errors clause

Functions can declare which errors they produce with an `errors` clause:

```ora
pub fn withdraw(amount: u256) -> !bool
    errors(InsufficientBalance, ZeroAmount)
{
    // ...
}
```

This is equivalent to `-> !bool | InsufficientBalance | ZeroAmount` but separates the error list from the return type for readability.

## The vault with error unions

Here is our vault with proper error handling:

```ora
error InsufficientBalance(required: u256, available: u256);
error ZeroAmount;

comptime const std = @import("std");

contract Vault {
    storage var totalDeposits: u256 = 0;
    storage var balances: map<address, u256>;

    pub fn deposit(amount: u256) -> !bool | ZeroAmount {
        if (amount == 0) { return ZeroAmount; }
        let sender: address = std.msg.sender();
        balances[sender] += amount;
        totalDeposits += amount;
        return true;
    }

    pub fn withdraw(amount: u256) -> !bool | InsufficientBalance {
        let sender: address = std.msg.sender();
        let current: u256 = balances[sender];
        if (current < amount) { return InsufficientBalance(amount, current); }
        balances[sender] = current - amount;
        totalDeposits -= amount;
        return true;
    }

    pub fn balanceOf(account: address) -> u256 {
        return balances[account];
    }

    pub fn getTotalDeposits() -> u256 {
        return totalDeposits;
    }
}
```

Compare this with the Chapter 4 version that returned `bool`. Now:
- `deposit` rejects zero amounts with a `ZeroAmount` error
- `withdraw` rejects insufficient balances with an `InsufficientBalance` error that carries the requested and available amounts
- Callers know at compile time what can go wrong

## Further reading

- [Error Unions and Try](../error-unions) — full reference including encoding details

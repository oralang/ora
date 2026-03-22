---
title: "Chapter 17: Extern Traits"
description: Type-safe cross-contract calls with mandatory gas and reentrancy checking.
sidebar_position: 17
---

# Extern Traits

Smart contracts call other contracts. Ora uses `extern trait` to declare typed interfaces for external calls, enforcing explicit opcodes (`call`/`staticcall`), mandatory gas budgets, and forced error handling at the type level.

## Declaring an extern trait

```ora
extern trait IERC20 {
    staticcall fn totalSupply(self) -> u256;
    staticcall fn balanceOf(self, owner: address) -> u256;
    call fn transfer(self, to: address, amount: u256) -> bool;
    call fn approve(self, spender: address, amount: u256) -> bool;
}
```

- `staticcall fn` — uses STATICCALL. No state changes, no reentrancy risk.
- `call fn` — uses CALL. State changes possible, reentrancy possible.
- The `call`/`staticcall` keyword is mandatory. The auditor sees the opcode.

You cannot `impl` an extern trait — you don't have the external contract's code.

## Calling an external contract

Inside a contract:

```ora
pub fn getTokenBalance(token: address, user: address) -> u256 {
    return try external<IERC20>(token, gas: 30000).balanceOf(user);
}
```

The syntax: `external<TraitName>(contract_address, gas: gas_amount).method(args)`

### Mandatory gas

Every external call requires an explicit `gas:` parameter:

```ora
external<IERC20>(token, gas: 100000).transfer(to, amount)   // OK
external<IERC20>(token).transfer(to, amount)                  // Compile error
```

No implicit gas forwarding. The 63/64 rule is not a substitute for intentional gas budgeting.

### Error union returns

External calls can fail (out of gas, external revert). The return type is always an error union:

```ora
error ExternalCallFailed;

pub fn safeTransfer(token: address, to: address, amount: u256) -> !bool | ExternalCallFailed {
    let success: bool = try external<IERC20>(token, gas: 100000).transfer(to, amount);
    return success;
}
```

### Errors clause on extern methods

Extern trait methods can declare which errors the external contract might return:

```ora
extern trait IERC20 {
    call fn transfer(self, to: address, amount: u256) -> bool
        errors(InsufficientBalance, InvalidRecipient);
}
```

## Reentrancy checking

The compiler tracks state writes around `call fn` invocations. If you write to storage after a `call fn` without a lock, the compiler warns about reentrancy risk:

```ora
pub fn unsafeDeposit(token: address, amount: u256) {
    // External call first
    try external<IERC20>(token, gas: 100000).transferFrom(sender, self, amount);
    // Storage write after external call — compiler warns about reentrancy
    balances[sender] += amount;
}
```

The safe pattern: commit state before the external call, or use `@lock`.

## The vault with extern traits

A vault that accepts ERC20 token deposits:

```ora
extern trait IERC20 {
    staticcall fn balanceOf(self, owner: address) -> u256;
    call fn transferFrom(self, from: address, to: address, amount: u256) -> bool;
}

error ExternalCallFailed;
error InsufficientBalance(required: u256, available: u256);

comptime const std = @import("std");

contract TokenVault {
    storage var token: address;
    storage var self_address: address;
    storage var balances: map<address, u256>;
    storage var reentrancyGuard: u256 = 0;

    pub fn deposit(amount: MinValue<u256, 1>) -> !bool | ExternalCallFailed {
        let sender: NonZeroAddress = std.msg.sender();

        @lock(reentrancyGuard);
        balances[sender] += amount;
        @unlock(reentrancyGuard);

        let ok: bool = try external<IERC20>(token, gas: 100000)
            .transferFrom(sender, self_address, amount);
        return ok;
    }

    pub fn getBalance(user: address) -> u256 {
        return try external<IERC20>(token, gas: 30000).balanceOf(user);
    }
}
```

State is committed (`balances[sender] += amount`) inside the lock, before the external call. This follows the checks-effects-interactions pattern, enforced by the compiler.

## Further reading

- [External Contract Calls](../extern-traits) — full extern trait reference

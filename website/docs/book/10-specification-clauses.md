---
title: "Chapter 10: Specification Clauses"
description: Preconditions, postconditions, and invariants for formal verification.
sidebar_position: 10
---

# Specification Clauses

Ora has built-in support for formal verification. You write `requires` and `ensures` clauses on functions, and the compiler sends them to a Z3 SMT solver to check whether the code satisfies them.

## Preconditions: requires

`requires` states what must be true when a function is called. Inside a contract, assuming `std` is imported:

```ora
pub fn withdraw(amount: u256)
    requires(amount > 0)
    requires(balances[std.msg.sender()] >= amount)
{
    // The compiler guarantees both conditions hold here
}
```

Multiple `requires` clauses are conjunctive — all must hold. If a caller violates a precondition, the compiler inserts a runtime guard or reports a verification error.

## Postconditions: ensures

`ensures` states what the function guarantees when it returns. Inside a contract:

```ora
pub fn deposit(amount: u256)
    requires(amount > 0)
    ensures(totalDeposits == old(totalDeposits) + amount)
{
    totalDeposits += amount;
}
```

The compiler checks that every code path satisfies the `ensures` clause. If it can't prove it, you get a verification error with a counterexample.

## The old() expression

`old(expr)` refers to the value of `expr` at function entry — before any mutations. Inside a contract, assuming `std` is imported:

```ora
pub fn deposit(amount: u256)
    ensures(totalDeposits == old(totalDeposits) + amount)
    ensures(balances[std.msg.sender()] == old(balances[std.msg.sender()]) + amount)
{
    let sender: address = std.msg.sender();
    balances[sender] += amount;
    totalDeposits += amount;
}
```

`old(totalDeposits)` is the value of `totalDeposits` before `deposit` executed. This lets you express relational postconditions — "the new value equals the old value plus the deposit."

## Assert and assume

- `assert(condition)` — checked at both runtime and verification time. If the condition can be false, the verifier reports it.
- `assume(condition)` — taken as true for verification only. No runtime check. Use for conditions the verifier can't derive but you know hold.

Inside a contract:

```ora
pub fn divide(a: u256, b: u256) -> u256 {
    assert(b != 0);    // runtime check + verification obligation
    return a / b;
}
```

## Loop invariants

Loops can carry `invariant` clauses. The verifier checks three things:
1. The invariant holds on loop entry
2. If the invariant holds before an iteration, it holds after (inductive step)
3. The invariant plus the loop exit condition implies the postcondition

Inside a contract:

```ora
pub fn sum(n: u256) -> u256
    ensures(result == n * (n + 1) / 2)
{
    var total: u256 = 0;
    var i: u256 = 0;
    while (i <= n)
        invariant(total == i * (i + 1) / 2)
    {
        total += i;
        i += 1;
    }
    return total;
}
```

## The vault with specifications

```ora
error InsufficientBalance(required: u256, available: u256);

comptime const std = @import("std");

contract Vault {
    storage var totalDeposits: u256 = 0;
    storage var balances: map<address, u256>;

    log Deposit(account: address, amount: u256);
    log Withdrawal(account: address, amount: u256);

    pub fn deposit(amount: MinValue<u256, 1>)
        requires(amount > 0)
        requires(totalDeposits <= std.constants.U256_MAX - amount)
        requires(balances[std.msg.sender()] <= std.constants.U256_MAX - amount)
        ensures(totalDeposits == old(totalDeposits) + amount)
        ensures(balances[std.msg.sender()] == old(balances[std.msg.sender()]) + amount)
    {
        let sender: NonZeroAddress = std.msg.sender();
        balances[sender] += amount;
        totalDeposits += amount;
        log Deposit(sender, amount);
    }

    pub fn withdraw(amount: MinValue<u256, 1>) -> !bool | InsufficientBalance
        requires(amount > 0)
        requires(balances[std.msg.sender()] >= amount)
        ensures(totalDeposits == old(totalDeposits) - amount)
        ensures(balances[std.msg.sender()] == old(balances[std.msg.sender()]) - amount)
    {
        let sender: NonZeroAddress = std.msg.sender();
        let current: u256 = balances[sender];
        if (current < amount) { return InsufficientBalance(amount, current); }
        balances[sender] = current - amount;
        totalDeposits -= amount;
        log Withdrawal(sender, amount);
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

The `requires` clauses on `deposit` prevent overflow: `totalDeposits + amount` must not exceed `U256_MAX`. The `ensures` clauses guarantee that the accounting is correct: after deposit, the balance increased by exactly `amount`.

The Z3 solver checks these automatically during compilation.

## Further reading

- [Formal Verification](../formal-verification) — verification model and pipeline

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
    requires amount > 0
    requires balances[std.msg.sender()] >= amount
{
    // The compiler guarantees both conditions hold here
}
```

Multiple `requires` clauses are conjunctive — all must hold. If a caller violates a precondition, the compiler inserts a runtime guard or reports a verification error.

## Postconditions: ensures

`ensures` states what the function guarantees when it returns. Inside a contract:

```ora
pub fn deposit(amount: u256)
    requires amount > 0
    ensures totalDeposits == old(totalDeposits) + amount
{
    totalDeposits += amount;
}
```

The compiler checks that every code path satisfies the `ensures` clause. If it can't prove it, you get a verification error with a counterexample.

## The old() expression

`old(expr)` refers to the value of `expr` at function entry — before any mutations. Inside a contract, assuming `std` is imported:

```ora
pub fn deposit(amount: u256)
    ensures totalDeposits == old(totalDeposits) + amount
    ensures balances[std.msg.sender()] == old(balances[std.msg.sender()]) + amount
{
    let sender: address = std.msg.sender();
    balances[sender] += amount;
    totalDeposits += amount;
}
```

`old(totalDeposits)` is the value of `totalDeposits` before `deposit` executed. This lets you express relational postconditions — "the new value equals the old value plus the deposit."

## Guards: runtime checks with verification

`guard` is a specification clause that generates a runtime check. It's the `requires` in spirit — but instead of trusting the caller to satisfy the condition, the function checks it at runtime and reverts if it fails.

```ora
pub fn transfer(to: address, amount: u256) -> bool
    guard amount > 0
    guard to != std.constants.ZERO_ADDRESS
{
    // Both conditions are checked at runtime (revert if false)
    // AND visible to the SMT solver (so downstream code can rely on them)
}
```

Without `guard`, you'd write the same logic manually:

```ora
pub fn transfer(to: address, amount: u256) -> bool {
    if (amount == 0) { return false; }
    if (to == std.constants.ZERO_ADDRESS) { return false; }
    // ...
}
```

`guard` writes both the check and the specification in one place. The compiler:
1. Emits a runtime check (reverts if the condition is false)
2. Tells the SMT solver the condition holds after the check — so the verifier can use it to prove downstream obligations like overflow safety or postconditions

### guard vs requires

| Clause | Who is responsible | Runtime behavior | SMT role |
|---|---|---|---|
| `requires(x > 0)` | **Caller** — must prove the condition before calling | No runtime check (proven statically, or caller gets verification error) | Assumption — the verifier assumes it holds |
| `guard(x > 0)` | **Function** — checks the condition itself | Runtime revert if false | Assumption after the check — the verifier knows it holds for everything after |

Use `requires` when the caller should guarantee the condition. Use `guard` when the function should enforce it.

### guard with requires and ensures

Guards compose with other spec clauses:

```ora
pub fn withdraw(amount: u256)
    requires amount <= balance              // caller guarantees no underflow
    guard amount > 0                       // function checks at runtime
    ensures balance == old(balance) - amount    // postcondition
{
    balance -= amount;
}
```

The verifier checks the `ensures` clause under the combined context of both the `requires` assumption and the `guard` assumption.

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
    ensures result == n * (n + 1) / 2
{
    var total: u256 = 0;
    var i: u256 = 0;
    while (i <= n)
        invariant total == i * (i + 1) / 2
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
        requires amount > 0
        requires totalDeposits <= std.constants.U256_MAX - amount
        requires balances[std.msg.sender()] <= std.constants.U256_MAX - amount
        ensures totalDeposits == old(totalDeposits) + amount
        ensures balances[std.msg.sender()] == old(balances[std.msg.sender()]) + amount
    {
        let sender: NonZeroAddress = std.msg.sender();
        balances[sender] += amount;
        totalDeposits += amount;
        log Deposit(sender, amount);
    }

    pub fn withdraw(amount: MinValue<u256, 1>) -> !bool | InsufficientBalance
        requires amount > 0
        requires balances[std.msg.sender()] >= amount
        ensures totalDeposits == old(totalDeposits) - amount
        ensures balances[std.msg.sender()] == old(balances[std.msg.sender()]) - amount
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

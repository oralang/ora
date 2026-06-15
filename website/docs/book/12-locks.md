---
title: "Chapter 12: Locks and Reentrancy"
description: Protecting against reentrancy with @lock and @unlock.
sidebar_position: 12
---

# Locks and Reentrancy

Reentrancy is the most exploited vulnerability in smart contract history. It occurs when an external call re-enters the contract before state updates complete.

Ora provides `@lock` and `@unlock` directives for transaction-scoped storage-path locking. A lock does not change the value stored at that path. It records that the path is locked for the current transaction, and writes to that locked path are rejected until it is unlocked or the transaction reverts.

## The lock pattern

```ora
contract Balances {
    storage var balances: map<address, u256>;

    pub fn read_then_write(user: address, amount: u256) {
        @lock(balances[user]);
        let current: u256 = balances[user];
        @unlock(balances[user]);

        balances[user] = current + amount;
    }
}
```

- `@lock(path)` locks the storage path identified by `path`.
- `@unlock(path)` releases that storage-path lock.
- The lock is on the storage path, not on a boolean value. `@lock(x)` does not set `x`, and `@unlock(x)` does not clear it.
- Writes to the same locked path are rejected. Writes to other storage paths may still be valid.
- If the current function needs to update that same path, update it before the lock or after the unlock.

## Lock discipline

The compiler tracks the active lock set while checking the function body:

- Direct writes to the same storage path while it is locked are rejected
- Runtime guards protect lock-participating paths across calls
- The lock is transaction-scoped and is released if the transaction reverts

## Lock the state you changed

For reentrancy-sensitive flows, commit the state update first, then lock the storage path that must not be rewritten during the external-call window.

```ora
extern trait ReentryHook {
    call fn onCall(self) -> bool;
}

comptime const std = @import("std");

contract ReentrancyVictim {
    storage var balances: map<address, u256>;

    pub fn deposit_and_call(hook: address, amount: u256) -> bool {
        let sender: NonZeroAddress = std.msg.sender();
        balances[sender] += amount;
        @lock(balances[sender]);
        let hook_result = external<ReentryHook>(hook, gas: 200000).onCall();
        @unlock(balances[sender]);

        return match (hook_result) {
            Ok(ok) => ok,
            Err(_) => false,
        };
    }
}
```

If the hook re-enters code that tries to write `balances[sender]`, the write targets a locked storage path and reverts. The lock does not change the balance value; it prevents the locked path from being rewritten while the lock is active.

Executable path-locking examples live in `ora-example/locks/` and `tests/conformance/lock_guard_revert.ora`.

## Further reading

- [Formal Verification](../formal-verification) — how lock analysis integrates with verification

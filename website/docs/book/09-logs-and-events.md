---
title: "Chapter 9: Logs and Events"
description: Declaring and emitting events for off-chain indexing.
sidebar_position: 9
---

# Logs and Events

Smart contracts communicate with the outside world through events. Ora calls them `log` declarations — they map directly to EVM LOG opcodes.

## Declaring logs

```ora
contract Token {
    log Transfer(sender: address, recipient: address, amount: u256);
    log Approval(owner: address, spender: address, amount: u256);
}
```

Log declarations list their fields with names and types. They're declared inside the contract body.

## Indexed fields

Mark fields as `indexed` for efficient off-chain filtering:

```ora
log Transfer(indexed sender: address, indexed recipient: address, amount: u256);
```

Indexed fields become LOG topics. The EVM supports up to 3 indexed fields per event (plus the event signature topic). Non-indexed fields are ABI-encoded in the LOG data.

## Emitting logs

Emit a log by calling it like a function. Inside a contract with `log Transfer` declared and `std` imported:

```ora
pub fn transfer(to: address, amount: u256) {
    let sender: address = std.msg.sender();
    // ... transfer logic ...
    log Transfer(sender, to, amount);
}
```

The arguments must match the log declaration's field types in order.

## Logs with various types

Logs can carry any supported type:

```ora
log ConfigChanged(key: string, old_value: u256, new_value: u256, changed_by: address);
log FlagSet(id: u256, active: bool);
```

## The vault with events

Our vault already has logs from the refinement types chapter:

```ora
log Deposit(account: address, amount: u256);
log Withdrawal(account: address, amount: u256);

pub fn deposit(amount: MinValue<u256, 1>) {
    let sender: NonZeroAddress = std.msg.sender();
    balances[sender] += amount;
    totalDeposits += amount;
    log Deposit(sender, amount);
}
```

Events let off-chain systems (indexers, frontends, analytics) track vault activity without polling storage.

## Further reading

- [Logs and Events](../logs-and-events) — full log reference

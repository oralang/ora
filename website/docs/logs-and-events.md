---
title: Logs and Events
description: Event-style logs for observability and indexable queries.
---

# Logs and Events

Logs are declared at contract scope and emitted with the `log` statement.

## Log declarations

```ora
contract Ledger {
    log Transfer(indexed from: address, indexed to: address, amount: u256);
    log Paused();
}
```

## Emitting logs

```ora
contract Ledger {
    log Transfer(indexed from: address, indexed to: address, amount: u256);

    pub fn send(to: NonZeroAddress, amount: u256) {
        let from: address = std.msg.sender();
        log Transfer(from, to, amount);
    }
}
```

## Indexed fields

Use `indexed` to mark fields for efficient filtering in downstream tooling.

```ora
log Approval(indexed owner: address, indexed spender: address, amount: u256);
```

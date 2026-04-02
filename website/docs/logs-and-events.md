---
title: Logs and Events
description: Event-style logs with indexed fields, type checking, and EVM LOG opcode lowering.
---

# Logs and Events

Logs are Ora's equivalent of Solidity events. They emit EVM log entries for off-chain indexing and observability.

## Declaring logs

Logs are declared at contract scope with typed fields:

```ora
contract Ledger {
    log Transfer(indexed from: address, indexed to: address, amount: u256);
    log Approval(indexed owner: address, indexed spender: address, amount: u256);
    log StatusChange(account: address, old_status: u8, new_status: u8);
    log Paused();
}
```

## Emitting logs

Use the `log` statement with positional arguments matching the declaration:

```ora
contract Ledger {
    log Transfer(indexed from: address, indexed to: address, amount: u256);

    pub fn send(to: address, amount: u256) {
        let from: address = std.msg.sender();
        log Transfer(from, to, amount);
    }
}
```

The compiler checks:
- The log name is declared in the current contract
- Argument count matches the declaration
- Argument types match the declared field types

## Indexed fields

Mark fields with `indexed` to make them filterable by off-chain indexers:

```ora
log Transfer(indexed from: address, indexed to: address, amount: u256);
```

### Limits

A log can have at most **3 indexed fields**. This is an EVM constraint — the `LOG` opcodes support up to 4 topics, and topic 0 is reserved for the event signature hash. Four or more indexed fields is a compile error:

```ora
// Compile error: too many indexed fields
log TooMany(indexed a: address, indexed b: address, indexed c: address, indexed d: address);
```

### Indexed field types

Indexed fields must be scalar types. The compiler rejects these types for indexed fields:

- `struct`
- `tuple`
- `array`
- `slice`
- `map`

Scalar types (`bool`, integers, `address`, `bytes`) are supported.

## EVM lowering

Logs compile to EVM `LOG0`–`LOG4` opcodes:

| Indexed fields | EVM opcode | Topics |
|---------------|------------|--------|
| 0 | `LOG1` | signature only |
| 1 | `LOG2` | signature + 1 indexed |
| 2 | `LOG3` | signature + 2 indexed |
| 3 | `LOG4` | signature + 3 indexed |

Topic 0 is the `keccak256` hash of the log signature (e.g., `keccak256("Transfer(address,address,uint256)")`), computed at compile time. Non-indexed fields are ABI-encoded into the log data payload.

## Supported field types

```ora
contract Events {
    log MixedEvent(id: u256, name: string, active: bool, data: bytes);
}
```

Fields can be any type the ABI encoder supports: integers, `bool`, `address`, `string`, `bytes`.

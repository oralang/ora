---
sidebar_position: 7
---

# Standard Library

Reference for Ora's built-in standard library (`std.*`). These built-ins expose
low-level EVM primitives with explicit behavior and strong typing.

## Overview

- The standard library is minimal and intentionally low-level.
- APIs may evolve as the compiler and backend stabilize.
- Examples may omit types for readability; the current compiler may require
  explicit type annotations for locals.

## Philosophy

1. Map closely to EVM semantics
2. Prefer explicitness over abstraction
3. Keep built-ins auditable and predictable
4. Avoid hidden runtime effects

---

## Block Data

Access current block information.

### `std.block.timestamp()`

Get the current block timestamp (seconds since Unix epoch).

```ora
pub fn getTimestamp() -> u256 {
    return std.block.timestamp();
}
```

**Returns**: `u256` - Current block timestamp  
**EVM Opcode**: `TIMESTAMP`  
**Gas Cost**: 2 gas (base cost)

**Example Use Case**:
```ora
// Check if deadline has passed
pub fn hasExpired(deadline: u256) -> bool {
    return std.block.timestamp() > deadline;
}
```

---

### `std.block.number()`

Get the current block number.

```ora
pub fn getBlockNumber() -> u256 {
    return std.block.number();
}
```

**Returns**: `u256` - Current block number  
**EVM Opcode**: `NUMBER`  
**Gas Cost**: 2 gas

**Example Use Case**:
```ora
// Check if enough blocks have passed
pub fn canExecute(lastBlock: u256) -> bool {
    return std.block.number() >= lastBlock + 100;
}
```

---

### `std.block.gaslimit()`

Get the current block's gas limit.

```ora
pub fn getGasLimit() -> u256 {
    return std.block.gaslimit();
}
```

**Returns**: `u256` - Block gas limit  
**EVM Opcode**: `GASLIMIT`  
**Gas Cost**: 2 gas

---

### `std.block.coinbase()`

Get the current block's miner address.

```ora
pub fn getMiner() -> address {
    return std.block.coinbase();
}
```

**Returns**: `address` - Miner's address  
**EVM Opcode**: `COINBASE`  
**Gas Cost**: 2 gas

**Example Use Case**:
```ora
// Reward the miner
pub fn rewardMiner(amount: u256) {
    let miner = std.block.coinbase();
    balances[miner] = balances[miner] + amount;
}
```

---

### `std.block.basefee()`

Get the current block's base fee (EIP-1559).

```ora
pub fn getBaseFee() -> u256 {
    return std.block.basefee();
}
```

**Returns**: `u256` - Base fee in wei  
**EVM Opcode**: `BASEFEE`  
**Gas Cost**: 2 gas  
**Availability**: Post-London hard fork

---

## Transaction Data

Access transaction-level information.

### `std.transaction.sender()`

Get the original transaction sender (origin).

```ora
pub fn getOrigin() -> address {
    return std.transaction.sender();
}
```

**Returns**: `address` - Transaction origin  
**EVM Opcode**: `ORIGIN`  
**Gas Cost**: 2 gas

⚠️ **Security Note**: Be careful with `std.transaction.sender()` - it returns the original EOA, not the immediate caller. For access control, usually prefer `std.msg.sender()`.

**Example Use Case**:
```ora
// Track which EOAs have interacted
storage visitedAddresses: map[address, bool];

pub fn recordVisit() {
    let origin = std.transaction.sender();
    visitedAddresses[origin] = true;
}
```

---

### `std.transaction.gasprice()`

Get the gas price for this transaction.

```ora
pub fn getGasPrice() -> u256 {
    return std.transaction.gasprice();
}
```

**Returns**: `u256` - Gas price in wei  
**EVM Opcode**: `GASPRICE`  
**Gas Cost**: 2 gas

---

## Message Data

Access call-level information.

### `std.msg.sender()`

Get the immediate caller's address.

```ora
pub fn recordCaller() -> address {
    return std.msg.sender();
}
```

**Returns**: `address` - Caller's address  
**EVM Opcode**: `CALLER`  
**Gas Cost**: 2 gas

✅ **Best Practice**: Use `std.msg.sender()` for access control and authentication.

**Example Use Case**:
```ora
storage owner: address;

pub fn onlyOwner() -> bool {
    let caller = std.msg.sender();
    if (caller != owner) {
        return false;
    }
    return true;
}
```

---

### `std.msg.value()`

Get the amount of wei sent with this call.

```ora
pub fn getValue() -> u256 {
    return std.msg.value();
}
```

**Returns**: `u256` - Wei sent with call  
**EVM Opcode**: `CALLVALUE`  
**Gas Cost**: 2 gas

**Example Use Case**:
```ora
pub fn deposit() -> bool {
    let caller = std.msg.sender();
    let amount = std.msg.value();
    
    balances[caller] = balances[caller] + amount;
    return true;
}
```

---

### `std.msg.data.size()`

Get the size of calldata in bytes.

```ora
pub fn getCalldataSize() -> u256 {
    return std.msg.data.size();
}
```

**Returns**: `u256` - Calldata size  
**EVM Opcode**: `CALLDATASIZE`  
**Gas Cost**: 2 gas

---

## Constants

Pre-defined constant values.

### `std.constants.ZERO_ADDRESS`

The zero address (`0x0000000000000000000000000000000000000000`).

```ora
pub fn isZeroAddress(addr: address) -> bool {
    return addr == std.constants.ZERO_ADDRESS;
}
```

**Type**: `address`  
**Value**: `0x0000000000000000000000000000000000000000`  
**Compilation**: Inlined as `arith.constant 0 : i160`

**Example Use Case**:
```ora
pub fn transfer(to: address, amount: u256) -> bool {
    // Reject transfers to zero address
    if (to == std.constants.ZERO_ADDRESS) {
        return false;
    }
    
    // Transfer logic...
    return true;
}
```

---

### `std.constants.U256_MAX`

Maximum value for a u256.

```ora
pub fn getMaxSupply() -> u256 {
    return std.constants.U256_MAX;
}
```

**Type**: `u256`  
**Value**: `115792089237316195423570985008687907853269984665640564039457584007913129639935`  
**Compilation**: Inlined as `arith.constant -1 : i256` (all bits set)

---

### `std.constants.U128_MAX`

Maximum value for a u128.

```ora
pub fn checkOverflow(value: u256) -> bool {
    return value > std.constants.U128_MAX;
}
```

**Type**: `u128`  
**Value**: `340282366920938463463374607431768211455`

---

### `std.constants.U64_MAX`

Maximum value for a u64.

```ora
pub fn withinU64Range(value: u256) -> bool {
    return value <= std.constants.U64_MAX;
}
```

**Type**: `u64`  
**Value**: `18446744073709551615`

---

### `std.constants.U32_MAX`

Maximum value for a u32.

```ora
pub fn withinU32Range(value: u256) -> bool {
    return value <= std.constants.U32_MAX;
}
```

**Type**: `u32`  
**Value**: `4294967295`

---

## Compilation and Performance

### Zero-Overhead

Standard library built-ins compile directly to EVM opcodes:

```ora
// Ora code
let timestamp = std.block.timestamp();
```

```sir
// Generated Sensei-IR (SIR)
fn main:
  entry -> result {
    result = timestamp
    iret
  }
```

The built-in is replaced with the raw EVM opcode at compile time - no function call overhead.

### Type Safety

Built-ins are type-checked at compile time:

```ora
// ✅ Correct
let time: u256 = std.block.timestamp();

// ❌ Compile error: type mismatch
let time: address = std.block.timestamp();
```

### Validation

The compiler validates built-in usage:

```ora
// ❌ Compile error: unknown built-in
let invalid = std.block.nonexistent();

// ❌ Compile error: missing ()
let sender = std.msg.sender;
```

---

## Complete Example: ERC20 Token

Here's a token contract using the standard library:

```ora
contract SimpleToken {
    storage totalSupply: u256;
    storage balances: map[address, u256];
    storage allowances: doublemap[address, address, u256];
    
    pub fn initialize(initialSupply: u256) -> bool {
        let deployer = std.msg.sender();
        totalSupply = initialSupply;
        balances[deployer] = initialSupply;
        return true;
    }
    
    pub fn transfer(recipient: address, amount: u256) -> bool {
        let sender = std.msg.sender();
        let senderBalance = balances[sender];
        
        // Validate recipient
        if (recipient == std.constants.ZERO_ADDRESS) {
            return false;
        }
        
        // Check balance
        if (senderBalance < amount) {
            return false;
        }
        
        // Perform transfer
        balances[sender] = senderBalance - amount;
        let recipientBalance = balances[recipient];
        balances[recipient] = recipientBalance + amount;
        
        return true;
    }
    
    pub fn approve(spender: address, amount: u256) -> bool {
        let owner = std.msg.sender();
        
        if (spender == std.constants.ZERO_ADDRESS) {
            return false;
        }
        
        allowances[owner][spender] = amount;
        return true;
    }
}
```

**Key Usage**:
- `std.msg.sender()` - Get the caller's address
- `std.constants.ZERO_ADDRESS` - Validate addresses

---

## How It Works

The standard library compiles in three stages:

```
Ora Source          →  MLIR IR          →  Sensei-IR (SIR)  →  EVM Bytecode
std.msg.sender()       ora.evm.caller()    caller operation    CALLER (0x33)
```

Each built-in maps to an EVM opcode or a small, direct sequence. The exact
lowering depends on the backend stage and target.

---

## Best Practices

### ✅ DO

- Use `std.msg.sender()` for access control
- Check for `std.constants.ZERO_ADDRESS` on address inputs
- Use `std.block.timestamp()` for time-based logic (with care)

### ❌ DON'T

- Don't use `std.transaction.sender()` for access control (use `std.msg.sender()`)
- Don't rely on `std.block.timestamp()` for critical randomness
- Don't assume `std.block.number()` increments by exactly 1

### Security Considerations

**Timestamp Manipulation**: Miners can manipulate `std.block.timestamp()` by ~15 seconds. Don't use it for critical randomness.

**Block Number**: Can be used for rough time estimates (1 block ≈ 12 seconds), but not precise.

**msg.sender vs transaction.sender**:
- `std.msg.sender()` = immediate caller (could be a contract)
- `std.transaction.sender()` = original EOA (can't be a contract)

Use `std.msg.sender()` for most access control!

---

## FAQ

### Q: Do I need to import the standard library?

**A**: No. The standard library is always available - just use `std.` prefix.

### Q: What's the gas cost?

**A**: Gas cost depends on the opcode and context. Built-ins aim to
lower directly without user-visible runtime wrappers.

### Q: Can I redefine `std`?

**A**: No. `std` is a reserved namespace.

### Q: Why so minimal?

**A**: Ora provides direct EVM access without abstraction. Higher-level utilities should be implemented as user libraries, not built-ins.

---

## Related Documentation

- [Language Basics](language-basics.md) - Core language features
- [MLIR Integration](specifications/mlir.md) - Compiler internals
- [Examples](examples.md) - More example contracts

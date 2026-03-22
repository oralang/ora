---
title: External Contract Calls
description: Type-safe cross-contract calls with extern traits, mandatory gas budgets, and re-entrancy checking.
---

# External Contract Calls

Ora uses `extern trait` to declare typed interfaces for calling external contracts. Unlike Solidity's `interface`, extern traits enforce:

- Explicit `call` or `staticcall` per method — the auditor sees the EVM opcode
- Mandatory gas budgets — no implicit gas forwarding
- Forced error handling — external calls always return error unions
- Re-entrancy checking — the compiler catches same-slot writes around effectful calls

## Declaring an Extern Trait

```ora
extern trait ERC20 {
    staticcall fn totalSupply(self) -> u256;
    staticcall fn balanceOf(self, owner: address) -> u256;
    call fn transfer(self, to: address, amount: u256) -> bool;
    call fn approve(self, spender: address, amount: u256) -> bool;
}
```

- `staticcall fn` — uses the `STATICCALL` opcode. No state changes possible, no re-entrancy risk.
- `call fn` — uses the `CALL` opcode. State changes possible, re-entrancy possible.
- The `call`/`staticcall` keyword is **mandatory**. The compiler rejects extern trait methods without it.

You cannot `impl` an extern trait — you don't have the external contract's code. You cannot add `ghost` specs — you can't verify code you don't own.

## Calling an External Contract

```ora
error ExternalCallFailed;

contract Vault {
    storage var token: address;
    storage var balances: map<address, u256>;

    pub fn deposit(amount: u256) {
        let sender: address = std.msg.sender();

        // Commit your state BEFORE the external call
        balances[sender] += amount;

        // External call — explicit gas, forced error handling
        let ok = try external<ERC20>(token, gas: 100000)
            .transferFrom(sender, std.address.self(), amount);
        if (!ok) revert;
    }

    pub fn getBalance(user: address) -> u256 {
        // staticcall — no re-entrancy risk, no lock needed
        return try external<ERC20>(token, gas: 30000).balanceOf(user);
    }
}
```

### Mandatory Gas

Every external call requires an explicit `gas:` parameter. The compiler rejects calls without it:

```ora
external<ERC20>(token).transfer(to, amount)              // Compile error: requires gas
external<ERC20>(token, gas: 100000).transfer(to, amount)  // OK
```

There is no default gas forwarding. The 63/64 rule (EIP-150) is not a substitute for intentional gas budgeting.

### Error Union Returns

External calls always return `!ReturnType | ExternalCallFailed`. The call itself can revert (out of gas, external revert), which is separate from the method's return value:

```ora
// The external contract returns bool, but the CALL can fail
let result = external<ERC20>(token, gas: 100000).transfer(to, amount);
// result is !bool | ExternalCallFailed

// Use try to propagate failure:
let ok = try external<ERC20>(token, gas: 100000).transfer(to, amount);
// ok is bool — call succeeded
```

## Declared Error Matching

Extern trait methods can declare which errors the external contract may produce:

```ora
error InsufficientBalance(required: u256, available: u256);

extern trait ERC20 {
    call fn transfer(self, to: address, amount: u256) -> bool
        errors(InsufficientBalance);
}
```

When the call reverts, Ora checks the returndata:
1. If the 4-byte selector matches a declared error → decode the payload → typed error
2. If no match → `ExternalCallFailed`

`ExternalCallFailed` is always in the error set — it covers out-of-gas, panics, and unmatched reverts. External contracts are black boxes; you can never exhaustively handle all their failure modes.

## Re-entrancy Checking

When a `call` method is invoked, the compiler tracks which storage slots were written before the call. If the **same slot** is written after the call, it's a compile error:

```ora
// Compile error: 'balances' written before AND after external call
balances[sender] -= amount;
let ok = try external<ERC20>(token, gas: 100000).transfer(to, amount);
balances[sender] += amount;  // re-entrancy risk on 'balances'
```

Writing to **different slots** after the call is allowed:

```ora
// OK: 'failedCount' not written before the call
let ok = try external<ERC20>(token, gas: 100000).transfer(to, amount);
if (!ok) { failedCount += 1; }
```

`staticcall` methods don't trigger this check — `STATICCALL` prevents re-entrancy by construction.

The check is **branch-sensitive**: if a slot is written in one branch but not another before the call, the post-call write is allowed (the slot wasn't definitely written on all paths).

## ABI Encoding

Ora computes ABI selectors at compile time using `@keccak256`. The encoding follows Solidity's ABI specification:
- Function selectors: `keccak256("transfer(address,uint256)")[:4]`
- Arguments: 32-byte ABI-encoded, left-padded
- Return values: ABI-decoded from returndata

The compiler handles this automatically — you write Ora types, and the compiler encodes/decodes them as standard Solidity ABI at the boundary.

## Comparison with Solidity

| | Solidity | Ora |
|---|---|---|
| Call type visibility | Hidden — must check source for `view` | Explicit `call`/`staticcall` keyword |
| Gas forwarding | Implicit (63/64 rule) | Mandatory explicit `gas:` parameter |
| Failure handling | Silent — `success` bit ignored by default | Forced — returns `!ReturnType \| ExternalCallFailed` |
| Re-entrancy protection | Convention (checks-effects-interactions) | Compiler-enforced same-slot write checking |
| Error decoding | Manual `abi.decode` in `catch` | Automatic via `errors(...)` clause |
| `delegatecall` | Available | Not exposed — breaks region system |

## What's Not Supported

- **`delegatecall`** — not exposed through extern traits. Delegate calls run external code in your storage context, which fundamentally breaks region tracking. If proxy patterns are needed, that's a separate unsafe feature.
- **Trait objects / dynamic proxy** — extern traits are resolved at compile time. There is no `dyn ExternTrait`.

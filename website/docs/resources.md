---
title: Resources
description: Conservation-aware resource domains, resource places, and the @amount, @create, @move, and @destroy builtins.
---

# Resources

Ora resources provide a conservation-aware way to represent balances, shares,
credits, allowances, debt, and other quantities stored at named places.

A resource system has two distinct types:

- A **resource domain** names a quantity, such as `TokenUnit`.
- A **resource place** stores that quantity, such as `balances[alice]`.

Resource places have a closed operation set. They can be observed with
`@amount` and changed only with `@create`, `@move`, or `@destroy`.

```ora
comptime const std = @import("std");

resource TokenUnit = u256;

contract Token {
    storage var balances: map<address, Resource<TokenUnit>>;
}
```

`TokenUnit` is a nominal quantity type backed by `u256`.
`Resource<TokenUnit>` is a storage or transient-storage place containing a
`TokenUnit` quantity.

## Operation summary

| Operation | Meaning | State effect |
|-----------|---------|--------------|
| `@amount(place)` | Read the quantity at a resource place | None |
| `@create(place, amount)` | Introduce quantity into a domain | `+amount` |
| `@move(from, to, amount)` | Relocate existing quantity | Conserves the touched pair |
| `@destroy(place, amount)` | Remove quantity from a domain | `-amount` |

There is deliberately no general update operation. Assignment to a resource
place would hide whether quantity was transferred, created, or destroyed.

## Resource domains

A declaration creates a nominal domain over an integer carrier:

```ora
resource TokenUnit = u256;
resource AllowanceUnit = u256;
resource DebtUnit = i256;
```

Domains remain distinct even when they use the same carrier. `TokenUnit` and
`AllowanceUnit` are not interchangeable merely because both erase to `u256`.

```ora
storage var balances: map<address, Resource<TokenUnit>>;
storage var allowances: map<address, Resource<AllowanceUnit>>;

// Compile error: the source and destination belong to different domains.
// @move(balances[user], allowances[user], amount);
```

At runtime, a domain quantity uses its carrier representation. Domain identity
is retained by the compiler for type checking, effects, verification, and audit
output. It does not add another storage word.

## Resource places

Resource places can be persistent or transient and may appear as scalar state,
map entries, nested map entries, or supported struct fields.

```ora
storage var reserve: Resource<TokenUnit>;
storage var balances: map<address, Resource<TokenUnit>>;
tstore var temporary_credit: Resource<TokenUnit>;
```

A resource place is not a first-class value. It cannot be copied into a local,
passed as a function parameter, returned, logged, exposed through the ABI, or
created with a cast. The quantity domain itself can be used as an ordinary
value and is ABI-encoded through its carrier.

Direct mutation is rejected:

```ora
// Compile errors:
// balances[user] = amount;
// balances[user] += amount;
// let place: Resource<TokenUnit> = balances[user];
```

This restriction is the accounting boundary: every change to a resource place
is classified as a transfer, creation, or destruction.

## Reading with `@amount`

`@amount(place)` returns the domain quantity currently stored at a place.
Reading is explicit so an auditor can distinguish observation from mutation.

```ora
pub fn balanceOf(owner: address) -> TokenUnit {
    return @amount(balances[owner]);
}
```

Resource places do not implicitly convert to their quantity:

```ora
// Compile error: a resource place is not a value.
// let balance: TokenUnit = balances[owner];

let balance: TokenUnit = @amount(balances[owner]);
```

`@amount` is pure and can be used in `requires`, `ensures`, and invariants. Use
`old(@amount(place))` to refer to the entry-state quantity.

```ora
pub fn issue(to: address, amount: TokenUnit)
    modifies balances[to]
    requires @amount(balances[to]) <=
        @cast(TokenUnit, std.constants.U256_MAX) - amount
    ensures @amount(balances[to]) ==
        old(@amount(balances[to])) + amount
{
    @create(balances[to], amount);
}
```

## Creating quantity

`@create(place, amount)` adds quantity to a resource place and records an
explicit positive domain delta.

```ora
@create(balances[to], amount);
```

For an unsigned carrier, the destination must have enough remaining range:

```ora
requires @amount(balances[to]) <=
    @cast(TokenUnit, std.constants.U256_MAX) - amount
```

For a signed carrier, the amount must be non-negative and the addition must
stay within the carrier range.

`@create` does not grant minting authority. Access control, supply caps, and
protocol policy remain ordinary Ora conditions that the contract must state.

## Moving quantity

`@move(from, to, amount)` performs one checked resource transition:

```ora
pub fn transfer(from: address, to: address, amount: TokenUnit)
    modifies balances[from], balances[to]
    requires @amount(balances[from]) >= amount
    requires @amount(balances[to]) <=
        @cast(TokenUnit, std.constants.U256_MAX) - amount
{
    @move(balances[from], balances[to], amount);
}
```

On successful execution with distinct places:

```text
new(from) = old(from) - amount
new(to)   = old(to) + amount
```

The sum of the two touched quantities is unchanged. `@move` does not create or
destroy quantity.

For unsigned carriers, the source must contain at least `amount`, and the
destination must not overflow. For signed carriers, `amount` must be
non-negative and both arithmetic results must remain inside the signed carrier
range. Signed resource balances may themselves be negative.

### Aliasing and self-moves

Dynamic places may resolve to the same storage location:

```ora
@move(balances[from], balances[to], amount); // `from` may equal `to`
```

The compiler compares comparable dynamic identities and takes an identity path
when both operands name the same place. A successful self-move leaves the state
unchanged.

A self-move is not an unchecked no-op. The source-side validity check still
runs. For example, an unsigned `@move(p, p, amount)` fails when `p` contains
less than `amount`. The destination overflow check is unnecessary on the
identity path because no addition is committed.

## Destroying quantity

`@destroy(place, amount)` removes quantity from a resource domain:

```ora
pub fn burn(owner: address, amount: TokenUnit)
    modifies balances[owner]
    requires @amount(balances[owner]) >= amount
    ensures @amount(balances[owner]) ==
        old(@amount(balances[owner])) - amount
{
    @destroy(balances[owner], amount);
}
```

For an unsigned carrier, the place must contain at least `amount`. For a signed
carrier, the amount must be non-negative and subtraction must remain within the
carrier range.

Like `@create`, `@destroy` is an accounting primitive, not an authorization
primitive. The contract decides who may burn or otherwise remove quantity.

## Effects, `modifies`, and locks

Resource operations participate in the normal effect system:

- `@amount` reads its place.
- `@create` and `@destroy` read and write their target.
- `@move` reads and writes both operands.

When a function declares `modifies`, every resource place it may write must be
covered. Resource writes also participate in lock and external-call checks.

```ora
pub fn sweep(amount: TokenUnit)
    modifies reserve, treasury
    requires @amount(reserve) >= amount
    requires @amount(treasury) <=
        @cast(TokenUnit, std.constants.U256_MAX) - amount
{
    @move(reserve, treasury, amount);
}
```

See [Memory Regions](./memory-regions) for persistent and transient storage.

## Failure and verification

Resource operations use checked arithmetic. An invalid operation reverts, and
the transition does not leave a partial resource update behind.

The verifier emits operation-specific obligations, including source validity,
destination range, amount polarity, same-place identity, and conservation. When
the compiler cannot prove a required runtime condition, the corresponding
runtime check remains in the bytecode.

The formal resource model proves local properties of the operations under their
guards, including:

- successful `@move` conserves the touched pair;
- successful self-move is the identity transition;
- `@create` and `@destroy` have the stated deltas;
- places outside an operation's operands are unchanged.

These facts do not automatically prove a contract's business invariants. Those
must still be stated in the contract.

## What resources do not guarantee

Resources intentionally do not provide the following automatically:

- **Authorization.** Anyone allowed by the surrounding function can execute an
  operation. Add owner, role, or capability checks in Ora code.
- **A global supply equation.** `@move` conserves its touched pair, but the
  compiler does not assume that one particular map is the complete domain.
- **Synchronization with a `total_supply` variable.** Update and specify that
  ordinary storage value alongside `@create` and `@destroy`.
- **Runtime asset identity.** Two separately declared resource domains are
  distinct statically; a runtime token-address model is a different abstraction.
- **Deferred commit semantics.** Resource writes currently take effect
  immediately, subject to normal EVM revert atomicity.

## Complete example

```ora
comptime const std = @import("std");

resource TokenUnit = u256;

contract ResourceToken {
    storage var total_supply: TokenUnit;
    storage var balances: map<address, Resource<TokenUnit>>;

    pub fn mint(to: address, amount: TokenUnit)
        modifies total_supply, balances[to]
        requires total_supply <=
            @cast(TokenUnit, std.constants.U256_MAX) - amount
        requires @amount(balances[to]) <=
            @cast(TokenUnit, std.constants.U256_MAX) - amount
        ensures total_supply == old(total_supply) + amount
        ensures @amount(balances[to]) ==
            old(@amount(balances[to])) + amount
    {
        total_supply += amount;
        @create(balances[to], amount);
    }

    pub fn transfer(from: address, to: address, amount: TokenUnit)
        modifies balances[from], balances[to]
        requires @amount(balances[from]) >= amount
        requires @amount(balances[to]) <=
            @cast(TokenUnit, std.constants.U256_MAX) - amount
        ensures total_supply == old(total_supply)
    {
        @move(balances[from], balances[to], amount);
    }

    pub fn burn(from: address, amount: TokenUnit)
        modifies total_supply, balances[from]
        requires total_supply >= amount
        requires @amount(balances[from]) >= amount
        ensures total_supply == old(total_supply) - amount
        ensures @amount(balances[from]) ==
            old(@amount(balances[from])) - amount
    {
        total_supply -= amount;
        @destroy(balances[from], amount);
    }

    pub fn balanceOf(owner: address) -> u256 {
        return @cast(u256, @amount(balances[owner]));
    }
}
```

The repository also contains a complete ERC-20-style implementation at
`ora-example/apps/erc20.ora`.

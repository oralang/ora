---
title: Refinement Types
description: Practical refinement types for safer contracts.
---

# Refinement Types

Refinement types express constraints like "non-zero address" or "amount >= 1" directly in types. The compiler checks them and can remove runtime guards when proofs succeed.

## Common refinements

```ora
let owner: NonZeroAddress = 0x1234567890123456789012345678901234567890;
let fee: BasisPoints<u256> = 250;      // 0..10000
let min: MinValue<u256, 1> = 1;
let cap: MaxValue<u256, 1_000_000> = 500_000;
let rate: InRange<u256, 0, 10000> = 150;
let scaled: Scaled<u256, 18> = 1_000_000_000_000_000_000;
```

## Storage safety

```ora
contract Vault {
    storage var balances: map[NonZeroAddress, u256];

    pub fn deposit(from: NonZeroAddress, amount: MinValue<u256, 1>) {
        balances[from] = balances[from] + amount;
    }
}
```

## Refinements with logic

When control flow proves a condition, the compiler can use it to satisfy a refinement without extra runtime checks.

```ora
fn safeWithdraw(balance: u256, amount: u256) -> u256 {
    if (amount == 0) return balance;
    if (balance < amount) return balance;

    let new_balance: MinValue<u256, 0> = balance - amount;
    return new_balance;
}
```

## Refinement + base type subtyping

Refined values are subtypes of their base types.

```ora
fn takeAddress(addr: address) { }

fn useNonZero() {
    let nz: NonZeroAddress = 0x1234567890123456789012345678901234567890;
    takeAddress(nz);
}
```

For verification details and SMT behavior, see the research writeup in `website/docs/research/refinement-types.md`.

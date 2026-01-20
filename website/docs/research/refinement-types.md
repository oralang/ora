---
sidebar_position: 6
---

# Refinement Types

Refinement types are a core Ora feature: types carry value-level constraints,
and the compiler enforces them through a proof-first, runtime-fallback model.
This is a major step forward for smart contract safety, verification, and
auditor-friendly codebases.

## Why refinements matter

- Encode safety invariants directly in types.
- Reduce boilerplate checks by moving constraints into the compiler.
- Allow proofs to remove runtime guards when constraints are satisfied.

## Implemented today

Refinements currently supported in the front end:

- `MinValue<T, N>`
- `MaxValue<T, N>`
- `InRange<T, Min, Max>`
- `Scaled<T, D>`
- `Exact<T>`
- `NonZeroAddress`

Guards are inserted at parameters, assignments, returns, and storage writes.
Guards are skipped when the type resolver can prove the refinement or when a
trusted source is used (for example, `std.msg.sender()` for `NonZeroAddress`).

## Example

```ora
pub fn transfer(to: NonZeroAddress, amount: MinValue<u256, 1>) -> bool
    requires balances[std.msg.sender()] >= amount
{
    balances[to] += amount;
    return true;
}
```

## More examples

### Range constraints

```ora
fn set_bps(bps: InRange<u256, 0, 10000>) {
    // 0..10000 inclusive
}
```

### Refinements proven by logic (no runtime guard)

```ora
fn accrue(balance: MinValue<u256, 1>, rate_bps: InRange<u256, 0, 10000>) -> MinValue<u256, 1> {
    var interest: u256 = (balance * rate_bps) / 10000;
    var next: MinValue<u256, 1> = balance + interest;
    return next; // SMT can discharge the MinValue guard
}
```

### Control-flow proof with refined locals

```ora
fn safe_withdraw(balance: MinValue<u256, 1>, amount: MinValue<u256, 1>) -> MinValue<u256, 1> {
    if (balance < amount) return balance;

    // Inside this branch: amount <= balance
    var new_balance: MinValue<u256, 1> = balance - amount;
    return new_balance; // SMT proves guard
}
```

### Zero-cost refinement with SMT proof

```ora
fn charge_fee(
    balance: InRange<u256, 0, 1_000_000>,
    amount: InRange<u256, 1, 10_000>,
    fee_bps: InRange<u256, 0, 1000>
) -> InRange<u256, 0, 1_000_000> {
    if (balance < amount) return balance;

    var fee: u256 = (amount * fee_bps) / 10_000;
    var next: InRange<u256, 0, 1_000_000> = balance - amount + fee;
    return next; // SMT proves bounds, guard eliminated
}
```

### Exact division

```ora
fn split_evenly(total: Exact<u256>, parts: MinValue<u256, 1>) -> u256 {
    return total / parts; // guard preserves exactness
}
```

### Scaled values

```ora
fn add_scaled(x: Scaled<u256, 18>, y: Scaled<u256, 18>) -> Scaled<u256, 18> {
    return x + y;
}
```

### Refinements on storage values

```ora
contract Treasury {
    storage var reserve: MinValue<u256, 1>;

    pub fn deposit(amount: MinValue<u256, 1>) {
        reserve = reserve + amount;
    }
}
```

### Non-zero address guard

```ora
fn set_owner(owner: NonZeroAddress) {
    // cannot be zero address
}
```

## How it works in the pipeline

1. **Type resolver** validates the refinement and checks subtyping.
2. **Lowering** emits refinement guards as runtime checks when needed.
3. **SMT pass** can discharge guards when proof succeeds.

Refinement constraints that cannot refine types are surfaced as SMT-only
assumptions instead of being dropped.

## Policy: proof-first, runtime-fallback

- Prove statically when possible.
- Emit a runtime guard only when proof is not available or too costly.
- Keep guard identity stable to allow SMT-based removal.

When the proof succeeds, the guard is removed, making the refinement cost-free
at runtime.

If function logic already establishes a constraint (for example, `if (x > 0)`),
the SMT pass can prove the refinement and skip any extra runtime guard.

```ora
fn ensure_positive(x: u256) -> MinValue<u256, 1> {
    if (x == 0) return 1;
    var y: MinValue<u256, 1> = x;
    return y; // SMT proves the guard from the branch condition
}
```

### Switch-based proof

```ora
fn classify(amount: u256) -> MinValue<u256, 1> {
    switch (amount) {
        0 => return 1,
        else => {
            var out: MinValue<u256, 1> = amount;
            return out; // SMT proves amount > 0 in else
        }
    }
}
```

## Implementation details

- Refinement validation: `src/ast/type_resolver/refinements/**`.
- Guard skip rules: `src/ast/type_resolver/core/statement.zig`.
- Guard emission: `src/mlir/statements/helpers.zig` and
  `src/mlir/declarations/refinements.zig`.
- SMT verification: `src/z3/verification.zig` with encoding in `src/z3/encoder.zig`.

## Evidence

- `docs/compiler/refinement-types-strategy.md`
- `TYPE_SYSTEM_STATE.md`
- `docs/formal-specs/ora-2.md`

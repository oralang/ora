---
sidebar_position: 4
---

# Examples

Ora’s examples live in the **`ora-example/`** directory in the repo. They show real contracts and feature-focused tests you can compile and verify. This page highlights what Ora can do and where to find it.

## What Ora can do

- **Full contracts** — ERC20-style tokens, DeFi-style lending pools, counters, and storage-heavy apps with maps, structs, and enums.
- **Formal verification** — `requires`/`ensures`/`invariant`, ghost state, loop invariants, and SMT-backed verification on real contracts.
- **Refinement types** — MinValue, MaxValue, InRange, BasisPoints, NonZero, Scaled, Exact, and NonZeroAddress in production-style token logic.
- **Comptime and generics** — Constant folding, `comptime T: type`, generic functions and structs, and contracts that combine bitfields with generics.
- **Signed arithmetic, bitfields, errors** — Signed ints with overflow handling, packed bitfields, and error unions with `try`/`catch`-style handling.
- **Verification and SMT** — Loop invariants, function contracts, state invariants, ghost functions, and multi-constraint proofs.

---

## Showcase: complex contracts

| Example | Path | What it shows |
|--------|------|----------------|
| **ERC20 + bitfields + generics** | `ora-example/apps/erc20_bitfield_comptime_generics.ora` | Token with bitfield config/flags, generic `TransferBook(T)`, comptime helpers, maps, logs. |
| **DeFi lending pool** | `ora-example/apps/defi_lending_pool.ora` | Large contract: enums, storage, interest/reserve logic, liquidation, multiple entrypoints. |
| **ERC20 with formal verification** | `ora-example/apps/erc20_verified.ora` | Invariants, ghost `sumOfBalances`, `requires`/`ensures` on init and transfer. |
| **ERC20 with refinements** | `ora-example/refinements/erc20_token.ora` | MinValue, MaxValue, InRange, BasisPoints, NonZero, Scaled, Exact, NonZeroAddress in token logic. |
| **DeFi pool + verification** | `ora-example/apps/defi_lending_pool_fv.ora` | Lending pool with formal verification annotations. |
| **Counter** | `ora-example/apps/counter.ora` | Minimal contract: storage, public functions. |
| **Basic ERC20** | `ora-example/apps/erc20.ora` | Straightforward ERC20-style token. |

---

## Examples by topic

### Apps (`ora-example/apps/`)

- `counter.ora` — Simple storage counter.
- `erc20.ora` — ERC20 token (balances, allowances, transfer, approve).
- `erc20_verified.ora` — ERC20 with invariants and function contracts.
- `erc20_bitfield_comptime_generics.ora` — ERC20 with bitfields, generics, and comptime.
- `defi_lending_pool.ora` — DeFi-style lending pool (deposits, borrows, liquidation).
- `defi_lending_pool_fv.ora` — Same with formal verification.

### Comptime and generics (`ora-example/comptime/`, `ora-example/comptime/generics/`)

- `comptime/generic_max.ora`, `comptime/generic_pair.ora` — Generic function and generic struct (e.g. `max(T, a, b)`, `Pair(T)`).
- `comptime/generics/generic_fn_identity_cache.ora` — Generic identity, cache behavior.
- `comptime/generics/generic_struct_expr_only.ora`, `generic_struct_multi_type_params.ora` — Generic structs and multiple type params.
- `comptime/generics/generic_nested_calls.ora` — Nested generic calls.
- `comptime/comptime_folds.ora`, `comptime_basics.ora` — Constant folding and comptime evaluation.
- `comptime/comptime_overflow_probe.ora`, `comptime_wrapping_ops.ora` — Comptime overflow and wrapping behavior.

### Refinements (`ora-example/refinements/`)

- `erc20_token.ora` — Full token using refinement types (see table above).
- `guards_showcase.ora` — Guard behavior and placement.
- `guard_placement_smoke.ora` — Guard placement and refinement flow.
- `patterns/fee_calculation.ora`, `patterns/supply_cap.ora` — Fee and supply-cap patterns with refinements.
- `dispatcher_refinement_e2e.ora` — Refinements through a dispatcher.

### Verification and SMT (`ora-example/smt/`, `ora-example/verification/`)

- `smt/verification/loop_invariants.ora` — Loop invariants and termination.
- `smt/verification/function_contracts_refinements.ora` — Function contracts with refinement types.
- `smt/verification/ghost_functions.ora`, `ghost_combined.ora` — Ghost functions and state.
- `smt/verification/state_invariants.ora` — Contract state invariants.
- `smt/complex/multiple_constraints.ora`, `arithmetic_proofs.ora` — Multiple constraints and arithmetic proofs.
- `verification/fv_test.ora` — Formal verification test.

### Bitfields (`ora-example/bitfields/`)

- `basic_bitfield_storage.ora` — Bitfield in storage.
- `contract_scoped_bitfields.ora` — Contract-scoped bitfield usage.
- `bitfield_map_values.ora` — Maps with bitfield values.

### Signed integers (`ora-example/signed/`)

- `basic_signed_ops.ora`, `wrapping_ops.ora` — Signed arithmetic and wrapping.
- `overflow_builtins.ora` — Overflow-checking builtins.
- `storage_ledger.ora` — Signed balances in storage.

### Error unions (`ora-example/errors/`)

- `basic_errors.ora`, `test_error_handling.ora` — Error definitions and handling.
- `try_expression.ora`, `try_catch_block.ora` — `try` and catch blocks.
- `error_union_split.ora` — Splitting and switching on error unions.

### Other

- **Regions:** `ora-example/regions/` — Storage, stack, effects, coercions.
- **Structs, enums, tuples:** `ora-example/types/`, `ora-example/tuples/tuple_basics.ora`.
- **Logs:** `ora-example/logs/`.
- **Loops:** `ora-example/loops/loop_basics.ora`.
- **Memory:** `ora-example/memory/`, `ora-example/transient/`.

---

## Quick snippets

Short illustrations of core syntax (full examples are in `ora-example/`).

**Contract + storage**

```ora
contract Counter {
    storage var value: u256;
    pub fn inc(delta: u256) { value = value + delta; }
    pub fn get() -> u256 { return value; }
}
```

**Refinement type**

```ora
fn withdraw(amount: MinValue<u256, 1>) -> bool { return true; }
```

**Switch expression**

```ora
var out: u32 = switch (x) { 0 => 0, 1...9 => 1, else => 2 };
```

**Requires / ensures**

```ora
pub fn transfer(to: address, amount: u256) -> bool
    requires amount > 0
    ensures amount > 0
{ return true; }
```

**Generic function**

```ora
fn max(comptime T: type, a: T, b: T) -> T {
    if (a > b) { return a; } else { return b; }
}
```

---

## Run examples

From the repo root after `zig build`:

```bash
# Validate all examples
./scripts/validate-examples.sh

# Compile and emit MLIR (with verification)
./zig-out/bin/ora --emit-mlir ora-example/apps/counter.ora

# Emit SMT report for a contract
./zig-out/bin/ora --emit-smt-report ora-example/apps/erc20_verified.ora
```

See [Getting Started](/docs/getting-started) and the [README](https://github.com/oralang/Ora) for install and more commands.

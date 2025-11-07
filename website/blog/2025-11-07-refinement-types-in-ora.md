---
slug: refinement-types-in-ora
title: Refinement Types: Making Smart Contracts Safer by Default
authors: [axe]
tags: [compiler, type-system, safety, refinement-types]
---

Smart contracts deal with money. That means every integer operation, every division, every transfer needs to be correct. Traditional type systems catch some bugs, but they don't catch the ones that actually matter: "is this value non-zero?", "is this amount above the minimum?", "is this rate within bounds?"

Refinement types fix that. They let you express constraints directly in the type system, and the compiler enforces them—at compile time when possible, at runtime when necessary.

<!-- truncate -->

## What Are Refinement Types?

A refinement type is a base type with a logical predicate attached. Instead of just `u256`, you can have `MinValue<u256, 1000>` (values ≥ 1000) or `InRange<u256, 0, 10000>` (values between 0 and 10000).

Think of it as a type that says "this is a `u256`, but it's also guaranteed to satisfy this constraint."

```ora
// Regular type: any u256 value
let amount: u256 = 0;  // OK, but maybe not what you want

// Refined type: u256 that's guaranteed ≥ 1000
let deposit: MinValue<u256, 1000> = 500;  // ❌ Compile error: 500 < 1000
let deposit: MinValue<u256, 1000> = 2000; // ✅ OK: 2000 ≥ 1000
```

The compiler checks these constraints. If you try to assign a value that doesn't satisfy the constraint, you get a compile-time error. No runtime checks needed—the compiler proved it's safe.

## Why This Matters for Smart Contracts

In financial code, most bugs come from edge cases that traditional types don't catch:

- **Division by zero**: `amount / fee_rate` where `fee_rate` might be zero
- **Dust attacks**: Accepting transfers of 0 or tiny amounts that cost more in gas than they're worth
- **Overflow/underflow**: Values exceeding maximums or going negative
- **Invalid ranges**: Fee rates outside 0-100%, percentages over 100%

Refinement types catch these at compile time. You declare the constraint once in the type, and the compiler enforces it everywhere.

```ora
// Without refinement types - manual checks everywhere
fn transfer(to: address, amount: u256) {
    require(amount >= 1000, "Amount too small");  // Manual check
    require(to != address(0), "Invalid recipient");  // Manual check
    // ... transfer logic
}

// With refinement types - constraints in the type
fn transfer(to: NonZeroAddress, amount: MinValue<u256, 1000>) {
    // Compiler guarantees: to != 0, amount >= 1000
    // No manual checks needed
    // ... transfer logic
}
```

## Refinement Types in Ora

Ora supports several built-in refinement types:

### `MinValue<T, N>` - Minimum Thresholds

Values that are guaranteed to be ≥ N. Perfect for minimum deposits, dust prevention, or any "at least X" constraint.

```ora
type DepositAmount = MinValue<u256, 1_000_000>;  // Minimum 1M wei

fn withdraw(amount: DepositAmount) {
    // Compiler knows: amount >= 1_000_000
    // No need to check
}
```

### `MaxValue<T, N>` - Maximum Bounds

Values that are guaranteed to be ≤ N. Useful for supply caps, rate limits, or fee ceilings.

```ora
type MaxSupply = MaxValue<u256, 1_000_000_000>;  // Cap at 1 billion

fn mint(amount: u256) -> MaxSupply {
    let new_supply = total_supply + amount;
    // Type system ensures new_supply <= 1_000_000_000
    return new_supply;
}
```

### `InRange<T, MIN, MAX>` - Bounded Values

Values guaranteed to be between MIN and MAX. Perfect for percentages, fee rates, or any bounded quantity.

```ora
type FeeRate = InRange<u256, 0, 10_000>;  // 0-100% in basis points

fn setFee(rate: FeeRate) {
    // Compiler knows: 0 <= rate <= 10_000
    transfer_fee_rate = rate;
}
```

### `NonZero<T>` - Non-Zero Guarantees

Alias for `MinValue<T, 1>`. Ensures a value is strictly positive—critical for division operations.

```ora
fn divide(a: u256, b: NonZero<u256>) -> u256 {
    return a / b;  // Safe: b is guaranteed non-zero
}
```

### `NonZeroAddress` - Valid Addresses

Prevents the zero address, which is a common source of bugs in token transfers and ownership.

```ora
fn transfer(to: NonZeroAddress, amount: u256) {
    // Compiler guarantees: to != address(0)
    balances[to] += amount;
}
```

### `Scaled<T, D>` - Fixed-Point Arithmetic

Tracks decimal scaling for fixed-point math. `Scaled<u256, 18>` represents values scaled by 10^18 (like WAD in DeFi).

```ora
type WAD = Scaled<u256, 18>;  // 18 decimal places

fn addWads(a: WAD, b: WAD) -> WAD {
    return a + b;  // Scale preserved: result is also Scaled<u256, 18>
}

fn multiplyWads(a: WAD, b: WAD) -> Scaled<u256, 36> {
    return a * b;  // Scale doubles: 18 + 18 = 36
}
```

The compiler prevents mixing different scales in addition/subtraction (compile error), but allows multiplication (scale doubles).

### `BasisPoints<T>` - Percentage Types

Specialized type for percentages: `InRange<T, 0, 10_000>` where 10,000 = 100%.

```ora
type FeeRate = BasisPoints<u256>;

fn calculateFee(amount: u256, rate: FeeRate) -> u256 {
    return (amount * rate) / 10_000;  // Type-safe percentage calculation
}
```

## Compile-Time vs Runtime

The compiler tries to prove constraints statically. If it can't, it inserts runtime guards automatically.

```ora
// Compile-time check: literal value
let x: MinValue<u256, 1000> = 500;  // ❌ Compile error

// Runtime guard: dynamic value
fn process(value: u256) -> MinValue<u256, 1000> {
    // Compiler inserts: if (value < 1000) revert("Refinement violation")
    return value;
}
```

This gives you the best of both worlds: compile-time guarantees when possible, runtime safety when necessary.

## Arithmetic with Refined Types

The compiler is smart about arithmetic operations. It infers refined result types from operands:

```ora
let x: MinValue<u256, 100> = 200;
let y: MinValue<u256, 50> = 100;

let sum = x + y;  // Type: MinValue<u256, 150> (100 + 50)
let diff = x - y; // Type: MinValue<u256, 50> (conservative: 100 - 50)
let prod = x * y; // Type: MinValue<u256, 5000> (100 * 50)
```

For `Scaled` types, the compiler preserves or adjusts scales:

```ora
let wad1: Scaled<u256, 18> = 1_000_000_000_000_000_000;
let wad2: Scaled<u256, 18> = 2_000_000_000_000_000_000;

let sum = wad1 + wad2;      // Scaled<u256, 18> (scale preserved)
let product = wad1 * wad2;  // Scaled<u256, 36> (scale doubles)
```

## Real Example: ERC20 Token

Here's how refinement types make a real contract safer:

```ora
contract ERC20Token {
    storage var total_supply: MaxValue<u256, 1_000_000_000> = 0;
    storage var transfer_fee_rate: BasisPoints<u256> = 0;
    storage var min_transfer: MinValue<u256, 1> = 1;
    storage var owner: NonZeroAddress;
    
    pub fn transfer(
        to: NonZeroAddress,
        amount: MinValue<u256, 1>
    ) -> !u256 | InsufficientBalance | TransferToZeroAddress {
        // Compiler guarantees:
        // - to != address(0) (NonZeroAddress)
        // - amount >= 1 (MinValue<u256, 1>)
        // - transfer_fee_rate in [0, 10000] (BasisPoints)
        
        if (amount < min_transfer) {
            return AmountBelowMinimum(amount, min_transfer);
        }
        
        // Type-safe arithmetic
        const amount_after_fee = amount - calculateFee(amount);
        // ...
    }
    
    fn calculateFee(amount: u256) -> u256 {
        // BasisPoints ensures transfer_fee_rate is in [0, 10000]
        return amount * transfer_fee_rate / 10_000;
    }
}
```

Every constraint is in the type system. The compiler enforces them. No scattered `require()` checks. No "I hope I remembered to validate this" moments.

## Subtyping and Compatibility

Refinement types form a subtype hierarchy. A more restrictive type is a subtype of a less restrictive one:

```ora
// MinValue<u256, 1000> is a subtype of MinValue<u256, 100>
// (more restrictive = subtype)
let strict: MinValue<u256, 1000> = 2000;
let loose: MinValue<u256, 100> = strict;  // ✅ OK: 1000 >= 100

// NonZeroAddress is a subtype of address
let nonzero: NonZeroAddress = 0x1234...;
let addr: address = nonzero;  // ✅ OK
```

This enables safe coercions and function overloading.

## Why Not Just Use `require()`?

You could write `require(amount >= 1000, "Too small")` everywhere. But:

1. **You have to remember**: Easy to forget a check, especially in complex code
2. **It's runtime-only**: Bugs only show up when the code runs
3. **It's verbose**: Repeating the same checks everywhere
4. **It's not composable**: Can't pass constraints through function calls

Refinement types solve all of this. The constraint is in the type, so:
- ✅ **Compiler enforces it**: Can't forget, can't bypass
- ✅ **Compile-time when possible**: Catch bugs before deployment
- ✅ **Declare once**: Type carries the constraint everywhere
- ✅ **Composable**: Functions preserve and propagate constraints

## The Philosophy: Comptime > Runtime

This fits Ora's core philosophy: **if something can be checked at compile time, it should be**.

Refinement types push validation from runtime to compile-time wherever possible. When the compiler can prove a constraint holds, no runtime check is needed. When it can't, a guard is inserted automatically.

The result: safer contracts, less gas spent on validation, and bugs caught before deployment.

## What's Next

Refinement types are fully implemented in Ora. You can use them today in:
- Function parameters and return types
- Variable declarations
- Arithmetic operations (with automatic result type inference)
- Storage variables

Future enhancements might include:
- User-defined predicates (beyond built-in types)
- SMT solver integration for more complex proofs
- Automatic unit rescaling for `Scaled` types
- Refined collections (arrays with length constraints)

But even with just the built-in types, refinement types make a huge difference. They turn "I hope this is correct" into "the compiler proved it's correct."

Try it out. Write a contract with `MinValue`, `NonZeroAddress`, and `BasisPoints`. You'll see how much cleaner and safer the code becomes when constraints are in the type system, not scattered in `require()` statements.

---

*Want to see refinement types in action? Check out [`ora-example/refinements/erc20_token.ora`](https://github.com/oralang/Ora/blob/main/ora-example/refinements/erc20_token.ora) for a complete example.*


# Ora Integer Arithmetic Safety Model (Checked / Wrapping / Reporting)

**Status:** Draft Spec  
**Scope:** Runtime machine integers (`uN`, `iN`)  
**Non-goals:** Fixed-point / floats, arbitrary-precision runtime integers (may exist in spec/ghost only)

---

## 1. Goals

1. Arithmetic is safe by default.
2. Escape hatches are explicit and easy to audit.
3. Verification is sound and predictable.
4. No hidden operator behavior changes per scope/file.

---

## 2. Terminology

For integer arithmetic, **overflow** means the mathematical result is out of range for the type.

- For unsigned types, this includes what is often called *underflow* (e.g. `0 - 1` on `u256`).
- For signed types, overflow includes both upper-bound and lower-bound violations.

Let `N = bitWidth(T)` for an integer type `T`.

- `uN` range: `[0, 2^N - 1]`
- `iN` range: `[-2^(N-1), 2^(N-1) - 1]` (two's complement)

---

## 3. Type-Level Baseline

1. Runtime numeric types are bounded machine integers: `uN` and `iN`.
2. (Optional) spec/ghost-only unbounded integers (`int`, `nat`) may exist for proof convenience; they do not exist at runtime and do not change runtime operator semantics.

---

## 4. Operator Semantics (Checked by Default)

The following operators are **checked** by default:

- `+` addition
- `-` subtraction
- `*` multiplication
- `/` division
- `%` modulo
- `<<` shift left
- `>>` shift right

Unary negation `-a` is defined **only for signed integers** (`iN`) and is checked (see below).

If a checked operation would overflow or is otherwise invalid, **execution traps/reverts**.

### 4.1 Checked operation trap conditions

For operands `a, b : T` and shift amount `s`:

**Addition (`a + b`)**
- traps iff result is not representable in `T`

**Subtraction (`a - b`)**
- traps iff result is not representable in `T`

**Multiplication (`a * b`)**
- traps iff result is not representable in `T`

**Division (`a / b`)**
- traps iff `b == 0`
- for signed types, also traps iff `a == MIN(T)` and `b == -1`

**Modulo (`a % b`)**
- traps iff `b == 0`

**Negation (`-a`)** (signed only)
- only valid for `iN`
- traps iff `a == MIN(iN)`
- unary `-` on unsigned types is a compile-time error (use `0 -% a` for wrapping negation)

**Shift left (`a << s`)**
Let `N = bitWidth(T)`. The operation traps iff:
1) `s >= N`, or
2) the result is not representable in `T`.

Representability is defined by round-tripping the wrapped shift:
- For `a << s` with `s < N`, let `r = a <<% s` (wrapping left shift).
- For `uN`, trap iff `(r >> s) != a` (logical shift right).
- For `iN`, trap iff `SAR(r, s) != a`, where `SAR` is arithmetic shift-right.

**Shift right (`a >> s`)**
- traps iff `s >= N`
- for `uN`: logical shift right (zero-fill)
- for `iN`: arithmetic shift right (sign-propagating)
- unlike `<<`, `>>` does not trap based on "bits shifted out"

---

## 5. Explicit Wrapping Operators (Modular Arithmetic)

Wrapping behavior is opt-in and explicit. Wrapping operations **never trap on overflow** and compute results modulo `2^N`.

- `+%` wrapping addition
- `-%` wrapping subtraction
- `*%` wrapping multiplication
- `<<%` wrapping shift left
- `>>%` wrapping shift right
- unary `-%` wrapping negation (signed and unsigned, but unary `-` itself is signed-only)

### 5.1 Wrapping semantics

For `uN`:
- results are reduced modulo `2^N`

For `iN`:
- results are computed in two's complement modulo `2^N`, then interpreted as `iN`

Shifts:
- `a <<% s`:
  - if `s >= N`, result is `0`
  - else result is `(a << s) mod 2^N`
- `a >>% s`:
  - if `s >= N`, result is `0` for `uN`
  - if `s >= N`, result is `-1` for negative `iN`, `0` for non-negative `iN` (sign-preserving saturation)
  - else standard right shift for the type (logical for `uN`, arithmetic for `iN`)

> Note: Wrapping shifts are included for completeness. Implementations may choose to provide wrapping shifts only via `@shlWithOverflow/@shrWithOverflow` if the surface operator set should remain minimal.

---

## 6. Overflow-Reporting Builtins (`@…WithOverflow`)

Ora provides builtins that return the **wrapped result** plus a boolean flag indicating whether the corresponding checked operator would trap/revert.

### 6.1 General contract

For each builtin `@opWithOverflow`, the returned pair `(value, overflow)` satisfies:

1. `value` equals the result of the corresponding wrapping computation (modulo `2^N` for integer types).
2. `overflow` is `true` iff the corresponding checked operator would trap/revert (for any trap reason defined for that operator).

The flag name is `overflow` for brevity; semantically it means **"checked op would trap."**

### 6.2 Required builtins

- `@addWithOverflow(a: T, b: T) -> (value: T, overflow: bool)`
- `@subWithOverflow(a: T, b: T) -> (value: T, overflow: bool)`
- `@mulWithOverflow(a: T, b: T) -> (value: T, overflow: bool)`
- `@divWithOverflow(a: T, b: T) -> (value: T, overflow: bool)`
- `@modWithOverflow(a: T, b: T) -> (value: T, overflow: bool)`
- `@negWithOverflow(a: T)        -> (value: T, overflow: bool)` (signed types only)
- `@shlWithOverflow(a: T, shift: u256) -> (value: T, overflow: bool)`
- `@shrWithOverflow(a: T, shift: u256) -> (value: T, overflow: bool)`

### 6.3 Division and modulo overflow definitions

For `@divWithOverflow` and `@modWithOverflow`, `overflow=true` iff the checked op would trap:
- `b == 0`, or (signed division only) `a == MIN(T)` and `b == -1`.

### 6.4 Shift overflow definitions

Let `N = bitWidth(T)` and `s` be the shift amount.

**`@shlWithOverflow(a, s)`**
- `overflow = (s >= N) OR (result not representable in T under checked-left-shift rules)`
- `value` equals `a <<% s`
- If `s >= N`, `value` is defined as `0`.

**`@shrWithOverflow(a, s)`**
- `overflow = (s >= N)`
- `value` equals `a >>% s`
- If `s >= N`, `value` is defined as:
  - `0` for unsigned types
  - `-1` for negative signed values, `0` for non-negative signed values (sign-preserving saturation)

---

## 7. Examples and Ergonomics

This section demonstrates the three-tier arithmetic model with practical Ora examples.

### 7.1 Checked operators (default)

Checked operators are concise and safe by default. They trap on overflow, making bugs explicit:

```ora
contract Token {
    storage var balances: map[address, u256];
    
    pub fn transfer(from: address, to: address, amount: u256) -> bool
        requires(balances[from] >= amount)
    {
        // Safe: traps if balances[from] < amount (underflow)
        balances[from] = balances[from] - amount;
        
        // Safe: traps if balances[to] + amount overflows
        var to_balance: u256 = balances[to];
        balances[to] = to_balance + amount;
        
        return true;
    }
    
    pub fn calculateInterest(principal: u256, rate: u256) -> u256 {
        // Safe: traps on overflow (common in DeFi)
        return principal * rate / 100;
    }
}
```

### 7.2 Wrapping operators (explicit escape hatches)

Wrapping operators are explicit and grep-friendly. Use them when modular arithmetic is intended:

```ora
contract HashUtils {
    pub fn hashCombine(h1: u256, h2: u256) -> u256 {
        // Explicit wrapping: intended modular arithmetic for hashing
        return h1 *% 31 +% h2;
    }
    
    pub fn ringBufferIndex(current: u256, offset: u256, size: u256) -> u256 {
        // Explicit wrapping: ring buffer arithmetic
        return (current +% offset) % size;
    }
    
    pub fn bitRotateLeft(value: u256, shift: u256) -> u256 {
        // Explicit wrapping: bit manipulation
        var N: u256 = 256;
        return (value <<% shift) | (value >>% (N -% shift));
    }
}
```

### 7.3 Overflow-reporting builtins (explicit control flow)

Use `@...WithOverflow` when you need to handle overflow explicitly without trapping:

```ora
error ArithmeticOverflow;

contract SafeArithmetic {
    pub fn safeAdd(a: u256, b: u256) -> !u256 | ArithmeticOverflow {
        var result: u256;
        var overflow: bool;
        result, overflow = @addWithOverflow(a, b);
        
        if (overflow) {
            return error.ArithmeticOverflow;
        }
        return result;
    }
    
    pub fn saturatingAdd(a: u256, b: u256) -> u256 {
        var result: u256;
        var overflow: bool;
        result, overflow = @addWithOverflow(a, b);
        
        if (overflow) {
            return std.constants.U256_MAX; // Saturate at maximum
        }
        return result;
    }
    
    pub fn multiPrecisionAdd(a_high: u256, a_low: u256, b: u256) -> (u256, u256) {
        // Low word addition
        var low_sum: u256;
        var low_overflow: bool;
        low_sum, low_overflow = @addWithOverflow(a_low, b);
        
        // Propagate carry to high word
        var high_sum: u256;
        if (low_overflow) {
            var h: u256;
            var h_overflow: bool;
            h, h_overflow = @addWithOverflow(a_high, 1);
            if (h_overflow) {
                revert("Multi-precision overflow");
            }
            high_sum = h;
        } else {
            high_sum = a_high;
        }
        
        return (high_sum, low_sum);
    }
    
    pub fn checkedShiftLeft(value: u256, shift: u256) -> !u256 | ArithmeticOverflow {
        var result: u256;
        var overflow: bool;
        result, overflow = @shlWithOverflow(value, shift);
        
        if (overflow) {
            return error.ArithmeticOverflow; // Shift amount too large or bits lost
        }
        return result;
    }
}
```

### 7.4 Comparison: three approaches

Here's the same operation using all three approaches:

```ora
contract ArithmeticComparison {
    // Approach 1: Checked (default) - concise, traps on overflow
    pub fn addChecked(a: u256, b: u256) -> u256 {
        return a + b; // Traps if overflow
    }
    
    // Approach 2: Wrapping - explicit, never traps
    pub fn addWrapping(a: u256, b: u256) -> u256 {
        return a +% b; // Always succeeds, wraps on overflow
    }
    
    // Approach 3: Reporting - explicit control flow
    error Overflow;
    pub fn addReporting(a: u256, b: u256) -> !u256 | Overflow {
        var result: u256;
        var overflow: bool;
        result, overflow = @addWithOverflow(a, b);
        
        if (overflow) {
            return error.Overflow;
        }
        return result;
    }
}
```

### 7.5 Signed arithmetic examples

```ora
error NegationOverflow;
error DivisionOverflow;

contract SignedArithmetic {
    pub fn negateSigned(value: i256) -> i256 {
        // Checked negation: traps if value == MIN(i256)
        return -value;
    }
    
    pub fn negateWrapping(value: i256) -> i256 {
        // Wrapping negation: always succeeds
        return -%value;
    }
    
    pub fn negateReporting(value: i256) -> !i256 | NegationOverflow {
        var result: i256;
        var overflow: bool;
        result, overflow = @negWithOverflow(value);
        
        if (overflow) {
            return error.NegationOverflow;
        }
        return result;
    }
    
    pub fn divideSigned(a: i256, b: i256) -> i256 {
        // Checked division: traps on div-by-zero or MIN/-1
        return a / b;
    }
    
    pub fn divideReporting(a: i256, b: i256) -> !i256 | DivisionOverflow {
        var result: i256;
        var overflow: bool;
        result, overflow = @divWithOverflow(a, b);
        
        if (overflow) {
            return error.DivisionOverflow;
        }
        return result;
    }
}
```

---

## 8. Ergonomics and Intended Use

This model provides three explicit tools, each optimized for a different style of code:

1. **Checked operators** are the default. They are concise and safe. They are ideal for invariants and business logic where overflow indicates a bug or invalid input.

2. **Wrapping operators** are explicit escape hatches for modular arithmetic (hashing, bit manipulation, ring buffers). They avoid hidden wrap-around in normal code and are easy to grep/audit.

3. **`@…WithOverflow` builtins** enable non-trapping arithmetic when the caller wants to branch on failure, implement low-level packing/unpacking, or write multi-precision routines. They avoid control-flow via trap and make failure handling explicit in the dataflow.

A `tryAdd` / `checkedAdd` helper can be provided later as *library sugar* over `@addWithOverflow` (returning an error union), but is not part of this spec.

---

## 9. Lowering Model (EVM)

### 8.1 Checked operators

Lower checked ops as:
1. compute wrapped result (EVM arithmetic is modular)
2. compute the overflow/trap predicate (including div-by-zero / invalid shift)
3. branch to a standardized trap/revert path if predicate is true
4. continue with the result otherwise

### 8.2 Wrapping operators

Lower directly to EVM arithmetic and bit operations with no trap path.

### 8.3 `@…WithOverflow`

Lower as:
1. compute wrapped result
2. compute overflow predicate for the corresponding checked op
3. construct the tuple `(value, overflow)` in the chosen IR representation

---

## 10. SMT and Verification Model

### 9.1 Soundness requirement

Even though checked operators trap at runtime, the verifier must model overflow/div-by-zero conditions to keep proofs sound.

### 9.2 Proof policy (recommended default)

1. Postconditions (`ensures`) are proved on successful (non-revert) paths only.
2. Trap paths from checked arithmetic are excluded from success postconditions.
3. Wrapping ops (`+%`, `-%`, `*%`, shifts) produce no "no-overflow" obligations.
4. `@…WithOverflow` requires proving correctness of the reported `overflow` flag and the returned wrapped value semantics.

### 9.3 Optional strict mode

An optional mode (e.g. `--prove-no-arithmetic-trap`) may require proving that checked arithmetic cannot trap on any reachable path, i.e. overflow/div-by-zero/invalid shift are unreachable.

This yields two useful modes:
- **Default:** safe runtime semantics + success-path correctness
- **Strict:** additionally prove arithmetic trap paths are unreachable

---

## 11. Implementation Notes (Non-Normative)

This section provides recommended overflow predicates for implementers.

Let all arithmetic be over `N`-bit words (bitvectors). Let `sign = 1 << (N-1)`.

### 10.1 Unsigned overflow

For `uN`, let `r_add = a +% b`, `r_sub = a -% b`:
- add overflow iff `r_add < a` (unsigned compare)
- sub underflow iff `r_sub > a` (unsigned compare)

### 10.2 Signed overflow (two's complement)

For `iN`, let `r_add = a +% b`, `r_sub = a -% b`:
- add overflow iff `((~(a ^ b) & (a ^ r_add)) & sign) != 0`
  - This detects when adding two positive numbers yields negative, or two negative numbers yields positive.
- sub overflow iff `(((a ^ b) & (a ^ r_sub)) & sign) != 0`
  - This detects when subtracting numbers with opposite signs causes sign change in the wrong direction.
- neg overflow iff `a == MIN(iN)` where `MIN(iN) = 1 << (N-1)`
- signed division additional trap iff `a == MIN(iN) && b == -1`

### 10.3 Checked left shift representability

For `a << s` with `s < N` and `r = a <<% s`:
- unsigned representable iff `(r >> s) == a` (logical shift right)
- signed representable iff `SAR(r, s) == a` (arithmetic shift right)

---

## 12. Rationale

- Safe-by-default behavior matches modern expectations and reduces accidental bugs.
- Escape hatches are explicit and grep-friendly.
- No scope-level "pragma" that changes operator meaning.
- Verification remains practical by default and can be strengthened when desired.

---
slug: state-analysis-and-verification
title: Automatic State Analysis and Z3 Integration
authors: [axe]
tags: [compiler, formal-verification, analysis]
---

The Ora compiler now includes automatic state analysis that tracks every storage operation in your contracts. It's also halfway through Z3 integration for formal verification. Both are turning out to be more useful than I expected.

<!-- truncate -->

## State Analysis: What It Does

Every time you compile a contract, the compiler analyzes which functions read and write to storage. It then warns you about issues that would otherwise bite you in production.

Here's what it catches:

**Dead stores** - variables written but never read:
```ora
storage unusedData: u256;

pub fn setData(value: u256) {
    unusedData = value;  // Written, but no function reads it
}
```

The compiler tracks this across the entire contract. If any function reads `unusedData`, no warning. If nothing reads it, you get:

```
Warning: Storage variable 'unusedData' is written by 'setData' 
         but never read by any function in the contract
```

This is contract-level analysis. Function-level would give false positives for legitimate setter functions.

**Missing validation** - functions that modify storage without checks:
```ora
pub fn approve(spender: address, amount: u256) {
    allowances[msg.sender()][spender] = amount;  // No validation
}
```

You get an info-level warning:
```
Info: Function 'approve' modifies storage without reading any state
      Consider adding validation checks
```

Maybe you want `require(spender != address(0))` or `require(amount > 0)`. Maybe you don't. But the compiler tells you to think about it.

**Constructor-specific checks** - different rules for `init`:
```ora
pub fn init(initialSupply: u256) {
    totalSupply = initialSupply;  // Expected for constructors
}
```

Gets a tailored warning:
```
Info: Constructor 'init' stores parameters without validation
      Consider checking: amount > 0, address != 0
```

Constructors are supposed to write without reading. The compiler knows this and checks for unvalidated parameters instead.

## Why This Matters

Storage operations are expensive. Every `SLOAD` costs 2100 gas (warm) or 2600 gas (cold). Every `SSTORE` costs 2900-20000 gas depending on the value change. Writing values you never read is burning gas for nothing.

More importantly, missing validation checks are security holes. The compiler can't know what validation you need, but it can flag functions that modify state without reading anything first. That's usually a smell.

The analysis runs automatically. Zero overhead during compilation. You just get warnings when there's a problem.

## How It Works

The compiler builds an AST visitor that tracks every expression and statement. When it sees storage access, it records whether it's a read or write. At the contract level, it aggregates all reads and writes across all functions.

Then it generates warnings:
1. Variables in writes but not in reads = dead store
2. Functions with writes but no reads = missing validation
3. Constructor (`init`) with writes = check parameters

The hard part was avoiding false positives. Initial implementation flagged legitimate patterns like ERC20 `approve()` writing to `allowances`. The contract-level analysis fixes this - `transferFrom()` reads what `approve()` writes, so no warning.

## Z3 Integration Status

The formal verification side is progressing. Grammar and AST support for all verification annotations is complete:

```ora
pub fn transfer(recipient: address, amount: u256) -> bool
    requires amount > 0
    requires balances[msg.sender()] >= amount
    ensures balances[msg.sender()] == old(balances[msg.sender()]) - amount
    ensures balances[recipient] == old(balances[recipient]) + amount
{
    // Implementation
}
```

The parser handles this. The AST represents it. Ghost code (specification-only variables and functions) is filtered out during code generation.

What's left:
- Verification condition generation
- Encoding constraints for Z3
- Counterexample reporting

The infrastructure is there. Z3 is in the build system. The language supports the syntax. Now it's about generating the right SMT-LIB2 constraints and feeding them to Z3.

## Ghost Code

One interesting piece: ghost code never makes it to bytecode. You can write specification-only logic that's purely for verification:

```ora
contract Token {
    ghost storage sumOfAllBalances: u256;
    
    ghost fn totalBalance() -> u256 {
        return sumOfAllBalances;
    }
    
    pub fn transfer(recipient: address, amount: u256)
        ensures totalBalance() == old(totalBalance())
    {
        // Transfer preserves total balance
    }
}
```

The compiler uses this for verification, then strips it out. Your deployed bytecode never includes it. No gas cost, no storage overhead.

This is useful for expressing complex properties that don't fit naturally in the contract's actual state.

## Implementation Details

State analysis happens after parsing but before MLIR generation. It's fast - microseconds for most contracts. The analysis builds hash maps of reads and writes per function, then aggregates them at the contract level.

The code is in `src/analysis/state_tracker.zig`. About 600 lines. It walks the AST and tracks:
- Direct storage access: `counter = 5`
- Map operations: `balances[addr] = 100`
- Compound assignments: `counter += 1`
- Multiple operations in one function

Z3 integration lives in `vendor/z3/` (system install preferred, vendor fallback available). The verification grammar is in `src/parser/` with special handling for `requires`, `ensures`, `invariant`, `ghost`, `old()`, `forall`, `exists`.

## What's Next

For state analysis: symbol table integration to replace storage variable heuristics. Right now it uses common names (`balances`, `counter`, etc.) to detect storage variables. The symbol table will make this precise.

For Z3: verification condition generation. This is the hard part. Taking AST + specifications and producing SMT-LIB2 constraints that Z3 can solve. Once this works, the rest follows.

## Try It

If you have the compiler built:

```bash
# Automatic analysis during compilation
ora contract.ora

# Detailed breakdown
ora --analyze-state contract.ora

# Check if verification annotations parse
ora --emit-ast contract_with_specs.ora
```

State analysis is production-ready. Z3 integration is in progress but the language side is done.

The compiler keeps getting smarter about catching bugs before deployment. That's the point.

---

*October 28, 2025*


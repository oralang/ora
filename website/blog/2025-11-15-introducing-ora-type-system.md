---
slug: introducing-ora-type-system
title: "Introducing Ora's Type System: Zig Meets Rust on the EVM"
authors: [axe]
tags: [type-system, design, compiler, safety]
---

Smart contracts need a type system that's both safe and predictable. After years of working with Solidity's implicit behaviors and Rust's complex ownership rules, we asked: what if we took the best parts of both and built something specifically for the EVM?

That's how Ora's type system came to be. It's Zig-first in philosophy (explicit, no hidden control flow), Rust-influenced in safety (affine types for resources), and EVM-native in execution (every construct maps clearly to bytecode).

<!-- truncate -->

## The Philosophy: Explicit Over Implicit

Ora's type system has one core principle: **if it matters, make it explicit**. 

In Solidity, you can accidentally create two mutable references to the same storage slot. The compiler won't stop you. Ora prevents this at compile time. Every variable declares its memory region: `storage`, `memory`, `calldata`, or `transient`. No guessing, no surprises.

```ora
// Ora: explicit regions everywhere
storage var u256 balance;
memory var u256 temp;
transient var u256 counter;

// Solidity: implicit and risky
uint256 balance;  // Is this storage? Memory? Who knows until runtime.
```

This explicitness extends to everything. Want to know if a value can be copied or must be moved? Check the type. Want to know if a function can modify storage? Look at the signature. There's no hidden behavior.

## Memory Regions: Where Your Data Lives

On the EVM, where data lives matters. A lot. Ora makes this explicit with four memory regions:

- **`storage`** - Persistent contract state (SSTORE/SLOAD)
- **`memory`** - Temporary, local (MSTORE/MLOAD)
- **`calldata`** - Immutable caller input (read-only)
- **`transient`** - Transaction-scoped scratch (EIP-1153)

Every variable declares its region. This prevents the classic Solidity bug where you accidentally modify storage when you meant to work with a local copy.

```ora
contract Token {
    storage var u256 totalSupply;
    
    pub fn mint(amount: u256) {
        // Explicit: this is a memory variable
        memory var u256 newSupply = totalSupply + amount;
        totalSupply = newSupply;  // Explicit assignment back to storage
    }
}
```

The compiler enforces region rules. You can't accidentally create two mutable references to the same storage slot. You can't modify `calldata`. The type system prevents these bugs before they reach the blockchain.

## Affine Types: Move-Only Resources

Rust's ownership system is powerful but complex. Ora takes a simpler approach: affine types for resources that shouldn't be duplicated.

Think of permission tokens, session handles, or proof objects. These are things you want to move, not copy. Ora's affine system is much simpler than Rust's—no borrow checker, no lifetime annotations. Just: "this value moves, it doesn't copy."

```ora
// Affine type: can't be duplicated
affine struct PermissionToken {
    owner: address;
    expires: u256;
}

fn transferPermission(token: PermissionToken) {
    // token is moved here, can't be used again
    // Prevents double-spending of permissions
}
```

For most values (integers, addresses, regular structs), copying is fine. Affine types are opt-in for resources that need move semantics. It's safety where it matters, simplicity everywhere else.

## Traits: Compile-Time Interfaces

Ora's traits are compile-time only. They're not runtime interfaces like Solidity, and they're not dynamic trait objects like Rust. They're Zig-style comptime polymorphism with Rust-like syntax.

```ora
trait ERC20 {
    fn totalSupply() -> u256;
    fn balanceOf(owner: address) -> u256;
    fn transfer(to: address, amount: u256) -> bool;
}

impl ERC20 for Token {
    fn balanceOf(owner: address) -> u256 {
        return self.balances[owner];
    }
}
```

Traits define behavior. Implementations bind behavior to storage. At runtime, traits don't exist—they're erased during compilation. This gives you abstraction without runtime cost.

For external contracts, Ora provides syntactic sugar:

```ora
let token = external<ERC20>(contractAddress);
token.transfer(alice, 100);  // Compiler generates ABI stubs
```

It looks like a trait object, but it's just ABI generation. No runtime conformance checks, no dynamic dispatch. Pure compile-time magic.

The trait system design was heavily influenced by working with [@philogy](https://twitter.com/philogy) and [@jtriley2p](https://twitter.com/jtriley2p). Their insights on compile-time interfaces and EVM-native abstractions shaped how Ora approaches traits.

## Refinement Types: Constraints in the Type

We've written about [refinement types](/blog/refinement-types-in-ora) before, but they're worth mentioning here. Ora's type system supports refinement predicates—constraints that are part of the type itself.

```ora
amount: { x: u256 | x <= self.balance }
```

These refinements are verified at compile time when possible, lowered to runtime guards when necessary. They're erased after verification, so there's no runtime overhead for proven constraints.

This is where Ora's type system gets really interesting: it's not just about preventing bugs, it's about proving correctness.

## No Inheritance, No Subtyping

Ora doesn't have inheritance. No multiple inheritance, no virtual functions, no diamond problems. The type system is flat and predictable.

This is intentional. Inheritance adds complexity and makes code harder to audit. Ora chooses composition over inheritance, traits over base classes. Every type stands on its own.

## How It Compares to Solidity

If you're coming from Solidity, here's what changes:

| What You're Used To | What Ora Does Instead |
|---------------------|----------------------|
| Implicit memory regions | Explicit `storage`/`memory`/`calldata` annotations |
| Storage aliasing allowed | Compiler prevents dangerous aliasing |
| `require()` everywhere | Refinement types + compile-time checks |
| Interfaces at runtime | Traits at compile-time only |
| No generics | Comptime generics (Zig-style) |
| Inheritance | Composition + traits |

Ora is stricter, but that's the point. The compiler catches bugs that Solidity would let through. You write more explicit code, but you get safety guarantees in return.

## Why This Matters

Smart contracts handle real money. A type system that prevents bugs at compile time is worth the extra explicitness.

Ora's type system gives you:
- **Predictability**: Every construct maps clearly to EVM behavior
- **Safety**: Compile-time prevention of common bugs (aliasing, invalid regions)
- **Auditability**: Explicit code is easier to review and verify
- **Correctness**: Refinement types let you prove properties, not just hope they hold

It's not just about catching bugs—it's about building confidence. When the compiler proves something is safe, you don't need to worry about it.

## What's Next

The type system design is documented in our [design documents](/docs/design-documents/type-system-v0.1). It's a working design, evolving as we build the compiler.

We're actively implementing these features. Memory regions are working. Affine types are in progress. Traits are being designed. Refinement types are already functional.

The type system is one of Ora's core differentiators. It's what makes Ora contracts safer, more auditable, and more predictable than what's possible in Solidity today.


---
slug: dropping-solc-and-yul
title: Dropping Solidity and Yul: Moving to a Community Open IR
authors: [axe]
tags: [announcement, compiler, architecture]
---

We're removing all Solidity compiler (`solc`) dependencies and dropping the Yul backend from Ora.  
Instead, we're building a **community-driven, open intermediate representation (IR)** that is not tied to any single language or toolchain.

<!-- truncate -->

## Why We're Dropping Solc and Yul

Ora originally used Solc to produce Yul and then emitted EVM bytecode from there.  
This was fine for bootstrapping, but it has now become a structural blocker.

Here’s why:

### 1. Solc locks us into its release cycle
We can’t iterate faster than Solidity’s compiler does. Every update, feature, or bug fix drags in Solc's entire dependency chain.

### 2. Build complexity is out of control  
Solc brings heavy C++ dependencies that slow down builds, CI pipelines, and contributor onboarding.  
Ora aims to be **fast, lightweight, and portable** — Solc is the opposite.

### 3. Yul isn’t a good fit for Ora  
Yul is tightly coupled to Solidity’s semantics and memory model.  
Ora’s:

- type system  
- control flow  
- memory regions  
- safety guarantees  
- verification model  

do **not** map cleanly to Yul.  
Every abstraction leak becomes a codegen hack.

### 4. No room for innovation
With Solc/Yul, language evolution is bottlenecked by someone else’s constraints.  
We want to:

- design better IR passes,  
- explore new verification workflows,  
- target multiple VMs (EVM, eWASM, FuelVM, etc.),  
- and push compilation research forward.

Yul doesn’t allow that.

The old backend has been moved to `_targets_backup/` for reference, but it's no longer part of the build.

---

## What We're Building Instead

A **community open IR** designed for modern smart-contract languages.

This IR:

- **Is multi-language** — not specific to Ora or Solidity  
- **Is multi-backend** — EVM first, but not EVM-only  
- **Avoids legacy baggage** — no solc, no yul, no forced semantics  
- **Enables real innovation** — better optimizations, better safety, better analysis  
- **Has a clean implementation** — minimal dependencies, easy to extend  
- **Is open to the ecosystem** — other language developers can adopt it  

### Think of it as:
A common low-level platform for smart contract compilers — the LLVM of Web3, but simpler and EVM-aware.

---

## What This Means for Ora Users

### ✔ Cleaner builds  
The project becomes dramatically easier to build and contribute to.  
No more C++ dependency chains.

### ✔ Better optimizations  
A purpose-built IR means we can implement optimizations that were impossible in Yul.

### ✔ Faster iteration  
We control our pipeline top to bottom — no external release cycles.

### ✔ Long-term stability  
We’re no longer tied to the internal design decisions of another language.

---

## What This Means for the Community

### ✔ Open compiler infrastructure  
Anyone can build on this IR — new languages, research projects, and experimental runtimes.

### ✔ A possible standard  
This can become the de-facto IR for next-generation EVM languages.

### ✔ A platform for innovation  
We eliminate the “Solidity monopoly” on compiler infrastructure.  
If the community wants to build something better than Yul, this is it.

---

## Technical Changes Already Done

- Removed all Solc dependencies from `build.zig`
- Deleted the Yul backend (preserved in `_targets_backup/`)
- Cleaned up all MLIR bindings referencing Yul
- Removed CLI flags that referenced Yul or Solc (`--target`, `--save-*`)

---

## What’s Next

We’re actively working on:

- Designing the IR (with community feedback)
- Implementing MLIR lowering into the new IR
- Building the EVM codegen backend from scratch
- Providing documentation so other languages can target the IR

The compiler still supports everything up to code generation:
- lexing  
- parsing  
- type checking  
- semantic analysis  
- MLIR generation  

Codegen will return once the new IR is stable.

---

## Get Involved

If you care about:

- IR design  
- compiler pipelines  
- MLIR  
- verification  
- multi-language targets  
- building a new generation of low-level smart contract tooling  

The repo is open.  
This is not an Ora-only initiative — it’s a community IR project.

Let’s build open, reusable compiler infrastructure for Web3 instead of inheriting the limitations of the past.

---
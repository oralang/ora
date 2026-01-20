# Welcome to the Ora Compiler

Ora is a multi-phase compiler. This guide is designed for **engineers**, not just compiler specialists.

## What success looks like (Day 1)

By the end of your first day, you should be able to:

- run the compiler and emit at least **tokens**, **AST**, and **MLIR**
- localize a bug to a single phase using the “artifact ladder”
- ship one small improvement (test, diagnostic, doc/example, or small internal change)

## The artifact ladder

You’ll learn Ora by looking at what it produces:

**Tokens → AST → Typed AST → Ora MLIR → (verification & passes) → Sensei-IR (SIR)**

This ladder is your map for both development and debugging.

## Lane markers

Throughout the book you’ll see callouts like:

- **Lane A (Contributor):** safe tasks that move the project forward without needing deep compiler knowledge.
- **Lane B (Compiler Engineer):** tasks that touch the compiler pipeline itself.

If you’re not sure which lane to choose, start with Lane A. You’ll naturally graduate into Lane B as you learn the pipeline.


> **Guiding rule:** Ora is a pipeline. Every phase produces an artifact. When something breaks, find the **first** artifact that looks wrong, then fix the phase that produced it.


## Reader promise

This guide is not a Wikipedia page about compilers.

Each chapter is written as:

1) **Goal** (what you are trying to accomplish)  
2) **Action** (what you do)  
3) **What “good” looks like** (expected output/shape)  
4) **If it’s wrong** (what phase to check next)  
5) **Where the code lives** (file pointers)

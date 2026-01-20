---
sidebar_position: 5
---

# Sensei-IR (SIR)

Sensei-IR (SIR) is the intermediate representation used for Ora's backend
lowering toward EVM bytecode. It replaces the earlier EVM-IR documentation and
serves as the target for Ora MLIR lowering.

## Purpose

- Provide a stable backend surface for Ora semantics
- Enable backend optimizations and legalization
- Offer a structured path to EVM bytecode generation

## Current status

SIR lowering is in active development. The front end emits Ora MLIR, and the
backend progressively lowers into SIR as features stabilize.

## Where to look

- Compiler source: `src/mlir/ora/lowering/OraToSIR/`
- Integration plans: `docs/tech-work/sensei-ir-integration-plan.md`


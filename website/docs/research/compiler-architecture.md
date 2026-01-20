---
sidebar_position: 2
---

# Compiler Architecture

Ora is structured as a multi-phase compiler with explicit artifacts at each
boundary. The primary research goal is to keep semantic commitments observable
and testable across phases.

## Phases (conceptual)

1. **Parsing and AST**: syntax and source spans are preserved.
2. **Type resolution**: types and refinements are committed in a typed AST.
3. **Ora MLIR**: semantic lowering into explicit operations and regions.
4. **Sensei-IR (SIR) and backend**: lowering toward EVM bytecode.

## Research emphasis

- **Traceability**: each phase produces artifacts that can be inspected.
- **Explicitness**: regions, refinements, and effects are surfaced in IR.
- **Verification hooks**: proof obligations are represented, not implied.

## Open questions

- How should legality boundaries be enforced across dialects?
- Which constraints should remain SMT-only vs runtime visible?
- What is the minimal IR surface area for formal reasoning?

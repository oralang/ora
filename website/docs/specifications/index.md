# Specifications

Formal specifications and technical references for the
Ora language and compiler. Aimed to be precise, but the
implementation evolves quickly, so treat them as living specs.

## Language specifications

### [Grammar](./grammar.md)
Formal BNF and EBNF grammar for Ora syntax.

### [MLIR Integration](./mlir.md)
Ora MLIR semantics, lowering strategy, and IR structure.

### [Sensei-IR (SIR)](./sensei-ir.md)
Backend IR used for Ora lowering toward EVM bytecode.

### [Formal Verification](../formal-verification.md)
Research-oriented description of the verification model and constraints.

### [API Documentation](./api.md)
Compiler CLI and library interfaces.

### [State Analysis](../state-analysis.md)
Experimental analysis pass for storage access and state effects.

## Status and alignment

For the implemented baseline, see:

- `TYPE_SYSTEM_STATE.md`
- `FIRST_PHASE_COMPLETENESS.md`
- `docs/ORA_FEATURE_TEST_REPORT.md`

For research context and rationale, see:

- [Type System](../research/type-system)
- [Comptime](../research/comptime)
- [SMT Verification](../research/smt-verification)
- [Refinement Types](../research/refinement-types)

These documents are closer to compiler reality than any single narrative page.

## Contributing

If you update a specification, also update the corresponding implementation
notes or tests. Specs should either describe the current behavior or clearly
label the gaps.

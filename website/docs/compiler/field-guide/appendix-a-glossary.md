# Appendix A — Glossary

**Artifact ladder** — Tokens → AST → Typed AST → MLIR → Sensei-IR (SIR).

**AST** — Abstract Syntax Tree. The structured representation of the program produced by the parser.

**SourceSpan** — Location data attached to nodes/tokens so diagnostics can point to precise places in source.

**Semantics Phase 1** — Declaration collection + scope construction used for name lookup.

**TypeInfo** — Type metadata attached to AST nodes after type resolution.

**Typed AST** — The AST after `TypeInfo` is populated.

**Ora MLIR** — The compiler’s internal IR in the Ora dialect.

**Dialect verification** — Structural invariants checked on IR (distinct from SMT verification).

**Legality** — Whether an op/program shape is supported and convertible (in practice, whether it can be converted to Sensei-IR (SIR)).

**Ora → Sensei-IR (SIR) conversion** — Lowering pass that converts Ora IR to Sensei-IR (SIR) IR (backend subset).

**SMT verification** — Constraint solving using a solver (e.g., Z3) to validate requires/ensures/invariants.

**Counterexample model** — A satisfying assignment produced by the solver that demonstrates a violated property.

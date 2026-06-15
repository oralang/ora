---
sidebar_position: 5
---

# Asuka v0.2 — Release Notes

Asuka v0.2 is Ora's proof-carrying contracts release: a compiler pipeline that
keeps Result values, ADTs, SMT verification, ABI lowering, metrics, and CFG
inspection in the normal developer workflow.

## What shipped

- **First-class Result and error-union values**: `Ok`/`Err`
  constructors, `match`, `try`, payloaded errors, multi-error support, public
  ABI reverts, and SMT encoding.
- **Unified ADT model**: product and sum ADTs, enums, error unions,
  source-level constructors, comptime parity, and sema-authoritative
  exhaustiveness.
- **SMT verification in the build**: Z3-backed obligations, counterexamples,
  explain-mode reports, proof capture hooks, vacuity checks, and fail-closed
  degradation handling.
- **Runtime and ABI improvements**: runtime `@abiEncode`, dynamic return ABI
  encoding, dispatcher decode coverage, custom-error selector reverts, and ABI
  layout unification.
- **Comptime expansion**: selector/reflection/encoder builtins, deterministic
  partial folding, and bounded loop unrolling.
- **Trait hardening**: duplicate visible impl detection, extern-call modifier
  rules, extern ABI layouts, and trusted extern summary verification.
- **Debugger, LSP, metrics, and CFG tooling**: source-level EVM debugging,
  production LSP features, `--metrics`/`--time-report`, Graphviz CFG output,
  and `cfg:sir-diff`.
- **Hardening and gates**: fail-closed type resolution, runtime ABI encode
  fixes, dispatcher error-union fixes, MLIR optimization through framework
  passes, conformance tests, SIR snapshots, and property gates.

## Contributing priorities

- More conformance coverage for ABI edge cases, verification examples, and
  multi-contract execution.
- Diagnostics and docs for unsupported shapes that intentionally fail closed.
- Performance profiling for SMT-heavy projects and compiler metrics baselines.
- Documentation updates

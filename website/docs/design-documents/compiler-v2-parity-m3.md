# Compiler V2 Parity M3

## Scope

This note tracks the first parity pass after `M2` made `--v2` usable on real packages.

The current corpus is:

- `ora-example/smoke.ora`
- `ora-example/no_return_test.ora`
- `ora-example/dce_test.ora`
- `ora-example/statements/contract_declaration.ora`
- `ora-example/apps/counter.ora`

For each file, we currently require:

- legacy lowering emits non-empty Ora MLIR
- compiler v2 lowering emits non-empty Ora MLIR
- both Ora MLIR modules convert through the existing Ora->SIR pass
- both paths expose the same lowered `func.func @...` symbol set

## Current Status

The current corpus passes those checks.

That gives us a useful first M3 baseline:

- v2 is not just lowering in isolation
- v2 and legacy both survive the current SIR pipeline on representative contracts
- simple top-level function symbol parity is holding across the selected corpus

## Known Expected Differences

Textual MLIR parity is not expected to be exact.

Current intentional differences include:

- v2 emits ABI/public-entry metadata directly on HIR functions
  - examples: `ora.selector`, `ora.abi_params`, `ora.abi_returns`
- v2 preserves newer sema/HIR correctness work that the legacy pipeline does not model the same way
  - regions
  - refinement flow conversions
  - trait/generic lowering
- unsupported ABI shapes currently skip ABI attrs on v2 instead of failing lowering
  - examples: maps, tuples, structs, enums, error unions

These should be classified as:

- expected improvement
- harmless structural difference

Only semantic regressions should block parity.

## Next M3 Expansions

- widen the corpus to import-heavy and generic-heavy examples
- add explicit parity checks for public entrypoints and storage globals
- record any per-file intentional divergences once they appear in the wider corpus

# SMT `modifies` Corpus

This directory contains named fixtures for the SMT-facing `modifies` and
`@lock` framing work. These files are intentionally small: each one pins one
proof boundary that also appears in unit tests.

Files prefixed with `pass_` must verify without degradation. Files prefixed
with `fail_` must fail for the named reason while still compiling.

Run the full corpus with:

```sh
scripts/check-smt-modifies-corpus.sh
```

The script first runs the source-level sema `modifies` corpus under
`ora-example/corpus/modifies`, then runs the SMT-facing fixtures through normal
full verification. It reruns the map-key, nested-map, and struct-field
internal-frame fixtures with `ORA_VERIFY_MAX_SUMMARY_INLINE_DEPTH=0` to exercise
opaque summary metadata, and checks that the negative controls fail for the
expected reason.

The script also reads each passing fixture's SMT report JSON and enforces a
per-query timing budget. The default budget is intentionally generous
(`ORA_SMT_MODIFIES_MAX_QUERY_MS=5000`) so normal CI noise does not matter, while
large solver regressions fail visibly. Override that environment variable when
triaging slower machines or tightening the corpus budget.

Current coverage:

- Sema-supported path syntax, `modifies()`, unsupported path fail-closed cases,
  and actual-write subset checks.
- Internal callee framing from sema-checked `modifies`.
- Path-precise map-key, nested-map, and struct-field framing.
- Empty `modifies()` interaction with staticcall and unresolved call paths.
- Runtime-lock-based framing across unresolved external calls.
- Negative controls for aliasing, unlocked storage, and unresolved callees.
- Per-query solver-time budget checks for the corpus pass cases.

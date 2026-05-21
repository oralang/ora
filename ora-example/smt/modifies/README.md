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
internal-frame fixtures with `--verify-max-summary-inline-depth=0` to exercise
opaque summary metadata through the public CLI surface, and checks that the
negative controls fail for the expected reason. It also runs an imported-module
fixture with `--verify-imported-summaries-only`, which marks imported call sites
as verifier summary boundaries and consumes their `modifies` metadata without
turning off exact inlining for ordinary internal calls.

The script also reads each passing fixture's SMT report JSON and enforces a
per-query timing budget. The default budget is intentionally generous
(`ORA_SMT_MODIFIES_MAX_QUERY_MS=5000`) so normal CI noise does not matter, while
large solver regressions fail visibly. Override that environment variable when
triaging slower machines or tightening the corpus budget.

The script writes a TSV performance report with per-fixture query counts,
total solver time, max query time, and the active max-query budget. It can also
compare the current run against a previous TSV via
`ORA_SMT_MODIFIES_PERF_BASELINE`, with
`ORA_SMT_MODIFIES_PERF_TOLERANCE_MS` controlling the allowed max-query drift.
This is the current performance-regression base for the SMT/modifies corpus.

Future work for broader verifier performance infrastructure:

- Historical storage: CI should persist or upload the TSV report somewhere
  durable.
- Trend dashboard: there is no charting or multi-commit comparison yet.
- Broader corpus coverage: this script tracks the SMT/modifies corpus, not the
  whole verifier suite.
- More metrics: no p95/p99, solver `UNKNOWN` count, fragment breakdown,
  SMTLIB byte totals per fixture, or per-query labels are emitted in the TSV.
- Regression policy: baseline comparison currently checks max query time, not
  query-count growth or total-time growth.
- Noise handling: tolerance is a simple absolute millisecond threshold, not a
  percentage threshold or rolling baseline.

Future work for precision-report UX:

- Add an encoder-level regression proving precision-note event provenance is
  populated from a real call site, not only from synthetic report-renderer
  fixtures.
- Deduplicate identical precision-note events before they consume the bounded
  report slots.
- Keep function-level attribution as the current v1 boundary. Exact
  query/proof attribution would require finer encoder provenance and is deferred
  until it becomes necessary.
- Keep both the top-level `precision_note_events` array and per-error
  `precision_context_events`: the former is the canonical complete list, while
  the latter is filtered context for human-facing diagnostics.

The corpus includes the focused fixtures in this directory and at least one
larger application fixture with real user-facing contracts annotated where the
current v1 path syntax can express the write set.

Current coverage:

- Sema-supported path syntax, `modifies()`, unsupported path fail-closed cases,
  and actual-write subset checks.
- Internal callee framing from sema-checked `modifies`.
- Path-precise map-key, nested-map, and struct-field framing.
- Imported-module summary-boundary framing through `ora.imported_call`.
- Empty `modifies()` interaction with staticcall and unresolved call paths.
- Runtime-lock-based framing across unresolved external calls.
- Negative controls for aliasing, unlocked storage, and unresolved callees.
- Per-query solver-time budget checks for the corpus pass cases.
- A larger ERC20 verification fixture with `modifies` annotations on initializer
  and mint write sets.

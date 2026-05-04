# Ora v2 — Release Report

**Range:** `a9ade6b` (Update Release) → `54f892e` (Fix debugger DAP breakpoint replay)
**Scope:** 147 commits · 185 files · +56,800 / −20,574 lines

This report groups the work into the major themes that landed for v2. Each theme lists the user-facing additions ("New features"), the correctness work ("Fixes & hardening"), and the supporting infra so reviewers can see how broad each change is.

---

## 1. EVM Debugger — first-class shipping product

The largest single workstream. The debugger went from a CLI probe + bare TUI to a productized tool with a DAP backend, compiler-aware overlays, and external docs.

### New features

- **DAP server (`ora-evm-debug-dap`)** — Content-Length framed JSON-RPC over stdio. Implements `initialize`, `launch`, `setBreakpoints`, `setExceptionBreakpoints`, `threads`, `stackTrace`, `scopes`, `variables`, `evaluate`, `continue`, `next`, `stepIn`, `stepOut`, `pause`, `disconnect`. Wired to root build and `ora debug --dap` (`build.zig:+16`, `lib/evm/build.zig:+53`, `lib/evm/src/debug_dap.zig:805`, `lib/evm/src/jsonrpc.zig:330`).
- **Storage watchpoints** — `:watch <binding-or-slot>` halts on SSTORE that changes the slot.
- **Conditional & hit-count breakpoints** — `:break <line> when <expr>`, `:break <line> hit <n>`, backed by a side-effect-free expression evaluator (`lib/evm/src/debug_eval.zig:343`).
- **ABI revert / event decoding** — Decoded `ErrorName(field: value, …)` on revert; decoded events on `LOG*`. Per-frame ABI lookup via cross-frame ABI registry.
- **Compiler-aware overlays** — Toggleable gutter overlays for folded values, hoist origins, region/iteration counters, gas heatmap, coverage, and **FV-proven dead branches** (UNSAT proofs from the verifier surface as dimmed source lines). Inlined-from chain rendering when the compiler emits it (`lib/evm/src/debug_overlay.zig:84`).
- **Coverage + gas heatmap** — Per-line execution counts and cumulative gas; `:cov export lcov` for VS Code Coverage Gutters.
- **EIP-3155 trace export** with Ora-decoded layer; richer backtrace.
- **CLI-configurable limits** — `--gas-limit`, `--max-steps`, `--deploy-step-cap`, `--artifact-max-bytes` on probe and TUI.
- **Replay determinism** — `SessionSeed.BlockEnv` pins `timestamp`, `number`, `blockhash`, `chain_id`, `coinbase`, `base_fee`, `prev_randao`, `gas_limit`. Two runs of contracts reading block opcodes now produce byte-identical traces.

### Correctness & hardening

- **Hash-indexed lookups** — `DebugInfo` and `SourceMap` previously linear-scanned per step/per render. Now `AutoHashMap` indices built once at load. `shouldIgnoreStatementBoundary` rewritten from O(N²) to constant-time. Bench harness reports < 100 µs/step (`lib/evm/test/bench/step_bench.zig`).
- **Unified stepping engine** — Four near-duplicate stepping loops collapsed into `runUntil(condition: StopCondition)`.
- **Quiet-failure cleanup** — `step_history.append`, `appendCommandLog`, `applyBreakpoints`, `pollWatchpoints` now surface failures to the status line; `instrumentation_degraded` sticky flag with reason text.
- **DAP correctness** — JSON escape helpers everywhere (no protocol corruption), explicit malformed-message responses, opt-in `ORA_DAP_LOG`, 16 MiB Content-Length cap, breakpoint-replay diff (the v2 tip commit fixes a bug where `setBreakpoints` didn't actually clear old breakpoints), step failures emit error responses.
- **Front-end-agnostic core** — `DebugController` extracted; DAP and TUI both route eval through it (`lib/evm/src/debug_controller.zig:98`).

### Architecture & docs

- **Module split** — `debug_session`, `debug_abi`, `debug_breakpoint`, `debug_overlay`, `debug_eval`, `debug_controller`, `jsonrpc`, `debug_tui_session`, `debug_tui_draw` extracted out of the 4716-LOC `debug_tui.zig`.
- **Dispatch tables** — `handleKey` and `executeCommand` driven by `(key, action)` and `(name, handler, help)` tables; `:help` is generated.
- **External docs** — `KEYBINDINGS.md`, `COMMANDS.md`, `README.md`, `getting-started.md`, `debug-a-failing-transaction.md` tutorial, `CHANGES.md` summary, `a3-snapshot-stepback-plan.md` and `b2-inliner-plan.md` for blocked work.
- **Test coverage** — `debugger_test.zig` (888 LOC) covers stepping, breakpoints, scope/binding, storage round-trip, replay determinism. Branch coverage of `debugger.zig` > 70 %.

---

## 2. Algebraic Data Types & pattern matching

A complete sum-type story landed: source-level constructors → AST → HIR → SMT datatypes → MLIR → SIR.

### New features

- **`Result` type** — Constructors and match lowering, public-ABI lowering, payload-error lowering, multi-error matching, named-error matching, helpers, dynamic-bytes input, struct flow, stateful flow, discard patterns, roundtrip.
- **Sum-match exhaustiveness** — Unified across enum/error-union/Result; warns on wildcard sum matches.
- **Or-patterns for enums** (`enum_or_patterns.ora`) and **storage enum flow**.
- **Explicit string and bytes enum values** end-to-end.
- **Comptime ADT** — Result match in comptime, enum discriminant expressions, ADT payload value persistence.
- **Phase 2a SMT datatypes** — Enums and error-unions migrated from product fallback to true SMT datatypes; **Phase 2b ADT IR** support.

### Fixes & hardening

- Wide ADT decode for handle-pointer inputs.
- Wide error-union catch propagation.
- Multi-error try propagation encoding.
- Named error payload roundtrip.
- Legacy error-return paths and enum constants with repr-backed storage.
- Recursive runtime ADTs rejected (slice-indirected recursive structs covered).
- Anonymous struct identity preserved through Phase 1 lowering.
- Source ADT constructors lowered through SIR; product coercion + map array state threading fixed.

### Refactors

- `convertOraReturn` split into per-type handlers.
- `ConvertUnrealizedConversionCastOp` split into named handlers.
- Shared ADT carrier helpers extracted; named helpers for ADT handle malloc/load.
- Centralised OraToSIR materialization-kind constants in one header.
- Phase 5 OraToSIR cast-cleanup walks collapsed; dead ADT patterns dropped.

---

## 3. SMT verification & explain mode

The verifier became substantially more honest, observable, and sound.

### New features

- **Explain-mode unsat cores** — `--explain` CLI flag; per-obligation cores cite ghost axioms, callee obligations/ensures, user assumptions, environment & frame assumptions, path guards, and imported-callee provenance.
- **Optional Z3 proof generation** plumbed; **optional unsat core minimization**.
- **Aggregate refinement guards** — Struct-field proofs, ADT-payload proofs, NonZeroAddress.
- **Phase 5 refinement aggregate validation** closed.
- **Imported callee tracking** — Obligations and ensures from imported callees flow into cores with provenance preserved.
- **Bounded loop unrolling** — Constant `for`, bounded `while`, labeled `while`, nested unroll provenance & policy.
- **`proven_guard_positions`** recorded on `VerificationResult` so the debugger can dim FV-dead branches.

### Fixes & hardening

- SMT width discipline tightened in encoder; coercion fixes; checked-power overflow encoder regression.
- Parallel SMT reload + Z3 soundness checks.
- Verification fail-closed on partial write-set recovery, struct metadata gaps, read-set, try-statements, tuples MLIR, dynamic bytes lowering.
- Dead verifier fallback removed; explain mode kept on canonical verifier path.
- Pure helper calls inlined consistently in SMT encoding.
- Control-flow summary fixes; verification replay unified.
- Error-union summary verification state corrected; scalar error-union memref normalization.
- Bounded SMT degradation reasons tracked; degradation diagnostics tightened.
- SMT axiom provenance + path guarding fixed.

### Test infrastructure

- `compiler.test.verification.zig` (1330 LOC) and `compiler.test.sema_verify.zig` (646 LOC) seeded.
- Fail-closed regression baseline expanded across SMT (`fail-closed/*.ora` corpus + verification corpus).
- Phase 1 product SMT coverage tightened; product SMT fallback paths retired in favour of hybrid SMT product sort.

---

## 4. Comptime evaluation

### New features

- **Phase 1 + Phase 2 partial folding** locked in.
- **Comptime nested struct field updates**.
- **Comptime enum discriminant expressions**.
- **Comptime Result ADT matching**.
- **Comptime ADT payload persistence**.
- **Limit diagnostics** — clean errors on comptime budget exhaustion.
- **Invalid comptime aggregate refinements rejected**.
- **`compiler.test.comptime.zig` (3939 LOC)** — comprehensive comptime test surface.

---

## 5. MLIR / OraToSIR

### New features

- **Tuples in MLIR** — divisions builtins corpus, full lowering.
- **Refinement guards for ADT match** allowed end-to-end.

### Refactors & hardening

- `ControlFlow.cpp` reworked (+3117 lines/-changes).
- `OraToSIRTypeConverter.cpp` rebalanced; `SIRDispatcher.cpp` reorganised.
- New per-pattern files: `Arithmetic.h/.cpp`, dedicated `Storage.h`, `Struct.h`, `MissingOps.h`.
- `OraOps.td` and `OraTypes.td` extended.
- `OraDialect.cpp` and `OraCAPI.cpp` updated for the ADT/tuple work.
- OraToSIR + verification query handling refined.

---

## 6. Stdlib

- **`std/bytes.ora`** — bytes/string `len`/`index` helpers.
- **`std/constants.ora`** — well-known constants module.
- **`std/result.ora`** — Result helpers exported.
- **`std/std.ora`** — top-level re-exports updated.
- **`stdlib_embedded.zig`** updated to ship the new modules.

---

## 7. Test infrastructure overhaul

`src/compiler.test.zig` (was 13,931 LOC monolith) split into focused topic files. Net: more tests, faster builds, clearer ownership.

| New file | LOC |
|---|---|
| `compiler.test.comptime.zig` | 3939 |
| `compiler.test.hir.zig` | 2822 |
| `compiler.test.match.zig` | 2672 |
| `compiler.test.misc.zig` | 2671 |
| `compiler.test.sema_regions.zig` | 1602 |
| `compiler.test.traits.zig` | 1556 |
| `compiler.test.verification.zig` | 1330 |
| `compiler.test.diagnostics.zig` | 870 |
| `compiler.test.oratosir.zig` | 653 |
| `compiler.test.sema_verify.zig` | 646 |
| `compiler.test.sema_infra.zig` | 496 |
| `compiler.test.syntax.zig` | 433 |
| `compiler.test.abi.zig` | 379 |
| `compiler.test.helpers.zig` | 364 |
| `compiler.test.debug_artifacts.zig` | 164 |

Plus: debug-artifact regression corpus (`lib/evm/test/specs/debug_artifacts/`) — golden `sourcemap.json` and `debug.json` per fixture, byte-equal asserted in CI.

---

## 8. ABI

- `src/abi.zig` substantially updated (+255 LOC); new `src/abi.test.zig` (195 LOC) seeded.
- Result/error-union ABI lowering reworked end-to-end.

---

## 9. Sema, HIR, parser

- **`sema/type_check.zig`** (+2726 LOC) — major work supporting Result, ADT, refinement aggregates.
- **`hir/control_flow.zig`** (+1125), **`hir/expr_lowering.zig`** (+1115), **`hir/module_lowering.zig`** (+713), **`hir/function_core.zig`** (+622), **`hir/mod.zig`** (+333) — broad HIR work for ADT/match/Result/comptime.
- **AST** — new walker (`ast/walk.zig`, +119), node additions, syntax lowering improvements (+250 LOC).
- **Parser** — explicit string/bytes enum values, ADT support, +227 LOC.

---

## 10. Documentation

- Website `docs/` — intro, getting-started, language-basics, imports, debugger, ora-vs-solidity, asuka roadmap and base document refreshed.
- New compiler docs: `docs/compiler/smt-audit/todo_unsat_cores.md` (450 LOC) and `implementation_team_smt.md` updates.

---

## Risk areas / known follow-ups

- **A3 — snapshot-based stepBack** (debugger): plan written, implementation deferred. Today `stepBack` re-runs from the start.
- **B2 — MLIR inliner** (compiler-side): plan written; the `inlined_from` schema slot exists and the TUI renders it, but the compiler doesn't yet emit the chain because the inliner isn't wired.
- **DB layer (`src/db/mod.zig`)** is not in this range but is the next obvious blocker for an LSP-paced workflow. See `src/db/REVIEW.md` (separate review).

---

## Bottom line

v2 ships:
1. A productized EVM debugger with DAP, decoded traces, watchpoints, conditional breakpoints, FV overlays, coverage, gas heatmaps — and external docs.
2. A real ADT + pattern-matching story end-to-end (source → SMT → MLIR), including `Result`.
3. An SMT explain mode with cores, provenance, and proof-aware refinements.
4. Bounded loop unrolling and a much stronger comptime evaluator.
5. A test split and large new corpus — every theme above has regression coverage.

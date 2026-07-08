# Quality Review Ledger — Formal Lane

Retroactive quality audit, 2026-07-07, covering the canonical-Z3/agreement-bridge
commit arc (~91b6c85b..e5943bb1). Prior reviews in this range verified
soundness/identity/polarity at line level; bloat, smells, and Lean-model
dimensions were audited retroactively and their findings live here.
**This file is the tracked work queue for those findings. Do not delete;
strike items through with the fixing commit hash.**

## Q17 (CRITICAL, found during Q1 implementation, FIXED in this commit)

`src/formal/obligation_from_z3.zig` and `src/formal/obligation_to_z3.zig`
in-file tests were **never collected by any test target** — Zig only collects
tests from files referenced in a `test`/`comptime` block, and no root
re-exported them. The promotion-table exhaustiveness pin, classifier matrices,
order-contract test, crosscheck fixtures, and Slice B identity tests had never
been built or executed; validation counts reported for those slices matched
same-named tests elsewhere. Fixed by the formal-lane import block in
`src/compiler.test.zig` (all 9 test-bearing formal files). On first-ever
execution all orphans PASSED (suite 1394 -> 1405) — the tests were correct,
only collection was broken. RECORD CORRECTION: slice B/C review validations
that cited these tests' names did not actually execute them.
Residual: HEAD-state validation of the collected orphans rides the first full
gate after the in-flight slice commits (WIP tree validated; worktree validation
blocked on submodules).

## Interrupt items — land BEFORE the next commit touching these files

- [x] **Q1 (FIXED this commit)** `obligation_from_z3.zig:334` `logicalRole()`:
  `stringToEnum(...) orelse unreachable` on a label produced by
  `z3_verification.formalLogicalRoleLabel` — a separately-maintained vocabulary
  in another module. Verified latent today (all 7 produced labels exist in
  `LogicalRole`), UB-on-drift tomorrow. Fix: error return + a lockstep pin test
  (inline-for over `formalLogicalRoleLabel` outputs; assert each parses into
  `obligation.LogicalRole`).
- [x] **Q2 (FIXED 87a90c70)** `obligation_to_z3.zig:1146`
  `canonicalQueryState()`: `query_state orelse unreachable` reachable through
  `pub` `encodeFormula`/`encodeTerm` if any external caller skips
  `appendQueryConstraints`. Fixed by returning a typed
  `MissingCanonicalQueryState` error and pinning direct `encodeFormula` misuse
  with a regression test.
- [x] **Q3 (FIXED this commit)** `formal/Ora/Obligation/Agreement.lean`:
  state the `canonicalEnv` invariant in a comment — denotability is
  environment-independent *given* an env total over the term arena's free
  variables; the fold provides exactly that totality. The module's correctness
  argument must live in the file.
- [x] **Q4 (FIXED this commit)** `formal/Ora/ObligationTotalitySync.lean`
  decoders: add per-tag decode pins (every Term/BinaryOp tag decodes to its
  expected constructor). Closes the tag-swap channel where the gate stays green
  while pinning the wrong term. Bounded risk (real proof fixtures exercise true
  semantics) but closeable for pennies.
- [x] **Q18 (FIXED 87a90c70)** `goal_skolem.zig`: goal-skolem name buffers must
  be structural, not caller-discipline. Fixed by exporting comptime-sized
  buffer constants, requiring pointer-to-sized-array callers, and proving the
  maximum generated name fits at comptime.
- [x] **Q19 (FIXED 87a90c70)** `obligation_to_z3.zig`:
  `encodeFunctionParamForallWrapper` must not silently erase a hypothetical
  wrapper condition. Fixed by rejecting conditioned function-param wrappers
  with the named `UnsupportedFunctionParamWrapperCondition` path and classifier
  reason.

## Structural — schedule as its own slice, design note first

- [ ] **Q5 (the strategic one)** `obligation_to_z3.zig`: five hand-synchronized
  term-tree walkers (`termCanonicalSupport` ~490, `collectTermPromotionFeatures`
  ~342, `termContainsResult` ~587, `staticTermTypeInfo` ~664,
  `Adapter.encodeTerm` ~1070) — same ten-variant switch, same fuel guard, five
  coordinated edits per new `Term` variant. Centralize guard + variant dispatch
  behind one visitor seam. **No new walker-growing slice starts in this file
  until this lands.** Mini design note first (visitor shape, per-walker leaf
  logic stays).
- [ ] **Q6** After Q5: split the file (2631 lines, four responsibilities) into
  canonical_support / canonical_encode / canonical_types, tests following each.

## Cleanup batch — one commit, no design needed

- [ ] **Q7** Generic `findById(slice, id)` in `obligation.zig` — six duplicated
  linear scans (`obligation_to_z3.zig:448,455,965,972,979`;
  `obligation_from_z3.zig:189`; `proof_check.zig:956`). Also the quadratic seam
  (find-per-id inside per-id loops; overlay is O(queries×rows) with nested
  scans) — fine at per-function manifest sizes, watch on growth.
- [ ] **Q8** Shared `formal_test_fixture.zig`: the single-obligation
  set-builder boilerplate is repeated across `obligation_to_z3.zig:1714-1875`,
  `obligation_from_z3.zig`, `proof_check.zig` tests — hundreds of lines.
- [ ] **Q9** One `kindFormula(kind) ?FormulaRef` helper replacing the
  8-kind unsupported arm-list repeated at `obligation_to_z3.zig` ~317, ~466,
  ~990, ~1006.
- [ ] **Q10** `encodeVariable`/`encodeStorageSymbol` call `namedConst` for the
  dupeZ→symbol→const→check tail (idiom triplicated ~1166, ~1185, ~1294).
- [ ] **Q11** `requireSupportedScalarPlace(place)` helper for the
  support→error mapping inlined 5× (~520, ~701, ~1097, ~1117, ~377).
- [ ] **Q12** `emit_obligation_totality_snapshot.zig:247` `writeRawTerm`:
  10 positional args → params-struct with defaults. (The `else => default`
  column-filler at ~452-471 is intentional, not fail-open — leave it.)
- [ ] **Q13** `proof_check.zig:1491` test re-inlines `leanTestProcessEnviron`
  (helper exists at 1344) — call it.
- [ ] **Q14** Consolidate id-list writers (`proof_check.zig:370,622`;
  `emit_obligation_totality_snapshot.zig:96`) and id-equality
  (`proof_check.zig:975` vs `obligation.equalIdSlices`) into `obligation.zig`.
- [ ] **Q15** Drop single-variant `LeanNatListMode` enum + param
  (`proof_check.zig:564`).
- [ ] **Q16** Extract `applyRowMetadata(*query, row)` from
  `overlayPreparedQueryResults` (~100-line body; metadata copy duplicated with
  `collectPreparedQueries`).

## Audited clean
`state_symbols.zig`, `runtime_checks.zig`, `measure-canonical-z3-corpus.py`.
Fail-closed discipline held across the whole range: no `catch {}` swallows, no
silent defaults, errdefer/allocator hygiene correct. The debt is cohesion, not
safety.

## Process (standing)
- Review verdicts carry three explicit sub-verdicts: Soundness / Code Quality /
  Lean Model — each with evidence or a named skip.
- Accretion audit (quality-auditor sweep over touched files) every ~5 slices;
  findings land here.
- Approvals in the audited range (quantifier parity and earlier parity slices)
  were soundness-scope; quality was retroactively audited 2026-07-07 into this
  ledger.

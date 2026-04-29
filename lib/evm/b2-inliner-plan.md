# B2: inlined-from chain — compiler-side implementation plan

The TUI render path (`Ui.currentInlinedFromChain`,
`Ui.appendInlinedChain`, `:where` integration) and the schema
slot (`SourceScope.inlined_from: []const InlinedFrame` in
`lib/evm/src/debug_info.zig`) shipped earlier — see commits
`1d27b891` and `9ec8c81c`. They're dormant because no compiler
pass populates the chain yet. This document is the gating spec
for that compiler work.

## Why nothing populates today

Verified by reading `src/`:

- `src/mlir/ora/lowering/OraToSIR/OraToSIR.cpp:2410` has a
  `// TODO: requires MLIR's InlinerInterface (not yet wired
  up).` comment. No `InlinerInterface` is registered for the
  Ora or SIR dialects.
- `src/z3/encoder.zig:208 inline_function_stack` is SMT-only
  (verifier inlines callee summaries for reasoning); doesn't
  rewrite emitted MLIR or affect source locations.
- `src/hir/module_lowering.zig:513` attaches `ora.inline =
  true` as a *hint* on imported private helpers but no
  downstream pass reads it as a directive to actually inline.

Result: every `func.call` in MLIR-SIR stays as a real call,
runtime EVM stays multi-frame, and `SourceScope.inlined_from`
is uniformly empty.

## Implementation paths

### Path A — MLIR InlinerInterface (recommended)

Wire MLIR's standard `InlinerInterface` for the Ora and SIR
dialects, then run the `inline` pass selectively.

1. **Register InlinerInterface for SIR**
   `src/mlir/ora/lowering/OraToSIR/SIRDialect.cpp` — add a
   `SIRInlinerInterface : public DialectInlinerInterface`
   override with at minimum:
   - `isLegalToInline(Region *, Region *, bool, IRMapping &)` —
     return true for everything outside untrusted untyped
     regions.
   - `isLegalToInline(Operation *, Region *, bool,
     IRMapping &)` — same.
   - `handleTerminator(Operation *, Block *)` — handle
     `sir.iret` by replacing with branch to the rewriter's
     dest block. Other terminators (cf.br, scf.yield) use
     base behavior.
   ~30-50 LOC.

2. **Annotate which calls are eligible**
   `src/hir/module_lowering.zig` already sets `ora.inline =
   true` on imported helpers. Extend the heuristic: also mark
   any `private fn` whose body is small (< K SIR ops) and
   whose call sites are countable. ~20 LOC.

3. **Run MLIR's inline pass**
   In the OraToSIR pipeline (`src/mlir/ora/lowering/OraToSIR/
   OraToSIR.cpp` after the lowering completes), add the
   `mlir::createInlinerPass()` to the pass manager, gated by
   the `ora.inline` attribute on functions. ~10 LOC.

4. **Track inlined_from at the call site rewrite**
   When MLIR's inliner clones the callee body into the
   caller, it can be configured to attach a `loc(callsite)`
   chain via `mlir::CallSiteLoc`. The SIR text emitter
   (`src/mlir/ora/lowering/OraToSIR/SIRTextEmitter.cpp`)
   already walks op locations; extend it to recognise
   `CallSiteLoc` and emit the chain into each scope's
   `inlined_from` array in `debug.json`. ~80 LOC.

5. **Refresh goldens**
   Add a regression fixture under `tests/debug_artifacts/`
   with a small inlined helper. Verify the emitted
   `inlined_from` chain has the expected outermost-first
   ordering matching the doc on `InlinedFrame`.

Estimate: 2–3 focused days. Path A reuses MLIR's mature
inliner infrastructure; the bulk of work is the dialect
interface and the tracking-through-emit.

### Path B — HIR-level call substitution (alternative)

If MLIR's InlinerInterface proves problematic for the SIR
type system, do the inlining one level up:

1. In `src/hir/module_lowering.zig`, after HIR is built, add
   a pass that walks every `Call` HIR node and, when the
   callee is marked `ora.inline`, substitutes the callee's
   HIR body inline.
2. While doing so, push the call site onto a per-region
   `inlined_from` stack.
3. When the lowering emits MLIR ops, attach the stack to the
   ops' source-scope metadata.
4. The SIR text emitter picks up the stack and writes
   `inlined_from` into the JSON.

Estimate: 3–5 focused days. Path B avoids MLIR-side
complexity but reimplements logic MLIR's inliner already
provides. Recommended only if Path A hits walls.

### Path C — Defer entirely (status quo)

Keep the schema slot empty. The TUI's render path stays
dormant. Document the limitation in `getting-started.md` so
users don't expect inlining to be visualized. No code change.
This is what's currently shipped.

## Acceptance for B2-close

- A regression fixture under `tests/debug_artifacts/` whose
  `debug.golden.json` has at least one source scope with a
  non-empty `inlined_from` chain.
- `:where` in the TUI surfaces the chain on a stop inside
  the inlined region. Verified by manual screenshot or
  in-process test against the fixture.
- The MLIR-level inliner is a real pass run during normal
  compilation (not a debug-only artifact path).

## Why this isn't closable in one debugger session

Path A is multi-day because each of the three MLIR pieces
(dialect interface, pass scheduling, location tracking) is
its own learning curve for a Zig+C++ codebase, and none of
them have existing precedent in `src/mlir/ora/lowering/`.
Path B is also multi-day because HIR substitution requires
careful handling of binding scope (a callee's locals
shadowing caller's) plus the same emit-side plumbing.

The TUI side is in place. The blocker is genuinely the
compiler-side transformation, not the debug-info layer.

# Ora EVM debugger — branch summary for review

Branch: `v0.2`. 138+ commits ahead of `main`. ~7k LOC added
under `lib/evm/` (25 files touched). The bulk of this document
is the debugger work; the branch also carries unrelated
compiler-side changes (ADT/IR fixes, OraToSIR refactors) that
aren't covered here. `lib/evm/src/debug_tui.zig` sits at ~5485
LOC after the partial split — see "Stopped early" below for
why phase 3 wasn't worth the churn, and "Reviewer round" for
the architectural split that did land.

**Headline metric:** 551/551 lib/evm unit tests pass; top-level
`zig build` and `zig build test-compiler` both green; bench
`step_bench` reports ~26 ns/step (budget 100 µs).

## How the work was organized

The branch follows a four-track plan
(`/Users/logic/.claude/plans/gentle-exploring-comet.md`):

- **Track A** — stability foundation (tests, perf, determinism,
  configurability).
- **Track B** — compiler-side debug info (regression corpus,
  inlined frames, statement-level liveness).
- **Track C** — differentiator features (watchpoints, decoded
  reverts, eval, conditional bps, overlays, coverage, trace
  export, DAP).
- **Track D** — docs and module split.

## Closed end-to-end

Everything below is shipped, tested, and documented.

### Track A — stability

| Item | Commit | What |
|---|---|---|
| A0 | `23700094` | Extracted `lib/evm/src/debug_session.zig` shared by probe + TUI. Constants `kArtifactMaxBytes`, `kDefaultGasLimit`, `kDeploymentStepCap`, `kDefaultMaxSteps`, `DebugLimits`. |
| A1 | `b46d0535` | New `lib/evm/src/debugger_test.zig` — 17 unit tests on stepIn/Over/Out/Back, breakpoints, statement-boundary skip, replay determinism, scope/binding round-trip. |
| A2 | `e12e06a3` | Hash-indexed `DebugInfo` and `SourceMap` lookups. `op_meta_by_idx`, `op_visibility_by_idx`, `scope_by_id`, `local_by_id` populated at load time. `SourceMap.FileIndex` indexes lines + statement-id lookups. Lone-invalid set precomputed. |
| A4 | `eef7781a` | `BlockEnv` pinned constants in `SessionSeed` so `TIMESTAMP`/`BLOCKHASH`/etc. are deterministic. Replay-determinism test. |
| A5 | `f6d98cf3` | `--gas-limit`, `--max-steps`, `--deploy-step-cap`, `--artifact-max-bytes` CLI flags on probe + TUI. `DebugLimits` struct. |
| A6 | `edfee119` | `lib/evm/KEYBINDINGS.md`, `lib/evm/COMMANDS.md` authored from current behavior; `:help` is generated from the same dispatch table. |
| A7 | `cc3a822e` | `handleKey` → `(key, action) → handler` table. `executeCommand` → `(name|prefix, handler, help)` table with exact + prefix matching. `runStep` collapsed via `runDebuggerCommand`. |
| A8 | `402e5f9d` | Surface OOM warnings on `step_history.append` and `appendCommandLog`. `applyBreakpoints` continues past per-line failures with a count. |

### Track B — compiler-side debug info

| Item | Commit | What |
|---|---|---|
| B1 | `5c62047b` | New `tests/debug_artifacts/` regression corpus driven by `src/compiler.test.debug_artifacts.zig`. `simple_counter` fixture + golden artifacts; byte-equality assertion gates compiler changes. Later joined by `liveness_dead_after_use` (B3). |
| B3 | `dcc11153` + `31c7550c` | New `src/ast/walk.zig:collectNamesInExpr` — recursive name-ref walker over all 26 `Expr` arms. New `computeNarrowedLiveEnd` in `src/main.zig` narrows `Local.live.end` from "end of body" to "end of last same-block use". Conservative fallback when later statements are body-bearing (no stmt walker yet). Verified narrowing on `liveness_dead_after_use` fixture. |

### Track C — differentiator features

| Item | Commit | What |
|---|---|---|
| C1 | `b37371b2` | Storage watchpoints. `Watchpoint` struct on `Debugger`, `addWatchpointBySlot`, `addWatchpointByBindingName`, `pollWatchpoints` after every opcode. New `:watch <slot|binding>`, `:unwatch <id>`, `:info watch`. New `StopReason.watchpoint_hit`. |
| C1b | `7935014c` | ABI-decoded reverts and events. `AbiDoc.findErrorBySelector`, `findEventByTopic0`. When `stop_reason == .execution_reverted`, status line decodes `frame.output` against the loaded ABI: `InsufficientBalance(have=0, need=100)` or `Error("...")` for Solidity-style. New `:print logs` walks `evm.logs`, decodes against event topic0. |
| C2 | `7935014c` | New `lib/evm/src/debug_eval.zig` — recursive-descent evaluator over numeric literals, visible bindings, `+ - * / %`, comparisons, `&& || !`, parens. Side-effect-free. `:eval <expr>`. 4 unit tests. |
| C3 | `7935014c` + `ee68c7ec` | Conditional + hit-count breakpoints. `:break <line> when <expr>` and `:break <line> hit <n>` (composable). Implemented as a TUI-layer gate: debugger core stays simple, TUI transparently resumes when the predicate fails or the hit target hasn't been reached. Sessions persist conditions and hit targets. Parser extracted to `lib/evm/src/debug_breakpoint.zig` with 9 unit tests. |
| C4 | `81914621` + `4d8aecbb` | Cross-frame ABI registry. `:bt` shows per-frame address, `static` flag, decoded function name when calldata selector matches. `--abi <hex-address>=<path>` repeatable flag binds external-callee ABIs. `Ui.secondary_abi` keyed by 20-byte address. |
| C5 (folded/hoist) | `f9a4d108` | Compiler-aware overlays for folded literals and hoist origins. `OverlayMode` extended; gutter shows `=<value>` for folded source declarations and `<-N` (origin_statement_id) for hoisted lines. |
| C5 (FV-dead-branch) | `94adb15e` + `a8566c90` | End-to-end FV-dead-branch overlay. `OpMeta.proof_status` schema slot. `ProvenGuardPosition` on `VerificationResult`; verifier records source positions for every Obligation it proves UNSAT. New `writeProofSidecar` writes `<stem>.proof.json` next to debug.json. TUI loads sidecar at init, populates `Ui.proof_lines`. New `OverlayMode.fv` renders ` S` / ` X` / ` ?` per line. **Smoke-tested against `arithmetic_test.ora` — 4 UNSAT proofs at lines 14/29/31/37 land in sidecar exactly.** |
| C6 | `d0c0a428` + `715b753b` + `5d944514` | Coverage + gas heatmap. `Debugger.line_hits: AutoHashMap(u32, u32)` bumped on every distinct statement-line transition. `Debugger.line_gas: AutoHashMap(u32, u64)` accumulates per-opcode gas attributed to `last_statement_line` (skipped when no source-map entry — correctly excludes callee opcodes). New `:cov [N]`, `:cov export lcov <path>`, `:gascov [N]`. Two new `OverlayMode` columns (`coverage`, `gas`). |
| C7 | `81914621` | EIP-3155 trace export. `:trace export <path>` builds a shadow `Session` from the seed, applies live breakpoints, attaches `Tracer`, replays `step_history`, writes the JSON. Doesn't disturb live session. |
| C8 | `ab563378` + `c6243137` + `c45f27c9` | DAP server (new binary `ora-evm-debug-dap`). Implemented requests: `initialize`, `launch` (takes pre-compiled artifact paths), `setBreakpoints` (stash + replay), `threads`, `stackTrace`, `continue`/`next`/`stepIn`/`stepOut`, `pause`, `disconnect`. Smoke-tested against `simple_counter` artifacts. **Known DAP gaps explicitly excluded from this round** — see "Deferred" below. |

### Track D — docs and module split

| Item | Commit | What |
|---|---|---|
| D1 | `98b22b1a` | `lib/evm/README.md`, `lib/evm/getting-started.md` (5-min walkthrough on counter contract). |
| D2 | `d0c0a428` | `lib/evm/debug-a-failing-transaction.md` — end-to-end tutorial exercising decoded reverts, watchpoints, conditional breakpoints, `:eval`. |
| D3 (phase 1) | `42072432` | `lib/evm/src/debug_tui_session.zig` — `SavedSession`, `writeSession`, `loadSession`, `exportTrace`. ~250 LOC out of the monolith. |
| D3 (phase 2) | `ae20cf4a` | `lib/evm/src/debug_tui_draw.zig` — pure drawing primitives (`seg`, `drawSegments`, all `style_*`, `ascii_border_glyphs`). |
| D3 (extractions) | `f40a83c1` + `ee68c7ec` + `b4c0fca5` | `lib/evm/src/debug_abi.zig` (AbiDoc + decoders + 4 tests), `lib/evm/src/debug_breakpoint.zig` (parser + 9 tests), `lib/evm/src/debug_overlay.zig` (OverlayMode + 4 tests). |

### Solidification round

`8de1fb7d` — six rough edges audited and fixed in one pass after
the first feature cycle:

1. Conditional/hit-count breakpoint `hit_count` reset on session
   rebuild (was getting "stuck" past N after step-back).
2. Gas accounting now skips opcodes with no source-map entry —
   correctly excludes callee opcodes from caller's tally.
3. `evalResolveBinding` propagates `OutOfMemory` instead of
   swallowing.
4. `exportTrace` shadow session applies live breakpoints so
   `:continue_` halts at the same place during replay.
5. `hoistOriginForLine` documented as inspecting only the first
   statement entry per line (rare edge case).
6. `abiDocForFrame` comment reframed honestly: the per-frame
   lookup seam is in place; the secondary registry was added in
   C4 (`4d8aecbb`).

Plus: `kMaxConditionalBreakpointGatingIters` named constant for
the spin guard; regression test for gas-accounting unmapped-PC
skip.

## Scaffolded with concrete plan, not closed

These items have all the debug-info / TUI scaffolding but
require compiler-side work that doesn't exist yet. Both have
written specs in the repo so the next session can pick them up
directly.

### A3 snapshot-based stepBack — `lib/evm/a3-snapshot-stepback-plan.md`

Today `stepBack` rebuilds the session from the seed and replays
every command in `step_history`. Cost is linear in user steps ×
opcodes-per-step. The plan specifies an `EvmSnapshot` struct that
captures the live `Frame` stack + `Storage` + `AccessListManager` +
`evm.logs` + debugger metadata at every K user-steps; `stepBack`
restores the nearest snapshot and replays only the trailing
K-or-fewer commands.

Five phases, ~14–18 focused hours total. Two alternatives
(replay-dedup, per-opcode undo log) considered and rejected with
reasons. Not closable inside a single debugger-feature session
because the cloning surface is broad enough that round-trip
tests at each layer are the gating quality concern.

### B2 inlined-from chain — `lib/evm/b2-inliner-plan.md`

Schema slot (`SourceScope.inlined_from: []const InlinedFrame`)
shipped (`1d27b891`); TUI render path shipped (`9ec8c81c`). Both
are dormant because no compiler pass currently inlines functions:
- `OraToSIR.cpp:2410` has the explicit `// TODO: requires
  MLIR's InlinerInterface (not yet wired up).` comment.
- `src/z3/encoder.zig:208 inline_function_stack` is SMT-only;
  doesn't rewrite emitted MLIR.
- `src/hir/module_lowering.zig:513` attaches `ora.inline = true`
  as a *hint*; no downstream pass reads it as a directive.

Plan lays out three implementation paths — Path A (MLIR
InlinerInterface, ~2–3 days), Path B (HIR substitution, ~3–5
days), Path C (defer entirely). Each with file:line attachments.

## Stopped early with rationale, not deferred

### D3 phase 3 — commands extraction

Phases 1+2 of the D3 split landed (session I/O + drawing
primitives, plus the four standalone modules already extracted).
Phase 3 would have moved the 37 `cmd*` handlers out of
`debug_tui.zig`, but each handler is a 1–5 line trampoline that
calls back into `Ui` machinery — extracting them would require
pub-ifying ~30 more `Ui` methods for ~700 LOC of moves, with no
real encapsulation gain. Documented in the D3 task closure.

`debug_tui.zig` ended at 5179 LOC (down from a 5491-LOC monolith
at branch start). Six new self-contained modules total:
`debug_session.zig`, `debug_eval.zig`, `debug_abi.zig`,
`debug_breakpoint.zig`, `debug_overlay.zig`, `debug_tui_draw.zig`,
`debug_tui_session.zig`.

## Reviewer round

After the first review pass, six findings landed in two
commits:

`0acc7c6c` — JSON safety, dedup, surface degraded paths:

- **DAP JSON safety** (#1). All response interpolations of
  user-controlled strings now go through `writeJsonEscaped` /
  `allocJsonString`. Verified by sending a request with
  `command:"weird\"name\\with/quote"` — response is well-formed
  JSON instead of a corrupt stream.
- **Stepping dedup** (#4). Four near-duplicate ~30-LOC
  stepping loops collapsed into a single `runUntil(condition:
  StopCondition)` engine on `Debugger`. Each public stepping
  method is now a 3–5 line wrapper. `step_opcode` arm folds
  in. Adding a new step mode plugs in without copying the
  harness.
- **Surface silent best-effort paths** (#5). `Debugger` gains
  a sticky `instrumentation_degraded` flag set on first
  `line_gas`/`watchpoint` failure. New pub getters
  `instrumentationDegraded()` /
  `instrumentationDegradedReason()`. TUI's status line picks
  up `[degraded: <reason>]` via a `defer` on
  `updateCommandStatusForCurrentStop`. Users no longer
  silently trust stale gas/watchpoint state.

**Reviewer round 3** — correctness + hardening:

- **High: setBreakpoints replacement** (#1). The DAP layer used
  to push new breakpoints into the debugger but never lift old
  ones, so a client unsetting a breakpoint via setBreakpoints
  still hit it. New `installed_breakpoints` tracking on
  `ServerState`; `replayPendingBreakpoints` now diffs pending
  vs installed and emits real `removeBreakpoint` calls for
  retired entries. Smoke-tested: install bp at line 5, unset
  via empty list, continue → finishes without hitting 5.
  Plus a `Debugger`-level regression test for the underlying
  set/remove primitives.
- **Medium: step-error responses** (#2). `continue`/`next`/
  `stepIn`/`stepOut` failures used to ack success and emit a
  `stopped` event regardless. Now propagate the error name as
  a failed response so the client doesn't desync UI state.
- **Medium: top-level wire-up** (#3). `zig build debug-dap`
  now exposed at the root build, mirroring `debug-tui`. New
  `ora debug --dap` flag spawns the DAP server with stdio
  inherited; explicitly skips the artifact pipeline so no
  unframed text leaks onto stdout (a stray "Bytecode saved"
  line ahead of a Content-Length header would corrupt the
  JSON-RPC stream). Workflow: `ora debug --no-tui` to
  produce artifacts, then DAP client passes those via the
  `launch` request.
- **Low: framing hardening** (#4). `kMaxMessageBytes = 16
  MiB` cap on Content-Length so a malformed announcement
  can't force unbounded allocation. `readDapMessage` factored
  through a `Source` abstraction so it's testable against
  in-memory buffers; 9 new framing tests cover simple body,
  no-op headers, zero-length body, missing Content-Length,
  EOF-before-any-byte, oversize Content-Length, header too
  long, short body, LF-only line endings.

`b35706ee` — DebugController extraction (#2/#3 partial):

- New `lib/evm/src/debug_controller.zig` —
  front-end-agnostic controller with `evaluateExpr` and
  `resolveBindingNumeric`. Generic over `EvmConfig` so it
  binds to the same `Debugger` instantiation the front-end
  uses.
- DAP `evaluate` request handler now calls
  `state.controller.evaluateExpr` directly. **No TUI
  dependency on the eval path.** End-to-end smoke tested:
  `evaluate "1 + 2 + 3"` → `"6"`; `evaluate "unknown_var"` →
  `success:false, message:"unknown identifier"`.
- DAP binary still imports `debug_tui.zig` for `Session` /
  `SessionSeed` / `AppConfig` machinery — that's the next
  lift target. The eval path is the first slice; ABI-param
  decoding (for SSA function args) and ABI document loading
  are tracked for the next round.

## Known DAP gaps remaining

- `scopes` / `variables` — would surface visible bindings as
  structured DAP variables. Needs ABI-param decoding +
  formatted-text rendering lifted onto `DebugController`.
- `restart`, `setExceptionBreakpoints`, `configurationDone` —
  capabilities advertise as unsupported.
- VS Code `package.json` `debuggers` contribution + `launch.json`
  schema (lives under `editors/vscode/`, not lib/evm/).

## Files of interest for review

The new modules are reasonable starting points — each is small,
self-contained, and has its own tests where applicable:

- `lib/evm/src/debug_eval.zig` (343 LOC, 4 tests) — expression
  evaluator.
- `lib/evm/src/debug_breakpoint.zig` (114 LOC, 9 tests) —
  conditional bp parser.
- `lib/evm/src/debug_abi.zig` (258 LOC, 4 tests) — AbiDoc +
  decoders.
- `lib/evm/src/debug_overlay.zig` (84 LOC, 4 tests) — OverlayMode.
- `lib/evm/src/debug_dap.zig` (~770 LOC) — DAP server +
  handlers including `evaluate` via DebugController.
- `lib/evm/src/debug_controller.zig` (~95 LOC) — front-end
  agnostic controller layer (eval path lifted out of TUI).
- `lib/evm/src/debug_tui_session.zig` (263 LOC) — Session JSON
  I/O + EIP-3155 trace export.
- `lib/evm/src/debug_tui_draw.zig` (94 LOC) — drawing primitives.

Tests:

- `lib/evm/src/debugger_test.zig` (814 LOC) — 17 debugger-core
  tests including regression coverage for the gas-accounting
  unmapped-PC skip and statement-level liveness narrowing.
- `lib/evm/test/bench/step_bench.zig` (191 LOC) — per-step
  wall-clock bench (gates regressions vs the 100µs budget).
- `tests/debug_artifacts/{simple_counter,liveness_dead_after_use}/`
  — golden artifact regression corpus.

External docs:

- `lib/evm/README.md`, `lib/evm/getting-started.md`,
  `lib/evm/debug-a-failing-transaction.md`, `lib/evm/COMMANDS.md`,
  `lib/evm/KEYBINDINGS.md`.

Plans:

- `lib/evm/a3-snapshot-stepback-plan.md`
- `lib/evm/b2-inliner-plan.md`

## Suggested review order

1. **One self-contained module first** — `debug_eval.zig` is the
   shortest piece with the cleanest interface; gives you a feel
   for the conventions.
2. **The Track-A foundation** — `debug_session.zig` (extracted
   harness), `debug_info.zig` and `source_map.zig` (the
   hash-index changes), `debugger_test.zig` (the unit-test
   coverage that was the gate for everything else).
3. **One end-to-end feature** — C5 FV-dead-branch is the most
   recent and the one that touches the most layers
   (`src/z3/errors.zig` → `src/z3/verification.zig` →
   `src/main.zig` → `lib/evm/src/debug_info.zig` → TUI overlay).
4. **The new binary** — `debug_dap.zig` can be reviewed against
   the DAP spec; smoke-tested but not exhaustively.
5. **The plan docs** — `a3-snapshot-stepback-plan.md` and
   `b2-inliner-plan.md` are the spec for the next round; review
   for correctness of effort estimates and approach.

## Verification before merge

Run:

```
cd /Users/logic/Ora/Ora/lib/evm
zig build test                                # 551/551 pass
zig build bench                               # ~25 ns/step

cd /Users/logic/Ora/Ora
zig build                                     # top-level green
zig build test-compiler                       # exit 0
```

Manual smoke (optional):

```
./zig-out/bin/ora debug --no-tui --verify ora-example/arithmetic_test.ora -o /tmp/fv
cat /tmp/fv/arithmetic_test.proof.json        # FV proof sidecar present
./zig-out/bin/ora-evm-debug-tui /tmp/fv/arithmetic_test.hex \
    /tmp/fv/arithmetic_test.sourcemap.json \
    ora-example/arithmetic_test.ora \
    --debug-info /tmp/fv/arithmetic_test.debug.json
# Press `O` four times to cycle into the `fv` overlay; lines
# with `S` are proved-safe by the verifier.
```

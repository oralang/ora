# Debugger TUI Redesign — Pipeline Three-Column Layout

**Status:** APPROVED design (2026-06-12). Layout direction "A + amendments" chosen by architect.
**Principle:** workflow-first. The UI exists to serve the debugging workflows below; every layout
and keybinding decision traces back to one of them. Visual multi-data-point support is the means,
the workflow is the test.

---

## 1. The Workflows (the spine)

Every pane, key, and sync rule below must justify itself against one of these. When a future
feature lands, it gets placed by asking "which workflow is this for?" — not "where is there room?"

### W1 — Step and watch (the default loop)
Step through Ora statements (`s`/`n`/`o`/`c`), watch bindings change, glance at gas.
**Needs:** Ora column with current-statement highlight + breakpoint gutter; bindings visible at
rest (dock home tab) with change-diff highlight; gas in header; binding-change pulse in header
when the dock is on another tab.

### W2 — Lowering trace ("why did this statement become this?")
Pick one Ora statement and read its lowering: statement → SIR ops → opcodes. The pipeline view.
This is the workflow the three-column layout exists for.
**Needs:** three-way synchronized highlight (ora line ↔ SIR range ↔ op rows) driven by the
current stop; statement-scoped SIR window (already provided by `currentMappingWindow()`);
full column height for all three views; provenance markers (synthetic/hoisted/folded) preserved.

### W3 — Machine forensics (revert hunting, stack archaeology)
Opcode-level stepping (`x`) around a revert or an unexpected value; stack always in view.
**Needs:** opcode disassembly with current-pc marker and scrollback; stack pane that is NOT
behind a tab (visible while stepping); raw inspectors (`:print stack[0]`, `:print slot`,
`:print mem`) unchanged.

### W4 — State deep-dive (storage/memory/calldata inspection)
Dump and read rooted storage, tstore, memory, calldata — wide hex content.
**Needs:** full-terminal-width dock tabs (today these get 58% width; the redesign improves them).

### W5 — Gas attribution
Where does gas go, per line and per op.
**Needs:** gas heatmap overlay on the Ora column (exists); gas-remaining + spent-since-stop in
header (exists); per-op gas delta annotation in the opcode column (phase 3, ring buffer).

### W6 — Time travel and session work
`p`/`:prev`, checkpoints, `:rerun`, sessions, trace export.
**Needs:** step index + checkpoint summary in header/footer (exists); Trace as a dock tab;
no layout interaction beyond that — time travel must not reshuffle panes.

---

## 2. Layout Spec

Wide mode (terminal width ≥ 150 cols AND sir_text present):

```
┌ Header (3 rows): contract.fn · stop line · pc/op · gas · depth · [pulse: total 3800→4000] ┐
│ ┌ Ora Source ──────────┐ ┌ SIR ─────────────────┐ ┌ Opcodes ─────────────────────┐        │
│ │ ~34% width           │ │ ~33% width           │ │ ~33% width                   │  ▲     │
│ │ gutter: breakpoints, │ │ statement-scoped     │ │ disasm, ▸ current pc,        │  │     │
│ │ overlays (fv/gas/cov)│ │ highlight window,    │ │ scrollback, gas/op (P3)      │ ~70%   │
│ │ current stmt >       │ │ idx/stmt/region title│ ├─ Stack (always visible) ────┤  │     │
│ │                      │ │                      │ │ 0│ 0x…0fa0 (4000)  Δ-marks   │  │     │
│ └──────────────────────┘ └──────────────────────┘ └──────────────────────────────┘  ▼     │
│ ┌ Dock: [Bindings]│Memory│Storage│TStore│Calldata│Trace ──────────────────────────┐ ~30%  │
│ │ full width; home tab = Bindings (diff highlight)                                │       │
│ └─────────────────────────────────────────────────────────────────────────────────┘       │
│ Footer (unchanged): key hints + `:` command console + result trail                        │
└────────────────────────────────────────────────────────────────────────────────────────────┘
```

Decisions locked in:
- **Columns are fixed-width** (no focus-zoom). Simplest layout model; least new view state.
- **Stack is a fixed sub-pane of column 3** (~40% of column height), never a tab. (W3)
- **Dock home tab is Bindings.** (W1)
- **Header pulse:** when a stop changes a binding and the dock is on another tab, render a
  compact `name old→new` (most recent change) in the header. One line of cost, keeps W1 alive
  from any tab.
- **Machine summary** (frame i/n, pc, opcode, depth, caller, calldata/memory size) moves from
  the old Machine pane into the header line + the Opcodes column title.
- **Narrow fallback (< 150 cols or no sir_text):** keep the CURRENT layout unchanged as the
  degraded mode. Three columns under ~45 chars each are unreadable slivers; don't force it.

## 3. Sync Model

Single source of truth: **the current stop's pc.** Everything else derives:
- Ora line: `pc_to_statement` (existing).
- SIR window: `currentMappingWindow()` (existing) — statement's sir line range + idx range.
- Opcode row: pc → row index via the disasm cache (new, §5).

Per-column `follow: bool` (generalizes today's `sir_follow`):
- A stop event re-centers every column with `follow == true`.
- Manually scrolling a column sets ITS follow to false (others unaffected).
- `=` re-enables follow on the focused column; `:sync` re-enables all three.

## 4. Key Map (delta from current)

Unchanged: `s n o c p x q :` · `[`/`]` cycle dock tabs · overlays · breakpoint keys.

| Key | Old | New |
|-----|-----|-----|
| `Tab` | — | cycle column focus Ora → SIR → Opcodes (scroll target only; no resize) |
| `j/k` | scroll Ora | scroll the FOCUSED column |
| `J/K` | scroll SIR | kept as SIR-scroll aliases (muscle memory; remove later) |
| `=` | resync SIR | re-enable follow on focused column |
| `1..6` | state tabs (5) | dock tabs: 1 Bindings, 2 Memory, 3 Storage, 4 TStore, 5 Calldata, 6 Trace |
| `B` | — | jump dock to Bindings |

Focused column gets a visual marker in its panel title.

## 5. Column 3: Opcodes + Stack

**Disasm cache (new, cheap):** bytecode is static per session — decode ONCE at session init into
`[]DisasmRow { pc: u32, op: u8, immediate: ?[]const u8 }` (PUSH-n immediates consumed correctly)
plus a pc→row index map. Rendering = window over rows. Primitives exist:
`frame.bytecode.getOpcode(pc)` + `opcode_utils.getOpName` (debug_tui.zig:4633).
- Current pc row marked `▸`; rows with a source-map entry get a subtle line-number annotation
  (lets W2 read op→ora attribution directly).
- Frame switch (`:frame N`) re-targets the column to the selected frame's bytecode (decode cache
  per contract address, lazily).

**Stack sub-pane:** top N entries (fills available rows), index-labeled, short-hex with decimal
annotation for small values, diff-highlight vs previous stop (same previous-snapshot pattern as
bindings). Selected frame's stack (follows `:frame`).

## 6. Implementation Plan — lands as debt paydown, not growth

Hard rule: this redesign must NOT grow debug_tui.zig (5,549L at time of writing). Each phase
extracts a module; new code goes in the modules.

**Precursor — concept cleanup (do BEFORE P1; see §6.1):** the DAP server currently imports
debug_tui.zig for Session/SessionSeed/AppConfig/loadSeedFromConfig (debug_dap.zig:40-44), and
the conditional/hit-count breakpoint model lives in the TUI layer where DAP can't reach it
(DAP silently drops `condition`/`hitCondition`). Moving those concepts to their proper owners
first shrinks debug_tui.zig and fixes the boundaries the column extraction lands on.

### 6.1 Concept cleanup (cross-frontend, precedes P1)
- **K1 — Session/launch concept out of the TUI.** Move `Session`, `SessionSeed`, `AppConfig`,
  `loadSeedFromConfig`, `decodeHexAlloc` from debug_tui.zig into debug_session.zig (its
  documented "single home"). Pure move; deletes the DAP→TUI import; probe/TUI/DAP all consume
  the shared module.
- **K2 — Breakpoint model into the engine.** Engine owns line/pc + condition + hit-count
  breakpoints, conditions evaluated inside `runUntil` via the existing `debug_eval.Resolver`
  callback pattern. TUI keeps only parsing (`debug_breakpoint.zig`) + gutter rendering; DAP maps
  protocol `condition`/`hitCondition` onto the same store. Interim (until K2 lands): DAP must
  report conditional requests as `verified: false` instead of silently installing them
  unconditional.
- **K3 — Execute debug_controller.zig's documented lift-list** (ABI doc loading, decoded value
  formatting, ABI param decoding, revert/log decode) → unblocks DAP `scopes`/`variables`, which
  is the DAP equivalent of the Bindings pane (W1 for IDE users).
- **K4 — Split the checkpoint concept:** replay-position checkpoint (shared, enables DAP
  `restart`) vs UI snapshot (stays TUI).

- **P0 — pure logic, no UI change:** `debug_disasm.zig` (decode cache + pc→row map, unit-tested);
  `debug_tui_layout.zig` (pure geometry: terminal size → pane rects for wide/narrow modes,
  unit-tested — today's inline percentage math moves here).
- **P1 — column framework, behavior-neutral:** extract `SourceColumn` / `SirColumn` renderers and
  a `Dock` module (today's drawEvmPane tabs + bindings + trace) from debug_tui.zig behind a small
  common interface (title, draw(rect), scroll, follow). Legacy layout still renders identically.
  Gate: manual probe parity on ora-example/debugger contracts.
- **P2 — new layout behind a toggle:** `MachineColumn` (disasm + stack); `:layout 3col` /
  `:layout classic` switch, default classic. Per-column follow + Tab focus land here.
- **P3 — flip the default:** 3col becomes default in wide mode; header pulse; per-op gas deltas
  (ring buffer of last N executed-op costs); remove `J/K` aliases and the classic layout after a
  bake period.

**Testing:** P0 modules fully unit-tested (pure). Column/dock scroll+follow state machines tested
as plain logic (no vaxis). Geometry: property test that pane rects tile the terminal exactly
(no gaps/overlap) across width×height ranges. Manual checklist per workflow W1–W6 using
DEBUGGER.md's probe contracts. debugger_test.zig stepping/replay tests are layout-independent
and must stay green untouched.

## 7. Non-goals / deferred

- Focus-zoom adaptive widths (proposal C) — rejected for now; revisit only with user demand.
- `:dock split` (two dock tabs side-by-side, e.g. Bindings + Storage) — the escape hatch if the
  "watch a slot while stepping" workflow materializes. Composes with this design; not in v1.
- SIR-line breakpoints, stack-value ABI decoding, multi-contract disasm browsing — later.
- DAP is unaffected (protocol layer has no layout).

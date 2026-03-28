# Ora Source-Level EVM Debugger

## Current Status

Phase 1 of the source map pipeline is already working end to end in the current branch.

- C++ extracts source locations from final SIR MLIR with op indices that match SIR text emission
- Sensei records per-op bytecode PCs and writes a sidecar op-index map
- Zig merges both streams into `<contract>.sourcemap.json`
- Statement boundaries are emitted via `stmt: true` on the first entry for each new source line

Observed working result from the current implementation:

- 84 source-location entries extracted from SIR MLIR
- 21 statement boundaries across the two test functions
- every Ora source line that produces bytecode has a corresponding source-map entry
- `ora emit --emit-bytecode --debug-info` now builds bytecode with debugger-safe lowering:
  - skips Ora MLIR canonicalization in the Zig driver
  - disables post-SIR optimization/cleanup passes in `oraConvertToSIR(...)`
  - keeps Sensei copy propagation disabled when source maps are requested

This means the branch now has a usable Phase 1.5 debugger build mode, not just a best-effort release source map.

There is now also a first debugger metadata sidecar:

- `ora emit --emit-bytecode --debug-info` writes `<contract>.debug.json`
- entries are keyed by the same serialized SIR op indices as the source map
- each entry records:
  - `idx`
  - `op`
  - `function`
  - `block`
  - optional `file` / `line` / `col`
  - `result_names`
  - `is_terminator`
  - `is_synthetic` for expanded ops such as `sir.select`

This is not full variable/scope debug info yet, but it gives the debugger a stable per-op metadata channel that survives the current debugger-safe lowering mode.

There is now also a first source-level scope/liveness layer in the same sidecar:

- `.debug.json` now includes `source_scopes`
- each scope records:
  - lexical parent
  - file / function / optional contract
  - scope kind (`function`, `if_then`, `while`, `for`, `catch`, etc.)
  - source range
  - declared locals in that scope
- each local records:
  - stable local id
  - name
  - local kind (`param`, `local`, `for_item`, `for_index`, `catch_error`)
  - optional binding/storage class
  - declaration range
  - lexical live range
- `.debug.json` also now includes `op_visibility`
  - keyed by serialized SIR op index
  - lists the scope ids and visible local ids for that stop point

This is still lexical scope/liveness, not exact runtime storage locations, but it is the first debugger-grade answer to “what names are in scope here?” that survives MLIR/SIR optimization.

This document should now be treated as the repo source of truth. The copy in `~/.claude/plans/` is useful as scratch history, but it should not lead the implementation.

## Context

No smart contract language has a real source-level debugger. Adding a Visual Studio-style debugger to Ora — where you set breakpoints on Ora source lines, step through statements, and inspect variables — is a killer feature.

The Ora compiler now preserves source locations and basic per-op metadata through the SIR text emission boundary when `--debug-info` is used. The EVM (lib/evm) already has single-stepping and full state observability. The remaining work is to enrich that metadata into real variable/scope debug info rather than merely line stepping.

## The Hard Problem: Expansion and Location Loss

A single line of Ora source undergoes massive expansion:

```
let c = a + b;        ← 1 Ora statement
    ↓
~6 Ora MLIR ops       ← arith.constant, arith.addi, bitcasts
    ↓
~15-20 SIR ops        ← sir.add, sir.bitcast (each inherits Ora location)
    ↓
~30-50 EVM opcodes    ← PUSH, MLOAD, ADD, PUSH, MSTORE, ...
```

**Key expansion drivers:**
- Every local variable access = PUSH addr + MLOAD/MSTORE (3-5 bytes each)
- A simple `sir.add` with 2 inputs, 1 output = 11 EVM bytes (2 loads + ADD + 1 store)
- `sir.mstore` (partial word) = 28-30 EVM bytes (mask + shift + OR + store)
- Branch = 18-24 EVM bytes (condition load + JUMPI + fallthrough JUMP)
- Function call = 24 + (N+M)*10 bytes (N inputs, M outputs transferred via memory)

**What this means for source maps:**
- Many EVM opcodes map to the same Ora source line (this is fine — debugger stops at statement boundaries)
- Some EVM opcodes have NO source location (compiler-internal: dispatch tables, memory init, trampolines)
- Optimization passes can eliminate or reorder ops — this is the hard problem
- The `SIRTextLegalizer` creates trampoline blocks with compiler-generated locations (~1-2% of ops)

## The Core Challenge: Traceability Through Optimizations

This is the hardest part of the entire debugger project. The Ora compiler runs 5+ optimization passes between source code and bytecode. Each pass can destroy, merge, or reorder the ops that carry our source locations.

### Pass Pipeline (in order)

```
Ora source
  ↓ parse + sema
Ora MLIR (every op has FileLineColLoc)
  ↓ createSimpleOraOptimizationPass()    ← canonicalization, CSE DISABLED
  ↓ createOraOptimizationPass()          ← constant dedup + constant folding
  ↓ createOraInliningPass()              ← function inlining
  ↓ createOraCleanupPass()               ← no-op placeholder
  ↓ applyPatternsGreedily() × 5 phases   ← Ora→SIR dialect conversion
SIR MLIR (ops inherit Ora locations)
  ↓ createSIROptimizationPass()          ← SIR constant dedup + folding
  ↓ createSIRCleanupPass()               ← dead code elimination
  ↓ createSimpleDCEPass()                ← removeDeadValuesPass
SIR MLIR (final)
  ↓ extractSIRLocations()                ← WE READ LOCATIONS HERE
  ↓ emitSIRText()
SIR text → Sensei → EVM bytecode
```

### Per-optimization analysis

| Optimization | Location Impact | Risk | Mitigation |
|-------------|----------------|------|------------|
| **Canonicalization** | Constant CSE would assign `UnknownLoc` to rehomed constants | **HIGH** | Already mitigated: `enableConstantCSE(false)` at `OraToSIR.cpp:2351`. But only for pre-conversion pass — post-SIR canonicalization is unprotected |
| **Constant Deduplication** | When two identical constants exist (e.g., `100` on line 3 and `100` on line 7), the duplicate is erased — its location is LOST. The surviving constant keeps line 3's location | **MEDIUM** | Unavoidable. The debugger may attribute the constant to the wrong line. Acceptable because the value is correct |
| **Constant Folding** | `let x = 1 + 2;` → the `add` op (line 5) is replaced by `const 3` which inherits the add's location (line 5) | **SAFE** | Location preserved by design: `builder.create<ConstOp>(addOp.getLoc(), ...)` |
| **Dead Code Elimination** | Entire dead statements are erased — all their ops and locations vanish | **SAFE** | No ambiguity. Dead line simply absent from source map. Debugger skips it during stepping |
| **OraToSIR Pattern Rewrites** | 1 Ora op → N SIR ops. All N ops get the same `op.getLoc()` | **SAFE** | Every pattern file explicitly passes `loc`. Verified in `Arithmetic.cpp`, `Storage.cpp`, `ControlFlow.cpp`, etc. |
| **Function Inlining** | Cloned ops keep their original locations from the callee | **SAFE but tricky** | Inlined code retains callee file:line. Debugger may jump between files. Standard behavior (same as C/C++ debuggers with inlined functions) |
| **SIR Cleanup (DCE)** | Removes unused ops (`op->use_empty()` → erase) | **SAFE** | Survivors' locations untouched |
| **SIRTextLegalizer** | Creates trampoline blocks for asymmetric conditional branches | **LOW** | Trampolines inherit the branch op's location. ~1-2% of ops |
| **Sensei Copy Propagation** | Eliminates `SetCopy` ops at Sensei IR level | **SAFE but index-shifting** | Copy prop happens AFTER SIR text parsing. If `--copy-propagation` is on, op indices shift. Must disable for source maps OR track the index mapping through copy prop |
| **GreedyPatternRewriteDriver** | Fires patterns iteratively until convergence | **SAFE** | Each rewrite is 1:1 location preserving |

### Critical risks that need active handling

**Risk 1: Constant CSE after SIR conversion**

The `createSIROptimizationPass()` runs constant deduplication on SIR ops. If MLIR's canonicalization also runs here with CSE enabled, constants get `UnknownLoc`.

**Action**: Verify that `createSIROptimizationPass()` either:
- Doesn't run canonicalization, OR
- Disables CSE like the pre-conversion pass does

**File**: `OraToSIR.cpp:1989` — read the SIR optimization pass setup

**Risk 2: Sensei copy propagation shifts op indices**

If `--copy-propagation` is enabled, Sensei removes `SetCopy` operations from the IR before code generation. This changes the operation count and shifts indices. Our source map correlation relies on op indices matching 1:1 between MLIR and Sensei.

**Action**: Two options:
- **Option A (simple)**: Disable copy propagation when `--source-map` is requested. The bytecode will be slightly larger but the mapping is correct.
- **Option B (better)**: Sensei's copy propagation should output a mapping of old indices → new indices. Then the merge step adjusts accordingly.

Recommendation: Ship with Option A, iterate to Option B.

**Risk 3: Constant dedup changes attribution**

If the same constant `100` appears at lines 3 and 7, after dedup only line 3's location survives. If the user sets a breakpoint on line 7 (expecting to hit the second use of 100), the debugger won't find a statement-entry PC for line 7.

**Action**: This is acceptable for v1. The constant's VALUE is correct, just attributed to the first occurrence. A future improvement could use `FusedLoc` to remember all original locations during dedup.

### Strategy: Debug mode vs Release mode

Like C/C++ compilers, we should support two modes:

**`ora build --debug` (default for `ora debug`):**
- Disable constant deduplication
- Disable copy propagation in Sensei
- Disable function inlining
- Keep canonicalization CSE disabled (already the case)
- Result: 1:1 source line mapping, no surprises, slightly larger bytecode

**`ora build --release` (default for `ora build`):**
- All optimizations enabled
- Source map is best-effort (some lines may be missing or misattributed)
- Good enough for production crash analysis, not great for interactive debugging

This is exactly what GCC/Clang do with `-O0 -g` vs `-O2 -g`. The debug info is more accurate at lower optimization levels.

**Implementation**: Add a `--debug-info` flag to the Ora compiler that:
1. Sets optimization config to disable problematic passes
2. Enables source map generation
3. Passes `--source-map` to Sensei (without `--copy-propagation`)

## Where Source Locations Exist Today

| Stage | Locations? | Mechanism | Notes |
|-------|-----------|-----------|-------|
| Lexer/Parser | Yes | `SourceSpan` on every token/AST node | `file_id`, line, col, byte_offset, length |
| Sema | Yes | Preserves AST ranges | Comptime folding keeps original location |
| HIR → Ora MLIR | Yes | `locationFromRange()` → `FileLineColLoc` | Uses real file path from SourceStore |
| MLIR optimization | Yes | MLIR preserves across passes | Dead ops removed, but survivors keep locations |
| Ora → SIR lowering | Yes | SIR ops inherit Ora op location | All patterns in `OraToSIR/patterns/*.cpp` do this |
| SIR Text Emitter | **NO** | `emitOperation()` ignores `op.getLoc()` | **THIS IS THE GAP** |
| Sensei backend | **NO** | SIR text has no location data | `mark_to_offset` computed but discarded |
| EVM bytecode | **NO** | Raw bytes | No metadata |

**Critical insight**: The location data is available in the SIR MLIR right before `SIRTextEmitter.cpp` runs. We don't need to thread locations through the SIR text format at all — we can extract them directly from MLIR and correlate with Sensei's bytecode offsets.

## Architecture

Two possible approaches:

### Approach A: Thread locations through SIR text (original plan)
```
SIR MLIR → SIRTextEmitter (add // loc: comments) → SIR text → Sensei parser (parse comments)
→ Sensei backend (per-op marks) → mark_to_offset → source map
```
**Problem**: Requires changes to 3 codebases (C++, Rust parser, Rust backend). The Sensei lexer skips comments, so we'd need to add a secondary comment-parsing pass. Fragile.

### Approach B: Sidecar source map from MLIR (recommended)
```
SIR MLIR → SIRTextEmitter → SIR text (unchanged)
    ↓ (parallel)                    ↓
    Extract locations           Sensei backend
    per SIR op index              ↓
    ↓                          bytecode + op-index→PC map
    ↓                              ↓
    Merge: SIR-op-index → (file, line, col)
         + SIR-op-index → bytecode-PC
         = bytecode-PC → (file, line, col)
```

**Why B is better:**
- SIR text format stays unchanged (no Sensei parser changes)
- Location extraction happens in C++ where MLIR API is native
- Sensei only needs to add per-op marks and expose the mapping (Rust changes only)
- The merge happens in Zig (main.zig) where we already orchestrate everything

## Phase 1: Source Map Pipeline

### 1.1 Extract locations from SIR MLIR (C++)

**File**: `src/mlir/ora/lowering/OraToSIR/SIRTextEmitter.cpp`

Add a new function alongside `emitSIRText()`:

```cpp
struct SIRLocationEntry {
    uint32_t op_index;     // Sequential index of this SIR op
    std::string filename;
    uint32_t line;
    uint32_t column;
};

// New: Extract per-op locations from MLIR before emitting text
std::vector<SIRLocationEntry> extractSIRLocations(ModuleOp module);
```

This iterates the same ops that `emitSIRText()` does (functions → blocks → operations) and for each op, checks `op.getLoc()`. If it's a `FileLineColLoc`, record `(op_index, file, line, col)`. If it's an unknown/fused location, record nothing for that index.

**New C API wrapper** in `OraCAPI.cpp`:
```c
// Returns JSON string: [{"idx":0,"file":"main.ora","line":3,"col":5}, ...]
MlirStringRef oraExtractSIRLocations(MlirContext ctx, MlirModule module);
```

**Op indexing must match Sensei's**: The op indices must correspond 1:1 with the operations in the SIR text. Since both `emitSIRText()` and `extractSIRLocations()` iterate the same MLIR module in the same order, this is guaranteed.

### 1.2 Add per-operation marks in Sensei (Rust)

**File**: `vendor/sensei/senseic/sir/crates/debug-backend/src/lib.rs`

In `translate_basic_blocks_from_entry_point` (around line 153), before each operation:

```rust
// Track source map: operation index → mark ID
let op_mark = self.mark_map.allocate_mark();
self.asm.push_mark(op_mark);
self.source_map_marks.push((op_mark, self.global_op_index));
self.global_op_index += 1;
```

After `assembler.assemble()` returns `mark_to_offset` (currently discarded at line 234), resolve:

```rust
let mark_to_offset = self.asm.assemble(result, ...)?;
let source_map: Vec<(u32, u32)> = self.source_map_marks.iter()
    .map(|(mark, op_idx)| (mark_to_offset[*mark], *op_idx))
    .collect();
```

### 1.3 Add `--source-map` CLI flag to Sensei (Rust)

**File**: `vendor/sensei/senseic/sir/crates/cli/src/main.rs`

Add `--source-map <path>` clap argument. When present, write JSON:

```json
{"ops": [{"idx": 0, "pc": 0}, {"idx": 1, "pc": 5}, {"idx": 2, "pc": 8}, ...]}
```

**Important**: The `idx` values must match the SIR text operation ordering. Sensei processes operations in basic-block order, which is the same order the SIR text emitter writes them (both do BFS/DFS from entry point). Need to verify this or enforce it.

### 1.4 Merge in Ora compiler (Zig)

**File**: `src/main.zig` (around `emitBytecodeFromSirText`, line 2529)

```
1. Call oraExtractSIRLocations(ctx, module) → JSON with (op_idx → file:line:col)
2. Call oraEmitSIRText(ctx, module) → SIR text (unchanged)
3. Write SIR text to temp file
4. Invoke sensei with --source-map /tmp/ops.json
5. Parse sensei's source map: (op_idx → bytecode_PC)
6. Merge: for each op_idx, if locations[op_idx] exists:
     emit (bytecode_PC, file, line, col, is_statement)
7. Write <Contract>.sourcemap.json
```

### 1.5 Statement boundary detection

Multiple SIR ops will share the same Ora source line. The first op for each new (file, line) is marked `is_statement: true`. The debugger stops only at these PCs during step-over.

**Edge case**: After optimization, some source lines may have NO ops (dead code eliminated). These lines are simply not in the source map — the debugger skips them.

**Edge case**: Compiler-internal ops (dispatch table, memory init, trampolines) have `UnknownLoc`. These get NO source map entry. The TUI shows "[internal]" for these PCs.

### Source Map Format

```json
{
  "version": 1,
  "sources": ["contracts/MyContract.ora"],
  "entries": [
    {"pc": 0,  "src": 0, "line": 10, "col": 4, "stmt": true},
    {"pc": 5,  "src": 0, "line": 10, "col": 4, "stmt": false},
    {"pc": 8,  "src": 0, "line": 10, "col": 12, "stmt": false},
    {"pc": 14, "src": 0, "line": 11, "col": 4, "stmt": true},
    {"pc": 45, "src": 0, "line": 12, "col": 4, "stmt": true}
  ]
}
```

- `src` indexes into `sources[]` (supports multi-file)
- `stmt: true` = debugger stop point for step-over
- Entries sorted by `pc`
- Gaps in PC coverage = compiler-internal code (no source)

### Op index ordering guarantee

**Critical**: The MLIR op iteration order in `extractSIRLocations()` must match the order Sensei processes them. Both traverse:
1. Functions sorted by name (init, then main, then helpers)
2. Within each function: basic blocks in BFS/DFS from entry
3. Within each block: operations in order, then terminator

The `SIRTextEmitter.cpp` already sorts functions at line 527 (`std::sort(funcOps.begin(), ...)`). Sensei parses the text top-to-bottom. As long as both use the same sort, indices align.

**Verification**: After implementation, compile a test contract and assert that the number of location entries equals the number of Sensei op entries. If they diverge, there's an ordering mismatch.

---

## Phase 2: EvmDebugger Backend

(Unchanged from previous plan — see `lib/evm/src/debugger.zig` design)

Key files:
- `lib/evm/src/source_map.zig` — source map loader + PC→line lookup
- `lib/evm/src/debugger.zig` — EvmDebugger wrapping Evm with step-over/in/out, breakpoints
- `lib/evm/src/root.zig` — export new modules

---

## Phase 3: TUI Debugger

(Unchanged from previous plan — terminal UI with source, stack, memory, storage panels)

Key files:
- `lib/evm/src/debugger_tui/main.zig`
- `lib/evm/src/debugger_tui/terminal.zig`
- `lib/evm/src/debugger_tui/layout.zig`

---

## Files Modified/Created Summary

### Phase 1 (Source Maps)
| File | Action | Language |
|------|--------|----------|
| `src/mlir/ora/lowering/OraToSIR/SIRTextEmitter.cpp` | Add `extractSIRLocations()` | C++ |
| `src/mlir/ora/lowering/OraToSIR/SIRTextEmitter.h` | Declare new function + struct | C++ |
| `src/mlir/ora/lib/OraCAPI.cpp` | Add `oraExtractSIRLocations()` C wrapper | C++ |
| `src/mlir/ora/include/OraDialectC.h` | Declare C API | C |
| `vendor/sensei/.../debug-backend/src/lib.rs` | Per-op marks, return mapping | Rust |
| `vendor/sensei/.../cli/src/main.rs` | `--source-map` flag | Rust |
| `src/main.zig` | Merge maps, write .sourcemap.json | Zig |

### Phase 2 (Debugger Backend)
| File | Action | Language |
|------|--------|----------|
| `lib/evm/src/source_map.zig` | Source map loader (already created) | Zig |
| `lib/evm/src/debugger.zig` | New: EvmDebugger engine | Zig |
| `lib/evm/src/root.zig` | Export new modules | Zig |

### Phase 3 (TUI)
| File | Action | Language |
|------|--------|----------|
| `lib/evm/src/debugger_tui/main.zig` | New: TUI entry point | Zig |
| `lib/evm/src/debugger_tui/terminal.zig` | New: raw terminal + ANSI | Zig |
| `lib/evm/src/debugger_tui/layout.zig` | New: panel rendering | Zig |
| `src/main.zig` | Add `ora debug` subcommand | Zig |

---

## The Comptime Problem

Comptime is 1/3 of Ora's story (alongside formal verification and the type system with regions). The debugger can't just ignore comptime — it needs to explain it.

### What happens to source code under comptime

```ora
fn foo() -> u256 {
    let a = 10;          // line 3 — folded away, no bytecode
    let b = 20;          // line 4 — folded away, no bytecode
    let c = a + b;       // line 5 — folded to const 30, no add opcode
    return c;            // line 6 — just PUSH 30 + RETURN
}
```

After comptime + constant folding, the entire function becomes:
```
sir.const 30    ← loc("file.ora":5:12)   (inherited from the folded add)
sir.return      ← loc("file.ora":6:4)
```

Lines 3 and 4 produce **zero bytecode**. Line 5's `a + b` became a constant. The source map has no entry for lines 3-4.

### Current comptime architecture

**Files**: `src/comptime/` (11 files)

Two-layer value model:
- `CtValue` — ephemeral, lives in `CtEnv` during evaluation (has heap-backed aggregates)
- `ConstValue` — persistent, interned in `ConstPool` (just the value, no metadata)

**What triggers comptime**: Both `const` AND `let` bindings are evaluated if all operands are comptime-known. The `Stage` system classifies expressions:
- `comptime_only` — must evaluate at comptime (@TypeOf, @sizeOf)
- `comptime_ok` — can evaluate at comptime (arithmetic, comparisons)
- `runtime_only` — can never be comptime (sload, msg.sender, block.*)

**The gap: NO provenance tracking.** When the evaluator computes `a + b = 30`:
- `a` is stored in a `CtEnv` slot as `integer: 10`
- `b` is stored as `integer: 20`
- The result is `integer: 30`
- **No record of which expressions or source locations contributed to the result**
- `ConstValue` is a bare union — `integer: u256`, `boolean: bool`, etc. No metadata fields.

**Partial trace infrastructure exists:**
- `EvalConfig.trace_enabled: bool = false` — flag exists but unused
- `EvalStats` tracks aggregate stats (total_steps, peak_recursion_depth)
- `Evaluator.step(span)` is called before each operation with the current `SourceSpan`
- But no event log, no dependency graph, no replay capability

### Strategy: Two debug experiences

The debugger needs to support two fundamentally different modes:

**Runtime Debugger** (stepping through EVM bytecode):
- PC → source line mapping
- Comptime-folded lines are absent from the source map
- TUI shows `[comptime: value = 30]` annotation for folded lines
- This is Phases 1-3 of the plan

**Comptime Trace Viewer** (understanding what the compiler computed):
- Shows the evaluation tree: which expressions fed into which results
- Not a runtime debugger — it replays the compiler's evaluation
- Requires adding provenance tracking to the evaluator

### Phase 5: Comptime Provenance (after Phases 1-3 ship)

#### 5.1 Add TraceEvent to the evaluator

**File**: `src/comptime/env.zig`

```zig
pub const TraceEvent = struct {
    step: u64,
    span: SourceSpan,           // source location of this evaluation
    expr_id: ExprId,            // AST expression that triggered it
    operation: []const u8,      // "add", "const", "field_access", etc.
    inputs: []const ConstId,    // values that fed in
    output: ConstId,            // value that came out
};
```

Add to `CtEnv`:
```zig
trace_events: std.ArrayList(TraceEvent),
```

#### 5.2 Record events during evaluation

**File**: `src/comptime/compiler_ast_eval.zig`

The evaluator already calls `self.step(span)` before each operation. Extend it:

```zig
fn evalBinaryOp(self: *ConstEvaluator, op: BinaryOp, lhs_id: ExprId, rhs_id: ExprId, span: SourceSpan) !?ConstValue {
    const lhs = self.evalExpr(lhs_id) orelse return null;
    const rhs = self.evalExpr(rhs_id) orelse return null;
    const result = /* compute */;

    if (self.trace_enabled) {
        try self.trace_events.append(.{
            .step = self.stats.total_steps,
            .span = span,
            .expr_id = /* current expr */,
            .operation = "add",
            .inputs = &.{ lhs_id_const, rhs_id_const },
            .output = result_id,
        });
    }

    return result;
}
```

#### 5.3 Persist trace in ConstEvalResult

**File**: `src/comptime/compiler_ast_eval.zig`

```zig
pub const ConstEvalResult = struct {
    arena: std.heap.ArenaAllocator,
    values: []?ConstValue,
    diagnostics: DiagnosticList,
    trace: ?[]const TraceEvent,     // NEW: comptime evaluation trace
};
```

#### 5.4 Emit comptime trace in source map

Extend the source map format:

```json
{
  "version": 2,
  "sources": ["contracts/MyContract.ora"],
  "entries": [
    {"pc": 0, "src": 0, "line": 6, "col": 4, "stmt": true}
  ],
  "comptime_folds": [
    {
      "result_line": 5,
      "result_value": "30",
      "trace": [
        {"line": 3, "expr": "let a = 10", "value": "10", "op": "const"},
        {"line": 4, "expr": "let b = 20", "value": "20", "op": "const"},
        {"line": 5, "expr": "a + b", "value": "30", "op": "add", "inputs": [0, 1]}
      ]
    }
  ]
}
```

#### 5.5 TUI comptime view

When the debugger encounters a comptime-folded region, it shows:

```
+--[ source.ora ]------------------+--[ Comptime Trace ]-------+
|  3    let a = 10;        [ct]    |  step 1: const 10         |
|  4    let b = 20;        [ct]    |    → a = 10 (line 3)      |
|  5    let c = a + b;     [ct]    |  step 2: const 20         |
|  6>   return c;          ←       |    → b = 20 (line 4)      |
|                                   |  step 3: add(a, b)        |
|                                   |    → c = 30 (line 5)      |
+-----------------------------------+---------------------------+
```

Lines 3-5 are grayed out with `[ct]` tag. The comptime trace panel shows the evaluation tree. The user can see exactly how the compiler arrived at `30`.

### What we ship first (Phases 1-3) vs later (Phase 5)

**Phases 1-3 (ship now):**
- Runtime debugger with source maps
- Comptime-folded lines show as `[comptime]` in TUI — the user sees they were folded but not *how*
- No changes to the comptime evaluator

**Phase 5 (ship after):**
- Add `TraceEvent` recording to the evaluator (~200 lines of Zig)
- Persist traces in `ConstEvalResult` and source map
- TUI comptime trace panel
- This is what makes Ora's debugger unique — no other language shows you the compile-time evaluation trace

---

## Known Limitations (ship and iterate)

1. **No variable inspection (v1)** — Source map gives PC→line, not variable→location. Adding ethdebug/format pointers is Phase 4 work.
2. **Optimized code has degraded source maps** — Constant dedup can misattribute lines, inlining mixes file locations. Mitigated by `--debug` mode which disables problematic passes. Release mode source maps are best-effort.
3. **Multi-contract calls** — The source map covers one contract. Cross-contract CALL enters unmapped bytecode. TUI shows "[external call]" for these.
4. **Inline assembly** — If Ora ever supports inline EVM, those ops won't have Ora source locations. Fine — show the raw opcodes.
5. **Sensei copy propagation + source maps (v1)** — Disable copy propagation when source maps are requested. Bytecode is slightly larger but mapping is correct.
6. **Comptime trace not in v1** — Folded lines show as `[comptime]` but without the evaluation tree. Full trace comes in Phase 5.

## Verification

1. **Op index alignment**: Compile test contract, assert `extractSIRLocations()` count == Sensei op count
2. **Source map correctness**: For a 5-line function, verify each `stmt:true` entry maps to the correct Ora line
3. **Debugger stepping**: step-over through a 5-line function, verify it visits lines in order
4. **Missing locations**: Compile contract with dispatch table, verify internal ops have no source entries
5. **Comptime folding**: Compile function where all lines are comptime, verify source map shows only the return line, with `[comptime]` annotations for folded lines
6. **End-to-end**: `ora debug examples/simple.ora` launches TUI at line 1 of the contract
7. **Phase 5 verification**: Compile comptime-heavy contract, verify trace events match manual evaluation, verify TUI shows correct fold tree

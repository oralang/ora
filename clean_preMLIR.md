# Pre-MLIR Compiler Cleanup Plan

Scope: frontend and semantic compiler modules before MLIR lowering. This includes
`lexer`, `syntax`, `ast`, `source`, `db`, `sema`, `comptime`, `abi`, `types`,
`imports`, and `driver`.

The practical direction is to create global reusable primitives first, then
simplify stages around them. Zig `comptime` should be used where it removes
classes of repeated local code, not as a micro-optimization pass.

This is a pre-MLIR architecture program, not a single cleanup PR. It touches
every frontend module and is months of work. It must be sequenced by value and
risk, not executed top-to-bottom. See "Sequencing By Value And Risk" below
before starting any phase.

## Core Compiler Principle: No Hidden Behavior, Fail Fast

This is a foundational principle of the Ora compiler, above any cleanup phase. It
applies to all current and future work, and every review gates on it.

Ora is a smart-contract language whose pillars are auditability and formal
verification. Transparency is required at all cost. The compiler must never
silently "be smart" about user code or invent behavior the user did not write.

Rules:

- **No default paths for semantic decisions.** Type, width, signedness, region,
  encoding, and layout decisions must come from explicit, resolved facts. An
  indeterminate value at a lowering, encoding, or codegen boundary **halts
  compilation with a diagnostic** — it never assumes a value (no
  `bits orelse 256`, no `signed orelse false`, no silent `orelse .unknown`).
- **Fail fast, fail closed.** When the compiler cannot prove what the user meant,
  it stops and says so. It does not guess, pad, truncate, widen, or pick a
  "reasonable" fallback. A wrong-but-plausible compile is the worst outcome for a
  contract.
- **Defaults that are genuine language rules live in exactly one place.** If a
  default is ever accepted as a language rule, it is applied once, explicitly, at
  a single resolution point, producing a fully resolved type. Downstream
  consumers receive resolved facts and error on anything indeterminate.
  Defaulting is never re-invented by individual consumers.
- **No hidden behavior in lowering.** Each stage lowers what the resolved facts
  say and nothing else. Convenience fallbacks that mask missing information are
  prohibited.

P2.2 integer-model decision (architect, 2026-06-01): an untyped integer literal
does **not** default to `u256`. It remains a comptime integer until a local
context resolves it. If no local context exists at a runtime boundary, the
compiler emits a diagnostic instead of inventing a width.

Resolved P2.2 enforcement stack:

- `IntegerType.bits` and `IntegerType.signed` are non-optional for resolved
  runtime integers; unresolved literals use a distinct `comptime_integer` kind.
- `resolveIntegerExpression` is the single sema gate that turns a comptime
  integer into a resolved integer through explicit context and fit checks.
- `scripts/check-no-width-defaults.sh` is wired into the build and rejects
  reintroduced `.bits orelse`, `.signed orelse`, and stale optional-integer
  construction patterns.
- The diagnostics matrix pins overflow, negative-to-unsigned, mixed resolved
  integer arithmetic/comparison, and peer-context overflow.

Historical remediation method — strictly 1-to-1, never a blanket sweep. For each
defaulting site, trace where its integer comes from and decide individually:

- if `bits`/`signed` are guaranteed resolved there, the `orelse` is dead code
  masking that guarantee — replace with a fail-closed error/diagnostic;
- if the value can actually be null there, that is an upstream bug (an
  unresolved type reached lowering) — fix it upstream; the site only surfaced it.

A mechanical find-replace of `orelse 256` would itself violate the principle by
"being smart" about sites it did not actually diagnose. End state is uniform
(zero defaults); the diagnosis is per-site.

Ratified integer model (architect, 2026-06-01) — the concrete application of this
principle to integers. Full spec, with coercion rules, fail-closed boundary,
enforcement, and test matrix:
**[docs/compiler/integer-type-model.md](docs/compiler/integer-type-model.md)**.
Implement from that spec; the decisions below are firm:

- A bare integer literal has **no type** — it is a `ComptimeInt` (arbitrary
  precision, comptime-only). It acquires a concrete type only from explicit
  context, fit-checked, or it is a compile error. It never defaults to a width.
- Two distinct kinds: `ComptimeInt` (no width/sign) and `ResolvedInt{bits,signed}`
  with **non-optional** fields — the only integer with a runtime representation.
  This split makes the ~10 silent defaults uncompilable, not merely removed.
- **D1:** mixed resolved widths/signs require an explicit cast. Current
  user-facing diagnostics report invalid binary operators for incompatible
  integer types. No implicit widening or sign conversion.
- **D2:** unannotated integers resolve from local context only (no whole-program
  inference); unresolvable forms diagnose at the binding/use site instead of
  defaulting.
- **D3:** comptime integer math is arbitrary precision; the `≤ i256`/`u256` bound
  is enforced only at binding to a runtime location.

These rules gate every future integer-resolution review.

## Zig Implementation Principles

Target Zig version: the package declares `.minimum_zig_version = "0.15.0"` and
the cleanup should use Zig 0.15 APIs and idioms.

This cleanup is not only about Zig `comptime`. `comptime` is useful for
generating stage views, visitors, typed-ID helpers, and repeated table-derived
APIs, but compiler speed should mostly come from explicit data layout,
allocation discipline, and predictable query ownership.

Use Zig broadly:

- Prefer flat arrays keyed by typed IDs over pointer-heavy object graphs.
- Store hot per-node facts in dense side tables indexed by `id.index()`.
- Keep walkers allocation-free; visitors and result builders own state.
- Use Zig 0.15 collection style consistently: initialize lists/maps explicitly,
  pass allocators to list operations such as `append`, `toOwnedSlice`, `writer`,
  and `deinit`, and keep allocator ownership visible at call sites.
- Allocate durable query results from their result arena, and drop the whole
  arena on invalidation.
- Use short-lived scratch lists/arenas only while building frozen result slices.
- Prefer packed structs/bitsets for boolean flags and dense membership facts.
- Keep stage boundaries explicit with small capability views instead of passing
  large mutable compiler state everywhere.
- Use `comptime` for table-derived APIs and restricted views, not for hiding
  runtime complexity.
- Measure hot paths before replacing simple dense loops with heavier generic
  abstractions.

## Implementation Rules For Cleanup PRs

This is the single normative rule list for code produced from this plan. The
"Zig Implementation Principles" section above is the narrative rationale; this
section is what PRs are held to. Where they overlap, this list wins.

The goal is correctness-preserving de-duplication that also avoids accidental
slowdowns — primarily by eliminating repeated scans and recomputation. It is not
a performance project: do not regress, do not add micro-optimization machinery.

Data layout and allocation:

- Follow Zig 0.15 APIs and idioms. Keep allocator ownership visible at call
  sites.
- Prefer data-oriented designs: typed IDs, dense arrays, packed facts, and
  cache-friendly sequential access.
- Prefer packed structs or integer bitsets for boolean fact groups. Avoid many
  independent boolean arrays unless separate storage is measurably useful.
- Prefer sorted slices plus binary search for stable indexes built once and
  queried many times. Use hash maps only for sparse, dynamic, or genuinely
  string-keyed data.
- Avoid per-node heap allocation in hot compiler stages. Allocate dense result
  arrays from the owning query result arena and temporary builder state from
  scratch lists/arenas.

Complexity discipline:

- Avoid accidental `O(n^2)` behavior from repeated local scans over AST nodes,
  members, fields, variants, methods, imports, or type records.
- Do not put an unbounded scan inside code that runs once per AST node unless the
  scanned collection is proven tiny, bounded by language rules, or documented as
  intentionally acceptable.
- Prefer forward fact collection plus indexed lookup over backtracking through
  AST, semantic, import, or module graphs to answer local questions. For graph
  traversal, use an explicit worklist/visited set and freeze the result into the
  owning query result or stage summary rather than rescanning.
- Cross-module facts must come from DB/query results, not repeated recomputation
  by each stage.
- New per-node analyses must state how they avoid nested unbounded scans, using
  dense facts, P1.6 indexes, query results, or a documented bounded collection.
- New graph-shaped analyses must state where visited state and computed facts are
  stored, and which later stages consume those facts instead of rescanning.
- Every new reusable index should document its build cost, lookup cost, owner,
  and invalidation lifetime. If a pass is intentionally worse than near-linear in
  an input dimension, document the feature bound and why the cost is acceptable.

`comptime` and helpers:

- Use `comptime` when it removes repeated hand-written code, creates restricted
  stage views, generates table-derived APIs, or specializes visitors/collections
  without runtime dispatch.
- Do not use `comptime` to hide large runtime behavior or create abstractions
  that are harder to inspect than the repeated code they replace.
- Keep generic helpers small and explicit. A helper should remove real
  duplication or encode a real invariant, not obscure a short local loop.

Out of scope:

- SIMD (`@Vector`, `std.simd`) and inline assembly are out of scope for this
  cleanup. Do not reach for SIMD or asm as a default implementation rule. If
  profiling later proves a specific pre-MLIR loop is hot after data-layout and
  Zig-level work, treat that as a separate, measured optimization with its own
  justification and a scalar fallback — not part of these PRs.

Ownership and traversal:

- Keep semantic ownership clear: AST owns syntax shape, sema owns semantic
  facts, comptime owns evaluation state, ABI owns encoding layout, and DB owns
  query caching/invalidation.
- New tables should be canonical and derive secondary APIs from one source of
  truth. Do not add another local switch if a global table should own the fact.
- New passes should reuse the shared AST walker and side-table/index helpers
  unless they have a documented reason to need custom traversal.

PR notes:

- PRs that remove repeated full-body walks or replace repeated linear lookup
  should mention the avoided work in the PR notes, even if no benchmark is added.
- PRs that claim performance wins should include before/after measurements or a
  narrow explanation of the eliminated work, such as one full-body walk removed
  or an `O(n)` lookup replaced by an indexed lookup.
- Every de-duplication slice must end with a mechanical `rg` proof that the
  collapsed pattern has no remaining definitions outside the canonical owner. If
  an exception remains, list it explicitly as tracked debt with owner, reason,
  and cleanup phase. Do not rely on the original audit list being exhaustive.
- When a duplicated pattern is easy to reintroduce and affects semantic
  correctness, add a build-time tripwire script, following the style of
  `check-abi-layout-ownership.sh`, so future local parsers/switches fail CI
  instead of relying on review memory.

## Roadmap

### P1 Findings

- P1.1: Canonical Ora type source of truth, including fixed-bytes handling.
  The fixed-bytes parser debt was reopened by sweep, then closed by routing the
  surviving parsers through `types/builtin.zig` and adding a duplicate-parser
  tripwire.
- P1.2: Reusable AST visitor/fold framework for repeated semantic tree walks.
- P1.3: Reusable effect-slot set/algebra instead of local list operations.
- P1.4: Normalized runtime/verification `Effect` model and behavior.
- P1.5: Centralized verification/spec fact model for ghost/spec constructs.
- P1.6: Canonical semantic lookup/index layer for fields, members, variants,
  trait methods, impl methods, and instantiated types.
- P1.7: Canonical ABI layout/support policy shared by sema, ABI, and HIR.

### P2 Findings

- P2.1: Consolidated compiler query capability interface.
- P2.2: Canonical type/value model plus interning/handles for comptime.
- P2.3: Optional syntax ID boilerplate cleanup.
- P2.4: Shared diagnostic builders/categories for repeated diagnostic shapes.

### Sequencing By Value And Risk

The findings above are not a linear to-do list. They differ greatly in value,
risk, and blast radius. Recommended ordering:

1. **Cheap, low-risk wins first.** Delete confirmed dead/transitional code
   (`TypeInfo`/`OraType` reframing below, unused `CommonTypes`), then the pure
   de-duplication items with obvious correctness: P1.1 fixed-bytes consolidation
   and P1.4 effect-flag normalization. These pay down real duplication without
   touching numbering or traversal architecture.
2. **Prerequisite audit before TypeId centralization.** Do not start the
   comptime `TypeId` portion of P1.1/P2.2 until the TypeId-stability audit
   (see "Prerequisite: Comptime TypeId Stability Audit") maps every current ID
   band and proves the blast radius. Numeric IDs are frozen unconditionally; the
   audit decides whether they need golden/snapshot guards and external-contract
   documentation.
3. **Well-defined architectural wins next.** P1.6 semantic lookup indexes and
   P1.7 ABI policy surface address genuine fragmentation and are well scoped.
4. **Speculative framework work last, behind a tripwire.** P1.2 generic walker
   and P2.2 type-model unification carry the most design risk; gate them as
   described in their sections.

No phase is "done" because it was started. Each lands as behavior-preserving
PRs with the validation listed for that phase.

## P1.1: Canonical Ora Type Source Of Truth

### Problem

The compiler currently has several competing type authorities:

- `src/lexer.zig` owns builtin type spellings as keyword entries.
- `src/types/type_info.zig` defines `TypeInfo`/`OraType` and claims to be a
  unified type system, but it is **not** a live competing authority for the main
  sema path: it has only a handful of users versus the hundreds for `sema.Type`,
  and `CommonTypes` (`u8_type`, etc.) is defined and re-exported but unused.
  Treat this as dead/transitional code, not active drift risk (see Phase 6).
- `src/sema/model.zig` defines the semantic `Type` union that the real compiler
  pipeline uses.
- `src/sema/type_descriptors.zig` maps names like `u256`, `address`, and
  `bytes32` into sema types.
- `src/sema/resolve.zig` separately recognizes type-value names.
- `src/comptime/value.zig` defines primitive `TypeId` constants.
- `src/comptime/compiler_ast_eval.zig` maps type IDs to source names, ABI names,
  byte sizes, and path lookup.
- `src/abi/layout.zig` repeats ABI classification and fixed-bytes parsing.
- Fixed-bytes handling is duplicated: `fixedBytesTypeId`,
  `fixedBytesLenForTypeId`, `fixedBytesLenFromName`, and `fixedBytesTypeName`
  live in comptime evaluation, while ABI has its own fixed-bytes helpers.

This creates drift risk. Adding or changing a builtin type requires updating
multiple local switches and parsers.

### Status Correction: Fixed-Bytes Cleanup Reopened And Closed

P1.1 was reopened after a post-slice sweep found surviving fixed-bytes parsers
outside `src/types/builtin.zig`. That sweep found:

| Site | Prior behavior | Cleanup |
| --- | --- | --- |
| `src/sema/type_check.zig` `parseFixedBytesSpelling` | strict-ish local parser | delegated to `builtin.parseFixedBytesName` |
| `src/abi.zig` `parseFixedBytesSpelling` | strict-ish local parser | delegated to `builtin.parseFixedBytesName` |
| `src/abi.zig` `isFixedBytesWireType` | strict-ish local parser | replaced with `builtin.parseFixedBytesName(wire_type) != null` |
| `src/sema/type_check.zig` `runtimeFixedBytesSpellingLen` | lenient `parseInt` path; accepted `bytes01`, `bytes+5`, and `bytes1_6` | replaced with the canonical strict parser and rejection tests |

The last site is a real behavioral fork. The accepted language decision is that
only `bytes1` through `bytes32` are fixed-bytes spellings; signed, zero-padded,
separator, or otherwise decorated forms are ordinary identifiers or invalid in
their later context. Cleanup of these sites is therefore a correctness fix, not
legacy compatibility.

The durable tripwire is `scripts/check-no-duplicate-fixed-bytes-parsers.sh`,
wired into `zig build test`. It fails if local fixed-bytes parsing reappears
outside `src/types/builtin.zig` or the explicitly allowlisted ABI delegation
wrappers. The same pattern should later cover local integer-width parsers and
per-payload effect-flag switches.

### Goal

There should be one low-level type authority, independent of AST, sema, ABI,
MLIR, and comptime evaluation.

The target shape is a canonical `src/types/` module that provides:

- builtin type metadata
- primitive type lookup by source spelling
- integer/fixed-bytes helpers
- canonical source/display names
- stable builtin IDs for comptime
- semantic primitive construction helpers

AST type expressions remain syntax. They should not become semantic types.
Sema should own resolution policy, but not duplicate builtin facts.

### Non-Goals

- Do not redesign green/red syntax.
- Do not replace the Salsa-like DB in this pass.
- Do not move MLIR dialect type definitions in this pass.
- Do not collapse runtime ABI layout trees into the type model.
- Do not make source tokens carry semantic types.

## Target Design

### Builtin Type Table

Create a single `comptime` table in `src/types`, for example
`src/types/builtin.zig`.

Expected shape:

```zig
pub const BuiltinTypeId = enum(u16) {
    u8,
    u16,
    u32,
    u64,
    u128,
    u160,
    u256,
    i8,
    i16,
    i32,
    i64,
    i128,
    i256,
    bool,
    address,
    string,
    bytes, // dynamic bytes
    void,
};

pub const BuiltinTypeSpec = struct {
    id: BuiltinTypeId,
    source_name: []const u8,
    category: TypeCategory,
    bit_width: ?u16 = null,
    byte_width: ?u16 = null,
    signed: ?bool = null,
    comptime_type_id: u32,
};

pub const builtin_types = [_]BuiltinTypeSpec{
    .{ .id = .u8, .source_name = "u8", .category = .integer, .bit_width = 8, .byte_width = 1, .signed = false, .comptime_type_id = 1 },
    .{ .id = .address, .source_name = "address", .category = .address, .bit_width = 160, .byte_width = 20, .comptime_type_id = 14 },
};
```

The exact fields can change, but each existing local fact must have one owner.
ABI names, static word counts, and support policy are owned by P1.7's ABI
policy layer, keyed by `BuiltinTypeId` where needed. Do not put ABI policy facts
in `BuiltinTypeSpec`.

Fixed bytes should be owned by the same module, even if represented as a
parameterized builtin family rather than 32 separate rows:

```zig
pub const FixedBytesSpec = struct {
    min_len: u8 = 1,
    max_len: u8 = 32,
    prefix: []const u8 = "bytes",
};
```

Derived APIs should include:

- `lookupBuiltinByName(name: []const u8) ?BuiltinTypeSpec`
- `lookupBuiltinById(id: BuiltinTypeId) BuiltinTypeSpec`
- `lookupBuiltinByComptimeTypeId(id: u32) ?BuiltinTypeSpec`
- `builtinName(id: BuiltinTypeId) []const u8`
- `builtinByteWidth(id: BuiltinTypeId) ?u16`
- `builtinBitWidth(id: BuiltinTypeId) ?u16`
- `builtinSignedness(id: BuiltinTypeId) ?bool`
- `parseIntegerBuiltin(name: []const u8) ?BuiltinTypeSpec`
- `parseFixedBytesName(name: []const u8) ?u8`
- `fixedBytesName(buffer_or_allocator, len: u8)`
- `isBuiltinTypeName(name: []const u8) bool`

Sema-owned helpers, not `src/types` APIs:

- `semanticTypeFromBuiltin(spec: BuiltinTypeSpec, spelling: []const u8)`
- `semanticFixedBytesType(name: []const u8, len: u8)`

### Canonical Semantic Type

The real semantic type model now lives in `src/types`, not `src/sema`.

Implemented direction:

- `src/types/semantic.zig` owns the semantic type union and related neutral
  type facts.
- `src/sema/model.zig` re-exports those neutral types for sema-owned result
  containers and compatibility callers.
- Retire or shrink `TypeInfo`/`OraType` so there is not a second semantic type
  universe.

Near-term compatibility is acceptable:

```zig
pub const Type = @import("../types/semantic.zig").Type;
pub const IntegerType = @import("../types/semantic.zig").IntegerType;
```

The important rule: sema can interpret AST and build semantic types, but builtin
type definitions come from `types/builtin.zig`.

## Migration Plan

### Prerequisite: Comptime TypeId Stability Audit

This audit is a hard gate. It must complete before any phase that centralizes,
aliases, or rewires comptime `TypeId` values (Phase 4 here, plus the related
P2.2 work). The cleanup must not renumber existing IDs.

Context: `src/comptime/value.zig` declares stable primitive `TypeId` constants
(`u8_id = 1` through `void_id = 17`, plus `u160_id = 18`), and
`src/comptime/compiler_ast_eval.zig` relies on fixed numeric ranges for fixed
bytes, ABI-decode error encoding, and named types. Current known bands are:

- primitive static builtins: `1..18` (`u160` is assigned `18`; existing IDs
  remain frozen)
- local test-only sentinels: `42` in the comptime heap tests and `100` in the
  comptime value/pool tests
- fixed bytes: `500_000 + len`, currently `500_001..500_032`
- ABI decode error ADT: `900_000`
- named types:
  `1_000_000 + module_component * 100_000 + item_component`

Roughly two hundred sites consume these IDs. If any ID value is baked into
emitted artifacts, reordering or regenerating the enum silently changes output.

The audit must answer, with evidence:

- Are any numeric `TypeId` values written into emitted MLIR attributes, ABI
  reflection output, debug artifacts, or serialized caches?
- Are any of them embedded in committed test snapshots / golden files?
- Are the fixed numeric ranges in `compiler_ast_eval.zig` (fixed bytes, ABI
  decode error, named types) assumed by any consumer outside comptime?
- Are local sentinel/test IDs such as `100` truly test-only, or do they need a
  documented reserved band?

Outcome:

- IDs remain frozen regardless of audit outcome. The value of this cleanup is
  one source of truth and generated lookup code, not new numeric assignments.
- If any ID is externally observable, record the numeric assignments as an
  external contract and add golden/snapshot regression coverage for the sink.
- If IDs are purely in-memory and not snapshot-sensitive, keep the same frozen
  assignments with lighter compile-time invariants.
- Add comptime assertions that static builtin IDs are unique, all known bands
  are disjoint, and formulas cannot collide with primitive or special bands.

Audit result, 2026-06-01:

- Numeric comptime `TypeId` values are currently an in-memory compiler contract,
  not an external artifact contract.
- `src/comptime/value.zig` and `src/comptime/compiler_ast_eval.zig` are the
  production numeric authorities today. `src/abi/comptime_decoder.zig` receives
  IDs through a resolver and stores them only inside evaluator `CtValue`
  enum/struct values.
- `src/sema/model.zig` `ConstValue` does not carry comptime `TypeId`, and HIR
  lowering emits MLIR attributes from that sema value model, not from comptime
  numeric IDs. Non-folded type values lower as the `"ora.type_value"`
  placeholder, not as a numeric ID.
- Debug artifacts format sema `ConstValue` through `formatConstDebugValue`;
  those formatted values do not include comptime `TypeId`.
- ABI manifest `typeId` fields in `src/abi.zig` are hashed string IDs from the
  ABI canonical payload (`t:...`), separate from comptime `u32` TypeIds.
- The old comptime persistence prototype preserved TypeIds, but it was not
  wired to a serializer or emitted cache and has since been removed as dead
  infrastructure.
- The only observed raw sentinel IDs are test-only: `42` in
  `src/comptime/heap.zig` tests and `100` in `src/comptime/value.zig` tests.

Guard decision:

- No external golden/snapshot guard is required for the current code because no
  emitted MLIR, ABI manifest, debug artifact, or serialized cache was found to
  contain numeric comptime `TypeId` values.
- The numeric assignments still remain frozen as an internal compiler ABI.
  Phase 4 may replace duplicated lookup code with table-derived code, but must
  preserve the explicit primitive IDs and the existing formulas for fixed bytes,
  ABI-decode error, and named types.
- If a future compiler cache or debug/ABI/MLIR sink starts serializing numeric
  comptime `TypeId`, that change must add a golden/snapshot guard and document
  the IDs as an external contract before it lands.

### Phase 1: Add Central Builtin Metadata

Add `src/types/builtin.zig` and export it from `src/types/root.zig`.

Include all primitive and fixed-bytes facts currently scattered across:

- `src/lexer.zig`
- `src/types/type_info.zig`
- `src/sema/type_descriptors.zig`
- `src/sema/resolve.zig`
- `src/comptime/value.zig`
- `src/comptime/compiler_ast_eval.zig`
- `src/abi/layout.zig`

Validation:

- `zig build test-types`
- Existing compiler tests should still pass.
- New unit tests for builtin lookup, integer metadata, and fixed-bytes parsing.

### Phase 2: Replace Sema Primitive Lookup

Update `descriptorFromPathName` in `src/sema/type_descriptors.zig` to use the
builtin table.

Current repeated logic:

- `void`
- `bool`
- `string`
- `address`
- `bytes`
- integer spellings
- fixed bytes spellings

Integer type spellings are a closed language set:

- `u8`, `u16`, `u32`, `u64`, `u128`, `u160`, `u256`
- `i8`, `i16`, `i32`, `i64`, `i128`, `i256`

Unsupported integer-looking names such as `u24`, `i96`, `u01`, or `u1_6` must
not fall through to `.named`. They should resolve to `unknown` and, when they
come from a source type expression, produce a specific diagnostic.

Target:

```zig
if (builtin.lookupBuiltinByName(trimmed)) |spec| {
    return semaBuiltinTypeFromSpec(spec, trimmed);
}
if (builtin.parseFixedBytesName(trimmed)) |len| {
    return fixedBytesType(trimmed, len);
}
if (invalidIntegerTypeName(trimmed)) {
    return .{ .unknown = {} };
}
```

Validation:

- Type descriptor tests.
- Sema tests covering primitive annotations, unsupported integer widths, fixed
  bytes, and named types.

### Phase 3: Replace Resolver Type-Value Recognition

Update `isRecognizedTypeValueName` in `src/sema/resolve.zig`.

Target behavior:

- Ask `types/builtin.zig` for builtin type names.
- Preserve special values like `std`, `Ok`, and `Err` outside the builtin type
  table.

This removes another hand-written primitive list.

### Phase 4: Replace Comptime TypeId And Fixed-Bytes Tables

Gated on the Comptime TypeId Stability Audit above. Do not start this phase
until that audit maps the existing bands and documents the guard level. This
phase must preserve the exact existing numeric assignments.

Update `src/comptime/value.zig` so primitive `TypeId` constants are declared in
one place as explicit frozen values from builtin specs. Generate lookup code
from the table; do not generate numeric IDs from row order.

The table owns only the static builtin ID band (`1..18`). Keep dynamic and
special bands as named constants/formulas:

- fixed bytes: `fixed_bytes_type_id_base + len`
- ABI decode error ADT: `abi_decode_error_type_id`
- named types: `named_type_id_base + module_offset + item_offset`

`typeNameForTypeId` remains a band dispatcher. It should delegate the primitive
band to table-generated lookup code, fixed bytes to the fixed-bytes formula, and
named types to existing named-type resolution.

Update `src/comptime/compiler_ast_eval.zig`:

- `typeNameForTypeId` uses builtin lookup by comptime type ID.
- `abiTypeNameForTypeId` uses ABI-owned primitive/fixed-bytes name helpers
  keyed by builtin classification; do not add ABI names to `BuiltinTypeSpec`.
  Note this introduces a deliberate `comptime -> abi` dependency edge (comptime
  asks the ABI policy layer for ABI names rather than owning them locally). That
  direction is correct per the ownership rules, but it is a new edge: confirm it
  does not create an import cycle with P1.7, and if it would, expose the ABI
  name helper through a small shared policy module both layers import.
- `typeByteSizeForTypeId` uses builtin byte widths.
- `pathTypeId` uses builtin lookup by source name.
- `fixedBytesTypeId`, `fixedBytesLenForTypeId`, `fixedBytesLenFromName`, and
  `fixedBytesTypeName` move to `src/types/builtin.zig` or a sibling
  `src/types/fixed_bytes.zig` re-exported by the builtin module.

Status, 2026-06-01:

- Implemented for the comptime evaluator. Primitive source names, ABI names,
  byte widths, type-name lookup, path-to-TypeId lookup, and model-type-to-TypeId
  lookup now delegate to the builtin table plus ABI-owned name helpers.
- Numeric assignments remain frozen in `src/types/builtin.zig`; `src/comptime/value.zig`
  derives its exported primitive constants from those explicit rows.
- `typeNameForTypeId` remains a band dispatcher: fixed bytes use the fixed-bytes
  formula, primitive builtins use table lookup, and named types keep the existing
  named-type formula.

Validation:

- Comptime tests.
- ABI reflection tests.
- Any tests covering `@typeName`, ABI names, `sizeof`, or type-valued comptime
  expressions.

### Phase 5: Replace ABI Primitive And Fixed-Bytes Classification

Update `src/abi/layout.zig` to use canonical builtin facts for:

- integer signedness
- integer width
- `bool`
- `address`
- `bytes`
- `string`
- fixed bytes parsing

ABI layout can still build ABI-specific layout trees, ABI names, static word
counts, and supportability decisions. The point is to stop re-parsing primitive
spellings and fixed-bytes names locally while keeping ABI policy in the ABI
layer.

Validation:

- ABI layout tests.
- ABI runtime encoder/decoder tests.
- Compiler ABI tests.

Additional gate:

- `src/abi/layout.zig` must fail closed if `IntegerType.bits` or
  `IntegerType.signed` is null. ABI layout may validate resolved integer facts
  against `types/builtin.zig`, but it must not infer missing width/signedness
  from spelling or silently fall back to `u256`.

### Phase 6: Decide Fate Of `TypeInfo`/`OraType`

Although numbered last, this is largely dead/transitional-code cleanup and can
be done early and cheaply (see "Sequencing By Value And Risk"). It is mostly a
deletion, not a migration.

Audit actual users of `src/types/type_info.zig`. Expect very few on the main
sema path. Take care with two consumer groups before deleting:

- `src/comptime/value.zig` and comptime evaluation, which may still reference
  these types or their constants.
- Legacy MLIR users that predate the canonical semantic `Type`.

Likely outcome:

- Keep `TypeInfo` only if it has a distinct role as metadata wrapper:
  source span, inferred/explicit source, optional region.
- Remove `OraType` as a parallel type universe, or make it a compatibility alias
  around canonical semantic `Type`.

Rules:

- There should not be two independent unions for the same language type facts.
- `TypeInfo` must not define primitive widths, names, ABI names, or category
  logic separately from the builtin table.

### Phase 7: Lexer Integration

Lexer keyword entries may remain a table, but builtin type keyword entries
should be generated from the builtin type table or appended from it at comptime.

This debt is now quantified. Adding `u160` required roughly nine token-layer
hand edits across `lexer.zig`, parser identifier/type-token switches,
`syntax_lowering.zig`, and LSP semantic token classification. That means the
"add a width by adding one builtin table row" invariant is not true at the
token layer yet, even though sema can now consume the table correctly.

Short-term mitigation is now in place: lexer exposes a builtin-type keyword
token set derived from `types/builtin.zig`, and parser/syntax/LSP consumers use
that predicate instead of repeating every integer width. The remaining manual
surface is the `TokenType` enum and keyword row itself.

If direct generation is awkward because `TokenType` lives in `lexer.zig`, a
short-term acceptable step is:

- keep `TokenType`
- add a compile-time validation test that every builtin type source name has a
  matching keyword/token entry

Long-term, the two candidate models were:

- builtin types do not need dedicated token kinds and parse as identifiers
  (prelude-identifier model, as in Zig/Rust), or
- builtin types stay reserved keyword tokens, with the keyword rows
  generated/validated from `types/builtin.zig` (keyword model).

Decision (architect, 2026-06-01): adopt the **keyword model**. Dedicated token
kinds stay; the short-term mitigation above is the permanent answer. Phase 7 is
considered closed under this decision. Rationale, specific to Ora:

- Ora already chose a closed, curated integer-width set (`u24`/`i96` are
  rejected). That is the keyword philosophy — the blessed set is the lexical
  set. The identifier model's main payoff is free arbitrary widths, which Ora
  deliberately discarded; adopting it would pay that model's costs without its
  benefit and force a "lex as identifier, reject in resolver" split.
- Auditability/verification is a pillar. Reserved primitive tokens make shadowing
  of `address`/`bool`/`u256`/`bytes32` impossible at the lexer, ironclad. The
  identifier model can only approximate that via resolver rules and risks holes.
- The original motivation (drift across ~9 enumerated switches) is already
  neutralized by the compile-time sync guard, so there is no maintenance pressure
  to switch. New widths are a rare, compile-enforced 3-edit event.

Revisit condition: if Ora later decides to support a large or arbitrary
integer-width family (e.g. odd widths like `u24` for gas-optimal EVM storage
packing), the economics flip toward the prelude-identifier model. At that point,
reopen Phase 7 as its own language-design slice with explicit reserved-word
handling — not as a refactoring chore.

## Parked Decision: Integer Width Family vs Bitfield Packing

Revisit at the END of this plan, with real contract use-cases in hand. This is a
language-design decision, not cleanup; it must not ride on a refactor slice.

Constraints that frame the decision:

- **ABI boundary is hard.** The EVM/Solidity ABI encodes integers only in
  multiples of 8 (`uint8`..`uint256`). Arbitrary widths (`u7`, `u23`) cannot be
  public params or `abi.encode`/`decode`d. Supporting them would create a
  two-tier type system (ABI-able vs internal-only) — rejected. Any integer-family
  expansion is multiples of 8 only.
- **Split the motivation first.** Sub-byte/bit-granular packing (`u3`, `u6`,
  flag sets) is `bitfield` territory, not the scalar integer family. Byte-granular
  arithmetic values (`u24`, `u48`, `u96` as the `address`+`u96` slot complement)
  are the only integer-type case. Decide whether the real need is storage density
  (use bitfields/packed structs — already in the language) or odd-byte arithmetic.
- **No curated subset.** Adding only `u96` recreates "why `u96` and not `u72`".
  It is binary: keep the minimal curated set (`u8/16/32/64/128/160/256`) and pack
  via bitfields, OR go full Solidity parity (every multiple of 8, `u8`..`u256` +
  signed, 64 types). No arbitrary cutoff.
- **Cost is conceptual, not technical.** Z3 handles any bitvector width; overflow
  is mechanical per width; EVM codegen already masks/extends sub-256 values
  (parameterized by width, no new code path); the table-driven design makes new
  rows trivial. The real cost of "all multiples of 8" is 64 integer types of
  surface for auditors/users to reason about vs 7.

Architect lean (not final): keep the curated set and use bitfields for packing,
unless concrete use-cases show genuine demand for odd-byte arithmetic across the
ABI. Expanding to 64 types is hard to reverse and is permanent surface; bitfields
are the EVM-idiomatic packing tool already present.

## Expected Deletions Or Shrinkage

The cleanup should remove or significantly reduce:

- `OraType.getCategory`
- `OraType.isInteger`
- `OraType.isUnsignedInteger`
- `OraType.isSignedInteger`
- `OraType.bitWidth`
- `OraType.toString` primitive cases
- `CommonTypes.u8_type`, `u16_type`, etc.
- the primitive-classification body of `comptime.typeNameForTypeId` (the
  function remains as a band dispatcher; only the hand-written primitive switch
  is replaced by table lookup)
- the primitive-classification body of `comptime.abiTypeNameForTypeId` (the
  function remains; its primitive/fixed-bytes naming delegates to ABI-owned
  helpers)
- `comptime.typeByteSizeForTypeId` primitive cases
- `comptime.pathTypeId` primitive chain
- `sema.descriptorFromPathName` primitive chain
- `resolve.isRecognizedTypeValueName` primitive chain
- duplicate fixed-bytes parsers in sema, comptime, and ABI
- ABI integer spelling helpers where possible

## Invariants

- Adding a primitive type changes one builtin table row.
- Adding `u512` or `i512` does not require hand-editing sema, comptime, ABI,
  resolver, and display code separately.
- `bytes32` parsing is owned by one helper.
- Source/display names are owned by builtin metadata; ABI names are owned by ABI
  policy.
- Comptime `TypeId` numeric assignments are frozen and declared explicitly in
  one place.
- Generated lookup code must not generate numeric `TypeId` assignments from row
  order.
- TypeId bands are disjoint by comptime assertion.
- AST `TypeExpr` remains syntactic and does not become semantic type state.
- Sema remains the stage that resolves names and produces semantic types.

## Tests To Add

- Every builtin source name resolves through sema type descriptors.
- Every builtin source name resolves to a comptime type ID.
- Every comptime builtin type ID maps back to source name.
- ABI policy tests cover integer/fixed-bytes ABI names; those names are not
  stored in `BuiltinTypeSpec`.
- Byte widths for integer/address/bool/void match expected values.
- Fixed bytes accepts `bytes1` through `bytes32`.
- Fixed bytes rejects `bytes0`, `bytes33`, `bytes01`, `bytes+5`, `bytes1_6`,
  and bare `bytes` as fixed bytes.
- Resolver recognizes builtin type-value names via the builtin table.
- Lexer keyword table contains every builtin type name, or parser accepts those
  names as identifiers if lexer token kinds are removed later.

## Remaining Findings

After P1.1, continue with the broader cleanup items:

### P1.2: Performance-Oriented AST Visitor/Fold Framework

`sema/type_check.zig` has multiple hand-written AST walks over almost the same
statement/expression structure. Compare external-call validation, lock
validation, effect collection, and direct-callee collection. These are the same
traversal skeleton with different hooks.

This is not parser-style backtracking over the AST. The current AST is already
flat, ID-addressed, and generally performance-friendly. The problem is repeated
semantic rediscovery: full subtree walks and local linear rescans rebuild the
same facts in several places instead of sharing traversal mechanics, lookup
indexes, and cheap summaries.

Status: closed for the pre-MLIR sema walkers.

Implemented state:

- `src/ast/walk.zig` owns generic `walkBody`, `walkStmt`, `walkExpr`,
  `walkPattern`, and `walkSwitchPattern` helpers.
- Visitors use optional comptime hooks (`enter*` / `exit*`) and return
  `WalkControl` to descend, skip children, or stop.
- `WalkOptions` makes traversal boundaries explicit: switch patterns,
  assignment target patterns, pattern bindings, comptime bodies, quantified
  bodies, external proxy operands, error-return args, and `old(...)` operands.
- `collectNamesInExpr` is now implemented on top of the walker and preserves
  its legacy behavior: it does not enter statement-bearing bodies and it does
  not collect switch-pattern expressions.
- Direct-callee collection, error-type collection, and effect collection now
  use walker visitors instead of local statement/expression/switch recursive
  skeletons.
- External-call and lock validation use the walker for expression and
  switch-pattern traversal, while their statement-level branch merge/freeze
  state machines remain explicit by design.
- `TypeCheckResult` owns dense `PatternId -> initializer ExprId` and
  `PatternId -> BindingKind` facts. This removes lookup-time scans over
  `file.statements` for pattern initializer context and keeps those facts
  module-local when typecheck results are copied for imports.

Resolved traversal targets:

- External-call validation: expression/switch-pattern traversal migrated;
  statement-level branch state remains explicit.
- Lock validation: expression/switch-pattern traversal migrated;
  statement-level lock-set branch state remains explicit.
- Effect collection: body/expression/switch traversal migrated; assignment
  target effects remain a domain-specific helper because they model writes and
  compound-assignment reads, not generic traversal.
- Direct-callee collection: migrated.
- Error-union collection: migrated.

Explicitly not migrated:

- Name resolution in `src/sema/resolve.zig` stays separate. It carries scope and
  environment state, so it needs a scoped-walker design if revisited.
- Branch-sensitive lock/external statement validators stay hand-written under
  the tripwire. Forcing their branch merge/freeze semantics into the generic
  walker would hide semantic state mutation behind traversal mechanics.

Performance risks:

- Multiple passes walk the same body independently to collect facts that could
  be computed once per body or per expression.
- Recursive boolean queries such as "contains result", "has external call", or
  "has direct callee" can rediscover the same subtree facts.
- Local reverse lookups over flat AST arrays turn typed IDs back into semantic
  context with `O(n)` scans.
- Contract-member and field/callee lookups can become repeated linear searches
  if they are performed inside expression visitors.
- Branch-sensitive validations still need their own state machines, but they
  should consume shared facts and indexes instead of reimplementing traversal.

Tripwire (read before building the framework):

The duplicated walks are real, but the generic walker is the highest-design-risk
item in this plan. It must prove itself on the read-only passes first
(`collectNamesInExpr`, direct-callee collection, error-type/expression facts).
The branch-sensitive validations — lock validation and external-call validation,
which carry merge/freeze state across branches — migrate last. If they do not
fit the walker cleanly, **stop**: do not force them through the framework. In
that case the walker must justify itself on the read-only passes alone, and the
branch-sensitive walks stay hand-written. The failure mode to avoid is shipping
the framework *and* keeping the hand-written walks, leaving both to maintain.

Tripwire outcome:

- Passed for read-only and mostly-read-only collectors: name collection,
  direct-callee collection, error-type collection, and effect collection.
- Partially passed for branch-sensitive validations: their expression traversal
  migrated cleanly, but their statement-level state machines intentionally did
  not.
- No generic body-summary cache was added. The migrated passes did not need it,
  so adding one now would be speculative bloat.

Goal:

- Put traversal mechanics in `src/ast/walk.zig`.
- Keep semantic decisions in `src/sema`.
- Make common traversal shape reusable without forcing every pass into one
  over-general visitor.
- Use Zig `comptime` to specialize hooks and avoid dynamic dispatch overhead.
- Preserve the existing flat AST storage model and add dense side tables where
  repeated lookup is currently being done by scan.
- Cache cheap, stable facts by typed AST ID so later validations can reuse them.
- Make allocator ownership follow compiler result lifetimes instead of hiding
  allocation inside walkers.

Non-goals:

- Do not merge syntax green/red walking with AST walking.
- Do not make AST walking scope-aware by default.
- Do not hide semantic state mutation behind a generic framework.
- Do not migrate all semantic walks in one patch.
- Do not replace the flat AST arrays with pointer-heavy node objects.
- Do not force every semantic pass into a single mega-pass.
- Do not allocate traversal state from `AstFile.arena` or from the generic
  walker.

Target design:

```zig
pub const WalkControl = enum {
    descend,
    skip_children,
    stop,
};

pub fn walkBody(
    comptime Visitor: type,
    visitor: *Visitor,
    file: *const AstFile,
    body_id: BodyId,
    comptime options: WalkOptions,
) anyerror!void;

pub fn walkStmt(
    comptime Visitor: type,
    visitor: *Visitor,
    file: *const AstFile,
    stmt_id: StmtId,
    comptime options: WalkOptions,
) anyerror!void;

pub fn walkExpr(
    comptime Visitor: type,
    visitor: *Visitor,
    file: *const AstFile,
    expr_id: ExprId,
    comptime options: WalkOptions,
) anyerror!void;

pub fn walkPattern(
    comptime Visitor: type,
    visitor: *Visitor,
    file: *const AstFile,
    pattern_id: PatternId,
    comptime options: WalkOptions,
) anyerror!void;

pub fn walkSwitchPattern(
    comptime Visitor: type,
    visitor: *Visitor,
    file: *const AstFile,
    pattern: SwitchPattern,
    comptime options: WalkOptions,
) anyerror!void;
```

The walker solves traversal duplication. It should be paired with small,
explicit ID-indexed helpers for facts and reverse lookups. The first one added
is the dense pattern fact table in `TypeCheckResult`; broader `IdMap` or
`BodySummary` helpers are deferred until a real second use appears.

Allocator policy:

- `src/ast/walk.zig` should not require an allocator for normal traversal.
  Visitors own any state they need.
- Durable typecheck facts and indexes should be allocated from
  `TypeCheckResult.arena`, matching existing arrays like `expr_types`,
  `call_resolutions`, `expr_effects`, and `body_types`.
- Temporary builder state can use short-lived `ArrayList`s or scratch arenas
  during typecheck, then freeze results into slices owned by
  `TypeCheckResult.arena`.
- Module-wide indexes that are independent of typecheck should live in their
  owning result arena, usually `ItemIndexResult.arena`.
- `AstFile.arena` remains AST-owned storage only; semantic caches should not be
  attached to the AST lifetime.
- The compiler DB allocator should remain the parent allocator for query result
  objects and arenas, not the allocator used for hot per-node facts.

Visitor hooks should be optional by convention:

```zig
pub const MyVisitor = struct {
    pub fn enterStmt(self: *@This(), file: *const AstFile, id: StmtId) !WalkControl {
        _ = self;
        _ = file;
        _ = id;
        return .descend;
    }

    pub fn exitExpr(self: *@This(), file: *const AstFile, id: ExprId) !void {
        _ = self;
        _ = file;
        _ = id;
    }
};
```

The walker can use `@hasDecl(Visitor, "enterExpr")` and `@hasDecl(Visitor,
"exitExpr")` to call only hooks that exist. This keeps simple visitors small and
lets the compiler specialize the traversal.

Traversal policy should be explicit:

- `walkExpr` should not automatically enter statement-bearing expression bodies
  unless the caller opts in.
- `walkBody` should recurse into nested block/labeled block bodies by default.
- Switch patterns need their own walker because `ast.SwitchPattern` is not stored
  behind `PatternId`.
- Assignment targets use `PatternId`, but their index expressions are `ExprId`;
  the walker must expose both.

Useful options:

```zig
pub const WalkOptions = struct {
    walk_switch_patterns: bool = true,
    walk_assignment_target_patterns: bool = true,
    walk_pattern_bindings: bool = true,
    enter_comptime_bodies: bool = false,
    enter_quantified_bodies: bool = false,
    walk_external_proxy_exprs: bool = true,
    walk_error_return_args: bool = true,
    walk_old_exprs: bool = true,
};
```

Indexes/caches added:

- `PatternId -> initializer ExprId`.
- `PatternId -> BindingKind`.

Indexes/caches deliberately not added in this slice:

- Callee resolution cache keyed by expression or call expression ID.
- Per-expression generic facts for common subtree predicates.
- Per-body summaries for direct callees/error types.

Those are not rejected; they are deferred until a measured or repeated use makes
them pay for their surface area.

Ownership boundary:

- Semantic member/field/variant/method indexes belong to P1.6. P1.2 should
  consume them when useful, but it should not own their design.

Migration result:

1. Generic AST walkers added in `src/ast/walk.zig`.
2. `collectNamesInExpr` migrated and pinned with a switch-pattern option test.
3. Direct-callee collection migrated.
4. Error-type collection migrated.
5. Effect collection migrated with explicit visitor state and an indexed-storage
   special case that preserves existing effect-summary behavior.
6. Lock/external expression and switch-pattern traversal migrated.
7. Branch-sensitive lock/external statement validators left explicit under the
   tripwire.
8. Dense pattern initializer/binding-kind facts added to remove lookup-time
   statement scans.
9. No generic `IdMap`, body summary cache, or callee cache added yet; no bloat
   without a concrete second use.

Tests to add:

- `walkExpr` switch-pattern option test added.
- Existing compiler/effect/lock/external suites cover the migrated sema
  visitors.
- More direct walker unit tests for every statement/expression shape are useful
  follow-up coverage, but not required to close P1.2 because the migrated passes
  exercise the live compiler traversal paths.

Acceptance criteria:

- Met: migrated sema analyses do not own complete expression/switch traversal
  skeletons.
- Met: traversal choices at semantic boundaries are explicit `WalkOptions`.
- Met: no lookup-time scan over `self.file.statements` remains for
  `PatternId` initializer context.
- Met: the AST walker has no allocator dependency for normal traversal.
- Met: durable pattern facts are arena-owned by `TypeCheckResult`.
- Met: no semantic behavior change intended; special cases were preserved
  explicitly rather than hidden in walker defaults.
- Deferred by design: branch-sensitive statement validators remain explicit,
  and broader body-summary/callee caches wait for concrete demand.

### P1.3: Effect-Slot Set Algebra

Effect-slot set logic is reimplemented locally. `cloneEffectSlots`,
`mergeLockedSlots`, `mergeStorageSlots`, `intersectLockedSlots`,
`intersectStorageSlots`, `appendUniqueSlot`, and `appendUniqueItemId` hand-roll
list/set behavior.

Plan:

- Add a generic small unique set helper parameterized by item type, equality,
  and optional filter.
- Use it for `EffectSlot` and `ast.ItemId`.
- Keep arena allocation explicit.

### P1.4: Centralized Effect Behavior

`effectWrites`, `effectHasExternal`, `effectHasLog`, `effectHasHavoc`,
`effectHasLock`, and `effectHasUnlock` are near-identical switches.

Plan:

- Normalize `Effect` into reads, writes, runtime flags, and verification flags
  instead of a union that repeats the same flags in every payload.
- Keep runtime flags focused on execution/frame/ordering behavior such as
  external calls, logs, locks, and unlocks.
- Keep current `havoc` behavior as a verification flag, not a runtime write.
- Move behavior to methods on `Effect` and `EffectFlags`.
- Adding a new effect flag should require changing the flag spec plus the
  producer/consumer logic, not every union payload and every switch.
- Keep reads/writes extraction explicit and test-covered.

Non-goals:

- Do not fold comptime evaluation behavior into `Effect` in this phase.
- Do not treat every ghost/spec construct as a runtime effect.

### P1.5: Verification/Spec Fact Model

`havoc`, `ghost`, `requires`, `ensures`, `guard`, `invariant`, `assert`,
`assume`, `old`, and quantified expressions are related verification constructs,
but they are currently handled across AST, sema, HIR lowering, and
verification-specific MLIR paths rather than through a clear pre-MLIR fact
model.

Termination measures (`decreases`/`increases`) are not first-class AST/sema
facts today — they surface only in lexer/LSP/MLIR-adjacent paths. Include them
in this fact model only if/when they are surfaced pre-MLIR and AST/sema actually
owns them; until then they are out of scope for P1.5.

This should not be solved by adding more booleans to runtime `Effect`.
Verification constructs need their own semantic facts and context so runtime
effect summaries stay focused on execution/frame behavior.

Plan:

- Define a sema-owned verification/spec fact model for constructs that affect
  proof obligations or verification context.
- Keep `havoc` visible through `Effect.verification` while preserving richer
  target information in verification facts.
- Represent ghost context explicitly for ghost functions, ghost fields,
  constants, trait ghost blocks, and ghost assertions.
- Represent function-level spec clauses by kind: `requires`, `guard`,
  `ensures`, `ensures_ok`, `ensures_err`, and `modifies`.
- Represent loop/contract invariants separately from ordinary runtime side
  effects. Represent termination measures only if/when AST/sema owns them as
  pre-MLIR facts.
- Make HIR/MLIR lowering consume these facts instead of reclassifying ghost/spec
  context locally where possible.

Audit baseline:

- `src/sema/verification.zig` already provides a cached sema query for
  verification facts. It is the correct owner to tighten first; do not create a
  second parallel fact model.
- The current query owns function spec clauses and contract/trait ghost-block
  facts, but the old fact shape was too lossy because every fact kind was an
  `ast.SpecClauseKind`. The model now distinguishes source spec clauses,
  contract invariants, trait ghost `assert`/`assume`, and trait ghost axioms with
  explicit verification context.
- `modifies` source clauses are verification facts, but their checked storage
  slot payload still lives in sema's write-set/framing state
  (`TypeCheckResult.item_modifies`) until the framing payload model is stable.
- HIR no longer locally classifies source-level contract/loop invariant
  lowering or `havoc` MLIR emission. Those source facts are collected by
  `src/sema/verification.zig` and consumed through owner-sorted HIR indexes.
- Remaining invariant/havoc AST uses are classified:
  `src/hir/control_flow.zig` and `src/hir/function_core.zig` still inspect
  `*.invariants.len` only as loop-unroll eligibility guards;
  `src/hir/analysis.zig` still reads `havoc_stmt.name` for carried-local
  mutation analysis, not verification fact classification; `src/hir/support.zig`
  uses `Havoc` only for statement ranges. Comptime invariant evaluation remains
  out of scope for this P1.5 runtime/HIR fact slice.
- Runtime/refinement-generated verification operations such as path assumptions,
  parameter refinement guards, slice/division assertions, and refinement-cleanup
  string handling are not first migration targets. They are lowering-generated
  facts or MLIR-adjacent cleanup, not source-level sema facts.
- Post-model audit result: refinement cleanup still correctly reads MLIR op
  names and `ora.verification_*` attributes because it runs after lowering and
  after Z3 guard proving. Those strings are MLIR metadata, not pre-MLIR source
  fact classification. The cleanup work here is to remove duplicated cleanup
  code, not to force post-lowering attributes back into sema facts.

Status:

- Ghost declaration facts now carry an owning item and distinguish ghost
  functions, fields, constants, and ghost blocks. Ghost assertion/assumption
  facts inside ghost functions and ghost blocks are collected under the owning
  ghost item; trait ghost-block expression axioms keep their existing fact
  surface.
- HIR lowering receives module verification facts and consumes them for ghost
  declaration attributes and ghost assertion context. The old direct
  `function.is_ghost` / `field.is_ghost` / `constant.is_ghost` HIR checks and
  the unused impl ghost-block local path were removed.
- Imported HIR lowering receives the imported module's verification facts
  through the HIR query capability instead of reusing the current module's facts.
- Module verification facts are aggregated recursively over declaration
  ownership (`contract.members`, `impl.methods`, struct/log metadata) with a
  visited bitmap, so HIR sees facts for nested member items without rescanning
  bodies or duplicating facts. Trait ghost blocks remain trait-owned to preserve
  their `trait_ghost_block` context.
- HIR builds an owner-sorted verification fact index once per lowered module;
  item lowering reads only that item's fact entries instead of repeatedly
  scanning the whole module fact list.
- Ordinary item-owned function clauses (`requires`, `guard`, `ensures`,
  `ensures_ok`, `ensures_err`) are lowered from sema verification facts instead
  of direct `function.clauses` walks in HIR. `modifies` lowering is gated by the
  source fact and still consumes the validated slot payload from
  `TypeCheckResult.item_modifies`.
- Verification facts now have a single owner union. Ordinary declarations use
  `.item`; trait method contracts use `.trait_method = (trait ItemId, method
  index)`, matching the existing trait-method lookup index.
- HIR builds a second owner-sorted fact index for trait-method facts. Impl
  method contract inheritance and extern-trait summaries now consume those sema
  facts instead of walking `TraitMethod.clauses` locally.
- Contract invariants and loop invariants are normalized in sema verification
  facts. The fact carries the lowered invariant expression, optional invariant
  label, source range, and context (`.contract` or `.loop`); HIR emits
  `ora.invariant` from the fact instead of reparsing invariant expression shape.
- Statement-level verification facts use `.statement = (item_id, stmt_id)` as
  the owner. HIR builds a third owner-sorted fact index for statement facts so
  each loop or `havoc` statement reads only its own facts.
- `havoc` facts now carry the target spelling in `target_name`. HIR `ora.havoc`
  lowering consumes that target from the sema fact and fails closed with
  `InvalidVerificationFact` if the fact is missing or malformed, instead of
  falling back to the AST statement string.
- HIR and the CLI MLIR path now share the same refinement/verification cleanup
  implementation in `src/mlir/refinement_guards.zig`; the duplicate
  `src/hir/refinement_cleanup.zig` implementation was removed. The shared pass
  still preserves the existing behavior: proven refinement guards are removed,
  unproven guards lower to `cf.assert`, ghost assertions remain verification-only,
  and other verification ops are erased before SIR emission.

Non-goals:

- Do not change verifier semantics in the first migration.
- Do not merge verification facts with runtime `Effect` or comptime evaluation
  state.
- Do not move MLIR verification lowering before the pre-MLIR facts are stable.
- Do not replace post-lowering MLIR attribute strings with sema facts; Z3 and
  SIR cleanup consume those attributes after HIR has already emitted MLIR.

Acceptance criteria:

- Runtime `Effect` does not grow flags for every spec/ghost construct.
- Verification constructs have one pre-MLIR classification point.
- HIR lowering needs fewer local checks to decide ghost/spec context.
- Existing verification MLIR output remains behaviorally unchanged during
  migration.

### P1.6: Semantic Lookup/Index Layer

Semantic lookup is currently implemented as local scans in sema, comptime
evaluation, and HIR lowering. This is separate from AST traversal: a shared
walker can find the expressions, but it should not decide how fields, members,
variants, trait methods, impl methods, or instantiated types are indexed.

Examples of duplicated lookup shape include:

- Field/member lookup in expression analysis.
- Enum variant lookup in sema, comptime evaluation, and HIR lowering.
- Contract member lookup by name and role.
- Trait/interface method lookup.
- Impl method lookup by trait and target type.
- Instantiated struct/enum/bitfield/interface lookup by name.

Current status:

- Struct-literal field lookup is indexed.
- Enum-variant lookup is indexed for sema/HIR/comptime consumers.
- Instantiated struct/enum/bitfield lookup is indexed both in the stable
  `TypeCheckResult` and in the live `TypeChecker` while generic instantiations
  are being created.
- Contract-member lookup by name and role is indexed for sema/HIR/comptime
  consumers; remaining `contract.members` loops are traversal or index
  construction, not local lookup policy.
- Trait/interface and impl method lookup is indexed through semantic interface
  records. AST trait-method lookups by `(trait item, method name)`, raw impl
  method lookups by `(impl item, method name)`, and reverse `method -> impl`
  lookup are indexed in `ItemIndexResult`.
- Struct/bitfield field lookup is indexed in `ItemIndexResult`; instantiated
  struct/bitfield records carry their own field indexes; anonymous-struct field
  lookup is routed through one shared semantic helper.
- Remaining `trait_item.methods`/`impl_item.methods` loops are classified as
  traversal/materialization/index construction: declaration validation,
  resolver binding, visiting/lowering every impl method, building semantic
  method signatures, ghost `self` discovery, and comptime `@traitMethods`
  reflection.
- Remaining field loops are classified as traversal/materialization, ABI
  support checks, struct/anonymous-struct shape construction, literal
  lowering, or bitfield layout prefix accumulation; they are not local by-name
  field lookup policy.
- `catch null` query-failure audit is complete for import/item-index paths:
  sema import helpers, HIR imported-field lowering, comptime callable
  resolution, comptime contract-member paths, and comptime anonymous-struct
  field lookup now propagate query errors instead of treating them as absent.
  The `enumVariantIndex` fallback can still answer from the AST and is an
  accepted equivalent fallback. The `abiDecode*` resolver callbacks now return
  `!?`, so real query failures propagate through ABI decode instead of becoming
  "empty type" answers. Other remaining `catch null` uses are parsers, integer
  conversion, arithmetic probes, allocation in optional formatting helpers, or
  best-effort constant folding and are outside this query-failure audit.

Plan:

- Add sema-owned lookup indexes built from durable query results and allocated
  from the result arena that owns their invalidation.
- Prefer sorted slices plus binary search for stable module facts. Use hash maps
  only for sparse or dynamic lookup sets where sorting is not a good fit.
- Index top-level items, contract members, fields, enum variants, bitfield
  fields, trait methods, impl methods, and instantiated type/interface records.
- Expose typed lookup helpers instead of making callers know storage layout.
- Make comptime evaluation and HIR lowering consume the same lookup surface
  through restricted query views instead of rebuilding local search logic.
- Keep AST walkers and semantic indexes separate: walkers provide traversal;
  indexes provide named semantic facts.

Non-goals:

- Do not introduce a global mutable symbol table.
- Do not replace name resolution scopes in this phase.
- Do not make P1.2 responsible for semantic lookup policy.

Acceptance criteria:

- Field/member/variant/method lookup no longer requires repeated local loops in
  sema, comptime evaluation, and HIR lowering.
- Contract member lookup by role and name is one shared API.
- Trait and impl method lookup use one indexed representation.
- Instantiated type lookup is not repeated as per-caller linear scans.
- Index lifetime is tied to the query result that produced the facts.

### P1.7: ABI Layout/Support Policy

ABI knowledge is not only primitive spelling and fixed-bytes metadata. Public
ABI support rules, runtime `abi.encode`/`abi.decode` support, static word
counts, result-carrier planning, and canonical ABI type strings are currently
owned in more than one place.

Plan:

- Keep primitive and fixed-bytes metadata sourced from P1.1.
- Move public ABI support checks into a single ABI policy surface.
- Move runtime `abi.encode`/`abi.decode` support checks into that same policy
  surface or an explicitly related runtime ABI policy object.
- Make `LayoutContext` the consumer of canonical semantic type information, not
  a parallel place to rediscover type spelling.
- Expose typed APIs for:
  - canonical ABI type name,
  - static word count,
  - public argument/return support,
  - runtime encode/decode support,
  - Result carrier planning.
- Make sema diagnostics call the ABI policy API instead of carrying local ABI
  classification switches.
- Make HIR lowering consume layout/policy results rather than recomputing public
  ABI attributes.

Status:

- Public function ABI validation in sema now calls the ABI public-policy
  surface instead of carrying local support/static-word switches.
- Result carrier planning is ABI-policy-owned; `LayoutContext` delegates to the
  policy while providing context-aware layout facts.
- Deliberate tightening: public-ABI integer word-count and Result dynamic-array
  element checks now fail closed for malformed or non-builtin integers instead
  of silently treating them as one ABI word. Valid builtin integers such as
  `u8` and `u256` still count as one word.
- Runtime `@abiEncode`/`@abiDecode` supportability now calls the ABI
  public-policy surface instead of carrying sema-local classifiers. Deliberate
  tightening: runtime encode/decode checks also fail closed for malformed,
  unresolved, or non-builtin integer types instead of accepting loose `uNNN`
  spellings or defaulting unresolved integers to `u256`.
- HIR lowering now consumes `LayoutContext` for public return ABI strings,
  error selectors/signatures, static word counts, parameter ABI names, and
  Result input modes. `src/hir/abi.zig` no longer owns semantic type-to-ABI
  classification; it only carries selector hashing, metadata wire names, and
  pre-rendered ABI signature string helpers.
- Canonical primitive/fixed-bytes ABI type names are derived through
  `src/abi/type_names.zig`, which consumes the P1.1 builtin table. `LayoutNode`
  rendering no longer formats `uintNNN`, `intNNN`, or `bytesNNN` locally, and
  public return marker names (`void`, `tuple`, `struct`, bitfield-as-`uint256`)
  are owned by `src/abi/policy.zig`.
- Deliberate tightening: direct ABI layout rendering now fails closed for
  invalid static encodings such as `uint24` or `bytes33` instead of rendering
  unsupported ABI names. Layouts produced from semantic types already failed
  closed; this applies the same rule to manually constructed `LayoutNode`s.
- P2.2/P2.1 neutrality tail complete at the ABI import boundary: `policy.zig`
  no longer imports sema query/index records or instantiated field structs, and
  `layout_context.zig` no longer imports sema result/index data or type
  descriptor logic. `LayoutContext` consumes a neutral provider vtable for
  type-expression resolution, named-type shape facts, struct/contract/error
  payload facts, and enum variant counts. The sema-owned adapter that implements
  that provider remains in `src/sema/abi_layout_provider.zig`, which is the
  correct ownership direction. The ABI policy API still uses
  support-classifier return shapes (`bool` / `?usize`) rather than a new
  error-propagating contract; widening that API would be a separate contract
  change.

Non-goals:

- Do not change emitted ABI layout in the first migration.
- Do not mix ABI policy with generic type metadata. The ABI layer can consume
  builtin metadata, but ABI supportability remains an ABI decision.
- Do not optimize with target-specific tricks before the ownership boundary is
  fixed and measured.

Acceptance criteria:

- Sema, ABI, and HIR agree through one ABI policy/layout API.
- Fixed-bytes and integer primitive classification are not duplicated in ABI
  support code.
- Runtime encode/decode support and public ABI support are explicit policies,
  not scattered local switches.
- Existing ABI tests and emitted ABI attributes remain behaviorally unchanged.

### P2.1: Query Capability Interface

`ImportQuery`, `TypeQuery`, and HIR's query struct repeat the same "ask the DB
for cross-module facts" concept with small differences.

Plan:

- Define a global compiler query capability set.
- Expose restricted stage views instead of separate mini-interfaces.
- Keep the existing Salsa-like DB shape for now.

Status: closed.

- The query view type definitions now live in `src/compiler_query.zig`.
  The old stage aliases (`sema.ImportQuery`, comptime `TypeQuery`, and HIR
  `ModuleQuery`) were removed; stage code uses the shared restricted view types
  directly.
- Query views expose methods (`astFile`, `itemIndex`, `moduleTypeCheck`,
  `lookupItem`, etc.), so stage code no longer reaches through
  `query.context` at call sites.
- `CompilerDb` owns construction of the three restricted views through helper
  constructors. Callback fields remain flat internally, but the DB wiring is now
  centralized instead of repeated at each query handoff.
- `zig build check-query-view-ownership` guards the ownership rule: no stage
  local query structs, no raw `query.context` calls outside `src/compiler_query.zig`,
  and DB callback wiring only inside the three view constructors.

### P2.2: Canonical Type/Value Model

`types.OraType`, `sema.Type`, comptime `TypeId`, and semantic `ConstValue`
encode overlapping facts.

The integer-resolution portion has a dedicated ratified spec:
**[docs/compiler/integer-type-model.md](docs/compiler/integer-type-model.md)** —
the `ComptimeInt` vs `ResolvedInt` split, the single resolution gate, coercion
rules, fit checks, fail-closed boundary, enforcement, and a test matrix. The
integer-resolution enforcement stack is implemented: non-optional resolved
integer facts, `comptime_integer` for unresolved literals, the single
`resolveIntegerExpression` gate, the `check-no-width-defaults` tripwire, and the
pinned diagnostics matrix.

Plan:

- Let P1.1 establish type metadata ownership.
- Semantic type ownership now lives in `src/types/semantic.zig`; `sema.model`
  re-exports the type model for sema-facing result containers and compatibility
  callers. Non-sema consumers that only need semantic type facts import
  `ora_types` directly.
- Refinement semantic facts now live in `src/types/refinement_semantics.zig`;
  the old `sema/refinements.zig` facade was removed, and ABI/comptime/HIR
  consume the neutral type-owned helpers instead of importing sema refinement
  logic.
- Keep integer width and signedness resolved exactly once through the sema gate.
  Downstream semantic types carry concrete width/signedness, while unresolved
  literal state is represented separately as `comptime_integer`.
- Keep the `check-no-width-defaults` tripwire active so ABI/HIR/comptime cannot
  reintroduce local `orelse false` / `orelse 256` defaults.
- Introduce interning/handles where comptime needs stable IDs. This is gated on
  the Comptime TypeId Stability Audit (see P1.1 Migration Plan). Do not
  renumber comptime IDs; preserve existing numeric assignments and generate only
  lookup/conversion code around them.
- Avoid parallel unions that need conversion glue.

Status: closed.

- Integer/type model: done and verified. Resolved runtime integers are concrete
  by construction; unresolved literals are represented as `comptime_integer` and
  must pass through `resolveIntegerExpression`.
- Semantic value model: done and verified. The cross-stage semantic constant
  representation lives in `src/types/value.zig`; `sema.model` only re-exports it
  as a compatibility facade.
- Dead comptime persistence prototype: removed. The unused persistence layer is
  no longer public surface and cannot be confused with the semantic
  `ora_types.ConstValue`.
- ABI neutrality tail: done at the ABI import boundary. ABI policy receives
  named-type shape facts through provider methods instead of importing sema
  query/index records. `LayoutContext` receives type-expression resolution and
  layout shape facts through a neutral provider vtable; the sema-owned adapter
  implements that provider outside the ABI layer.
- Cosmetic facade tail: done for the P2.2 boundary. `src/hir/support.zig`
  imports neutral semantic type/refinement facts directly from `ora_types`, and
  `src/comptime/compiler_ast_eval.zig` spells semantic `Type`,
  `IntegerType`, `TypeKind`, and anonymous-struct fields through `ora_types`
  while retaining `sema.model` only for real sema result/key containers.
  Remaining non-sema imports of `sema/mod.zig` or `sema/model.zig` are
  classified as result/query/lowering boundaries (`TypeCheckResult`,
  `ItemIndexResult`, `NameResolutionResult`, `VerificationFact`,
  `ResolvedCall`, `TypeCheckKey`, or DB/driver/compiler public facade wiring),
  not neutral-type ownership leaks.

Value-model scoping note:

- Audit before code changes: enumerate all `sema_model.ConstValue`,
  `comptime.CtValue`, the dead comptime persistence prototype, bridge,
  debug-format, and ABI encoder consumers. Classify each as
  semantic/persistent/evaluator-local before moving anything.
- Keep evaluator-local and persistent concerns separate. `CtValue` may carry
  heap references and belongs to the comptime evaluator. A persistent const value
  must be stable outside one evaluator heap and must not depend on live heap IDs.
- Decide the canonical persistent/semantic value representation first. Either
  move the sema const value shape into a neutral owner under `src/types`, or
  replace sema consumers with the existing comptime persistent value model plus
  explicit handles. Do not add another parallel union.
- Make conversion boundaries explicit and narrow: evaluator value -> persistent
  value at the comptime bridge, persistent/semantic value -> ABI encoding at the
  ABI boundary, and debug formatting from the canonical representation. No local
  ad-hoc conversions.
- Preserve numeric and aggregate semantics. Big integer range/fit information,
  aggregate ordering, struct/enum identity, error-union payload shape, and
  `TypeId` stability must not change as a side effect of unification.
- Acceptance: no behavior change for existing constants, no silent truncation or
  defaulting, full compiler and MLIR suites green, and a sweep proving no
  duplicate public semantic `ConstValue` model remains outside the chosen owner.

Value-model audit result:

- `sema_model.ConstValue` is the active cross-stage semantic constant value. It
  is stored in `ConstEvalResult.values`, read by sema fit/enum checks, HIR
  constant lowering, debug output, tests, and ABI comptime encoding. Its integer
  payload is `std.math.big.int.Managed`, which preserves arbitrary precision and
  negative intermediate constants before type-resolution fit checks.
- `comptime.CtValue` is evaluator-local and may contain `HeapId` references.
  It remains owned by the comptime evaluator and must not become the semantic
  persistent value exposed to sema/HIR/ABI.
- The comptime persistence prototype was not wired into the live compiler DB
  and was removed. It was not a safe drop-in replacement for semantic
  `ConstValue` because its integer and aggregate representation did not preserve
  the full semantic constant shape.
- `src/comptime/compiler_const_bridge.zig` is the current conversion boundary:
  semantic constants <-> evaluator values for the cases that can cross that
  boundary. `src/abi/comptime_encoder.zig` is a second boundary that currently
  accepts either `CtValue` or semantic `ConstValue`.

Value-model decision:

- The canonical semantic constant representation moved to the neutral owner
  `src/types/value.zig`, preserving the previous `sema_model.ConstValue` shape
  and `BigInt` integer semantics. `sema.model` re-exports it for compatibility.
- `ConstEvalResult` remains a query/result container for now because it owns a
  `diagnostics.DiagnosticList`; the standalone `ora_types` package cannot import
  diagnostics without breaking its module boundary. Its `values` field should use
  the neutral semantic `ConstValue`, but moving the result container itself
  belongs with the query/result-interface cleanup, not the pure type package.
- `comptime.CtValue` stays evaluator-local. Do not move heap-backed evaluator
  values into `src/types`.
- Do not reintroduce a public comptime-side semantic constant alias; there must
  not be two public `ConstValue` meanings that look interchangeable.
- Implementation order:
  1. Move semantic `ConstValue` to the neutral type/value owner while keeping
     `sema.ConstValue` as a temporary re-export. Keep `ConstEvalResult` as a
     sema/query result container whose `values` use neutral `ConstValue`.
  2. Retarget HIR lowering, debug formatting, the const bridge, ABI comptime
     encoder, and DB const-eval storage to import the neutral semantic value
     directly.
  3. Remove or quarantine the unused comptime persistence prototype so no public
     API exposes it as the semantic constant model. Done: removed.
  4. Sweep for duplicate `ConstValue` definitions/re-exports and remove stale
     compatibility aliases once call sites are migrated. Production users now
     import `ora_types.ConstValue`; the remaining sema facade is compatibility
     surface, not a second owner.

Separate reliability close-out:

- The intermittent native OraToSIR abort was root-caused to pass-invocation
  cache lifetime. It is closed by replacing the `thread_local` map/memref
  caches and manual clear guard with invocation-owned cache objects passed into
  the relevant lowering patterns (`789cc762`).
- Native sanitizer coverage is available through
  `zig build test-mlir -Dnative-sanitize=address --summary all`. Sanitized SIR
  and Ora dialect artifacts install to a separate `vendor/mlir-*` prefix and
  are linked before the normal MLIR install, so reliability runs cover
  project-owned OraToSIR C++ without rebuilding all of LLVM/MLIR.
- ABI type-shape-only helpers (`layout`, runtime encode/decode, comptime
  decode/test support) now import `ora_types.SemanticType` directly instead of
  reaching through the sema compatibility facade. ABI policy and layout context
  are sema-neutral at the import boundary: policy receives named-type shape
  facts through provider methods, and layout context receives type-expression
  resolution plus layout shape facts through a neutral provider vtable. The
  sema-owned adapter implements that provider in `src/sema/abi_layout_provider.zig`.

### P2.3: Syntax ID Boilerplate

Status: closed.

This finding was overstated in the original draft and was narrowed. AST IDs
already reuse `source.defineId` (`src/ast/ids.zig:6`), and AST builder/accessor
helpers are already centralized through `support.mixin`
(`src/ast/support.zig:18`). That part was already the clean state we wanted.

The genuine remaining issue was syntax green IDs hand-rolling the same enum
shape. `GreenNodeId` and `GreenTokenId` now use `source.defineId`, so source,
syntax, and AST IDs share one ID generator.

Closure:

- `src/syntax/green.zig` uses `source.defineId` for green node/token IDs.
- Existing syntax accessors continue to use `.fromIndex()` and `.index()`;
  the shared formatter is additive and does not change ID values.

### P2.4: Diagnostic Builders

Status: closed.

Local diagnostic helpers repeat shape: locked writes, external-call writes,
overflow, generic arity, and field/method diagnostics.

Closure:

- `src/diagnostics/messages.zig` owns repeated message text for generic arity,
  expected/found type or region, integer fit, ADT named-payload field shape,
  locked writes, external-call write ordering, and missing `modifies` coverage.
- `src/sema/type_check.zig` keeps the source range, severity, and recovery
  decisions local, but routes the repeated wording through narrow emit helpers.
- Domain-specific context remains local as subject text, so messages such as
  `log field 'x' expects type ...` and `error 'E' argument 'code' expects type
  ...` keep their local meaning without duplicating the expected/found format.
- A duplicate-string sweep confirms the migrated repeated sema format strings
  now only exist in the shared diagnostics message builder.

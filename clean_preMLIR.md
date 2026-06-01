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

## Roadmap

### P1 Findings

- P1.1: Canonical Ora type source of truth, including fixed-bytes handling.
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

The real semantic type model should live in `src/types`, not `src/sema`.

Proposed direction:

- Move or mirror `sema.model.Type` into `src/types/semantic.zig`.
- Let `src/sema/model.zig` re-export it during migration.
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
- `src/comptime/value.zig` `ConstValue` and `src/comptime/pool.zig` preserve
  TypeIds, but the pool is not currently wired to a serializer or emitted cache.
- The only observed raw sentinel IDs are test-only: `42` in
  `src/comptime/heap.zig` tests and `100` in `src/comptime/value.zig` /
  `src/comptime/pool.zig` tests.

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

Long-term, either:

- builtin types do not need dedicated token kinds and parse as identifiers, or
- `BuiltinTypeSpec` includes enough token information to generate/check lexer
  keyword rows.

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
- duplicate fixed-bytes parsers in comptime and ABI
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

Current state:

- `src/ast/walk.zig` only provides `collectNamesInExpr`.
- `collectNamesInExpr` walks expressions only and explicitly stops at
  statement-bearing boundaries like `ComptimeExpr.body`.
- Sema currently owns several independent recursive traversals over the same
  `BodyId`, `StmtId`, `ExprId`, `PatternId`, and `SwitchPattern` shapes.
- The AST module already owns the node storage and typed IDs, so traversal shape
  belongs there, not inside every semantic analysis.
- Several semantic helpers do side scans during or near traversal, such as
  scanning all statements to find a pattern initializer or scanning contract
  members to resolve a field/callee. These should become explicit indexes or
  caches, not ad hoc loops inside analyses.

Repeated traversal targets:

- External-call validation:
  `validateStmtExternalCalls`, `validateExprExternalCalls`,
  `validateSwitchPatternExternalCalls`.
- Lock validation:
  `validateStmtLocks`, `validateExprLocks`, `validateSwitchPatternLocks`.
- Effect collection:
  `collectStmtEffects`, `collectExprEffects`, `collectSwitchPatternEffects`,
  `collectPatternTargetEffects`.
- Direct-callee collection:
  `collectStmtDirectCallees`, `collectExprDirectCallees`,
  `collectPatternDirectCallees`.
- Error-union collection and pattern-binding validation have the same smell.
- Name resolution also has a separate statement/expression traversal in
  `src/sema/resolve.zig`.

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
) anyerror!void;

pub fn walkStmt(
    comptime Visitor: type,
    visitor: *Visitor,
    file: *const AstFile,
    stmt_id: StmtId,
) anyerror!void;

pub fn walkExpr(
    comptime Visitor: type,
    visitor: *Visitor,
    file: *const AstFile,
    expr_id: ExprId,
) anyerror!void;

pub fn walkPattern(
    comptime Visitor: type,
    visitor: *Visitor,
    file: *const AstFile,
    pattern_id: PatternId,
) anyerror!void;
```

The walker solves traversal duplication. It should be paired with small,
explicit ID-indexed helpers for facts and reverse lookups:

```zig
pub fn IdMap(comptime Id: type, comptime T: type) type {
    return struct {
        values: []T,

        pub fn get(self: @This(), id: Id) T {
            return self.values[id.index()];
        }

        pub fn set(self: *@This(), id: Id, value: T) void {
            self.values[id.index()] = value;
        }
    };
}

pub const ExprFacts = packed struct {
    contains_result: bool = false,
    has_external_call: bool = false,
    has_call: bool = false,
    may_write_storage: bool = false,
};

pub const BodySummary = struct {
    direct_callees: []const ItemId,
    error_types: []const Type,
    expr_facts: IdMap(ExprId, ExprFacts),
};
```

The exact home for semantic summaries should stay in `src/sema`; the reusable
typed-ID storage helper can live beside AST infrastructure if it stays generic.

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
    enter_comptime_bodies: bool = false,
    enter_quantified_bodies: bool = false,
    walk_type_exprs: bool = false,
    walk_spec_clauses: bool = true,
};
```

Pragmatic indexes/caches to add as migration exposes the need:

- `PatternId -> initializer ExprId` so assignment/pattern analyses stop scanning
  all statements to recover initializer context.
- Callee resolution cache keyed by expression or call expression ID.
- Per-expression facts for common subtree predicates.
- Per-body summaries for direct callees, error types, and other read-only facts.
- Optional storage/effect summaries after P1.3 introduces reusable effect-slot
  sets.
- Semantic member/field/variant/method indexes belong to P1.6. P1.2 should
  consume them when useful, but it should not own their design.

Migration plan:

1. Extend `src/ast/walk.zig` with generic body/statement/expression/pattern
   walkers and tests that assert traversal order.
2. Add a tiny dense `IdMap(Id, T)` helper and first sema-local summary structs.
   Keep the helper generic; keep semantic facts out of `src/ast`; allocate
   durable arrays from `TypeCheckResult.arena`.
3. Add reverse lookup indexes that remove local scans, starting with
   `PatternId -> initializer ExprId`.
4. Reimplement `collectNamesInExpr` on top of the generic walker. This proves
   the framework supports the existing lightweight use case.
5. Migrate direct-callee collection using the walker plus callee resolution
   cache. It is mostly read-only and has limited branch-state semantics.
6. Migrate error-type collection and cheap expression facts next. These should
   expose gaps in switch-pattern traversal and remove repeated subtree queries.
7. Merge compatible read-only summaries per body where it is simple and
   measurable. Do not combine branch-sensitive validation state prematurely.
8. Migrate effect collection after P1.3 starts, because it should benefit from
   reusable effect-slot sets.
9. Migrate lock and external-call validation last. They have branch merge/freeze
   semantics and should use explicit visitor state rather than hidden walker
   behavior.
10. Consider name resolution only after sema traversals prove the API. Name
   resolution has scope/environment behavior and may need a scoped visitor
   extension.

Tests to add:

- Walk order for bodies, nested blocks, `if`, `while`, `for`, `switch`, and
  `try/catch`.
- Expression walking for tuple, array, struct literal, switch expression,
  external proxy, call, builtin, field, index, group, old, and quantified.
- Pattern walking for name, field, index, destructure, and switch patterns.
- Options controlling `ComptimeExpr.body` and quantified bodies.
- `collectNamesInExpr` behavior remains unchanged.

Acceptance criteria:

- New semantic analyses do not need to write their own complete AST traversal.
- Existing migrated passes become smaller and mostly contain semantic hook logic.
- Traversal choices at body boundaries are explicit in options.
- No pass scans all `file.statements` just to recover context for a `PatternId`.
- Caches owned by P1.2 are traversal/body facts, not global semantic lookup
  tables.
- Reused summaries avoid repeated full-body walks for read-only facts.
- The AST walker has no allocator dependency for normal traversal.
- Durable sema facts are arena-owned by the sema result that invalidates them.
- No semantic behavior changes after each migration step.

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

Non-goals:

- Do not change verifier semantics in the first migration.
- Do not merge verification facts with runtime `Effect` or comptime evaluation
  state.
- Do not move MLIR verification lowering before the pre-MLIR facts are stable.

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

### P2.2: Canonical Type/Value Model

`types.OraType`, `sema.Type`, comptime `TypeId`, sema `ConstValue`, and
comptime `ConstValue` encode overlapping facts.

Plan:

- Let P1.1 establish type metadata ownership.
- Then migrate semantic type ownership into `src/types`.
- Introduce interning/handles where comptime needs stable IDs. This is gated on
  the Comptime TypeId Stability Audit (see P1.1 Migration Plan). Do not
  renumber comptime IDs; preserve existing numeric assignments and generate only
  lookup/conversion code around them.
- Avoid parallel unions that need conversion glue.

### P2.3: Syntax ID Boilerplate (Optional / Low Priority)

This finding was overstated in the original draft and is now narrowed. AST IDs
already reuse `source.defineId` (`src/ast/ids.zig:6`), and AST builder/accessor
helpers are already centralized through `support.mixin`
(`src/ast/support.zig:18`). That part is the clean state we want — no work
needed there.

What genuinely remains: syntax green IDs still hand-roll the enum shape
(`src/syntax/green.zig:9`) instead of reusing the shared `defineId` generator.

Plan (optional, low priority):

- Migrate syntax green IDs onto the shared `defineId` generator so source,
  syntax, and AST IDs use one mechanism.
- If any truly repetitive accessor boilerplate remains after that, fold it into
  the existing mixin pattern.

This is the lowest-value item in the plan. Treat it as optional cleanup, not a
required phase.

### P2.4: Diagnostic Builders

Local diagnostic helpers repeat shape: locked writes, external-call writes,
overflow, generic arity, and field/method diagnostics.

Plan:

- Add reusable diagnostic builders for common categories.
- Keep domain-specific wording close to the domain logic.

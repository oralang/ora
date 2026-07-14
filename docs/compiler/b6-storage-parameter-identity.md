# B6 Storage Slice 2B Prerequisite - Parameter Key Identity

**Status**: Implemented - 2026-07-05.
**One-line scope**: make storage `PlaceKey.parameter` use the same compiler binding identity as `FreeVarRef`, without yet using parameter inequality to prove storage disjointness.

## 0. Ground Truth

- `FreeVarId` is the compiler-grade identity for free source variables in the
  obligation manifest: `{ file_id, pattern_id }` in
  `src/formal/obligation.zig` and `formal/Ora/Obligation/Manifest.lean`.
- The formal collector reads `ora.param_binding_ids` on `func.func` and stores
  them in `Collector.function_param_binding_ids`.
- General term projection already maps function entry block arguments to
  `FreeVarRef` with `freeVarIdForFunctionParam(index)`.
- Storage key projection currently maps the same entry block arguments to
  `PlaceKey.parameter index`. The entry-block pointer check prevents loop or
  region block arguments from becoming function parameters, but the payload is
  still only an ordinal.
- Effect paths parsed from attributes such as `balances[param#0]` also become
  `PlaceKey.parameter index` through `placeKeyFromSegment`.
- Lean treats `PlaceRef` as an opaque identity. It cannot infer that
  `.parameter 0` in a place key is the same runtime value as a free variable
  with `{ file_id := 13, pattern_id := 9 }`.
- Slice 2A correctly keeps parameter-vs-parameter disjointness unsupported.
  The generated truth table currently pins `parameter_vs_parameter = false`.

### 0.1 Query Scoping Audit

The ordinal collision is not reachable across functions in today's formal
query assembly.

The collector creates each normal query through `addQuery`, which gathers
assumptions with `assumptionIdsForOwner(owner)`. That helper includes only rows
where `ownerEqual(item.owner, owner)` is true. `appendQuery` then creates a
query with one target obligation id, and all current callers pass the same
owner used for the obligation row. Effect-frame queries use
`addQueryNoAssumptions`, so they carry no cross-owner assumptions. Base queries
have no obligation ids and no assumptions. The Lean emitter then defines
`emittedQuery_N` only from that query's `obligation_ids` and `assumption_ids`.

So two functions that both contain `balances[param#0]` can currently produce
identical ordinal-keyed `PlaceRef` values in the global manifest, but no Lean
query environment consumes both functions' rows together. That makes the bug
latent today, not a live cross-function proof channel.

This scoping is now part of the ground truth this design relies on: future
query assembly must not mix assumptions or obligations from different owners
without first making place-key identities globally stable. Add a regression
with this slice that asserts every non-empty query's obligation rows and
assumption rows have the same owner as the query.

## 1. Problem

The later proof-backed storage frame wants to use specifications like:

```ora
storage balances: map<u256, u256>;

pub fn f(user: u256, other: u256, value: u256)
    requires user != other
    ensures balances[other] == old(balances[other])
{
    balances[user] = value;
}
```

The `requires user != other` formula is over free variables identified by
`FreeVarId`. The read/write paths are currently keyed by parameter ordinals:
`balances[parameter 1]` and `balances[parameter 0]`.

Those two identity systems are not the same. If the compiler lets a proof over
free variables justify a frame decision over ordinal place keys, it recreates
the block-argument identity bug at a higher level: a value-level proof can be
about one identity while the storage place uses another.

## 2. Decision

Change `PlaceKey.parameter` from an ordinal to `FreeVarId`.

```zig
pub const PlaceKey = union(PlaceKeyTag) {
    parameter: FreeVarId,
    comptime_parameter: u32,
    comptime_range_parameter: u32,
    constant: []const u8,
    msg_sender,
    tx_origin,
    unknown,
};
```

Lean mirrors the same shape:

```lean
inductive PlaceKey where
  | parameter : FreeVarId -> PlaceKey
  | comptimeParameter : Nat -> PlaceKey
  | comptimeRangeParameter : Nat -> PlaceKey
  | constant : String -> PlaceKey
  | msgSender
  | txOrigin
  | unknown
```

This makes a parameter-keyed place and a free-variable formula speak about the
same compiler binding identity. It does not, by itself, prove any two parameter
keys are distinct.

## 3. Rejected Alternative: Side Table

An alternative is to keep `PlaceKey.parameter Nat` and add a manifest table:

```text
parameter_index -> FreeVarId
```

Reject that for now.

The table creates two identities that must be kept in sync everywhere:
`PlaceKey.parameter 0` and the table row for index `0`. Every consumer would
need to remember to consult the table before comparing or serializing keys.
That is exactly the kind of parallel identity channel that caused the previous
block-argument bug.

Putting `FreeVarId` directly in the key makes the identity local, serializable,
and visible to Lean, JSON dumps, generated snapshots, and reviewers.

## 4. Collector Rules

### 4.1 MLIR Value Keys

`placeKeyFromValue` keeps the existing entry-block check:

```text
if value is a block argument:
  arg = functionEntryBlockArgumentNumber(value) orelse unknown
  id = functionParamBindingId(arg) orelse unknown
  return .parameter(id)
```

Loop-carried block arguments, quantifier region arguments, and any non-entry
block argument still become `.unknown` and block projection.

The important tightening: place keys must not use the synthetic fallback id.
If `ora.param_binding_ids` is missing or too short, a parameter place key is
`.unknown`, and the collector records the existing fail-closed diagnostic path.
Synthetic ids can remain for legacy term projection if required, but they must
not enter storage place identity.

### 4.2 Effect Path Keys

`placeKeyFromSegment("param#N")` must also resolve through the current
function's `ora.param_binding_ids`:

```text
param#N -> .parameter(function_param_binding_ids[N])
```

If the parser sees `param#N` outside a function scope, with no binding-id
attribute, or with `N` out of range, it returns `.unknown` or records a blocking
diagnostic. It must not silently fall back to ordinal identity.

This keeps walker-produced keys and effect-attribute keys byte-identical after
serialization.

### 4.3 Names Are Display Only

Parameter names never participate in identity. Renaming `owner` to `z` must not
change whether a place key matches the corresponding free variable. The binding
id is the identity; the source name is diagnostic text.

## 5. Lean Semantics

The Lean change is structural first:

- `PlaceKey.parameter` stores `FreeVarId`;
- `BEq` and `DecidableEq` compare the full id;
- `placeKeysDefinitelyDistinct` still returns `false` for parameter-vs-parameter;
- `placeDefinitelyDisjoint` behavior does not expand in this prerequisite slice.

That last point is deliberate. The identity tie makes later proof-backed
disjointness possible, but it does not authorize a new frame decision.

The storage disjointness snapshot must update to encode parameter ids, for
example:

```lean
("parameter", "file:13:pattern:9")
```

or another canonical data-only spelling with the same information. The decoder
must parse the same `FreeVarId` fields Lean uses for `FreeVarRef`.

## 5.1 Serialization and Compatibility Surfaces

Changing the `PlaceKey.parameter` payload is a breaking manifest-shape change
for every surface that serializes place identities.

Surfaces that must change deliberately:

- `obligation_dump.zig` JSON-lines output: parameter keys currently serialize
  as `{"tag":"parameter","index":N}`. They should become an object carrying
  the `FreeVarId`, for example
  `{"tag":"parameter","id":{"file_id":F,"pattern_id":P}}`. This is a
  versioned external-ish surface, so bump `obligation_dump_schema_version` in
  the same implementation slice.
- `obligation_to_lean.zig` generated obligation modules: `.parameter N`
  becomes `.parameter { file_id := F, pattern_id := P }`. This is not a JSON
  schema, but it is the userland proof surface; existing proof files that name
  concrete places may need source edits.
- `emit_storage_disjointness_snapshot.zig` and
  `formal/Ora/Generated/StorageDisjointnessSnapshot.lean`: the snapshot
  encoding must carry the full id and `check-formal-sync` must regenerate it.
- `StorageDisjointnessSync.lean`: the decoder must parse the same canonical
  id representation used in the generated snapshot.
- Tests and golden strings that embed place identities, including the formal
  Lean emitter tests, storage path fixtures, and any JSON dump tests.

Surfaces that should not change schema just because of this slice:

- proof manifests (`proofs.json`) still target query ids, obligation ids, and
  assumption ids;
- proof certificates may get different content hashes after generated Lean
  text changes, but the certificate schema does not change;
- the SMT erasure decision path must not learn any new frame rule from this
  representation change.

## 6. Later Slice: Proof-Backed Key Disjointness

Only after this identity tie lands should the compiler add proof-backed
parameter disjointness.

That later slice needs a first-class evidence path. It must not simply let Zig
scan assumptions and return `true` from `placeDefinitelyDisjoint`.

The required shape is:

1. Extract supported key inequality evidence from manifest terms whose operands
   are free variables with `FreeVarId`.
2. Connect that evidence to `PlaceKey.parameter FreeVarId`.
3. Keep anti-vacuity: the query assumptions must still pass the existing
   satisfiability gate before any proof can unblock artifacts.
4. Make the user-visible recipe show the frame fact being used.

Until that exists, `requires user != other` remains a Z3-only path for storage
frame disjointness.

## 7. Non-Goals

- Do not make parameter-vs-parameter disjoint in this slice.
- Do not consume `requires user != other`.
- Do not add value injectivity for `Env.placeValue`.
- Do not relate distinct place identities to distinct stored values.
- Do not support address-typed free-variable inequalities unless the term
  fragment already denotes those values.
- Do not change post-state or versioned storage semantics.
- Do not infer anything from parameter names.

## 8. Acceptance Tests

1. Manifest shape:
   - `PlaceKey.parameter` serializes a `FreeVarId`, not an ordinal;
   - Lean generated obligations render `.parameter { file_id := ..., pattern_id := ... }`.

2. Term/key identity tie:
   - a source fixture with `requires user != other` and `balances[user]` emits
     the same `FreeVarId` for the `FreeVarRef user` term and the
     `PlaceKey.parameter` inside the `balances[user]` place.

3. Effect parser parity:
   - walker-produced `balances[user]` and effect-path parsed
     `balances[param#N]` produce byte-identical `PlaceRef` values after dump.

4. Missing binding ids fail closed:
   - hand-written MLIR with `param#0` but no `ora.param_binding_ids` does not
     produce `.parameter synthetic`;
   - the query remains outside the Lean fragment or records a blocking
     diagnostic.

5. Loop/block-argument regression:
   - a loop-carried block argument used as a map key still becomes `.unknown`;
   - the function parameter in the same function still becomes
     `.parameter FreeVarId`.

6. Nested function scope:
   - nested functions save and restore parameter binding context;
   - an inner `param#0` cannot reuse the outer function's `FreeVarId`.

7. Disjointness unchanged:
   - the storage disjointness truth table still says parameter-vs-parameter is
     not disjoint;
   - the parameter-inequality source fixture remains recipe-unavailable.

8. Sync gate:
   - generated storage disjointness snapshot updates;
   - `check-formal-sync` proves Zig/Lean relation agreement with the new key
     encoding.

9. Query owner scoping:
   - a collector-level regression asserts every query's obligation ids and
     assumption ids, when present, resolve to rows with the same owner as the
     query;
   - a two-function fixture where both functions use `balances[param#0]` does
     not produce a Lean-supported query that combines rows from both owners.

10. Serialization versioning:
   - obligation JSONL parameter keys carry `FreeVarId`;
   - `obligation_dump_schema_version` is bumped for that breaking shape change;
   - proof manifest and proof certificate schema versions remain unchanged.

## 9. Design Challenge Verdict

APPROVE the identity-tie direction with one hard boundary: this slice must be
representation-only for parameter keys.

The correct next move is to replace ordinal parameter place keys with
`FreeVarId` everywhere they cross the formal trust boundary. That directly
solves the mismatch that blocks proof-backed key disjointness, and it reuses
the existing compiler binding identity rather than inventing a custom aliasing
scheme.

CHALLENGE any implementation that also changes `parameter_vs_parameter` to
disjoint or lets source `requires` clauses drive projection in the same patch.
That would combine identity plumbing with a new proof rule, making review much
harder and risking a hidden frame axiom. The proof-backed rule is the next
slice, not this prerequisite.

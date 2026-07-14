# B6 Storage Slice 2 - Path Disjointness

**Status**: REFINE-approved design - 2026-07-05. No code yet.
**One-line scope**: define the conservative path-disjointness relation needed before Lean projection may treat a read under a written storage root as stable.

## 0. Ground Truth

- Storage slice 1 is closed. The formal collector can project scalar and keyed
  storage reads to `placeRead` only when the function has complete write
  metadata, no external call, and the entire root is not written.
- The closed slice deliberately uses `storageWriteSlotsContain(write_slots,
  root)`, so a write to `balances[user]` blocks every read under `balances`.
  This is conservative and sound.
- `PlaceRef { root, region, fields, keys }` is already the manifest identity
  for storage places. `PlaceKey` currently has parameter ordinals, comptime
  parameter ordinals, constants, `msg_sender`, `tx_origin`, and `unknown`.
- `placeKeyFromValue` is now hardened for Lean identity use: `.parameter N`
  is emitted only for MLIR block arguments owned by the current function entry
  block; loop and region block arguments become `.unknown` and fail closed.
- `ora.evm.caller` and `ora.evm.origin` now project to `.msg_sender` and
  `.tx_origin`, type-gated on address results.
- Lean treats `PlaceRef` as an opaque key. `Env.placeValue` is a total function
  over `.stable place` and `.entry place`; Lean does not know map arrays,
  keccak slots, storage layout, or write sets.
- Lean currently has `placeListDisjoint = actual.all (!declared.contains)`.
  That is exact-identity negation, not a path-disjointness relation.
- The SMT lane already proves path-precise map frames in fixtures such as
  `ora-example/smt/modifies/pass_internal_map_key_frame.ora`, using source
  assumptions like `requires(user != other)`. That does not automatically make
  the same claim safe in the Lean lane.
- `PlaceKey.parameter` currently stores only an ordinal. User assumptions over
  parameters use free-variable binding ids. Until those identities are tied
  together in the manifest, Lean cannot safely consume `requires user != other`
  as proof that `.parameter 0` and `.parameter 1` are distinct.
- Storage root separation rests on the compiler's storage layout model. The
  current anchor is `assignGlobalSlots` in
  `src/mlir/ora/lowering/OraToSIR/OraToSIR.cpp`, the strict metadata checks in
  `src/mlir/ora/lowering/OraToSIR/patterns/Storage.cpp`, and the
  `compiler storage layout manifest matches SIR slot usage` regression in
  `src/compiler.test.oratosir.zig`.

## 1. Problem

Root-conservative projection blocks useful true frames:

```ora
storage balances: map<u256, u256>;

pub fn set_one()
    ensures balances[2] == old(balances[2])
{
    balances[1] = 7;
}
```

The read path `balances[2]` is statically disjoint from the write path
`balances[1]`. Slice 1 still blocks it because both paths share the root
`balances`.

The tempting broader case is:

```ora
pub fn f(user: address, other: address, value: u256)
    requires user != other
    ensures balances[other] == old(balances[other])
{
    balances[user] = value;
}
```

This is valid in the SMT lane, but it must not be smuggled into Lean by simply
projecting both sides to the same stable place. The disjointness proof depends
on runtime parameter values and on a faithful connection between place keys and
function-parameter identities.

## 2. Decision

Implement path-disjointness in two stages.

### 2.1 Slice 2A: Static Path Disjointness

Support only disjointness that is true from manifest place identity alone.

This allows same-root projection when every write under that root is statically
disjoint from the read, for example:

```ora
ensures buckets[2] == old(buckets[2])
{
    buckets[1] = value;
}
```

It does not use user assumptions, SMT results, names, or inferred inequality
facts. If the relation cannot prove disjointness from the `PlaceRef` values
alone, it returns "may alias" and projection remains unavailable.

### 2.2 Later Slice: Proof-Backed Key Disjointness

Support `requires user != other` only after two prerequisites exist:

1. `PlaceKey.parameter` is made compiler-grade stable, either by carrying the
   corresponding `FreeVarId` or by a manifest-level parameter table that Lean
   can check against free-variable bindings.
2. The query carries first-class key-inequality evidence derived from supported
   assumptions, with anti-vacuity still enforced by the existing artifact gate.

Until then, parameter-vs-parameter disjointness remains unsupported even when
the source has `requires user != other`.

## 3. One Relation

There must be one shared relation, not separate ad hoc checks:

```text
placeDefinitelyDisjoint(read: PlaceRef, write: PlaceRef) -> bool
```

Consumers:

- Lean projection eligibility for stable `placeRead` under written roots.
- `old(map_read)` collapse eligibility.
- Effect-frame `read_preserved_by_frame` row construction.
- Lean `placeListDisjoint` semantics.
- User-facing diagnostics and coverage reasons.

The relation should be conservative:

- `true` means the two places cannot alias.
- `false` means may alias, unknown, or unsupported.

Do not introduce a relation where one caller treats "unknown" as disjoint while
another treats it as blocking.

## 4. Static Relation Rules

For this slice, `placeDefinitelyDisjoint(read, write)` returns `true` only for
these cases.

### 4.1 Region and Root

- Different regions are disjoint only when both regions are concrete. If either
  side has `.none`, the relation returns may-alias.
- Same region with different roots is disjoint.
- Same storage root with a whole-root write is not disjoint from any path under
  that root.
- Any `$computed_storage` root or `.unknown` key blocks disjointness.

### 4.1.1 Storage Model Assumption

Different-root disjointness is the one model assumption in this relation. It
trusts two facts outside the Boolean checker itself:

1. The compiler assigns non-overlapping base slots to storage roots. Today that
   guarantee is enforced by `assignGlobalSlots`, strict `ora.slot_index`
   metadata validation in `ConvertGlobalOp`, and the SIR slot-usage regression
   named in section 0.
2. Keyed storage follows the standard EVM model: a map or array domain is
   derived from its base slot with domain-separated keccak-style addressing, so
   two distinct base roots do not alias through their keyed children.

Everything else in `placeDefinitelyDisjoint` is decidable from manifest
identity. This paragraph names the axiom-like storage physics the relation
trusts; it must not silently grow to include value-level claims.

### 4.2 Keys

Compare keys from left to right.

- If all compared keys are equal and one key list is a prefix of the other,
  the places may alias. A write to `allowances[owner]` may affect
  `allowances[owner][spender]`.
- If keys differ at position `i`, the paths are disjoint only when
  `placeKeysDefinitelyDistinct(lhs[i], rhs[i])` is true.
- `placeKeysDefinitelyDistinct` is true only for normalized, typed constants
  that are proven unequal by the same canonical key parser used for compiler
  effect paths. In the current collector this is `placeKeyFromSegment` in
  `src/formal/obligation_from_mlir.zig`; if implementation moves it, this note
  should move with the shared parser name.
- Raw string inequality is not enough. `"1"` and `"01"` must not become
  disjoint just because their bytes differ.
- `.msg_sender` vs `.tx_origin` is not disjoint. They may be equal in a direct
  transaction.
- `.parameter i` vs `.parameter j` is not disjoint in slice 2A.
- `.parameter` vs `.constant`, `.msg_sender` vs `.constant`,
  `.tx_origin` vs `.constant`, and all comptime-key cases are not disjoint in
  slice 2A unless a later canonical value proof is added.

### 4.3 Fields

Fields are not the first target of this slice. Field-name inequality is not
slot disjointness: packed storage can place two fields in the same physical
word.

The safe implementation choice is:

- preserve existing exact-identity behavior for field-only effect-frame rows
  until storage-layout-backed field disjointness is designed;
- do not use different field names to justify Lean storage projection for map
  reads in this slice;
- do not add any theorem that distinct field identities imply distinct storage
  values.

If field disjointness is added later, it must be tied to the storage-layout
work, not inferred from field-name strings alone.

## 5. Projection Rule

Replace root-only projection with:

```text
canProjectStablePlace(read_place):
  complete write summary is present
  no external call in the function
  every write slot is definitely disjoint from read_place
```

Important details:

- Missing `ora.write_slots` still blocks.
- Missing `ora.write_slots_complete = true` still blocks.
- External calls still block.
- A write slot with the same root and no keys blocks every keyed read under
  that root.
- A write slot with `.unknown` blocks every read under the same root.
- A written same path remains unsupported for bare post-state reads. This slice
  is not versioned/post-state storage.

`old(map_read)` uses the same rule:

- if the read path is stable under all writes, collapse to stable `placeRead`;
- otherwise stay unsupported.

Do not expose `old(written_map_path)` as entry-only in this slice. That belongs
with versioned/post-state storage.

## 6. Lean Semantics

Lean should learn the same static relation used by Zig:

```lean
def placeKeysDefinitelyDistinct : PlaceKey -> PlaceKey -> Bool
def placeDefinitelyDisjoint : PlaceRef -> PlaceRef -> Bool
def placeListDisjoint (writes reads : List PlaceRef) : Bool :=
  reads.all (fun read => writes.all (fun write =>
    placeDefinitelyDisjoint read write))
```

This is a Boolean checker over manifest identities, not a storage-value theorem.

Theorems should be small and shape-level:

- different roots are disjoint;
- unequal canonical constants at the first differing key are disjoint;
- same key prefix is not disjoint;
- unknown keys are not disjoint;
- parameter-vs-parameter is not disjoint in slice 2A.

The Zig/Lean agreement test must not be two hand-written truth tables. Emit one
shared fixture from Zig - a list of `PlaceRef` pairs plus expected verdicts -
into a generated Lean data file, consume the same fixture from the Zig unit
test, and prove the Lean relation agrees by kernel `decide`. This is the
CompilerSnapshot pattern applied to the disjointness relation, and should be
the template for the later Zig/Lean totality and identity lockstep guard.

Do not add `placeValue` injectivity or a theorem saying disjoint places have
different values. Disjointness only justifies a frame/collapse decision already
made by the compiler; it is not a value inequality fact.

## 7. Zig Shape

The implementation should centralize the relation near the manifest data model.

Preferred shape:

```zig
fn placeDefinitelyDisjoint(lhs: obligation.PlaceRef, rhs: obligation.PlaceRef) bool
fn placeKeysDefinitelyDistinct(lhs: obligation.PlaceKey, rhs: obligation.PlaceKey) bool
```

The collector then uses the relation for:

- `canProjectStablePlace(place)`;
- `readsDisjointFromWrites`;
- any future recipe diagnostic reason for path aliasing.

Avoid adding a second path parser. The relation consumes already-decoded
`PlaceRef` values. If constants need normalization, use the same canonical
logic that produced the `PlaceKey.constant` value, currently
`placeKeyFromSegment`, or return false.

Add an implementation test proving byte identity between walker-produced
constant keys and effect-parser-produced constant keys for the same source key.
A mismatch is fail-safe because it withholds disjointness, but it would silently
erase the fragment coverage this slice is meant to add.

## 8. Explicit Non-Goals

- Proving `parameter i != parameter j` from `requires`.
- Connecting `PlaceKey.parameter` to `FreeVarId`.
- Post-state reads of written map paths.
- `old(written_map_path)` entry-only support.
- Keccak or physical slot computation in Lean.
- General map array semantics in Lean.
- Field-layout disjointness.
- Computed-storage range disjointness.
- Relaxing the external-call gate.

## 9. Exploit Shapes To Block

### 9.1 Parameter Inequality Smuggling

```ora
requires user != other
ensures balances[other] == old(balances[other])
{
    balances[user] = value;
}
```

Unsupported in slice 2A. This remains Z3-only until proof-backed key
disjointness exists.

### 9.2 Parameter Ordinal Collision

Loop-carried block arguments must never become parameter keys:

```ora
while (i < n)
    invariant balances[i] <= cap
{
    i = i + 1;
}
```

This is already pinned by `3cae3cb2`, and this slice must keep that behavior.

### 9.3 Raw Constant String Mismatch

Hand-written MLIR must not make these disjoint unless the canonical parser
proves they are different values:

```text
buckets[1]
buckets[01]
```

### 9.4 Prefix Alias

```ora
allowances[owner] = replacement;
ensures allowances[owner][spender] == old(allowances[owner][spender])
```

Unsupported. Parent writes may affect child paths.

### 9.5 Environment Key Assumption

```ora
ensures balances[tx.origin] == old(balances[tx.origin])
{
    balances[msg.sender] = value;
}
```

Unsupported. `msg.sender` and `tx.origin` may be equal.

## 10. Acceptance Tests

1. Relation truth table in Zig:
   - different roots -> disjoint;
   - same root exact path -> not disjoint;
   - whole-root write vs keyed read -> not disjoint;
   - normalized unequal constants -> disjoint;
   - raw noncanonical constants -> not disjoint unless normalized safely;
   - parameter-vs-parameter -> not disjoint;
   - msg.sender-vs-tx.origin -> not disjoint;
   - unknown key -> not disjoint;
   - nested same-prefix paths -> not disjoint.

2. Matching Lean truth table:
   - the cases come from one generated Zig fixture, not hand-written twins;
   - the same fixture feeds the Zig unit test and generated Lean data;
   - Lean decides the relation by kernel `decide`;
   - this pins Zig/Lean relation agreement until the dual-emission bridge exists.

3. Parser agreement:
   - walker-produced and effect-parser-produced constant keys are byte-identical
     for the same source key;
   - `1` and any noncanonical spelling such as `01` cannot become disjoint by
     raw byte inequality.

4. Source-level static positive:
   - write `buckets[1]`;
   - ensure `buckets[2] == old(buckets[2])`;
   - collector projects the read as a stable keyed `placeRead`;
   - query is Lean-supported.

5. Source-level parameter negative:
   - write `balances[user]`;
   - ensure `balances[other] == old(balances[other])`;
   - even with `requires user != other`, the Lean recipe is unavailable in
     slice 2A, with a path-aliasing reason.

6. Prefix negative:
   - write parent or shorter nested path;
   - read longer child path;
   - unsupported.

7. Environment negative:
   - write `balances[std.msg.sender()]`;
   - read `balances[std.tx.origin()]`;
   - unsupported.

8. Written same path negative:
   - write `balances[user]`;
   - ensure `balances[user] == old(balances[user])`;
   - unsupported, proving this slice did not accidentally add versioned
     post-state reads.

9. Region precision:
   - concrete different regions are disjoint;
   - `.none` on either side returns may-alias.

10. Effect-frame parity:
   - `read_preserved_by_frame` rows are emitted only when the same relation
     says the read is disjoint from every write;
   - no exact-identity shortcut remains for map paths.

11. E2E Lean recipe:
   - static positive forced to Z3 UNKNOWN;
   - valid Lean proof unblocks artifact emission;
   - bytecode remains guard-preserving;
   - sorry proof fails.

12. Coverage counter:
   - the static positive moves from `origin_value` to `Term`;
   - parameter negative remains outside the fragment with an explicit reason.

## 11. Design Challenge Verdict

REFINE before implementation.

The direction is correct, but the implementation must not jump directly from
root-conservative projection to parameter-inequality frames. The current
manifest does not carry enough stable identity in `PlaceKey.parameter` for Lean
to check that a source assumption over free variables proves two place keys are
different. Treating parameter ordinals as enough would recreate the identity
class of the block-argument bug at a higher level.

Approved implementation shape after this refinement:

1. Add one conservative static disjointness relation over decoded `PlaceRef`.
2. Mirror that relation in Lean with a small truth-table sync test.
3. Use it for projection, `old()` collapse, and effect-frame row construction.
4. Keep parameter inequality, field-layout disjointness, and versioned
   post-state storage explicitly out of scope.
5. Add source-level positives and negatives that prove the boundary is visible
   to users.

This slice is valuable even though it is narrow: it removes the root-only false
negative for statically distinct constant paths while keeping the runtime-value
alias cases honest and fail-closed.

## 12. Implementation review (2026-07-05) — APPROVED after one round

Slice verified: relation core correct on all probed edges (both-concrete regions, $computed_storage poisoning, field-inequality never justifies disjointness, prefix-may-alias, "1"/"01" by value, unparseable-constant fail-safe); dangerous write_slots migration handled by DELETING the root-containment helpers (compile-enforced consumer migration, 5 consumers on one relation); Z3 call-summary normalizes paths to roots (effectSlotPathRoot) — no under-havoc; generated truth-table sync = 6th snapshot in check-formal-sync, kernel-decide over all rows — first working Zig↔Lean lockstep-guard instance. ONE blocker found+fixed: parseDecimalU256 accepted Zig-isms (underscores, '+') Lean rejects — strict digit pre-scan added + underscore_constant_blocks fixture row pins parser agreement in the gate forever. Final: 18/18 obligation unit, sync gate 4/4 stages OK, 355 declarations audited. All four design amendments landed incl. the named model-axiom paragraph (assignGlobalSlots / ConvertGlobalOp / SIR slot regression). Outstanding process item: erc20.ora dirty again (flag 8) with a manual restore-dance during validation — commit-or-revert required before next slice.

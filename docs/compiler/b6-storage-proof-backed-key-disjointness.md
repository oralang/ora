# B6 Storage Slice 2C - Proof-Backed Key Disjointness

**Status**: Implemented - 2026-07-05.
**One-line scope**: let supported `requires lhs != rhs` evidence discharge storage frame obligations for parameter-keyed places, without turning user assumptions into hidden compiler facts.

This is the "proof-backed key disjointness" slice named by
`b6-storage-parameter-identity.md`.

## 0. Ground Truth

- Static path disjointness is already implemented and gate-enforced through the
  generated storage-disjointness snapshot. `parameter` vs `parameter` remains
  may-alias in that static relation.
- `PlaceKey.parameter` now carries the same `FreeVarId` used by `FreeVarRef`.
  The identity mismatch that blocked proof-backed parameter disjointness is
  closed.
- Query assembly is owner-scoped: normal queries collect assumptions only from
  the same function owner, and the owner-scoping invariant is now pinned by a
  regression.
- Lean query propositions already include the anti-vacuity shape:
  `assumptionsSatisfiable manifest assumptions` and then
  `forall env, assumptions -> obligation`.
- Current effect-frame rows are assumption-free. The collector emits
  `read_preserved_by_frame` through `addQueryNoAssumptions`, and Lean denotes it
  as the pure Boolean `placeListDisjoint declared actual = true`.
- Current stable storage projection and `old(map_read)` collapse are compiler
  side conditions. They rely on static disjointness only; they do not carry
  evidence into Lean.
- The Lean value fragment can denote direct disequality over supported carrier
  values through `.ne` and `Value.eqProp?`. Unsupported syntax returns `none`
  and therefore fails closed.
- This slice introduces a stronger instance of the storage-physics axiom family
  than static different-root disjointness: within one map domain, unequal
  concrete key values address disjoint storage slots.

## 1. Problem

Static path disjointness cannot prove this common frame:

```ora
storage balances: map<u256, u256>;

pub fn f(user: u256, other: u256, value: u256)
    requires user != other
{
    let observed = balances[other];
    balances[user] = value;
    // The read of balances[other] is preserved only under user != other.
}
```

The dangerous shortcut is to let Zig scan `requires user != other` and return
`true` from `placeDefinitelyDisjoint(balances[other], balances[user])`.

That would hide a semantic dependency in the compiler:

- inconsistent assumptions could prove a frame unless anti-vacuity is preserved;
- a future refactor could use evidence in one caller but not another;
- Lean would see only a Boolean fact, not the assumption that made it true;
- the old block-argument identity bug would reappear as an assumption-to-place
  identity bug.

The evidence must therefore be first-class manifest data and must be discharged
by the same query theorem shape as every other Lean-backed obligation.

## 1.1 Storage Model Assumption

Evidence-backed key disjointness trusts one storage-physics fact that Lean does
not prove from bit-vector disequality alone:

```text
within the same keyed storage domain, different key values derive different
physical storage slots.
```

For a map root `balances`, this is the step from:

```text
user != other
```

to:

```text
balances[user] and balances[other] cannot alias.
```

Lean checks the value-level disequality and the identity connection between
`FreeVarId` and `PlaceKey.parameter`. The last step relies on the compiler's
storage layout model: keyed storage paths are derived from the root's base slot
with the standard domain-separated EVM map-slot derivation, so two unequal keys
inside the same root/key-prefix domain do not collide.

This is deliberately narrower than a general storage injectivity theorem. It
does not say distinct `PlaceRef` identities imply distinct values, does not
apply across unknown/computed storage, and does not cover fields or prefixes.
The implementation must point this assumption at the same slot-derivation code
and layout checks named by the static path-disjointness note.

## 2. Decision

Add a separate evidence-backed frame lane. Do not change the meaning of the
static relation.

Static relation stays:

```text
placeDefinitelyDisjoint(read, write) -> Bool
```

and remains assumption-free. In particular:

```text
parameter(user) vs parameter(other) == false
```

The new lane is a distinct effect-frame relation, tentatively:

```zig
read_preserved_by_key_evidence
```

It carries the read/write places plus a list of evidence records. The collector
may emit this row only when every read/write pair is either statically disjoint
or covered by supported evidence.

Evidence-backed rows must use `addQuery`, not `addQueryNoAssumptions`, so the
owner's `requires` rows are in the query premise and the existing
`assumptionsSatisfiable` anti-vacuity guard applies.

## 3. Non-Goals

- Do not make `placeDefinitelyDisjoint` consult assumptions.
- Do not change the static disjointness truth table.
- Do not use evidence to erase runtime guards or alter codegen.
- Do not use evidence to project logical storage reads or collapse
  `old(map_read)` in this slice.
- Do not add `Env.placeValue` injectivity or any theorem that distinct place
  identities imply distinct stored values.
- Do not support path-sensitive `assume(...)` evidence in this slice.
- Do not support address/u160 evidence until the Lean value fragment denotes
  those values explicitly.

The important limitation: a source-level postcondition like
`ensures balances[other] == old(balances[other])` remains outside the Lean
fragment when it needs assumption-backed frame equality. This slice proves
frame obligations; formula-level conditional post-state equality is a later
design.

## 4. Manifest Shape

Add a first-class evidence record to the manifest model.

Sketch:

```zig
pub const KeyDisjointEvidenceKind = enum(u8) {
    free_var_disequality,
};

pub const KeyDisjointEvidence = struct {
    kind: KeyDisjointEvidenceKind,
    assumption_id: Id,
    lhs: FreeVarId,
    rhs: FreeVarId,
    read: PlaceRef,
    write: PlaceRef,
    key_index: u32,
};

pub const EffectFrameGoal = struct {
    relation: EffectFrameRelation,
    declared: []const PlaceRef = &.{},
    actual: []const PlaceRef = &.{},
    evidence: []const KeyDisjointEvidence = &.{},
};
```

Lean mirrors the same data in `formal/Ora/Obligation/Manifest.lean`.

The existing relations keep `evidence = []`. Only
`read_preserved_by_key_evidence` may carry evidence.

The JSONL obligation dump format changes because effect-frame rows gain a new
field and parameter evidence becomes part of the public manifest. Bump the dump
schema version with the implementation.

## 5. Supported Evidence

First slice supports exactly this evidence shape:

```ora
requires lhs != rhs
```

where:

- the assumption row kind is `requires`;
- the assumption belongs to the same owner as the frame query;
- the formula is a direct binary `.ne`;
- both operands are free variables;
- both free variables have denotable 256-bit carrier types;
- the two `FreeVarId` values match the first differing
  `PlaceKey.parameter` values in the read/write path;
- matching is symmetric: `requires user != other` and
  `requires other != user` both cover the same pair;
- the read and write have the same concrete region, root, fields, and key-list
  length;
- every key before `key_index` is exactly equal;
- no prefix relation is involved.

One covered differing key is enough. For example:

```text
allowances[user][spender] vs allowances[other][spender]
```

can be covered by `requires user != other`.

These stay unsupported:

- `requires !(user == other)`;
- `requires user < other`;
- `assume(user != other)`;
- `requires user != 42`;
- `requires msg.sender != tx.origin`;
- address/u160 variables until the value fragment supports them;
- different field names;
- same-key paths;
- prefix paths such as `allowances[user]` vs `allowances[user][spender]`.

Unsupported means no evidence row is emitted and the Lean recipe reports the
reason. It must not silently fall back to a weaker proof.

`requires user != 42` is excluded by slice scope, not because it would be
unsound in principle. Parameter-vs-constant evidence can be added later under
the same storage model assumption once the manifest and Lean support a shared
constant-key evidence shape.

## 6. Lean Semantics

Change effect-frame denotation so it can see the manifest and environment:

```lean
def effectFrameGoalDenotes? (manifest : Manifest) (env : Env)
    (goal : EffectFrameGoal) : Option Prop
```

For existing relations, this returns the current static meanings:

```lean
writeCoveredByModifies     -> placeListCovers declared actual = true
readPreservedByFrame       -> placeListDisjoint declared actual = true
```

For the new relation:

```lean
readPreservedByKeyEvidence ->
  every actual read is separated from every declared write either by
  placeDefinitelyDisjoint = true or by a denoted KeyDisjointEvidence
```

An evidence record denotes only when Lean can check all of these:

- the referenced assumption row exists;
- the row kind is `requires`;
- the row formula is denotable in the current `env`;
- the row formula has the supported `free(lhs) != free(rhs)` shape;
- the formula's `FreeVarId` values match the evidence record in either order;
- the evidence record's places and `key_index` match the first differing
  parameter keys of the read/write pair;
- the denoted formula proposition holds.

If any lookup, shape check, type check, or denotation fails, the evidence
returns `none`, and `obligationDenotesInEnv` turns that into `False`.

This is the load-bearing property: Lean sees the same assumption formula that
the user wrote. Zig is allowed to identify candidate evidence, but the proof
does not trust Zig's conclusion that the evidence is true.

## 7. Collector Rules

Keep static reads exactly as they are.

Add a second path for frame rows:

1. Build the normal static `read_preserved_by_frame` row for reads that are
   statically disjoint from all writes.
2. For remaining reads, try to cover every non-static read/write pair with a
   supported key-disequality evidence record.
3. If all pairs are covered, emit a
   `read_preserved_by_key_evidence` effect-frame row.
4. Use `addQuery`, not `addQueryNoAssumptions`, for that row.
5. If any pair is not statically disjoint and has no supported evidence, emit
   no evidence-backed row for that read.

Do not reuse the evidence result in:

- `canProjectStablePlace`;
- `oldTerm`;
- static `placeIsDefinitelyDisjointFromAll`;
- `placeDefinitelyDisjoint`;
- runtime guard erasure.

Those consumers remain static-only in this slice.

## 8. Supportability And UX

`obligation_to_lean` must support the new relation only when every evidence
record is itself denotable.

Add precise unsupported reasons:

- `missing_key_disjoint_evidence`;
- `unsupported_key_disjoint_evidence_formula`;
- `unsupported_key_disjoint_evidence_kind`;
- `key_disjoint_evidence_type_unsupported`;
- `key_disjoint_evidence_owner_mismatch`;
- `key_disjoint_evidence_path_mismatch`.

The proof recipe for an UNKNOWN evidence-backed frame should show:

- the theorem name, `emittedQuery_N`;
- the read place and write place;
- the assumption row id and source location used as evidence;
- the exact key index being separated;
- the reason if evidence was unavailable.

This is not just UX. It is how reviewers see that a proof is using
`requires user != other` as a premise rather than a compiler-invented frame
fact.

## 9. Anti-Vacuity

Contradictory assumptions must not unlock evidence-backed frames.

Because the new row is queried with `addQuery`, Lean proofs target:

```lean
assumptionsSatisfiable emittedManifest assumptions ∧
  ∀ env, assumptionsDenoteInEnv emittedManifest env assumptions →
    obligationDenotesInEnv emittedManifest env row
```

So `requires user != other` plus `requires user == other` cannot discharge the
row unless the proof can also produce a satisfying environment. It cannot.

The artifact gate must keep rejecting vacuous, vacuity-unknown, degraded, or
caveated targets before accepting a proof row. This slice must not introduce a
second proof-acceptance path.

## 10. Acceptance Tests

1. Static truth table unchanged:
   `parameter(file:0:pattern:0)` vs `parameter(file:0:pattern:1)` remains
   not statically disjoint.
2. Positive source fixture:
   a function reads `balances[other]`, writes `balances[user]`, and has
   `requires user != other`; the collector emits
   `read_preserved_by_key_evidence` with the correct assumption id and
   `key_index`.
3. End-to-end proof fixture:
   build without `--lean-proofs` emits the unchanged bytecode; with
   `--lean-proofs`, the structural frame target fails without a proof, a valid
   Lean proof succeeds, emitted bytecode is byte-identical to the no-flag
   reference, `.lean.proof.json` is written, and a `sorry` proof fails the
   axiom audit.
4. No-evidence negative:
   the same read/write paths without `requires user != other` produce no
   evidence-backed Lean recipe.
5. Contradictory-assumptions negative:
   `requires user != other` plus `requires user == other` cannot be accepted as
   a proof target because anti-vacuity fails.
6. Same-key negative:
   `requires user != other` cannot preserve `balances[user]` across a write to
   `balances[user]`.
7. Prefix negative:
   no evidence can prove `allowances[user]` disjoint from
   `allowances[user][spender]`.
8. Multi-key positive:
   `allowances[user][spender]` vs `allowances[other][spender]` is covered by
   `requires user != other`.
9. Symmetry positive:
   `requires other != user` covers the same `balances[user]` vs
   `balances[other]` pair as `requires user != other`.
10. All-pairs negative:
    one read `balances[other]` and two writes `balances[user]` and
    `balances[admin]` with only `requires user != other` emits no
    evidence-backed row, because the `admin` pair is uncovered.
11. Owner-scope negative:
    an assumption from another function cannot be referenced by evidence.
12. Unsupported shape diagnostics:
    `!(user == other)`, `user != 42`, `assume(user != other)`, and address/u160
    variables each report the precise unsupported reason and do not emit an
    evidence-backed row.
13. Formula projection boundary:
    a source-level `ensures balances[other] == old(balances[other])` that
    needs assumption-backed frame equality remains outside the Lean fragment in
    this slice, with a diagnostic explaining that conditional post-state
    storage is deferred.
14. Gate discipline:
    the schema-version bump is tested, and `check-formal-sync` remains green.

## 11. Review Checklist

Before implementation, review these points explicitly:

1. The new relation is separate from `placeDefinitelyDisjoint`.
2. Evidence-backed rows carry assumptions through `addQuery`.
3. Lean validates the evidence shape and formula denotation; Zig does not just
   assert it.
4. Contradictory assumptions fail through the existing anti-vacuity gate.
5. Runtime guard erasure and codegen are byte-identical with and without a Lean
   proof.
6. Logical storage projection remains static-only until a separate conditional
   post-state design exists.

## 12. Verdict

Direction: implementable after review.

The narrow design is intentional. This slice lets user-written disequality
evidence discharge frame obligations, but it does not yet claim source-level
storage postconditions are Lean-denotable under that evidence. That keeps the
first assumption-backed storage slice on the existing manifest/proof-gate
architecture and avoids smuggling a conditional frame axiom through term
projection.

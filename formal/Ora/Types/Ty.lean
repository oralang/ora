/-
Ora type system — the composite type universe.

Builds the full Ora type lattice over the primitive core (`Ora.Types.PrimTy`)
and regions (`Ora.Types.Region`). This is still the STATIC type layer: it says
which types exist and how they compose, not which runtime values inhabit them.

Source of truth: `src/types/semantic.zig` (`TypeKind` / `Type`). The 28
`TypeKind`s are modeled as:

  * the 7 PRIMITIVE kinds (void, bool, integer, string, address, bytes,
    fixed_bytes) are embedded via `Ty.prim : PrimTy → Ty`;
  * the 17 COMPOSITE / structural / nominal kinds get constructors below;
  * 4 kinds are intentionally EXCLUDED from this SURFACE/user-visible universe;
    they belong to a future `Internal.lean` CORE layer, and they are NOT all the
    same thing:
      - `never`   — a REAL bottom type ⊥ (the type of diverging / abort /
                    compile-error expressions; fits anywhere; the lattice ⊥, see
                    `type_check.zig:7071`, `type_descriptors.zig:215`). Not
                    surface-spellable, but genuinely part of the CORE type
                    system — it will live in the internal layer, e.g.
                    `InternalTy | runtime : Ty → InternalTy | never`.
      - `unknown` — a fail-closed ERROR sentinel; only appears in ill-typed /
                    error states, never in a well-typed program. An ELABORATION
                    state.
      - `named`   — an UNRESOLVED-name placeholder: a name not yet resolved to a
                    `struct_`/`enum_`/`contract`/… Also an elaboration state.
                    Together `unknown` and `named` form a separate
                    `ElabTy | unknown | named : Name → ElabTy | resolved : Ty → ElabTy`.
      - `comptime_integer` — comptime-only; lowers to a sized int before runtime
                    (a `ComptimeTy`/`StaticTy` layer).

  DECISION: `Ty` is the surface/user-visible universe (no `never`). The bottom
  type, elaboration states, and comptime types live in a separate core/internal
  layer added when the typing rules and lattice actually need them.

Nominal vs structural:
  The nominal kinds (`struct`, `enum`, `bitfield`, `contract`) are NAME
  references — the compiler stores them as `NamedType{name}` and resolves the
  definition in a declaration environment. We carry just the name here; the
  field/variant definitions live in a later declaration-context layer, which
  also keeps the `Ty` recursion well-founded. `anonStruct` is the one STRUCTURAL
  record (inline fields), matching `anonymous_struct`.
-/

import Ora.Types.Region
import Ora.Types.Prim

namespace Ora.Types

/-- A user-level identifier (struct/enum/contract/field/trait/refinement name).
    The compiler interns `[]const u8`; `String` is the faithful model. -/
abbrev Name := String

/--
Argument to a registry-backed refinement type. Mirrors
`semantic.zig:RefinementArg = union { Type, Integer: { text } }`.

`typeMarker` is the compiler's payload-less `.Type` arm: a marker that this
argument POSITION is a type — the concrete type lives in the refinement's
`base_type`, not in the arg (confirmed at `type_descriptors.zig:453`). `integer`
is `.Integer{text}`, the textual constant of an integer argument (the `100` in
`MinValue<u256, 100>`). Equality follows `refinementArgEql` (`:474`): same
variant, and same text for integers — hence `DecidableEq`.
-/
inductive RefinementArg where
  | typeMarker
  | integer : String → RefinementArg
  deriving Repr, DecidableEq

/--
The Ora type universe.

Recursive: aggregates, error unions, refinements, functions, and resource types
carry sub-`Ty`s. Nominal kinds carry only a `Name` (resolved elsewhere), so the
recursion stays structurally well-founded.
-/
inductive Ty where
  /-- Primitive core (`PrimTy`): void, bool, ints, address, bytes, bytesN, string. -/
  | prim : PrimTy → Ty
  -- aggregates (product)
  /-- `tuple` — positional product. -/
  | tuple : List Ty → Ty
  /-- `anonymous_struct` — inline named-field record (structural). -/
  | anonStruct : List (Name × Ty) → Ty
  /-- `array` — element type with an OPTIONAL fixed length (`none` = unsized). -/
  | array : Ty → Option Nat → Ty
  /-- `slice` — dynamically sized view of an element type. -/
  | slice : Ty → Ty
  /-- `map` — key ⇒ value association. -/
  | map : Ty → Ty → Ty
  -- sum / fallible
  /-- `error_union` — a success payload plus the set of possible error types. -/
  | errorUnion : Ty → List Ty → Ty
  -- refinement (registry-backed: name + base + parameters)
  /-- `refinement` — a named refinement over a base type with type/integer args. -/
  | refinement : Name → Ty → List RefinementArg → Ty
  -- nominal (name references; definitions resolved in a declaration env)
  /-- `struct_` — nominal struct (underscore matches the compiler `TypeKind`
      and avoids visual collision with Lean's `structure`). -/
  | struct_ : Name → Ty
  /-- `enum_` — nominal enum (underscore matches the compiler `TypeKind`). -/
  | enum_ : Name → Ty
  /-- `bitfield` — nominal packed bitfield. -/
  | bitfield : Name → Ty
  /-- `contract` — nominal contract type. -/
  | contract : Name → Ty
  -- callable
  /-- `function` — value-level function: optional name, parameter types ⇒ return
      types (Ora functions may return multiple values). The optional name is
      part of compiler `typeEql`, so it is part of the model. -/
  | function : Option Name → List Ty → List Ty → Ty
  -- resource / linear
  /-- `resource_domain` — a linear resource domain over a carrier type. -/
  | resourceDomain : Name → Ty → Ty
  /-- `resource_place` — a place within a resource domain. -/
  | resourcePlace : Ty → Ty
  -- external
  /-- `external_proxy` — an external-contract proxy named by its trait. -/
  | externalProxy : Name → Ty
  -- storage handles
  /-- `storage_slot` — a first-class storage slot handle. -/
  | storageSlot : Ty
  /-- `storage_range` — a first-class storage range handle. -/
  | storageRange : Ty
  -- NOTE: no `deriving Repr`/`DecidableEq` here. Lean's auto-deriving does not
  -- synthesize `Repr`/`DecidableEq` through the nested `List Ty` occurrences;
  -- those instances will be added manually (or via mathlib) once a comparison /
  -- display need arises. Same open decision as `PrimTy`.

/-! ## Structural induction principle

Because `Ty` recurses *through* `List Ty` (`tuple`, `function`, `errorUnion`) and
`List (Name × Ty)` (`anonStruct`), it is a NESTED inductive: the auto-generated
`Ty.rec` exposes those children through three separate companion motives
(`motive_2 : List Ty → _`, `motive_3 : List (Name × Ty) → _`, `motive_4 : Name ×
Ty → _`). That is why a bare `induction t` is rejected ("multiple motives") and
why proofs over `Ty` were forced to hand-supply motives.

`Ty.recAux` discharges that obligation ONCE: it pins the companion motives to the
element-wise predicate (`∀ t ∈ ts, motive t`) so every downstream proof sees a
single `motive : Ty → Prop` and gets an ordinary element-wise induction
hypothesis for each aggregate. Use it via `induction t using Ty.recAux`. -/

/-- Structural induction for `Ty`. Each aggregate constructor hands you an
    induction hypothesis for *every element* of its child list.

    `motive` is `Prop`-valued: this is a proof principle, not a data recursor —
    the element-wise IH cases on `List.Mem`, which lives in `Prop`. -/
@[elab_as_elim]
def Ty.recAux {motive : Ty → Prop}
    (prim : ∀ p, motive (.prim p))
    (tuple : ∀ ts, (∀ t ∈ ts, motive t) → motive (.tuple ts))
    (anonStruct : ∀ fs, (∀ f ∈ fs, motive f.2) → motive (.anonStruct fs))
    (array : ∀ e n, motive e → motive (.array e n))
    (slice : ∀ e, motive e → motive (.slice e))
    (map : ∀ k v, motive k → motive v → motive (.map k v))
    (errorUnion : ∀ p es, motive p → (∀ e ∈ es, motive e) → motive (.errorUnion p es))
    (refinement : ∀ n b as, motive b → motive (.refinement n b as))
    (struct_ : ∀ n, motive (.struct_ n)) (enum_ : ∀ n, motive (.enum_ n))
    (bitfield : ∀ n, motive (.bitfield n)) (contract : ∀ n, motive (.contract n))
    (function : ∀ n ps rs, (∀ t ∈ ps, motive t) → (∀ t ∈ rs, motive t) →
      motive (.function n ps rs))
    (resourceDomain : ∀ n c, motive c → motive (.resourceDomain n c))
    (resourcePlace : ∀ e, motive e → motive (.resourcePlace e))
    (externalProxy : ∀ n, motive (.externalProxy n))
    (storageSlot : motive .storageSlot) (storageRange : motive .storageRange)
    (t : Ty) : motive t :=
  Ty.rec (motive_1 := motive)
    (motive_2 := fun ts => ∀ t ∈ ts, motive t)
    (motive_3 := fun fs => ∀ f ∈ fs, motive f.2)
    (motive_4 := fun f => motive f.2)
    prim tuple anonStruct array slice map errorUnion refinement
    struct_ enum_ bitfield contract function resourceDomain resourcePlace
    externalProxy storageSlot storageRange
    (fun _ h => nomatch h)
    (fun _ _ ih iht t hm => by cases hm with | head => exact ih | tail _ h => exact iht _ h)
    (fun _ h => nomatch h)
    (fun _ _ ih iht f hm => by cases hm with | head => exact ih | tail _ h => exact iht _ h)
    (fun _ _ ih => ih) t

/-! ## Located types — σ ::= τ @ ρ -/

/--
A located type: a type, the region it lives in, and its provenance (where the
value came from — a SEPARATE axis from region). Mirrors the compiler's
`LocatedType` (`semantic.zig:468`); `provenance` defaults to `.local`.

`σ ::= τ @ ρ` (`docs/formal-specs/type_system_spec_v1.md` §1.4). Mutability `µ` (on bindings)
and effects `ϵ` (on functions) are NOT part of σ — they live in other layers.
-/
structure Located where
  ty         : Ty
  region     : Region
  provenance : Provenance := .local

/-- An unlocated type: default region (`stack` ≡ compiler `.none`) and `local`
    provenance. Mirrors `LocatedType.unlocated`. -/
def Located.unlocated (t : Ty) : Located :=
  { ty := t, region := .stack }

/-- A type at a given region, with `local` provenance. Mirrors
    `LocatedType.withRegion`. -/
def Located.atRegion (t : Ty) (ρ : Region) : Located :=
  { ty := t, region := ρ }

/-! ## Composition sanity checks

    These `example`s force the kernel to typecheck representative compositions,
    confirming the constructors compose as intended. -/

/-- `map<address, u256>`. -/
example : Ty := .map (.prim .address) (.prim u256)

/-- `[]u256` (unsized array of `u256`). -/
example : Ty := .array (.prim u256) none

/-- `[32]bytes32` (fixed array). -/
example : Ty := .array (.prim (.fixedBytes ⟨32, by decide, by decide⟩)) (some 32)

/-- `u256 ! {ErrOverflow}` — a success payload with one error type (nominal). -/
example : Ty := .errorUnion (.prim u256) [.enum_ "ErrOverflow"]

/-- `(u256, address) -> (bool)` — multi-arg, single-return function. -/
example : Ty := .function none [.prim u256, .prim .address] [.prim .bool]

/-- `u256 @ storage` — a located type. -/
example : Located := { ty := .prim u256, region := .storage }

end Ora.Types

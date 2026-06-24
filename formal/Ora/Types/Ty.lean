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
Argument to a registry-backed refinement type.

Mirrors `semantic.zig:RefinementArg = union { Type, Integer: { text } }`.
NOTE: the compiler's `.Type` arm is payload-less — its precise meaning (a
type-parameter slot?) is worth confirming; modeled here as a nullary `type`.
-/
inductive RefinementArg where
  /-- The compiler's payload-less `.Type` arm. Named `typeMarker` (not `type`)
      because it does NOT carry a type — its precise meaning (a type-parameter
      slot?) still needs confirming against the compiler's usage. -/
  | typeMarker
  | integer : String → RefinementArg
  deriving Repr

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
  /-- `function` — value-level function: parameter types ⇒ return types (Ora
      functions may return multiple values). -/
  | function : List Ty → List Ty → Ty
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

/-! ## Located types — σ ::= τ @ ρ -/

/--
A located type packages a type with the region it lives in.

`σ ::= τ @ ρ`  (`docs/formal-specs/ora-2.md` §4.4). Mutability `µ` and effects
`ϵ` join this in a later layer.
-/
structure Located where
  ty     : Ty
  region : Region

/-- Embed a primitive at the default (stack) region. -/
def Located.ofPrimAtStack (p : PrimTy) : Located :=
  { ty := .prim p, region := .stack }

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
example : Ty := .function [.prim u256, .prim .address] [.prim .bool]

/-- `u256 @ storage` — a located type. -/
example : Located := { ty := .prim u256, region := .storage }

end Ora.Types

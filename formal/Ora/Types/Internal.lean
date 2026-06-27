/-
Ora type system — the core / internal type layer.

The surface universe (`Ty`) is what a user can write. The compiler also uses
types that are NOT surface-spellable; this file models them, kept distinct from
`Ty` so the surface metatheory stays clean (review §1/§3).

  * `InternalTy` — the surface types PLUS the bottom type `never`. `never` is a
    REAL type (the type of diverging / abort / compile-error expressions and the
    ⊥ of the assignability lattice — compiler `TypeKind.never`), just not
    surface-spellable. Assignability (next layer) ranges over `InternalTy`.

  * `ElabTy` — ELABORATION states, not real types: an unresolved name (`named`),
    a fail-closed error sentinel (`unknown`), or a resolved surface type. A
    well-typed elaboration result is `resolved _`; `unknown`/`named` only occur
    mid-elaboration. The `resolve` step below turns a `named` into a `resolved`
    nominal type against the declaration context, or `unknown` if undeclared.

`comptime_integer` (the other excluded `TypeKind`) is a comptime-only literal
type; it belongs to a separate comptime/static layer and is not modeled here.
-/

import Ora.Types.Ty
import Ora.Types.Decl

namespace Ora.Types

/-! ## Core types: surface ∪ `never` -/

/-- The core type universe: a surface type, or the bottom type `never`. -/
inductive InternalTy where
  | runtime : Ty → InternalTy
  | never

/-- Is this the bottom type? -/
def InternalTy.isNever : InternalTy → Bool
  | .never => true
  | _      => false

theorem runtime_not_never (t : Ty) : (InternalTy.runtime t).isNever = false := rfl
theorem never_isNever : InternalTy.never.isNever = true := rfl

/-! ## Elaboration states -/

/-- Elaboration state: an unresolved name, an error sentinel, or a resolved type. -/
inductive ElabTy where
  | unknown
  | named : Name → ElabTy
  | resolved : Ty → ElabTy

/-- A fully-elaborated state carries a real surface type. -/
def ElabTy.isResolved : ElabTy → Bool
  | .resolved _ => true
  | _           => false

/-- Extract the surface type from a resolved state. -/
def ElabTy.toTy? : ElabTy → Option Ty
  | .resolved t => some t
  | _           => none

/--
Resolve a name against the declaration context: a `named n` becomes the matching
nominal surface type when `n` is declared, else `unknown` (fail-closed). Already
`resolved`/`unknown` states pass through.
-/
def ElabTy.resolve (Γ : DeclEnv) : ElabTy → ElabTy
  | .named n =>
      match Γ.lookup n with
      | some (.struct_ _)  => .resolved (.struct_ n)
      | some (.enum_ _)    => .resolved (.enum_ n)
      | some (.bitfield _) => .resolved (.bitfield n)
      | some (.contract _) => .resolved (.contract n)
      | some (.resource c) => .resolved (.resourceDomain n c)
      | none               => .unknown
  | other => other

/-! ## Theorems -/

theorem unknown_not_resolved : ElabTy.isResolved .unknown = false := rfl
theorem named_not_resolved (n : Name) : ElabTy.isResolved (.named n) = false := rfl

/-- A declared name resolves to its nominal surface type. -/
theorem point_name_resolves :
    ElabTy.resolve examplePoint (.named "Point") = .resolved (.struct_ "Point") := rfl

/-- An undeclared name fails closed to `unknown`. -/
theorem missing_name_is_unknown :
    ElabTy.resolve examplePoint (.named "Nope") = .unknown := rfl

/-- Resolution lands in a well-formed elaboration state for a declared name. -/
theorem point_resolution_is_resolved :
    (ElabTy.resolve examplePoint (.named "Point")).isResolved = true := rfl

end Ora.Types

/-
Ora type system — coherence between structural equality and assignability.

`TypeEqLawful` made `Ty.beq` decide `=`; `AssignableLawful` made `Ty.assignable` a
preorder. This file relates the two and pins down exactly where they diverge.

  * `Ty.assignable_of_beq` / `Ty.mutual_assignable_of_beq` — equality implies (mutual)
    assignability: `beq ⊆ assignable`.

The inclusion is STRICT, and the gap has exactly two sources:

  1. INTEGER WIDENING. `assignable` accepts `u8 ↦ u256` (not symmetric) where `beq`
     does not. But `IntTy.assignable_antisymm` shows widening is *antisymmetric*:
     mutual widening forces equality, so it never conflates two distinct int types.

  2. REFINEMENT BOUNDS. `Ty.assignable`'s refinement arm only checks name + base —
     it IGNORES the bound args (the deferred "name + base only" modeling). So
     `MinValue<_,1>` and `MinValue<_,100>` are mutually assignable yet unequal. Hence
     `assignable` is NOT antisymmetric, and mutual assignability does NOT imply `beq`
     (`Ty.assignable_not_antisymm`, `Ty.beq_not_of_mutual_assignable`).

So the SOLE obstruction to "`beq a b ↔ assignable a b ∧ assignable b a`" is the
ignored refinement bounds — exactly the gap the bounds-aware subtyping in
`RefinementSubtype` closes. Wiring that into `Ty.assignable` would make mutual
assignability coincide with equality (modulo bound-literal text representation).
-/

import Ora.Types.AssignableLawful

namespace Ora.Types

/-! ## Equality implies assignability -/

theorem Ty.assignable_of_beq {a b : Ty} (h : Ty.beq a b = true) : Ty.assignable a b = true := by
  have : a = b := Ty.eq_of_beq a b h
  subst this; exact Ty.assignable_refl a

/-- Equal types are MUTUALLY assignable. -/
theorem Ty.mutual_assignable_of_beq {a b : Ty} (h : Ty.beq a b = true) :
    Ty.assignable a b = true ∧ Ty.assignable b a = true := by
  have : a = b := Ty.eq_of_beq a b h
  subst this; exact ⟨Ty.assignable_refl a, Ty.assignable_refl a⟩

/-! ## Divergence (1): integer widening — strict, asymmetric, but antisymmetric -/

example : Ty.assignable (.prim u256) (.prim u8) = true := rfl   -- u8 widens to u256
example : Ty.beq (.prim u256) (.prim u8) = false := rfl         -- but not equal
example : Ty.assignable (.prim u8) (.prim u256) = false := rfl  -- and not symmetric

/-- Integer assignability IS antisymmetric — mutual widening forces equality, so
    widening alone never conflates two distinct int types. -/
theorem IntTy.assignable_antisymm {e a : IntTy}
    (h1 : IntTy.assignable e a = true) (h2 : IntTy.assignable a e = true) : e = a := by
  cases e <;> cases a <;>
    first
    | exact Bool.noConfusion h1
    | exact Bool.noConfusion h2
    | (rename_i we wa; cases we <;> cases wa <;> simp_all [IntTy.assignable, IntTy.bits])

/-! ## Divergence (2): refinement bounds — `assignable` ignores them -/

example :
    Ty.assignable (.refinement "MinValue" (.prim u256) [.integer "1"])
                  (.refinement "MinValue" (.prim u256) [.integer "100"]) = true := rfl
example :
    Ty.assignable (.refinement "MinValue" (.prim u256) [.integer "100"])
                  (.refinement "MinValue" (.prim u256) [.integer "1"]) = true := rfl
example :
    Ty.beq (.refinement "MinValue" (.prim u256) [.integer "1"])
           (.refinement "MinValue" (.prim u256) [.integer "100"]) = false := rfl

/-- `Ty.assignable` is NOT antisymmetric — mutual assignability does not imply
    equality. The sole obstruction is the ignored refinement bounds (witnessed by
    `MinValue<_,1>` vs `MinValue<_,100>`); the bounds-aware subtyping in
    `RefinementSubtype` is what distinguishes them. -/
theorem Ty.assignable_not_antisymm :
    ¬ ∀ a b : Ty, Ty.assignable a b = true → Ty.assignable b a = true → a = b := by
  intro h
  have key := h (.refinement "MinValue" (.prim u256) [.integer "1"])
                (.refinement "MinValue" (.prim u256) [.integer "100"]) rfl rfl
  exact absurd key (by decide)

/-- Equivalently: mutual assignability does NOT imply `beq`. -/
theorem Ty.beq_not_of_mutual_assignable :
    ¬ ∀ a b : Ty, Ty.assignable a b = true → Ty.assignable b a = true → Ty.beq a b = true := by
  intro h
  have key := h (.refinement "MinValue" (.prim u256) [.integer "1"])
                (.refinement "MinValue" (.prim u256) [.integer "100"]) rfl rfl
  exact Bool.noConfusion key

end Ora.Types

/-
Ora type system — coherence between structural equality and assignability.

`TypeEqLawful` made `Ty.beq` decide `=`; `AssignableLawful` made `Ty.assignable` a
preorder. This file relates the two and pins down exactly where they diverge.

  * `Ty.assignable_of_beq` / `Ty.mutual_assignable_of_beq` — equality implies (mutual)
    assignability: `beq ⊆ assignable`.

The inclusion is STRICT. Where they diverge:

  1. INTEGER WIDENING. `assignable` accepts `u8 ↦ u256` (not symmetric) where `beq`
     does not. But `IntTy.assignable_antisymm` shows widening is *antisymmetric*:
     mutual widening forces equality, so it never conflates two distinct int types.

  2. REFINEMENT BOUNDS. `assignable` does proper interval-containment subtyping
     (`MinValue<1> ↦ MinValue<100>` one-way) — directional, but bound-aware (no longer
     "ignored"). Mutual refinement assignability forces equal bounds.

  3. ERROR-SET ORDER. `assignable` compares error sets by SUBSET; `beq` is positional.
     So an error union and its reordering are mutually assignable yet `beq`-distinct —
     which is why `assignable` is NOT antisymmetric and mutual assignability does NOT
     imply `beq` (`Ty.assignable_not_antisymm`, `Ty.beq_not_of_mutual_assignable`).
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

/-! ## Divergence (2): refinement bound-subtyping — directional, but bound-aware -/

-- tighter ↦ looser is accepted; the reverse is rejected (no longer "ignored")
example : Ty.assignable (.refinement "MinValue" (.prim u256) [.integer "1"])
                        (.refinement "MinValue" (.prim u256) [.integer "100"]) = true := by decide
example : Ty.assignable (.refinement "MinValue" (.prim u256) [.integer "100"])
                        (.refinement "MinValue" (.prim u256) [.integer "1"]) = false := by decide

/-! ## Divergence (3): error-set ORDER — `assignable` is subset, `beq` is positional -/

private def euA : Ty := .errorUnion (.prim u256) [.enum_ "E1", .enum_ "E2"]
private def euB : Ty := .errorUnion (.prim u256) [.enum_ "E2", .enum_ "E1"]

-- same error SET, different order: mutually assignable, yet `beq`-distinct
example : Ty.assignable euA euB = true := by decide
example : Ty.assignable euB euA = true := by decide
example : Ty.beq euA euB = false := by decide

/-- `Ty.assignable` is NOT antisymmetric — mutual assignability does not imply equality.
    Witness: an error union and its reordering (subset both ways, but not equal). -/
theorem Ty.assignable_not_antisymm :
    ¬ ∀ a b : Ty, Ty.assignable a b = true → Ty.assignable b a = true → a = b := by
  intro h
  exact absurd (h euA euB (by decide) (by decide)) (by decide)

/-- Equivalently: mutual assignability does NOT imply `beq`. -/
theorem Ty.beq_not_of_mutual_assignable :
    ¬ ∀ a b : Ty, Ty.assignable a b = true → Ty.assignable b a = true → Ty.beq a b = true := by
  intro h
  exact Bool.noConfusion (h euA euB (by decide) (by decide))

end Ora.Types

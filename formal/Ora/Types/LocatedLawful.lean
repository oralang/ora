/-
Ora type system — lawfulness of the located/internal assignability layer (`τ@ρ`).

`AssignableLawful` proved the TYPE axis (`Ty.assignable`) is a preorder. A located
type `σ ::= τ @ ρ` also has a REGION axis, and `InternalTy` adds the `never` ⊥. This
file establishes how those compose — and where they DON'T.

KEY RESULT — region coercion is single-step:
  * `Region.assignableTo` is REFLEXIVE but NOT TRANSITIVE. `calldata → memory` and
    `memory → storage` both hold, yet `calldata → storage` is rejected. This mirrors
    the compiler's `regionAssignable` arm-for-arm: implicit region coercion is a
    single step, not a transitively-closed order (you copy `calldata → memory`, then
    `memory → storage`, explicitly).

CONSEQUENCES:
  * `InternalTy.assignable` (the type axis + `never` ⊥) IS a preorder — a clean lift
    of `Ty.assignable_refl`/`_trans` (`never` as bottom never breaks transitivity:
    nothing coerces *from* `never` except into `never`).
  * `Located.assignable` is REFLEXIVE but, like its region component, NOT a preorder.
    Crucially the TYPE axis always composes: `Located.assignable_trans_of_region`
    shows the *only* obstruction to located transitivity is the region step — supply
    the direct region coercion and transitivity holds.
-/

import Ora.Types.Region
import Ora.Types.Assignable
import Ora.Types.AssignableLawful

namespace Ora.Types

/-! ## Region coercion: reflexive, but NOT transitive (single-step) -/

/-- Every region coerces to itself. -/
theorem Region.assignableTo_refl (r : Region) : Region.assignableTo r r = true := by
  cases r <;> rfl

/-- Region coercion is NOT transitive: `calldata → memory → storage` holds step-wise,
    but `calldata → storage` is rejected — implicit region coercion is single-step. -/
theorem Region.assignableTo_not_trans :
    ¬ ∀ a b c : Region, Region.assignableTo a b = true → Region.assignableTo b c = true →
      Region.assignableTo a c = true := by
  intro h
  exact Bool.noConfusion (h .calldata .memory .storage rfl rfl)

/-! ## Internal assignability (with `never` ⊥) IS a preorder

    (Bottom facts `never_bottom` / `never_only_from_never` live in `Assignable.lean`.) -/

/-- `InternalTy.assignable` is reflexive. -/
theorem InternalTy.assignable_refl (t : InternalTy) : InternalTy.assignable t t = true := by
  cases t with
  | never => rfl
  | runtime ty => exact Ty.assignable_refl ty

/-- `InternalTy.assignable` is transitive — the `never` ⊥ does not break it. -/
theorem InternalTy.assignable_trans {a b c : InternalTy}
    (h1 : InternalTy.assignable a b = true) (h2 : InternalTy.assignable b c = true) :
    InternalTy.assignable a c = true := by
  cases a <;> cases b <;> cases c <;>
    first
    | rfl
    | exact Ty.assignable_trans _ _ _ h1 h2
    | exact Bool.noConfusion h1
    | exact Bool.noConfusion h2

/-! ## Located assignability: reflexive, not a preorder; the type axis composes -/

/-- A located type is assignable to itself (type refl ∧ region refl). -/
theorem Located.assignable_refl (l : Located) : Located.assignable l l = true := by
  simp [Located.assignable, Ty.assignable_refl, Region.assignableTo_refl]

/-- Located transitivity FAILS — inherited from region intransitivity (same type,
    regions `calldata → memory → storage`). -/
theorem Located.assignable_not_trans :
    ¬ ∀ a b c : Located, Located.assignable a b = true → Located.assignable b c = true →
      Located.assignable a c = true := by
  intro h
  have key := h { ty := .prim u256, region := .calldata }
                { ty := .prim u256, region := .memory }
                { ty := .prim u256, region := .storage } (by decide) (by decide)
  exact absurd key (by decide)

/-- The TYPE axis always composes: the only obstruction to located transitivity is
    the region step. Given the direct region coercion `a.region → c.region`, located
    assignability is transitive. -/
theorem Located.assignable_trans_of_region {a b c : Located}
    (hr : Region.assignableTo a.region c.region = true)
    (h1 : Located.assignable a b = true) (h2 : Located.assignable b c = true) :
    Located.assignable a c = true := by
  simp only [Located.assignable, Bool.and_eq_true] at *
  exact ⟨Ty.assignable_trans _ _ _ h2.1 h1.1, hr⟩

end Ora.Types

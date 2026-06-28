/-
Ora refinement subtyping SOUNDNESS — the bound-subtyping decided by `Ty.assignable`
entails the right denotational containment.

`RefinementValue.lean` gives the predicates and the `∧`-projection entailments
(`InRange → MinValue ∧ MaxValue`, `NonZeroAddress → NonZero`, `BasisPoints →
InRange`) — none of which need order axioms. This file adds the entailments that DO
need transitivity: **monotonicity of the bounds**, i.e. "a tighter refinement
entails the looser one." These are the *meaning* of the `Ty.assignable` refinement
arm (§4.2 of the type-system spec): the subtyping condition `actual_min ≥
expected_min` / `actual_max ≤ expected_max` is sound precisely because it implies
the value set shrinks.

Core Lean has no `Preorder`, and we stay mathlib-free, so transitivity comes from a
4-line local `LePreorder` class with instances for the carriers we actually use
(`Int` — the bound type `Ty.assignable` parses to — and `Nat`). The bridge lemmas
at the end connect the *decidable* `boundOk` test (over `Int`) to the *denotational*
entailment, closing the loop: `Ty.assignable` says yes ⟹ the predicate really is
entailed.
-/

import Ora.Types.RefinementValue
import Ora.Types.Assignable

namespace Ora.Refine

universe u

/-- Minimal, mathlib-free preorder on `≤` (reflexive + transitive). This is the SINGLE
    home for order transitivity in the formal layer — do not reintroduce a second one. -/
class LePreorder (α : Type u) [LE α] where
  le_refl  : ∀ a : α, a ≤ a
  le_trans : ∀ a b c : α, a ≤ b → b ≤ c → a ≤ c

instance : LePreorder Int := ⟨Int.le_refl, fun _ _ _ h1 h2 => Int.le_trans h1 h2⟩
instance : LePreorder Nat := ⟨Nat.le_refl, fun _ _ _ h1 h2 => Nat.le_trans h1 h2⟩

variable {α : Type u} [LE α] [LePreorder α]

/-! ## Bound-subtyping soundness — a tighter refinement entails the looser -/

/-- `MinValue` is ANTITONE in its bound: raising the floor (`a ≤ b`) shrinks the value
    set, so a value meeting the higher floor `b` also meets the lower floor `a`.
    This is the soundness of the `MinValue` arm: `expected_min ≤ actual_min`. -/
theorem minValue_subsume {a b x : α} (hab : a ≤ b) (hx : MinValue b x) : MinValue a x :=
  LePreorder.le_trans a b x hab hx

/-- `MaxValue` is MONOTONE in its bound: lowering the ceiling (`b ≤ a`) shrinks the
    value set. Soundness of the `MaxValue` arm: `actual_max ≤ expected_max`. -/
theorem maxValue_subsume {a b x : α} (hba : b ≤ a) (hx : MaxValue b x) : MaxValue a x :=
  LePreorder.le_trans x b a hx hba

/-- `InRange` containment: `[lo₂,hi₂] ⊆ [lo₁,hi₁]` (`lo₁ ≤ lo₂`, `hi₂ ≤ hi₁`) means
    a value in the tighter interval lies in the wider one. -/
theorem inRange_subsume {lo1 hi1 lo2 hi2 x : α}
    (hlo : lo1 ≤ lo2) (hhi : hi2 ≤ hi1) (hx : InRange lo2 hi2 x) : InRange lo1 hi1 x :=
  ⟨LePreorder.le_trans lo1 lo2 x hlo hx.1, LePreorder.le_trans x hi2 hi1 hx.2 hhi⟩

/-! ## Cross-kind soundness (the gate's `cross_*` rows) -/

/-- A value in `[lo, hi]` meets a `MinValue` floor `a ≤ lo` — `MinValue` accepts
    `InRange` when the floor is below the interval. -/
theorem inRange_to_minValue {a lo hi x : α} (h : a ≤ lo) (hx : InRange lo hi x) :
    MinValue a x :=
  LePreorder.le_trans a lo x h hx.1

/-- A value in `[lo, hi]` meets a `MaxValue` ceiling `hi ≤ b`. -/
theorem inRange_to_maxValue {b lo hi x : α} (h : hi ≤ b) (hx : InRange lo hi x) :
    MaxValue b x :=
  LePreorder.le_trans x hi b hx.2 h

/-! ## Bridge — the decidable `boundOk` test (`Int`) ⟹ the denotational entailment

    `Ty.assignable`'s refinement arm reduces to `Ord.Types.boundOk` over the `Int`
    bounds parsed from the type. These lemmas show that a passing `boundOk` is exactly
    the hypothesis the subsumption lemmas need, instantiated at `α = Int`. -/

open Ora.Types in
/-- If the `MinValue` bound check passes (`boundOk (some e) (some a) true`), then a
    value satisfying the actual floor `a` satisfies the expected floor `e`. -/
theorem minValue_of_boundOk {e a x : Int}
    (h : boundOk (some e) (some a) true = true) (hx : MinValue a x) : MinValue e x := by
  have hea : e ≤ a := by simpa [boundOk] using h
  exact minValue_subsume hea hx

open Ora.Types in
/-- If the `MaxValue` bound check passes (`boundOk (some e) (some a) false`), then a
    value satisfying the actual ceiling `a` satisfies the expected ceiling `e`. -/
theorem maxValue_of_boundOk {e a x : Int}
    (h : boundOk (some e) (some a) false = true) (hx : MaxValue a x) : MaxValue e x := by
  have hae : a ≤ e := by simpa [boundOk] using h
  exact maxValue_subsume hae hx

end Ora.Refine

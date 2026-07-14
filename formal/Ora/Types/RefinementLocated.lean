/-
Ora refinement × (InternalTy / Located) interaction.

The refinement subtyping (`Ty.assignable`'s refinement arm, §4.2) and its soundness
(`RefinementSubsumption.lean`) live on the bare `Ty`. This file lifts them through the
two layers above `Ty`:

  * `InternalTy` — adds the bottom `never`. A refined slot accepts `never`
    (vacuously refined), and refinement subtyping IS `InternalTy` assignability for
    refined runtimes.
  * `Located` — adds region + provenance. Region coercion and the refinement
    predicate are ORTHOGONAL: a refined located value assigns iff the refinement is a
    subtype AND the region coerces; the region never affects the predicate.

The payoff (`Located.minValue_sound`): when a `MinValue<e>@ρ'` slot accepts a
`MinValue<a>@ρ` value, a runtime value satisfying the actual floor satisfies the
expected floor — region-agnostically. This ties the §4 decision to the §8 meaning
through the `Located` layer, end to end.
-/

import Ora.Types.RefinementSubsumption
import Ora.Types.LocatedLawful

namespace Ora.Types

/-! ## InternalTy: refinement subtyping and the bottom `never` -/

/-- Refinement subtyping IS `InternalTy` assignability for refined runtimes. -/
@[simp] theorem InternalTy.assignable_refinement (ne be ae na ba aa) :
    InternalTy.assignable (.runtime (.refinement ne be ae)) (.runtime (.refinement na ba aa))
      = Ty.assignable (.refinement ne be ae) (.refinement na ba aa) := rfl

/-- `never` (⊥) flows into any refined slot — a diverging/abort value vacuously
    satisfies every refinement. -/
theorem InternalTy.never_into_refinement (ne be ae) :
    InternalTy.assignable (.runtime (.refinement ne be ae)) .never = true := rfl

/-! ## Located: region ⊥ refinement -/

/-- A refined located value assigns iff (refinement subtyping) AND (region coercion);
    the two axes factor cleanly. -/
theorem Located.assignable_refinement {ne be ae na ba aa} {ρ ρ' : Region} {π π' : Provenance} :
    Located.assignable ⟨.refinement na ba aa, ρ, π⟩ ⟨.refinement ne be ae, ρ', π'⟩
      = (Ty.assignable (.refinement ne be ae) (.refinement na ba aa)
          && Region.assignableTo ρ ρ') :=
  rfl

/-- Region-orthogonality: located assignability gives the type-axis subtyping,
    independent of regions/provenance. -/
theorem Located.refined_type_subtyping {E A : Ty} {ρ ρ' : Region} {π π' : Provenance}
    (h : Located.assignable ⟨A, ρ, π⟩ ⟨E, ρ', π'⟩ = true) : Ty.assignable E A = true := by
  simp only [Located.assignable, Bool.and_eq_true] at h
  exact h.1

/-! ## End-to-end: a `Located` `MinValue` assignment entails the floor predicate -/

/-- Extract the floor inequality `E ≤ A` from a passing `MinValue` refinement
    subtyping condition (both the semantic-equal and the bounds branch give it). -/
theorem refCond_minValue_le {e a : String} {E A : Int}
    (he : parseInt? e = some E) (ha : parseInt? a = some A)
    (h : refCond "MinValue" [.integer e] "MinValue" [.integer a] = true) : E ≤ A := by
  unfold refCond at h
  rw [show isNZA "MinValue" = false from by decide] at h
  simp only [Bool.false_eq_true, if_false] at h
  split at h
  · -- semantic-equal branch: the args are string-equal, hence parse-equal
    rename_i hsem
    rw [Bool.and_eq_true] at hsem
    have harg := hsem.2
    simp only [refArgsEq, refArgEq, Bool.and_true] at harg
    rw [beq_iff_eq] at harg
    subst harg
    rw [he] at ha
    have hEA : E = A := Option.some.inj ha
    omega
  · -- bounds branch
    simp only [refBounds,
      show RefinementName.ofString? "MinValue" = some .minValue from by decide,
      intArgs, he, ha, Option.map, boundOk, if_true, Bool.and_true, decide_eq_true_eq] at h
    exact h

/-- SOUNDNESS through `Located`: if a `MinValue<e>@ρ'` slot accepts a `MinValue<a>@ρ`
    value, then any runtime value meeting the actual floor `A` meets the expected
    floor `E` — the region is irrelevant to the predicate. -/
theorem Located.minValue_sound {base : Ty} {e a : String} {E A : Int} {x : Int}
    {ρ ρ' : Region} {π π' : Provenance}
    (he : parseInt? e = some E) (ha : parseInt? a = some A)
    (hassign : Located.assignable ⟨.refinement "MinValue" base [.integer a], ρ, π⟩
                                  ⟨.refinement "MinValue" base [.integer e], ρ', π'⟩ = true)
    (hx : Refine.MinValue A x) : Refine.MinValue E x := by
  have htype : Ty.assignable (.refinement "MinValue" base [.integer e])
                             (.refinement "MinValue" base [.integer a]) = true :=
    Located.refined_type_subtyping hassign
  rw [asg_refine, Bool.and_eq_true] at htype
  exact Refine.minValue_subsume (refCond_minValue_le he ha htype.2) hx

end Ora.Types

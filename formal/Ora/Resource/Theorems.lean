/-
Reusable theorems for the core resource-state model.

These facts intentionally cover local deltas and frame preservation over the
touched places. They do not claim global finite-domain conservation.
-/

import Ora.Resource.Model

namespace Ora.Resource

open Ora.Obligation

theorem amount_eq (s : State P) (p : P) :
    amount s p = s p := by
  rfl

/--
Resource carriers are unsigned. This theorem is a denotation-case witness:
resource `amount_non_negative` maps to this proposition, never to bare `True`.
-/
theorem amount_non_negative (x : Carrier) :
    amountNonNegative x := by
  unfold amountNonNegative
  exact Nat.zero_le x.toNat

theorem create_delta [DecidableEq P]
    (s : State P) (p : P) (x : Carrier)
    (hGuard : destinationNoOverflow s p x) :
    (create s p x p).toNat = (s p).toNat + x.toNat := by
  unfold create update
  simp
  exact U256.toNat_add_noOverflow (s p) x hGuard

theorem destroy_delta [DecidableEq P]
    (s : State P) (p : P) (x : Carrier)
    (hGuard : sourceSufficient s p x) :
    (destroy s p x p).toNat + x.toNat = (s p).toNat := by
  unfold destroy update
  simp
  exact U256.toNat_sub_noUnderflow (s p) x hGuard

theorem move_self [DecidableEq P] (s : State P) (p : P) (x : Carrier) :
    move s p p x = s := by
  unfold move
  simp

theorem move_destination_no_overflow_self
    (s : State P) (p : P) (x : Carrier) :
    moveDestinationNoOverflow s p p x := by
  unfold moveDestinationNoOverflow
  exact Or.inl rfl

theorem move_source_sufficient_self_iff
    (s : State P) (p : P) (x : Carrier) :
    moveSourceSufficient s p p x ↔ sourceSufficient s p x := by
  rfl

theorem move_source_sufficient_self_insufficient_counterexample :
    ¬ moveSourceSufficient
      (fun _ : Nat => BitVec.ofNat 256 0)
      0
      0
      (BitVec.ofNat 256 1) := by
  unfold moveSourceSufficient sourceSufficient
  simp

theorem source_sufficient_of_move
    (s : State P) (source destination : P) (x : Carrier)
    (hGuard : moveSourceSufficient s source destination x) :
    sourceSufficient s source x := by
  exact hGuard

theorem destination_no_overflow_of_move_distinct
    (s : State P) (source destination : P) (x : Carrier)
    (hDistinct : source ≠ destination)
    (hGuard : moveDestinationNoOverflow s source destination x) :
    destinationNoOverflow s destination x := by
  unfold moveDestinationNoOverflow at hGuard
  cases hGuard with
  | inl hSame => exact False.elim (hDistinct hSame)
  | inr hDestination => exact hDestination

theorem move_conserves_distinct [DecidableEq P]
    (s : State P) (source destination : P) (x : Carrier)
    (hDistinct : source ≠ destination)
    (hSource : sourceSufficient s source x)
    (hDestination : destinationNoOverflow s destination x) :
    (move s source destination x source).toNat +
        (move s source destination x destination).toNat =
      (s source).toNat + (s destination).toNat := by
  unfold move update
  simp [hDistinct, hDistinct.symm]
  have hSub := U256.toNat_sub_noUnderflow (s source) x hSource
  have hAdd := U256.toNat_add_noOverflow (s destination) x hDestination
  rw [hAdd]
  omega

theorem create_frame [DecidableEq P]
    (s : State P) (p untouched : P) (x : Carrier)
    (hUntouched : untouched ≠ p) :
    create s p x untouched = s untouched := by
  unfold create update
  simp [hUntouched]

theorem destroy_frame [DecidableEq P]
    (s : State P) (p untouched : P) (x : Carrier)
    (hUntouched : untouched ≠ p) :
    destroy s p x untouched = s untouched := by
  unfold destroy update
  simp [hUntouched]

theorem move_frame [DecidableEq P]
    (s : State P) (source destination untouched : P) (x : Carrier)
    (hSource : untouched ≠ source)
    (hDestination : untouched ≠ destination) :
    move s source destination x untouched = s untouched := by
  unfold move update
  by_cases hSame : source = destination
  · simp [hSame]
  · simp [hSame, hSource, hDestination]

end Ora.Resource

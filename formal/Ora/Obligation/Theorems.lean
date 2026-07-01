/-
Reviewed obligation theorem foundation.

This module intentionally contains only reusable facts over the supported
semantic fragment. Generated-manifest adapters belong in small sync modules
once the compiler emits reproducible Lean fixtures for the free/bound variable
model.
-/

import Ora.Obligation.Semantics

namespace Ora.Obligation

/-! ## Reusable u256 facts -/

theorem u256_ult_implies_ule (i x : U256) :
    U256.ult i x → U256.ule i x :=
  U256.ult_implies_ule i x

theorem u256_inrange_implies_minvalue (lo hi x : U256) :
    (U256.ule lo x ∧ U256.ule x hi) → U256.ule lo x := by
  intro h
  exact h.1

theorem u256_inrange_implies_maxvalue (lo hi x : U256) :
    (U256.ule lo x ∧ U256.ule x hi) → U256.ule x hi := by
  intro h
  exact h.2

theorem u256_basispoints_implies_inrange (x : U256) :
    (U256.ule (BitVec.ofNat 256 0) x ∧
      U256.ule x (BitVec.ofNat 256 10000)) →
    (U256.ule (BitVec.ofNat 256 0) x ∧
      U256.ule x (BitVec.ofNat 256 10000)) := by
  intro h
  exact h

theorem u256_nonzeroaddress_implies_nonzero (x : U256) :
    x ≠ BitVec.ofNat 256 0 → x ≠ BitVec.ofNat 256 0 := by
  intro h
  exact h

theorem u256_bounded_successor_preserves_order (i x : U256) :
    U256.ult x U256.max →
    U256.ult i x →
    U256.ule i (U256.add x (BitVec.ofNat 256 1)) :=
  U256.lt_max_succ_ule i x

theorem u256_bounded_add_preserves_order (i x bound step : U256) :
    bound.toNat + step.toNat ≤ 2^256 →
    U256.ult x bound →
    U256.ult i x →
    U256.ule i (U256.add x step) :=
  U256.lt_bound_add_ule i x bound step

theorem u256_bounded_symbolic_add_preserves_order (i x step : U256) :
    U256.ult x (U256.sub U256.max step) →
    U256.ult i x →
    U256.ule i (U256.add x step) :=
  U256.lt_max_sub_add_ule i x step

/-! ## Effect-frame facts over decoded manifests -/

theorem effect_frame_write_covered_denotes
    (declared actual : List PlaceRef)
    (hCovered : placeListCovers declared actual = true) :
    match EffectFrameGoal.denotes?
      { relation := .writeCoveredByModifies, declared := declared, actual := actual } with
    | some proposition => proposition
    | none => False := by
  simp [EffectFrameGoal.denotes?, hCovered]

theorem effect_frame_read_preserved_denotes
    (declared actual : List PlaceRef)
    (hDisjoint : placeListDisjoint declared actual = true) :
    match EffectFrameGoal.denotes?
      { relation := .readPreservedByFrame, declared := declared, actual := actual } with
    | some proposition => proposition
    | none => False := by
  simp [EffectFrameGoal.denotes?, hDisjoint]

theorem unsupported_effect_frame_relation_fails_closed
    (declared actual : List PlaceRef) :
    (EffectFrameGoal.denotes?
      { relation := .lockCoversWrite, declared := declared, actual := actual }) = none := by
  rfl

end Ora.Obligation

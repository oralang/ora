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

/-! ## Storage-place facts -/

theorem stable_place_read_self_eq_denotes (env : Env) (place : PlaceRef) :
    match Value.eqProp? (env.placeValue (.stable place)) (env.placeValue (.stable place)) with
    | some proposition => proposition
    | none => False := by
  cases hValue : env.placeValue (.stable place) with
  | bool value => simp [Value.eqProp?]
  | integer value =>
      rcases value with ⟨shape, bits⟩
      simp [Value.eqProp?, Ora.Integer.Value.eqProp?]

/-! ## Effect-frame facts over decoded manifests -/

theorem option_prop_and_true_left_intro (candidate : Option Prop) :
    (match candidate with
    | some proposition => proposition
    | none => False) →
    (match optionPropAnd? (some True) candidate with
    | some proposition => proposition
    | none => False) := by
  cases candidate <;> simp [optionPropAnd?]

theorem effect_frame_write_covered_denotes
    (manifest : Manifest)
    (env : Env)
    (declared actual : List PlaceRef)
    (hCovered : placeListCovers declared actual = true) :
    match effectFrameGoalDenotes? manifest env
      { relation := .writeCoveredByModifies, declared := declared, actual := actual } with
    | some proposition => proposition
    | none => False := by
  simp [effectFrameGoalDenotes?, hCovered]

theorem effect_frame_read_preserved_denotes
    (manifest : Manifest)
    (env : Env)
    (declared actual : List PlaceRef)
    (hDisjoint : placeListDisjoint declared actual = true) :
    match effectFrameGoalDenotes? manifest env
      { relation := .readPreservedByFrame, declared := declared, actual := actual } with
    | some proposition => proposition
    | none => False := by
  simp [effectFrameGoalDenotes?, hDisjoint]

theorem unsupported_effect_frame_relation_fails_closed
    (manifest : Manifest)
    (env : Env)
    (declared actual : List PlaceRef) :
    (effectFrameGoalDenotes? manifest env
      { relation := .lockCoversWrite, declared := declared, actual := actual }) = none := by
  rfl

/-- A checked witness for stronger assumptions establishes original nonvacuity. -/
theorem strengthened_sat_implies_original {State : Type} (assumptions pin : State → Prop)
    (witness : ∃ state, assumptions state ∧ pin state) :
    ∃ state, assumptions state := by
  obtain ⟨state, h, _⟩ := witness
  exact ⟨state, h⟩

/-- Failure to find the pinned witness gives no contradictory-assumptions verdict. -/
theorem pinned_unsat_does_not_establish_original_unsat :
    ¬ (∀ (assumptions pin : Bool → Prop),
      (¬ ∃ state, assumptions state ∧ pin state) → ¬ ∃ state, assumptions state) := by
  intro h
  exact h (fun _ => True) (fun _ => False) (by simp) ⟨true, trivial⟩

/-- The counterexample search must remain unpinned. -/
theorem pinned_safety_does_not_establish_original_safety :
    ¬ (∀ (assumptions pin safe : Bool → Prop),
      (∀ state, assumptions state → pin state → safe state) →
      ∀ state, assumptions state → safe state) := by
  intro h
  have bad := h (fun _ => True) (fun state => state = false)
    (fun state => state = false) (fun _ _ hp => hp) true trivial
  cases bad

end Ora.Obligation

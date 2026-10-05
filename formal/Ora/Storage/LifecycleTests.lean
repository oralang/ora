import Ora.Storage.Lifecycle

namespace Ora.StorageLifecycle
open Ora.Loop Ora.Obligation

def capClaim : Claim := {
  location := { root := "contract:Probe/cap" }
  name := "MaxValue"
  bounds := [.integer (.ofNat (.unsigned .w8) 100)]
}

def u8 (n : Nat) : Value := .integer (.ofNat (.unsigned .w8) n)

theorem default_zero_valid : capClaim.holds (u8 0) := by
  change (0 : Nat) ≤ 100
  decide
theorem safe_write_valid : capClaim.holds (u8 100) := by
  change (100 : Nat) ≤ 100
  decide
theorem violating_write_invalid : ¬ capClaim.holds (u8 101) := by
  change ¬ (101 : Nat) ≤ 100
  decide
theorem wrong_width_invalid :
    ¬ capClaim.holds (.integer (.ofNat (.unsigned .w256) 0)) := by
  change ¬ False
  decide
theorem positive_minimum_rejects_zero :
    ¬ ({ capClaim with name := "MinValue", bounds := [u8 1] }).holds (u8 0) := by
  change ¬ (1 : Nat) ≤ 0
  decide

theorem aliased_bad_write_breaks :
    ¬ Invariant [capClaim] (StorageState.set (fun _ => u8 0) capClaim.location (u8 101)) := by
  intro h
  have bad := h capClaim (by simp)
  exact violating_write_invalid (by simpa using bad)

theorem omitted_writer_cannot_cover :
    ¬ Covers (fun before after => after = before.set capClaim.location (u8 101)) [] := by
  intro h
  rcases h (fun _ => u8 0) (StorageState.set (fun _ => u8 0) capClaim.location (u8 101)) rfl with
    ⟨_, hMember, _⟩
  simp at hMember

/-- A final-state check cannot justify a reentrant observation of the prefix. -/
theorem restored_final_state_does_not_rescue_bad_prefix :
    ¬ Invariant [capClaim]
      (StorageState.set (fun _ => u8 0) capClaim.location (u8 101)) ∧
    Invariant [capClaim]
      (StorageState.set
        (StorageState.set (fun _ => u8 0) capClaim.location (u8 101))
        capClaim.location (u8 0)) := by
  constructor
  · exact aliased_bad_write_breaks
  · intro claim hMember
    have hEqual : claim = capClaim := by simpa using hMember
    subst claim
    simpa using default_zero_valid

/-- Concrete nonempty transition family, with no supplied solver verdict. -/
def checkedCapWrite : Transition := fun before after =>
  ∃ value, capClaim.holds value ∧ after = before.set capClaim.location value

theorem checked_cap_certificate :
    Certificate [capClaim] (fun store => store = fun _ => u8 0)
      [checkedCapWrite] := by
  constructor
  · exact ⟨fun _ => u8 0, rfl⟩
  · intro store hInitial
    subst store
    intro claim hMember
    have hEqual : claim = capClaim := by simpa using hMember
    subst claim
    exact default_zero_valid
  · intro step hStep before after hInvariant hExec
    have hEqual : step = checkedCapWrite := by simpa using hStep
    subst step
    rcases hExec with ⟨value, hValue, rfl⟩
    apply checked_write_preserves _ _ _ _ hInvariant
    intro claim hMember _
    have hEqual : claim = capClaim := by simpa using hMember
    subst claim
    exact hValue

end Ora.StorageLifecycle

#print axioms Ora.StorageLifecycle.checked_write_preserves
#print axioms Ora.StorageLifecycle.action_preserves
#print axioms Ora.StorageLifecycle.certificate_induction
#print axioms Ora.StorageLifecycle.reached_getter_fact
#print axioms Ora.StorageLifecycle.covered_step_preserves
#print axioms Ora.StorageLifecycle.default_zero_valid
#print axioms Ora.StorageLifecycle.aliased_bad_write_breaks
#print axioms Ora.StorageLifecycle.omitted_writer_cannot_cover
#print axioms Ora.StorageLifecycle.restored_final_state_does_not_rescue_bad_prefix
#print axioms Ora.StorageLifecycle.checked_cap_certificate
#print axioms Ora.StorageLifecycle.safe_write_valid
#print axioms Ora.StorageLifecycle.violating_write_invalid
#print axioms Ora.StorageLifecycle.wrong_width_invalid
#print axioms Ora.StorageLifecycle.positive_minimum_rejects_zero
#print axioms Ora.StorageLifecycle.frame_preserves

namespace Ora.StorageLifecycle
open Ora.Obligation Ora.Loop

-- SL2B changes a canonical identity only where both names already denote Bool.
theorem boolean_type_identity (value : Value) :
    storageValueMatchesTy value (.compilerTypeId 13) =
      storageValueMatchesTy value (.spelling "bool") := by
  cases value <;> rfl

theorem boolean_accepts_exact (b : Bool) :
    storageValueMatchesTy (.bool b) (.spelling "bool") = true := by rfl

theorem integer_is_not_boolean (value : Ora.Integer.Value) :
    storageValueMatchesTy (.integer value) (.spelling "bool") = false := by rfl

#print axioms boolean_type_identity
#print axioms boolean_accepts_exact
#print axioms integer_is_not_boolean
end Ora.StorageLifecycle

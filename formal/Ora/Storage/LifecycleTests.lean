import Ora.Storage.Lifecycle
import Ora.Storage.Address

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

namespace Ora.StorageAddressTests
open Ora.Obligation Ora.Loop Ora.Storage

def addressContext (value : Value) : StorageFormulaContext := {
  manifest := { terms := [.variable (.free {
    id := { file_id := 0, pattern_id := 0 }, name := "key", ty := some (.compilerTypeId 14)
  })] }
  entryEnv := { freeBindings := [({ file_id := 0, pattern_id := 0 }, value)] }
  bindLocals := fun env _ => some env
  resolvePlace := fun _ _ => none
}

def addressState (store : StorageState) : StorageActionState := {
  loop := { locals := [], storage := store }, entryStorage := store
}

def addressMap : StorageLocationExpr := {
  root := "balances"
  projections := [.mappingKey { formula := .term 0, ty := .compilerTypeId 14 }]
  valueTy := .compilerTypeId 6
}

theorem addressKeyResolves (word : BitVec 160) (store : StorageState) :
    (addressContext (addressValue word)).evalLocation? addressMap (addressState store) =
      some { root := "balances", projections := [.mappingKey (addressValue word)] } := rfl

theorem wrongKeyRefused (value : Bool) (store : StorageState) :
    (addressContext (.bool value)).evalLocation? addressMap (addressState store) = none := rfl

theorem wideKeyRefused (value : BitVec 256) (store : StorageState) :
    (addressContext (.u256 value)).evalLocation? addressMap (addressState store) = none := rfl

theorem signedAddressRefused (width : Ora.Types.SIntWidth)
    (word : Ora.Integer.Carrier (.signed width)) :
    storageValueMatchesTy (.integer { shape := .signed width, bits := word })
      (.compilerTypeId 14) = false := by cases width <;> rfl

theorem missingKeyRefused (store : StorageState) :
    ({ addressContext (addressValue 0) with entryEnv := Env.empty }).evalLocation?
      addressMap (addressState store) = none := rfl

theorem addressIsNotArithmetic : (TyRef.compilerTypeId 14).integerShape? = none := rfl

theorem zeroIsAnAddress : storageValueMatchesTy (addressValue 0) (.compilerTypeId 14) = true := rfl

theorem addressWrite (word : BitVec 160) (store : StorageState) :
    execStorageAction? (addressContext (addressValue word)).actionSemantics
      (.write { root := "owner", valueTy := .compilerTypeId 14 } (.term 0)) (addressState store) =
      some { state := { loop := { locals := [], storage := store.set { root := "owner" } (addressValue word) }, entryStorage := store },
             writeTrace := [{ root := "owner" }] } := rfl

theorem wrongAddressWriteRefused (store : StorageState) :
    execStorageAction? (addressContext (.bool false)).actionSemantics
      (.write { root := "owner", valueTy := .compilerTypeId 14 } (.term 0)) (addressState store) = none := rfl

#print axioms Ora.Storage.nonAddressTypingUnchanged
#print axioms Ora.Storage.nonAddressLocationUnchanged
#print axioms Ora.Storage.addressTypingExact
#print axioms Ora.Storage.addressValueTyped
#print axioms Ora.Storage.addressSpellingAgrees
#print axioms Ora.Storage.addressEquality
#print axioms Ora.Storage.addressValueInjective
#print axioms Ora.Storage.addressMappingKeyAlias
#print axioms Ora.Storage.addressCastRoundTrip
#print axioms Ora.Storage.addressCastMask
#print axioms Ora.Storage.directAddressLocation
#print axioms addressKeyResolves
#print axioms wrongKeyRefused
#print axioms wideKeyRefused
#print axioms signedAddressRefused
#print axioms missingKeyRefused
#print axioms addressIsNotArithmetic
#print axioms zeroIsAnAddress
#print axioms addressWrite
#print axioms wrongAddressWriteRefused
end Ora.StorageAddressTests

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

import Ora.Loop.StorageDenotation

namespace Ora.Storage
open Ora.Obligation Ora.Loop

def addressValue (word : BitVec 160) : Value :=
  .integer { shape := .unsigned .w160, bits := word }

theorem nonAddressTypingUnchanged (value : Value) (ty : TyRef)
    (h : storageAddressType ty = false) :
    storageValueMatchesTy value ty = (match value with
      | .bool _ => ty.isBool
      | .integer number => ty.integerShape? == some number.shape) := by
  cases value <;> simp only [storageValueMatchesTy, h, Bool.false_eq_true, ↓reduceIte]

theorem nonAddressLocationUnchanged (ty : TyRef)
    (h : storageAddressType ty = false) :
    storageTypeSupported ty = (ty.isBool || ty.integerShape?.isSome) := by
  simp only [storageTypeSupported, h, Bool.or_false]

theorem addressTypingExact (value : Value) :
    storageValueMatchesTy value (.compilerTypeId 14) = true ↔
      ∃ word, value = addressValue word := by
  cases value with
  | bool b => simp [storageValueMatchesTy, TyRef.isBool, compilerTypeIdBool, Ora.Spec.expectedCompilerTypeIdBool, addressValue]
  | integer number =>
      cases number with
      | mk shape bits =>
          cases shape <;> rename_i width <;> cases width <;>
            simp [storageValueMatchesTy, storageAddressType, addressValue]

theorem addressValueTyped (word : BitVec 160) :
    storageValueMatchesTy (addressValue word) (.compilerTypeId 14) = true := rfl

theorem addressSpellingAgrees (value : Value) :
    storageValueMatchesTy value (.compilerTypeId 14) =
      storageValueMatchesTy value (.spelling "address") := by cases value <;> rfl

theorem addressEquality (left right : BitVec 160) :
    (addressValue left).eqProp? (addressValue right) = some (left = right) := rfl

theorem addressValueInjective (left right : BitVec 160) :
    addressValue left = addressValue right ↔ left = right := by simp [addressValue]

theorem addressMappingKeyAlias (left right : BitVec 160) :
    StorageProjection.mappingKey (addressValue left) =
      StorageProjection.mappingKey (addressValue right) ↔ left = right := by
  simp [addressValueInjective]

theorem addressCastRoundTrip (word : BitVec 160) :
    (word.setWidth 256).setWidth 160 = word := by
  simp [BitVec.setWidth_setWidth_of_le word (by decide : 160 ≤ 256)]

theorem addressCastMask (word : BitVec 160) :
    word.setWidth 256 &&& BitVec.ofNat 256 (2 ^ 160 - 1) = word.setWidth 256 := by
  apply BitVec.eq_of_toNat_eq
  rw [BitVec.toNat_and]
  change (word.setWidth 256).toNat &&& (2 ^ 160 - 1) = (word.setWidth 256).toNat
  rw [Ora.Integer.lowMaskIsWidth, addressCastRoundTrip]
  simp only [BitVec.toNat_setWidth]
  exact (Nat.mod_eq_of_lt (Nat.lt_trans word.isLt (by decide : 2 ^ 160 < 2 ^ 256))).symm

theorem directAddressLocation (context : StorageFormulaContext)
    (state : StorageActionState) (root : String) (hRoot : root.isEmpty = false) :
    context.evalLocation? { root, valueTy := .compilerTypeId 14 } state =
      some { root } := by
  simp [StorageFormulaContext.evalLocation?, storageTypeSupported, storageAddressType,
    hRoot, StorageFormulaContext.evalProjections?]

end Ora.Storage

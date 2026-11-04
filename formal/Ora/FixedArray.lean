/-
Checked fixed-array indexing semantics.

This module is independent of compiler rows and lowering.  It defines the
language-level safety condition consumed by runtime checks and formal
verification: a canonical unsigned index must be strictly below the exact
array extent.  Whether the access reads or writes does not change that
condition.
-/

import Ora.Integer.Arithmetic

namespace Ora.FixedArray

/-- Ora canonicalizes fixed-array indices to the EVM-width unsigned carrier. -/
abbrev Index := Ora.Integer.Carrier (.unsigned .w256)

/-- Checked semantic input for one fixed-array projection.  The extent is
separate from the resolved location because it is proof-relevant type data,
not part of the element identity. -/
structure CheckedIndex where
  index : Index
  extent : Nat
  deriving Repr, DecidableEq

inductive AccessKind where
  | read
  | write
  deriving Repr, DecidableEq

def CheckedIndex.Safe (input : CheckedIndex) : Prop :=
  input.index.toNat < input.extent

/-- Shape validity is checked independently from access safety so a malformed
extent cannot turn into an executable default. -/
def CheckedIndex.WellFormed (input : CheckedIndex) : Prop :=
  0 < input.extent ∧ input.extent < 2 ^ 256

def AccessKind.Safe (_ : AccessKind) (input : CheckedIndex) : Prop :=
  input.Safe

def CheckedIndex.ofNat (index extent : Nat) : CheckedIndex := {
  index := BitVec.ofNat 256 index
  extent
}

@[simp] theorem access_safe_iff
    (kind : AccessKind) (input : CheckedIndex) :
    kind.Safe input ↔ input.Safe :=
  Iff.rfl

theorem read_safe_iff_write_safe (input : CheckedIndex) :
    AccessKind.read.Safe input ↔ AccessKind.write.Safe input :=
  Iff.rfl

@[simp] theorem zero_safe_iff (extent : Nat) :
    (CheckedIndex.ofNat 0 extent).Safe ↔ 0 < extent := by
  simp [CheckedIndex.Safe, CheckedIndex.ofNat]

theorem ofNat_safe
    (index extent : Nat)
    (hIndex : index < extent)
    (hFits : index < 2 ^ 256) :
    (CheckedIndex.ofNat index extent).Safe := by
  simpa [CheckedIndex.Safe, CheckedIndex.ofNat, Nat.mod_eq_of_lt hFits]
    using hIndex

theorem last_safe
    (extent : Nat)
    (hPositive : 0 < extent)
    (hFits : extent < 2 ^ 256) :
    (CheckedIndex.ofNat (extent - 1) extent).Safe := by
  apply ofNat_safe
  · omega
  · omega

theorem extent_not_safe
    (extent : Nat)
    (hFits : extent < 2 ^ 256) :
    ¬(CheckedIndex.ofNat extent extent).Safe := by
  simp [CheckedIndex.Safe, CheckedIndex.ofNat, Nat.mod_eq_of_lt hFits]

theorem above_not_safe
    (index extent : Nat)
    (hOutside : extent ≤ index)
    (hFits : index < 2 ^ 256) :
    ¬(CheckedIndex.ofNat index extent).Safe := by
  simp [CheckedIndex.Safe, CheckedIndex.ofNat, Nat.mod_eq_of_lt hFits,
    Nat.not_lt.mpr hOutside]

end Ora.FixedArray

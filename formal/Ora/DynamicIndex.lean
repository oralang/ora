/-
Checked indexing semantics for dynamically sized Ora values.

This module is independent of compiler rows, regions, and lowering. It models
the language-level condition shared by slices, bytes, and strings: the
canonical unsigned index must be strictly below the runtime logical length of
the exact value being indexed. Physical region effects remain outside this
module.
-/

import Ora.FixedArray
import Ora.Loop.Timeline

namespace Ora.DynamicIndex

/-- Dynamic indices and lengths use Ora's canonical EVM-width carrier. -/
abbrev Word := Ora.Integer.Carrier (.unsigned .w256)

/-- Logical access granularity. `elements n` records the byte stride of one
element without changing the logical `index < length` rule. -/
inductive AccessKind where
  | byte
  | elements (strideBytes : Nat)
  deriving Repr, DecidableEq

def AccessKind.strideBytes : AccessKind → Nat
  | .byte => 1
  | .elements stride => stride

def AccessKind.WellFormed (kind : AccessKind) : Prop :=
  0 < kind.strideBytes

inductive Direction where
  | read
  | write
  deriving Repr, DecidableEq

/-- One checked dynamic access. `lengthOwner` is separate from `value` so a
malformed row that borrows another value's length is representable and can be
rejected rather than silently normalized. -/
structure CheckedAccess (ValueId : Type) where
  value : ValueId
  lengthOwner : ValueId
  index : Word
  length : Word
  kind : AccessKind
  deriving Repr, DecidableEq

/-- The bounds condition is exactly the unsigned logical comparison. -/
def CheckedAccess.BoundsSafe (access : CheckedAccess ValueId) : Prop :=
  access.index.toNat < access.length.toNat

/-- Shape validity is separate from bounds safety. In particular, a borrowed
length or zero element stride cannot become a valid access by satisfying the
numeric comparison. -/
def CheckedAccess.WellFormed (access : CheckedAccess ValueId) : Prop :=
  access.value = access.lengthOwner ∧ access.kind.WellFormed

def CheckedAccess.Authorized (access : CheckedAccess ValueId) : Prop :=
  access.WellFormed ∧ access.BoundsSafe

def Direction.Safe (_ : Direction) (access : CheckedAccess ValueId) : Prop :=
  access.Authorized

/-- The dynamic rule is the existing fixed-index rule instantiated with the
runtime length's mathematical value. -/
def CheckedAccess.asFixedIndex
    (access : CheckedAccess ValueId) : Ora.FixedArray.CheckedIndex := {
  index := access.index
  extent := access.length.toNat
}

@[simp] theorem boundsSafe_iff_fixedIndex_safe
    (access : CheckedAccess ValueId) :
    access.BoundsSafe ↔ access.asFixedIndex.Safe :=
  Iff.rfl

@[simp] theorem direction_safe_iff
    (direction : Direction) (access : CheckedAccess ValueId) :
    direction.Safe access ↔ access.Authorized :=
  Iff.rfl

theorem read_safe_iff_write_safe (access : CheckedAccess ValueId) :
    Direction.read.Safe access ↔ Direction.write.Safe access :=
  Iff.rfl

@[simp] theorem zero_bounds_safe_iff
    (value : ValueId) (length : Word) (kind : AccessKind) :
    ({ value, lengthOwner := value, index := 0, length, kind } : CheckedAccess ValueId).BoundsSafe ↔
      0 < length.toNat := by
  simp [CheckedAccess.BoundsSafe]

theorem last_bounds_safe
    (value : ValueId) (length : Word) (kind : AccessKind)
    (hPositive : 0 < length.toNat) :
    ({
      value
      lengthOwner := value
      index := BitVec.ofNat 256 (length.toNat - 1)
      length
      kind
    } : CheckedAccess ValueId).BoundsSafe := by
  change (BitVec.ofNat 256 (length.toNat - 1)).toNat < length.toNat
  rw [BitVec.toNat_ofNat]
  have hFits : length.toNat - 1 < 2 ^ 256 := by
    have hLengthFits : length.toNat < 2 ^ 256 := by
      simpa [Word, Ora.Integer.Carrier] using length.isLt
    omega
  rw [Nat.mod_eq_of_lt hFits]
  omega

theorem length_not_bounds_safe
    (value : ValueId) (length : Word) (kind : AccessKind) :
    ¬({ value, lengthOwner := value, index := length, length, kind } : CheckedAccess ValueId).BoundsSafe := by
  simp [CheckedAccess.BoundsSafe]

theorem above_not_bounds_safe
    (access : CheckedAccess ValueId)
    (hOutside : access.length.toNat ≤ access.index.toNat) :
    ¬access.BoundsSafe := by
  exact Nat.not_lt.mpr hOutside

theorem not_authorized_of_wrong_length_owner
    (access : CheckedAccess ValueId)
    (hWrong : access.value ≠ access.lengthOwner) :
    ¬access.Authorized := by
  intro hAuthorized
  exact hWrong hAuthorized.1.1

theorem byte_bounds_iff_element_bounds
    (value lengthOwner : ValueId) (index length : Word) (stride : Nat) :
    ({ value, lengthOwner, index, length, kind := .byte } : CheckedAccess ValueId).BoundsSafe ↔
      ({ value, lengthOwner, index, length, kind := .elements stride } : CheckedAccess ValueId).BoundsSafe :=
  Iff.rfl

/-- A later assumption cannot authorize an earlier unsafe access.  The theorem
uses the shared prefix-staged timeline semantics rather than a second executor
for dynamic indexing. -/
theorem unsafe_checkpoint_not_rescued_by_later_assumption
    (semantics : Ora.Loop.TimelineSemantics
      (CheckedAccess ValueId) LocationExpr ValueExpr Child)
    (access : CheckedAccess ValueId)
    (state : Ora.Loop.StorageActionState)
    (hEval : semantics.evalFormula access state = some access.Authorized)
    (hUnsafe : ¬access.Authorized) :
    ¬Ora.Loop.TimelineSafe semantics
      [.check access, .assume access] state := by
  intro hSafe
  have hRequired := hSafe [] access [.assume access] rfl
    { state } (.nil state)
  rcases hRequired with ⟨proposition, hRequiredEval, hHolds⟩
  have hSame : proposition = access.Authorized :=
    Option.some.inj (hRequiredEval.symm.trans hEval)
  exact hUnsafe (hSame ▸ hHolds)

/-- Non-wrapping address arithmetic for contiguous memory representations.
This is an explicit construction/decoder premise, not a consequence of the
bit-vector address wrapping to some value. -/
structure ContiguousLayout where
  baseAddress : Nat
  headerBytes : Nat
  strideBytes : Nat
  deriving Repr, DecidableEq

def ContiguousLayout.WellFormedFor
    (layout : ContiguousLayout) (length : Word) : Prop :=
  0 < layout.strideBytes ∧
    layout.baseAddress + layout.headerBytes + length.toNat * layout.strideBytes ≤ 2 ^ 256

def ContiguousLayout.accessEnd
    (layout : ContiguousLayout) (index : Word) : Nat :=
  layout.baseAddress + layout.headerBytes + (index.toNat + 1) * layout.strideBytes

theorem safe_access_end_fits
    (access : CheckedAccess ValueId)
    (layout : ContiguousLayout)
    (hLayout : layout.WellFormedFor access.length)
    (hStride : layout.strideBytes = access.kind.strideBytes)
    (hSafe : access.BoundsSafe) :
    layout.accessEnd access.index ≤ 2 ^ 256 := by
  unfold ContiguousLayout.WellFormedFor at hLayout
  unfold ContiguousLayout.accessEnd
  rw [hStride] at hLayout ⊢
  change access.index.toNat < access.length.toNat at hSafe
  have hIndex : access.index.toNat + 1 ≤ access.length.toNat := by omega
  have hScaled :
      (access.index.toNat + 1) * access.kind.strideBytes ≤
        access.length.toNat * access.kind.strideBytes :=
    Nat.mul_le_mul_right access.kind.strideBytes hIndex
  exact Nat.le_trans
    (Nat.add_le_add_left hScaled (layout.baseAddress + layout.headerBytes))
    hLayout.2

theorem not_wellFormedFor_of_wrapping_offset
    (layout : ContiguousLayout)
    (length : Word)
    (hWraps : 2 ^ 256 <
      layout.baseAddress + layout.headerBytes + length.toNat * layout.strideBytes) :
    ¬layout.WellFormedFor length := by
  intro hWellFormed
  exact (Nat.not_lt_of_ge hWellFormed.2) hWraps

end Ora.DynamicIndex

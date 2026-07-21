/-
Ora integer arithmetic semantics.

This module owns the width-polymorphic meaning of the checked and wrapping
integer operators.  The carrier is a bit-vector at the width selected by the
trusted `IntegerShape`; signedness changes mathematical interpretation and
right shift, but never changes the stored bits.

Checked arithmetic is modeled as a partial operation.  Add, subtract,
multiply, and power succeed exactly when their mathematical result is
representable.  Ora's checked shifts guard the shift amount only; the shifted
bit pattern is intentionally modular.  Wrapping operations are total.
-/

import Ora.Spec.Facts

namespace Ora.Integer

open Ora.Spec

abbrev Carrier (shape : IntegerShape) := BitVec shape.bits

/--
An Ora integer together with its compiler-visible width and signedness.

Keeping the shape in the value prevents a proof from equating or operating on
two integers merely because their underlying bit patterns happen to agree.
-/
structure Value where
  shape : IntegerShape
  bits : Carrier shape
  deriving Repr

def Value.ofNat (shape : IntegerShape) (value : Nat) : Value :=
  { shape, bits := BitVec.ofNat shape.bits value }

def Value.ofInt (shape : IntegerShape) (value : Int) : Value :=
  { shape, bits := BitVec.ofInt shape.bits value }

def Value.bitsFor? (value : Value) (shape : IntegerShape) : Option (Carrier shape) :=
  if h : value.shape = shape then
    some (h ▸ value.bits)
  else
    none

def Value.sameShape (lhs rhs : Value) : Bool :=
  decide (lhs.shape = rhs.shape)

def Value.eqProp? (lhs rhs : Value) : Option Prop :=
  if h : lhs.shape = rhs.shape then
    some (lhs.bits = h ▸ rhs.bits)
  else
    none

@[simp] theorem Value.eqProp?_sameShape
    (shape : IntegerShape)
    (lhs rhs : Carrier shape) :
    Value.eqProp? { shape, bits := lhs } { shape, bits := rhs } =
      some (lhs = rhs) := by
  simp [Value.eqProp?]

theorem Value.bitsFor?_self (value : Value) :
    value.bitsFor? value.shape = some value.bits := by
  simp [Value.bitsFor?]

inductive BinaryOp where
  | add
  | sub
  | mul
  | pow
  | shl
  | shr
  deriving Repr, DecidableEq

def allBinaryOps : List BinaryOp :=
  [.add, .sub, .mul, .pow, .shl, .shr]

theorem allBinaryOps_complete (op : BinaryOp) : op ∈ allBinaryOps := by
  cases op <;> decide

def unsignedValue {shape : IntegerShape} (value : Carrier shape) : Nat :=
  value.toNat

def mathematicalValue {shape : IntegerShape} (value : Carrier shape) : Int :=
  if shape.isSigned then value.toInt else value.toNat

def signedLowerBound (width : Nat) : Int :=
  -((2 : Int) ^ (width - 1))

def signedUpperBound (width : Nat) : Int :=
  (2 : Int) ^ (width - 1)

def representable (shape : IntegerShape) (value : Int) : Prop :=
  if shape.isSigned then
    signedLowerBound shape.bits ≤ value ∧ value < signedUpperBound shape.bits
  else
    0 ≤ value ∧ value < (2 : Int) ^ shape.bits

instance representableDecidable (shape : IntegerShape) (value : Int) :
    Decidable (representable shape value) := by
  unfold representable
  split <;> infer_instance

def exactArithmeticResult
    (shape : IntegerShape)
    (op : BinaryOp)
    (lhs rhs : Carrier shape) : Int :=
  match op with
  | .add => mathematicalValue lhs + mathematicalValue rhs
  | .sub => mathematicalValue lhs - mathematicalValue rhs
  | .mul => mathematicalValue lhs * mathematicalValue rhs
  -- Exponents are always the unsigned carrier bits. A signed source exponent
  -- therefore keeps its two's-complement bit pattern; signedness applies to
  -- the base and result, not to exponent interpretation.
  | .pow => mathematicalValue lhs ^ rhs.toNat
  | .shl => mathematicalValue lhs * (2 : Int) ^ rhs.toNat
  | .shr => mathematicalValue lhs >>> rhs.toNat

/--
Arithmetic right shift with Ora's shaped-integer boundary made explicit.

The EVM executes on 256-bit words, but narrow signed values are sign-extended
before `SAR`.  Consequently an amount at least the source width is already a
complete sign fill, even when the amount is below 256.
-/
def signedShiftRight
    (shape : IntegerShape)
    (lhs : Carrier shape)
    (amount : Nat) : Carrier shape :=
  if shape.bits ≤ amount then
    if lhs.msb then BitVec.allOnes shape.bits else BitVec.ofNat shape.bits 0
  else
    lhs.sshiftRight amount

def wrappingResult
    (shape : IntegerShape)
    (op : BinaryOp)
    (lhs rhs : Carrier shape) : Carrier shape :=
  match op with
  | .add => lhs + rhs
  | .sub => lhs - rhs
  | .mul => lhs * rhs
  | .pow => BitVec.ofNat shape.bits (lhs.toNat ^ rhs.toNat)
  | .shl => lhs <<< rhs.toNat
  | .shr =>
      if shape.isSigned then
        signedShiftRight shape lhs rhs.toNat
      else
        lhs >>> rhs.toNat

def unsignedLt {shape : IntegerShape} (lhs rhs : Carrier shape) : Prop :=
  lhs.toNat < rhs.toNat

def unsignedLe {shape : IntegerShape} (lhs rhs : Carrier shape) : Prop :=
  lhs.toNat ≤ rhs.toNat

def signedLt {shape : IntegerShape} (lhs rhs : Carrier shape) : Prop :=
  mathematicalValue lhs < mathematicalValue rhs

def signedLe {shape : IntegerShape} (lhs rhs : Carrier shape) : Prop :=
  mathematicalValue lhs ≤ mathematicalValue rhs

@[simp] theorem unsignedLe_refl
    {shape : IntegerShape}
    (value : Carrier shape) :
    unsignedLe value value := by
  exact Nat.le_refl _

@[simp] theorem signedLe_refl
    {shape : IntegerShape}
    (value : Carrier shape) :
    signedLe value value := by
  exact Int.le_refl _

def lt (shape : IntegerShape) (lhs rhs : Carrier shape) : Prop :=
  match shape with
  | .unsigned _ => unsignedLt lhs rhs
  | .signed _ => signedLt lhs rhs

def le (shape : IntegerShape) (lhs rhs : Carrier shape) : Prop :=
  match shape with
  | .unsigned _ => unsignedLe lhs rhs
  | .signed _ => signedLe lhs rhs

def zero (shape : IntegerShape) : Carrier shape :=
  BitVec.ofNat shape.bits 0

def minSigned (width : Ora.Types.SIntWidth) : Carrier (.signed width) :=
  BitVec.ofNat (sintBits width) (2 ^ (sintBits width - 1))

def negOne (width : Ora.Types.SIntWidth) : Carrier (.signed width) :=
  BitVec.ofNat (sintBits width) (2 ^ sintBits width - 1)

def truncDivInt (lhs rhs : Int) : Int :=
  let quotient : Nat := lhs.natAbs / rhs.natAbs
  if (lhs < 0) == (rhs < 0) then
    quotient
  else
    -((quotient : Nat) : Int)

def signedRemInt (lhs rhs : Int) : Int :=
  let remainder : Nat := lhs.natAbs % rhs.natAbs
  if lhs < 0 then
    -((remainder : Nat) : Int)
  else
    remainder

def unsignedDivTotal (shape : IntegerShape) (lhs rhs : Carrier shape) : Carrier shape :=
  if rhs = zero shape then
    zero shape
  else
    BitVec.ofNat shape.bits (lhs.toNat / rhs.toNat)

def unsignedRemTotal (shape : IntegerShape) (lhs rhs : Carrier shape) : Carrier shape :=
  if rhs = zero shape then
    zero shape
  else
    BitVec.ofNat shape.bits (lhs.toNat % rhs.toNat)

def signedDivTotal
    (width : Ora.Types.SIntWidth)
    (lhs rhs : Carrier (.signed width)) : Carrier (.signed width) :=
  if rhs = zero (.signed width) then
    zero (.signed width)
  else if lhs = minSigned width && rhs = negOne width then
    minSigned width
  else
    BitVec.ofInt (sintBits width) (truncDivInt lhs.toInt rhs.toInt)

def signedRemTotal
    (width : Ora.Types.SIntWidth)
    (lhs rhs : Carrier (.signed width)) : Carrier (.signed width) :=
  if rhs = zero (.signed width) then
    zero (.signed width)
  else if lhs = minSigned width && rhs = negOne width then
    zero (.signed width)
  else
    BitVec.ofInt (sintBits width) (signedRemInt lhs.toInt rhs.toInt)

def arithmeticResult
    (shape : IntegerShape)
    (op : BinaryOp)
    (lhs rhs : Carrier shape) : Carrier shape :=
  wrappingResult shape op lhs rhs

def divResult
    (shape : IntegerShape)
    (lhs rhs : Carrier shape) : Carrier shape :=
  match shape with
  | .unsigned width => unsignedDivTotal (.unsigned width) lhs rhs
  | .signed width => signedDivTotal width lhs rhs

def remResult
    (shape : IntegerShape)
    (lhs rhs : Carrier shape) : Carrier shape :=
  match shape with
  | .unsigned width => unsignedRemTotal (.unsigned width) lhs rhs
  | .signed width => signedRemTotal width lhs rhs

def bitAndResult
    (shape : IntegerShape)
    (lhs rhs : Carrier shape) : Carrier shape :=
  lhs &&& rhs

def bitXorResult
    (shape : IntegerShape)
    (lhs rhs : Carrier shape) : Carrier shape :=
  lhs ^^^ rhs

def Value.binary? (shape : IntegerShape)
    (op : Carrier shape → Carrier shape → Carrier shape)
    (lhs rhs : Value) : Option Value := do
  let lhsBits ← lhs.bitsFor? shape
  let rhsBits ← rhs.bitsFor? shape
  some { shape, bits := op lhsBits rhsBits }

def Value.relation? (shape : IntegerShape)
    (relation : Carrier shape → Carrier shape → Prop)
    (lhs rhs : Value) : Option Prop := do
  let lhsBits ← lhs.bitsFor? shape
  let rhsBits ← rhs.bitsFor? shape
  some (relation lhsBits rhsBits)

@[simp] theorem Value.binary_sameShape
    (shape : IntegerShape)
    (op : Carrier shape → Carrier shape → Carrier shape)
    (lhs rhs : Carrier shape) :
    Value.binary? shape op
      { shape, bits := lhs }
      { shape, bits := rhs } =
      some { shape, bits := op lhs rhs } := by
  cases shape <;> simp [Value.binary?, Value.bitsFor?]

@[simp] theorem Value.relation_sameShape
    (shape : IntegerShape)
    (relation : Carrier shape → Carrier shape → Prop)
    (lhs rhs : Carrier shape) :
    Value.relation? shape relation
      { shape, bits := lhs }
      { shape, bits := rhs } =
      some (relation lhs rhs) := by
  cases shape <;> simp [Value.relation?, Value.bitsFor?]

@[simp] theorem Value.binary_u256
    (op : Carrier (.unsigned .w256) → Carrier (.unsigned .w256) →
      Carrier (.unsigned .w256))
    (lhs rhs : Carrier (.unsigned .w256)) :
    Value.binary? (.unsigned .w256) op
      { shape := .unsigned .w256, bits := lhs }
      { shape := .unsigned .w256, bits := rhs } =
      some { shape := .unsigned .w256, bits := op lhs rhs } := by
  exact Value.binary_sameShape (.unsigned .w256) op lhs rhs

@[simp] theorem Value.relation_u256
    (relation : Carrier (.unsigned .w256) → Carrier (.unsigned .w256) → Prop)
    (lhs rhs : Carrier (.unsigned .w256)) :
    Value.relation? (.unsigned .w256) relation
      { shape := .unsigned .w256, bits := lhs }
      { shape := .unsigned .w256, bits := rhs } =
      some (relation lhs rhs) := by
  exact Value.relation_sameShape (.unsigned .w256) relation lhs rhs

def checkedSafe
    (shape : IntegerShape)
    (op : BinaryOp)
    (lhs rhs : Carrier shape) : Prop :=
  match op with
  | .add | .sub | .mul | .pow =>
      representable shape (exactArithmeticResult shape op lhs rhs)
  | .shl | .shr =>
      rhs.toNat < shape.bits

instance checkedSafeDecidable
    (shape : IntegerShape)
    (op : BinaryOp)
    (lhs rhs : Carrier shape) :
    Decidable (checkedSafe shape op lhs rhs) := by
  cases op <;> simp only [checkedSafe] <;> infer_instance

def checkedResult
    (shape : IntegerShape)
    (op : BinaryOp)
    (lhs rhs : Carrier shape) : Option (Carrier shape) :=
  if checkedSafe shape op lhs rhs then
    some (wrappingResult shape op lhs rhs)
  else
    none

theorem checkedResult_eq_some_iff
    (shape : IntegerShape)
    (op : BinaryOp)
    (lhs rhs : Carrier shape) :
    checkedResult shape op lhs rhs = some (wrappingResult shape op lhs rhs) ↔
      checkedSafe shape op lhs rhs := by
  simp [checkedResult]

theorem checkedResult_eq_none_iff
    (shape : IntegerShape)
    (op : BinaryOp)
    (lhs rhs : Carrier shape) :
    checkedResult shape op lhs rhs = none ↔
      ¬ checkedSafe shape op lhs rhs := by
  simp [checkedResult]

theorem wrapping_add_toNat
    (shape : IntegerShape)
    (lhs rhs : Carrier shape) :
    (wrappingResult shape .add lhs rhs).toNat =
      (lhs.toNat + rhs.toNat) % 2 ^ shape.bits := by
  simp [wrappingResult]

theorem wrapping_sub_toNat
    (shape : IntegerShape)
    (lhs rhs : Carrier shape) :
    (wrappingResult shape .sub lhs rhs).toNat =
      (lhs.toNat + (2 ^ shape.bits - rhs.toNat)) % 2 ^ shape.bits := by
  simp [wrappingResult, Nat.add_comm]

theorem wrapping_mul_toNat
    (shape : IntegerShape)
    (lhs rhs : Carrier shape) :
    (wrappingResult shape .mul lhs rhs).toNat =
      (lhs.toNat * rhs.toNat) % 2 ^ shape.bits := by
  simp [wrappingResult]

theorem wrapping_pow_toNat
    (shape : IntegerShape)
    (lhs rhs : Carrier shape) :
    (wrappingResult shape .pow lhs rhs).toNat =
      (lhs.toNat ^ rhs.toNat) % 2 ^ shape.bits := by
  simp [wrappingResult]

theorem wrapping_shl_toNat
    (shape : IntegerShape)
    (lhs rhs : Carrier shape) :
    (wrappingResult shape .shl lhs rhs).toNat =
      (lhs.toNat * 2 ^ rhs.toNat) % 2 ^ shape.bits := by
  simp [wrappingResult, Nat.shiftLeft_eq]

theorem wrapping_unsigned_shr_toNat
    (width : Ora.Types.UIntWidth)
    (lhs rhs : Carrier (.unsigned width)) :
    (wrappingResult (.unsigned width) .shr lhs rhs).toNat =
      lhs.toNat / 2 ^ rhs.toNat := by
  cases width <;>
    simp [wrappingResult, IntegerShape.isSigned, IntegerShape.bits,
      uintBits, Nat.shiftRight_eq_div_pow]

theorem wrapping_signed_shr_below_width_is_arithmetic
    (width : Ora.Types.SIntWidth)
    (lhs rhs : Carrier (.signed width))
    (hAmount : rhs.toNat < sintBits width) :
    wrappingResult (.signed width) .shr lhs rhs =
      BitVec.ofInt (sintBits width) (lhs.toInt >>> rhs.toNat) := by
  cases width <;>
    simp [sintBits] at hAmount <;>
    simp only [wrappingResult, IntegerShape.isSigned, if_true,
      IntegerShape.bits, sintBits] <;>
    unfold signedShiftRight <;>
    split <;>
    simp [IntegerShape.bits] at * <;>
    first | omega | rfl

theorem checked_arithmetic_safe_iff_representable
    (shape : IntegerShape)
    (op : BinaryOp)
    (hop : op = .add ∨ op = .sub ∨ op = .mul ∨ op = .pow)
    (lhs rhs : Carrier shape) :
    checkedSafe shape op lhs rhs ↔
      representable shape (exactArithmeticResult shape op lhs rhs) := by
  rcases hop with rfl | rfl | rfl | rfl <;> rfl

theorem checked_shift_safe_iff
    (shape : IntegerShape)
    (op : BinaryOp)
    (hop : op = .shl ∨ op = .shr)
    (lhs rhs : Carrier shape) :
    checkedSafe shape op lhs rhs ↔ rhs.toNat < shape.bits := by
  rcases hop with rfl | rfl <;> rfl

theorem checked_shift_rejects_width
    (shape : IntegerShape)
    (op : BinaryOp)
    (hop : op = .shl ∨ op = .shr)
    (lhs : Carrier shape) :
    checkedResult shape op lhs (BitVec.ofNat shape.bits shape.bits) = none := by
  apply (checkedResult_eq_none_iff shape op lhs
    (BitVec.ofNat shape.bits shape.bits)).mpr
  rw [checked_shift_safe_iff shape op hop]
  simp

theorem wrapping_unsigned_shifts_by_width_are_zero
    (width : Ora.Types.UIntWidth)
    (lhs : Carrier (.unsigned width)) :
    wrappingResult (.unsigned width) .shl lhs
        (BitVec.ofNat (uintBits width) (uintBits width)) =
        BitVec.ofNat (uintBits width) 0 ∧
      wrappingResult (.unsigned width) .shr lhs
        (BitVec.ofNat (uintBits width) (uintBits width)) =
        BitVec.ofNat (uintBits width) 0 := by
  cases width <;> constructor <;>
    apply BitVec.eq_of_getLsbD_eq <;>
    intro i <;>
    simp [wrappingResult, IntegerShape.isSigned, IntegerShape.bits, uintBits,
      BitVec.getLsbD_shiftLeft, BitVec.getLsbD_ushiftRight]

theorem wrapping_signed_shr_by_width_is_sign_fill
    (width : Ora.Types.SIntWidth)
    (lhs : Carrier (.signed width)) :
    wrappingResult (.signed width) .shr lhs
        (BitVec.ofNat (sintBits width) (sintBits width)) =
      if lhs.msb then BitVec.allOnes (sintBits width)
      else BitVec.ofNat (sintBits width) 0 := by
  cases width <;>
    simp [wrappingResult, signedShiftRight, sintBits]

theorem unsigned_and_signed_right_shift_are_distinct :
    wrappingResult (.unsigned .w8) .shr (BitVec.ofNat 8 248) (BitVec.ofNat 8 8) =
        BitVec.ofNat 8 0 ∧
      wrappingResult (.signed .w8) .shr (BitVec.ofNat 8 248) (BitVec.ofNat 8 8) =
        BitVec.ofNat 8 255 := by
  decide

theorem wrapping_u8_add_example :
    wrappingResult (.unsigned .w8) .add (BitVec.ofNat 8 255) (BitVec.ofNat 8 1) =
      BitVec.ofNat 8 0 := by
  decide

theorem wrapping_u160_add_example :
    wrappingResult (.unsigned .w160) .add
        (BitVec.ofNat 160 (2 ^ 160 - 1)) (BitVec.ofNat 160 1) =
      BitVec.ofNat 160 0 := by
  decide

theorem wrapping_i8_mul_example :
    wrappingResult (.signed .w8) .mul (BitVec.ofInt 8 (-16)) (BitVec.ofNat 8 16) =
      BitVec.ofNat 8 0 := by
  decide

theorem wrapping_u8_pow_example :
    wrappingResult (.unsigned .w8) .pow (BitVec.ofNat 8 16) (BitVec.ofNat 8 2) =
      BitVec.ofNat 8 0 := by
  decide

theorem signed_power_exponent_uses_unsigned_carrier_bits :
    wrappingResult (.signed .w8) .pow
        (BitVec.ofInt 8 (-2)) (BitVec.ofInt 8 (-1)) =
      BitVec.ofNat 8 ((254 : Nat) ^ 255) := by
  decide

theorem checked_u8_add_rejects_overflow :
    checkedResult (.unsigned .w8) .add (BitVec.ofNat 8 255) (BitVec.ofNat 8 1) = none := by
  decide

theorem checked_i8_add_accepts_representable :
    checkedResult (.signed .w8) .add (BitVec.ofInt 8 (-5)) (BitVec.ofNat 8 3) =
      some (BitVec.ofInt 8 (-2)) := by
  decide

end Ora.Integer

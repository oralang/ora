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
  deriving Repr, DecidableEq

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

-- Restore a widened addition before observing its source-width overflow.
theorem maskBit (n i : Nat) : Nat.testBit (2^n - 1) i = decide (i < n) := by
  induction n generalizing i with
  | zero => simp only [Nat.pow_zero, Nat.sub_self, Nat.zero_testBit, Nat.not_lt_zero, decide_false]
  | succ n ih =>
    have pos : 0 < 2^n := Nat.two_pow_pos n
    cases i with
    | zero =>
      simp only [Nat.testBit_zero, Nat.pow_succ]
      have hmod : (2^n * 2 - 1) % 2 = 1 := by omega
      simp only [hmod, decide_true, Nat.zero_lt_succ]
    | succ i =>
      rw [Nat.testBit_succ]
      have hdiv : (2^(n+1)-1) / 2 = 2^n-1 := by rw [Nat.pow_succ]; omega
      rw [hdiv, ih]
      simp only [Nat.succ_lt_succ_iff]

theorem lowMaskIsWidth (word : BitVec 256) (n : Nat) :
    word.toNat &&& (2^n-1) = (word.setWidth n).toNat := by
  simp only [BitVec.toNat_setWidth]
  apply Nat.eq_of_testBit_eq
  intro i
  simp only [Nat.testBit_and, Nat.testBit_mod_two_pow, maskBit]
  exact Bool.and_comm _ _

theorem truncateWideAdd (n : Nat) (h : n ≤ 256) (a b : BitVec n) :
    (a.setWidth 256 + b.setWidth 256).setWidth n = a + b := by
  rw [BitVec.setWidth_add _ _ h]
  simp only [BitVec.setWidth_setWidth_of_le _ h, BitVec.setWidth_eq]

theorem maskedWideAdd (n : Nat) (h : n ≤ 256) (a b : BitVec n) :
    (a.setWidth 256 + b.setWidth 256).toNat &&& (2^n - 1) = (a+b).toNat := by
  rw [lowMaskIsWidth, truncateWideAdd n h]

theorem wrappedUnsignedAddOverflow {n : Nat} (a b : BitVec n) :
    (a+b).toNat < a.toNat ↔ 2^n ≤ a.toNat + b.toNat := by
  have ha := a.isLt
  have hb := b.isLt
  simp only [BitVec.toNat_add]
  cases Nat.lt_or_ge (a.toNat+b.toNat) (2^n) with
  | inl h =>
    rw [Nat.mod_eq_of_lt h]
    constructor <;> intro hbad <;> omega
  | inr h =>
    rw [Nat.mod_eq_sub_mod h]
    rw [Nat.mod_eq_of_lt (by omega)]
    constructor <;> intro hbound <;> omega

-- Checked unsigned multiplication guard; divisor orientation does not change overflow.
-- The nonzero factor masks either EVM or SMT division-by-zero totalization.
theorem unsignedMulOverflow_iff (M a b : Nat) :
    (b ≠ 0 ∧ M / b < a) ↔ M < a * b := by
  by_cases hb : b = 0
  · simp [hb]
  · simp [hb, Nat.div_lt_iff_lt_mul (Nat.pos_of_ne_zero hb)]

theorem unsignedMulOverflow_comm (M a b : Nat) :
    (b ≠ 0 ∧ M / b < a) ↔ (a ≠ 0 ∧ M / a < b) := by
  rw [unsignedMulOverflow_iff, unsignedMulOverflow_iff, Nat.mul_comm a b]

theorem unsignedMulOverflow_bits {w : Nat} (a b : BitVec w) :
    ((b != 0#w) && (BitVec.allOnes w / b).ult a) =
      decide (2 ^ w - 1 < a.toNat * b.toNat) := by
  apply Bool.eq_iff_iff.mpr
  simp only [Bool.and_eq_true, bne_iff_ne, BitVec.toNat_ne,
    BitVec.toNat_ofNat, Nat.zero_mod, BitVec.ult, decide_eq_true_eq,
    BitVec.toNat_udiv, BitVec.toNat_allOnes]
  exact unsignedMulOverflow_iff _ _ _

theorem unsignedMulOverflow_bits_comm {w : Nat} (a b : BitVec w) :
    ((b != 0#w) && (BitVec.allOnes w / b).ult a) =
      ((a != 0#w) && (BitVec.allOnes w / a).ult b) := by
  rw [unsignedMulOverflow_bits, unsignedMulOverflow_bits, Nat.mul_comm a.toNat b.toNat]

theorem unsignedMulOverflow_zero_quotient_irrelevant {w : Nat} (a b onZero : BitVec w) :
    ((b != 0#w) && (if b = 0#w then onZero else BitVec.allOnes w / b).ult a) =
      ((b != 0#w) && (BitVec.allOnes w / b).ult a) := by
  by_cases hb : b = 0#w <;> simp [hb]

def unsignedLt {shape : IntegerShape} (lhs rhs : Carrier shape) : Prop :=
  lhs.toNat < rhs.toNat

def unsignedLe {shape : IntegerShape} (lhs rhs : Carrier shape) : Prop :=
  lhs.toNat ≤ rhs.toNat

def signedLt {shape : IntegerShape} (lhs rhs : Carrier shape) : Prop :=
  lhs.slt rhs = true

def signedLe {shape : IntegerShape} (lhs rhs : Carrier shape) : Prop :=
  lhs.sle rhs = true

theorem signedLt_iff_toInt_lt
    {shape : IntegerShape}
    (lhs rhs : Carrier shape) :
    signedLt lhs rhs ↔ lhs.toInt < rhs.toInt := by
  simp [signedLt, BitVec.slt]

theorem signedLe_iff_toInt_le
    {shape : IntegerShape}
    (lhs rhs : Carrier shape) :
    signedLe lhs rhs ↔ lhs.toInt ≤ rhs.toInt := by
  simp [signedLe, BitVec.sle]

@[simp] theorem unsignedLe_refl
    {shape : IntegerShape}
    (value : Carrier shape) :
    unsignedLe value value :=
  Nat.le_refl _

@[simp] theorem signedLe_refl
    {shape : IntegerShape}
    (value : Carrier shape) :
    signedLe value value := by
  simp [signedLe, BitVec.sle]

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
      some { shape := .unsigned .w256, bits := op lhs rhs } :=
  Value.binary_sameShape (.unsigned .w256) op lhs rhs

@[simp] theorem Value.relation_u256
    (relation : Carrier (.unsigned .w256) → Carrier (.unsigned .w256) → Prop)
    (lhs rhs : Carrier (.unsigned .w256)) :
    Value.relation? (.unsigned .w256) relation
      { shape := .unsigned .w256, bits := lhs }
      { shape := .unsigned .w256, bits := rhs } =
      some (relation lhs rhs) :=
  Value.relation_sameShape (.unsigned .w256) relation lhs rhs

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

theorem bmod_eq_self_of_signed_bounds
    (value : Int)
    (half : Nat)
    (hHalf : 0 < half)
    (hLower : -((half : Nat) : Int) ≤ value)
    (hUpper : value < (half : Nat)) :
    Int.bmod value (2 * half) = value := by
  rw [Int.bmod_def]
  have hModulus : (0 : Int) < 2 * half := by omega
  by_cases hNonnegative : 0 ≤ value
  · rw [Int.emod_eq_of_lt hNonnegative (by omega)]
    split <;> omega
  · have hMod : value % (2 * half : Nat) = value + 2 * half := by
      rw [← Int.add_emod_self]
      exact Int.emod_eq_of_lt (by omega) (by omega)
    rw [hMod]
    split <;> omega

theorem signed_add_toInt_eq_of_representable
    (width : Ora.Types.SIntWidth)
    (lhs rhs : Carrier (.signed width))
    (hSafe : representable (.signed width) (lhs.toInt + rhs.toInt)) :
    (lhs + rhs).toInt = lhs.toInt + rhs.toInt := by
  rw [BitVec.toInt_add]
  have hWidthPositive : 0 < sintBits width := by
    cases width <;> decide
  have hModulus :
      2 ^ (IntegerShape.signed width).bits =
        2 * 2 ^ (sintBits width - 1) := by
    simp only [IntegerShape.bits]
    rw [← Nat.two_pow_pred_mul_two hWidthPositive, Nat.mul_comm]
  rw [hModulus]
  apply bmod_eq_self_of_signed_bounds
      (lhs.toInt + rhs.toInt) (2 ^ (sintBits width - 1))
  · exact Nat.two_pow_pos _
  · simp only [representable, IntegerShape.isSigned, if_true,
      signedLowerBound, signedUpperBound, IntegerShape.bits] at hSafe
    exact_mod_cast hSafe.1
  · simp only [representable, IntegerShape.isSigned, if_true,
      signedLowerBound, signedUpperBound, IntegerShape.bits] at hSafe
    exact_mod_cast hSafe.2

theorem toInt_nonnegative_iff_msb_false
    {width : Nat}
    (hWidth : 0 < width)
    (value : BitVec width) :
    0 ≤ value.toInt ↔ value.msb = false := by
  have hValue := value.isLt
  have hPow : 2 ^ width = 2 * 2 ^ (width - 1) := by
    rw [← Nat.two_pow_pred_mul_two hWidth, Nat.mul_comm]
  rw [BitVec.toInt_eq_toNat_cond, BitVec.msb_eq_decide]
  split
  · constructor
    · intro _
      apply decide_eq_false
      omega
    · intro _
      exact Int.ofNat_zero_le _
  · constructor
    · intro hNonnegative
      apply decide_eq_false
      omega
    · intro hDecision
      have hBelow := of_decide_eq_false hDecision
      omega

theorem signed_add_overflow_bit_nonnegative_of_checkedSafe
    (width : Ora.Types.SIntWidth)
    (lhs rhs : Carrier (.signed width)) :
    checkedSafe (.signed width) .add lhs rhs →
    ¬ signedLt
      (bitAndResult (.signed width)
        (bitXorResult (.signed width)
          (wrappingResult (.signed width) .add lhs rhs) lhs)
        (bitXorResult (.signed width)
          (wrappingResult (.signed width) .add lhs rhs) rhs))
      (BitVec.ofNat (sintBits width) 0) := by
  intro hSafe
  have hRepresentable :
      representable (.signed width) (lhs.toInt + rhs.toInt) := by
    exact hSafe
  have hExact := signed_add_toInt_eq_of_representable width lhs rhs hRepresentable
  rw [signedLt_iff_toInt_lt]
  rw [show (BitVec.ofNat (sintBits width) 0).toInt = 0 by
    cases width <;> decide]
  simp only [Int.not_lt]
  have hWidth : 0 < sintBits width := by cases width <;> decide
  apply (toInt_nonnegative_iff_msb_false hWidth _).mpr
  simp only [bitAndResult, bitXorResult, wrappingResult, BitVec.msb_and,
    BitVec.msb_xor]
  have hLhsSign := toInt_nonnegative_iff_msb_false hWidth lhs
  have hRhsSign := toInt_nonnegative_iff_msb_false hWidth rhs
  have hResultSign := toInt_nonnegative_iff_msb_false hWidth (lhs + rhs)
  rw [hExact] at hResultSign
  cases hLhs : lhs.msb <;>
    cases hRhs : rhs.msb <;>
    cases hResult : (lhs + rhs).msb <;>
    simp only [hLhs, hRhs, hResult, Bool.false_xor, Bool.xor_false,
      Bool.true_xor, Bool.xor_true, Bool.false_and, Bool.true_and] <;>
    simp [hLhs] at hLhsSign <;>
    simp [hRhs] at hRhsSign <;>
    simp [hResult] at hResultSign <;>
    first | decide | omega

theorem toInt_lt_signedUpperBound
    (width : Ora.Types.SIntWidth)
    (value : Carrier (.signed width)) :
    value.toInt < signedUpperBound (sintBits width) := by
  cases width <;>
    rw [BitVec.toInt_eq_toNat_cond] <;>
    split <;>
    simp only [signedUpperBound, IntegerShape.bits, sintBits] at * <;>
    have hValue := value.isLt <;>
    omega

/-- A nonnegative signed addend is safe when the emitted arithmetic facts bound
the exact result by another value of the same signed type.  The `signedLt`
hypothesis identifies that formula-owned bound; the proof does not recognize a
particular increment or literal. -/
theorem signed_add_checkedSafe_of_nonnegative_bounded_result
    (width : Ora.Types.SIntWidth)
    (lhs rhs bound : Carrier (.signed width))
    (hRhs : 0 ≤ rhs.toInt)
    (hLt : signedLt lhs bound)
    (hResultBound : lhs.toInt + rhs.toInt ≤ bound.toInt) :
    checkedSafe (.signed width) .add lhs rhs := by
  simp only [checkedSafe, representable, IntegerShape.isSigned, if_true,
    exactArithmeticResult, mathematicalValue, IntegerShape.bits]
  constructor
  · have hLower := BitVec.toInt_intMin_le lhs
    change (BitVec.intMin (sintBits width)).toInt ≤ lhs.toInt at hLower
    have hMin : (BitVec.intMin (sintBits width)).toInt =
        signedLowerBound (sintBits width) := by
      have hWidth : 0 < sintBits width := by cases width <;> decide
      simp only [BitVec.toInt_intMin, signedLowerBound]
      rw [Nat.two_pow_pred_mod_two_pow hWidth]
      congr 1
      exact_mod_cast rfl
    rw [hMin] at hLower
    omega
  · have hUpper := toInt_lt_signedUpperBound width bound
    rw [signedLt_iff_toInt_lt] at hLt
    omega

/-- Under checked-add safety, the wrapped carrier denotes the exact signed
sum.  This is the generic bridge used when a formula requires an order fact
about the emitted next-state expression. -/
theorem signed_add_le_of_checkedSafe
    (width : Ora.Types.SIntWidth)
    (lhs rhs bound : Carrier (.signed width))
    (hSafe : checkedSafe (.signed width) .add lhs rhs)
    (hLe : lhs.toInt + rhs.toInt ≤ bound.toInt) :
    signedLe (wrappingResult (.signed width) .add lhs rhs) bound := by
  rw [signedLe_iff_toInt_le]
  simp only [wrappingResult]
  rw [signed_add_toInt_eq_of_representable width lhs rhs hSafe]
  exact hLe

theorem signed_add_one_checkedSafe_of_lt
    (width : Ora.Types.SIntWidth)
    (value bound : Carrier (.signed width))
    (h : signedLt value bound) :
    checkedSafe (.signed width) .add value
      (BitVec.ofNat (sintBits width) 1) := by
  rw [signedLt_iff_toInt_lt] at h
  have hLower := BitVec.toInt_intMin_le value
  have hUpper := toInt_lt_signedUpperBound width bound
  simp only [checkedSafe, representable, IntegerShape.isSigned, if_true,
    exactArithmeticResult, mathematicalValue, IntegerShape.bits]
  have hOne : (BitVec.ofNat (sintBits width) 1).toInt = 1 := by
    cases width <;> decide
  rw [hOne]
  constructor
  · change (BitVec.intMin (sintBits width)).toInt ≤ value.toInt at hLower
    have hMin : (BitVec.intMin (sintBits width)).toInt =
        signedLowerBound (sintBits width) := by
      have hWidth : 0 < sintBits width := by cases width <;> decide
      simp only [BitVec.toInt_intMin, signedLowerBound]
      rw [Nat.two_pow_pred_mod_two_pow hWidth]
      congr 1
      exact_mod_cast rfl
    rw [hMin] at hLower
    omega
  · omega

theorem wrapping_add_toNat
    (shape : IntegerShape)
    (lhs rhs : Carrier shape) :
    (wrappingResult shape .add lhs rhs).toNat =
      (lhs.toNat + rhs.toNat) % 2 ^ shape.bits := by
  simp [wrappingResult]

theorem wrapping_add_one_le_of_unsigned_lt
    (shape : IntegerShape)
    (value bound : Carrier shape)
    (hBound : unsignedLt value bound) :
    unsignedLe
      (wrappingResult shape .add value (BitVec.ofNat shape.bits 1))
      bound := by
  cases shape with
  | unsigned width =>
      cases width <;> simp [unsignedLt, unsignedLe, wrappingResult] at * <;> omega
  | signed width =>
      cases width <;> simp [unsignedLt, unsignedLe, wrappingResult] at * <;> omega

theorem wrapping_add_one_not_lt_self_of_unsigned_lt
    (shape : IntegerShape)
    (value bound : Carrier shape)
    (hBound : unsignedLt value bound) :
    ¬ unsignedLt
      (wrappingResult shape .add value (BitVec.ofNat shape.bits 1))
      value := by
  cases shape with
  | unsigned width =>
      cases width <;> simp [unsignedLt, wrappingResult] at * <;> omega
  | signed width =>
      cases width <;> simp [unsignedLt, wrappingResult] at * <;> omega

theorem wrapping_add_one_not_lt_self_of_room
    (shape : IntegerShape)
    (value : Carrier shape)
    (hRoom : value.toNat + 1 < 2 ^ shape.bits) :
    ¬ unsignedLt
      (wrappingResult shape .add value (BitVec.ofNat shape.bits 1))
      value := by
  cases shape with
  | unsigned width =>
      cases width <;> simp [unsignedLt, wrappingResult] at * <;> omega
  | signed width =>
      cases width <;> simp [unsignedLt, wrappingResult] at * <;> omega

/--
Adding one to a signed integer strictly below another signed value preserves
the corresponding non-strict bound. The proof is uniform over Ora's complete
registered signed-width universe; it is not a loop-specific fact.
-/
theorem signed_add_one_le_of_lt
    (width : Ora.Types.SIntWidth)
    (value bound : Carrier (.signed width))
    (h : signedLt value bound) :
    signedLe
      (wrappingResult (.signed width) .add value
        (BitVec.ofNat (sintBits width) 1))
      bound := by
  have hSafe := signed_add_one_checkedSafe_of_lt width value bound h
  have hExact := signed_add_toInt_eq_of_representable width value
    (BitVec.ofNat (sintBits width) 1) hSafe
  rw [signedLt_iff_toInt_lt] at h
  rw [signedLe_iff_toInt_le]
  simp only [wrappingResult]
  rw [hExact]
  cases width <;> simp at * <;> omega

/--
The source-owned signed-add overflow predicate is false when adding one below
a strictly greater signed bound. This connects the shared arithmetic fact to
the explicit runtime safety formula without giving proof-driven erasure any
authority.
-/
theorem signed_add_one_guard_safe_of_lt
    (width : Ora.Types.SIntWidth)
    (value bound : Carrier (.signed width))
    (h : signedLt value bound) :
    ¬ signedLt
      (bitAndResult (.signed width)
        (bitXorResult (.signed width)
          (wrappingResult (.signed width) .add value
            (BitVec.ofNat (sintBits width) 1))
          value)
        (bitXorResult (.signed width)
          (wrappingResult (.signed width) .add value
            (BitVec.ofNat (sintBits width) 1))
          (BitVec.ofNat (sintBits width) 1)))
      (BitVec.ofNat (sintBits width) 0) := by
  apply signed_add_overflow_bit_nonnegative_of_checkedSafe
  exact signed_add_one_checkedSafe_of_lt width value bound h

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

/-- Unsigned zero is an exact total lower bound, independently of a value's producer. -/
theorem unsigned_zero_lower_bound (width : Nat) (value : BitVec width) :
    (0 : BitVec width) ≤ value := by
  change (0 : BitVec width).toNat ≤ value.toNat
  simp

/-- The rewrite supplies a genuine value-level proposition, not a loop result. -/
theorem unsigned_zero_predicate (width : Nat) (value : BitVec width) :
    decide ((0 : BitVec width) ≤ value) = true := by
  exact decide_eq_true (unsigned_zero_lower_bound width value)

end Ora.Integer

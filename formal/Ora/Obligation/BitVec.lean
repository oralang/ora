/-
Unsigned bit-vector facts used by userland obligation proofs.

This module is intentionally small: it defines the unsigned interpretation Ora
needs for the first Lean-backed UNKNOWN slice without depending on mathlib or on
solver-specific SMT syntax. More bit-vector operations should be added here only
when the compiler emits obligations that need them.
-/

namespace Ora.Obligation

abbrev U256 := BitVec 256

namespace U256

def ult (lhs rhs : U256) : Prop :=
  lhs.toNat < rhs.toNat

def ule (lhs rhs : U256) : Prop :=
  lhs.toNat ≤ rhs.toNat

def ugt (lhs rhs : U256) : Prop :=
  rhs.ult lhs

def uge (lhs rhs : U256) : Prop :=
  rhs.ule lhs

def toInt256 (value : U256) : Int :=
  if value.toNat < 2^255 then
    value.toNat
  else
    (value.toNat : Int) - 2^256

def slt (lhs rhs : U256) : Prop :=
  lhs.toInt256 < rhs.toInt256

def sle (lhs rhs : U256) : Prop :=
  lhs.toInt256 ≤ rhs.toInt256

def sgt (lhs rhs : U256) : Prop :=
  rhs.slt lhs

def sge (lhs rhs : U256) : Prop :=
  rhs.sle lhs

def max : U256 :=
  BitVec.ofNat 256 (2^256 - 1)

theorem slt_max_zero_and_not_ult :
    slt max (BitVec.ofNat 256 0) ∧ ¬ ult max (BitVec.ofNat 256 0) := by
  constructor
  · unfold slt toInt256 max
    have hMaxLt : 2^256 - 1 < 2^256 := by omega
    have hMaxHigh : ¬ (BitVec.ofNat 256 (2^256 - 1)).toNat < 2^255 := by
      simp [Nat.mod_eq_of_lt hMaxLt]
    simp [Nat.mod_eq_of_lt hMaxLt, hMaxHigh]
  · unfold ult max
    have hMaxLt : 2^256 - 1 < 2^256 := by omega
    simp [Nat.mod_eq_of_lt hMaxLt]

def add (lhs rhs : U256) : U256 :=
  BitVec.ofNat 256 (lhs.toNat + rhs.toNat)

theorem toNat_add_noOverflow (lhs rhs : U256) :
    lhs.toNat + rhs.toNat < 2^256 →
      (lhs.add rhs).toNat = lhs.toNat + rhs.toNat := by
  intro h
  unfold add
  simp [Nat.mod_eq_of_lt h]

def sub (lhs rhs : U256) : U256 :=
  BitVec.ofNat 256 (lhs.toNat + 2^256 - rhs.toNat)

theorem toNat_sub_noUnderflow (lhs rhs : U256) :
    rhs.toNat ≤ lhs.toNat →
      (lhs.sub rhs).toNat + rhs.toNat = lhs.toNat := by
  intro h
  unfold sub
  have hDiffLt : lhs.toNat - rhs.toNat < 2^256 := by omega
  have hRewrite :
      lhs.toNat + 2^256 - rhs.toNat = 2^256 + (lhs.toNat - rhs.toNat) := by
    omega
  rw [hRewrite]
  simp [Nat.mod_eq_of_lt hDiffLt]
  omega

theorem sub_underflow_wrap_example :
    sub (BitVec.ofNat 256 3) (BitVec.ofNat 256 5) =
      BitVec.ofNat 256 (2^256 - 2) := by
  decide

def mul (lhs rhs : U256) : U256 :=
  BitVec.ofNat 256 (lhs.toNat * rhs.toNat)

def zero : U256 :=
  BitVec.ofNat 256 0

def minSigned : U256 :=
  BitVec.ofNat 256 (2^255)

def negOne : U256 :=
  BitVec.ofNat 256 (2^256 - 1)

def modulusInt : Int :=
  (2 : Int)^256

def ofInt256 (value : Int) : U256 :=
  BitVec.ofNat 256 ((value % modulusInt).toNat)

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

def udivTotal (lhs rhs : U256) : U256 :=
  if rhs = zero then
    zero
  else
    BitVec.ofNat 256 (lhs.toNat / rhs.toNat)

def uremTotal (lhs rhs : U256) : U256 :=
  if rhs = zero then
    zero
  else
    BitVec.ofNat 256 (lhs.toNat % rhs.toNat)

def sdivTotal (lhs rhs : U256) : U256 :=
  if rhs = zero then
    zero
  else if lhs = minSigned && rhs = negOne then
    minSigned
  else
    ofInt256 (truncDivInt lhs.toInt256 rhs.toInt256)

def sremTotal (lhs rhs : U256) : U256 :=
  if rhs = zero then
    zero
  else if lhs = minSigned && rhs = negOne then
    zero
  else
    ofInt256 (signedRemInt lhs.toInt256 rhs.toInt256)

theorem udiv_zero_divisor_is_zero (lhs : U256) :
    lhs.udivTotal zero = zero := by
  simp [udivTotal]

theorem urem_zero_divisor_is_zero (lhs : U256) :
    lhs.uremTotal zero = zero := by
  simp [uremTotal]

theorem sdiv_min_neg_one_is_min :
    sdivTotal minSigned negOne = minSigned := by
  decide

theorem srem_min_neg_one_is_zero :
    sremTotal minSigned negOne = zero := by
  decide

set_option maxRecDepth 100000 in
theorem srem_dividend_sign_examples :
    sremTotal (BitVec.ofNat 256 (2^256 - 10)) (BitVec.ofNat 256 3) =
        BitVec.ofNat 256 (2^256 - 1) ∧
      sremTotal (BitVec.ofNat 256 10) (BitVec.ofNat 256 (2^256 - 3)) =
        BitVec.ofNat 256 1 := by
  decide

theorem ult_implies_ule (lhs rhs : U256) :
    lhs.ult rhs → lhs.ule rhs := by
  intro h
  exact Nat.le_of_lt h

theorem lt_max_succ_ule (i x : U256) :
    x.ult max → i.ult x → i.ule (x.add (BitVec.ofNat 256 1)) := by
  intro hx hi
  unfold ult ule add max at *
  have hlt : x.toNat + 1 < 2^256 := by omega
  simp [Nat.mod_eq_of_lt hlt]
  omega

theorem lt_bound_add_ule (i x bound step : U256) :
    bound.toNat + step.toNat ≤ 2^256 →
    x.ult bound →
    i.ult x →
    i.ule (x.add step) := by
  intro hSafe hx hi
  unfold ult ule add at *
  have hlt : x.toNat + step.toNat < 2^256 := by omega
  simp [Nat.mod_eq_of_lt hlt]
  omega

theorem lt_max_sub_add_ule (i x step : U256) :
    x.ult (max.sub step) →
    i.ult x →
    i.ule (x.add step) := by
  intro hx hi
  unfold ult ule add sub max at *
  have hMax :
      (BitVec.ofNat 256 (2^256 - 1)).toNat = 2^256 - 1 := by
    have hMaxLt : 2^256 - 1 < 2^256 := by omega
    simp [Nat.mod_eq_of_lt hMaxLt]
  have hStepLeMax : step.toNat ≤ 2^256 - 1 := by omega
  have hBoundLt : 2^256 - 1 - step.toNat < 2^256 := by omega
  have hBoundWrapped :
      2^256 - 1 + 2^256 - step.toNat =
        2^256 + (2^256 - 1 - step.toNat) := by
    omega
  have hBound :
      (BitVec.ofNat 256 (2^256 - 1 + 2^256 - step.toNat)).toNat =
        2^256 - 1 - step.toNat := by
    rw [hBoundWrapped]
    simp [Nat.mod_eq_of_lt hBoundLt]
  rw [hMax] at hx
  rw [hBound] at hx
  have hAddLt : x.toNat + step.toNat < 2^256 := by omega
  simp [Nat.mod_eq_of_lt hAddLt]
  omega

end U256

end Ora.Obligation

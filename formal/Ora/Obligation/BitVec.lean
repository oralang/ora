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

def max : U256 :=
  BitVec.ofNat 256 (2^256 - 1)

def add (lhs rhs : U256) : U256 :=
  BitVec.ofNat 256 (lhs.toNat + rhs.toNat)

def sub (lhs rhs : U256) : U256 :=
  BitVec.ofNat 256 (lhs.toNat - rhs.toNat)

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
  have hBound :
      (BitVec.ofNat 256 (2^256 - 1 - step.toNat)).toNat =
        2^256 - 1 - step.toNat := by
    simp [Nat.mod_eq_of_lt hBoundLt]
  rw [hMax] at hx
  rw [hBound] at hx
  have hAddLt : x.toNat + step.toNat < 2^256 := by omega
  simp [Nat.mod_eq_of_lt hAddLt]
  omega

end U256

end Ora.Obligation

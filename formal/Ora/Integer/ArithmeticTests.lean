import Ora.Integer.Arithmetic

namespace Ora.Integer

#print axioms maskBit
#print axioms lowMaskIsWidth
#print axioms truncateWideAdd
#print axioms maskedWideAdd
#print axioms wrappedUnsignedAddOverflow

#print axioms unsignedMulOverflow_iff
#print axioms unsignedMulOverflow_comm
#print axioms unsignedMulOverflow_bits
#print axioms unsignedMulOverflow_bits_comm
#print axioms unsignedMulOverflow_zero_quotient_irrelevant

-- Tests include zero on either side, no overflow at MAX, and overflow just beyond.
example (M a : Nat) : ¬ (0 ≠ 0 ∧ M / 0 < a) := by simp
example (M b : Nat) : ¬ (b ≠ 0 ∧ M / b < 0) := by simp
example : (decide (100 ≠ 0 ∧ (2^256 - 1) / 100 < (2^256 - 1) / 100)) = false := by decide
example : (decide (100 ≠ 0 ∧ (2^256 - 1) / 100 < (2^256 - 1) / 100 + 1)) = true := by decide
example : ((100#256 != 0#256) && (BitVec.allOnes 256 / 100#256).ult
    (BitVec.ofNat 256 ((2^256 - 1) / 100))) = false := by decide
example : ((100#256 != 0#256) && (BitVec.allOnes 256 / 100#256).ult
    (BitVec.ofNat 256 ((2^256 - 1) / 100 + 1))) = true := by decide
example (a : BitVec 256) : ((0#256 != 0#256) && (BitVec.allOnes 256).ult a) = false := by simp
example (a : BitVec 256) : ((0#256 != 0#256) && (0#256).ult a) = false := by simp
example : ((1#8 != 0#8) && (BitVec.allOnes 8 / 1#8).ult 255#8) = false := by decide
example : ((2#8 != 0#8) && (BitVec.allOnes 8 / 2#8).ult 128#8) = true := by decide
example (a b : BitVec 0) : ((b != 0#0) && (BitVec.allOnes 0 / b).ult a) = false := by
  rw [unsignedMulOverflow_bits]
  simp [BitVec.of_length_zero]

-- Removing or misattaching the nonzero guard is observably incorrect.
example : ¬ ((0 ≠ 0 ∧ 255 / 0 < 1) ↔ 255 / 0 < 1) := by decide
example : ¬ ((0 ≠ 0 ∧ 255 / 0 < 1) ↔ (1 ≠ 0 ∧ 255 / 0 < 1)) := by decide
end Ora.Integer

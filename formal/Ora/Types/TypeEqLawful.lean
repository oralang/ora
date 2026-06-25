/-
Ora type system — lawfulness of structural equality (`Ty.beq`).

Reflexivity of `Ty.beq` (and onward toward `beq ↔ =`, i.e. `DecidableEq Ty`).

ISOLATED ON PURPOSE: this module is NOT imported by the `Ora` root, so the
default `lake build` and `check-formal-sync.sh` do NOT compile it. The proofs
unfold the mutual `Ty.beq` definition, whose `whnf` is expensive (a single full
build of this file costs ~40s), and that cost should not tax every gate run.

It is still a real, kernel-checked proof. Build/verify it on demand:

    lake build Ora.Types.TypeEqLawful

(and it should be built in CI alongside the gate). `induction t` is rejected on
the nested `List Ty`, so these are equation-style mutual structural recursions
mirroring `Ty.beq`'s own shape; the `maxHeartbeats` bump is for the slow (finite)
`whnf`, not a loop.
-/

import Ora.Types.TypeEq

namespace Ora.Types

-- NOTE: `unfold`, `simp [Ty.beq]`, and `simp only [Ty.beq]` were all tried and
-- ALL time out at `whnf` — the cost is intrinsic to unfolding the mutual def's
-- compiled `brecOn`, not the tactic. The real fix is to RESTRUCTURE `Ty.beq`
-- (non-mutual recursion scheme), which is a deliberate follow-up; for now the
-- heartbeat bump + isolation is the accepted cost.
set_option maxHeartbeats 1000000 in
mutual
/-- `Ty.beq` is reflexive. -/
theorem Ty.beq_self : (t : Ty) → Ty.beq t t = true
  | .prim p => by cases p <;> simp [Ty.beq, primBeq]
  | .tuple ts => by simpa [Ty.beq] using beqList_self ts
  | .anonStruct fs => by simpa [Ty.beq] using beqFields_self fs
  | .array e (some n) => by simp [Ty.beq, Ty.beq_self e]
  | .array e none => by simpa [Ty.beq] using Ty.beq_self e
  | .slice e => by simpa [Ty.beq] using Ty.beq_self e
  | .map k v => by simp [Ty.beq, Ty.beq_self k, Ty.beq_self v]
  | .errorUnion p es => by simp [Ty.beq, Ty.beq_self p, beqList_self es]
  | .refinement n b as => by simp [Ty.beq, Ty.beq_self b]
  | .struct_ n => by simp [Ty.beq]
  | .enum_ n => by simp [Ty.beq]
  | .bitfield n => by simp [Ty.beq]
  | .contract n => by simp [Ty.beq]
  | .externalProxy n => by simp [Ty.beq]
  | .storageSlot => rfl
  | .storageRange => rfl
  | .function n ps rs => by simp [Ty.beq, beqList_self ps, beqList_self rs]
  | .resourceDomain n c => by simp [Ty.beq, Ty.beq_self c]
  | .resourcePlace e => by simpa [Ty.beq] using Ty.beq_self e
theorem beqList_self : (ts : List Ty) → beqList ts ts = true
  | [] => rfl
  | t :: ts => by simp [beqList, Ty.beq_self t, beqList_self ts]
theorem beqFields_self : (fs : List (Name × Ty)) → beqFields fs fs = true
  | [] => rfl
  | f :: fs => by simp [beqFields, Ty.beq_self f.2, beqFields_self fs]
end

end Ora.Types

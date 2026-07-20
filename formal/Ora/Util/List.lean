/-
Shared executable list predicates used by trusted formal models.

These definitions are intentionally independent of dispatcher and source-
accounting schemas. A single implementation prevents sync modules from
silently acquiring different notions of duplicate-free natural-number rows.
-/

namespace Ora.Util

def noDuplicateNat : List Nat → Bool
  | [] => true
  | value :: rest => !(rest.contains value) && noDuplicateNat rest

theorem noDuplicateNat_sound (values : List Nat)
    (h : noDuplicateNat values = true) : values.Nodup := by
  induction values with
  | nil => simp
  | cons value rest ih =>
      simp [noDuplicateNat] at h
      simp [List.nodup_cons, h.1, ih h.2]

end Ora.Util

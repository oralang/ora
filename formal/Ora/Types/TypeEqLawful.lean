/-
Ora type system — lawfulness of structural equality (`Ty.beq`), up to a real
`DecidableEq Ty`.

PROVEN here: `Ty.beq` is reflexive (`Ty.beq_self`), reflects equality
(`Ty.eq_of_beq`), hence decides it (`Ty.beq_iff_eq`) — yielding `DecidableEq Ty`.

PROOF TECHNIQUE — the genuine fix to the old `whnf` blowup, on two axes:

  * STRUCTURE: induct with `Ty.recAux` (defined in `Ora/Types/Ty.lean`), the
    nested-inductive's proper structural principle. One `motive`, an element-wise
    IH per aggregate — no hand-supplied `motive_1/2/3`, no functional-induction
    juggling.
  * REDUCTION: unfold `Ty.beq` ONLY through the cheap per-constructor `@[simp]`
    `rfl`-lemmas (`beq_tuple`, `beq_eu`, …, in `TypeEq.lean`), NEVER `simp
    [Ty.beq]` (which forces the expensive auto-generated equation lemma).

Result: the whole chain compiles at the DEFAULT heartbeat budget in ~1.5s — no
`set_option maxHeartbeats`, no isolation cost. Mismatched constructor pairs in the
forward direction reduce to `false` by `rfl`, so a stray `Ty.beq a b = true`
hypothesis is killed by `nomatch h`.
-/

import Ora.Types.TypeEq

namespace Ora.Types

/-! ## Reflexivity (`Ty.beq t t = true`) -/

/-- Reflexivity for the leaf primitives (`int`/`fixedBytes` need `BEq` reflexivity,
    not `rfl`). -/
theorem primBeq_self (p : PrimTy) : primBeq p p = true := by cases p <;> simp [primBeq]

/-- `beqList` reflexivity, GIVEN element-wise reflexivity (the `Ty.recAux` IH). -/
theorem beqList_self_of : (ts : List Ty) → (∀ t ∈ ts, Ty.beq t t = true) → beqList ts ts = true
  | [], _ => rfl
  | t :: ts, h => by
      simp only [beqList_cons, Bool.and_eq_true]
      exact ⟨h t (.head _), beqList_self_of ts (fun x hx => h x (.tail _ hx))⟩

/-- `beqFields` reflexivity, given element-wise reflexivity. -/
theorem beqFields_self_of :
    (fs : List (Name × Ty)) → (∀ f ∈ fs, Ty.beq f.2 f.2 = true) → beqFields fs fs = true
  | [], _ => rfl
  | f :: fs, h => by
      simp only [beqFields_cons, Bool.and_eq_true, beq_self_eq_true, true_and]
      exact ⟨h f (.head _), beqFields_self_of fs (fun x hx => h x (.tail _ hx))⟩

/-- `Ty.beq` is reflexive. Plain structural induction via `Ty.recAux`. -/
theorem Ty.beq_self (t : Ty) : Ty.beq t t = true := by
  induction t using Ty.recAux with
  | prim p => simpa using primBeq_self p
  | tuple ts ih => simpa using beqList_self_of ts ih
  | anonStruct fs ih => simpa using beqFields_self_of fs ih
  | array e n ih => cases n <;> simp [ih]
  | slice e ih => simpa using ih
  | map k v ihk ihv => simp [ihk, ihv]
  | errorUnion p es ihp ih => simp [ihp, beqList_self_of es ih]
  | refinement n b as ih => simp [ih]
  | function n ps rs ihp ihr => simp [beqList_self_of ps ihp, beqList_self_of rs ihr]
  | resourceDomain n c ih => simp [ih]
  | resourcePlace e ih => simpa using ih
  | struct_ n | enum_ n | bitfield n | contract n | externalProxy n => simp
  | storageSlot | storageRange => rfl

/-- Convenience: unconditional `beqList` reflexivity. -/
theorem beqList_self (ts : List Ty) : beqList ts ts = true :=
  beqList_self_of ts (fun t _ => Ty.beq_self t)

/-- Convenience: unconditional `beqFields` reflexivity. -/
theorem beqFields_self (fs : List (Name × Ty)) : beqFields fs fs = true :=
  beqFields_self_of fs (fun f _ => Ty.beq_self f.2)

/-! ## Forward direction (`Ty.beq a b = true → a = b`) -/

/-- `FixedBytesLen` equality is its length (the bound proofs are irrelevant). -/
theorem FixedBytesLen.eq_iff {m n : FixedBytesLen} : m = n ↔ m.n = n.n :=
  ⟨fun h => h ▸ rfl, fun h => by cases m; cases n; cases h; rfl⟩

theorem primBeq_iff (p q : PrimTy) : primBeq p q = true ↔ p = q := by
  cases p <;> cases q <;> simp [primBeq, FixedBytesLen.eq_iff, beq_iff_eq]

/-- `beqList` reflects equality, given element-wise reflection (the IH). -/
theorem eq_of_beqList : (as bs : List Ty) →
    (∀ a ∈ as, ∀ b, Ty.beq a b = true → a = b) → beqList as bs = true → as = bs
  | [], [], _, _ => rfl
  | a :: as, b :: bs, ih, h => by
      simp only [beqList_cons, Bool.and_eq_true] at h
      have hb := ih a (.head _) b h.1
      have ht := eq_of_beqList as bs (fun x hx => ih x (.tail _ hx)) h.2
      subst hb; subst ht; rfl

/-- `beqFields` reflects equality, given element-wise reflection. -/
theorem eq_of_beqFields : (as bs : List (Name × Ty)) →
    (∀ a ∈ as, ∀ b, Ty.beq a.2 b = true → a.2 = b) → beqFields as bs = true → as = bs
  | [], [], _, _ => rfl
  | a :: as, b :: bs, ih, h => by
      simp only [beqFields_cons, Bool.and_eq_true, beq_iff_eq] at h
      obtain ⟨⟨hn, hv⟩, ht⟩ := h
      have hv2 := ih a (.head _) b.2 hv
      have ht2 := eq_of_beqFields as bs (fun x hx => ih x (.tail _ hx)) ht
      subst ht2; cases a; cases b; simp_all

/-- `Ty.beq` reflects equality. Structural induction via `Ty.recAux` on the first
    type, then `cases` on the second: the matching arm reduces through the cheap
    `@[simp]` lemmas and the IH; every mismatched arm has `Ty.beq a b` reduce to
    `false` by `rfl`, so the `= true` hypothesis is impossible (`nomatch h`). -/
theorem Ty.eq_of_beq (t : Ty) : ∀ b, Ty.beq t b = true → t = b := by
  induction t using Ty.recAux with
  | prim p => intro b; cases b <;> intro h <;>
      first | (rw [beq_prim, primBeq_iff] at h; exact h ▸ rfl) | nomatch h
  | tuple ts ih => intro b; cases b <;> intro h <;>
      first | exact congrArg _ (eq_of_beqList ts _ ih (by simpa using h)) | nomatch h
  | anonStruct fs ih => intro b; cases b <;> intro h <;>
      first | exact congrArg _ (eq_of_beqFields fs _ ih (by simpa using h)) | nomatch h
  | array e n ihe => intro b; cases b with
      | array a m =>
          cases n <;> cases m <;> intro h <;>
            first
            | (simp only [beq_arrayS, Bool.and_eq_true, _root_.beq_iff_eq] at h;
               exact h.1 ▸ ihe _ h.2 ▸ rfl)
            | (simp only [beq_arrayN] at h; exact ihe _ h ▸ rfl)
            | nomatch h
      | _ => cases n <;> intro h <;> nomatch h
  | slice e ihe => intro b; cases b <;> intro h <;>
      first | (rw [beq_slice] at h; exact ihe _ h ▸ rfl) | nomatch h
  | map k v ihk ihv => intro b; cases b <;> intro h <;>
      first
      | (simp only [beq_map, Bool.and_eq_true] at h; exact ihk _ h.1 ▸ ihv _ h.2 ▸ rfl)
      | nomatch h
  | errorUnion p es ihp ih => intro b; cases b <;> intro h <;>
      first | (simp only [beq_eu, Bool.and_eq_true] at h;
               exact ihp _ h.1 ▸ eq_of_beqList es _ ih h.2 ▸ rfl) | nomatch h
  | refinement n b as ihb => intro c; cases c <;> intro h <;>
      first | (simp only [beq_refine, Bool.and_eq_true, _root_.beq_iff_eq] at h;
               obtain ⟨⟨hn, hb⟩, ha⟩ := h; exact hn ▸ ihb _ hb ▸ ha ▸ rfl) | nomatch h
  | function n ps rs ihp ihr => intro b; cases b <;> intro h <;>
      first | (simp only [beq_function, Bool.and_eq_true, _root_.beq_iff_eq] at h;
               obtain ⟨⟨hn, hp⟩, hr⟩ := h;
               exact hn ▸ eq_of_beqList ps _ ihp hp ▸ eq_of_beqList rs _ ihr hr ▸ rfl) | nomatch h
  | resourceDomain n c ihc => intro b; cases b <;> intro h <;>
      first | (simp only [beq_rdom, Bool.and_eq_true, _root_.beq_iff_eq] at h;
               exact h.1 ▸ ihc _ h.2 ▸ rfl) | nomatch h
  | resourcePlace e ihe => intro b; cases b <;> intro h <;>
      first | (rw [beq_rplace] at h; exact ihe _ h ▸ rfl) | nomatch h
  | struct_ n | enum_ n | bitfield n | contract n | externalProxy n =>
      intro b; cases b <;> intro h <;>
      first
      | (simp only [beq_struct, beq_enum, beq_bitfield, beq_contract, beq_extproxy,
                    _root_.beq_iff_eq] at h;
         exact h ▸ rfl)
      | nomatch h
  | storageSlot | storageRange => intro b; cases b <;> intro h <;> first | rfl | nomatch h

/-! ## `beq ↔ =` and `DecidableEq Ty` -/

/-- Structural `Ty.beq` is lawful: it decides propositional equality. -/
theorem Ty.beq_iff_eq (a b : Ty) : Ty.beq a b = true ↔ a = b :=
  ⟨Ty.eq_of_beq a b, fun h => by subst h; exact Ty.beq_self a⟩

/-- The `List Ty` analogue (componentwise via `Ty.beq_iff_eq`). -/
theorem beqList_iff_eq : (as bs : List Ty) → (beqList as bs = true ↔ as = bs)
  | [], [] => by simp
  | [], _ :: _ => by simp
  | _ :: _, [] => by simp
  | a :: as, b :: bs => by
      simp only [beqList_cons, Bool.and_eq_true, Ty.beq_iff_eq, beqList_iff_eq as bs,
        List.cons.injEq]

/-- The payoff: a real `DecidableEq Ty`, backed by the structural `Ty.beq`. -/
instance : DecidableEq Ty := fun a b =>
  decidable_of_iff (Ty.beq a b = true) (Ty.beq_iff_eq a b)

/-- `Ty.beq` is the lawful `BEq` (it decides `=`). This makes `List Ty` membership
    decidable and lets the standard `List` API work for `Ty`. -/
instance : LawfulBEq Ty where
  eq_of_beq := Ty.eq_of_beq _ _
  rfl := Ty.beq_self _

end Ora.Types

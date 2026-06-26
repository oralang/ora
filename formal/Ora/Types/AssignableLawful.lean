/-
Ora type system — lawfulness of (the faithful, structural) `Ty.assignable`.

PROVEN: `Ty.assignable` is a PREORDER — `Ty.assignable_refl` + `Ty.assignable_trans`.
The refinement arm composes via `refCond_trans` (NonZeroAddress / semantic-equality /
interval-containment by `boundOk_trans`, i.e. `Int.le_trans`); the error-union arm via
`errorSubsetB_trans`; integers via `IntTy.assignable_trans`; aggregates via the IH.
All structural ⇒ axioms `propext, Quot.sound` only (no `sorry`, no `native_decide`).

(The full compiler relation is NOT a preorder — but our structural under-approximation is;
see the cross-constructor boundary noted in `Assignable.lean`.)
-/

import Ora.Types.Assignable
import Ora.Types.TypeEqLawful

namespace Ora.Types

theorem refArgEq_self : ∀ a, refArgEq a a = true
  | .integer s => by simp [refArgEq]
  | .typeMarker => rfl
theorem refArgsEq_self : ∀ as, refArgsEq as as = true
  | [] => rfl
  | a :: as => by simp [refArgsEq, refArgEq_self a, refArgsEq_self as]

theorem IntTy.assignable_refl (e : IntTy) : IntTy.assignable e e = true := by simp [IntTy.assignable]

theorem errorMem_self {e : Ty} {ee : List Ty} (h : e ∈ ee) : errorMem e ee = true := by
  simp only [errorMem, List.any_eq_true]; exact ⟨e, h, Ty.beq_self e⟩
theorem errorSubsetB_self : ∀ ee, errorSubsetB ee ee = true
  | [] => rfl
  | e :: es => by
      simp only [errorSubsetB, List.all_eq_true]
      intro x hx; exact errorMem_self hx

theorem assignableList_refl_of : (ts : List Ty) → (∀ t ∈ ts, Ty.assignable t t = true) → assignableList ts ts = true
  | [], _ => rfl
  | t :: ts, h => by
      simp only [asgList_cons, Bool.and_eq_true]
      exact ⟨h t (.head _), assignableList_refl_of ts (fun x hx => h x (.tail _ hx))⟩
theorem assignableFields_refl_of : (fs : List (Name × Ty)) → (∀ f ∈ fs, Ty.assignable f.2 f.2 = true) → assignableFields fs fs = true
  | [], _ => rfl
  | (n, t) :: fs, h => by
      simp only [asgFields_cons, Bool.and_eq_true, beq_self_eq_true, true_and]
      exact ⟨h (n, t) (.head _), assignableFields_refl_of fs (fun x hx => h x (.tail _ hx))⟩

theorem Ty.assignable_refl (t : Ty) : Ty.assignable t t = true := by
  induction t using Ty.recAux with
  | prim p => cases p <;> simp [IntTy.assignable_refl]
  | tuple ts ih => simpa using assignableList_refl_of ts ih
  | anonStruct fs ih => simpa using assignableFields_refl_of fs ih
  | array e n ih => cases n <;> simp [ih]
  | slice e ih => simpa using ih
  | map k v ihk ihv => simp [ihk, ihv]
  | errorUnion p es ihp ih => simp [ihp, errorSubsetB_self]
  | refinement n b as ih =>
      simp only [asg_refine, Ty.beq_self, Bool.true_and, refCond]
      split <;> simp_all [refArgsEq_self]
  | function n ps rs _ _ =>
      simp only [asg_func, beq_self_eq_true, beqList_self, Bool.and_self]
  | resourceDomain n c _ => simp [Ty.beq_self]
  | resourcePlace e _ => simpa using Ty.beq_self e
  | struct_ n | enum_ n | bitfield n | contract n | externalProxy n => simp
  | storageSlot | storageRange => rfl

/-! ## transitivity helpers -/

theorem errorMem_iff {e : Ty} {es : List Ty} : errorMem e es = true ↔ e ∈ es := by
  simp only [errorMem, List.any_eq_true]
  constructor
  · rintro ⟨x, hx, hb⟩; rw [Ty.beq_iff_eq] at hb; exact hb ▸ hx
  · intro h; exact ⟨e, h, Ty.beq_self e⟩

theorem errorSubsetB_trans {a b c : List Ty}
    (h1 : errorSubsetB a b = true) (h2 : errorSubsetB b c = true) : errorSubsetB a c = true := by
  simp only [errorSubsetB, List.all_eq_true] at *
  intro e he
  exact h2 e (errorMem_iff.1 (h1 e he))

theorem boundOk_trans {e1 e2 e3 : Option Int} {isMin : Bool}
    (h1 : boundOk e1 e2 isMin = true) (h2 : boundOk e2 e3 isMin = true) : boundOk e1 e3 isMin = true := by
  cases isMin <;> cases e1 <;> cases e2 <;> cases e3 <;>
    simp_all [boundOk, decide_eq_true_eq] <;> omega

theorem refArgEq_eq {a b : RefinementArg} (h : refArgEq a b = true) : a = b := by
  cases a <;> cases b <;> simp_all [refArgEq]
theorem refArgsEq_eq : (a b : List RefinementArg) → refArgsEq a b = true → a = b
  | [], [], _ => rfl
  | x :: xs, y :: ys, h => by
      simp only [refArgsEq, Bool.and_eq_true] at h
      rw [refArgEq_eq h.1, refArgsEq_eq xs ys h.2]
  | [], _ :: _, h => by simp [refArgsEq] at h
  | _ :: _, [], h => by simp [refArgsEq] at h

theorem refBounds_nza {ne : Name} {aa : List RefinementArg} (h : isNZA ne = true) :
    refBounds ne aa = none := by
  simp only [isNZA, beq_iff_eq] at h
  simp only [refBounds, h]

theorem refCond_trans {ne1 ne2 ne3 : Name} {ae1 ae2 ae3 : List RefinementArg}
    (h1 : refCond ne1 ae1 ne2 ae2 = true) (h2 : refCond ne2 ae2 ne3 ae3 = true) :
    refCond ne1 ae1 ne3 ae3 = true := by
  unfold refCond at h1 h2 ⊢
  by_cases hz1 : isNZA ne1 = true
  · rw [if_pos hz1] at h1 ⊢; rw [if_pos h1] at h2; exact h2
  · rw [if_neg hz1] at h1 ⊢
    by_cases hs1 : (ne1 == ne2 && refArgsEq ae1 ae2) = true
    · rw [Bool.and_eq_true] at hs1; obtain ⟨hn, ha⟩ := hs1
      rw [beq_iff_eq] at hn; subst hn; have hae := refArgsEq_eq _ _ ha; subst hae
      rw [if_neg hz1] at h2; exact h2
    · rw [if_neg hs1] at h1
      by_cases hs13 : (ne1 == ne3 && refArgsEq ae1 ae3) = true
      · rw [if_pos hs13]
      · rw [if_neg hs13]
        by_cases hz2 : isNZA ne2 = true
        · rw [refBounds_nza hz2] at h1; cases refBounds ne1 ae1 <;> simp_all
        · rw [if_neg hz2] at h2
          by_cases hs2 : (ne2 == ne3 && refArgsEq ae2 ae3) = true
          · rw [Bool.and_eq_true] at hs2; obtain ⟨hn2, ha2⟩ := hs2
            rw [beq_iff_eq] at hn2; subst hn2; have hae2 := refArgsEq_eq _ _ ha2; subst hae2
            exact h1
          · rw [if_neg hs2] at h2
            cases hr1 : refBounds ne1 ae1 with
            | none => rw [hr1] at h1; simp at h1
            | some b1 => cases hr2 : refBounds ne2 ae2 with
                | none => rw [hr1, hr2] at h1; simp at h1
                | some b2 => cases hr3 : refBounds ne3 ae3 with
                    | none => rw [hr2, hr3] at h2; simp at h2
                    | some b3 =>
                        obtain ⟨e1, m1⟩ := b1; obtain ⟨e2, m2⟩ := b2; obtain ⟨e3, m3⟩ := b3
                        simp only [hr1, hr2, Bool.and_eq_true] at h1
                        simp only [hr2, hr3, Bool.and_eq_true] at h2
                        simp only [hr1, hr3, Bool.and_eq_true]
                        exact ⟨boundOk_trans h1.1 h2.1, boundOk_trans h1.2 h2.2⟩

theorem Ty.beq_trans {a b c : Ty} (h1 : Ty.beq a b = true) (h2 : Ty.beq b c = true) :
    Ty.beq a c = true := by rw [Ty.beq_iff_eq] at h1 h2 ⊢; exact h1.trans h2

theorem IntTy.assignable_trans {e1 e2 e3 : IntTy}
    (h1 : IntTy.assignable e1 e2 = true) (h2 : IntTy.assignable e2 e3 = true) :
    IntTy.assignable e1 e3 = true := by
  simp only [IntTy.assignable, Bool.and_eq_true, _root_.beq_iff_eq, decide_eq_true_eq] at h1 h2 ⊢
  exact ⟨h1.1.trans h2.1, Nat.le_trans h2.2 h1.2⟩

theorem prim_trans (p q r : PrimTy) :
    Ty.assignable (.prim p) (.prim q) = true → Ty.assignable (.prim q) (.prim r) = true →
    Ty.assignable (.prim p) (.prim r) = true := by
  cases p <;> cases q <;> cases r <;> intro h1 h2 <;>
    first
    | exact Bool.noConfusion h1
    | exact Bool.noConfusion h2
    | exact IntTy.assignable_trans h1 h2
    | (simp_all only [asg_fbytes, _root_.beq_iff_eq]; omega)
    | simp_all

theorem assignableList_trans : (as bs cs : List Ty) →
    (∀ a ∈ as, ∀ b c, Ty.assignable a b = true → Ty.assignable b c = true → Ty.assignable a c = true) →
    assignableList as bs = true → assignableList bs cs = true → assignableList as cs = true
  | [], [], [], _, _, _ => rfl
  | [], [], _ :: _, _, _, h2 => by nomatch h2
  | [], _ :: _, _, _, h1, _ => by nomatch h1
  | _ :: _, [], _, _, h1, _ => by nomatch h1
  | _ :: _, _ :: _, [], _, _, h2 => by nomatch h2
  | a :: as, b :: bs, c :: cs, ih, h1, h2 => by
      simp only [asgList_cons, Bool.and_eq_true] at h1 h2 ⊢
      exact ⟨ih a (.head _) b c h1.1 h2.1,
             assignableList_trans as bs cs (fun x hx => ih x (.tail _ hx)) h1.2 h2.2⟩

theorem assignableFields_trans : (as bs cs : List (Name × Ty)) →
    (∀ a ∈ as, ∀ b c, Ty.assignable a.2 b = true → Ty.assignable b c = true → Ty.assignable a.2 c = true) →
    assignableFields as bs = true → assignableFields bs cs = true → assignableFields as cs = true
  | [], [], [], _, _, _ => rfl
  | [], [], _ :: _, _, _, h2 => by nomatch h2
  | [], _ :: _, _, _, h1, _ => by nomatch h1
  | _ :: _, [], _, _, h1, _ => by nomatch h1
  | _ :: _, _ :: _, [], _, _, h2 => by nomatch h2
  | a :: as, b :: bs, c :: cs, ih, h1, h2 => by
      simp only [asgFields_cons, Bool.and_eq_true, _root_.beq_iff_eq] at h1 h2 ⊢
      obtain ⟨⟨hn1, hv1⟩, ht1⟩ := h1
      obtain ⟨⟨hn2, hv2⟩, ht2⟩ := h2
      exact ⟨⟨hn1.trans hn2, ih a (.head _) b.2 c.2 hv1 hv2⟩,
             assignableFields_trans as bs cs (fun x hx => ih x (.tail _ hx)) ht1 ht2⟩

theorem Ty.assignable_trans (a : Ty) :
    ∀ b c, Ty.assignable a b = true → Ty.assignable b c = true → Ty.assignable a c = true := by
  induction a using Ty.recAux with
  | prim p =>
      intro b c h1 h2
      cases b with
      | prim q => cases c with
          | prim r => exact prim_trans p q r h1 h2
          | _ => cases q <;> exact Bool.noConfusion h2
      | _ => cases p <;> exact Bool.noConfusion h1
  | tuple ts ih =>
      intro b c h1 h2
      cases b <;> first
      | exact Bool.noConfusion h1
      | (cases c <;> first
          | (simp only [asg_tuple] at h1 h2 ⊢; exact assignableList_trans ts _ _ ih h1 h2)
          | exact Bool.noConfusion h2)
  | anonStruct fs ih =>
      intro b c h1 h2
      cases b <;> first
      | exact Bool.noConfusion h1
      | (cases c <;> first
          | (simp only [asg_anon] at h1 h2 ⊢; exact assignableFields_trans fs _ _ ih h1 h2)
          | exact Bool.noConfusion h2)
  | array e n ihe =>
      intro b c h1 h2
      cases b with
      | array a m => cases c with
          | array a' k =>
              cases n <;> cases m <;> cases k <;> first
              | (simp only [asg_arrayS, Bool.and_eq_true, _root_.beq_iff_eq] at h1 h2 ⊢;
                 exact ⟨h1.1.trans h2.1, ihe _ _ h1.2 h2.2⟩)
              | (simp only [asg_arrayN] at h1 h2 ⊢; exact ihe _ _ h1 h2)
              | exact Bool.noConfusion h1
              | exact Bool.noConfusion h2
          | _ => cases m <;> exact Bool.noConfusion h2
      | _ => cases n <;> exact Bool.noConfusion h1
  | slice e ihe =>
      intro b c h1 h2
      cases b <;> first
      | exact Bool.noConfusion h1
      | (cases c <;> first
          | (simp only [asg_slice] at h1 h2 ⊢; exact ihe _ _ h1 h2)
          | exact Bool.noConfusion h2)
  | map k v ihk ihv =>
      intro b c h1 h2
      cases b <;> first
      | exact Bool.noConfusion h1
      | (cases c <;> first
          | (simp only [asg_map, Bool.and_eq_true] at h1 h2 ⊢;
             exact ⟨ihk _ _ h1.1 h2.1, ihv _ _ h1.2 h2.2⟩)
          | exact Bool.noConfusion h2)
  | errorUnion p es ihp _ =>
      intro b c h1 h2
      cases b <;> first
      | exact Bool.noConfusion h1
      | (cases c <;> first
          | (simp only [asg_eu, Bool.and_eq_true] at h1 h2 ⊢;
             exact ⟨ihp _ _ h1.1 h2.1, errorSubsetB_trans h2.2 h1.2⟩)
          | exact Bool.noConfusion h2)
  | refinement n be ae _ =>
      intro c d h1 h2
      cases c <;> first
      | exact Bool.noConfusion h1
      | (cases d <;> first
          | (simp only [asg_refine, Bool.and_eq_true] at h1 h2 ⊢;
             exact ⟨Ty.beq_trans h1.1 h2.1, refCond_trans h1.2 h2.2⟩)
          | exact Bool.noConfusion h2)
  | function n ps rs _ _ =>
      intro b c h1 h2
      cases b <;> first
      | exact Bool.noConfusion h1
      | (cases c <;> first
          | (simp only [asg_func, Bool.and_eq_true, _root_.beq_iff_eq, beqList_iff_eq] at h1 h2 ⊢;
             obtain ⟨⟨hn1, hp1⟩, hr1⟩ := h1; obtain ⟨⟨hn2, hp2⟩, hr2⟩ := h2;
             exact ⟨⟨hn1.trans hn2, hp1.trans hp2⟩, hr1.trans hr2⟩)
          | exact Bool.noConfusion h2)
  | resourceDomain n c _ =>
      intro b d h1 h2
      cases b <;> first
      | exact Bool.noConfusion h1
      | (cases d <;> first
          | (simp only [asg_rdom, Bool.and_eq_true, _root_.beq_iff_eq, Ty.beq_iff_eq] at h1 h2 ⊢;
             exact ⟨h1.1.trans h2.1, h1.2.trans h2.2⟩)
          | exact Bool.noConfusion h2)
  | resourcePlace e _ =>
      intro b c h1 h2
      cases b <;> first
      | exact Bool.noConfusion h1
      | (cases c <;> first
          | (simp only [asg_rplace, Ty.beq_iff_eq] at h1 h2 ⊢; exact h1.trans h2)
          | exact Bool.noConfusion h2)
  | struct_ n | enum_ n | bitfield n | contract n | externalProxy n =>
      intro b c h1 h2
      cases b <;> first
      | exact Bool.noConfusion h1
      | (cases c <;> first
          | (simp only [asg_struct, asg_enum, asg_bitfield, asg_contract, asg_extproxy,
               _root_.beq_iff_eq] at h1 h2 ⊢; exact h1.trans h2)
          | exact Bool.noConfusion h2)
  | storageSlot | storageRange =>
      intro b c h1 h2
      cases b <;> first
      | exact Bool.noConfusion h1
      | (cases c <;> first | rfl | exact Bool.noConfusion h2)

end Ora.Types

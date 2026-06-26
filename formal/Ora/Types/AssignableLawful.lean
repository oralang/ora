/-
Ora type system — lawfulness of assignability (`Ty.assignable`).

PROVEN here: assignability is a PREORDER on `Ty`:
  * `Ty.assignable_refl`  — reflexive (every type assigns to itself);
  * `Ty.assignable_trans` — transitive (widening / structural / nominal compose).

The transitive core is `IntTy.assignable_trans` (integer widening: same signedness,
width bounds chain by `≤`) — the only non-trivial leaf relation.

PROOF TECHNIQUE (same as `TypeEqLawful.lean`, on the same foundation):
  * STRUCTURE: induct with `Ty.recAux` (in `Ty.lean`) — one `motive`, element-wise
    IH per aggregate; no hand-supplied motives.
  * REDUCTION: unfold `Ty.assignable` ONLY through the cheap per-constructor
    `@[simp]` `rfl`-lemmas in `Assignable.lean`, NEVER `simp [Ty.assignable]`.
  * Transitivity inducts on the first type and `cases` the others: the matching arm
    reduces through the cheap lemmas + IH / `IntTy.assignable_trans`; mismatched
    constructor pairs make `Ty.assignable` reduce to `false` by `rfl`, so a stray
    `= true` hypothesis is impossible (`Bool.noConfusion`). The `prim` and `array`
    arms case their payloads (`PrimTy` / `Option`) explicitly, since those block
    reduction of mismatches otherwise.

The `function` / `resource_*` arms are identity-only in the compiler, carried via
`Ty.beq`; their transitivity is equality-transitivity (`Ty.beq_iff_eq` /
`beqList_iff_eq`).

NOTE: write `_root_.beq_iff_eq` (not bare `beq_iff_eq`) — the bare name resolves
ambiguously in this namespace and silently no-ops inside `simp only`.

Builds at the DEFAULT heartbeat budget in ~1.5s; part of the default `Ora` build.
-/

import Ora.Types.Assignable
import Ora.Types.TypeEqLawful

namespace Ora.Types

/-! ## Reflexivity -/

theorem IntTy.assignable_refl (e : IntTy) : IntTy.assignable e e = true := by
  simp [IntTy.assignable]

/-- `assignableList` reflexivity, GIVEN element-wise reflexivity (the `recAux` IH). -/
theorem assignableList_refl_of :
    (ts : List Ty) → (∀ t ∈ ts, Ty.assignable t t = true) → assignableList ts ts = true
  | [], _ => rfl
  | t :: ts, h => by
      simp only [asgList_cons, Bool.and_eq_true]
      exact ⟨h t (.head _), assignableList_refl_of ts (fun x hx => h x (.tail _ hx))⟩

/-- `assignableFields` reflexivity, given element-wise reflexivity. -/
theorem assignableFields_refl_of :
    (fs : List (Name × Ty)) → (∀ f ∈ fs, Ty.assignable f.2 f.2 = true) →
      assignableFields fs fs = true
  | [], _ => rfl
  | f :: fs, h => by
      simp only [asgFields_cons, Bool.and_eq_true, beq_self_eq_true, true_and]
      exact ⟨h f (.head _), assignableFields_refl_of fs (fun x hx => h x (.tail _ hx))⟩

/-- Assignability is reflexive: any type is assignable to itself. -/
theorem Ty.assignable_refl (t : Ty) : Ty.assignable t t = true := by
  induction t using Ty.recAux with
  | prim p => cases p <;> simp [IntTy.assignable_refl]
  | tuple ts ih => simpa using assignableList_refl_of ts ih
  | anonStruct fs ih => simpa using assignableFields_refl_of fs ih
  | array e n ih => cases n <;> simp [ih]
  | slice e ih => simpa using ih
  | map k v ihk ihv => simp [ihk, ihv]
  | errorUnion p es ihp ih => simp [ihp, assignableList_refl_of es ih]
  | refinement n b as ih => simp [ih]
  | function n ps rs _ _ => simp [beqList_self]
  | resourceDomain n c _ => simp [Ty.beq_self]
  | resourcePlace e _ => simpa using Ty.beq_self e
  | struct_ n | enum_ n | bitfield n | contract n | externalProxy n => simp
  | storageSlot | storageRange => rfl

/-! ## Transitivity -/

/-- Integer widening is transitive: same signedness composes, width bounds chain by
    `≤`. The only non-trivial leaf relation in assignability. -/
theorem IntTy.assignable_trans {e₁ e₂ e₃ : IntTy}
    (h₁ : IntTy.assignable e₁ e₂ = true) (h₂ : IntTy.assignable e₂ e₃ = true) :
    IntTy.assignable e₁ e₃ = true := by
  simp only [IntTy.assignable, Bool.and_eq_true, _root_.beq_iff_eq, decide_eq_true_eq] at h₁ h₂ ⊢
  exact ⟨h₁.1.trans h₂.1, Nat.le_trans h₂.2 h₁.2⟩

/-- Transitivity for the primitive leaves (the only widening case is `int`). -/
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

/-- `assignableList` transitivity, given element-wise transitivity (the IH). -/
theorem assignableList_trans : (as bs cs : List Ty) →
    (∀ a ∈ as, ∀ b c, Ty.assignable a b = true → Ty.assignable b c = true →
      Ty.assignable a c = true) →
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

/-- `assignableFields` transitivity, given element-wise transitivity. -/
theorem assignableFields_trans : (as bs cs : List (Name × Ty)) →
    (∀ a ∈ as, ∀ b c, Ty.assignable a.2 b = true → Ty.assignable b c = true →
      Ty.assignable a.2 c = true) →
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

/-- Assignability is transitive — full structure. Induct on the first type with
    `Ty.recAux`; the matching arm reduces via the cheap `@[simp]` lemmas + the IH /
    `IntTy.assignable_trans`, mismatches die by `Bool.noConfusion`. -/
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
  | errorUnion p es ihp ih =>
      intro b c h1 h2
      cases b <;> first
      | exact Bool.noConfusion h1
      | (cases c <;> first
          | (simp only [asg_eu, Bool.and_eq_true] at h1 h2 ⊢;
             exact ⟨ihp _ _ h1.1 h2.1, assignableList_trans es _ _ ih h1.2 h2.2⟩)
          | exact Bool.noConfusion h2)
  | refinement n b as ihb =>
      intro c d h1 h2
      cases c <;> first
      | exact Bool.noConfusion h1
      | (cases d <;> first
          | (simp only [asg_refine, Bool.and_eq_true, _root_.beq_iff_eq] at h1 h2 ⊢;
             exact ⟨h1.1.trans h2.1, ihb _ _ h1.2 h2.2⟩)
          | exact Bool.noConfusion h2)
  | function n ps rs _ _ =>
      intro b c h1 h2
      cases b <;> first
      | exact Bool.noConfusion h1
      | (cases c <;> first
          | (simp only [asg_function, Bool.and_eq_true, _root_.beq_iff_eq,
               beqList_iff_eq] at h1 h2 ⊢;
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

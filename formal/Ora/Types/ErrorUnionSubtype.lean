/-
Ora type system — error-union subtyping (`p ! es`).

`Ty.assignable`'s error-union arm currently checks payload + POSITIONAL
`assignableList` on the error list — the deferred "error subset" rule isn't modeled,
and positional comparison is order-sensitive. This file models the intended rule and
proves it sound, the way `RefinementSubtype` did for refinements.

The rule: `p ! es <: q ! fs` when the payload widens (`p` assignable into `q`) AND the
error set SHRINKS — `es ⊆ fs` (covariant). Intuition: a value that can fail in fewer
ways is usable where more failure modes are admitted (a `T ! {E1}` result fits a
`T ! {E1, E2}` slot). Error sets are SETS: order and duplication are irrelevant.

  * `EuSubtype` — the relation; `EuSubtype.refl` / `.trans` make it a PREORDER (payload
    via `assignable_refl`/`assignable_trans`, errors via subset refl/trans).
  * `ErrorSubset.of_perm` / `EuSubtype.errorPerm_mutual` — reordering the error set is a
    subtyping in BOTH directions (the set semantics).

This makes precise where the current positional arm is wrong: it REJECTS reordered
and widened error sets that the proper subset rule accepts (witnessed below).
-/

import Ora.Types.AssignableLawful

namespace Ora.Types

/-! ## Error-set subset (order-independent) -/

/-- Every error type of `sub` appears in `sup`. -/
def ErrorSubset (sub sup : List Ty) : Prop := ∀ e ∈ sub, e ∈ sup

/-- Decidable via `DecidableEq Ty` / `LawfulBEq Ty` (`TypeEqLawful`). -/
instance (sub sup : List Ty) : Decidable (ErrorSubset sub sup) :=
  inferInstanceAs (Decidable (∀ e ∈ sub, e ∈ sup))

theorem ErrorSubset.refl (es : List Ty) : ErrorSubset es es := fun _ h => h

theorem ErrorSubset.trans {a b c : List Ty}
    (h1 : ErrorSubset a b) (h2 : ErrorSubset b c) : ErrorSubset a c :=
  fun e he => h2 e (h1 e he)

/-- Reordering the error set (a permutation) is a subset — error sets are SETS. -/
theorem ErrorSubset.of_perm {es fs : List Ty} (h : es.Perm fs) : ErrorSubset es fs :=
  fun _ he => h.mem_iff.1 he

/-! ## Error-union subtyping: payload widens, error set shrinks -/

/-- `EuSubtype p es q fs` — a `p ! es` value is usable where `q ! fs` is expected: the
    payload `p` is assignable into `q`, and every way `p!es` can fail is admitted by
    `q!fs` (error-set subset, the covariant direction). -/
def EuSubtype (p : Ty) (es : List Ty) (q : Ty) (fs : List Ty) : Prop :=
  Ty.assignable q p = true ∧ ErrorSubset es fs

/-- Reflexive. -/
theorem EuSubtype.refl (p : Ty) (es : List Ty) : EuSubtype p es p es :=
  ⟨Ty.assignable_refl p, ErrorSubset.refl es⟩

/-- Transitive (payload via `assignable_trans`, errors via subset transitivity). -/
theorem EuSubtype.trans {p es q fs r gs}
    (h1 : EuSubtype p es q fs) (h2 : EuSubtype q fs r gs) : EuSubtype p es r gs :=
  ⟨Ty.assignable_trans r q p h2.1 h1.1, ErrorSubset.trans h1.2 h2.2⟩

/-- Subtyping lifted to the error-union TYPES. -/
def Ty.euSubtype : Ty → Ty → Prop
  | .errorUnion p es, .errorUnion q fs => EuSubtype p es q fs
  | _, _ => False

theorem Ty.euSubtype_refl (p : Ty) (es : List Ty) :
    Ty.euSubtype (.errorUnion p es) (.errorUnion p es) := EuSubtype.refl p es

theorem Ty.euSubtype_trans {a b c : Ty}
    (h1 : Ty.euSubtype a b) (h2 : Ty.euSubtype b c) : Ty.euSubtype a c := by
  cases a with
  | errorUnion p es => cases b with
      | errorUnion q fs => cases c with
          | errorUnion r gs => exact EuSubtype.trans h1 h2
          | _ => exact (h2 : False).elim
      | _ => exact (h1 : False).elim
  | _ => exact (h1 : False).elim

/-! ## The subset rule is order-independent — the positional arm is not -/

/-- A permuted error union is mutually a subtype (both directions). -/
theorem EuSubtype.errorPerm_mutual (p : Ty) {es fs : List Ty} (h : es.Perm fs) :
    EuSubtype p es p fs ∧ EuSubtype p fs p es :=
  ⟨⟨Ty.assignable_refl p, ErrorSubset.of_perm h⟩,
   ⟨Ty.assignable_refl p, ErrorSubset.of_perm h.symm⟩⟩

/-- Swapping two error types is a subtype under the SET rule… -/
example : EuSubtype (.prim u256) [.enum_ "E1", .enum_ "E2"]
                    (.prim u256) [.enum_ "E2", .enum_ "E1"] :=
  (EuSubtype.errorPerm_mutual _ (List.Perm.swap _ _ [])).1

/-- …but `Ty.assignable`'s CURRENT (positional) error arm REJECTS the reorder —
    `assignableList [E1,E2] [E2,E1]` needs `E1 ↦ E2` componentwise, which fails. -/
example :
    Ty.assignable (.errorUnion (.prim u256) [.enum_ "E1", .enum_ "E2"])
                  (.errorUnion (.prim u256) [.enum_ "E2", .enum_ "E1"]) = false := rfl

/-- Error-set WIDENING is a subtype: `p ! {E1}` is usable where `p ! {E1, E2}` is
    wanted (fewer failure modes) — also rejected by the positional arm. -/
example : EuSubtype (.prim u256) [.enum_ "E1"]
                    (.prim u256) [.enum_ "E1", .enum_ "E2"] :=
  ⟨Ty.assignable_refl _, by decide⟩

end Ora.Types

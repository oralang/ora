/-
Ora type system — refinement subtyping (`refinementSubtypeAssignable`).

`Ty.assignable`'s refinement arm only checks name + base (bounds ignored). The
compiler's real rule compares BOUNDS: a value under a tighter refinement may be
used where a looser one is wanted. This file models that as a relation on
refinements with TYPED bounds and proves the two properties that make it a
trustworthy subtyping relation:

  * PREORDER — `subsumes` is reflexive and transitive (well-formed subtyping);
  * SOUNDNESS — `sub.subsumes sup → ∀ x, sub.denote x → sup.denote x`: a value
    satisfying the subtype's predicate satisfies the supertype's. This is the
    property that justifies the coercion.

Bounds are TYPED (carrier `α`), not the type-level strings — deliberately. The
string→value interpretation is the deferred step `RefinementTie` parametrizes
(`lit : Nat → α`); proving subtyping over typed bounds keeps `String.toNat?` (not
kernel-reducible) out of the order reasoning. Same-name bound comparison is exactly
`refinementSubtypeAssignable`'s "min/max bounds comparison"; the CROSS-name
entailments (`NonZeroAddress ⟹ NonZero`, `BasisPoints ⟹ InRange 0 10000`, …) are
the separate subsumption lemmas in `RefinementValue`.

Order structure: core Lean has no `Preorder`, and `RefinementValue` is deliberately
axiom-free. `LePreorder` below is the MINIMAL `≤` structure soundness needs (refl +
trans), instantiable at any concrete carrier (the `Nat` instance witnesses it).
-/

import Ora.Types.RefinementValue
import Ora.Types.Refinement

namespace Ora.Refine

universe u
variable {α : Type u}

/-- Minimal mathlib-free preorder on `≤` — exactly the two facts refinement
    bound-subtyping needs. Concrete carriers (`Nat`, the Ora integer types, …)
    instantiate it. -/
class LePreorder (α : Type u) [LE α] : Prop where
  le_refl : ∀ a : α, a ≤ a
  le_trans : ∀ {a b c : α}, a ≤ b → b ≤ c → a ≤ c

instance : LePreorder Nat := ⟨Nat.le_refl, Nat.le_trans⟩

/-- The six runtime-guarded refinements, with TYPED bounds. (Exactly the set
    `RefinementValue` models — `Exact`/`Scaled` are compile-time-only.) -/
inductive Refined (α : Type u) where
  | minValue (n : α)
  | maxValue (n : α)
  | inRange (lo hi : α)
  | nonZero
  | nonZeroAddress
  | basisPoints

/-- What each refined type MEANS as a value predicate (the `Ora.Refine` predicates). -/
def Refined.denote [LE α] [Zero α] [OfNat α 10000] : Refined α → (α → Prop)
  | .minValue n => MinValue n
  | .maxValue n => MaxValue n
  | .inRange lo hi => InRange lo hi
  | .nonZero => NonZero
  | .nonZeroAddress => NonZeroAddress
  | .basisPoints => BasisPoints

/-- Same-name bound subtyping (`refinementSubtypeAssignable`): `sub` subtypes `sup`
    when its constraint is at least as strong. `MinValue` tightens upward (`b ≤ a`),
    `MaxValue` downward, `InRange` by interval containment. -/
def Refined.subsumes [LE α] : Refined α → Refined α → Prop
  | .minValue a, .minValue b => b ≤ a
  | .maxValue a, .maxValue b => a ≤ b
  | .inRange la ha, .inRange lb hb => lb ≤ la ∧ ha ≤ hb
  | .nonZero, .nonZero => True
  | .nonZeroAddress, .nonZeroAddress => True
  | .basisPoints, .basisPoints => True
  | _, _ => False

/-! ## Preorder -/

/-- Subtyping is reflexive. -/
theorem Refined.subsumes_refl [LE α] [LePreorder α] (r : Refined α) : r.subsumes r := by
  cases r <;> simp [Refined.subsumes, LePreorder.le_refl]

/-- Subtyping is transitive. `MinValue`'s contravariance flips the chaining order. -/
theorem Refined.subsumes_trans [LE α] [LePreorder α] {a b c : Refined α}
    (h1 : a.subsumes b) (h2 : b.subsumes c) : a.subsumes c := by
  cases a <;> cases b <;> cases c <;>
    first
    | exact (h1 : False).elim
    | exact (h2 : False).elim
    | exact LePreorder.le_trans h2 h1
    | exact LePreorder.le_trans h1 h2
    | exact ⟨LePreorder.le_trans h2.1 h1.1, LePreorder.le_trans h1.2 h2.2⟩
    | trivial

/-! ## Soundness -/

/-- The payoff: subtyping implies denotational entailment. A value proven under the
    subtype's predicate satisfies the supertype's — so the coercion is sound. -/
theorem Refined.subsumes_sound [LE α] [Zero α] [OfNat α 10000] [LePreorder α]
    {sub sup : Refined α} (h : sub.subsumes sup) (x : α) :
    sub.denote x → sup.denote x := by
  cases sub <;> cases sup <;>
    first
    | exact (h : False).elim
    | (intro hx; exact LePreorder.le_trans h hx)
    | (intro hx; exact LePreorder.le_trans hx h)
    | (intro hx; exact ⟨LePreorder.le_trans h.1 hx.1, LePreorder.le_trans hx.2 h.2⟩)
    | (intro hx; exact hx)

/-! ## Registry tie -/

/-- The registry name a typed refinement carries. -/
def Refined.name : Refined α → Ora.Types.RefinementName
  | .minValue _ => .minValue
  | .maxValue _ => .maxValue
  | .inRange _ _ => .inRange
  | .nonZero => .nonZero
  | .nonZeroAddress => .nonZeroAddress
  | .basisPoints => .basisPoints

/-- Coherence: every refinement modeled here is one the registry classifies
    `hasRuntimeGuard` (this layer models exactly the runtime-guarded six). -/
theorem Refined.name_hasRuntimeGuard (r : Refined α) :
    Ora.Types.HasRuntimeGuard r.name := by cases r <;> rfl

/-! ## Concrete smoke tests (over `Nat`) -/

/-- `MinValue 100` is a subtype of `MinValue 1` (tighter lower bound). -/
example : (Refined.minValue 100).subsumes (Refined.minValue (1 : Nat)) := by
  show (1 : Nat) ≤ 100; decide
/-- …and soundly: a value ≥ 100 is ≥ 1. -/
example (x : Nat) (hx : (Refined.minValue 100).denote x) : (Refined.minValue 1).denote x :=
  Refined.subsumes_sound (sub := .minValue 100) (sup := .minValue 1)
    (show (1 : Nat) ≤ 100 by decide) x hx
/-- `[10, 20] ⊆ [0, 100]` as `InRange` subtyping. -/
example : (Refined.inRange 10 20).subsumes (Refined.inRange 0 (100 : Nat)) := by
  show (0 : Nat) ≤ 10 ∧ (20 : Nat) ≤ 100; decide
/-- Narrowing is NOT subtyping: `MinValue 1` does not subtype `MinValue 100`. -/
example : ¬ (Refined.minValue 1).subsumes (Refined.minValue (100 : Nat)) := by
  show ¬ (100 : Nat) ≤ 1; decide

end Ora.Refine

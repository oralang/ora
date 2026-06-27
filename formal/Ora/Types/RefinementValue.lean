/-
Ora refinement value layer — what the built-in refinements MEAN.

The registry layer (`Refinement.lean`) says which refinements exist and how the
compiler classifies them. This file gives the runtime PREDICATE for each
value-constraining refinement, plus the subsumption lemmas between them.

Carrier: predicates are generic over a carrier `α` with only the structure each
one needs (`LE`, `Zero`, the `10000` literal). They instantiate later at the
concrete Ora value types (`u256`, `i256`, address, …); nothing here commits to a
value representation, and the subsumption lemmas need no order axioms — only
`∧`-projection and definitional unfolding — so this stays mathlib-free.

DECISION — `Exact` / `Scaled` are NOT modeled here. The registry classifies them
`compileTimeOnly = true`, `hasRuntimeGuard = false`: they are compile-time /
type-level metadata (exact representability; a fixed-point scale tag), not a
predicate a runtime value satisfies. They belong to the comptime / type layer
(the `Scaled<T,S>` fixed-point design), not this value layer. So this file models
exactly the six `hasRuntimeGuard` refinements:
`MinValue`, `MaxValue`, `InRange`, `NonZero`, `NonZeroAddress`, `BasisPoints`.
-/

namespace Ora.Refine

universe u
variable {α : Type u}

/-! ## Predicates -/

/-- `MinValue n x` — the value is at least `n`. -/
def MinValue [LE α] (n x : α) : Prop := n ≤ x

/-- `MaxValue n x` — the value is at most `n`. -/
def MaxValue [LE α] (n x : α) : Prop := x ≤ n

/-- `InRange lo hi x` — the value lies in the closed interval `[lo, hi]`. -/
def InRange [LE α] (lo hi x : α) : Prop := lo ≤ x ∧ x ≤ hi

/-- `NonZero x` — the value is not zero. -/
def NonZero [Zero α] (x : α) : Prop := x ≠ 0

/-- `NonZeroAddress a` — an address value is not zero. The address refinement of
    `NonZero`: the same constraint, named for its domain. -/
def NonZeroAddress [Zero α] (a : α) : Prop := a ≠ 0

/-- `BasisPoints x` — a basis-points value lies in `[0, 10000]`. -/
def BasisPoints [LE α] [Zero α] [OfNat α 10000] (x : α) : Prop :=
  (0 : α) ≤ x ∧ x ≤ (10000 : α)

/-! ## Subsumption

    These record the refinement hierarchy: a value proven under a stronger
    refinement satisfies the weaker ones it entails. -/

/-- `InRange` entails its lower bound. -/
theorem inRange_minValue [LE α] {lo hi x : α} (h : InRange lo hi x) : MinValue lo x := h.1

/-- `InRange` entails its upper bound. -/
theorem inRange_maxValue [LE α] {lo hi x : α} (h : InRange lo hi x) : MaxValue hi x := h.2

/-- A non-zero address is non-zero. -/
theorem nonZeroAddress_nonZero [Zero α] {a : α} (h : NonZeroAddress a) : NonZero a := h

/-- Basis points are exactly the range `[0, 10000]`. -/
theorem basisPoints_inRange [LE α] [Zero α] [OfNat α 10000] {x : α}
    (h : BasisPoints x) : InRange (0 : α) (10000 : α) x := h

end Ora.Refine

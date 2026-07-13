/-
Ora refinement bridge — registry ⇄ semantics.

`Refinement.lean` (registry) says which refinements exist and how the compiler
classifies them (`hasRuntimeGuard` / `compileTimeOnly`); `RefinementValue.lean`
(semantics) says what each means as a runtime predicate. This file connects a
resolved registry name to a typed runtime predicate:

  * `RefinementName.runtimeDenotation` — the predicate a registry name imposes on
    a value, given its typed bounds; and
  * the COHERENCE theorems: the registry's `hasRuntimeGuard` classification is
    exactly "has a runtime predicate". The six guarded refinements denote to their
    `Ora.Refine` predicate; `Exact` / `Scaled` (compile-time-only) denote to
    nothing — and nothing the registry marks compile-time-only can ever denote.

Bounds are typed (`α`) here. The type-level string-name and string-bound bridge
is handled by `RefinementTie.lean`, where `Ty.runtimeDenotation?` resolves a
`Ty.refinement` name through the registry and interprets integer bounds through
an explicit carrier function `Nat -> α`.

Remaining boundary: `Exact` and `Scaled` are compile-time-only in this runtime
denotation layer and intentionally denote to `none`; carrier-specific numeric
validity beyond the explicit `Nat -> α` interpretation is not proven here.
-/

import Ora.Types.Refinement
import Ora.Types.RefinementValue

namespace Ora.Types

open Ora.Refine

universe u
variable {α : Type u} [LE α] [Zero α] [OfNat α 10000]

/-- The runtime predicate a refinement imposes on a value, given its typed bounds.
    `none` for the compile-time-only refinements (`Exact` / `Scaled`) and for any
    arity mismatch. -/
def RefinementName.runtimeDenotation : RefinementName → List α → Option (α → Prop)
  | .minValue,       [n]      => some (MinValue n)
  | .maxValue,       [n]      => some (MaxValue n)
  | .inRange,        [lo, hi] => some (InRange lo hi)
  | .nonZero,        []       => some NonZero
  | .nonZeroAddress, []       => some NonZeroAddress
  | .basisPoints,    []       => some BasisPoints
  | _,               _        => none

/-! ## Each guarded refinement denotes to its `Ora.Refine` predicate -/

theorem minValue_denotes (n : α) :
    RefinementName.minValue.runtimeDenotation [n] = some (MinValue n) := rfl

theorem maxValue_denotes (n : α) :
    RefinementName.maxValue.runtimeDenotation [n] = some (MaxValue n) := rfl

theorem inRange_denotes (lo hi : α) :
    RefinementName.inRange.runtimeDenotation [lo, hi] = some (InRange lo hi) := rfl

theorem nonZero_denotes :
    (RefinementName.nonZero.runtimeDenotation ([] : List α)) = some NonZero := rfl

theorem nonZeroAddress_denotes :
    (RefinementName.nonZeroAddress.runtimeDenotation ([] : List α)) = some NonZeroAddress := rfl

theorem basisPoints_denotes :
    (RefinementName.basisPoints.runtimeDenotation ([] : List α)) = some BasisPoints := rfl

/-! ## Coherence with the registry classification -/

/-- The compile-time-only refinements never denote a runtime predicate. -/
theorem exact_no_denotation (bounds : List α) :
    RefinementName.exact.runtimeDenotation bounds = none := by
  cases bounds <;> rfl

theorem scaled_no_denotation (bounds : List α) :
    RefinementName.scaled.runtimeDenotation bounds = none := by
  cases bounds <;> rfl

/-- SOUNDNESS: the denotation never contradicts the registry — anything that
    denotes a runtime predicate is classified `hasRuntimeGuard`. (Equivalently:
    no `compileTimeOnly` refinement can denote.) -/
theorem hasRuntimeGuard_of_denotes {r : RefinementName} {bounds : List α}
    {p : α → Prop} (h : r.runtimeDenotation bounds = some p) : HasRuntimeGuard r := by
  cases r <;>
    simp_all [HasRuntimeGuard, RefinementName.info, RefinementName.runtimeDenotation]

end Ora.Types

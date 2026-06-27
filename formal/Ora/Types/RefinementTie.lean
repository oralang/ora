/-
Ora type system — the refinement type-tie (type ⇄ registry ⇄ value).

Three refinement layers existed but weren't connected:
  * TYPE level    — `Ty.refinement name base args`, where `name : Name` is a STRING
                    and `args : List RefinementArg` carry integer bounds as STRINGS;
  * REGISTRY      — the closed `RefinementName` enum (+ `compilerName`, classification);
  * VALUE         — `Ora.Refine` predicates (`MinValue`, `InRange`, …) and, via
                    `RefinementBridge`, `RefinementName.runtimeDenotation`.

`RefinementBridge` flagged the gap: "interpreting the type-level `RefinementArg`
string literals into concrete values is a separate step (deferred)". This file
closes it.

  * NAME tie: `RefinementName.ofString?` is the inverse of `compilerName` — it
    resolves a refinement type's string name to its registry entry. Round-trip and
    injectivity are proven, so the type-level name and the closed registry agree.
  * VALUE tie: `Ty.runtimeDenotation?` glues all three layers — type name →
    registry (`ofString?`) → value predicate (`runtimeDenotation`), interpreting the
    integer bounds through a carrier interpretation `lit : Nat → α` (the deferred
    string→value step, now an explicit parameter).
  * COHERENCE: the type-level denotation equals the registry name's denotation at
    the interpreted bounds, and can never contradict the registry's `hasRuntimeGuard`
    classification (the capstone, via `RefinementBridge.hasRuntimeGuard_of_denotes`).

Coherence is stated GENERICALLY (over the resolved name + interpreted bounds), so it
never evaluates a concrete string literal — `String.toNat?` is not kernel-reducible
(only `native_decide` evaluates it, which we avoid). Axioms stay at `propext`.
-/

import Ora.Types.TypeEq
import Ora.Types.Refinement
import Ora.Types.RefinementBridge

namespace Ora.Types

open Ora.Refine

/-! ## Name tie — `ofString?` is the inverse of `compilerName` -/

/-- Resolve a compiler spelling back to its registry name (inverse of
    `compilerName`). `none` for any string that is not a built-in refinement. -/
def RefinementName.ofString? (s : String) : Option RefinementName :=
  allRefinementNames.find? (fun r => r.compilerName == s)

/-- Round-trip: every registry name resolves from its own spelling. -/
theorem ofString?_compilerName (r : RefinementName) :
    RefinementName.ofString? r.compilerName = some r := by
  cases r <;> rfl

/-- `compilerName` is injective: distinct registry names have distinct spellings. -/
theorem compilerName_inj {a b : RefinementName}
    (h : a.compilerName = b.compilerName) : a = b := by
  cases a <;> cases b <;> simp_all [RefinementName.compilerName]

/-- Characterization: `ofString?` resolves `s` to `r` exactly when `r` spells `s`. -/
theorem ofString?_eq_some_iff {s : String} {r : RefinementName} :
    RefinementName.ofString? s = some r ↔ r.compilerName = s := by
  constructor
  · intro h; simpa using List.find?_some h
  · intro h; subst h; exact ofString?_compilerName r

/-! ## Type-level tie -/

/-- The registry name a refinement TYPE refers to (via its compiler spelling).
    `none` for non-refinement types or unknown names. -/
def Ty.refinementName? : Ty → Option RefinementName
  | .refinement n _ _ => RefinementName.ofString? n
  | _ => none

/-! ## Value tie — bound interpretation + denotation -/

/-- Read a refinement argument's integer literal as a `Nat` (`typeMarker` carries no
    value). The string→value step `RefinementBridge` left deferred. -/
def RefinementArg.bound? : RefinementArg → Option Nat
  | .integer s => s.toNat?
  | .typeMarker => none

@[simp] theorem RefinementArg.bound?_integer (s) :
    RefinementArg.bound? (.integer s) = s.toNat? := rfl
@[simp] theorem RefinementArg.bound?_typeMarker :
    RefinementArg.bound? .typeMarker = none := rfl

/-- The runtime predicate a refinement TYPE denotes, given an interpretation
    `lit : Nat → α` of its integer bounds into the carrier. Glues all three layers:
    type name → registry (`ofString?`) → value predicate (`runtimeDenotation`). -/
def Ty.runtimeDenotation? {α : Type u} [LE α] [Zero α] [OfNat α 10000]
    (lit : Nat → α) : Ty → Option (α → Prop)
  | .refinement n _ args =>
      match RefinementName.ofString? n with
      | some r => r.runtimeDenotation ((args.filterMap RefinementArg.bound?).map lit)
      | none => none
  | _ => none

/-- GENERIC coherence: a refinement type's denotation IS its registry name's
    denotation at the type's interpreted bounds. The full type→registry→value tie. -/
theorem Ty.runtimeDenotation?_refinement {α} [LE α] [Zero α] [OfNat α 10000]
    (lit : Nat → α) (name : Name) (base : Ty) (args : List RefinementArg) (r : RefinementName)
    (h : RefinementName.ofString? name = some r) :
    Ty.runtimeDenotation? lit (.refinement name base args)
      = r.runtimeDenotation ((args.filterMap RefinementArg.bound?).map lit) := by
  simp only [Ty.runtimeDenotation?, h]

/-- Non-refinement types (and unresolved names) denote nothing. -/
theorem Ty.runtimeDenotation?_not_refinement {α} [LE α] [Zero α] [OfNat α 10000]
    (lit : Nat → α) {t : Ty} (h : Ty.refinementName? t = none) :
    Ty.runtimeDenotation? lit t = none := by
  cases t <;> simp_all [Ty.runtimeDenotation?, Ty.refinementName?]

/-! ## Concrete coherence (the boundless refinements; rfl-clean, no string parse) -/

variable {α : Type u} [LE α] [Zero α] [OfNat α 10000] (lit : Nat → α) (base : Ty)

theorem nonZero_type_denotes :
    Ty.runtimeDenotation? lit (.refinement "NonZero" base []) = some NonZero := rfl
theorem nonZeroAddress_type_denotes :
    Ty.runtimeDenotation? lit (.refinement "NonZeroAddress" base []) = some NonZeroAddress := rfl
theorem basisPoints_type_denotes :
    Ty.runtimeDenotation? lit (.refinement "BasisPoints" base []) = some BasisPoints := rfl
theorem exact_type_no_denotation :
    Ty.runtimeDenotation? lit (.refinement "Exact" base []) = none := rfl
theorem scaled_type_no_denotation :
    Ty.runtimeDenotation? lit (.refinement "Scaled" base []) = none := rfl
theorem unknown_type_no_denotation :
    Ty.runtimeDenotation? lit (.refinement "Bogus" base []) = none := rfl

/-! ## Capstone — type-level denotation never contradicts the registry -/

/-- If a refinement TYPE denotes a runtime predicate, its name resolves to a
    registry refinement classified `hasRuntimeGuard`. (Via `RefinementBridge`'s
    `hasRuntimeGuard_of_denotes`.) -/
theorem Ty.hasRuntimeGuard_of_runtimeDenotation {α} [LE α] [Zero α] [OfNat α 10000]
    (lit : Nat → α) {t : Ty} {p : α → Prop} (h : Ty.runtimeDenotation? lit t = some p) :
    ∃ r, Ty.refinementName? t = some r ∧ HasRuntimeGuard r := by
  cases t with
  | refinement name base args =>
      simp only [Ty.runtimeDenotation?] at h
      cases hr : RefinementName.ofString? name with
      | none => rw [hr] at h; exact absurd h (by simp)
      | some r => rw [hr] at h; exact ⟨r, hr, hasRuntimeGuard_of_denotes h⟩
  | _ => exact absurd h (by simp [Ty.runtimeDenotation?])

end Ora.Types

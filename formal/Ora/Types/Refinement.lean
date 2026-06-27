/-
Ora type system — closed refinement registry.

This file defines the trusted representation of Ora's built-in refinement
lexicon. It is intentionally a closed enum with named metadata fields, rather
than a string/boolean table, so invalid refinement names are unrepresentable and
human review does not depend on remembering positional columns.

This is the registry layer, not the predicate/value layer. It says which
refinements exist and how the compiler classifies them. Later value semantics
will define predicates such as `MinValue n x`, `NonZero x`, etc.
-/

namespace Ora.Types

/-- Closed set of built-in refinement names. -/
inductive RefinementName where
  | minValue
  | maxValue
  | inRange
  | nonZeroAddress
  | nonZero
  | basisPoints
  | exact
  | scaled
  deriving Repr, DecidableEq

/-- Reviewable metadata for one built-in refinement. -/
structure RefinementInfo where
  hasRuntimeGuard : Bool
  compileTimeOnly : Bool
  hasNativeMlirType : Bool
  pathForm : Bool
  boundsBacked : Bool
  deriving Repr, DecidableEq

/--
Canonical metadata for each built-in refinement.

Keep this definition readable and explicit. The spec facts project this
registry into the data-only sync tables.
-/
def RefinementName.info : RefinementName → RefinementInfo
  | .minValue =>
      { hasRuntimeGuard := true,
        compileTimeOnly := false,
        hasNativeMlirType := true,
        pathForm := false,
        boundsBacked := true }
  | .maxValue =>
      { hasRuntimeGuard := true,
        compileTimeOnly := false,
        hasNativeMlirType := true,
        pathForm := false,
        boundsBacked := true }
  | .inRange =>
      { hasRuntimeGuard := true,
        compileTimeOnly := false,
        hasNativeMlirType := true,
        pathForm := false,
        boundsBacked := true }
  | .nonZeroAddress =>
      { hasRuntimeGuard := true,
        compileTimeOnly := false,
        hasNativeMlirType := true,
        pathForm := true,
        boundsBacked := false }
  | .nonZero =>
      { hasRuntimeGuard := true,
        compileTimeOnly := false,
        hasNativeMlirType := false,
        pathForm := false,
        boundsBacked := true }
  | .basisPoints =>
      { hasRuntimeGuard := true,
        compileTimeOnly := false,
        hasNativeMlirType := false,
        pathForm := false,
        boundsBacked := true }
  | .exact =>
      { hasRuntimeGuard := false,
        compileTimeOnly := true,
        hasNativeMlirType := true,
        pathForm := false,
        boundsBacked := false }
  | .scaled =>
      { hasRuntimeGuard := false,
        compileTimeOnly := true,
        hasNativeMlirType := true,
        pathForm := false,
        boundsBacked := false }

/-- Compiler spelling for each refinement name. -/
def RefinementName.compilerName : RefinementName → String
  | .minValue => "MinValue"
  | .maxValue => "MaxValue"
  | .inRange => "InRange"
  | .nonZeroAddress => "NonZeroAddress"
  | .nonZero => "NonZero"
  | .basisPoints => "BasisPoints"
  | .exact => "Exact"
  | .scaled => "Scaled"

/-- Registry order, matching the compiler registry. -/
def allRefinementNames : List RefinementName :=
  [.minValue, .maxValue, .inRange, .nonZeroAddress,
   .nonZero, .basisPoints, .exact, .scaled]

abbrev HasRuntimeGuard (r : RefinementName) : Prop :=
  ((RefinementName.info r).hasRuntimeGuard) = true

abbrev CompileTimeOnly (r : RefinementName) : Prop :=
  ((RefinementName.info r).compileTimeOnly) = true

abbrev HasNativeMlirType (r : RefinementName) : Prop :=
  ((RefinementName.info r).hasNativeMlirType) = true

abbrev IsPathForm (r : RefinementName) : Prop :=
  ((RefinementName.info r).pathForm) = true

abbrev BoundsBacked (r : RefinementName) : Prop :=
  ((RefinementName.info r).boundsBacked) = true

theorem minValue_has_runtime_guard :
    HasRuntimeGuard .minValue := rfl

theorem exact_is_compile_time_only :
    CompileTimeOnly .exact := rfl

theorem nonZeroAddress_is_path_form :
    IsPathForm .nonZeroAddress := rfl

theorem nonZero_is_bounds_backed :
    BoundsBacked .nonZero := rfl

theorem exact_has_no_runtime_guard :
    ¬ (HasRuntimeGuard .exact) := fun h => nomatch h

end Ora.Types

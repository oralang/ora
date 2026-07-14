/-
`check-formal-sync` — the TRUSTED checks.

Each theorem proves, by `decide` (kernel-checked, NOT `native_decide`), that a
fact the compiler emitted in `Ora/Generated/CompilerSnapshot.lean` matches the
spec's expectation in `Ora/Spec/Facts.lean`. If the compiler drifts from the
spec, the relevant `decide` fails and `lake build` goes red.

This file is hand-written and trusted; the snapshot it checks is data-only.
-/

import Ora.Spec.Facts
import Ora.Generated.CompilerSnapshot

namespace Ora.Sync

open Ora.Spec Ora.Generated Ora.Types

/-! ## 1. Spellable scalar builtins — no missing, no extra -/

theorem builtin_types_match : compilerBuiltinTypeIds = expectedBuiltinTypeIds := by decide

theorem builtin_type_comptime_ids_match :
    compilerBuiltinTypeComptimeIds = expectedBuiltinTypeComptimeIds := by decide

theorem obligation_semantics_u256_type_id_pinned :
    ("u256", expectedCompilerTypeIdU256) ∈ compilerBuiltinTypeComptimeIds := by decide

theorem obligation_semantics_i256_type_id_pinned :
    ("i256", expectedCompilerTypeIdI256) ∈ compilerBuiltinTypeComptimeIds := by decide

theorem obligation_semantics_bool_type_id_pinned :
    ("bool", expectedCompilerTypeIdBool) ∈ compilerBuiltinTypeComptimeIds := by decide

/-! ## 2. Integer widths — no missing, no extra -/

theorem uint_widths_match : compilerUIntWidths = expectedUIntWidths := by decide
theorem sint_widths_match : compilerSIntWidths = expectedSIntWidths := by decide

/-! ## 3. `i160` is absent from the signed widths -/

theorem i160_absent : 160 ∉ compilerSIntWidths := by decide

/-! ## 4. `address` is a distinct primitive from `u160`

    This is a modeling fact (the compiler does not "emit" it), re-exported from
    `Prim.lean` so it sits with the other sync guarantees. -/

theorem address_ne_u160 : PrimTy.address ≠ PrimTy.int (.uint .w160) :=
  Ora.Types.address_ne_u160

/-! ## Fixed-bytes bounds -/

theorem fixed_bytes_bounds_match :
    compilerFixedBytesMin = expectedFixedBytesMin ∧
    compilerFixedBytesMax = expectedFixedBytesMax := by decide

/-! ## TypeKind universe — complete accounting of all 28 -/

theorem typekinds_match : compilerTypeKinds = expectedTypeKinds := by decide

/-- The excluded internal kinds are a subset of the universe — so adding/removing
    an exclusion is a deliberate, checked change. -/
theorem excluded_typekinds_are_real : ∀ k ∈ excludedTypeKinds, k ∈ expectedTypeKinds := by decide

/-! ## Closed refinement registry — checked via self-describing per-property name lists -/

theorem refinement_names_match :
    compilerRefinementNames = expectedRefinementNames := by decide

theorem runtime_guard_refinement_names_match :
    compilerRuntimeGuardRefinementNames = expectedRuntimeGuardRefinementNames := by decide

theorem compile_time_only_refinement_names_match :
    compilerCompileTimeOnlyRefinementNames = expectedCompileTimeOnlyRefinementNames := by decide

theorem native_mlir_refinement_names_match :
    compilerNativeMlirRefinementNames = expectedNativeMlirRefinementNames := by decide

theorem path_form_refinement_names_match :
    compilerPathFormRefinementNames = expectedPathFormRefinementNames := by decide

theorem bounds_backed_refinement_names_match :
    compilerBoundsBackedRefinementNames = expectedBoundsBackedRefinementNames := by decide

/-! ## 5. Regions + the assignability table match the formal spec -/

theorem regions_match : compilerRegions = expectedRegions := by decide

theorem region_table_matches : compilerRegionTable = expectedRegionTable := by decide

end Ora.Sync

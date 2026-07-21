/-
Kernel check for the compiler-emitted scalar-loop support matrix.

The generated module contains data only. This module reconstructs every loop
shape, pins the compiler's named rejection reason, and checks that Lean
denotation is total exactly for the shapes that reach the denotation boundary.
Query owner/id mismatches are rejected by the compiler target gate while the
underlying summary remains denotable, so those two rows intentionally separate
target eligibility from summary denotation.
-/

import Ora.Loop.Denotation
import Ora.Generated.LoopTotalitySnapshot

namespace Ora.LoopTotalitySync

open Ora.Obligation Ora.Loop Ora.Generated

abbrev RawRow := String × String × Bool × String × Bool

def loopId : FreeVarId := { file_id := 10, pattern_id := 1 }
def otherId : FreeVarId := { file_id := 10, pattern_id := 2 }

def baseManifest : Manifest := {
  terms := [
    .intLit { value := "0", ty := some (.compilerTypeId 6) },
    .boolLit true
  ]
}

def typeIdForMutation : String → Nat
  | "supported_u8" => 1
  | "supported_u16" => 2
  | "supported_u32" => 3
  | "supported_u64" => 4
  | "supported_u128" => 5
  | "supported_u160" => 18
  | "supported_i8" => 7
  | "supported_i16" => 8
  | "supported_i32" => 9
  | "supported_i64" => 10
  | "supported_i128" => 11
  | "supported_i256" => 12
  | "unregistered_integer_variable" => 999
  | _ => 6

def manifestForMutation (mutation : String) : Manifest := {
  terms := [
    .intLit { value := "0", ty := some (.compilerTypeId (typeIdForMutation mutation)) },
    .boolLit true
  ]
}

def baseVariable : VariableRow := {
  index := 0, id := loopId, name := "i", ty := .compilerTypeId 6
}

def baseAssignment : StepAssignmentRow := {
  variableIndex := 0, target := loopId, value := .term 0
}

def baseSummary : SummaryRow := {
  id := 100
  owner := "run"
  kind := .scfWhile
  variables := [baseVariable]
  init := [.term 0]
  guard := some (.term 1)
  invariants := [.term 1]
  step := [baseAssignment]
  bodySafety := [.term 1]
  post := [.term 1]
}

def summaryForMutation : String → SummaryRow
  | "supported_u8" => { baseSummary with variables := [{ baseVariable with ty := .compilerTypeId 1 }] }
  | "supported_u16" => { baseSummary with variables := [{ baseVariable with ty := .compilerTypeId 2 }] }
  | "supported_u32" => { baseSummary with variables := [{ baseVariable with ty := .compilerTypeId 3 }] }
  | "supported_u64" => { baseSummary with variables := [{ baseVariable with ty := .compilerTypeId 4 }] }
  | "supported_u128" => { baseSummary with variables := [{ baseVariable with ty := .compilerTypeId 5 }] }
  | "supported_u160" => { baseSummary with variables := [{ baseVariable with ty := .compilerTypeId 18 }] }
  | "supported_i8" => { baseSummary with variables := [{ baseVariable with ty := .compilerTypeId 7 }] }
  | "supported_i16" => { baseSummary with variables := [{ baseVariable with ty := .compilerTypeId 8 }] }
  | "supported_i32" => { baseSummary with variables := [{ baseVariable with ty := .compilerTypeId 9 }] }
  | "supported_i64" => { baseSummary with variables := [{ baseVariable with ty := .compilerTypeId 10 }] }
  | "supported_i128" => { baseSummary with variables := [{ baseVariable with ty := .compilerTypeId 11 }] }
  | "supported_i256" => { baseSummary with variables := [{ baseVariable with ty := .compilerTypeId 12 }] }
  | "supported_for" => { baseSummary with kind := .scfFor }
  | "storage_write" => { baseSummary with unsupportedReasons := ["loop_has_storage_write"] }
  | "storage_read" => { baseSummary with unsupportedReasons := ["loop_has_storage_read"] }
  | "call" => { baseSummary with unsupportedReasons := ["loop_has_call"] }
  | "external_call" => { baseSummary with unsupportedReasons := ["loop_has_external_call"] }
  | "resource_operation" => { baseSummary with unsupportedReasons := ["loop_has_resource_operation"] }
  | "break_or_continue" => { baseSummary with unsupportedReasons := ["loop_has_break_or_continue"] }
  | "error_control_flow" => { baseSummary with unsupportedReasons := ["loop_has_error_control_flow"] }
  | "nested_loop" => { baseSummary with unsupportedReasons := ["loop_has_nested_loop"] }
  | "branching_body" => { baseSummary with unsupportedReasons := ["loop_has_branching_body"] }
  | "missing_guard" => { baseSummary with guard := none, unsupportedReasons := ["loop_guard_missing"] }
  | "missing_invariant" => { baseSummary with invariants := [], unsupportedReasons := ["loop_invariant_missing"] }
  | "missing_body_safety" => { baseSummary with bodySafety := [], unsupportedReasons := ["loop_body_safety_missing"] }
  | "unsupported_kind" => { baseSummary with kind := .other, unsupportedReasons := ["loop_kind_unsupported"] }
  | "unregistered_integer_variable" =>
      { baseSummary with variables := [{ baseVariable with ty := .compilerTypeId 999 }] }
  | "bad_update_target" =>
      { baseSummary with step := [{ baseAssignment with target := otherId }] }
  | "unsupported_formula" => { baseSummary with guard := some (.term 0) }
  | "identity_collision" => { baseSummary with contextVariables := [baseVariable] }
  | _ => baseSummary

def expectedCompilerSupport : String → Bool
  | "supported_while" | "supported_for" |
    "supported_u8" | "supported_u16" | "supported_u32" | "supported_u64" |
    "supported_u128" | "supported_u160" |
    "supported_i8" | "supported_i16" | "supported_i32" | "supported_i64" |
    "supported_i128" | "supported_i256" => true
  | _ => false

def expectedReason : String → String
  | "supported_while" | "supported_for" |
    "supported_u8" | "supported_u16" | "supported_u32" | "supported_u64" |
    "supported_u128" | "supported_u160" |
    "supported_i8" | "supported_i16" | "supported_i32" | "supported_i64" |
    "supported_i128" | "supported_i256" => ""
  | "storage_write" => "loop_has_storage_write"
  | "storage_read" => "loop_has_storage_read"
  | "call" => "loop_has_call"
  | "external_call" => "loop_has_external_call"
  | "resource_operation" => "loop_has_resource_operation"
  | "break_or_continue" => "loop_has_break_or_continue"
  | "error_control_flow" => "loop_has_error_control_flow"
  | "nested_loop" => "loop_has_nested_loop"
  | "branching_body" => "loop_has_branching_body"
  | "missing_guard" => "loop_guard_missing"
  | "missing_invariant" => "loop_invariant_missing"
  | "missing_body_safety" => "loop_body_safety_missing"
  | "unsupported_kind" => "loop_kind_unsupported"
  | "unregistered_integer_variable" => "loop_variable_not_registered_integer"
  | "bad_update_target" => "loop_update_target_not_loop_variable"
  | "unsupported_formula" => "loop_formula_unsupported"
  | "identity_collision" => "loop_identity_missing"
  | "query_owner_mismatch" | "query_id_mismatch" => "loop_summary_query_mismatch"
  | _ => "unknown_fixture"

def expectedLeanDenotable : String → Bool
  | "supported_while" | "supported_for" |
    "supported_u8" | "supported_u16" | "supported_u32" | "supported_u64" |
    "supported_u128" | "supported_u160" |
    "supported_i8" | "supported_i16" | "supported_i32" | "supported_i64" |
    "supported_i128" | "supported_i256" |
    "query_owner_mismatch" | "query_id_mismatch" => true
  | _ => false

def rowMatches : RawRow → Bool
  | (name, mutation, compilerSupported, reason, leanDenotable) =>
      name == mutation &&
      compilerSupported == expectedCompilerSupport mutation &&
      reason == expectedReason mutation &&
      leanDenotable == expectedLeanDenotable mutation &&
      (denoteLoopSummary? (manifestForMutation mutation) Env.empty
        (summaryForMutation mutation)).isSome ==
        leanDenotable

theorem loop_totality_matrix_matches : loopTotalityRows.all rowMatches = true := by
  decide

def unsupportedSummaryFails : RawRow → Bool
  | (_, mutation, _, _, leanDenotable) =>
      leanDenotable ||
        (denoteLoopSummary? (manifestForMutation mutation) Env.empty
          (summaryForMutation mutation)).isNone

theorem negative_shapes_fail_closed :
    loopTotalityRows.all unsupportedSummaryFails = true := by
  decide

end Ora.LoopTotalitySync

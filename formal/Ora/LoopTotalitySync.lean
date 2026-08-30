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

def baseBinding : StateBindingRow := {
  variableIndex := 0, target := loopId, sourceKey := "loop-state:0", value := .term 0
}

def baseSafetyTimelineEvent : TimelineEventRow := {
  id := 1
  structuralKey := "body/ora.assert:0"
  payload := .safetyCheckpoint { formulas := [{ formula := .term 1 }] }
}

def baseBackedgeTimeline : List TimelineEventRow := [{
  id := 0
  structuralKey := "body/straight-line-yield/path"
  payload := .pathAssumption { formulas := [{ formula := .term 1 }] }
}, baseSafetyTimelineEvent]

def baseExitTimeline : List TimelineEventRow := [{
  id := 0
  structuralKey := "guard/false-successor/path"
  payload := .pathAssumption { formulas := [{ formula := .term 1 }] }
}]

def baseBackedge : EdgeRow := {
  id := 0
  structuralKey := "body/straight-line-yield"
  timeline := baseBackedgeTimeline
  nextState := [baseBinding]
  target := .backedge
}

def baseExit : EdgeRow := {
  id := 1
  structuralKey := "guard/false-successor"
  timeline := baseExitTimeline
  nextState := [baseBinding]
  target := .successor
}

def baseSummary : SummaryRow := {
  id := 100
  owner := "run"
  kind := .scfWhile
  variables := [baseVariable]
  init := [.term 0]
  invariants := [.term 1]
  edges := [baseBackedge, baseExit]
  projectionComplete := true
  post := [.term 1]
}

def timelineReadId : FreeVarId := { file_id := 10, pattern_id := 90 }
def timelineForeignId : FreeVarId := { file_id := 10, pattern_id := 91 }

def timelineLocation : StorageLocationExprRow := {
  root := "counter", valueTy := .compilerTypeId 6
}

def timelineDependency : SummaryDependencyRow := {
  id := 0
  structuralKey := "body/child"
  summaryId := 2
  loopVcSha256 := "0000000000000000000000000000000000000000000000000000000000000000"
  target := .successor
  writeFootprintComplete := true
}

def timelinePathEvent : TimelineEventRow := {
  id := 0
  structuralKey := "body/path"
  payload := .pathAssumption { formulas := [{ formula := .term 0 }] }
}

def timelineReadEvent : TimelineEventRow := {
  id := 1
  structuralKey := "body/read"
  payload := .storageRead { result := timelineReadId, location := timelineLocation }
}

def timelineWriteEvent : TimelineEventRow := {
  id := 2
  structuralKey := "body/write"
  payload := .storageWrite {
    location := timelineLocation
    value := { formula := .term 0, readResults := [timelineReadId] }
  }
}

def timelineSafetyEvent : TimelineEventRow := {
  id := 3
  structuralKey := "body/safety"
  payload := .safetyCheckpoint { formulas := [{ formula := .term 0 }] }
}

def timelineDependencyEvent : TimelineEventRow := {
  id := 4
  structuralKey := "body/child"
  payload := .summaryDependency timelineDependency
}

def timelineBaseEdge : EdgeRow := {
  id := 0
  structuralKey := "body"
  timeline := [timelinePathEvent, timelineReadEvent, timelineWriteEvent,
    timelineSafetyEvent, timelineDependencyEvent]
  nextState := [{
    variableIndex := 0
    target := loopId
    sourceKey := "loop-state:0"
    value := .term 0
    readResults := [timelineReadId]
  }]
  target := .backedge
}

def timelineEdgeForMutation : String → EdgeRow
  | "timeline_missing" => { timelineBaseEdge with timeline := [] }
  | "event_order_invalid" =>
      { timelineBaseEdge with timeline := [timelinePathEvent,
          { timelineReadEvent with id := 7 }, timelineWriteEvent,
          timelineSafetyEvent, timelineDependencyEvent] }
  | "event_identity_missing" =>
      { timelineBaseEdge with timeline := [timelinePathEvent,
          { timelineReadEvent with structuralKey := "" }, timelineWriteEvent,
          timelineSafetyEvent, timelineDependencyEvent] }
  | "event_identity_duplicate" =>
      { timelineBaseEdge with timeline := [timelinePathEvent,
          { timelineReadEvent with structuralKey := timelinePathEvent.structuralKey },
          timelineWriteEvent, timelineSafetyEvent, timelineDependencyEvent] }
  | "path_missing" =>
      { timelineBaseEdge with timeline := [{ timelineSafetyEvent with id := 0 }] }
  | "storage_root_missing" =>
      { timelineBaseEdge with timeline := [timelinePathEvent,
          { timelineReadEvent with payload := .storageRead {
              result := timelineReadId
              location := { timelineLocation with root := "" }
            } },
          timelineWriteEvent, timelineSafetyEvent, timelineDependencyEvent] }
  | "storage_projection_invalid" =>
      { timelineBaseEdge with timeline := [timelinePathEvent,
          { timelineReadEvent with payload := .storageRead {
              result := timelineReadId
              location := { timelineLocation with projections := [.field ""] }
            } },
          timelineWriteEvent, timelineSafetyEvent, timelineDependencyEvent] }
  | "storage_type_invalid" =>
      { timelineBaseEdge with timeline := [timelinePathEvent,
          { timelineReadEvent with payload := .storageRead {
              result := timelineReadId
              location := { timelineLocation with valueTy := .compilerTypeId 0 }
            } },
          timelineWriteEvent, timelineSafetyEvent, timelineDependencyEvent] }
  | "read_identity_duplicate" =>
      { timelineBaseEdge with timeline := [timelinePathEvent,
          { timelineReadEvent with payload := .storageRead {
              result := loopId, location := timelineLocation
            } },
          timelineWriteEvent, timelineSafetyEvent, timelineDependencyEvent] }
  | "read_identity_foreign" =>
      { timelineBaseEdge with timeline := [timelinePathEvent, timelineReadEvent,
          { timelineWriteEvent with payload := .storageWrite {
              location := timelineLocation
              value := { formula := .term 0, readResults := [timelineForeignId] }
            } },
          timelineSafetyEvent, timelineDependencyEvent] }
  | "read_identity_forward" =>
      { timelineBaseEdge with timeline := [timelinePathEvent,
          { timelineWriteEvent with id := 1 }, { timelineReadEvent with id := 2 },
          timelineSafetyEvent, timelineDependencyEvent] }
  | "child_footprint_missing" =>
      { timelineBaseEdge with timeline := [timelinePathEvent, timelineReadEvent,
          timelineWriteEvent, timelineSafetyEvent,
          { timelineDependencyEvent with payload := .summaryDependency {
              timelineDependency with writeFootprintComplete := false
            } }] }
  | "dependency_order_invalid" =>
      { timelineBaseEdge with timeline := [timelinePathEvent, timelineReadEvent,
          timelineWriteEvent, timelineSafetyEvent,
          { timelineDependencyEvent with payload := .summaryDependency {
              timelineDependency with id := 1
            } }] }
  | "dependency_target_unsupported" =>
      { timelineBaseEdge with timeline := [timelinePathEvent, timelineReadEvent,
          timelineWriteEvent, timelineSafetyEvent,
          { timelineDependencyEvent with payload := .summaryDependency {
              timelineDependency with target := .backedge
            } }] }
  | "body_safety_missing" =>
      { timelineBaseEdge with timeline := [timelinePathEvent, timelineReadEvent,
          timelineWriteEvent,
          { timelineSafetyEvent with payload := .safetyCheckpoint { formulas := [] } },
          timelineDependencyEvent] }
  | _ => timelineBaseEdge

def timelineMutationNames : List String := [
  "supported", "timeline_missing", "event_order_invalid",
  "event_identity_missing", "event_identity_duplicate", "path_missing",
  "storage_root_missing", "storage_projection_invalid",
  "storage_type_invalid", "read_identity_duplicate", "read_identity_foreign",
  "read_identity_forward", "child_footprint_missing",
  "dependency_order_invalid", "dependency_target_unsupported",
  "body_safety_missing"
]

def timelineReasonForMutation (mutation : String) : String :=
  match edgeTimelineReason? [loopId] (timelineEdgeForMutation mutation) with
  | some reason => reason.name
  | none => ""

def timelineValidationRowMatches : String × String → Bool
  | (mutation, compilerReason) =>
      timelineMutationNames.contains mutation &&
        compilerReason == timelineReasonForMutation mutation

theorem loop_timeline_event_tags_pinned :
    loopTimelineEventTags == ["path_assumption", "safety_checkpoint",
      "storage_read", "storage_write", "summary_dependency"] := by
  decide

theorem loop_storage_projection_tags_pinned :
    loopStorageProjectionTags == ["field", "fixed_index", "mapping_key"] := by
  decide

theorem loop_timeline_validation_mutations_pinned :
    loopTimelineValidationRows.map Prod.fst == timelineMutationNames := by
  decide

theorem loop_timeline_validation_matches :
    loopTimelineValidationRows.all timelineValidationRowMatches = true := by
  decide

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
  | "missing_guard" =>
      { baseSummary with
        edges := [{ baseBackedge with timeline := [baseSafetyTimelineEvent] }, baseExit]
        unsupportedReasons := ["loop_guard_missing"] }
  | "missing_invariant" => { baseSummary with invariants := [], unsupportedReasons := ["loop_invariant_missing"] }
  | "missing_body_safety" =>
      { baseSummary with
        edges := [{ baseBackedge with timeline := baseBackedgeTimeline.take 1 }, baseExit]
        unsupportedReasons := ["loop_body_safety_missing"] }
  | "unsupported_kind" => { baseSummary with kind := .other, unsupportedReasons := ["loop_kind_unsupported"] }
  | "unregistered_integer_variable" =>
      { baseSummary with variables := [{ baseVariable with ty := .compilerTypeId 999 }] }
  | "bad_update_target" =>
      { baseSummary with
        edges := [
          { baseBackedge with nextState := [{ baseBinding with target := otherId }] },
          baseExit
        ] }
  | "unsupported_formula" =>
      { baseSummary with
        edges := [
          { baseBackedge with
            timeline := [{
              id := 0
              structuralKey := "body/straight-line-yield/path"
              payload := .pathAssumption { formulas := [{ formula := .term 0 }] }
            }, baseSafetyTimelineEvent]
          },
          baseExit
        ]
      }
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
  | "bad_update_target" => "loop_state_binding_foreign"
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

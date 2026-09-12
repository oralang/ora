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
      "storage_read", "indexed_region_read", "storage_write",
      "summary_dependency"] := by
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

inductive IndexedRegionMutation
  | supported
  | readMissingShapeValid
  | readDuplicated
  | readOrderInvalid
  | checkpointMissing
  | checkpointAfterRead
  | checkpointWrongEvent
  | checkpointNotDynamic
  | ownerChangedShapeValid
  | indexChangedShapeValid
  | regionChangedShapeValid
  | elementTypeChangedShapeValid
  | elementTypeInvalid
  | readIdentityDuplicate
  | readIdentityCrossDomain
  | indexReadForeign
  | indexReadForward
  deriving DecidableEq

def IndexedRegionMutation.name : IndexedRegionMutation → String
  | .supported => "supported"
  | .readMissingShapeValid => "read_missing_shape_valid"
  | .readDuplicated => "read_duplicated"
  | .readOrderInvalid => "read_order_invalid"
  | .checkpointMissing => "checkpoint_missing"
  | .checkpointAfterRead => "checkpoint_after_read"
  | .checkpointWrongEvent => "checkpoint_wrong_event"
  | .checkpointNotDynamic => "checkpoint_not_dynamic"
  | .ownerChangedShapeValid => "owner_changed_shape_valid"
  | .indexChangedShapeValid => "index_changed_shape_valid"
  | .regionChangedShapeValid => "region_changed_shape_valid"
  | .elementTypeChangedShapeValid => "element_type_changed_shape_valid"
  | .elementTypeInvalid => "element_type_invalid"
  | .readIdentityDuplicate => "read_identity_duplicate"
  | .readIdentityCrossDomain => "read_identity_cross_domain"
  | .indexReadForeign => "index_read_foreign"
  | .indexReadForward => "index_read_forward"

def indexedRegionMutations : List IndexedRegionMutation := [
  .supported, .readMissingShapeValid, .readDuplicated, .readOrderInvalid,
  .checkpointMissing, .checkpointAfterRead, .checkpointWrongEvent,
  .checkpointNotDynamic, .ownerChangedShapeValid, .indexChangedShapeValid,
  .regionChangedShapeValid, .elementTypeChangedShapeValid, .elementTypeInvalid,
  .readIdentityDuplicate,
  .readIdentityCrossDomain, .indexReadForeign, .indexReadForward
]

def indexedRegionForwardId : FreeVarId := { file_id := 10, pattern_id := 91 }

def indexedRegionOwnerPattern (mutation : IndexedRegionMutation) : Nat :=
  if mutation == .ownerChangedShapeValid then 8 else 7

def indexedRegionIndexTerm (mutation : IndexedRegionMutation) : Nat :=
  if mutation == .indexChangedShapeValid then 1 else 0

def indexedRegionCheckpoint (dynamic : Bool := true) : TimelineEventRow := {
  id := 0
  structuralKey := "body/dynamic-check"
  payload := .safetyCheckpoint {
    formulas := [{ formula := .term 0 }]
    dynamicIndexBounds := dynamic
  }
}

def indexedRegionRead (mutation : IndexedRegionMutation) : TimelineEventRow := {
  id := 1
  structuralKey := "body/region-read"
  payload := .indexedRegionRead {
    result := if mutation == .readIdentityDuplicate then loopId else timelineReadId
    region := if mutation == .regionChangedShapeValid then .memory else .calldata
    owner := { file_id := 10, pattern_id := indexedRegionOwnerPattern mutation }
    index := {
      formula := .term (indexedRegionIndexTerm mutation)
      readResults := match mutation with
        | .indexReadForeign => [timelineForeignId]
        | .indexReadForward => [indexedRegionForwardId]
        | _ => []
    }
    elementTy := match mutation with
      | .elementTypeChangedShapeValid => .compilerTypeId 1
      | .elementTypeInvalid => .compilerTypeId 0
      | _ => .compilerTypeId 6
    checkpointStructuralKey := if mutation == .checkpointMissing then ""
      else "body/dynamic-check"
  }
}

def indexedRegionForwardRead : TimelineEventRow := {
  id := 2
  structuralKey := "body/forward-read"
  payload := .indexedRegionRead {
    result := indexedRegionForwardId
    region := .calldata
    owner := { file_id := 10, pattern_id := 7 }
    index := { formula := .term 0 }
    elementTy := .compilerTypeId 6
    checkpointStructuralKey := "body/dynamic-check"
  }
}

def indexedRegionWrongCheckpointEvent : TimelineEventRow := {
  id := 0
  structuralKey := "body/dynamic-check"
  payload := .storageRead {
    result := indexedRegionForwardId
    location := timelineLocation
  }
}

def indexedRegionTimeline : IndexedRegionMutation → List TimelineEventRow
  | .readMissingShapeValid => [indexedRegionCheckpoint]
  | .readDuplicated => [indexedRegionCheckpoint, indexedRegionRead .readDuplicated,
      { indexedRegionRead .readDuplicated with id := 2 }]
  | .readOrderInvalid => [indexedRegionCheckpoint,
      { indexedRegionRead .readOrderInvalid with id := 7 }]
  | .checkpointAfterRead => [
      { indexedRegionRead .checkpointAfterRead with id := 0 },
      { indexedRegionCheckpoint with id := 1 }]
  | .checkpointWrongEvent => [indexedRegionWrongCheckpointEvent,
      indexedRegionRead .checkpointWrongEvent]
  | .checkpointNotDynamic => [indexedRegionCheckpoint false,
      indexedRegionRead .checkpointNotDynamic]
  | .readIdentityCrossDomain => [
      { timelineReadEvent with id := 0, structuralKey := "body/storage-read" },
      indexedRegionRead .readIdentityCrossDomain]
  | .indexReadForward => [indexedRegionCheckpoint,
      indexedRegionRead .indexReadForward, indexedRegionForwardRead]
  | mutation => [indexedRegionCheckpoint, indexedRegionRead mutation]

def indexedRegionReasonForMutation (mutation : IndexedRegionMutation) : String :=
  match preheaderTimelineReason? [loopId] (indexedRegionTimeline mutation) with
  | some reason => reason.name
  | none => ""

def expectedIndexedRegionValidationRows : List (String × String) :=
  indexedRegionMutations.map fun mutation =>
    (mutation.name, indexedRegionReasonForMutation mutation)

theorem loop_indexed_region_validation_mutations_pinned :
    loopIndexedRegionValidationRows.map Prod.fst ==
      indexedRegionMutations.map (·.name) := by
  decide

theorem loop_indexed_region_validation_matches :
    loopIndexedRegionValidationRows == expectedIndexedRegionValidationRows := by
  decide

inductive FixedIndexMutation
  | supported
  | extentInvalid
  | checkpointMissing
  | checkpointLate
  | checkpointWrongShape
  | checkpointWrongIndex
  | checkpointWrongType
  | checkpointInconsistent
  | checkpointNoncanonicalLiteral
  deriving DecidableEq

def FixedIndexMutation.name : FixedIndexMutation → String
  | .supported => "supported"
  | .extentInvalid => "extent_invalid"
  | .checkpointMissing => "checkpoint_missing"
  | .checkpointLate => "checkpoint_late"
  | .checkpointWrongShape => "checkpoint_wrong_shape"
  | .checkpointWrongIndex => "checkpoint_wrong_index"
  | .checkpointWrongType => "checkpoint_wrong_type"
  | .checkpointInconsistent => "checkpoint_inconsistent"
  | .checkpointNoncanonicalLiteral => "checkpoint_noncanonical_literal"

inductive FixedIndexTimelineShape
  | normal
  | missing
  | late

inductive FixedIndexFormulaShape
  | canonical
  | wrongOperator
  | wrongIndex
  | wrongType

structure FixedIndexFixtureConfig where
  literalValue : String := "4"
  extent : Nat := 4
  timelineShape : FixedIndexTimelineShape := .normal
  formulaShape : FixedIndexFormulaShape := .canonical

def fixedIndexFixtureConfig : FixedIndexMutation → FixedIndexFixtureConfig
  | .supported => {}
  | .extentInvalid => { extent := 0 }
  | .checkpointMissing => { timelineShape := .missing }
  | .checkpointLate => { timelineShape := .late }
  | .checkpointWrongShape => { formulaShape := .wrongOperator }
  | .checkpointWrongIndex => { formulaShape := .wrongIndex }
  | .checkpointWrongType => { formulaShape := .wrongType }
  | .checkpointInconsistent => { extent := 3 }
  | .checkpointNoncanonicalLiteral => { literalValue := "04" }

def fixedIndexMutations : List FixedIndexMutation := [
  .supported, .extentInvalid, .checkpointMissing, .checkpointLate,
  .checkpointWrongShape, .checkpointWrongIndex, .checkpointWrongType,
  .checkpointInconsistent, .checkpointNoncanonicalLiteral
]

def fixedIndexManifest (mutation : FixedIndexMutation) : Manifest := {
  terms := [
    .variable (.free {
      id := timelineReadId
      name := "i"
      ty := some (.compilerTypeId 6) }),
    .intLit {
      value := (fixedIndexFixtureConfig mutation).literalValue
      ty := some (.compilerTypeId 6) },
    .binary {
      op := match (fixedIndexFixtureConfig mutation).formulaShape with
        | .wrongOperator => .le
        | _ => .lt
      lhs := match (fixedIndexFixtureConfig mutation).formulaShape with
        | .wrongIndex => 1
        | _ => 0
      rhs := 1
      ty := some (match (fixedIndexFixtureConfig mutation).formulaShape with
        | .wrongType => .compilerTypeId 1
        | _ => .compilerTypeId 6) }
  ]
}

def fixedIndexProjectionFor (mutation : FixedIndexMutation) : StorageProjectionRow :=
  .fixedIndex {
    index := { value := { formula := .term 0 }, ty := .compilerTypeId 6 }
    extent := (fixedIndexFixtureConfig mutation).extent
  }

def fixedIndexCheckpoint : TimelineEventRow := {
  id := 0
  structuralKey := "body/check"
  payload := .safetyCheckpoint { formulas := [{ formula := .term 2 }] }
}

def fixedIndexWrite (mutation : FixedIndexMutation) : TimelineEventRow := {
  id := 1
  structuralKey := "body/write"
  payload := .storageWrite {
    location := {
      root := "values"
      projections := [fixedIndexProjectionFor mutation]
      valueTy := .compilerTypeId 6
    }
    value := { formula := .term 0 }
  }
}

def fixedIndexTimelineFor (mutation : FixedIndexMutation) : List TimelineEventRow :=
  match (fixedIndexFixtureConfig mutation).timelineShape with
  | .missing => [fixedIndexWrite mutation]
  | .late => [
      { fixedIndexWrite mutation with id := 0 },
      { fixedIndexCheckpoint with id := 1 }]
  | .normal => [fixedIndexCheckpoint, fixedIndexWrite mutation]

def fixedIndexReasonForMutation (mutation : FixedIndexMutation) : String :=
  match timelineFixedIndexBoundsReason? (fixedIndexManifest mutation)
      (fixedIndexTimelineFor mutation) with
  | some reason => reason.name
  | none => ""

def expectedFixedIndexBoundsRows : List (String × String) :=
  fixedIndexMutations.map fun mutation =>
    (mutation.name, fixedIndexReasonForMutation mutation)

theorem loop_fixed_index_validation_mutations_pinned :
    loopFixedIndexBoundsRows.map Prod.fst == fixedIndexMutations.map (·.name) := by
  decide

theorem loop_fixed_index_validation_matches :
    loopFixedIndexBoundsRows == expectedFixedIndexBoundsRows := by
  decide

inductive HeaderFactMutation
  | supported
  | rowIdInvalid
  | rowOrderInvalid
  | identityMissing
  | identityDuplicate
  | sourceUnresolved
  | blockMismatch
  | sourceNotEarlier
  | targetMismatch
  | valueIdentityDuplicate
  | valueTypeMissing
  | assumptionMissing
  | assumptionUnexpected
  | unmatchedAssumption
  | extraAssumption
  | formulaMismatch
  | ownerMismatch
  deriving DecidableEq

def HeaderFactMutation.name : HeaderFactMutation → String
  | .supported => "supported"
  | .rowIdInvalid => "row_id_invalid"
  | .rowOrderInvalid => "row_order_invalid"
  | .identityMissing => "identity_missing"
  | .identityDuplicate => "identity_duplicate"
  | .sourceUnresolved => "source_unresolved"
  | .blockMismatch => "block_mismatch"
  | .sourceNotEarlier => "source_not_earlier"
  | .targetMismatch => "target_mismatch"
  | .valueIdentityDuplicate => "value_identity_duplicate"
  | .valueTypeMissing => "value_type_missing"
  | .assumptionMissing => "assumption_missing"
  | .assumptionUnexpected => "assumption_unexpected"
  | .unmatchedAssumption => "unmatched_assumption"
  | .extraAssumption => "extra_assumption"
  | .formulaMismatch => "formula_mismatch"
  | .ownerMismatch => "owner_mismatch"

def headerFactMutations : List HeaderFactMutation := [
  .supported, .rowIdInvalid, .rowOrderInvalid, .identityMissing, .identityDuplicate,
  .sourceUnresolved, .blockMismatch, .sourceNotEarlier, .targetMismatch,
  .valueIdentityDuplicate, .valueTypeMissing, .assumptionMissing, .assumptionUnexpected,
  .unmatchedAssumption, .extraAssumption, .formulaMismatch, .ownerMismatch
]

def baseHeaderFact : CanonicalHeaderFactRow := {
  id := 0
  structuralKey := "header-fact:0"
  formula := .term 1
  sourceKind := .functionRequires
  sourceStructuralKey := "requires:0"
  sourceBlockKey := "block:0"
  sourcePosition := 1
  targetStructuralKey := "loop:0"
  targetBlockKey := "block:0"
  targetPosition := 4
  stableValues := [{ id := loopId, ty := .compilerTypeId 6 }]
  assumptionId := some 7
}

def secondHeaderFact : CanonicalHeaderFactRow := {
  baseHeaderFact with id := 1, sourceStructuralKey := "requires:1", sourcePosition := 2
}

def headerFactsForMutation : HeaderFactMutation → List CanonicalHeaderFactRow
  | .rowIdInvalid => [{ baseHeaderFact with id := 1 }]
  | .rowOrderInvalid => [baseHeaderFact,
      { secondHeaderFact with sourcePosition := 1 }]
  | .identityMissing => [{ baseHeaderFact with structuralKey := "" }]
  | .identityDuplicate => [baseHeaderFact,
      { secondHeaderFact with structuralKey := baseHeaderFact.structuralKey }]
  | .sourceUnresolved => [{ baseHeaderFact with sourceKind := .unresolved }]
  | .blockMismatch => [{ baseHeaderFact with sourceBlockKey := "block:1" }]
  | .sourceNotEarlier => [{ baseHeaderFact with sourcePosition := 4 }]
  | .targetMismatch => [baseHeaderFact,
      { secondHeaderFact with structuralKey := "header-fact:1", targetStructuralKey := "loop:1" }]
  | .valueIdentityDuplicate => [{ baseHeaderFact with stableValues := [
      { id := loopId, ty := .compilerTypeId 6 },
      { id := loopId, ty := .compilerTypeId 6 }] }]
  | .valueTypeMissing => [{ baseHeaderFact with stableValues := [
      { id := loopId, ty := .compilerTypeId 0 }] }]
  | .assumptionMissing => [{ baseHeaderFact with assumptionId := none }]
  | .assumptionUnexpected => [{ baseHeaderFact with sourceKind := .pathAssume }]
  | _ => [baseHeaderFact]

def headerFactManifest (mutation : HeaderFactMutation) : Manifest := {
  baseManifest with assumptions := [{
    id := 7
    owner := if mutation == .ownerMismatch then "other" else "run"
    kind := .requires
    formula := some (.term (if mutation == .formulaMismatch then 0 else 1))
  }, {
    id := 8
    owner := "run"
    kind := .requires
    formula := some (.term 1)
  }]
}

def headerFactAssumptionIds : HeaderFactMutation → List Nat
  | .unmatchedAssumption => []
  | .extraAssumption => [7, 8]
  | _ => [7]

def expectedHeaderFactValidationRows : List (String × Bool × Bool) :=
  headerFactMutations.map fun mutation =>
    let rows := headerFactsForMutation mutation
    (mutation.name, headerFactsShapeSupported rows,
      headerFactsMatchAssumptions (headerFactManifest mutation) "run" rows
        (headerFactAssumptionIds mutation))

theorem loop_header_fact_validation_mutations_pinned :
    (loopHeaderFactValidationRows.map fun row => row.1) ==
      headerFactMutations.map (·.name) := by
  decide

theorem loop_header_fact_validation_matches :
    loopHeaderFactValidationRows == expectedHeaderFactValidationRows := by
  decide

def preheaderReadEvent : TimelineEventRow :=
  { timelineReadEvent with id := 0, structuralKey := "preheader/read" }
def preheaderWriteEvent : TimelineEventRow :=
  { timelineWriteEvent with id := 1, structuralKey := "preheader/write" }
def preheaderSafetyEvent : TimelineEventRow :=
  { timelineSafetyEvent with id := 2, structuralKey := "preheader/check" }

def preheaderTimelineForMutation : String → List TimelineEventRow
  | "read_identity_forward" => [{ preheaderWriteEvent with id := 0 },
      { preheaderReadEvent with id := 1 }, preheaderSafetyEvent]
  | "dependency_unsupported" =>
      [{ timelineDependencyEvent with id := 0, structuralKey := "preheader/child" },
       preheaderWriteEvent, preheaderSafetyEvent]
  | _ => [preheaderReadEvent, preheaderWriteEvent, preheaderSafetyEvent]

def preheaderMutationNames : List String := [
  "supported", "read_identity_forward", "dependency_unsupported"
]

def preheaderValidationRowMatches : String × String → Bool
  | (mutation, compilerReason) =>
      preheaderMutationNames.contains mutation &&
        compilerReason == match preheaderTimelineReason? [loopId]
            (preheaderTimelineForMutation mutation) with
          | some reason => reason.name
          | none => ""

theorem loop_preheader_validation_mutations_pinned :
    loopPreheaderValidationRows.map Prod.fst == preheaderMutationNames := by decide

theorem loop_preheader_validation_matches :
    loopPreheaderValidationRows.all preheaderValidationRowMatches = true := by decide

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

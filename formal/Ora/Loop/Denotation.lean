/-
Fail-closed denotation of compiler-emitted loop verification conditions.

The compiler emits only data. Unsupported identities, types, references, edge
targets, or incomplete state bindings make `denoteLoopSummary?` return `none`.
-/

import Ora.Loop.StorageDenotation
import Ora.Loop.HeaderFacts
import Ora.Loop.IntegerTypeSimpAttr

namespace Ora.Loop

open Ora.Obligation

attribute [ora_loop_integer_types] TyRef.integerShape?
  integerShapeForCompilerTypeId_u8 integerShapeForCompilerTypeId_u16
  integerShapeForCompilerTypeId_u32 integerShapeForCompilerTypeId_u64
  integerShapeForCompilerTypeId_u128 integerShapeForCompilerTypeId_u160
  integerShapeForCompilerTypeId_u256 integerShapeForCompilerTypeId_i8
  integerShapeForCompilerTypeId_i16 integerShapeForCompilerTypeId_i32
  integerShapeForCompilerTypeId_i64 integerShapeForCompilerTypeId_i128
  integerShapeForCompilerTypeId_i256

inductive LoopKind where
  | scfWhile
  | scfFor
  | other
  deriving Repr, BEq, DecidableEq

inductive ControlTargetRow where
  | backedge
  | successor
  | functionReturn
  | errorReturn
  | explicitRevert
  | unresolved
  deriving Repr, BEq, DecidableEq

structure VariableRow where
  index : Nat
  id : FreeVarId
  name : String := ""
  ty : TyRef
  deriving Repr, BEq, DecidableEq

def storageTyExplicit : TyRef → Bool
  | .spelling name => !name.isEmpty
  | .compilerTypeId id => id != 0

structure HeaderFactValueRow where
  id : FreeVarId
  ty : TyRef
  deriving Repr, BEq, DecidableEq

structure CanonicalHeaderFactRow where
  id : Nat
  structuralKey : String
  formula : FormulaRef
  sourceKind : HeaderFactSource
  sourceStructuralKey : String
  sourceBlockKey : String
  sourcePosition : Nat
  targetStructuralKey : String
  targetBlockKey : String
  targetPosition : Nat
  stableValues : List HeaderFactValueRow := []
  assumptionId : Option Nat := none
  deriving Repr, BEq, DecidableEq

def CanonicalHeaderFactRow.semanticRow
    (row : CanonicalHeaderFactRow) : HeaderFactRow FormulaRef := {
  id := row.id
  structuralKey := row.structuralKey
  formula := row.formula
  sourceKind := row.sourceKind
  source := { block := row.sourceBlockKey, position := row.sourcePosition }
  target := { block := row.targetBlockKey, position := row.targetPosition }
}

def rowsIndexedFrom {Row : Type} (rowIndex : Row → Nat) : Nat → List Row → Bool
  | _, [] => true
  | expected, row :: rest =>
      rowIndex row == expected && rowsIndexedFrom rowIndex (expected + 1) rest

def headerFactsShapeSupported (rows : List CanonicalHeaderFactRow) : Bool :=
  rowsIndexedFrom CanonicalHeaderFactRow.id 0 rows &&
    rows.all (fun row => !row.structuralKey.isEmpty &&
        !row.sourceStructuralKey.isEmpty && !row.sourceBlockKey.isEmpty &&
        !row.targetStructuralKey.isEmpty && !row.targetBlockKey.isEmpty &&
        row.semanticRow.shapeSupported &&
        row.stableValues.all (storageTyExplicit ·.ty) &&
        decide (row.stableValues.map HeaderFactValueRow.id).Nodup &&
        ((row.sourceKind == .functionRequires && row.assumptionId.isSome) ||
          (row.sourceKind == .pathAssume && row.assumptionId.isNone))) &&
    decide (rows.map CanonicalHeaderFactRow.structuralKey).Nodup &&
    decide (rows.map CanonicalHeaderFactRow.sourceStructuralKey).Nodup &&
    decide ((rows.map CanonicalHeaderFactRow.sourcePosition).Pairwise (fun lhs rhs => lhs < rhs)) &&
    match rows with
      | [] => true
      | first :: rest => rest.all fun row =>
          row.targetStructuralKey == first.targetStructuralKey &&
            row.targetBlockKey == first.targetBlockKey &&
            row.targetPosition == first.targetPosition

def headerFactsMatchAssumptions
    (manifest : Manifest) (owner : String)
    (rows : List CanonicalHeaderFactRow) (assumptionIds : List Nat) : Bool :=
  headerFactsShapeSupported rows &&
    rows.filterMap (fun row =>
      if row.sourceKind == .functionRequires then row.assumptionId else none) == assumptionIds &&
    rows.all fun row => match row.sourceKind, row.assumptionId with
      | .functionRequires, some id => manifest.assumptions.any fun assumption =>
          assumption.id == id && assumption.owner == owner &&
            assumption.kind == .requires &&
            match assumption.formula, row.formula with
              | some (.term assumptionTerm), .term factTerm => assumptionTerm == factTerm
              | none, _ => false
      | .pathAssume, none => true
      | _, _ => false

structure StateBindingRow where
  variableIndex : Nat
  target : FreeVarId
  sourceKey : String := ""
  value : FormulaRef
  readResults : List FreeVarId := []
  deriving Repr, BEq, DecidableEq

structure SummaryDependencyRow where
  id : Nat
  structuralKey : String
  summaryId : Nat
  loopVcSha256 : String
  target : ControlTargetRow
  writeFootprint : List PlaceRef := []
  writeFootprintComplete : Bool := false
  deriving Repr, BEq, DecidableEq

structure FormulaExprRow where
  formula : FormulaRef
  readResults : List FreeVarId := []
  deriving Repr, BEq, DecidableEq

structure TypedFormulaRow where
  value : FormulaExprRow
  ty : TyRef
  deriving Repr, BEq, DecidableEq

structure FixedIndexProjectionRow where
  index : TypedFormulaRow
  extent : Nat
  deriving Repr, BEq, DecidableEq

inductive StorageProjectionRow where
  | field : String → StorageProjectionRow
  | fixedIndex : FixedIndexProjectionRow → StorageProjectionRow
  | mappingKey : TypedFormulaRow → StorageProjectionRow
  deriving Repr, BEq, DecidableEq

structure StorageLocationExprRow where
  root : String
  projections : List StorageProjectionRow := []
  valueTy : TyRef
  deriving Repr, BEq, DecidableEq

structure TimelineFormulaEventRow where
  formulas : List FormulaExprRow
  /-- Marks the formula as the source-owned dynamic-index bounds assertion. -/
  dynamicIndexBounds : Bool := false
  deriving Repr, BEq, DecidableEq

structure StorageReadEventRow where
  result : FreeVarId
  location : StorageLocationExprRow
  deriving Repr, BEq, DecidableEq

structure StorageWriteEventRow where
  location : StorageLocationExprRow
  value : FormulaExprRow
  deriving Repr, BEq, DecidableEq

inductive TimelineEventPayloadRow where
  | pathAssumption : TimelineFormulaEventRow → TimelineEventPayloadRow
  | safetyCheckpoint : TimelineFormulaEventRow → TimelineEventPayloadRow
  | storageRead : StorageReadEventRow → TimelineEventPayloadRow
  | storageWrite : StorageWriteEventRow → TimelineEventPayloadRow
  | summaryDependency : SummaryDependencyRow → TimelineEventPayloadRow
  deriving Repr, BEq, DecidableEq

structure TimelineEventRow where
  id : Nat
  structuralKey : String
  payload : TimelineEventPayloadRow
  deriving Repr, BEq, DecidableEq

inductive TimelineShapeReason where
  | timelineMissing
  | eventOrderInvalid
  | eventIdentityMissing
  | eventIdentityDuplicate
  | pathMissing
  | storageRootMissing
  | storageProjectionInvalid
  | fixedIndexExtentInvalid
  | fixedIndexBoundsCheckpointMissing
  | fixedIndexBoundsFormulaShapeMismatch
  | fixedIndexBoundsIndexIdentityMismatch
  | fixedIndexBoundsTypeMismatch
  | fixedIndexBoundsExtentLiteralMismatch
  | storageTypeInvalid
  | readIdentityDuplicate
  | readIdentityForeign
  | readIdentityForward
  | childFootprintMissing
  | dependencyOrderInvalid
  | dependencyTargetUnsupported
  | bodySafetyMissing
  | preheaderUnsupported
  deriving Repr, BEq, DecidableEq

def TimelineShapeReason.name : TimelineShapeReason → String
  | .timelineMissing => "loop_timeline_missing"
  | .eventOrderInvalid => "loop_timeline_event_order_invalid"
  | .eventIdentityMissing => "loop_timeline_event_identity_missing"
  | .eventIdentityDuplicate => "loop_timeline_event_identity_duplicate"
  | .pathMissing => "loop_timeline_path_missing"
  | .storageRootMissing => "loop_storage_root_missing"
  | .storageProjectionInvalid => "loop_storage_projection_invalid"
  | .fixedIndexExtentInvalid => "loop_fixed_index_extent_invalid"
  | .fixedIndexBoundsCheckpointMissing => "loop_fixed_index_bounds_checkpoint_missing"
  | .fixedIndexBoundsFormulaShapeMismatch =>
      "loop_fixed_index_bounds_formula_shape_mismatch"
  | .fixedIndexBoundsIndexIdentityMismatch =>
      "loop_fixed_index_bounds_index_identity_mismatch"
  | .fixedIndexBoundsTypeMismatch => "loop_fixed_index_bounds_type_mismatch"
  | .fixedIndexBoundsExtentLiteralMismatch =>
      "loop_fixed_index_bounds_extent_literal_mismatch"
  | .storageTypeInvalid => "loop_storage_type_invalid"
  | .readIdentityDuplicate => "loop_storage_read_identity_duplicate"
  | .readIdentityForeign => "loop_storage_read_identity_foreign"
  | .readIdentityForward => "loop_storage_read_identity_forward"
  | .childFootprintMissing => "loop_storage_child_footprint_missing"
  | .dependencyOrderInvalid => "loop_summary_dependency_order_invalid"
  | .dependencyTargetUnsupported => "loop_summary_dependency_target_unsupported"
  | .bodySafetyMissing => "loop_body_safety_missing"
  | .preheaderUnsupported => "loop_preheader_storage_relation_unsupported"

def timelineReadIds (events : List TimelineEventRow) : List FreeVarId :=
  events.filterMap fun event => match event.payload with
    | .storageRead read => some read.result
    | _ => none

def timelineReadReferencesReason?
    (events : List TimelineEventRow)
    (eventIndex : Nat)
    (references : List FreeVarId) : Option TimelineShapeReason :=
  if !(decide references.Nodup) then some .readIdentityDuplicate
  else
    references.findSome? fun reference =>
      if (timelineReadIds (events.take eventIndex)).contains reference then none
      else if (timelineReadIds (events.drop eventIndex)).contains reference then
        some .readIdentityForward
      else some .readIdentityForeign

def fixedIndexExtentValid (extent : Nat) : Bool :=
  -- Compiler extents are `u64`; Lean rows use unbounded `Nat`, so the upper
  -- guard rejects hand-authored rows outside the canonical carrier.
  extent != 0 && decide (extent < 2 ^ 256)

def storageProjectionReason?
    (events : List TimelineEventRow)
    (eventIndex : Nat) : StorageProjectionRow → Option TimelineShapeReason
  | .field name => if name.isEmpty then some .storageProjectionInvalid else none
  | .fixedIndex projection =>
      if !fixedIndexExtentValid projection.extent then some .fixedIndexExtentInvalid
      else if !storageTyExplicit projection.index.ty then some .storageTypeInvalid
      else timelineReadReferencesReason? events eventIndex projection.index.value.readResults
  | .mappingKey value =>
      if !storageTyExplicit value.ty then some .storageTypeInvalid
      else timelineReadReferencesReason? events eventIndex value.value.readResults

def storageLocationReason?
    (events : List TimelineEventRow)
    (eventIndex : Nat)
    (location : StorageLocationExprRow) : Option TimelineShapeReason :=
  if location.root.isEmpty then some .storageRootMissing
  else if !storageTyExplicit location.valueTy then some .storageTypeInvalid
  else location.projections.findSome? (storageProjectionReason? events eventIndex)

def timelineFormulaEventReason?
    (events : List TimelineEventRow)
    (eventIndex : Nat)
    (formulas : List FormulaExprRow)
    (missing : TimelineShapeReason) : Option TimelineShapeReason :=
  if formulas.isEmpty then some missing
  else formulas.findSome? fun formula =>
    timelineReadReferencesReason? events eventIndex formula.readResults

def dependencyTargetSupported : ControlTargetRow → Bool
  | .successor | .functionReturn | .errorReturn => true
  | .backedge | .explicitRevert | .unresolved => false

def footprintPlaceSupported (place : PlaceRef) : Bool :=
  place.region == .storage && !place.root.isEmpty

def timelineEventsReason?
    (events : List TimelineEventRow)
    (variableIds : List FreeVarId)
    (requirePath allowDependencies : Bool) : Option TimelineShapeReason :=
  let rec visit : Nat → Nat → Bool → List String → List TimelineEventRow →
      Option TimelineShapeReason
    | _, _, hasPath, _, [] => if requirePath && !hasPath then some .pathMissing else none
    | expected, dependencyIndex, hasPath, keys, event :: rest =>
        if event.id != expected then some .eventOrderInvalid
        else if event.structuralKey.isEmpty then some .eventIdentityMissing
        else if keys.contains event.structuralKey then some .eventIdentityDuplicate
        else
          let next := visit (expected + 1) dependencyIndex hasPath
            (keys ++ [event.structuralKey]) rest
          match event.payload with
          | .pathAssumption path =>
              match timelineFormulaEventReason? events expected path.formulas .pathMissing with
              | some reason => some reason
              | none => visit (expected + 1) dependencyIndex true
                  (keys ++ [event.structuralKey]) rest
          | .safetyCheckpoint checkpoint =>
              match timelineFormulaEventReason? events expected checkpoint.formulas
                  .bodySafetyMissing with
              | some reason => some reason
              | none => next
          | .storageRead read =>
              match storageLocationReason? events expected read.location with
              | some reason => some reason
              | none =>
                  if variableIds.contains read.result ||
                      (timelineReadIds (events.take expected)).contains read.result then
                    some .readIdentityDuplicate
                  else next
          | .storageWrite write =>
              match storageLocationReason? events expected write.location with
              | some reason => some reason
              | none =>
                  match timelineReadReferencesReason? events expected write.value.readResults with
                  | some reason => some reason
                  | none => next
          | .summaryDependency dependency =>
              if !allowDependencies then some .preheaderUnsupported
              else if dependency.id != dependencyIndex then some .dependencyOrderInvalid
              else if !dependencyTargetSupported dependency.target then
                some .dependencyTargetUnsupported
              else if !dependency.writeFootprintComplete ||
                  !dependency.writeFootprint.all footprintPlaceSupported then
                some .childFootprintMissing
              else visit (expected + 1) (dependencyIndex + 1) hasPath
                (keys ++ [event.structuralKey]) rest
  if events.isEmpty then if requirePath then some .timelineMissing else none
  else visit 0 0 false [] events

structure SafetyCheckpointRow where
  id : Nat
  structuralKey : String
  afterDependencyCount : Nat
  formulas : List FormulaRef
  deriving Repr, BEq, DecidableEq

structure EdgeRow where
  id : Nat
  structuralKey : String := ""
  timeline : List TimelineEventRow := []
  nextState : List StateBindingRow := []
  target : ControlTargetRow := .unresolved
  targetFormulas : List FormulaRef := []
  deriving Repr, BEq, DecidableEq

def timelinePathView : List TimelineEventRow → List FormulaRef
  | [] => []
  | event :: rest =>
      match event.payload with
      | .pathAssumption path =>
          path.formulas.map FormulaExprRow.formula ++ timelinePathView rest
      | _ => timelinePathView rest

def timelineDependencyView : List TimelineEventRow → List SummaryDependencyRow
  | [] => []
  | event :: rest =>
      match event.payload with
      | .summaryDependency dependency => dependency :: timelineDependencyView rest
      | _ => timelineDependencyView rest

def timelineSafetyViewFrom : Nat → Nat → List TimelineEventRow → List SafetyCheckpointRow
  | _, _, [] => []
  | dependencyCount, checkpointId, event :: rest =>
      match event.payload with
      | .summaryDependency _ =>
          timelineSafetyViewFrom (dependencyCount + 1) checkpointId rest
      | .safetyCheckpoint checkpoint =>
          {
            id := checkpointId
            structuralKey := event.structuralKey
            afterDependencyCount := dependencyCount
            formulas := checkpoint.formulas.map FormulaExprRow.formula
          } :: timelineSafetyViewFrom dependencyCount (checkpointId + 1) rest
      | _ => timelineSafetyViewFrom dependencyCount checkpointId rest

def timelineSafetyView (events : List TimelineEventRow) : List SafetyCheckpointRow :=
  timelineSafetyViewFrom 0 0 events

def EdgeRow.path (edge : EdgeRow) : List FormulaRef :=
  timelinePathView edge.timeline

def EdgeRow.safetyCheckpoints (edge : EdgeRow) : List SafetyCheckpointRow :=
  timelineSafetyView edge.timeline

def EdgeRow.summaryDependencies (edge : EdgeRow) : List SummaryDependencyRow :=
  timelineDependencyView edge.timeline

theorem EdgeRow.path_eq_timeline (edge : EdgeRow) :
    edge.path = timelinePathView edge.timeline := rfl

theorem EdgeRow.safety_eq_timeline (edge : EdgeRow) :
    edge.safetyCheckpoints = timelineSafetyView edge.timeline := rfl

theorem EdgeRow.dependencies_eq_timeline (edge : EdgeRow) :
    edge.summaryDependencies = timelineDependencyView edge.timeline := rfl

def edgeTimelineReason?
    (variableIds : List FreeVarId)
    (edge : EdgeRow) : Option TimelineShapeReason :=
  match timelineEventsReason? edge.timeline variableIds true true with
  | some reason => some reason
  | none =>
      match edge.nextState.findSome? fun binding =>
          timelineReadReferencesReason? edge.timeline edge.timeline.length
            binding.readResults with
      | some reason => some reason
      | none => none

def preheaderTimelineReason?
    (variableIds : List FreeVarId)
    (events : List TimelineEventRow) : Option TimelineShapeReason :=
  timelineEventsReason? events variableIds false false

def edgeTimelineShapeSupported
    (variableIds : List FreeVarId)
    (edge : EdgeRow) : Bool :=
  (edgeTimelineReason? variableIds edge).isNone

structure SummaryRow where
  id : Nat
  owner : String
  kind : LoopKind
  contextVariables : List VariableRow := []
  variables : List VariableRow := []
  init : List FormulaRef := []
  invariants : List FormulaRef := []
  headerFacts : List CanonicalHeaderFactRow := []
  preheaderTimeline : List TimelineEventRow := []
  preheaderProjectionComplete : Bool := false
  edges : List EdgeRow := []
  projectionComplete : Bool := false
  post : List FormulaRef := []
  unsupportedReasons : List String := []
  deriving Repr, BEq, DecidableEq

def idsUnique (variables : List VariableRow) : Bool :=
  decide (variables.map VariableRow.id).Nodup

def variablesAreIntegers (variables : List VariableRow) : Bool :=
  variables.all (fun loopVar => loopVar.ty.integerShape?.isSome)

def variableSetsDisjoint (lhs rhs : List VariableRow) : Bool :=
  lhs.all (fun left => rhs.all (fun right => left.id != right.id))

def bindingsMatchVariables : List VariableRow → List StateBindingRow → Bool
  | [], [] => true
  | loopVar :: variables, binding :: bindings =>
      binding.variableIndex == loopVar.index &&
        binding.target == loopVar.id &&
        !binding.sourceKey.isEmpty &&
        bindingsMatchVariables variables bindings
  | _, _ => false

def edgeStructuralKeysValid (edges : List EdgeRow) : Bool :=
  edges.all (fun edge => !edge.structuralKey.isEmpty) &&
    decide (edges.map EdgeRow.structuralKey).Nodup

def edgeTargetSupported : ControlTargetRow → Bool
  | .backedge | .successor | .functionReturn | .errorReturn => true
  | .explicitRevert | .unresolved => false

def edgeTargetFormulasValid (edge : EdgeRow) : Bool :=
  match edge.target with
  | .backedge | .successor => edge.targetFormulas.isEmpty
  | .functionReturn | .errorReturn => true
  | .explicitRevert | .unresolved => false

def safetyCheckpointsIndexedAndStagedFrom : Nat → Nat → Nat →
    List SafetyCheckpointRow → Bool
  | _, _, _, [] => true
  | expected, previousStage, dependencyCount, checkpoint :: rest =>
      checkpoint.id == expected &&
        !checkpoint.structuralKey.isEmpty &&
        !checkpoint.formulas.isEmpty &&
        previousStage <= checkpoint.afterDependencyCount &&
        checkpoint.afterDependencyCount <= dependencyCount &&
        safetyCheckpointsIndexedAndStagedFrom (expected + 1)
          checkpoint.afterDependencyCount dependencyCount rest

def safetyCheckpointStructuralKeysValid (checkpoints : List SafetyCheckpointRow) : Bool :=
  decide (checkpoints.map SafetyCheckpointRow.structuralKey).Nodup

def safetyCheckpointsShapeSupported (edge : EdgeRow) : Bool :=
  safetyCheckpointsIndexedAndStagedFrom 0 0 edge.summaryDependencies.length
      edge.safetyCheckpoints &&
    safetyCheckpointStructuralKeysValid edge.safetyCheckpoints

def edgeBaseShapeSupported
    (contextVariables variables : List VariableRow)
    (edge : EdgeRow) : Bool :=
  !edge.path.isEmpty && edgeTargetSupported edge.target &&
    edgeTargetFormulasValid edge &&
    safetyCheckpointsShapeSupported edge &&
    bindingsMatchVariables variables edge.nextState &&
    edgeTimelineShapeSupported
      ((contextVariables ++ variables).map VariableRow.id) edge

def edgeShapeSupported
    (contextVariables variables : List VariableRow)
    (edge : EdgeRow) : Bool :=
  edgeBaseShapeSupported contextVariables variables edge &&
    edge.summaryDependencies.isEmpty

def lowerHexDigit (value : Char) : Bool :=
  "0123456789abcdef".contains value

def dependencyDigestValid (dependency : SummaryDependencyRow) : Bool :=
  dependency.loopVcSha256.length == 64 &&
    dependency.loopVcSha256.toList.all lowerHexDigit

def dependencyStructuralKeysValid (dependencies : List SummaryDependencyRow) : Bool :=
  dependencies.all (fun dependency => !dependency.structuralKey.isEmpty) &&
    decide (dependencies.map SummaryDependencyRow.structuralKey).Nodup

def findSummaryRow? : List SummaryRow → Nat → Option SummaryRow
  | [], _ => none
  | candidate :: rest, id =>
      if candidate.id == id then some candidate else findSummaryRow? rest id

def summaryIdsUnique (rows : List SummaryRow) : Bool :=
  decide (rows.map SummaryRow.id).Nodup

def summaryHasTarget (row : SummaryRow) (target : ControlTargetRow) : Bool :=
  row.edges.any (fun edge => edge.target == target)

def dependencyShapeSupported
    (rows : List SummaryRow)
    (parent : SummaryRow)
    (dependency : SummaryDependencyRow) : Bool :=
  dependency.summaryId > parent.id &&
    edgeTargetSupported dependency.target &&
    dependency.target != .backedge &&
    match findSummaryRow? rows dependency.summaryId with
    | some inner => summaryHasTarget inner dependency.target
    | none => false

/- The exact dependency digest and its lowercase hexadecimal serialization are
checked by the compiler-kernel obligation and artifact gates before this
denotation can authorize output.  The digest remains part of the emitted row
and certificate identity, but it has no role in the child transition relation;
rechecking `String.contains` here would make semantic denotation depend on an
opaque string implementation rather than on proof-relevant loop data. -/

def edgeDependenciesSupported
    (rows : List SummaryRow)
    (parent : SummaryRow)
    (edge : EdgeRow) : Bool :=
  rowsIndexedFrom SummaryDependencyRow.id 0 edge.summaryDependencies &&
    dependencyStructuralKeysValid edge.summaryDependencies &&
    edge.summaryDependencies.all (dependencyShapeSupported rows parent)

def hasBackedge (edges : List EdgeRow) : Bool :=
  edges.any (fun edge => edge.target == .backedge)

def hasExit (edges : List EdgeRow) : Bool :=
  edges.any (fun edge => edge.target != .backedge && edgeTargetSupported edge.target)

def valueForTy : TyRef → Value
  | ty =>
      match ty.integerShape? with
      | some shape => .integer (Ora.Integer.Value.ofNat shape 0)
      | none => .bool true

def bindSupportVariable (env : Env) (rowVariable : VariableRow) : Env :=
  env.setFree rowVariable.id (valueForTy rowVariable.ty)

def SummaryRow.supportEnv (row : SummaryRow) : Env :=
  (row.contextVariables ++ row.variables).foldl bindSupportVariable Env.empty

def dependencyVariables (rows : List SummaryRow) (row : SummaryRow) : List VariableRow :=
  row.edges.flatMap fun edge =>
    edge.summaryDependencies.flatMap fun dependency =>
      match findSummaryRow? rows dependency.summaryId with
      | some inner => inner.variables
      | none => []

def dependencyVariablesFor
    (rows : List SummaryRow)
    (dependencies : List SummaryDependencyRow) : List VariableRow :=
  dependencies.flatMap fun dependency =>
    match findSummaryRow? rows dependency.summaryId with
    | some inner => inner.variables
    | none => []

/-- The support environment exposes only the parent variables and variables
declared by its exact summary dependencies.  Unrelated loop rows cannot make
an otherwise dangling formula reference denotable. -/
def SummaryRow.supportEnvIn (rows : List SummaryRow) (row : SummaryRow) : Env :=
  (row.contextVariables ++ row.variables ++ dependencyVariables rows row).foldl
    bindSupportVariable Env.empty

def formulaDenotable (manifest : Manifest) (env : Env) (formula : FormulaRef) : Bool :=
  (formulaDenotes? manifest env formula).isSome

def valueDenotable (manifest : Manifest) (env : Env) (formula : FormulaRef) : Bool :=
  (formulaValue? manifest env formula).isSome

def storageLocationFormulasDenotable
    (manifest : Manifest)
    (env : Env)
    (location : StorageLocationExprRow) : Bool :=
  location.projections.all fun projection => match projection with
    | .field _ => true
    | .fixedIndex projection =>
        valueDenotable manifest env projection.index.value.formula
    | .mappingKey key => valueDenotable manifest env key.value.formula

inductive FixedIndexBoundsMatch where
  | matched
  | formulaShapeMismatch
  | indexIdentityMismatch
  | typeMismatch
  | extentLiteralMismatch
  deriving Repr, BEq, DecidableEq

def FixedIndexBoundsMatch.rank : FixedIndexBoundsMatch → Nat
  | .matched => 0
  | .formulaShapeMismatch => 1
  | .indexIdentityMismatch => 2
  | .typeMismatch => 3
  | .extentLiteralMismatch => 4

def FixedIndexBoundsMatch.reason? : FixedIndexBoundsMatch → Option TimelineShapeReason
  | .matched => none
  | .formulaShapeMismatch => some .fixedIndexBoundsFormulaShapeMismatch
  | .indexIdentityMismatch => some .fixedIndexBoundsIndexIdentityMismatch
  | .typeMismatch => some .fixedIndexBoundsTypeMismatch
  | .extentLiteralMismatch => some .fixedIndexBoundsExtentLiteralMismatch

def fixedIndexBoundsFormulaMatch
    (manifest : Manifest)
    (candidate : FormulaExprRow)
    (projection : FixedIndexProjectionRow) : FixedIndexBoundsMatch :=
  match candidate.formula, projection.index.value.formula with
  | .term formulaId, .term indexId =>
      match manifest.terms[formulaId]? with
      | some (.binary binary) =>
          if binary.op != .lt then .formulaShapeMismatch
          else if binary.lhs != indexId then .indexIdentityMismatch
          else if binary.ty != some projection.index.ty then .typeMismatch
          else match manifest.terms[binary.rhs]? with
            | some (.intLit literal) =>
                if literal.ty != some projection.index.ty then .typeMismatch
                else if literal.value != toString projection.extent then
                  .extentLiteralMismatch
                else .matched
            | _ => .formulaShapeMismatch
      | _ => .formulaShapeMismatch

def bestFixedIndexBoundsMatch? :
    List FixedIndexBoundsMatch → Option FixedIndexBoundsMatch
  | [] => none
  | result :: rest =>
      if result == .matched then some .matched
      else match bestFixedIndexBoundsMatch? rest with
        | some .matched => some .matched
        | some later =>
            if result.rank >= later.rank then some result else some later
        | none => some result

def priorSafetyFormulas : List TimelineEventRow → List FormulaExprRow
  | [] => []
  | event :: rest =>
      match event.payload with
      | .safetyCheckpoint checkpoint => checkpoint.formulas ++ priorSafetyFormulas rest
      | _ => priorSafetyFormulas rest

def fixedIndexProjectionBoundsReason?
    (manifest : Manifest)
    (prior : List TimelineEventRow)
    (projection : FixedIndexProjectionRow) : Option TimelineShapeReason :=
  if !fixedIndexExtentValid projection.extent then
    some .fixedIndexExtentInvalid
  else
    match bestFixedIndexBoundsMatch?
        ((priorSafetyFormulas prior).map
          (fixedIndexBoundsFormulaMatch manifest · projection)) with
    | some .matched => none
    | some mismatch => mismatch.reason?
    | none => some .fixedIndexBoundsCheckpointMissing

def storageLocationFixedIndexBoundsReason?
    (manifest : Manifest)
    (prior : List TimelineEventRow)
    (location : StorageLocationExprRow) : Option TimelineShapeReason :=
  location.projections.findSome? fun projection => match projection with
    | .fixedIndex fixed => fixedIndexProjectionBoundsReason? manifest prior fixed
    | .field _ | .mappingKey _ => none

def timelineFixedIndexBoundsReason?
    (manifest : Manifest) : List TimelineEventRow → Option TimelineShapeReason :=
  let rec visit (prior : List TimelineEventRow) :
      List TimelineEventRow → Option TimelineShapeReason
    | [] => none
    | event :: rest =>
        let reason := match event.payload with
          | .storageRead read =>
              storageLocationFixedIndexBoundsReason? manifest prior read.location
          | .storageWrite write =>
              storageLocationFixedIndexBoundsReason? manifest prior write.location
          | _ => none
        match reason with
        | some reason => some reason
        | none => visit (prior ++ [event]) rest
  visit []

def SummaryRow.fixedIndexBoundsSupported
    (manifest : Manifest)
    (row : SummaryRow) : Bool :=
  (timelineFixedIndexBoundsReason? manifest row.preheaderTimeline).isNone &&
    row.edges.all fun edge =>
      (timelineFixedIndexBoundsReason? manifest edge.timeline).isNone

def preheaderTimelineEnv?
    (manifest : Manifest) : Env → List TimelineEventRow → Option Env
  | env, [] => some env
  | env, event :: rest => do
      let next ← match event.payload with
        | .pathAssumption path | .safetyCheckpoint path =>
            if path.formulas.all (fun formula => formulaDenotable manifest env formula.formula)
              then some env else none
        | .storageRead read =>
            if storageLocationFormulasDenotable manifest env read.location then
              some (env.setFree read.result (valueForTy read.location.valueTy))
            else none
        | .storageWrite write =>
            if storageLocationFormulasDenotable manifest env write.location &&
                valueDenotable manifest env write.value.formula then some env else none
        | .summaryDependency _ => none
      preheaderTimelineEnv? manifest next rest

def preheaderTimelineFormulasDenotable
    (manifest : Manifest)
    (row : SummaryRow) : Bool :=
  let context := row.contextVariables.foldl bindSupportVariable Env.empty
  (preheaderTimelineEnv? manifest context row.preheaderTimeline).isSome

def safetyCheckpointDenotable
    (manifest : Manifest)
    (rows : List SummaryRow)
    (row : SummaryRow)
    (edge : EdgeRow)
    (checkpoint : SafetyCheckpointRow) : Bool :=
  let variables := dependencyVariablesFor rows
    (edge.summaryDependencies.take checkpoint.afterDependencyCount)
  let env := (row.contextVariables ++ row.variables ++ variables).foldl
    bindSupportVariable Env.empty
  checkpoint.formulas.all (formulaDenotable manifest env)

def edgeFormulasDenotable (manifest : Manifest) (env : Env) (edge : EdgeRow) : Bool :=
  edge.path.all (formulaDenotable manifest env) &&
    edge.safetyCheckpoints.all (fun checkpoint =>
      checkpoint.formulas.all (formulaDenotable manifest env)) &&
    edge.targetFormulas.all (formulaDenotable manifest env) &&
    edge.nextState.all (fun binding => valueDenotable manifest env binding.value)

def edgeControlFormulasDenotable
    (manifest : Manifest)
    (rows : List SummaryRow)
    (row : SummaryRow)
    (env : Env)
    (edge : EdgeRow) : Bool :=
  edge.path.all (formulaDenotable manifest env) &&
    edge.safetyCheckpoints.all (safetyCheckpointDenotable manifest rows row edge) &&
    edge.targetFormulas.all (formulaDenotable manifest env)

def edgeBindingsDenotable
    (manifest : Manifest)
    (env : Env)
    (edge : EdgeRow) : Bool :=
  edge.nextState.all (fun binding => valueDenotable manifest env binding.value)

def SummaryRow.baseShapeSupported (row : SummaryRow) : Bool :=
  row.projectionComplete &&
    row.unsupportedReasons.isEmpty &&
    (row.kind == .scfWhile || row.kind == .scfFor) &&
    !row.variables.isEmpty &&
    row.init.length == row.variables.length &&
    !row.invariants.isEmpty &&
    !row.edges.isEmpty &&
    rowsIndexedFrom VariableRow.index 0 row.variables &&
    idsUnique row.variables &&
    idsUnique row.contextVariables &&
    variablesAreIntegers row.variables &&
    variablesAreIntegers row.contextVariables &&
    variableSetsDisjoint row.variables row.contextVariables &&
    rowsIndexedFrom EdgeRow.id 0 row.edges &&
    edgeStructuralKeysValid row.edges &&
    hasBackedge row.edges && hasExit row.edges

def SummaryRow.shapeSupported (row : SummaryRow) : Bool :=
  row.baseShapeSupported &&
    headerFactsShapeSupported row.headerFacts &&
    (preheaderTimelineReason?
      ((row.contextVariables ++ row.variables).map VariableRow.id)
      row.preheaderTimeline).isNone &&
    row.edges.all (edgeShapeSupported row.contextVariables row.variables)

def SummaryRow.shapeSupportedIn (rows : List SummaryRow) (row : SummaryRow) : Bool :=
  summaryIdsUnique rows &&
    row.baseShapeSupported &&
    headerFactsShapeSupported row.headerFacts &&
    (preheaderTimelineReason?
      ((row.contextVariables ++ row.variables).map VariableRow.id)
      row.preheaderTimeline).isNone &&
    row.edges.all (fun edge =>
      edgeBaseShapeSupported row.contextVariables row.variables edge &&
        edgeDependenciesSupported rows row edge)

def SummaryRow.formulasDenotable (manifest : Manifest) (row : SummaryRow) : Bool :=
  let env := row.supportEnv
  row.init.all (valueDenotable manifest env) &&
    row.headerFacts.all (formulaDenotable manifest env ·.formula) &&
    preheaderTimelineFormulasDenotable manifest row &&
    row.invariants.all (formulaDenotable manifest env) &&
    row.edges.all (edgeFormulasDenotable manifest env) &&
    row.post.all (formulaDenotable manifest env)

def SummaryRow.formulasDenotableIn
    (manifest : Manifest)
    (rows : List SummaryRow)
    (row : SummaryRow) : Bool :=
  let base := row.supportEnv
  let dependent := row.supportEnvIn rows
  row.init.all (valueDenotable manifest base) &&
    row.headerFacts.all (formulaDenotable manifest base ·.formula) &&
    preheaderTimelineFormulasDenotable manifest row &&
    row.invariants.all (formulaDenotable manifest base) &&
    row.edges.all (edgeControlFormulasDenotable manifest rows row base) &&
    row.edges.all (edgeBindingsDenotable manifest dependent) &&
    row.post.all (formulaDenotable manifest base)

def SummaryRow.supported (manifest : Manifest) (row : SummaryRow) : Bool :=
  manifest.wf && manifest.termsTopological && row.shapeSupported &&
    row.fixedIndexBoundsSupported manifest && row.formulasDenotable manifest

def SummaryRow.supportedIn
    (manifest : Manifest)
    (rows : List SummaryRow)
    (row : SummaryRow) : Bool :=
  manifest.wf && manifest.termsTopological &&
    row.shapeSupportedIn rows && row.fixedIndexBoundsSupported manifest &&
    row.formulasDenotableIn manifest rows

def contextReady (env : Env) (variables : List VariableRow) : Prop :=
  ∀ loopVar ∈ variables, ∃ value,
    loopVar.ty.integerShape? = some value.shape ∧
      env.lookupFree loopVar.id = some (.integer value)

/- Loop state contains only loop-carried integer values. The obligation
environment is captured by the denoted VC and remains unchanged; binding the
state shadows only the declared loop variables. -/
abbrev ScalarLoopState := List Ora.Integer.Value

def valuesMatchVariables : List VariableRow → ScalarLoopState → Bool
  | [], [] => true
  | loopVar :: variables, value :: values =>
      loopVar.ty.integerShape? == some value.shape &&
        valuesMatchVariables variables values
  | _, _ => false

def stateReady (variables : List VariableRow) (state : ScalarLoopState) : Prop :=
  valuesMatchVariables variables state = true

def bindState? : Env → List VariableRow → ScalarLoopState → Option Env
  | env, [], [] => some env
  | env, loopVar :: variables, value :: values =>
      if loopVar.ty.integerShape? = some value.shape then
        bindState? (env.setFree loopVar.id (.integer value)) variables values
      else
        none
  | _, _, _ => none

theorem bindState_some_of_stateReady
    (base : Env)
    (variables : List VariableRow)
    (state : ScalarLoopState)
    (hReady : stateReady variables state) :
    ∃ bound, bindState? base variables state = some bound := by
  induction variables generalizing base state with
  | nil =>
      cases state with
      | nil => exact ⟨base, rfl⟩
      | cons _ _ => simp [stateReady, valuesMatchVariables] at hReady
  | cons loopVar rest ih =>
      cases state with
      | nil => simp [stateReady, valuesMatchVariables] at hReady
      | cons value values =>
          simp only [stateReady, valuesMatchVariables, Bool.and_eq_true,
            beq_iff_eq] at hReady
          rcases hReady with ⟨hShape, hRest⟩
          rcases ih (base.setFree loopVar.id (.integer value)) values hRest with
            ⟨bound, hBound⟩
          refine ⟨bound, ?_⟩
          simp only [bindState?, hShape, hBound, if_pos]

theorem bindState_lookupFree_of_not_mem
    (env bound : Env)
    (variables : List VariableRow)
    (state : ScalarLoopState)
    (id : FreeVarId)
    (hFresh : id ∉ variables.map VariableRow.id)
    (hBind : bindState? env variables state = some bound) :
    bound.lookupFree id = env.lookupFree id := by
  induction variables generalizing env bound state with
  | nil =>
      cases state with
      | nil =>
          simp only [bindState?, Option.some.injEq] at hBind
          subst bound
          rfl
      | cons _ _ => simp [bindState?] at hBind
  | cons row rest ih =>
      cases state with
      | nil => simp [bindState?] at hBind
      | cons value values =>
          simp only [List.map_cons, List.mem_cons, not_or] at hFresh
          simp only [bindState?] at hBind
          split at hBind
          · rw [ih (env := env.setFree row.id (.integer value))
              (bound := bound) (state := values) hFresh.2 hBind]
            unfold Env.setFree Env.lookupFree
            simp only [lookupFreeBinding]
            have hDifferent : (row.id == id) = false := by
              apply Bool.eq_false_iff.mpr
              intro hEqual
              exact hFresh.1 (beq_iff_eq.mp hEqual).symm
            simp only [hDifferent, Bool.false_eq_true, if_false]
          · cases hBind

def formulaHoldsInState
    (manifest : Manifest)
    (base : Env)
    (variables : List VariableRow)
    (state : ScalarLoopState)
    (formula : FormulaRef) : Prop :=
  match bindState? base variables state with
  | some env =>
      match formulaDenotes? manifest env formula with
      | some proposition => proposition
      | none => False
  | none => False

def formulasHoldInState
    (manifest : Manifest)
    (base : Env)
    (variables : List VariableRow)
    (state : ScalarLoopState)
    (formulas : List FormulaRef) : Prop :=
  ∀ formula ∈ formulas, formulaHoldsInState manifest base variables state formula

def formulaHoldsInEnv
    (manifest : Manifest)
    (env : Env)
    (formula : FormulaRef) : Prop :=
  match formulaDenotes? manifest env formula with
  | some proposition => proposition
  | none => False

def formulasHoldInEnv
    (manifest : Manifest)
    (env : Env)
    (formulas : List FormulaRef) : Prop :=
  ∀ formula ∈ formulas, formulaHoldsInEnv manifest env formula

/-- Task 3 restricts these formulas to immutable context leaves, so their
truth is independent of the changing loop state. -/
def headerFactsHoldInEnv
    (manifest : Manifest) (env : Env) (rows : List CanonicalHeaderFactRow) : Prop :=
  ∀ row ∈ rows, formulaHoldsInEnv manifest env row.formula

def integerFormulaValue?
    (manifest : Manifest)
    (env : Env)
    (formula : FormulaRef) : Option Ora.Integer.Value :=
  match formulaValue? manifest env formula with
  | some (.integer value) => some value
  | _ => none

def evaluateIntegerFormulas?
    (manifest : Manifest)
    (env : Env) : List FormulaRef → Option (List Ora.Integer.Value)
  | [] => some []
  | formula :: formulas => do
      let value ← integerFormulaValue? manifest env formula
      let values ← evaluateIntegerFormulas? manifest env formulas
      pure (value :: values)

def initializeState?
    (manifest : Manifest)
    (base : Env)
    (variables : List VariableRow)
    (formulas : List FormulaRef) : Option ScalarLoopState := do
  let values ← evaluateIntegerFormulas? manifest base formulas
  if valuesMatchVariables variables values then
    pure values
  else
    none

def bindingFormulas (bindings : List StateBindingRow) : List FormulaRef :=
  bindings.map StateBindingRow.value

def applyStateBindingsInEnv?
    (manifest : Manifest)
    (variables : List VariableRow)
    (env : Env)
    (bindings : List StateBindingRow) : Option ScalarLoopState :=
  if bindingsMatchVariables variables bindings then do
    let values ← evaluateIntegerFormulas? manifest env (bindingFormulas bindings)
    if valuesMatchVariables variables values then
      pure values
    else
      none
  else
    none

def applyStateBindings?
    (manifest : Manifest)
    (base : Env)
    (variables : List VariableRow)
    (current : ScalarLoopState)
    (bindings : List StateBindingRow) : Option ScalarLoopState :=
  if bindingsMatchVariables variables bindings then do
    let env ← bindState? base variables current
    let values ← evaluateIntegerFormulas? manifest env (bindingFormulas bindings)
    if valuesMatchVariables variables values then
      pure values
    else
      none
  else
    none

theorem bound_state_in_env_ready
    (manifest : Manifest)
    (variables : List VariableRow)
    (env : Env)
    (bindings : List StateBindingRow)
    (following : ScalarLoopState)
    (hNext : applyStateBindingsInEnv? manifest variables env bindings = some following) :
    stateReady variables following := by
  unfold applyStateBindingsInEnv? at hNext
  by_cases hBindings : bindingsMatchVariables variables bindings = true
  · cases hValues : evaluateIntegerFormulas? manifest env (bindingFormulas bindings) with
    | none => simp [hBindings, hValues] at hNext
    | some values =>
        by_cases hReady : valuesMatchVariables variables values = true
        · simp [hBindings, hValues, hReady] at hNext
          subst following
          exact hReady
        · cases hReadyValue : valuesMatchVariables variables values <;>
            simp_all
  · cases hBindingsValue : bindingsMatchVariables variables bindings <;>
      simp_all

def valuesInitializeState
    (manifest : Manifest)
    (base : Env)
    (variables : List VariableRow)
    (formulas : List FormulaRef)
    (state : ScalarLoopState) : Prop :=
  initializeState? manifest base variables formulas = some state

theorem initialized_state_ready
    (manifest : Manifest)
    (base : Env)
    (variables : List VariableRow)
    (formulas : List FormulaRef)
    (state : ScalarLoopState)
    (hInitial : valuesInitializeState manifest base variables formulas state) :
    stateReady variables state := by
  unfold valuesInitializeState initializeState? at hInitial
  cases hValues : evaluateIntegerFormulas? manifest base formulas with
  | none => simp [hValues] at hInitial
  | some values =>
      cases hReady : valuesMatchVariables variables values with
      | false => simp [hValues, hReady] at hInitial
      | true =>
        simp [hValues, hReady] at hInitial
        subst state
        exact hReady

def stateBindingsProduceState
    (manifest : Manifest)
    (base : Env)
    (variables : List VariableRow)
    (current : ScalarLoopState)
    (bindings : List StateBindingRow)
    (following : ScalarLoopState) : Prop :=
  applyStateBindings? manifest base variables current bindings = some following

theorem bound_state_ready
    (manifest : Manifest)
    (base : Env)
    (variables : List VariableRow)
    (current following : ScalarLoopState)
    (bindings : List StateBindingRow)
    (hNext : stateBindingsProduceState manifest base variables current bindings following) :
    stateReady variables following := by
  unfold stateBindingsProduceState applyStateBindings? at hNext
  by_cases hBindings : bindingsMatchVariables variables bindings = true
  · cases hEnv : bindState? base variables current with
    | none => simp [hBindings, hEnv] at hNext
    | some env =>
        cases hValues : evaluateIntegerFormulas? manifest env (bindingFormulas bindings) with
        | none => simp [hBindings, hEnv, hValues] at hNext
        | some values =>
            by_cases hReady : valuesMatchVariables variables values = true
            · simp [hBindings, hEnv, hValues, hReady] at hNext
              subst following
              exact hReady
            · cases hReadyValue : valuesMatchVariables variables values <;>
                simp_all
  · cases hBindingsValue : bindingsMatchVariables variables bindings <;>
      simp_all

abbrev assignmentsProduceState := stateBindingsProduceState

def denoteControlTarget? : ControlTargetRow → Option ControlTarget
  | .backedge => some .backedge
  | .successor => some .successor
  | .functionReturn => some .functionReturn
  | .errorReturn => some .errorReturn
  | .explicitRevert | .unresolved => none

theorem explicit_revert_target_fails_closed :
    denoteControlTarget? .explicitRevert = none := rfl

abbrev SummaryResolver := Env → SummaryRow → Option (LoopVC ScalarLoopState)

def summaryDependencyHolds
    (manifest : Manifest)
    (resolver : SummaryResolver)
    (inner : SummaryRow)
    (target : ControlTarget)
    (current following : Env) : Prop :=
  match resolver current inner with
  | none => False
  | some summary =>
      ∃ initial final bound,
        valuesInitializeState manifest current inner.variables inner.init initial ∧
        summary.CertifiedControlRelation target initial final ∧
        bindState? current inner.variables final = some bound ∧
        following = bound

structure EnvironmentRelation where
  holds : Env → Env → Prop

/-- Denote one certified inner-loop summary as an environment relation.  The
resolver is checked while constructing the parent VC, so an unavailable child
summary makes parent denotation fail instead of turning the edge relation into
`False` and making preservation vacuous. -/
def denoteSummaryDependency?
    (manifest : Manifest)
    (rows : List SummaryRow)
    (resolver : SummaryResolver)
    (supportBase : Env)
    (dependency : SummaryDependencyRow) : Option EnvironmentRelation := do
  let inner ← findSummaryRow? rows dependency.summaryId
  let target ← denoteControlTarget? dependency.target
  if (resolver supportBase inner).isSome then
    pure {
      holds := summaryDependencyHolds manifest resolver inner target
    }
  else
    none

def denoteSummaryDependencies?
    (manifest : Manifest)
    (rows : List SummaryRow)
    (resolver : SummaryResolver)
    (supportBase : Env) :
    List SummaryDependencyRow → Option (List EnvironmentRelation)
  | [] => some []
  | dependency :: rest => do
      let relation ← denoteSummaryDependency? manifest rows resolver supportBase dependency
      let relations ← denoteSummaryDependencies? manifest rows resolver supportBase rest
      pure (relation :: relations)

def environmentRelationsProduce :
    List EnvironmentRelation → Env → Env → Prop
  | [], current, following => current = following
  | relation :: rest, current, following =>
      ∃ intermediate,
        relation.holds current intermediate ∧
        environmentRelationsProduce rest intermediate following

def safetyCheckpointHolds
    (manifest : Manifest)
    (base : Env)
    (variables : List VariableRow)
    (current : ScalarLoopState)
    (dependencies : List EnvironmentRelation)
    (checkpoint : SafetyCheckpointRow) : Prop :=
  ∀ beforeDependencies checkpointEnv,
    bindState? base variables current = some beforeDependencies →
    environmentRelationsProduce
        (dependencies.take checkpoint.afterDependencyCount)
        beforeDependencies checkpointEnv →
    formulasHoldInEnv manifest checkpointEnv checkpoint.formulas

def safetyCheckpointsHold
    (manifest : Manifest)
    (base : Env)
    (variables : List VariableRow)
    (current : ScalarLoopState)
    (dependencies : List EnvironmentRelation)
    (checkpoints : List SafetyCheckpointRow) : Prop :=
  ∀ checkpoint ∈ checkpoints,
    safetyCheckpointHolds manifest base variables current dependencies checkpoint

theorem safetyCheckpointsHold_no_dependencies_formula
    (manifest : Manifest)
    (base : Env)
    (variables : List VariableRow)
    (current : ScalarLoopState)
    (checkpoints : List SafetyCheckpointRow)
    (checkpoint : SafetyCheckpointRow)
    (formula : FormulaRef)
    (hReady : stateReady variables current)
    (hChecks : safetyCheckpointsHold manifest base variables current [] checkpoints)
    (hCheckpoint : checkpoint ∈ checkpoints)
    (hFormula : formula ∈ checkpoint.formulas) :
    formulaHoldsInState manifest base variables current formula := by
  rcases bindState_some_of_stateReady base variables current hReady with
    ⟨bound, hBound⟩
  have hCheckpointHolds := hChecks checkpoint hCheckpoint
  unfold safetyCheckpointHolds at hCheckpointHolds
  have hFormulas := hCheckpointHolds bound bound hBound (by
    simp [environmentRelationsProduce])
  unfold formulaHoldsInState
  rw [hBound]
  exact hFormulas formula hFormula

/-- Partial correctness does not demand safety for code following a dependency
that has no terminating result. If the dependency does terminate, the
checkpoint remains universally quantified over every result. -/
theorem safety_checkpoint_after_nonterminating_dependency
    (manifest : Manifest)
    (base : Env)
    (variables : List VariableRow)
    (current : ScalarLoopState)
    (relation : EnvironmentRelation)
    (checkpoint : SafetyCheckpointRow)
    (before : Env)
    (hBound : bindState? base variables current = some before)
    (hStage : checkpoint.afterDependencyCount = 1)
    (hNever : ∀ following, ¬ relation.holds before following) :
    safetyCheckpointHolds manifest base variables current [relation] checkpoint := by
  intro actualBefore checkpointEnv hActualBound hProduce
  have hBefore : actualBefore = before :=
    Option.some.inj (hActualBound.symm.trans hBound)
  subst actualBefore
  rw [hStage] at hProduce
  simp only [List.take, environmentRelationsProduce] at hProduce
  rcases hProduce with ⟨following, hRelation, _⟩
  exact False.elim (hNever following hRelation)

def stateBindingsProduceAfterDependencies
    (manifest : Manifest)
    (base : Env)
    (variables : List VariableRow)
    (current : ScalarLoopState)
    (dependencies : List EnvironmentRelation)
    (bindings : List StateBindingRow)
    (following : ScalarLoopState) : Prop :=
  ∃ beforeDependencies afterDependencies,
    bindState? base variables current = some beforeDependencies ∧
    environmentRelationsProduce dependencies beforeDependencies afterDependencies ∧
    applyStateBindingsInEnv? manifest variables afterDependencies bindings = some following

theorem bound_state_after_dependencies_ready
    (manifest : Manifest)
    (base : Env)
    (variables : List VariableRow)
    (current : ScalarLoopState)
    (dependencies : List EnvironmentRelation)
    (bindings : List StateBindingRow)
    (following : ScalarLoopState)
    (hNext : stateBindingsProduceAfterDependencies manifest base variables current
      dependencies bindings following) :
    stateReady variables following := by
  rcases hNext with ⟨_, afterDependencies, _, _, hBindings⟩
  exact bound_state_in_env_ready manifest variables afterDependencies bindings following hBindings

theorem stateBindingsProduceAfterNoDependencies_iff
    (manifest : Manifest)
    (base : Env)
    (variables : List VariableRow)
    (current : ScalarLoopState)
    (bindings : List StateBindingRow)
    (following : ScalarLoopState) :
    stateBindingsProduceAfterDependencies manifest base variables current [] bindings following ↔
      stateBindingsProduceState manifest base variables current bindings following := by
  unfold stateBindingsProduceAfterDependencies stateBindingsProduceState
  cases hBind : bindState? base variables current <;>
    simp [environmentRelationsProduce, applyStateBindings?, applyStateBindingsInEnv?, hBind]

def denoteEdge?
    (manifest : Manifest)
    (base : Env)
    (variables : List VariableRow)
    (successorPost : List FormulaRef)
    (row : EdgeRow) : Option (Edge ScalarLoopState) := do
  let target ← denoteControlTarget? row.target
  let targetFormulas := match row.target with
    | .successor => successorPost
    | .functionReturn | .errorReturn => row.targetFormulas
    | .backedge | .explicitRevert | .unresolved => []
  pure {
    path := fun state => formulasHoldInState manifest base variables state row.path
    safe := fun state => safetyCheckpointsHold manifest base variables state []
      row.safetyCheckpoints
    next := fun current following =>
      stateBindingsProduceState manifest base variables current row.nextState following
    target := target
    targetAssertion := fun state =>
      formulasHoldInState manifest base variables state targetFormulas
  }

def denoteEdgeWithResolver?
    (manifest : Manifest)
    (rows : List SummaryRow)
    (resolver : SummaryResolver)
    (base : Env)
    (variables : List VariableRow)
    (successorPost : List FormulaRef)
    (row : EdgeRow) : Option (Edge ScalarLoopState) := do
  let target ← denoteControlTarget? row.target
  let dependencies ← denoteSummaryDependencies? manifest rows resolver base
    row.summaryDependencies
  let targetFormulas := match row.target with
    | .successor => successorPost
    | .functionReturn | .errorReturn => row.targetFormulas
    | .backedge | .explicitRevert | .unresolved => []
  pure {
    path := fun state => formulasHoldInState manifest base variables state row.path
    safe := fun state => safetyCheckpointsHold manifest base variables state dependencies
      row.safetyCheckpoints
    next := fun current following =>
      stateBindingsProduceAfterDependencies manifest base variables current
        dependencies row.nextState following
    target := target
    targetAssertion := fun state =>
      formulasHoldInState manifest base variables state targetFormulas
  }

def denoteEdges?
    (manifest : Manifest)
    (base : Env)
    (variables : List VariableRow)
    (successorPost : List FormulaRef) : List EdgeRow → Option (List (Edge ScalarLoopState))
  | [] => some []
  | edge :: rest => do
      let denoted ← denoteEdge? manifest base variables successorPost edge
      let denotedRest ← denoteEdges? manifest base variables successorPost rest
      pure (denoted :: denotedRest)

def denoteEdgesWithResolver?
    (manifest : Manifest)
    (rows : List SummaryRow)
    (resolver : SummaryResolver)
    (base : Env)
    (variables : List VariableRow)
    (successorPost : List FormulaRef) :
    List EdgeRow → Option (List (Edge ScalarLoopState))
  | [] => some []
  | edge :: rest => do
      let denoted ← denoteEdgeWithResolver? manifest rows resolver base variables
        successorPost edge
      let denotedRest ← denoteEdgesWithResolver? manifest rows resolver base variables
        successorPost rest
      pure (denoted :: denotedRest)

def denoteLoopSummary? (manifest : Manifest) (base : Env) (row : SummaryRow) :
    Option (LoopVC ScalarLoopState) :=
  if row.supported manifest then do
    let edges ← denoteEdges? manifest base row.variables row.post row.edges
    pure {
      initial := fun state => headerFactsHoldInEnv manifest base row.headerFacts ∧
        valuesInitializeState manifest base row.variables row.init state
      invariant := fun state =>
        stateReady row.variables state ∧
          headerFactsHoldInEnv manifest base row.headerFacts ∧
          formulasHoldInState manifest base row.variables state row.invariants
      edges := edges
    }
  else
    none

def denoteLoopSummaryWithResolver?
    (manifest : Manifest)
    (rows : List SummaryRow)
    (resolver : SummaryResolver)
    (base : Env)
    (row : SummaryRow) : Option (LoopVC ScalarLoopState) :=
  if row.supportedIn manifest rows then do
    let edges ← denoteEdgesWithResolver? manifest rows resolver base row.variables row.post
      row.edges
    pure {
      initial := fun state => headerFactsHoldInEnv manifest base row.headerFacts ∧
        valuesInitializeState manifest base row.variables row.init state
      invariant := fun state =>
        stateReady row.variables state ∧
          headerFactsHoldInEnv manifest base row.headerFacts ∧
          formulasHoldInState manifest base row.variables state row.invariants
      edges := edges
    }
  else
    none

/-- Fuel is consumed only when following a declared inner-summary dependency.
Unique, strictly-forward summary ids bound the required depth by `rows.length`;
the public entry point supplies one additional unit for the root summary. -/
def denoteLoopSummaryWithFuel?
    (fuel : Nat)
    (manifest : Manifest)
    (rows : List SummaryRow)
    (base : Env)
    (row : SummaryRow) : Option (LoopVC ScalarLoopState) :=
  match fuel with
  | 0 => none
  | remaining + 1 =>
      denoteLoopSummaryWithResolver? manifest rows
        (fun innerBase inner =>
          denoteLoopSummaryWithFuel? remaining manifest rows innerBase inner)
        base row
termination_by fuel

def denoteLoopSummaryIn?
    (manifest : Manifest)
    (rows : List SummaryRow)
    (base : Env)
    (row : SummaryRow) : Option (LoopVC ScalarLoopState) :=
  denoteLoopSummaryWithFuel? (rows.length + 1) manifest rows base row

def loopInitialPremises (manifest : Manifest) (base : Env) (row : SummaryRow)
    (state : ScalarLoopState) : Prop :=
  match denoteLoopSummary? manifest base row with
  | some summary => summary.initial state
  | none => False

def loopInductionObligationsAtEnv
    (manifest : Manifest)
    (base : Env)
    (row : SummaryRow) : Prop :=
  match denoteLoopSummary? manifest base row with
  | some summary => summary.InductionObligations
  | none => False

def loopVerifiedAtEnv
    (manifest : Manifest)
    (base : Env)
    (row : SummaryRow) : Prop :=
  match denoteLoopSummary? manifest base row with
  | some summary => summary.Verified
  | none => False

def loopInitialPremisesIn
    (manifest : Manifest)
    (rows : List SummaryRow)
    (base : Env)
    (row : SummaryRow)
    (state : ScalarLoopState) : Prop :=
  match denoteLoopSummaryIn? manifest rows base row with
  | some summary => summary.initial state
  | none => False

def loopInductionObligationsAtEnvIn
    (manifest : Manifest)
    (rows : List SummaryRow)
    (base : Env)
    (row : SummaryRow) : Prop :=
  match denoteLoopSummaryIn? manifest rows base row with
  | some summary => summary.InductionObligations
  | none => False

def loopVerifiedAtEnvIn
    (manifest : Manifest)
    (rows : List SummaryRow)
    (base : Env)
    (row : SummaryRow) : Prop :=
  match denoteLoopSummaryIn? manifest rows base row with
  | some summary => summary.Verified
  | none => False

/-- The denoted pre/post relation for one exact loop control target.  Keeping
the target in the relation prevents an enclosing loop from treating a return
or error return as an ordinary successor. -/
def loopControlRelationAtEnv
    (manifest : Manifest)
    (base : Env)
    (row : SummaryRow)
    (target : ControlTarget)
    (initial final : ScalarLoopState) : Prop :=
  match denoteLoopSummary? manifest base row with
  | some summary => summary.ControlRelation target initial final
  | none => False

def loopCertifiedControlRelationAtEnv
    (manifest : Manifest)
    (base : Env)
    (row : SummaryRow)
    (target : ControlTarget)
    (initial final : ScalarLoopState) : Prop :=
  match denoteLoopSummary? manifest base row with
  | some summary => summary.CertifiedControlRelation target initial final
  | none => False

def loopControlTargetAssertionAtEnv
    (manifest : Manifest)
    (base : Env)
    (row : SummaryRow)
    (target : ControlTarget)
    (final : ScalarLoopState) : Prop :=
  match denoteLoopSummary? manifest base row with
  | some summary =>
      ∃ edge,
        edge ∈ summary.edges ∧
        edge.target = target ∧
        edge.targetAssertion final
  | none => False

def loopSpecificationNonvacuous
    (manifest : Manifest)
    (assumptions : List AssumptionRow)
    (row : SummaryRow) : Prop :=
  ∃ env state,
    assumptionsDenoteInEnv manifest env assumptions ∧
    contextReady env row.contextVariables ∧
    loopInitialPremises manifest env row state

def loopInductionCertificateFromAssumptions
    (manifest : Manifest)
    (assumptions : List AssumptionRow)
    (row : SummaryRow) : Prop :=
  ∀ env,
    assumptionsDenoteInEnv manifest env assumptions →
    contextReady env row.contextVariables →
    loopInductionObligationsAtEnv manifest env row

def loopProofAuthorizationFromAssumptions
    (manifest : Manifest)
    (assumptions : List AssumptionRow)
    (row : SummaryRow) : Prop :=
  loopSpecificationNonvacuous manifest assumptions row ∧
    loopInductionCertificateFromAssumptions manifest assumptions row

def loopSpecificationNonvacuousIn
    (manifest : Manifest)
    (assumptions : List AssumptionRow)
    (rows : List SummaryRow)
    (row : SummaryRow) : Prop :=
  ∃ env state,
    assumptionsDenoteInEnv manifest env assumptions ∧
    contextReady env row.contextVariables ∧
    loopInitialPremisesIn manifest rows env row state

def loopInductionCertificateFromAssumptionsIn
    (manifest : Manifest)
    (assumptions : List AssumptionRow)
    (rows : List SummaryRow)
    (row : SummaryRow) : Prop :=
  ∀ env,
    assumptionsDenoteInEnv manifest env assumptions →
    contextReady env row.contextVariables →
    loopInductionObligationsAtEnvIn manifest rows env row

def loopProofAuthorizationFromAssumptionsIn
    (manifest : Manifest)
    (assumptions : List AssumptionRow)
    (rows : List SummaryRow)
    (row : SummaryRow) : Prop :=
  loopSpecificationNonvacuousIn manifest assumptions rows row ∧
    loopInductionCertificateFromAssumptionsIn manifest assumptions rows row

def loopVerifiedFromAssumptions
    (manifest : Manifest)
    (assumptions : List AssumptionRow)
    (row : SummaryRow) : Prop :=
  ∀ env,
    assumptionsDenoteInEnv manifest env assumptions →
    contextReady env row.contextVariables →
    loopVerifiedAtEnv manifest env row

theorem loop_induction_certificate_sound
    (manifest : Manifest)
    (assumptions : List AssumptionRow)
    (row : SummaryRow)
    (proof : loopInductionCertificateFromAssumptions manifest assumptions row) :
    loopVerifiedFromAssumptions manifest assumptions row := by
  intro env hAssumptions hContext
  have hObligations := proof env hAssumptions hContext
  unfold loopInductionObligationsAtEnv at hObligations
  unfold loopVerifiedAtEnv
  cases hSummary : denoteLoopSummary? manifest env row with
  | none => simp [hSummary] at hObligations
  | some summary =>
      simp only [hSummary] at hObligations ⊢
      exact verified_of_induction summary hObligations

theorem loop_authorization_sound
    (manifest : Manifest)
    (assumptions : List AssumptionRow)
    (row : SummaryRow)
    (proof : loopProofAuthorizationFromAssumptions manifest assumptions row) :
    loopVerifiedFromAssumptions manifest assumptions row :=
  loop_induction_certificate_sound manifest assumptions row proof.2

def loopVerifiedFromAssumptionsIn
    (manifest : Manifest)
    (assumptions : List AssumptionRow)
    (rows : List SummaryRow)
    (row : SummaryRow) : Prop :=
  ∀ env,
    assumptionsDenoteInEnv manifest env assumptions →
    contextReady env row.contextVariables →
    loopVerifiedAtEnvIn manifest rows env row

theorem loop_induction_certificate_sound_in
    (manifest : Manifest)
    (assumptions : List AssumptionRow)
    (rows : List SummaryRow)
    (row : SummaryRow)
    (proof : loopInductionCertificateFromAssumptionsIn manifest assumptions rows row) :
    loopVerifiedFromAssumptionsIn manifest assumptions rows row := by
  intro env hAssumptions hContext
  have hObligations := proof env hAssumptions hContext
  unfold loopInductionObligationsAtEnvIn at hObligations
  unfold loopVerifiedAtEnvIn
  cases hSummary : denoteLoopSummaryIn? manifest rows env row with
  | none => simp [hSummary] at hObligations
  | some summary =>
      simp only [hSummary] at hObligations ⊢
      exact verified_of_induction summary hObligations

theorem loop_authorization_sound_in
    (manifest : Manifest)
    (assumptions : List AssumptionRow)
    (rows : List SummaryRow)
    (row : SummaryRow)
    (proof : loopProofAuthorizationFromAssumptionsIn manifest assumptions rows row) :
    loopVerifiedFromAssumptionsIn manifest assumptions rows row :=
  loop_induction_certificate_sound_in manifest assumptions rows row proof.2

theorem loop_authorization_has_nonvacuous_initial_state
    (manifest : Manifest)
    (assumptions : List AssumptionRow)
    (row : SummaryRow)
    (proof : loopProofAuthorizationFromAssumptions manifest assumptions row) :
    ∃ env state,
      assumptionsDenoteInEnv manifest env assumptions ∧
      contextReady env row.contextVariables ∧
      loopInitialPremises manifest env row state :=
  proof.1

/-- An authorized compiler row exposes a sound binary relation for each exact
control target.  The proof reuses the ordinary loop induction certificate; it
does not introduce a nested-loop proof rule. -/
theorem loop_authorization_control_relation_sound
    (manifest : Manifest)
    (assumptions : List AssumptionRow)
    (row : SummaryRow)
    (proof : loopProofAuthorizationFromAssumptions manifest assumptions row)
    (env : Env)
    (hAssumptions : assumptionsDenoteInEnv manifest env assumptions)
    (hContext : contextReady env row.contextVariables)
    (target : ControlTarget)
    (initial final : ScalarLoopState)
    (hRelation : loopControlRelationAtEnv manifest env row target initial final) :
    loopControlTargetAssertionAtEnv manifest env row target final := by
  have hCertificate := proof.2 env hAssumptions hContext
  unfold loopInductionObligationsAtEnv at hCertificate
  unfold loopControlRelationAtEnv at hRelation
  unfold loopControlTargetAssertionAtEnv
  cases hSummary : denoteLoopSummary? manifest env row with
  | none => simp [hSummary] at hRelation
  | some summary =>
      simp only [hSummary] at hCertificate hRelation ⊢
      exact control_relation_target_holds summary hCertificate target initial final hRelation

theorem loop_authorization_control_relation_is_certified
    (manifest : Manifest)
    (assumptions : List AssumptionRow)
    (row : SummaryRow)
    (proof : loopProofAuthorizationFromAssumptions manifest assumptions row)
    (env : Env)
    (hAssumptions : assumptionsDenoteInEnv manifest env assumptions)
    (hContext : contextReady env row.contextVariables)
    (target : ControlTarget)
    (initial final : ScalarLoopState)
    (hRelation : loopControlRelationAtEnv manifest env row target initial final) :
    loopCertifiedControlRelationAtEnv manifest env row target initial final := by
  have hCertificate := proof.2 env hAssumptions hContext
  unfold loopInductionObligationsAtEnv at hCertificate
  unfold loopControlRelationAtEnv at hRelation
  unfold loopCertifiedControlRelationAtEnv
  cases hSummary : denoteLoopSummary? manifest env row with
  | none => simp [hSummary] at hRelation
  | some summary =>
      simp only [hSummary] at hCertificate hRelation ⊢
      exact control_relation_contained_in_certified_relation summary hCertificate
        target initial final hRelation

theorem unsupported_kind_fails_closed
    (manifest : Manifest) (env : Env) (row : SummaryRow)
    (h : row.kind = .other) :
    denoteLoopSummary? manifest env row = none := by
  have hKind : (row.kind == .scfWhile || row.kind == .scfFor) = false := by
    rw [h]
    decide
  simp [denoteLoopSummary?, SummaryRow.supported, SummaryRow.shapeSupported,
    SummaryRow.baseShapeSupported, hKind]

end Ora.Loop

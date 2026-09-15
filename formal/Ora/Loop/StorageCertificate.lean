/- Canonical direct-root storage timelines interpreted by the generic loop
induction theorem.  This module adds no loop rule: it only turns emitted rows
into `LoopVC LoopState` data. -/

import Ora.Loop.Denotation

namespace Ora.Loop

open Ora.Obligation

abbrev DirectStorageEvent :=
  TimelineEvent FormulaRef StorageLocationExpr FormulaRef PEmpty

def directStorageProjectionData : StorageProjectionRow → StorageProjectionExpr
  | .field name => .field name
  | .fixedIndex projection => .fixedIndex {
      index := {
        formula := projection.index.value.formula
        ty := projection.index.ty
      }
      extent := projection.extent
    }
  | .mappingKey expression => .mappingKey {
      formula := expression.value.formula, ty := expression.ty }

def directStorageProjection? : StorageProjectionRow → Option StorageProjectionExpr
  | .field name => if name.isEmpty then none else some (.field name)
  | projection => some (directStorageProjectionData projection)

def directStorageProjections? :
    List StorageProjectionRow → Option (List StorageProjectionExpr)
  | [] => some []
  | projection :: rest => do
      pure ((← directStorageProjection? projection) ::
        (← directStorageProjections? rest))

def directStorageLocation? (row : StorageLocationExprRow) : Option StorageLocationExpr :=
  if row.root == "" then none
  else do
    pure {
      root := row.root
      projections := ← directStorageProjections? row.projections
      valueTy := row.valueTy
    }

def directIndexedRegionKind : IndexedRegionKindRow → IndexedRegionKind
  | .calldata => .calldata
  | .memory => .memory

def directIndexedRegionRead? (row : IndexedRegionReadEventRow) :
    Option (IndexedRegionRead FormulaRef) := do
  let elementType ← row.elementTy.integerShape?
  pure {
    id := row.result
    region := directIndexedRegionKind row.region
    owner := row.owner
    index := row.index.formula
    elementType
  }

def directStorageEvent? (row : TimelineEventRow) : Option (List DirectStorageEvent) :=
  match row.payload with
  | .pathAssumption event => some (event.formulas.map fun formula => .assume formula.formula)
  | .safetyCheckpoint event => some (event.formulas.map fun formula => .check formula.formula)
  | .storageRead event => do
      let location ← directStorageLocation? event.location
      pure [.storage (.read event.result location)]
  | .storageWrite event => do
      let location ← directStorageLocation? event.location
      pure [.storage (.write location event.value.formula)]
  | .indexedRegionRead event => do
      pure [.indexedRead (← directIndexedRegionRead? event)]
  | .summaryDependency _ => none

def directStorageTimeline? : List TimelineEventRow → Option (List DirectStorageEvent)
  | [] => some []
  | event :: rest => do
      let first ← directStorageEvent? event
      pure (first ++ (← directStorageTimeline? rest))

def directStorageResolvePlace
    (lookup : FreeVarId → Option Value) (place : PlaceRef) : Option StorageLocation :=
  if place.region != .storage || !place.fields.isEmpty || place.root.isEmpty then
    none
  else
    match place.keys with
    | [] =>
        if place.projectionDomain == .mapping then
          some { root := place.root }
        else
          none
    | [.parameter id] => do
        let key ← lookup id
        match place.projectionDomain with
        | .mapping =>
            pure { root := place.root, projections := [.mappingKey key] }
        | .fixedIndex =>
            match key with
            | .integer value =>
                if value.shape == .unsigned .w256 then
                  pure { root := place.root, projections := [.fixedIndex key] }
                else
                  none
            | _ => none
    | _ => none

theorem directStorageMappingLocations_ne
    (root : String) (readKey writeKey : Value)
    (hNe : readKey ≠ writeKey) :
    ({ root := root, projections := [.mappingKey readKey] } : StorageLocation) ≠
      { root := root, projections := [.mappingKey writeKey] } := by
  intro hEqual
  exact hNe (StorageProjection.mappingKey.inj
    (List.cons.inj (StorageLocation.mk.inj hEqual).2).1)

theorem directStorageFixedIndexLocations_ne
    (root : String) (readIndex writeIndex : Value)
    (hNe : readIndex ≠ writeIndex) :
    ({ root := root, projections := [.fixedIndex readIndex] } : StorageLocation) ≠
      { root := root, projections := [.fixedIndex writeIndex] } := by
  intro hEqual
  exact hNe (StorageProjection.fixedIndex.inj
    (List.cons.inj (StorageLocation.mk.inj hEqual).2).1)

/-- Once exact projection coverage resolves every timeline write into the
checked frame, frame disjointness is sufficient for store preservation. -/
theorem TimelineExec.preserves_outside_resolved_frame
    (semantics : TimelineSemantics FormulaExpr LocationExpr ValueExpr Child)
    (events : List (EventOf FormulaExpr LocationExpr ValueExpr Child))
    (initial : StorageActionState)
    (result : TimelineResult)
    (frame : List StorageLocation)
    (observed : StorageLocation)
    (hExec : TimelineExec semantics events initial result)
    (hCovered : ∀ written, result.writes written → written ∈ frame)
    (hOutside : ∀ written ∈ frame, observed ≠ written) :
    result.state.loop.storage observed = initial.loop.storage observed := by
  apply TimelineExec.preserves_outside semantics events initial result observed hExec
  intro hWritten
  exact hOutside observed (hCovered observed hWritten) rfl

def storageStateFromEnv (env : Env) (entry : Bool) : StorageState :=
  fun location =>
    let place : PlaceRef := { root := location.root, region := .storage }
    env.placeValue (if entry then .entry place else .stable place)

def directStoragePreheaderState (env : Env) (row : SummaryRow) : StorageState :=
  storageStateFromEnv env row.preheaderProjectionComplete

def directStorageContext
    (manifest : Manifest) (base : Env) (variables : List VariableRow) :
    StorageFormulaContext := {
  manifest
  entryEnv := base
  bindLocals := fun env locals => bindState? env variables locals
  resolvePlace := directStorageResolvePlace
}

/-- A function-context value not shadowed by a loop variable or timeline read
keeps the same identity in the exact environment used to resolve storage
locations.  Mapping-frame proofs use this lemma instead of replaying the two
binding implementations. -/
theorem directStorageContext_lookupFree_of_not_bound
    (manifest : Manifest)
    (base bound : Env)
    (variables : List VariableRow)
    (state : StorageActionState)
    (id : FreeVarId)
    (hLocalFresh : id ∉ variables.map VariableRow.id)
    (hReadFresh : id ∉ state.reads.map StorageReadBinding.id)
    (hBound : (directStorageContext manifest base variables).stateEnv? state =
      some bound) :
    bound.lookupFree id = base.lookupFree id := by
  unfold StorageFormulaContext.stateEnv? directStorageContext at hBound
  cases hLocals : bindState? base variables state.loop.locals with
  | none => simp [hLocals] at hBound
  | some localsBound =>
      simp only [hLocals, Option.bind_eq_bind, Option.some_bind,
        Option.pure_def] at hBound
      have hEqual := Option.some.inj hBound
      subst bound
      rw [StorageFormulaContext.withStoreViews_lookupFree,
        bindStorageReads_lookupFree_of_not_mem _ _ _ hReadFresh,
        bindState_lookupFree_of_not_mem base localsBound variables
          state.loop.locals id hLocalFresh hLocals]

def restrictStoragePreheaderEnv (base : Env) (variables : List VariableRow) : Env :=
  { base with freeBindings := base.freeBindings.filter fun binding =>
      variables.any fun declared => declared.id == binding.1 }

/-- Preheader evaluation reuses the ordinary storage semantics with an empty
local binder.  Its free environment contains only declared function-context
variables; executed reads are added by `stateEnv?` in timeline order. -/
def directStoragePreheaderContext
    (manifest : Manifest)
    (base : Env)
    (variables : List VariableRow) : StorageFormulaContext := {
  manifest
  entryEnv := restrictStoragePreheaderEnv base variables
  bindLocals := fun env locals => if locals.isEmpty then some env else none
  resolvePlace := directStorageResolvePlace
}

def directInitialValues?
    (context : StorageFormulaContext)
    (entry current : StorageState) : List FormulaRef → Option (List Ora.Integer.Value)
  | [] => some []
  | formula :: rest => do
      let value ← context.evalPreheaderValue? entry current formula
      let integer ← match value with | .integer value => some value | .bool _ => none
      pure (integer :: (← directInitialValues? context entry current rest))

def directInitialValuesAfterPreheader?
    (context : StorageFormulaContext)
    (state : StorageActionState) : List FormulaRef → Option (List Ora.Integer.Value)
  | [] => some []
  | formula :: rest => do
      let value ← context.evalValue? formula state
      let integer ← match value with | .integer value => some value | .bool _ => none
      pure (integer :: (← directInitialValuesAfterPreheader? context state rest))

theorem direct_initial_values_after_empty_preheader
    (context : StorageFormulaContext)
    (entry current : StorageState)
    (formulas : List FormulaRef)
    (hEmpty : context.bindLocals context.entryEnv [] = some context.entryEnv) :
    directInitialValuesAfterPreheader? context {
      loop := { locals := [], storage := current }
      entryStorage := entry
    } formulas = directInitialValues? context entry current formulas := by
  induction formulas with
  | nil => rfl
  | cons formula rest ih =>
      simp [directInitialValuesAfterPreheader?, directInitialValues?,
        StorageFormulaContext.evalValue?, StorageFormulaContext.stateEnv?,
        StorageFormulaContext.evalPreheaderValue?,
        StorageFormulaContext.preheaderEnv, bindStorageReads,
        hEmpty, ih]

def directFormulasHold
    (context : StorageFormulaContext)
    (entry : StorageState)
    (state : LoopState)
    (formulas : List FormulaRef) : Prop :=
  ∀ formula ∈ formulas,
    ∃ proposition,
      context.evalFormula? formula { loop := state, entryStorage := entry } = some proposition ∧
      proposition

def directStorageSemantics (context : StorageFormulaContext) :
    TimelineSemantics FormulaRef StorageLocationExpr FormulaRef PEmpty := {
  storage := context.actionSemantics
  indexedRegion := {
    snapshot := fun location => match location.region with
      | .calldata => some (context.entryEnv.indexedRegionValue
          location.owner location.index location.elementType)
      | .memory => none
    evalIndex := fun formula state => do
      match ← context.evalValue? formula state with
      | .integer value => some value
      | .bool _ => none
  }
  -- Control formulas in this authority boundary are read-independent. Storage
  -- write values still use `actionSemantics`, which observes bound read IDs.
  evalFormula := fun formula state => context.evalFormula? formula { state with reads := [] }
  childRelation := fun child => nomatch child
  childFrame := fun child => nomatch child
}

abbrev directStoragePreheaderSemantics := directStorageSemantics

def directStoragePreheaderInitialState (base : Env) : StorageActionState :=
  let entry := storageStateFromEnv base true
  {
    loop := { locals := [], storage := entry }
    entryStorage := entry
  }

/-- Exact preheader execution is part of the loop's initial predicate.  The
`TimelineSafe` conjunct prevents an always-failing prefix from manufacturing a
vacuous loop certificate. -/
def directStoragePreheaderProducesInitial
    (context : StorageFormulaContext)
    (events : List DirectStorageEvent)
    (initial : StorageActionState)
    (formulas : List FormulaRef)
    (following : LoopState) : Prop :=
  ∃ result,
    TimelineExec (directStoragePreheaderSemantics context) events initial result ∧
      TimelineSafe (directStoragePreheaderSemantics context) events initial ∧
      directInitialValuesAfterPreheader? context result.state formulas =
        some following.locals ∧
      following.storage = result.state.loop.storage

/-- Lean-first denotation seam only.  The live storage-summary denotation keeps
its existing empty-prefix support predicate until the later authority task. -/
def denoteDirectStoragePreheaderInitial?
    (manifest : Manifest)
    (base : Env)
    (row : SummaryRow) : Option (LoopState → Prop) := do
  if !row.preheaderProjectionComplete then none else pure ()
  let events ← directStorageTimeline? row.preheaderTimeline
  let context := directStoragePreheaderContext manifest base row.contextVariables
  let initial := directStoragePreheaderInitialState base
  pure (directStoragePreheaderProducesInitial context events initial row.init)

theorem direct_storage_preheader_empty_iff
    (context : StorageFormulaContext)
    (initial : StorageActionState)
    (formulas : List FormulaRef)
    (following : LoopState) :
    directStoragePreheaderProducesInitial context [] initial formulas following ↔
      directInitialValuesAfterPreheader? context initial formulas =
          some following.locals ∧
        following.storage = initial.loop.storage := by
  constructor
  · rintro ⟨result, hExec, _, hValues, hStorage⟩
    cases hExec
    exact ⟨hValues, hStorage⟩
  · rintro ⟨hValues, hStorage⟩
    refine ⟨{ state := initial }, .nil initial, ?_, hValues, hStorage⟩
    simp [TimelineSafe]

theorem direct_storage_preheader_empty_matches_legacy
    (context : StorageFormulaContext)
    (entry current : StorageState)
    (formulas : List FormulaRef)
    (following : LoopState)
    (hEmpty : context.bindLocals context.entryEnv [] = some context.entryEnv) :
    directStoragePreheaderProducesInitial context [] {
      loop := { locals := [], storage := current }
      entryStorage := entry
    } formulas following ↔
      directInitialValues? context entry current formulas = some following.locals ∧
        following.storage = current := by
  rw [direct_storage_preheader_empty_iff,
    direct_initial_values_after_empty_preheader context entry current formulas hEmpty]

theorem direct_storage_preheader_initial_has_safe_execution
    (context : StorageFormulaContext)
    (events : List DirectStorageEvent)
    (initial : StorageActionState)
    (formulas : List FormulaRef)
    (following : LoopState)
    (hInitial : directStoragePreheaderProducesInitial
      context events initial formulas following) :
    ∃ result,
      TimelineExec (directStoragePreheaderSemantics context) events initial result ∧
        TimelineSafe (directStoragePreheaderSemantics context) events initial := by
  rcases hInitial with ⟨result, hExec, hSafe, _, _⟩
  exact ⟨result, hExec, hSafe⟩

theorem direct_storage_preheader_initial_preserves_entry
    (context : StorageFormulaContext)
    (events : List DirectStorageEvent)
    (initial : StorageActionState)
    (formulas : List FormulaRef)
    (following : LoopState)
    (hInitial : directStoragePreheaderProducesInitial
      context events initial formulas following) :
    ∃ result,
      TimelineExec (directStoragePreheaderSemantics context) events initial result ∧
        result.state.entryStorage = initial.entryStorage ∧
        following.storage = result.state.loop.storage := by
  rcases hInitial with ⟨result, hExec, _, _, hStorage⟩
  exact ⟨result, hExec,
    TimelineExec.preserves_entryStorage _ _ _ _ hExec, hStorage⟩

def storageReadsForIds : List StorageReadId → List StorageReadBinding →
    List StorageReadBinding
  | [], _ => []
  | _, [] => []
  | ids, read :: rest =>
      if ids.contains read.id then read :: storageReadsForIds ids rest
      else storageReadsForIds ids rest

def storageReadsForBindings
    (bindings : List StateBindingRow)
    (reads : List StorageReadBinding) : List StorageReadBinding :=
  let used := bindings.flatMap StateBindingRow.readResults
  storageReadsForIds used reads

def directStorageFinalLocals?
    (context : StorageFormulaContext)
    (variables : List VariableRow)
    (bindings : List StateBindingRow)
    (state : StorageActionState) : Option (List Ora.Integer.Value) := do
  let env ← context.stateEnv? {
    state with reads := storageReadsForBindings bindings state.reads
  }
  applyStateBindingsInEnv? context.manifest variables env bindings

def directFormulaControlCompatible
    (manifest : Manifest)
    (base : Env)
    (variables : List VariableRow)
    (formula : FormulaRef) : Prop :=
  ∀ state entryStorage,
    timelineFormulaHolds
        (directStorageSemantics (directStorageContext manifest base variables))
        formula { loop := state, entryStorage := entryStorage, reads := [] } ↔
      formulaHoldsInState manifest base variables state.locals formula

def directInitialControlCompatible
    (manifest : Manifest)
    (base : Env)
    (variables : List VariableRow)
    (formulas : List FormulaRef) : Prop :=
  ∀ entryStorage currentStorage,
    directInitialValues? (directStorageContext manifest base variables)
        entryStorage currentStorage formulas =
      initializeState? manifest base variables formulas

theorem option_prop_witness_iff (result : Option Prop) :
    (∃ proposition, result = some proposition ∧ proposition) ↔
      match result with
      | some proposition => proposition
      | none => False := by
  cases result with
  | none => simp
  | some proposition =>
      constructor
      · rintro ⟨candidate, hCandidate, hHolds⟩
        have hEqual : proposition = candidate := Option.some.inj hCandidate
        exact hEqual ▸ hHolds
      · intro hHolds
        exact ⟨proposition, rfl, hHolds⟩

theorem direct_formulas_control_compatible
    (manifest : Manifest)
    (base : Env)
    (variables : List VariableRow)
    (formulas : List FormulaRef)
    (hCompatible : ∀ formula, formula ∈ formulas →
      directFormulaControlCompatible manifest base variables formula)
    (state : LoopState)
    (entryStorage : StorageState) :
    directFormulasHold (directStorageContext manifest base variables)
        entryStorage state formulas ↔
      formulasHoldInState manifest base variables state.locals formulas := by
  constructor
  · intro hExact formula hMember
    exact (hCompatible formula hMember state entryStorage).mp
      (hExact formula hMember)
  · intro hControl formula hMember
    exact (hCompatible formula hMember state entryStorage).mpr
      (hControl formula hMember)

def directTargetFormulas
    (successorPost : List FormulaRef) (row : EdgeRow) : List FormulaRef :=
  match row.target with
  | .successor => successorPost
  | .functionReturn | .errorReturn => row.targetFormulas
  | .backedge | .explicitRevert | .unresolved => []

/- All proof-search facts for one emitted direct-storage edge.  The row itself
selects the formulas, timeline, bindings, and target; no edge id or source
shape is represented in this interface. -/
def directStorageEventData (row : TimelineEventRow) : List DirectStorageEvent :=
  match row.payload with
  | .pathAssumption event => event.formulas.map fun formula => .assume formula.formula
  | .safetyCheckpoint event => event.formulas.map fun formula => .check formula.formula
  | .storageWrite event =>
      [.storage (.write {
        root := event.location.root
        projections := event.location.projections.map directStorageProjectionData
        valueTy := event.location.valueTy
      } event.value.formula)]
  | .storageRead event =>
      [.storage (.read event.result {
        root := event.location.root
        projections := event.location.projections.map directStorageProjectionData
        valueTy := event.location.valueTy
      })]
  | .indexedRegionRead event =>
      match directIndexedRegionRead? event with
      | some read => [.indexedRead read]
      | none => []
  | .summaryDependency _ => []

def directStorageEvents (row : EdgeRow) : List DirectStorageEvent :=
  row.timeline.flatMap directStorageEventData

theorem directStorageCheck_mem_iff
    (events : List TimelineEventRow)
    (dependencyCount checkpointId : Nat)
    (formula : FormulaRef) :
    TimelineEvent.check formula ∈ events.flatMap directStorageEventData ↔
      ∃ checkpoint,
        checkpoint ∈ timelineSafetyViewFrom dependencyCount checkpointId events ∧
          formula ∈ checkpoint.formulas := by
  induction events generalizing dependencyCount checkpointId with
  | nil => simp [timelineSafetyViewFrom]
  | cons event rest ih =>
      rcases event with ⟨id, key, payload⟩
      cases payload with
      | pathAssumption path =>
          simp only [List.flatMap_cons, directStorageEventData,
            timelineSafetyViewFrom, List.mem_append, List.mem_map,
            TimelineEvent.check.injEq]
          rw [ih dependencyCount checkpointId]
          simp
      | safetyCheckpoint checkpoint =>
          simp only [List.flatMap_cons, directStorageEventData,
            timelineSafetyViewFrom, List.mem_append, List.mem_map,
            TimelineEvent.check.injEq, List.mem_cons]
          rw [ih dependencyCount (checkpointId + 1)]
          constructor
          · rintro (⟨eventFormula, hMember, hEqual⟩ | hRest)
            · subst formula
              exact ⟨_, Or.inl rfl, List.mem_map.mpr ⟨eventFormula, hMember, rfl⟩⟩
            · rcases hRest with ⟨row, rowMember, formulaMember⟩
              exact ⟨row, Or.inr rowMember, formulaMember⟩
          · rintro ⟨row, rfl | rowMember, formulaMember⟩
            · rcases List.mem_map.mp formulaMember with
                ⟨eventFormula, hMember, hEqual⟩
              exact Or.inl ⟨eventFormula, hMember, hEqual⟩
            · exact Or.inr ⟨row, rowMember, formulaMember⟩
      | storageRead read =>
          simpa only [List.flatMap_cons, directStorageEventData,
            timelineSafetyViewFrom, List.mem_append, List.mem_singleton,
            TimelineEvent.check_ne_storage, false_or] using
            ih (dependencyCount := dependencyCount)
              (checkpointId := checkpointId)
      | storageWrite write =>
          simpa only [List.flatMap_cons, directStorageEventData,
            timelineSafetyViewFrom, List.mem_append, List.mem_singleton,
            TimelineEvent.check_ne_storage, false_or] using
            ih (dependencyCount := dependencyCount)
              (checkpointId := checkpointId)
      | indexedRegionRead read =>
          cases hRead : directIndexedRegionRead? read <;>
            simpa only [List.flatMap_cons, directStorageEventData, hRead,
              timelineSafetyViewFrom, List.nil_append, List.mem_append,
              List.mem_singleton, TimelineEvent.check_ne_indexedRead,
              false_or] using
              ih (dependencyCount := dependencyCount)
                (checkpointId := checkpointId)
      | summaryDependency dependency =>
          simpa only [List.flatMap_cons, directStorageEventData,
            timelineSafetyViewFrom, List.nil_append] using
            ih (dependencyCount := dependencyCount + 1)
              (checkpointId := checkpointId)

theorem directStorageCheck_mem_iff_row
    (row : EdgeRow) (formula : FormulaRef) :
    TimelineEvent.check formula ∈ directStorageEvents row ↔
      ∃ checkpoint,
        checkpoint ∈ row.safetyCheckpoints ∧ formula ∈ checkpoint.formulas := by
  exact directStorageCheck_mem_iff row.timeline 0 0 formula

def directStorageTarget (row : EdgeRow) : ControlTarget :=
  match row.target with
  | .backedge => .backedge
  | .successor => .successor
  | .functionReturn => .functionReturn
  | .errorReturn => .errorReturn
  | .explicitRevert | .unresolved => .explicitRevert

structure DirectStorageEdgeCompatible
    (manifest : Manifest)
    (base : Env)
    (variables : List VariableRow)
    (successorPost : List FormulaRef)
    (row : EdgeRow) : Prop where
  timeline : directStorageTimeline? row.timeline = some (directStorageEvents row)
  targetDenotes : denoteControlTarget? row.target = some (directStorageTarget row)
  assumptionPrefix : directStorageEvents row =
    row.path.map TimelineEvent.assume ++
      (directStorageEvents row).drop row.path.length
  pathFormulaCompatible : ∀ formula, formula ∈ row.path →
    directFormulaControlCompatible manifest base variables formula
  checkFormulaCompatible : ∀ formula,
    TimelineEvent.check formula ∈ directStorageEvents row →
      directFormulaControlCompatible manifest base variables formula
  finalLocalsAfterTrace : ∀ current result locals,
    TimelineExec
      (directStorageSemantics (directStorageContext manifest base variables))
      (directStorageEvents row)
      { loop := current, entryStorage := storageStateFromEnv base true }
      result →
    directStorageFinalLocals?
      (directStorageContext manifest base variables) variables row.nextState
      result.state = some locals →
    stateBindingsProduceState manifest base variables current.locals
      row.nextState locals
  targetFormulaCompatible : ∀ formula,
    formula ∈ directTargetFormulas successorPost row →
      directFormulaControlCompatible manifest base variables formula

theorem DirectStorageEdgeCompatible.timelineSafety
    (proof : DirectStorageEdgeCompatible manifest base variables successorPost row)
    (current : LoopState)
    (hReady : stateReady variables current.locals)
    (hControlSafety :
      formulasHoldInState manifest base variables current.locals row.path →
        safetyCheckpointsHold manifest base variables current.locals []
          row.safetyCheckpoints) :
    TimelineSafe
      (directStorageSemantics (directStorageContext manifest base variables))
      (directStorageEvents row)
      { loop := current, entryStorage := storageStateFromEnv base true } := by
  rw [proof.assumptionPrefix]
  apply timelineSafe_of_leading_control
    (Path := formulasHoldInState manifest base variables current.locals row.path)
  · intro before formula after result hEvents hExec hAtInitial
    have hOriginal : TimelineEvent.check formula ∈ directStorageEvents row := by
      rw [proof.assumptionPrefix, hEvents]
      simp
    have hLocals := TimelineExec.preserves_locals _ _ _ _
      (by intro child; nomatch child) hExec
    have hCompatible := proof.checkFormulaCompatible formula hOriginal
    have hInitialControl := (hCompatible current
      (storageStateFromEnv base true)).mp hAtInitial
    have hResultControl : formulaHoldsInState manifest base variables
        result.state.loop.locals formula := by
      rw [hLocals]
      exact hInitialControl
    exact (hCompatible result.state.loop result.state.entryStorage).mpr hResultControl
  · intro hAssumptions formula hMember
    exact (proof.pathFormulaCompatible formula hMember current
      (storageStateFromEnv base true)).mp (hAssumptions formula hMember)
  · intro hPath formula hMember
    have hOriginal : TimelineEvent.check formula ∈ directStorageEvents row := by
      rw [proof.assumptionPrefix]
      exact hMember
    rcases (directStorageCheck_mem_iff_row row formula).mp hOriginal with
      ⟨checkpoint, hCheckpoint, hFormula⟩
    have hScalar := safetyCheckpointsHold_no_dependencies_formula
      manifest base variables current.locals row.safetyCheckpoints
      checkpoint formula hReady (hControlSafety hPath) hCheckpoint hFormula
    exact (proof.checkFormulaCompatible formula hOriginal current
      (storageStateFromEnv base true)).mpr hScalar

theorem DirectStorageEdgeCompatible.refinement
    (proof : DirectStorageEdgeCompatible manifest base variables successorPost row)
    (abstractVC : LoopVC ScalarLoopState)
    (concreteVC : LoopVC LoopState)
    (hAbstractReady : ∀ state,
      abstractVC.invariant state → stateReady variables state) :
    Edge.Refinement abstractVC concreteVC (fun state => state.locals)
      {
        path := fun state =>
          formulasHoldInState manifest base variables state row.path
        safe := fun state =>
          safetyCheckpointsHold manifest base variables state [] row.safetyCheckpoints
        next := fun current following =>
          stateBindingsProduceState manifest base variables current row.nextState following
        target := directStorageTarget row
        targetAssertion := fun state => formulasHoldInState manifest base variables
          state (directTargetFormulas successorPost row)
      }
      (timelineEdge
        (directStorageSemantics (directStorageContext manifest base variables))
        (directStorageEvents row) (storageStateFromEnv base true)
        (directStorageFinalLocals?
          (directStorageContext manifest base variables) variables row.nextState)
        (directStorageTarget row)
        (fun state => directFormulasHold
          (directStorageContext manifest base variables)
          (storageStateFromEnv base true) state
          (directTargetFormulas successorPost row))) := by
  apply timelineEdge_refinement
  · rfl
  · intro state _ hAbstractInvariant hAbstractSafety
    exact proof.timelineSafety state
      (hAbstractReady state.locals hAbstractInvariant) hAbstractSafety
  · intro current result hExec
    rw [proof.assumptionPrefix] at hExec
    have hAssumptions := TimelineExec.leading_assumptions_hold _ row.path
      ((directStorageEvents row).drop row.path.length)
      { loop := current, entryStorage := storageStateFromEnv base true }
      result hExec
    intro formula hMember
    exact (proof.pathFormulaCompatible formula hMember current
      (storageStateFromEnv base true)).mp (hAssumptions formula hMember)
  · intro current result locals hExec hLocals
    exact proof.finalLocalsAfterTrace current result locals hExec hLocals
  · intro following hAssertion
    exact (direct_formulas_control_compatible manifest base variables
      (directTargetFormulas successorPost row) proof.targetFormulaCompatible
      following (storageStateFromEnv base true)).mpr hAssertion

def denoteDirectStorageEdge?
    (manifest : Manifest)
    (base : Env)
    (variables : List VariableRow)
    (successorPost : List FormulaRef)
    (row : EdgeRow) : Option (Edge LoopState) := do
  let events ← directStorageTimeline? row.timeline
  let target ← denoteControlTarget? row.target
  let context := directStorageContext manifest base variables
  let entry := storageStateFromEnv base true
  let targetFormulas := match row.target with
    | .successor => successorPost
    | .functionReturn | .errorReturn => row.targetFormulas
    | .backedge | .explicitRevert | .unresolved => []
  pure (timelineEdge (directStorageSemantics context) events entry
    (directStorageFinalLocals? context variables row.nextState) target
    (fun state => directFormulasHold context entry state targetFormulas))

theorem DirectStorageEdgeCompatible.denotes
    (proof : DirectStorageEdgeCompatible manifest base variables successorPost row) :
    ∃ edge,
      denoteDirectStorageEdge? manifest base variables successorPost row = some edge := by
  refine ⟨timelineEdge
    (directStorageSemantics (directStorageContext manifest base variables))
    (directStorageEvents row) (storageStateFromEnv base true)
    (directStorageFinalLocals?
      (directStorageContext manifest base variables) variables row.nextState)
    (directStorageTarget row)
    (fun state => directFormulasHold
      (directStorageContext manifest base variables)
      (storageStateFromEnv base true) state
      (directTargetFormulas successorPost row)), ?_⟩
  unfold denoteDirectStorageEdge?
  rw [proof.timeline, proof.targetDenotes]
  rfl

def denoteDirectStorageEdges?
    (manifest : Manifest) (base : Env) (variables : List VariableRow)
    (successorPost : List FormulaRef) : List EdgeRow → Option (List (Edge LoopState))
  | [] => some []
  | edge :: rest => do
      pure ((← denoteDirectStorageEdge? manifest base variables successorPost edge) ::
        (← denoteDirectStorageEdges? manifest base variables successorPost rest))

theorem direct_storage_edges_refinements
    (manifest : Manifest)
    (base : Env)
    (variables : List VariableRow)
    (successorPost : List FormulaRef)
    (rows : List EdgeRow)
    (abstractVC : LoopVC ScalarLoopState)
    (concreteVC : LoopVC LoopState)
    (hAbstractReady : ∀ state,
      abstractVC.invariant state → stateReady variables state)
    (compatible : ∀ row, row ∈ rows →
      DirectStorageEdgeCompatible manifest base variables successorPost row)
    (abstractEdges : List (Edge ScalarLoopState))
    (concreteEdges : List (Edge LoopState))
    (hAbstract : denoteEdges? manifest base variables successorPost rows =
      some abstractEdges)
    (hConcrete : denoteDirectStorageEdges? manifest base variables successorPost rows =
      some concreteEdges) :
    LoopVC.EdgeRefinements abstractVC concreteVC (fun state => state.locals)
      abstractEdges concreteEdges := by
  induction rows generalizing abstractEdges concreteEdges with
  | nil =>
      simp only [denoteEdges?, denoteDirectStorageEdges?, Option.some.injEq]
        at hAbstract hConcrete
      subst abstractEdges
      subst concreteEdges
      exact .nil
  | cons row rest ih =>
      have head := compatible row (by simp)
      have tailCompatible : ∀ tail, tail ∈ rest →
          DirectStorageEdgeCompatible manifest base variables successorPost tail := by
        intro tail hMember
        exact compatible tail (by simp [hMember])
      unfold denoteEdges? at hAbstract
      unfold denoteDirectStorageEdges? at hConcrete
      unfold denoteEdge? at hAbstract
      unfold denoteDirectStorageEdge? at hConcrete
      rw [head.targetDenotes] at hAbstract hConcrete
      rw [head.timeline] at hConcrete
      simp only [Option.bind_eq_bind, Option.some_bind, Option.pure_def]
        at hAbstract hConcrete
      cases hAbstractRest : denoteEdges? manifest base variables successorPost rest with
      | none => simp [hAbstractRest] at hAbstract
      | some abstractRest =>
          cases hConcreteRest : denoteDirectStorageEdges? manifest base variables
              successorPost rest with
          | none => simp [hConcreteRest] at hConcrete
          | some concreteRest =>
              simp only [hAbstractRest, hConcreteRest, Option.some_bind,
                Option.some.injEq] at hAbstract hConcrete
              subst abstractEdges
              subst concreteEdges
              apply LoopVC.EdgeRefinements.cons
              · simpa only [directTargetFormulas] using
                  head.refinement abstractVC concreteVC hAbstractReady
              · exact ih tailCompatible abstractRest concreteRest
                  hAbstractRest hConcreteRest

theorem direct_storage_edges_denote
    (manifest : Manifest)
    (base : Env)
    (variables : List VariableRow)
    (successorPost : List FormulaRef)
    (rows : List EdgeRow)
    (timelines : ∀ row, row ∈ rows →
      directStorageTimeline? row.timeline = some (directStorageEvents row))
    (targets : ∀ row, row ∈ rows →
      denoteControlTarget? row.target = some (directStorageTarget row)) :
    ∃ edges,
      denoteDirectStorageEdges? manifest base variables successorPost rows =
        some edges := by
  induction rows with
  | nil => exact ⟨[], rfl⟩
  | cons row rest ih =>
      have tailTimelines : ∀ tail, tail ∈ rest →
          directStorageTimeline? tail.timeline =
            some (directStorageEvents tail) := by
        intro tail hMember
        exact timelines tail (by simp [hMember])
      have tailTargets : ∀ tail, tail ∈ rest →
          denoteControlTarget? tail.target =
            some (directStorageTarget tail) := by
        intro tail hMember
        exact targets tail (by simp [hMember])
      rcases ih tailTimelines tailTargets with ⟨tailEdges, hTail⟩
      let headEdge := timelineEdge
        (directStorageSemantics (directStorageContext manifest base variables))
        (directStorageEvents row) (storageStateFromEnv base true)
        (directStorageFinalLocals?
          (directStorageContext manifest base variables) variables row.nextState)
        (directStorageTarget row)
        (fun state => directFormulasHold
          (directStorageContext manifest base variables)
          (storageStateFromEnv base true) state
          (directTargetFormulas successorPost row))
      have hHead : denoteDirectStorageEdge? manifest base variables
          successorPost row = some headEdge := by
        unfold denoteDirectStorageEdge?
        rw [timelines row (by simp), targets row (by simp)]
        rfl
      refine ⟨headEdge :: tailEdges, ?_⟩
      unfold denoteDirectStorageEdges?
      rw [hHead, hTail]
      rfl

def directStorageSummarySupported (manifest : Manifest) (row : SummaryRow) : Bool :=
  row.supported manifest &&
    (row.preheaderTimeline.isEmpty || row.preheaderProjectionComplete) &&
    (row.preheaderTimeline.any (fun event =>
      match event.payload with
      | .storageRead _ | .indexedRegionRead _ | .storageWrite _ => true
      | _ => false) || row.edges.any (fun edge => edge.timeline.any fun event =>
      match event.payload with
      | .storageRead _ | .indexedRegionRead _ | .storageWrite _ => true
      | _ => false)) &&
    row.edges.all (fun edge => edge.timeline.all fun event =>
      match event.payload with
      | .summaryDependency _ => false
      | _ => true)

def denoteDirectStorageLoopSummary?
    (manifest : Manifest) (base : Env) (row : SummaryRow) : Option (LoopVC LoopState) :=
  if directStorageSummarySupported manifest row then do
    let edges ← denoteDirectStorageEdges? manifest base row.variables row.post row.edges
    let context := directStorageContext manifest base row.variables
    let entry := storageStateFromEnv base true
    let current := directStoragePreheaderState base row
    let initial ← if row.preheaderTimeline.isEmpty then
      pure (fun state => state.storage = current ∧
        directInitialValues? context entry current row.init = some state.locals)
    else
      denoteDirectStoragePreheaderInitial? manifest base row
    pure {
      initial := fun state => headerFactsHoldInEnv manifest base row.headerFacts ∧ initial state
      invariant := fun state =>
        stateReady row.variables state.locals ∧
          headerFactsHoldInEnv manifest base row.headerFacts ∧
          directFormulasHold context entry state row.invariants
      edges
    }
  else none

theorem direct_storage_summary_denotes
    (manifest : Manifest)
    (base : Env)
    (row : SummaryRow)
    (hDirect : directStorageSummarySupported manifest row = true)
    (hEmpty : row.preheaderTimeline.isEmpty = true)
    (hTimelines : ∀ edge, edge ∈ row.edges →
      directStorageTimeline? edge.timeline = some (directStorageEvents edge))
    (hTargets : ∀ edge, edge ∈ row.edges →
      denoteControlTarget? edge.target = some (directStorageTarget edge)) :
    ∃ summary, denoteDirectStorageLoopSummary? manifest base row = some summary := by
  rcases direct_storage_edges_denote manifest base row.variables row.post
    row.edges hTimelines hTargets with ⟨edges, hDenotes⟩
  let context := directStorageContext manifest base row.variables
  let entry := storageStateFromEnv base true
  let current := directStoragePreheaderState base row
  refine ⟨{
    initial := fun state => headerFactsHoldInEnv manifest base row.headerFacts ∧
      state.storage = current ∧
      directInitialValues? context entry current row.init = some state.locals
    invariant := fun state => stateReady row.variables state.locals ∧
      headerFactsHoldInEnv manifest base row.headerFacts ∧
      directFormulasHold context entry state row.invariants
    edges := edges
  }, ?_⟩
  unfold denoteDirectStorageLoopSummary?
  rw [hDirect, hDenotes, hEmpty]
  rfl

theorem scalar_initial_values_of_denotation
    (manifest : Manifest)
    (base : Env)
    (row : SummaryRow)
    (summary : LoopVC ScalarLoopState)
    (locals : ScalarLoopState)
    (hSummary : denoteLoopSummary? manifest base row = some summary)
    (hInitial : summary.initial locals) :
    headerFactsHoldInEnv manifest base row.headerFacts ∧
      valuesInitializeState manifest base row.variables row.init locals := by
  unfold denoteLoopSummary? at hSummary
  have hSupported : row.supported manifest = true := by
    cases hValue : row.supported manifest <;> simp [hValue] at hSummary ⊢
  simp only [hSupported, if_true] at hSummary
  cases hEdges : denoteEdges? manifest base row.variables row.post row.edges with
  | none => simp [hEdges] at hSummary
  | some edges =>
      simp only [hEdges, Option.bind_eq_bind, Option.some_bind,
        Option.pure_def, Option.some.injEq] at hSummary
      subst summary
      exact hInitial

theorem direct_initial_of_control_denotation
    (manifest : Manifest)
    (base : Env)
    (row : SummaryRow)
    (summary : LoopVC LoopState)
    (locals : ScalarLoopState)
    (hSummary : denoteDirectStorageLoopSummary? manifest base row = some summary)
    (hEmpty : row.preheaderTimeline.isEmpty = true)
    (hCompatible : directInitialControlCompatible manifest base row.variables row.init)
    (hInitial : headerFactsHoldInEnv manifest base row.headerFacts ∧
      valuesInitializeState manifest base row.variables row.init locals) :
    summary.initial {
      locals := locals
      storage := directStoragePreheaderState base row
    } := by
  unfold denoteDirectStorageLoopSummary? at hSummary
  have hDirect : directStorageSummarySupported manifest row = true := by
    cases hValue : directStorageSummarySupported manifest row <;>
      simp [hValue] at hSummary ⊢
  simp only [hDirect, if_true, hEmpty] at hSummary
  cases hEdges : denoteDirectStorageEdges? manifest base row.variables row.post row.edges with
  | none => simp [hEdges] at hSummary
  | some edges =>
      simp only [hEdges, Option.bind_eq_bind, Option.some_bind,
        Option.pure_def, Option.some.injEq] at hSummary
      subst summary
      refine ⟨hInitial.1, rfl, ?_⟩
      unfold valuesInitializeState at hInitial
      rw [hCompatible]
      exact hInitial.2

theorem direct_storage_summary_refinement
    (manifest : Manifest)
    (base : Env)
    (row : SummaryRow)
    (abstractVC : LoopVC ScalarLoopState)
    (concreteVC : LoopVC LoopState)
    (hAbstract : denoteLoopSummary? manifest base row = some abstractVC)
    (hConcrete : denoteDirectStorageLoopSummary? manifest base row = some concreteVC)
    (hEmpty : row.preheaderTimeline.isEmpty = true)
    (hInitial : directInitialControlCompatible manifest base row.variables row.init)
    (hInvariant : ∀ formula, formula ∈ row.invariants →
      directFormulaControlCompatible manifest base row.variables formula)
    (hEdges : ∀ edge, edge ∈ row.edges →
      DirectStorageEdgeCompatible manifest base row.variables row.post edge) :
    LoopVC.Refinement abstractVC concreteVC (fun state => state.locals) := by
  unfold denoteLoopSummary? at hAbstract
  unfold denoteDirectStorageLoopSummary? at hConcrete
  have hSupported : row.supported manifest = true := by
    cases hValue : row.supported manifest <;> simp [hValue] at hAbstract ⊢
  have hDirect : directStorageSummarySupported manifest row = true := by
    cases hValue : directStorageSummarySupported manifest row <;>
      simp [hValue] at hConcrete ⊢
  simp only [hSupported, hDirect, if_true, hEmpty] at hAbstract hConcrete
  cases hAbstractEdges : denoteEdges? manifest base row.variables row.post row.edges with
  | none => simp [hAbstractEdges] at hAbstract
  | some abstractEdges =>
      cases hConcreteEdges : denoteDirectStorageEdges? manifest base row.variables
          row.post row.edges with
      | none => simp [hConcreteEdges] at hConcrete
      | some concreteEdges =>
          simp only [hAbstractEdges, hConcreteEdges, Option.some_bind,
            Option.bind_eq_bind, Option.pure_def, Option.some.injEq]
            at hAbstract hConcrete
          subst abstractVC
          subst concreteVC
          refine {
            initial := ?_
            invariant := ?_
            edges := direct_storage_edges_refinements manifest base row.variables
              row.post row.edges _ _
              (fun _ (hInvariant : stateReady row.variables _ ∧ _) => hInvariant.1)
              hEdges abstractEdges concreteEdges
              hAbstractEdges hConcreteEdges
          }
          · intro state hExact
            refine ⟨hExact.1, ?_⟩
            unfold valuesInitializeState
            rw [← hInitial (storageStateFromEnv base true)
              (directStoragePreheaderState base row)]
            exact hExact.2.2
          · intro state
            exact and_congr Iff.rfl (and_congr Iff.rfl
              (direct_formulas_control_compatible manifest base row.variables
                row.invariants hInvariant state (storageStateFromEnv base true)))

def directStorageLoopProofAuthorizationFromAssumptions
    (manifest : Manifest) (assumptions : List AssumptionRow) (row : SummaryRow) : Prop :=
  (∃ env state,
    assumptionsDenoteInEnv manifest env assumptions ∧
    contextReady env row.contextVariables ∧
    match denoteDirectStorageLoopSummary? manifest env row with
    | some summary => summary.initial state
    | none => False) ∧
  (∀ env,
    assumptionsDenoteInEnv manifest env assumptions →
    contextReady env row.contextVariables →
    match denoteDirectStorageLoopSummary? manifest env row with
    | some summary => summary.InductionObligations
    | none => False)

def directStorageLoopVerifiedFromAssumptions
    (manifest : Manifest) (assumptions : List AssumptionRow) (row : SummaryRow) : Prop :=
  ∀ env,
    assumptionsDenoteInEnv manifest env assumptions →
    contextReady env row.contextVariables →
    match denoteDirectStorageLoopSummary? manifest env row with
    | some summary => summary.Verified
    | none => False

theorem direct_storage_loop_authorization_sound
    (manifest : Manifest) (assumptions : List AssumptionRow) (row : SummaryRow)
    (proof : directStorageLoopProofAuthorizationFromAssumptions manifest assumptions row) :
    directStorageLoopVerifiedFromAssumptions manifest assumptions row := by
  intro env hAssumptions hContext
  have hInduction := proof.2 env hAssumptions hContext
  cases hSummary : denoteDirectStorageLoopSummary? manifest env row with
  | none => simp [hSummary] at hInduction
  | some summary =>
      simp only [hSummary] at hInduction ⊢
      exact verified_of_induction summary hInduction

end Ora.Loop

/- Canonical direct-root storage timelines interpreted by the generic loop
induction theorem.  This module adds no loop rule: it only turns emitted rows
into `LoopVC LoopState` data. -/

import Ora.Loop.Denotation

namespace Ora.Loop

open Ora.Obligation

abbrev DirectStorageEvent :=
  TimelineEvent FormulaRef StorageLocationExpr FormulaRef PEmpty

def directStorageLocation? (row : StorageLocationExprRow) : Option StorageLocationExpr :=
  if row.root == "" || !row.projections.isEmpty then none
  else some { root := row.root, valueTy := row.valueTy }

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
  | .summaryDependency _ => none

def directStorageTimeline? : List TimelineEventRow → Option (List DirectStorageEvent)
  | [] => some []
  | event :: rest => do
      let first ← directStorageEvent? event
      pure (first ++ (← directStorageTimeline? rest))

def directStorageResolvePlace
    (_ : FreeVarId → Option Value) (place : PlaceRef) : Option StorageLocation :=
  if place.region == .storage && place.fields.isEmpty && place.keys.isEmpty &&
      !place.root.isEmpty then
    some { root := place.root }
  else
    none

def storageStateFromEnv (env : Env) (entry : Bool) : StorageState :=
  fun location =>
    let place : PlaceRef := { root := location.root, region := .storage }
    env.placeValue (if entry then .entry place else .stable place)

def directStorageContext
    (manifest : Manifest) (base : Env) (variables : List VariableRow) :
    StorageFormulaContext := {
  manifest
  entryEnv := base
  bindLocals := fun env locals => bindState? env variables locals
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
  evalFormula := context.evalFormula?
  childRelation := fun child => nomatch child
  childFrame := fun child => nomatch child
}

def directStorageFinalLocals?
    (context : StorageFormulaContext)
    (variables : List VariableRow)
    (bindings : List StateBindingRow)
    (state : StorageActionState) : Option (List Ora.Integer.Value) := do
  let env ← context.stateEnv? state
  applyStateBindingsInEnv? context.manifest variables env bindings

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

def denoteDirectStorageEdges?
    (manifest : Manifest) (base : Env) (variables : List VariableRow)
    (successorPost : List FormulaRef) : List EdgeRow → Option (List (Edge LoopState))
  | [] => some []
  | edge :: rest => do
      pure ((← denoteDirectStorageEdge? manifest base variables successorPost edge) ::
        (← denoteDirectStorageEdges? manifest base variables successorPost rest))

def directStorageSummarySupported (manifest : Manifest) (row : SummaryRow) : Bool :=
  row.supported manifest &&
    row.edges.any (fun edge => edge.timeline.any fun event =>
      match event.payload with
      | .storageRead _ | .storageWrite _ => true
      | _ => false) &&
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
    let current := storageStateFromEnv base false
    pure {
      initial := fun state =>
        state.storage = current ∧
          directInitialValues? context entry current row.init = some state.locals
      invariant := fun state =>
        stateReady row.variables state.locals ∧
          directFormulasHold context entry state row.invariants
      edges
    }
  else none

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

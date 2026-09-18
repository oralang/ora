import Ora.Loop.Auto
import Ora.Loop.Call

namespace Ora.Loop
open Ora.Obligation

/-- Digests are compiler-owned selectors, never semantic premises. A certificate
must additionally prove the body denoted by the selected row and manifest. -/
structure CallAdapterManifest where
  formulas : Manifest
  functions : List (String × FunctionSummaryRow)
  loops : List (String × SummaryRow)

def adapterKey (id : Nat) (digest : String) : String := s!"{id}:{digest}"

def adapterRequire (value : Option α) (reason : String) : Except String α :=
  match value with | some result => .ok result | none => .error reason

def CallAdapterManifest.wellFormed (manifest : CallAdapterManifest) : Bool :=
  manifest.formulas.wf && manifest.formulas.termsTopological &&
    decide (manifest.functions.map (·.2.id)).Nodup && decide (manifest.loops.map (·.2.id)).Nodup &&
    (manifest.functions.map Prod.fst ++ manifest.loops.map Prod.fst).all (fun digest =>
      digest.length == 64 && digest.toList.all ("0123456789abcdef".toList.contains ·) &&
        !digest.toList.all (· == '0'))

structure AdapterLoop where
  ready : Env → StorageActionState → Prop
  relation : Env → ControlTarget → StorageActionState → LoopState → Prop
  frame : ∀ (env : Env) (target : ControlTarget) (context : StorageActionState), CertifiedWriteFrame
    (fun before after => relation env target { context with loop := before } after)

structure CallAdapterDependencies where
  calls : List (String × CertifiedCall) := []
  loops : List (String × AdapterLoop) := []

abbrev AdapterChild := AdapterLoop × ControlTarget
abbrev AdapterFormula := Sum FormulaRef AdapterChild
abbrev AdapterEvent := EventOf (CallFormula AdapterFormula FormulaRef AdapterChild)
  StorageLocationExpr FormulaRef AdapterChild
abbrev AdapterBody := CallBody (CallFormula AdapterFormula FormulaRef AdapterChild)
  StorageLocationExpr FormulaRef AdapterChild

def adapterSemantics (dependencies : CallAdapterDependencies)
    (context : StorageFormulaContext) :
    TimelineSemantics AdapterFormula StorageLocationExpr FormulaRef AdapterChild := {
  storage := context.actionSemantics
  evalFormula := fun formula state => match formula with
    | .inl formula => context.evalFormula? formula state
    | .inr (child, _) => some (∃ env, context.stateEnv? state = some env ∧ child.ready env state)
  childRelation := fun (child, target) state after =>
    ∃ env, context.stateEnv? state = some env ∧ child.relation env target state after
  childFrame := fun (child, target) state => {
    footprint := fun location => ∃ env, (child.frame env target state).footprint location
    preservesOutside := by
      rintro before after ⟨env, _, related⟩ location outside
      exact (child.frame env target state).preservesOutside before after related location
        (fun written => outside ⟨env, written⟩) }
  calls := fun key => (dependencies.calls.find? (·.1 == key)).map Prod.snd
}

def CallAdapterManifest.event (manifest : CallAdapterManifest)
    (dependencies : CallAdapterDependencies) (row : TimelineEventRow) :
    Except String (List AdapterEvent) := do
  match row.payload with
  | .call invocation =>
      let (digest, callee) ← adapterRequire (manifest.functions.find? (·.2.id == invocation.calleeSummaryId))
        "call_adapter_callee_missing"
      if digest != invocation.calleeVcSha256 then throw "call_adapter_callee_digest_mismatch"
      if !callSignatureSupported callee invocation then throw "call_adapter_signature_mismatch"
      let key := adapterKey callee.id digest
      if !(dependencies.calls.any (·.1 == key)) then throw "call_adapter_certificate_missing"
      return checkedCallEvents {
        callee := key, arguments := invocation.arguments.map (·.value.formula)
        outcome := invocation.outcome, results := invocation.results.map (·.id) }
  | .summaryDependency dependency =>
      let (digest, child) ← adapterRequire (manifest.loops.find? (·.2.id == dependency.summaryId))
        "call_adapter_loop_missing"
      if digest != dependency.loopVcSha256 then throw "call_adapter_loop_digest_mismatch"
      if dependency.target == .backedge || !child.edges.any (·.target == dependency.target) then
        throw "call_adapter_loop_target"
      let target ← adapterRequire (denoteControlTarget? dependency.target) "call_adapter_loop_target"
      let bindings ← adapterRequire dependency.resultBindings "call_adapter_loop_bindings_missing"
      if !bindings.all (fun binding => (child.variables[binding.index]?).any
          (fun slot => slot.ty == binding.ty)) then throw "call_adapter_loop_binding_type"
      let (_, certificate) ← adapterRequire (dependencies.loops.find?
        (·.1 == adapterKey child.id digest)) "call_adapter_loop_certificate_missing"
      let selected := (certificate, target)
      return .check (.ordinary (.inr selected)) :: checkedLoopResultEvents selected
        (bindings.map fun binding => { id := binding.id, index := binding.index, ty := binding.ty })
  | .pathAssumption event => return event.formulas.map fun f => .assume (.ordinary (.inl f.formula))
  | .safetyCheckpoint event => return event.formulas.map fun f => .check (.ordinary (.inl f.formula))
  | .storageRead event =>
      let location ← adapterRequire (directStorageLocation? event.location) "call_adapter_storage_location"
      return [.storage (.read event.result location)]
  | .storageWrite event =>
      let location ← adapterRequire (directStorageLocation? event.location) "call_adapter_storage_location"
      return [.storage (.write location event.value.formula)]
  | .indexedRegionRead _ => throw "call_adapter_indexed_region_unsupported"

def CallAdapterManifest.timeline (manifest : CallAdapterManifest)
    (dependencies : CallAdapterDependencies) (rows : List TimelineEventRow) :
    Except String (List AdapterEvent) := List.flatten <$> rows.mapM (manifest.event dependencies)

def adapterReturn : CallReturnExprRow → CallReturnExpr FormulaRef
  | .direct outcome values => .direct outcome (values.map (·.value.formula))
  | .loopExit => .loopExit

def CallAdapterManifest.function (manifest : CallAdapterManifest)
    (dependencies : CallAdapterDependencies) (digest : String) (row : FunctionSummaryRow) :
    Except String AdapterBody := do
  if !manifest.wellFormed then throw "call_adapter_manifest_malformed"
  if dependencies.calls.any (·.1 == adapterKey row.id digest) then throw "call_adapter_duplicate_certificate"
  let (expectedDigest, expected) ← adapterRequire (manifest.functions.find? (·.2.id == row.id))
    "call_adapter_function_missing"
  if expectedDigest != digest then throw "call_adapter_function_digest_mismatch"
  if expected != row then throw "call_adapter_function_foreign"
  if row.owner.isEmpty || !row.projectionComplete || !row.writeFootprintComplete || !row.shapeSupported then
    throw "call_adapter_function_malformed"
  if (row.entryFormulas? manifest.formulas).isNone then throw "call_adapter_entry_scope_invalid"
  let paths : List (CallPath (CallFormula AdapterFormula FormulaRef AdapterChild)
      StorageLocationExpr FormulaRef AdapterChild) ← row.paths.mapM fun path => do
    return { events := ← manifest.timeline dependencies path.timeline
             returns := adapterReturn path.returns }
  return {
    signature := row.callSignature, paths, requires := row.entryFactsHold manifest.formulas
    semantics := fun _ => (adapterSemantics dependencies
      (directStorageContext manifest.formulas Env.empty [])).withCallChecks }

abbrev AdapterLoopVC := Env → StorageActionState → LoopVC LoopState

def adapterEntryFacts (manifest : Manifest) (owner : FunctionSummaryRow) (env : Env)
    (entry : StorageState) : Prop := ∃ arguments,
  owner.parameters.mapM (fun parameter => env.lookupFree parameter.id) = some arguments ∧
    owner.entryFactsHold manifest { arguments, storage := entry }

def adapterLoopEdge (manifest : Manifest) (dependencies : CallAdapterDependencies)
    (row : SummaryRow) (edge : EdgeRow) (events : List AdapterEvent) (target : ControlTarget)
    (base : Env) (caller : StorageActionState) : Edge LoopState := Id.run do
  let context := directStorageContext manifest base row.variables
  let semantics := (adapterSemantics dependencies context).withCallChecks
  let state := fun current => { caller with loop := current }
  let finish := fun result => do
    let locals ← directStorageFinalLocals? context row.variables edge.nextState result
    let returned ← match edge.returnValue with
      | none => some result.loop.returned
      | some expression => some <$> callEvaluateReturn? context.evalValue? (adapterReturn expression) result
    return { result.loop with locals, returned }
  {
    path := fun _ => True
    safe := fun current => TimelineSafe semantics events (state current) ∧
      ∀ result, TimelineExec semantics events (state current) result → (finish result.state).isSome = true
    next := fun current following => ∃ result,
      TimelineExec semantics events (state current) result ∧ finish result.state = some following
    target
    targetAssertion := fun following => directFormulasHold context caller.entryStorage following
      (directTargetFormulas row.post edge) }

/-- Lexical entry is the actual enclosing timeline prefix. Its operations have
already executed there; replaying the function preheader would duplicate calls. -/
def CallAdapterManifest.loop (manifest : CallAdapterManifest)
    (dependencies : CallAdapterDependencies) (digest : String) (row : SummaryRow) :
    Except String AdapterLoopVC := do
  if !manifest.wellFormed then throw "call_adapter_manifest_malformed"
  if dependencies.loops.any (·.1 == adapterKey row.id digest) then throw "call_adapter_duplicate_certificate"
  let (expectedDigest, expected) ← adapterRequire (manifest.loops.find? (·.2.id == row.id))
    "call_adapter_loop_missing"
  if expectedDigest != digest then throw "call_adapter_loop_digest_mismatch"
  if expected != row then throw "call_adapter_loop_foreign"
  let owner ← match manifest.functions.filter (·.2.owner == row.owner) with
    | [(_, owner)] => pure owner
    | [] => throw "call_adapter_loop_owner_missing"
    | _ => throw "call_adapter_loop_owner_ambiguous"
  if !owner.projectionComplete || !owner.shapeSupported || (owner.entryFormulas? manifest.formulas).isNone then
    throw "call_adapter_loop_owner_invalid"
  if !row.preheaderProjectionComplete || !row.baseShapeSupported || !headerFactsShapeSupported row.headerFacts then
    throw "call_adapter_loop_malformed"
  let edges ← row.edges.mapM fun edge => do
    if !(edgeTimelineReason? ((row.contextVariables ++ row.variables).map (·.id)) edge).isNone ||
        !bindingsMatchVariables row.variables edge.nextState then throw "call_adapter_edge_malformed"
    let events ← manifest.timeline dependencies edge.timeline
    let target ← adapterRequire (denoteControlTarget? edge.target) "call_adapter_loop_target"
    return adapterLoopEdge manifest.formulas dependencies row edge events target
  return fun base caller =>
    let initialContext := { directStorageContext manifest.formulas base [] with
      bindLocals := fun env _ => some env }
    let facts := contextReady base row.contextVariables ∧
      headerFactsHoldInEnv manifest.formulas base row.headerFacts ∧
      adapterEntryFacts manifest.formulas owner base caller.entryStorage ∧
      ∀ edge ∈ row.edges, (caller.reads.map (·.id) ++ timelineReadIds edge.timeline).Nodup
    {
      initial := fun initial => facts ∧ stateReady row.variables initial.locals ∧
        directInitialValuesAfterPreheader? initialContext caller row.init = some initial.locals ∧
        initial.storage = caller.loop.storage ∧ initial.returned = none
      invariant := fun current => facts ∧ stateReady row.variables current.locals ∧
        directFormulasHold (directStorageContext manifest.formulas base row.variables)
          caller.entryStorage current row.invariants
      edges := edges.map (fun edge => edge base caller) }

def adapterLoopRelation (vc : AdapterLoopVC) (env : Env) (target : ControlTarget)
    (caller : StorageActionState) (following : LoopState) : Prop :=
  ∃ initial, (vc env caller).ControlRelation target initial following

def adapterLoopCertificate (vc : AdapterLoopVC)
    (frames : ∀ (env : Env) (target : ControlTarget) (caller : StorageActionState), CertifiedWriteFrame
      (fun before after => adapterLoopRelation vc env target { caller with loop := before } after)) :
    AdapterLoop := {
  ready := fun env caller => ∃ initial, (vc env caller).initial initial
  relation := adapterLoopRelation vc
  frame := frames }

/- A dependency can only be installed after its exact row-derived body has a
kernel proof. Constructors refer to earlier entries, so mixed cycles cannot
be built. No digest string, declared frame, or Z3 verdict is proof evidence. -/
inductive CallAdapterBuilt (manifest : CallAdapterManifest) : CallAdapterDependencies → Prop where
  | empty : CallAdapterBuilt manifest {}
  | function {dependencies} (previous : CallAdapterBuilt manifest dependencies)
      (digest : String) (row : FunctionSummaryRow) (body : AdapterBody)
      (bound : manifest.function dependencies digest row = .ok body)
      (certificate : body.Certificate) :
      CallAdapterBuilt manifest { dependencies with calls :=
        (adapterKey row.id digest, body.certified certificate) :: dependencies.calls }
  | loop {dependencies} (previous : CallAdapterBuilt manifest dependencies)
      (digest : String) (row : SummaryRow) (vc : AdapterLoopVC)
      (bound : manifest.loop dependencies digest row = .ok vc)
      (certificate : ∀ env caller, (vc env caller).InductionObligations)
      (frames : ∀ (env : Env) (target : ControlTarget) (caller : StorageActionState), CertifiedWriteFrame
        (fun before after => adapterLoopRelation vc env target { caller with loop := before } after)) :
      CallAdapterBuilt manifest { dependencies with loops :=
        (adapterKey row.id digest, adapterLoopCertificate vc frames) :: dependencies.loops }

/-- Automatic assembly consumes kernel proof terms, not a success bit. This
type is the consumption boundary; the raw denotation helpers grant nothing. -/
structure CallAdapterCertificates (manifest : CallAdapterManifest) where
  dependencies : CallAdapterDependencies
  checked : CallAdapterBuilt manifest dependencies

def CallAdapterCertificates.empty (manifest : CallAdapterManifest) : CallAdapterCertificates manifest :=
  ⟨{}, .empty⟩

def CallAdapterCertificates.addFunction (previous : CallAdapterCertificates manifest)
    (digest : String) (row : FunctionSummaryRow) (body : AdapterBody)
    (bound : manifest.function previous.dependencies digest row = .ok body)
    (proof : body.Certificate) : CallAdapterCertificates manifest :=
  ⟨{ previous.dependencies with calls :=
      (adapterKey row.id digest, body.certified proof) :: previous.dependencies.calls },
    .function previous.checked digest row body bound proof⟩

def CallAdapterCertificates.addLoop (previous : CallAdapterCertificates manifest)
    (digest : String) (row : SummaryRow) (vc : AdapterLoopVC)
    (bound : manifest.loop previous.dependencies digest row = .ok vc)
    (proof : ∀ env caller, (vc env caller).InductionObligations)
    (frames : ∀ (env : Env) (target : ControlTarget) (caller : StorageActionState), CertifiedWriteFrame
      (fun before after => adapterLoopRelation vc env target { caller with loop := before } after)) :
    CallAdapterCertificates manifest :=
  ⟨{ previous.dependencies with loops :=
      (adapterKey row.id digest, adapterLoopCertificate vc frames) :: previous.dependencies.loops },
    .loop previous.checked digest row vc bound proof frames⟩

end Ora.Loop

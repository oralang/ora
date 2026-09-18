import Ora.Loop.CallAdapterTests

namespace Ora.Loop.CallAdapterLoopTests
open Ora.Obligation

def digest : String := String.mk (List.replicate 64 'c')
def ident : FreeVarId := { file_id := 90, pattern_id := 1 }
def zero : Ora.Integer.Value := Ora.Integer.Value.ofNat (.unsigned .w8) 0
def variableRow : VariableRow := { index := 0, id := ident, ty := .spelling "u8" }
def assignment : StateBindingRow := {
  variableIndex := 0, target := ident, sourceKey := "zero", value := .term 1 }
def back : EdgeRow := {
  id := 0, structuralKey := "back"
  timeline := [{ id := 0, structuralKey := "guard:false", payload := .pathAssumption { formulas := [{ formula := .term 2 }] } }]
  nextState := [assignment], target := .backedge }
def done : EdgeRow := {
  id := 1, structuralKey := "return"
  timeline := [{ id := 0, structuralKey := "guard:true", payload := .pathAssumption { formulas := [{ formula := .term 0 }] } }]
  nextState := [assignment], target := .functionReturn
  returnValue := some (.direct .success [{ value := { formula := .term 0 }, ty := .spelling "bool" }]) }
def row : SummaryRow := {
  id := 2, owner := "looper", kind := .scfWhile, variables := [variableRow]
  init := [.term 1], invariants := [.term 0], edges := [back, done]
  projectionComplete := true, preheaderProjectionComplete := true }
def functionDigest : String := String.mk (List.replicate 64 'd')
def dependency : SummaryDependencyRow := {
  id := 0, structuralKey := "loop:2", summaryId := 2, loopVcSha256 := digest
  target := .functionReturn, writeFootprintComplete := true, resultBindings := some [] }
def functionRow : FunctionSummaryRow := {
  id := 3, owner := "looper", structuralKey := "looper"
  outcomes := [{ outcome := .success, types := [.spelling "bool"] }]
  paths := [{
    id := 0, structuralKey := "loop:return", returns := .loopExit
    timeline := [{ id := 0, structuralKey := "loop:2", payload := .summaryDependency dependency }] }]
  projectionComplete := true, writeFootprintComplete := true }
def manifest : CallAdapterManifest := {
  formulas := { terms := [.boolLit true, .intLit { value := "0", ty := some (.spelling "u8") }, .boolLit false] }
  functions := [(functionDigest, functionRow)], loops := [(digest, row)] }
def events (id : Nat) : List AdapterEvent := [.assume (.ordinary (.inl (.term id)))]
def context (env : Env) : StorageFormulaContext := directStorageContext manifest.formulas env row.variables
def semantics (env : Env) := (adapterSemantics {} (context env)).withCallChecks
def facts (env : Env) (caller : StorageActionState) : Prop :=
  contextReady env [] ∧ headerFactsHoldInEnv manifest.formulas env [] ∧
    adapterEntryFacts manifest.formulas functionRow env caller.entryStorage ∧
    ∀ edge ∈ row.edges, (caller.reads.map (·.id) ++ timelineReadIds edge.timeline).Nodup
def vc : AdapterLoopVC := fun env caller => {
  initial := fun initial => facts env caller ∧
    stateReady row.variables initial.locals ∧
    directInitialValuesAfterPreheader? {
      directStorageContext manifest.formulas env [] with bindLocals := fun env _ => some env
    } caller row.init = some initial.locals ∧ initial.storage = caller.loop.storage ∧ initial.returned = none
  invariant := fun current => facts env caller ∧
    stateReady row.variables current.locals ∧ directFormulasHold (context env) caller.entryStorage current [.term 0]
  edges := [adapterLoopEdge manifest.formulas {} row back (events 2) .backedge env caller,
    adapterLoopEdge manifest.formulas {} row done (events 0) .functionReturn env caller] }

theorem bound : manifest.loop {} digest row = .ok vc := by rfl

theorem traceState (env : Env) (term : Nat) (initial : StorageActionState) (result : TimelineResult)
    (trace : TimelineExec (semantics env) (events term) initial result) : result.state = initial := by
  cases trace with
  | cons event rest => cases rest; cases event; rfl

theorem envBound (env : Env) (state : StorageActionState)
    (ready : stateReady row.variables state.loop.locals) :
    ∃ bound, (context env).stateEnv? state = some bound := by
  rcases bindState_some_of_stateReady env row.variables state.loop.locals ready with ⟨bound, hBound⟩
  refine ⟨(context env).withStoreViews (bindStorageReads bound state.reads) state, ?_⟩
  unfold StorageFormulaContext.stateEnv?
  change ((bindState? env row.variables state.loop.locals).bind _) = _
  rw [hBound]
  rfl

theorem formulasTrue (env : Env) (entry : StorageState) (state : LoopState)
    (ready : stateReady row.variables state.locals) :
    directFormulasHold (context env) entry state [.term 0] := by
  intro formula member
  simp only [List.mem_singleton] at member
  subst formula
  rcases envBound env { loop := state, entryStorage := entry } ready with ⟨bound, hBound⟩
  exact ⟨True, StorageFormulaContext.evalFormula_bool_true _ _ bound 0 hBound rfl, trivial⟩

theorem localValues (env : Env) (state : StorageActionState)
    (ready : stateReady row.variables state.loop.locals) :
    directStorageFinalLocals? (context env) row.variables [assignment] state = some [zero] := by
  rcases envBound env { state with reads := [] } ready with ⟨bound, hBound⟩
  have noReads : storageReadsForBindings [assignment] state.reads = [] := by
    cases state.reads <;> rfl
  unfold directStorageFinalLocals?
  rw [noReads]
  change ((context env).stateEnv? { state with reads := [] }).bind
    (fun env => applyStateBindingsInEnv? manifest.formulas row.variables env [assignment]) = _
  rw [hBound]
  rfl

theorem returnValue (env : Env) (state : StorageActionState)
    (ready : stateReady row.variables state.loop.locals) :
    callEvaluateReturn? (context env).evalValue? (.direct .success [.term 0]) state =
      some { outcome := .success, payload := [.bool true] } := by
  rcases envBound env state ready with ⟨bound, hBound⟩
  have eval : (context env).evalValue? (.term 0) state = some (.bool true) := by
    unfold StorageFormulaContext.evalValue?
    rw [hBound]
    rfl
  simp [callEvaluateReturn?, callEvaluateValues?, eval]

theorem edgeSafety (env : Env) (caller : StorageActionState) (edge : EdgeRow) (term : Nat)
    (which : edge = back ∨ edge = done) (state : LoopState)
    (ready : stateReady row.variables state.locals) :
    (adapterLoopEdge manifest.formulas {} row edge (events term)
      (directStorageTarget edge) env caller).safe state := by
  constructor
  · intro before formula after eventEquality
    cases before with
    | nil => simp [events] at eventEquality
    | cons head tail => simp [events] at eventEquality
  · intro result trace
    rw [traceState env term { caller with loop := state } result trace]
    have hValues := localValues env { caller with loop := state } ready
    have hReturn := returnValue env { caller with loop := state } ready
    dsimp [context] at hValues hReturn
    rcases which with rfl | rfl <;>
      dsimp only [back, done, adapterReturn, List.map] <;>
      rw [hValues]
    · rfl
    · simp only [Option.bind_some, hReturn]
      rfl

theorem nextShape (env : Env) (caller : StorageActionState) (edge : EdgeRow) (term : Nat)
    (which : edge = back ∨ edge = done) (current following : LoopState)
    (ready : stateReady row.variables current.locals)
    (next : (adapterLoopEdge manifest.formulas {} row edge (events term)
      (directStorageTarget edge) env caller).next current following) :
    following.locals = [zero] ∧ following.storage = current.storage := by
  rcases next with ⟨result, trace, finish⟩
  rw [traceState env term { caller with loop := current } result trace] at finish
  have values := localValues env { caller with loop := current } ready
  have returned := returnValue env { caller with loop := current } ready
  dsimp [context] at values returned
  rcases which with rfl | rfl <;>
    dsimp only [back, done, adapterReturn, List.map] at finish <;>
    rw [values] at finish
  · cases finish
    exact ⟨rfl, rfl⟩
  · simp only [Option.bind_some, returned, Option.map_some', Option.some.injEq] at finish
    cases finish
    exact ⟨rfl, rfl⟩

theorem proof (env : Env) (caller : StorageActionState) : (vc env caller).InductionObligations where
  base := fun _ initial => ⟨initial.1, initial.2.1, formulasTrue env _ _ initial.2.1⟩
  backedge := by
    intro edge current following member target invariant path safe next
    simp only [vc, List.mem_cons, List.not_mem_nil, or_false] at member
    rcases member with rfl | rfl
    · have shape := nextShape env caller back 2 (Or.inl rfl) current following invariant.2.1 next
      have ready : stateReady row.variables following.locals := by rw [shape.1]; rfl
      exact ⟨invariant.1, ready, formulasTrue env _ _ ready⟩
    · cases target
  edgeSafety := by
    intro edge current member invariant path
    simp only [vc, List.mem_cons, List.not_mem_nil, or_false] at member
    rcases member with rfl | rfl
    · exact edgeSafety env caller back 2 (Or.inl rfl) current invariant.2.1
    · exact edgeSafety env caller done 0 (Or.inr rfl) current invariant.2.1
  exit := by
    intro edge current following member target invariant path safe next
    simp only [vc, List.mem_cons, List.not_mem_nil, or_false] at member
    rcases member with rfl | rfl
    · cases target rfl
    · intro formula member
      cases member

theorem storagePreserved (env : Env) (caller : StorageActionState) (target : ControlTarget)
    (following : LoopState) (run : adapterLoopRelation vc env target caller following) :
    following.storage = caller.loop.storage := by
  rcases run with ⟨initial, notBackedge, initialized, current, edge, reached, member, targetEq, path, safe, next⟩
  have readyAt : ∀ state, Reaches (vc env caller) initial state → (vc env caller).invariant state :=
    fun _ reached => invariant_holds_on_reached_state (vc env caller) (proof env caller).base
      (proof env caller).backedge initialized reached
  have preserved : current.storage = initial.storage := by
    clear member targetEq path safe next
    induction reached with
    | refl => rfl
    | @next current following edge reached member target path safe next ih =>
        have ready := (readyAt current reached).2.1
        simp only [vc, List.mem_cons, List.not_mem_nil, or_false] at member
        rcases member with rfl | rfl
        · exact (nextShape env caller back 2 (Or.inl rfl) current following ready next).2.trans ih
        · cases target
  have ready := (readyAt current reached).2.1
  simp only [vc, List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl
  · exact False.elim (notBackedge targetEq.symm)
  · exact (nextShape env caller done 0 (Or.inr rfl) current following ready next).2.trans
      (preserved.trans initialized.2.2.2.1)

def frames (env : Env) (target : ControlTarget) (caller : StorageActionState) :
    CertifiedWriteFrame (fun before after => adapterLoopRelation vc env target { caller with loop := before } after) := {
  footprint := fun _ => False
  preservesOutside := fun before after run location _ =>
    congrFun (storagePreserved env { caller with loop := before } target after run) location }

def certificates : CallAdapterCertificates manifest :=
  (CallAdapterCertificates.empty manifest).addLoop digest row vc bound proof frames

theorem returnedTrue (env : Env) (caller : StorageActionState) (following : LoopState)
    (run : adapterLoopRelation vc env .functionReturn caller following) :
    following.returned = some { outcome := .success, payload := [.bool true] } := by
  rcases run with ⟨initial, _, initialized, current, edge, reached, member, targetEq, _, _, next⟩
  have ready := (invariant_holds_on_reached_state (vc env caller) (proof env caller).base
    (proof env caller).backedge initialized reached).2.1
  simp only [vc, List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl
  · cases targetEq
  · rcases next with ⟨result, trace, finish⟩
    rw [traceState env 0 { caller with loop := current } result trace] at finish
    have values := localValues env { caller with loop := current } ready
    have returned := returnValue env { caller with loop := current } ready
    dsimp [context] at values returned
    dsimp only [done, adapterReturn, List.map] at finish
    rw [values] at finish
    simp only [Option.bind_some, returned, Option.map_some', Option.some.injEq] at finish
    cases finish
    rfl

theorem initialized (env : Env) (caller : StorageActionState)
    (fresh : (caller.reads.map (·.id)).Nodup) : ∃ initial, (vc env caller).initial initial := by
  refine ⟨{ locals := [zero], storage := caller.loop.storage }, ?_, rfl, rfl, rfl, rfl⟩
  refine ⟨by simp [contextReady], by simp [headerFactsHoldInEnv], ?_, ?_⟩
  · refine ⟨[], rfl, ?_⟩
    change ∀ formula ∈ ([] : List FormulaRef), _
    simp
  · intro edge member
    simp only [row, List.mem_cons, List.not_mem_nil, or_false] at member
    rcases member with rfl | rfl <;> simpa [timelineReadIds, back, done] using fresh

def child : AdapterChild := (adapterLoopCertificate vc frames, .functionReturn)
def functionEvents : List AdapterEvent :=
  .check (.ordinary (.inr child)) :: checkedLoopResultEvents child []
def functionContext : StorageFormulaContext := directStorageContext manifest.formulas Env.empty []
def functionSemantics := adapterSemantics certificates.dependencies functionContext
def functionBody : AdapterBody := {
  signature := functionRow.callSignature, requires := functionRow.entryFactsHold manifest.formulas
  paths := [{ events := functionEvents, returns := .loopExit }]
  semantics := fun _ => functionSemantics.withCallChecks }

theorem functionBound : manifest.function certificates.dependencies functionDigest functionRow = .ok functionBody := by rfl

def functionEntry (input : CallInput) : StorageActionState := {
  loop := { locals := [], storage := input.storage }, entryStorage := input.storage }

theorem childReady (input : CallInput) :
    timelineFormulaHolds functionSemantics.withCallChecks (.ordinary (.inr child)) (functionEntry input) := by
  refine ⟨_, rfl, ?_⟩
  refine ⟨functionContext.withStoreViews Env.empty (functionEntry input), rfl, ?_⟩
  exact initialized _ _ (by simp [functionEntry])

theorem childResults (input : CallInput) : loopResultsReady functionSemantics child [] (functionEntry input) := by
  exact ⟨by simp [functionEntry], fun _ _ => rfl⟩

theorem functionSafe (input : CallInput) :
    TimelineSafe functionSemantics.withCallChecks functionEvents (functionEntry input) := by
  apply TimelineSafe.cons
  · intro formula equal
    cases equal
    exact childReady input
  · intro result step
    cases step
    apply TimelineSafe.cons
    · intro formula equal
      cases equal
      exact ⟨_, rfl, childResults input⟩
    · intro result step
      cases step
      apply TimelineSafe.cons
      · intro formula equal
        cases equal
      · intro result step before formula after equal
        simp at equal

theorem functionCertificate : functionBody.Certificate where
  safety := by
    intro input typed requires initial entry path member
    change functionRow.callSignature.inputTyped input = true at typed
    simp only [functionBody, CallSignature.entry?, typed, if_true, Option.some.injEq] at entry
    cases entry
    simp only [functionBody, List.mem_singleton] at member
    subst path
    exact functionSafe input
  calls := by
    intro input typed requires initial entry path member before invocation after equal
    simp only [functionBody, List.mem_singleton] at member
    subst path
    cases before with
    | nil => cases equal
    | cons a tail => cases tail with
      | nil => cases equal
      | cons b tail => cases tail with
        | nil => cases equal
        | cons c tail => simp [functionEvents, checkedLoopResultEvents] at equal
  loopResults := by
    intro input typed requires initial entry path member before dependency bindings after equal result trace
    change functionRow.callSignature.inputTyped input = true at typed
    simp only [functionBody, CallSignature.entry?, typed, if_true, Option.some.injEq] at entry
    cases entry
    simp only [functionBody, List.mem_singleton] at member
    subst path
    cases before with
    | nil => cases equal
    | cons a tail => cases tail with
      | nil => cases equal
      | cons b tail => cases tail with
        | cons c tail => simp [functionEvents, checkedLoopResultEvents] at equal
        | nil =>
          simp only [functionEvents, checkedLoopResultEvents, List.cons_append, List.nil_append,
            List.cons.injEq, TimelineEvent.childResult.injEq, List.nil_eq] at equal
          rcases equal with ⟨rfl, rfl, ⟨rfl, rfl⟩, rfl⟩
          cases trace with
          | cons first rest => cases first; cases rest with
            | cons second rest => cases second; cases rest
                                  exact childResults input
  returns := by
    intro input initial path result typed requires entry member trace
    simp only [functionBody, List.mem_singleton] at member
    subst path
    cases trace with
    | cons first rest => cases first; cases rest with
      | cons second rest => cases second; cases rest with
        | cons step rest => cases rest; cases step with
          | childResult related resumed =>
            rcases related with ⟨env, _, run⟩
            have returned := returnedTrue env initial _ run
            have effects := resumeLoop_effects _ _ _ _ resumed
            refine ⟨{ outcome := .success, payload := [.bool true] }, ?_, rfl⟩
            exact effects.2.2.2.1.trans returned

def complete : CallAdapterCertificates manifest :=
  certificates.addFunction functionDigest functionRow functionBody functionBound functionCertificate

theorem nonemptyLoop (env : Env) (caller : StorageActionState)
    (fresh : (caller.reads.map (·.id)).Nodup) :
    ∃ initial following, (vc env caller).ControlRelation .functionReturn initial following := by
  rcases initialized env caller fresh with ⟨initial, hInitial⟩
  let state : StorageActionState := { caller with loop := initial }
  rcases envBound env state hInitial.2.1 with ⟨bound, hBound⟩
  have eval := StorageFormulaContext.evalFormula_bool_true (context env) state bound 0 hBound rfl
  let result := ({ state } : TimelineResult).comp { state }
  have trace : TimelineExec (semantics env) (events 0) state result :=
    .cons (.assume eval trivial) (.nil _)
  refine ⟨initial, { locals := [zero], storage := initial.storage, returned := some { outcome := .success, payload := [.bool true] } }, by decide, hInitial,
    initial, adapterLoopEdge manifest.formulas {} row done (events 0) .functionReturn env caller,
    .refl _, by simp [vc], rfl, trivial,
    edgeSafety env caller done 0 (Or.inr rfl) initial hInitial.2.1, ?_⟩
  refine ⟨result, trace, ?_⟩
  have values := localValues env state hInitial.2.1
  have returned := returnValue env state hInitial.2.1
  dsimp [context] at values returned
  dsimp only [done, adapterReturn, List.map, result, TimelineResult.comp]
  rw [values]
  simp only [Option.bind_some, returned, Option.map_some']
  rfl

theorem actualExitIsCertified (env : Env) (caller : StorageActionState)
    (fresh : (caller.reads.map (·.id)).Nodup) :
    ∃ initial following, (vc env caller).CertifiedControlRelation .functionReturn initial following := by
  rcases nonemptyLoop env caller fresh with ⟨initial, following, run⟩
  exact ⟨initial, following, control_relation_contained_in_certified_relation _ (proof env caller) _ _ _ run⟩

def event (dep : SummaryDependencyRow) : TimelineEventRow := {
  id := 0, structuralKey := "loop:2", payload := .summaryDependency dep }

example : (manifest.event certificates.dependencies (event dependency)).map (fun _ => ()) = .ok () := by rfl
example : (manifest.event {} (event dependency)).map (fun _ => ()) =
    .error "call_adapter_loop_certificate_missing" := by rfl
example : (manifest.event certificates.dependencies (event { dependency with loopVcSha256 := functionDigest })).map (fun _ => ()) =
    .error "call_adapter_loop_digest_mismatch" := by rfl
example : (manifest.event certificates.dependencies (event { dependency with target := .errorReturn })).map (fun _ => ()) =
    .error "call_adapter_loop_target" := by rfl
example : (manifest.event certificates.dependencies (event { dependency with target := .backedge })).map (fun _ => ()) =
    .error "call_adapter_loop_target" := by rfl
example : (manifest.event certificates.dependencies (event { dependency with resultBindings := none })).map (fun _ => ()) =
    .error "call_adapter_loop_bindings_missing" := by rfl
example : (manifest.event certificates.dependencies (event { dependency with resultBindings := some [{ variableRow with index := 1 }] })).map (fun _ => ()) =
    .error "call_adapter_loop_binding_type" := by rfl
example : (manifest.event certificates.dependencies (event { dependency with resultBindings := some [{ variableRow with ty := .spelling "i8" }] })).map (fun _ => ()) =
    .error "call_adapter_loop_binding_type" := by rfl
example : (manifest.loop {} digest { row with owner := "foreign" }).map (fun _ => ()) =
    .error "call_adapter_loop_foreign" := by rfl
example : ({ manifest with functions := [] }.loop {} digest row).map (fun _ => ()) =
    .error "call_adapter_loop_owner_missing" := by rfl
example : ({ manifest with functions := [(functionDigest, functionRow), (digest, { functionRow with id := 99 })] }.loop {} digest row).map (fun _ => ()) =
    .error "call_adapter_loop_owner_ambiguous" := by rfl

#print axioms nonemptyLoop
#print axioms actualExitIsCertified

#print axioms functionCertificate
#print axioms complete

#print axioms proof
#print axioms storagePreserved
#print axioms certificates

#print axioms bound
#print axioms edgeSafety

end Ora.Loop.CallAdapterLoopTests

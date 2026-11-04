-- Supplied proofs over compiler-emitted rows, including the call inside the loop.

namespace G9cSuppliedLeaf
open Ora.Loop Ora.Obligation
open Ora.Generated.Obligations.Source_call_adapter_leaf_a1f737dc97a8081
abbrev row := emittedLoopSummary_1
attribute [local ora_loop_row] row emittedLoopSummary_1
attribute [local ora_loop_manifest] emitted_manifest_wf emitted_terms_topological
attribute [local ora_loop_data] emitted_terms_length
attribute [local ora_loop_data] emitted_term_0_lookup emitted_term_1_lookup emitted_term_2_lookup emitted_term_3_lookup emitted_term_4_lookup emitted_term_5_lookup emitted_term_6_lookup emitted_term_7_lookup emitted_term_8_lookup emitted_term_9_lookup emitted_term_10_lookup emitted_term_11_lookup emitted_term_12_lookup emitted_term_13_lookup emitted_term_14_lookup emitted_term_15_lookup emitted_term_16_lookup emitted_term_17_lookup emitted_term_18_lookup emitted_term_19_lookup emitted_term_20_lookup emitted_term_21_lookup emitted_term_22_lookup

def word (bits : BitVec 256) : Ora.Integer.Value := ⟨.unsigned .w256, bits⟩
def leafContext := directStorageContext emittedManifest Env.empty []
def leafBody : AdapterBody := {
  signature := emittedFunctionSummary_2.callSignature
  requires := emittedFunctionSummary_2.entryFactsHold emittedManifest
  paths := [{ events := [], returns := .direct .success [.term 22] }]
  semantics := fun _ => (adapterSemantics {} leafContext).withCallChecks }
theorem leafBound : emittedCallManifest.function {} emittedFunctionVcSha256_2 emittedFunctionSummary_2 = .ok leafBody := by rfl
theorem leafCertificate : leafBody.Certificate where
  safety := by
    intro input typed required initial entry path member before formula after events
    simp only [leafBody, List.mem_singleton] at member
    subst path
    simp at events
  calls := by
    intro input typed required initial entry path member before invocation after events
    simp only [leafBody, List.mem_singleton] at member
    subst path
    simp at events
  loopResults := by
    intro input typed required initial entry path member before child bindings after events
    simp only [leafBody, List.mem_singleton] at member
    subst path
    simp at events
  returns := by
    intro input initial path result typed required entry member trace
    simp only [leafBody, List.mem_singleton] at member
    subst path
    cases trace
    change emittedFunctionSummary_2.callSignature.inputTyped input = true at typed
    simp only [leafBody, CallSignature.entry?, typed, if_true, Option.some.injEq] at entry
    subst initial
    exact ⟨{ outcome := .success, payload := [.integer (word 1)] }, rfl, rfl⟩
def leafRegistry := (CallAdapterCertificates.empty emittedCallManifest).addFunction
  emittedFunctionVcSha256_2 emittedFunctionSummary_2 leafBody leafBound leafCertificate
theorem leafCustody : emittedCallCustody_2 := ⟨leafRegistry, rfl⟩

theorem leafJoint (bits : BitVec 256) (store : StorageState) (output : CallOutput)
    (related : leafBody.execution { arguments := [.integer (word bits)], storage := store } output) :
    output = { outcome := .success, payload := [.integer (word 1)], storage := store } := by
  rcases related with ⟨initial, path, result, returned, entry, member, trace, evaluated, rfl⟩
  have selected : leafBody.signature.entry? { arguments := [.integer (word bits)], storage := store } =
      some {
        loop := { locals := [], storage := store }
        entryStorage := store
        reads := [{ id := { file_id := 0, pattern_id := 0 }, value := .integer (word bits) }] } := rfl
  have equal := Option.some.inj (selected.symm.trans entry)
  subst initial
  simp only [leafBody, List.mem_singleton] at member
  subst path
  cases trace
  have payload : some { outcome := .success, payload := [.integer (word 1)] } = some returned := evaluated
  cases Option.some.inj payload
  rfl

def back : EdgeRow := row.edges[0]'(by decide)
def done : EdgeRow := row.edges[1]'(by decide)
def context (env : Env) := directStorageContext emittedManifest env row.variables
def invocation : CallInvocation FormulaRef := {
  callee := adapterKey 2 emittedFunctionVcSha256_2, arguments := [.term 14]
  outcome := .success, results := [{ file_id := 4294967277, pattern_id := 0 }] }
def backEvents : List AdapterEvent := [
  .assume (.ordinary (.inl (.term 13))), .check (.ready invocation), .call invocation ]
def doneEvents : List AdapterEvent := [.assume (.ordinary (.inl (.term 16)))]
def semantics (env : Env) := (adapterSemantics leafRegistry.dependencies (context env)).withCallChecks

theorem resolved (env : Env) : (semantics env).calls invocation.callee =
    some (leafBody.certified leafCertificate) := by rfl

theorem typedState (locals : ScalarLoopState) (ready : stateReady row.variables locals) :
    ∃ bits, locals = [word bits] := by
  ora_loop_unpack_ready ready
  exact ⟨_, rfl⟩

theorem typedContext (env : Env) (ready : contextReady env row.contextVariables) :
    ∃ bits, env.lookupFree { file_id := 0, pattern_id := 1 } = some (.integer (word bits)) := by
  have slot := ready (row.contextVariables[0]'(by decide)) (by simp [row, emittedLoopSummary_1])
  rcases slot with ⟨⟨shape, bits⟩, typed, lookup⟩
  change some (.unsigned .w256) = some shape at typed
  cases Option.some.inj typed
  exact ⟨bits, lookup⟩

theorem argumentValue (env : Env) (state : LoopState) (store : StorageState)
    (bits input : BitVec 256) (locals : state.locals = [word bits])
    (lookup : env.lookupFree { file_id := 0, pattern_id := 1 } = some (.integer (word input))) :
    (context env).evalValue? (.term 14) { loop := state, entryStorage := store } = some (.integer (word input)) := by
  cases state with | mk actual storage returned =>
    dsimp at locals
    subst actual
    change env.lookupFree { file_id := 0, pattern_id := 1 } = _
    exact lookup

theorem invocationReady (env : Env) (state : LoopState) (store : StorageState)
    (ready : contextReady env row.contextVariables) (typed : stateReady row.variables state.locals) :
    invocation.ready (semantics env).calls (semantics env).storage.evalValue
      { loop := state, entryStorage := store } := by
  rcases typedState _ typed with ⟨bits, locals⟩
  rcases typedContext env ready with ⟨input, lookup⟩
  refine ⟨leafBody.certified leafCertificate, [.integer (word input)], [.compilerTypeId 6],
    resolved env, ?_, rfl, ?_, ?_, rfl, rfl⟩
  · change callEvaluateValues? (context env).evalValue? [.term 14] _ = _
    simp only [callEvaluateValues?, List.mapM_cons, argumentValue env state store bits input locals lookup,
      List.mapM_nil, Option.bind_eq_bind, Option.some_bind, Option.pure_def]
  · intro formula member; cases member
  · change ([({ file_id := 4294967277, pattern_id := 0 } : FreeVarId)]).Nodup
    decide

def calledState (state : LoopState) (store : StorageState) : StorageActionState := {
  loop := state, entryStorage := store
  reads := [{ id := { file_id := 4294967277, pattern_id := 0 }, value := .integer (word 1) }] }

theorem callResult (env : Env) (state : LoopState) (store : StorageState)
    (ready : contextReady env row.contextVariables) (typed : stateReady row.variables state.locals)
    (result : TimelineResult) (step : TimelineEventExec (semantics env) (.call invocation)
      { loop := state, entryStorage := store } result) : result.state = calledState state store := by
  rcases typedState _ typed with ⟨bits, locals⟩
  rcases typedContext env ready with ⟨input, lookup⟩
  cases step with
  | call found arguments related outcome arity fresh =>
    rw [resolved env] at found
    cases Option.some.inj found
    have evaluated : callEvaluateValues? (semantics env).storage.evalValue invocation.arguments
        { loop := state, entryStorage := store } = some [.integer (word input)] := by
      change callEvaluateValues? (context env).evalValue? [.term 14] _ = _
      simp only [callEvaluateValues?, List.mapM_cons, argumentValue env state store bits input locals lookup,
        List.mapM_nil, Option.bind_eq_bind, Option.some_bind, Option.pure_def]
    have equal := Option.some.inj (evaluated.symm.trans arguments)
    subst_vars
    rw [leafJoint input state.storage _ related]
    cases state
    rfl

#print axioms leafCertificate
#print axioms leafJoint
#print axioms typedContext
#print axioms argumentValue
#print axioms invocationReady
#print axioms callResult
theorem emptySafe (sem : TimelineSemantics F L V C) (state : StorageActionState) :
    TimelineSafe sem [] state := by
  intro before formula after equal
  cases before <;> simp at equal

theorem checkSafe (sem : TimelineSemantics F L V C) (formula : F)
    (tail : List (EventOf F L V C)) (state : StorageActionState)
    (checked : timelineFormulaHolds sem formula state) (rest : TimelineSafe sem tail state) :
    TimelineSafe sem (.check formula :: tail) state := by
  apply TimelineSafe.cons
  · intro actual equal
    cases equal
    exact checked
  · intro result step
    cases step
    exact rest


theorem backTrace (env : Env) (state : LoopState) (store : StorageState)
    (ready : contextReady env row.contextVariables) (typed : stateReady row.variables state.locals)
    (result : TimelineResult) (trace : TimelineExec (semantics env) backEvents
      { loop := state, entryStorage := store } result) :
    result.state = calledState state store ∧ timelineFormulaHolds (semantics env)
      (.ordinary (.inl (.term 13))) { loop := state, entryStorage := store } := by
  cases trace with
  | cons first rest =>
    cases first with
    | assume evaluated holds =>
      cases rest with
      | cons check rest =>
        cases check
        cases rest with
        | cons step rest =>
          cases rest
          have same := callResult env state store ready typed _ step
          exact ⟨same, _, evaluated, holds⟩

theorem doneTrace (env : Env) (initial : StorageActionState) (result : TimelineResult)
    (trace : TimelineExec (semantics env) doneEvents initial result) : result.state = initial := by
  cases trace with
  | cons first rest => cases first; cases rest; rfl

def facts (env : Env) (caller : StorageActionState) : Prop :=
  contextReady env row.contextVariables ∧ headerFactsHoldInEnv emittedManifest env row.headerFacts ∧
    adapterEntryFacts emittedManifest emittedFunctionSummary_1 env caller.entryStorage ∧
    ∀ edge ∈ row.edges, (caller.reads.map (·.id) ++ timelineReadIds edge.timeline).Nodup

def vc : AdapterLoopVC := fun env caller =>
  let captured := bindStorageReads env caller.reads
  { initial := fun initial => facts captured caller ∧ stateReady row.variables initial.locals ∧
      directInitialValuesAfterPreheader? { directStorageContext emittedManifest captured [] with
        bindLocals := fun env _ => some env } caller row.init = some initial.locals ∧
      initial.storage = caller.loop.storage ∧ initial.returned = none
    invariant := fun current => facts captured caller ∧ stateReady row.variables current.locals ∧
      directFormulasHold (context captured) caller.entryStorage current row.invariants
    edges := [
      adapterLoopEdge emittedManifest leafRegistry.dependencies row back backEvents .backedge captured { caller with reads := [] },
      adapterLoopEdge emittedManifest leafRegistry.dependencies row done doneEvents .successor captured { caller with reads := [] }] }
theorem bound : emittedCallManifest.loop leafRegistry.dependencies emittedLoopVcSha256_1 row = .ok vc := by rfl

theorem backValues (env : Env) (entry : StorageState) (current : LoopState) (bits : BitVec 256)
    (locals : current.locals = [word bits]) :
    directStorageFinalLocals? (context env) row.variables back.nextState
      (calledState current entry) = some [word (bits + 1)] := by
  cases current with | mk actual storage payload => dsimp at locals; subst actual; rfl
theorem doneValues (env : Env) (entry : StorageState) (current : LoopState) (bits : BitVec 256)
    (locals : current.locals = [word bits]) :
    directStorageFinalLocals? (context env) row.variables done.nextState
      { loop := current, entryStorage := entry } = some [word bits] := by
  cases current with | mk actual storage payload => dsimp at locals; subst actual; rfl

theorem invariantCompatible (env : Env) :
    directFormulaControlCompatible emittedManifest env row.variables (.term 6) := by
  ora_loop_direct_formula_control_in row env

theorem invariantAt (env : Env) (entry : StorageState) (current : LoopState)
    (bits : BitVec 256) (locals : current.locals = [word bits]) :
    directFormulasHold (context env) entry current [.term 6] ↔ bits.toNat ≤ 1 := by
  change (∀ formula ∈ [FormulaRef.term 6], _) ↔ _
  simp only [List.forall_mem_cons, List.forall_mem_nil, and_true]
  rw [show (∃ proposition, (context env).evalFormula? (.term 6)
      { loop := current, entryStorage := entry } = some proposition ∧ proposition) ↔
      formulaHoldsInState emittedManifest env row.variables current.locals (.term 6)
    from invariantCompatible env current entry]
  rw [locals]
  ora_loop_formula_context_reduce
  ora_loop_value_reduce
  simp only [word, if_true, lookupFreeBinding, beq_self_eq_true, ora_loop_data, Option.some_bind]
  ora_loop_normalize
  simp only [List.not_mem_nil, false_implies, implies_true, and_true]
  rfl

theorem guardCompatible (env : Env) :
    directFormulaControlCompatible emittedManifest env row.variables (.term 13) := by
  ora_loop_direct_formula_control_in row env

theorem guardAt (env : Env) (caller : StorageActionState) (current : LoopState)
    (bits : BitVec 256) (locals : current.locals = [word bits])
    (holds : timelineFormulaHolds (semantics env) (.ordinary (.inl (.term 13)))
      { loop := current, entryStorage := caller.entryStorage }) : bits.toNat < 1 := by
  have scalar := (guardCompatible env current caller.entryStorage).mp holds
  rw [locals] at scalar
  ora_loop_formula_context_reduce at scalar
  ora_loop_value_reduce at scalar
  simp only [word, if_true, lookupFreeBinding, beq_self_eq_true, ora_loop_data, Option.some_bind] at scalar
  ora_loop_normalize at scalar
  exact scalar


theorem backNext (env : Env) (caller : StorageActionState) (current following : LoopState)
    (bits : BitVec 256) (locals : current.locals = [word bits])
    (ready : contextReady env row.contextVariables)
    (next : (adapterLoopEdge emittedManifest leafRegistry.dependencies row back backEvents .backedge
      env { caller with reads := [] }).next current following) :
    following = { current with locals := [word (bits + 1)] } ∧ bits.toNat < 1 := by
  rcases next with ⟨result, trace, finish⟩
  have values := backValues env caller.entryStorage current bits locals
  have completed := backTrace env current caller.entryStorage ready (by rw [locals]; rfl) result trace
  rw [completed.1] at finish
  dsimp [context] at values
  change ((directStorageFinalLocals? _ _ _ _).bind _) = some following at finish
  rw [values] at finish
  exact ⟨(Option.some.inj finish).symm, guardAt env caller current bits locals completed.2⟩

theorem doneNext (env : Env) (caller : StorageActionState) (current following : LoopState)
    (bits : BitVec 256) (locals : current.locals = [word bits])
    (next : (adapterLoopEdge emittedManifest leafRegistry.dependencies row done doneEvents .successor
      env { caller with reads := [] }).next current following) : following = current := by
  rcases next with ⟨result, trace, finish⟩
  rw [doneTrace env { caller with loop := current, reads := [] } result trace] at finish
  have values := doneValues env caller.entryStorage current bits locals
  dsimp [context] at values
  change ((directStorageFinalLocals? _ _ _ _).bind _) = some following at finish
  rw [values] at finish
  have equal := (Option.some.inj finish).symm
  simpa only [← locals] using equal

theorem backSafe (env : Env) (caller : StorageActionState) (current : LoopState)
    (ready : contextReady env row.contextVariables) (typed : stateReady row.variables current.locals) :
    (adapterLoopEdge emittedManifest leafRegistry.dependencies row back backEvents .backedge
      env { caller with reads := [] }).safe current := by
  constructor
  · apply TimelineSafe.cons
    · intro formula equal; cases equal
    · intro result step
      cases step
      refine checkSafe _ _ _ _ ⟨_, rfl, invocationReady env current caller.entryStorage ready typed⟩ ?_
      apply TimelineSafe.cons
      · intro formula equal; cases equal
      · intro result step; exact emptySafe _ _
  · intro result trace
    rw [(backTrace env current caller.entryStorage ready typed result trace).1]
    rcases typedState _ typed with ⟨bits, locals⟩
    have values := backValues env caller.entryStorage current bits locals
    dsimp [context] at values
    change ((directStorageFinalLocals? _ _ _ _).bind _).isSome = true
    rw [values]
    rfl

theorem doneSafe (env : Env) (caller : StorageActionState) (current : LoopState)
    (typed : stateReady row.variables current.locals) :
    (adapterLoopEdge emittedManifest leafRegistry.dependencies row done doneEvents .successor
      env { caller with reads := [] }).safe current := by
  constructor
  · intro before formula after equal
    have member : TimelineEvent.check formula ∈ before ++ .check formula :: after := by simp
    rw [← equal] at member
    simp [doneEvents] at member
  · intro result trace
    rw [doneTrace env { caller with loop := current, reads := [] } result trace]
    rcases typedState _ typed with ⟨bits, locals⟩
    have values := doneValues env caller.entryStorage current bits locals
    dsimp [context] at values
    change ((directStorageFinalLocals? _ _ _ _).bind _).isSome = true
    rw [values]
    rfl
theorem stepBound (bits : BitVec 256) (less : bits.toNat < 1) : (bits + 1).toNat ≤ 1 := by
  rw [BitVec.toNat_add]
  change (bits.toNat + 1) % 2 ^ 256 ≤ 1
  rw [Nat.mod_eq_of_lt (by omega)]
  omega


theorem proof (env : Env) (caller : StorageActionState) : (vc env caller).InductionObligations where
  base := by
    intro current initial
    have locals : current.locals = [word 0] := by
      have initialized := initial.2.2.1
      change some [word 0] = some current.locals at initialized
      exact (Option.some.inj initialized).symm
    refine ⟨initial.1, initial.2.1, ?_⟩
    exact (invariantAt _ _ _ 0 locals).mpr (by decide)
  backedge := by
    intro edge current following member target invariant path safe next
    simp only [vc, List.mem_cons, List.not_mem_nil, or_false] at member
    rcases member with rfl | rfl
    · rcases typedState _ invariant.2.1 with ⟨bits, locals⟩
      have shape := backNext _ caller current following bits locals invariant.1.1 next
      refine ⟨invariant.1, ?_, ?_⟩
      · rw [shape.1]; rfl
      · exact (invariantAt _ _ _ (bits + 1) (by rw [shape.1])).mpr (stepBound bits shape.2)
    · cases target
  edgeSafety := by
    intro edge current member invariant path
    simp only [vc, List.mem_cons, List.not_mem_nil, or_false] at member
    rcases member with rfl | rfl
    · exact backSafe _ caller current invariant.1.1 invariant.2.1
    · exact doneSafe _ caller current invariant.2.1
  exit := by
    intro edge current following member target invariant path safe next
    simp only [vc, List.mem_cons, List.not_mem_nil, or_false] at member
    rcases member with rfl | rfl
    · exact (target rfl).elim
    · intro formula member; cases member

#print axioms bound
#print axioms backTrace
#print axioms backValues
#print axioms invariantAt
#print axioms backNext
#print axioms backSafe
#print axioms proof

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
      have inv := readyAt current reached
      rcases typedState _ inv.2.1 with ⟨bits, locals⟩
      simp only [vc, List.mem_cons, List.not_mem_nil, or_false] at member
      rcases member with rfl | rfl
      · rw [(backNext _ caller current following bits locals inv.1.1 next).1]
        exact ih
      · cases target
  rcases typedState _ (readyAt current reached).2.1 with ⟨bits, locals⟩
  simp only [vc, List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl
  · exact False.elim (notBackedge targetEq.symm)
  · rw [doneNext _ caller current following bits locals next]
    exact preserved.trans initialized.2.2.2.1

def frames (env : Env) (target : ControlTarget) (caller : StorageActionState) :
    CertifiedWriteFrame (fun before after => adapterLoopRelation vc env target { caller with loop := before } after) := {
  footprint := fun _ => False
  preservesOutside := fun before after run location _ =>
    congrFun (storagePreserved env { caller with loop := before } target after run) location }

def supplied : AdapterLoopProof emittedCallManifest emittedLoopVcSha256_1 row where
  previous := leafRegistry
  vc := vc
  bound := bound
  induction := proof
  frames := frames
  witness := by
    let env : Env := Env.empty.setFree { file_id := 0, pattern_id := 1 } (.integer (word 0))
    let store : StorageState := fun _ => .bool false
    let caller : StorageActionState := { loop := { locals := [], storage := store }, entryStorage := store }
    refine ⟨env, caller, { locals := [word 0], storage := store }, ?_, rfl, rfl, rfl, rfl⟩
    constructor
    · intro slot member
      simp only [row, emittedLoopSummary_1, List.mem_singleton] at member
      subst slot
      exact ⟨word 0, rfl, rfl⟩
    constructor
    · intro fact member; cases member
    constructor
    · refine ⟨[.integer (word 0)], rfl, ?_⟩
      intro formula member; cases member
    · intro edge member
      simp only [row, emittedLoopSummary_1, List.mem_cons, List.not_mem_nil, or_false] at member
      rcases member with rfl | rfl <;> decide

theorem loopAuthorization : emittedCallLoop_1 := ⟨supplied⟩

theorem exitLocals (env : Env) (caller : StorageActionState) (target : ControlTarget) (following : LoopState)
    (run : adapterLoopRelation vc env target caller following) : ∃ bits, following.locals = [word bits] := by
  rcases run with ⟨initial, notBackedge, initialized, current, edge, reached, member, targetEq, path, safe, next⟩
  have invariant := invariant_holds_on_reached_state (vc env caller) (proof env caller).base
    (proof env caller).backedge initialized reached
  rcases typedState _ invariant.2.1 with ⟨bits, locals⟩
  simp only [vc, List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl
  · exact False.elim (notBackedge targetEq.symm)
  · exact ⟨bits, (congrArg LoopState.locals (doneNext _ caller current following bits locals next)).trans locals⟩

#print axioms frames
#print axioms loopAuthorization
#print axioms exitLocals
def certificates := supplied.install
def child (target : ControlTarget) : AdapterChild := (adapterLoopCertificate vc frames, target)
def bindings : List LoopResultBinding := [{ id := { file_id := 4294967284, pattern_id := 0 }, index := 0, ty := .compilerTypeId 6 }]
def functionContext := directStorageContext emittedManifest Env.empty []
def functionSemantics := adapterSemantics certificates.dependencies functionContext
def entryEvents (target : ControlTarget) (results : List LoopResultBinding) : List AdapterEvent := [
  .check (.ordinary (.inr (child target))), .check (.loopResultsReady (child target) results),
  .childResult (child target) results ]
def functionEvents : List AdapterEvent := entryEvents .successor bindings
def functionBody : AdapterBody := {
  signature := emittedFunctionSummary_1.callSignature
  requires := emittedFunctionSummary_1.entryFactsHold emittedManifest
  paths := [{ events := functionEvents, returns := .direct .success [.term 21] }]
  semantics := fun _ => functionSemantics.withCallChecks }
theorem functionBound : emittedCallManifest.function certificates.dependencies
    emittedFunctionVcSha256_1 emittedFunctionSummary_1 = .ok functionBody := by rfl

def functionEntry (bits : BitVec 256) (store : StorageState) : StorageActionState := {
  loop := { locals := [], storage := store }, entryStorage := store
  reads := [{ id := { file_id := 0, pattern_id := 1 }, value := .integer (word bits) }] }
def functionEnv (bits : BitVec 256) (store : StorageState) :=
  functionContext.withStoreViews (bindStorageReads Env.empty (functionEntry bits store).reads)
    (functionEntry bits store)

theorem childReady (bits : BitVec 256) (store : StorageState) (target : ControlTarget) :
    timelineFormulaHolds functionSemantics.withCallChecks (.ordinary (.inr (child target))) (functionEntry bits store) := by
  refine ⟨_, rfl, functionEnv bits store, rfl, ?_⟩
  refine ⟨{ locals := [word 0], storage := store }, ?_, rfl, rfl, rfl, rfl⟩
  constructor
  · intro slot member
    simp only [row, emittedLoopSummary_1, List.mem_singleton] at member
    subst slot
    exact ⟨word bits, rfl, rfl⟩
  constructor
  · intro fact member
    cases member
  constructor
  · refine ⟨[.integer (word bits)], rfl, ?_⟩
    intro formula member
    cases member
  · intro edge member
    simp only [row, emittedLoopSummary_1, List.mem_cons, List.not_mem_nil, or_false] at member
    rcases member with rfl | rfl
    · change ([({ file_id := 0, pattern_id := 1 } : FreeVarId), { file_id := 4294967277, pattern_id := 0 }]).Nodup
      decide
    · change ([({ file_id := 0, pattern_id := 1 } : FreeVarId)]).Nodup
      decide

theorem childValues (bits : BitVec 256) (store : StorageState) (target : ControlTarget) (following : LoopState)
    (related : functionSemantics.childRelation (child target) (functionEntry bits store) following) :
    ∃ result, following.locals = [word result] := by
  rcases related with ⟨env, _, run⟩
  exact exitLocals env (functionEntry bits store) target following run

theorem childResults (bits : BitVec 256) (store : StorageState) (target : ControlTarget) :
    loopResultsReady functionSemantics (child target) bindings (functionEntry bits store) := by
  refine ⟨?_, ?_⟩
  · change ([({ file_id := 0, pattern_id := 1 } : FreeVarId), { file_id := 4294967284, pattern_id := 0 }]).Nodup
    decide
  intro following related
  rcases childValues bits store target following related with ⟨result, locals⟩
  simp [loopResultBindings?, bindings, locals, word, storageValueMatchesTy]

theorem emptyResults (bits : BitVec 256) (store : StorageState) (target : ControlTarget) :
    loopResultsReady functionSemantics (child target) [] (functionEntry bits store) := by
  refine ⟨?_, fun _ _ => rfl⟩
  change ([({ file_id := 0, pattern_id := 1 } : FreeVarId)]).Nodup
  decide

theorem typedInput (input : CallInput) (typed : functionBody.signature.inputTyped input = true) :
    ∃ bits, input.arguments = [.integer (word bits)] := by
  have arity := typed
  change (decide ([({ file_id := 0, pattern_id := 1 } : FreeVarId)]).Nodup &&
    callValuesTyped [.compilerTypeId 6] input.arguments) = true at arity
  simp only [callValuesTyped, Bool.and_eq_true, beq_iff_eq, List.length] at arity
  have length : input.arguments.length = 1 := arity.2.1.symm
  cases arguments : input.arguments with
  | nil => simp [arguments] at length
  | cons value tail =>
    have tailLength : tail.length + 1 = 1 := by
      simpa only [arguments, List.length_cons] using length
    have empty : tail = [] := List.length_eq_zero.mp (Nat.succ.inj tailLength)
    subst tail
    have typedValue : storageValueMatchesTy value (.compilerTypeId 6) = true := by
      simpa [arguments] using arity.2.2
    cases value <;> try contradiction
    rename_i integer
    rcases integer with ⟨shape, bits⟩
    change (some (.unsigned .w256) == some shape) = true at typedValue
    have equal : some (.unsigned .w256) = some shape := by simpa only [beq_iff_eq] using typedValue
    cases Option.some.inj equal
    exact ⟨bits, rfl⟩

theorem childResultSafe (sem : TimelineSemantics F L V C) (child : C)
    (bindings : List LoopResultBinding) (tail : List (EventOf F L V C))
    (state : StorageActionState)
    (rest : ∀ following resumed, sem.childRelation child state following →
      resumeLoop? bindings state following = some resumed → TimelineSafe sem tail resumed) :
    TimelineSafe sem (.childResult child bindings :: tail) state := by
  apply TimelineSafe.cons
  · intro formula equal
    cases equal
  · intro result step
    cases step with
    | childResult related resume => exact rest _ _ related resume

theorem checkTrace (sem : TimelineSemantics F L V C) (formula : F)
    (tail : List (EventOf F L V C)) (initial : StorageActionState) (result : TimelineResult)
    (trace : TimelineExec sem (.check formula :: tail) initial result) :
    ∃ following, TimelineExec sem tail initial following ∧ result.state = following.state := by
  cases trace with
  | cons first rest => cases first; exact ⟨_, rest, rfl⟩

#print axioms exitLocals
#print axioms functionBound
#print axioms childReady
#print axioms childResults
#print axioms typedInput


theorem entrySafe (bits : BitVec 256) (store : StorageState) (target : ControlTarget)
    (results : List LoopResultBinding) (tail : List AdapterEvent)
    (ready : loopResultsReady functionSemantics (child target) results (functionEntry bits store))
    (safe : ∀ state, TimelineSafe functionSemantics.withCallChecks tail state) :
    TimelineSafe functionSemantics.withCallChecks (entryEvents target results ++ tail) (functionEntry bits store) := by
  simp only [entryEvents, List.cons_append, List.nil_append]
  refine checkSafe _ _ _ _ (childReady bits store target) ?_
  refine checkSafe _ _ _ _ ⟨_, rfl, ready⟩ ?_
  exact childResultSafe _ _ _ _ _ (fun _ resumed _ _ => safe resumed)


theorem functionSafe (bits : BitVec 256) (store : StorageState) :
    TimelineSafe functionSemantics.withCallChecks functionEvents (functionEntry bits store) := by
  simpa only [List.append_nil] using entrySafe bits store .successor bindings []
    (childResults bits store .successor) (emptySafe _)
theorem loopPrefixResults (sem : TimelineSemantics AdapterFormula StorageLocationExpr FormulaRef AdapterChild)
    (child : AdapterChild) (bindings : List LoopResultBinding) (tail : List AdapterEvent)
    (initial : StorageActionState) (ready : loopResultsReady sem child bindings initial)
    (noChild : ∀ dependency results, TimelineEvent.childResult dependency results ∉ tail)
    (before : List AdapterEvent) (dependency : AdapterChild) (results : List LoopResultBinding)
    (after : List AdapterEvent)
    (equal : [.check (.ordinary (.inr child)), .check (.loopResultsReady child bindings),
      .childResult child bindings] ++ tail = before ++ .childResult dependency results :: after)
    (result : TimelineResult) (trace : TimelineExec sem.withCallChecks before initial result) :
    loopResultsReady sem.withCallChecks dependency results result.state := by
  cases before with
  | nil => cases equal
  | cons a rest => cases rest with
    | nil => cases equal
    | cons b rest => cases rest with
      | nil =>
        simp only [List.cons_append, List.nil_append, List.cons.injEq, TimelineEvent.childResult.injEq] at equal
        rcases equal with ⟨rfl, rfl, ⟨rfl, rfl⟩, rfl⟩
        rcases checkTrace _ _ _ _ _ trace with ⟨r1, t1, e1⟩
        rcases checkTrace _ _ _ _ _ t1 with ⟨r2, t2, e2⟩
        cases t2
        rw [e1, e2]
        exact ready
      | cons c rest =>
        simp only [List.cons_append, List.nil_append, List.cons.injEq] at equal
        rcases equal with ⟨_, _, _, later⟩
        have present : TimelineEvent.childResult dependency results ∈ rest ++ .childResult dependency results :: after := by simp
        rw [← later] at present
        exact (noChild dependency results present).elim

def resumedState (bits value : BitVec 256) (store : StorageState) (following : LoopState) : StorageActionState := {
  loop := { locals := [], storage := following.storage, returned := following.returned }
  entryStorage := store
  reads := [{ id := { file_id := 0, pattern_id := 1 }, value := .integer (word bits) },
    { id := { file_id := 4294967284, pattern_id := 0 }, value := .integer (word value) }] }

theorem resumedValues (bits : BitVec 256) (store : StorageState) (following : LoopState)
    (resumed : StorageActionState)
    (related : functionSemantics.childRelation (child .successor) (functionEntry bits store) following)
    (resume : resumeLoop? bindings (functionEntry bits store) following = some resumed) :
    ∃ value, resumed = resumedState bits value store following := by
  rcases childValues bits store .successor following related with ⟨value, locals⟩
  refine ⟨value, ?_⟩
  have evaluated : resumeLoop? bindings (functionEntry bits store) following =
      some (resumedState bits value store following) := by
    simp [resumeLoop?, loopResultBindings?, bindings, functionEntry, locals, word,
      storageValueMatchesTy, resumedState]
  exact Option.some.inj (resume.symm.trans evaluated)


theorem completed (bits : BitVec 256) (store : StorageState) (result : TimelineResult)
    (trace : TimelineExec functionSemantics.withCallChecks functionEvents (functionEntry bits store) result) :
    ∃ value following, result.state = resumedState bits value store following := by
  rcases checkTrace _ _ _ _ _ trace with ⟨r1, t1, e1⟩
  rcases checkTrace _ _ _ _ _ t1 with ⟨r2, t2, e2⟩
  cases t2 with
  | cons step rest =>
    cases rest
    cases step with
    | childResult related resume =>
      rcases resumedValues bits store _ _ related resume with ⟨value, equal⟩
      exact ⟨value, _, e1.trans (e2.trans equal)⟩
theorem returnedValue (bits value : BitVec 256) (store : StorageState) (following : LoopState) :
    functionContext.evalValue? (.term 21) (resumedState bits value store following) =
      some (.integer (word value)) := by
  let state := resumedState bits value store following
  let env := functionContext.withStoreViews (bindStorageReads Env.empty state.reads) state
  apply StorageFormulaContext.evalValue_free_variable functionContext state env 21
    { id := { file_id := 4294967284, pattern_id := 0 }, name := "i", ty := some (.compilerTypeId 6) }
  · rfl
  · exact emitted_term_21_lookup
  · simp only [env, StorageFormulaContext.withStoreViews_lookupFree, state, resumedState,
      bindStorageReads, List.foldl, Env.setFree, Env.lookupFree, lookupFreeBinding,
      beq_self_eq_true, if_true]

theorem entryTyped (input : CallInput) (bits : BitVec 256)
    (typed : functionBody.signature.inputTyped input = true)
    (arguments : input.arguments = [.integer (word bits)]) :
    functionBody.signature.entry? input = some (functionEntry bits input.storage) := by
  simp only [CallSignature.entry?, typed, if_true]
  rw [arguments]
  rfl


theorem functionCertificate : functionBody.Certificate where
  safety := by
    intro input typed required initial entry path member
    rcases typedInput input typed with ⟨bits, arguments⟩
    have equal := Option.some.inj ((entryTyped input bits typed arguments).symm.trans entry)
    subst initial
    simp only [functionBody, List.mem_singleton] at member
    subst path
    exact functionSafe bits input.storage
  calls := by
    intro input typed required initial entry path member before invocation after equal
    have contains : TimelineEvent.call invocation ∈ before ++ .call invocation :: after := by simp
    rw [← equal] at contains
    simp only [functionBody, List.mem_singleton] at member
    subst path
    simp [functionEvents, entryEvents] at contains
  loopResults := by
    intro input typed required initial entry path member before dependency results after equal result trace
    rcases typedInput input typed with ⟨bits, arguments⟩
    have same := Option.some.inj ((entryTyped input bits typed arguments).symm.trans entry)
    subst initial
    simp only [functionBody, List.mem_singleton] at member
    subst path
    exact loopPrefixResults functionSemantics (child .successor) bindings [] _
      (childResults bits input.storage .successor)
      (by intro dependency results; simp) _ _ _ _ equal _ trace
  returns := by
    intro input initial path result typed required entry member trace
    rcases typedInput input typed with ⟨bits, arguments⟩
    have equal := Option.some.inj ((entryTyped input bits typed arguments).symm.trans entry)
    subst initial
    simp only [functionBody, List.mem_singleton] at member
    subst path
    rcases completed bits input.storage result trace with ⟨value, following, equal⟩
    rw [equal]
    refine ⟨{ outcome := .success, payload := [.integer (word value)] }, ?_, rfl⟩
    change callEvaluateReturn? functionContext.evalValue? (.direct .success [.term 21])
      (resumedState bits value input.storage following) = _
    simp only [callEvaluateReturn?, callEvaluateValues?, List.mapM_cons, returnedValue,
      List.mapM_nil, Option.bind_eq_bind, Option.some_bind, Option.pure_def, Option.map_some']

def installed := certificates.addFunction emittedFunctionVcSha256_1 emittedFunctionSummary_1
  functionBody functionBound functionCertificate
theorem custody : emittedCallCustody_1 := ⟨installed, rfl⟩
#print axioms functionBound
#print axioms childReady
#print axioms functionCertificate
#print axioms custody
end G9cSuppliedLeaf

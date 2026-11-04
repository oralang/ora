-- Supplied proofs over the compiler-emitted rows; no expected manifest is copied here.
-- Check through call_projection_loop_return.proofs.json.

namespace G9cSuppliedReturn
open Ora.Loop Ora.Obligation
open Ora.Generated.Obligations.Source_call_projection_loop_return_6670e546cbd63471
abbrev row := emittedLoopSummary_1
attribute [local ora_loop_row] row emittedLoopSummary_1
attribute [local ora_loop_manifest] emitted_manifest_wf emitted_terms_topological
attribute [local ora_loop_data] emitted_terms_length
attribute [local ora_loop_data] emitted_term_0_lookup emitted_term_1_lookup emitted_term_2_lookup emitted_term_3_lookup emitted_term_4_lookup emitted_term_5_lookup emitted_term_6_lookup emitted_term_7_lookup emitted_term_8_lookup emitted_term_9_lookup emitted_term_10_lookup emitted_term_11_lookup emitted_term_12_lookup emitted_term_13_lookup emitted_term_14_lookup emitted_term_15_lookup emitted_term_16_lookup emitted_term_17_lookup emitted_term_18_lookup emitted_term_19_lookup emitted_term_20_lookup emitted_term_21_lookup emitted_term_22_lookup emitted_term_23_lookup emitted_term_24_lookup emitted_term_25_lookup emitted_term_26_lookup emitted_term_27_lookup emitted_term_28_lookup emitted_term_29_lookup emitted_term_30_lookup emitted_term_31_lookup emitted_term_32_lookup emitted_term_33_lookup emitted_term_34_lookup emitted_term_35_lookup emitted_term_36_lookup emitted_term_37_lookup emitted_term_38_lookup

def word (bits : BitVec 256) : Ora.Integer.Value := ⟨.unsigned .w256, bits⟩
def returned : EdgeRow := row.edges[0]'(by decide)
def back : EdgeRow := row.edges[1]'(by decide)
def done : EdgeRow := row.edges[2]'(by decide)
def context (env : Env) := directStorageContext emittedManifest env row.variables
def events (terms : List Nat) : List AdapterEvent := terms.map fun term => .assume (.ordinary (.inl (.term term)))
def semantics (env : Env) := (adapterSemantics {} (context env)).withCallChecks

theorem traceState (env : Env) (terms : List Nat) (initial : StorageActionState) (result : TimelineResult)
    (trace : TimelineExec (semantics env) (events terms) initial result) : result.state = initial := by
  induction terms generalizing result with
  | nil => cases trace; rfl
  | cons term terms ih =>
    cases trace with
    | cons first rest => cases first; have equal := ih _ rest; exact equal

theorem noChecks (sem : TimelineSemantics F L V C) (formulas : List F) (initial : StorageActionState) :
    TimelineSafe sem (formulas.map .assume) initial := by
  intro before formula after equal
  have member : TimelineEvent.check formula ∈ before ++ .check formula :: after := by simp
  rw [← equal] at member
  simp only [List.mem_map] at member
  rcases member with ⟨_, _, impossible⟩
  cases impossible

theorem typedState (locals : ScalarLoopState) (ready : stateReady row.variables locals) :
    ∃ bits, locals = [word bits] := by
  ora_loop_unpack_ready ready
  exact ⟨_, rfl⟩

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
      adapterLoopEdge emittedManifest {} row returned (events [19, 22]) .functionReturn captured { caller with reads := [] },
      adapterLoopEdge emittedManifest {} row back (events [19, 23, 27]) .backedge captured { caller with reads := [] },
      adapterLoopEdge emittedManifest {} row done (events [31]) .successor captured { caller with reads := [] }] }
theorem bound : emittedCallManifest.loop {} emittedLoopVcSha256_1 row = .ok vc := by rfl

theorem returnValues (env : Env) (entry : StorageState) (current : LoopState) (bits : BitVec 256)
    (locals : current.locals = [word bits]) :
    directStorageFinalLocals? (context env) row.variables returned.nextState
      { loop := current, entryStorage := entry } = some [word bits] := by
  cases current with | mk actual storage payload => dsimp at locals; subst actual; rfl

theorem backValues (env : Env) (entry : StorageState) (current : LoopState) (bits : BitVec 256)
    (locals : current.locals = [word bits]) :
    directStorageFinalLocals? (context env) row.variables back.nextState
      { loop := current, entryStorage := entry } = some [word (bits + 1)] := by
  cases current with | mk actual storage payload => dsimp at locals; subst actual; rfl

theorem doneValues (env : Env) (entry : StorageState) (current : LoopState) (bits : BitVec 256)
    (locals : current.locals = [word bits]) :
    directStorageFinalLocals? (context env) row.variables done.nextState
      { loop := current, entryStorage := entry } = some [word bits] := by
  cases current with | mk actual storage payload => dsimp at locals; subst actual; rfl

theorem returnPayload (env : Env) (entry : StorageState) (current : LoopState) (bits : BitVec 256)
    (locals : current.locals = [word bits]) :
    callEvaluateReturn? (context env).evalValue? (.direct .success [.term 33])
      { loop := current, entryStorage := entry } = some { outcome := .success, payload := [.integer (word bits)] } := by
  cases current with | mk actual storage payload => dsimp at locals; subst actual; rfl

#print axioms bound
#print axioms returnValues
#print axioms backValues
#print axioms doneValues
#print axioms returnPayload


theorem invariantCompatible (env : Env) :
    directFormulaControlCompatible emittedManifest env row.variables (.term 6) := by
  ora_loop_direct_formula_control_in row env

theorem invariantAt (env : Env) (entry : StorageState) (current : LoopState)
    (bits : BitVec 256) (locals : current.locals = [word bits]) :
    directFormulasHold (context env) entry current [.term 6] ↔ bits.toNat ≤ 2 := by
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
    directFormulaControlCompatible emittedManifest env row.variables (.term 19) := by
  ora_loop_direct_formula_control_in row env

theorem guardAt (env : Env) (caller : StorageActionState) (current : LoopState)
    (bits : BitVec 256) (locals : current.locals = [word bits])
    (holds : timelineFormulaHolds (semantics env) (.ordinary (.inl (.term 19)))
      { loop := current, entryStorage := caller.entryStorage }) : bits.toNat < 2 := by
  have scalar := (guardCompatible env current caller.entryStorage).mp holds
  rw [locals] at scalar
  ora_loop_formula_context_reduce at scalar
  ora_loop_value_reduce at scalar
  simp only [word, if_true, lookupFreeBinding, beq_self_eq_true, ora_loop_data, Option.some_bind] at scalar
  ora_loop_normalize at scalar
  exact scalar

theorem returnNext (env : Env) (caller : StorageActionState) (current following : LoopState)
    (bits : BitVec 256) (locals : current.locals = [word bits])
    (next : (adapterLoopEdge emittedManifest {} row returned (events [19, 22]) .functionReturn
      env { caller with reads := [] }).next current following) :
    following = { current with
      locals := [word bits]
      returned := some { outcome := .success, payload := [.integer (word bits)] } } := by
  rcases next with ⟨result, trace, finish⟩
  rw [traceState env [19, 22] { caller with loop := current, reads := [] } result trace] at finish
  have values := returnValues env caller.entryStorage current bits locals
  have payload := returnPayload env caller.entryStorage current bits locals
  dsimp [context] at values payload
  change ((directStorageFinalLocals? _ _ _ _).bind _) = some following at finish
  rw [values] at finish
  simp only [Option.bind_eq_bind, Option.some_bind, Option.pure_def] at finish
  simp only [returned, row, emittedLoopSummary_1, List.getElem_cons_zero, adapterReturn, List.map] at finish
  simp only [row, emittedLoopSummary_1] at payload
  rw [payload] at finish
  exact (Option.some.inj finish).symm

theorem backNext (env : Env) (caller : StorageActionState) (current following : LoopState)
    (bits : BitVec 256) (locals : current.locals = [word bits])
    (next : (adapterLoopEdge emittedManifest {} row back (events [19, 23, 27]) .backedge
      env { caller with reads := [] }).next current following) :
    following = { current with locals := [word (bits + 1)] } ∧
      timelineFormulaHolds (semantics env) (.ordinary (.inl (.term 19)))
        { loop := current, entryStorage := caller.entryStorage } := by
  rcases next with ⟨result, trace, finish⟩
  rw [traceState env [19, 23, 27] { caller with loop := current, reads := [] } result trace] at finish
  have values := backValues env caller.entryStorage current bits locals
  dsimp [context] at values
  change ((directStorageFinalLocals? _ _ _ _).bind _) = some following at finish
  rw [values] at finish
  refine ⟨(Option.some.inj finish).symm, ?_⟩
  cases trace with
  | cons step rest => cases step with
    | assume evaluated holds => exact ⟨_, evaluated, holds⟩

theorem doneNext (env : Env) (caller : StorageActionState) (current following : LoopState)
    (bits : BitVec 256) (locals : current.locals = [word bits])
    (next : (adapterLoopEdge emittedManifest {} row done (events [31]) .successor
      env { caller with reads := [] }).next current following) : following = current := by
  rcases next with ⟨result, trace, finish⟩
  rw [traceState env [31] { caller with loop := current, reads := [] } result trace] at finish
  have values := doneValues env caller.entryStorage current bits locals
  dsimp [context] at values
  change ((directStorageFinalLocals? _ _ _ _).bind _) = some following at finish
  rw [values] at finish
  have equal := (Option.some.inj finish).symm
  simpa only [← locals] using equal

#print axioms invariantAt
#print axioms guardAt
#print axioms returnNext
#print axioms backNext
#print axioms doneNext


theorem edgeSafety (env : Env) (caller : StorageActionState) (edge : EdgeRow) (terms : List Nat)
    (which : (edge = returned ∧ terms = [19, 22]) ∨ (edge = back ∧ terms = [19, 23, 27]) ∨
      (edge = done ∧ terms = [31])) (current : LoopState) (ready : stateReady row.variables current.locals) :
    (adapterLoopEdge emittedManifest {} row edge (events terms) (directStorageTarget edge)
      env { caller with reads := [] }).safe current := by
  rcases typedState current.locals ready with ⟨bits, locals⟩
  constructor
  · simpa only [events, List.map_map, Function.comp_def] using noChecks (semantics env)
      (terms.map fun term => (.ordinary (.inl (.term term)) : CallFormula AdapterFormula FormulaRef AdapterChild))
      { caller with loop := current, reads := [] }
  · intro result trace
    rw [traceState env terms { caller with loop := current, reads := [] } result trace]
    rcases which with ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩
    · have values := returnValues env caller.entryStorage current bits locals
      have payload := returnPayload env caller.entryStorage current bits locals
      dsimp [context] at values payload
      change ((directStorageFinalLocals? _ _ _ _).bind _).isSome = true
      rw [values]
      simp only [Option.bind_eq_bind, Option.some_bind, Option.pure_def]
      simp only [returned, row, emittedLoopSummary_1, List.getElem_cons_zero, adapterReturn, List.map]
      simp only [row, emittedLoopSummary_1] at payload
      rw [payload]
      rfl
    · have values := backValues env caller.entryStorage current bits locals
      dsimp [context] at values
      change ((directStorageFinalLocals? _ _ _ _).bind _).isSome = true
      rw [values]
      rfl
    · have values := doneValues env caller.entryStorage current bits locals
      dsimp [context] at values
      change ((directStorageFinalLocals? _ _ _ _).bind _).isSome = true
      rw [values]
      rfl

theorem stepBound (bits : BitVec 256) (less : bits.toNat < 2) : (bits + 1).toNat ≤ 2 := by
  rw [BitVec.toNat_add]
  change (bits.toNat + 1) % 2 ^ 256 ≤ 2
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
    rcases member with rfl | rfl | rfl
    · cases target
    · rcases typedState _ invariant.2.1 with ⟨bits, locals⟩
      have shape := backNext _ caller current following bits locals next
      have guard := guardAt _ caller current bits locals shape.2
      refine ⟨invariant.1, ?_, ?_⟩
      · rw [shape.1]
        rfl
      · exact (invariantAt _ _ _ (bits + 1) (by rw [shape.1])).mpr (stepBound bits guard)
    · cases target
  edgeSafety := by
    intro edge current member invariant path
    simp only [vc, List.mem_cons, List.not_mem_nil, or_false] at member
    rcases member with rfl | rfl | rfl
    · exact edgeSafety _ caller returned [19, 22] (Or.inl ⟨rfl, rfl⟩) current invariant.2.1
    · exact edgeSafety _ caller back [19, 23, 27] (Or.inr (Or.inl ⟨rfl, rfl⟩)) current invariant.2.1
    · exact edgeSafety _ caller done [31] (Or.inr (Or.inr ⟨rfl, rfl⟩)) current invariant.2.1
  exit := by
    intro edge current following member target invariant path safe next
    simp only [vc, List.mem_cons, List.not_mem_nil, or_false] at member
    rcases member with rfl | rfl | rfl
    · intro formula member
      cases member
    · exact (target rfl).elim
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
      rcases typedState _ (readyAt current reached).2.1 with ⟨bits, locals⟩
      simp only [vc, List.mem_cons, List.not_mem_nil, or_false] at member
      rcases member with rfl | rfl | rfl
      · cases target
      · rw [(backNext _ caller current following bits locals next).1]
        exact ih
      · cases target
  rcases typedState _ (readyAt current reached).2.1 with ⟨bits, locals⟩
  simp only [vc, List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl | rfl
  · rw [returnNext _ caller current following bits locals next]
    exact preserved.trans initialized.2.2.2.1
  · exact False.elim (notBackedge targetEq.symm)
  · rw [doneNext _ caller current following bits locals next]
    exact preserved.trans initialized.2.2.2.1

def frames (env : Env) (target : ControlTarget) (caller : StorageActionState) :
    CertifiedWriteFrame (fun before after => adapterLoopRelation vc env target { caller with loop := before } after) := {
  footprint := fun _ => False
  preservesOutside := fun before after run location _ =>
    congrFun (storagePreserved env { caller with loop := before } target after run) location }

def supplied : AdapterLoopProof emittedCallManifest emittedLoopVcSha256_1 row where
  previous := CallAdapterCertificates.empty emittedCallManifest
  vc := vc
  bound := bound
  induction := proof
  frames := frames
  witness := by
    let env : Env := Env.empty.setFree { file_id := 0, pattern_id := 0 } (.integer (word 0))
    let store : StorageState := fun _ => .bool false
    let caller : StorageActionState := { loop := { locals := [], storage := store }, entryStorage := store }
    refine ⟨env, caller, { locals := [word 0], storage := store }, ?_, rfl, rfl, rfl, rfl⟩
    constructor
    · intro slot member
      simp only [row, emittedLoopSummary_1, List.mem_singleton] at member
      subst slot
      exact ⟨word 0, rfl, rfl⟩
    constructor
    · intro fact member
      cases member
    constructor
    · refine ⟨[.integer (word 0)], rfl, ?_⟩
      intro formula member
      cases member
    · intro edge member
      simp only [row, emittedLoopSummary_1, List.mem_cons, List.not_mem_nil, or_false] at member
      rcases member with rfl | rfl | rfl <;> decide

theorem loopAuthorization : emittedCallLoop_1 := ⟨supplied⟩
#print axioms proof
#print axioms frames
#print axioms loopAuthorization


theorem exitLocals (env : Env) (caller : StorageActionState) (target : ControlTarget) (following : LoopState)
    (run : adapterLoopRelation vc env target caller following) : ∃ bits, following.locals = [word bits] := by
  rcases run with ⟨initial, notBackedge, initialized, current, edge, reached, member, targetEq, path, safe, next⟩
  have invariant := invariant_holds_on_reached_state (vc env caller) (proof env caller).base
    (proof env caller).backedge initialized reached
  rcases typedState _ invariant.2.1 with ⟨bits, locals⟩
  simp only [vc, List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl | rfl
  · exact ⟨bits, congrArg LoopState.locals (returnNext _ caller current following bits locals next)⟩
  · exact False.elim (notBackedge targetEq.symm)
  · exact ⟨bits, (congrArg LoopState.locals (doneNext _ caller current following bits locals next)).trans locals⟩

theorem exitPayload (env : Env) (caller : StorageActionState) (following : LoopState)
    (run : adapterLoopRelation vc env .functionReturn caller following) :
    ∃ bits, following.returned = some { outcome := .success, payload := [.integer (word bits)] } := by
  rcases run with ⟨initial, _, initialized, current, edge, reached, member, target, path, safe, next⟩
  have invariant := invariant_holds_on_reached_state (vc env caller) (proof env caller).base
    (proof env caller).backedge initialized reached
  rcases typedState _ invariant.2.1 with ⟨bits, locals⟩
  simp only [vc, List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl | rfl
  · exact ⟨bits, congrArg LoopState.returned (returnNext _ caller current following bits locals next)⟩
  · cases target
  · cases target

def certificates := supplied.install
def child (target : ControlTarget) : AdapterChild := (adapterLoopCertificate vc frames, target)
def bindings : List LoopResultBinding := [{ id := { file_id := 4294967283, pattern_id := 0 }, index := 0, ty := .compilerTypeId 6 }]
def functionContext := directStorageContext emittedManifest Env.empty []
def functionSemantics := adapterSemantics certificates.dependencies functionContext
def entryEvents (target : ControlTarget) (results : List LoopResultBinding) : List AdapterEvent := [
  .check (.ordinary (.inr (child target))), .check (.loopResultsReady (child target) results),
  .childResult (child target) results ]
def successorEvents : List AdapterEvent := entryEvents .successor bindings ++ [.assume (.ordinary (.inl (.term 14)))]
def returnEvents : List AdapterEvent := entryEvents .functionReturn []
def functionBody : AdapterBody := {
  signature := emittedFunctionSummary_1.callSignature
  requires := emittedFunctionSummary_1.entryFactsHold emittedManifest
  paths := [{ events := successorEvents, returns := .direct .success [.term 35] },
    { events := returnEvents, returns := .loopExit }]
  semantics := fun _ => functionSemantics.withCallChecks }
theorem functionBound : emittedCallManifest.function certificates.dependencies
    emittedFunctionVcSha256_1 emittedFunctionSummary_1 = .ok functionBody := by rfl

def functionEntry (bits : BitVec 256) (store : StorageState) : StorageActionState := {
  loop := { locals := [], storage := store }, entryStorage := store
  reads := [{ id := { file_id := 0, pattern_id := 0 }, value := .integer (word bits) }] }
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
    rcases member with rfl | rfl | rfl <;> simp [functionEntry, timelineReadIds]

theorem childValues (bits : BitVec 256) (store : StorageState) (target : ControlTarget) (following : LoopState)
    (related : functionSemantics.childRelation (child target) (functionEntry bits store) following) :
    ∃ result, following.locals = [word result] := by
  rcases related with ⟨env, _, run⟩
  exact exitLocals env (functionEntry bits store) target following run

theorem childResults (bits : BitVec 256) (store : StorageState) (target : ControlTarget) :
    loopResultsReady functionSemantics (child target) bindings (functionEntry bits store) := by
  refine ⟨?_, ?_⟩
  · change ([({ file_id := 0, pattern_id := 0 } : FreeVarId), { file_id := 4294967283, pattern_id := 0 }]).Nodup
    decide
  intro following related
  rcases childValues bits store target following related with ⟨result, locals⟩
  simp [loopResultBindings?, bindings, locals, word, storageValueMatchesTy]

theorem emptyResults (bits : BitVec 256) (store : StorageState) (target : ControlTarget) :
    loopResultsReady functionSemantics (child target) [] (functionEntry bits store) := by
  refine ⟨?_, fun _ _ => rfl⟩
  change ([({ file_id := 0, pattern_id := 0 } : FreeVarId)]).Nodup
  decide

theorem typedInput (input : CallInput) (typed : functionBody.signature.inputTyped input = true) :
    ∃ bits, input.arguments = [.integer (word bits)] := by
  have arity := typed
  change (decide ([({ file_id := 0, pattern_id := 0 } : FreeVarId)]).Nodup &&
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
#print axioms exitPayload
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

theorem successorSafe (bits : BitVec 256) (store : StorageState) :
    TimelineSafe functionSemantics.withCallChecks successorEvents (functionEntry bits store) :=
  entrySafe bits store .successor bindings _ (childResults bits store .successor)
    (fun state => noChecks functionSemantics.withCallChecks [.ordinary (.inl (.term 14))] state)

theorem returnSafe (bits : BitVec 256) (store : StorageState) :
    TimelineSafe functionSemantics.withCallChecks returnEvents (functionEntry bits store) := by
  simpa only [List.append_nil] using entrySafe bits store .functionReturn [] []
    (emptyResults bits store .functionReturn) (emptySafe _)

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
  reads := [{ id := { file_id := 0, pattern_id := 0 }, value := .integer (word bits) },
    { id := { file_id := 4294967283, pattern_id := 0 }, value := .integer (word value) }] }

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

theorem assumeTrace (sem : TimelineSemantics F L V C) (formula : F)
    (tail : List (EventOf F L V C)) (initial : StorageActionState) (result : TimelineResult)
    (trace : TimelineExec sem (.assume formula :: tail) initial result) :
    ∃ following, TimelineExec sem tail initial following ∧ result.state = following.state := by
  cases trace with
  | cons first rest => cases first; exact ⟨_, rest, rfl⟩

theorem successorCompleted (bits : BitVec 256) (store : StorageState) (result : TimelineResult)
    (trace : TimelineExec functionSemantics.withCallChecks successorEvents (functionEntry bits store) result) :
    ∃ value following, result.state = resumedState bits value store following := by
  rcases checkTrace _ _ _ _ _ trace with ⟨r1, t1, e1⟩
  rcases checkTrace _ _ _ _ _ t1 with ⟨r2, t2, e2⟩
  cases t2 with
  | cons step rest =>
    cases step with
    | childResult related resume =>
      rcases resumedValues bits store _ _ related resume with ⟨value, equal⟩
      rcases assumeTrace _ _ _ _ _ rest with ⟨final, trace, finalEq⟩
      cases trace
      exact ⟨value, _, e1.trans (e2.trans (finalEq.trans equal))⟩

theorem returnCompleted (bits : BitVec 256) (store : StorageState) (result : TimelineResult)
    (trace : TimelineExec functionSemantics.withCallChecks returnEvents (functionEntry bits store) result) :
    ∃ value, result.state.loop.returned = some { outcome := .success, payload := [.integer (word value)] } := by
  rcases checkTrace _ _ _ _ _ trace with ⟨r1, t1, e1⟩
  rcases checkTrace _ _ _ _ _ t1 with ⟨r2, t2, e2⟩
  cases t2 with
  | cons step rest =>
    cases rest
    cases step with
    | childResult related resume =>
      rcases related with ⟨env, _, run⟩
      rcases exitPayload env (functionEntry bits store) _ run with ⟨value, payload⟩
      have effects := resumeLoop_effects _ _ _ _ resume
      rw [e1, e2]
      exact ⟨value, effects.2.2.2.1.trans payload⟩

theorem returnedValue (bits value : BitVec 256) (store : StorageState) (following : LoopState) :
    functionContext.evalValue? (.term 35) (resumedState bits value store following) =
      some (.integer (word value)) := by
  let state := resumedState bits value store following
  let env := functionContext.withStoreViews (bindStorageReads Env.empty state.reads) state
  apply StorageFormulaContext.evalValue_free_variable functionContext state env 35
    { id := { file_id := 4294967283, pattern_id := 0 }, name := "i", ty := some (.compilerTypeId 6) }
  · rfl
  · exact emitted_term_35_lookup
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

#print axioms successorSafe
#print axioms returnSafe
#print axioms loopPrefixResults
#print axioms successorCompleted
#print axioms returnCompleted
#print axioms returnedValue


theorem functionCertificate : functionBody.Certificate where
  safety := by
    intro input typed required initial entry path member
    rcases typedInput input typed with ⟨bits, arguments⟩
    have equal := Option.some.inj ((entryTyped input bits typed arguments).symm.trans entry)
    subst initial
    simp only [functionBody, List.mem_cons, List.not_mem_nil, or_false] at member
    rcases member with rfl | rfl
    · exact successorSafe bits input.storage
    · exact returnSafe bits input.storage
  calls := by
    intro input typed required initial entry path member before invocation after equal
    have contains : TimelineEvent.call invocation ∈ before ++ .call invocation :: after := by simp
    rw [← equal] at contains
    simp only [functionBody, List.mem_cons, List.not_mem_nil, or_false] at member
    rcases member with rfl | rfl <;> simp [successorEvents, returnEvents, entryEvents] at contains
  loopResults := by
    intro input typed required initial entry path member before dependency results after equal result trace
    rcases typedInput input typed with ⟨bits, arguments⟩
    have same := Option.some.inj ((entryTyped input bits typed arguments).symm.trans entry)
    subst initial
    simp only [functionBody, List.mem_cons, List.not_mem_nil, or_false] at member
    rcases member with rfl | rfl
    · exact loopPrefixResults functionSemantics (child .successor) bindings
        [.assume (.ordinary (.inl (.term 14)))] _ (childResults bits input.storage .successor)
        (by intro dependency results; simp) _ _ _ _ equal _ trace
    · exact loopPrefixResults functionSemantics (child .functionReturn) [] [] _
        (emptyResults bits input.storage .functionReturn)
        (by intro dependency results; simp) _ _ _ _ equal _ trace
  returns := by
    intro input initial path result typed required entry member trace
    rcases typedInput input typed with ⟨bits, arguments⟩
    have equal := Option.some.inj ((entryTyped input bits typed arguments).symm.trans entry)
    subst initial
    simp only [functionBody, List.mem_cons, List.not_mem_nil, or_false] at member
    rcases member with rfl | rfl
    · rcases successorCompleted bits input.storage result trace with ⟨value, following, equal⟩
      rw [equal]
      refine ⟨{ outcome := .success, payload := [.integer (word value)] }, ?_, rfl⟩
      change callEvaluateReturn? functionContext.evalValue? (.direct .success [.term 35])
        (resumedState bits value input.storage following) = _
      simp only [callEvaluateReturn?, callEvaluateValues?, List.mapM_cons, returnedValue,
        List.mapM_nil, Option.bind_eq_bind, Option.some_bind, Option.pure_def, Option.map_some']
    · rcases returnCompleted bits input.storage result trace with ⟨value, payload⟩
      exact ⟨{ outcome := .success, payload := [.integer (word value)] }, payload, rfl⟩

def installed := certificates.addFunction emittedFunctionVcSha256_1 emittedFunctionSummary_1
  functionBody functionBound functionCertificate
theorem custody : emittedCallCustody_1 := ⟨installed, rfl⟩
#print axioms functionCertificate
#print axioms custody


theorem functionJoint (bits : BitVec 256) (store : StorageState) (output : CallOutput)
    (execution : functionBody.execution { arguments := [.integer (word bits)], storage := store } output)
    (outcome : output.outcome = .success) : ∃ value, output.payload = [.integer (word value)] := by
  have typed := functionCertificate.outputTyped functionBody _ output rfl
    (by intro formula member; cases member) execution
  unfold CallSignature.outputTyped at typed
  rw [outcome] at typed
  exact typedInput { arguments := output.payload, storage := output.storage } typed

def invocation : CallInvocation FormulaRef := {
  callee := adapterKey 1 emittedFunctionVcSha256_1, arguments := [.term 36]
  outcome := .success, results := [{ file_id := 4294967245, pattern_id := 0 }] }
def callerSemantics := adapterSemantics installed.dependencies functionContext
def callerEvents : List AdapterEvent := [.check (.ready invocation), .call invocation]
def callerBody : AdapterBody := {
  signature := emittedFunctionSummary_2.callSignature
  requires := emittedFunctionSummary_2.entryFactsHold emittedManifest
  paths := [{ events := callerEvents, returns := .direct .success [.term 38] }]
  semantics := fun _ => callerSemantics.withCallChecks }
theorem callerBound : emittedCallManifest.function installed.dependencies
    emittedFunctionVcSha256_2 emittedFunctionSummary_2 = .ok callerBody := by rfl

def callerEntry (bits : BitVec 256) (store : StorageState) : StorageActionState := {
  loop := { locals := [], storage := store }, entryStorage := store
  reads := [{ id := { file_id := 0, pattern_id := 3 }, value := .integer (word bits) }] }

theorem resolved : callerSemantics.calls invocation.callee =
    some (functionBody.certified functionCertificate) := by rfl

theorem invocationReady (bits : BitVec 256) (store : StorageState) :
    invocation.ready callerSemantics.calls callerSemantics.storage.evalValue (callerEntry bits store) := by
  refine ⟨functionBody.certified functionCertificate, [.integer (word bits)], [.compilerTypeId 6],
    resolved, rfl, rfl, ?_, ?_, rfl, rfl⟩
  · intro formula member
    cases member
  · change ([({ file_id := 0, pattern_id := 3 } : FreeVarId), { file_id := 4294967245, pattern_id := 0 }]).Nodup
    decide

def callerReturned (bits result : BitVec 256) (store finalStore : StorageState) :
    StorageActionState := {
  loop := { locals := [], storage := finalStore }, entryStorage := store
  reads := [{ id := { file_id := 0, pattern_id := 3 }, value := .integer (word bits) },
    { id := { file_id := 4294967245, pattern_id := 0 }, value := .integer (word result) }] }

theorem invocationResult (bits : BitVec 256) (store : StorageState) (result : TimelineResult)
    (step : TimelineEventExec callerSemantics.withCallChecks (.call invocation) (callerEntry bits store) result) :
    ∃ output finalStore, result.state = callerReturned bits output store finalStore := by
  cases step with
  | call found arguments related outcome arity fresh =>
    change callerSemantics.calls invocation.callee = some _ at found
    rw [resolved] at found
    cases Option.some.inj found
    have evaluated : callEvaluateValues? callerSemantics.storage.evalValue invocation.arguments
        (callerEntry bits store) = some [.integer (word bits)] := rfl
    have equal := Option.some.inj (evaluated.symm.trans arguments)
    subst_vars
    rename_i returned
    rcases functionJoint bits store returned related outcome with ⟨output, payload⟩
    refine ⟨output, returned.storage, ?_⟩
    dsimp [CallInvocation.resume, invocation, callerEntry, callerReturned]
    rw [payload]
    rfl

theorem callerSafe (bits : BitVec 256) (store : StorageState) :
    TimelineSafe callerSemantics.withCallChecks callerEvents (callerEntry bits store) := by
  unfold callerEvents
  refine checkSafe _ _ _ _ ⟨_, rfl, invocationReady bits store⟩ ?_
  apply TimelineSafe.cons
  · intro formula equal
    cases equal
  · intro result step
    exact emptySafe _ _

theorem callerCompleted (bits : BitVec 256) (store : StorageState) (result : TimelineResult)
    (trace : TimelineExec callerSemantics.withCallChecks callerEvents (callerEntry bits store) result) :
    ∃ output finalStore, result.state = callerReturned bits output store finalStore := by
  rcases checkTrace _ _ _ _ _ trace with ⟨r1, t1, e1⟩
  cases t1 with
  | cons step rest =>
    cases rest
    rcases invocationResult bits store _ step with ⟨output, finalStore, equal⟩
    exact ⟨output, finalStore, e1.trans equal⟩

theorem callerTyped (input : CallInput) (typed : callerBody.signature.inputTyped input = true) :
    ∃ bits, input.arguments = [.integer (word bits)] := typedInput input typed

theorem callerEntryTyped (input : CallInput) (bits : BitVec 256)
    (typed : callerBody.signature.inputTyped input = true)
    (arguments : input.arguments = [.integer (word bits)]) :
    callerBody.signature.entry? input = some (callerEntry bits input.storage) := by
  simp only [CallSignature.entry?, typed, if_true]
  rw [arguments]
  rfl

theorem callerReturnedValue (bits result : BitVec 256) (store finalStore : StorageState) :
    functionContext.evalValue? (.term 38) (callerReturned bits result store finalStore) =
      some (.integer (word result)) := by
  let state := callerReturned bits result store finalStore
  let env := functionContext.withStoreViews (bindStorageReads Env.empty state.reads) state
  apply StorageFormulaContext.evalValue_free_variable functionContext state env 38
    { id := { file_id := 4294967245, pattern_id := 0 }, name := "call_result", ty := some (.compilerTypeId 6) }
  · rfl
  · exact emitted_term_38_lookup
  · simp only [env, StorageFormulaContext.withStoreViews_lookupFree, state, callerReturned,
      bindStorageReads, List.foldl, Env.setFree, Env.lookupFree, lookupFreeBinding,
      beq_self_eq_true, if_true]

theorem callerCertificate : callerBody.Certificate where
  safety := by
    intro input typed required initial entry path member
    rcases callerTyped input typed with ⟨bits, arguments⟩
    have equal := Option.some.inj ((callerEntryTyped input bits typed arguments).symm.trans entry)
    subst initial
    simp only [callerBody, List.mem_singleton] at member
    subst path
    exact callerSafe bits input.storage
  calls := by
    intro input typed required initial entry path member before call after equal result trace
    rcases callerTyped input typed with ⟨bits, arguments⟩
    have same := Option.some.inj ((callerEntryTyped input bits typed arguments).symm.trans entry)
    subst initial
    simp only [callerBody, List.mem_singleton] at member
    subst path
    cases before with
    | nil => cases equal
    | cons a tail => cases tail with
      | nil =>
        simp only [callerEvents, List.cons_append, List.nil_append,
          List.cons.injEq, TimelineEvent.call.injEq] at equal
        rcases equal with ⟨rfl, rfl, rfl⟩
        rcases checkTrace _ _ _ _ _ trace with ⟨r1, t1, e1⟩
        cases t1
        rw [e1]
        exact invocationReady bits input.storage
      | cons b tail => simp [callerEvents] at equal
  loopResults := by
    intro input typed required initial entry path member before dependency results after equal
    simp only [callerBody, List.mem_singleton] at member
    subst path
    have contains : TimelineEvent.childResult dependency results ∈ before ++ .childResult dependency results :: after := by simp
    rw [← equal] at contains
    simp [callerEvents] at contains
  returns := by
    intro input initial path result typed required entry member trace
    rcases callerTyped input typed with ⟨bits, arguments⟩
    have equal := Option.some.inj ((callerEntryTyped input bits typed arguments).symm.trans entry)
    subst initial
    simp only [callerBody, List.mem_singleton] at member
    subst path
    rcases callerCompleted bits input.storage result trace with ⟨output, finalStore, equal⟩
    rw [equal]
    refine ⟨{ outcome := .success, payload := [.integer (word output)] }, ?_, rfl⟩
    change callEvaluateReturn? functionContext.evalValue? (.direct .success [.term 38])
      (callerReturned bits output input.storage finalStore) = _
    simp only [callEvaluateReturn?, callEvaluateValues?, List.mapM_cons, callerReturnedValue, List.mapM_nil,
      Option.bind_eq_bind, Option.some_bind, Option.pure_def, Option.map_some']

def complete := installed.addFunction emittedFunctionVcSha256_2 emittedFunctionSummary_2
  callerBody callerBound callerCertificate
theorem callerCustody : emittedCallCustody_2 := ⟨complete, rfl⟩
#print axioms functionJoint
#print axioms callerBound
#print axioms invocationReady
#print axioms invocationResult
#print axioms callerReturnedValue
#print axioms callerCertificate
#print axioms callerCustody
end G9cSuppliedReturn

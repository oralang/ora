-- Supplied proofs over the compiler-emitted rows; no expected manifest is copied here.
-- Check through call_projection_callee_loop_value.proofs.json.

namespace G9cSuppliedRepeat
open Ora.Loop Ora.Obligation
open Ora.Generated.Obligations.Source_call_projection_callee_loop_value_9b67d620b2daaec5
abbrev autoSummary := emittedLoopSummary_1
abbrev autoAssumptions := [emittedAssumptionRow_1]
attribute [local ora_loop_row] autoSummary emittedLoopSummary_1 autoAssumptions emittedAssumptionRow_1
attribute [local ora_loop_data] emitted_terms_length
attribute [local ora_loop_data] emitted_term_0_lookup emitted_term_1_lookup emitted_term_2_lookup emitted_term_3_lookup emitted_term_4_lookup emitted_term_5_lookup emitted_term_6_lookup emitted_term_7_lookup emitted_term_8_lookup emitted_term_9_lookup emitted_term_10_lookup emitted_term_11_lookup emitted_term_12_lookup emitted_term_13_lookup emitted_term_14_lookup emitted_term_15_lookup emitted_term_16_lookup emitted_term_17_lookup emitted_term_18_lookup emitted_term_19_lookup emitted_term_20_lookup emitted_term_21_lookup emitted_term_22_lookup emitted_term_23_lookup emitted_term_24_lookup emitted_term_25_lookup emitted_term_26_lookup emitted_term_27_lookup emitted_term_28_lookup emitted_term_29_lookup emitted_term_30_lookup emitted_term_31_lookup emitted_term_32_lookup emitted_term_33_lookup emitted_term_34_lookup emitted_term_35_lookup emitted_term_36_lookup emitted_term_37_lookup emitted_term_38_lookup emitted_term_39_lookup emitted_term_40_lookup emitted_term_41_lookup emitted_term_42_lookup emitted_term_43_lookup emitted_term_44_lookup emitted_term_45_lookup
attribute [local ora_loop_manifest] emitted_manifest_wf emitted_terms_topological
private theorem eq_0_0 : (({ file_id := 0, pattern_id := 0 } : FreeVarId) == { file_id := 0, pattern_id := 0 }) = true := by rfl
private theorem eq_0_1 : (({ file_id := 0, pattern_id := 0 } : FreeVarId) == { file_id := 4294967294, pattern_id := 0 }) = false := by rfl
private theorem eq_1_0 : (({ file_id := 4294967294, pattern_id := 0 } : FreeVarId) == { file_id := 0, pattern_id := 0 }) = false := by rfl
private theorem eq_1_1 : (({ file_id := 4294967294, pattern_id := 0 } : FreeVarId) == { file_id := 4294967294, pattern_id := 0 }) = true := by rfl
attribute [local ora_loop_data] eq_0_0 eq_0_1 eq_1_0 eq_1_1
theorem supported : autoSummary.supported emittedManifest = true := by ora_loop_support
attribute [local ora_loop_supported] supported
theorem scalar : loopProofAuthorizationFromAssumptions emittedManifest autoAssumptions autoSummary := by
  ora_loop_auto
#print axioms scalar

def word (bits : BitVec 256) : Ora.Integer.Value := ⟨.unsigned .w256, bits⟩
def back : EdgeRow := emittedLoopSummary_1.edges[0]'(by decide)
def done : EdgeRow := emittedLoopSummary_1.edges[1]'(by decide)
def context (env : Env) := directStorageContext emittedManifest env autoSummary.variables
def events (term : Nat) : List AdapterEvent := [.assume (.ordinary (.inl (.term term)))]
def semantics (env : Env) := (adapterSemantics {} (context env)).withCallChecks

theorem traceState (env : Env) (term : Nat) (initial : StorageActionState) (result : TimelineResult)
    (trace : TimelineExec (semantics env) (events term) initial result) : result.state = initial := by
  cases trace with
  | cons event rest => cases rest; cases event; rfl

theorem invariantCompatible (env : Env) :
    directFormulaControlCompatible emittedManifest env autoSummary.variables (.term 9) := by
  ora_loop_direct_formula_control_in autoSummary env

theorem invariantAt (env : Env) (entry : StorageState) (state : LoopState)
    (bits limit : BitVec 256)
    (locals : state.locals = [word bits])
    (lookup : env.lookupFree { file_id := 0, pattern_id := 0 } = some (.integer (word limit))) :
    directFormulasHold (context env) entry state [.term 9] ↔ bits.toNat ≤ limit.toNat := by
  change (∀ formula ∈ [FormulaRef.term 9], _) ↔ _
  simp only [List.forall_mem_cons, List.forall_mem_nil, and_true]
  rw [show (∃ proposition, (context env).evalFormula? (.term 9)
      { loop := state, entryStorage := entry } = some proposition ∧ proposition) ↔
      formulaHoldsInState emittedManifest env autoSummary.variables state.locals (.term 9)
    from invariantCompatible env state entry]
  rw [locals]
  ora_loop_formula_context_reduce
  ora_loop_value_reduce
  simp only [word, Env.lookupFree] at lookup
  simp only [word, if_true, lookupFreeBinding, ora_loop_data, Option.some_bind, lookup,
    List.not_mem_nil, false_implies, implies_true, and_true]
  ora_loop_normalize
  rfl

theorem backValues (env : Env) (entry : StorageState) (state : LoopState) (bits : BitVec 256)
    (locals : state.locals = [word bits]) :
    directStorageFinalLocals? (context env) autoSummary.variables back.nextState
      { loop := state, entryStorage := entry } = some [word (bits + 1)] := by
  cases state with
  | mk actual store returned =>
    simp only at locals
    subst actual
    rfl

theorem doneValues (env : Env) (entry : StorageState) (state : LoopState) (bits : BitVec 256)
    (locals : state.locals = [word bits]) :
    directStorageFinalLocals? (context env) autoSummary.variables done.nextState
      { loop := state, entryStorage := entry } = some [word bits] := by
  cases state with
  | mk actual store returned =>
    simp only at locals
    subst actual
    rfl

#print axioms invariantAt
#print axioms backValues
#print axioms doneValues

theorem typedState (locals : ScalarLoopState) (ready : stateReady autoSummary.variables locals) :
    ∃ bits, locals = [word bits] := by
  ora_loop_unpack_ready ready
  exact ⟨_, rfl⟩

theorem typedContext (env : Env) (ready : contextReady env autoSummary.contextVariables) :
    ∃ bits, env.lookupFree { file_id := 0, pattern_id := 0 } = some (.integer (word bits)) := by
  rcases ready { index := 0, id := { file_id := 0, pattern_id := 0 }, name := "limit", ty := .compilerTypeId 6 }
    (List.mem_singleton_self _) with ⟨⟨shape, bits⟩, typed, lookup⟩
  change some (.unsigned .w256) = some shape at typed
  cases Option.some.inj typed
  exact ⟨bits, lookup⟩

theorem guardCompatible (env : Env) :
    directFormulaControlCompatible emittedManifest env autoSummary.variables (.term 20) := by
  ora_loop_direct_formula_control_in autoSummary env

theorem guardAt (env : Env) (entry : StorageState) (state : LoopState)
    (bits limit : BitVec 256) (locals : state.locals = [word bits])
    (lookup : env.lookupFree { file_id := 0, pattern_id := 0 } = some (.integer (word limit)))
    (holds : timelineFormulaHolds (semantics env) (.ordinary (.inl (.term 20)))
      { loop := state, entryStorage := entry }) : bits.toNat < limit.toNat := by
  have scalar := (guardCompatible env state entry).mp holds
  rw [locals] at scalar
  ora_loop_formula_context_reduce at scalar
  ora_loop_value_reduce at scalar
  simp only [word, Env.lookupFree] at lookup
  simp only [word, if_true, lookupFreeBinding, ora_loop_data, Option.some_bind, lookup] at scalar
  ora_loop_normalize at scalar
  exact scalar

def facts (env : Env) (caller : StorageActionState) : Prop :=
  contextReady env autoSummary.contextVariables ∧
    headerFactsHoldInEnv emittedManifest env autoSummary.headerFacts ∧
    adapterEntryFacts emittedManifest emittedFunctionSummary_1 env caller.entryStorage ∧
    ∀ edge ∈ autoSummary.edges, (caller.reads.map (·.id) ++ timelineReadIds edge.timeline).Nodup

def vc : AdapterLoopVC := fun env caller =>
  let captured := bindStorageReads env caller.reads
  { initial := fun initial => facts captured caller ∧
      stateReady autoSummary.variables initial.locals ∧
      directInitialValuesAfterPreheader? {
        directStorageContext emittedManifest captured [] with bindLocals := fun env _ => some env
      } caller autoSummary.init = some initial.locals ∧
      initial.storage = caller.loop.storage ∧ initial.returned = none
    invariant := fun current => facts captured caller ∧
      stateReady autoSummary.variables current.locals ∧
      directFormulasHold (context captured) caller.entryStorage current autoSummary.invariants
    edges := [adapterLoopEdge emittedManifest {} autoSummary back (events 20) .backedge captured { caller with reads := [] },
      adapterLoopEdge emittedManifest {} autoSummary done (events 21) .successor captured { caller with reads := [] }] }

theorem bound : emittedCallManifest.loop {} emittedLoopVcSha256_1 autoSummary = .ok vc := by rfl

theorem edgeSafety (env : Env) (caller : StorageActionState) (edge : EdgeRow) (term : Nat)
    (which : edge = back ∨ edge = done) (current : LoopState)
    (ready : stateReady autoSummary.variables current.locals) :
    (adapterLoopEdge emittedManifest {} autoSummary edge (events term)
      (directStorageTarget edge) env { caller with reads := [] }).safe current := by
  rcases typedState current.locals ready with ⟨bits, locals⟩
  constructor
  · intro before formula after equation
    cases before with
    | nil => simp [events] at equation
    | cons head tail => simp [events] at equation
  · intro result trace
    rw [traceState env term { caller with loop := current, reads := [] } result trace]
    have hBack := backValues env caller.entryStorage current bits locals
    have hDone := doneValues env caller.entryStorage current bits locals
    dsimp [context] at hBack hDone
    rcases which with rfl | rfl
    · change ((directStorageFinalLocals? _ _ _ _).bind _).isSome = true
      rw [hBack]
      rfl
    · change ((directStorageFinalLocals? _ _ _ _).bind _).isSome = true
      rw [hDone]
      rfl

theorem backNext (env : Env) (caller : StorageActionState) (current following : LoopState)
    (bits : BitVec 256) (locals : current.locals = [word bits])
    (next : (adapterLoopEdge emittedManifest {} autoSummary back (events 20)
      .backedge env { caller with reads := [] }).next current following) :
    following = { current with locals := [word (bits + 1)] } ∧
      timelineFormulaHolds (semantics env) (.ordinary (.inl (.term 20)))
        { loop := current, entryStorage := caller.entryStorage } := by
  rcases next with ⟨result, trace, finish⟩
  have resultState := traceState env 20 { caller with loop := current, reads := [] } result trace
  rw [resultState] at finish
  have values := backValues env caller.entryStorage current bits locals
  dsimp [context] at values
  change ((directStorageFinalLocals? _ _ _ _).bind _) = some following at finish
  rw [values] at finish
  have shape : { current with locals := [word (bits + 1)] } = following := Option.some.inj finish
  refine ⟨shape.symm, ?_⟩
  cases trace with
  | cons event rest =>
    cases event with
    | assume evaluated holds => exact ⟨_, evaluated, holds⟩

theorem doneNext (env : Env) (caller : StorageActionState) (current following : LoopState)
    (bits : BitVec 256) (locals : current.locals = [word bits])
    (next : (adapterLoopEdge emittedManifest {} autoSummary done (events 21)
      .successor env { caller with reads := [] }).next current following) : following = current := by
  rcases next with ⟨result, trace, finish⟩
  rw [traceState env 21 { caller with loop := current, reads := [] } result trace] at finish
  have values := doneValues env caller.entryStorage current bits locals
  dsimp [context] at values
  change ((directStorageFinalLocals? _ _ _ _).bind _) = some following at finish
  rw [values] at finish
  have shape : { current with locals := [word bits] } = following := Option.some.inj finish
  rw [← locals] at shape
  exact shape.symm

#print axioms bound
#print axioms edgeSafety
#print axioms backNext
#print axioms doneNext

theorem stepBound (bits limit : BitVec 256) (guard : bits.toNat < limit.toNat) :
    (bits + 1).toNat ≤ limit.toNat := by
  have bound := limit.isLt
  rw [BitVec.toNat_add]
  change (bits.toNat + 1) % 2 ^ 256 ≤ limit.toNat
  rw [Nat.mod_eq_of_lt (by omega)]
  omega

theorem postFromInvariant (env : Env) (entry : StorageState) (current : LoopState)
    (valid : directFormulasHold (context env) entry current autoSummary.invariants) :
    directFormulasHold (context env) entry current autoSummary.post := by
  intro formula member
  simp only [autoSummary, emittedLoopSummary_1, List.mem_singleton] at member
  subst formula
  exact valid (.term 9) (List.mem_singleton_self _)

theorem proof (env : Env) (caller : StorageActionState) : (vc env caller).InductionObligations where
  base := by
    intro current initial
    rcases typedContext _ initial.1.1 with ⟨limit, lookup⟩
    have locals : current.locals = [word 0] := by
      have initialized := initial.2.2.1
      change some [word 0] = some current.locals at initialized
      exact (Option.some.inj initialized).symm
    refine ⟨initial.1, initial.2.1, ?_⟩
    exact (invariantAt _ _ _ 0 limit locals lookup).mpr (Nat.zero_le _)
  backedge := by
    intro edge current following member target invariant path safe next
    simp only [vc, List.mem_cons, List.not_mem_nil, or_false] at member
    rcases member with rfl | rfl
    · rcases typedState _ invariant.2.1 with ⟨bits, locals⟩
      rcases typedContext _ invariant.1.1 with ⟨limit, lookup⟩
      have shape := backNext (bindStorageReads env caller.reads) caller current following bits locals next
      have guard := guardAt _ _ _ bits limit locals lookup shape.2
      refine ⟨invariant.1, ?_, ?_⟩
      · rw [shape.1]
        rfl
      · exact (invariantAt _ _ _ (bits + 1) limit (by rw [shape.1]) lookup).mpr (stepBound bits limit guard)
    · cases target
  edgeSafety := by
    intro edge current member invariant path
    simp only [vc, List.mem_cons, List.not_mem_nil, or_false] at member
    rcases member with rfl | rfl
    · exact edgeSafety _ caller back 20 (Or.inl rfl) current invariant.2.1
    · exact edgeSafety _ caller done 21 (Or.inr rfl) current invariant.2.1
  exit := by
    intro edge current following member target invariant path safe next
    simp only [vc, List.mem_cons, List.not_mem_nil, or_false] at member
    rcases member with rfl | rfl
    · exact (target rfl).elim
    · rcases typedState _ invariant.2.1 with ⟨bits, locals⟩
      rw [doneNext (bindStorageReads env caller.reads) caller current following bits locals next]
      exact postFromInvariant _ _ _ invariant.2.2

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
        rcases member with rfl | rfl
        · rw [(backNext _ caller current following bits locals next).1]
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

def supplied : AdapterLoopProof emittedCallManifest emittedLoopVcSha256_1 autoSummary where
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
      simp only [autoSummary, emittedLoopSummary_1, List.mem_singleton] at member
      subst slot
      exact ⟨word 0, rfl, rfl⟩
    constructor
    · intro fact member
      simp only [autoSummary, emittedLoopSummary_1, List.mem_singleton] at member
      subst fact
      change (0 : Nat) ≤ 2
      decide
    constructor
    · refine ⟨[.integer (word 0)], rfl, ?_⟩
      intro formula member
      change formula ∈ [FormulaRef.term 2] at member
      simp only [List.mem_singleton] at member
      subst formula
      change (0 : Nat) ≤ 2
      decide
    · intro edge member
      simp only [autoSummary, emittedLoopSummary_1, List.mem_cons, List.not_mem_nil, or_false] at member
      rcases member with rfl | rfl <;> decide

theorem loopAuthorization : emittedCallLoop_1 := ⟨supplied⟩
#print axioms proof
#print axioms frames
#print axioms loopAuthorization


theorem exitInvariant (env : Env) (caller : StorageActionState) (following : LoopState)
    (run : adapterLoopRelation vc env .successor caller following) :
    (vc env caller).invariant following := by
  rcases run with ⟨initial, _, initialized, current, edge, reached, member, target, path, safe, next⟩
  have invariant := invariant_holds_on_reached_state (vc env caller) (proof env caller).base
    (proof env caller).backedge initialized reached
  rcases typedState _ invariant.2.1 with ⟨bits, locals⟩
  simp only [vc, List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl
  · cases target
  · rw [doneNext _ caller current following bits locals next]
    exact invariant

def certificates := supplied.install
def child : AdapterChild := (adapterLoopCertificate vc frames, .successor)
def bindings : List LoopResultBinding := [{ id := { file_id := 4294967283, pattern_id := 0 }, index := 0, ty := .compilerTypeId 6 }]
def functionContext := directStorageContext emittedManifest Env.empty []
def functionSemantics := adapterSemantics certificates.dependencies functionContext
def functionEvents : List AdapterEvent := [
  .check (.ordinary (.inl (.term 34))), .check (.ordinary (.inr child)),
  .check (.loopResultsReady child bindings), .childResult child bindings,
  .check (.ordinary (.inl (.term 37))) ]
def functionBody : AdapterBody := {
  signature := emittedFunctionSummary_1.callSignature
  requires := emittedFunctionSummary_1.entryFactsHold emittedManifest
  paths := [{ events := functionEvents, returns := .direct .success [.term 35] }]
  semantics := fun _ => functionSemantics.withCallChecks }
theorem functionBound : emittedCallManifest.function certificates.dependencies
    emittedFunctionVcSha256_1 emittedFunctionSummary_1 = .ok functionBody := by rfl

def functionEntry (bits : BitVec 256) (store : StorageState) : StorageActionState := {
  loop := { locals := [], storage := store }, entryStorage := store
  reads := [{ id := { file_id := 0, pattern_id := 0 }, value := .integer (word bits) }] }
def functionEnv (bits : BitVec 256) (store : StorageState) :=
  functionContext.withStoreViews (bindStorageReads Env.empty (functionEntry bits store).reads)
    (functionEntry bits store)

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

theorem requiresBound (bits : BitVec 256) (store : StorageState)
    (required : functionBody.requires { arguments := [.integer (word bits)], storage := store }) :
    bits.toNat ≤ 2 := by
  have holds := required (.term 2) (List.mem_singleton_self _)
  exact holds

theorem childReady (bits : BitVec 256) (store : StorageState) (bound : bits.toNat ≤ 2) :
    timelineFormulaHolds functionSemantics.withCallChecks (.ordinary (.inr child)) (functionEntry bits store) := by
  refine ⟨_, rfl, functionEnv bits store, rfl, ?_⟩
  refine ⟨{ locals := [word 0], storage := store }, ?_, rfl, rfl, rfl, rfl⟩
  constructor
  · intro slot member
    simp only [autoSummary, emittedLoopSummary_1, List.mem_singleton] at member
    subst slot
    exact ⟨word bits, rfl, rfl⟩
  constructor
  · intro fact member
    simp only [autoSummary, emittedLoopSummary_1, List.mem_singleton] at member
    subst fact
    exact bound
  constructor
  · refine ⟨[.integer (word bits)], rfl, ?_⟩
    intro formula member
    change formula ∈ [FormulaRef.term 2] at member
    simp only [List.mem_singleton] at member
    subst formula
    exact bound
  · intro edge member
    simp only [autoSummary, emittedLoopSummary_1, List.mem_cons, List.not_mem_nil, or_false] at member
    rcases member with rfl | rfl <;> simp [functionEntry, timelineReadIds]

theorem childValues (bits : BitVec 256) (store : StorageState) (following : LoopState)
    (related : functionSemantics.childRelation child (functionEntry bits store) following) :
    ∃ result, following.locals = [word result] ∧ result.toNat ≤ bits.toNat := by
  rcases related with ⟨env, evaluated, run⟩
  have same : env = functionEnv bits store := (Option.some.inj evaluated).symm
  subst env
  have invariant := exitInvariant (functionEnv bits store) (functionEntry bits store) following run
  rcases typedState _ invariant.2.1 with ⟨result, locals⟩
  refine ⟨result, locals, ?_⟩
  exact (invariantAt _ store following result bits locals rfl).mp invariant.2.2

theorem childResults (bits : BitVec 256) (store : StorageState) :
    loopResultsReady functionSemantics child bindings (functionEntry bits store) := by
  refine ⟨?_, ?_⟩
  · change ([({ file_id := 0, pattern_id := 0 } : FreeVarId), { file_id := 4294967283, pattern_id := 0 }]).Nodup
    decide
  intro following related
  rcases childValues bits store following related with ⟨result, locals, _⟩
  simp [loopResultBindings?, bindings, locals, word, storageValueMatchesTy]

#print axioms functionBound
#print axioms typedInput
#print axioms childReady
#print axioms childValues

def returnedState (bits result : BitVec 256) (store : StorageState) (following : LoopState) :
    StorageActionState := {
  loop := { locals := [], storage := following.storage, returned := following.returned }
  entryStorage := store
  reads := [{ id := { file_id := 0, pattern_id := 0 }, value := .integer (word bits) },
    { id := { file_id := 4294967283, pattern_id := 0 }, value := .integer (word result) }] }

theorem resumedValues (bits : BitVec 256) (store : StorageState) (following : LoopState)
    (resumed : StorageActionState)
    (related : functionSemantics.childRelation child (functionEntry bits store) following)
    (resume : resumeLoop? bindings (functionEntry bits store) following = some resumed) :
    ∃ result, resumed = returnedState bits result store following ∧ result.toNat ≤ bits.toNat := by
  rcases childValues bits store following related with ⟨result, locals, below⟩
  refine ⟨result, ?_, below⟩
  have evaluated : resumeLoop? bindings (functionEntry bits store) following =
      some (returnedState bits result store following) := by
    simp [resumeLoop?, loopResultBindings?, bindings, functionEntry, locals, word,
      storageValueMatchesTy, returnedState]
  exact (Option.some.inj (resume.symm.trans evaluated))

#print axioms resumedValues

theorem entryCheck (bits : BitVec 256) (store : StorageState) (bound : bits.toNat ≤ 2) :
    timelineFormulaHolds functionSemantics.withCallChecks (.ordinary (.inl (.term 34)))
      (functionEntry bits store) := ⟨bits.toNat ≤ 2, rfl, bound⟩
theorem returnedCheck (bits result : BitVec 256) (store : StorageState) (following : LoopState)
    (below : result.toNat ≤ bits.toNat) :
    timelineFormulaHolds functionSemantics.withCallChecks (.ordinary (.inl (.term 37)))
      (returnedState bits result store following) := by
  refine ⟨result.toNat ≤ bits.toNat, ?_, below⟩
  change functionContext.evalFormula? (.term 37) (returnedState bits result store following) = _
  simp only [functionContext, returnedState, directStorageContext,
    StorageFormulaContext.evalFormula?, StorageFormulaContext.stateEnv?, bindState?,
    bindStorageReads, List.foldl, Option.bind_eq_bind, Option.some_bind,
    Option.pure_def, StorageFormulaContext.formulaStorageReady,
    StorageFormulaContext.termStorageReady, ora_loop_data,
    StorageFormulaContext.withStoreViews]
  ora_loop_value_reduce
  simp only [word, lookupFreeBinding, ora_loop_data, if_true, if_false, Option.some_bind, reduceCtorEq]
  have different : ({ file_id := 4294967283, pattern_id := 0 } : FreeVarId) ≠
      { file_id := 0, pattern_id := 0 } := by
    intro equal
    have impossible := congrArg FreeVarId.file_id equal
    contradiction
  simp only [different, if_false, Bool.not_true, Bool.false_eq_true]
  change some (result.toNat ≤ bits.toNat) = some True
  simp only [below]

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

theorem functionSafe (bits : BitVec 256) (store : StorageState) (bound : bits.toNat ≤ 2) :
    TimelineSafe functionSemantics.withCallChecks functionEvents (functionEntry bits store) := by
  unfold functionEvents
  refine checkSafe _ _ _ _ (entryCheck bits store bound) ?_
  refine checkSafe _ _ _ _ (childReady bits store bound) ?_
  refine checkSafe _ _ _ _ ⟨_, rfl, childResults bits store⟩ ?_
  refine childResultSafe _ _ _ _ _ ?_
  intro following resumed related resume
  rcases resumedValues bits store following resumed related resume with ⟨output, equal, below⟩
  subst resumed
  exact checkSafe _ _ _ _ (returnedCheck bits output store following below) (emptySafe _ _)

#print axioms functionSafe


theorem checkTrace (sem : TimelineSemantics F L V C) (formula : F)
    (tail : List (EventOf F L V C)) (initial : StorageActionState) (result : TimelineResult)
    (trace : TimelineExec sem (.check formula :: tail) initial result) :
    ∃ following, TimelineExec sem tail initial following ∧ result.state = following.state := by
  cases trace with
  | cons first rest => cases first; exact ⟨_, rest, rfl⟩

theorem completed (bits : BitVec 256) (store : StorageState) (result : TimelineResult)
    (trace : TimelineExec functionSemantics.withCallChecks functionEvents (functionEntry bits store) result) :
    ∃ output following, result.state = returnedState bits output store following ∧ output.toNat ≤ bits.toNat := by
  rcases checkTrace _ _ _ _ _ trace with ⟨r1, t1, e1⟩
  rcases checkTrace _ _ _ _ _ t1 with ⟨r2, t2, e2⟩
  rcases checkTrace _ _ _ _ _ t2 with ⟨r3, t3, e3⟩
  cases t3 with
  | cons step rest =>
    cases step with
    | childResult related resume =>
      rcases resumedValues bits store _ _ related resume with ⟨output, equal, below⟩
      rcases checkTrace _ _ _ _ _ rest with ⟨final, trace, finalEq⟩
      cases trace
      exact ⟨output, _, e1.trans (e2.trans (e3.trans (finalEq.trans equal))), below⟩

theorem entryTyped (input : CallInput) (bits : BitVec 256)
    (typed : functionBody.signature.inputTyped input = true)
    (arguments : input.arguments = [.integer (word bits)]) :
    functionBody.signature.entry? input = some (functionEntry bits input.storage) := by
  simp only [CallSignature.entry?, typed, if_true]
  rw [arguments]
  rfl

theorem inputBound (input : CallInput) (bits : BitVec 256)
    (arguments : input.arguments = [.integer (word bits)])
    (required : functionBody.requires input) : bits.toNat ≤ 2 := by
  cases input with
  | mk values store =>
    dsimp at arguments
    subst values
    exact requiresBound bits store required

theorem safetyCertificate : ∀ input, functionBody.signature.inputTyped input = true →
    functionBody.requires input → functionBody.safe input := by
  intro input typed required initial entry path member
  rcases typedInput input typed with ⟨bits, arguments⟩
  have equal := Option.some.inj ((entryTyped input bits typed arguments).symm.trans entry)
  subst initial
  simp only [functionBody, List.mem_singleton] at member
  subst path
  exact functionSafe bits input.storage (inputBound input bits arguments required)

#print axioms safetyCertificate

theorem callsCertificate : ∀ input, functionBody.signature.inputTyped input = true →
    functionBody.requires input → functionBody.callsSafe input := by
  intro input typed required initial entry path member before invocation after equal
  simp only [functionBody, List.mem_singleton] at member
  subst path
  have contains : TimelineEvent.call invocation ∈ before ++ .call invocation :: after := by simp
  rw [← equal] at contains
  simp [functionEvents] at contains

#print axioms callsCertificate

theorem loopResultsCertificate : ∀ input, functionBody.signature.inputTyped input = true →
    functionBody.requires input → functionBody.loopResultsSafe input := by
  intro input typed required initial entry path member before dependency results after equal result trace
  rcases typedInput input typed with ⟨bits, arguments⟩
  have same := Option.some.inj ((entryTyped input bits typed arguments).symm.trans entry)
  subst initial
  simp only [functionBody, List.mem_singleton] at member
  subst path
  cases before with
  | nil => cases equal
  | cons a tail => cases tail with
    | nil => cases equal
    | cons b tail => cases tail with
      | nil => cases equal
      | cons c tail => cases tail with
        | cons d tail =>
          cases tail with
          | nil => cases equal
          | cons e tail => simp [functionEvents] at equal
        | nil =>
          simp only [functionEvents, List.cons_append, List.nil_append,
            List.cons.injEq, TimelineEvent.childResult.injEq] at equal
          rcases equal with ⟨rfl, rfl, rfl, ⟨rfl, rfl⟩, rfl⟩
          rcases checkTrace _ _ _ _ _ trace with ⟨r1, t1, e1⟩
          rcases checkTrace _ _ _ _ _ t1 with ⟨r2, t2, e2⟩
          rcases checkTrace _ _ _ _ _ t2 with ⟨r3, t3, e3⟩
          cases t3
          rw [e1, e2, e3]
          exact childResults bits input.storage

#print axioms loopResultsCertificate

theorem returnedValue (bits result : BitVec 256) (store : StorageState) (following : LoopState) :
    functionContext.evalValue? (.term 35) (returnedState bits result store following) =
      some (.integer (word result)) := by
  let state := returnedState bits result store following
  let env := functionContext.withStoreViews (bindStorageReads Env.empty state.reads) state
  apply StorageFormulaContext.evalValue_free_variable functionContext state env 35
    { id := { file_id := 4294967283, pattern_id := 0 }, name := "i", ty := some (.compilerTypeId 6) }
  · rfl
  · exact emitted_term_35_lookup
  · simp only [env, StorageFormulaContext.withStoreViews_lookupFree, state, returnedState,
      bindStorageReads, List.foldl, Env.setFree, Env.lookupFree, lookupFreeBinding,
      beq_self_eq_true, if_true]

theorem returnsCertificate : ∀ input initial path result,
    functionBody.signature.inputTyped input = true → functionBody.requires input →
    functionBody.signature.entry? input = some initial → path ∈ functionBody.paths →
    TimelineExec (functionBody.semantics input) path.events initial result →
    ∃ returned, callEvaluateReturn? (functionBody.semantics input).storage.evalValue path.returns
      result.state = some returned ∧ functionBody.signature.outputTyped {
        toCallReturn := returned, storage := result.state.loop.storage } = true := by
  intro input initial path result typed required entry member trace
  rcases typedInput input typed with ⟨bits, arguments⟩
  have equal := Option.some.inj ((entryTyped input bits typed arguments).symm.trans entry)
  subst initial
  simp only [functionBody, List.mem_singleton] at member
  subst path
  rcases completed bits input.storage result trace with ⟨output, following, equal, below⟩
  rw [equal]
  refine ⟨{ outcome := .success, payload := [.integer (word output)] }, ?_, rfl⟩
  change callEvaluateReturn? functionContext.evalValue? (.direct .success [.term 35])
    (returnedState bits output input.storage following) = _
  simp only [callEvaluateReturn?, callEvaluateValues?, List.mapM_cons, returnedValue, List.mapM_nil,
    Option.bind_eq_bind, Option.some_bind, Option.pure_def, Option.map_some']

theorem functionCertificate : functionBody.Certificate := {
  safety := safetyCertificate, calls := callsCertificate,
  loopResults := loopResultsCertificate, returns := returnsCertificate }
def installed := certificates.addFunction emittedFunctionVcSha256_1 emittedFunctionSummary_1
  functionBody functionBound functionCertificate
theorem custody : emittedCallCustody_1 := ⟨installed, rfl⟩

#print axioms functionCertificate
#print axioms custody


theorem returnedEvaluation (bits result : BitVec 256) (store : StorageState) (following : LoopState) :
    callEvaluateReturn? functionContext.evalValue? (.direct .success [.term 35])
      (returnedState bits result store following) =
        some { outcome := .success, payload := [.integer (word result)] } := by
  simp only [callEvaluateReturn?, callEvaluateValues?, List.mapM_cons, returnedValue, List.mapM_nil,
    Option.bind_eq_bind, Option.some_bind, Option.pure_def, Option.map_some']

theorem functionJoint (bits : BitVec 256) (store : StorageState) (output : CallOutput)
    (execution : functionBody.execution { arguments := [.integer (word bits)], storage := store } output) :
    ∃ result, output.outcome = .success ∧ output.payload = [.integer (word result)] ∧
      result.toNat ≤ bits.toNat := by
  rcases execution with ⟨initial, path, result, returned, entry, member, trace, evaluated, rfl⟩
  have selected : functionBody.signature.entry? { arguments := [.integer (word bits)], storage := store } =
      some (functionEntry bits store) := rfl
  have equal := Option.some.inj (selected.symm.trans entry)
  subst initial
  simp only [functionBody, List.mem_singleton] at member
  subst path
  rcases completed bits store result trace with ⟨value, following, state, below⟩
  rw [state] at evaluated
  have payload := Option.some.inj ((returnedEvaluation bits value store following).symm.trans evaluated)
  subst returned
  exact ⟨value, rfl, rfl, below⟩

def invocation : CallInvocation FormulaRef := {
  callee := adapterKey 1 emittedFunctionVcSha256_1, arguments := [.term 41]
  outcome := .success, results := [{ file_id := 4294967267, pattern_id := 0 }] }
def callerSemantics := adapterSemantics installed.dependencies functionContext
def callerEvents : List AdapterEvent := [
  .check (.ordinary (.inl (.term 40))), .check (.ready invocation),
  .call invocation, .check (.ordinary (.inl (.term 44))) ]
def callerBody : AdapterBody := {
  signature := emittedFunctionSummary_2.callSignature
  requires := emittedFunctionSummary_2.entryFactsHold emittedManifest
  paths := [{ events := callerEvents, returns := .direct .success [.term 45] }]
  semantics := fun _ => callerSemantics.withCallChecks }
theorem callerBound : emittedCallManifest.function installed.dependencies
    emittedFunctionVcSha256_2 emittedFunctionSummary_2 = .ok callerBody := by rfl

def callerEntry (bits : BitVec 256) (store : StorageState) : StorageActionState := {
  loop := { locals := [], storage := store }, entryStorage := store
  reads := [{ id := { file_id := 0, pattern_id := 3 }, value := .integer (word bits) }] }

theorem resolved : callerSemantics.calls invocation.callee =
    some (functionBody.certified functionCertificate) := by rfl

theorem invocationReady (bits : BitVec 256) (store : StorageState) (bound : bits.toNat ≤ 2) :
    invocation.ready callerSemantics.calls callerSemantics.storage.evalValue (callerEntry bits store) := by
  refine ⟨functionBody.certified functionCertificate, [.integer (word bits)], [.compilerTypeId 6],
    resolved, rfl, rfl, ?_, ?_, rfl, rfl⟩
  · intro formula member
    change formula ∈ [FormulaRef.term 2] at member
    simp only [List.mem_singleton] at member
    subst formula
    exact bound
  · change ([({ file_id := 0, pattern_id := 3 } : FreeVarId), { file_id := 4294967267, pattern_id := 0 }]).Nodup
    decide

#print axioms functionJoint
#print axioms callerBound
#print axioms invocationReady


def callerReturned (bits result : BitVec 256) (store finalStore : StorageState) :
    StorageActionState := {
  loop := { locals := [], storage := finalStore }, entryStorage := store
  reads := [{ id := { file_id := 0, pattern_id := 3 }, value := .integer (word bits) },
    { id := { file_id := 4294967267, pattern_id := 0 }, value := .integer (word result) }] }

theorem invocationResult (bits : BitVec 256) (store : StorageState) (result : TimelineResult)
    (step : TimelineEventExec callerSemantics.withCallChecks (.call invocation) (callerEntry bits store) result) :
    ∃ output finalStore, result.state = callerReturned bits output store finalStore ∧ output.toNat ≤ bits.toNat := by
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
    rcases functionJoint bits store returned related with ⟨output, _, payload, below⟩
    refine ⟨output, returned.storage, ?_, below⟩
    dsimp [CallInvocation.resume, invocation, callerEntry, callerReturned]
    rw [payload]
    rfl

theorem callerEntryCheck (bits : BitVec 256) (store : StorageState) (bound : bits.toNat ≤ 2) :
    timelineFormulaHolds callerSemantics.withCallChecks (.ordinary (.inl (.term 40)))
      (callerEntry bits store) := ⟨bits.toNat ≤ 2, rfl, bound⟩

theorem callerReturnedCheck (bits result : BitVec 256) (store finalStore : StorageState)
    (below : result.toNat ≤ bits.toNat) :
    timelineFormulaHolds callerSemantics.withCallChecks (.ordinary (.inl (.term 44)))
      (callerReturned bits result store finalStore) := by
  refine ⟨result.toNat ≤ bits.toNat, ?_, below⟩
  change functionContext.evalFormula? (.term 44) (callerReturned bits result store finalStore) = _
  simp only [functionContext, callerReturned, directStorageContext,
    StorageFormulaContext.evalFormula?, StorageFormulaContext.stateEnv?, bindState?,
    bindStorageReads, List.foldl, Option.bind_eq_bind, Option.some_bind,
    Option.pure_def, StorageFormulaContext.formulaStorageReady,
    StorageFormulaContext.termStorageReady, ora_loop_data, StorageFormulaContext.withStoreViews]
  ora_loop_value_reduce
  simp only [word, lookupFreeBinding, ora_loop_data, if_true, if_false, Option.some_bind]
  have different : ({ file_id := 4294967267, pattern_id := 0 } : FreeVarId) ≠
      { file_id := 0, pattern_id := 3 } := by
    intro equal
    have impossible := congrArg FreeVarId.file_id equal
    contradiction
  simp only [different, if_false, Bool.not_true, Bool.false_eq_true]
  change some (result.toNat ≤ bits.toNat) = some True
  simp only [below]

theorem callerSafe (bits : BitVec 256) (store : StorageState) (bound : bits.toNat ≤ 2) :
    TimelineSafe callerSemantics.withCallChecks callerEvents (callerEntry bits store) := by
  unfold callerEvents
  refine checkSafe _ _ _ _ (callerEntryCheck bits store bound) ?_
  refine checkSafe _ _ _ _ ⟨_, rfl, invocationReady bits store bound⟩ ?_
  apply TimelineSafe.cons
  · intro formula equal
    cases equal
  · intro result step
    rcases invocationResult bits store result step with ⟨output, finalStore, equal, below⟩
    rw [equal]
    exact checkSafe _ _ _ _ (callerReturnedCheck bits output store finalStore below) (emptySafe _ _)

#print axioms invocationResult
#print axioms callerReturnedCheck
#print axioms callerSafe


theorem callerCompleted (bits : BitVec 256) (store : StorageState) (result : TimelineResult)
    (trace : TimelineExec callerSemantics.withCallChecks callerEvents (callerEntry bits store) result) :
    ∃ output finalStore, result.state = callerReturned bits output store finalStore ∧ output.toNat ≤ bits.toNat := by
  rcases checkTrace _ _ _ _ _ trace with ⟨r1, t1, e1⟩
  rcases checkTrace _ _ _ _ _ t1 with ⟨r2, t2, e2⟩
  cases t2 with
  | cons step rest =>
    rcases invocationResult bits store _ step with ⟨output, finalStore, equal, below⟩
    rcases checkTrace _ _ _ _ _ rest with ⟨final, trace, finalEq⟩
    cases trace
    exact ⟨output, finalStore, e1.trans (e2.trans (finalEq.trans equal)), below⟩

theorem callerTyped (input : CallInput) (typed : callerBody.signature.inputTyped input = true) :
    ∃ bits, input.arguments = [.integer (word bits)] := typedInput input typed

theorem callerEntryTyped (input : CallInput) (bits : BitVec 256)
    (typed : callerBody.signature.inputTyped input = true)
    (arguments : input.arguments = [.integer (word bits)]) :
    callerBody.signature.entry? input = some (callerEntry bits input.storage) := by
  simp only [CallSignature.entry?, typed, if_true]
  rw [arguments]
  rfl

theorem callerInputBound (input : CallInput) (bits : BitVec 256)
    (arguments : input.arguments = [.integer (word bits)])
    (required : callerBody.requires input) : bits.toNat ≤ 2 := by
  cases input with
  | mk values store =>
    dsimp at arguments
    subst values
    exact required (.term 31) (List.mem_singleton_self _)

theorem callerReturnedValue (bits result : BitVec 256) (store finalStore : StorageState) :
    functionContext.evalValue? (.term 45) (callerReturned bits result store finalStore) =
      some (.integer (word result)) := by
  let state := callerReturned bits result store finalStore
  let env := functionContext.withStoreViews (bindStorageReads Env.empty state.reads) state
  apply StorageFormulaContext.evalValue_free_variable functionContext state env 45
    { id := { file_id := 4294967267, pattern_id := 0 }, name := "call_result", ty := some (.compilerTypeId 6) }
  · rfl
  · exact emitted_term_45_lookup
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
    exact callerSafe bits input.storage (callerInputBound input bits arguments required)
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
      | nil => cases equal
      | cons b tail => cases tail with
        | cons c tail => cases tail with
          | nil => cases equal
          | cons d tail => simp [callerEvents] at equal
        | nil =>
          simp only [callerEvents, List.cons_append, List.nil_append,
            List.cons.injEq, TimelineEvent.call.injEq] at equal
          rcases equal with ⟨rfl, rfl, rfl, rfl⟩
          rcases checkTrace _ _ _ _ _ trace with ⟨r1, t1, e1⟩
          rcases checkTrace _ _ _ _ _ t1 with ⟨r2, t2, e2⟩
          cases t2
          rw [e1, e2]
          exact invocationReady bits input.storage (callerInputBound input bits arguments required)
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
    rcases callerCompleted bits input.storage result trace with ⟨output, finalStore, equal, below⟩
    rw [equal]
    refine ⟨{ outcome := .success, payload := [.integer (word output)] }, ?_, rfl⟩
    change callEvaluateReturn? functionContext.evalValue? (.direct .success [.term 45])
      (callerReturned bits output input.storage finalStore) = _
    simp only [callEvaluateReturn?, callEvaluateValues?, List.mapM_cons, callerReturnedValue, List.mapM_nil,
      Option.bind_eq_bind, Option.some_bind, Option.pure_def, Option.map_some']

def complete := installed.addFunction emittedFunctionVcSha256_2 emittedFunctionSummary_2
  callerBody callerBound callerCertificate
theorem callerCustody : emittedCallCustody_2 := ⟨complete, rfl⟩

#print axioms callerCompleted
#print axioms callerCertificate
#print axioms callerCustody
end G9cSuppliedRepeat

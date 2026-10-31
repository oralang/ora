namespace Ora.UserProof.DirectStoragePreheader

open Ora.Obligation
open Ora.Loop
open Ora.Generated.Obligations.Source_direct_storage_preheader_certificate_ff499ad866a88c04

attribute [local simp] emitted_manifest_wf emitted_terms_topological emitted_terms_length
attribute [local simp]
  emitted_term_0_lookup emitted_term_1_lookup emitted_term_2_lookup
  emitted_term_3_lookup emitted_term_4_lookup emitted_term_5_lookup
  emitted_term_6_lookup emitted_term_7_lookup emitted_term_8_lookup
  emitted_term_9_lookup emitted_term_10_lookup emitted_term_11_lookup
  emitted_term_12_lookup emitted_term_13_lookup emitted_term_14_lookup
  emitted_term_15_lookup emitted_term_16_lookup emitted_term_17_lookup

def zero : Value :=
  .integer (Ora.Integer.Value.ofNat (.unsigned .w256) 0)

def witness : Env := {
  placeValue := fun _ => zero
}

theorem evalValueIntegerLiteral
    (context : StorageFormulaContext)
    (state : StorageActionState)
    (bound : Env)
    (termId : TermId)
    (literal : IntegerLiteralTerm)
    (value : Ora.Integer.Value)
    (hBound : context.stateEnv? state = some bound)
    (hTerm : context.manifest.terms[termId]? = some (.intLit literal))
    (hValue : literal.asInteger? = some value) :
    context.evalValue? (.term termId) state = some (.integer value) := by
  unfold StorageFormulaContext.evalValue?
    StorageFormulaContext.formulaStorageReady
    StorageFormulaContext.termStorageReady
  simp only [hBound, Option.bind_eq_bind, Option.some_bind, hTerm,
    Bool.not_true, Bool.false_eq_true, if_false]
  unfold formulaValue? denoteValue?
  simp only [hTerm, hValue, Option.map_some', Option.pure_def,
    Option.some_bind]

theorem preheaderReadWriteSamePreserves
    (manifest : Manifest)
    (base : Env)
    (contextVariables : List VariableRow)
    (root : String)
    (valueTy : TyRef)
    (readId : FreeVarId)
    (termId : TermId)
    (reference : FreeVarRef)
    (result : TimelineResult)
    (hRoot : root.isEmpty = false)
    (hValueTy : (valueTy.isBool || valueTy.integerShape?.isSome) = true)
    (hReference : reference.id = readId)
    (hTerm : manifest.terms[termId]? = some (.variable (.free reference)))
    (hExec : TimelineExec
      (directStoragePreheaderSemantics
        (directStoragePreheaderContext manifest base contextVariables))
      [.storage (.read readId { root, valueTy }),
       .storage (.write { root, valueTy } (.term termId))]
      (directStoragePreheaderInitialState base) result) :
    result.state.loop.storage =
      (directStoragePreheaderInitialState base).loop.storage := by
  cases hExec with
  | cons hRead hRest =>
    cases hRest with
    | cons hWrite hEnd =>
      cases hEnd
      cases hRead with
      | @storage _ _ readResult hReadAction =>
        cases hWrite with
        | @storage _ _ writeResult hWriteAction =>
          let context := directStoragePreheaderContext
            manifest base contextVariables
          let initial := directStoragePreheaderInitialState base
          let locationExpr : StorageLocationExpr := { root, valueTy }
          let location : StorageLocation := { root }
          change execStorageAction? context.actionSemantics
            (.read readId locationExpr) initial = some readResult at hReadAction
          have hLocation : context.evalLocation? locationExpr initial =
              some location := by
            unfold StorageFormulaContext.evalLocation? storageTypeSupported
            simp only [locationExpr, hRoot, Bool.false_eq_true, if_false,
              StorageFormulaContext.evalProjections?, Option.pure_def,
              Option.some_bind]
            rw [hValueTy]
            rfl
          unfold execStorageAction? at hReadAction
          simp only [StorageFormulaContext.actionSemantics, hLocation,
            Option.bind_eq_bind, Option.some_bind] at hReadAction
          split at hReadAction
          · simp only [Option.none_bind] at hReadAction
            cases hReadAction
          · simp only [Option.pure_def, Option.some_bind,
              StorageActionState.bindRead?] at hReadAction
            split at hReadAction
            · simp only [Option.none_bind] at hReadAction
              cases hReadAction
            · simp only [Option.some_bind, Option.some.injEq] at hReadAction
              subst readResult
              let afterRead : StorageActionState := {
                loop := initial.loop
                entryStorage := initial.entryStorage
                reads := initial.reads ++ [{
                  id := readId, value := initial.loop.storage location }]
              }
              change execStorageAction? context.actionSemantics
                (.write locationExpr (.term termId)) afterRead =
                  some writeResult at hWriteAction
              have hWriteLocation : context.evalLocation? locationExpr afterRead =
                  some location := by
                unfold StorageFormulaContext.evalLocation? storageTypeSupported
                simp only [locationExpr, hRoot, Bool.false_eq_true, if_false,
                  StorageFormulaContext.evalProjections?, Option.pure_def]
                rw [hValueTy]
                rfl
              let bound := context.withStoreViews
                (context.entryEnv.setFree readId (initial.loop.storage location))
                afterRead
              have hBound : context.stateEnv? afterRead = some bound := by rfl
              have hWriteValue : context.evalValue? (.term termId) afterRead =
                  some (initial.loop.storage location) := by
                apply context.evalValue_free_variable afterRead bound termId reference
                  (initial.loop.storage location) hBound hTerm
                rw [hReference]
                change (context.entryEnv.setFree readId
                  (initial.loop.storage location)).lookupFree readId =
                    some (initial.loop.storage location)
                simp [Env.setFree, Env.lookupFree, lookupFreeBinding]
              unfold execStorageAction? at hWriteAction
              simp only [StorageFormulaContext.actionSemantics, hWriteLocation,
                hWriteValue, Option.bind_eq_bind, Option.some_bind] at hWriteAction
              split at hWriteAction
              · simp only [Option.none_bind] at hWriteAction
                cases hWriteAction
              · simp only [Option.pure_def, Option.some_bind,
                  Option.some.injEq] at hWriteAction
                subst writeResult
                funext observed
                by_cases hObserved : observed = location
                · subst observed
                  simp [TimelineResult.comp, StorageState.set]
                · simp [TimelineResult.comp, StorageState.set, hObserved]

theorem counter_invariant_iff
    (env : Env)
    (state : Ora.Loop.LoopState)
    (hReady : Ora.Loop.stateReady emittedLoopSummary_1.variables state.locals) :
    Ora.Loop.directFormulasHold
        (Ora.Loop.directStorageContext emittedManifest env
          emittedLoopSummary_1.variables)
        (Ora.Loop.storageStateFromEnv env true) state [.term 8] ↔
      match Value.eqProp? (state.storage { root := "counter" })
          ((Ora.Loop.storageStateFromEnv env true) { root := "counter" }) with
      | some proposition => proposition
      | none => False := by
  let context := Ora.Loop.directStorageContext emittedManifest env
    emittedLoopSummary_1.variables
  let entryStorage := Ora.Loop.storageStateFromEnv env true
  let place : PlaceRef := { root := "counter", region := .storage }
  let location : Ora.Loop.StorageLocation := { root := "counter" }
  rcases Ora.Loop.bindState_some_of_stateReady env
      emittedLoopSummary_1.variables state.locals hReady with
    ⟨localsBound, hLocalsBound⟩
  let actionState : Ora.Loop.StorageActionState := {
    loop := state
    entryStorage
  }
  let bound := context.withStoreViews localsBound actionState
  have hBound : context.stateEnv? actionState = some bound := by
    unfold Ora.Loop.StorageFormulaContext.stateEnv?
    change (Ora.Loop.bindState? env emittedLoopSummary_1.variables state.locals).bind
      (fun locals => some (context.withStoreViews
        (Ora.Loop.bindStorageReads locals []) actionState)) = some bound
    rw [hLocalsBound]
    rfl
  have hCurrentLocation : context.resolvePlace bound.lookupFree place =
      some location := by
    change Ora.Loop.directStorageResolvePlace bound.lookupFree place =
      some location
    simp [place, location, Ora.Loop.directStorageResolvePlace]
    decide
  have hEntryLocation : context.resolvePlace context.entryEnv.lookupFree place =
      some location := by
    change Ora.Loop.directStorageResolvePlace context.entryEnv.lookupFree place =
      some location
    simp [place, location, Ora.Loop.directStorageResolvePlace]
    decide
  have hLocalLocation : context.resolvePlace localsBound.lookupFree place =
      some location := by
    change Ora.Loop.directStorageResolvePlace localsBound.lookupFree place =
      some location
    simp [place, location, Ora.Loop.directStorageResolvePlace]
    decide
  have hCurrentValue : bound.placeValue (.stable place) =
      actionState.loop.storage location := by
    simp [bound, Ora.Loop.StorageFormulaContext.withStoreViews,
      hLocalLocation, Ora.Loop.StorageActionState.currentValue]
  have hEntryValue : bound.placeValue (.entry place) =
      actionState.entryStorage location := by
    simp [bound, Ora.Loop.StorageFormulaContext.withStoreViews,
      hEntryLocation, Ora.Loop.StorageActionState.oldValue]
  have hCharacterization := context.evalFormula_place_eq_old_self
    actionState bound 17 8 5 7 6 (.compilerTypeId 6) place location hBound
    (by change emittedManifest.terms.length = 18; exact emitted_terms_length)
    emitted_term_8_lookup emitted_term_5_lookup emitted_term_7_lookup
    emitted_term_6_lookup hCurrentLocation hEntryLocation hCurrentValue hEntryValue
  constructor
  · intro hInvariant
    exact hCharacterization.mp (hInvariant (.term 8) (by simp))
  · intro hSemantic formula hMember
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hMember
    subst formula
    exact hCharacterization.mpr hSemantic

theorem discharge_q10 : emittedQuery_10 := by
  unfold emittedQuery_10 Ora.Loop.directStorageLoopProofAuthorizationFromAssumptions
  have hSupported : emittedLoopSummary_1.supported emittedManifest = true := by decide
  have hDirect : Ora.Loop.directStorageSummarySupported
      emittedManifest emittedLoopSummary_1 = true := by
    simp only [Ora.Loop.directStorageSummarySupported, emittedLoopSummary_1, hSupported]
    decide
  have hCounterNotEmpty : ("counter" == "") = false := by decide
  have hCounterRootNotEmpty : "counter".isEmpty = false := by decide
  constructor
  · refine ⟨witness, {
      locals := [Ora.Integer.Value.ofNat (.unsigned .w256) 0]
      storage := Ora.Loop.storageStateFromEnv witness true
    }, ?_, ?_, ?_⟩
    · simp [assumptionsDenoteInEnv, assumptionsDenoteInEnv?]
    · simp [Ora.Loop.contextReady, emittedLoopSummary_1]
    · simp only [Ora.Loop.denoteDirectStorageLoopSummary?, hDirect, if_true]
      simp only [Ora.Loop.denoteDirectStorageEdges?,
        Ora.Loop.denoteDirectStorageEdge?, Ora.Loop.directStorageTimeline?,
        Ora.Loop.directStorageEvent?, Ora.Loop.directStorageLocation?,
        Ora.Loop.directStorageProjection?, Ora.Loop.directStorageProjections?,
        Ora.Loop.denoteControlTarget?, emittedLoopSummary_1,
        Ora.Loop.denoteDirectStoragePreheaderInitial?,
        Option.bind_eq_bind, Option.some_bind, Option.pure_def,
        List.isEmpty, List.map, List.append, Bool.not_false, Bool.not_true,
        Bool.false_or, Bool.false_eq_true, hCounterNotEmpty, if_false]
      refine ⟨by simp [Ora.Loop.headerFactsHoldInEnv], ?_⟩
      unfold Ora.Loop.directStoragePreheaderProducesInitial
      let context := Ora.Loop.directStoragePreheaderContext
        emittedManifest witness []
      let initial := Ora.Loop.directStoragePreheaderInitialState witness
      let locationExpr : Ora.Loop.StorageLocationExpr := {
        root := "counter"
        valueTy := .compilerTypeId 6
      }
      let location : Ora.Loop.StorageLocation := { root := "counter" }
      let readId : FreeVarId := { file_id := 4294967293, pattern_id := 4 }
      let readResult : Ora.Loop.StorageActionResult := {
        state := {
          loop := initial.loop
          entryStorage := initial.entryStorage
          reads := [{ id := readId, value := zero }]
        }
      }
      have hRead : Ora.Loop.execStorageAction? context.actionSemantics
          (.read readId locationExpr) initial = some readResult := by
        simp [context, initial, locationExpr, location, readId, readResult,
          Ora.Loop.directStoragePreheaderContext,
          Ora.Loop.directStoragePreheaderInitialState,
          Ora.Loop.StorageFormulaContext.actionSemantics,
          Ora.Loop.StorageFormulaContext.evalLocation?, Ora.Loop.storageTypeSupported,
          Ora.Loop.StorageFormulaContext.evalProjections?,
          Ora.Loop.directStorageResolvePlace, Ora.Loop.execStorageAction?,
          Ora.Loop.StorageActionState.bindRead?, Ora.Loop.storageReadIdBound,
          Ora.Loop.storageValueMatchesTy, Ora.Loop.storageStateFromEnv,
          Ora.Integer.Value.ofNat, hCounterRootNotEmpty, witness, zero]
      let writeResult : Ora.Loop.StorageActionResult := {
        state := {
          loop := { initial.loop with
            storage := initial.loop.storage.set location zero }
          entryStorage := initial.entryStorage
          reads := readResult.state.reads
        }
        writeTrace := [location]
      }
      have hWriteLocation : context.evalLocation? locationExpr readResult.state =
          some location := by
        simp [context, locationExpr, location, readResult,
          Ora.Loop.directStoragePreheaderContext,
          Ora.Loop.StorageFormulaContext.evalLocation?, Ora.Loop.storageTypeSupported,
          Ora.Loop.StorageFormulaContext.evalProjections?,
          Ora.Loop.directStorageResolvePlace, hCounterRootNotEmpty]
      have hWriteValue : context.evalValue? (.term 0) readResult.state = some zero := by
        let bound := context.withStoreViews
          (context.entryEnv.setFree readId zero) readResult.state
        have hBound : context.stateEnv? readResult.state = some bound := by rfl
        apply context.evalValue_free_variable readResult.state bound 0
          { id := readId, name := "counter", ty := some (.compilerTypeId 6) }
          zero hBound emitted_term_0_lookup
        change (context.entryEnv.setFree readId zero).lookupFree readId = some zero
        simp [Env.setFree, Env.lookupFree, lookupFreeBinding]
      have hWriteType : Ora.Loop.storageValueMatchesTy zero locationExpr.valueTy = true := by
        decide
      have hWrite : Ora.Loop.execStorageAction? context.actionSemantics
          (.write locationExpr (.term 0)) readResult.state = some writeResult := by
        simp only [Ora.Loop.execStorageAction?,
          Ora.Loop.StorageFormulaContext.actionSemantics,
          hWriteLocation, hWriteValue, hWriteType,
          Option.bind_eq_bind, Option.some_bind, Bool.true_eq_false, if_false,
          Option.pure_def, Option.some.injEq]
        rfl
      refine ⟨_, .cons (.storage hRead) (.cons (.storage hWrite) (.nil _)), ?_, ?_, ?_⟩
      · intro before formula after hEvents
        change [Ora.Loop.TimelineEvent.storage (.read readId locationExpr),
          Ora.Loop.TimelineEvent.storage (.write locationExpr (.term 0))] =
            before ++ Ora.Loop.TimelineEvent.check formula :: after at hEvents
        have hCheck : (Ora.Loop.TimelineEvent.check formula :
            Ora.Loop.DirectStorageEvent) ∈
            [Ora.Loop.TimelineEvent.storage (.read readId locationExpr),
              Ora.Loop.TimelineEvent.storage (.write locationExpr (.term 0))] := by
          rw [hEvents]
          simp
        simp at hCheck
      · change Ora.Loop.directInitialValuesAfterPreheader? context
          writeResult.state [.term 1] =
            some [Ora.Integer.Value.ofNat (.unsigned .w256) 0]
        unfold Ora.Loop.directInitialValuesAfterPreheader?
        let bound := context.withStoreViews
          (Ora.Loop.bindStorageReads context.entryEnv writeResult.state.reads)
          writeResult.state
        have hBound : context.stateEnv? writeResult.state = some bound := by rfl
        have hValue : context.evalValue? (.term 1) writeResult.state = some zero := by
          unfold Ora.Loop.StorageFormulaContext.evalValue?
          rw [hBound]
          simp only [Ora.Loop.StorageFormulaContext.formulaStorageReady,
            Ora.Loop.StorageFormulaContext.termStorageReady,
            emitted_terms_length, emitted_term_1_lookup,
            Bool.not_true, Bool.false_eq_true, if_false,
            Option.bind_eq_bind, Option.some_bind]
          unfold formulaValue? denoteValue?
          simp only [emitted_term_1_lookup, Option.pure_def,
            TyRef.integerShape?, integerShapeForCompilerTypeId_u256,
            Option.some_bind]
          rfl
        rw [hValue]
        rfl
      · change (Ora.Loop.storageStateFromEnv witness true) =
          writeResult.state.loop.storage
        funext observed
        simp [writeResult, readResult, initial, location,
          Ora.Loop.directStoragePreheaderInitialState,
          Ora.Loop.StorageState.set, Ora.Loop.storageStateFromEnv,
          witness, zero]
  · intro env _ _
    simp only [Ora.Loop.denoteDirectStorageLoopSummary?, hDirect, if_true]
    simp only [Ora.Loop.denoteDirectStorageEdges?,
      Ora.Loop.denoteDirectStorageEdge?, Ora.Loop.directStorageTimeline?,
      Ora.Loop.directStorageEvent?, Ora.Loop.directStorageLocation?,
      Ora.Loop.directStorageProjection?, Ora.Loop.directStorageProjections?,
      Ora.Loop.denoteControlTarget?, emittedLoopSummary_1,
      Option.bind_eq_bind, Option.some_bind, Option.pure_def,
      List.isEmpty, List.map, List.append, Bool.not_false, Bool.not_true,
      Bool.false_or, Bool.false_eq_true, hCounterNotEmpty, if_false]
    refine { base := ?_, backedge := ?_, edgeSafety := ?_, exit := ?_ }
    · intro state hInitial
      rcases hInitial with ⟨hHeaderFacts, hInitial⟩
      change Ora.Loop.directStoragePreheaderProducesInitial
        (Ora.Loop.directStoragePreheaderContext emittedManifest env [])
        [Ora.Loop.TimelineEvent.storage (.read
          { file_id := 4294967293, pattern_id := 4 }
          { root := "counter", valueTy := .compilerTypeId 6 }),
         Ora.Loop.TimelineEvent.storage (.write
          { root := "counter", valueTy := .compilerTypeId 6 } (.term 0))]
        (Ora.Loop.directStoragePreheaderInitialState env) [.term 1] state at hInitial
      change Ora.Loop.stateReady emittedLoopSummary_1.variables state.locals ∧
        Ora.Loop.headerFactsHoldInEnv emittedManifest env
          emittedLoopSummary_1.headerFacts ∧ Ora.Loop.directFormulasHold
          (Ora.Loop.directStorageContext emittedManifest env
            emittedLoopSummary_1.variables)
          (Ora.Loop.storageStateFromEnv env true) state [.term 8]
      rcases hInitial with ⟨result, hExec, _, hLocals, hStorage⟩
      have hPreserved := preheaderReadWriteSamePreserves
        emittedManifest env [] "counter" (.compilerTypeId 6)
        { file_id := 4294967293, pattern_id := 4 } 0
        { id := { file_id := 4294967293, pattern_id := 4 },
          name := "counter", ty := some (.compilerTypeId 6) }
        result hCounterRootNotEmpty (by decide) rfl emitted_term_0_lookup hExec
      have hResultStorage : state.storage =
          (Ora.Loop.directStoragePreheaderInitialState env).loop.storage :=
        hStorage.trans hPreserved
      let preContext := Ora.Loop.directStoragePreheaderContext emittedManifest env []
      have hResultLocals : state.locals =
          [Ora.Integer.Value.ofNat (.unsigned .w256) 0] := by
        unfold Ora.Loop.directInitialValuesAfterPreheader? at hLocals
        cases hBound : preContext.stateEnv? result.state with
        | none =>
            simp [preContext, hBound,
              Ora.Loop.StorageFormulaContext.evalValue?] at hLocals
        | some bound =>
            have hValue : preContext.evalValue? (.term 1) result.state =
                some (.integer (Ora.Integer.Value.ofNat (.unsigned .w256) 0)) := by
              apply evalValueIntegerLiteral preContext result.state bound 1
                { value := "0", ty := some (.compilerTypeId 6) }
                (Ora.Integer.Value.ofNat (.unsigned .w256) 0)
                hBound emitted_term_1_lookup
              exact integerLiteral_u256_zero
            rw [hValue] at hLocals
            exact (Option.some.inj hLocals).symm
      rcases state with ⟨locals, storage⟩
      change storage = Ora.Loop.storageStateFromEnv env true at hResultStorage
      change locals = [Ora.Integer.Value.ofNat (.unsigned .w256) 0] at hResultLocals
      subst storage
      subst locals
      constructor
      · simp [Ora.Loop.stateReady, Ora.Loop.valuesMatchVariables,
          emittedLoopSummary_1, TyRef.integerShape?,
          integerShapeForCompilerTypeId_u256, Ora.Integer.Value.ofNat]
      · refine ⟨hHeaderFacts, ?_⟩
        intro formula hFormula
        simp only [List.mem_cons, List.not_mem_nil, or_false] at hFormula
        subst formula
        let context := Ora.Loop.directStorageContext emittedManifest env
          emittedLoopSummary_1.variables
        let actionState : Ora.Loop.StorageActionState := {
          loop := {
            locals := [Ora.Integer.Value.ofNat (.unsigned .w256) 0]
            storage := Ora.Loop.storageStateFromEnv env true
          }
          entryStorage := Ora.Loop.storageStateFromEnv env true
        }
        let localsBound := env.setFree
          { file_id := 4294967294, pattern_id := 0 }
          (.integer (Ora.Integer.Value.ofNat (.unsigned .w256) 0))
        let bound := context.withStoreViews localsBound actionState
        let place : PlaceRef := {
          root := "counter"
          region := .storage
        }
        let location : Ora.Loop.StorageLocation := { root := "counter" }
        have hBound : context.stateEnv? actionState = some bound := by rfl
        have hCurrentLocation : context.resolvePlace bound.lookupFree place =
            some location := by
          change Ora.Loop.directStorageResolvePlace bound.lookupFree place =
            some location
          simp [place, location, Ora.Loop.directStorageResolvePlace,
            hCounterRootNotEmpty]
          decide
        have hEntryLocation : context.resolvePlace context.entryEnv.lookupFree place =
            some location := by
          change Ora.Loop.directStorageResolvePlace context.entryEnv.lookupFree place =
            some location
          simp [place, location, Ora.Loop.directStorageResolvePlace,
            hCounterRootNotEmpty]
          decide
        have hLocalLocation : context.resolvePlace localsBound.lookupFree place =
            some location := by
          change Ora.Loop.directStorageResolvePlace localsBound.lookupFree place =
            some location
          simp [place, location, Ora.Loop.directStorageResolvePlace,
            hCounterRootNotEmpty]
          decide
        have hCurrentValue : bound.placeValue (.stable place) =
            actionState.loop.storage location := by
          simp [bound, Ora.Loop.StorageFormulaContext.withStoreViews,
            hLocalLocation, Ora.Loop.StorageActionState.currentValue]
        have hEntryValue : bound.placeValue (.entry place) =
            actionState.entryStorage location := by
          simp [bound, Ora.Loop.StorageFormulaContext.withStoreViews,
            hEntryLocation, Ora.Loop.StorageActionState.oldValue]
        apply (context.evalFormula_place_eq_old_self actionState bound
          17 8 5 7 6 (.compilerTypeId 6) place location hBound
          (by change emittedManifest.terms.length = 18; exact emitted_terms_length)
          emitted_term_8_lookup
          emitted_term_5_lookup emitted_term_7_lookup emitted_term_6_lookup
          hCurrentLocation hEntryLocation hCurrentValue hEntryValue).mpr
        have hSame : actionState.loop.storage location =
            actionState.entryStorage location := rfl
        rw [hSame]
        generalize hValue : actionState.entryStorage location = value
        cases value with
        | bool value => simp [hValue, Value.eqProp?]
        | integer value =>
            rcases value with ⟨shape, bits⟩
            simp [hValue, Value.eqProp?, Ora.Integer.Value.eqProp?]
    · intro edge current following hMember hTarget hInvariant hPath hSafe hNext
      simp only [List.mem_cons, List.not_mem_nil, or_false] at hMember
      rcases hMember with rfl | rfl
      · dsimp at hNext ⊢
        rcases hInvariant with
          ⟨hInvariantReady, hHeaderFacts, hInvariantFormulas⟩
        rcases hNext with ⟨result, locals, returned, hExec, hLocals, hReturn, rfl⟩
        let variables := emittedLoopSummary_1.variables
        let bindings := emittedLoopSummary_1.edges[0].nextState
        let context := Ora.Loop.directStorageContext emittedManifest env variables
        change Ora.Loop.directStorageFinalLocals? context variables bindings
          result.state = some locals at hLocals
        unfold Ora.Loop.directStorageFinalLocals? at hLocals
        cases hEnv : context.stateEnv? {
          result.state with
          reads := Ora.Loop.storageReadsForBindings bindings result.state.reads
        } with
        | none => simp [hEnv] at hLocals
        | some bound =>
            simp only [hEnv, Option.bind_eq_bind, Option.some_bind] at hLocals
            have hFollowingReady := Ora.Loop.bound_state_in_env_ready
              emittedManifest variables bound bindings locals hLocals
            let location : Ora.Loop.StorageLocation := { root := "counter" }
            have hPreserved : result.state.loop.storage location =
                current.storage location := by
              apply Ora.Loop.TimelineExec.preserves_outside _ _ _ _ _ hExec
              intro hWritten
              cases hExec with
              | cons h0 hTail0 =>
                  cases hTail0 with
                  | cons h1 hTail1 =>
                      cases hTail1 with
                      | cons h2 hTail2 =>
                          cases hTail2
                          cases h0 with
                          | assume =>
                              cases h1 with
                              | @storage action0 state0 read0Result hRead0 =>
                                  cases h2 with
                                  | @storage action1 state1 read1Result hRead1 =>
                                      have hRead0Empty :=
                                        Ora.Loop.execStorageRead_writeTrace_nil
                                          _ _ _ _ _ hRead0
                                      have hRead1Empty :=
                                        Ora.Loop.execStorageRead_writeTrace_nil
                                          _ _ _ _ _ hRead1
                                      simp [Ora.Loop.TimelineResult.comp,
                                        hRead0Empty, hRead1Empty] at hWritten
            have hCurrentSemantic :=
              (counter_invariant_iff env current hInvariantReady).mp
                hInvariantFormulas
            refine ⟨hFollowingReady, hHeaderFacts, ?_⟩
            apply (counter_invariant_iff env
                { locals, storage := result.state.loop.storage }
                hFollowingReady).mpr
            rw [hPreserved]
            exact hCurrentSemantic
      · simp [Ora.Loop.timelineEdge] at hTarget
    · intro edge state hMember _ _
      simp only [List.mem_cons, List.not_mem_nil, or_false] at hMember
      rcases hMember with rfl | rfl
      · dsimp
        intro before formula after hEvents _ _
        have hCheck : .check formula ∈ before ++ .check formula :: after := by simp
        rw [← hEvents] at hCheck
        simp at hCheck
      · dsimp
        intro before formula after hEvents _ _
        have hCheck : .check formula ∈ before ++ .check formula :: after := by simp
        rw [← hEvents] at hCheck
        simp at hCheck
    · intro edge current following hMember hNotBackedge _ _ _ _
      simp only [List.mem_cons, List.not_mem_nil, or_false] at hMember
      rcases hMember with rfl | rfl
      · exact False.elim (hNotBackedge rfl)
      · intro formula hFormula
        simp at hFormula

theorem discharge_q2 : emittedQuery_2 := by
  unfold emittedQuery_2 obligationFollowsFromAssumptions
  constructor
  · exact ⟨Env.empty, by simp [assumptionsDenoteInEnv,
      assumptionsDenoteInEnv?]⟩
  · intro env _
    unfold obligationDenotesInEnv obligationDenotesInEnv?
    simp only [emittedObligationRow_1]
    exact effect_frame_write_covered_denotes emittedManifest env
      [{ root := "counter", region := .storage }]
      [{ root := "counter", region := .storage }] (by decide)

end Ora.UserProof.DirectStoragePreheader

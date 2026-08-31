namespace Ora.UserProof.MappingStoragePreservation

open Ora.Obligation
open Ora.Generated.Obligations.Source_mapping_storage_preservation_certificate_7bad0625ff5820f7

attribute [local simp] emitted_manifest_wf emitted_terms_topological emitted_terms_length
attribute [local ora_loop_manifest] emitted_manifest_wf emitted_terms_topological
attribute [local ora_loop_row] emittedLoopSummary_1
attribute [local simp]
  emitted_term_0_lookup emitted_term_1_lookup emitted_term_2_lookup
  emitted_term_3_lookup emitted_term_4_lookup emitted_term_5_lookup
  emitted_term_6_lookup emitted_term_7_lookup emitted_term_8_lookup
  emitted_term_9_lookup emitted_term_10_lookup emitted_term_11_lookup
  emitted_term_12_lookup emitted_term_13_lookup emitted_term_14_lookup
  emitted_term_15_lookup emitted_term_16_lookup emitted_term_17_lookup
  emitted_term_18_lookup emitted_term_19_lookup emitted_term_20_lookup
  emitted_term_21_lookup emitted_term_22_lookup emitted_term_23_lookup
  emitted_term_24_lookup emitted_term_25_lookup emitted_term_26_lookup
  emitted_term_27_lookup emitted_term_28_lookup emitted_term_29_lookup
  emitted_term_30_lookup emitted_term_31_lookup emitted_term_32_lookup
  emitted_term_33_lookup emitted_term_34_lookup emitted_term_35_lookup
attribute [local ora_loop_data]
  emitted_terms_length emitted_term_0_lookup emitted_term_1_lookup
  emitted_term_2_lookup emitted_term_3_lookup emitted_term_4_lookup
  emitted_term_5_lookup emitted_term_6_lookup emitted_term_7_lookup
  emitted_term_8_lookup emitted_term_9_lookup emitted_term_10_lookup
  emitted_term_11_lookup emitted_term_12_lookup emitted_term_13_lookup
  emitted_term_14_lookup emitted_term_15_lookup emitted_term_16_lookup
  emitted_term_17_lookup emitted_term_18_lookup emitted_term_19_lookup
  emitted_term_20_lookup emitted_term_21_lookup emitted_term_22_lookup
  emitted_term_23_lookup emitted_term_24_lookup emitted_term_25_lookup
  emitted_term_26_lookup emitted_term_27_lookup emitted_term_28_lookup
  emitted_term_29_lookup emitted_term_30_lookup emitted_term_31_lookup
  emitted_term_32_lookup emitted_term_33_lookup emitted_term_34_lookup
  emitted_term_35_lookup

theorem storageReadWritesNone
    (semantics : Ora.Loop.StorageActionSemantics LocationExpr ValueExpr)
    (id : Ora.Loop.StorageReadId)
    (location : LocationExpr)
    (initial : Ora.Loop.StorageActionState)
    (result : Ora.Loop.StorageActionResult)
    (hExec : Ora.Loop.execStorageAction? semantics (.read id location) initial =
      some result) :
    result.writeTrace = [] := by
  simp only [Ora.Loop.execStorageAction?, Option.bind_eq_bind,
    Option.bind_eq_some] at hExec
  rcases hExec with ⟨resolved, _, hExec⟩
  split at hExec
  · simp only [Option.none_bind] at hExec
    cases hExec
  · simp only [Option.pure_def, Option.some_bind,
      Ora.Loop.StorageActionState.bindRead?] at hExec
    split at hExec
    · simp only [Option.none_bind] at hExec
      cases hExec
    · simp only [Option.some_bind, Option.some.injEq] at hExec
      subst result
      rfl

theorem storageWriteTrace
    (semantics : Ora.Loop.StorageActionSemantics LocationExpr ValueExpr)
    (location : LocationExpr)
    (value : ValueExpr)
    (initial : Ora.Loop.StorageActionState)
    (result : Ora.Loop.StorageActionResult)
    (hExec : Ora.Loop.execStorageAction? semantics (.write location value) initial =
      some result) :
    ∃ resolved,
      semantics.evalLocation location initial = some resolved ∧
        result.writeTrace = [resolved] := by
  simp only [Ora.Loop.execStorageAction?, Option.bind_eq_bind,
    Option.bind_eq_some] at hExec
  rcases hExec with ⟨resolved, hResolved, stored, _, hExec⟩
  split at hExec
  · simp only [Option.none_bind] at hExec
    cases hExec
  · simp only [Option.pure_def, Option.some_bind,
      Option.some.injEq] at hExec
    subst result
    exact ⟨resolved, hResolved, rfl⟩

theorem bindStateLookupFreeOfNotMem
    (env bound : Env)
    (variables : List Ora.Loop.VariableRow)
    (state : List Ora.Integer.Value)
    (id : FreeVarId)
    (hFresh : id ∉ variables.map Ora.Loop.VariableRow.id)
    (hBind : Ora.Loop.bindState? env variables state = some bound) :
    bound.lookupFree id = env.lookupFree id := by
  induction variables generalizing env bound state with
  | nil =>
      cases state with
      | nil =>
          simp only [Ora.Loop.bindState?, Option.some.injEq] at hBind
          subst bound
          rfl
      | cons _ _ => simp [Ora.Loop.bindState?] at hBind
  | cons row rest ih =>
      cases state with
      | nil => simp [Ora.Loop.bindState?] at hBind
      | cons value values =>
          simp only [List.map_cons, List.mem_cons, not_or] at hFresh
          simp only [Ora.Loop.bindState?] at hBind
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

theorem discharge_q9 : emittedQuery_9 := by
  unfold emittedQuery_9 Ora.Loop.directStorageLoopProofAuthorizationFromAssumptions
  have hSupported : emittedLoopSummary_1.supported emittedManifest = true := by decide
  have hDirect : Ora.Loop.directStorageSummarySupported emittedManifest emittedLoopSummary_1 = true := by
    simp only [Ora.Loop.directStorageSummarySupported, hSupported,
      emittedLoopSummary_1]
    decide
  constructor
  · let zero : Value := .integer (Ora.Integer.Value.ofNat (.unsigned .w256) 0)
    let one : Value := .integer (Ora.Integer.Value.ofNat (.unsigned .w256) 1)
    let witness := ((Env.empty
      |>.setFree { file_id := 0, pattern_id := 0 } zero)
      |>.setFree { file_id := 0, pattern_id := 1 } one)
      |>.setFree { file_id := 0, pattern_id := 2 } zero
    refine ⟨witness, {
      locals := [Ora.Integer.Value.ofNat (.unsigned .w256) 0]
      storage := Ora.Loop.storageStateFromEnv witness true
    }, ?_, ?_, ?_⟩
    · dsimp [witness, zero, one]
      simp only [assumptionsDenoteInEnv, assumptionsDenoteInEnv?,
        assumptionDenotesInEnv?, assumptionAnd?, emittedAssumptionRow_1,
        formulaDenotes?, denoteFormula?, denoteValue?, emitted_term_0_lookup,
        emitted_term_1_lookup, emitted_term_2_lookup, Env.setFree,
        Env.lookupVar, Env.lookupFree, lookupFreeBinding, Value.eqProp?,
        Ora.Integer.Value.ofNat, Ora.Integer.Value.eqProp?,
        Ora.Integer.Value.relation?,
        Option.bind_eq_bind, Option.some_bind, Option.map_some',
        List.foldl, List.isEmpty, Bool.false_eq_true, beq_iff_eq,
        if_pos, if_false]
      simp
    · simp [Ora.Loop.contextReady, emittedLoopSummary_1, witness, zero, one,
        Env.setFree, Env.lookupFree, lookupFreeBinding,
        TyRef.integerShape?, integerShapeForCompilerTypeId_u256]
      repeat' first | constructor | rfl
    · simp only [Ora.Loop.denoteDirectStorageLoopSummary?, hDirect, if_true]
      simp only [Ora.Loop.denoteDirectStorageEdges?,
        Ora.Loop.denoteDirectStorageEdge?, Ora.Loop.directStorageTimeline?,
        Ora.Loop.directStorageEvent?, Ora.Loop.directStorageLocation?,
        Ora.Loop.directStorageProjection?, Ora.Loop.directStorageProjections?,
        Ora.Loop.denoteControlTarget?, Option.bind_eq_bind,
        Option.some_bind, Option.pure_def, emittedLoopSummary_1,
        List.isEmpty, List.map, List.append, Bool.not_false, Bool.not_true,
        Bool.false_or, Bool.false_eq_true, if_false]
      constructor <;> rfl
  · intro env hAssumptions hContext
    have hWrite := hContext {
      index := 0
      id := { file_id := 0, pattern_id := 0 }
      name := "write_key"
      ty := .compilerTypeId 6
    } (by simp [emittedLoopSummary_1])
    have hObserved := hContext {
      index := 1
      id := { file_id := 0, pattern_id := 1 }
      name := "observed_key"
      ty := .compilerTypeId 6
    } (by simp [emittedLoopSummary_1])
    have hValueBinding := hContext {
      index := 2
      id := { file_id := 0, pattern_id := 2 }
      name := "value"
      ty := .compilerTypeId 6
    } (by simp [emittedLoopSummary_1])
    simp only [TyRef.integerShape?, integerShapeForCompilerTypeId_u256]
      at hWrite hObserved hValueBinding
    rcases hWrite with ⟨writeKey, hWriteShape, hWriteKey⟩
    rcases hObserved with ⟨observedKey, hObservedShape, hObservedKey⟩
    rcases hValueBinding with ⟨value, hValueShape, hValue⟩
    have hAssumptions' := hAssumptions
    simp only [assumptionsDenoteInEnv, assumptionsDenoteInEnv?,
      assumptionDenotesInEnv?, assumptionAnd?, emittedAssumptionRow_1,
      formulaDenotes?, denoteFormula?, denoteValue?, emitted_term_0_lookup,
      emitted_term_1_lookup, emitted_term_2_lookup, Env.lookupVar,
      Env.lookupFree, Option.bind_eq_bind, Option.some_bind,
      Option.none_bind, Option.pure_def] at hAssumptions'
    simp only [Env.lookupFree] at hWriteKey hObservedKey
    simp only [hWriteKey, hObservedKey, Value.eqProp?,
      Ora.Integer.Value.eqProp?] at hAssumptions'
    have hShapes : writeKey.shape = observedKey.shape :=
      (Option.some.inj hWriteShape).symm.trans (Option.some.inj hObservedShape)
    have hWriteUnsigned : writeKey.shape = .unsigned .w256 :=
      (Option.some.inj hWriteShape).symm
    simp [hShapes] at hAssumptions'
    have hKeysNe : (Value.integer writeKey) ≠ .integer observedKey := by
      intro hEqual
      cases hEqual
      exact hAssumptions' rfl
    simp only [Ora.Loop.denoteDirectStorageLoopSummary?, hDirect, if_true]
    simp only [Ora.Loop.denoteDirectStorageEdges?,
      Ora.Loop.denoteDirectStorageEdge?, Ora.Loop.directStorageTimeline?,
      Ora.Loop.directStorageEvent?, Ora.Loop.directStorageLocation?,
      Ora.Loop.directStorageProjection?, Ora.Loop.directStorageProjections?,
      Ora.Loop.denoteControlTarget?, Option.bind_eq_bind,
      Option.some_bind, Option.pure_def, emittedLoopSummary_1,
      List.isEmpty, List.map, List.append, Bool.not_false, Bool.not_true,
      Bool.false_or, Bool.false_eq_true, if_false]
    refine { base := ?_, backedge := ?_, edgeSafety := ?_, exit := ?_ }
    · intro state hInitial
      rcases hInitial with ⟨hStorage, hLocals⟩
      rcases state with ⟨locals, storage⟩
      change some [Ora.Integer.Value.ofNat (.unsigned .w256) 0] =
        some locals at hLocals
      change storage = Ora.Loop.storageStateFromEnv env true at hStorage
      have hState := (Option.some.inj hLocals).symm
      subst locals
      subst storage
      constructor
      · simp [Ora.Loop.stateReady, Ora.Loop.valuesMatchVariables,
          emittedLoopSummary_1, TyRef.integerShape?,
          integerShapeForCompilerTypeId_u256, Ora.Integer.Value.ofNat]
      · intro formula hFormula
        simp only [List.mem_cons, List.not_mem_nil, or_false] at hFormula
        rcases hFormula with rfl
        unfold Ora.Loop.StorageFormulaContext.evalFormula?
          Ora.Loop.StorageFormulaContext.stateEnv?
          Ora.Loop.directStorageContext
        simp only [Ora.Loop.bindStorageReads, List.foldl,
          Option.bind_eq_bind, Option.some_bind, Option.none_bind]
        ora_loop_reduce
        simp only [Ora.Loop.StorageFormulaContext.formulaStorageReady,
          Ora.Loop.StorageFormulaContext.termStorageReady,
          Ora.Loop.StorageFormulaContext.withStoreViews,
          Ora.Loop.directStorageResolvePlace, ora_loop_data,
          formulaDenotes?, denoteFormula?, denoteValue?, Env.lookupVar,
          Env.lookupFree, Option.bind_eq_bind, Option.some_bind,
          Option.none_bind, Option.pure_def]
        simp only [hObservedKey, Env.lookupFree, Env.lookupPlace,
          Env.lookupEntryPlace, lookupFreeBinding,
          Ora.Loop.StorageFormulaContext.withStoreViews,
          Ora.Loop.directStorageResolvePlace,
          Ora.Loop.StorageActionState.currentValue,
          Ora.Loop.StorageActionState.oldValue,
          Ora.Loop.storageStateFromEnv, Bool.false_eq_true,
          Bool.true_eq_false, if_false, if_pos,
          Option.bind_eq_bind, Option.some_bind, Option.none_bind,
          Option.pure_def, List.isEmpty, Bool.not_true,
          Bool.not_false, Bool.false_or, Bool.true_or]
        have hRegion : (RegionRef.storage != RegionRef.storage) = false := by decide
        have hRoot : "values".isEmpty = false := by decide
        simp only [hRegion, hRoot, Bool.false_eq_true, if_false]
        generalize hStored : env.placeValue (.entry ({
          root := "values"
          region := .storage
        } : PlaceRef)) = stored
        cases stored <;> simp [hStored] <;> exact ⟨True, trivial⟩
    · simp only [Ora.Loop.LoopVC.BackedgeObligation,
        Ora.Loop.timelineEdge]
      intro edge current following hMember hTarget hInvariant hPath hSafe hNext
      simp only [List.mem_cons, List.not_mem_nil, or_false] at hMember
      rcases hMember with rfl | rfl
      · dsimp at hNext ⊢
        rcases hNext with ⟨result, locals, hExec, hLocals, rfl⟩
        let variables := emittedLoopSummary_1.variables
        let bindings := emittedLoopSummary_1.edges[0].nextState
        let context := Ora.Loop.directStorageContext emittedManifest env variables
        change Ora.Loop.directStorageFinalLocals? context variables bindings
          result.state = some locals at hLocals
        unfold Ora.Loop.directStorageFinalLocals? at hLocals
        cases hEnv : context.stateEnv? result.state with
        | none => simp [hEnv] at hLocals
        | some bound =>
            simp only [hEnv, Option.bind_eq_bind, Option.some_bind] at hLocals
            have hReady := Ora.Loop.bound_state_in_env_ready emittedManifest
              variables bound bindings locals hLocals
            let observedLocation : Ora.Loop.StorageLocation := {
              root := "values"
              projections := [.mappingKey (.integer observedKey)]
            }
            have hPreserved : result.state.loop.storage observedLocation =
                current.storage observedLocation := by
              apply Ora.Loop.TimelineExec.preserves_outside _ _ _ _ _ hExec
              intro hWritten
              cases hExec with
              | cons h0 hTail0 =>
                  cases hTail0 with
                  | cons h1 hTail1 =>
                      cases hTail1 with
                      | cons h2 hTail2 =>
                          cases hTail2 with
                          | cons h3 hTail3 =>
                              cases hTail3
                              cases h0 with
                              | assume hEval hHolds =>
                                  cases h1 with
                                  | @storage action0 state0 read0Result hRead0 =>
                                      cases h2 with
                                      | @storage action1 state1 read1Result hRead1 =>
                                          cases h3 with
                                          | @storage action2 state2 writeResult hWrite =>
                                              have hRead0Empty :=
                                                storageReadWritesNone _ _ _ _ _ hRead0
                                              have hRead1Empty :=
                                                storageReadWritesNone _ _ _ _ _ hRead1
                                              simp [Ora.Loop.TimelineResult.comp,
                                                hRead0Empty, hRead1Empty] at hWritten
                                              rcases storageWriteTrace _ _ _ _ _ hWrite with
                                                ⟨writtenLocation, hLocation, hWriteTrace⟩
                                              rw [hWriteTrace] at hWritten
                                              simp only [List.mem_singleton] at hWritten
                                              change context.evalLocation? {
                                                root := "values"
                                                projections := [.mappingKey {
                                                  formula := .term 29
                                                  ty := .compilerTypeId 6
                                                }]
                                                valueTy := .compilerTypeId 6
                                              } read1Result.state = some writtenLocation
                                                at hLocation
                                              have hRead0Ids :=
                                                Ora.Loop.execStorageRead_appends_read_id
                                                  _ _ _ _ _ hRead0
                                              have hRead1Ids :=
                                                Ora.Loop.execStorageRead_appends_read_id
                                                  _ _ _ _ _ hRead1
                                              simp only [List.map_nil,
                                                List.nil_append] at hRead0Ids
                                              rw [hRead0Ids] at hRead1Ids
                                              have hFresh :
                                                  { file_id := 0, pattern_id := 0 } ∉
                                                    read1Result.state.reads.map
                                                      Ora.Loop.StorageReadBinding.id := by
                                                rw [hRead1Ids]
                                                decide
                                              cases hBoundState : context.bindLocals
                                                  context.entryEnv
                                                  read1Result.state.loop.locals with
                                              | none =>
                                                  simp [Ora.Loop.StorageFormulaContext.evalLocation?,
                                                    Ora.Loop.StorageFormulaContext.evalProjections?,
                                                    Ora.Loop.StorageFormulaContext.evalTypedValue?,
                                                    Ora.Loop.StorageFormulaContext.evalValue?,
                                                    Ora.Loop.StorageFormulaContext.stateEnv?,
                                                    hBoundState] at hLocation
                                              | some locationBound =>
                                                  have hBoundLookup :=
                                                    bindStateLookupFreeOfNotMem
                                                      env locationBound variables
                                                      read1Result.state.loop.locals
                                                      { file_id := 0, pattern_id := 0 }
                                                      (by simp [variables,
                                                        emittedLoopSummary_1])
                                                      hBoundState
                                                  have hReadLookup :=
                                                    Ora.Loop.bindStorageReads_lookupFree_of_not_mem
                                                      locationBound read1Result.state.reads
                                                      { file_id := 0, pattern_id := 0 }
                                                      hFresh
                                                  have hLookup :
                                                      (Ora.Loop.bindStorageReads locationBound
                                                        read1Result.state.reads).lookupFree
                                                          { file_id := 0, pattern_id := 0 } =
                                                        some (.integer writeKey) := by
                                                    rw [hReadLookup, hBoundLookup]
                                                    exact hWriteKey
                                                  let formulaEnv :=
                                                    context.withStoreViews
                                                      (Ora.Loop.bindStorageReads locationBound
                                                        read1Result.state.reads)
                                                      read1Result.state
                                                  have hFormulaReady :
                                                      context.formulaStorageReady formulaEnv
                                                        (.term 29) = true := by
                                                    unfold Ora.Loop.StorageFormulaContext.formulaStorageReady
                                                      Ora.Loop.StorageFormulaContext.termStorageReady
                                                    simp only [context,
                                                      Ora.Loop.directStorageContext,
                                                      emitted_terms_length,
                                                      emitted_term_29_lookup]
                                                  have hFormulaValue :
                                                      formulaValue? context.manifest formulaEnv
                                                        (.term 29) =
                                                        some (.integer writeKey) := by
                                                    unfold formulaValue? denoteValue?
                                                    simp only [context,
                                                      Ora.Loop.directStorageContext,
                                                      emitted_terms_length,
                                                      emitted_term_29_lookup,
                                                      Env.lookupVar,
                                                      Ora.Loop.StorageFormulaContext.withStoreViews_lookupFree,
                                                      hLookup]
                                                    simpa only [formulaEnv,
                                                      Ora.Loop.StorageFormulaContext.withStoreViews_lookupFree]
                                                      using hLookup
                                                  simp only [Ora.Loop.StorageFormulaContext.evalLocation?,
                                                    Ora.Loop.StorageFormulaContext.evalProjections?,
                                                    Ora.Loop.StorageFormulaContext.evalTypedValue?,
                                                    Ora.Loop.StorageFormulaContext.evalValue?,
                                                    Ora.Loop.StorageFormulaContext.stateEnv?,
                                                    Ora.Loop.directStorageProjectionData,
                                                    Ora.Loop.storageValueMatchesTy,
                                                    hBoundState, hFormulaReady,
                                                    hFormulaValue,
                                                    hWriteUnsigned,
                                                    TyRef.integerShape?,
                                                    integerShapeForCompilerTypeId_u256,
                                                    Option.bind_eq_bind, Option.some_bind,
                                                    Option.none_bind, Option.pure_def,
                                                    List.isEmpty, Bool.not_false,
                                                    Bool.not_true,
                                                    Bool.false_eq_true, Bool.false_or,
                                                    beq_self_eq_true, if_pos,
                                                    if_false] at hLocation
                                                  cases hLocation
                                                  exact hKeysNe
                                                    (Ora.Loop.StorageProjection.mappingKey.inj
                                                      (List.cons.inj
                                                        (Ora.Loop.StorageLocation.mk.inj hWritten).2).1).symm
            constructor
            · exact hReady
            · intro formula hFormula
              simp only [List.mem_cons, List.not_mem_nil, or_false] at hFormula
              rcases hFormula with rfl
              have hCurrent := hInvariant.2 (.term 10) (by simp)
              rcases hCurrent with ⟨proposition, hCurrentEval, hHolds⟩
              refine ⟨proposition, ?_, hHolds⟩
              rw [← hCurrentEval]
              rcases Ora.Loop.bindState_some_of_stateReady env variables
                  current.locals hInvariant.1 with
                ⟨currentBound, hCurrentBound⟩
              rcases Ora.Loop.bindState_some_of_stateReady env variables
                  locals hReady with
                ⟨followingBound, hFollowingBound⟩
              have hObservedFresh :
                  { file_id := 0, pattern_id := 1 } ∉
                    variables.map Ora.Loop.VariableRow.id := by
                simp [variables, emittedLoopSummary_1]
              have hCurrentObserved := bindStateLookupFreeOfNotMem
                env currentBound variables current.locals
                { file_id := 0, pattern_id := 1 }
                hObservedFresh hCurrentBound
              have hFollowingObserved := bindStateLookupFreeOfNotMem
                env followingBound variables locals
                { file_id := 0, pattern_id := 1 }
                hObservedFresh hFollowingBound
              have hCurrentLookup : currentBound.lookupFree
                  { file_id := 0, pattern_id := 1 } =
                    some (.integer observedKey) := by
                rw [hCurrentObserved]
                exact hObservedKey
              have hFollowingLookup : followingBound.lookupFree
                  { file_id := 0, pattern_id := 1 } =
                    some (.integer observedKey) := by
                rw [hFollowingObserved]
                exact hObservedKey
              dsimp [variables, emittedLoopSummary_1] at hCurrentBound hFollowingBound
              dsimp [observedLocation] at hPreserved
              simp only [Env.lookupFree] at hCurrentLookup hFollowingLookup
              have hRegion : (RegionRef.storage != RegionRef.storage) = false := by decide
              have hRoot : "values".isEmpty = false := by decide
              unfold Ora.Loop.StorageFormulaContext.evalFormula?
                Ora.Loop.StorageFormulaContext.stateEnv?
                Ora.Loop.directStorageContext
              simp only [Ora.Loop.bindStorageReads, List.foldl,
                hCurrentBound, hFollowingBound, Option.bind_eq_bind,
                Option.some_bind, Option.none_bind]
              ora_loop_reduce
              simp only [Ora.Loop.StorageFormulaContext.formulaStorageReady,
                Ora.Loop.StorageFormulaContext.termStorageReady,
                Ora.Loop.StorageFormulaContext.withStoreViews,
                Ora.Loop.directStorageResolvePlace, ora_loop_data,
                formulaDenotes?, denoteFormula?, denoteValue?, Env.lookupVar,
                Env.lookupFree, Option.bind_eq_bind, Option.some_bind,
                Option.none_bind, Option.pure_def]
              simp only [hRegion, hRoot, hObservedKey, hCurrentLookup,
                hFollowingLookup, Env.lookupPlace, Env.lookupEntryPlace,
                lookupFreeBinding,
                Ora.Loop.StorageFormulaContext.withStoreViews,
                Ora.Loop.directStorageResolvePlace,
                Ora.Loop.StorageActionState.currentValue,
                Ora.Loop.StorageActionState.oldValue,
                observedLocation, hPreserved, Bool.false_eq_true,
                Bool.true_eq_false, if_false, if_pos,
                Option.bind_eq_bind, Option.some_bind, Option.none_bind,
                Option.pure_def, List.isEmpty, Bool.not_true,
                Bool.not_false, Bool.false_or, Bool.true_or]
      · simp at hTarget
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

theorem discharge_q11 : emittedQuery_11 := by
  unfold emittedQuery_11 obligationFollowsFromAssumptions
  constructor
  · rcases discharge_q9.1 with ⟨env, state, hAssumptions, _, _⟩
    exact ⟨env, hAssumptions⟩
  · intro env hAssumptions
    have hRequired :
        match formulaDenotes? emittedManifest env (.term 2) with
        | some proposition => proposition
        | none => False := by
      cases hFormula : formulaDenotes? emittedManifest env (.term 2) <;>
        simpa only [assumptionsDenoteInEnv, assumptionsDenoteInEnv?,
          assumptionDenotesInEnv?, assumptionAnd?, emittedAssumptionRow_1,
          optionPropAnd?, hFormula, and_true] using hAssumptions
    let readPlace : PlaceRef := {
      root := "values"
      region := .storage
      keys := [.parameter { file_id := 0, pattern_id := 1 }]
    }
    let writePlace : PlaceRef := {
      root := "values"
      region := .storage
      keys := [.parameter { file_id := 0, pattern_id := 0 }]
    }
    let evidence : KeyDisjointEvidence := {
      kind := .freeVarDisequality
      assumptionId := 1
      lhs := { file_id := 0, pattern_id := 0 }
      rhs := { file_id := 0, pattern_id := 1 }
      read := readPlace
      write := writePlace
      keyIndex := 0
    }
    have hNotDisjoint : placeDefinitelyDisjoint readPlace writePlace = false := by
      decide
    have hCovered : pairCoveredByEvidence readPlace writePlace [evidence] = true := by
      decide
    have hPath : keyEvidencePathMatches readPlace writePlace 0 evidence.lhs evidence.rhs = true := by
      decide
    have hAssumptionLookup :
        emittedManifest.assumptionById 1 = some emittedAssumptionRow_1 := by
      rfl
    have hRequiresKind :
        (AssumptionKind.requires != AssumptionKind.requires) = false := by
      decide
    have hNeOp : (BinaryOp.ne != BinaryOp.ne) = false := by
      decide
    have hEvidence :
        keyDisjointEvidenceDenotes? emittedManifest env evidence =
          formulaDenotes? emittedManifest env (.term 2) := by
      unfold keyDisjointEvidenceDenotes? keyDisjointEvidenceFormulaDenotes?
      simp only [evidence, hPath, Bool.true_eq_false, Bool.false_eq_true, if_false,
        hAssumptionLookup, emittedAssumptionRow_1,
        emitted_term_2_lookup, emitted_term_0_lookup, emitted_term_1_lookup,
        termFreeVarId?, freeVarPairMatches, beq_self_eq_true,
        Bool.true_and, Bool.true_or, if_true, hRequiresKind, hNeOp]
    have option_prop_and_true_right_intro (candidate : Option Prop) :
        (match candidate with
        | some proposition => proposition
        | none => False) →
        (match optionPropAnd? candidate (some True) with
        | some proposition => proposition
        | none => False) := by
      cases candidate <;> simp [optionPropAnd?]
    unfold obligationDenotesInEnv obligationDenotesInEnv?
    simp only [emittedObligationRow_10, effectFrameGoalDenotes?,
      placeListDisjointWithEvidence?, List.foldl]
    apply option_prop_and_true_left_intro
    simp only [placePairDisjointWithEvidence?, readPlace, writePlace,
      hNotDisjoint, Bool.false_eq_true, if_false, hCovered, if_true]
    apply option_prop_and_true_right_intro
    rw [hEvidence]
    exact hRequired

end Ora.UserProof.MappingStoragePreservation

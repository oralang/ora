namespace Ora.UserProof.FixedStoragePreservation

open Ora.Obligation
open Ora.Generated.Obligations.Source_fixed_storage_preservation_certificate_ff6a0546fe5678b9

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
  emitted_term_125_lookup emitted_term_126_lookup emitted_term_127_lookup
  emitted_term_129_lookup emitted_term_130_lookup emitted_term_132_lookup
  emitted_term_133_lookup emitted_term_134_lookup emitted_term_135_lookup
  emitted_term_138_lookup emitted_term_140_lookup
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
  emitted_term_35_lookup emitted_term_125_lookup emitted_term_126_lookup
  emitted_term_127_lookup emitted_term_129_lookup emitted_term_130_lookup
  emitted_term_132_lookup emitted_term_133_lookup emitted_term_134_lookup
  emitted_term_135_lookup emitted_term_138_lookup emitted_term_140_lookup

theorem third_assumption_holds
    (manifest : Manifest) (env : Env) (first second third : AssumptionRow)
    (hAll : assumptionsDenoteInEnv manifest env [first, second, third]) :
    match assumptionDenotesInEnv? manifest env third with
    | some proposition => proposition
    | none => False := by
  unfold assumptionsDenoteInEnv at hAll
  simp only [assumptionsDenoteInEnv?] at hAll
  cases hFirst : assumptionDenotesInEnv? manifest env first <;>
    simp [hFirst, assumptionAnd?] at hAll
  cases hSecond : assumptionDenotesInEnv? manifest env second <;>
    simp [hSecond, assumptionAnd?] at hAll
  cases hThird : assumptionDenotesInEnv? manifest env third <;>
    simp [hThird, assumptionAnd?] at hAll ⊢
  exact hAll.2.2

theorem first_assumption_holds
    (manifest : Manifest) (env : Env) (first second third : AssumptionRow)
    (hAll : assumptionsDenoteInEnv manifest env [first, second, third]) :
    match assumptionDenotesInEnv? manifest env first with
    | some proposition => proposition
    | none => False := by
  unfold assumptionsDenoteInEnv at hAll
  simp only [assumptionsDenoteInEnv?] at hAll
  cases hFirst : assumptionDenotesInEnv? manifest env first <;>
    simp [hFirst, assumptionAnd?] at hAll ⊢
  cases hSecond : assumptionDenotesInEnv? manifest env second <;>
    simp [hSecond, assumptionAnd?] at hAll
  cases hThird : assumptionDenotesInEnv? manifest env third <;>
    simp [hThird, assumptionAnd?] at hAll
  exact hAll.1

theorem second_assumption_holds
    (manifest : Manifest) (env : Env) (first second third : AssumptionRow)
    (hAll : assumptionsDenoteInEnv manifest env [first, second, third]) :
    match assumptionDenotesInEnv? manifest env second with
    | some proposition => proposition
    | none => False := by
  unfold assumptionsDenoteInEnv at hAll
  simp only [assumptionsDenoteInEnv?] at hAll
  cases hFirst : assumptionDenotesInEnv? manifest env first <;>
    simp [hFirst, assumptionAnd?] at hAll
  cases hSecond : assumptionDenotesInEnv? manifest env second <;>
    simp [hSecond, assumptionAnd?] at hAll ⊢
  cases hThird : assumptionDenotesInEnv? manifest env third <;>
    simp [hThird, assumptionAnd?] at hAll
  exact hAll.2.1

set_option maxRecDepth 4096 in
private theorem headerFacts_of_assumptions
    (env : Env)
    (hAssumptions : assumptionsDenoteInEnv emittedManifest env
      [emittedAssumptionRow_1, emittedAssumptionRow_2, emittedAssumptionRow_3]) :
    Ora.Loop.headerFactsHoldInEnv emittedManifest env emittedLoopSummary_1.headerFacts := by
  intro fact hFact
  simp only [emittedLoopSummary_1, List.mem_cons, List.not_mem_nil, or_false] at hFact
  rcases hFact with rfl | rfl | rfl
  · simpa [Ora.Loop.formulaHoldsInEnv, assumptionDenotesInEnv?, emittedAssumptionRow_1]
      using first_assumption_holds emittedManifest env emittedAssumptionRow_1
        emittedAssumptionRow_2 emittedAssumptionRow_3 hAssumptions
  · simpa [Ora.Loop.formulaHoldsInEnv, assumptionDenotesInEnv?, emittedAssumptionRow_2]
      using second_assumption_holds emittedManifest env emittedAssumptionRow_1
        emittedAssumptionRow_2 emittedAssumptionRow_3 hAssumptions
  · simpa [Ora.Loop.formulaHoldsInEnv, assumptionDenotesInEnv?, emittedAssumptionRow_3]
      using third_assumption_holds emittedManifest env emittedAssumptionRow_1
        emittedAssumptionRow_2 emittedAssumptionRow_3 hAssumptions

theorem some_prop_of_match_holds (value : Option Prop)
    (hValue : match value with | some proposition => proposition | none => False) :
    ∃ proposition, value = some proposition ∧ proposition := by
  cases hValueOption : value with
  | none => simp [hValueOption] at hValue
  | some proposition =>
      exact ⟨proposition, rfl, by simpa [hValueOption] using hValue⟩

-- The generated term table contains function-level quantified obligations in
-- addition to the loop row; this allowance covers data reduction only.
set_option maxRecDepth 4096 in
theorem discharge_q17 : emittedQuery_17 := by
  unfold emittedQuery_17 Ora.Loop.directStorageLoopProofAuthorizationFromAssumptions
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
    have hWitnessAssumptions : assumptionsDenoteInEnv emittedManifest witness
        [emittedAssumptionRow_1, emittedAssumptionRow_2, emittedAssumptionRow_3] := by
      simp only [assumptionsDenoteInEnv, assumptionsDenoteInEnv?,
        assumptionDenotesInEnv?, assumptionAnd?, emittedAssumptionRow_1,
        emittedAssumptionRow_2, emittedAssumptionRow_3, formulaDenotes?,
        denoteFormula?, denoteValue?, emitted_term_0_lookup, emitted_term_1_lookup,
        emitted_term_2_lookup, emitted_term_3_lookup, emitted_term_4_lookup,
        emitted_term_5_lookup, emitted_term_6_lookup, emitted_term_7_lookup,
        emitted_term_8_lookup, Env.lookupVar, Env.lookupFree,
        Option.bind_eq_bind, Option.some_bind, Option.none_bind, Option.pure_def]
      simp [witness, zero, one, Env.setFree, lookupFreeBinding,
        TyRef.integerShape?, integerShapeForCompilerTypeId_u256,
        IntegerLiteralTerm.asInteger?, Ora.Integer.Value.ofNat]
      ora_loop_value_reduce
      repeat' apply And.intro
      all_goals ora_loop_close
    refine ⟨witness, {
      locals := [Ora.Integer.Value.ofNat (.unsigned .w256) 0]
      storage := Ora.Loop.storageStateFromEnv witness true
    }, hWitnessAssumptions, ?_, ?_⟩
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
      constructor
      · exact headerFacts_of_assumptions witness hWitnessAssumptions
      · constructor <;> rfl
  · intro env hAssumptions hContext
    have hWrite := hContext {
      index := 0
      id := { file_id := 0, pattern_id := 0 }
      name := "write_index"
      ty := .compilerTypeId 6
    } (by simp [emittedLoopSummary_1])
    have hObserved := hContext {
      index := 1
      id := { file_id := 0, pattern_id := 1 }
      name := "observed_index"
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
    have hAssumptions' := third_assumption_holds emittedManifest env
      emittedAssumptionRow_1 emittedAssumptionRow_2 emittedAssumptionRow_3
      (by simpa [emittedAssumptionRow_1, emittedAssumptionRow_2,
        emittedAssumptionRow_3] using hAssumptions)
    simp only [assumptionDenotesInEnv?, emittedAssumptionRow_3,
      formulaDenotes?, denoteFormula?, denoteValue?, emitted_term_6_lookup,
      emitted_term_7_lookup, emitted_term_8_lookup, Env.lookupVar,
      Option.bind_eq_bind, Option.some_bind, Option.none_bind,
      Option.pure_def] at hAssumptions'
    simp only [hWriteKey, hObservedKey, Value.eqProp?,
      Ora.Integer.Value.eqProp?] at hAssumptions'
    have hShapes : writeKey.shape = observedKey.shape :=
      (Option.some.inj hWriteShape).symm.trans (Option.some.inj hObservedShape)
    have hWriteUnsigned : writeKey.shape = .unsigned .w256 :=
      (Option.some.inj hWriteShape).symm
    have hObservedUnsigned : observedKey.shape = .unsigned .w256 :=
      (Option.some.inj hObservedShape).symm
    have hObservedBinding := hObservedKey
    simp only [Env.lookupFree] at hObservedBinding
    have hWriteBound := first_assumption_holds emittedManifest env
      emittedAssumptionRow_1 emittedAssumptionRow_2 emittedAssumptionRow_3
      (by simpa [emittedAssumptionRow_1, emittedAssumptionRow_2,
        emittedAssumptionRow_3] using hAssumptions)
    have hObservedBound := second_assumption_holds emittedManifest env
      emittedAssumptionRow_1 emittedAssumptionRow_2 emittedAssumptionRow_3
      (by simpa [emittedAssumptionRow_1, emittedAssumptionRow_2,
        emittedAssumptionRow_3] using hAssumptions)
    simp only [assumptionDenotesInEnv?, emittedAssumptionRow_1,
      formulaDenotes?, denoteFormula?, denoteValue?, emitted_term_0_lookup,
      emitted_term_1_lookup, emitted_term_2_lookup, Env.lookupVar,
      Option.bind_eq_bind, Option.some_bind, Option.none_bind,
      Option.pure_def] at hWriteBound
    simp only [assumptionDenotesInEnv?, emittedAssumptionRow_2,
      formulaDenotes?, denoteFormula?, denoteValue?, emitted_term_3_lookup,
      emitted_term_4_lookup, emitted_term_5_lookup, Env.lookupVar,
      Option.bind_eq_bind, Option.some_bind, Option.none_bind,
      Option.pure_def] at hObservedBound
    have hWriteBound' : match writeKey.relation? (.unsigned .w256)
        Ora.Integer.unsignedLt (Ora.Integer.Value.ofNat (.unsigned .w256) 4) with
      | some proposition => proposition
      | none => False := by
      simpa [hWriteKey, hWriteUnsigned, TyRef.integerShape?,
      integerShapeForCompilerTypeId_u256, IntegerLiteralTerm.asInteger?,
      parseDecimalNat?, parseDecimalNatAux, decimalDigit?,
      Ora.Integer.Value.ofNat] using hWriteBound
    have hObservedBound' : match observedKey.relation? (.unsigned .w256)
        Ora.Integer.unsignedLt (Ora.Integer.Value.ofNat (.unsigned .w256) 4) with
      | some proposition => proposition
      | none => False := by
      simpa [hObservedKey, hObservedUnsigned, TyRef.integerShape?,
      integerShapeForCompilerTypeId_u256, IntegerLiteralTerm.asInteger?,
      parseDecimalNat?, parseDecimalNatAux, decimalDigit?,
      Ora.Integer.Value.ofNat] using hObservedBound
    rcases some_prop_of_match_holds _ hWriteBound' with
      ⟨writeBoundProposition, hWriteRelation, hWriteBoundHolds⟩
    rcases some_prop_of_match_holds _ hObservedBound' with
      ⟨observedBoundProposition, hObservedRelation, hObservedBoundHolds⟩
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
      rcases hInitial with ⟨hHeaderFacts, hStorage, hLocals⟩
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
      · constructor
        · exact hHeaderFacts
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
          simp only [hObservedKey, hObservedUnsigned, Env.lookupFree, Env.lookupPlace,
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
          cases stored <;> simp [hStored, hObservedBinding,
            hObservedUnsigned] <;>
            exact ⟨True, trivial⟩
    · simp only [Ora.Loop.LoopVC.BackedgeObligation,
        Ora.Loop.timelineEdge]
      intro edge current following hMember hTarget hInvariant hPath hSafe hNext
      simp only [List.mem_cons, List.not_mem_nil, or_false] at hMember
      rcases hMember with rfl | rfl
      · dsimp at hNext ⊢
        rcases hInvariant with ⟨hInvariantReady, hHeaderFacts, hInvariantFormulas⟩
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
            have hReady := Ora.Loop.bound_state_in_env_ready emittedManifest
              variables bound bindings locals hLocals
            let observedLocation : Ora.Loop.StorageLocation := {
              root := "values"
              projections := [.fixedIndex (.integer observedKey)]
            }
            let writeLocation : Ora.Loop.StorageLocation := {
              root := "values"
              projections := [.fixedIndex (.integer writeKey)]
            }
            have hPreserved : result.state.loop.storage observedLocation =
                current.storage observedLocation := by
              apply Ora.Loop.TimelineExec.preserves_outside_resolved_frame
                _ _ _ _ [writeLocation] observedLocation hExec
              · intro written hWritten
                cases hExec with
                | cons h0 hTail0 =>
                  cases hTail0 with
                  | cons h1 hTail1 =>
                      cases hTail1 with
                      | cons h2 hTail2 =>
                          cases hTail2 with
                          | cons h3 hTail3 =>
                              cases hTail3 with
                              | cons h4 hTail4 =>
                                cases hTail4 with
                                | cons h5 hTail5 =>
                                  cases hTail5 with
                                  | cons h6 hTail6 =>
                                    cases hTail6
                                    cases h0 with
                                    | assume hEval hHolds =>
                                      cases h1 with
                                      | check hCheck0 =>
                                        cases h2 with
                                        | @storage action0 state0 read0Result hRead0 =>
                                          cases h3 with
                                          | check hCheck1 =>
                                            cases h4 with
                                            | @storage action1 state1 read1Result hRead1 =>
                                              cases h5 with
                                              | check hCheck2 =>
                                                cases h6 with
                                                | @storage action2 state2 writeResult hWrite =>
                                              have hRead0Empty :=
                                                Ora.Loop.execStorageRead_writeTrace_nil
                                                  _ _ _ _ _ hRead0
                                              have hRead1Empty :=
                                                Ora.Loop.execStorageRead_writeTrace_nil
                                                  _ _ _ _ _ hRead1
                                              simp [Ora.Loop.TimelineResult.comp,
                                                hRead0Empty, hRead1Empty] at hWritten
                                              rcases Ora.Loop.execStorageWrite_writeTrace_singleton
                                                  _ _ _ _ _ hWrite with
                                                ⟨writtenLocation, hLocation, hWriteTrace⟩
                                              rw [hWriteTrace] at hWritten
                                              simp only [List.mem_singleton] at hWritten
                                              change context.evalLocation? {
                                                root := "values"
                                                projections := [.fixedIndex {
                                                  index := {
                                                    formula := .term 132
                                                    ty := .compilerTypeId 6
                                                  }
                                                  extent := 4
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
                                                    Ora.Loop.bindState_lookupFree_of_not_mem
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
                                                  have hLookupRaw :
                                                      (Ora.Loop.bindStorageReads locationBound
                                                        read1Result.state.reads).lookupFree
                                                          { file_id := 0, pattern_id := 0 } =
                                                        some (.integer writeKey) := by
                                                    rw [hReadLookup, hBoundLookup]
                                                    exact hWriteKey
                                                  let bound := Ora.Loop.bindStorageReads
                                                    locationBound read1Result.state.reads
                                                  let formulaBound := context.withStoreViews
                                                    bound read1Result.state
                                                  have hStateEnv : context.stateEnv?
                                                      read1Result.state = some formulaBound := by
                                                    simp [Ora.Loop.StorageFormulaContext.stateEnv?,
                                                      hBoundState, bound, formulaBound]
                                                  have hLookup : formulaBound.lookupFree
                                                      { file_id := 0, pattern_id := 0 } =
                                                        some (.integer writeKey) := by
                                                    simpa [formulaBound, bound] using hLookupRaw
                                                  have hKeyType : Ora.Loop.storageValueMatchesTy
                                                      (.integer writeKey) (.compilerTypeId 6) = true := by
                                                    simp [Ora.Loop.storageValueMatchesTy,
                                                      TyRef.integerShape?,
                                                      integerShapeForCompilerTypeId_u256,
                                                      hWriteUnsigned]
                                                  have hResolved :=
                                                    context.evalLocation_fixed_free_variable
                                                      read1Result.state formulaBound 132 {
                                                        id := { file_id := 0, pattern_id := 0 }
                                                        name := "write_index"
                                                        ty := some (.compilerTypeId 6)
                                                      } writeKey (.compilerTypeId 6)
                                                      (.compilerTypeId 6) "values" 4
                                                      hStateEnv emitted_term_132_lookup hLookup
                                                      hKeyType hWriteUnsigned (by decide)
                                                      (by decide) (by decide) (by decide)
                                                  have hLocationEqual :
                                                      writtenLocation = writeLocation :=
                                                    Option.some.inj (hLocation.symm.trans hResolved)
                                                  rw [hLocationEqual] at hWritten
                                                  simpa only [List.mem_singleton] using hWritten
              · intro written hMember
                simp only [List.mem_singleton] at hMember
                subst written
                exact Ora.Loop.directStorageFixedIndexLocations_ne "values"
                  (.integer observedKey) (.integer writeKey) hKeysNe.symm
            refine ⟨hReady, hHeaderFacts, ?_⟩
            intro formula hFormula
            simp only [List.mem_cons, List.not_mem_nil, or_false] at hFormula
            rcases hFormula with rfl
            have hCurrent := hInvariantFormulas (.term 16) (by simp)
            rcases hCurrent with ⟨proposition, hCurrentEval, hHolds⟩
            refine ⟨proposition, ?_, hHolds⟩
            rw [← hCurrentEval]
            rcases Ora.Loop.bindState_some_of_stateReady env variables
                current.locals hInvariantReady with
              ⟨currentBound, hCurrentBound⟩
            rcases Ora.Loop.bindState_some_of_stateReady env variables
                locals hReady with
              ⟨followingBound, hFollowingBound⟩
            have hObservedFresh :
                { file_id := 0, pattern_id := 1 } ∉
                  variables.map Ora.Loop.VariableRow.id := by
              simp [variables, emittedLoopSummary_1]
            have hCurrentObserved := Ora.Loop.bindState_lookupFree_of_not_mem
              env currentBound variables current.locals
              { file_id := 0, pattern_id := 1 }
              hObservedFresh hCurrentBound
            have hFollowingObserved := Ora.Loop.bindState_lookupFree_of_not_mem
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
            simp only [hRegion, hRoot, hObservedKey, hObservedBinding,
              hObservedUnsigned, hCurrentLookup,
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
              Bool.not_false, Bool.false_or, Bool.true_or,
              beq_self_eq_true]
      · simp at hTarget
    · intro edge state hMember hInvariant _
      simp only [List.mem_cons, List.not_mem_nil, or_false] at hMember
      rcases hMember with rfl | rfl
      · dsimp
        intro before formula after hEvents beforeResult hExec
        have hCheck : .check formula ∈ before ++ .check formula :: after := by simp
        rw [← hEvents] at hCheck
        simp at hCheck
        let variables := emittedLoopSummary_1.variables
        let context := Ora.Loop.directStorageContext emittedManifest env variables
        have hLocals := Ora.Loop.TimelineExec.preserves_locals _ _ _ _
          (by intro child; nomatch child) hExec
        have hReady : Ora.Loop.stateReady variables
            beforeResult.state.loop.locals := by
          rw [hLocals]
          exact hInvariant.1
        rcases Ora.Loop.bindState_some_of_stateReady env variables
            beforeResult.state.loop.locals hReady with
          ⟨localsBound, hLocalsBound⟩
        let controlState : Ora.Loop.StorageActionState := {
          beforeResult.state with reads := []
        }
        let bound := context.withStoreViews localsBound controlState
        have hStateEnv : context.stateEnv? controlState = some bound := by
          simp [context, Ora.Loop.StorageFormulaContext.stateEnv?,
            Ora.Loop.directStorageContext, Ora.Loop.bindStorageReads,
            hLocalsBound, controlState, bound]
        have hWriteFresh : { file_id := 0, pattern_id := 0 } ∉
            controlState.reads.map Ora.Loop.StorageReadBinding.id := by
          simp [controlState]
        have hObservedFresh : { file_id := 0, pattern_id := 1 } ∉
            controlState.reads.map Ora.Loop.StorageReadBinding.id := by
          simp [controlState]
        have hWriteLookup : bound.lookupFree { file_id := 0, pattern_id := 0 } =
            some (.integer writeKey) := by
          rw [Ora.Loop.directStorageContext_lookupFree_of_not_bound
            emittedManifest env bound variables controlState
            { file_id := 0, pattern_id := 0 }
            (by simp [variables, emittedLoopSummary_1]) hWriteFresh hStateEnv]
          exact hWriteKey
        have hObservedLookup : bound.lookupFree { file_id := 0, pattern_id := 1 } =
            some (.integer observedKey) := by
          rw [Ora.Loop.directStorageContext_lookupFree_of_not_bound
            emittedManifest env bound variables controlState
            { file_id := 0, pattern_id := 1 }
            (by simp [variables, emittedLoopSummary_1]) hObservedFresh hStateEnv]
          exact hObservedKey
        have hTermFuel : 0 < context.manifest.terms.length := by
          change 0 < emittedManifest.terms.length
          simp [emitted_terms_length]
        rcases hCheck with rfl | rfl | rfl
        · refine ⟨observedBoundProposition,
              context.evalFormula_u256_lt_free_literal
                controlState bound 125 126 127 observedKey
                (Ora.Integer.Value.ofNat (.unsigned .w256) 4)
                observedBoundProposition (.compilerTypeId 6) hStateEnv
                emitted_term_127_lookup ?_ ?_ ?_ ?_ (by decide)
                hObservedRelation,
              hObservedBoundHolds⟩
          · exact context.termStorageReady_variable bound _ 125 _
              hTermFuel emitted_term_125_lookup
          · exact context.termStorageReady_intLit bound _ 126 _
              hTermFuel emitted_term_126_lookup
          · exact Ora.Loop.denoteValue_free_variable
              context.manifest bound _ 125 _ _ hTermFuel
              emitted_term_125_lookup hObservedLookup
          · exact Ora.Loop.denoteValue_intLit
              context.manifest bound _ 126 _ _ hTermFuel
              emitted_term_126_lookup (by decide)
        · refine ⟨observedBoundProposition,
              context.evalFormula_u256_lt_free_literal
                controlState bound 125 129 130 observedKey
                (Ora.Integer.Value.ofNat (.unsigned .w256) 4)
                observedBoundProposition (.compilerTypeId 6) hStateEnv
                emitted_term_130_lookup ?_ ?_ ?_ ?_ (by decide)
                hObservedRelation,
              hObservedBoundHolds⟩
          · exact context.termStorageReady_variable bound _ 125 _
              hTermFuel emitted_term_125_lookup
          · exact context.termStorageReady_intLit bound _ 129 _
              hTermFuel emitted_term_129_lookup
          · exact Ora.Loop.denoteValue_free_variable
              context.manifest bound _ 125 _ _ hTermFuel
              emitted_term_125_lookup hObservedLookup
          · exact Ora.Loop.denoteValue_intLit
              context.manifest bound _ 129 _ _ hTermFuel
              emitted_term_129_lookup (by decide)
        · refine ⟨writeBoundProposition,
              context.evalFormula_u256_lt_free_literal
                controlState bound 132 133 134 writeKey
                (Ora.Integer.Value.ofNat (.unsigned .w256) 4)
                writeBoundProposition (.compilerTypeId 6) hStateEnv
                emitted_term_134_lookup ?_ ?_ ?_ ?_ (by decide)
                hWriteRelation,
              hWriteBoundHolds⟩
          · exact context.termStorageReady_variable bound _ 132 _
              hTermFuel emitted_term_132_lookup
          · exact context.termStorageReady_intLit bound _ 133 _
              hTermFuel emitted_term_133_lookup
          · exact Ora.Loop.denoteValue_free_variable
              context.manifest bound _ 132 _ _ hTermFuel
              emitted_term_132_lookup hWriteLookup
          · exact Ora.Loop.denoteValue_intLit
              context.manifest bound _ 133 _ _ hTermFuel
              emitted_term_133_lookup (by decide)
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

-- The generated term table contains function-level quantified obligations in
-- addition to the frame row; this allowance covers data reduction only.
set_option maxRecDepth 4096 in
theorem discharge_q19 : emittedQuery_19 := by
  unfold emittedQuery_19 obligationFollowsFromAssumptions
  constructor
  · rcases discharge_q17.1 with ⟨env, state, hAssumptions, _, _⟩
    exact ⟨env, hAssumptions⟩
  · intro env hAssumptions
    have hRequired :
        match formulaDenotes? emittedManifest env (.term 8) with
        | some proposition => proposition
        | none => False := by
      have hThird := third_assumption_holds emittedManifest env
        emittedAssumptionRow_1 emittedAssumptionRow_2 emittedAssumptionRow_3
        hAssumptions
      simpa [assumptionDenotesInEnv?, emittedAssumptionRow_3] using hThird
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
      assumptionId := 3
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
    have hPath : keyEvidencePathMatches readPlace writePlace 0
        evidence.lhs evidence.rhs = true := by
      decide
    have hAssumptionLookup :
        emittedManifest.assumptionById 3 = some emittedAssumptionRow_3 := by
      rfl
    have hRequiresKind :
        (AssumptionKind.requires != AssumptionKind.requires) = false := by
      decide
    have hNeOp : (BinaryOp.ne != BinaryOp.ne) = false := by
      decide
    have hEvidence :
        keyDisjointEvidenceDenotes? emittedManifest env evidence =
          formulaDenotes? emittedManifest env (.term 8) := by
      unfold keyDisjointEvidenceDenotes? keyDisjointEvidenceFormulaDenotes?
      simp only [evidence, hPath, Bool.true_eq_false, Bool.false_eq_true,
        if_false, hAssumptionLookup, emittedAssumptionRow_3,
        emitted_term_8_lookup, emitted_term_6_lookup, emitted_term_7_lookup,
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
    simp only [emittedObligationRow_18, effectFrameGoalDenotes?,
      placeListDisjointWithEvidence?, List.foldl]
    apply option_prop_and_true_left_intro
    simp only [placePairDisjointWithEvidence?, readPlace, writePlace,
      hNotDisjoint, Bool.false_eq_true, if_false, hCovered, if_true]
    apply option_prop_and_true_right_intro
    rw [hEvidence]
    exact hRequired

end Ora.UserProof.FixedStoragePreservation

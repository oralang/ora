namespace Ora.UserProof.MappingStoragePreheader

open Ora.Obligation
open Ora.Loop
open Ora.Generated.Obligations.Source_mapping_storage_preheader_certificate_a4241569d0136b2a

attribute [local simp] emitted_manifest_wf emitted_terms_topological emitted_terms_length
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
  emitted_term_36_lookup emitted_term_37_lookup emitted_term_38_lookup
  emitted_term_39_lookup

def zero : Value :=
  .integer (Ora.Integer.Value.ofNat (.unsigned .w256) 0)

def one : Value :=
  .integer (Ora.Integer.Value.ofNat (.unsigned .w256) 1)

def witness : Env :=
  ((Env.empty.setFree { file_id := 0, pattern_id := 0 } zero)
    |>.setFree { file_id := 0, pattern_id := 1 } one)
    |>.setFree { file_id := 0, pattern_id := 2 } zero

private theorem headerFacts_of_assumption
    (env : Env)
    (hAssumptions : assumptionsDenoteInEnv emittedManifest env
      [emittedAssumptionRow_3]) :
    headerFactsHoldInEnv emittedManifest env emittedLoopSummary_1.headerFacts := by
  intro fact hFact
  simp only [emittedLoopSummary_1, List.mem_cons, List.not_mem_nil, or_false] at hFact
  subst fact
  cases hDenotes : formulaDenotes? emittedManifest env (.term 2) with
  | none =>
      simp [assumptionsDenoteInEnv, assumptionsDenoteInEnv?,
        assumptionDenotesInEnv?, assumptionAnd?, emittedAssumptionRow_3,
        hDenotes] at hAssumptions
  | some proposition =>
      unfold formulaHoldsInEnv
      rw [hDenotes]
      simpa [assumptionsDenoteInEnv, assumptionsDenoteInEnv?,
        assumptionDenotesInEnv?, assumptionAnd?, emittedAssumptionRow_3,
        hDenotes] using hAssumptions

theorem restrictStoragePreheaderLookup
    (bindings : List (FreeVarId × Value))
    (variables : List VariableRow)
    (id : FreeVarId)
    (hDeclared : variables.any (fun row => row.id == id) = true) :
    lookupFreeBinding
        (bindings.filter fun binding =>
          variables.any fun row => row.id == binding.1) id =
      lookupFreeBinding bindings id := by
  induction bindings with
  | nil => rfl
  | cons binding rest ih =>
      simp only [List.filter_cons, lookupFreeBinding]
      by_cases hKeep : variables.any (fun row => row.id == binding.1) = true
      · simp only [hKeep, if_true, lookupFreeBinding]
        by_cases hSame : (binding.1 == id) = true
        · simp only [hSame, if_true]
        · simp only [hSame, if_false, ih]
      · have hDifferent : (binding.1 == id) = false := by
          apply Bool.eq_false_iff.mpr
          intro hEqual
          have hIds : binding.1 = id := beq_iff_eq.mp hEqual
          rw [hIds, hDeclared] at hKeep
          exact hKeep rfl
        have hKeepFalse : variables.any (fun row => row.id == binding.1) =
            false := Bool.eq_false_iff.mpr hKeep
        simp only [hKeepFalse, Bool.false_eq_true, if_false, hDifferent, ih]

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

theorem mappingInvariantIff
    (env : Env)
    (state : LoopState)
    (observedKey : Ora.Integer.Value)
    (hObserved : env.lookupFree { file_id := 0, pattern_id := 1 } =
      some (.integer observedKey))
    (hReady : stateReady emittedLoopSummary_1.variables state.locals) :
    directFormulasHold
        (directStorageContext emittedManifest env emittedLoopSummary_1.variables)
        (storageStateFromEnv env true) state [.term 18] ↔
      match Value.eqProp?
          (state.storage {
            root := "values"
            projections := [.mappingKey (.integer observedKey)]
          })
          ((storageStateFromEnv env true) {
            root := "values"
            projections := [.mappingKey (.integer observedKey)]
          }) with
      | some proposition => proposition
      | none => False := by
  let context := directStorageContext emittedManifest env
    emittedLoopSummary_1.variables
  let entryStorage := storageStateFromEnv env true
  let place : PlaceRef := {
    root := "values"
    region := .storage
    keys := [.parameter { file_id := 0, pattern_id := 1 }]
  }
  let location : StorageLocation := {
    root := "values"
    projections := [.mappingKey (.integer observedKey)]
  }
  rcases bindState_some_of_stateReady env emittedLoopSummary_1.variables
      state.locals hReady with ⟨localsBound, hLocalsBound⟩
  have hFresh : { file_id := 0, pattern_id := 1 } ∉
      emittedLoopSummary_1.variables.map VariableRow.id := by
    simp [emittedLoopSummary_1]
  have hLocalObserved : localsBound.lookupFree
      { file_id := 0, pattern_id := 1 } = some (.integer observedKey) := by
    rw [bindState_lookupFree_of_not_mem env localsBound
      emittedLoopSummary_1.variables state.locals _ hFresh hLocalsBound]
    exact hObserved
  let actionState : StorageActionState := {
    loop := state
    entryStorage
  }
  let bound := context.withStoreViews localsBound actionState
  have hBound : context.stateEnv? actionState = some bound := by
    unfold StorageFormulaContext.stateEnv?
    change (bindState? env emittedLoopSummary_1.variables state.locals).bind
      (fun locals => some (context.withStoreViews
        (bindStorageReads locals []) actionState)) = some bound
    rw [hLocalsBound]
    rfl
  have hCurrentLocation : context.resolvePlace bound.lookupFree place =
      some location := by
    change directStorageResolvePlace bound.lookupFree place = some location
    have hRegion : (RegionRef.storage != RegionRef.storage) = false := by decide
    have hRoot : "values".isEmpty = false := by decide
    simp [place, location, directStorageResolvePlace, bound, hLocalObserved,
      hRegion, hRoot]
  have hEntryLocation : context.resolvePlace context.entryEnv.lookupFree place =
      some location := by
    change directStorageResolvePlace env.lookupFree place = some location
    have hRegion : (RegionRef.storage != RegionRef.storage) = false := by decide
    have hRoot : "values".isEmpty = false := by decide
    simp [place, location, directStorageResolvePlace, hObserved, hRegion, hRoot]
  have hLocalLocation : context.resolvePlace localsBound.lookupFree place =
      some location := by
    change directStorageResolvePlace localsBound.lookupFree place = some location
    have hRegion : (RegionRef.storage != RegionRef.storage) = false := by decide
    have hRoot : "values".isEmpty = false := by decide
    simp [place, location, directStorageResolvePlace, hLocalObserved,
      hRegion, hRoot]
  have hCurrentValue : bound.placeValue (.stable place) =
      actionState.loop.storage location := by
    simp [bound, StorageFormulaContext.withStoreViews, hLocalLocation,
      StorageActionState.currentValue]
  have hEntryValue : bound.placeValue (.entry place) =
      actionState.entryStorage location := by
    simp [bound, StorageFormulaContext.withStoreViews, hEntryLocation,
      StorageActionState.oldValue]
  have hCharacterization := context.evalFormula_place_eq_old_self
    actionState bound 39 18 15 17 16 (.compilerTypeId 6) place location hBound
    (by change emittedManifest.terms.length = 40; exact emitted_terms_length)
    emitted_term_18_lookup emitted_term_15_lookup emitted_term_17_lookup
    emitted_term_16_lookup hCurrentLocation hEntryLocation hCurrentValue hEntryValue
  constructor
  · intro hInvariant
    exact hCharacterization.mp (hInvariant (.term 18) (by simp))
  · intro hSemantic formula hMember
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hMember
    subst formula
    exact hCharacterization.mpr hSemantic

theorem preheaderPreservesObserved
    (env : Env)
    (writeKey observedKey : Ora.Integer.Value)
    (hWrite : env.lookupFree { file_id := 0, pattern_id := 0 } =
      some (.integer writeKey))
    (hWriteShape : writeKey.shape = .unsigned .w256)
    (hDistinct : (Value.integer writeKey) ≠ .integer observedKey)
    (result : TimelineResult)
    (hExec : TimelineExec
      (directStoragePreheaderSemantics
        (directStoragePreheaderContext emittedManifest env
          emittedLoopSummary_1.contextVariables))
      (directStorageTimeline? emittedLoopSummary_1.preheaderTimeline).get!
      (directStoragePreheaderInitialState env) result) :
    result.state.loop.storage {
      root := "values"
      projections := [.mappingKey (.integer observedKey)]
    } =
      (directStoragePreheaderInitialState env).loop.storage {
        root := "values"
        projections := [.mappingKey (.integer observedKey)]
      } := by
  let observedLocation : StorageLocation := {
    root := "values"
    projections := [.mappingKey (.integer observedKey)]
  }
  apply TimelineExec.preserves_outside _ _ _ _ observedLocation hExec
  intro hWritten
  simp only [emittedLoopSummary_1, directStorageTimeline?, directStorageEvent?,
    directStorageLocation?, directStorageProjection?, directStorageProjections?,
    Option.bind_eq_bind, Option.some_bind, Option.pure_def, List.map,
    Bool.not_false, Bool.false_eq_true, if_false] at hExec
  cases hExec with
  | cons hPath hTail0 =>
    cases hTail0 with
    | cons hCheck hTail1 =>
      cases hTail1 with
      | cons hWriteEvent hTail2 =>
        cases hTail2
        cases hPath
        cases hCheck
        cases hWriteEvent with
        | @storage _ _ writeResult hWriteAction =>
          rcases execStorageWrite_writeTrace_singleton _ _ _ _ _ hWriteAction with
            ⟨writtenLocation, hLocation, hTrace⟩
          simp [TimelineResult.comp, hTrace] at hWritten
          let context := directStoragePreheaderContext emittedManifest env
            emittedLoopSummary_1.contextVariables
          let initial := directStoragePreheaderInitialState env
          let expectedLocation : StorageLocation := {
            root := "values"
            projections := [.mappingKey (.integer writeKey)]
          }
          let bound := context.withStoreViews context.entryEnv initial
          have hBound : context.stateEnv? initial = some bound := by rfl
          have hDeclared : emittedLoopSummary_1.contextVariables.any
              (fun row => row.id == ({ file_id := 0, pattern_id := 0 } : FreeVarId)) =
              true := by decide
          have hRestricted : context.entryEnv.lookupFree
              { file_id := 0, pattern_id := 0 } = some (.integer writeKey) := by
            change lookupFreeBinding
                (env.freeBindings.filter fun binding =>
                  emittedLoopSummary_1.contextVariables.any fun row =>
                    row.id == binding.1)
                { file_id := 0, pattern_id := 0 } = some (.integer writeKey)
            rw [restrictStoragePreheaderLookup env.freeBindings
              emittedLoopSummary_1.contextVariables _ hDeclared]
            exact hWrite
          have hBoundWrite : bound.lookupFree
              { file_id := 0, pattern_id := 0 } = some (.integer writeKey) := by
            simpa [bound] using hRestricted
          have hKeyValue : context.evalValue? (.term 9) initial =
              some (.integer writeKey) := by
            apply context.evalValue_free_variable initial bound 9
              { id := { file_id := 0, pattern_id := 0 }, name := "write_key",
                ty := some (.compilerTypeId 6) }
              (.integer writeKey) hBound emitted_term_9_lookup hBoundWrite
          have hExpected : context.evalLocation? {
              root := "values"
              projections := [.mappingKey {
                formula := .term 9
                ty := .compilerTypeId 6
              }]
              valueTy := .compilerTypeId 6
            } initial = some expectedLocation := by
            have hRoot : "values".isEmpty = false := by decide
            have hValueTy : ((TyRef.compilerTypeId 6).isBool ||
                (TyRef.compilerTypeId 6).integerShape?.isSome) = true := by decide
            have hEmpty : context.evalProjections? initial [] = some [] := by rfl
            unfold StorageFormulaContext.evalLocation?
              StorageFormulaContext.evalProjections?
              StorageFormulaContext.evalTypedValue?
            simp [hRoot, hValueTy, hKeyValue, storageValueMatchesTy,
              hWriteShape, expectedLocation, hEmpty]
          change context.evalLocation? {
              root := "values"
              projections := [.mappingKey {
                formula := .term 9
                ty := .compilerTypeId 6
              }]
              valueTy := .compilerTypeId 6
            } initial = some writtenLocation at hLocation
          rw [hExpected] at hLocation
          have hLocationEq : expectedLocation = writtenLocation :=
            Option.some.inj hLocation
          rw [← hLocationEq] at hWritten
          apply hDistinct
          exact (StorageProjection.mappingKey.inj
            (List.cons.inj (StorageLocation.mk.inj hWritten).2).1).symm

theorem discharge_q11 : emittedQuery_11 := by
  unfold emittedQuery_11 directStorageLoopProofAuthorizationFromAssumptions
  have hSupported : emittedLoopSummary_1.supported emittedManifest = true := by decide
  have hDirect : directStorageSummarySupported emittedManifest
      emittedLoopSummary_1 = true := by
    simp only [directStorageSummarySupported, emittedLoopSummary_1, hSupported]
    decide
  constructor
  · have hWitnessAssumptions : assumptionsDenoteInEnv emittedManifest witness
        [emittedAssumptionRow_3] := by
      dsimp [witness, zero, one]
      simp only [assumptionsDenoteInEnv, assumptionsDenoteInEnv?,
        assumptionDenotesInEnv?, assumptionAnd?, emittedAssumptionRow_3,
        formulaDenotes?, denoteFormula?, denoteValue?, emitted_term_2_lookup,
        emitted_term_0_lookup, emitted_term_1_lookup, Env.setFree,
        Env.lookupVar, Env.lookupFree, lookupFreeBinding, Value.eqProp?,
        Ora.Integer.Value.ofNat, Ora.Integer.Value.eqProp?,
        Ora.Integer.Value.relation?, Option.bind_eq_bind, Option.some_bind,
        Option.map_some', List.foldl, List.isEmpty, Bool.false_eq_true,
        beq_iff_eq, if_pos, if_false]
      simp
    refine ⟨witness, {
      locals := [Ora.Integer.Value.ofNat (.unsigned .w256) 0]
      storage := (storageStateFromEnv witness true).set {
        root := "values"
        projections := [.mappingKey zero]
      } zero
    }, hWitnessAssumptions, ?_, ?_⟩
    · dsimp [witness, zero, one]
      simp [contextReady, emittedLoopSummary_1,
        Env.setFree, Env.lookupFree, lookupFreeBinding, TyRef.integerShape?,
        integerShapeForCompilerTypeId_u256]
      repeat' first | constructor | rfl
    · simp only [denoteDirectStorageLoopSummary?, hDirect, if_true]
      simp only [denoteDirectStorageEdges?, denoteDirectStorageEdge?,
        directStorageTimeline?, directStorageEvent?, directStorageLocation?,
        directStorageProjection?, directStorageProjections?, denoteControlTarget?,
        emittedLoopSummary_1, denoteDirectStoragePreheaderInitial?,
        Option.bind_eq_bind, Option.some_bind, Option.pure_def, List.isEmpty,
        List.map, List.append, Bool.not_false, Bool.not_true, Bool.false_or,
        Bool.false_eq_true, if_false]
      refine ⟨headerFacts_of_assumption witness hWitnessAssumptions, ?_⟩
      unfold directStoragePreheaderProducesInitial
      let context := directStoragePreheaderContext emittedManifest witness
        emittedLoopSummary_1.contextVariables
      let initial := directStoragePreheaderInitialState witness
      let bound := context.withStoreViews context.entryEnv initial
      have hBound : context.stateEnv? initial = some bound := by rfl
      have hBoundWrite : bound.lookupFree { file_id := 0, pattern_id := 0 } =
          some zero := by
        simp [bound, context, directStoragePreheaderContext,
          restrictStoragePreheaderEnv, emittedLoopSummary_1, witness, zero,
          one, Env.setFree, Env.lookupFree, lookupFreeBinding]
      have hBoundObserved : bound.lookupFree { file_id := 0, pattern_id := 1 } =
          some one := by
        simp [bound, context, directStoragePreheaderContext,
          restrictStoragePreheaderEnv, emittedLoopSummary_1, witness, zero,
          one, Env.setFree, Env.lookupFree, lookupFreeBinding]
      have hBoundValue : bound.lookupFree { file_id := 0, pattern_id := 2 } =
          some zero := by
        simp [bound, context, directStoragePreheaderContext,
          restrictStoragePreheaderEnv, emittedLoopSummary_1, witness, zero,
          one, Env.setFree, Env.lookupFree, lookupFreeBinding]
      have hPath : (directStoragePreheaderSemantics context).evalFormula
          (.term 5) initial = some True := by
        change context.evalFormula? (.term 5) initial = some True
        have hReady : context.formulaStorageReady bound (.term 5) = true := by
          unfold context directStoragePreheaderContext
          unfold StorageFormulaContext.formulaStorageReady
          rw [emitted_terms_length]
          simp only [StorageFormulaContext.termStorageReady,
            emitted_term_5_lookup, emitted_term_3_lookup,
            emitted_term_4_lookup, Bool.true_and]
        unfold StorageFormulaContext.evalFormula?
        simp only [hBound, Option.bind_eq_bind, Option.some_bind, hReady,
          Bool.not_true, Bool.false_eq_true, if_false]
        change formulaDenotes? emittedManifest bound (.term 5) = some True
        have hLeft : denoteValue? emittedManifest bound 40 3 = some zero := by
          unfold denoteValue?
          simp only [emitted_term_3_lookup, Env.lookupVar, hBoundWrite]
        have hRight : denoteValue? emittedManifest bound 40 4 = some one := by
          unfold denoteValue?
          simp only [emitted_term_4_lookup, Env.lookupVar, hBoundObserved]
        unfold formulaDenotes? denoteFormula?
        simp only [emitted_terms_length, emitted_term_5_lookup, hLeft, hRight,
          Value.eqProp?, Ora.Integer.Value.eqProp?, zero, one,
          Ora.Integer.Value.ofNat, Option.pure_def, Option.some.injEq]
        rw [dif_pos trivial]
        change some (¬(Ora.Integer.Value.ofNat (.unsigned .w256) 0).bits =
          (Ora.Integer.Value.ofNat (.unsigned .w256) 1).bits) = some True
        apply congrArg some
        apply propext
        decide
      have hCheck : (directStoragePreheaderSemantics context).evalFormula
          (.term 8) initial = some True := by
        change context.evalFormula? (.term 8) initial = some True
        have hReady : context.formulaStorageReady bound (.term 8) = true := by
          unfold context directStoragePreheaderContext
          unfold StorageFormulaContext.formulaStorageReady
          rw [emitted_terms_length]
          simp only [StorageFormulaContext.termStorageReady,
            emitted_term_8_lookup, emitted_term_6_lookup,
            emitted_term_7_lookup, Bool.true_and]
        unfold StorageFormulaContext.evalFormula?
        simp only [hBound, Option.bind_eq_bind, Option.some_bind, hReady,
          Bool.not_true, Bool.false_eq_true, if_false]
        change formulaDenotes? emittedManifest bound (.term 8) = some True
        have hLeft : denoteValue? emittedManifest bound 40 6 = some zero := by
          unfold denoteValue?
          simp only [emitted_term_6_lookup, Env.lookupVar, hBoundWrite]
        have hRight : denoteValue? emittedManifest bound 40 7 = some one := by
          unfold denoteValue?
          simp only [emitted_term_7_lookup, Env.lookupVar, hBoundObserved]
        unfold formulaDenotes? denoteFormula?
        simp only [emitted_terms_length, emitted_term_8_lookup, hLeft, hRight,
          Value.eqProp?, Ora.Integer.Value.eqProp?, zero, one,
          Ora.Integer.Value.ofNat, Option.pure_def, Option.some.injEq]
        rw [dif_pos trivial]
        change some (¬(Ora.Integer.Value.ofNat (.unsigned .w256) 0).bits =
          (Ora.Integer.Value.ofNat (.unsigned .w256) 1).bits) = some True
        apply congrArg some
        apply propext
        decide
      let locationExpr : StorageLocationExpr := {
        root := "values"
        projections := [.mappingKey { formula := .term 9, ty := .compilerTypeId 6 }]
        valueTy := .compilerTypeId 6
      }
      let location : StorageLocation := {
        root := "values"
        projections := [.mappingKey zero]
      }
      let writeResult : StorageActionResult := {
        state := { initial with loop := { initial.loop with
          storage := initial.loop.storage.set location zero } }
        writeTrace := [location]
      }
      have hKeyValue : context.evalValue? (.term 9) initial = some zero := by
        apply context.evalValue_free_variable initial bound 9
          { id := { file_id := 0, pattern_id := 0 }, name := "write_key",
            ty := some (.compilerTypeId 6) }
          zero hBound emitted_term_9_lookup hBoundWrite
      have hStoredValue : context.evalValue? (.term 10) initial = some zero := by
        apply context.evalValue_free_variable initial bound 10
          { id := { file_id := 0, pattern_id := 2 }, name := "value",
            ty := some (.compilerTypeId 6) }
          zero hBound emitted_term_10_lookup hBoundValue
      have hLocation : context.evalLocation? locationExpr initial = some location := by
        have hRoot : "values".isEmpty = false := by decide
        have hValueTy : ((TyRef.compilerTypeId 6).isBool ||
            (TyRef.compilerTypeId 6).integerShape?.isSome) = true := by decide
        have hEmpty : context.evalProjections? initial [] = some [] := by rfl
        have hZeroShape : (Ora.Integer.Value.ofNat
            (.unsigned .w256) 0).shape = .unsigned .w256 := rfl
        unfold StorageFormulaContext.evalLocation?
          StorageFormulaContext.evalProjections?
          StorageFormulaContext.evalTypedValue?
        simp [locationExpr, location, hRoot, hValueTy, hKeyValue,
          storageValueMatchesTy, zero, hEmpty, hZeroShape]
      have hWrite : execStorageAction? context.actionSemantics
          (.write locationExpr (.term 10)) initial = some writeResult := by
        unfold execStorageAction?
        simp only [StorageFormulaContext.actionSemantics, hLocation,
          hStoredValue, Option.bind_eq_bind, Option.some_bind,
          storageValueMatchesTy, zero, TyRef.integerShape?,
          integerShapeForCompilerTypeId_u256, beq_self_eq_true,
          Bool.true_eq_false, if_false, Option.pure_def, Option.some.injEq]
        rfl
      have hWriteAfterControl : execStorageAction? context.actionSemantics
          (.write locationExpr (.term 10))
          ({ state := ({ state := initial } : TimelineResult).state } :
            TimelineResult).state = some writeResult := by
        simpa using hWrite
      let storageEventResult : TimelineResult := {
            state := writeResult.state
            writes := fun observed => observed ∈ writeResult.writeTrace
          }
      let storageTailResult : TimelineResult := storageEventResult.comp {
        state := writeResult.state
      }
      let checkTailResult : TimelineResult :=
        ({ state := initial } : TimelineResult).comp storageTailResult
      let finalResult : TimelineResult :=
        ({ state := initial } : TimelineResult).comp checkTailResult
      have hTimeline : TimelineExec (directStoragePreheaderSemantics context)
          [.assume (.term 5), .check (.term 8),
            .storage (.write locationExpr (.term 10))]
          initial finalResult :=
        .cons (.assume hPath trivial)
          (.cons (.check hCheck)
            (.cons (.storage hWriteAfterControl) (.nil writeResult.state)))
      refine ⟨finalResult, ?_, ?_, ?_, ?_⟩
      · simpa [locationExpr, storageEventResult, storageTailResult,
          checkTailResult, finalResult, directStorageProjectionData,
          TimelineResult.comp] using hTimeline
      · intro before formula after hEvents beforeResult hBefore
        simp only [List.append_nil, List.nil_append, List.cons_append] at hEvents
        rcases before with _ | ⟨head, tail⟩
        · have hFirst := (List.cons.inj hEvents).1
          cases hFirst
        · rcases tail with _ | ⟨second, rest⟩
          · have hFirst := List.cons.inj hEvents
            have hSecond := List.cons.inj hFirst.2
            cases hFirst.1
            cases hSecond.1
            cases hBefore with
            | cons hEvent hTail =>
              cases hEvent
              cases hTail
              exact ⟨True, hCheck, trivial⟩
          · rcases rest with _ | ⟨third, remaining⟩
            · have hFirst := List.cons.inj hEvents
              have hSecond := List.cons.inj hFirst.2
              have hThird := List.cons.inj hSecond.2
              cases hThird.1
            · have hFirst := List.cons.inj hEvents
              have hSecond := List.cons.inj hFirst.2
              have hThird := List.cons.inj hSecond.2
              cases remaining <;> cases hThird.2
      · change directInitialValuesAfterPreheader? context writeResult.state
          [.term 11] = some [Ora.Integer.Value.ofNat (.unsigned .w256) 0]
        unfold directInitialValuesAfterPreheader?
        let bound := context.withStoreViews context.entryEnv writeResult.state
        have hBound : context.stateEnv? writeResult.state = some bound := by rfl
        have hValue : context.evalValue? (.term 11) writeResult.state = some zero := by
          apply evalValueIntegerLiteral context writeResult.state bound 11
            { value := "0", ty := some (.compilerTypeId 6) }
            (Ora.Integer.Value.ofNat (.unsigned .w256) 0)
            hBound emitted_term_11_lookup
          exact integerLiteral_u256_zero
        rw [hValue]
        rfl
      · rfl
  · intro env hAssumptions hContext
    have hWriteBinding := hContext {
      index := 0
      id := { file_id := 0, pattern_id := 0 }
      name := "write_key"
      ty := .compilerTypeId 6
    } (by simp [emittedLoopSummary_1])
    have hObservedBinding := hContext {
      index := 1
      id := { file_id := 0, pattern_id := 1 }
      name := "observed_key"
      ty := .compilerTypeId 6
    } (by simp [emittedLoopSummary_1])
    simp only [TyRef.integerShape?, integerShapeForCompilerTypeId_u256]
      at hWriteBinding hObservedBinding
    rcases hWriteBinding with ⟨writeKey, hWriteShape, hWrite⟩
    rcases hObservedBinding with ⟨observedKey, hObservedShape, hObserved⟩
    have hAssumptions' := hAssumptions
    simp only [assumptionsDenoteInEnv, assumptionsDenoteInEnv?,
      assumptionDenotesInEnv?, assumptionAnd?, emittedAssumptionRow_3,
      formulaDenotes?, denoteFormula?, denoteValue?, emitted_term_0_lookup,
      emitted_term_1_lookup, emitted_term_2_lookup, Env.lookupVar,
      Env.lookupFree, Option.bind_eq_bind, Option.some_bind,
      Option.none_bind, Option.pure_def] at hAssumptions'
    simp only [Env.lookupFree] at hWrite hObserved
    simp only [hWrite, hObserved, Value.eqProp?, Ora.Integer.Value.eqProp?]
      at hAssumptions'
    have hShapes : writeKey.shape = observedKey.shape :=
      (Option.some.inj hWriteShape).symm.trans (Option.some.inj hObservedShape)
    have hWriteUnsigned : writeKey.shape = .unsigned .w256 :=
      (Option.some.inj hWriteShape).symm
    simp [hShapes] at hAssumptions'
    have hDistinct : (Value.integer writeKey) ≠ .integer observedKey := by
      intro hEqual
      cases hEqual
      exact hAssumptions' rfl
    simp only [denoteDirectStorageLoopSummary?, hDirect, if_true]
    simp only [denoteDirectStorageEdges?, denoteDirectStorageEdge?,
      directStorageTimeline?, directStorageEvent?, directStorageLocation?,
      directStorageProjection?, directStorageProjections?, denoteControlTarget?,
      emittedLoopSummary_1, Option.bind_eq_bind, Option.some_bind,
      Option.pure_def, List.isEmpty, List.map, List.append, Bool.not_false,
      Bool.not_true, Bool.false_or, Bool.false_eq_true, if_false]
    refine { base := ?_, backedge := ?_, edgeSafety := ?_, exit := ?_ }
    · intro state hInitial
      rcases hInitial with ⟨hHeaderFacts, hInitial⟩
      change directStoragePreheaderProducesInitial
        (directStoragePreheaderContext emittedManifest env
          emittedLoopSummary_1.contextVariables)
        (directStorageTimeline? emittedLoopSummary_1.preheaderTimeline).get!
        (directStoragePreheaderInitialState env) [.term 11] state at hInitial
      rcases hInitial with ⟨result, hExec, _, hLocals, hStorage⟩
      have hPreserved := preheaderPreservesObserved env writeKey observedKey
        hWrite hWriteUnsigned hDistinct result hExec
      let preContext := directStoragePreheaderContext emittedManifest env
        emittedLoopSummary_1.contextVariables
      have hResultLocals : state.locals =
          [Ora.Integer.Value.ofNat (.unsigned .w256) 0] := by
        unfold directInitialValuesAfterPreheader? at hLocals
        cases hBound : preContext.stateEnv? result.state with
        | none => simp [preContext, hBound, StorageFormulaContext.evalValue?] at hLocals
        | some bound =>
            have hValue : preContext.evalValue? (.term 11) result.state =
                some zero := by
              apply evalValueIntegerLiteral preContext result.state bound 11
                { value := "0", ty := some (.compilerTypeId 6) }
                (Ora.Integer.Value.ofNat (.unsigned .w256) 0)
                hBound emitted_term_11_lookup
              exact integerLiteral_u256_zero
            rw [hValue] at hLocals
            exact (Option.some.inj hLocals).symm
      have hObservedStorage : state.storage {
          root := "values"
          projections := [.mappingKey (.integer observedKey)]
        } = (storageStateFromEnv env true) {
          root := "values"
          projections := [.mappingKey (.integer observedKey)]
        } := by
        rw [hStorage, hPreserved]
        rfl
      rcases state with ⟨locals, storage⟩
      change locals = [Ora.Integer.Value.ofNat (.unsigned .w256) 0]
        at hResultLocals
      subst locals
      have hReady : stateReady emittedLoopSummary_1.variables
          [Ora.Integer.Value.ofNat (.unsigned .w256) 0] := by
        simp [stateReady, valuesMatchVariables, emittedLoopSummary_1,
          TyRef.integerShape?, integerShapeForCompilerTypeId_u256,
          Ora.Integer.Value.ofNat]
      constructor
      · exact hReady
      · constructor
        · exact hHeaderFacts
        · apply (mappingInvariantIff env
          { locals := [Ora.Integer.Value.ofNat (.unsigned .w256) 0], storage }
          observedKey hObserved hReady).mpr
          rw [hObservedStorage]
          generalize hValue : (storageStateFromEnv env true) {
            root := "values"
            projections := [.mappingKey (.integer observedKey)]
          } = value
          cases value with
          | bool value => simp [hValue, Value.eqProp?]
          | integer value =>
              rcases value with ⟨shape, bits⟩
              simp [hValue, Value.eqProp?, Ora.Integer.Value.eqProp?]
    · intro edge current following hMember hTarget hInvariant _ _ hNext
      simp only [List.mem_cons, List.not_mem_nil, or_false] at hMember
      rcases hMember with rfl | rfl
      · dsimp at hNext ⊢
        rcases hNext with ⟨result, locals, returned, hExec, hLocals, hReturn, rfl⟩
        rcases hInvariant with
          ⟨hInvariantReady, hHeaderFacts, hInvariantFormulas⟩
        let variables := emittedLoopSummary_1.variables
        let bindings := emittedLoopSummary_1.edges[0].nextState
        let context := directStorageContext emittedManifest env variables
        change directStorageFinalLocals? context variables bindings result.state =
          some locals at hLocals
        unfold directStorageFinalLocals? at hLocals
        cases hEnv : context.stateEnv? {
          result.state with
          reads := storageReadsForBindings bindings result.state.reads
        } with
        | none => simp [hEnv] at hLocals
        | some bound =>
            simp only [hEnv, Option.bind_eq_bind, Option.some_bind] at hLocals
            have hReady := bound_state_in_env_ready emittedManifest variables bound
              bindings locals hLocals
            let observedLocation : StorageLocation := {
              root := "values"
              projections := [.mappingKey (.integer observedKey)]
            }
            have hPreserved : result.state.loop.storage observedLocation =
                current.storage observedLocation := by
              apply TimelineExec.preserves_outside _ _ _ _ _ hExec
              intro hWritten
              cases hExec with
              | cons h0 hTail0 =>
                cases hTail0 with
                | cons h1 hTail1 =>
                  cases hTail1 with
                  | cons h2 hTail2 =>
                    cases hTail2
                    cases h0
                    cases h1 with
                    | @storage _ _ read0Result hRead0 =>
                      cases h2 with
                      | @storage _ _ read1Result hRead1 =>
                        have hRead0Empty :=
                          execStorageRead_writeTrace_nil _ _ _ _ _ hRead0
                        have hRead1Empty :=
                          execStorageRead_writeTrace_nil _ _ _ _ _ hRead1
                        simp [TimelineResult.comp, hRead0Empty,
                          hRead1Empty] at hWritten
            have hCurrentSemantic :=
              (mappingInvariantIff env current observedKey hObserved
                hInvariantReady).mp hInvariantFormulas
            constructor
            · exact hReady
            · constructor
              · exact hHeaderFacts
              · apply (mappingInvariantIff env
                  { locals, storage := result.state.loop.storage }
                  observedKey hObserved hReady).mpr
                rw [hPreserved]
                exact hCurrentSemantic
      · simp [timelineEdge] at hTarget
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
    let declared : PlaceRef := {
      root := "values"
      region := .storage
      keys := [.parameter { file_id := 0, pattern_id := 0 }]
    }
    let actual : PlaceRef := {
      root := "values"
      region := .storage
      keys := [.parameter { file_id := 0, pattern_id := 0 }]
    }
    exact effect_frame_write_covered_denotes emittedManifest env
      [declared] [actual] (by decide)

theorem discharge_q13 : emittedQuery_13 := by
  unfold emittedQuery_13 obligationFollowsFromAssumptions
  constructor
  · rcases discharge_q11.1 with ⟨env, _, hAssumptions, _, _⟩
    exact ⟨env, hAssumptions⟩
  · intro env hAssumptions
    have hRequired :
        match formulaDenotes? emittedManifest env (.term 2) with
        | some proposition => proposition
        | none => False := by
      cases hFormula : formulaDenotes? emittedManifest env (.term 2) <;>
        simpa only [assumptionsDenoteInEnv, assumptionsDenoteInEnv?,
          assumptionDenotesInEnv?, assumptionAnd?, emittedAssumptionRow_3,
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
    have hPath : keyEvidencePathMatches readPlace writePlace 0 evidence.lhs
        evidence.rhs = true := by
      decide
    have hAssumptionLookup : emittedManifest.assumptionById 3 =
        some emittedAssumptionRow_3 := by rfl
    have hRequiresKind :
        (AssumptionKind.requires != AssumptionKind.requires) = false := by decide
    have hNeOp : (BinaryOp.ne != BinaryOp.ne) = false := by decide
    have hEvidence : keyDisjointEvidenceDenotes? emittedManifest env evidence =
        formulaDenotes? emittedManifest env (.term 2) := by
      unfold keyDisjointEvidenceDenotes? keyDisjointEvidenceFormulaDenotes?
      simp only [evidence, hPath, Bool.true_eq_false, Bool.false_eq_true,
        if_false, hAssumptionLookup, emittedAssumptionRow_3,
        emitted_term_2_lookup, emitted_term_0_lookup, emitted_term_1_lookup,
        termFreeVarId?, freeVarPairMatches, beq_self_eq_true, Bool.true_and,
        Bool.true_or, if_true, hRequiresKind, hNeOp]
    have option_prop_and_true_right_intro (candidate : Option Prop) :
        (match candidate with | some proposition => proposition | none => False) →
        (match optionPropAnd? candidate (some True) with
        | some proposition => proposition
        | none => False) := by
      cases candidate <;> simp [optionPropAnd?]
    unfold obligationDenotesInEnv obligationDenotesInEnv?
    simp only [emittedObligationRow_12, effectFrameGoalDenotes?,
      placeListDisjointWithEvidence?, List.foldl]
    apply option_prop_and_true_left_intro
    simp only [placePairDisjointWithEvidence?, readPlace, writePlace,
      hNotDisjoint, Bool.false_eq_true, if_false, hCovered, if_true]
    apply option_prop_and_true_right_intro
    rw [hEvidence]
    exact hRequired

end Ora.UserProof.MappingStoragePreheader

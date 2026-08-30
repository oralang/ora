namespace Ora.UserProof.DirectStorageLoop

open Ora.Obligation
open Ora.Generated.Obligations.Source_direct_storage_loop_certificate_becafa6bdc0dbf90

attribute [local simp] emitted_manifest_wf emitted_terms_topological emitted_terms_length
attribute [local simp] emitted_term_0_lookup emitted_term_1_lookup emitted_term_2_lookup
  emitted_term_3_lookup emitted_term_4_lookup emitted_term_5_lookup emitted_term_6_lookup
  emitted_term_7_lookup emitted_term_8_lookup emitted_term_9_lookup emitted_term_10_lookup
  emitted_term_11_lookup

theorem directFormulasHold_single_bool_true
    (context : Ora.Loop.StorageFormulaContext)
    (entry : Ora.Loop.StorageState)
    (state : Ora.Loop.LoopState)
    (termId : TermId)
    (bound : Env)
    (hBound : context.stateEnv? { loop := state, entryStorage := entry } = some bound)
    (hTerm : context.manifest.terms[termId]? = some (.boolLit true)) :
    Ora.Loop.directFormulasHold context entry state [.term termId] := by
  intro formula hFormula
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hFormula
  subst formula
  refine ⟨True, ?_, trivial⟩
  exact context.evalFormula_bool_true
    { loop := state, entryStorage := entry } bound termId hBound hTerm

theorem discharge_q8 : emittedQuery_8 := by
  unfold emittedQuery_8 Ora.Loop.directStorageLoopProofAuthorizationFromAssumptions
  have hSupported : emittedLoopSummary_1.supported emittedManifest = true := by decide
  have hDirect : Ora.Loop.directStorageSummarySupported emittedManifest emittedLoopSummary_1 = true := by
    simp only [Ora.Loop.directStorageSummarySupported, hSupported, emittedLoopSummary_1]
    decide
  have hCounterNotEmpty : "counter".isEmpty = false := by decide
  constructor
  · refine ⟨Env.empty, {
      locals := [Ora.Integer.Value.ofNat (.unsigned .w256) 0]
      storage := Ora.Loop.storageStateFromEnv Env.empty false
    }, ?_, ?_, ?_⟩
    · simp [assumptionsDenoteInEnv, assumptionsDenoteInEnv?]
    · simp [Ora.Loop.contextReady, emittedLoopSummary_1]
    · simp only [Ora.Loop.denoteDirectStorageLoopSummary?, hDirect, if_true]
      simp only [
        Ora.Loop.denoteDirectStorageEdges?, Ora.Loop.denoteDirectStorageEdge?,
        Ora.Loop.directStorageTimeline?, Ora.Loop.directStorageEvent?,
        Ora.Loop.directStorageLocation?, Ora.Loop.denoteControlTarget?,
        Option.bind_eq_bind, Option.some_bind, Option.pure_def,
        emittedLoopSummary_1, hCounterNotEmpty, List.isEmpty, List.map,
        List.append, Bool.not_false, Bool.not_true, Bool.false_or, Bool.false_eq_true,
        if_false]
      constructor
      · trivial
      · rfl
  · intro env _ _
    simp only [Ora.Loop.denoteDirectStorageLoopSummary?, hDirect, if_true]
    simp only [
      Ora.Loop.denoteDirectStorageEdges?, Ora.Loop.denoteDirectStorageEdge?,
      Ora.Loop.directStorageTimeline?, Ora.Loop.directStorageEvent?,
      Ora.Loop.directStorageLocation?, Ora.Loop.denoteControlTarget?,
      Option.bind_eq_bind, Option.some_bind, Option.pure_def,
      emittedLoopSummary_1, hCounterNotEmpty, List.isEmpty, List.map,
      List.append, Bool.not_false, Bool.not_true, Bool.false_or, Bool.false_eq_true,
      if_false]
    constructor
    · simp only [Ora.Loop.LoopVC.BaseObligation]
      intro state hInitial
      rcases hInitial with ⟨_, hLocals⟩
      rcases state with ⟨locals, storage⟩
      change some [Ora.Integer.Value.ofNat (.unsigned .w256) 0] =
        some locals at hLocals
      have hState := (Option.some.inj hLocals).symm
      subst locals
      constructor
      · rfl
      · let context := Ora.Loop.directStorageContext emittedManifest env
          emittedLoopSummary_1.variables
        let actionState : Ora.Loop.StorageActionState := {
          loop := {
            locals := [Ora.Integer.Value.ofNat (.unsigned .w256) 0]
            storage
          }
          entryStorage := Ora.Loop.storageStateFromEnv env true
        }
        let bound := context.withStoreViews
          (env.setFree { file_id := 4294967294, pattern_id := 0 }
            (.integer (Ora.Integer.Value.ofNat (.unsigned .w256) 0))) actionState
        apply directFormulasHold_single_bool_true context
          (Ora.Loop.storageStateFromEnv env true) actionState.loop 4 bound
        · rfl
        · exact emitted_term_4_lookup
    · simp only [Ora.Loop.LoopVC.BackedgeObligation, Ora.Loop.timelineEdge]
      intro edge current following hMember hTarget _ _ _ hNext
      simp only [List.mem_cons, List.not_mem_nil, or_false] at hMember
      rcases hMember with rfl | rfl
      · dsimp at hNext ⊢
        rcases hNext with ⟨result, locals, hExec, hLocals, rfl⟩
        let variables := emittedLoopSummary_1.variables
        let bindings := emittedLoopSummary_1.edges[0].nextState
        let context := Ora.Loop.directStorageContext emittedManifest env variables
        change Ora.Loop.directStorageFinalLocals? context variables bindings result.state =
          some locals at hLocals
        unfold Ora.Loop.directStorageFinalLocals? at hLocals
        cases hEnv : context.stateEnv? result.state with
        | none => simp [hEnv] at hLocals
        | some bound =>
            simp only [hEnv, Option.bind_eq_bind, Option.some_bind] at hLocals
            have hReady := Ora.Loop.bound_state_in_env_ready emittedManifest
              variables bound bindings locals hLocals
            constructor
            · exact hReady
            · rcases locals with _ | ⟨value, tail⟩
              · simp [Ora.Loop.stateReady, Ora.Loop.valuesMatchVariables] at hReady
              · rcases tail with _ | ⟨extra, rest⟩
                · rcases value with ⟨shape, bits⟩
                  simp [Ora.Loop.stateReady, Ora.Loop.valuesMatchVariables,
                    TyRef.integerShape?] at hReady
                  subst shape
                  let actionState : Ora.Loop.StorageActionState := {
                    loop := {
                      locals := [⟨.unsigned .w256, bits⟩]
                      storage := result.state.loop.storage
                    }
                    entryStorage := Ora.Loop.storageStateFromEnv env true
                  }
                  let formulaBound := context.withStoreViews
                    (env.setFree { file_id := 4294967294, pattern_id := 0 }
                      (.integer ⟨.unsigned .w256, bits⟩)) actionState
                  apply directFormulasHold_single_bool_true context
                    (Ora.Loop.storageStateFromEnv env true) actionState.loop 4 formulaBound
                  · rfl
                  · exact emitted_term_4_lookup
                · simp [Ora.Loop.stateReady, Ora.Loop.valuesMatchVariables] at hReady
      · simp at hTarget
    · simp only [Ora.Loop.LoopVC.EdgeSafetyObligation, Ora.Loop.timelineEdge]
      intro edge state hMember _ _
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
    · simp only [Ora.Loop.LoopVC.ExitTargetObligation, Ora.Loop.timelineEdge]
      intro edge current following hMember hNotBackedge _ _ _ _
      simp only [List.mem_cons, List.not_mem_nil, or_false] at hMember
      rcases hMember with rfl | rfl
      · exact False.elim (hNotBackedge rfl)
      · intro formula hFormula
        simp at hFormula

end Ora.UserProof.DirectStorageLoop

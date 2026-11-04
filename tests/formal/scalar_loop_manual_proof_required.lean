namespace Ora.UserProof.ScalarLoopManual
open Ora.Obligation
open Ora.Generated.Obligations.Source_scalar_loop_manual_proof_required_f3f3e5cd452071ac

namespace Query_12
section
private abbrev autoSummary_q12 : Ora.Loop.SummaryRow := emittedLoopSummary_1
local notation "autoSummary" => autoSummary_q12
attribute [local simp] autoSummary_q12 emittedLoopSummary_1
attribute [local irreducible] emittedManifest
attribute [local simp] emitted_manifest_wf emitted_terms_topological emitted_terms_length emitted_term_0_lookup emitted_term_1_lookup emitted_term_2_lookup emitted_term_3_lookup emitted_term_4_lookup emitted_term_5_lookup emitted_term_6_lookup emitted_term_7_lookup emitted_term_8_lookup emitted_term_9_lookup emitted_term_10_lookup emitted_term_11_lookup emitted_term_33_lookup emitted_term_34_lookup emitted_term_35_lookup emitted_term_36_lookup emitted_term_37_lookup emitted_term_38_lookup emitted_term_39_lookup emitted_term_40_lookup emitted_term_41_lookup emitted_term_42_lookup emitted_term_43_lookup
theorem discharge_q12 : emittedQuery_12 := by
  unfold emittedQuery_12 Ora.Loop.loopProofAuthorizationFromAssumptions
  have h_auto_free_0_0 : (({ file_id := 4294967294, pattern_id := 0 } : FreeVarId) == { file_id := 4294967294, pattern_id := 0 }) = true := by rfl
  have hSupported : emittedLoopSummary_1.supported emittedManifest = true := by
    have hShape : emittedLoopSummary_1.shapeSupported = true := by decide
    have hFormulas : emittedLoopSummary_1.formulasDenotable emittedManifest = true := by
      simp only [Ora.Loop.SummaryRow.formulasDenotable,
        Ora.Loop.SummaryRow.supportEnv, Ora.Loop.bindSupportVariable,
        Ora.Loop.valueForTy, Ora.Loop.edgeFormulasDenotable,
        Ora.Loop.formulaDenotable, Ora.Loop.valueDenotable, List.all,
        emittedLoopSummary_1, TyRef.integerShape?, integerShapeForCompilerTypeId_u8, integerShapeForCompilerTypeId_u16, integerShapeForCompilerTypeId_u32, integerShapeForCompilerTypeId_u64, integerShapeForCompilerTypeId_u128, integerShapeForCompilerTypeId_u160, integerShapeForCompilerTypeId_u256, integerShapeForCompilerTypeId_i8, integerShapeForCompilerTypeId_i16, integerShapeForCompilerTypeId_i32, integerShapeForCompilerTypeId_i64, integerShapeForCompilerTypeId_i128, integerShapeForCompilerTypeId_i256, h_auto_free_0_0, Env.setFree, Env.lookupFree, lookupFreeBinding, if_pos, if_false,
        formulaDenotes?, formulaValue?, denoteFormula?, denoteValue?,
        IntegerLiteralTerm.asInteger?, Ora.Integer.Value.ofNat,
        Ora.Integer.Value.relation?, Ora.Integer.Value.binary?,
        Ora.Integer.Value.bitsFor?, Ora.Integer.wrappingResult,
        Ora.Integer.bitAndResult, Ora.Integer.bitXorResult,
        Ora.Spec.IntegerShape.isSigned, Env.lookupVar, emitted_terms_length, emitted_term_0_lookup, emitted_term_1_lookup, emitted_term_2_lookup, emitted_term_3_lookup, emitted_term_4_lookup, emitted_term_5_lookup, emitted_term_6_lookup, emitted_term_7_lookup, emitted_term_8_lookup, emitted_term_9_lookup, emitted_term_10_lookup, emitted_term_11_lookup, emitted_term_33_lookup, emitted_term_34_lookup, emitted_term_35_lookup, emitted_term_36_lookup, emitted_term_37_lookup, emitted_term_38_lookup, emitted_term_39_lookup, emitted_term_40_lookup, emitted_term_41_lookup, emitted_term_42_lookup, emitted_term_43_lookup, Bool.false_eq_true]
      decide
    have hFixedIndexBounds :
        emittedLoopSummary_1.fixedIndexBoundsSupported emittedManifest = true := by
      decide
    simp only [Ora.Loop.SummaryRow.supported, emitted_manifest_wf,
      emitted_terms_topological, hShape, hFixedIndexBounds, hFormulas, Bool.true_and]
  constructor
  · let autoWitnessEnv : Env := Env.empty
    have hInitialSome :
        (Ora.Loop.initializeState? emittedManifest autoWitnessEnv
          (autoSummary).variables (autoSummary).init).isSome = true := by
      simp only [Ora.Loop.initializeState?,
        Ora.Loop.evaluateIntegerFormulas?, Ora.Loop.integerFormulaValue?,
        Ora.Loop.valuesMatchVariables, autoSummary_q12, emittedLoopSummary_1]
      simp only [formulaValue?, denoteValue?, IntegerLiteralTerm.asInteger?,
        Ora.Integer.Value.ofNat, Ora.Integer.Value.binary?,
        Ora.Integer.Value.bitsFor?, Ora.Integer.wrappingResult,
        Env.setFree, Env.lookupVar, Env.lookupFree, lookupFreeBinding, TyRef.integerShape?, integerShapeForCompilerTypeId_u8, integerShapeForCompilerTypeId_u16, integerShapeForCompilerTypeId_u32, integerShapeForCompilerTypeId_u64, integerShapeForCompilerTypeId_u128, integerShapeForCompilerTypeId_u160, integerShapeForCompilerTypeId_u256, integerShapeForCompilerTypeId_i8, integerShapeForCompilerTypeId_i16, integerShapeForCompilerTypeId_i32, integerShapeForCompilerTypeId_i64, integerShapeForCompilerTypeId_i128, integerShapeForCompilerTypeId_i256, emitted_terms_length, emitted_term_0_lookup, emitted_term_1_lookup, emitted_term_2_lookup, emitted_term_3_lookup, emitted_term_4_lookup, emitted_term_5_lookup, emitted_term_6_lookup, emitted_term_7_lookup, emitted_term_8_lookup, emitted_term_9_lookup, emitted_term_10_lookup, emitted_term_11_lookup, emitted_term_33_lookup, emitted_term_34_lookup, emitted_term_35_lookup, emitted_term_36_lookup, emitted_term_37_lookup, emitted_term_38_lookup, emitted_term_39_lookup, emitted_term_40_lookup, emitted_term_41_lookup, emitted_term_42_lookup, emitted_term_43_lookup, h_auto_free_0_0, if_pos, if_false, Bool.false_eq_true]
      decide
    cases hInitial : Ora.Loop.initializeState? emittedManifest autoWitnessEnv
        (autoSummary).variables (autoSummary).init with
    | none =>
        rw [hInitial] at hInitialSome
        simp at hInitialSome
    | some initial =>
        refine ⟨autoWitnessEnv, initial, ?_⟩
        constructor
        · simp [assumptionsDenoteInEnv, assumptionsDenoteInEnv?]
        constructor
        ·
          simp [Ora.Loop.contextReady]
        · simp only [Ora.Loop.loopInitialPremises,
            Ora.Loop.denoteLoopSummary?, hSupported, Ora.Loop.denoteEdges?,
            Ora.Loop.denoteEdge?, Ora.Loop.denoteControlTarget?,
            Ora.Loop.EdgeRow.path_eq_timeline,
            Ora.Loop.EdgeRow.safety_eq_timeline,
            Ora.Loop.timelinePathView, Ora.Loop.timelineSafetyView,
            Ora.Loop.timelineSafetyViewFrom]
          exact ⟨by simp [Ora.Loop.headerFactsHoldInEnv], hInitial⟩
  · intro env hAssumptions hContext
    simp only [Ora.Loop.loopInductionObligationsAtEnv,
      Ora.Loop.denoteLoopSummary?, hSupported, Ora.Loop.denoteEdges?,
      Ora.Loop.denoteEdge?, Ora.Loop.denoteControlTarget?,
      Ora.Loop.EdgeRow.path_eq_timeline,
      Ora.Loop.EdgeRow.safety_eq_timeline,
      Ora.Loop.timelinePathView, Ora.Loop.timelineSafetyView,
      Ora.Loop.timelineSafetyViewFrom]
    constructor
    · intro state hInitial
      dsimp at hInitial ⊢
      rcases hInitial with ⟨hHeaderFacts, hInitial⟩
      have hReady := Ora.Loop.initialized_state_ready emittedManifest env
        (autoSummary).variables (autoSummary).init state hInitial
      simp only [Ora.Loop.valuesInitializeState, Ora.Loop.initializeState?, Ora.Loop.evaluateIntegerFormulas?, Ora.Loop.integerFormulaValue?, Ora.Loop.valuesMatchVariables] at hInitial
      simp only [formulaValue?, denoteValue?, IntegerLiteralTerm.asInteger?, Ora.Integer.Value.ofNat, Ora.Integer.Value.binary?, Ora.Integer.Value.bitsFor?, Ora.Integer.wrappingResult, Ora.Integer.bitAndResult, Ora.Integer.bitXorResult, Env.setFree, Env.lookupVar, Env.lookupFree, lookupFreeBinding, TyRef.integerShape?, integerShapeForCompilerTypeId_u8, integerShapeForCompilerTypeId_u16, integerShapeForCompilerTypeId_u32, integerShapeForCompilerTypeId_u64, integerShapeForCompilerTypeId_u128, integerShapeForCompilerTypeId_u160, integerShapeForCompilerTypeId_u256, integerShapeForCompilerTypeId_i8, integerShapeForCompilerTypeId_i16, integerShapeForCompilerTypeId_i32, integerShapeForCompilerTypeId_i64, integerShapeForCompilerTypeId_i128, integerShapeForCompilerTypeId_i256, emitted_terms_length, emitted_term_0_lookup, emitted_term_1_lookup, emitted_term_2_lookup, emitted_term_3_lookup, emitted_term_4_lookup, emitted_term_5_lookup, emitted_term_6_lookup, emitted_term_7_lookup, emitted_term_8_lookup, emitted_term_9_lookup, emitted_term_10_lookup, emitted_term_11_lookup, emitted_term_33_lookup, emitted_term_34_lookup, emitted_term_35_lookup, emitted_term_36_lookup, emitted_term_37_lookup, emitted_term_38_lookup, emitted_term_39_lookup, emitted_term_40_lookup, emitted_term_41_lookup, emitted_term_42_lookup, emitted_term_43_lookup, h_auto_free_0_0, if_pos, if_false, Bool.false_eq_true] at hInitial
      change some [Ora.Integer.Value.mk (.unsigned .w256) (BitVec.ofNat 256 0)] = some state at hInitial
      have hState := (Option.some.inj hInitial).symm
      subst state
      let auto_following_bits_0 := (BitVec.ofNat 256 0)
      refine ⟨hReady, hHeaderFacts, ?_⟩
      intro formula hFormula
      simp at hFormula
      rcases hFormula with hFormula | hFormula <;> subst formula
      ·
        simp only [Ora.Loop.formulaHoldsInState, Ora.Loop.bindState?, TyRef.integerShape?, integerShapeForCompilerTypeId_u8, integerShapeForCompilerTypeId_u16, integerShapeForCompilerTypeId_u32, integerShapeForCompilerTypeId_u64, integerShapeForCompilerTypeId_u128, integerShapeForCompilerTypeId_u160, integerShapeForCompilerTypeId_u256, integerShapeForCompilerTypeId_i8, integerShapeForCompilerTypeId_i16, integerShapeForCompilerTypeId_i32, integerShapeForCompilerTypeId_i64, integerShapeForCompilerTypeId_i128, integerShapeForCompilerTypeId_i256, h_auto_free_0_0, if_pos]
        simp only [formulaDenotes?, denoteFormula?, denoteValue?, IntegerLiteralTerm.asInteger?, Ora.Integer.Value.ofNat, Ora.Integer.Value.relation?, Ora.Integer.Value.bitsFor?, Ora.Integer.Value.binary?, Ora.Integer.wrappingResult, Ora.Integer.bitAndResult, Ora.Integer.bitXorResult, Ora.Spec.IntegerShape.isSigned, Env.setFree, Env.lookupVar, Env.lookupFree, lookupFreeBinding, TyRef.integerShape?, integerShapeForCompilerTypeId_u8, integerShapeForCompilerTypeId_u16, integerShapeForCompilerTypeId_u32, integerShapeForCompilerTypeId_u64, integerShapeForCompilerTypeId_u128, integerShapeForCompilerTypeId_u160, integerShapeForCompilerTypeId_u256, integerShapeForCompilerTypeId_i8, integerShapeForCompilerTypeId_i16, integerShapeForCompilerTypeId_i32, integerShapeForCompilerTypeId_i64, integerShapeForCompilerTypeId_i128, integerShapeForCompilerTypeId_i256, emitted_terms_length, emitted_term_0_lookup, emitted_term_1_lookup, emitted_term_2_lookup, emitted_term_3_lookup, emitted_term_4_lookup, emitted_term_5_lookup, emitted_term_6_lookup, emitted_term_7_lookup, emitted_term_8_lookup, emitted_term_9_lookup, emitted_term_10_lookup, emitted_term_11_lookup, emitted_term_33_lookup, emitted_term_34_lookup, emitted_term_35_lookup, emitted_term_36_lookup, emitted_term_37_lookup, emitted_term_38_lookup, emitted_term_39_lookup, emitted_term_40_lookup, emitted_term_41_lookup, emitted_term_42_lookup, emitted_term_43_lookup, h_auto_free_0_0, if_pos, if_false, Bool.false_eq_true]
        change (Ora.Integer.unsignedLe (shape := (.unsigned .w256)) auto_following_bits_0 (BitVec.ofNat 256 3))
        first
        | bv_decide
        | (simp only [auto_following_bits_0]; apply Ora.Integer.signed_add_one_le_of_lt; assumption)
        | (apply Ora.Integer.signed_add_one_guard_safe_of_lt; assumption)
        | (simp only [auto_following_bits_0]; apply Ora.Integer.wrapping_add_one_le_of_unsigned_lt; simp_all only [])
        | (apply Ora.Integer.wrapping_add_one_not_lt_self_of_room; simp_all [Ora.Integer.unsignedLt, Ora.Spec.IntegerShape.bits, Ora.Spec.uintBits] <;> omega)
        | (simp_all only [auto_following_bits_0] <;> done)
        | (simp [auto_following_bits_0, Ora.Integer.unsignedLt, Ora.Integer.unsignedLe, Ora.Integer.signedLt, Ora.Integer.signedLe, Ora.Integer.mathematicalValue, Ora.Integer.wrappingResult] at * <;> omega)
      ·
        simp only [Ora.Loop.formulaHoldsInState, Ora.Loop.bindState?, TyRef.integerShape?, integerShapeForCompilerTypeId_u8, integerShapeForCompilerTypeId_u16, integerShapeForCompilerTypeId_u32, integerShapeForCompilerTypeId_u64, integerShapeForCompilerTypeId_u128, integerShapeForCompilerTypeId_u160, integerShapeForCompilerTypeId_u256, integerShapeForCompilerTypeId_i8, integerShapeForCompilerTypeId_i16, integerShapeForCompilerTypeId_i32, integerShapeForCompilerTypeId_i64, integerShapeForCompilerTypeId_i128, integerShapeForCompilerTypeId_i256, h_auto_free_0_0, if_pos]
        simp only [formulaDenotes?, denoteFormula?, denoteValue?, IntegerLiteralTerm.asInteger?, Ora.Integer.Value.ofNat, Ora.Integer.Value.relation?, Ora.Integer.Value.bitsFor?, Ora.Integer.Value.binary?, Ora.Integer.wrappingResult, Ora.Integer.bitAndResult, Ora.Integer.bitXorResult, Ora.Spec.IntegerShape.isSigned, Env.setFree, Env.lookupVar, Env.lookupFree, lookupFreeBinding, TyRef.integerShape?, integerShapeForCompilerTypeId_u8, integerShapeForCompilerTypeId_u16, integerShapeForCompilerTypeId_u32, integerShapeForCompilerTypeId_u64, integerShapeForCompilerTypeId_u128, integerShapeForCompilerTypeId_u160, integerShapeForCompilerTypeId_u256, integerShapeForCompilerTypeId_i8, integerShapeForCompilerTypeId_i16, integerShapeForCompilerTypeId_i32, integerShapeForCompilerTypeId_i64, integerShapeForCompilerTypeId_i128, integerShapeForCompilerTypeId_i256, emitted_terms_length, emitted_term_0_lookup, emitted_term_1_lookup, emitted_term_2_lookup, emitted_term_3_lookup, emitted_term_4_lookup, emitted_term_5_lookup, emitted_term_6_lookup, emitted_term_7_lookup, emitted_term_8_lookup, emitted_term_9_lookup, emitted_term_10_lookup, emitted_term_11_lookup, emitted_term_33_lookup, emitted_term_34_lookup, emitted_term_35_lookup, emitted_term_36_lookup, emitted_term_37_lookup, emitted_term_38_lookup, emitted_term_39_lookup, emitted_term_40_lookup, emitted_term_41_lookup, emitted_term_42_lookup, emitted_term_43_lookup, h_auto_free_0_0, if_pos, if_false, Bool.false_eq_true]
        change (Ora.Integer.unsignedLe (shape := (.unsigned .w256)) (Ora.Integer.wrappingResult (.unsigned .w256) .mul auto_following_bits_0 (BitVec.ofNat 256 1)) (BitVec.ofNat 256 3))
        first
        | bv_decide
        | (simp only [auto_following_bits_0]; apply Ora.Integer.signed_add_one_le_of_lt; assumption)
        | (apply Ora.Integer.signed_add_one_guard_safe_of_lt; assumption)
        | (simp only [auto_following_bits_0]; apply Ora.Integer.wrapping_add_one_le_of_unsigned_lt; simp_all only [])
        | (apply Ora.Integer.wrapping_add_one_not_lt_self_of_room; simp_all [Ora.Integer.unsignedLt, Ora.Spec.IntegerShape.bits, Ora.Spec.uintBits] <;> omega)
        | (simp_all only [auto_following_bits_0] <;> done)
        | (simp [auto_following_bits_0, Ora.Integer.unsignedLt, Ora.Integer.unsignedLe, Ora.Integer.signedLt, Ora.Integer.signedLe, Ora.Integer.mathematicalValue, Ora.Integer.wrappingResult] at * <;> omega)
    · intro edge current following hMember hTarget hInvariant hPath hSafe hNext
      simp at hMember
      rcases hMember with hMember | hMember
      · subst edge
        dsimp at hTarget hInvariant hPath hSafe hNext ⊢
        rcases hInvariant with ⟨hReady, hHeaderFacts, hInvariant⟩
        have hFollowingReady := Ora.Loop.bound_state_ready emittedManifest env
          (autoSummary).variables current following
          (autoSummary).edges[0].nextState hNext
        rcases current with _ | ⟨auto_state_value_0, auto_state_tail_0⟩
        · simp [Ora.Loop.stateReady, Ora.Loop.valuesMatchVariables] at hReady
        ·
          rcases auto_state_tail_0 with _ | ⟨auto_state_extra, auto_state_rest⟩
          ·
            rcases auto_state_value_0 with ⟨auto_state_shape_0, auto_state_bits_0⟩
            simp [Ora.Loop.stateReady, Ora.Loop.valuesMatchVariables] at hReady
            subst auto_state_shape_0
            have h_auto_invariant_0 := hInvariant (.term 6) (by simp)
            simp only [Ora.Loop.formulaHoldsInState, Ora.Loop.bindState?, TyRef.integerShape?, integerShapeForCompilerTypeId_u8, integerShapeForCompilerTypeId_u16, integerShapeForCompilerTypeId_u32, integerShapeForCompilerTypeId_u64, integerShapeForCompilerTypeId_u128, integerShapeForCompilerTypeId_u160, integerShapeForCompilerTypeId_u256, integerShapeForCompilerTypeId_i8, integerShapeForCompilerTypeId_i16, integerShapeForCompilerTypeId_i32, integerShapeForCompilerTypeId_i64, integerShapeForCompilerTypeId_i128, integerShapeForCompilerTypeId_i256, h_auto_free_0_0, if_pos] at h_auto_invariant_0
            simp only [formulaDenotes?, denoteFormula?, denoteValue?, IntegerLiteralTerm.asInteger?, Ora.Integer.Value.ofNat, Ora.Integer.Value.relation?, Ora.Integer.Value.bitsFor?, Ora.Integer.Value.binary?, Ora.Integer.wrappingResult, Ora.Integer.bitAndResult, Ora.Integer.bitXorResult, Ora.Spec.IntegerShape.isSigned, Env.setFree, Env.lookupVar, Env.lookupFree, lookupFreeBinding, TyRef.integerShape?, integerShapeForCompilerTypeId_u8, integerShapeForCompilerTypeId_u16, integerShapeForCompilerTypeId_u32, integerShapeForCompilerTypeId_u64, integerShapeForCompilerTypeId_u128, integerShapeForCompilerTypeId_u160, integerShapeForCompilerTypeId_u256, integerShapeForCompilerTypeId_i8, integerShapeForCompilerTypeId_i16, integerShapeForCompilerTypeId_i32, integerShapeForCompilerTypeId_i64, integerShapeForCompilerTypeId_i128, integerShapeForCompilerTypeId_i256, emitted_terms_length, emitted_term_0_lookup, emitted_term_1_lookup, emitted_term_2_lookup, emitted_term_3_lookup, emitted_term_4_lookup, emitted_term_5_lookup, emitted_term_6_lookup, emitted_term_7_lookup, emitted_term_8_lookup, emitted_term_9_lookup, emitted_term_10_lookup, emitted_term_11_lookup, emitted_term_33_lookup, emitted_term_34_lookup, emitted_term_35_lookup, emitted_term_36_lookup, emitted_term_37_lookup, emitted_term_38_lookup, emitted_term_39_lookup, emitted_term_40_lookup, emitted_term_41_lookup, emitted_term_42_lookup, emitted_term_43_lookup, h_auto_free_0_0, if_pos, if_false, Bool.false_eq_true] at h_auto_invariant_0
            change (Ora.Integer.unsignedLe (shape := (.unsigned .w256)) auto_state_bits_0 (BitVec.ofNat 256 3)) at h_auto_invariant_0
            have h_auto_invariant_1 := hInvariant (.term 11) (by simp)
            simp only [Ora.Loop.formulaHoldsInState, Ora.Loop.bindState?, TyRef.integerShape?, integerShapeForCompilerTypeId_u8, integerShapeForCompilerTypeId_u16, integerShapeForCompilerTypeId_u32, integerShapeForCompilerTypeId_u64, integerShapeForCompilerTypeId_u128, integerShapeForCompilerTypeId_u160, integerShapeForCompilerTypeId_u256, integerShapeForCompilerTypeId_i8, integerShapeForCompilerTypeId_i16, integerShapeForCompilerTypeId_i32, integerShapeForCompilerTypeId_i64, integerShapeForCompilerTypeId_i128, integerShapeForCompilerTypeId_i256, h_auto_free_0_0, if_pos] at h_auto_invariant_1
            simp only [formulaDenotes?, denoteFormula?, denoteValue?, IntegerLiteralTerm.asInteger?, Ora.Integer.Value.ofNat, Ora.Integer.Value.relation?, Ora.Integer.Value.bitsFor?, Ora.Integer.Value.binary?, Ora.Integer.wrappingResult, Ora.Integer.bitAndResult, Ora.Integer.bitXorResult, Ora.Spec.IntegerShape.isSigned, Env.setFree, Env.lookupVar, Env.lookupFree, lookupFreeBinding, TyRef.integerShape?, integerShapeForCompilerTypeId_u8, integerShapeForCompilerTypeId_u16, integerShapeForCompilerTypeId_u32, integerShapeForCompilerTypeId_u64, integerShapeForCompilerTypeId_u128, integerShapeForCompilerTypeId_u160, integerShapeForCompilerTypeId_u256, integerShapeForCompilerTypeId_i8, integerShapeForCompilerTypeId_i16, integerShapeForCompilerTypeId_i32, integerShapeForCompilerTypeId_i64, integerShapeForCompilerTypeId_i128, integerShapeForCompilerTypeId_i256, emitted_terms_length, emitted_term_0_lookup, emitted_term_1_lookup, emitted_term_2_lookup, emitted_term_3_lookup, emitted_term_4_lookup, emitted_term_5_lookup, emitted_term_6_lookup, emitted_term_7_lookup, emitted_term_8_lookup, emitted_term_9_lookup, emitted_term_10_lookup, emitted_term_11_lookup, emitted_term_33_lookup, emitted_term_34_lookup, emitted_term_35_lookup, emitted_term_36_lookup, emitted_term_37_lookup, emitted_term_38_lookup, emitted_term_39_lookup, emitted_term_40_lookup, emitted_term_41_lookup, emitted_term_42_lookup, emitted_term_43_lookup, h_auto_free_0_0, if_pos, if_false, Bool.false_eq_true] at h_auto_invariant_1
            change (Ora.Integer.unsignedLe (shape := (.unsigned .w256)) (Ora.Integer.wrappingResult (.unsigned .w256) .mul auto_state_bits_0 (BitVec.ofNat 256 1)) (BitVec.ofNat 256 3)) at h_auto_invariant_1
            have h_auto_path_0 := hPath (.term 3) (by simp)
            simp only [Ora.Loop.formulaHoldsInState, Ora.Loop.bindState?, TyRef.integerShape?, integerShapeForCompilerTypeId_u8, integerShapeForCompilerTypeId_u16, integerShapeForCompilerTypeId_u32, integerShapeForCompilerTypeId_u64, integerShapeForCompilerTypeId_u128, integerShapeForCompilerTypeId_u160, integerShapeForCompilerTypeId_u256, integerShapeForCompilerTypeId_i8, integerShapeForCompilerTypeId_i16, integerShapeForCompilerTypeId_i32, integerShapeForCompilerTypeId_i64, integerShapeForCompilerTypeId_i128, integerShapeForCompilerTypeId_i256, h_auto_free_0_0, if_pos] at h_auto_path_0
            simp only [formulaDenotes?, denoteFormula?, denoteValue?, IntegerLiteralTerm.asInteger?, Ora.Integer.Value.ofNat, Ora.Integer.Value.relation?, Ora.Integer.Value.bitsFor?, Ora.Integer.Value.binary?, Ora.Integer.wrappingResult, Ora.Integer.bitAndResult, Ora.Integer.bitXorResult, Ora.Spec.IntegerShape.isSigned, Env.setFree, Env.lookupVar, Env.lookupFree, lookupFreeBinding, TyRef.integerShape?, integerShapeForCompilerTypeId_u8, integerShapeForCompilerTypeId_u16, integerShapeForCompilerTypeId_u32, integerShapeForCompilerTypeId_u64, integerShapeForCompilerTypeId_u128, integerShapeForCompilerTypeId_u160, integerShapeForCompilerTypeId_u256, integerShapeForCompilerTypeId_i8, integerShapeForCompilerTypeId_i16, integerShapeForCompilerTypeId_i32, integerShapeForCompilerTypeId_i64, integerShapeForCompilerTypeId_i128, integerShapeForCompilerTypeId_i256, emitted_terms_length, emitted_term_0_lookup, emitted_term_1_lookup, emitted_term_2_lookup, emitted_term_3_lookup, emitted_term_4_lookup, emitted_term_5_lookup, emitted_term_6_lookup, emitted_term_7_lookup, emitted_term_8_lookup, emitted_term_9_lookup, emitted_term_10_lookup, emitted_term_11_lookup, emitted_term_33_lookup, emitted_term_34_lookup, emitted_term_35_lookup, emitted_term_36_lookup, emitted_term_37_lookup, emitted_term_38_lookup, emitted_term_39_lookup, emitted_term_40_lookup, emitted_term_41_lookup, emitted_term_42_lookup, emitted_term_43_lookup, h_auto_free_0_0, if_pos, if_false, Bool.false_eq_true] at h_auto_path_0
            change (Ora.Integer.unsignedLt (shape := (.unsigned .w256)) auto_state_bits_0 (BitVec.ofNat 256 3)) at h_auto_path_0
            simp only [Ora.Loop.stateBindingsProduceState, Ora.Loop.applyStateBindings?, Ora.Loop.bindingsMatchVariables, Ora.Loop.bindState?, Ora.Loop.bindingFormulas, Ora.Loop.evaluateIntegerFormulas?, Ora.Loop.integerFormulaValue?, Ora.Loop.valuesMatchVariables] at hNext
            simp only [formulaValue?, denoteValue?, IntegerLiteralTerm.asInteger?, Ora.Integer.Value.ofNat, Ora.Integer.Value.binary?, Ora.Integer.Value.bitsFor?, Ora.Integer.wrappingResult, Ora.Integer.bitAndResult, Ora.Integer.bitXorResult, Env.setFree, Env.lookupVar, Env.lookupFree, lookupFreeBinding, TyRef.integerShape?, integerShapeForCompilerTypeId_u8, integerShapeForCompilerTypeId_u16, integerShapeForCompilerTypeId_u32, integerShapeForCompilerTypeId_u64, integerShapeForCompilerTypeId_u128, integerShapeForCompilerTypeId_u160, integerShapeForCompilerTypeId_u256, integerShapeForCompilerTypeId_i8, integerShapeForCompilerTypeId_i16, integerShapeForCompilerTypeId_i32, integerShapeForCompilerTypeId_i64, integerShapeForCompilerTypeId_i128, integerShapeForCompilerTypeId_i256, emitted_terms_length, emitted_term_0_lookup, emitted_term_1_lookup, emitted_term_2_lookup, emitted_term_3_lookup, emitted_term_4_lookup, emitted_term_5_lookup, emitted_term_6_lookup, emitted_term_7_lookup, emitted_term_8_lookup, emitted_term_9_lookup, emitted_term_10_lookup, emitted_term_11_lookup, emitted_term_33_lookup, emitted_term_34_lookup, emitted_term_35_lookup, emitted_term_36_lookup, emitted_term_37_lookup, emitted_term_38_lookup, emitted_term_39_lookup, emitted_term_40_lookup, emitted_term_41_lookup, emitted_term_42_lookup, emitted_term_43_lookup, h_auto_free_0_0, if_pos, if_false, Bool.false_eq_true] at hNext
            change some [Ora.Integer.Value.mk (.unsigned .w256) (Ora.Integer.wrappingResult (.unsigned .w256) .add auto_state_bits_0 (BitVec.ofNat 256 1))] = some following at hNext
            have hFollowing := (Option.some.inj hNext).symm
            subst following
            let auto_following_bits_0 := (Ora.Integer.wrappingResult (.unsigned .w256) .add auto_state_bits_0 (BitVec.ofNat 256 1))
            refine ⟨hFollowingReady, hHeaderFacts, ?_⟩
            intro formula hFormula
            simp at hFormula
            rcases hFormula with hFormula | hFormula <;> subst formula
            ·
              simp only [Ora.Loop.formulaHoldsInState, Ora.Loop.bindState?, TyRef.integerShape?, integerShapeForCompilerTypeId_u8, integerShapeForCompilerTypeId_u16, integerShapeForCompilerTypeId_u32, integerShapeForCompilerTypeId_u64, integerShapeForCompilerTypeId_u128, integerShapeForCompilerTypeId_u160, integerShapeForCompilerTypeId_u256, integerShapeForCompilerTypeId_i8, integerShapeForCompilerTypeId_i16, integerShapeForCompilerTypeId_i32, integerShapeForCompilerTypeId_i64, integerShapeForCompilerTypeId_i128, integerShapeForCompilerTypeId_i256, h_auto_free_0_0, if_pos]
              simp only [formulaDenotes?, denoteFormula?, denoteValue?, IntegerLiteralTerm.asInteger?, Ora.Integer.Value.ofNat, Ora.Integer.Value.relation?, Ora.Integer.Value.bitsFor?, Ora.Integer.Value.binary?, Ora.Integer.wrappingResult, Ora.Integer.bitAndResult, Ora.Integer.bitXorResult, Ora.Spec.IntegerShape.isSigned, Env.setFree, Env.lookupVar, Env.lookupFree, lookupFreeBinding, TyRef.integerShape?, integerShapeForCompilerTypeId_u8, integerShapeForCompilerTypeId_u16, integerShapeForCompilerTypeId_u32, integerShapeForCompilerTypeId_u64, integerShapeForCompilerTypeId_u128, integerShapeForCompilerTypeId_u160, integerShapeForCompilerTypeId_u256, integerShapeForCompilerTypeId_i8, integerShapeForCompilerTypeId_i16, integerShapeForCompilerTypeId_i32, integerShapeForCompilerTypeId_i64, integerShapeForCompilerTypeId_i128, integerShapeForCompilerTypeId_i256, emitted_terms_length, emitted_term_0_lookup, emitted_term_1_lookup, emitted_term_2_lookup, emitted_term_3_lookup, emitted_term_4_lookup, emitted_term_5_lookup, emitted_term_6_lookup, emitted_term_7_lookup, emitted_term_8_lookup, emitted_term_9_lookup, emitted_term_10_lookup, emitted_term_11_lookup, emitted_term_33_lookup, emitted_term_34_lookup, emitted_term_35_lookup, emitted_term_36_lookup, emitted_term_37_lookup, emitted_term_38_lookup, emitted_term_39_lookup, emitted_term_40_lookup, emitted_term_41_lookup, emitted_term_42_lookup, emitted_term_43_lookup, h_auto_free_0_0, if_pos, if_false, Bool.false_eq_true]
              change (Ora.Integer.unsignedLe (shape := (.unsigned .w256)) auto_following_bits_0 (BitVec.ofNat 256 3))
              first
              | bv_decide
              | (simp only [auto_following_bits_0]; apply Ora.Integer.signed_add_one_le_of_lt; assumption)
              | (apply Ora.Integer.signed_add_one_guard_safe_of_lt; assumption)
              | (simp only [auto_following_bits_0]; apply Ora.Integer.wrapping_add_one_le_of_unsigned_lt; simp_all only [])
              | (apply Ora.Integer.wrapping_add_one_not_lt_self_of_room; simp_all [Ora.Integer.unsignedLt, Ora.Spec.IntegerShape.bits, Ora.Spec.uintBits] <;> omega)
              | (simp_all only [auto_following_bits_0] <;> done)
              | (simp [auto_following_bits_0, Ora.Integer.unsignedLt, Ora.Integer.unsignedLe, Ora.Integer.signedLt, Ora.Integer.signedLe, Ora.Integer.mathematicalValue, Ora.Integer.wrappingResult] at * <;> omega)
            ·
              simp only [Ora.Loop.formulaHoldsInState, Ora.Loop.bindState?, TyRef.integerShape?, integerShapeForCompilerTypeId_u8, integerShapeForCompilerTypeId_u16, integerShapeForCompilerTypeId_u32, integerShapeForCompilerTypeId_u64, integerShapeForCompilerTypeId_u128, integerShapeForCompilerTypeId_u160, integerShapeForCompilerTypeId_u256, integerShapeForCompilerTypeId_i8, integerShapeForCompilerTypeId_i16, integerShapeForCompilerTypeId_i32, integerShapeForCompilerTypeId_i64, integerShapeForCompilerTypeId_i128, integerShapeForCompilerTypeId_i256, h_auto_free_0_0, if_pos]
              simp only [formulaDenotes?, denoteFormula?, denoteValue?, IntegerLiteralTerm.asInteger?, Ora.Integer.Value.ofNat, Ora.Integer.Value.relation?, Ora.Integer.Value.bitsFor?, Ora.Integer.Value.binary?, Ora.Integer.wrappingResult, Ora.Integer.bitAndResult, Ora.Integer.bitXorResult, Ora.Spec.IntegerShape.isSigned, Env.setFree, Env.lookupVar, Env.lookupFree, lookupFreeBinding, TyRef.integerShape?, integerShapeForCompilerTypeId_u8, integerShapeForCompilerTypeId_u16, integerShapeForCompilerTypeId_u32, integerShapeForCompilerTypeId_u64, integerShapeForCompilerTypeId_u128, integerShapeForCompilerTypeId_u160, integerShapeForCompilerTypeId_u256, integerShapeForCompilerTypeId_i8, integerShapeForCompilerTypeId_i16, integerShapeForCompilerTypeId_i32, integerShapeForCompilerTypeId_i64, integerShapeForCompilerTypeId_i128, integerShapeForCompilerTypeId_i256, emitted_terms_length, emitted_term_0_lookup, emitted_term_1_lookup, emitted_term_2_lookup, emitted_term_3_lookup, emitted_term_4_lookup, emitted_term_5_lookup, emitted_term_6_lookup, emitted_term_7_lookup, emitted_term_8_lookup, emitted_term_9_lookup, emitted_term_10_lookup, emitted_term_11_lookup, emitted_term_33_lookup, emitted_term_34_lookup, emitted_term_35_lookup, emitted_term_36_lookup, emitted_term_37_lookup, emitted_term_38_lookup, emitted_term_39_lookup, emitted_term_40_lookup, emitted_term_41_lookup, emitted_term_42_lookup, emitted_term_43_lookup, h_auto_free_0_0, if_pos, if_false, Bool.false_eq_true]
              change (Ora.Integer.unsignedLe (shape := (.unsigned .w256)) (Ora.Integer.wrappingResult (.unsigned .w256) .mul auto_following_bits_0 (BitVec.ofNat 256 1)) (BitVec.ofNat 256 3))
              first
              | bv_decide
              | (simp only [auto_following_bits_0]; apply Ora.Integer.signed_add_one_le_of_lt; assumption)
              | (apply Ora.Integer.signed_add_one_guard_safe_of_lt; assumption)
              | (simp only [auto_following_bits_0]; apply Ora.Integer.wrapping_add_one_le_of_unsigned_lt; simp_all only [])
              | (apply Ora.Integer.wrapping_add_one_not_lt_self_of_room; simp_all [Ora.Integer.unsignedLt, Ora.Spec.IntegerShape.bits, Ora.Spec.uintBits] <;> omega)
              | (simp_all only [auto_following_bits_0] <;> done)
              | (simp [auto_following_bits_0, Ora.Integer.unsignedLt, Ora.Integer.unsignedLe, Ora.Integer.signedLt, Ora.Integer.signedLe, Ora.Integer.mathematicalValue, Ora.Integer.wrappingResult] at * <;> omega)
          · simp [Ora.Loop.stateReady, Ora.Loop.valuesMatchVariables] at hReady
      · subst edge
        dsimp at hTarget
        simp at hTarget
    · intro edge state hMember hInvariant hPath
      simp at hMember
      rcases hMember with hMember | hMember
      · subst edge
        dsimp at hInvariant hPath ⊢
        rcases hInvariant with ⟨hReady, _, hInvariant⟩
        rcases state with _ | ⟨auto_state_value_0, auto_state_tail_0⟩
        · simp [Ora.Loop.stateReady, Ora.Loop.valuesMatchVariables] at hReady
        ·
          rcases auto_state_tail_0 with _ | ⟨auto_state_extra, auto_state_rest⟩
          ·
            rcases auto_state_value_0 with ⟨auto_state_shape_0, auto_state_bits_0⟩
            simp [Ora.Loop.stateReady, Ora.Loop.valuesMatchVariables] at hReady
            subst auto_state_shape_0
            have h_auto_invariant_0 := hInvariant (.term 6) (by simp)
            simp only [Ora.Loop.formulaHoldsInState, Ora.Loop.bindState?, TyRef.integerShape?, integerShapeForCompilerTypeId_u8, integerShapeForCompilerTypeId_u16, integerShapeForCompilerTypeId_u32, integerShapeForCompilerTypeId_u64, integerShapeForCompilerTypeId_u128, integerShapeForCompilerTypeId_u160, integerShapeForCompilerTypeId_u256, integerShapeForCompilerTypeId_i8, integerShapeForCompilerTypeId_i16, integerShapeForCompilerTypeId_i32, integerShapeForCompilerTypeId_i64, integerShapeForCompilerTypeId_i128, integerShapeForCompilerTypeId_i256, h_auto_free_0_0, if_pos] at h_auto_invariant_0
            simp only [formulaDenotes?, denoteFormula?, denoteValue?, IntegerLiteralTerm.asInteger?, Ora.Integer.Value.ofNat, Ora.Integer.Value.relation?, Ora.Integer.Value.bitsFor?, Ora.Integer.Value.binary?, Ora.Integer.wrappingResult, Ora.Integer.bitAndResult, Ora.Integer.bitXorResult, Ora.Spec.IntegerShape.isSigned, Env.setFree, Env.lookupVar, Env.lookupFree, lookupFreeBinding, TyRef.integerShape?, integerShapeForCompilerTypeId_u8, integerShapeForCompilerTypeId_u16, integerShapeForCompilerTypeId_u32, integerShapeForCompilerTypeId_u64, integerShapeForCompilerTypeId_u128, integerShapeForCompilerTypeId_u160, integerShapeForCompilerTypeId_u256, integerShapeForCompilerTypeId_i8, integerShapeForCompilerTypeId_i16, integerShapeForCompilerTypeId_i32, integerShapeForCompilerTypeId_i64, integerShapeForCompilerTypeId_i128, integerShapeForCompilerTypeId_i256, emitted_terms_length, emitted_term_0_lookup, emitted_term_1_lookup, emitted_term_2_lookup, emitted_term_3_lookup, emitted_term_4_lookup, emitted_term_5_lookup, emitted_term_6_lookup, emitted_term_7_lookup, emitted_term_8_lookup, emitted_term_9_lookup, emitted_term_10_lookup, emitted_term_11_lookup, emitted_term_33_lookup, emitted_term_34_lookup, emitted_term_35_lookup, emitted_term_36_lookup, emitted_term_37_lookup, emitted_term_38_lookup, emitted_term_39_lookup, emitted_term_40_lookup, emitted_term_41_lookup, emitted_term_42_lookup, emitted_term_43_lookup, h_auto_free_0_0, if_pos, if_false, Bool.false_eq_true] at h_auto_invariant_0
            change (Ora.Integer.unsignedLe (shape := (.unsigned .w256)) auto_state_bits_0 (BitVec.ofNat 256 3)) at h_auto_invariant_0
            have h_auto_invariant_1 := hInvariant (.term 11) (by simp)
            simp only [Ora.Loop.formulaHoldsInState, Ora.Loop.bindState?, TyRef.integerShape?, integerShapeForCompilerTypeId_u8, integerShapeForCompilerTypeId_u16, integerShapeForCompilerTypeId_u32, integerShapeForCompilerTypeId_u64, integerShapeForCompilerTypeId_u128, integerShapeForCompilerTypeId_u160, integerShapeForCompilerTypeId_u256, integerShapeForCompilerTypeId_i8, integerShapeForCompilerTypeId_i16, integerShapeForCompilerTypeId_i32, integerShapeForCompilerTypeId_i64, integerShapeForCompilerTypeId_i128, integerShapeForCompilerTypeId_i256, h_auto_free_0_0, if_pos] at h_auto_invariant_1
            simp only [formulaDenotes?, denoteFormula?, denoteValue?, IntegerLiteralTerm.asInteger?, Ora.Integer.Value.ofNat, Ora.Integer.Value.relation?, Ora.Integer.Value.bitsFor?, Ora.Integer.Value.binary?, Ora.Integer.wrappingResult, Ora.Integer.bitAndResult, Ora.Integer.bitXorResult, Ora.Spec.IntegerShape.isSigned, Env.setFree, Env.lookupVar, Env.lookupFree, lookupFreeBinding, TyRef.integerShape?, integerShapeForCompilerTypeId_u8, integerShapeForCompilerTypeId_u16, integerShapeForCompilerTypeId_u32, integerShapeForCompilerTypeId_u64, integerShapeForCompilerTypeId_u128, integerShapeForCompilerTypeId_u160, integerShapeForCompilerTypeId_u256, integerShapeForCompilerTypeId_i8, integerShapeForCompilerTypeId_i16, integerShapeForCompilerTypeId_i32, integerShapeForCompilerTypeId_i64, integerShapeForCompilerTypeId_i128, integerShapeForCompilerTypeId_i256, emitted_terms_length, emitted_term_0_lookup, emitted_term_1_lookup, emitted_term_2_lookup, emitted_term_3_lookup, emitted_term_4_lookup, emitted_term_5_lookup, emitted_term_6_lookup, emitted_term_7_lookup, emitted_term_8_lookup, emitted_term_9_lookup, emitted_term_10_lookup, emitted_term_11_lookup, emitted_term_33_lookup, emitted_term_34_lookup, emitted_term_35_lookup, emitted_term_36_lookup, emitted_term_37_lookup, emitted_term_38_lookup, emitted_term_39_lookup, emitted_term_40_lookup, emitted_term_41_lookup, emitted_term_42_lookup, emitted_term_43_lookup, h_auto_free_0_0, if_pos, if_false, Bool.false_eq_true] at h_auto_invariant_1
            change (Ora.Integer.unsignedLe (shape := (.unsigned .w256)) (Ora.Integer.wrappingResult (.unsigned .w256) .mul auto_state_bits_0 (BitVec.ofNat 256 1)) (BitVec.ofNat 256 3)) at h_auto_invariant_1
            have h_auto_path_0 := hPath (.term 3) (by simp)
            simp only [Ora.Loop.formulaHoldsInState, Ora.Loop.bindState?, TyRef.integerShape?, integerShapeForCompilerTypeId_u8, integerShapeForCompilerTypeId_u16, integerShapeForCompilerTypeId_u32, integerShapeForCompilerTypeId_u64, integerShapeForCompilerTypeId_u128, integerShapeForCompilerTypeId_u160, integerShapeForCompilerTypeId_u256, integerShapeForCompilerTypeId_i8, integerShapeForCompilerTypeId_i16, integerShapeForCompilerTypeId_i32, integerShapeForCompilerTypeId_i64, integerShapeForCompilerTypeId_i128, integerShapeForCompilerTypeId_i256, h_auto_free_0_0, if_pos] at h_auto_path_0
            simp only [formulaDenotes?, denoteFormula?, denoteValue?, IntegerLiteralTerm.asInteger?, Ora.Integer.Value.ofNat, Ora.Integer.Value.relation?, Ora.Integer.Value.bitsFor?, Ora.Integer.Value.binary?, Ora.Integer.wrappingResult, Ora.Integer.bitAndResult, Ora.Integer.bitXorResult, Ora.Spec.IntegerShape.isSigned, Env.setFree, Env.lookupVar, Env.lookupFree, lookupFreeBinding, TyRef.integerShape?, integerShapeForCompilerTypeId_u8, integerShapeForCompilerTypeId_u16, integerShapeForCompilerTypeId_u32, integerShapeForCompilerTypeId_u64, integerShapeForCompilerTypeId_u128, integerShapeForCompilerTypeId_u160, integerShapeForCompilerTypeId_u256, integerShapeForCompilerTypeId_i8, integerShapeForCompilerTypeId_i16, integerShapeForCompilerTypeId_i32, integerShapeForCompilerTypeId_i64, integerShapeForCompilerTypeId_i128, integerShapeForCompilerTypeId_i256, emitted_terms_length, emitted_term_0_lookup, emitted_term_1_lookup, emitted_term_2_lookup, emitted_term_3_lookup, emitted_term_4_lookup, emitted_term_5_lookup, emitted_term_6_lookup, emitted_term_7_lookup, emitted_term_8_lookup, emitted_term_9_lookup, emitted_term_10_lookup, emitted_term_11_lookup, emitted_term_33_lookup, emitted_term_34_lookup, emitted_term_35_lookup, emitted_term_36_lookup, emitted_term_37_lookup, emitted_term_38_lookup, emitted_term_39_lookup, emitted_term_40_lookup, emitted_term_41_lookup, emitted_term_42_lookup, emitted_term_43_lookup, h_auto_free_0_0, if_pos, if_false, Bool.false_eq_true] at h_auto_path_0
            change (Ora.Integer.unsignedLt (shape := (.unsigned .w256)) auto_state_bits_0 (BitVec.ofNat 256 3)) at h_auto_path_0
            intro checkpoint hCheckpoint
            simp at hCheckpoint
            subst checkpoint
            unfold Ora.Loop.safetyCheckpointHolds
            intro beforeDependencies checkpointEnv hParentBound hDependencies
            change some (Env.setFree env { file_id := 4294967294, pattern_id := 0 }
              (.integer (Ora.Integer.Value.mk (.unsigned .w256) auto_state_bits_0))) =
              some beforeDependencies at hParentBound
            have hBeforeDependencies := (Option.some.inj hParentBound).symm
            rw [hBeforeDependencies] at hDependencies
            simp only [List.take, Ora.Loop.environmentRelationsProduce] at hDependencies
            subst checkpointEnv
            intro formula hFormula
            simp at hFormula
            subst formula
            simp only [Ora.Loop.formulaHoldsInEnv]
            simp only [formulaDenotes?, denoteFormula?, denoteValue?, IntegerLiteralTerm.asInteger?, Ora.Integer.Value.ofNat, Ora.Integer.Value.relation?, Ora.Integer.Value.bitsFor?, Ora.Integer.Value.binary?, Ora.Integer.wrappingResult, Ora.Integer.bitAndResult, Ora.Integer.bitXorResult, Ora.Spec.IntegerShape.isSigned, Env.setFree, Env.lookupVar, Env.lookupFree, lookupFreeBinding, TyRef.integerShape?, integerShapeForCompilerTypeId_u8, integerShapeForCompilerTypeId_u16, integerShapeForCompilerTypeId_u32, integerShapeForCompilerTypeId_u64, integerShapeForCompilerTypeId_u128, integerShapeForCompilerTypeId_u160, integerShapeForCompilerTypeId_u256, integerShapeForCompilerTypeId_i8, integerShapeForCompilerTypeId_i16, integerShapeForCompilerTypeId_i32, integerShapeForCompilerTypeId_i64, integerShapeForCompilerTypeId_i128, integerShapeForCompilerTypeId_i256, emitted_terms_length, emitted_term_0_lookup, emitted_term_1_lookup, emitted_term_2_lookup, emitted_term_3_lookup, emitted_term_4_lookup, emitted_term_5_lookup, emitted_term_6_lookup, emitted_term_7_lookup, emitted_term_8_lookup, emitted_term_9_lookup, emitted_term_10_lookup, emitted_term_11_lookup, emitted_term_33_lookup, emitted_term_34_lookup, emitted_term_35_lookup, emitted_term_36_lookup, emitted_term_37_lookup, emitted_term_38_lookup, emitted_term_39_lookup, emitted_term_40_lookup, emitted_term_41_lookup, emitted_term_42_lookup, emitted_term_43_lookup, h_auto_free_0_0, if_pos, if_false, Bool.false_eq_true]
            change (¬ (Ora.Integer.unsignedLt (shape := (.unsigned .w256)) (Ora.Integer.wrappingResult (.unsigned .w256) .add auto_state_bits_0 (BitVec.ofNat 256 1)) auto_state_bits_0))
            first
            | bv_decide
            | (apply Ora.Integer.signed_add_one_le_of_lt; assumption)
            | (apply Ora.Integer.signed_add_one_guard_safe_of_lt; assumption)
            | (apply Ora.Integer.wrapping_add_one_le_of_unsigned_lt; simp_all only [])
            | (apply Ora.Integer.wrapping_add_one_not_lt_self_of_room; simp_all [Ora.Integer.unsignedLt, Ora.Spec.IntegerShape.bits, Ora.Spec.uintBits] <;> omega)
            | (simp [Ora.Integer.unsignedLt, Ora.Integer.unsignedLe, Ora.Integer.signedLt, Ora.Integer.signedLe, Ora.Integer.mathematicalValue, Ora.Integer.wrappingResult] at * <;> omega)
          · simp [Ora.Loop.stateReady, Ora.Loop.valuesMatchVariables] at hReady
      · subst edge
        dsimp at hInvariant hPath ⊢
        simp [Ora.Loop.safetyCheckpointsHold]
    · intro edge current following hMember hTarget hInvariant hPath hSafe hNext
      simp at hMember
      rcases hMember with hMember | hMember
      · subst edge
        dsimp at hTarget
        simp at hTarget
      · subst edge
        dsimp at hTarget hInvariant hPath hSafe hNext ⊢
        rcases hInvariant with ⟨hReady, _, hInvariant⟩
        rcases current with _ | ⟨auto_state_value_0, auto_state_tail_0⟩
        · simp [Ora.Loop.stateReady, Ora.Loop.valuesMatchVariables] at hReady
        ·
          rcases auto_state_tail_0 with _ | ⟨auto_state_extra, auto_state_rest⟩
          ·
            rcases auto_state_value_0 with ⟨auto_state_shape_0, auto_state_bits_0⟩
            simp [Ora.Loop.stateReady, Ora.Loop.valuesMatchVariables] at hReady
            subst auto_state_shape_0
            have h_auto_invariant_0 := hInvariant (.term 6) (by simp)
            simp only [Ora.Loop.formulaHoldsInState, Ora.Loop.bindState?, TyRef.integerShape?, integerShapeForCompilerTypeId_u8, integerShapeForCompilerTypeId_u16, integerShapeForCompilerTypeId_u32, integerShapeForCompilerTypeId_u64, integerShapeForCompilerTypeId_u128, integerShapeForCompilerTypeId_u160, integerShapeForCompilerTypeId_u256, integerShapeForCompilerTypeId_i8, integerShapeForCompilerTypeId_i16, integerShapeForCompilerTypeId_i32, integerShapeForCompilerTypeId_i64, integerShapeForCompilerTypeId_i128, integerShapeForCompilerTypeId_i256, h_auto_free_0_0, if_pos] at h_auto_invariant_0
            simp only [formulaDenotes?, denoteFormula?, denoteValue?, IntegerLiteralTerm.asInteger?, Ora.Integer.Value.ofNat, Ora.Integer.Value.relation?, Ora.Integer.Value.bitsFor?, Ora.Integer.Value.binary?, Ora.Integer.wrappingResult, Ora.Integer.bitAndResult, Ora.Integer.bitXorResult, Ora.Spec.IntegerShape.isSigned, Env.setFree, Env.lookupVar, Env.lookupFree, lookupFreeBinding, TyRef.integerShape?, integerShapeForCompilerTypeId_u8, integerShapeForCompilerTypeId_u16, integerShapeForCompilerTypeId_u32, integerShapeForCompilerTypeId_u64, integerShapeForCompilerTypeId_u128, integerShapeForCompilerTypeId_u160, integerShapeForCompilerTypeId_u256, integerShapeForCompilerTypeId_i8, integerShapeForCompilerTypeId_i16, integerShapeForCompilerTypeId_i32, integerShapeForCompilerTypeId_i64, integerShapeForCompilerTypeId_i128, integerShapeForCompilerTypeId_i256, emitted_terms_length, emitted_term_0_lookup, emitted_term_1_lookup, emitted_term_2_lookup, emitted_term_3_lookup, emitted_term_4_lookup, emitted_term_5_lookup, emitted_term_6_lookup, emitted_term_7_lookup, emitted_term_8_lookup, emitted_term_9_lookup, emitted_term_10_lookup, emitted_term_11_lookup, emitted_term_33_lookup, emitted_term_34_lookup, emitted_term_35_lookup, emitted_term_36_lookup, emitted_term_37_lookup, emitted_term_38_lookup, emitted_term_39_lookup, emitted_term_40_lookup, emitted_term_41_lookup, emitted_term_42_lookup, emitted_term_43_lookup, h_auto_free_0_0, if_pos, if_false, Bool.false_eq_true] at h_auto_invariant_0
            change (Ora.Integer.unsignedLe (shape := (.unsigned .w256)) auto_state_bits_0 (BitVec.ofNat 256 3)) at h_auto_invariant_0
            have h_auto_invariant_1 := hInvariant (.term 11) (by simp)
            simp only [Ora.Loop.formulaHoldsInState, Ora.Loop.bindState?, TyRef.integerShape?, integerShapeForCompilerTypeId_u8, integerShapeForCompilerTypeId_u16, integerShapeForCompilerTypeId_u32, integerShapeForCompilerTypeId_u64, integerShapeForCompilerTypeId_u128, integerShapeForCompilerTypeId_u160, integerShapeForCompilerTypeId_u256, integerShapeForCompilerTypeId_i8, integerShapeForCompilerTypeId_i16, integerShapeForCompilerTypeId_i32, integerShapeForCompilerTypeId_i64, integerShapeForCompilerTypeId_i128, integerShapeForCompilerTypeId_i256, h_auto_free_0_0, if_pos] at h_auto_invariant_1
            simp only [formulaDenotes?, denoteFormula?, denoteValue?, IntegerLiteralTerm.asInteger?, Ora.Integer.Value.ofNat, Ora.Integer.Value.relation?, Ora.Integer.Value.bitsFor?, Ora.Integer.Value.binary?, Ora.Integer.wrappingResult, Ora.Integer.bitAndResult, Ora.Integer.bitXorResult, Ora.Spec.IntegerShape.isSigned, Env.setFree, Env.lookupVar, Env.lookupFree, lookupFreeBinding, TyRef.integerShape?, integerShapeForCompilerTypeId_u8, integerShapeForCompilerTypeId_u16, integerShapeForCompilerTypeId_u32, integerShapeForCompilerTypeId_u64, integerShapeForCompilerTypeId_u128, integerShapeForCompilerTypeId_u160, integerShapeForCompilerTypeId_u256, integerShapeForCompilerTypeId_i8, integerShapeForCompilerTypeId_i16, integerShapeForCompilerTypeId_i32, integerShapeForCompilerTypeId_i64, integerShapeForCompilerTypeId_i128, integerShapeForCompilerTypeId_i256, emitted_terms_length, emitted_term_0_lookup, emitted_term_1_lookup, emitted_term_2_lookup, emitted_term_3_lookup, emitted_term_4_lookup, emitted_term_5_lookup, emitted_term_6_lookup, emitted_term_7_lookup, emitted_term_8_lookup, emitted_term_9_lookup, emitted_term_10_lookup, emitted_term_11_lookup, emitted_term_33_lookup, emitted_term_34_lookup, emitted_term_35_lookup, emitted_term_36_lookup, emitted_term_37_lookup, emitted_term_38_lookup, emitted_term_39_lookup, emitted_term_40_lookup, emitted_term_41_lookup, emitted_term_42_lookup, emitted_term_43_lookup, h_auto_free_0_0, if_pos, if_false, Bool.false_eq_true] at h_auto_invariant_1
            change (Ora.Integer.unsignedLe (shape := (.unsigned .w256)) (Ora.Integer.wrappingResult (.unsigned .w256) .mul auto_state_bits_0 (BitVec.ofNat 256 1)) (BitVec.ofNat 256 3)) at h_auto_invariant_1
            have h_auto_path_0 := hPath (.term 43) (by simp)
            simp only [Ora.Loop.formulaHoldsInState, Ora.Loop.bindState?, TyRef.integerShape?, integerShapeForCompilerTypeId_u8, integerShapeForCompilerTypeId_u16, integerShapeForCompilerTypeId_u32, integerShapeForCompilerTypeId_u64, integerShapeForCompilerTypeId_u128, integerShapeForCompilerTypeId_u160, integerShapeForCompilerTypeId_u256, integerShapeForCompilerTypeId_i8, integerShapeForCompilerTypeId_i16, integerShapeForCompilerTypeId_i32, integerShapeForCompilerTypeId_i64, integerShapeForCompilerTypeId_i128, integerShapeForCompilerTypeId_i256, h_auto_free_0_0, if_pos] at h_auto_path_0
            simp only [formulaDenotes?, denoteFormula?, denoteValue?, IntegerLiteralTerm.asInteger?, Ora.Integer.Value.ofNat, Ora.Integer.Value.relation?, Ora.Integer.Value.bitsFor?, Ora.Integer.Value.binary?, Ora.Integer.wrappingResult, Ora.Integer.bitAndResult, Ora.Integer.bitXorResult, Ora.Spec.IntegerShape.isSigned, Env.setFree, Env.lookupVar, Env.lookupFree, lookupFreeBinding, TyRef.integerShape?, integerShapeForCompilerTypeId_u8, integerShapeForCompilerTypeId_u16, integerShapeForCompilerTypeId_u32, integerShapeForCompilerTypeId_u64, integerShapeForCompilerTypeId_u128, integerShapeForCompilerTypeId_u160, integerShapeForCompilerTypeId_u256, integerShapeForCompilerTypeId_i8, integerShapeForCompilerTypeId_i16, integerShapeForCompilerTypeId_i32, integerShapeForCompilerTypeId_i64, integerShapeForCompilerTypeId_i128, integerShapeForCompilerTypeId_i256, emitted_terms_length, emitted_term_0_lookup, emitted_term_1_lookup, emitted_term_2_lookup, emitted_term_3_lookup, emitted_term_4_lookup, emitted_term_5_lookup, emitted_term_6_lookup, emitted_term_7_lookup, emitted_term_8_lookup, emitted_term_9_lookup, emitted_term_10_lookup, emitted_term_11_lookup, emitted_term_33_lookup, emitted_term_34_lookup, emitted_term_35_lookup, emitted_term_36_lookup, emitted_term_37_lookup, emitted_term_38_lookup, emitted_term_39_lookup, emitted_term_40_lookup, emitted_term_41_lookup, emitted_term_42_lookup, emitted_term_43_lookup, h_auto_free_0_0, if_pos, if_false, Bool.false_eq_true] at h_auto_path_0
            change (¬ (Ora.Integer.unsignedLt (shape := (.unsigned .w256)) auto_state_bits_0 (BitVec.ofNat 256 3))) at h_auto_path_0
            simp only [Ora.Loop.stateBindingsProduceState, Ora.Loop.applyStateBindings?, Ora.Loop.bindingsMatchVariables, Ora.Loop.bindState?, Ora.Loop.bindingFormulas, Ora.Loop.evaluateIntegerFormulas?, Ora.Loop.integerFormulaValue?, Ora.Loop.valuesMatchVariables] at hNext
            simp only [formulaValue?, denoteValue?, IntegerLiteralTerm.asInteger?, Ora.Integer.Value.ofNat, Ora.Integer.Value.binary?, Ora.Integer.Value.bitsFor?, Ora.Integer.wrappingResult, Ora.Integer.bitAndResult, Ora.Integer.bitXorResult, Env.setFree, Env.lookupVar, Env.lookupFree, lookupFreeBinding, TyRef.integerShape?, integerShapeForCompilerTypeId_u8, integerShapeForCompilerTypeId_u16, integerShapeForCompilerTypeId_u32, integerShapeForCompilerTypeId_u64, integerShapeForCompilerTypeId_u128, integerShapeForCompilerTypeId_u160, integerShapeForCompilerTypeId_u256, integerShapeForCompilerTypeId_i8, integerShapeForCompilerTypeId_i16, integerShapeForCompilerTypeId_i32, integerShapeForCompilerTypeId_i64, integerShapeForCompilerTypeId_i128, integerShapeForCompilerTypeId_i256, emitted_terms_length, emitted_term_0_lookup, emitted_term_1_lookup, emitted_term_2_lookup, emitted_term_3_lookup, emitted_term_4_lookup, emitted_term_5_lookup, emitted_term_6_lookup, emitted_term_7_lookup, emitted_term_8_lookup, emitted_term_9_lookup, emitted_term_10_lookup, emitted_term_11_lookup, emitted_term_33_lookup, emitted_term_34_lookup, emitted_term_35_lookup, emitted_term_36_lookup, emitted_term_37_lookup, emitted_term_38_lookup, emitted_term_39_lookup, emitted_term_40_lookup, emitted_term_41_lookup, emitted_term_42_lookup, emitted_term_43_lookup, h_auto_free_0_0, if_pos, if_false, Bool.false_eq_true] at hNext
            change some [Ora.Integer.Value.mk (.unsigned .w256) auto_state_bits_0] = some following at hNext
            have hFollowing := (Option.some.inj hNext).symm
            subst following
            let auto_following_bits_0 := auto_state_bits_0
            simp [Ora.Loop.formulasHoldInState]
          · simp [Ora.Loop.stateReady, Ora.Loop.valuesMatchVariables] at hReady
end

end Query_12

end Ora.UserProof.ScalarLoopManual

/-
Repository-owned automation for the already-supported scalar LoopVC fragment.

The tactic is proof search only.  It does not change denotation, eligibility,
or artifact authority: Lean still checks the resulting term against the exact
compiler-emitted `loopProofAuthorizationFromAssumptions` proposition.

The generated call site must bind the exact local names `autoSummary` and
`emittedManifest`.  The macros resolve those names deliberately; this is the
stable Zig-to-Lean emission contract, not an implicit search convention.
-/

import Ora.Loop.Denotation
import Lean.Meta.Tactic.Simp.RegisterCommand

namespace Ora.Loop

open Ora.Obligation

register_simp_attr ora_loop_data
register_simp_attr ora_loop_row
register_simp_attr ora_loop_supported
register_simp_attr ora_loop_manifest

theorem lookupFreeBinding_cons_of_beq_true
    (bindings : List (FreeVarId × Value))
    (candidate id : FreeVarId)
    (value : Value)
    (hEqual : (candidate == id) = true) :
    lookupFreeBinding ((candidate, value) :: bindings) id = some value := by
  simp only [lookupFreeBinding, hEqual, if_true]

theorem lookupFreeBinding_cons_of_beq_false
    (bindings : List (FreeVarId × Value))
    (candidate id : FreeVarId)
    (value : Value)
    (hDifferent : (candidate == id) = false) :
    lookupFreeBinding ((candidate, value) :: bindings) id =
      lookupFreeBinding bindings id := by
  simp only [lookupFreeBinding, hDifferent, Bool.false_eq_true, if_false]

theorem specification_nonvacuous_from_support_env
    (manifest : Manifest)
    (assumptions : List AssumptionRow)
    (row : SummaryRow)
    (hAssumptions : assumptionsDenoteInEnv manifest row.supportEnv assumptions)
    (hContext : contextReady row.supportEnv row.contextVariables)
    (hInitial : ∃ state, loopInitialPremises manifest row.supportEnv row state) :
    loopSpecificationNonvacuous manifest assumptions row := by
  rcases hInitial with ⟨state, hState⟩
  exact ⟨row.supportEnv, state, hAssumptions, hContext, hState⟩

macro tk:"ora_loop_support" : tactic => do
  let autoSummary := Lean.mkIdentFrom tk `autoSummary
  let emittedManifest := Lean.mkIdentFrom tk `emittedManifest
  `(tactic| {
    have hShape : ($autoSummary).shapeSupported = true := by decide
    have hFormulas : ($autoSummary).formulasDenotable $emittedManifest = true := by
      simp only [ora_loop_data, ora_loop_row, Ora.Loop.SummaryRow.formulasDenotable,
        Ora.Loop.SummaryRow.supportEnv, Ora.Loop.bindSupportVariable,
        Ora.Loop.valueForTy, Ora.Loop.edgeFormulasDenotable,
        Ora.Loop.formulaDenotable, Ora.Loop.valueDenotable, List.all,
        TyRef.integerShape?, integerShapeForCompilerTypeId_u8,
        integerShapeForCompilerTypeId_u16, integerShapeForCompilerTypeId_u32,
        integerShapeForCompilerTypeId_u64, integerShapeForCompilerTypeId_u128,
        integerShapeForCompilerTypeId_u160, integerShapeForCompilerTypeId_u256,
        integerShapeForCompilerTypeId_i8, integerShapeForCompilerTypeId_i16,
        integerShapeForCompilerTypeId_i32, integerShapeForCompilerTypeId_i64,
        integerShapeForCompilerTypeId_i128, integerShapeForCompilerTypeId_i256,
        Env.setFree, Env.lookupFree, lookupFreeBinding, if_pos, if_false,
        formulaDenotes?, formulaValue?, denoteFormula?, denoteValue?,
        IntegerLiteralTerm.asInteger?, Ora.Integer.Value.ofNat,
        Ora.Integer.Value.relation?, Ora.Integer.Value.binary?,
        Ora.Integer.Value.bitsFor?, Ora.Integer.wrappingResult,
        Ora.Integer.bitAndResult, Ora.Integer.bitXorResult,
        Ora.Spec.IntegerShape.isSigned, Env.lookupVar, Bool.false_eq_true]
      decide
    have hSupported : ($autoSummary).supported $emittedManifest = true := by
      simp only [ora_loop_manifest, ora_loop_data, Ora.Loop.SummaryRow.supported,
        hShape, hFormulas, Bool.true_and]
    exact hSupported })

macro tk:"ora_loop_auto" : tactic => do
  let autoSummary := Lean.mkIdentFrom tk `autoSummary
  let emittedManifest := Lean.mkIdentFrom tk `emittedManifest
  `(tactic| {
    have hSupported : ($autoSummary).supported $emittedManifest = true := by
      simp only [ora_loop_supported]
    constructor
    · apply Ora.Loop.specification_nonvacuous_from_support_env
      · simp only [ora_loop_data, assumptionsDenoteInEnv, assumptionsDenoteInEnv?]
      · intro loopVar hLoopVar
        simp only [ora_loop_row, List.mem_singleton] at hLoopVar
        subst loopVar
        exact ⟨Ora.Integer.Value.ofNat
            (Ora.Spec.IntegerShape.unsigned Ora.Types.UIntWidth.w256) 0, rfl, by
          change some (Ora.Obligation.Value.integer (Ora.Integer.Value.ofNat
              (Ora.Spec.IntegerShape.unsigned Ora.Types.UIntWidth.w256) 0)) =
            some (Ora.Obligation.Value.integer (Ora.Integer.Value.ofNat
              (Ora.Spec.IntegerShape.unsigned Ora.Types.UIntWidth.w256) 0))
          rfl⟩
      · simp only [Ora.Loop.loopInitialPremises, Ora.Loop.denoteLoopSummary?,
          hSupported, if_true, Ora.Loop.denoteEdges?, Ora.Loop.denoteEdge?,
          Ora.Loop.denoteControlTarget?]
        refine ⟨[Ora.Integer.Value.ofNat
          (Ora.Spec.IntegerShape.unsigned Ora.Types.UIntWidth.w256) 0], ?_⟩
        simp only [ora_loop_data, ora_loop_row, Ora.Loop.valuesInitializeState,
          Ora.Loop.initializeState?,
          Ora.Loop.evaluateIntegerFormulas?, Ora.Loop.integerFormulaValue?,
          Ora.Loop.valuesMatchVariables, Ora.Loop.edgeFormulasDenotable,
          Ora.Loop.formulaDenotable, Ora.Loop.valueDenotable,
          formulaValue?, formulaDenotes?, denoteValue?, denoteFormula?,
          IntegerLiteralTerm.asInteger?, Env.lookupVar, Env.lookupFree,
          lookupFreeBinding, TyRef.integerShape?, Ora.Integer.Value.ofNat,
          Ora.Integer.Value.binary?, Ora.Integer.Value.relation?,
          Ora.Integer.Value.bitsFor?, Ora.Integer.wrappingResult,
          Ora.Integer.bitAndResult, Ora.Integer.bitXorResult,
          if_pos, if_false, Bool.false_eq_true]
        rfl
    · intro env hAssumptions hContext
      change Ora.Loop.loopInductionObligationsAtEnv
        $emittedManifest env $autoSummary
      unfold Ora.Loop.loopInductionObligationsAtEnv Ora.Loop.denoteLoopSummary?
      rw [hSupported]
      unfold $autoSummary
      simp only [Ora.Loop.denoteEdges?, Ora.Loop.denoteEdge?,
        Ora.Loop.denoteControlTarget?, Option.bind, if_pos, ora_loop_row]
      have hContextMember : ($autoSummary).contextVariables[0] ∈
          ($autoSummary).contextVariables := by
        unfold $autoSummary
        simp only [ora_loop_row, List.mem_singleton]
        rfl
      rcases hContext ($autoSummary).contextVariables[0] hContextMember with
        ⟨autoContextValue, hAutoContextShape, hAutoContextLookup⟩
      rcases autoContextValue with ⟨autoContextShape, autoContextBits⟩
      unfold $autoSummary at hAutoContextShape hAutoContextLookup
      simp only [ora_loop_row] at hAutoContextShape hAutoContextLookup
      dsimp at hAutoContextShape hAutoContextLookup
      unfold Env.lookupFree at hAutoContextLookup
      have hAutoContextShapeValue :
          autoContextShape = (.unsigned .w256) := by
        have hSome : some (.unsigned .w256) = some autoContextShape := by
          simpa only [ora_loop_row, TyRef.integerShape?,
            integerShapeForCompilerTypeId_u256] using hAutoContextShape
        exact (Option.some.inj hSome).symm
      subst autoContextShape
      refine { base := ?_, backedge := ?_, edgeSafety := ?_, exit := ?_ }
      · intro state hInitial
        have hReady := Ora.Loop.initialized_state_ready
          $emittedManifest env ($autoSummary).variables
          ($autoSummary).init state hInitial
        simp only [Ora.Loop.valuesInitializeState, Ora.Loop.initializeState?,
          Ora.Loop.evaluateIntegerFormulas?, Ora.Loop.integerFormulaValue?,
          Ora.Loop.valuesMatchVariables] at hInitial
        simp only [ora_loop_data, ora_loop_row, formulaValue?, denoteValue?,
          IntegerLiteralTerm.asInteger?, Ora.Integer.Value.ofNat,
          Ora.Integer.Value.binary?, Ora.Integer.Value.bitsFor?,
          Ora.Integer.wrappingResult, Ora.Integer.bitAndResult,
          Ora.Integer.bitXorResult, Env.setFree, Env.lookupVar,
          Env.lookupFree, lookupFreeBinding, TyRef.integerShape?,
          integerShapeForCompilerTypeId_u256, if_pos, if_false,
          Bool.false_eq_true] at hInitial
        change some [Ora.Integer.Value.mk (.unsigned .w256)
          (BitVec.ofNat 256 0)] = some state at hInitial
        have hState := (Option.some.inj hInitial).symm
        subst state
        refine ⟨hReady, ?_⟩
        intro formula hFormula
        simp only [ora_loop_row, List.mem_singleton] at hFormula
        subst formula
        simp only [Ora.Loop.formulaHoldsInState, Ora.Loop.bindState?,
          ora_loop_data, ora_loop_row, formulaDenotes?, denoteFormula?,
          denoteValue?, IntegerLiteralTerm.asInteger?,
          Ora.Integer.Value.ofNat, Ora.Integer.Value.relation?,
          Ora.Integer.Value.bitsFor?, Ora.Integer.Value.binary?,
          Ora.Integer.wrappingResult, Ora.Integer.bitAndResult,
          Ora.Integer.bitXorResult, Ora.Spec.IntegerShape.isSigned,
          Env.setFree, Env.lookupVar, Env.lookupFree,
          lookupFreeBinding_cons_of_beq_true,
          lookupFreeBinding_cons_of_beq_false,
          TyRef.integerShape?,
          integerShapeForCompilerTypeId_u256, hAutoContextLookup,
          if_pos, if_false, Bool.false_eq_true]
        change Ora.Integer.unsignedLe (shape := (.unsigned .w256))
          (BitVec.ofNat 256 0) autoContextBits
        clear hSupported hAssumptions hContext hContextMember
          hAutoContextShape hAutoContextLookup
        change 0 ≤ autoContextBits.toNat
        exact Nat.zero_le _
      · intro edge current following hMember hTarget hInvariant hPath hSafe hNext
        simp only [List.mem_cons, List.not_mem_nil, or_false] at hMember
        rcases hMember with hMember | hMember
        · subst edge
          dsimp at hTarget hInvariant hPath hSafe hNext ⊢
          rcases hInvariant with ⟨hReady, hInvariant⟩
          have hFollowingReady := Ora.Loop.bound_state_ready
            $emittedManifest env ($autoSummary).variables current following
            ($autoSummary).edges[0].nextState hNext
          rcases current with _ | ⟨stateValue, stateTail⟩
          · simp only [Ora.Loop.stateReady, ora_loop_row,
              Ora.Loop.valuesMatchVariables, Bool.false_eq_true] at hReady
          · rcases stateTail with _ | ⟨stateExtra, stateRest⟩
            · rcases stateValue with ⟨stateShape, stateBits⟩
              simp only [Ora.Loop.stateReady, ora_loop_row,
                Ora.Loop.valuesMatchVariables, TyRef.integerShape?,
                integerShapeForCompilerTypeId_u256, Bool.and_true,
                beq_iff_eq] at hReady
              have hStateShape : stateShape = (.unsigned .w256) :=
                (Option.some.inj hReady).symm
              subst stateShape
              have hInvariantFormula := hInvariant
                (($autoSummary).invariants[0]) (by
                  unfold $autoSummary
                  simp only [ora_loop_row, List.mem_singleton]
                  rfl)
              have hInvariantDenoted : Ora.Integer.unsignedLe
                  (shape := (.unsigned .w256)) stateBits autoContextBits := by
                unfold $autoSummary at hInvariantFormula
                simp only [ora_loop_row] at hInvariantFormula
                dsimp at hInvariantFormula
                simpa only [Ora.Loop.formulaHoldsInState, Ora.Loop.bindState?,
                  ora_loop_data, ora_loop_row, formulaDenotes?, denoteFormula?,
                  denoteValue?, IntegerLiteralTerm.asInteger?,
                  Ora.Integer.Value.ofNat, Ora.Integer.Value.relation?,
                  Ora.Integer.Value.bitsFor?, Ora.Integer.Value.binary?,
                  Ora.Integer.wrappingResult, Ora.Integer.bitAndResult,
                  Ora.Integer.bitXorResult, Ora.Spec.IntegerShape.isSigned,
                  Env.setFree, Env.lookupVar, Env.lookupFree,
                  lookupFreeBinding_cons_of_beq_true,
                  lookupFreeBinding_cons_of_beq_false,
                  TyRef.integerShape?, integerShapeForCompilerTypeId_u256,
                  hAutoContextLookup, if_pos, if_false, Bool.false_eq_true]
                  using hInvariantFormula
              have hPathFormula := hPath (($autoSummary).edges[0].path[0]) (by
                unfold $autoSummary
                simp only [ora_loop_row, List.mem_singleton]
                rfl)
              have hPathDenoted : Ora.Integer.unsignedLt
                  (shape := (.unsigned .w256)) stateBits autoContextBits := by
                unfold $autoSummary at hPathFormula
                simp only [ora_loop_row] at hPathFormula
                dsimp at hPathFormula
                simpa only [Ora.Loop.formulaHoldsInState, Ora.Loop.bindState?,
                  ora_loop_data, ora_loop_row, formulaDenotes?, denoteFormula?,
                  denoteValue?, IntegerLiteralTerm.asInteger?,
                  Ora.Integer.Value.ofNat, Ora.Integer.Value.relation?,
                  Ora.Integer.Value.bitsFor?, Ora.Integer.Value.binary?,
                  Ora.Integer.wrappingResult, Ora.Integer.bitAndResult,
                  Ora.Integer.bitXorResult, Ora.Spec.IntegerShape.isSigned,
                  Env.setFree, Env.lookupVar, Env.lookupFree,
                  lookupFreeBinding_cons_of_beq_true,
                  lookupFreeBinding_cons_of_beq_false,
                  TyRef.integerShape?, integerShapeForCompilerTypeId_u256,
                  hAutoContextLookup, if_pos, if_false, Bool.false_eq_true]
                  using hPathFormula
              simp only [Ora.Loop.stateBindingsProduceState,
                Ora.Loop.applyStateBindings?, Ora.Loop.bindingsMatchVariables,
                Ora.Loop.bindState?, Ora.Loop.bindingFormulas,
                Ora.Loop.evaluateIntegerFormulas?, Ora.Loop.integerFormulaValue?,
                Ora.Loop.valuesMatchVariables] at hNext
              simp only [ora_loop_data, ora_loop_row, formulaValue?, denoteValue?,
                IntegerLiteralTerm.asInteger?, Ora.Integer.Value.ofNat,
                Ora.Integer.Value.binary?, Ora.Integer.Value.bitsFor?,
                Ora.Integer.wrappingResult, Ora.Integer.bitAndResult,
                Ora.Integer.bitXorResult, Env.setFree, Env.lookupVar,
                Env.lookupFree, lookupFreeBinding_cons_of_beq_true,
                lookupFreeBinding_cons_of_beq_false,
                TyRef.integerShape?, integerShapeForCompilerTypeId_u256,
                if_pos, if_false, Bool.false_eq_true] at hNext
              change some [Ora.Integer.Value.mk (.unsigned .w256)
                (Ora.Integer.wrappingResult (.unsigned .w256) .add stateBits
                  (BitVec.ofNat 256 1))] = some following at hNext
              have hFollowing := (Option.some.inj hNext).symm
              subst following
              refine ⟨hFollowingReady, ?_⟩
              intro formula hFormula
              simp only [ora_loop_row, List.mem_singleton] at hFormula
              subst formula
              simp only [Ora.Loop.formulaHoldsInState, Ora.Loop.bindState?,
                ora_loop_data, ora_loop_row, formulaDenotes?, denoteFormula?,
                denoteValue?, IntegerLiteralTerm.asInteger?,
                Ora.Integer.Value.ofNat, Ora.Integer.Value.relation?,
                Ora.Integer.Value.bitsFor?, Ora.Integer.Value.binary?,
                Ora.Integer.wrappingResult, Ora.Integer.bitAndResult,
                Ora.Integer.bitXorResult, Ora.Spec.IntegerShape.isSigned,
                Env.setFree, Env.lookupVar, Env.lookupFree,
                lookupFreeBinding_cons_of_beq_true,
                lookupFreeBinding_cons_of_beq_false,
                TyRef.integerShape?, integerShapeForCompilerTypeId_u256,
                hAutoContextLookup, if_pos, if_false, Bool.false_eq_true]
              exact Ora.Integer.wrapping_add_one_le_of_unsigned_lt
                (.unsigned .w256) stateBits autoContextBits hPathDenoted
            · simp only [Ora.Loop.stateReady, ora_loop_row,
                Ora.Loop.valuesMatchVariables, Bool.and_false,
                Bool.false_eq_true] at hReady
        · subst edge
          dsimp at hTarget
          contradiction
      · intro edge state hMember hInvariant hPath
        simp only [List.mem_cons, List.not_mem_nil, or_false] at hMember
        rcases hMember with hMember | hMember
        · subst edge
          dsimp at hInvariant hPath ⊢
          rcases hInvariant with ⟨hReady, hInvariant⟩
          rcases state with _ | ⟨stateValue, stateTail⟩
          · simp only [Ora.Loop.stateReady, ora_loop_row,
              Ora.Loop.valuesMatchVariables, Bool.false_eq_true] at hReady
          · rcases stateTail with _ | ⟨stateExtra, stateRest⟩
            · rcases stateValue with ⟨stateShape, stateBits⟩
              simp only [Ora.Loop.stateReady, ora_loop_row,
                Ora.Loop.valuesMatchVariables, TyRef.integerShape?,
                integerShapeForCompilerTypeId_u256, Bool.and_true,
                beq_iff_eq] at hReady
              have hStateShape : stateShape = (.unsigned .w256) :=
                (Option.some.inj hReady).symm
              subst stateShape
              have hPathFormula := hPath (($autoSummary).edges[0].path[0]) (by
                unfold $autoSummary
                simp only [ora_loop_row, List.mem_singleton]
                rfl)
              have hPathDenoted : Ora.Integer.unsignedLt
                  (shape := (.unsigned .w256)) stateBits autoContextBits := by
                unfold $autoSummary at hPathFormula
                simp only [ora_loop_row] at hPathFormula
                dsimp at hPathFormula
                simpa only [Ora.Loop.formulaHoldsInState, Ora.Loop.bindState?,
                  ora_loop_data, ora_loop_row, formulaDenotes?, denoteFormula?,
                  denoteValue?, IntegerLiteralTerm.asInteger?,
                  Ora.Integer.Value.ofNat, Ora.Integer.Value.relation?,
                  Ora.Integer.Value.bitsFor?, Ora.Integer.Value.binary?,
                  Ora.Integer.wrappingResult, Ora.Integer.bitAndResult,
                  Ora.Integer.bitXorResult, Ora.Spec.IntegerShape.isSigned,
                  Env.setFree, Env.lookupVar, Env.lookupFree,
                  lookupFreeBinding_cons_of_beq_true,
                  lookupFreeBinding_cons_of_beq_false,
                  TyRef.integerShape?, integerShapeForCompilerTypeId_u256,
                  hAutoContextLookup, if_pos, if_false, Bool.false_eq_true]
                  using hPathFormula
              intro checkpoint hCheckpoint
              simp only [ora_loop_row, List.mem_singleton] at hCheckpoint
              subst checkpoint
              unfold Ora.Loop.safetyCheckpointHolds
              intro beforeDependencies checkpointEnv hParentBound hDependencies
              simp only [ora_loop_row, Ora.Loop.bindState?, Env.setFree,
                TyRef.integerShape?, integerShapeForCompilerTypeId_u256,
                if_pos] at hParentBound
              have hBeforeDependencies := (Option.some.inj hParentBound).symm
              rw [hBeforeDependencies] at hDependencies
              simp only [List.take, Ora.Loop.environmentRelationsProduce] at hDependencies
              subst checkpointEnv
              intro formula hFormula
              simp only [ora_loop_row, List.mem_singleton] at hFormula
              subst formula
              simp only [Ora.Loop.formulaHoldsInEnv, ora_loop_data,
                ora_loop_row, formulaDenotes?, denoteFormula?, denoteValue?,
                IntegerLiteralTerm.asInteger?, Ora.Integer.Value.ofNat,
                Ora.Integer.Value.relation?, Ora.Integer.Value.bitsFor?,
                Ora.Integer.Value.binary?, Ora.Integer.wrappingResult,
                Ora.Integer.bitAndResult, Ora.Integer.bitXorResult,
                Ora.Spec.IntegerShape.isSigned, Env.setFree, Env.lookupVar,
                Env.lookupFree, lookupFreeBinding_cons_of_beq_true,
                lookupFreeBinding_cons_of_beq_false,
                TyRef.integerShape?, integerShapeForCompilerTypeId_u256,
                hAutoContextLookup, if_pos, if_false, Bool.false_eq_true]
              exact Ora.Integer.wrapping_add_one_not_lt_self_of_unsigned_lt
                (.unsigned .w256) stateBits autoContextBits hPathDenoted
            · simp only [Ora.Loop.stateReady, ora_loop_row,
                Ora.Loop.valuesMatchVariables, Bool.and_false,
                Bool.false_eq_true] at hReady
        · subst edge
          dsimp
          intro checkpoint hCheckpoint
          exact (List.not_mem_nil checkpoint hCheckpoint).elim
      · intro edge current following hMember hTarget hInvariant hPath hSafe hNext
        simp only [List.mem_cons, List.not_mem_nil, or_false] at hMember
        rcases hMember with hMember | hMember
        · subst edge
          dsimp at hTarget
          contradiction
        · subst edge
          dsimp at hTarget hInvariant hPath hSafe hNext ⊢
          rcases hInvariant with ⟨hReady, hInvariant⟩
          rcases current with _ | ⟨stateValue, stateTail⟩
          · simp only [Ora.Loop.stateReady, ora_loop_row,
              Ora.Loop.valuesMatchVariables, Bool.false_eq_true] at hReady
          · rcases stateTail with _ | ⟨stateExtra, stateRest⟩
            · rcases stateValue with ⟨stateShape, stateBits⟩
              simp only [Ora.Loop.stateReady, ora_loop_row,
                Ora.Loop.valuesMatchVariables, TyRef.integerShape?,
                integerShapeForCompilerTypeId_u256, Bool.and_true,
                beq_iff_eq] at hReady
              have hStateShape : stateShape = (.unsigned .w256) :=
                (Option.some.inj hReady).symm
              subst stateShape
              have hInvariantFormula := hInvariant
                (($autoSummary).invariants[0]) (by
                  unfold $autoSummary
                  simp only [ora_loop_row, List.mem_singleton]
                  rfl)
              have hInvariantDenoted : Ora.Integer.unsignedLe
                  (shape := (.unsigned .w256)) stateBits autoContextBits := by
                unfold $autoSummary at hInvariantFormula
                simp only [ora_loop_row] at hInvariantFormula
                dsimp at hInvariantFormula
                simpa only [Ora.Loop.formulaHoldsInState, Ora.Loop.bindState?,
                  ora_loop_data, ora_loop_row, formulaDenotes?, denoteFormula?,
                  denoteValue?, IntegerLiteralTerm.asInteger?,
                  Ora.Integer.Value.ofNat, Ora.Integer.Value.relation?,
                  Ora.Integer.Value.bitsFor?, Ora.Integer.Value.binary?,
                  Ora.Integer.wrappingResult, Ora.Integer.bitAndResult,
                  Ora.Integer.bitXorResult, Ora.Spec.IntegerShape.isSigned,
                  Env.setFree, Env.lookupVar, Env.lookupFree,
                  lookupFreeBinding_cons_of_beq_true,
                  lookupFreeBinding_cons_of_beq_false,
                  TyRef.integerShape?, integerShapeForCompilerTypeId_u256,
                  hAutoContextLookup, if_pos, if_false, Bool.false_eq_true]
                  using hInvariantFormula
              simp only [Ora.Loop.stateBindingsProduceState,
                Ora.Loop.applyStateBindings?, Ora.Loop.bindingsMatchVariables,
                Ora.Loop.bindState?, Ora.Loop.bindingFormulas,
                Ora.Loop.evaluateIntegerFormulas?, Ora.Loop.integerFormulaValue?,
                Ora.Loop.valuesMatchVariables] at hNext
              simp only [ora_loop_data, ora_loop_row, formulaValue?, denoteValue?,
                IntegerLiteralTerm.asInteger?, Ora.Integer.Value.ofNat,
                Ora.Integer.Value.binary?, Ora.Integer.Value.bitsFor?,
                Ora.Integer.wrappingResult, Ora.Integer.bitAndResult,
                Ora.Integer.bitXorResult, Env.setFree, Env.lookupVar,
                Env.lookupFree, lookupFreeBinding_cons_of_beq_true,
                lookupFreeBinding_cons_of_beq_false, TyRef.integerShape?,
                integerShapeForCompilerTypeId_u256, if_pos, if_false,
                Bool.false_eq_true] at hNext
              change some [Ora.Integer.Value.mk (.unsigned .w256) stateBits] =
                some following at hNext
              have hFollowing := (Option.some.inj hNext).symm
              subst following
              intro formula hFormula
              simp only [ora_loop_row, List.mem_singleton] at hFormula
              subst formula
              simp only [Ora.Loop.formulaHoldsInState, Ora.Loop.bindState?,
                ora_loop_data, ora_loop_row, formulaDenotes?, denoteFormula?,
                denoteValue?, IntegerLiteralTerm.asInteger?,
                Ora.Integer.Value.ofNat, Ora.Integer.Value.relation?,
                Ora.Integer.Value.bitsFor?, Ora.Integer.Value.binary?,
                Ora.Integer.wrappingResult, Ora.Integer.bitAndResult,
                Ora.Integer.bitXorResult, Ora.Spec.IntegerShape.isSigned,
                Env.setFree, Env.lookupVar, Env.lookupFree,
                lookupFreeBinding_cons_of_beq_true,
                lookupFreeBinding_cons_of_beq_false,
                TyRef.integerShape?, integerShapeForCompilerTypeId_u256,
                hAutoContextLookup, if_pos, if_false, Bool.false_eq_true]
              exact hInvariantDenoted
            · simp only [Ora.Loop.stateReady, ora_loop_row,
                Ora.Loop.valuesMatchVariables, Bool.and_false,
                Bool.false_eq_true] at hReady })

end Ora.Loop

/-
Repository-owned automation for the already-supported scalar LoopVC fragment.

The tactic is proof search only.  It does not change denotation, eligibility,
or artifact authority: Lean still checks the resulting term against the exact
compiler-emitted `loopProofAuthorizationFromAssumptions` proposition.

The generated call site must bind the exact local names `autoSummary` and
`emittedManifest`.  The macros resolve those names deliberately; this is the
stable Zig-to-Lean emission contract, not an implicit search convention.
-/

import Ora.Loop.StorageCertificate
import Lean.Elab.Tactic
import Lean.Meta.Tactic.Simp.RegisterCommand

namespace Ora.Loop

open Ora.Obligation

register_simp_attr ora_loop_data
register_simp_attr ora_loop_row
register_simp_attr ora_loop_supported
register_simp_attr ora_loop_manifest
register_simp_attr ora_loop_nested

open Lean Elab Tactic Meta

elab "ora_loop_all_goals" tac:tacticSeq : tactic => do
  let mut result : List MVarId := []
  for goal in ← getGoals do
    result := result ++ (← evalTacticAt tac goal)
  setGoals result

private def readyStateCarrier? (type : Expr) : MetaM Bool := do
  let type ← whnf type
  return type.isAppOfArity ``List 1 || type.isConstOf ``Ora.Integer.Value

private def readyStateLocal? (hReadyType : Expr) : TacticM (Option LocalDecl) := do
  for localDecl in ← getLCtx do
    if !localDecl.isAuxDecl &&
        hReadyType.containsFVar localDecl.fvarId &&
        (← readyStateCarrier? localDecl.type)
    then
      return some localDecl
  return none

private partial def unpackReadyStateAt
    (goal : MVarId)
    (hReadySyntax : Ident)
    (depth : Nat) : TacticM (List MVarId) := do
  _ := depth
  let hReadyLocation : TSyntax `Lean.Parser.Tactic.locationHyp :=
    ⟨Syntax.node SourceInfo.none `Lean.Parser.Tactic.locationHyp #[
      Syntax.node SourceInfo.none `null #[hReadySyntax.raw],
      Syntax.node SourceInfo.none `null #[]]⟩
  let reduced ← evalTacticAt (← `(tactic|
    try simp only [Ora.Loop.stateReady, Ora.Loop.valuesMatchVariables,
      ora_loop_row, TyRef.integerShape?, integerShapeForCompilerTypeId_u8,
      integerShapeForCompilerTypeId_u16, integerShapeForCompilerTypeId_u32,
      integerShapeForCompilerTypeId_u64, integerShapeForCompilerTypeId_u128,
      integerShapeForCompilerTypeId_u160, integerShapeForCompilerTypeId_u256,
      integerShapeForCompilerTypeId_i8, integerShapeForCompilerTypeId_i16,
      integerShapeForCompilerTypeId_i32, integerShapeForCompilerTypeId_i64,
      integerShapeForCompilerTypeId_i128, integerShapeForCompilerTypeId_i256,
      Bool.and_true, Bool.and_false, Bool.and_eq_true,
      Bool.false_eq_true, beq_iff_eq, Option.some.injEq,
      and_true, and_false, true_and, false_and, False.elim]
      at $hReadyLocation)) goal
  let mut result : List MVarId := []
  for reducedGoal in reduced do
    let next? ← reducedGoal.withContext do
      let saved ← getGoals
      try
        setGoals [reducedGoal]
        let hReadyId ← getFVarId hReadySyntax
        readyStateLocal? (← hReadyId.getType)
      finally
        setGoals saved
    match next? with
    | none => result := result ++ [reducedGoal]
    | some localDecl =>
        let branches ← reducedGoal.cases localDecl.fvarId
        for branch in branches do
          result := result ++ (← unpackReadyStateAt branch.mvarId hReadySyntax (depth + 1))
  return result

elab "ora_loop_unpack_state" hReady:ident : tactic => do
  let mut result : List MVarId := []
  for goal in ← getGoals do
    result := result ++ (← unpackReadyStateAt goal hReady 0)
  setGoals result

private def decomposableProofLocal? : TacticM (Option LocalDecl) := do
  for localDecl in ← getLCtx do
    if localDecl.isAuxDecl then continue
    let type := localDecl.type
    if type.isAppOfArity ``And 2 || type.isAppOfArity ``Exists 2 then
      return some localDecl
  return none

private partial def unpackConjunctionsAt (goal : MVarId) : TacticM (List MVarId) := do
  let next? ← goal.withContext do
    let saved ← getGoals
    try
      setGoals [goal]
      decomposableProofLocal?
    finally
      setGoals saved
  match next? with
  | none => return [goal]
  | some localDecl =>
      let mut result : List MVarId := []
      for branch in ← goal.cases localDecl.fvarId do
        result := result ++ (← unpackConjunctionsAt branch.mvarId)
      return result

elab "ora_loop_unpack_conjunctions" : tactic => do
  let mut result : List MVarId := []
  for goal in ← getGoals do
    result := result ++ (← unpackConjunctionsAt goal)
  setGoals result

macro "ora_loop_expand_membership" _value:ident hypothesis:ident : tactic =>
  `(tactic| repeat' first |
    rcases $hypothesis:ident with $hypothesis:ident | $hypothesis:ident)

private def integerValueLocal? : TacticM (Option LocalDecl) := do
  for localDecl in ← getLCtx do
    if !localDecl.isAuxDecl &&
        localDecl.type.isConstOf ``Ora.Integer.Value
    then
      return some localDecl
  return none

private partial def unpackIntegerValuesAt (goal : MVarId) : TacticM (List MVarId) := do
  let next? ← goal.withContext do
    let saved ← getGoals
    try
      setGoals [goal]
      integerValueLocal?
    finally
      setGoals saved
  match next? with
  | none => return [goal]
  | some localDecl =>
      let mut result : List MVarId := []
      for branch in ← goal.cases localDecl.fvarId do
        result := result ++ (← unpackIntegerValuesAt branch.mvarId)
      return result

elab "ora_loop_unpack_integer_values" : tactic => do
  let mut result : List MVarId := []
  for goal in ← getGoals do
    result := result ++ (← unpackIntegerValuesAt goal)
  setGoals result

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

theorem unsigned_zero_le
    (shape : Ora.Spec.IntegerShape)
    (bound : Ora.Integer.Carrier shape) :
    Ora.Integer.unsignedLe (BitVec.ofNat shape.bits 0) bound := by
  cases shape with
  | unsigned width =>
      cases width <;> simp only [Ora.Integer.unsignedLe,
        Ora.Spec.IntegerShape.bits, Ora.Spec.uintBits,
        BitVec.toNat_ofNat, Nat.reducePow, Nat.reduceMod, Nat.zero_le]
  | signed width =>
      cases width <;> simp only [Ora.Integer.unsignedLe,
        Ora.Spec.IntegerShape.bits, Ora.Spec.sintBits,
        BitVec.toNat_ofNat, Nat.reducePow, Nat.reduceMod, Nat.zero_le]

/-
The automatic proof reducer uses this closed allowlist.  Unlike an unrestricted
`simp`, its behavior cannot acquire new rewrites when unrelated imports add simp
lemmas.  Compiler-emitted term lookups, row definitions, and equality facts are
supplied separately through the local `ora_loop_data` and `ora_loop_row`
attributes.
-/
macro "ora_loop_reduce" loc:(Lean.Parser.Tactic.location)? : tactic =>
  `(tactic| simp only [ora_loop_data, ora_loop_row,
    assumptionsDenoteInEnv, assumptionsDenoteInEnv?, assumptionAnd?,
    assumptionDenotesInEnv?, contextReady, SummaryRow.supportEnv,
    bindSupportVariable, valueForTy, loopInitialPremises, denoteLoopSummary?,
    denoteEdges?, denoteEdge?, denoteControlTarget?, valuesInitializeState,
    initializeState?, stateReady, valuesMatchVariables, formulasHoldInState,
    formulaHoldsInState, safetyCheckpointsHold, safetyCheckpointHolds,
    environmentRelationsProduce, formulasHoldInEnv, formulaHoldsInEnv,
    stateBindingsProduceState, applyStateBindings?, bindingsMatchVariables,
    bindingFormulas, bindState?, evaluateIntegerFormulas?, integerFormulaValue?,
    formulaValue?, formulaDenotes?, denoteFormula?, denoteValue?,
    IntegerLiteralTerm.asInteger?, Ora.Integer.Value.ofNat,
    Ora.Integer.Value.relation?, Ora.Integer.Value.eqProp?,
    Ora.Integer.Value.binary?, Ora.Integer.Value.bitsFor?,
    Ora.Integer.wrappingResult, Ora.Integer.bitAndResult,
    Ora.Integer.bitXorResult, Ora.Integer.unsignedLt, Ora.Integer.unsignedLe,
    Ora.Integer.signedLt, Ora.Integer.signedLe, Ora.Integer.mathematicalValue,
    Ora.Spec.IntegerShape.isSigned, Ora.Spec.IntegerShape.bits,
    Ora.Spec.uintBits, Ora.Spec.sintBits, Env.setFree, Env.lookupVar,
    Env.lookupFree, lookupFreeBinding, lookupFreeBinding_cons_of_beq_true,
    lookupFreeBinding_cons_of_beq_false, TyRef.integerShape?,
    integerShapeForCompilerTypeId_u8, integerShapeForCompilerTypeId_u16,
    integerShapeForCompilerTypeId_u32, integerShapeForCompilerTypeId_u64,
    integerShapeForCompilerTypeId_u128, integerShapeForCompilerTypeId_u160,
    integerShapeForCompilerTypeId_u256, integerShapeForCompilerTypeId_i8,
    integerShapeForCompilerTypeId_i16, integerShapeForCompilerTypeId_i32,
    integerShapeForCompilerTypeId_i64, integerShapeForCompilerTypeId_i128,
    integerShapeForCompilerTypeId_i256, parseDecimalNat?, parseDecimalNatAux,
    decimalDigit?, Value.eqProp?, List.take, Option.pure_def,
    Option.bind_eq_bind, Option.some_bind, Option.none_bind, Option.map_some',
    Option.map_none', Option.some.injEq, Option.isSome, String.isEmpty,
    Option.ite_none_right_eq_some,
    List.isEmpty, List.not_mem_nil, Nat.zero_mul, Nat.mul_zero, Nat.zero_add,
    Nat.add_zero, Bool.not_false, Bool.not_true, Bool.and_true, Bool.true_and,
    Bool.and_false, Bool.false_and, Bool.false_eq_true, beq_iff_eq,
    beq_self_eq_true, and_true, and_false, true_and, false_and,
    if_pos, if_false, dif_pos, dif_neg] $[$loc:location]?)

macro "ora_loop_split_formulas" loc:(Lean.Parser.Tactic.location)? : tactic =>
  `(tactic| simp only [ora_loop_row, EdgeRow.path_eq_timeline, timelinePathView,
    formulasHoldInState, formulasHoldInEnv,
    List.forall_mem_cons, List.not_mem_nil, false_implies, and_true]
    $[$loc:location]?)

macro "ora_loop_formula_context_reduce" loc:(Lean.Parser.Tactic.location)? : tactic =>
  `(tactic| simp only [ora_loop_row, formulaHoldsInState, formulaHoldsInEnv,
    bindState?, TyRef.integerShape?, integerShapeForCompilerTypeId_u8,
    integerShapeForCompilerTypeId_u16, integerShapeForCompilerTypeId_u32,
    integerShapeForCompilerTypeId_u64, integerShapeForCompilerTypeId_u128,
    integerShapeForCompilerTypeId_u160, integerShapeForCompilerTypeId_u256,
    integerShapeForCompilerTypeId_i8, integerShapeForCompilerTypeId_i16,
    integerShapeForCompilerTypeId_i32, integerShapeForCompilerTypeId_i64,
    integerShapeForCompilerTypeId_i128, integerShapeForCompilerTypeId_i256,
    beq_iff_eq, beq_self_eq_true, Option.some.injEq, if_pos, if_false]
    $[$loc:location]?)

/-- Connect one store-independent exact-timeline formula to the existing
scalar formula denotation. Any current or entry storage read prevents the
closed reduction from finishing. -/
macro "ora_loop_direct_formula_control_in"
    autoSummary:ident env:ident : tactic => do
  `(tactic| {
    unfold Ora.Loop.directFormulaControlCompatible
      Ora.Loop.timelineFormulaHolds Ora.Loop.formulaHoldsInState
    intro state entryStorage
    cases state with
    | mk locals storage =>
        dsimp
        unfold Ora.Loop.directStorageSemantics
          Ora.Loop.StorageFormulaContext.evalFormula?
          Ora.Loop.StorageFormulaContext.stateEnv?
          Ora.Loop.directStorageContext
        simp only [Ora.Loop.bindStorageReads, List.foldl,
          Option.bind_eq_bind, Option.some_bind,
          Option.none_bind, Option.pure_def]
        cases hBound : Ora.Loop.bindState? $env
            ($autoSummary).variables locals with
        | none =>
            simp only [ora_loop_row] at hBound
            simp only [hBound, Option.none_bind]
            simp
        | some bound =>
            simp only [ora_loop_row] at hBound
            simp only [hBound, Option.some_bind]
            simp only [
              Ora.Loop.StorageFormulaContext.formulaStorageReady,
              Ora.Loop.StorageFormulaContext.termStorageReady,
              Ora.Loop.StorageFormulaContext.withStoreViews,
              ora_loop_data, formulaDenotes?, denoteFormula?, denoteValue?,
              Env.lookupVar, Env.lookupFree]
            exact Ora.Loop.option_prop_witness_iff _ })

/-- Connect store-independent preheader values to the scalar initializer.
The generated row and term graph determine the proof; storage reads or type
mismatches remain as open goals. -/
macro "ora_loop_direct_initial_control" : tactic =>
  `(tactic| {
    unfold Ora.Loop.directInitialControlCompatible
      Ora.Loop.initializeState? Ora.Loop.directInitialValues?
      Ora.Loop.StorageFormulaContext.evalPreheaderValue?
      Ora.Loop.directStorageContext Ora.Loop.evaluateIntegerFormulas?
      Ora.Loop.integerFormulaValue? formulaValue?
    intro entryStorage currentStorage
    simp only [ora_loop_row,
      Ora.Loop.StorageFormulaContext.formulaStorageReady,
      Ora.Loop.StorageFormulaContext.termStorageReady,
      Ora.Loop.StorageFormulaContext.withStoreViews,
      ora_loop_data, formulaDenotes?, denoteFormula?, denoteValue?,
      Env.lookupVar, Env.lookupFree, Option.bind_eq_bind,
      Option.some_bind, Option.none_bind, Option.pure_def,
      Option.map_some', Option.map_none']
    ora_loop_reduce
    simp only [Ora.Loop.directInitialValues?, Option.some_bind] })

macro "ora_loop_direct_timeline_reduce"
    loc:(Lean.Parser.Tactic.location)? : tactic =>
  `(tactic| simp only [ora_loop_row, Ora.Loop.EdgeRow.path_eq_timeline,
    Ora.Loop.timelinePathView, Ora.Loop.directStorageTimeline?,
    Ora.Loop.directStorageEvent?, Ora.Loop.directStorageLocation?,
    Ora.Loop.directStorageProjection?, Ora.Loop.directStorageProjections?,
    Ora.Loop.directStorageProjectionData,
    Ora.Loop.directStorageEventData, Ora.Loop.directStorageEvents,
    Ora.Loop.directStorageTarget, Ora.Loop.TimelineEvent.NoStorageRead,
    Ora.Loop.timelineReadIdsOf, Ora.Loop.TimelineEvent.readIds,
    Ora.Loop.TimelineEvent.check.injEq,
    Ora.Loop.TimelineEvent.assume.injEq,
    Ora.Loop.TimelineEvent.check_ne_assume,
    Ora.Loop.TimelineEvent.check_ne_storage,
    Ora.Loop.TimelineEvent.assume_ne_storage,
    List.flatMap, List.flatten, List.mem_append, List.mem_singleton,
    List.map, List.append, List.append_nil, List.append_assoc, List.mem_cons,
    List.not_mem_nil, List.drop, List.length, or_false,
    Bool.not_false, Bool.not_true,
    Bool.false_or, Bool.false_eq_true, if_false]
    $[$loc:location]?)

elab "ora_loop_try_no_read_state"
    result:ident hReadIds:ident hFinal:ident : tactic => do
  let mut transformed : List MVarId := []
  for goal in ← getGoals do
    let noReads ← goal.withContext do
      let readIdsId ← getFVarId hReadIds
      let readIdsType ← instantiateMVars (← readIdsId.getType)
      unless readIdsType.isAppOfArity ``Eq 3 do
        throwError "loop automation expected a read-identity equality"
      let rhs ← whnf readIdsType.getAppArgs[2]!
      return rhs.isAppOfArity ``List.nil 1
    if noReads then
      transformed := transformed ++ (← evalTacticAt (← `(tactic| (
        have hMappedReadsEmpty :
            ($result:ident).state.reads.map
              Ora.Loop.StorageReadBinding.id = [] := by
          rw [$hReadIds:ident]
          ora_loop_direct_timeline_reduce
        have hReadsEmpty : ($result:ident).state.reads = [] := by
          cases hReads : ($result:ident).state.reads with
          | nil => rfl
          | cons head tail =>
              simp only [hReads, List.map] at hMappedReadsEmpty
              cases hMappedReadsEmpty
        simp only [hReadsEmpty, Ora.Loop.bindStorageReads,
          List.foldl] at $hFinal:ident
      ))) goal)
    else
      transformed := transformed ++ [goal]
  setGoals transformed

macro "ora_loop_direct_edge_compatible"
    autoSummary:ident env:ident : tactic => do
  `(tactic| {
    exact {
      timeline := by rfl
      targetDenotes := by rfl
      assumptionPrefix := by
        ora_loop_direct_timeline_reduce
        all_goals rfl
      pathFormulaCompatible := by
        intro formula hMember
        all_goals ora_loop_direct_timeline_reduce at hMember
        all_goals ora_loop_expand_membership formula hMember
        all_goals ora_loop_direct_formula_control_in $autoSummary $env
      checkFormulaCompatible := by
        intro formula hMember
        ora_loop_direct_timeline_reduce at hMember
        ora_loop_expand_membership formula hMember
        all_goals ora_loop_direct_formula_control_in $autoSummary $env
      readOrderSupported := by
        ora_loop_direct_timeline_reduce
        decide
      finalLocalsAfterTrace := by
        intro current result locals hExec hFinal
        have hLocals := Ora.Loop.TimelineExec.preserves_locals _ _ _ _
          (by intro child; nomatch child) hExec
        have hReadIds := Ora.Loop.TimelineExec.read_ids _ _ _ _ hExec
        simp only [List.map_nil, List.nil_append] at hReadIds
        have hLookup : ∀ (bound : Env) (id : FreeVarId),
            id ∈ ((($autoSummary).contextVariables ++
              ($autoSummary).variables).map VariableRow.id) →
            (Ora.Loop.bindStorageReads bound result.state.reads).lookupFree id =
              bound.lookupFree id := by
          intro bound id hKnown
          apply Ora.Loop.bindStorageReads_lookupFree_of_not_mem
          rw [hReadIds]
          ora_loop_direct_timeline_reduce
          simp only [ora_loop_row, List.map, List.append,
            List.mem_cons, List.not_mem_nil, or_false] at hKnown
          ora_loop_expand_membership id hKnown
          all_goals decide
        simp only [ora_loop_row, List.map, List.append,
          List.mem_cons, List.not_mem_nil, or_false] at hLookup
        unfold Ora.Loop.stateBindingsProduceState
        rw [← hLocals]
        unfold Ora.Loop.directStorageFinalLocals?
          Ora.Loop.StorageFormulaContext.stateEnv?
          Ora.Loop.directStorageContext at hFinal
        unfold Ora.Loop.applyStateBindingsInEnv? at hFinal
        unfold Ora.Loop.applyStateBindings?
        simp only [Option.bind_eq_bind, Option.some_bind,
          Option.none_bind, Option.pure_def] at hFinal ⊢
        simp only [ora_loop_row] at hFinal ⊢
        cases hBound : Ora.Loop.bindState? $env ($autoSummary).variables
            result.state.loop.locals with
        | none =>
            simp only [ora_loop_row] at hBound
            simp only [hBound, Option.none_bind] at hFinal
            cases hFinal
        | some bound =>
            simp only [ora_loop_row] at hBound
            simp only [hBound, Option.some_bind] at hFinal ⊢
            ora_loop_try_no_read_state result hReadIds hFinal
            simp only [Ora.Loop.StorageFormulaContext.withStoreViews_lookupFree,
              Ora.Loop.bindingFormulas, Ora.Loop.bindingsMatchVariables,
              Ora.Loop.evaluateIntegerFormulas?, Ora.Loop.integerFormulaValue?,
              formulaValue?, ora_loop_row, ora_loop_data, denoteValue?,
              Env.lookupVar, hLookup,
              List.mem_cons, List.not_mem_nil, or_false,
              Option.bind_eq_bind, Option.some_bind, Option.none_bind,
              Option.pure_def, Option.map_some', Option.map_none'] at hFinal ⊢
            try simp only [hBound, Option.some_bind] at hFinal
            exact hFinal
      targetFormulaCompatible := by
        intro formula hMember
        all_goals simp only [Ora.Loop.directTargetFormulas, ora_loop_row,
          List.mem_cons, List.not_mem_nil, or_false] at hMember
        all_goals ora_loop_expand_membership formula hMember
        all_goals ora_loop_direct_formula_control_in $autoSummary $env
    } })

macro "ora_loop_value_reduce" loc:(Lean.Parser.Tactic.location)? : tactic =>
  `(tactic| set_option maxRecDepth 1024 in
    simp only [ora_loop_data, ora_loop_row, formulaValue?, formulaDenotes?,
    denoteFormula?, denoteValue?, IntegerLiteralTerm.asInteger?,
    Ora.Integer.Value.ofNat, Ora.Integer.Value.relation?,
    Ora.Integer.Value.bitsFor?, Ora.Integer.Value.binary?,
    Ora.Integer.wrappingResult, Ora.Integer.bitAndResult,
    Ora.Integer.bitXorResult, Ora.Spec.IntegerShape.isSigned,
    Env.setFree, Env.lookupVar, Env.lookupFree, lookupFreeBinding,
    lookupFreeBinding_cons_of_beq_true, lookupFreeBinding_cons_of_beq_false,
    TyRef.integerShape?, integerShapeForCompilerTypeId_u8,
    integerShapeForCompilerTypeId_u16, integerShapeForCompilerTypeId_u32,
    integerShapeForCompilerTypeId_u64, integerShapeForCompilerTypeId_u128,
    integerShapeForCompilerTypeId_u160, integerShapeForCompilerTypeId_u256,
    integerShapeForCompilerTypeId_i8, integerShapeForCompilerTypeId_i16,
    integerShapeForCompilerTypeId_i32, integerShapeForCompilerTypeId_i64,
    integerShapeForCompilerTypeId_i128, integerShapeForCompilerTypeId_i256,
    parseDecimalNat?, parseDecimalNatAux, decimalDigit?, Option.pure_def,
    Option.bind_eq_bind, Option.some_bind, Option.none_bind, Option.map_some',
    Option.map_none', Option.some.injEq, Nat.zero_mul, Nat.mul_zero,
    Nat.zero_add, Nat.add_zero, Bool.and_true, Bool.true_and,
    beq_self_eq_true, if_pos, if_false, Bool.false_eq_true]
    $[$loc:location]?)

macro "ora_loop_transition_reduce" loc:(Lean.Parser.Tactic.location)? : tactic =>
  `(tactic| simp only [ora_loop_row, stateBindingsProduceState, applyStateBindings?,
      applyStateBindingsInEnv?,
      bindingsMatchVariables, bindingFormulas, evaluateIntegerFormulas?,
      integerFormulaValue?, valuesMatchVariables, Option.pure_def,
      Option.bind_eq_bind, Option.some_bind, Option.none_bind,
      Option.ite_none_right_eq_some, Option.some.injEq, String.isEmpty,
      Bool.and_true, Bool.true_and, Bool.and_false, Bool.false_and,
      Bool.false_eq_true, if_pos, if_false] $[$loc:location]?)

macro "ora_loop_state_shape_reduce" loc:(Lean.Parser.Tactic.location)? : tactic =>
  `(tactic| simp only [ora_loop_row, valuesMatchVariables,
    TyRef.integerShape?, integerShapeForCompilerTypeId_u8,
    integerShapeForCompilerTypeId_u16, integerShapeForCompilerTypeId_u32,
    integerShapeForCompilerTypeId_u64, integerShapeForCompilerTypeId_u128,
    integerShapeForCompilerTypeId_u160, integerShapeForCompilerTypeId_u256,
    integerShapeForCompilerTypeId_i8, integerShapeForCompilerTypeId_i16,
    integerShapeForCompilerTypeId_i32, integerShapeForCompilerTypeId_i64,
    integerShapeForCompilerTypeId_i128, integerShapeForCompilerTypeId_i256,
    beq_self_eq_true, Bool.and_true, Option.ite_none_right_eq_some,
    Option.some.injEq] $[$loc:location]?)

elab "ora_loop_normalize_context_shapes" hContext:ident : tactic => do
  let hContextId ← getFVarId hContext
  let some hContextDecl := (← getLCtx).find? hContextId | return
  unless hContextDecl.type.isAppOfArity ``And 2 do return
  let location : TSyntax `Lean.Parser.Tactic.locationHyp :=
    ⟨Syntax.node SourceInfo.none `Lean.Parser.Tactic.locationHyp #[
      Syntax.node SourceInfo.none `null #[hContext.raw],
      Syntax.node SourceInfo.none `null #[]]⟩
  evalTactic (← `(tactic| ora_loop_state_shape_reduce at $location))

private def hasContextLookupFact : TacticM Bool := do
  let goal ← getMainGoal
  goal.withContext do
    for localDecl in ← getLCtx do
      let type ← whnf localDecl.type
      unless type.isAppOfArity ``Eq 3 do continue
      let args := type.getAppArgs
      let lhs := args[1]!
      let rhs := args[2]!
      if lhs.getAppFn.isConstOf ``Env.lookupFree ||
          lhs.getAppFn.isConstOf ``lookupFreeBinding ||
          rhs.getAppFn.isConstOf ``Env.lookupFree ||
          rhs.getAppFn.isConstOf ``lookupFreeBinding then
        return true
    return false

elab "ora_loop_apply_context" : tactic => do
  unless ← hasContextLookupFact do return
  evalTactic (← `(tactic| simp_all only [Env.lookupFree, TyRef.integerShape?,
    integerShapeForCompilerTypeId_u8, integerShapeForCompilerTypeId_u16,
    integerShapeForCompilerTypeId_u32, integerShapeForCompilerTypeId_u64,
    integerShapeForCompilerTypeId_u128, integerShapeForCompilerTypeId_u160,
    integerShapeForCompilerTypeId_u256, integerShapeForCompilerTypeId_i8,
    integerShapeForCompilerTypeId_i16, integerShapeForCompilerTypeId_i32,
    integerShapeForCompilerTypeId_i64, integerShapeForCompilerTypeId_i128,
    integerShapeForCompilerTypeId_i256, Option.some.injEq]))

macro "ora_loop_normalize" loc:(Lean.Parser.Tactic.location)? : tactic =>
  `(tactic| simp only [Value.eqProp?, Ora.Integer.Value.eqProp?,
    Ora.Integer.Value.relation?, Ora.Integer.Value.binary?,
    Ora.Integer.Value.bitsFor?, Ora.Spec.IntegerShape.isSigned,
    Ora.Spec.IntegerShape.bits, Ora.Spec.uintBits, Ora.Spec.sintBits,
    TyRef.integerShape?, integerShapeForCompilerTypeId_u8,
    integerShapeForCompilerTypeId_u16, integerShapeForCompilerTypeId_u32,
    integerShapeForCompilerTypeId_u64, integerShapeForCompilerTypeId_u128,
    integerShapeForCompilerTypeId_u160, integerShapeForCompilerTypeId_u256,
    integerShapeForCompilerTypeId_i8, integerShapeForCompilerTypeId_i16,
    integerShapeForCompilerTypeId_i32, integerShapeForCompilerTypeId_i64,
    integerShapeForCompilerTypeId_i128, integerShapeForCompilerTypeId_i256,
    Option.pure_def, Option.bind_eq_bind, Option.some_bind, Option.none_bind,
    Option.map_some', Option.map_none', Option.some.injEq,
    if_pos, if_false, dif_pos, dif_neg, beq_iff_eq, beq_self_eq_true,
    Bool.and_true, Bool.true_and, Bool.and_false, Bool.false_and,
    Bool.false_eq_true, and_true, true_and, implies_true]
    $[$loc:location]?)

macro "ora_loop_linear_close" : tactic =>
  `(tactic| (
    simp_all only [Ora.Integer.unsignedLt, Ora.Integer.unsignedLe,
      Ora.Integer.signedLt, Ora.Integer.signedLe,
      Ora.Integer.mathematicalValue, Ora.Spec.IntegerShape.isSigned,
      Ora.Spec.IntegerShape.bits, Ora.Spec.uintBits, Ora.Spec.sintBits,
      BitVec.toNat_ofNat, Nat.reducePow, Nat.reduceMod]
    omega))

macro "ora_loop_signed_add_safe" : tactic =>
  `(tactic| (
    apply Ora.Integer.signed_add_checkedSafe_of_nonnegative_bounded_result
    · simp only [Ora.Spec.IntegerShape.bits, Ora.Spec.sintBits,
        BitVec.toInt_ofNat, Int.Nat.cast_ofNat_Int, Nat.reducePow,
        Int.reduceBmod]
      omega
    · assumption
    · simp_all only [Ora.Integer.signedLt_iff_toInt_lt,
        Ora.Integer.signedLe_iff_toInt_le,
        Ora.Spec.IntegerShape.bits, Ora.Spec.sintBits,
        BitVec.toInt_ofNat, Int.Nat.cast_ofNat_Int, Nat.reducePow,
        Int.reduceBmod]
      omega))

macro "ora_loop_close" : tactic =>
  `(tactic| first
    | assumption
    | rfl
    | trivial
    | (apply Ora.Integer.signed_add_overflow_bit_nonnegative_of_checkedSafe
       ora_loop_signed_add_safe)
    | (apply Ora.Integer.signed_add_le_of_checkedSafe
       · ora_loop_signed_add_safe
       · simp_all only [Ora.Integer.signedLt_iff_toInt_lt,
           Ora.Integer.signedLe_iff_toInt_le,
           Ora.Spec.IntegerShape.bits, Ora.Spec.sintBits,
           BitVec.toInt_ofNat, Int.Nat.cast_ofNat_Int, Nat.reducePow,
           Int.reduceBmod]
         omega)
    | (simp_all only [Ora.Integer.unsignedLt, Ora.Integer.unsignedLe,
         Ora.Integer.signedLt, Ora.Integer.signedLe,
         Ora.Integer.mathematicalValue, Ora.Integer.wrappingResult,
         Ora.Spec.IntegerShape.isSigned, Ora.Spec.IntegerShape.bits,
         Ora.Spec.uintBits, Ora.Spec.sintBits,
         BitVec.toNat_ofNat, Nat.reducePow, Nat.reduceMod]
       first
       | (rw [BitVec.toNat_add_of_lt (by
            simp_all only [BitVec.toNat_ofNat, Nat.reducePow, Nat.reduceMod]
            omega)]
          simp_all only [BitVec.toNat_ofNat, Nat.reducePow, Nat.reduceMod]
          omega)
       | (rw [Ora.Integer.signed_add_toInt_eq_of_representable] <;>
           simp_all only [Ora.Integer.representable,
             Ora.Integer.mathematicalValue,
             Ora.Spec.IntegerShape.isSigned, Ora.Spec.IntegerShape.bits,
             Ora.Spec.sintBits, BitVec.toNat_ofNat,
             Nat.reducePow, Nat.reduceMod] <;> omega)
       | omega)
    | ora_loop_linear_close)

/- This reducer traverses the complete emitted nested-summary data.  Its local
recursion allowance is isolated from the induction and arithmetic proof steps. -/
macro "ora_loop_nested_reduce" : tactic =>
  `(tactic| set_option maxRecDepth 4096 in (
    first
      | unfold loopInitialPremisesIn denoteLoopSummaryIn?
      | unfold loopInductionObligationsAtEnvIn denoteLoopSummaryIn?
    simp only [ora_loop_nested]
    unfold denoteLoopSummaryWithFuel?
    simp only [denoteLoopSummaryWithResolver?, ora_loop_nested, if_true]
    simp only [denoteEdgesWithResolver?, denoteEdgeWithResolver?,
      denoteSummaryDependencies?]
    simp only [denoteSummaryDependency?]
    simp only [ora_loop_nested]
    simp only [denoteControlTarget?, Option.bind_eq_bind,
      Option.some_bind, Option.none_bind]
    simp only [ora_loop_nested]
    simp only [Option.isSome, if_true]
    simp only [Option.pure_def, Option.bind_eq_bind,
      Option.some_bind, Option.none_bind]
    simp only [ora_loop_row, EdgeRow.path_eq_timeline,
      EdgeRow.safety_eq_timeline, EdgeRow.dependencies_eq_timeline,
      timelinePathView, timelineSafetyView, timelineSafetyViewFrom,
      timelineDependencyView, Option.pure_def, Option.bind_eq_bind,
      Option.some_bind, Option.none_bind, Option.isSome,
      if_pos, if_false, Bool.false_eq_true]))

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
    have hFixedIndexBounds :
        ($autoSummary).fixedIndexBoundsSupported $emittedManifest = true := by
      set_option maxRecDepth 4096 in
        decide
    have hSupported : ($autoSummary).supported $emittedManifest = true := by
      simp only [ora_loop_manifest, ora_loop_data, Ora.Loop.SummaryRow.supported,
        hShape, hFixedIndexBounds, hFormulas, Bool.true_and]
    exact hSupported })

/-
Nested support is a deterministic computation over emitted row and term data.
The local recursion allowance covers that data traversal; it does not increase
the budget of the induction proof or of any arithmetic closer.
-/
macro "ora_loop_support_in"
    manifest:ident rows:ident row:ident : tactic =>
  `(tactic| {
    have hShape : ($row).shapeSupportedIn $rows = true := by
      set_option maxRecDepth 4096 in
        decide
    have hFormulas : ($row).formulasDenotableIn $manifest $rows = true := by
      set_option maxRecDepth 4096 in
        simp only [ora_loop_data, ora_loop_row, ora_loop_nested,
          Ora.Loop.SummaryRow.formulasDenotableIn,
          Ora.Loop.SummaryRow.supportEnv, Ora.Loop.SummaryRow.supportEnvIn,
          Ora.Loop.dependencyVariables, Ora.Loop.dependencyVariablesFor,
          Ora.Loop.safetyCheckpointDenotable, List.take,
          Ora.Loop.bindSupportVariable, Ora.Loop.valueForTy,
          Ora.Loop.edgeControlFormulasDenotable,
          Ora.Loop.edgeBindingsDenotable,
          Ora.Loop.formulaDenotable,
          Ora.Loop.valueDenotable, Ora.Loop.findSummaryRow?, List.all,
          TyRef.integerShape?, integerShapeForCompilerTypeId_u8,
          integerShapeForCompilerTypeId_u16,
          integerShapeForCompilerTypeId_u32,
          integerShapeForCompilerTypeId_u64,
          integerShapeForCompilerTypeId_u128,
          integerShapeForCompilerTypeId_u160,
          integerShapeForCompilerTypeId_u256,
          integerShapeForCompilerTypeId_i8,
          integerShapeForCompilerTypeId_i16,
          integerShapeForCompilerTypeId_i32,
          integerShapeForCompilerTypeId_i64,
          integerShapeForCompilerTypeId_i128,
          integerShapeForCompilerTypeId_i256,
          Env.setFree, Env.lookupFree, lookupFreeBinding,
          if_pos, if_false, formulaDenotes?, formulaValue?, denoteFormula?,
          denoteValue?, IntegerLiteralTerm.asInteger?,
          Ora.Integer.Value.ofNat, Ora.Integer.Value.relation?,
          Ora.Integer.Value.binary?, Ora.Integer.Value.bitsFor?,
          Ora.Integer.wrappingResult, Ora.Integer.bitAndResult,
          Ora.Integer.bitXorResult, Ora.Spec.IntegerShape.isSigned,
          Env.lookupVar, Bool.false_eq_true]
        decide
    have hFixedIndexBounds :
        ($row).fixedIndexBoundsSupported $manifest = true := by
      set_option maxRecDepth 4096 in
        decide
    simp only [Ora.Loop.SummaryRow.supportedIn, ora_loop_manifest,
      hShape, hFixedIndexBounds, hFormulas, Bool.true_and] })

macro tk:"ora_loop_prove_base" : tactic => do
  let autoSummary := Lean.mkIdentFrom tk `autoSummary
  let emittedManifest := Lean.mkIdentFrom tk `emittedManifest
  `(tactic| {
    intro state hInitial
    have hReady := Ora.Loop.initialized_state_ready
      $emittedManifest _ ($autoSummary).variables
      ($autoSummary).init state hInitial
    simp only [ora_loop_data, ora_loop_row,
      Ora.Loop.valuesInitializeState, Ora.Loop.initializeState?,
      Ora.Loop.evaluateIntegerFormulas?, Ora.Loop.integerFormulaValue?,
      Ora.Loop.valuesMatchVariables, formulaValue?, denoteValue?,
      IntegerLiteralTerm.asInteger?, Ora.Integer.Value.ofNat,
      Ora.Integer.Value.binary?, Ora.Integer.Value.bitsFor?,
      Ora.Integer.wrappingResult, Ora.Integer.bitAndResult,
      Ora.Integer.bitXorResult, Env.setFree, Env.lookupVar,
      Env.lookupFree, lookupFreeBinding, TyRef.integerShape?,
      integerShapeForCompilerTypeId_u8,
      integerShapeForCompilerTypeId_u16,
      integerShapeForCompilerTypeId_u32,
      integerShapeForCompilerTypeId_u64,
      integerShapeForCompilerTypeId_u128,
      integerShapeForCompilerTypeId_u160,
      integerShapeForCompilerTypeId_u256,
      integerShapeForCompilerTypeId_i8,
      integerShapeForCompilerTypeId_i16,
      integerShapeForCompilerTypeId_i32,
      integerShapeForCompilerTypeId_i64,
      integerShapeForCompilerTypeId_i128,
      integerShapeForCompilerTypeId_i256,
      parseDecimalNat?, parseDecimalNatAux, decimalDigit?,
      Option.pure_def, Option.bind_eq_bind, Option.some_bind,
      Option.none_bind, Option.map_some', Option.map_none',
      Bool.and_true, Bool.true_and, beq_iff_eq, beq_self_eq_true,
      if_pos, if_false, Bool.false_eq_true] at hInitial
    have hState := Option.some.inj hInitial
    subst state
    refine ⟨hReady, ?_⟩
    intro formula hFormula
    simp only [ora_loop_row, List.mem_cons, List.not_mem_nil, or_false]
      at hFormula
    repeat' first
      | subst formula
      | rcases hFormula with hFormula | hFormula
    all_goals
      simp only [Ora.Loop.formulaHoldsInState,
        Ora.Loop.bindState?, ora_loop_data, ora_loop_row,
        formulaDenotes?, denoteFormula?, denoteValue?,
        IntegerLiteralTerm.asInteger?, Ora.Integer.Value.ofNat,
        Ora.Integer.Value.relation?, Ora.Integer.Value.binary?,
        Ora.Integer.Value.bitsFor?, Ora.Integer.wrappingResult,
        Ora.Integer.bitAndResult, Ora.Integer.bitXorResult,
        Ora.Spec.IntegerShape.isSigned, Env.setFree, Env.lookupVar,
        Env.lookupFree, lookupFreeBinding, TyRef.integerShape?,
        integerShapeForCompilerTypeId_u8,
        integerShapeForCompilerTypeId_u16,
        integerShapeForCompilerTypeId_u32,
        integerShapeForCompilerTypeId_u64,
        integerShapeForCompilerTypeId_u128,
        integerShapeForCompilerTypeId_u160,
        integerShapeForCompilerTypeId_u256,
        integerShapeForCompilerTypeId_i8,
        integerShapeForCompilerTypeId_i16,
        integerShapeForCompilerTypeId_i32,
        integerShapeForCompilerTypeId_i64,
        integerShapeForCompilerTypeId_i128,
        integerShapeForCompilerTypeId_i256,
        parseDecimalNat?, parseDecimalNatAux, decimalDigit?,
        Value.eqProp?, Ora.Integer.Value.eqProp?,
        Option.pure_def, Option.bind_eq_bind, Option.some_bind,
        Option.none_bind, Option.map_some', Option.map_none',
        beq_self_eq_true, Bool.and_true, Bool.true_and,
        Ora.Integer.unsignedLt, Ora.Integer.unsignedLe,
        Ora.Integer.signedLt, Ora.Integer.signedLe,
        Ora.Integer.mathematicalValue, Ora.Spec.IntegerShape.bits,
        Ora.Spec.uintBits, Ora.Spec.sintBits,
        if_pos, if_false, dif_pos, dif_neg, Bool.false_eq_true]
    all_goals try ora_loop_apply_context
    all_goals first
      | assumption
      | exact Nat.zero_le _
      | apply unsigned_zero_le
      | decide
      | ora_loop_linear_close })

macro tk:"ora_loop_auto_plain" : tactic => do
  let autoSummary := Lean.mkIdentFrom tk `autoSummary
  let emittedManifest := Lean.mkIdentFrom tk `emittedManifest
  `(tactic| {
    have hSupported : ($autoSummary).supported $emittedManifest = true := by
      simp only [ora_loop_supported]
    constructor
    · let autoWitnessEnv : Env := ($autoSummary).supportEnv
      have hInitialSome :
          (Ora.Loop.initializeState? $emittedManifest autoWitnessEnv
            ($autoSummary).variables ($autoSummary).init).isSome = true := by
        dsimp [autoWitnessEnv]
        ora_loop_reduce
      cases hInitial : Ora.Loop.initializeState? $emittedManifest autoWitnessEnv
          ($autoSummary).variables ($autoSummary).init with
      | none =>
          simp only [hInitial, Option.isSome, Bool.false_eq_true] at hInitialSome
      | some initial =>
          refine ⟨autoWitnessEnv, initial, ?_⟩
          constructor
          · ora_loop_reduce
          constructor
          · dsimp [autoWitnessEnv]
            intro loopVar hLoopVar
            simp only [ora_loop_row, List.mem_cons, List.not_mem_nil, or_false]
              at hLoopVar
            repeat' first
              | subst loopVar
              | rcases hLoopVar with hLoopVar | hLoopVar
            all_goals try ora_loop_reduce
            all_goals try exact ⟨_, rfl, rfl⟩
          · simp only [Ora.Loop.loopInitialPremises,
              Ora.Loop.denoteLoopSummary?, hSupported, if_true,
              Ora.Loop.denoteEdges?, Ora.Loop.denoteEdge?,
              Ora.Loop.denoteControlTarget?]
            exact hInitial
    · intro env hAssumptions hContext
      try simp only [Ora.Loop.contextReady, ora_loop_row,
        List.forall_mem_cons, List.forall_mem_nil] at hContext
      ora_loop_normalize_context_shapes hContext
      ora_loop_unpack_conjunctions
      ora_loop_unpack_integer_values
      ora_loop_apply_context
      subst_vars
      change Ora.Loop.loopInductionObligationsAtEnv
        $emittedManifest env $autoSummary
      unfold Ora.Loop.loopInductionObligationsAtEnv
        Ora.Loop.denoteLoopSummary?
      rw [hSupported]
      unfold $autoSummary
      simp only [ora_loop_row, EdgeRow.path_eq_timeline,
        EdgeRow.safety_eq_timeline, timelinePathView, timelineSafetyView,
        timelineSafetyViewFrom, Ora.Loop.denoteEdges?, Ora.Loop.denoteEdge?,
        Ora.Loop.denoteControlTarget?, Option.bind, if_pos]
      refine { base := ?_, backedge := ?_, edgeSafety := ?_, exit := ?_ }
      · ora_loop_prove_base
      · intro edge current following hMember hTarget hInvariant hPath hSafe hNext
        simp only [ora_loop_row, List.mem_cons, List.not_mem_nil, or_false]
          at hMember
        repeat' first
          | subst edge
          | rcases hMember with hMember | hMember
        all_goals dsimp at hTarget hInvariant hPath hSafe hNext ⊢
        all_goals try contradiction
        all_goals (
          rcases hInvariant with ⟨hReady, hInvariant⟩
          have hFollowingReady :
              Ora.Loop.stateReady ($autoSummary).variables following := by
            apply Ora.Loop.bound_state_ready $emittedManifest env
              ($autoSummary).variables current following
            exact hNext
          ora_loop_unpack_state hReady)
        all_goals (
          repeat' first | rcases hReady with ⟨hShape, hReady⟩
          subst_vars
          ora_loop_transition_reduce at hNext
          ora_loop_formula_context_reduce at hNext
          ora_loop_value_reduce at hNext
          try simp only [if_pos True.intro, dif_pos True.intro,
            Option.some_bind, Option.map_some',
            Option.pure_def, Option.bind_eq_bind, beq_self_eq_true,
            Bool.and_true] at hNext
          try ora_loop_state_shape_reduce at hNext
          rcases hNext with ⟨hSourcesReady, hNextState⟩
          rcases hNextState with ⟨hConditionReady, hNextState⟩
          subst_vars
          constructor
          · exact hFollowingReady
          · ora_loop_split_formulas at hInvariant hPath ⊢
            all_goals ora_loop_formula_context_reduce at hInvariant hPath ⊢
            all_goals ora_loop_value_reduce at hInvariant hPath ⊢
            all_goals try ora_loop_apply_context
            all_goals ora_loop_normalize at *
            all_goals repeat' apply And.intro
            all_goals ora_loop_unpack_conjunctions
            all_goals subst_vars
            all_goals ora_loop_close)
      · intro edge state hMember hInvariant hPath
        simp only [ora_loop_row, List.mem_cons, List.not_mem_nil, or_false]
          at hMember
        repeat' first
          | subst edge
          | rcases hMember with hMember | hMember
        all_goals (
          dsimp at hInvariant hPath ⊢
          rcases hInvariant with ⟨hReady, hInvariant⟩
          ora_loop_unpack_state hReady
          repeat' first | rcases hReady with ⟨hShape, hReady⟩
          subst_vars
          simp only [ora_loop_row, EdgeRow.safety_eq_timeline,
            timelineSafetyView, timelineSafetyViewFrom,
            Ora.Loop.safetyCheckpointsHold,
            List.forall_mem_cons, List.forall_mem_nil])
        all_goals repeat' apply And.intro
        all_goals try (intro checkpoint hCheckpoint; contradiction)
        all_goals (
            unfold Ora.Loop.safetyCheckpointHolds
            intro beforeDependencies checkpointEnv hBound hProduce
            simp only [List.take, Ora.Loop.environmentRelationsProduce]
              at hProduce
            subst checkpointEnv
            ora_loop_formula_context_reduce at hBound
            subst beforeDependencies
            ora_loop_split_formulas at hInvariant hPath ⊢
            all_goals ora_loop_formula_context_reduce at hInvariant hPath ⊢
            all_goals ora_loop_value_reduce at hInvariant hPath ⊢
            all_goals try ora_loop_apply_context
            all_goals ora_loop_normalize at hInvariant hPath ⊢
            all_goals ora_loop_unpack_conjunctions
            all_goals subst_vars
            all_goals ora_loop_close)
      · intro edge current following hMember hTarget hInvariant hPath hSafe hNext
        simp only [ora_loop_row, List.mem_cons, List.not_mem_nil, or_false]
          at hMember
        repeat' first
          | subst edge
          | rcases hMember with hMember | hMember
        all_goals dsimp at hTarget hInvariant hPath hSafe hNext ⊢
        all_goals try contradiction
        all_goals (
          rcases hInvariant with ⟨hReady, hInvariant⟩
          ora_loop_unpack_state hReady)
        all_goals (
          repeat' first | rcases hReady with ⟨hShape, hReady⟩
          subst_vars
          ora_loop_transition_reduce at hNext
          ora_loop_formula_context_reduce at hNext
          ora_loop_value_reduce at hNext
          try simp only [if_pos True.intro, dif_pos True.intro,
            Option.some_bind, Option.map_some',
            Option.pure_def, Option.bind_eq_bind, beq_self_eq_true,
            Bool.and_true] at hNext
          try ora_loop_state_shape_reduce at hNext
          rcases hNext with ⟨hSourcesReady, hNextState⟩
          rcases hNextState with ⟨hConditionReady, hNextState⟩
          subst_vars
          ora_loop_split_formulas at hInvariant hPath ⊢
          all_goals ora_loop_formula_context_reduce at hInvariant hPath ⊢
          all_goals ora_loop_value_reduce at hInvariant hPath ⊢
          all_goals try ora_loop_apply_context
          all_goals ora_loop_normalize at *
          all_goals ora_loop_unpack_conjunctions
          all_goals subst_vars
          all_goals ora_loop_close) })

private partial def containsConstant (expr : Expr) (name : Name) : Bool :=
  match expr with
  | .const candidate _ => candidate == name
  | .app fn arg => containsConstant fn name || containsConstant arg name
  | .lam _ domain body _ | .forallE _ domain body _ =>
      containsConstant domain name || containsConstant body name
  | .letE _ type value body _ =>
      containsConstant type name || containsConstant value name ||
        containsConstant body name
  | .mdata _ inner => containsConstant inner name
  | .proj _ _ inner => containsConstant inner name
  | _ => false


private def dependencyLocation
    (ident : Ident) : TSyntax `Lean.Parser.Tactic.locationHyp :=
  ⟨Syntax.node SourceInfo.none `Lean.Parser.Tactic.locationHyp #[
    Syntax.node SourceInfo.none `null #[ident.raw],
    Syntax.node SourceInfo.none `null #[]]⟩

private def reduceSummaryDependencyAt
    (goal : MVarId)
    (localDecl : LocalDecl) : TacticM (List MVarId) := do
  let dependencyName := `hAutoDependency
  let renamedGoal ← goal.rename localDecl.fvarId dependencyName
  let hDependency : Ident := ⟨mkIdent dependencyName⟩
  let hDependencyLocation := dependencyLocation hDependency
  evalTacticAt (← `(tactic| (
    unfold Ora.Loop.summaryDependencyHolds at $hDependencyLocation
    unfold Ora.Loop.denoteLoopSummaryWithFuel? at $hDependencyLocation
    simp only [Ora.Loop.denoteLoopSummaryWithResolver?, ora_loop_nested,
      if_true] at $hDependencyLocation
    simp only [Ora.Loop.denoteEdgesWithResolver?,
      Ora.Loop.denoteEdgeWithResolver?,
      Ora.Loop.denoteSummaryDependencies?, EdgeRow.path_eq_timeline,
      EdgeRow.safety_eq_timeline, EdgeRow.dependencies_eq_timeline,
      timelinePathView, timelineSafetyView, timelineSafetyViewFrom,
      timelineDependencyView] at $hDependencyLocation
    simp only [Ora.Loop.denoteControlTarget?, Option.bind_eq_bind,
      Option.some_bind, Option.none_bind, Option.pure_def] at $hDependencyLocation
    split at $hDependencyLocation <;> simp_all only
    rcases $hDependency:ident with ⟨childInitial, childFinal, childBound,
      hChildInitial, hChildCertified, hBindChildFinal, hAfterChild⟩
    split at *
    all_goals try simp_all only [Option.some.injEq]
    all_goals try contradiction
    all_goals subst_vars
    all_goals rcases hChildCertified with
      ⟨hChildNotBackedge, hChildInitialRelation, childCurrent, childEdge,
        hChildInvariant, hChildMember, hChildTarget, hChildPath,
        hChildSafe, hChildNext⟩
    all_goals
      simp only [List.mem_cons, List.not_mem_nil, or_false] at hChildMember
    all_goals repeat' first
      | subst childEdge
      | rcases hChildMember with hChildMember | hChildMember
    all_goals
      (dsimp at hChildTarget hChildInvariant hChildPath hChildSafe hChildNext)
    all_goals try contradiction
    all_goals (
      rcases hChildInvariant with ⟨hChildReady, hChildInvariant⟩
      ora_loop_unpack_state hChildReady
      repeat' first | rcases hChildReady with ⟨hShape, hChildReady⟩
      subst_vars
      rw [Ora.Loop.stateBindingsProduceAfterNoDependencies_iff]
        at hChildNext
      ora_loop_transition_reduce at hChildNext
      ora_loop_formula_context_reduce at hChildNext
      ora_loop_value_reduce at hChildNext
      try simp only [if_pos True.intro, dif_pos True.intro,
        Option.some_bind, Option.map_some', Option.pure_def,
        Option.bind_eq_bind, beq_self_eq_true, Bool.and_true]
        at hChildNext
      try ora_loop_state_shape_reduce at hChildNext
      rcases hChildNext with ⟨hChildSourcesReady, hChildNextState⟩
      rcases hChildNextState with
        ⟨hChildConditionReady, hChildNextState⟩
      subst childFinal
      ora_loop_formula_context_reduce at hBindChildFinal
      ora_loop_value_reduce at hBindChildFinal
      subst childBound
      ora_loop_split_formulas at hChildInvariant hChildPath
      all_goals
        (ora_loop_formula_context_reduce at hChildInvariant hChildPath)
      all_goals (ora_loop_value_reduce at hChildInvariant hChildPath)
      all_goals try ora_loop_apply_context
      all_goals (ora_loop_normalize at hChildInvariant hChildPath)
      all_goals ora_loop_unpack_conjunctions
      all_goals subst_vars)))) renamedGoal

private def tailLocal? (goal : MVarId) : TacticM (Option LocalDecl) :=
  goal.withContext do
    for localDecl in ← getLCtx do
      if localDecl.userName == `hAutoDependencyTail then
        return some localDecl
    return none

private partial def reduceEnvironmentRelationsAt
    (goal : MVarId)
    (chainId : FVarId) : TacticM (List MVarId) := do
  let chainType ← goal.withContext do
    whnf (← instantiateMVars (← getFVarLocalDecl (mkFVar chainId)).type)
  if chainType.isEq then
    return ← evalTacticAt (← `(tactic| subst_vars)) goal
  unless chainType.isAppOfArity ``Exists 2 do
    throwError "loop automation expected an environment-relation chain"
  let existsCases ← goal.cases chainId
  let existsCase ← match existsCases.toList with
    | [branch] => pure branch
    | _ => throwError "loop automation could not decompose an environment relation"
  unless existsCase.fields.size >= 2 do
    throwError "loop automation did not expose the relation witness"
  let relationPairId := existsCase.fields[existsCase.fields.size - 1]!.fvarId!
  let relationCases ← existsCase.mvarId.cases relationPairId
  let relationCase ← match relationCases.toList with
    | [branch] => pure branch
    | _ => throwError "loop automation expected a dependency and relation tail"
  unless relationCase.fields.size == 2 do
    throwError "loop automation did not expose both relation components"
  let dependencyId := relationCase.fields[0]!.fvarId!
  let tailId := relationCase.fields[1]!.fvarId!
  let renamedGoal ← relationCase.mvarId.rename tailId `hAutoDependencyTail
  let dependency ← renamedGoal.withContext do
    getFVarLocalDecl (mkFVar dependencyId)
  let mut result : List MVarId := []
  for reduced in ← reduceSummaryDependencyAt renamedGoal dependency do
    match ← tailLocal? reduced with
    | some tailDecl =>
        result := result ++
          (← reduceEnvironmentRelationsAt reduced tailDecl.fvarId)
    | none => result := result ++ [reduced]
  return result

elab "ora_loop_reduce_environment_relations" hChain:ident : tactic => do
  let mut result : List MVarId := []
  for goal in ← getGoals do
    let chainId ← goal.withContext do getFVarId hChain
    result := result ++ (← reduceEnvironmentRelationsAt goal chainId)
  setGoals result

macro "ora_loop_finish_nested_backedges"
    hApply:ident following:ident hFollowingReady:ident : tactic => do
  let hApplyLocation := dependencyLocation hApply
  `(tactic| {
    ora_loop_transition_reduce at $hApplyLocation
    ora_loop_value_reduce at $hApplyLocation
    try simp only [if_pos True.intro, dif_pos True.intro,
      Option.some_bind, Option.map_some', Option.pure_def,
      Option.bind_eq_bind, beq_self_eq_true, Bool.and_true]
      at $hApplyLocation
    try ora_loop_state_shape_reduce at $hApplyLocation
    rcases $hApply:ident with ⟨hSourcesReady, hNextState⟩
    rcases hNextState with ⟨hConditionReady, hNextState⟩
    subst $following:ident
    refine ⟨$hFollowingReady:term, ?_⟩
    ora_loop_split_formulas
    all_goals ora_loop_formula_context_reduce
    all_goals ora_loop_value_reduce
    all_goals try ora_loop_apply_context
    all_goals ora_loop_normalize
    all_goals repeat' apply And.intro
    all_goals ora_loop_unpack_conjunctions
    all_goals subst_vars
    all_goals ora_loop_close })

macro "ora_loop_finish_nested_safety" : tactic =>
  `(tactic| {
    subst_vars
    ora_loop_split_formulas
    all_goals ora_loop_formula_context_reduce
    all_goals ora_loop_value_reduce
    all_goals try ora_loop_apply_context
    all_goals ora_loop_normalize
    all_goals ora_loop_unpack_conjunctions
    all_goals subst_vars
    all_goals ora_loop_close })

macro tk:"ora_loop_auto_nested" : tactic => do
  let autoSummary := Lean.mkIdentFrom tk `autoSummary
  let emittedManifest := Lean.mkIdentFrom tk `emittedManifest
  let emittedLoopSummaries := Lean.mkIdentFrom tk `emittedLoopSummaries
  `(tactic| {
    have hSupported : ($autoSummary).supportedIn $emittedManifest
        $emittedLoopSummaries = true := by
      simp only [ora_loop_nested]
    constructor
    · let autoWitnessEnv : Env :=
        ($autoSummary).supportEnvIn $emittedLoopSummaries
      have hInitialSome :
          (Ora.Loop.initializeState? $emittedManifest autoWitnessEnv
            ($autoSummary).variables ($autoSummary).init).isSome = true := by
        dsimp [autoWitnessEnv]
        simp only [ora_loop_nested, Ora.Loop.SummaryRow.supportEnvIn,
          Ora.Loop.dependencyVariables, Ora.Loop.findSummaryRow?,
          Option.pure_def, Option.bind_eq_bind, Option.some_bind,
          Option.none_bind, if_pos, if_false]
        simp only [ora_loop_row]
        ora_loop_reduce
      cases hInitial : Ora.Loop.initializeState? $emittedManifest autoWitnessEnv
          ($autoSummary).variables ($autoSummary).init with
      | none =>
          simp only [hInitial, Option.isSome, Bool.false_eq_true] at hInitialSome
      | some initial =>
          refine ⟨autoWitnessEnv, initial, ?_⟩
          constructor
          · ora_loop_reduce
          constructor
          · dsimp [autoWitnessEnv]
            intro loopVar hLoopVar
            simp only [Ora.Loop.SummaryRow.supportEnvIn,
              Ora.Loop.dependencyVariables, Ora.Loop.findSummaryRow?,
              ora_loop_nested, Option.pure_def, Option.bind_eq_bind,
              Option.some_bind, Option.none_bind, if_pos, if_false]
              at hLoopVar ⊢
            simp only [ora_loop_row, List.mem_cons, List.not_mem_nil, or_false]
              at hLoopVar ⊢
            repeat' first
              | subst loopVar
              | rcases hLoopVar with hLoopVar | hLoopVar
            all_goals try ora_loop_reduce
            all_goals try exact ⟨_, rfl, rfl⟩
          · ora_loop_nested_reduce
            exact hInitial
    · intro env hAssumptions hContext
      change Ora.Loop.loopInductionObligationsAtEnvIn
        $emittedManifest $emittedLoopSummaries env $autoSummary
      ora_loop_nested_reduce
      constructor
      · ora_loop_prove_base
      · intro edge current following hMember hTarget hInvariant hPath hSafe hNext
        simp only [ora_loop_row, List.mem_cons, List.not_mem_nil, or_false]
          at hMember
        repeat' first
          | subst edge
          | rcases hMember with hMember | hMember
        all_goals dsimp at hTarget hInvariant hPath hSafe hNext ⊢
        all_goals try contradiction
        ora_loop_all_goals (
          rcases hInvariant with ⟨hReady, hInvariant⟩
          have hFollowingReady :
              Ora.Loop.stateReady ($autoSummary).variables following := by
            apply Ora.Loop.bound_state_after_dependencies_ready
              $emittedManifest env ($autoSummary).variables current _ _ following
            exact hNext
          ora_loop_unpack_state hReady
          repeat' first | rcases hReady with ⟨hShape, hReady⟩
          subst_vars
          rcases hNext with ⟨beforeDependencies, afterDependencies,
            hParentBound, hDependencies, hApply⟩
          ora_loop_formula_context_reduce at hParentBound
          ora_loop_value_reduce at hParentBound
          subst beforeDependencies
          ora_loop_split_formulas at hInvariant hPath
          all_goals (ora_loop_formula_context_reduce at hInvariant hPath)
          all_goals (ora_loop_value_reduce at hInvariant hPath)
          all_goals try ora_loop_apply_context
          all_goals (ora_loop_normalize at hInvariant hPath)
          all_goals ora_loop_unpack_conjunctions
          simp only [Ora.Loop.environmentRelationsProduce] at hDependencies
          ora_loop_reduce_environment_relations hDependencies
          all_goals
            ora_loop_finish_nested_backedges hApply following hFollowingReady)
      · intro edge state hMember hInvariant hPath
        simp only [ora_loop_row, List.mem_cons, List.not_mem_nil, or_false]
          at hMember
        repeat' first
          | subst edge
          | rcases hMember with hMember | hMember
        all_goals (
          dsimp at hInvariant hPath ⊢
          rcases hInvariant with ⟨hReady, hInvariant⟩
          ora_loop_unpack_state hReady
          repeat' first | rcases hReady with ⟨hShape, hReady⟩
          subst_vars
          simp only [ora_loop_row, EdgeRow.safety_eq_timeline,
            timelineSafetyView, timelineSafetyViewFrom,
            Ora.Loop.safetyCheckpointsHold,
            List.forall_mem_cons, List.forall_mem_nil])
        all_goals repeat' apply And.intro
        all_goals try (intro checkpoint hCheckpoint; contradiction)
        ora_loop_all_goals (
          unfold Ora.Loop.safetyCheckpointHolds
          intro beforeDependencies checkpointEnv hParentBound hDependencies
          ora_loop_formula_context_reduce at hParentBound
          ora_loop_value_reduce at hParentBound
          subst beforeDependencies
          ora_loop_split_formulas at hInvariant hPath
          all_goals (ora_loop_formula_context_reduce at hInvariant hPath)
          all_goals (ora_loop_value_reduce at hInvariant hPath)
          all_goals try ora_loop_apply_context
          all_goals (ora_loop_normalize at hInvariant hPath)
          all_goals ora_loop_unpack_conjunctions
          simp only [List.take, Ora.Loop.environmentRelationsProduce]
            at hDependencies
          ora_loop_reduce_environment_relations hDependencies
          all_goals ora_loop_finish_nested_safety)
      · intro edge current following hMember hTarget hInvariant hPath hSafe hNext
        simp only [ora_loop_row, List.mem_cons, List.not_mem_nil, or_false]
          at hMember
        repeat' first
          | subst edge
          | rcases hMember with hMember | hMember
        all_goals dsimp at hTarget hInvariant hPath hSafe hNext ⊢
        all_goals try contradiction
        all_goals (
          rcases hInvariant with ⟨hReady, hInvariant⟩
          ora_loop_unpack_state hReady
          repeat' first | rcases hReady with ⟨hShape, hReady⟩
          subst_vars
          rw [Ora.Loop.stateBindingsProduceAfterNoDependencies_iff] at hNext
          ora_loop_transition_reduce at hNext
          ora_loop_formula_context_reduce at hNext
          ora_loop_value_reduce at hNext
          try simp only [if_pos True.intro, dif_pos True.intro,
            Option.some_bind, Option.map_some', Option.pure_def,
            Option.bind_eq_bind, beq_self_eq_true, Bool.and_true] at hNext
          try ora_loop_state_shape_reduce at hNext
          rcases hNext with ⟨hSourcesReady, hNextState⟩
          rcases hNextState with ⟨hConditionReady, hNextState⟩
          subst following
          ora_loop_split_formulas at hInvariant hPath ⊢
          all_goals ora_loop_formula_context_reduce at hInvariant hPath ⊢
          all_goals ora_loop_value_reduce at hInvariant hPath ⊢
          all_goals try ora_loop_apply_context
          all_goals ora_loop_normalize at hInvariant hPath ⊢
          all_goals ora_loop_unpack_conjunctions
          all_goals subst_vars
          all_goals ora_loop_close) })

macro tk:"ora_loop_auto_direct_storage" : tactic => do
  let autoSummary := Lean.mkIdentFrom tk `autoSummary
  let emittedManifest := Lean.mkIdentFrom tk `emittedManifest
  `(tactic| {
    have hScalar : Ora.Loop.loopProofAuthorizationFromAssumptions
        $emittedManifest [] $autoSummary := by
      ora_loop_auto_plain
    change Ora.Loop.directStorageLoopProofAuthorizationFromAssumptions
      $emittedManifest [] $autoSummary
    have hDirect : Ora.Loop.directStorageSummarySupported
        $emittedManifest $autoSummary = true := by
      have hSupported : ($autoSummary).supported $emittedManifest = true := by
        ora_loop_support
      unfold Ora.Loop.directStorageSummarySupported
      rw [hSupported]
      simp only [Bool.true_and, ora_loop_row, List.any, List.all]
      decide
    have hPreheaderEmpty : ($autoSummary).preheaderTimeline.isEmpty = true := by
      simp only [ora_loop_row, List.isEmpty]
    constructor
    · rcases hScalar.1 with
        ⟨env, locals, hAssumptions, hContext, hInitial⟩
      refine ⟨env, {
        locals := locals
        storage := Ora.Loop.directStoragePreheaderState env $autoSummary
      }, hAssumptions, hContext, ?_⟩
      have hEdgeCompatible : ∀ edge, edge ∈ ($autoSummary).edges →
          Ora.Loop.DirectStorageEdgeCompatible $emittedManifest env
            ($autoSummary).variables ($autoSummary).post edge := by
        intro edge hMember
        simp only [ora_loop_row, List.mem_cons, List.not_mem_nil, or_false]
          at hMember
        ora_loop_expand_membership edge hMember
        all_goals ora_loop_direct_edge_compatible $autoSummary env
      rcases Ora.Loop.direct_storage_summary_denotes $emittedManifest env
          $autoSummary hDirect hPreheaderEmpty hEdgeCompatible with
          ⟨concreteVC, hConcrete⟩
      rw [hConcrete]
      unfold Ora.Loop.loopInitialPremises at hInitial
      cases hAbstract : Ora.Loop.denoteLoopSummary?
          $emittedManifest env $autoSummary with
      | none => simp [hAbstract] at hInitial
      | some abstractVC =>
          simp only [hAbstract] at hInitial
          have hValues := Ora.Loop.scalar_initial_values_of_denotation
            $emittedManifest env $autoSummary abstractVC locals hAbstract hInitial
          apply Ora.Loop.direct_initial_of_control_denotation
            $emittedManifest env $autoSummary concreteVC locals hConcrete
          · exact hPreheaderEmpty
          · ora_loop_direct_initial_control
          · exact hValues
    · intro env hAssumptions hContext
      have hScalarInduction := hScalar.2 env hAssumptions hContext
      unfold Ora.Loop.loopInductionObligationsAtEnv at hScalarInduction
      cases hAbstract : Ora.Loop.denoteLoopSummary?
          $emittedManifest env $autoSummary with
      | none => simp [hAbstract] at hScalarInduction
      | some abstractVC =>
          simp only [hAbstract] at hScalarInduction
          have hEdgeCompatible : ∀ edge, edge ∈ ($autoSummary).edges →
              Ora.Loop.DirectStorageEdgeCompatible $emittedManifest env
                ($autoSummary).variables ($autoSummary).post edge := by
            intro edge hMember
            simp only [ora_loop_row, List.mem_cons, List.not_mem_nil, or_false]
              at hMember
            ora_loop_expand_membership edge hMember
            all_goals ora_loop_direct_edge_compatible $autoSummary env
          rcases Ora.Loop.direct_storage_summary_denotes $emittedManifest env
              $autoSummary hDirect hPreheaderEmpty hEdgeCompatible with
              ⟨concreteVC, hConcrete⟩
          simp only [hConcrete]
          apply Ora.Loop.induction_obligations_of_refinement
            abstractVC concreteVC (fun state => state.locals)
            hScalarInduction
          apply Ora.Loop.direct_storage_summary_refinement
            $emittedManifest env $autoSummary abstractVC concreteVC
            hAbstract hConcrete
          · exact hPreheaderEmpty
          · ora_loop_direct_initial_control
          · intro formula hMember
            simp only [ora_loop_row, List.mem_cons, List.not_mem_nil,
              or_false] at hMember
            ora_loop_expand_membership formula hMember
            all_goals ora_loop_direct_formula_control_in $autoSummary env
          · exact hEdgeCompatible })

elab _tk:"ora_loop_auto" : tactic => do
  let goal ← getMainGoal
  let target ← whnf (← goal.getType)
  if containsConstant target
      ``Ora.Loop.directStorageLoopProofAuthorizationFromAssumptions ||
      containsConstant target ``Ora.Loop.denoteDirectStorageLoopSummary? then
    evalTactic (← `(tactic| ora_loop_auto_direct_storage))
  else if containsConstant target ``Ora.Loop.loopInductionCertificateFromAssumptionsIn ||
      containsConstant target ``Ora.Loop.loopSpecificationNonvacuousIn then
    evalTactic (← `(tactic| ora_loop_auto_nested))
  else
    evalTactic (← `(tactic| ora_loop_auto_plain))

end Ora.Loop

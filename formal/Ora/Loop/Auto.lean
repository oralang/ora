/-
Repository-owned automation for the already-supported scalar LoopVC fragment.

The tactic is proof search only.  It does not change denotation, eligibility,
or artifact authority: Lean still checks the resulting term against the exact
compiler-emitted `loopProofAuthorizationFromAssumptions` proposition.

The generated call site must bind `autoSummary`, `autoAssumptions`, and
`emittedManifest`. The macros resolve those names deliberately; this is the
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
register_simp_attr ora_loop_shape
register_simp_attr ora_loop_manifest
register_simp_attr ora_loop_nested

open Lean Elab Tactic Meta

/-- Candidate search only: half-open mathematical intervals cannot wrap at a
carrier boundary. Unsupported atoms contribute no bound, never a proof. -/
def loopWitnessInterval? (op : Ora.Obligation.BinaryOp) (value : Nat)
    (bounds : Nat × Nat) : Option (Nat × Nat) := do
  let next := match op with
    | .eq => (max bounds.1 value, min bounds.2 (value + 1))
    | .lt => (bounds.1, min bounds.2 value)
    | .le => (bounds.1, min bounds.2 (value + 1))
    | .gt => (max bounds.1 (value + 1), bounds.2)
    | .ge => (max bounds.1 value, bounds.2)
    | _ => bounds
  if next.1 < next.2 then some next else none

def loopWitnessTermBounds? (manifest : Manifest) (loopVar : VariableRow) :
    Nat → TermId → Nat × Nat → Option (Nat × Nat)
  | 0, _, _ => none
  | fuel + 1, id, bounds => do
      let term ← manifest.terms[id]?
      match term with
      | .binary binary =>
          if binary.op == .and_ then
            let narrowed ← loopWitnessTermBounds? manifest loopVar fuel binary.lhs bounds
            loopWitnessTermBounds? manifest loopVar fuel binary.rhs narrowed
          else if ![.eq, .lt, .le, .gt, .ge].contains binary.op then
            some bounds
          else do
            let lhs ← manifest.terms[binary.lhs]?
            let rhs ← manifest.terms[binary.rhs]?
            let pair := match lhs, rhs with
              | .variable (.free free), .intLit literal => some (free, literal, false)
              | .intLit literal, .variable (.free free) => some (free, literal, true)
              | _, _ => none
            match pair with
            | none => some bounds
            | some (free, literal, reverse) => do
                if free.id != loopVar.id then return bounds
                if free.ty != some loopVar.ty || literal.ty != some loopVar.ty then none
                else if binary.op != .eq && binary.ty != some loopVar.ty then none
                else do
                  let shape ← loopVar.ty.integerShape?
                  if shape.isSigned then none else do
                    let value ← parseDecimalNat? literal.value
                    if value ≥ 2 ^ shape.bits then none else
                      let op := if reverse then match binary.op with
                        | .lt => .gt | .le => .ge | .gt => .lt | .ge => .le
                        | op => op
                      else binary.op
                      loopWitnessInterval? op value bounds
      | _ => some bounds

/-- Overrides only unsigned context bindings in a supplied support environment.
All assumptions and the complete initial predicate still have to be proved;
neither this data nor a successful interval intersection authorizes anything. -/
def loopWitnessCandidate? (manifest : Manifest) (assumptions : List AssumptionRow)
    (row : SummaryRow) (base : Env) : Option Env := do
  let assumptionFormulas ← assumptions.mapM (·.formula)
  let formulas := assumptionFormulas ++ row.headerFacts.map (·.formula)
  let mut env := base
  for loopVar in row.contextVariables do
    let shape ← loopVar.ty.integerShape?
    if !shape.isSigned then
      let mut bounds := (0, 2 ^ shape.bits)
      for formula in formulas do
        let .term id := formula
        bounds ← loopWitnessTermBounds? manifest loopVar (id + 1) id bounds
      env := env.setFree loopVar.id (.integer (Ora.Integer.Value.ofNat shape bounds.1))
  return env

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
      ora_loop_row, ora_loop_integer_types,
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

theorem bitvec_zero_toNat_le (bound : BitVec width) :
    (0#width).toNat ≤ bound.toNat := by
  simp only [BitVec.toNat_ofNat, Nat.zero_mod, Nat.zero_le]

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
    headerFactsHoldInEnv,
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
    lookupFreeBinding_cons_of_beq_false, ora_loop_integer_types,
    parseDecimalNat?, parseDecimalNatAux,
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
    List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true]
    $[$loc:location]?)

/-- A literal-false path is impossible even before its state bindings are
reduced. This avoids replaying transitions for short-circuit dead branches. -/
theorem formulaHoldsInState_false_literal (manifest : Manifest) (base : Env)
    (variables : List VariableRow) (state : ScalarLoopState) (id : TermId)
    (hTerm : manifest.terms[id]? = some (.boolLit false)) :
    formulaHoldsInState manifest base variables state (.term id) ↔ False := by
  unfold formulaHoldsInState
  cases hBound : bindState? base variables state <;>
    simp [formulaDenotes?, denoteFormula?, hTerm]

macro "ora_loop_false_path" h:ident : tactic =>
  `(tactic| solve | simp (disch := simp only [ora_loop_data, reduceCtorEq]) only
    [ora_loop_row, EdgeRow.path_eq_timeline, timelinePathView,
      List.map, List.cons_append, List.nil_append,
      formulasHoldInState, List.forall_mem_cons, List.forall_mem_nil,
      formulaHoldsInState_false_literal, and_false, false_and] at $h:ident)

macro "ora_loop_split_header_facts" h:ident : tactic =>
  `(tactic| simp only [headerFactsHoldInEnv, ora_loop_row,
    List.forall_mem_cons, List.not_mem_nil, false_implies, forall_const, and_true]
    at $h:ident)

macro "ora_loop_formula_context_reduce" loc:(Lean.Parser.Tactic.location)? : tactic =>
  `(tactic| simp only [ora_loop_row, formulaHoldsInState, formulaHoldsInEnv,
    bindState?, ora_loop_integer_types,
    beq_iff_eq, beq_self_eq_true, Option.some.injEq, if_pos, if_false]
    $[$loc:location]?)

/-- Connect one store-independent exact-timeline formula to the existing
scalar formula denotation. Any current or entry storage read prevents the
closed reduction from finishing. -/
macro "ora_loop_direct_formula_control_in"
    autoSummary:ident env:ident : tactic => do
  `(tactic| first
  | solve
    | apply Ora.Loop.directFormulaControlCompatible_bool_literal
      simp only [ora_loop_data]
  | {
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
            simp only [ora_loop_row, hBound, Option.none_bind]
            simp
        | some bound =>
            simp only [ora_loop_row] at hBound
            simp only [hBound, Option.some_bind]
            set_option maxRecDepth 800 in
              simp only [Ora.Loop.StorageFormulaContext.formulaStorageReady]
            set_option maxRecDepth 800 in
              simp only [Ora.Loop.StorageFormulaContext.termStorageReady,
                ora_loop_data]
            set_option maxRecDepth 800 in
              simp only [Bool.not_true, Bool.false_eq_true, if_false,
                Ora.Loop.StorageFormulaContext.withStoreViews,
                formulaDenotes?, denoteFormula?, denoteValue?,
                ora_loop_data, Env.lookupVar, Env.lookupFree]
            set_option maxRecDepth 800 in
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
    Ora.Loop.directIndexedRegionKind, Ora.Loop.directIndexedRegionRead?,
    ora_loop_integer_types,
    Option.bind_eq_bind, Option.some_bind, Option.none_bind, Option.pure_def,
    Ora.Loop.directStorageProjection?, Ora.Loop.directStorageProjections?,
    Ora.Loop.directStorageProjectionData,
    Ora.Loop.directStorageEventData, Ora.Loop.directStorageEvents,
    Ora.Loop.directStorageTarget, Ora.Loop.TimelineEvent.NoRead,
    Ora.Loop.timelineReadIdsOf, Ora.Loop.TimelineEvent.readIds,
    Ora.Loop.TimelineEvent.check.injEq,
    Ora.Loop.TimelineEvent.assume.injEq,
    Ora.Loop.TimelineEvent.check_ne_assume,
    Ora.Loop.TimelineEvent.check_ne_storage,
    Ora.Loop.TimelineEvent.check_ne_indexedRead,
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

syntax "ora_loop_prefix_checks" : tactic

macro "ora_loop_direct_edge_compatible"
    autoSummary:ident env:ident : tactic => do
  `(tactic| {
    exact {
      timeline := by rfl
      targetDenotes := by rfl
      pathMatches := by
        intro formula
        ora_loop_direct_timeline_reduce
        all_goals simp only [reduceCtorEq, false_or, or_false]
      formulaCompatible := by
        intro formula hMember
        ora_loop_direct_timeline_reduce at hMember
        try simp only [reduceCtorEq, false_or, or_false] at hMember
        all_goals ora_loop_expand_membership formula hMember
        ora_loop_all_goals (ora_loop_direct_formula_control_in $autoSummary $env)
      controlChecks := by ora_loop_prefix_checks
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
          Ora.Loop.storageReadsForBindings Ora.Loop.storageReadsForIds
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
              List.flatMap, List.map, List.flatten,
              List.nil_append, List.append_nil,
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
        ora_loop_all_goals (ora_loop_direct_formula_control_in $autoSummary $env)
    } })

macro "ora_loop_value_reduce" loc:(Lean.Parser.Tactic.location)? : tactic =>
  `(tactic| set_option maxRecDepth 1024 in
    simp only [*, ora_loop_data, ora_loop_row, formulaValue?, formulaDenotes?,
    denoteFormula?, denoteValue?, IntegerLiteralTerm.asInteger?,
    Ora.Integer.Value.ofNat, Ora.Integer.Value.relation?,
    Ora.Integer.Value.bitsFor?, Ora.Integer.Value.binary?,
    Ora.Integer.wrappingResult, Ora.Integer.bitAndResult,
    Ora.Integer.bitXorResult, Ora.Spec.IntegerShape.isSigned,
    SummaryRow.supportEnv, bindSupportVariable, valueForTy,
    Env.setFree, Env.lookupVar, Env.lookupFree, lookupFreeBinding,
    lookupFreeBinding_cons_of_beq_true, lookupFreeBinding_cons_of_beq_false,
    ora_loop_integer_types,
    parseDecimalNat?, parseDecimalNatAux, decimalDigit?, Option.pure_def,
    Option.bind_eq_bind, Option.some_bind, Option.none_bind, Option.map_some',
    Option.map_none', Option.some.injEq, Nat.zero_mul, Nat.mul_zero,
    Nat.zero_add, Nat.add_zero, Bool.and_true, Bool.true_and,
    beq_self_eq_true, if_pos, if_false, dif_pos, dif_neg, Bool.false_eq_true]
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
    ora_loop_integer_types,
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
  evalTactic (← `(tactic| set_option maxRecDepth 4096 in
    simp_all only [Env.lookupFree, ora_loop_integer_types, Option.some.injEq]))

-- Plain and dependency-bearing proofs consume the same typed input premise.
macro "ora_loop_prepare_context" hContext:ident : tactic =>
  `(tactic| (
    try simp only [Ora.Loop.contextReady, ora_loop_row,
      TyRef.integerShape?, ora_loop_integer_types,
      List.forall_mem_cons, List.forall_mem_nil] at $hContext:ident
    ora_loop_normalize_context_shapes $hContext:ident
    ora_loop_unpack_conjunctions
    ora_loop_unpack_integer_values
    ora_loop_apply_context
    subst_vars))

macro "ora_loop_normalize" loc:(Lean.Parser.Tactic.location)? : tactic =>
  `(tactic| simp only [Value.eqProp?, Ora.Integer.Value.eqProp?,
    Ora.Integer.Value.relation?, Ora.Integer.Value.binary?,
    Ora.Integer.Value.bitsFor?, Ora.Spec.IntegerShape.isSigned,
    Ora.Spec.IntegerShape.bits, Ora.Spec.uintBits, Ora.Spec.sintBits,
    ora_loop_integer_types,
    Option.pure_def, Option.bind_eq_bind, Option.some_bind, Option.none_bind,
    Option.map_some', Option.map_none', Option.some.injEq,
    if_pos, if_false, dif_pos, dif_neg, beq_iff_eq, beq_self_eq_true,
    Bool.and_true, Bool.true_and, Bool.and_false, Bool.false_and,
    Bool.false_eq_true, and_true, true_and, implies_true]
    $[$loc:location]?)

/-- Derive ordinary bounds for every natural bitwise conjunction encountered;
no mask, term identity or source shape is selected by this search. -/
elab "ora_loop_bitwise_bounds" : tactic => withMainContext do
  let mut expressions := #[(← getMainTarget)]
  for decl in ← getLCtx do
    unless decl.isImplementationDetail do expressions := expressions.push decl.type
  let seen ← IO.mkRef ({} : ExprSet)
  for expression in expressions do
    expression.forEach fun subexpression => do
      let_expr HAnd.hAnd α _ _ _ lhs rhs := subexpression | return
      unless α.isConstOf ``Nat do return
      if (← seen.get).contains subexpression then return
      seen.modify (·.insert subexpression)
      for lemma in [``Nat.and_le_left, ``Nat.and_le_right] do
        let proof ← mkAppOptM lemma #[some lhs, some rhs]
        let goal ← (← getMainGoal).assert (← mkFreshUserName `hBitwiseBound)
          (← inferType proof) proof
        let (_, goal) ← goal.intro1
        replaceMainGoal [goal]

theorem bitvec_sub_toNat_of_le (x y : BitVec n) (h : y.toNat ≤ x.toNat) :
    (x - y).toNat = x.toNat - y.toNat :=
  BitVec.toNat_sub_of_le h

macro "ora_loop_unsigned_bound" : tactic => `(tactic| (
  try simp only [BitVec.toNat_and, BitVec.toNat_ofNat, Nat.reducePow, Nat.reduceMod]
  omega))

macro "ora_loop_linear_close" : tactic =>
  `(tactic| (
    try simp_all only [Ora.Integer.unsignedLt, Ora.Integer.unsignedLe,
      Ora.Integer.signedLt, Ora.Integer.signedLe,
      Ora.Integer.mathematicalValue, Ora.Spec.IntegerShape.isSigned,
      Ora.Spec.IntegerShape.bits, Ora.Spec.uintBits, Ora.Spec.sintBits,
      ne_eq, BitVec.toNat_eq, BitVec.toNat_and, BitVec.ofNat_eq_ofNat, BitVec.toNat_ofNat,
      Nat.reducePow, Nat.reduceMod]
    ora_loop_bitwise_bounds
    try simp (disch := ora_loop_unsigned_bound) only [BitVec.toNat_add_of_lt, bitvec_sub_toNat_of_le,
      BitVec.toNat_and, BitVec.toNat_ofNat, Nat.reducePow, Nat.reduceMod] at *
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
         ne_eq, BitVec.toNat_eq, BitVec.toNat_and, BitVec.ofNat_eq_ofNat, BitVec.toNat_ofNat,
         Nat.reducePow, Nat.reduceMod]
       ora_loop_bitwise_bounds
       try simp (disch := ora_loop_unsigned_bound) only [BitVec.toNat_add_of_lt, bitvec_sub_toNat_of_le,
         BitVec.toNat_and, BitVec.toNat_ofNat, Nat.reducePow, Nat.reduceMod] at *
       first
       | apply bitvec_zero_toNat_le
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

macro "ora_loop_prove_support_header_facts" env:ident : tactic =>
  `(tactic| {
    try dsimp [$env:ident]
    unfold headerFactsHoldInEnv
    intro fact hFact
    simp only [ora_loop_row, List.mem_cons, List.not_mem_nil, or_false] at hFact
    repeat' first | subst fact | rcases hFact with hFact | hFact
    all_goals ora_loop_formula_context_reduce
    all_goals ora_loop_value_reduce
    all_goals simp only [Ora.Integer.unsignedLt, Ora.Integer.unsignedLe,
      Ora.Spec.IntegerShape.bits, Ora.Spec.uintBits,
      BitVec.toNat_ofNat, Nat.reducePow, Nat.reduceMod]
    all_goals first | decide | apply bitvec_zero_toNat_le })

macro "ora_loop_select_emitted_row" row:ident member:ident : tactic =>
  `(tactic| (
    simp only [ora_loop_row, List.mem_cons, List.not_mem_nil, or_false]
      at $member:ident
    repeat' first
      | subst $row:ident
      | rcases $member:ident with $member:ident | $member:ident))

macro "ora_loop_unpack_ready" ready:ident : tactic =>
  `(tactic| (
    ora_loop_unpack_state $ready:ident
    repeat' first | rcases $ready:ident with ⟨hShape, $ready:ident⟩
    subst_vars))

macro_rules
  | `(tactic| ora_loop_prefix_checks) => `(tactic| first
    | solve
      | intro locals hInvariant
        try ora_loop_direct_timeline_reduce
        simp only [Ora.Loop.timelineControlChecks, implies_true]
    | {
      intro locals hInvariant
      rcases hInvariant with ⟨hReady, hHeaderFacts, hInvariant⟩
      ora_loop_unpack_ready hReady
      ora_loop_direct_timeline_reduce
      simp only [Ora.Loop.timelineControlChecks]
      ora_loop_split_header_facts hHeaderFacts
      ora_loop_split_formulas at hInvariant
      ora_loop_formula_context_reduce at hHeaderFacts hInvariant ⊢
      ora_loop_value_reduce at hHeaderFacts hInvariant ⊢
      try ora_loop_apply_context
      ora_loop_normalize at hHeaderFacts hInvariant ⊢
      repeat' first | intro hPrefix | apply And.intro
      all_goals ora_loop_close })

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
    have hShape : ($autoSummary).shapeSupported = true := by
      first
      | solve | simp only [ora_loop_shape]
      | decide +kernel
    have hFormulas : ($autoSummary).formulasDenotable $emittedManifest = true := by
      set_option maxRecDepth 4096 in
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
        simp only [ora_loop_row, ora_loop_data,
          Ora.Loop.SummaryRow.fixedIndexBoundsSupported,
          Ora.Loop.timelineFixedIndexBoundsReason?,
          Ora.Loop.timelineFixedIndexBoundsReason?.visit,
          Ora.Loop.storageLocationFixedIndexBoundsReason?,
          Ora.Loop.fixedIndexProjectionBoundsReason?,
          Ora.Loop.priorSafetyFormulas,
          Ora.Loop.bestFixedIndexBoundsMatch?,
          Ora.Loop.fixedIndexBoundsFormulaMatch,
          Ora.Loop.fixedIndexExtentValid, List.all, Option.isNone]
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
        simp only [ora_loop_row, ora_loop_data,
          Ora.Loop.SummaryRow.fixedIndexBoundsSupported,
          Ora.Loop.timelineFixedIndexBoundsReason?,
          Ora.Loop.timelineFixedIndexBoundsReason?.visit,
          Ora.Loop.storageLocationFixedIndexBoundsReason?,
          Ora.Loop.fixedIndexProjectionBoundsReason?,
          Ora.Loop.priorSafetyFormulas,
          Ora.Loop.bestFixedIndexBoundsMatch?,
          Ora.Loop.fixedIndexBoundsFormulaMatch,
          Ora.Loop.fixedIndexExtentValid, List.all, Option.isNone]
        decide
    simp only [Ora.Loop.SummaryRow.supportedIn, ora_loop_manifest,
      hShape, hFixedIndexBounds, hFormulas, Bool.true_and] })

macro tk:"ora_loop_prove_base" : tactic => do
  let autoSummary := Lean.mkIdentFrom tk `autoSummary
  let emittedManifest := Lean.mkIdentFrom tk `emittedManifest
  `(tactic| {
    intro state hInitial
    rcases hInitial with ⟨hHeaderFacts, hInitial⟩
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
    refine ⟨hReady, ?_, ?_⟩
    · simpa only [headerFactsHoldInEnv, ora_loop_row,
        List.forall_mem_cons, List.not_mem_nil, false_implies,
        forall_const, and_true] using hHeaderFacts
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

macro tk:"ora_loop_prove_witness" base:term : tactic => do
  let autoSummary := Lean.mkIdentFrom tk `autoSummary
  let emittedManifest := Lean.mkIdentFrom tk `emittedManifest
  `(tactic| {
    let autoWitnessEnv : Env := $base
    have hInitialSome :
        (Ora.Loop.initializeState? $emittedManifest autoWitnessEnv
          ($autoSummary).variables ($autoSummary).init).isSome = true := by
      dsimp [autoWitnessEnv]
      try simp only [ora_loop_nested, SummaryRow.supportEnvIn,
        dependencyVariables, findSummaryRow?, Option.pure_def,
        Option.bind_eq_bind, Option.some_bind, Option.none_bind, if_pos, if_false]
      ora_loop_reduce
    cases hInitial : Ora.Loop.initializeState? $emittedManifest autoWitnessEnv
        ($autoSummary).variables ($autoSummary).init with
    | none =>
        simp only [hInitial, Option.isSome, Bool.false_eq_true] at hInitialSome
    | some initial =>
        refine ⟨autoWitnessEnv, initial, ?_⟩
        constructor
        · ora_loop_reduce
          all_goals decide
        constructor
        · dsimp [autoWitnessEnv]
          intro loopVar hLoopVar
          simp only [ora_loop_row, List.mem_cons, List.not_mem_nil, or_false] at hLoopVar
          repeat' first | subst loopVar | rcases hLoopVar with hLoopVar | hLoopVar
          all_goals try ora_loop_reduce
          all_goals try exact ⟨_, rfl, rfl⟩
        · first
            | ora_loop_nested_reduce
            | simp only [Ora.Loop.loopInitialPremises, Ora.Loop.denoteLoopSummary?,
                ora_loop_supported, if_true, Ora.Loop.denoteEdges?,
                Ora.Loop.denoteEdge?, Ora.Loop.denoteControlTarget?]
          refine ⟨?_, hInitial⟩
          ora_loop_prove_support_header_facts autoWitnessEnv })

/-- Only the candidate is reduced to data. The original existential goal must
be solved in full; failed default attempts restore every metavariable first. -/
elab "ora_loop_witness" base:term : tactic => withMainContext do
  let saved ← Tactic.saveState
  try
    withoutRecover <| evalTactic (← `(tactic| solve | ora_loop_prove_witness $base))
    return
  catch _ => saved.restore
  let manifest := mkIdent `emittedManifest
  let assumptions := mkIdent `autoAssumptions
  let row := mkIdent `autoSummary
  -- Emitted manifests are large lists. This allowance reduces candidate data
  -- only; it does not enlarge the universal proof search or assume a result.
  let candidate ← withOptions (fun options => options.setNat `maxRecDepth 4096) do
    let candidate ← Term.elabTerm
      (← `(loopWitnessCandidate? $manifest $assumptions $row $base)) none
    withTransparency .all (Meta.reduce candidate)
  unless candidate.isAppOfArity ``Option.some 2 do
    throwError "automatic_loop_witness_unavailable: no typed interval candidate"
  let value ← Term.exprToSyntax candidate.appArg!
  try
    withoutRecover <| evalTactic (← `(tactic| solve | ora_loop_prove_witness $value))
  catch ex =>
    throwError m!"automatic_loop_witness_rejected: all assumptions and the complete initial predicate are required\n{ex.toMessageData}"

macro tk:"ora_loop_auto_plain" : tactic => do
  let autoSummary := Lean.mkIdentFrom tk `autoSummary
  let emittedManifest := Lean.mkIdentFrom tk `emittedManifest
  `(tactic| {
    have hSupported : ($autoSummary).supported $emittedManifest = true := by
      simp only [ora_loop_supported]
    constructor
    · ora_loop_witness ($autoSummary).supportEnv
    · intro env hAssumptions hContext
      ora_loop_prepare_context hContext
      change Ora.Loop.loopInductionObligationsAtEnv
        $emittedManifest env $autoSummary
      unfold Ora.Loop.loopInductionObligationsAtEnv
        Ora.Loop.denoteLoopSummary?
      rw [hSupported]
      clear hSupported
      unfold $autoSummary
      simp only [ora_loop_row, EdgeRow.path_eq_timeline,
        EdgeRow.safety_eq_timeline, timelinePathView, timelineSafetyView,
        timelineSafetyViewFrom, Ora.Loop.denoteEdges?, Ora.Loop.denoteEdge?,
        Ora.Loop.denoteControlTarget?, Option.bind, if_pos]
      refine { base := ?_, backedge := ?_, edgeSafety := ?_, exit := ?_ }
      · ora_loop_prove_base
      · intro edge current following hMember hTarget hInvariant hPath hSafe hNext
        ora_loop_select_emitted_row edge hMember
        all_goals dsimp at hTarget
        all_goals try contradiction
        all_goals try ora_loop_false_path hPath
        all_goals dsimp at hInvariant hPath hSafe hNext ⊢
        all_goals (
          rcases hInvariant with ⟨hReady, hHeaderFacts, hInvariant⟩
          have hFactPremises := hHeaderFacts
          ora_loop_split_header_facts hFactPremises
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
          refine ⟨hFollowingReady, hHeaderFacts, ?_⟩
          clear hHeaderFacts
          ora_loop_split_formulas at hInvariant hPath ⊢
          all_goals try (ora_loop_formula_context_reduce at hFactPremises)
          all_goals ora_loop_formula_context_reduce at hInvariant hPath ⊢
          all_goals try (ora_loop_value_reduce at hFactPremises)
          all_goals ora_loop_value_reduce at hInvariant hPath ⊢
          all_goals try ora_loop_apply_context
          all_goals ora_loop_normalize at *
          all_goals repeat' apply And.intro
          all_goals ora_loop_unpack_conjunctions
          all_goals subst_vars
          all_goals ora_loop_close)
      · intro edge state hMember hInvariant hPath
        ora_loop_select_emitted_row edge hMember
        all_goals try solve | simp only [ora_loop_row, EdgeRow.safety_eq_timeline,
          timelineSafetyView, timelineSafetyViewFrom, Ora.Loop.safetyCheckpointsHold,
          List.not_mem_nil, false_implies, implies_true]
        all_goals try ora_loop_false_path hPath
        all_goals (
          dsimp at hInvariant hPath ⊢
          rcases hInvariant with ⟨hReady, hHeaderFacts, hInvariant⟩
          ora_loop_split_header_facts hHeaderFacts
          ora_loop_unpack_ready hReady
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
            all_goals try (ora_loop_formula_context_reduce at hHeaderFacts)
            all_goals ora_loop_formula_context_reduce at hInvariant hPath ⊢
            all_goals try (ora_loop_value_reduce at hHeaderFacts)
            all_goals ora_loop_value_reduce at hInvariant hPath ⊢
            all_goals try ora_loop_apply_context
            all_goals try (ora_loop_normalize at hHeaderFacts)
            -- Context reduction may already normalize a checkpoint completely.
            -- The following closure still has to prove the unchanged goal.
            all_goals try (ora_loop_normalize at hInvariant hPath ⊢)
            all_goals ora_loop_unpack_conjunctions
            all_goals subst_vars
            all_goals ora_loop_close)
      · intro edge current following hMember hTarget hInvariant hPath hSafe hNext
        ora_loop_select_emitted_row edge hMember
        all_goals dsimp at hTarget
        all_goals try contradiction
        -- Empty exit assertions need no transition reduction. Edge safety
        -- remains a separate obligation, including unreachable exit paths.
        all_goals try solve | ora_loop_split_formulas
        all_goals dsimp at hInvariant hPath hSafe hNext ⊢
        all_goals (
          rcases hInvariant with ⟨hReady, hHeaderFacts, hInvariant⟩
          ora_loop_split_header_facts hHeaderFacts
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
          all_goals try (ora_loop_formula_context_reduce at hHeaderFacts)
          all_goals ora_loop_formula_context_reduce at hInvariant hPath ⊢
          all_goals try (ora_loop_value_reduce at hHeaderFacts)
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
    hApply:ident following:ident hFollowingReady:ident hHeaderFacts:ident : tactic => do
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
    have hFactPremises := $hHeaderFacts:ident
    ora_loop_split_header_facts hFactPremises
    refine ⟨$hFollowingReady:term, ?_, ?_⟩
    · simpa only [headerFactsHoldInEnv, ora_loop_row,
        List.forall_mem_cons, List.not_mem_nil, false_implies,
        forall_const, and_true] using $hHeaderFacts:term
    clear $hHeaderFacts:ident
    ora_loop_split_formulas
    all_goals try (ora_loop_formula_context_reduce at hFactPremises)
    all_goals ora_loop_formula_context_reduce
    all_goals try (ora_loop_value_reduce at hFactPremises)
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
    · ora_loop_witness (($autoSummary).supportEnvIn $emittedLoopSummaries)
    · intro env hAssumptions hContext
      ora_loop_prepare_context hContext
      change Ora.Loop.loopInductionObligationsAtEnvIn
        $emittedManifest $emittedLoopSummaries env $autoSummary
      ora_loop_nested_reduce
      constructor
      · ora_loop_prove_base
      · intro edge current following hMember hTarget hInvariant hPath hSafe hNext
        ora_loop_select_emitted_row edge hMember
        all_goals dsimp at hTarget hInvariant hPath hSafe hNext ⊢
        all_goals try contradiction
        ora_loop_all_goals (
          rcases hInvariant with ⟨hReady, hHeaderFacts, hInvariant⟩
          have hFollowingReady :
              Ora.Loop.stateReady ($autoSummary).variables following := by
            apply Ora.Loop.bound_state_after_dependencies_ready
              $emittedManifest env ($autoSummary).variables current _ _ following
            exact hNext
          ora_loop_unpack_ready hReady
          rcases hNext with ⟨beforeDependencies, afterDependencies,
            hParentBound, hDependencies, hApply⟩
          ora_loop_formula_context_reduce at hParentBound
          ora_loop_value_reduce at hParentBound
          subst beforeDependencies
          ora_loop_split_formulas at hInvariant hPath
          all_goals try (ora_loop_formula_context_reduce at hHeaderFacts)
          all_goals (ora_loop_formula_context_reduce at hInvariant hPath)
          all_goals try (ora_loop_value_reduce at hHeaderFacts)
          all_goals (ora_loop_value_reduce at hInvariant hPath)
          all_goals try ora_loop_apply_context
          all_goals try (ora_loop_normalize at hHeaderFacts)
          -- Context simplification may already normalize these hypotheses.
          all_goals try (ora_loop_normalize at hInvariant hPath)
          all_goals ora_loop_unpack_conjunctions
          simp only [Ora.Loop.environmentRelationsProduce] at hDependencies
          ora_loop_reduce_environment_relations hDependencies
          all_goals
            ora_loop_finish_nested_backedges hApply following hFollowingReady hHeaderFacts)
      · intro edge state hMember hInvariant hPath
        ora_loop_select_emitted_row edge hMember
        all_goals (
          dsimp at hInvariant hPath ⊢
          rcases hInvariant with ⟨hReady, hHeaderFacts, hInvariant⟩
          ora_loop_split_header_facts hHeaderFacts
          ora_loop_unpack_ready hReady
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
          all_goals try (ora_loop_formula_context_reduce at hHeaderFacts)
          all_goals (ora_loop_formula_context_reduce at hInvariant hPath)
          all_goals try (ora_loop_value_reduce at hHeaderFacts)
          all_goals (ora_loop_value_reduce at hInvariant hPath)
          all_goals try ora_loop_apply_context
          all_goals try (ora_loop_normalize at hHeaderFacts)
          all_goals try (ora_loop_normalize at hInvariant hPath)
          all_goals ora_loop_unpack_conjunctions
          simp only [List.take, Ora.Loop.environmentRelationsProduce]
            at hDependencies
          ora_loop_reduce_environment_relations hDependencies
          all_goals ora_loop_finish_nested_safety)
      · intro edge current following hMember hTarget hInvariant hPath hSafe hNext
        ora_loop_select_emitted_row edge hMember
        all_goals dsimp at hTarget hInvariant hPath hSafe hNext ⊢
        all_goals try contradiction
        all_goals (
          rcases hInvariant with ⟨hReady, hHeaderFacts, hInvariant⟩
          ora_loop_split_header_facts hHeaderFacts
          ora_loop_unpack_ready hReady
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
          all_goals try (ora_loop_formula_context_reduce at hHeaderFacts)
          all_goals ora_loop_formula_context_reduce at hInvariant hPath ⊢
          all_goals try (ora_loop_value_reduce at hHeaderFacts)
          all_goals ora_loop_value_reduce at hInvariant hPath ⊢
          all_goals try ora_loop_apply_context
          all_goals ora_loop_normalize at hInvariant hPath ⊢
          all_goals ora_loop_unpack_conjunctions
          all_goals subst_vars
          all_goals ora_loop_close) })

macro tk:"ora_loop_auto_direct_storage_from" hScalar:ident hEdges:(ident)? : tactic => do
  let autoSummary := Lean.mkIdentFrom tk `autoSummary
  let emittedManifest := Lean.mkIdentFrom tk `emittedManifest
  let hCachedEdges := Lean.mkIdentFrom tk `hCachedEdges
  let prepareEdges ← match hEdges with
    | some proof => `(tactic| have $hCachedEdges := $proof env hAssumptions hContext)
    | none => `(tactic| have $hCachedEdges : True := True.intro)
  `(tactic| {
    change Ora.Loop.directStorageLoopProofAuthorizationFromAssumptions
      $emittedManifest _ $autoSummary
    have hDirect : Ora.Loop.directStorageSummarySupported
        $emittedManifest $autoSummary = true := by
      have hSupported : ($autoSummary).supported $emittedManifest = true := by
        first
        | solve | simp only [ora_loop_supported]
        | ora_loop_support
      unfold Ora.Loop.directStorageSummarySupported
      rw [hSupported]
      simp only [Bool.true_and, ora_loop_row, List.any, List.all]
      decide
    have hPreheaderEmpty : ($autoSummary).preheaderTimeline.isEmpty = true := by
      simp only [ora_loop_row, List.isEmpty]
    have hEdgeTimelines : ∀ edge, edge ∈ ($autoSummary).edges →
        Ora.Loop.directStorageTimeline? edge.timeline =
          some (Ora.Loop.directStorageEvents edge) := by
      intro edge hMember
      simp only [ora_loop_row, List.mem_cons, List.not_mem_nil, or_false]
        at hMember
      ora_loop_expand_membership edge hMember
      all_goals rfl
    have hEdgeTargets : ∀ edge, edge ∈ ($autoSummary).edges →
        Ora.Loop.denoteControlTarget? edge.target =
          some (Ora.Loop.directStorageTarget edge) := by
      intro edge hMember
      simp only [ora_loop_row, List.mem_cons, List.not_mem_nil, or_false]
        at hMember
      ora_loop_expand_membership edge hMember
      all_goals rfl
    constructor
    · rcases $hScalar.1 with
        ⟨env, locals, hAssumptions, hContext, hInitial⟩
      refine ⟨env, {
        locals := locals
        storage := Ora.Loop.directStoragePreheaderState env $autoSummary
      }, hAssumptions, hContext, ?_⟩
      rcases Ora.Loop.direct_storage_summary_denotes $emittedManifest env
          $autoSummary hDirect hPreheaderEmpty hEdgeTimelines hEdgeTargets with
          ⟨concreteVC, hConcrete⟩
      rw [hConcrete]
      unfold Ora.Loop.loopInitialPremises at hInitial
      cases hAbstract : Ora.Loop.denoteLoopSummary?
          $emittedManifest env $autoSummary with
      | none => simp [hAbstract] at hInitial
      | some abstractVC =>
          simp only [hAbstract] at hInitial
          have hScalarInitial := Ora.Loop.scalar_initial_values_of_denotation
            $emittedManifest env $autoSummary abstractVC locals hAbstract hInitial
          apply Ora.Loop.direct_initial_of_control_denotation
            $emittedManifest env $autoSummary concreteVC locals hConcrete
          · exact hPreheaderEmpty
          · ora_loop_direct_initial_control
          · exact hScalarInitial
    · intro env hAssumptions hContext
      have hScalarInduction := $hScalar.2 env hAssumptions hContext
      $prepareEdges
      ora_loop_prepare_context hContext
      unfold Ora.Loop.loopInductionObligationsAtEnv at hScalarInduction
      cases hAbstract : Ora.Loop.denoteLoopSummary?
          $emittedManifest env $autoSummary with
      | none => simp [hAbstract] at hScalarInduction
      | some abstractVC =>
          simp only [hAbstract] at hScalarInduction
          have hEdgeCompatible : ∀ edge, edge ∈ ($autoSummary).edges →
              Ora.Loop.DirectStorageEdgeCompatible $emittedManifest env
                ($autoSummary).variables ($autoSummary).post edge
                (fun locals => Ora.Loop.stateReady ($autoSummary).variables locals ∧
                  Ora.Loop.headerFactsHoldInEnv $emittedManifest env ($autoSummary).headerFacts ∧
                  Ora.Loop.formulasHoldInState $emittedManifest env
                    ($autoSummary).variables locals ($autoSummary).invariants) := by
            first
            | exact $hCachedEdges
            | intro edge hMember
              simp only [ora_loop_row, List.mem_cons, List.not_mem_nil, or_false]
                at hMember
              ora_loop_expand_membership edge hMember
              all_goals ora_loop_direct_edge_compatible $autoSummary env
          rcases Ora.Loop.direct_storage_summary_denotes $emittedManifest env
              $autoSummary hDirect hPreheaderEmpty hEdgeTimelines hEdgeTargets with
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
            ora_loop_all_goals (ora_loop_direct_formula_control_in $autoSummary env)
          · exact hEdgeCompatible })

macro tk:"ora_loop_auto_direct_storage" : tactic => do
  let autoSummary := Lean.mkIdentFrom tk `autoSummary
  let autoAssumptions := Lean.mkIdentFrom tk `autoAssumptions
  let emittedManifest := Lean.mkIdentFrom tk `emittedManifest
  let hScalar := Lean.mkIdentFrom tk `hScalar
  `(tactic| {
    have $hScalar : Ora.Loop.loopProofAuthorizationFromAssumptions
        $emittedManifest $autoAssumptions $autoSummary := by
      set_option maxRecDepth 4096 in ora_loop_auto_plain
    ora_loop_auto_direct_storage_from $hScalar
  })

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

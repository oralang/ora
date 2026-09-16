/- Candidate construction is data; these tests do not replace authorization. -/
import Ora.Loop.Auto

namespace Ora.Loop
open Ora.Obligation

private def witnessVariable (id : Nat) (ty : TyRef := .compilerTypeId 6) : VariableRow :=
  { index := id, id := { file_id := 0, pattern_id := id }, ty }

private def witnessVariableTerm (loopVar : VariableRow) : Term :=
  .variable (.free { id := loopVar.id, name := "input", ty := some loopVar.ty })

private def witnessRow (variables : List VariableRow) : SummaryRow :=
  { id := 0, owner := "generic", kind := .scfWhile, contextVariables := variables }

private def witnessAssumption (id : Nat) : AssumptionRow :=
  { id, owner := "generic", kind := .requires, formula := some (.term id) }

private def witnessComparison (op : BinaryOp) (value : Nat)
    (reverse : Bool := false) (ty : TyRef := .compilerTypeId 6) : Manifest := {
  terms := [witnessVariableTerm (witnessVariable 0 ty),
    .intLit { value := toString value, ty := some ty },
    .binary {
      op
      lhs := if reverse then 1 else 0
      rhs := if reverse then 0 else 1
      ty := some ty }]
}

private def witnessNumbers? (manifest : Manifest) (assumptions : List AssumptionRow)
    (variables : List VariableRow) : Option (List Nat) := do
  let row := witnessRow variables
  let env ← loopWitnessCandidate? manifest assumptions row row.supportEnv
  variables.mapM fun loopVar => do
    let .integer value ← env.lookupFree loopVar.id | none
    pure value.bits.toNat

example : witnessNumbers? (witnessComparison .ge 1) [witnessAssumption 2]
    [witnessVariable 0] = some [1] := by decide
example : witnessNumbers? (witnessComparison .gt 7) [witnessAssumption 2]
    [witnessVariable 0] = some [8] := by decide
example : witnessNumbers? (witnessComparison .eq 3) [witnessAssumption 2]
    [witnessVariable 0] = some [3] := by decide
example : witnessNumbers? (witnessComparison .le 1 true) [witnessAssumption 2]
    [witnessVariable 0] = some [1] := by decide
example : witnessNumbers? (witnessComparison .lt 1 true) [witnessAssumption 2]
    [witnessVariable 0] = some [2] := by decide
example : witnessNumbers? (witnessComparison .ge 1 false (.compilerTypeId 3))
    [witnessAssumption 2] [witnessVariable 0 (.compilerTypeId 3)] = some [1] := by decide
example : witnessNumbers? (witnessComparison .eq 255 false (.compilerTypeId 1))
    [witnessAssumption 2] [witnessVariable 0 (.compilerTypeId 1)] = some [255] := by decide
example : witnessNumbers? (witnessComparison .gt 255 false (.compilerTypeId 1))
    [witnessAssumption 2] [witnessVariable 0 (.compilerTypeId 1)] = none := by decide
example : witnessNumbers? (witnessComparison .eq 256 false (.compilerTypeId 1))
    [witnessAssumption 2] [witnessVariable 0 (.compilerTypeId 1)] = none := by decide
set_option maxRecDepth 1024 in
example : witnessNumbers? (witnessComparison .eq (2^256 - 1)) [witnessAssumption 2]
    [witnessVariable 0] = some [2^256 - 1] := by decide
set_option maxRecDepth 1024 in
example : witnessNumbers? (witnessComparison .gt (2^256 - 1)) [witnessAssumption 2]
    [witnessVariable 0] = none := by decide
example : witnessNumbers? (witnessComparison .lt 0) [witnessAssumption 2]
    [witnessVariable 0] = none := by decide
example : witnessNumbers? (witnessComparison .ge 1) [witnessAssumption 99]
    [witnessVariable 0] = none := by decide
example : witnessNumbers? (witnessComparison .ge 1)
    [{ id := 0, owner := "generic", kind := .requires }]
    [witnessVariable 0] = none := by decide
example : witnessNumbers? (witnessComparison .ge 1) [witnessAssumption 2]
    [witnessVariable 0 (.compilerTypeId 3)] = none := by decide

private def witnessConjunction : Manifest := {
  terms := [witnessVariableTerm (witnessVariable 0),
    .intLit { value := "1", ty := some (.compilerTypeId 6) },
    .binary { op := .ge, lhs := 0, rhs := 1, ty := some (.compilerTypeId 6) },
    .intLit { value := "4", ty := some (.compilerTypeId 6) },
    .binary { op := .le, lhs := 0, rhs := 3, ty := some (.compilerTypeId 6) },
    .binary { op := .and_, lhs := 2, rhs := 4 },
    .binary { op := .lt, lhs := 0, rhs := 1, ty := some (.compilerTypeId 6) },
    witnessVariableTerm (witnessVariable 1),
    .binary { op := .ge, lhs := 7, rhs := 3, ty := some (.compilerTypeId 6) },
    .intLit { value := "0", ty := some (.compilerTypeId 6) },
    .binary { op := .ne, lhs := 7, rhs := 9 }]
}

example : witnessNumbers? witnessConjunction [witnessAssumption 5]
    [witnessVariable 0] = some [1] := by decide
example : witnessNumbers? witnessConjunction [witnessAssumption 4, witnessAssumption 2]
    [witnessVariable 0] = witnessNumbers? witnessConjunction
      [witnessAssumption 2, witnessAssumption 4] [witnessVariable 0] := by decide
example : witnessNumbers? witnessConjunction [witnessAssumption 5, witnessAssumption 6]
    [witnessVariable 0] = none := by decide
example : witnessNumbers? witnessConjunction [witnessAssumption 5, witnessAssumption 8]
    [witnessVariable 0, witnessVariable 1] = some [1, 4] := by decide

-- Mixed environment: unbounded context values retain zero. The unhandled
-- disequality does not change the candidate, but still rejects that witness.
example : witnessNumbers? witnessConjunction [witnessAssumption 5]
    [witnessVariable 0, witnessVariable 1] = some [1, 0] := by decide
example : witnessNumbers? witnessConjunction [witnessAssumption 5, witnessAssumption 10]
    [witnessVariable 0, witnessVariable 1] = some [1, 0] := by decide

private def mixedWitness : Env := (witnessRow [witnessVariable 0, witnessVariable 1]).supportEnv
  |>.setFree (witnessVariable 0).id (.integer (Ora.Integer.Value.ofNat (.unsigned .w256) 1))
  |>.setFree (witnessVariable 1).id (.integer (Ora.Integer.Value.ofNat (.unsigned .w256) 0))

example : assumptionsDenoteInEnv witnessConjunction mixedWitness [witnessAssumption 5] := by
  change ((1 : Nat) ≤ 1 ∧ 1 ≤ 4) ∧ True
  decide
example : ¬ assumptionsDenoteInEnv witnessConjunction mixedWitness
    [witnessAssumption 5, witnessAssumption 10] := by
  change ¬ ((((1 : Nat) ≤ 1 ∧ 1 ≤ 4)) ∧ ((0#256 ≠ 0#256) ∧ True))
  decide

-- Candidate generation must not affect the environment used for support checks.
example : ((witnessRow [witnessVariable 0]).supportEnv.lookupFree (witnessVariable 0).id) =
    some (.integer (Ora.Integer.Value.ofNat (.unsigned .w256) 0)) := by decide

-- Candidate construction keeps an existing child binding, while changing the
-- bounded context value. Child bindings are not extra search variables.
example : ((loopWitnessCandidate? witnessConjunction [witnessAssumption 5]
    (witnessRow [witnessVariable 0])
    (Env.empty.setFree (witnessVariable 2).id
      (.integer (Ora.Integer.Value.ofNat (.unsigned .w256) 7)))).bind
        (·.lookupFree (witnessVariable 2).id)) =
    some (.integer (Ora.Integer.Value.ofNat (.unsigned .w256) 7)) := by decide

-- A well-typed candidate is not permission to invent a missing initializer.
example : initializeState? witnessConjunction mixedWitness
    [witnessVariable 2] [.term 99] = none := by decide
example : ¬ assumptionsDenoteInEnv witnessConjunction
    (witnessRow [witnessVariable 0]).supportEnv [witnessAssumption 5] := by
  change ¬ (((1 : Nat) ≤ 0 ∧ 0 ≤ 4) ∧ True)
  decide

namespace WitnessTactic

private def emittedManifest : Manifest := witnessComparison .ge 1
private def autoAssumptions : List AssumptionRow := [witnessAssumption 2]
@[local ora_loop_data] private theorem variableLookup :
    emittedManifest.terms[0]? = some (witnessVariableTerm (witnessVariable 0)) := rfl
@[local ora_loop_data] private theorem literalLookup :
    emittedManifest.terms[1]? = some (.intLit { value := "1", ty := some (.compilerTypeId 6) }) := rfl
@[local ora_loop_data] private theorem comparisonLookup :
    emittedManifest.terms[2]? = some
      (.binary { op := .ge, lhs := 0, rhs := 1, ty := some (.compilerTypeId 6) }) := rfl
private def stateVariable : VariableRow := { witnessVariable 2 with index := 0 }
private def witnessEdge (id : Nat) (target : ControlTargetRow) : EdgeRow := {
  id := id
  structuralKey := toString id
  target := target
  timeline := [{
    id := 0
    structuralKey := "path"
    payload := .pathAssumption { formulas := [{ formula := .term 2 }] }
  }]
  nextState := [{
    variableIndex := 0
    target := stateVariable.id
    sourceKey := "literal"
    value := .term 1
  }]
}
private def autoSummary : SummaryRow := {
  id := 0, owner := "generic", kind := .scfWhile
  contextVariables := [witnessVariable 0]
  variables := [stateVariable], init := [.term 1], invariants := [.term 2]
  edges := [witnessEdge 0 .backedge, witnessEdge 1 .successor]
  projectionComplete := true
}
attribute [local ora_loop_row] autoAssumptions autoSummary
  stateVariable witnessVariable witnessVariableTerm witnessEdge witnessAssumption

@[local ora_loop_supported] private theorem supported :
    autoSummary.supported emittedManifest = true := by decide

-- A setup helper must leave the proof goal to its caller, not create a block
-- that demands a completed proof after unpacking the input context.
example (env : Env) (hContext : contextReady env autoSummary.contextVariables) :
    ∃ n : Nat, n + n = 2 := by
  ora_loop_prepare_context hContext
  exact ⟨1, rfl⟩

-- The failed default must backtrack, not admit an alternative. The candidate
-- then proves the original existential, including assumptions and initial state.
example : loopSpecificationNonvacuous emittedManifest autoAssumptions autoSummary := by
  fail_if_success solve | ora_loop_prove_witness autoSummary.supportEnv
  ora_loop_witness autoSummary.supportEnv

namespace WithChild

private def childVariable : VariableRow := { witnessVariable 3 with index := 0 }
private def childSummary : SummaryRow := {
  WitnessTactic.autoSummary with
  id := 1
  variables := [childVariable]
  edges := (WitnessTactic.autoSummary.edges).map fun edge =>
    { edge with nextState := [{
        variableIndex := 0
        target := childVariable.id
        sourceKey := "literal"
        value := .term 1
      }] }
}
private def childEvent : TimelineEventRow := {
  id := 1
  structuralKey := "child"
  payload := .summaryDependency {
    id := 0
    structuralKey := "child"
    summaryId := 1
    target := .successor
    writeFootprintComplete := true
    -- Model fixture only; certificate digest binding is checked by CLI tests.
    loopVcSha256 := "0000000000000000000000000000000000000000000000000000000000000000"
  }
}
private def autoSummary : SummaryRow := {
  WitnessTactic.autoSummary with
  edges := [{ witnessEdge 0 .backedge with timeline :=
    (witnessEdge 0 .backedge).timeline ++ [childEvent] },
    witnessEdge 1 .successor]
}
private def emittedLoopSummaries : List SummaryRow := [autoSummary, childSummary]
attribute [local ora_loop_row] autoSummary childSummary childVariable childEvent
@[local ora_loop_nested] private theorem summaryCount :
    emittedLoopSummaries.length = 2 := rfl
@[local ora_loop_nested] private theorem childFind :
    findSummaryRow? emittedLoopSummaries 1 = some childSummary := rfl
@[local ora_loop_nested] private theorem supportedIn :
    autoSummary.supportedIn emittedManifest emittedLoopSummaries = true := by decide
@[local ora_loop_nested] private theorem childSupportedIn :
    childSummary.supportedIn emittedManifest emittedLoopSummaries = true := by decide
@[local ora_loop_nested] private theorem childDenotes (base : Env) :
    (denoteLoopSummaryWithFuel? 2 emittedManifest emittedLoopSummaries base childSummary).isSome = true := by
  unfold denoteLoopSummaryWithFuel?
  simp only [denoteLoopSummaryWithResolver?, childSupportedIn, if_true]
  simp only [ora_loop_row, denoteEdgesWithResolver?, denoteEdgeWithResolver?,
    EdgeRow.summaryDependencies, timelineDependencyView, denoteSummaryDependencies?,
    denoteControlTarget?, Option.bind_eq_bind, Option.some_bind, Option.pure_def,
    List.map, Option.isSome]

private def checkedNestedWitness : Env :=
  (autoSummary.supportEnvIn emittedLoopSummaries).setFree (witnessVariable 0).id
    (.integer (Ora.Integer.Value.ofNat (.unsigned .w256) 1))

example : loopWitnessCandidate? emittedManifest autoAssumptions autoSummary
    (autoSummary.supportEnvIn emittedLoopSummaries) = some checkedNestedWitness := rfl

-- Exercise the same automatic witness path as nested certificates, against
-- the complete predicate. The CLI's separate assumption policy is unchanged.
theorem nestedCandidateChecksInitial : loopSpecificationNonvacuousIn emittedManifest autoAssumptions
    emittedLoopSummaries autoSummary := by
  fail_if_success solve | ora_loop_prove_witness (autoSummary.supportEnvIn emittedLoopSummaries)
  ora_loop_witness (autoSummary.supportEnvIn emittedLoopSummaries)

-- The positive input must not overwrite the child's independently bound state.
example : checkedNestedWitness.lookupFree childVariable.id =
    some (.integer (Ora.Integer.Value.ofNat (.unsigned .w256) 0)) := by decide

end WithChild

end WitnessTactic

end Ora.Loop

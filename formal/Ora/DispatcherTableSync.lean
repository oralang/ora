/-
Trusted sync check for compiler-emitted dispatcher table facts.

`Ora/Generated/DispatcherTableSnapshot.lean` is data-only: each row is emitted
from actual Ora source compiled through `oraBuildSIRDispatcher`, then classified
with Sinora's real switch planner/index functions. This module decodes those
rows and checks that they satisfy the premises used by `Ora.Dispatcher`:

* known selectors resolve in the corresponding abstract dispatcher model;
* dense rows have injective route indices over known selectors;
* sparse rows resolve through bucket scan by exact selector equality;
* every switch has a named default destination (`guarded = true` in the
  raw snapshot schema);
* Lean independently regenerates the complete scored candidate search,
  including all 512 multiplicative constants per table size, and checks the
  emitted trace and chosen plan byte-for-byte as data.

This is representative fixture coverage for the repository snapshot and a
per-contract userland gate for every row emitted by `ora --lean-proofs`. It
proves the concrete contract's chosen plan equals an independent Lean execution
of the bounded planner model. It does not prove the Zig source implementation
equivalent for inputs that were never compiled through this gate.
-/


import Ora.Dispatcher.Gate

namespace Ora.DispatcherTableSync

open Ora.Dispatcher.PlannerArithmetic Ora.Generated Ora.Spec.DispatcherFacts

/- Zig emits these names into every per-contract checker. Keeping the complete
   public surface in this gate-collected block makes a Lean-side rename fail the
   repository build before it can become a runtime checker error. -/
#check @Gate.networkMatches
#check @Gate.rowsHaveNamedDefault
#check @Gate.rowsCovered
#check @Gate.denseRowsInjective
#check @Gate.rowsMatchModel
#check @Gate.planShapesValid
#check @Gate.plansAdmissible
#check @Gate.plannerMatches
#check @Gate.plannerReferenceMatches
#check @Gate.plannerSearchesValid
#check @Gate.plannerCoreMatches
#check @Gate.planIndicesMatch
#check @Gate.manifestRowsMatch
#check @Gate.manifestBaseRowsMatch
#check @Gate.rowStrategyWF
#check @Gate.builderCorrect
#check @Gate.plannerMatchesOfParts
#check @Gate.manifestRowsMatchOfParts

def compilerDispatcherRowsHaveNamedDefault : Bool :=
  Gate.rowsHaveNamedDefault compilerDispatcherTableRows

def compilerDispatcherRowsCovered : Bool :=
  Gate.rowsCovered compilerDispatcherTableRows

def compilerDenseRowsInjective : Bool :=
  Gate.denseRowsInjective compilerDispatcherTableRows

def compilerDispatcherPlanShapesValid : Bool :=
  Gate.planShapesValid compilerDispatcherTableRows

def compilerSparseRowsExactScan : Bool :=
  compilerDispatcherTableRows.all fun row =>
    if row.plan.strategy == "sparse" then
      sparseRunKnownOk row && allHasNamedDefault row
    else true

def compilerDispatcherTableRowsMatch : Bool :=
  Gate.rowsMatchModel compilerDispatcherTableRows

def compilerDispatcherPlansAdmissible : Bool :=
  Gate.plansAdmissible compilerDispatcherTableRows

def compilerDispatcherPlannerMatches : Bool :=
  Gate.plannerMatches compilerDispatcherTableRows

def compilerDispatcherPlanIndicesMatch : Bool :=
  Gate.planIndicesMatch compilerDispatcherTableRows

def compilerDispatcherManifestRowsMatch : Bool :=
  Gate.manifestRowsMatch compilerDispatcherTableRows

def compilerDispatcherPlannerSearchesValid : Bool :=
  Gate.plannerSearchesValid compilerDispatcherTableRows

def compilerDispatcherPlannerCoreMatches : Bool :=
  Gate.plannerCoreMatches compilerDispatcherTableRows

def compilerDispatcherManifestBaseRowsMatch : Bool :=
  Gate.manifestBaseRowsMatch compilerDispatcherTableRows

private def compilerDispatcherPlannerCoreChecksAt (index : Nat) : PlannerCoreChecks :=
  match compilerDispatcherTableRows[index]? with
  | some row => plannerCoreChecks row
  | none =>
      { policy := false, preconditions := false, linearScore := false
        denseCandidateCount := false, bestDense := false
        sparseCandidateCount := false, bestSparse := false, plan := false }

private def compilerDispatcherPlannerCoreMatchesAt (index : Nat) : Bool :=
  (compilerDispatcherPlannerCoreChecksAt index).all

private theorem compiler_dispatcher_planner_core_row_0 :
    compilerDispatcherPlannerCoreMatchesAt 0 = true := by decide

private theorem compiler_dispatcher_planner_core_row_1 :
    compilerDispatcherPlannerCoreMatchesAt 1 = true := by decide

-- Rows 2 and 3 contain the complete multiplicative-search traces; their
-- concrete `decide` reductions need data-proportional recursion depth.
set_option maxRecDepth 1000000 in
private theorem compiler_dispatcher_planner_core_row_2 :
    compilerDispatcherPlannerCoreMatchesAt 2 = true := by decide

set_option maxRecDepth 1000000 in
private theorem compiler_dispatcher_planner_core_row_3_policy :
    (compilerDispatcherPlannerCoreChecksAt 3).policy = true := by decide

set_option maxRecDepth 1000000 in
private theorem compiler_dispatcher_planner_core_row_3_preconditions :
    (compilerDispatcherPlannerCoreChecksAt 3).preconditions = true := by decide

set_option maxRecDepth 1000000 in
private theorem compiler_dispatcher_planner_core_row_3_linear_score :
    (compilerDispatcherPlannerCoreChecksAt 3).linearScore = true := by decide

set_option maxRecDepth 1000000 in
private theorem compiler_dispatcher_planner_core_row_3_dense_count :
    (compilerDispatcherPlannerCoreChecksAt 3).denseCandidateCount = true := by decide

set_option maxRecDepth 1000000 in
private theorem compiler_dispatcher_planner_core_row_3_best_dense :
    (compilerDispatcherPlannerCoreChecksAt 3).bestDense = true := by decide

set_option maxRecDepth 1000000 in
private theorem compiler_dispatcher_planner_core_row_3_sparse_count :
    (compilerDispatcherPlannerCoreChecksAt 3).sparseCandidateCount = true := by decide

set_option maxRecDepth 1000000 in
private theorem compiler_dispatcher_planner_core_row_3_best_sparse :
    (compilerDispatcherPlannerCoreChecksAt 3).bestSparse = true := by decide

set_option maxRecDepth 1000000 in
private theorem compiler_dispatcher_planner_core_row_3_plan :
    (compilerDispatcherPlannerCoreChecksAt 3).plan = true := by decide

private theorem compiler_dispatcher_planner_core_row_3 :
    compilerDispatcherPlannerCoreMatchesAt 3 = true := by
  apply PlannerCoreChecks.all_of_fields
  · exact compiler_dispatcher_planner_core_row_3_policy
  · exact compiler_dispatcher_planner_core_row_3_preconditions
  · exact compiler_dispatcher_planner_core_row_3_linear_score
  · exact compiler_dispatcher_planner_core_row_3_dense_count
  · exact compiler_dispatcher_planner_core_row_3_best_dense
  · exact compiler_dispatcher_planner_core_row_3_sparse_count
  · exact compiler_dispatcher_planner_core_row_3_best_sparse
  · exact compiler_dispatcher_planner_core_row_3_plan

set_option maxRecDepth 1000000 in
private theorem compiler_dispatcher_planner_core_rows_match :
    Gate.plannerCoreMatches compilerDispatcherTableRows = true := by
  have hlen : compilerDispatcherTableRows.length = 4 := by decide
  generalize hrows : compilerDispatcherTableRows = rows at hlen ⊢
  cases rows with
  | nil => simp at hlen
  | cons first rest =>
      cases rest with
      | nil => simp at hlen
      | cons second rest =>
          cases rest with
          | nil => simp at hlen
          | cons third rest =>
              cases rest with
              | nil => simp at hlen
              | cons fourth rest =>
                  cases rest with
                  | cons fifth rest => simp at hlen
                  | nil =>
                      have hfirst : rowPlannerCoreMatches first = true := by
                        simpa [compilerDispatcherPlannerCoreMatchesAt,
                          compilerDispatcherPlannerCoreChecksAt, rowPlannerCoreMatches, hrows] using
                          compiler_dispatcher_planner_core_row_0
                      have hsecond : rowPlannerCoreMatches second = true := by
                        simpa [compilerDispatcherPlannerCoreMatchesAt,
                          compilerDispatcherPlannerCoreChecksAt, rowPlannerCoreMatches, hrows] using
                          compiler_dispatcher_planner_core_row_1
                      have hthird : rowPlannerCoreMatches third = true := by
                        simpa [compilerDispatcherPlannerCoreMatchesAt,
                          compilerDispatcherPlannerCoreChecksAt, rowPlannerCoreMatches, hrows] using
                          compiler_dispatcher_planner_core_row_2
                      have hfourth : rowPlannerCoreMatches fourth = true := by
                        simpa [compilerDispatcherPlannerCoreMatchesAt,
                          compilerDispatcherPlannerCoreChecksAt, rowPlannerCoreMatches, hrows] using
                          compiler_dispatcher_planner_core_row_3
                      simp [Gate.plannerCoreMatches, hfirst, hsecond, hthird, hfourth]

-- `decide` reduces the generated 512-entry candidate vector; this is a data-
-- depth allowance, not additional elaboration search.
set_option maxRecDepth 1000000 in
theorem dispatcher_multiplicative_candidates_match :
    compilerDispatcherMultiplicativeCandidates =
      (List.range expectedMultiplicativeSearchBudget).map
        multiplicativeCandidate := by decide

theorem multiplicative_candidate_discriminators :
    multiplicativeCandidate 0 = 2462723855 ∧
      multiplicativeCandidate 1 = 2527132011 ∧
      multiplicativeCandidate 2 = 3024231355 := by decide

theorem dispatcher_policy_lambda_discriminators :
    policyLambda? "gas" = some 0 ∧
      policyLambda? "balanced" = some 5 ∧
      policyLambda? "size" = some 50 ∧
      policyLambda? "unknown" = none := by decide

theorem multiplicative_search_rejects_false_collision_witness :
    multiplicativeSearchValid
      [1447852734, 832491607, 3309386683, 2561671559]
      { tableSlots := 4, selected := some 1,
        rejected := [{ constant := 2462723855, firstCase := 0, secondCase := 1 }] } =
      false := by decide

theorem dispatcher_table_rows_have_named_default :
    compilerDispatcherRowsHaveNamedDefault = true := by decide

theorem dispatcher_table_rows_covered :
    compilerDispatcherRowsCovered = true := by decide

theorem dense_dispatcher_rows_injective :
    compilerDenseRowsInjective = true := by decide

theorem dispatcher_plan_shapes_valid :
    compilerDispatcherPlanShapesValid = true := by decide

theorem sparse_dispatcher_rows_exact_scan :
    compilerSparseRowsExactScan = true := by decide

theorem dispatcher_table_rows_match_model :
    compilerDispatcherTableRowsMatch = true := by decide

theorem dispatcher_plans_admissible :
    compilerDispatcherPlansAdmissible = true := by decide

-- Planner equality evaluates the complete generated search traces, including
-- all multiplicative candidates; only recursive reduction depth is raised.
set_option maxRecDepth 1000000 in
theorem dispatcher_planner_matches :
    compilerDispatcherPlannerMatches = true := by
  apply Gate.plannerMatchesOfParts
  · decide
  · exact compiler_dispatcher_planner_core_rows_match

theorem dispatcher_plan_indices_match :
    compilerDispatcherPlanIndicesMatch = true := by decide

-- Manifest equality reuses the data-scale planner theorem above and evaluates
-- the remaining generated rows without enlarging the heartbeat budget.
set_option maxRecDepth 1000000 in
theorem dispatcher_manifest_rows_match :
    compilerDispatcherManifestRowsMatch = true := by
  apply Gate.manifestRowsMatchOfParts
  · decide
  · exact dispatcher_planner_matches

end Ora.DispatcherTableSync

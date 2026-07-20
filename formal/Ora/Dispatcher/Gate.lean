/-
Stable compiler-kernel gate surface for dispatcher rows and networks.

The Zig-emitted per-contract checker consumes the names in namespace Gate.
Keep this module focused on composition and proof authority; decoding and
planner recomputation live in sibling modules.
-/

import Ora.Dispatcher.Network
import Ora.Dispatcher.PlannerCore

namespace Ora.DispatcherTableSync

def rowStrategyWF (row : RawRow) : Prop :=
  match rowBuilderPlan? row with
  | some plan => Ora.Dispatcher.StrategyWF (rowCasePairs row) plan
  | none => False

def rowManifestBaseOk (row : RawRow) : Bool :=
  strategyKnown row &&
    selectorsDistinct row &&
    routeIndicesInRange row &&
    rowPlanShapeValid row &&
    allHasNamedDefault row &&
    denseInjectiveIfDense row &&
    rowPlanIndicesMatch row

def rowManifestOk (row : RawRow) : Bool :=
  rowManifestBaseOk row && rowPlannerMatches row

theorem row_builder_correct_of_plan_admissible (row : RawRow)
    (h : rowPlanAdmissible row = true) :
    rowStrategyWF row :=
  by
    unfold rowPlanAdmissible at h
    unfold rowStrategyWF
    cases hplan : rowBuilderPlan? row with
    | none =>
        simp [hplan] at h
    | some plan =>
        simp [hplan] at h ⊢
        exact Ora.Dispatcher.builder_correct (rowCasePairs row) plan h

theorem row_builder_correct_of_reference_match (row : RawRow)
    (h : rowPlannerReferenceMatches row = true) :
    rowStrategyWF row := by
  unfold rowPlannerReferenceMatches at h
  cases hpolicy : plannerPolicy? row.trace.policy with
  | none => simp [hpolicy] at h
  | some policy =>
      simp only [hpolicy] at h
      let input : Ora.DispatcherPlannerSpec.Input String :=
        { cases := rowCasePairs row,
          hasDefault := allHasNamedDefault row,
          policy := policy }
      have hplan :
          encodePlannerPlan input.cases.length
              (Ora.SinoraPlanner.choosePlanReference input) = row.plan := by
        simpa [input] using h
      have hwf := Ora.SinoraPlanner.planner_reference_builder_correct input
      have hshape := Ora.SinoraPlanner.choosePlanReference_wellShaped input
      have hlen : (rowCasePairs row).length = row.cases.length := by
        simp [rowCasePairs]
      dsimp [input] at hplan hwf hshape
      unfold rowStrategyWF rowBuilderPlan?
      rw [← hplan]
      rw [← hlen]
      rw [planBuilder_encodePlannerPlan _ _ hshape]
      exact hwf

theorem rows_builder_correct_of_plan_admissible (rows : List RawRow)
    (h : rows.all rowPlanAdmissible = true) :
    ∀ row, row ∈ rows → rowStrategyWF row := by
  intro row hmem
  exact row_builder_correct_of_plan_admissible row ((List.all_eq_true).mp h row hmem)

theorem rows_builder_correct_of_reference_match (rows : List RawRow)
    (h : rows.all rowPlannerReferenceMatches = true) :
    ∀ row, row ∈ rows → rowStrategyWF row := by
  intro row hmem
  exact row_builder_correct_of_reference_match row ((List.all_eq_true).mp h row hmem)

/- Stable surface consumed by the Zig-emitted per-contract Lean checker.
    Keep the runtime gate on these names so `lake build` catches helper renames
    here instead of letting string-generated Lean drift until a CLI run. -/
namespace Gate

def networkMatches
    (intents : List RawIntent)
    (switches : List RawNetworkSwitch)
    (entry : String) : Bool :=
  dispatcherNetworkMatches intents switches entry

def rowsHaveNamedDefault (rows : List RawRow) : Bool :=
  rows.all allHasNamedDefault

def rowsCovered (rows : List RawRow) : Bool :=
  rows.all modelRunKnownOk

def denseRowsInjective (rows : List RawRow) : Bool :=
  rows.all denseInjectiveIfDense

def rowsMatchModel (rows : List RawRow) : Bool :=
  rows.all rowMatchesDispatcherModel

def planShapesValid (rows : List RawRow) : Bool :=
  rows.all rowPlanShapeValid

def plansAdmissible (rows : List RawRow) : Bool :=
  rows.all fun row =>
    rawPlanAdmissible (rowCasePairs row) row.cases.length row.plan

def plannerMatches (rows : List RawRow) : Bool :=
  rows.all rowPlannerMatches

def plannerReferenceMatches (rows : List RawRow) : Bool :=
  rows.all rowPlannerReferenceMatches

def plannerSearchesValid (rows : List RawRow) : Bool :=
  rows.all rowMultiplicativeSearchesValid

def plannerCoreMatches (rows : List RawRow) : Bool :=
  rows.all rowPlannerCoreMatches

def planIndicesMatch (rows : List RawRow) : Bool :=
  rows.all rowPlanIndicesMatch

def manifestRowsMatch (rows : List RawRow) : Bool :=
  rows.all rowManifestOk

def manifestBaseRowsMatch (rows : List RawRow) : Bool :=
  rows.all rowManifestBaseOk

def rowStrategyWF := Ora.DispatcherTableSync.rowStrategyWF

theorem builderCorrect (rows : List RawRow)
    (h : plannerReferenceMatches rows = true) :
    ∀ row, row ∈ rows → rowStrategyWF row :=
  rows_builder_correct_of_reference_match rows h

theorem plannerMatchesOfParts (rows : List RawRow)
    (hsearches : plannerSearchesValid rows = true)
    (hcore : plannerCoreMatches rows = true) :
    plannerMatches rows = true := by
  induction rows with
  | nil => rfl
  | cons row rest ih =>
      simp only [plannerSearchesValid, plannerCoreMatches, plannerMatches,
        rowPlannerMatches, List.all_cons, Bool.and_eq_true] at hsearches hcore ⊢
      exact ⟨⟨hsearches.1, hcore.1⟩, ih hsearches.2 hcore.2⟩

theorem manifestRowsMatchOfParts (rows : List RawRow)
    (hbase : manifestBaseRowsMatch rows = true)
    (hplanner : plannerMatches rows = true) :
    manifestRowsMatch rows = true := by
  induction rows with
  | nil => rfl
  | cons row rest ih =>
      simp only [manifestBaseRowsMatch, plannerMatches, manifestRowsMatch,
        rowManifestOk, List.all_cons, Bool.and_eq_true] at hbase hplanner ⊢
      exact ⟨⟨hbase.1, hplanner.1⟩, ih hbase.2 hplanner.2⟩

end Gate

end Ora.DispatcherTableSync

/-
Executable Lean reference for the Sinora dispatcher planner.

`choosePlanReference` executes the exact finite search and branch order stated
by `Ora.DispatcherPlannerSpec`.  The universal theorem proves the Lean planner
model always returns a plan accepted by the abstract dispatcher builder.  The
separate per-contract checker remains responsible for establishing that the
production Zig planner emitted this reference plan.
-/

import Ora.DispatcherPlannerSpec

namespace Ora.SinoraPlanner

open Ora.DispatcherPlannerSpec

def chooseCertifiedPlan (input : Input Label) : CertifiedPlan input.cases :=
  if _hpre : preconditionsMet input then
    match _hdense : acceptedDense? input with
    | some candidate =>
        { plan := .dense candidate.plan,
          admissible := candidate.admissible,
          wellShaped := candidate.wellShaped }
    | none =>
        match _hsparse : acceptedSparse? input with
        | some candidate =>
            { plan := .sparse candidate.plan,
              admissible := rfl,
              wellShaped := candidate.wellShaped }
        | none => { plan := .linear, admissible := rfl, wellShaped := rfl }
  else
    { plan := .linear, admissible := rfl, wellShaped := rfl }

def choosePlanReference (input : Input Label) : Plan :=
  (chooseCertifiedPlan input).plan

theorem choosePlanReference_admissible (input : Input Label) :
    Plan.admissible input.cases (choosePlanReference input) = true :=
  (chooseCertifiedPlan input).admissible

theorem choosePlanReference_wellShaped (input : Input Label) :
    Plan.wellShaped (choosePlanReference input) = true :=
  (chooseCertifiedPlan input).wellShaped

theorem choosePlanReference_spec (input : Input Label) :
    PlannerChooses input (choosePlanReference input) := by
  unfold choosePlanReference chooseCertifiedPlan
  by_cases hpre : preconditionsMet input = true
  · simp only [hpre, if_pos]
    cases hdense : acceptedDense? input with
    | some candidate =>
        exact PlannerChooses.dense hpre hdense
    | none =>
        cases hsparse : acceptedSparse? input with
        | some candidate =>
            exact PlannerChooses.sparse hpre hdense hsparse
        | none =>
            exact PlannerChooses.linear hpre hdense hsparse
  · have hfalse : preconditionsMet input = false := by
      cases h : preconditionsMet input <;> simp_all
    simp only [hfalse, if_false]
    exact PlannerChooses.preconditionFailure hfalse

theorem planner_spec_deterministic {input : Input Label} {first second : Plan}
    (hfirst : PlannerChooses input first) (hsecond : PlannerChooses input second) :
    first = second := by
  cases hfirst <;> cases hsecond <;> simp_all

/-- Universal planner+builder correctness for the Lean specification.

This theorem has no per-contract admissibility premise. It does not claim the
Zig source is verified; production correspondence is checked for each compiled
contract by `Ora.DispatcherTableSync`.
-/
theorem planner_reference_builder_correct [Inhabited Label] (input : Input Label) :
    Ora.Dispatcher.StrategyWF input.cases (choosePlanReference input).toBuilderPlan :=
  Ora.Dispatcher.builder_correct input.cases _ (choosePlanReference_admissible input)

theorem four_sequential_balanced_keeps_linear :
    choosePlanReference
      { cases := [(0, "a"), (1, "b"), (2, "c"), (3, "d")],
        hasDefault := true,
        policy := .balanced } =
      .linear := by decide

theorem missing_default_selects_linear :
    choosePlanReference
      { cases := [(0, "a"), (1, "b"), (2, "c"), (3, "d")],
        hasDefault := false,
        policy := .balanced } = .linear := by decide

theorem non_u32_selector_selects_linear :
    choosePlanReference
      { cases := [(0, "a"), (1, "b"), (2, "c"), (2^32, "d")],
        hasDefault := true,
        policy := .balanced } = .linear := by decide

end Ora.SinoraPlanner

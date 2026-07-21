/-
Generic partial-correctness model for scalar loops.

The model deliberately says nothing about termination. `Reaches` describes a
finite number of loop steps, and the main theorem only characterizes a state
that is known to be an exit reached from an initial state.
-/

import Ora.Integer.Arithmetic

namespace Ora.Loop

abbrev ScalarLoopState := List Ora.Integer.Value

structure ScalarLoopSummary where
  initial : ScalarLoopState → Prop
  guard : ScalarLoopState → Prop
  invariant : ScalarLoopState → Prop
  step : ScalarLoopState → ScalarLoopState → Prop
  safe : ScalarLoopState → Prop
  post : ScalarLoopState → Prop

inductive Reaches (summary : ScalarLoopSummary) :
    ScalarLoopState → ScalarLoopState → Prop where
  | refl (state) : Reaches summary state state
  | next {initial current following} :
      Reaches summary initial current →
      summary.guard current →
      summary.step current following →
      Reaches summary initial following

def ScalarLoopSummary.BaseObligation (summary : ScalarLoopSummary) : Prop :=
  ∀ state, summary.initial state → summary.invariant state

def ScalarLoopSummary.StepObligation (summary : ScalarLoopSummary) : Prop :=
  ∀ current following,
    summary.invariant current →
    summary.guard current →
    summary.step current following →
    summary.invariant following

def ScalarLoopSummary.ExitObligation (summary : ScalarLoopSummary) : Prop :=
  ∀ state, summary.invariant state → ¬summary.guard state → summary.post state

def ScalarLoopSummary.SafetyObligation (summary : ScalarLoopSummary) : Prop :=
  ∀ state, summary.invariant state → summary.guard state → summary.safe state

def ScalarLoopSummary.PartialCorrect (summary : ScalarLoopSummary) : Prop :=
  ∀ initial exit,
    summary.initial initial →
    Reaches summary initial exit →
    ¬summary.guard exit →
    summary.post exit

def ScalarLoopSummary.SafeOnReachableStates (summary : ScalarLoopSummary) : Prop :=
  ∀ initial current,
    summary.initial initial →
    Reaches summary initial current →
    summary.guard current →
    summary.safe current

def ScalarLoopSummary.Verified (summary : ScalarLoopSummary) : Prop :=
  summary.PartialCorrect ∧ summary.SafeOnReachableStates

structure ScalarLoopSummary.InductionObligations
    (summary : ScalarLoopSummary) : Prop where
  base : summary.BaseObligation
  step : summary.StepObligation
  safety : summary.SafetyObligation
  exit : summary.ExitObligation

theorem invariant_holds_on_reached_state
    (summary : ScalarLoopSummary)
    (hBase : summary.BaseObligation)
    (hStep : summary.StepObligation)
    {initial current : ScalarLoopState}
    (hInitial : summary.initial initial)
    (hReaches : Reaches summary initial current) :
    summary.invariant current := by
  induction hReaches with
  | refl => exact hBase initial hInitial
  | next hPrefix hGuard hBody ih =>
      exact hStep _ _ ih hGuard hBody

theorem invariant_preserved_to_exit
    (summary : ScalarLoopSummary)
    (hBase : summary.BaseObligation)
    (hStep : summary.StepObligation)
    (hExit : summary.ExitObligation) :
    summary.PartialCorrect := by
  intro initial exit hInitial hReaches hNotGuard
  exact hExit exit
    (invariant_holds_on_reached_state summary hBase hStep hInitial hReaches)
    hNotGuard

theorem safety_holds_on_reached_state
    (summary : ScalarLoopSummary)
    (hBase : summary.BaseObligation)
    (hStep : summary.StepObligation)
    (hSafety : summary.SafetyObligation) :
    summary.SafeOnReachableStates := by
  intro initial current hInitial hReaches hGuard
  exact hSafety current
    (invariant_holds_on_reached_state summary hBase hStep hInitial hReaches)
    hGuard

theorem verified_of_induction
    (summary : ScalarLoopSummary)
    (proof : summary.InductionObligations) :
    summary.Verified := by
  exact ⟨
    invariant_preserved_to_exit summary proof.base proof.step proof.exit,
    safety_holds_on_reached_state summary proof.base proof.step proof.safety
  ⟩

/-! A tiny one-step scalar-loop instantiation. -/

private def zero : Ora.Integer.Value := Ora.Integer.Value.ofNat (.unsigned .w8) 0
private def one : Ora.Integer.Value := Ora.Integer.Value.ofNat (.unsigned .w8) 1

private def tinyLoop : ScalarLoopSummary where
  initial := fun state => state = [zero]
  guard := fun state => state = [zero]
  invariant := fun state => state = [zero] ∨ state = [one]
  step := fun current following => current = [zero] ∧ following = [one]
  safe := fun _ => True
  post := fun state => state = [one]

private theorem tinyLoop_base : tinyLoop.BaseObligation := by
  intro state hInitial
  exact Or.inl hInitial

private theorem tinyLoop_step : tinyLoop.StepObligation := by
  intro current following _ _ hBody
  exact Or.inr hBody.2

private theorem tinyLoop_exit : tinyLoop.ExitObligation := by
  intro state hInvariant hNotGuard
  cases hInvariant with
  | inl hZero => exact False.elim (hNotGuard hZero)
  | inr hOne => exact hOne

private theorem tinyLoop_safety : tinyLoop.SafetyObligation := by
  simp [ScalarLoopSummary.SafetyObligation, tinyLoop]

private def tinyLoop_induction : tinyLoop.InductionObligations where
  base := tinyLoop_base
  step := tinyLoop_step
  safety := tinyLoop_safety
  exit := tinyLoop_exit

example : tinyLoop.Verified :=
  verified_of_induction tinyLoop tinyLoop_induction

end Ora.Loop

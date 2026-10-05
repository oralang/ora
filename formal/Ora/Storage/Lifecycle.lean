import Ora.Loop.Storage

/- Conditional storage lifecycle model. It consumes no solver verdicts.
   Compiler extraction, writer coverage, representation correspondence and
   digest custody must be established separately before any authority grant. -/
namespace Ora.StorageLifecycle
open Ora.Loop Ora.Obligation

structure Claim where
  location : StorageLocation
  name : String
  bounds : List Value

def Claim.holds (claim : Claim) (value : Value) : Prop :=
  match value.refinementPredicate? claim.name claim.bounds with
  | some predicate => predicate
  | none => False

def Invariant (claims : List Claim) (store : StorageState) : Prop :=
  ∀ claim, claim ∈ claims → claim.holds (store claim.location)

/-- An alias is allowed only if the new value satisfies every claim at it. -/
def WriteChecked (claims : List Claim) (location : StorageLocation)
    (value : Value) : Prop :=
  ∀ claim, claim ∈ claims → claim.location = location → claim.holds value

theorem checked_write_preserves
    (before : StorageState) (claims : List Claim)
    (location : StorageLocation) (value : Value)
    (hBefore : Invariant claims before)
    (hWrite : WriteChecked claims location value) :
    Invariant claims (before.set location value) := by
  intro claim hMember
  by_cases hAlias : claim.location = location
  · simpa [StorageState.set, hAlias] using hWrite claim hMember hAlias
  · simpa [StorageState.set, hAlias] using hBefore claim hMember

/-- A frame fact must name all observed locations. Unknown interference does
    not acquire this hypothesis merely because an external call returns. -/
theorem frame_preserves (claims : List Claim) (before after : StorageState)
    (hBefore : Invariant claims before)
    (hFrame : ∀ claim, claim ∈ claims → after claim.location = before claim.location) :
    Invariant claims after := by
  intro claim hMember
  rw [hFrame claim hMember]
  exact hBefore claim hMember

def ActionChecked (claims : List Claim)
    (semantics : StorageActionSemantics LocationExpr ValueExpr)
    (action : StorageAction LocationExpr ValueExpr) (state : StorageActionState) : Prop :=
  match action with
  | .read _ _ => True
  | .write locationExpr valueExpr =>
      ∀ location value,
        semantics.evalLocation locationExpr state = some location →
        semantics.evalValue valueExpr state = some value →
        WriteChecked claims location value

theorem action_preserves
    (claims : List Claim) (semantics : StorageActionSemantics LocationExpr ValueExpr)
    (action : StorageAction LocationExpr ValueExpr)
    (before : StorageActionState) (after : StorageActionResult)
    (hBefore : Invariant claims before.loop.storage)
    (hChecked : ActionChecked claims semantics action before)
    (hExec : execStorageAction? semantics action before = some after) :
    Invariant claims after.state.loop.storage := by
  cases action with
  | read id place =>
      cases hPlace : semantics.evalLocation place before with
      | none => simp [execStorageAction?, hPlace] at hExec
      | some location =>
          cases hMatch : semantics.valueMatchesLocation place (before.loop.storage location) with
          | false => simp [execStorageAction?, hPlace, hMatch] at hExec
          | true =>
              cases hBind : before.bindRead? id (before.loop.storage location) with
              | none => simp [execStorageAction?, hPlace, hMatch, hBind] at hExec
              | some following =>
                  simp [execStorageAction?, hPlace, hMatch, hBind] at hExec
                  subst after
                  simp [StorageActionState.bindRead?] at hBind
                  rcases hBind with ⟨_, rfl⟩
                  exact hBefore
  | write place expression =>
      cases hPlace : semantics.evalLocation place before with
      | none => simp [execStorageAction?, hPlace] at hExec
      | some location =>
          cases hValue : semantics.evalValue expression before with
          | none => simp [execStorageAction?, hPlace, hValue] at hExec
          | some value =>
              cases hMatch : semantics.valueMatchesLocation place value with
              | false => simp [execStorageAction?, hPlace, hValue, hMatch] at hExec
              | true =>
                  simp [execStorageAction?, hPlace, hValue, hMatch] at hExec
                  subst after
                  exact checked_write_preserves before.loop.storage claims location value
                    hBefore (hChecked location value hPlace hValue)

abbrev Transition := StorageState → StorageState → Prop

def transitionEdge (step : Transition) : Edge StorageState := {
  path := fun _ => True
  safe := fun _ => True
  next := step
  target := .backedge
  targetAssertion := fun _ => True
}

/-- The existing loop reachability theorem also expresses repeated transactions
    and checked interference. This proves slot invariants only, not call safety,
    termination, non-reentrancy, or bytecode correctness. -/
def lifecycleVC (claims : List Claim) (initial : StorageState → Prop)
    (steps : List Transition) : LoopVC StorageState := {
  initial
  invariant := Invariant claims
  edges := steps.map transitionEdge
}

structure Certificate (claims : List Claim) (initial : StorageState → Prop)
    (steps : List Transition) : Prop where
  witness : ∃ store, initial store
  bootstrap : ∀ store, initial store → Invariant claims store
  preserve : ∀ step, step ∈ steps → ∀ before after,
    Invariant claims before → step before after → Invariant claims after

theorem certificate_induction
    (certificate : Certificate claims initial steps) :
    (lifecycleVC claims initial steps).InductionObligations := by
  constructor
  · exact certificate.bootstrap
  · intro edge before after hMember _ hInvariant _ _ hNext
    rcases List.mem_map.mp hMember with ⟨step, hStep, rfl⟩
    exact certificate.preserve step hStep before after hInvariant hNext
  · intro edge _ hMember _ _
    rcases List.mem_map.mp hMember with ⟨_, _, rfl⟩
    trivial
  · intro edge _ _ hMember hTarget _ _ _ _
    rcases List.mem_map.mp hMember with ⟨_, _, rfl⟩
    exact False.elim (hTarget rfl)

theorem reached_getter_fact
    (certificate : Certificate claims initial steps)
    (hInitial : initial start)
    (hReach : Reaches (lifecycleVC claims initial steps) start current)
    (hMember : claim ∈ claims) : claim.holds (current claim.location) := by
  have hInvariant := invariant_holds_on_reached_state
    (lifecycleVC claims initial steps) (certificate_induction certificate).base
    (certificate_induction certificate).backedge hInitial hReach
  exact hInvariant claim hMember

/-- Semantic coverage must be proved against the actual HIR execution step.
    An incomplete list of approved writers cannot provide this witness. -/
def Covers (actual : Transition) (steps : List Transition) : Prop :=
  ∀ before after, actual before after → ∃ step, step ∈ steps ∧ step before after

theorem covered_step_preserves
    (certificate : Certificate claims initial steps)
    (hCoverage : Covers actual steps)
    (hBefore : Invariant claims before) (hStep : actual before after) :
    Invariant claims after := by
  rcases hCoverage before after hStep with ⟨step, hMember, hNext⟩
  exact certificate.preserve step hMember before after hBefore hNext

end Ora.StorageLifecycle

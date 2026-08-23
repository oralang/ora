/-
Generic partial correctness and safety for loop control-flow edges.

The theorem is independent of source loop syntax and says nothing about
termination. `Reaches` contains only a finite sequence of certified backedges.
-/

namespace Ora.Loop

inductive ControlTarget where
  | backedge
  | successor
  | functionReturn
  | errorReturn
  | explicitRevert
  deriving Repr, BEq, DecidableEq

structure Edge (State : Type) where
  path : State → Prop
  safe : State → Prop
  next : State → State → Prop
  target : ControlTarget

structure LoopVC (State : Type) where
  initial : State → Prop
  invariant : State → Prop
  edges : List (Edge State)
  targetAssertion : ControlTarget → State → Prop

inductive Reaches (vc : LoopVC State) : State → State → Prop where
  | refl (state) : Reaches vc state state
  | next {initial current following edge} :
      Reaches vc initial current →
      edge ∈ vc.edges →
      edge.target = .backedge →
      edge.path current →
      edge.safe current →
      edge.next current following →
      Reaches vc initial following

def LoopVC.BaseObligation (vc : LoopVC State) : Prop :=
  ∀ state, vc.initial state → vc.invariant state

def LoopVC.EdgeSafetyObligation (vc : LoopVC State) : Prop :=
  ∀ edge state,
    edge ∈ vc.edges →
    vc.invariant state →
    edge.path state →
    edge.safe state

def LoopVC.BackedgeObligation (vc : LoopVC State) : Prop :=
  ∀ edge current following,
    edge ∈ vc.edges →
    edge.target = .backedge →
    vc.invariant current →
    edge.path current →
    edge.safe current →
    edge.next current following →
    vc.invariant following

def LoopVC.ExitTargetObligation (vc : LoopVC State) : Prop :=
  ∀ edge current following,
    edge ∈ vc.edges →
    edge.target ≠ .backedge →
    vc.invariant current →
    edge.path current →
    edge.safe current →
    edge.next current following →
    vc.targetAssertion edge.target following

def LoopVC.TargetsHoldOnReachableEdges (vc : LoopVC State) : Prop :=
  ∀ initial current following edge,
    vc.initial initial →
    Reaches vc initial current →
    edge ∈ vc.edges →
    edge.path current →
    edge.safe current →
    edge.next current following →
    edge.target ≠ .backedge →
    vc.targetAssertion edge.target following

def LoopVC.SafeOnReachableEdges (vc : LoopVC State) : Prop :=
  ∀ initial current edge,
    vc.initial initial →
    Reaches vc initial current →
    edge ∈ vc.edges →
    edge.path current →
    edge.safe current

def LoopVC.Verified (vc : LoopVC State) : Prop :=
  vc.TargetsHoldOnReachableEdges ∧ vc.SafeOnReachableEdges

structure LoopVC.InductionObligations (vc : LoopVC State) : Prop where
  base : vc.BaseObligation
  backedge : vc.BackedgeObligation
  edgeSafety : vc.EdgeSafetyObligation
  exit : vc.ExitTargetObligation

theorem invariant_holds_on_reached_state
    (vc : LoopVC State)
    (hBase : vc.BaseObligation)
    (hBackedgeObligation : vc.BackedgeObligation)
    {initial current : State}
    (hInitial : vc.initial initial)
    (hReaches : Reaches vc initial current) :
    vc.invariant current := by
  induction hReaches with
  | refl => exact hBase initial hInitial
  | @next previous following edge hPrefix hMember hBackedge hPath hSafe hNext ih =>
      exact hBackedgeObligation edge previous following hMember hBackedge ih hPath hSafe hNext

theorem targets_hold_on_reachable_edges
    (vc : LoopVC State)
    (hBase : vc.BaseObligation)
    (hBackedge : vc.BackedgeObligation)
    (hExit : vc.ExitTargetObligation) :
    vc.TargetsHoldOnReachableEdges := by
  intro initial current following edge hInitial hReaches hMember hPath hSafe hNext hNotBackedge
  exact hExit edge current following hMember hNotBackedge
    (invariant_holds_on_reached_state vc hBase hBackedge hInitial hReaches)
    hPath hSafe hNext

theorem safety_holds_on_reachable_edges
    (vc : LoopVC State)
    (hBase : vc.BaseObligation)
    (hBackedge : vc.BackedgeObligation)
    (hSafety : vc.EdgeSafetyObligation) :
    vc.SafeOnReachableEdges := by
  intro initial current edge hInitial hReaches hMember hPath
  exact hSafety edge current hMember
    (invariant_holds_on_reached_state vc hBase hBackedge hInitial hReaches)
    hPath

theorem verified_of_induction
    (vc : LoopVC State)
    (proof : vc.InductionObligations) :
    vc.Verified := by
  exact ⟨
    targets_hold_on_reachable_edges vc proof.base proof.backedge proof.exit,
    safety_holds_on_reachable_edges vc proof.base proof.backedge proof.edgeSafety
  ⟩

private inductive TinyState where
  | zero
  | one
  deriving DecidableEq

private def tinyBackedge : Edge TinyState where
  path := fun state => state = .zero
  safe := fun _ => True
  next := fun current following => current = .zero ∧ following = .one
  target := .backedge

private def tinyExit : Edge TinyState where
  path := fun state => state = .one
  safe := fun _ => True
  next := fun current following => current = following
  target := .successor

private def tinyLoop : LoopVC TinyState where
  initial := fun state => state = .zero
  invariant := fun state => state = .zero ∨ state = .one
  edges := [tinyBackedge, tinyExit]
  targetAssertion := fun target state => target = .successor ∧ state = .one

private theorem tinyLoop_base : tinyLoop.BaseObligation := by
  intro state hInitial
  exact Or.inl hInitial

private theorem tinyLoop_safety : tinyLoop.EdgeSafetyObligation := by
  intro edge state hMember _ _
  simp [tinyLoop] at hMember
  rcases hMember with hBackedge | hExit
  · rw [hBackedge]
    trivial
  · rw [hExit]
    trivial

private theorem tinyLoop_backedge : tinyLoop.BackedgeObligation := by
  intro edge current following hMember hTarget _ hPath _ hNext
  simp [tinyLoop] at hMember
  rcases hMember with hBackedge | hExit
  · rw [hBackedge] at hPath hNext
    exact Or.inr hNext.2
  · rw [hExit] at hTarget
    contradiction

private theorem tinyLoop_exit : tinyLoop.ExitTargetObligation := by
  intro edge current following hMember hTarget _ hPath _ hNext
  simp [tinyLoop] at hMember
  rcases hMember with hBackedge | hExit
  · rw [hBackedge] at hTarget
    contradiction
  · rw [hExit] at hPath hNext ⊢
    exact ⟨rfl, hNext ▸ hPath⟩

private def tinyLoop_induction : tinyLoop.InductionObligations where
  base := tinyLoop_base
  backedge := tinyLoop_backedge
  edgeSafety := tinyLoop_safety
  exit := tinyLoop_exit

example : tinyLoop.Verified :=
  verified_of_induction tinyLoop tinyLoop_induction

example : Reaches tinyLoop .zero .zero :=
  Reaches.refl TinyState.zero

example : Reaches tinyLoop .zero .one := by
  apply Reaches.next (edge := tinyBackedge) (Reaches.refl TinyState.zero)
  · simp [tinyLoop]
  · rfl
  · rfl
  · trivial
  · exact ⟨rfl, rfl⟩

private def falseBaseLoop : LoopVC Bool where
  initial := fun state => state = true
  invariant := fun _ => False
  edges := []
  targetAssertion := fun _ _ => True

example : ¬ falseBaseLoop.BaseObligation := by
  intro hBase
  exact hBase true rfl

private def falseSafetyEdge : Edge Bool where
  path := fun _ => True
  safe := fun _ => False
  next := fun current following => current = following
  target := .backedge

private def falseSafetyLoop : LoopVC Bool where
  initial := fun _ => True
  invariant := fun _ => True
  edges := [falseSafetyEdge]
  targetAssertion := fun _ _ => True

example : ¬ falseSafetyLoop.EdgeSafetyObligation := by
  intro hSafety
  exact hSafety falseSafetyEdge true (by simp [falseSafetyLoop]) trivial trivial

private def falseBackedgeEdge : Edge Bool where
  path := fun _ => True
  safe := fun _ => True
  next := fun current following => current = true ∧ following = false
  target := .backedge

private def falseBackedgeLoop : LoopVC Bool where
  initial := fun state => state = true
  invariant := fun state => state = true
  edges := [falseBackedgeEdge]
  targetAssertion := fun _ _ => True

example : ¬ falseBackedgeLoop.BackedgeObligation := by
  intro hBackedge
  have hInvariant := hBackedge falseBackedgeEdge true false
    (by simp [falseBackedgeLoop]) rfl rfl trivial trivial ⟨rfl, rfl⟩
  contradiction

private def falseExitEdge : Edge Bool where
  path := fun _ => True
  safe := fun _ => True
  next := fun current following => current = following
  target := .successor

private def falseExitLoop : LoopVC Bool where
  initial := fun _ => True
  invariant := fun _ => True
  edges := [falseExitEdge]
  targetAssertion := fun _ _ => False

example : ¬ falseExitLoop.ExitTargetObligation := by
  intro hExit
  exact hExit falseExitEdge true true (by simp [falseExitLoop]) (by decide)
    trivial trivial trivial rfl

end Ora.Loop

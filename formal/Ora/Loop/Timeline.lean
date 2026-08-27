/-
Relational execution of one canonical loop edge timeline.

The timeline is deliberately independent of compiler rows.  Direct storage
actions are deterministic, while certified child summaries may relate one
input state to several output states.  Path assumptions restrict executions;
safety checkpoints do not.  Safety is instead required at every reachable
prefix, so a false check cannot disappear by deleting a transition.
-/

import Ora.Loop.Storage

namespace Ora.Loop

/-- Canonical event order for an edge.  Formula, location, value, and child
types remain abstract until the compiler manifest instantiates this model. -/
inductive TimelineEvent
    (FormulaExpr LocationExpr ValueExpr Child : Type) where
  | assume : FormulaExpr → TimelineEvent FormulaExpr LocationExpr ValueExpr Child
  | check : FormulaExpr → TimelineEvent FormulaExpr LocationExpr ValueExpr Child
  | storage : StorageAction LocationExpr ValueExpr →
      TimelineEvent FormulaExpr LocationExpr ValueExpr Child
  | child : Child → TimelineEvent FormulaExpr LocationExpr ValueExpr Child

/-- Meaning of the expressions and certified child relations carried by an
edge timeline.  Every evaluator observes the exact state at its event. -/
structure TimelineSemantics
    (FormulaExpr LocationExpr ValueExpr Child : Type) where
  storage : StorageActionSemantics LocationExpr ValueExpr
  evalFormula : FormulaExpr → StorageActionState → Option Prop
  childRelation : Child → LoopState → LoopState → Prop
  childFrame : ∀ child, CertifiedWriteFrame (childRelation child)

/-- A relational execution result.  `writes` is an over-approximation for the
exact execution: direct actions contribute their resolved locations and child
events contribute their certified footprints. -/
structure TimelineResult where
  state : StorageActionState
  writes : StorageLocation → Prop := fun _ => False

def TimelineResult.comp
    (first following : TimelineResult) : TimelineResult := {
  state := following.state
  writes := fun location => first.writes location ∨ following.writes location
}

abbrev EventOf (FormulaExpr LocationExpr ValueExpr Child : Type) :=
  TimelineEvent FormulaExpr LocationExpr ValueExpr Child

/-- One event step.  Notice that `.check` requires the formula to denote but
does not require it to be true.  Truth is owned by `TimelineSafe`. -/
inductive TimelineEventExec
    (semantics : TimelineSemantics FormulaExpr LocationExpr ValueExpr Child) :
    EventOf FormulaExpr LocationExpr ValueExpr Child →
      StorageActionState → TimelineResult → Prop where
  | assume {formula state proposition}
      (hEval : semantics.evalFormula formula state = some proposition)
      (hHolds : proposition) :
      TimelineEventExec semantics (.assume formula) state { state }
  | check {formula state proposition}
      (hEval : semantics.evalFormula formula state = some proposition) :
      TimelineEventExec semantics (.check formula) state { state }
  | storage {action state result}
      (hExec : execStorageAction? semantics.storage action state = some result) :
      TimelineEventExec semantics (.storage action) state {
        state := result.state
        writes := fun location => location ∈ result.writeTrace
      }
  | child {child state following}
      (hRelation : semantics.childRelation child state.loop following) :
      TimelineEventExec semantics (.child child) state {
        state := { state with loop := following }
        writes := (semantics.childFrame child).footprint
      }

/-- Relational execution of a timeline.  Composition retains the footprint of
every event in execution order. -/
inductive TimelineExec
    (semantics : TimelineSemantics FormulaExpr LocationExpr ValueExpr Child) :
    List (EventOf FormulaExpr LocationExpr ValueExpr Child) →
      StorageActionState → TimelineResult → Prop where
  | nil (state) : TimelineExec semantics [] state { state }
  | cons {event rest initial first following}
      (hEvent : TimelineEventExec semantics event initial first)
      (hRest : TimelineExec semantics rest first.state following) :
      TimelineExec semantics (event :: rest) initial (first.comp following)

def timelineFormulaHolds
    (semantics : TimelineSemantics FormulaExpr LocationExpr ValueExpr Child)
    (formula : FormulaExpr)
    (state : StorageActionState) : Prop :=
  ∃ proposition, semantics.evalFormula formula state = some proposition ∧ proposition

/-- Every checkpoint is true in every state admitted by the exact preceding
event prefix.  This universal quantification is what prevents one safe child
result from hiding a second unsafe result. -/
def TimelineSafe
    (semantics : TimelineSemantics FormulaExpr LocationExpr ValueExpr Child)
    (events : List (EventOf FormulaExpr LocationExpr ValueExpr Child))
    (initial : StorageActionState) : Prop :=
  ∀ beforeEvents formula afterEvents,
    events = beforeEvents ++ .check formula :: afterEvents →
    ∀ beforeResult,
      TimelineExec semantics beforeEvents initial beforeResult →
      timelineFormulaHolds semantics formula beforeResult.state

theorem TimelineEventExec.preserves_entryStorage
    (semantics : TimelineSemantics FormulaExpr LocationExpr ValueExpr Child)
    (event : EventOf FormulaExpr LocationExpr ValueExpr Child)
    (initial : StorageActionState)
    (result : TimelineResult)
    (hExec : TimelineEventExec semantics event initial result) :
    result.state.entryStorage = initial.entryStorage := by
  cases hExec with
  | assume => rfl
  | check => rfl
  | storage hAction =>
      exact execStorageAction_preserves_entryStorage semantics.storage _ _ _ hAction
  | child => rfl

theorem TimelineEventExec.preserves_outside
    (semantics : TimelineSemantics FormulaExpr LocationExpr ValueExpr Child)
    (event : EventOf FormulaExpr LocationExpr ValueExpr Child)
    (initial : StorageActionState)
    (result : TimelineResult)
    (location : StorageLocation)
    (hExec : TimelineEventExec semantics event initial result)
    (hOutside : ¬ result.writes location) :
    result.state.loop.storage location = initial.loop.storage location := by
  cases hExec with
  | assume => rfl
  | check => rfl
  | storage hAction =>
      exact execStorageAction_preserves_unwritten semantics.storage _ _ _ location
        hAction hOutside
  | @child child _ following hRelation =>
      exact (semantics.childFrame child).preservesOutside initial.loop following
        hRelation location hOutside

theorem TimelineExec.preserves_entryStorage
    (semantics : TimelineSemantics FormulaExpr LocationExpr ValueExpr Child)
    (events : List (EventOf FormulaExpr LocationExpr ValueExpr Child))
    (initial : StorageActionState)
    (result : TimelineResult)
    (hExec : TimelineExec semantics events initial result) :
    result.state.entryStorage = initial.entryStorage := by
  induction hExec with
  | nil => rfl
  | @cons event rest initial first following hEvent hRest ih =>
      exact ih.trans (TimelineEventExec.preserves_entryStorage semantics event
        initial first hEvent)

theorem TimelineExec.preserves_outside
    (semantics : TimelineSemantics FormulaExpr LocationExpr ValueExpr Child)
    (events : List (EventOf FormulaExpr LocationExpr ValueExpr Child))
    (initial : StorageActionState)
    (result : TimelineResult)
    (location : StorageLocation)
    (hExec : TimelineExec semantics events initial result)
    (hOutside : ¬ result.writes location) :
    result.state.loop.storage location = initial.loop.storage location := by
  induction hExec with
  | nil => rfl
  | @cons event rest initial first following hEvent hRest ih =>
      have hOutsideFirst : ¬ first.writes location := by
        intro hWritten
        exact hOutside (Or.inl hWritten)
      have hOutsideRest : ¬ following.writes location := by
        intro hWritten
        exact hOutside (Or.inr hWritten)
      exact (ih hOutsideRest).trans
        (TimelineEventExec.preserves_outside semantics event initial first
          location hEvent hOutsideFirst)

/-- Final local bindings are evaluated after the complete trace.  They may
read the final store and preceding read identities, but cannot replace the
store produced by that same trace. -/
def timelineProducesLoopState
    (semantics : TimelineSemantics FormulaExpr LocationExpr ValueExpr Child)
    (events : List (EventOf FormulaExpr LocationExpr ValueExpr Child))
    (evalFinalLocals : StorageActionState → Option (List Ora.Integer.Value))
    (initial : StorageActionState)
    (following : LoopState) : Prop :=
  ∃ result locals,
    TimelineExec semantics events initial result ∧
    evalFinalLocals result.state = some locals ∧
    following = { locals, storage := result.state.loop.storage }

theorem timelineProducesLoopState_preserves_entry_old
    (semantics : TimelineSemantics FormulaExpr LocationExpr ValueExpr Child)
    (events : List (EventOf FormulaExpr LocationExpr ValueExpr Child))
    (evalFinalLocals : StorageActionState → Option (List Ora.Integer.Value))
    (initial : StorageActionState)
    (following : LoopState)
    (hProduces : timelineProducesLoopState semantics events evalFinalLocals
      initial following) :
    ∃ result,
      TimelineExec semantics events initial result ∧
      result.state.entryStorage = initial.entryStorage ∧
      following.storage = result.state.loop.storage := by
  rcases hProduces with ⟨result, locals, hExec, hLocals, rfl⟩
  exact ⟨result, hExec, TimelineExec.preserves_entryStorage semantics events
    initial result hExec, rfl⟩

/-- Bridge into the existing generic loop theorem.  Assumptions are executed
inside `next`; `path` stays true so an absent final child result cannot erase a
checkpoint reachable before that child. -/
def timelineEdge
    (semantics : TimelineSemantics FormulaExpr LocationExpr ValueExpr Child)
    (events : List (EventOf FormulaExpr LocationExpr ValueExpr Child))
    (entryStorage : StorageState)
    (evalFinalLocals : StorageActionState → Option (List Ora.Integer.Value))
    (target : ControlTarget)
    (targetAssertion : LoopState → Prop) : Edge LoopState where
  path := fun _ => True
  safe := fun current => TimelineSafe semantics events {
    loop := current, entryStorage := entryStorage
  }
  next := fun current following => timelineProducesLoopState semantics events
    evalFinalLocals { loop := current, entryStorage := entryStorage } following
  target := target
  targetAssertion := targetAssertion

theorem false_checkpoint_does_not_disappear
    (semantics : TimelineSemantics FormulaExpr LocationExpr ValueExpr Child)
    (formula : FormulaExpr)
    (state : StorageActionState)
    (hFalse : semantics.evalFormula formula state = some False) :
    TimelineEventExec semantics (.check formula) state { state } ∧
      ¬ TimelineSafe semantics [.check formula] state := by
  constructor
  · exact .check hFalse
  · intro hSafe
    have hRequired := hSafe [] formula [] (by rfl) { state } (.nil state)
    rcases hRequired with ⟨proposition, hEval, hHolds⟩
    have : proposition = False := Option.some.inj (hEval.symm.trans hFalse)
    subst proposition
    exact hHolds

end Ora.Loop

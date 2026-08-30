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

def TimelineEvent.NoStorageRead :
    EventOf FormulaExpr LocationExpr ValueExpr Child → Prop
  | .storage (.read _ _) => False
  | _ => True

def TimelineEvent.isStorageRead :
    EventOf FormulaExpr LocationExpr ValueExpr Child → Bool
  | .storage (.read _ _) => true
  | _ => false

def TimelineEvent.isCheck :
    EventOf FormulaExpr LocationExpr ValueExpr Child → Bool
  | .check _ => true
  | _ => false

def timelineHasCheck
    (events : List (EventOf FormulaExpr LocationExpr ValueExpr Child)) : Bool :=
  events.any TimelineEvent.isCheck

/-- A read may follow the last checkpoint, but it may not precede a checkpoint
handled by the control-only safety bridge. -/
def timelineReadBeforeCheckSupported :
    List (EventOf FormulaExpr LocationExpr ValueExpr Child) → Bool
  | [] => true
  | event :: rest =>
      (if event.isStorageRead then !timelineHasCheck rest else true) &&
        timelineReadBeforeCheckSupported rest

theorem timelineHasCheck_of_mem_check
    (events : List (EventOf FormulaExpr LocationExpr ValueExpr Child))
    (formula : FormulaExpr)
    (hMember : TimelineEvent.check formula ∈ events) :
    timelineHasCheck events = true := by
  unfold timelineHasCheck
  rw [List.any_eq_true]
  exact ⟨TimelineEvent.check formula, hMember, rfl⟩

theorem timelineReadBeforeCheckSupported_no_read
    (events : List (EventOf FormulaExpr LocationExpr ValueExpr Child))
    (hSupported : timelineReadBeforeCheckSupported events = true)
    (before : List (EventOf FormulaExpr LocationExpr ValueExpr Child))
    (formula : FormulaExpr)
    (after : List (EventOf FormulaExpr LocationExpr ValueExpr Child))
    (hEvents : events = before ++ TimelineEvent.check formula :: after)
    (observed : EventOf FormulaExpr LocationExpr ValueExpr Child)
    (hMember : observed ∈ before) :
    observed.NoStorageRead := by
  induction events generalizing before with
  | nil => simp at hEvents
  | cons head tail ih =>
      cases before with
      | nil => simp at hMember
      | cons beforeHead beforeTail =>
          simp only [List.cons_append, List.cons.injEq] at hEvents
          rcases hEvents with ⟨hHead, hTail⟩
          subst beforeHead
          simp only [List.mem_cons] at hMember
          have hTailSupported : timelineReadBeforeCheckSupported tail = true := by
            unfold timelineReadBeforeCheckSupported at hSupported
            simp only [Bool.and_eq_true] at hSupported
            exact hSupported.2
          rcases hMember with hHeadMember | hMember
          · subst observed
            cases head with
            | assume => trivial
            | check => trivial
            | child => trivial
            | storage action =>
                cases action with
                | write => trivial
                | read =>
                    have hHasCheck := timelineHasCheck_of_mem_check tail formula (by
                      rw [hTail]
                      simp)
                    unfold timelineReadBeforeCheckSupported at hSupported
                    simp only [TimelineEvent.isStorageRead, if_true,
                      Bool.and_eq_true] at hSupported
                    rw [hHasCheck] at hSupported
                    simp at hSupported
          · exact ih hTailSupported beforeTail hTail hMember

def TimelineEvent.readIds :
    EventOf FormulaExpr LocationExpr ValueExpr Child → List StorageReadId
  | .storage (.read id _) => [id]
  | _ => []

def timelineReadIdsOf
    (events : List (EventOf FormulaExpr LocationExpr ValueExpr Child)) :
    List StorageReadId :=
  events.flatMap TimelineEvent.readIds

theorem TimelineEvent.check_ne_assume
    (check : FormulaExpr) (assumption : FormulaExpr) :
    TimelineEvent.check check ≠
      (TimelineEvent.assume assumption : EventOf FormulaExpr LocationExpr ValueExpr Child) := by
  intro hEqual
  cases hEqual

theorem TimelineEvent.check_ne_storage
    (check : FormulaExpr) (action : StorageAction LocationExpr ValueExpr) :
    TimelineEvent.check check ≠
      (TimelineEvent.storage action : EventOf FormulaExpr LocationExpr ValueExpr Child) := by
  intro hEqual
  cases hEqual

theorem TimelineEvent.assume_ne_storage
    (assumption : FormulaExpr) (action : StorageAction LocationExpr ValueExpr) :
    TimelineEvent.assume assumption ≠
      (TimelineEvent.storage action : EventOf FormulaExpr LocationExpr ValueExpr Child) := by
  intro hEqual
  cases hEqual

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

def TimelineChecksHold
    (semantics : TimelineSemantics FormulaExpr LocationExpr ValueExpr Child)
    (events : List (EventOf FormulaExpr LocationExpr ValueExpr Child))
    (initial : StorageActionState) : Prop :=
  ∀ formula, .check formula ∈ events →
    timelineFormulaHolds semantics formula initial

/-- A canonical leading path prefix is witnessed by execution at the exact
entry state. Later storage reads cannot retroactively affect these formulas. -/
theorem TimelineExec.leading_assumptions_hold
    (semantics : TimelineSemantics FormulaExpr LocationExpr ValueExpr Child)
    (formulas : List FormulaExpr)
    (rest : List (EventOf FormulaExpr LocationExpr ValueExpr Child))
    (initial : StorageActionState)
    (result : TimelineResult)
    (hExec : TimelineExec semantics
      (formulas.map TimelineEvent.assume ++ rest) initial result) :
    ∀ formula ∈ formulas,
      timelineFormulaHolds semantics formula initial := by
  induction formulas generalizing initial result with
  | nil => simp
  | cons head tail ih =>
      simp only [List.map_cons, List.cons_append] at hExec
      cases hExec with
      | @cons event events _ first following hEvent hRest =>
          cases hEvent with
          | assume hEval hHolds =>
              intro formula hMember
              simp only [List.mem_cons] at hMember
              rcases hMember with rfl | hTail
              · exact ⟨_, hEval, hHolds⟩
              · exact ih _ _ hRest formula hTail

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

theorem TimelineEventExec.preserves_locals
    (semantics : TimelineSemantics FormulaExpr LocationExpr ValueExpr Child)
    (event : EventOf FormulaExpr LocationExpr ValueExpr Child)
    (initial : StorageActionState)
    (result : TimelineResult)
    (hChildPreserves : ∀ child before after,
      semantics.childRelation child before after → after.locals = before.locals)
    (hExec : TimelineEventExec semantics event initial result) :
    result.state.loop.locals = initial.loop.locals := by
  cases hExec with
  | assume => rfl
  | check => rfl
  | storage hAction =>
      exact execStorageAction_preserves_locals semantics.storage _ _ _ hAction
  | child hRelation => exact hChildPreserves _ _ _ hRelation

theorem TimelineEventExec.preserves_reads
    (semantics : TimelineSemantics FormulaExpr LocationExpr ValueExpr Child)
    (event : EventOf FormulaExpr LocationExpr ValueExpr Child)
    (initial : StorageActionState)
    (result : TimelineResult)
    (hNoRead : event.NoStorageRead)
    (hExec : TimelineEventExec semantics event initial result) :
    result.state.reads = initial.reads := by
  cases hExec with
  | assume => rfl
  | check => rfl
  | @storage action _ actionResult hAction =>
      cases action with
      | read => exact False.elim hNoRead
      | write location value =>
          exact execStorageWrite_preserves_reads semantics.storage location value
            initial actionResult hAction
  | child => rfl

theorem TimelineEventExec.read_ids
    (semantics : TimelineSemantics FormulaExpr LocationExpr ValueExpr Child)
    (event : EventOf FormulaExpr LocationExpr ValueExpr Child)
    (initial : StorageActionState)
    (result : TimelineResult)
    (hExec : TimelineEventExec semantics event initial result) :
    result.state.reads.map StorageReadBinding.id =
      initial.reads.map StorageReadBinding.id ++ event.readIds := by
  cases hExec with
  | assume => simp [TimelineEvent.readIds]
  | check => simp [TimelineEvent.readIds]
  | @storage action _ actionResult hAction =>
      cases action with
      | read id location =>
          simpa [TimelineEvent.readIds] using
            execStorageRead_appends_read_id semantics.storage id location
              initial actionResult hAction
      | write location value =>
          rw [execStorageWrite_preserves_reads semantics.storage location value
            initial actionResult hAction]
          simp [TimelineEvent.readIds]
  | child => simp [TimelineEvent.readIds]

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

theorem TimelineExec.preserves_locals
    (semantics : TimelineSemantics FormulaExpr LocationExpr ValueExpr Child)
    (events : List (EventOf FormulaExpr LocationExpr ValueExpr Child))
    (initial : StorageActionState)
    (result : TimelineResult)
    (hChildPreserves : ∀ child before after,
      semantics.childRelation child before after → after.locals = before.locals)
    (hExec : TimelineExec semantics events initial result) :
    result.state.loop.locals = initial.loop.locals := by
  induction hExec with
  | nil => rfl
  | @cons event rest initial first following hEvent hRest ih =>
      exact ih.trans
        (TimelineEventExec.preserves_locals semantics event initial first
          hChildPreserves hEvent)

theorem TimelineExec.preserves_reads
    (semantics : TimelineSemantics FormulaExpr LocationExpr ValueExpr Child)
    (events : List (EventOf FormulaExpr LocationExpr ValueExpr Child))
    (initial : StorageActionState)
    (result : TimelineResult)
    (hNoReads : ∀ event, event ∈ events → event.NoStorageRead)
    (hExec : TimelineExec semantics events initial result) :
    result.state.reads = initial.reads := by
  induction hExec with
  | nil => rfl
  | @cons event rest initial first following hEvent hRest ih =>
      have hHead : event.NoStorageRead := hNoReads event (by simp)
      have hTail : ∀ tailEvent, tailEvent ∈ rest → tailEvent.NoStorageRead := by
        intro tailEvent hMember
        exact hNoReads tailEvent (by simp [hMember])
      exact (ih hTail).trans
        (TimelineEventExec.preserves_reads semantics event initial first
          hHead hEvent)

theorem TimelineExec.read_ids
    (semantics : TimelineSemantics FormulaExpr LocationExpr ValueExpr Child)
    (events : List (EventOf FormulaExpr LocationExpr ValueExpr Child))
    (initial : StorageActionState)
    (result : TimelineResult)
    (hExec : TimelineExec semantics events initial result) :
    result.state.reads.map StorageReadBinding.id =
      initial.reads.map StorageReadBinding.id ++ timelineReadIdsOf events := by
  induction hExec with
  | nil => simp [timelineReadIdsOf]
  | @cons event rest initial first following hEvent hRest ih =>
      change following.state.reads.map StorageReadBinding.id = _
      rw [ih, TimelineEventExec.read_ids semantics event initial first hEvent]
      simp only [timelineReadIdsOf, List.flatMap_cons, List.append_assoc]

/-- A check occurring after a canonical leading assumption block has that
whole block in its execution prefix.  This is a list-shape fact only; it does
not assign semantics to either assumptions or checks. -/
theorem prefix_before_check_contains_leading_assumptions
    (formulas : List FormulaExpr)
    (rest before after : List (EventOf FormulaExpr LocationExpr ValueExpr Child))
    (formula : FormulaExpr)
    (hEvents : formulas.map TimelineEvent.assume ++ rest =
      before ++ .check formula :: after) :
    ∃ following, before = formulas.map TimelineEvent.assume ++ following := by
  induction formulas generalizing rest before with
  | nil => exact ⟨before, rfl⟩
  | cons head tail ih =>
      cases before with
      | nil =>
          simp only [List.map_cons, List.cons_append, List.nil_append] at hEvents
          cases hEvents
      | cons beforeHead beforeTail =>
          simp only [List.map_cons, List.cons_append, List.cons.injEq] at hEvents
          rcases hEvents with ⟨hHead, hTail⟩
          subst beforeHead
          rcases ih rest beforeTail hTail with ⟨following, rfl⟩
          exact ⟨following, rfl⟩

/-- Lift control-level path and checkpoint facts to an exact timeline whose
canonical path assumptions form the leading prefix.  This theorem handles
storage writes between assumptions and checks without replaying the action
semantics.  Read-dependent checks require the corresponding read-binding
bridge and are deliberately not hidden behind this no-read lemma. -/
theorem timelineSafe_of_leading_control
    (semantics : TimelineSemantics FormulaExpr LocationExpr ValueExpr Child)
    (path : List FormulaExpr)
    (rest : List (EventOf FormulaExpr LocationExpr ValueExpr Child))
    (initial : StorageActionState)
    (hFormulaStable : ∀ before formula after result,
      path.map TimelineEvent.assume ++ rest =
        before ++ .check formula :: after →
      TimelineExec semantics before initial result →
      timelineFormulaHolds semantics formula initial →
      timelineFormulaHolds semantics formula result.state)
    (hPath : (∀ formula ∈ path,
      timelineFormulaHolds semantics formula initial) → Path)
    (hChecks : Path →
      TimelineChecksHold semantics (path.map TimelineEvent.assume ++ rest) initial) :
    TimelineSafe semantics (path.map TimelineEvent.assume ++ rest) initial := by
  intro before formula after hEvents beforeResult hExec
  rcases prefix_before_check_contains_leading_assumptions
      path rest before after formula hEvents with ⟨following, rfl⟩
  have hLeading := TimelineExec.leading_assumptions_hold semantics path following
    initial beforeResult hExec
  have hControlPath := hPath hLeading
  have hAtInitial := hChecks hControlPath formula (by
    rw [hEvents]
    simp)
  exact hFormulaStable _ formula _ beforeResult hEvents hExec hAtInitial

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

/-- Package a proof about the exact trace into the generic ordered-edge
refinement used by loop induction. The trace is opened once here; emitted
proofs provide only compact safety, path, final-local, and target bridges. -/
theorem timelineEdge_refinement
    (abstractVC : LoopVC (List Ora.Integer.Value))
    (concreteVC : LoopVC LoopState)
    (abstractEdge : Edge (List Ora.Integer.Value))
    (semantics : TimelineSemantics FormulaExpr LocationExpr ValueExpr Child)
    (events : List (EventOf FormulaExpr LocationExpr ValueExpr Child))
    (entryStorage : StorageState)
    (evalFinalLocals : StorageActionState → Option (List Ora.Integer.Value))
    (target : ControlTarget)
    (targetAssertion : LoopState → Prop)
    (hTarget : abstractEdge.target = target)
    (hSafety : ∀ state,
      concreteVC.invariant state →
      abstractVC.invariant state.locals →
      (abstractEdge.path state.locals → abstractEdge.safe state.locals) →
      TimelineSafe semantics events { loop := state, entryStorage := entryStorage })
    (hPath : ∀ current result,
      TimelineExec semantics events
        { loop := current, entryStorage := entryStorage } result →
      abstractEdge.path current.locals)
    (hNext : ∀ current result locals,
      TimelineExec semantics events
        { loop := current, entryStorage := entryStorage } result →
      evalFinalLocals result.state = some locals →
      abstractEdge.next current.locals locals)
    (hAssertion : ∀ following,
      abstractEdge.targetAssertion following.locals →
      targetAssertion following) :
    Edge.Refinement abstractVC concreteVC (fun state => state.locals)
      abstractEdge
      (timelineEdge semantics events entryStorage evalFinalLocals target
        targetAssertion) where
  target := hTarget
  safety := by
    intro state hInvariant _ hAbstractInvariant hAbstractSafety
    exact hSafety state hInvariant hAbstractInvariant hAbstractSafety
  transition := by
    intro current following _ _ _ hProduces
    rcases hProduces with ⟨result, locals, hExec, hLocals, rfl⟩
    exact ⟨hPath current result hExec,
      hNext current result locals hExec hLocals⟩
  targetAssertion := hAssertion

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

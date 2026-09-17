/-
Relational execution of one canonical loop edge timeline.

The timeline is deliberately independent of compiler rows.  Direct storage
actions are deterministic, while certified child summaries may relate one
input state to several output states.  Path assumptions restrict executions;
safety checkpoints do not.  Safety is instead required at every reachable
prefix, so a false check cannot disappear by deleting a transition.
-/

import Ora.Loop.IndexedRegion
import Ora.Loop.CallTypes

namespace Ora.Loop
open Ora.Obligation

/-- A lexical loop exports only declared SSA results. The loop retains the
function-entry snapshot; it does not start a new function-call scope. -/
structure LoopResultBinding where
  id : FreeVarId
  index : Nat
  ty : TyRef

def loopResultBindings? (bindings : List LoopResultBinding) (following : LoopState) :
    Option (List StorageReadBinding) := bindings.mapM fun binding => do
  let value := Value.integer (← following.locals[binding.index]?)
  if storageValueMatchesTy value binding.ty then some { id := binding.id, value } else none

def resumeLoop? (bindings : List LoopResultBinding)
    (caller : StorageActionState) (following : LoopState) : Option StorageActionState := do
  if !(decide (caller.reads.map StorageReadBinding.id ++ bindings.map (·.id)).Nodup) then none
  else pure ()
  let results ← loopResultBindings? bindings following
  return { caller with
    loop := { caller.loop with storage := following.storage, returned := following.returned }
    reads := caller.reads ++ results }

theorem loopResultBindings_ids (bindings : List LoopResultBinding) (following : LoopState)
    (results : List StorageReadBinding) (evaluated : loopResultBindings? bindings following = some results) :
    results.map StorageReadBinding.id = bindings.map (·.id) := by
  induction bindings generalizing results with
  | nil => simpa [loopResultBindings?] using evaluated.symm
  | cons binding rest ih =>
    simp only [loopResultBindings?, List.mapM_cons, Option.bind_eq_bind] at evaluated
    cases value : following.locals[binding.index]? with
    | none => simp [value] at evaluated
    | some val =>
      simp [value] at evaluated
      split at evaluated <;> simp_all
      change ((loopResultBindings? rest following).bind fun tail =>
        some ({ id := binding.id, value := Value.integer val } :: tail)) = some results at evaluated
      cases tail : loopResultBindings? rest following <;> simp [tail] at evaluated
      cases evaluated
      simp [ih _ tail]

theorem resumeLoop_effects (bindings : List LoopResultBinding) (caller : StorageActionState)
    (following : LoopState) (resumed : StorageActionState)
    (evaluated : resumeLoop? bindings caller following = some resumed) :
    resumed.entryStorage = caller.entryStorage ∧ resumed.loop.locals = caller.loop.locals ∧
    resumed.loop.storage = following.storage ∧ resumed.loop.returned = following.returned ∧
    resumed.reads.map StorageReadBinding.id = caller.reads.map StorageReadBinding.id ++ bindings.map (·.id) ∧
    ∃ appended, resumed.reads = caller.reads ++ appended := by
  unfold resumeLoop? at evaluated
  split at evaluated <;> simp_all
  cases results : loopResultBindings? bindings following <;> simp [results] at evaluated
  cases evaluated
  exact ⟨rfl, rfl, rfl, rfl, by simp [loopResultBindings_ids _ _ _ results], _, rfl⟩


/-- Canonical event order for an edge.  Formula, location, value, and child
types remain abstract until the compiler manifest instantiates this model. -/
inductive TimelineEvent
    (FormulaExpr LocationExpr ValueExpr Child : Type) where
  | assume : FormulaExpr → TimelineEvent FormulaExpr LocationExpr ValueExpr Child
  | check : FormulaExpr → TimelineEvent FormulaExpr LocationExpr ValueExpr Child
  | storage : StorageAction LocationExpr ValueExpr →
      TimelineEvent FormulaExpr LocationExpr ValueExpr Child
  | indexedRead : IndexedRegionRead ValueExpr →
      TimelineEvent FormulaExpr LocationExpr ValueExpr Child
  | child : Child → TimelineEvent FormulaExpr LocationExpr ValueExpr Child
  | childResult : Child → List LoopResultBinding → TimelineEvent FormulaExpr LocationExpr ValueExpr Child
  | call : CallInvocation ValueExpr → TimelineEvent FormulaExpr LocationExpr ValueExpr Child

/-- Meaning of the expressions and certified child relations carried by an
edge timeline.  Every evaluator observes the exact state at its event. -/
structure TimelineSemantics
    (FormulaExpr LocationExpr ValueExpr Child : Type) where
  storage : StorageActionSemantics LocationExpr ValueExpr
  indexedRegion : IndexedRegionReadSemantics ValueExpr :=
    IndexedRegionReadSemantics.unsupported
  evalFormula : FormulaExpr → StorageActionState → Option Prop
  /-- A loop after a call observes the actual prefix reads and caller-entry
  store. Its frame quantifies over stores with those bindings held fixed. -/
  childRelation : Child → StorageActionState → LoopState → Prop
  childFrame : ∀ child (context : StorageActionState), CertifiedWriteFrame
    (fun before after => childRelation child { context with loop := before } after)
  calls : String → Option CertifiedCall := fun _ => none

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

def TimelineEvent.NoRead :
    EventOf FormulaExpr LocationExpr ValueExpr Child → Prop
  | .storage (.read _ _) => False
  | .indexedRead _ => False
  | .childResult _ _ | .call _ => False
  | _ => True

def TimelineEvent.isRead :
    EventOf FormulaExpr LocationExpr ValueExpr Child → Bool
  | .storage (.read _ _) => true
  | .indexedRead _ => true
  | .childResult _ _ | .call _ => true
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
      (if event.isRead then !timelineHasCheck rest else true) &&
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
    observed.NoRead := by
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
            | call _ | childResult _ _ =>
                have hHasCheck := timelineHasCheck_of_mem_check tail formula (by
                  rw [hTail]; simp)
                simp [timelineReadBeforeCheckSupported, TimelineEvent.isRead,
                  hHasCheck] at hSupported
            | indexedRead read =>
                have hHasCheck := timelineHasCheck_of_mem_check tail formula (by
                  rw [hTail]
                  simp)
                unfold timelineReadBeforeCheckSupported at hSupported
                simp only [TimelineEvent.isRead, if_true,
                  Bool.and_eq_true] at hSupported
                rw [hHasCheck] at hSupported
                simp at hSupported
            | storage action =>
                cases action with
                | write => trivial
                | read =>
                    have hHasCheck := timelineHasCheck_of_mem_check tail formula (by
                      rw [hTail]
                      simp)
                    unfold timelineReadBeforeCheckSupported at hSupported
                    simp only [TimelineEvent.isRead, if_true,
                      Bool.and_eq_true] at hSupported
                    rw [hHasCheck] at hSupported
                    simp at hSupported
          · exact ih hTailSupported beforeTail hTail hMember

def TimelineEvent.readIds :
    EventOf FormulaExpr LocationExpr ValueExpr Child → List StorageReadId
  | .storage (.read id _) => [id]
  | .indexedRead read => [read.id]
  | .childResult _ bindings => bindings.map (·.id)
  | .call invocation => invocation.results
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

theorem TimelineEvent.check_ne_indexedRead
    (check : FormulaExpr) (read : IndexedRegionRead ValueExpr) :
    TimelineEvent.check check ≠
      (TimelineEvent.indexedRead read : EventOf FormulaExpr LocationExpr ValueExpr Child) := by
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
  | indexedRead {read state following}
      (hExec : IndexedRegionReadExec semantics.indexedRegion read state following) :
      TimelineEventExec semantics (.indexedRead read) state { state := following }
  | child {child state following}
      (hRelation : semantics.childRelation child state following) :
      TimelineEventExec semantics (.child child) state {
        state := { state with loop := following }
        writes := (semantics.childFrame child state).footprint
      }
  | childResult {child bindings state following resumed}
      (hRelation : semantics.childRelation child state following)
      (hResume : resumeLoop? bindings state following = some resumed) :
      TimelineEventExec semantics (.childResult child bindings) state {
        state := resumed
        writes := (semantics.childFrame child state).footprint
      }
  | call {call state callee arguments output}
      (hResolve : semantics.calls call.callee = some callee)
      (hArguments : callEvaluateValues? semantics.storage.evalValue call.arguments state =
        some arguments)
      (hRelation : callee.relation { arguments, storage := state.loop.storage } output)
      (hOutcome : output.outcome = call.outcome)
      (hArity : output.payload.length = call.results.length)
      (hFresh : call.fresh state) :
      TimelineEventExec semantics (.call call) state {
        state := call.resume state output
        writes := callee.footprint { arguments, storage := state.loop.storage }
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

/-- A local checkpoint is proved before universally considering the next
event result. This also applies when the event has no completed execution. -/
theorem TimelineSafe.cons (semantics : TimelineSemantics F L V C)
    (event : EventOf F L V C) (rest : List (EventOf F L V C)) (initial : StorageActionState)
    (head : ∀ formula, event = .check formula → timelineFormulaHolds semantics formula initial)
    (tail : ∀ result, TimelineEventExec semantics event initial result →
      TimelineSafe semantics rest result.state) : TimelineSafe semantics (event :: rest) initial := by
  intro before formula after events result trace
  cases before with
  | nil =>
    simp only [List.nil_append, List.cons.injEq] at events
    cases trace
    exact head formula events.1
  | cons first following =>
    simp only [List.cons_append, List.cons.injEq] at events
    rcases events with ⟨rfl, hRest⟩
    cases trace with
    | @cons _ _ _ firstResult restResult hEvent hTrace =>
      exact tail firstResult hEvent following formula after hRest restResult hTrace

/-- A proof-obligation fold, not an executor: each assumption scopes only the
remaining checks. It supplies no values, states, transitions or authority. -/
def timelineControlChecks (holds : FormulaExpr → Prop) :
    List (EventOf FormulaExpr LocationExpr ValueExpr Child) → Prop
  | [] => True
  | .assume formula :: rest => holds formula → timelineControlChecks holds rest
  | .check formula :: rest => holds formula ∧ timelineControlChecks holds rest
  | _ :: rest => timelineControlChecks holds rest

/-- Transport requires every admitted event execution to preserve `P`, and
every relevant formula to agree with the control proposition in every such
state. Neither premise can use a later path fact or a completed transition. -/
theorem timelineSafe_of_control_checks
    (semantics : TimelineSemantics FormulaExpr LocationExpr ValueExpr Child)
    (P : StorageActionState → Prop) (holds : FormulaExpr → Prop)
    (hStep : ∀ event before result, P before →
      TimelineEventExec semantics event before result → P result.state)
    (events : List (EventOf FormulaExpr LocationExpr ValueExpr Child))
    (hFormula : ∀ formula,
      (.assume formula ∈ events ∨ .check formula ∈ events) →
      ∀ state, P state → (timelineFormulaHolds semantics formula state ↔ holds formula))
    (initial : StorageActionState) (hInitial : P initial)
    (hChecks : timelineControlChecks holds events) :
    TimelineSafe semantics events initial := by
  induction events generalizing initial with
  | nil => intro before formula after hEvents; cases before <;> simp at hEvents
  | cons event rest ih =>
      intro before formula after hEvents result hExec
      cases before with
      | nil =>
          simp only [List.nil_append, List.cons.injEq] at hEvents
          rcases hEvents with ⟨rfl, rfl⟩
          cases hExec
          exact (hFormula formula (Or.inr (by simp)) initial hInitial).mpr hChecks.1
      | cons head tail =>
          simp only [List.cons_append, List.cons.injEq] at hEvents
          rcases hEvents with ⟨rfl, hRest⟩
          cases hExec with
          | cons hEvent hTail =>
              apply ih (fun f hMember => hFormula f
                (hMember.imp (List.mem_cons_of_mem _) (List.mem_cons_of_mem _)))
                _ (hStep _ _ _ hInitial hEvent) ?_ tail formula after hRest _ hTail
              cases hEvent with
              | assume hEval hHolds =>
                  exact hChecks ((hFormula _ (Or.inl (by simp)) _ hInitial).mp
                    ⟨_, hEval, hHolds⟩)
              | check => exact hChecks.2
              | storage | indexedRead | child | childResult | call => exact hChecks

/-- Extract only assumptions actually executed. Stability is explicit; the
result is not inferred from row membership or a declared effect inventory. -/
theorem TimelineExec.control_assumptions
    (semantics : TimelineSemantics FormulaExpr LocationExpr ValueExpr Child)
    (P : StorageActionState → Prop) (holds : FormulaExpr → Prop)
    (hStep : ∀ event before result, P before →
      TimelineEventExec semantics event before result → P result.state)
    (events : List (EventOf FormulaExpr LocationExpr ValueExpr Child))
    (initial : StorageActionState) (result : TimelineResult)
    (hInitial : P initial)
    (hFormula : ∀ formula, .assume formula ∈ events →
      ∀ state, P state → (timelineFormulaHolds semantics formula state → holds formula))
    (hExec : TimelineExec semantics events initial result) :
    ∀ formula, .assume formula ∈ events → holds formula := by
  induction hExec with
  | nil => simp
  | cons hEvent hRest ih =>
      intro formula hMember
      rcases List.mem_cons.mp hMember with hHead | hTail
      · subst hHead
        cases hEvent with
        | assume hEval hHolds => exact hFormula _ (by simp) _ hInitial ⟨_, hEval, hHolds⟩
      · exact ih (hStep _ _ _ hInitial hEvent)
          (fun f hf => hFormula f (List.mem_cons_of_mem _ hf)) formula hTail

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
  | indexedRead hRead =>
      exact (IndexedRegionReadExec.state_effects _ _ _ _ hRead).2.2.1
  | child => rfl
  | childResult _ hResume => exact (resumeLoop_effects _ _ _ _ hResume).1
  | call => rfl

theorem TimelineEventExec.preserves_locals
    (semantics : TimelineSemantics FormulaExpr LocationExpr ValueExpr Child)
    (event : EventOf FormulaExpr LocationExpr ValueExpr Child)
    (initial : StorageActionState)
    (result : TimelineResult)
    (hChildPreserves : ∀ child before after,
      semantics.childRelation child before after → after.locals = before.loop.locals)
    (hExec : TimelineEventExec semantics event initial result) :
    result.state.loop.locals = initial.loop.locals := by
  cases hExec with
  | assume => rfl
  | check => rfl
  | storage hAction =>
      exact execStorageAction_preserves_locals semantics.storage _ _ _ hAction
  | indexedRead hRead =>
      exact congrArg LoopState.locals
        (IndexedRegionReadExec.state_effects _ _ _ _ hRead).2.1
  | child hRelation => exact hChildPreserves _ _ _ hRelation
  | childResult _ hResume => exact (resumeLoop_effects _ _ _ _ hResume).2.1
  | call => rfl

theorem TimelineEventExec.preserves_reads
    (semantics : TimelineSemantics FormulaExpr LocationExpr ValueExpr Child)
    (event : EventOf FormulaExpr LocationExpr ValueExpr Child)
    (initial : StorageActionState)
    (result : TimelineResult)
    (hNoRead : event.NoRead)
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
  | indexedRead => exact False.elim hNoRead
  | child => rfl
  | childResult | call => exact False.elim hNoRead

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
  | indexedRead hRead =>
      simpa [TimelineEvent.readIds] using
        (IndexedRegionReadExec.state_effects _ _ _ _ hRead).2.2.2
  | child => simp [TimelineEvent.readIds]
  | childResult _ hResume => exact (resumeLoop_effects _ _ _ _ hResume).2.2.2.2.1
  | call _ _ _ _ hArity _ =>
      simp only [CallInvocation.resume, List.map_append, List.map_map,
        TimelineEvent.readIds]
      congr 1
      simpa using List.map_fst_zip _ _ (by omega)

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
  | indexedRead hRead =>
      exact congrArg (fun loop => loop.storage location)
        (IndexedRegionReadExec.state_effects _ _ _ _ hRead).2.1
  | @child child _ following hRelation =>
      exact (semantics.childFrame child initial).preservesOutside initial.loop following
        hRelation location hOutside
  | @childResult child bindings _ following resumed hRelation hResume =>
      rw [(resumeLoop_effects _ _ _ _ hResume).2.2.1]
      exact (semantics.childFrame child initial).preservesOutside initial.loop following
        hRelation location hOutside
  | call _ _ hRelation _ _ _ =>
      exact CertifiedCall.preservesOutside _ _ _ hRelation location hOutside

theorem TimelineEventExec.read_writes_false
    (semantics : TimelineSemantics FormulaExpr LocationExpr ValueExpr Child)
    (id : StorageReadId)
    (location : LocationExpr)
    (initial : StorageActionState)
    (result : TimelineResult)
    (hExec : TimelineEventExec semantics (.storage (.read id location))
      initial result) :
    ∀ written, ¬ result.writes written := by
  cases hExec with
  | storage hAction =>
      have hTrace := execStorageRead_writeTrace_nil _ _ _ _ _ hAction
      simp [hTrace]

theorem TimelineEventExec.indexed_read_writes_false
    (semantics : TimelineSemantics FormulaExpr LocationExpr ValueExpr Child)
    (read : IndexedRegionRead ValueExpr)
    (initial : StorageActionState)
    (result : TimelineResult)
    (hExec : TimelineEventExec semantics (.indexedRead read) initial result) :
    ∀ written, ¬ result.writes written := by
  cases hExec
  simp

theorem TimelineEventExec.write_within_resolved
    (semantics : TimelineSemantics FormulaExpr LocationExpr ValueExpr Child)
    (location : LocationExpr)
    (value : ValueExpr)
    (resolved : StorageLocation)
    (initial : StorageActionState)
    (result : TimelineResult)
    (hResolved : semantics.storage.evalLocation location initial = some resolved)
    (hExec : TimelineEventExec semantics (.storage (.write location value))
      initial result) :
    ∀ written, result.writes written → written = resolved := by
  cases hExec with
  | storage hAction =>
      rcases execStorageWrite_writeTrace_singleton _ _ _ _ _ hAction with
        ⟨actual, hActual, hTrace⟩
      have hEqual : actual = resolved := Option.some.inj (hActual.symm.trans hResolved)
      subst actual
      simp [hTrace]

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
      semantics.childRelation child before after → after.locals = before.loop.locals)
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
    (hNoReads : ∀ event, event ∈ events → event.NoRead)
    (hExec : TimelineExec semantics events initial result) :
    result.state.reads = initial.reads := by
  induction hExec with
  | nil => rfl
  | @cons event rest initial first following hEvent hRest ih =>
      have hHead : event.NoRead := hNoReads event (by simp)
      have hTail : ∀ tailEvent, tailEvent ∈ rest → tailEvent.NoRead := by
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

/-- Lift independently established per-event write frames to the complete
timeline.  The premise is state-polymorphic: coverage must hold for every
execution admitted by an event, so later events cannot justify earlier
writes and omitted writes cannot disappear in composition. -/
theorem TimelineExec.writes_within
    (semantics : TimelineSemantics FormulaExpr LocationExpr ValueExpr Child)
    (events : List (EventOf FormulaExpr LocationExpr ValueExpr Child))
    (initial : StorageActionState)
    (result : TimelineResult)
    (frame : List StorageLocation)
    (hEventFrame : ∀ event, event ∈ events →
      ∀ before after,
        TimelineEventExec semantics event before after →
        ∀ written, after.writes written → written ∈ frame)
    (hExec : TimelineExec semantics events initial result) :
    ∀ written, result.writes written → written ∈ frame := by
  induction hExec with
  | nil => simp [TimelineResult.comp]
  | @cons event rest before first following hEvent hRest ih =>
      intro written hWritten
      rcases hWritten with hFirst | hFollowing
      · exact hEventFrame event (by simp) before first hEvent written hFirst
      · apply ih
        · intro tail hMember
          exact hEventFrame tail (by simp [hMember])
        · exact hFollowing

/-- A complete per-event frame plus disjointness proves preservation without
requiring a certificate author to destruct the timeline execution tree. -/
theorem TimelineExec.preserves_outside_event_frame
    (semantics : TimelineSemantics FormulaExpr LocationExpr ValueExpr Child)
    (events : List (EventOf FormulaExpr LocationExpr ValueExpr Child))
    (initial : StorageActionState)
    (result : TimelineResult)
    (frame : List StorageLocation)
    (observed : StorageLocation)
    (hEventFrame : ∀ event, event ∈ events →
      ∀ before after,
        TimelineEventExec semantics event before after →
        ∀ written, after.writes written → written ∈ frame)
    (hOutside : ∀ written ∈ frame, observed ≠ written)
    (hExec : TimelineExec semantics events initial result) :
    result.state.loop.storage observed = initial.loop.storage observed := by
  apply TimelineExec.preserves_outside semantics events initial result observed
    hExec
  intro hWritten
  have hMember := TimelineExec.writes_within semantics events initial result frame
    hEventFrame hExec observed hWritten
  exact hOutside observed hMember rfl

/-- Final local bindings are evaluated after the complete trace.  They may
read the final store and preceding read identities, but cannot replace the
store produced by that same trace. -/
def timelineProducesLoopState
    (semantics : TimelineSemantics FormulaExpr LocationExpr ValueExpr Child)
    (events : List (EventOf FormulaExpr LocationExpr ValueExpr Child))
    (evalFinalLocals : StorageActionState → Option (List Ora.Integer.Value))
    (initial : StorageActionState)
    (following : LoopState)
    (evalReturn : StorageActionState → Option (Option CallReturn) := fun _ => some none) : Prop :=
  ∃ result locals returned,
    TimelineExec semantics events initial result ∧
    evalFinalLocals result.state = some locals ∧
    evalReturn result.state = some returned ∧
    following = { locals, storage := result.state.loop.storage, returned }

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
  rcases hProduces with ⟨result, locals, returned, hExec, hLocals, hReturn, rfl⟩
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
    (targetAssertion : LoopState → Prop)
    (evalReturn : StorageActionState → Option (Option CallReturn) := fun _ => some none) :
    Edge LoopState where
  path := fun _ => True
  safe := fun current => TimelineSafe semantics events {
    loop := current, entryStorage := entryStorage
  }
  next := fun current following => timelineProducesLoopState semantics events
    evalFinalLocals { loop := current, entryStorage := entryStorage } following evalReturn
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
    rcases hProduces with ⟨result, locals, returned, hExec, hLocals, hReturn, rfl⟩
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

import Ora.Storage.Lifecycle
import Ora.Loop.CallAdapter

namespace Ora.StorageLifecycle
open Ora.Loop Ora.Obligation

theorem direct_empty_env (manifest : Manifest) (state : StorageActionState)
    (empty : state.loop.locals = []) :
    (directStorageContext manifest Env.empty []).stateEnv? state =
      some ((directStorageContext manifest Env.empty []).withStoreViews
        (bindStorageReads Env.empty state.reads) state) := by
  simp only [StorageFormulaContext.stateEnv?, directStorageContext, empty, bindState?,
    Option.pure_def, Option.bind_eq_bind, Option.some_bind]

/-- A selector is checked against the exact original event. Only live retained
    checks authenticated by the producer may occur in this list. -/
def successfulView (events : List (EventOf F L V C))
    (selected : List Nat) : Option (List (EventOf F L V C)) := do
  if !(decide selected.Nodup) || selected.any (fun i =>
      !(events[i]?.map TimelineEvent.isCheck).getD false) then none
  else events.enum.mapM fun (index, event) =>
    if index ∈ selected then
      match event with
      | .check formula => some (.assume formula)
      | _ => none
    else some event

/-- Every reached prefix, including one before a later false guard or a
    missing return. This is a predicate over the existing executor. -/
def PrefixesHold (claims : List Claim) (semantics : TimelineSemantics F L V C)
    (events : List (EventOf F L V C)) (state : StorageActionState) : Prop :=
  ∀ count result, TimelineExec semantics (events.take count) state result →
    Invariant claims result.state.loop.storage

theorem prefixes_nil (valid : Invariant claims state.loop.storage) :
    PrefixesHold claims semantics [] state := by
  intro count result trace
  simp only [List.take_nil] at trace
  cases trace
  exact valid

theorem prefixes_cons (valid : Invariant claims state.loop.storage)
    (tail : ∀ result, TimelineEventExec semantics event state result →
      PrefixesHold claims semantics rest result.state) :
    PrefixesHold claims semantics (event :: rest) state := by
  intro count result trace
  cases count with
  | zero => cases trace; exact valid
  | succ count =>
    cases trace with
    | @cons _ _ _ firstResult followingResult first following =>
      exact tail firstResult first count followingResult following

/-- Reads and control events frame every slot. Writes must establish their
    exact resolved non-aliasing relation, independently of later events. -/
def EventFrames (semantics : TimelineSemantics F L V C)
    (observed : StorageLocation) : EventOf F L V C → Prop
  | .assume _ | .check _ | .storage (.read _ _) => True
  | .storage (.write place _) => ∀ state location,
      semantics.storage.evalLocation place state = some location → observed ≠ location
  | _ => False

theorem event_frame (frame : EventFrames semantics observed event)
    (trace : TimelineEventExec semantics event state result) :
    result.state.loop.storage observed = state.loop.storage observed := by
  apply TimelineEventExec.preserves_outside _ _ _ _ _ trace
  cases trace with
  | assume | check => exact False.elim
  | @storage action _ _ execution =>
    cases action with
    | read id place =>
      have empty := execStorageRead_writeTrace_nil _ _ _ _ _ execution
      simp [empty]
    | write place value =>
      obtain ⟨location, evaluates, writes⟩ :=
        execStorageWrite_writeTrace_singleton _ _ _ _ _ execution
      simpa [writes] using frame _ _ evaluates
  | indexedRead | child | childResult | call => exact False.elim frame

theorem direct_location (context : StorageFormulaContext) (root : String)
    (ty : TyRef) (state : StorageActionState) (location : StorageLocation)
    (evaluates : context.evalLocation? { root, projections := [], valueTy := ty }
      state = some location) : location = { root } := by
  simp [StorageFormulaContext.evalLocation?, StorageFormulaContext.evalProjections?] at evaluates
  exact evaluates.2.2.symm

theorem prefixes_frame
    (frames : ∀ event ∈ events, EventFrames semantics observed event)
    (trace : TimelineExec semantics events state result) :
    result.state.loop.storage observed = state.loop.storage observed := by
  induction trace with
  | nil => rfl
  | cons first following ih =>
    exact (ih (fun event member => frames event (List.mem_cons_of_mem _ member))).trans
      (event_frame (frames _ (by simp)) first)

theorem framed_prefixes
    (frames : ∀ claim ∈ claims, ∀ event ∈ events,
      EventFrames semantics claim.location event)
    (valid : Invariant claims state.loop.storage) :
    PrefixesHold claims semantics events state := by
  intro count result trace claim member
  rw [prefixes_frame (fun event inPrefix =>
    frames claim member event (List.mem_of_mem_take inPrefix)) trace]
  exact valid claim member

theorem guarded_write_prefixes (semantics : TimelineSemantics F L V C)
    (formula : F)
    (valid : Invariant [claim] state.loop.storage)
    (guarded : semantics.evalFormula formula state = some (claim.holds value))
    (written : execStorageAction? semantics.storage (.write place expression) state = some after)
    (stored : after.state.loop.storage = state.loop.storage.set claim.location value)
    (frames : ∀ event ∈ rest, EventFrames semantics claim.location event) :
    PrefixesHold [claim] semantics (.assume formula :: .storage (.write place expression) :: rest) state := by
  apply prefixes_cons valid
  intro result execution
  cases execution with
  | assume evaluates holds =>
    rw [guarded] at evaluates
    have bounded : claim.holds value := Option.some.inj evaluates ▸ holds
    apply prefixes_cons valid
    intro result execution
    cases execution with
    | storage executes =>
      rw [written] at executes
      cases Option.some.inj executes
      apply framed_prefixes
      · intro observed member
        have equal : observed = claim := by simpa using member
        subst observed
        exact frames
      · rw [stored]
        apply checked_write_preserves _ _ _ _ valid
        intro observed member _
        have equal : observed = claim := by simpa using member
        subst observed
        exact bounded

structure Root (F L V C : Type) where
  semantics : TimelineSemantics F L V C
  events : List (EventOf F L V C)
  entry : StorageActionState → Prop

def Root.step (root : Root F L V C) : Transition := fun before after =>
  ∃ state count result, root.entry state ∧ state.loop.storage = before ∧
    TimelineExec root.semantics (root.events.take count) state result ∧
    result.state.loop.storage = after

def Root.Checked (claims : List Claim) (root : Root F L V C) : Prop :=
  ∀ state, root.entry state → Invariant claims state.loop.storage →
    PrefixesHold claims root.semantics root.events state

/-- All enumerated roots are checked, with a bootstrap witness. Actual HIR
    coverage remains a separate obligation, never inferred from this list. -/
theorem prefix_certificate (roots : List (Root F L V C))
    (witness : ∃ store, initial store)
    (bootstrap : ∀ store, initial store → Invariant claims store)
    (checked : ∀ root ∈ roots, root.Checked claims) :
    Certificate claims initial (roots.map Root.step) := by
  refine ⟨witness, bootstrap, ?_⟩
  intro step member before after valid execution
  obtain ⟨root, inRoots, rfl⟩ := List.mem_map.mp member
  obtain ⟨state, count, result, entry, rfl, trace, rfl⟩ := execution
  exact checked root inRoots state entry valid count result trace

end Ora.StorageLifecycle

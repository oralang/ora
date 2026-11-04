import Ora.Storage.LifecycleAdapter

/- Isolated sub-gate 1 model. Uses existing call-entry denotation and timeline
   execution; adds no source assumption, executor, solver premise or grant. -/
namespace Ora.StorageLifecycle
open Ora.Loop Ora.Obligation

def sourceEntry (manifest : Manifest) (row : FunctionSummaryRow)
    (state : StorageActionState) : Prop :=
  ∃ input, row.callSignature.entry? input = some state ∧
    row.entryFactsHold manifest input

def sourceRoot (manifest : Manifest) (row : FunctionSummaryRow)
    (semantics : TimelineSemantics F L V C) (events : List (EventOf F L V C)) :
    Root F L V C := { semantics, events, entry := sourceEntry manifest row }

/-- Same complete entry predicate as preservation; every retained checkpoint
    is demanded by the strict witness timeline. Compiler binding is separate. -/
def SourceWitness (manifest : Manifest) (row : FunctionSummaryRow)
    (claims : List Claim) (semantics : TimelineSemantics F L V C)
    (strictEvents : List (EventOf F L V C)) : Prop :=
  ∃ input state result, row.callSignature.entry? input = some state ∧
    row.entryFactsHold manifest input ∧ Invariant claims state.loop.storage ∧
    TimelineExec semantics strictEvents state result

variable {F L V C : Type} {manifest : Manifest} {row : FunctionSummaryRow}
variable {claims : List Claim} {state : StorageActionState} {actual : Transition}
variable {semantics : TimelineSemantics F L V C} {formula : F}
variable {events strictEvents rest : List (EventOf F L V C)}

theorem source_entry_typed (entered : sourceEntry manifest row state) :
    ∃ input, row.callSignature.inputTyped input = true ∧
      row.callSignature.entry? input = some state ∧ row.entryFactsHold manifest input := by
  obtain ⟨input, entry, facts⟩ := entered
  exact ⟨input, row.entryFactsHold_requires_inputTyped manifest input facts, entry, facts⟩

theorem source_root_checked
    (preserved : ∀ input state, row.callSignature.entry? input = some state →
      row.entryFactsHold manifest input → Invariant claims state.loop.storage →
      PrefixesHold claims semantics events state) :
    (sourceRoot manifest row semantics events).Checked claims := by
  rintro state ⟨input, entry, facts⟩ valid
  exact preserved input state entry facts valid

theorem source_witness_has_complete_entry
    (witness : SourceWitness manifest row claims semantics strictEvents) :
    ∃ state, sourceEntry manifest row state ∧ Invariant claims state.loop.storage := by
  obtain ⟨input, state, _, entry, facts, valid, _⟩ := witness
  exact ⟨state, ⟨input, entry, facts⟩, valid⟩

theorem malformed_entry_cannot_witness
    (invalid : row.entryFormulas? manifest = none) :
    ¬ SourceWitness manifest row claims semantics strictEvents := by
  rintro ⟨input, _, _, _, facts, _⟩
  simp [FunctionSummaryRow.entryFactsHold, invalid] at facts

theorem contradictory_entry_cannot_witness
    (impossible : ∀ input, ¬ row.entryFactsHold manifest input) :
    ¬ SourceWitness manifest row claims semantics strictEvents := by
  rintro ⟨input, _, _, _, facts, _⟩
  exact impossible input facts

theorem source_witness_requires_strict_trace
    (impossible : ∀ input state result, row.callSignature.entry? input = some state →
      row.entryFactsHold manifest input → Invariant claims state.loop.storage →
      ¬ TimelineExec semantics strictEvents state result) :
    ¬ SourceWitness manifest row claims semantics strictEvents := by
  rintro ⟨input, state, result, entry, facts, valid, trace⟩
  exact impossible input state result entry facts valid trace

/-- Universal preservation still quantifies over every actual invocation.
    A witness for a chosen argument never supplies this coverage premise. -/
theorem source_step_covered
    (authenticated : ∀ before after, actual before after →
      ∃ input state count result, row.callSignature.entry? input = some state ∧
        row.entryFactsHold manifest input ∧ state.loop.storage = before ∧
        TimelineExec semantics (events.take count) state result ∧
        result.state.loop.storage = after) :
    Covers actual [(sourceRoot manifest row semantics events).step] := by
  intro before after executed
  obtain ⟨input, state, count, result, entry, facts, start, trace, finish⟩ :=
    authenticated before after executed
  exact ⟨(sourceRoot manifest row semantics events).step, by simp,
    state, count, result, ⟨input, entry, facts⟩, start, trace, finish⟩

/-- Source assertions between a retained guard and its write do not mutate
    the state. Safety is still an independent FunctionVC obligation. -/
theorem check_prefixes
    (valid : Invariant claims state.loop.storage)
    (tail : PrefixesHold claims semantics rest state) :
    PrefixesHold claims semantics (.check formula :: rest) state := by
  apply prefixes_cons valid
  intro result execution
  cases execution
  exact tail

theorem checks_prefixes (checks : List F)
    (valid : Invariant claims state.loop.storage)
    (tail : PrefixesHold claims semantics rest state) :
    PrefixesHold claims semantics (checks.map TimelineEvent.check ++ rest) state := by
  induction checks with
  | nil => exact tail
  | cons formula checks ih => exact check_prefixes valid ih

theorem guarded_write_after_checks (semantics : TimelineSemantics F L V C)
    (formula : F) (checks : List F)
    (valid : Invariant [claim] state.loop.storage)
    (guarded : semantics.evalFormula formula state = some (claim.holds value))
    (written : execStorageAction? semantics.storage (.write place expression) state = some after)
    (stored : after.state.loop.storage = state.loop.storage.set claim.location value)
    (frames : ∀ event ∈ rest, EventFrames semantics claim.location event) :
    PrefixesHold [claim] semantics
      (.assume formula :: checks.map TimelineEvent.check ++ .storage (.write place expression) :: rest) state := by
  apply prefixes_cons valid
  intro result execution
  cases execution with
  | assume evaluates holds =>
      have bounded : claim.holds value := Option.some.inj (guarded.symm.trans evaluates) ▸ holds
      apply checks_prefixes checks valid
      apply prefixes_cons valid
      intro result execution
      cases execution with
      | storage executes =>
          rw [written] at executes
          cases Option.some.inj executes
          apply framed_prefixes (fun observed member => by
            have same : observed = claim := by simpa using member
            subst observed
            exact frames)
          rw [stored]
          apply checked_write_preserves _ _ _ _ valid
          intro observed member _
          have same : observed = claim := by simpa using member
          subst observed
          exact bounded

end Ora.StorageLifecycle

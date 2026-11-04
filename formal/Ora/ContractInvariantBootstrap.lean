import Ora.Storage.Lifecycle

/- Diagnosis model only. Writer relations stand for actual executions, including
   observable prefixes when those can expose storage. No compiler authority. -/
namespace Ora.InvariantBootstrap

abbrev Writer (State : Type) := State → State → Prop

inductive EntryReachable (initial : State → Prop) (writers : List (Writer State)) : State → Prop
  | initial {state} : initial state → EntryReachable initial writers state
  | write {before after writer} : EntryReachable initial writers before →
      writer ∈ writers → writer before after → EntryReachable initial writers after

def EntrySound (initial invariant : State → Prop) (writers : List (Writer State)) : Prop :=
  ∀ state, EntryReachable initial writers state → invariant state

def Bootstrap (initial invariant : State → Prop) : Prop :=
  ∀ state, initial state → invariant state

def AllWritersPreserve (invariant : State → Prop) (writers : List (Writer State)) : Prop :=
  ∀ writer ∈ writers, ∀ before after, invariant before → writer before after → invariant after

def ReachedWritersPreserve (initial invariant : State → Prop) (writers : List (Writer State)) : Prop :=
  ∀ writer ∈ writers, ∀ before after, EntryReachable initial writers before →
    invariant before → writer before after → invariant after

/-- Exact iff: preservation ranges over every actual writer at reachable entry.
    Preservation on ALL invariant states is the stronger sufficient rule below. -/
theorem entry_sound_iff (initial invariant : State → Prop) (writers : List (Writer State)) :
    EntrySound initial invariant writers ↔
      Bootstrap initial invariant ∧ ReachedWritersPreserve initial invariant writers := by
  constructor
  · intro sound
    constructor
    · intro state start
      exact sound state (.initial start)
    · intro writer member before after reached _ executed
      exact sound after (.write reached member executed)
  · rintro ⟨base, preserve⟩ state reached
    induction reached with
    | initial start => exact base _ start
    | write reached member executed ih => exact preserve _ member _ _ reached ih executed

theorem inductive_rule_sound (base : Bootstrap initial invariant)
    (preserve : AllWritersPreserve invariant writers) : EntrySound initial invariant writers := by
  apply (entry_sound_iff initial invariant writers).mpr
  exact ⟨base, fun writer member before after _ good step => preserve writer member before after good step⟩

theorem missing_bootstrap_forbids_entry_assumption (bad : initial state ∧ ¬ invariant state) :
    ¬ EntrySound initial invariant writers := by
  intro sound
  exact bad.2 (sound state (.initial bad.1))

theorem bad_reached_writer_forbids_entry_assumption
    (reached : EntryReachable initial writers before) (member : writer ∈ writers)
    (executed : writer before after) (bad : ¬ invariant after) :
    ¬ EntrySound initial invariant writers := by
  intro sound
  exact bad (sound after (.write reached member executed))

def zeroInitial (state : Nat) : Prop := state = 0
def positive (state : Nat) : Prop := 1 ≤ state
def unchanged (before after : Nat) : Prop := after = before

theorem unchanged_preserves_positive : AllWritersPreserve positive [unchanged] := by
  intro writer member before after good step
  have same : writer = unchanged := by simpa using member
  subst writer
  change after = before at step
  exact step.symm ▸ good

theorem positive_has_witness : ∃ state, positive state := ⟨1, by unfold positive; decide⟩

/-- Nonempty actual initialization and a perfectly preserving writer do not
    justify the assumed positive entry fact: the missing base is decisive. -/
theorem assume_without_base_case :
    (∃ state, zeroInitial state) ∧ (∃ state, positive state) ∧
    AllWritersPreserve positive [unchanged] ∧ ¬ EntrySound zeroInitial positive [unchanged] := by
  refine ⟨⟨0, rfl⟩, positive_has_witness, unchanged_preserves_positive, ?_⟩
  exact missing_bootstrap_forbids_entry_assumption (state := 0) ⟨rfl, by unfold positive; decide⟩

theorem explicit_one_bootstrap_control :
    EntrySound (fun state : Nat => state = 1) positive [unchanged] := by
  apply inductive_rule_sound _ unchanged_preserves_positive
  intro state one
  subst state
  unfold positive
  decide

theorem omitted_writer_is_not_covered :
    ¬ EntrySound (fun state : Nat => state = 1) positive [unchanged, fun _ after => after = 0] := by
  apply bad_reached_writer_forbids_entry_assumption
    (before := 1) (after := 0) (writer := fun (_ : Nat) after => after = 0)
  · exact .initial rfl
  · simp
  · rfl
  · unfold positive; decide

/-- A declaration's claimed initializer is not the actual deployment relation. -/
theorem claimed_initializer_cannot_replace_actual :
    Bootstrap (fun state : Nat => state = 1) positive ∧ ¬ Bootstrap zeroInitial positive := by
  constructor
  · intro state one; subst state; unfold positive; decide
  · intro base; exact (by unfold positive; decide : ¬ positive 0) (base 0 rfl)

end Ora.InvariantBootstrap

/-
Ora dispatcher abstract model.

This file proves generic facts about selector dispatch models. A checked indexed
dispatcher guards each jump with `record.selector == sel`; that guard is sound
(`run_sound`), known selectors resolve when stored at their own index
(`run_complete`), and an unknown selector aliasing a known one is mis-dispatched
without the guard (`alias_misdispatched_without_check`). Linear and sparse
models are likewise proved sound because they scan by exact selector equality.

This file does NOT, by itself, prove that every Ora source-level dispatcher
lowering emits a table satisfying those premises. Representative compiled Ora
fixtures are checked separately in `Ora/DispatcherTableSync.lean`, where rows
extracted after `oraBuildSIRDispatcher` are decoded and proved to have
known-selector coverage, dense route injectivity, sparse exact-scan behavior,
and exact-selector guard presence.

Scope: the existence claim — that compression (`|Idx| < |Sel|`) forces an alias
— is a pigeonhole result, not proved here. Lean can express it (`Fin`/finite
types); this project's mathlib-free layer just doesn't yet include the
finite-cardinality machinery, so it is deferred rather than stubbed.
Axiom-light (`propext`).
-/

namespace Ora.Dispatcher

variable {Sel Idx Label : Type} [DecidableEq Sel]

/-- An indexed dispatcher: an index function and a record table (selector + jump label). -/
structure Dispatcher (Sel Idx Label : Type) where
  index : Sel → Idx
  table : Idx → Option (Sel × Label)

/-- Dispatch with the selector-equality guard (`if record.selector == sel then …`). -/
def Dispatcher.run (d : Dispatcher Sel Idx Label) (sel : Sel) : Option Label :=
  match d.table (d.index sel) with
  | some (s, l) => if s = sel then some l else none
  | none        => none

/-- Dispatch without the guard — jump on an index hit alone. -/
def Dispatcher.runNoCheck (d : Dispatcher Sel Idx Label) (sel : Sel) : Option Label :=
  (d.table (d.index sel)).map Prod.snd

/-! ## §2 collision classes (`Σ` modeled by `known : Sel → Prop`) -/

/-- Two distinct known selectors sharing an index — compile-time; dense dispatch forbids it. -/
def KnownCollision (H : Sel → Idx) (known : Sel → Prop) (s₁ s₂ : Sel) : Prop :=
  known s₁ ∧ known s₂ ∧ s₁ ≠ s₂ ∧ H s₁ = H s₂

/-- An unknown selector sharing a known selector's index — a runtime input fact. -/
def UnknownAlias (H : Sel → Idx) (known : Sel → Prop) (s sf : Sel) : Prop :=
  ¬ known s ∧ known sf ∧ H s = H sf

/-- §5 dense premise: forbidding known collisions is exactly index-injectivity over `Σ`. -/
theorem noKnownCollision_iff_injOn (H : Sel → Idx) (known : Sel → Prop) :
    (∀ s₁ s₂, ¬ KnownCollision H known s₁ s₂) ↔
      (∀ s₁ s₂, known s₁ → known s₂ → H s₁ = H s₂ → s₁ = s₂) := by
  constructor
  · intro hno s₁ s₂ h1 h2 heq
    by_cases hne : s₁ = s₂
    · exact hne
    · exact absurd ⟨h1, h2, hne, heq⟩ (hno s₁ s₂)
  · intro hinj s₁ s₂ hkc
    obtain ⟨h1, h2, hne, heq⟩ := hkc
    exact hne (hinj s₁ s₂ h1 h2 heq)

/-! ## Soundness of the guarded path -/

/-- A returned label's record is keyed by exactly the input selector. -/
theorem run_sound (d : Dispatcher Sel Idx Label) {sel : Sel} {l : Label}
    (h : d.run sel = some l) : d.table (d.index sel) = some (sel, l) := by
  rcases hrec : d.table (d.index sel) with _ | ⟨s, lbl⟩
  · simp [Dispatcher.run, hrec] at h
  · by_cases hs : s = sel
    · have hl : lbl = l := by simpa [Dispatcher.run, hrec, hs] using h
      rw [hs, hl]
    · simp [Dispatcher.run, hrec, hs] at h

/-- A selector stored at its own index resolves to that record's label. -/
theorem run_complete (d : Dispatcher Sel Idx Label) {sel : Sel} {l : Label}
    (h : d.table (d.index sel) = some (sel, l)) : d.run sel = some l := by
  simp [Dispatcher.run, h]

/-! ## Dense table construction — tying completeness to the compiler-built table -/

/-- Well-formedness of a compiler-generated dense table for the known set `Σ` and label
    assignment `labelOf`: every known selector is stored at its own index (`covered`), and
    the index is injective over `Σ` (`injOn` — no known collision, §5). -/
structure DenseWF (d : Dispatcher Sel Idx Label) (known : Sel → Prop)
    (labelOf : Sel → Label) : Prop where
  covered : ∀ s, known s → d.table (d.index s) = some (s, labelOf s)
  injOn   : ∀ s₁ s₂, known s₁ → known s₂ → d.index s₁ = d.index s₂ → s₁ = s₂

/-- For a well-formed dense table, every known selector dispatches to its own label —
    completeness is now tied to the construction, not an arbitrary table premise. -/
theorem DenseWF.run_known {d : Dispatcher Sel Idx Label} {known : Sel → Prop}
    {labelOf : Sel → Label} (wf : DenseWF d known labelOf) {s : Sel} (hs : known s) :
    d.run s = some (labelOf s) :=
  run_complete d (wf.covered s hs)

/-- A well-formed dense table admits no known collision (§5 dense admissibility). -/
theorem DenseWF.noKnownCollision {d : Dispatcher Sel Idx Label} {known : Sel → Prop}
    {labelOf : Sel → Label} (wf : DenseWF d known labelOf) :
    ∀ s₁ s₂, ¬ KnownCollision d.index known s₁ s₂ :=
  (noKnownCollision_iff_injOn d.index known).mpr wf.injOn

/-- The guard only removes hits: it refines the no-guard path. -/
theorem run_le_runNoCheck (d : Dispatcher Sel Idx Label) {sel : Sel} {l : Label}
    (h : d.run sel = some l) : d.runNoCheck sel = some l := by
  simp [Dispatcher.runNoCheck, run_sound d h]

/-! ## §6 necessity: an unknown alias breaks the no-guard path -/

/-- An unknown selector `s` aliasing a known `sf` (whose slot stores `sf`) is mis-dispatched
    to `sf`'s label without the guard, but correctly rejected (`none`) with it. -/
theorem alias_misdispatched_without_check
    (d : Dispatcher Sel Idx Label) {known : Sel → Prop} {s sf : Sel} {l : Label}
    (halias : UnknownAlias d.index known s sf)
    (hslot : d.table (d.index sf) = some (sf, l)) :
    d.runNoCheck s = some l ∧ d.run s = none := by
  obtain ⟨hns, hksf, hH⟩ := halias
  have hne : sf ≠ s := fun heq => hns (heq ▸ hksf)
  exact ⟨by simp [Dispatcher.runNoCheck, hH, hslot], by simp [Dispatcher.run, hH, hslot, hne]⟩

/-- The dual of `DenseWF.run_known`: a well-formed dense table also REJECTS unknown aliases.
    An unknown `s` aliasing a known `sf` is mis-dispatched to `sf`'s label without the guard
    but returns `none` with it — so a WF dense table dispatches knowns correctly *and*
    rejects unknown aliases (the construction-level §6 necessity). -/
theorem DenseWF.alias_rejected {d : Dispatcher Sel Idx Label} {known : Sel → Prop}
    {labelOf : Sel → Label} (wf : DenseWF d known labelOf) {s sf : Sel}
    (halias : UnknownAlias d.index known s sf) :
    d.runNoCheck s = some (labelOf sf) ∧ d.run s = none :=
  alias_misdispatched_without_check d halias (wf.covered sf halias.2.1)

/-- Concrete instance: the no-guard path accepts an alias the guard rejects. -/
theorem runNoCheck_unsound :
    ∃ (d : Dispatcher Bool Unit Nat) (s sf : Bool),
      s ≠ sf ∧ d.runNoCheck s = some 7 ∧ d.run s = none :=
  ⟨⟨fun _ => (), fun _ => some (true, 7)⟩, false, true, by decide, rfl, rfl⟩

/-! ## Linear dispatch — the guard is both lookup and validation (§3) -/

/-- Linear dispatch: scan for the first exact selector match. -/
def linearRun (tbl : List (Sel × Label)) (sel : Sel) : Option Label :=
  (tbl.find? (fun r => r.1 = sel)).map Prod.snd

/-- A linear-dispatch result is keyed by the exact input selector. -/
theorem linearRun_sound (tbl : List (Sel × Label)) {sel : Sel} {l : Label}
    (h : linearRun tbl sel = some l) : (sel, l) ∈ tbl := by
  unfold linearRun at h
  rcases hf : tbl.find? (fun r => r.1 = sel) with _ | rec
  · simp [hf] at h
  · simp only [hf, Option.map_some, Option.some.injEq] at h
    have hmem := List.mem_of_find?_eq_some hf
    have hp : rec.1 = sel := by simpa using List.find?_some hf
    have hrec : rec = (sel, l) := by cases rec; simp_all
    rw [hrec] at hmem; exact hmem

/-! ## Sparse / bucketed dispatch (§4) -/

/-- A sparse dispatcher: an index selects a bucket; the bucket is scanned by exact selector
    equality. Known-selector collisions are allowed — colliding known selectors share a
    bucket (that is why buckets exist). -/
structure SparseDispatcher (Sel Idx Label : Type) where
  index   : Sel → Idx
  buckets : Idx → List (Sel × Label)

/-- Sparse dispatch: pick the bucket, then scan it for an exact selector match. -/
def SparseDispatcher.run (d : SparseDispatcher Sel Idx Label) (sel : Sel) : Option Label :=
  linearRun (d.buckets (d.index sel)) sel

/-- The degenerate no-scan path: take the bucket's first record without the equality check.
    Real sparse dispatch always scans; this models dropping that scan (`runNoCheck_unsound`
    shows it is wrong even for a singleton bucket). -/
def SparseDispatcher.runNoCheck (d : SparseDispatcher Sel Idx Label) (sel : Sel) : Option Label :=
  (d.buckets (d.index sel)).head?.map Prod.snd

/-- SOUNDNESS (§4): a sparse result is keyed by exactly the input selector and lives in the
    selected bucket — the bucket only narrows the candidates; the equality scan identifies. -/
theorem SparseDispatcher.run_sound (d : SparseDispatcher Sel Idx Label) {sel : Sel} {l : Label}
    (h : d.run sel = some l) : (sel, l) ∈ d.buckets (d.index sel) :=
  linearRun_sound _ h

/-- The dual of `run_sound`: a selector keyed by no bucket entry is REJECTED (`none`) — the
    scan rejects unknown selectors that index into an occupied bucket. -/
theorem SparseDispatcher.run_none_of_not_mem (d : SparseDispatcher Sel Idx Label) {sel : Sel}
    (h : ∀ l, (sel, l) ∉ d.buckets (d.index sel)) : d.run sel = none := by
  rcases hr : d.run sel with _ | l
  · rfl
  · exact absurd (run_sound d hr) (h l)

/-- §4: known-selector collisions are ALLOWED — two known selectors share a bucket and the
    equality scan still resolves each to its own label. -/
example :
    let d : SparseDispatcher Bool Unit Nat := ⟨fun _ => (), fun _ => [(true, 1), (false, 2)]⟩
    d.run true = some 1 ∧ d.run false = some 2 := by decide

/-- §4: even a SINGLETON bucket needs the equality check — without it an unknown selector
    indexing into an occupied bucket is mis-dispatched to that bucket's first record. -/
theorem SparseDispatcher.runNoCheck_unsound :
    ∃ (d : SparseDispatcher Bool Unit Nat) (s sf : Bool),
      s ≠ sf ∧ d.runNoCheck s = some 7 ∧ d.run s = none :=
  ⟨⟨fun _ => (), fun _ => [(true, 7)]⟩, false, true, by decide, rfl, rfl⟩

/-! ## Dispatcher builder correctness

These theorems separate construction correctness from planning optimality.  A
planner may choose linear, sparse, or dense routing; the builder theorem says
that, once a plan is chosen, the emitted abstract dispatcher satisfies the
corresponding model premise.  Dense routing is intentionally guarded by an
admissibility premise: a caller must prove the dense index is injective over the
known selector set.  Without that premise, the statement is false.
-/

/-- Function selector: 4-byte keccak256 hash, modeled as Nat. -/
abbrev Selector := Nat

/-- The strategy shape selected by the planner. Dense and sparse plans carry only the
    routing index here; cost fields belong to the planner/optimality layer, not the
    builder-correctness theorem. -/
inductive BuilderPlan (Idx : Type) where
  | linear
  | dense (index : Selector → Idx)
  | sparse (index : Selector → Idx)

/-- Linear builder postcondition: any returned label came from the exact selector row. -/
def LinearWF (cases : List (Selector × Label)) : Prop :=
  ∀ {s : Selector} {l : Label}, linearRun cases s = some l → (s, l) ∈ cases

/-- The selector set represented by a concrete case list. -/
def casesKnown (cases : List (Selector × Label)) (selector : Selector) : Prop :=
  ∃ label, (selector, label) ∈ cases

/-- Dense table construction: the first case routed to an index owns that slot. -/
def buildDenseTable [DecidableEq Idx]
    (cases : List (Selector × Label)) (index : Selector → Idx) :
    Idx → Option (Selector × Label) :=
  fun idx => cases.find? (fun r => decide (index r.1 = idx))

/-- Dense builder: one abstract slot per route index.  The concrete compiler table is
    checked against this model from emitted rows; this construction theorem proves the
    model itself is well-formed when the chosen dense index is admissible. -/
def buildDenseDispatcher [DecidableEq Idx]
    (cases : List (Selector × Label)) (index : Selector → Idx) :
    Dispatcher Selector Idx Label :=
  { index := index,
    table := buildDenseTable cases index }

/-- The label assignment induced by the dense builder. It is only used under `casesKnown`,
    where `buildDenseDispatcher_wf` proves the table contains the selector's slot. -/
def denseBuiltLabelOf [DecidableEq Idx] [Inhabited Label]
    (cases : List (Selector × Label)) (index : Selector → Idx) : Selector → Label :=
  fun selector =>
    match buildDenseTable cases index (index selector) with
    | some (_, label) => label
    | none => default

/-- Sparse builder: route to a bucket, then scan that bucket by exact selector equality. -/
def buildSparseDispatcher [DecidableEq Idx]
    (cases : List (Selector × Label)) (index : Selector → Idx) :
    SparseDispatcher Selector Idx Label :=
  { index := index,
    buckets := fun idx => cases.filter (fun r => index r.1 = idx) }

/-- Sparse builder postcondition: a result must be present in the selected bucket. -/
def SparseWF [DecidableEq Idx]
    (cases : List (Selector × Label)) (index : Selector → Idx) : Prop :=
  let d := buildSparseDispatcher cases index
  ∀ {s : Selector} {l : Label}, d.run s = some l → (s, l) ∈ d.buckets (d.index s)

/-- The admissibility condition a planner must establish before handing a plan to the
    builder. Linear and sparse dispatch scan by exact equality; dense dispatch needs
    collision freedom over the known selector set. Returns Bool so it can be used
    with `List.find?`. -/
def PlanAdmissible [DecidableEq Idx] (cases : List (Selector × Label)) : BuilderPlan Idx → Bool
  | .linear => true
  | .dense index =>
      cases.all (fun (s₁, _) =>
        cases.all (fun (s₂, _) =>
          decide (index s₁ = index s₂ → s₁ = s₂)))
  | .sparse _ => true

/-- Extract the injectivity condition from a dense `PlanAdmissible` result.
    Since `PlanAdmissible` checks all pairs in `cases`, we can convert it to
    a `∀` statement over known selectors. -/
theorem PlanAdmissible_dense_inj [DecidableEq Idx]
    (cases : List (Selector × Label)) (index : Selector → Idx)
    (h : PlanAdmissible cases (.dense index) = true) :
    ∀ s₁ s₂,
      casesKnown cases s₁ → casesKnown cases s₂ →
        index s₁ = index s₂ → s₁ = s₂ := by
  simp [PlanAdmissible] at h
  intro s₁ s₂ h1 h2 heq
  rcases h1 with ⟨l₁, hmem₁⟩
  rcases h2 with ⟨l₂, hmem₂⟩
  have hcheck := h s₁ l₁ hmem₁ s₂ l₂ hmem₂
  rcases hcheck with (hne | heq')
  · exact (hne heq).elim
  · exact heq'

/-- Preconditions for dispatcher planning: at least 4 cases and a default target. -/
def preconditionsMet (cases : List (Selector × Label)) (hasDefault : Bool) : Bool :=
  cases.length ≥ 4 && hasDefault

/-- A small strategy-agnostic planner used to prove the generic composition
    pattern. The production scoring planner is specified separately in
    `Ora.DispatcherPlannerSpec` and implemented by `Ora.SinoraPlanner`. -/
def choosePlan [DecidableEq Idx] (cases : List (Selector × Label)) (hasDefault : Bool)
    (denseCandidates : List (Selector → Idx)) (sparseCandidates : List (Selector → Idx)) :
    BuilderPlan Idx :=
  if preconditionsMet cases hasDefault then
    match denseCandidates.find? (fun index => PlanAdmissible cases (.dense index)) with
    | some index => .dense index
    | none =>
        match sparseCandidates.find? (fun index => PlanAdmissible cases (.sparse index)) with
        | some index => .sparse index
        | none => .linear
  else
    .linear

/-- The planner returns an admissible plan.  `find?` only selects elements
    that satisfy the predicate, so the returned plan is admissible by construction. -/
theorem choosePlan_admissible [DecidableEq Idx]
    (cases : List (Selector × Label)) (hasDefault : Bool)
    (denseCandidates : List (Selector → Idx)) (sparseCandidates : List (Selector → Idx)) :
    PlanAdmissible cases (choosePlan cases hasDefault denseCandidates sparseCandidates) = true := by
  unfold choosePlan
  split
  · cases hd : denseCandidates.find? (fun index => PlanAdmissible cases (.dense index)) with
    | none =>
        cases hs : sparseCandidates.find? (fun index => PlanAdmissible cases (.sparse index)) with
        | none => rfl
        | some _ => simp [hd, hs, PlanAdmissible]
    | some di =>
        have hadm := List.find?_some hd
        simp [hd, hadm]
  · rfl

/-- Strategy-specific builder postcondition. -/
def StrategyWF [DecidableEq Idx] [Inhabited Label]
    (cases : List (Selector × Label)) :
    BuilderPlan Idx → Prop
  | .linear => LinearWF cases
  | .dense index =>
      DenseWF (buildDenseDispatcher cases index) (casesKnown cases) (denseBuiltLabelOf cases index)
  | .sparse index => SparseWF cases index

/-- Dense builder correctness: injecting known selectors into route slots is exactly the
    `DenseWF` premise consumed by the dispatcher theorems above. -/
theorem buildDenseDispatcher_wf [DecidableEq Idx] [Inhabited Label]
    (cases : List (Selector × Label)) (index : Selector → Idx)
    (hinj : ∀ s₁ s₂, casesKnown cases s₁ → casesKnown cases s₂ →
      index s₁ = index s₂ → s₁ = s₂) :
    DenseWF
      (buildDenseDispatcher cases index) (casesKnown cases) (denseBuiltLabelOf cases index) := by
  constructor
  · intro s hs
    rcases hs with ⟨label, hmem⟩
    unfold buildDenseDispatcher buildDenseTable denseBuiltLabelOf
    rcases hfind : cases.find? (fun r => decide (index r.1 = index s)) with _ | rec
    · have hnone := (List.find?_eq_none).mp hfind (s, label) hmem
      have hpred :
          (fun r : Selector × Label => decide (index r.1 = index s)) (s, label) = true := by
        simp
      exact False.elim (hnone hpred)
    · have hmem_rec := List.mem_of_find?_eq_some hfind
      have hpred_rec := List.find?_some hfind
      have hindex : index rec.1 = index s := of_decide_eq_true hpred_rec
      have hknown_rec : casesKnown cases rec.1 := by
        cases rec with
        | mk selector rec_label =>
            exact ⟨rec_label, hmem_rec⟩
      have hsel : rec.1 = s := hinj rec.1 s hknown_rec ⟨label, hmem⟩ hindex
      cases rec with
      | mk selector rec_label =>
          have hsel' : selector = s := by simpa using hsel
          simp [buildDenseTable, hfind, hsel']
  · exact hinj

/-- Builder correctness for every strategy shape. This is intentionally not an
    optimality theorem: it proves that an admissible chosen plan is built into the
    right dispatcher model, not that the planner chose the cheapest admissible plan. -/
theorem builder_correct [DecidableEq Idx] [Inhabited Label]
    (cases : List (Selector × Label)) (plan : BuilderPlan Idx)
    (hadm : PlanAdmissible cases plan = true) :
    StrategyWF cases plan := by
  cases plan with
  | linear =>
      intro s l h
      exact linearRun_sound cases h
  | dense index =>
      have hinj := PlanAdmissible_dense_inj cases index hadm
      exact buildDenseDispatcher_wf cases index hinj
  | sparse index =>
      intro s l h
      exact SparseDispatcher.run_sound (buildSparseDispatcher cases index) h

/-- Planner + builder correctness for all selector sets.  Given any list of
    known selectors, any default flag, and any lists of dense/sparse candidate
    index functions, the planner's chosen plan is admissible and the builder
    produces a dispatcher that satisfies the strategy postcondition.

    This is the universal correctness theorem: for every possible input selector
    set the planner could receive, the planner never returns an inadmissible plan,
    and the builder never corrupts that admissibility into an incorrect dispatcher.

    This does NOT prove that the Zig planner's candidate generation is exhaustive
    or that the emitted SIR table matches the model — those are translation-level
    checks handled by the per-contract sync gate.  What this removes is the
    per-contract assumption that `DenseWF` (or `LinearWF` / `SparseWF`) holds
    as a hypothesis rather than a derived fact. -/
theorem planner_builder_correct [DecidableEq Idx] [Inhabited Label]
    (cases : List (Selector × Label)) (hasDefault : Bool)
    (denseCandidates : List (Selector → Idx)) (sparseCandidates : List (Selector → Idx)) :
    StrategyWF cases (choosePlan cases hasDefault denseCandidates sparseCandidates) :=
  builder_correct cases (choosePlan cases hasDefault denseCandidates sparseCandidates)
    (choosePlan_admissible cases hasDefault denseCandidates sparseCandidates)

end Ora.Dispatcher

/-
Ora dispatcher: soundness and necessity of the runtime selector-equality check

A checked indexed dispatcher guards each jump with `record.selector == sel`. The guard
is sound (`run_sound`), known selectors resolve (`run_complete`), and it is necessary:
an unknown selector aliasing a known one is mis-dispatched without it
(`alias_misdispatched_without_check`). The collision classes are `KnownCollision`
(compile-time; forbidden by dense dispatch — `noKnownCollision_iff_injOn`) and
`UnknownAlias` (runtime; handled by the guard).

Scope: the existence claim — that compression (`|Idx| < |Sel|`) forces an alias — is a
pigeonhole result, not proved here. Lean can express it (`Fin`/finite types); this
project's mathlib-free layer just doesn't yet include the finite-cardinality machinery,
so it's deferred to a later layer rather than stubbed. Axiom-light (`propext`).
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

end Ora.Dispatcher

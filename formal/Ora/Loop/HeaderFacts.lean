/-
Canonical facts that hold before a loop header.

This module models only the proof-facing data.  A later compiler projection
must show that each row comes from a semantic assumption in the same canonical
block and strictly before the target loop.  Fact execution reuses the existing
timeline relation; there is no second preheader executor.
-/

import Ora.Loop.Timeline

namespace Ora.Loop

inductive HeaderFactSource where
  | pathAssume
  | functionRequires
  | unresolved
  deriving Repr, BEq, DecidableEq

structure HeaderPoint where
  block : String
  position : Nat
  deriving Repr, BEq, DecidableEq

/-- The first reviewed dominance boundary: an earlier operation in the exact
same enclosing block executes on every execution that reaches the later loop
operation. -/
def HeaderPoint.dominates (source target : HeaderPoint) : Prop :=
  source.block = target.block ∧ source.position < target.position

structure HeaderFactRow (FormulaExpr : Type) where
  id : Nat
  structuralKey : String
  formula : FormulaExpr
  sourceKind : HeaderFactSource
  source : HeaderPoint
  target : HeaderPoint

inductive HeaderFactShapeReason where
  | sourceKindUnresolved
  | sourceBlockMismatch
  | sourceNotEarlier
  deriving Repr, BEq, DecidableEq

def HeaderFactRow.shapeReason?
    (row : HeaderFactRow FormulaExpr) : Option HeaderFactShapeReason :=
  if row.sourceKind = .unresolved then
    some .sourceKindUnresolved
  else if row.source.block != row.target.block then
    some .sourceBlockMismatch
  else if row.source.position < row.target.position then
    none
  else
    some .sourceNotEarlier

def HeaderFactRow.ShapeSupported (row : HeaderFactRow FormulaExpr) : Prop :=
  row.shapeReason? = none

def HeaderFactRow.shapeSupported (row : HeaderFactRow FormulaExpr) : Bool :=
  row.shapeReason?.isNone

theorem HeaderFactRow.shapeSupported_iff
    (row : HeaderFactRow FormulaExpr) :
    row.ShapeSupported ↔
      row.sourceKind ≠ .unresolved ∧ row.source.dominates row.target := by
  by_cases hKind : row.sourceKind = .unresolved
  · simp [HeaderFactRow.ShapeSupported, HeaderFactRow.shapeReason?,
      HeaderPoint.dominates, hKind]
  · by_cases hBlock : row.source.block = row.target.block
    · simp [HeaderFactRow.ShapeSupported, HeaderFactRow.shapeReason?,
        HeaderPoint.dominates, hKind, hBlock]
    · simp [HeaderFactRow.ShapeSupported, HeaderFactRow.shapeReason?,
        HeaderPoint.dominates, hKind, hBlock]

theorem HeaderFactRow.shapeSupported_eq_true_iff
    (row : HeaderFactRow FormulaExpr) :
    row.shapeSupported = true ↔ row.ShapeSupported := by
  simp [HeaderFactRow.shapeSupported, HeaderFactRow.ShapeSupported]

def headerFactFormulas (rows : List (HeaderFactRow FormulaExpr)) :
    List FormulaExpr :=
  rows.map HeaderFactRow.formula

def HeaderFactsHold
    (semantics : TimelineSemantics FormulaExpr LocationExpr ValueExpr Child)
    (rows : List (HeaderFactRow FormulaExpr))
    (state : StorageActionState) : Prop :=
  ∀ row ∈ rows, timelineFormulaHolds semantics row.formula state

def HeaderFactsStable
    (semantics : TimelineSemantics FormulaExpr LocationExpr ValueExpr Child)
    (rows : List (HeaderFactRow FormulaExpr)) : Prop :=
  ∀ row ∈ rows, ∀ before after,
    timelineFormulaHolds semantics row.formula before ↔
      timelineFormulaHolds semantics row.formula after

theorem headerFactsHold_append_iff
    (semantics : TimelineSemantics FormulaExpr LocationExpr ValueExpr Child)
    (first following : List (HeaderFactRow FormulaExpr))
    (state : StorageActionState) :
    HeaderFactsHold semantics (first ++ following) state ↔
      HeaderFactsHold semantics first state ∧
        HeaderFactsHold semantics following state := by
  constructor
  · intro hHold
    exact ⟨fun row h => hHold row (List.mem_append_left following h),
      fun row h => hHold row (List.mem_append_right first h)⟩
  · rintro ⟨hFirst, hFollowing⟩ row h
    exact (List.mem_append.mp h).elim (hFirst row) (hFollowing row)

theorem HeaderFactsHold.transport
    (semantics : TimelineSemantics FormulaExpr LocationExpr ValueExpr Child)
    (rows : List (HeaderFactRow FormulaExpr))
    (before after : StorageActionState)
    (hStable : HeaderFactsStable semantics rows)
    (hHold : HeaderFactsHold semantics rows before) :
    HeaderFactsHold semantics rows after := by
  intro row hMember
  exact (hStable row hMember before after).mp (hHold row hMember)

/-- A canonical prefix made only from the validated header facts establishes
every fact at the exact prefix-entry state.  This is the existing timeline
executor's leading-assumption theorem, not a parallel execution model. -/
theorem TimelineExec.establishes_header_facts
    (semantics : TimelineSemantics FormulaExpr LocationExpr ValueExpr Child)
    (rows : List (HeaderFactRow FormulaExpr))
    (initial : StorageActionState)
    (result : TimelineResult)
    (hExec : TimelineExec semantics
      ((headerFactFormulas rows).map TimelineEvent.assume) initial result) :
    HeaderFactsHold semantics rows initial := by
  have hLeading := TimelineExec.leading_assumptions_hold semantics
    (headerFactFormulas rows) [] initial result (by simpa using hExec)
  intro row hMember
  exact hLeading row.formula (by
    simp [headerFactFormulas]
    exact ⟨row, hMember, rfl⟩)

/-- Header facts are ambient, stable premises for the existing induction
obligations.  Discharging this implication yields the original obligation
structure; it does not introduce another loop correctness rule. -/
theorem LoopVC.induction_obligations_of_header_facts
    (vc : LoopVC State)
    (facts : Prop)
    (hFacts : facts)
    (proof : facts → vc.InductionObligations) :
    vc.InductionObligations :=
  proof hFacts

def LoopInitialWithHeaderFacts
    (facts : Prop) (vc : LoopVC State) (state : State) : Prop :=
  facts ∧ vc.initial state

theorem loopInitialWithEmptyHeaderFacts_iff
    (semantics : TimelineSemantics FormulaExpr LocationExpr ValueExpr Child)
    (vc : LoopVC State)
    (context : StorageActionState)
    (state : State) :
    LoopInitialWithHeaderFacts (HeaderFactsHold semantics [] context) vc state ↔
      vc.initial state := by
  simp [LoopInitialWithHeaderFacts, HeaderFactsHold]

theorem no_loop_initial_without_header_facts
    (facts : Prop)
    (vc : LoopVC State)
    (hMissing : ¬ facts) :
    ¬ ∃ state, LoopInitialWithHeaderFacts facts vc state := by
  intro hWitness
  rcases hWitness with ⟨state, hInitial⟩
  exact hMissing hInitial.1

private def headerFactTestTarget : HeaderPoint := {
  block := "function-entry"
  position := 7
}

private def headerFactTestRow : HeaderFactRow Bool := {
  id := 0
  structuralKey := "op:2/ora.assume"
  formula := true
  sourceKind := .pathAssume
  source := { block := "function-entry", position := 2 }
  target := headerFactTestTarget
}

example : headerFactTestRow.shapeReason? = none := rfl

example :
    ({ headerFactTestRow with
      source := { block := "nested", position := 2 } }).shapeReason? =
        some .sourceBlockMismatch := rfl

example :
    ({ headerFactTestRow with
      source := { block := "function-entry", position := 7 } }).shapeReason? =
        some .sourceNotEarlier := rfl

example :
    ({ headerFactTestRow with sourceKind := .unresolved }).shapeReason? =
      some .sourceKindUnresolved := rfl

end Ora.Loop

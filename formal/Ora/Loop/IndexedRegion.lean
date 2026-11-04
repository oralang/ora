/- Lean-first semantics for immutable indexed-region reads in loop timelines.
The snapshot is immutable; read results share the owner-scoped namespace with
storage reads. No compiler row or proof-planner rule lives here.
-/

import Ora.Loop.Storage

namespace Ora.Loop

open Ora.Obligation Ora.Spec

inductive IndexedRegionKind where
  | calldata
  | memory
  deriving Repr, DecidableEq

/-- A resolved location in one immutable indexed input snapshot. -/
structure IndexedRegionLocation where
  region : IndexedRegionKind
  owner : FreeVarId
  index : Ora.Integer.Value
  elementType : IntegerShape
  deriving Repr, DecidableEq

/-- Canonical read data; the index evaluates in its exact timeline-prefix state. -/
structure IndexedRegionRead (IndexExpr : Type) where
  id : StorageReadId
  region : IndexedRegionKind
  owner : FreeVarId
  index : IndexExpr
  elementType : IntegerShape

/-- Immutable input contents and the evaluator for the canonical index term.
`none` is a fail-closed missing location, never a default element value. -/
structure IndexedRegionReadSemantics (IndexExpr : Type) where
  snapshot : IndexedRegionLocation → Option Value
  evalIndex : IndexExpr → StorageActionState → Option Ora.Integer.Value

def IndexedRegionReadSemantics.unsupported : IndexedRegionReadSemantics IndexExpr where
  snapshot := fun _ => none
  evalIndex := fun _ _ => none

def indexedRegionValueMatches (elementType : IntegerShape) : Value → Bool
  | .integer value => decide (value.shape = elementType)
  | .bool _ => false

/-- One immutable-region read step.  It changes only the shared timeline read
bindings.  The snapshot is semantic input, not mutable execution state. -/
inductive IndexedRegionReadExec
    (semantics : IndexedRegionReadSemantics IndexExpr) :
    IndexedRegionRead IndexExpr →
      StorageActionState → StorageActionState → Prop where
  | read {event initial index value following}
      (hIndex : semantics.evalIndex event.index initial = some index)
      (hIndexType : index.shape = .unsigned .w256)
      (hSnapshot : semantics.snapshot {
        region := event.region
        owner := event.owner
        index
        elementType := event.elementType
      } = some value)
      (hElementType : indexedRegionValueMatches event.elementType value = true)
      (hBind : initial.bindRead? event.id value = some following) :
      IndexedRegionReadExec semantics event initial following

theorem IndexedRegionReadExec.exact_snapshot_value
    (semantics : IndexedRegionReadSemantics IndexExpr)
    (event : IndexedRegionRead IndexExpr) (initial following : StorageActionState)
    (hExec : IndexedRegionReadExec semantics event initial following) :
    ∃ index value,
      semantics.evalIndex event.index initial = some index ∧
      semantics.snapshot {
        region := event.region
        owner := event.owner
        index
        elementType := event.elementType
      } = some value := by
  cases hExec with
  | read hIndex _ hSnapshot _ _ => exact ⟨_, _, hIndex, hSnapshot⟩

theorem IndexedRegionReadExec.state_effects
    (semantics : IndexedRegionReadSemantics IndexExpr)
    (event : IndexedRegionRead IndexExpr) (initial following : StorageActionState)
    (hExec : IndexedRegionReadExec semantics event initial following) :
    storageReadIdBound initial.reads event.id = false ∧
      following.loop = initial.loop ∧
      following.entryStorage = initial.entryStorage ∧
      following.reads.map StorageReadBinding.id =
        initial.reads.map StorageReadBinding.id ++ [event.id] := by
  cases hExec with
  | read _ _ _ _ hBind =>
      simp only [StorageActionState.bindRead?] at hBind
      split at hBind
      · contradiction
      · simp only [Option.some.injEq] at hBind
        subst following
        simp_all

theorem IndexedRegionReadExec.same_location_same_value
    (semantics : IndexedRegionReadSemantics IndexExpr) (location : IndexedRegionLocation)
    (first second : Value)
    (hFirst : semantics.snapshot location = some first)
    (hSecond : semantics.snapshot location = some second) :
    first = second := by
  exact Option.some.inj (hFirst.symm.trans hSecond)

@[simp] theorem indexedRegionLocation_eq_iff (lhs rhs : IndexedRegionLocation) :
    lhs = rhs ↔
      lhs.region = rhs.region ∧
      lhs.owner = rhs.owner ∧
      lhs.index = rhs.index ∧
      lhs.elementType = rhs.elementType := by
  cases lhs
  cases rhs
  simp_all

end Ora.Loop

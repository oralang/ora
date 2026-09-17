/-
Lean-first semantic model for ordered persistent-storage actions in LoopVC.

This module deliberately contains no compiler row, MLIR operation, fixture, or
proof-planner definition.  The action executor is parameterized by the exact
location and value expression types plus their evaluators.  The canonical
manifest instantiation lands only after this model and its theorems are
kernel-checked, as required by the formal-lane Lean-first rule.

The supported value fragment is `Ora.Obligation.Value`: booleans and shaped Ora
integers.  `StorageLocation` is an abstract verifier-input Ora MLIR location;
it does not denote a physical EVM slot and assumes nothing about Keccak.
-/

import Ora.Loop.Core
import Ora.Obligation.Semantics

namespace Ora.Loop

open Ora.Obligation

def storageValueMatchesTy : Value → TyRef → Bool
  | .bool _, ty => ty.isBool
  | .integer value, ty => ty.integerShape? == some value.shape


/-- One ordered projection in a resolved semantic storage location.  Fixed
array and mapping keys remain distinct even when their typed values agree. -/
inductive StorageProjection where
  | field : String → StorageProjection
  | fixedIndex : Value → StorageProjection
  | mappingKey : Value → StorageProjection
  deriving Repr, DecidableEq

/-- A resolved persistent-storage location in the verifier-input Ora MLIR
model.  Projection order is semantic, and key values retain their Ora type. -/
structure StorageLocation where
  root : String
  projections : List StorageProjection := []
  deriving Repr, DecidableEq

/-- The current semantic persistent store. -/
abbrev StorageState := StorageLocation → Value

def StorageState.set
    (storage : StorageState)
    (location : StorageLocation)
    (value : Value) : StorageState :=
  fun candidate => if candidate = location then value else storage candidate

@[simp] theorem StorageState.set_self
    (storage : StorageState)
    (location : StorageLocation)
    (value : Value) :
    (storage.set location value) location = value := by
  simp [StorageState.set]

@[simp] theorem StorageState.set_other
    (storage : StorageState)
    (written observed : StorageLocation)
    (value : Value)
    (hNe : observed ≠ written) :
    (storage.set written value) observed = storage observed := by
  simp [StorageState.set, hNe]

inductive CallOutcome where
  | success
  | error : Nat → CallOutcome
  deriving Repr, BEq, DecidableEq

structure CallReturn where
  outcome : CallOutcome
  payload : List Value

/-- Loop-carried locals and storage, with a typed terminal return channel.
The exit timeline produces the payload together with this same final store. -/
structure LoopState where
  locals : List Ora.Integer.Value
  storage : StorageState
  returned : Option CallReturn := none

/- Read results share the canonical owner-scoped free-variable namespace.  A
plain counter would permit two functions or expansions to alias accidentally. -/
abbrev StorageReadId := FreeVarId

structure StorageReadBinding where
  id : StorageReadId
  value : Value
  deriving Repr, DecidableEq

def lookupStorageRead : List StorageReadBinding → StorageReadId → Option Value
  | [], _ => none
  | binding :: rest, id =>
      if binding.id = id then some binding.value else lookupStorageRead rest id

def storageReadIdBound (bindings : List StorageReadBinding) (id : StorageReadId) : Bool :=
  bindings.any (fun binding => binding.id == id)

/-- Execution state for direct storage actions.  `entryStorage` is immutable
and is the only store consulted by the later `old(place)` denotation.  Read
results are temporary timeline bindings, not loop-carried state. -/
structure StorageActionState where
  loop : LoopState
  entryStorage : StorageState
  reads : List StorageReadBinding := []

def StorageActionState.currentValue
    (state : StorageActionState) (location : StorageLocation) : Value :=
  state.loop.storage location

def StorageActionState.oldValue
    (state : StorageActionState) (location : StorageLocation) : Value :=
  state.entryStorage location

def StorageActionState.readValue?
    (state : StorageActionState) (id : StorageReadId) : Option Value :=
  lookupStorageRead state.reads id

def StorageActionState.bindRead?
    (state : StorageActionState)
    (id : StorageReadId)
    (value : Value) : Option StorageActionState :=
  if storageReadIdBound state.reads id then
    none
  else
    some { state with reads := state.reads ++ [{ id, value }] }

inductive StorageAction (LocationExpr ValueExpr : Type) where
  | read : StorageReadId → LocationExpr → StorageAction LocationExpr ValueExpr
  | write : LocationExpr → ValueExpr → StorageAction LocationExpr ValueExpr

/-- The semantic interface implemented later by canonical manifest formulas.
Both evaluators receive the exact state immediately before the action. -/
structure StorageActionSemantics (LocationExpr ValueExpr : Type) where
  evalLocation : LocationExpr → StorageActionState → Option StorageLocation
  evalValue : ValueExpr → StorageActionState → Option Value
  valueMatchesLocation : LocationExpr → Value → Bool

structure StorageActionResult where
  state : StorageActionState
  /-- Resolved writes in execution order. -/
  writeTrace : List StorageLocation := []

@[simp] def StorageActionResult.then
    (first following : StorageActionResult) : StorageActionResult := {
  state := following.state
  writeTrace := first.writeTrace ++ following.writeTrace
}

@[simp] theorem StorageActionResult.then_assoc
    (first middle final : StorageActionResult) :
    first.then (middle.then final) = (first.then middle).then final := by
  cases first
  cases middle
  cases final
  simp [StorageActionResult.then, List.append_assoc]

def execStorageAction?
    (semantics : StorageActionSemantics LocationExpr ValueExpr)
    (action : StorageAction LocationExpr ValueExpr)
    (state : StorageActionState) : Option StorageActionResult :=
  match action with
  | .read id locationExpr => do
      let location ← semantics.evalLocation locationExpr state
      let value := state.loop.storage location
      if !semantics.valueMatchesLocation locationExpr value then none else pure ()
      let following ← state.bindRead? id value
      pure { state := following }
  | .write locationExpr valueExpr => do
      let location ← semantics.evalLocation locationExpr state
      let value ← semantics.evalValue valueExpr state
      if !semantics.valueMatchesLocation locationExpr value then none else pure ()
      let followingLoop := { state.loop with
        storage := state.loop.storage.set location value }
      pure {
        state := { state with loop := followingLoop }
        writeTrace := [location]
      }

def execStorageActions?
    (semantics : StorageActionSemantics LocationExpr ValueExpr) :
    List (StorageAction LocationExpr ValueExpr) →
      StorageActionState → Option StorageActionResult
  | [], state => some { state }
  | action :: rest, state => do
      let first ← execStorageAction? semantics action state
      let following ← execStorageActions? semantics rest first.state
      pure (first.then following)

@[simp] theorem execStorageActions_empty
    (semantics : StorageActionSemantics LocationExpr ValueExpr)
    (state : StorageActionState) :
    execStorageActions? semantics [] state = some { state := state } := by
  rfl

theorem execStorageActions_append
    (semantics : StorageActionSemantics LocationExpr ValueExpr)
    (beforeActions afterActions : List (StorageAction LocationExpr ValueExpr))
    (initial : StorageActionState) :
    execStorageActions? semantics (beforeActions ++ afterActions) initial =
      (do
        let beforeResult ← execStorageActions? semantics beforeActions initial
        let afterResult ← execStorageActions? semantics afterActions beforeResult.state
        pure (beforeResult.then afterResult)) := by
  induction beforeActions generalizing initial with
  | nil =>
      cases hAfter : execStorageActions? semantics afterActions initial with
      | none => simp [execStorageActions?, hAfter]
      | some afterResult => simp [execStorageActions?, hAfter]
  | cons action rest ih =>
      cases hFirst : execStorageAction? semantics action initial with
      | none => simp [execStorageActions?, hFirst]
      | some firstResult =>
          cases hRest : execStorageActions? semantics rest firstResult.state with
          | none =>
              have hCombined := ih firstResult.state
              simp [hRest] at hCombined
              simp [execStorageActions?, hFirst, hRest, hCombined]
          | some restResult =>
              cases hAfter : execStorageActions? semantics afterActions restResult.state with
              | none =>
                  have hCombined := ih firstResult.state
                  simp [hRest, hAfter] at hCombined
                  simp [execStorageActions?, hFirst, hRest, hAfter, hCombined]
              | some afterResult =>
                  have hCombined := ih firstResult.state
                  simp [hRest, hAfter] at hCombined
                  simp [execStorageActions?, hFirst, hRest, hAfter, hCombined,
                    StorageActionResult.then_assoc]

theorem execStorageAction_preserves_unwritten
    (semantics : StorageActionSemantics LocationExpr ValueExpr)
    (action : StorageAction LocationExpr ValueExpr)
    (initial : StorageActionState)
    (result : StorageActionResult)
    (observed : StorageLocation)
    (hExec : execStorageAction? semantics action initial = some result)
    (hNotWritten : observed ∉ result.writeTrace) :
    result.state.loop.storage observed = initial.loop.storage observed := by
  cases action with
  | read id locationExpr =>
      simp only [execStorageAction?] at hExec
      cases hLocation : semantics.evalLocation locationExpr initial with
      | none => simp [hLocation] at hExec
      | some location =>
          simp only [hLocation, Option.bind_some] at hExec
          cases hMatch : semantics.valueMatchesLocation locationExpr
              (initial.loop.storage location) with
          | false => simp [hMatch] at hExec
          | true =>
              cases hBind : initial.bindRead? id (initial.loop.storage location) with
              | none => simp [hMatch, hBind] at hExec
              | some following =>
                  simp [hMatch, hBind] at hExec
                  subst result
                  simp [StorageActionState.bindRead?] at hBind
                  rcases hBind with ⟨_, rfl⟩
                  rfl
  | write locationExpr valueExpr =>
      simp only [execStorageAction?] at hExec
      cases hLocation : semantics.evalLocation locationExpr initial with
      | none => simp [hLocation] at hExec
      | some written =>
          simp only [hLocation, Option.bind_some] at hExec
          cases hValue : semantics.evalValue valueExpr initial with
          | none => simp [hValue] at hExec
          | some value =>
              simp only [hValue, Option.bind_some] at hExec
              cases hMatch : semantics.valueMatchesLocation locationExpr value with
              | false => simp [hMatch] at hExec
              | true =>
                  simp [hMatch] at hExec
                  subst result
                  simp only [List.mem_cons, List.not_mem_nil, or_false] at hNotWritten
                  exact StorageState.set_other initial.loop.storage written observed value
                    hNotWritten

theorem execStorageAction_preserves_entryStorage
    (semantics : StorageActionSemantics LocationExpr ValueExpr)
    (action : StorageAction LocationExpr ValueExpr)
    (initial : StorageActionState)
    (result : StorageActionResult)
    (hExec : execStorageAction? semantics action initial = some result) :
    result.state.entryStorage = initial.entryStorage := by
  cases action with
  | read id locationExpr =>
      cases hLocation : semantics.evalLocation locationExpr initial with
      | none => simp [execStorageAction?, hLocation] at hExec
      | some location =>
          cases hMatch : semantics.valueMatchesLocation locationExpr
              (initial.loop.storage location) with
          | false => simp [execStorageAction?, hLocation, hMatch] at hExec
          | true =>
              cases hBind : initial.bindRead? id (initial.loop.storage location) with
              | none => simp [execStorageAction?, hLocation, hMatch, hBind] at hExec
              | some following =>
                  simp [execStorageAction?, hLocation, hMatch, hBind] at hExec
                  subst result
                  simp [StorageActionState.bindRead?] at hBind
                  rcases hBind with ⟨_, rfl⟩
                  rfl
  | write locationExpr valueExpr =>
      cases hLocation : semantics.evalLocation locationExpr initial with
      | none => simp [execStorageAction?, hLocation] at hExec
      | some location =>
          cases hValue : semantics.evalValue valueExpr initial with
          | none => simp [execStorageAction?, hLocation, hValue] at hExec
          | some value =>
              cases hMatch : semantics.valueMatchesLocation locationExpr value with
              | false => simp [execStorageAction?, hLocation, hValue, hMatch] at hExec
              | true =>
                  simp [execStorageAction?, hLocation, hValue, hMatch] at hExec
                  subst result
                  rfl

theorem execStorageAction_preserves_locals
    (semantics : StorageActionSemantics LocationExpr ValueExpr)
    (action : StorageAction LocationExpr ValueExpr)
    (initial : StorageActionState)
    (result : StorageActionResult)
    (hExec : execStorageAction? semantics action initial = some result) :
    result.state.loop.locals = initial.loop.locals := by
  cases action <;>
    simp only [execStorageAction?, Option.bind_eq_bind,
      Option.bind_eq_some] at hExec
  case read =>
    rcases hExec with ⟨location, _, hExec⟩
    split at hExec
    · simp only [Option.none_bind] at hExec
      cases hExec
    · simp only [Option.pure_def, Option.some_bind,
        StorageActionState.bindRead?] at hExec
      split at hExec
      · simp only [Option.none_bind] at hExec
        cases hExec
      · simp only [Option.some_bind, Option.some.injEq] at hExec
        subst result
        rfl
  case write =>
    rcases hExec with ⟨location, _, value, _, hExec⟩
    split at hExec
    · simp only [Option.none_bind] at hExec
      cases hExec
    · simp only [Option.pure_def, Option.some_bind,
        Option.some.injEq] at hExec
      subst result
      rfl

theorem execStorageWrite_preserves_reads
    (semantics : StorageActionSemantics LocationExpr ValueExpr)
    (location : LocationExpr)
    (value : ValueExpr)
    (initial : StorageActionState)
    (result : StorageActionResult)
    (hExec : execStorageAction? semantics (.write location value) initial =
      some result) :
    result.state.reads = initial.reads := by
  simp only [execStorageAction?, Option.bind_eq_bind,
    Option.bind_eq_some] at hExec
  rcases hExec with ⟨resolved, _, stored, _, hExec⟩
  split at hExec
  · simp only [Option.none_bind] at hExec
    cases hExec
  · simp only [Option.pure_def, Option.some_bind,
      Option.some.injEq] at hExec
    subst result
    rfl

theorem execStorageRead_appends_read_id
    (semantics : StorageActionSemantics LocationExpr ValueExpr)
    (id : StorageReadId)
    (location : LocationExpr)
    (initial : StorageActionState)
    (result : StorageActionResult)
    (hExec : execStorageAction? semantics (.read id location) initial =
      some result) :
    result.state.reads.map StorageReadBinding.id =
      initial.reads.map StorageReadBinding.id ++ [id] := by
  simp only [execStorageAction?, Option.bind_eq_bind,
    Option.bind_eq_some] at hExec
  rcases hExec with ⟨resolved, _, hExec⟩
  split at hExec
  · simp only [Option.none_bind] at hExec
    cases hExec
  · simp only [Option.pure_def, Option.some_bind,
      StorageActionState.bindRead?] at hExec
    split at hExec
    · simp only [Option.none_bind] at hExec
      cases hExec
    · simp only [Option.some_bind, Option.some.injEq] at hExec
      subst result
      simp only [List.map_append, List.map_singleton,
        StorageReadBinding.id]

theorem execStorageRead_writeTrace_nil
    (semantics : StorageActionSemantics LocationExpr ValueExpr)
    (id : StorageReadId)
    (location : LocationExpr)
    (initial : StorageActionState)
    (result : StorageActionResult)
    (hExec : execStorageAction? semantics (.read id location) initial =
      some result) :
    result.writeTrace = [] := by
  simp only [execStorageAction?, Option.bind_eq_bind,
    Option.bind_eq_some] at hExec
  rcases hExec with ⟨resolved, _, hExec⟩
  split at hExec
  · simp only [Option.none_bind] at hExec
    cases hExec
  · simp only [Option.pure_def, Option.some_bind,
      StorageActionState.bindRead?] at hExec
    split at hExec
    · simp only [Option.none_bind] at hExec
      cases hExec
    · simp only [Option.some_bind, Option.some.injEq] at hExec
      subst result
      rfl

theorem execStorageWrite_writeTrace_singleton
    (semantics : StorageActionSemantics LocationExpr ValueExpr)
    (location : LocationExpr)
    (value : ValueExpr)
    (initial : StorageActionState)
    (result : StorageActionResult)
    (hExec : execStorageAction? semantics (.write location value) initial =
      some result) :
    ∃ resolved,
      semantics.evalLocation location initial = some resolved ∧
        result.writeTrace = [resolved] := by
  simp only [execStorageAction?, Option.bind_eq_bind,
    Option.bind_eq_some] at hExec
  rcases hExec with ⟨resolved, hResolved, stored, _, hExec⟩
  split at hExec
  · simp only [Option.none_bind] at hExec
    cases hExec
  · simp only [Option.pure_def, Option.some_bind,
      Option.some.injEq] at hExec
    subst result
    exact ⟨resolved, hResolved, rfl⟩

theorem execStorageActions_preserves_locals
    (semantics : StorageActionSemantics LocationExpr ValueExpr)
    (actions : List (StorageAction LocationExpr ValueExpr))
    (initial : StorageActionState)
    (result : StorageActionResult)
    (hExec : execStorageActions? semantics actions initial = some result) :
    result.state.loop.locals = initial.loop.locals := by
  induction actions generalizing initial result with
  | nil =>
      simp only [execStorageActions?] at hExec
      cases hExec
      rfl
  | cons action rest ih =>
      simp only [execStorageActions?] at hExec
      cases hAction : execStorageAction? semantics action initial with
      | none => simp [hAction] at hExec
      | some actionResult =>
          simp only [hAction, Option.bind_some] at hExec
          cases hRest : execStorageActions? semantics rest actionResult.state with
          | none => simp [hRest] at hExec
          | some restResult =>
              simp [hRest] at hExec
              cases hExec
              exact (ih actionResult.state restResult hRest).trans
                (execStorageAction_preserves_locals semantics action initial
                  actionResult hAction)

theorem execStorageActions_preserves_unwritten
    (semantics : StorageActionSemantics LocationExpr ValueExpr)
    (actions : List (StorageAction LocationExpr ValueExpr))
    (initial : StorageActionState)
    (result : StorageActionResult)
    (observed : StorageLocation)
    (hExec : execStorageActions? semantics actions initial = some result)
    (hNotWritten : observed ∉ result.writeTrace) :
    result.state.loop.storage observed = initial.loop.storage observed := by
  induction actions generalizing initial result with
  | nil =>
      simp only [execStorageActions?] at hExec
      cases hExec
      rfl
  | cons action rest ih =>
      simp only [execStorageActions?] at hExec
      cases hAction : execStorageAction? semantics action initial with
      | none => simp [hAction] at hExec
      | some actionResult =>
          simp only [hAction, Option.bind_some] at hExec
          cases hRest : execStorageActions? semantics rest actionResult.state with
          | none => simp [hRest] at hExec
          | some restResult =>
              simp [hRest] at hExec
              cases hExec
              have hNotAction : observed ∉ actionResult.writeTrace := by
                intro hMember
                exact hNotWritten (List.mem_append_left _ hMember)
              have hNotRest : observed ∉ restResult.writeTrace := by
                intro hMember
                exact hNotWritten (List.mem_append_right _ hMember)
              exact (ih actionResult.state restResult hRest hNotRest).trans
                (execStorageAction_preserves_unwritten semantics action initial
                  actionResult observed hAction hNotAction)

theorem execStorageActions_deterministic
    (semantics : StorageActionSemantics LocationExpr ValueExpr)
    (actions : List (StorageAction LocationExpr ValueExpr))
    (initial : StorageActionState)
    (first second : StorageActionResult)
    (hFirst : execStorageActions? semantics actions initial = some first)
    (hSecond : execStorageActions? semantics actions initial = some second) :
    first = second := by
  rw [hFirst] at hSecond
  exact Option.some.inj hSecond

/-- The locations written by some successful execution of an action list. -/
def storageActionFootprint
    (semantics : StorageActionSemantics LocationExpr ValueExpr)
    (actions : List (StorageAction LocationExpr ValueExpr))
    (location : StorageLocation) : Prop :=
  ∃ initial result,
    execStorageActions? semantics actions initial = some result ∧
      location ∈ result.writeTrace

def storageActionsRelation
    (semantics : StorageActionSemantics LocationExpr ValueExpr)
    (actions : List (StorageAction LocationExpr ValueExpr))
    (entryStorage : StorageState)
    (before after : LoopState) : Prop :=
  ∃ result,
    execStorageActions? semantics actions {
      loop := before, entryStorage := entryStorage
    } = some result ∧ result.state.loop = after

/-- A relation together with a certified write-footprint over-approximation. -/
structure CertifiedWriteFrame (relation : LoopState → LoopState → Prop) where
  footprint : StorageLocation → Prop
  preservesOutside : ∀ before after,
    relation before after →
    ∀ location, ¬ footprint location → after.storage location = before.storage location

def storageActionsCertifiedWriteFrame
    (semantics : StorageActionSemantics LocationExpr ValueExpr)
    (actions : List (StorageAction LocationExpr ValueExpr))
    (entryStorage : StorageState) :
    CertifiedWriteFrame (storageActionsRelation semantics actions entryStorage) := by
  refine {
    footprint := storageActionFootprint semantics actions
    preservesOutside := ?_
  }
  intro before after hRelation location hOutside
  rcases hRelation with ⟨result, hExec, rfl⟩
  apply execStorageActions_preserves_unwritten semantics actions
    { loop := before, entryStorage := entryStorage } result location hExec
  intro hMember
  exact hOutside ⟨{ loop := before, entryStorage := entryStorage }, result,
    hExec, hMember⟩

def relationComp
    (first second : LoopState → LoopState → Prop)
    (before after : LoopState) : Prop :=
  ∃ middle, first before middle ∧ second middle after

def CertifiedWriteFrame.comp
    {first second : LoopState → LoopState → Prop}
    (firstFrame : CertifiedWriteFrame first)
    (secondFrame : CertifiedWriteFrame second) :
    CertifiedWriteFrame (relationComp first second) := by
  refine {
    footprint := fun location =>
      firstFrame.footprint location ∨ secondFrame.footprint location
    preservesOutside := ?_
  }
  intro before after hRelation location hOutside
  rcases hRelation with ⟨middle, hFirst, hSecond⟩
  have hOutsideFirst : ¬ firstFrame.footprint location := by
    intro hMember
    exact hOutside (Or.inl hMember)
  have hOutsideSecond : ¬ secondFrame.footprint location := by
    intro hMember
    exact hOutside (Or.inr hMember)
  exact (secondFrame.preservesOutside middle after hSecond location
    hOutsideSecond).trans (firstFrame.preservesOutside before middle hFirst
      location hOutsideFirst)

end Ora.Loop

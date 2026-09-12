/- Executable examples for immutable indexed-region timeline reads. -/

import Ora.Loop.Timeline

namespace Ora.Loop

open Ora.Obligation Ora.Spec

private def owner : FreeVarId := { file_id := 1, pattern_id := 2 }
private def readId : StorageReadId := { file_id := 1, pattern_id := 3 }
private def index := Ora.Integer.Value.ofNat (.unsigned .w256) 7
private def value : Value :=
  .integer (Ora.Integer.Value.ofNat (.unsigned .w256) 42)

private def location : IndexedRegionLocation := {
  region := .calldata
  owner
  index
  elementType := .unsigned .w256
}

private def read : IndexedRegionRead Nat := {
  id := readId
  region := .calldata
  owner
  index := 7
  elementType := .unsigned .w256
}

private def state : StorageActionState := {
  loop := { locals := [index], storage := fun _ => .bool false }
  entryStorage := fun _ => .bool true
}

private def following : StorageActionState := {
  state with reads := [{ id := readId, value }]
}

private def semantics : IndexedRegionReadSemantics Nat := {
  snapshot := fun candidate => if candidate = location then some value else none
  evalIndex := fun candidate _ =>
    some (Ora.Integer.Value.ofNat (.unsigned .w256) candidate)
}

private theorem readExec : IndexedRegionReadExec semantics read state following := by
  apply IndexedRegionReadExec.read
  all_goals rfl

/- The event reads the exact typed location and only appends its globally
unique read binding; the immutable snapshot is semantic input, not state. -/
example :
    semantics.snapshot location = some value ∧
      storageReadIdBound state.reads readId = false ∧
      following.loop = state.loop ∧
      following.entryStorage = state.entryStorage ∧
      following.reads.map StorageReadBinding.id = [readId] := by
  refine ⟨rfl, ?_⟩
  simpa using IndexedRegionReadExec.state_effects _ _ _ _ readExec

example : ∃ resolved observed,
    semantics.evalIndex read.index state = some resolved ∧
      semantics.snapshot {
        region := read.region
        owner := read.owner
        index := resolved
        elementType := read.elementType
      } = some observed :=
  IndexedRegionReadExec.exact_snapshot_value _ _ _ _ readExec

/- Snapshot function extensionality gives same-location stability. -/
example (first second : Value)
    (hFirst : semantics.snapshot location = some first)
    (hSecond : semantics.snapshot location = some second) : first = second :=
  IndexedRegionReadExec.same_location_same_value _ _ _ _ hFirst hSecond

/- Region, owner, index, and element type are all location identity. -/
example :
    location ≠ { location with region := .memory } ∧
    location ≠ { location with owner := { file_id := 1, pattern_id := 4 } } ∧
    location ≠ { location with
      index := Ora.Integer.Value.ofNat (.unsigned .w256) 8 } ∧
    location ≠ { location with elementType := .unsigned .w128 } := by
  decide

/- A result identity is absent before its event and present afterward. -/
example : state.readValue? readId = none ∧ following.readValue? readId = some value := by
  decide

example : timelineReadBeforeCheckSupported
    ([.indexedRead read, .check true] : List (EventOf Bool PEmpty Nat PEmpty)) = false := by
  decide

private def storageSemantics : StorageActionSemantics PEmpty Nat := {
  evalLocation := fun impossible => nomatch impossible
  evalValue := fun _ _ => none
  valueMatchesLocation := fun impossible => nomatch impossible
}

private def timelineSemantics : TimelineSemantics Bool PEmpty Nat PEmpty := {
  storage := storageSemantics
  indexedRegion := semantics
  evalFormula := fun formula _ => some formula
  childRelation := fun impossible => nomatch impossible
  childFrame := fun impossible => nomatch impossible
}

private theorem eventExec : TimelineEventExec timelineSemantics
    (.indexedRead read) state { state := following } :=
  .indexedRead readExec

/- Indexed reads compose through the existing timeline executor. -/
example : TimelineExec timelineSemantics [.indexedRead read]
    state { state := following } :=
  by simpa [TimelineResult.comp] using
    TimelineExec.cons eventExec (.nil following)

/- A later indexed read cannot make an earlier false checkpoint true. -/
example : ¬ TimelineSafe timelineSemantics
    [.check false, .indexedRead read] state := by
  intro hSafe
  have hRequired := hSafe [] false [.indexedRead read] rfl
    { state } (.nil state)
  rcases hRequired with ⟨proposition, hEval, hHolds⟩
  have hFalse : proposition = False := by
    simpa [timelineSemantics] using hEval
  subst proposition
  exact hHolds

end Ora.Loop

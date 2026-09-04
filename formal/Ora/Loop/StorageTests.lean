/- Executable examples for the Lean-first storage timeline model. -/

import Ora.Loop.StorageCertificate

namespace Ora.Loop

open Ora.Obligation

abbrev ResolvedStorageAction := StorageAction StorageLocation Value

def resolvedStorageActionSemantics : StorageActionSemantics StorageLocation Value := {
  evalLocation := fun location _ => some location
  evalValue := fun value _ => some value
  valueMatchesLocation := fun _ _ => true
}

def execResolvedStorageActions? :=
  execStorageActions? resolvedStorageActionSemantics

def constantStorage (value : Value) : StorageState := fun _ => value

def testReadId (patternId : Nat) : StorageReadId :=
  { file_id := 0, pattern_id := patternId }

def testU256 (value : Nat) : Value :=
  .integer (Ora.Integer.Value.ofNat (.unsigned .w256) value)

def stableMappingKeyId : FreeVarId := { file_id := 0, pattern_id := 40 }

def stableMappingKeyManifest : Manifest := {
  terms := [.variable (.free {
    id := stableMappingKeyId
    name := "key"
    ty := some (.compilerTypeId 6)
  })]
}

example :
    let key := testU256 7
    let base := Env.empty.setFree stableMappingKeyId key
    let context := directStorageContext stableMappingKeyManifest base []
    let state : StorageActionState := {
      loop := { locals := [], storage := constantStorage (testU256 0) }
      entryStorage := constantStorage (testU256 0)
    }
    context.evalLocation? {
      root := "values"
      projections := [.mappingKey {
        formula := .term 0, ty := .compilerTypeId 6 }]
      valueTy := .compilerTypeId 6
    } state = some { root := "values", projections := [.mappingKey key] } := by
  decide

def directTestPlace : PlaceRef :=
  { root := "counter", region := .storage }

def directTestLocation : StorageLocation :=
  { root := "counter" }

def directTestResolver
    (_ : FreeVarId → Option Value)
    (place : PlaceRef) : Option StorageLocation :=
  if place == directTestPlace then some directTestLocation else none

def noLocalFormulaContext (manifest : Manifest) : StorageFormulaContext := {
  manifest
  entryEnv := Env.empty
  bindLocals := fun env locals => if locals.isEmpty then some env else none
  resolvePlace := directTestResolver
}

def keyedTestPlace : PlaceRef := {
  root := "values"
  region := .storage
  keys := [.parameter (testReadId 9)]
}

def keyedTestResolver
    (lookup : FreeVarId → Option Value)
    (place : PlaceRef) : Option StorageLocation :=
  match place.region, place.fields, place.keys with
  | .storage, [], [.parameter id] => do
      let key ← lookup id
      pure { root := place.root, projections := [.mappingKey key] }
  | _, _, _ => none

def currentAndOldManifest : Manifest := {
  terms := [.placeRead directTestPlace, .old 0]
}

example :
    let state : StorageActionState := {
      loop := { locals := [], storage := constantStorage (testU256 2) }
      entryStorage := constantStorage (testU256 1)
    }
    (noLocalFormulaContext currentAndOldManifest).evalValue? (.term 0) state =
        some (testU256 2) ∧
      (noLocalFormulaContext currentAndOldManifest).evalValue? (.term 1) state =
        some (testU256 1) := by
  decide

def currentAndOldKeyedManifest : Manifest := {
  terms := [.placeRead keyedTestPlace, .old 0]
}

example :
    let state : StorageActionState := {
      loop := { locals := [], storage := constantStorage (testU256 2) }
      entryStorage := constantStorage (testU256 1)
      reads := [{ id := testReadId 9, value := testU256 7 }]
    }
    let context : StorageFormulaContext := {
      manifest := currentAndOldKeyedManifest
      entryEnv := Env.empty
      bindLocals := fun env locals => if locals.isEmpty then some env else none
      resolvePlace := keyedTestResolver
    }
    context.evalValue? (.term 0) state = some (testU256 2) ∧
      context.evalValue? (.term 1) state = none := by
  decide

example :
    (noLocalFormulaContext currentAndOldManifest).evalPreheaderValue?
      (constantStorage (testU256 1)) (constantStorage (testU256 2)) (.term 0) =
        some (testU256 2) := by
  decide

def readResultManifest : Manifest := {
  terms := [.variable (.free {
    id := testReadId 9
    name := "read_result"
    ty := some (.compilerTypeId 6)
  })]
}

example :
    let state : StorageActionState := {
      loop := { locals := [], storage := constantStorage (testU256 0) }
      entryStorage := constantStorage (testU256 0)
      reads := [{ id := testReadId 9, value := testU256 7 }]
    }
    directInitialValuesAfterPreheader?
      (noLocalFormulaContext readResultManifest) state [.term 0] =
        some [Ora.Integer.Value.ofNat (.unsigned .w256) 7] := by
  decide

def preheaderTestManifest : Manifest := {
  terms := [
    .intLit { value := "7", ty := some (.compilerTypeId 6) },
    .variable (.free {
      id := testReadId 9
      name := "preheader_read"
      ty := some (.compilerTypeId 6)
    }),
    .boolLit true,
    .boolLit false
  ]
}

def preheaderContextId : FreeVarId := { file_id := 0, pattern_id := 20 }
def preheaderExcludedLoopId : FreeVarId := { file_id := 0, pattern_id := 21 }

example :
    let base := (Env.empty.setFree preheaderExcludedLoopId (testU256 9)).setFree
      preheaderContextId (testU256 8)
    let restricted := restrictStoragePreheaderEnv base [{
      index := 0
      id := preheaderContextId
      name := "limit"
      ty := .compilerTypeId 6
    }]
    restricted.lookupFree preheaderContextId = some (testU256 8) ∧
      restricted.lookupFree preheaderExcludedLoopId = none := by
  decide

def preheaderTestLocationRow : StorageLocationExprRow := {
  root := "counter"
  valueTy := .compilerTypeId 6
}

def preheaderTestRow : SummaryRow := {
  id := 1
  owner := "preheaderTest"
  kind := .scfWhile
  init := [.term 1]
  preheaderTimeline := [{
    id := 0
    structuralKey := "preheader/check"
    payload := .safetyCheckpoint { formulas := [{ formula := .term 2 }] }
  }, {
    id := 1
    structuralKey := "preheader/write"
    payload := .storageWrite {
      location := preheaderTestLocationRow
      value := { formula := .term 0 }
    }
  }, {
    id := 2
    structuralKey := "preheader/read"
    payload := .storageRead {
      result := testReadId 9
      location := preheaderTestLocationRow
    }
  }]
  preheaderProjectionComplete := true
}

def preheaderTestEvents : List DirectStorageEvent := [
  .check (.term 2),
  .storage (.write {
    root := "counter"
    valueTy := .compilerTypeId 6
  } (.term 0)),
  .storage (.read (testReadId 9) {
    root := "counter"
    valueTy := .compilerTypeId 6
  })
]

example : preheaderTimelineReason? [] preheaderTestRow.preheaderTimeline = none := by decide

example : preheaderTimelineFormulasDenotable preheaderTestManifest preheaderTestRow = true := by decide

example : storageLocationFormulasDenotable preheaderTestManifest
    (Env.empty.setFree (testReadId 9) (testU256 7)) {
      root := "values"
      projections := [.mappingKey {
        value := { formula := .term 1, readResults := [testReadId 9] }
        ty := .compilerTypeId 6
      }]
      valueTy := .compilerTypeId 6
    } = true := by decide

example : directStorageTimeline? preheaderTestRow.preheaderTimeline =
    some preheaderTestEvents := by
  rfl

example (base : Env) :
    denoteDirectStoragePreheaderInitial? preheaderTestManifest base
      preheaderTestRow = some
        (directStoragePreheaderProducesInitial
          (directStoragePreheaderContext preheaderTestManifest base
            preheaderTestRow.contextVariables)
          preheaderTestEvents (directStoragePreheaderInitialState base)
          preheaderTestRow.init) := by
  rfl

example
    (context : StorageFormulaContext)
    (initial : StorageActionState)
    (formulas : List FormulaRef)
    (following : LoopState) :
    directStoragePreheaderProducesInitial context [] initial formulas following ↔
      directInitialValuesAfterPreheader? context initial formulas =
          some following.locals ∧
        following.storage = initial.loop.storage :=
  direct_storage_preheader_empty_iff context initial formulas following

example (storage entryStorage : StorageState) :
    let initial : StorageActionState := {
      loop := { locals := [], storage }
      entryStorage
    }
    directStoragePreheaderProducesInitial
      (noLocalFormulaContext { terms := [.boolLit true] })
      [.check (.term 0)] initial [] {
        locals := []
        storage := initial.loop.storage
      } := by
  dsimp only
  let initial : StorageActionState := {
    loop := { locals := [], storage }
    entryStorage
  }
  let context := noLocalFormulaContext { terms := [.boolLit true] }
  have hEval : (directStoragePreheaderSemantics context).evalFormula
      (.term 0) initial = some True := by
    simp [directStoragePreheaderSemantics, directStorageSemantics, context,
      noLocalFormulaContext, StorageFormulaContext.evalFormula?,
      StorageFormulaContext.stateEnv?, bindStorageReads,
      StorageFormulaContext.formulaStorageReady,
      StorageFormulaContext.termStorageReady,
      StorageFormulaContext.withStoreViews, formulaDenotes?, denoteFormula?]
  let finalResult : TimelineResult :=
    ({ state := initial } : TimelineResult).comp { state := initial }
  refine ⟨finalResult, .cons (.check hEval) (.nil initial), ?_, rfl, rfl⟩
  intro before formula after hEvents beforeResult hExec
  cases before with
  | nil =>
      simp at hEvents
      rcases hEvents with ⟨rfl, rfl⟩
      cases hExec
      exact ⟨True, hEval, trivial⟩
  | cons head tail =>
      simp at hEvents

example (storage entryStorage : StorageState) (following : LoopState) :
    let initial : StorageActionState := {
      loop := { locals := [], storage }
      entryStorage
    }
    ¬ directStoragePreheaderProducesInitial
      (noLocalFormulaContext { terms := [.boolLit false] })
      [.check (.term 0)] initial [] following := by
  dsimp only
  let initial : StorageActionState := {
    loop := { locals := [], storage }
    entryStorage
  }
  intro hInitial
  rcases direct_storage_preheader_initial_has_safe_execution _ _ _ _ _ hInitial with
    ⟨_, _, hSafe⟩
  have hFalse := false_checkpoint_does_not_disappear
    (directStoragePreheaderSemantics
      (noLocalFormulaContext { terms := [.boolLit false] }))
    (.term 0) initial (by
      simp [directStoragePreheaderSemantics, directStorageSemantics,
        noLocalFormulaContext, StorageFormulaContext.evalFormula?,
        StorageFormulaContext.stateEnv?, bindStorageReads,
        StorageFormulaContext.formulaStorageReady,
        StorageFormulaContext.termStorageReady,
        StorageFormulaContext.withStoreViews, formulaDenotes?, denoteFormula?])
  exact hFalse.2 hSafe

example :
    let state : StorageActionState := {
      loop := { locals := [], storage := constantStorage (testU256 0) }
      entryStorage := constantStorage (testU256 0)
      reads := [{ id := testReadId 9, value := testU256 7 }]
    }
    (noLocalFormulaContext readResultManifest).evalValue? (.term 0) state =
      some (testU256 7) := by
  decide

example :
    let context := noLocalFormulaContext {}
    let location : StorageLocationExpr := {
      root := "counter"
      valueTy := .compilerTypeId 6
    }
    execStorageAction? context.actionSemantics
      (.read (testReadId 1) location) {
        loop := { locals := [], storage := constantStorage (.bool true) }
        entryStorage := constantStorage (.bool true)
      } = none := by
  rfl

example (state : StorageActionState) :
    execResolvedStorageActions? [] state = some { state } := by
  rfl

example
    (value : Value)
    (locals : List Ora.Integer.Value)
    (location : StorageLocation) :
    ∃ result,
      execResolvedStorageActions? [.read (testReadId 7) location] {
        loop := { locals, storage := constantStorage value }
        entryStorage := constantStorage value
      } = some result ∧
      lookupStorageRead result.state.reads (testReadId 7) = some value := by
  refine ⟨{
    state := {
      loop := { locals := locals, storage := constantStorage value }
      entryStorage := constantStorage value
      reads := [{ id := testReadId 7, value := value }]
    }
  }, ?_⟩
  simp [execResolvedStorageActions?, execStorageActions?, execStorageAction?,
    resolvedStorageActionSemantics, StorageActionState.bindRead?,
    storageReadIdBound, lookupStorageRead, constantStorage]

example
    (initialValue writtenValue : Value)
    (locals : List Ora.Integer.Value)
    (location : StorageLocation) :
    ∃ result,
      execResolvedStorageActions? [.write location writtenValue] {
        loop := { locals, storage := constantStorage initialValue }
        entryStorage := constantStorage initialValue
      } = some result ∧
      result.state.loop.storage location = writtenValue ∧
      result.writeTrace = [location] := by
  refine ⟨{
    state := {
      loop := {
        locals := locals
        storage := (constantStorage initialValue).set location writtenValue
      }
      entryStorage := constantStorage initialValue
    }
    writeTrace := [location]
  }, ?_⟩
  simp [execResolvedStorageActions?, execStorageActions?, execStorageAction?,
    resolvedStorageActionSemantics]

example
    (initialValue writtenValue : Value)
    (locals : List Ora.Integer.Value)
    (location : StorageLocation) :
    ∃ result,
      execResolvedStorageActions?
        [.write location writtenValue, .read (testReadId 7) location] {
          loop := { locals, storage := constantStorage initialValue }
          entryStorage := constantStorage initialValue
        } = some result ∧
      lookupStorageRead result.state.reads (testReadId 7) = some writtenValue := by
  refine ⟨{
    state := {
      loop := {
        locals := locals
        storage := (constantStorage initialValue).set location writtenValue
      }
      entryStorage := constantStorage initialValue
      reads := [{ id := testReadId 7, value := writtenValue }]
    }
    writeTrace := [location]
  }, ?_⟩
  simp [execResolvedStorageActions?, execStorageActions?, execStorageAction?,
    resolvedStorageActionSemantics, StorageActionState.bindRead?,
    storageReadIdBound, lookupStorageRead]

example
    (initial first second : Value)
    (locals : List Ora.Integer.Value)
    (location : StorageLocation) :
    ∃ result,
      execResolvedStorageActions?
        [.write location first, .write location second] {
          loop := { locals, storage := constantStorage initial }
          entryStorage := constantStorage initial
        } = some result ∧
      result.state.loop.storage location = second ∧
      result.writeTrace = [location, location] := by
  refine ⟨{
    state := {
      loop := {
        locals := locals
        storage := ((constantStorage initial).set location first).set location second
      }
      entryStorage := constantStorage initial
    }
    writeTrace := [location, location]
  }, ?_⟩
  simp [execResolvedStorageActions?, execStorageActions?, execStorageAction?,
    resolvedStorageActionSemantics]

example
    (initial written : Value)
    (locals : List Ora.Integer.Value)
    (writtenLocation observedLocation : StorageLocation)
    (hNe : observedLocation ≠ writtenLocation) :
    ∃ result,
      execResolvedStorageActions? [.write writtenLocation written] {
        loop := { locals, storage := constantStorage initial }
        entryStorage := constantStorage initial
      } = some result ∧
      result.state.loop.storage observedLocation = initial := by
  refine ⟨{
    state := {
      loop := {
        locals := locals
        storage := (constantStorage initial).set writtenLocation written
      }
      entryStorage := constantStorage initial
    }
    writeTrace := [writtenLocation]
  }, ?_⟩
  simp [execResolvedStorageActions?, execStorageActions?, execStorageAction?,
    resolvedStorageActionSemantics, constantStorage, hNe]

def fixedTestLocation (root : String) (key : Value) : StorageLocation :=
  { root, projections := [.fixedIndex key] }

def mappingTestLocation (root : String) (key : Value) : StorageLocation :=
  { root, projections := [.mappingKey key] }

def resolvedTimelineSemantics :
    TimelineSemantics Bool StorageLocation Value PEmpty := {
  storage := resolvedStorageActionSemantics
  evalFormula := fun formula _ => some formula
  childRelation := fun child => nomatch child
  childFrame := fun child => nomatch child
}

example
    (initial : StorageActionState)
    (result : TimelineResult)
    (written observed : StorageLocation)
    (value : Value)
    (hNe : observed ≠ written)
    (hExec : TimelineExec resolvedTimelineSemantics
      [.storage (.write written value)] initial result) :
    result.state.loop.storage observed = initial.loop.storage observed := by
  apply TimelineExec.preserves_outside_resolved_frame _ _ _ _ [written]
    observed hExec
  · intro candidate hWritten
    cases hExec with
    | cons hEvent hRest =>
        cases hRest
        cases hEvent with
        | storage hAction =>
            simp [resolvedTimelineSemantics, resolvedStorageActionSemantics,
              execStorageAction?] at hAction
            subst_vars
            simpa [TimelineResult.comp] using hWritten
  · intro candidate hMember
    simp only [List.mem_singleton] at hMember
    subst candidate
    exact hNe

/-- Reads and repeated writes to one mapping location preserve a distinct
mapping location through the generic whole-timeline frame theorem. -/
example
    (initial : StorageActionState)
    (result : TimelineResult)
    (source written observed : StorageLocation)
    (readId : StorageReadId)
    (first second : Value)
    (hNe : observed ≠ written)
    (hExec : TimelineExec resolvedTimelineSemantics
      [.storage (.read readId source),
       .storage (.write written first),
       .storage (.write written second)] initial result) :
    result.state.loop.storage observed = initial.loop.storage observed := by
  apply TimelineExec.preserves_outside_event_frame _ _ _ _ [written]
    observed ?_ ?_ hExec
  · intro event hMember before after hEvent candidate hCandidate
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hMember
    rcases hMember with rfl | rfl | rfl
    · exact False.elim
        (TimelineEventExec.read_writes_false _ _ _ _ _ hEvent _ hCandidate)
    · simpa using TimelineEventExec.write_within_resolved _ _ _ written
        before after rfl hEvent candidate hCandidate
    · simpa using TimelineEventExec.write_within_resolved _ _ _ written
        before after rfl hEvent candidate hCandidate
  · intro candidate hMember
    simp only [List.mem_singleton] at hMember
    subst candidate
    exact hNe

/-- A later write to the observed mapping key cannot be hidden by an earlier
disjoint write: the singleton frame premise itself is false. -/
example
    (initial : StorageActionState)
    (written observed : StorageLocation)
    (later : Value)
    (hNe : observed ≠ written) :
    ¬ (∀ after,
      TimelineEventExec resolvedTimelineSemantics
        (.storage (.write observed later)) initial after →
      ∀ location, after.writes location → location ∈ [written]) := by
  intro hFrame
  let followingState : StorageActionState := {
    initial with loop := { initial.loop with
      storage := initial.loop.storage.set observed later }
  }
  have hLater : execStorageAction? resolvedStorageActionSemantics
      (.write observed later) initial = some {
        state := followingState, writeTrace := [observed]
      } := by
    simp [resolvedStorageActionSemantics, execStorageAction?, followingState]
  have hLaterEvent : TimelineEventExec resolvedTimelineSemantics
      (.storage (.write observed later)) initial {
        state := followingState, writes := fun location => location ∈ [observed]
      } := by
    exact @TimelineEventExec.storage Bool StorageLocation Value PEmpty
      resolvedTimelineSemantics (.write observed later) initial
      { state := followingState, writeTrace := [observed] }
      (by simpa [resolvedTimelineSemantics] using hLater)
  have hObserved := hFrame _ hLaterEvent observed (by simp)
  simp only [List.mem_singleton] at hObserved
  exact hNe hObserved

example (root : String) (i j : Value) (hNe : i ≠ j) :
    mappingTestLocation root i ≠ mappingTestLocation root j := by
  exact directStorageMappingLocations_ne root i j hNe

example
    (initial : StorageActionState)
    (result : TimelineResult)
    (first second observed : StorageLocation)
    (events : List (TimelineEvent Bool StorageLocation Value PEmpty))
    (hExec : TimelineExec resolvedTimelineSemantics events initial result)
    (hCovered : ∀ written, result.writes written → written ∈ [first, second])
    (hFirst : observed ≠ first)
    (hSecond : observed ≠ second) :
    result.state.loop.storage observed = initial.loop.storage observed := by
  apply TimelineExec.preserves_outside_resolved_frame _ _ _ _ [first, second]
    observed hExec hCovered
  intro written hMember
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hMember
  rcases hMember with rfl | rfl
  · exact hFirst
  · exact hSecond

example (root : String) (i j : Value) (hNe : i ≠ j) :
    fixedTestLocation root i ≠ fixedTestLocation root j := by
  intro hEqual
  exact hNe (StorageProjection.fixedIndex.inj
    (List.cons.inj (StorageLocation.mk.inj hEqual).2).1)

example (left right : String) (key : Value) (hNe : left ≠ right) :
    mappingTestLocation left key ≠ mappingTestLocation right key := by
  intro hEqual
  exact hNe (StorageLocation.mk.inj hEqual).1

example (root : String) (key : Value) :
    fixedTestLocation root key ≠ mappingTestLocation root key := by
  intro hEqual
  cases (List.cons.inj (StorageLocation.mk.inj hEqual).2).1

example (root : String) (key initial written : Value)
    (hNe : initial ≠ written) :
    let location := mappingTestLocation root key
    ((constantStorage initial).set location written) location ≠ initial := by
  simp [Ne.symm hNe]

example (root : String) (i j initial first later : Value)
    (hKeys : i ≠ j) (hChanged : initial ≠ later) :
    let written := mappingTestLocation root i
    let observed := mappingTestLocation root j
    let afterFirst := (constantStorage initial).set written first
    afterFirst observed = initial ∧
      (afterFirst.set observed later) observed ≠ initial := by
  have hLocations : mappingTestLocation root j ≠ mappingTestLocation root i := by
    intro hEqual
    exact hKeys (StorageProjection.mappingKey.inj
      (List.cons.inj (StorageLocation.mk.inj hEqual).2).1).symm
  simp [constantStorage, hLocations, Ne.symm hChanged]

example
    (entryValue currentValue writtenValue : Value)
    (locals : List Ora.Integer.Value)
    (location : StorageLocation) :
    ∃ result,
      execResolvedStorageActions? [.write location writtenValue] {
        loop := { locals, storage := constantStorage currentValue }
        entryStorage := constantStorage entryValue
      } = some result ∧
      result.state.entryStorage location = entryValue := by
  refine ⟨{
    state := {
      loop := {
        locals := locals
        storage := (constantStorage currentValue).set location writtenValue
      }
      entryStorage := constantStorage entryValue
    }
    writeTrace := [location]
  }, ?_⟩
  simp [execResolvedStorageActions?, execStorageActions?, execStorageAction?,
    resolvedStorageActionSemantics, constantStorage]

example
    (value : Value)
    (locals : List Ora.Integer.Value)
    (source target : StorageLocation) :
    let semantics : StorageActionSemantics StorageLocation
        (Sum Value StorageReadId) := {
      evalLocation := fun location _ => some location
      evalValue := fun expression state => match expression with
        | .inl literal => some literal
        | .inr readId => lookupStorageRead state.reads readId
      valueMatchesLocation := fun _ _ => true
    }
    ∃ result,
      execStorageActions? semantics
        [.read (testReadId 5) source, .write target (.inr (testReadId 5))] {
          loop := { locals, storage := constantStorage value }
          entryStorage := constantStorage value
        } = some result ∧
      result.state.loop.storage target = value := by
  dsimp
  refine ⟨{
    state := {
      loop := {
        locals := locals
        storage := (constantStorage value).set target value
      }
      entryStorage := constantStorage value
      reads := [{ id := testReadId 5, value := value }]
    }
    writeTrace := [target]
  }, ?_⟩
  simp [execStorageActions?, execStorageAction?, StorageActionState.bindRead?,
    storageReadIdBound, lookupStorageRead, constantStorage]

example
    (value : Value)
    (locals : List Ora.Integer.Value)
    (location : StorageLocation) :
    execResolvedStorageActions?
      [.read (testReadId 3) location, .read (testReadId 3) location] {
      loop := { locals, storage := constantStorage value }
      entryStorage := constantStorage value
    } = none := by
  have hSame : testReadId 3 == testReadId 3 := by decide
  simp [execResolvedStorageActions?, execStorageActions?, execStorageAction?,
    resolvedStorageActionSemantics, StorageActionState.bindRead?,
    storageReadIdBound, hSame]

example
    (value : Value)
    (location : StorageLocation) :
    ∃ storage : StorageState, storage location = value := by
  exact ⟨constantStorage value, rfl⟩

example
    (first second : Value)
    (location : StorageLocation)
    (hNe : first ≠ second) :
    ¬ ∃ storage : StorageState,
      storage location = first ∧ storage location = second := by
  intro witness
  rcases witness with ⟨storage, hFirst, hSecond⟩
  exact hNe (hFirst.symm.trans hSecond)

def branchingChildRelation (before after : LoopState) : Prop :=
  after.storage = before.storage ∧
    (after.locals = before.locals ∨ after.locals = [])

def branchingChildFrame : CertifiedWriteFrame branchingChildRelation where
  footprint := fun _ => False
  preservesOutside := by
    intro before after hRelation location _
    exact congrFun hRelation.1 location

def branchingTimelineSemantics :
    TimelineSemantics (StorageActionState → Prop) StorageLocation Value Unit where
  storage := resolvedStorageActionSemantics
  evalFormula := fun formula state => some (formula state)
  childRelation := fun _ => branchingChildRelation
  childFrame := fun _ => branchingChildFrame

/-- A checkpoint after a relational child is required for every admitted child
result, not merely for one safe result. -/
example
    (value : Value)
    (carried : Ora.Integer.Value) :
    let initial : StorageActionState := {
      loop := { locals := [carried], storage := constantStorage value }
      entryStorage := constantStorage value
    }
    let formula : StorageActionState → Prop := fun state => state.loop.locals ≠ []
    ¬ TimelineSafe branchingTimelineSemantics
      [.child (), .check formula] initial := by
  dsimp
  intro hSafe
  let initial : StorageActionState := {
    loop := { locals := [carried], storage := constantStorage value }
    entryStorage := constantStorage value
  }
  let emptyLoop : LoopState := {
    locals := []
    storage := initial.loop.storage
  }
  let first : TimelineResult := {
    state := { initial with loop := emptyLoop }
    writes := branchingChildFrame.footprint
  }
  let final : TimelineResult := first.comp { state := first.state }
  have hChild : TimelineEventExec branchingTimelineSemantics (.child ())
      initial first := by
    apply TimelineEventExec.child
    exact ⟨rfl, Or.inr rfl⟩
  have hPrefix : TimelineExec branchingTimelineSemantics [.child ()]
      initial final := TimelineExec.cons hChild (.nil first.state)
  have hRequired := hSafe [.child ()]
    (fun state : StorageActionState => state.loop.locals ≠ []) [] rfl final hPrefix
  rcases hRequired with ⟨proposition, hEval, hHolds⟩
  have hExpected : branchingTimelineSemantics.evalFormula
      (fun state : StorageActionState => state.loop.locals ≠ []) final.state =
      some (final.state.loop.locals ≠ []) := rfl
  have hProposition : proposition = (final.state.loop.locals ≠ []) :=
    Option.some.inj (hEval.symm.trans hExpected)
  subst proposition
  apply hHolds
  rfl

/-- An incomplete empty footprint cannot certify a relation that changes a
location. -/
example
    (location : StorageLocation)
    (beforeValue afterValue : Value)
    (hNe : beforeValue ≠ afterValue) :
    let before : LoopState := {
      locals := []
      storage := constantStorage beforeValue
    }
    let after : LoopState := {
      locals := []
      storage := (constantStorage beforeValue).set location afterValue
    }
    let relation := fun lhs rhs : LoopState => lhs = before ∧ rhs = after
    ¬ ∃ frame : CertifiedWriteFrame relation,
      ∀ candidate, ¬ frame.footprint candidate := by
  dsimp
  intro hExists
  rcases hExists with ⟨frame, hEmpty⟩
  have hRelation : (fun lhs rhs : LoopState =>
      lhs = { locals := [], storage := constantStorage beforeValue } ∧
      rhs = {
        locals := []
        storage := (constantStorage beforeValue).set location afterValue
      })
      { locals := [], storage := constantStorage beforeValue }
      {
        locals := []
        storage := (constantStorage beforeValue).set location afterValue
      } := ⟨rfl, rfl⟩
  have hPreserved := frame.preservesOutside _ _ hRelation location (hEmpty location)
  simp [constantStorage] at hPreserved
  exact hNe hPreserved.symm

private def refinementControlEdge : Edge Nat where
  path := fun state => state = 0
  safe := fun _ => True
  next := fun current following => current = 0 ∧ following = 1
  target := .backedge
  targetAssertion := fun _ => True

private def refinementControlVC : LoopVC Nat where
  initial := fun state => state = 0
  invariant := fun state => state = 0 ∨ state = 1
  edges := [refinementControlEdge]

private def refinementExactEdge : Edge (Nat × Bool) where
  path := fun _ => True
  safe := fun _ => True
  next := fun current following =>
    current.1 = 0 ∧ following.1 = 1 ∧ following.2 = !current.2
  target := .backedge
  targetAssertion := fun _ => True

private def refinementExactVC : LoopVC (Nat × Bool) where
  initial := fun state => state.1 = 0
  invariant := fun state => state.1 = 0 ∨ state.1 = 1
  edges := [refinementExactEdge]

private theorem refinementControlInduction :
    refinementControlVC.InductionObligations where
  base := by
    intro state hInitial
    exact Or.inl hInitial
  backedge := by
    intro edge current following hMember _ _ _ _ hNext
    simp only [refinementControlVC, List.mem_cons,
      List.not_mem_nil, or_false] at hMember
    subst edge
    exact Or.inr hNext.2
  edgeSafety := by
    intro edge _ hMember _ _
    simp only [refinementControlVC, List.mem_cons,
      List.not_mem_nil, or_false] at hMember
    subst edge
    trivial
  exit := by
    intro edge _ _ hMember hNotBackedge
    simp only [refinementControlVC, List.mem_cons,
      List.not_mem_nil, or_false] at hMember
    subst edge
    exact False.elim (hNotBackedge rfl)

private theorem refinementExactRefinesControl :
    LoopVC.Refinement refinementControlVC refinementExactVC Prod.fst where
  initial := by
    intro state hInitial
    exact hInitial
  invariant := by
    intro state
    rfl
  edges := .cons {
    target := rfl
    safety := by
      intro _ _ _ _ _
      trivial
    transition := by
      intro current following _ _ _ hNext
      exact ⟨hNext.1, hNext.1, hNext.2.1⟩
    targetAssertion := by
      intro _ _
      trivial
  } .nil

/-- Exact timelines may keep `path = True`: a completed transition recovers
the abstract control predicate, while safety remains independently required. -/
example : refinementExactVC.InductionObligations :=
  induction_obligations_of_refinement refinementControlVC refinementExactVC
    Prod.fst refinementControlInduction refinementExactRefinesControl

end Ora.Loop

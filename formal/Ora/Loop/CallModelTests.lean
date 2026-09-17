import Ora.Loop.Call

set_option linter.unusedVariables false
namespace Ora.Loop.CallTests
open Ora.Obligation

def flag : StorageLocation := { root := "flag" }
def boolSignature : CallSignature := {
  parameters := []
  results := fun outcome => if outcome = .success then some [.spelling "bool"] else none
}
abbrev Expr := StorageActionState → Option Value
abbrev Formula := StorageActionState → Option Prop

def leafSemantics : TimelineSemantics Formula StorageLocation Expr Empty := {
  storage := {
    evalLocation := fun location _ => some location
    evalValue := fun value state => value state
    valueMatchesLocation := fun _ value => storageValueMatchesTy value (.spelling "bool")
  }
  evalFormula := fun formula state => formula state
  childRelation := fun child => nomatch child
  childFrame := fun child => nomatch child
}

def leaf : CallBody Formula StorageLocation Expr Empty := {
  signature := boolSignature
  semantics := fun _ => leafSemantics
  paths := [{
    events := [.storage (.write flag (fun _ => some (.bool true)))]
    returns := .direct .success [fun state => some (state.loop.storage flag)]
  }]
  requires := fun _ => True
}

theorem leafTrace (initial : StorageActionState) (result : TimelineResult)
    (trace : TimelineExec leafSemantics
      [.storage (.write flag (fun _ => some (.bool true)))] initial result) :
    result.state.loop.storage flag = .bool true := by
  cases trace with
  | cons event rest =>
    cases rest
    cases event with
    | storage action =>
      simp [execStorageAction?, leafSemantics, storageValueMatchesTy, TyRef.isBool] at action
      subst_vars
      simp [TimelineResult.comp]

def leafCertificate : leaf.Certificate where
  loopResults := by
    intro input typed required initial entry path member before child
    cases child
  calls := by
    intro input typed requires initial entry path member before invocation after events
    simp only [leaf, List.mem_singleton] at member
    subst path
    cases before with
    | nil => simp at events
    | cons head tail =>
      simp only [List.cons_append, List.cons.injEq] at events
      have : tail ++ TimelineEvent.call invocation :: after = [] := events.2.symm
      simp at this
  safety := by
    intro input typed requires initial entry path member before formula after events
    simp only [leaf, List.mem_singleton] at member
    subst path
    cases before with
    | nil => simp at events
    | cons head tail =>
      simp only [List.cons_append, List.cons.injEq] at events
      have : tail ++ TimelineEvent.check formula :: after = [] := events.2.symm
      simp at this
  returns := by
    intro input initial path result typed requires entry member trace
    simp only [leaf, List.mem_singleton] at member
    subst path
    have stored := leafTrace initial result trace
    refine ⟨{ outcome := .success, payload := [.bool true] }, ?_, ?_⟩
    · simp [callEvaluateReturn?, callEvaluateValues?, leaf, leafSemantics, stored]
    · rfl

theorem leafJoint (input : CallInput) (output : CallOutput) (run : leaf.execution input output) :
    output.payload = [.bool true] ∧ output.storage flag = .bool true := by
  rcases run with ⟨initial, path, result, payload, entry, member, trace, evaluated, rfl⟩
  simp only [leaf, List.mem_singleton] at member
  subst path
  have stored := leafTrace initial result trace
  simp [callEvaluateReturn?, callEvaluateValues?, leaf, leafSemantics, stored] at evaluated
  subst payload
  exact ⟨rfl, stored⟩

def caller : StorageActionState := {
  loop := { locals := [], storage := fun _ => .bool true }
  entryStorage := fun _ => .bool false
  reads := [{ id := { file_id := 1, pattern_id := 0 }, value := .bool false }]
}
def invocation : CallInvocation Expr := {
  callee := "leaf", arguments := [], outcome := .success
  results := [{ file_id := 1, pattern_id := 1 }]
}

example :
    ((boolSignature.entry? { arguments := [], storage := caller.loop.storage }).getD caller).oldValue flag = .bool true ∧
    (invocation.resume caller {
      outcome := .success, payload := [.bool true], storage := caller.loop.storage }).oldValue flag = .bool false := by decide

example : callValuesTyped [.spelling "bool"] [.bool false] = true := by decide
example : callValuesTyped [.spelling "bool"] [] = false := by decide
example : callValuesTyped [] [.bool true] = false := by decide
example : callValuesTyped [.spelling "u256"] [.bool true] = false := by decide
example : invocation.fresh caller := by unfold CallInvocation.fresh; decide

def backedge : Edge LoopState := {
  path := fun state => state.storage flag = .bool false
  safe := fun _ => True
  next := fun before after => after = {
    before with storage := before.storage.set flag (.bool true) }
  target := .backedge
  targetAssertion := fun _ => True
}
def exitEdge : Edge LoopState := {
  path := fun state => state.storage flag = .bool true
  safe := fun _ => True
  next := Eq
  target := .successor
  targetAssertion := fun _ => True
}
def loopVC (entry : StorageState) : LoopVC LoopState := {
  initial := fun state => state.storage = entry
  invariant := fun state => ∀ location, location ≠ flag →
    state.storage location = entry location
  edges := [backedge, exitEdge]
}
def loopProof (entry : StorageState) : (loopVC entry).InductionObligations where
  base := by intro state initial location outside; rw [initial]
  backedge := by
    intro edge current following member target invariant path safe next
    simp only [loopVC, List.mem_cons, List.not_mem_nil, or_false] at member
    rcases member with rfl | rfl
    · subst following
      intro location outside
      exact (StorageState.set_other _ _ _ _ outside).trans (invariant location outside)
    · cases target
  edgeSafety := by
    intro edge state member _ _
    simp only [loopVC, List.mem_cons, List.not_mem_nil, or_false] at member
    rcases member with rfl | rfl <;> trivial
  exit := by
    intro edge current following member _ _ _ _ _
    simp only [loopVC, List.mem_cons, List.not_mem_nil, or_false] at member
    rcases member with rfl | rfl <;> trivial

def loopRelation (before after : LoopState) : Prop :=
  (loopVC before.storage).CertifiedControlRelation .successor before after

theorem loopResult (before after : LoopState) (run : loopRelation before after) :
    after.storage flag = .bool true ∧
    ∀ location, location ≠ flag → after.storage location = before.storage location := by
  rcases run with ⟨_, _, current, edge, invariant, member, target, path, _, next⟩
  simp only [loopVC, List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl
  · cases target
  · cases next
    exact ⟨path, invariant⟩

def loopFrame : CertifiedWriteFrame loopRelation := {
  footprint := fun location => location = flag
  preservesOutside := fun before after run location outside =>
    (loopResult before after run).2 location outside
}

def loopSemantics : TimelineSemantics Formula StorageLocation Expr Unit := {
  storage := leafSemantics.storage
  evalFormula := leafSemantics.evalFormula
  childRelation := fun _ before => loopRelation before.loop
  childFrame := fun _ _ => loopFrame
}
def loopCallee : CallBody Formula StorageLocation Expr Unit := {
  signature := boolSignature
  semantics := fun _ => loopSemantics
  paths := [{
    events := [.child ()]
    returns := .direct .success [fun state => some (state.loop.storage flag)]
  }]
  requires := fun _ => True
}

def loopCertificate : loopCallee.Certificate where
  loopResults := by
    intro input typed required initial entry path member before child bindings after events
    simp only [loopCallee, List.mem_singleton] at member
    subst path
    have present : TimelineEvent.childResult child bindings ∈ before ++ .childResult child bindings :: after := by simp
    rw [← events] at present
    simp at present
  calls := by
    intro input typed requires initial entry path member before invocation after events
    simp only [loopCallee, List.mem_singleton] at member
    subst path
    cases before with
    | nil => simp at events
    | cons head tail =>
      simp only [List.cons_append, List.cons.injEq] at events
      have : tail ++ TimelineEvent.call invocation :: after = [] := events.2.symm
      simp at this
  safety := by
    intro input typed requires initial entry path member before formula after events
    simp only [loopCallee, List.mem_singleton] at member
    subst path
    cases before with
    | nil => simp at events
    | cons head tail =>
      simp only [List.cons_append, List.cons.injEq] at events
      have : tail ++ TimelineEvent.check formula :: after = [] := events.2.symm
      simp at this
  returns := by
    intro input initial path result typed requires entry member trace
    simp only [loopCallee, List.mem_singleton] at member
    subst path
    cases trace with
    | cons event rest =>
      cases rest
      cases event with
      | child run =>
        have stored := (loopResult _ _ run).1
        refine ⟨{ outcome := .success, payload := [.bool true] }, ?_, ?_⟩
        · simp [callEvaluateReturn?, callEvaluateValues?, loopCallee, loopSemantics, leafSemantics,
            TimelineResult.comp, stored]
        · rfl

-- The concrete loop uses the one existing induction theorem, then the same
-- joint function interface as the leaf callee above.
example (before after : LoopState)
    (run : (loopVC before.storage).ControlRelation .successor before after) :
    loopRelation before after :=
  control_relation_contained_in_certified_relation _ (loopProof _) _ _ _ run

example : CertifiedCall := leaf.certified leafCertificate
example : CertifiedCall := loopCallee.certified loopCertificate

def callingSemantics := { leafSemantics with calls := fun _ => some (leaf.certified leafCertificate) }

theorem invocationReady : invocation.ready callingSemantics.calls
    callingSemantics.storage.evalValue caller := by
  refine ⟨leaf.certified leafCertificate, [], [.spelling "bool"], rfl, rfl, rfl, ?_, ?_, rfl, rfl⟩
  · trivial
  · unfold CallInvocation.fresh
    decide

def resultFormula : CallFormula Formula Expr Empty := .ordinary fun state =>
  some (state.readValue? { file_id := 1, pattern_id := 1 } = some (.bool true))

theorem resultAfterCall (result : TimelineResult)
    (run : TimelineEventExec callingSemantics.withCallChecks (.call invocation) caller result) :
    timelineFormulaHolds callingSemantics.withCallChecks resultFormula result.state := by
  cases run with
  | call resolved arguments relation outcome arity fresh =>
    simp only [callingSemantics, TimelineSemantics.withCallChecks, Option.some.injEq] at resolved
    subst_vars
    have joint := leafJoint _ _ relation
    refine ⟨_, rfl, ?_⟩
    simp [resultFormula, StorageActionState.readValue?, CallInvocation.resume,
      invocation, caller, joint.1, lookupStorageRead]

-- The post-call checkpoint consumes the returned value of that same relation.
example : TimelineSafe callingSemantics.withCallChecks
    (checkedCallEvents invocation ++ [.check resultFormula]) caller := by
  apply TimelineSafe.cons
  · intro formula equal
    cases equal
    exact ⟨_, rfl, invocationReady⟩
  · intro result run
    cases run
    apply TimelineSafe.cons
    · intro formula equal; cases equal
    · intro result run
      apply TimelineSafe.cons
      · intro formula equal
        cases equal
        exact resultAfterCall result run
      · intro following trace
        simp [TimelineSafe]

-- A returned fact cannot discharge a checkpoint preceding the invocation.
example : ¬ TimelineSafe callingSemantics.withCallChecks
    (.check (.ordinary (fun _ => some False)) :: checkedCallEvents invocation) caller := by
  intro safe
  rcases safe [] _ _ rfl { state := caller } (.nil caller) with ⟨proposition, evaluated, holds⟩
  have equal : proposition = False := (Option.some.inj evaluated).symm
  exact equal ▸ holds

-- A wrong arity or aliased result slot cannot establish invocation readiness.
example : ¬ ({ invocation with results := [] }).ready callingSemantics.calls
    callingSemantics.storage.evalValue caller := by
  rintro ⟨callee, arguments, types, resolved, evaluated, typed, requires, fresh, returns, arity⟩
  simp [callingSemantics] at resolved
  subst callee
  simp [leaf, boolSignature, CallBody.certified, invocation] at returns
  subst types
  cases arity

example : ¬ ({ invocation with results := [{ file_id := 1, pattern_id := 0 }] }).ready
    callingSemantics.calls callingSemantics.storage.evalValue caller := by
  rintro ⟨callee, arguments, types, resolved, evaluated, typed, requires, fresh, returns, arity⟩
  simp [CallInvocation.fresh, caller] at fresh

def errorLeaf : CallBody Formula StorageLocation Expr Empty := {
  leaf with
  signature := { boolSignature with
    results := fun outcome => if outcome = .error 7 then some [.spelling "bool"] else none }
  paths := [{
    events := [.storage (.write flag (fun _ => some (.bool true)))]
    returns := .direct (.error 7) [fun state => some (state.loop.storage flag)]
  }]
}

def errorLeafCertificate : errorLeaf.Certificate where
  loopResults := by
    intro input typed required initial entry path member before child
    cases child
  safety := by
    intro input typed requires
    simpa [CallBody.safe, errorLeaf, leaf] using leafCertificate.safety input typed requires
  calls := by
    intro input typed requires
    simpa [CallBody.callsSafe, errorLeaf, leaf] using leafCertificate.calls input typed requires
  returns := by
    intro input initial path result typed requires entry member trace
    simp only [errorLeaf, List.mem_singleton] at member
    subst path
    have stored := leafTrace initial result trace
    refine ⟨{ outcome := .error 7, payload := [.bool true] }, ?_, rfl⟩
    simp [callEvaluateReturn?, callEvaluateValues?, errorLeaf, leaf, leafSemantics, stored]

def zeroStore : StorageState := fun _ => .bool false
def writeResult : StorageState := zeroStore.set flag (.bool true)

-- An error and its payload are produced together with the write. There is
-- no implicit rollback when the internal error value is returned.
example : errorLeaf.execution { arguments := [], storage := zeroStore } {
    outcome := .error 7, payload := [.bool true], storage := writeResult } := by
  let initial : StorageActionState := {
    loop := { locals := [], storage := zeroStore }, entryStorage := zeroStore }
  let following : StorageActionState := {
    initial with loop := { initial.loop with storage := writeResult } }
  refine ⟨initial, {
    events := [.storage (.write flag (fun _ => some (.bool true)))]
    returns := .direct (.error 7) [fun state => some (state.loop.storage flag)]
  }, { state := following, writes := fun location => location ∈ [flag] ∨ False },
    { outcome := .error 7, payload := [.bool true] }, rfl, by simp [errorLeaf], ?_, rfl, rfl⟩
  have action : execStorageAction? leafSemantics.storage
      (.write flag (fun _ => some (.bool true))) initial =
      some { state := following, writeTrace := [flag] } := rfl
  exact .cons (.storage action) (.nil following)

example : errorLeaf.signature.outputTyped {
    outcome := .success, payload := [.bool true], storage := writeResult } = false := by rfl

def loopInitial : LoopState := { locals := [], storage := zeroStore }
def loopFinal : LoopState := { loopInitial with storage := writeResult }

theorem loopActuallyExecutes : (loopVC zeroStore).ControlRelation .successor loopInitial loopFinal := by
  refine ⟨by decide, rfl, loopFinal, exitEdge, ?_, by simp [loopVC], rfl, rfl, trivial, rfl⟩
  exact .next (edge := backedge) (.refl loopInitial)
    (by simp [loopVC]) rfl rfl trivial rfl

-- This is a completed, one-iteration loop, rather than an empty relation.
example : loopCallee.execution { arguments := [], storage := zeroStore } {
    outcome := .success, payload := [.bool true], storage := writeResult } := by
  let initial : StorageActionState := {
    loop := loopInitial, entryStorage := zeroStore }
  let following : StorageActionState := { initial with loop := loopFinal }
  refine ⟨initial, {
    events := [.child ()]
    returns := .direct .success [fun state => some (state.loop.storage flag)]
  }, { state := following, writes := fun location => location = flag ∨ False },
    { outcome := .success, payload := [.bool true] }, rfl, by simp [loopCallee], ?_, rfl, rfl⟩
  have child : TimelineEventExec loopSemantics (.child ()) initial {
      state := following, writes := fun location => location = flag } :=
    .child (control_relation_contained_in_certified_relation _
      (loopProof _) _ _ _ loopActuallyExecutes)
  exact .cons child (.nil following)

example : boolSignature.inputTyped {
    arguments := [.bool true], storage := zeroStore } = false := by rfl
example : callValuesTyped [] [] = true := by rfl
example : callValuesTyped [] [.bool false] = false := by rfl

def restrictedLeaf : CallBody Formula StorageLocation Expr Empty := {
  leaf with requires := fun input => input.storage flag = .bool false }
def restrictedCertificate : restrictedLeaf.Certificate where
  loopResults := by
    intro input typed required initial entry path member before child
    cases child
  safety := fun input typed _ => leafCertificate.safety input typed trivial
  calls := fun input typed _ => leafCertificate.calls input typed trivial
  returns := fun input initial path result typed _ =>
    leafCertificate.returns input initial path result typed trivial

example : ¬ invocation.ready (fun _ => some (restrictedLeaf.certified restrictedCertificate))
    callingSemantics.storage.evalValue caller := by
  rintro ⟨callee, arguments, types, resolved, evaluated, typed, requires, _⟩
  cases resolved
  cases requires

example : ¬ invocation.ready (fun _ => none) callingSemantics.storage.evalValue caller := by
  rintro ⟨callee, arguments, types, resolved, _⟩
  cases resolved

def parameterId : FreeVarId := { file_id := 2, pattern_id := 1 }
def parameterSignature : CallSignature := {
  boolSignature with parameters := [(parameterId, .spelling "bool")] }

example : ((parameterSignature.entry? {
    arguments := [.bool true], storage := zeroStore }).getD caller).readValue? parameterId =
      some (.bool true) := by decide
example : parameterSignature.entry? { arguments := [], storage := zeroStore } = none := by rfl
example : parameterSignature.entry? {
    arguments := [.bool false, .bool true], storage := zeroStore } = none := by rfl
example : parameterSignature.entry? {
    arguments := [.integer (Ora.Integer.Value.ofNat (.unsigned .w256) 1)],
    storage := zeroStore } = none := by rfl

def voidBody : CallBody Formula StorageLocation Expr Empty := {
  signature := { boolSignature with
    results := fun outcome => if outcome = .success then some [] else none }
  semantics := fun _ => leafSemantics
  paths := [{ events := [], returns := .direct .success [] }]
  requires := fun _ => True
}
def voidCertificate : voidBody.Certificate where
  loopResults := by
    intro input typed required initial entry path member before child
    cases child
  safety := by
    intro input typed requires initial entry path member
    simp only [voidBody, List.mem_singleton] at member
    subst path
    simp [TimelineSafe]
  calls := by
    intro input typed requires initial entry path member before invocation after events
    simp only [voidBody, List.mem_singleton] at member
    subst path
    simp at events
  returns := by
    intro input initial path result typed requires entry member trace
    simp only [voidBody, List.mem_singleton] at member
    subst path
    exact ⟨{ outcome := .success, payload := [] }, rfl, rfl⟩

example : voidBody.execution { arguments := [], storage := zeroStore } {
    outcome := .success, payload := [], storage := zeroStore } := by
  let initial : StorageActionState := {
    loop := { locals := [], storage := zeroStore }, entryStorage := zeroStore }
  exact ⟨initial, { events := [], returns := .direct .success [] }, { state := initial },
    { outcome := .success, payload := [] }, rfl, by simp [voidBody], .nil initial, rfl, rfl⟩
example : voidBody.signature.outputTyped {
    outcome := .success, payload := [.bool false], storage := zeroStore } = false := by rfl

def errorExit (entry : StorageState) : Edge LoopState :=
  timelineEdge leafSemantics [] entry (fun state => some state.loop.locals)
    .errorReturn (fun _ => True) (fun state => some (some {
      outcome := .error 7, payload := [state.loop.storage flag] }))

def errorLoopVC (entry : StorageState) : LoopVC LoopState := {
  initial := fun state => state.storage = entry ∧ ∃ bit, entry flag = .bool bit
  invariant := fun state =>
    (∀ location, location ≠ flag → state.storage location = entry location) ∧
      ∃ bit, state.storage flag = .bool bit
  edges := [backedge, errorExit entry]
}

def errorLoopProof (entry : StorageState) : (errorLoopVC entry).InductionObligations where
  base := by
    rintro state ⟨equal, bit, typed⟩
    exact ⟨by intro location _; rw [equal], bit, equal ▸ typed⟩
  backedge := by
    intro edge current following member target invariant path safe next
    simp only [errorLoopVC, List.mem_cons, List.not_mem_nil, or_false] at member
    rcases member with rfl | rfl
    · subst following
      exact ⟨fun location outside =>
        (StorageState.set_other _ _ _ _ outside).trans (invariant.1 location outside),
        true, StorageState.set_self current.storage flag (.bool true)⟩
    · cases target
  edgeSafety := by
    intro edge state member invariant path
    simp only [errorLoopVC, List.mem_cons, List.not_mem_nil, or_false] at member
    rcases member with rfl | rfl
    · trivial
    · simp [errorExit, timelineEdge, TimelineSafe]
  exit := by
    intro edge current following member target invariant path safe next
    simp only [errorLoopVC, List.mem_cons, List.not_mem_nil, or_false] at member
    rcases member with rfl | rfl <;> trivial

def errorLoopRelation (before after : LoopState) : Prop :=
  (errorLoopVC before.storage).CertifiedControlRelation .errorReturn before after

theorem errorLoopResult (before after : LoopState) (run : errorLoopRelation before after) :
    (∃ bit, after.returned = some { outcome := .error 7, payload := [.bool bit] }) ∧
    ∀ location, location ≠ flag → after.storage location = before.storage location := by
  rcases run with ⟨_, _, current, edge, invariant, member, target, path, safe, next⟩
  simp only [errorLoopVC, List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl
  · cases target
  · rcases next with ⟨result, locals, returned, trace, hLocals, hReturned, hAfter⟩
    cases trace
    simp only [Option.some.injEq] at hLocals hReturned
    subst locals
    subst returned
    subst after
    rcases invariant with ⟨outside, bit, typed⟩
    exact ⟨⟨bit, by simp only [typed]⟩, outside⟩

def errorLoopFrame : CertifiedWriteFrame errorLoopRelation := {
  footprint := fun location => location = flag
  preservesOutside := fun before after run location outside =>
    (errorLoopResult before after run).2 location outside
}
def errorLoopSemantics : TimelineSemantics Formula StorageLocation Expr Unit := {
  storage := leafSemantics.storage
  evalFormula := leafSemantics.evalFormula
  childRelation := fun _ before => errorLoopRelation before.loop
  childFrame := fun _ _ => errorLoopFrame
}
def earlyErrorCallee : CallBody Formula StorageLocation Expr Unit := {
  signature := errorLeaf.signature
  semantics := fun _ => errorLoopSemantics
  paths := [{ events := [.child ()], returns := .loopExit }]
  requires := fun input => ∃ bit, input.storage flag = .bool bit
}
def earlyErrorCertificate : earlyErrorCallee.Certificate where
  loopResults := by
    intro input typed required initial entry path member before child bindings after events
    simp only [earlyErrorCallee, List.mem_singleton] at member
    subst path
    have present : TimelineEvent.childResult child bindings ∈ before ++ .childResult child bindings :: after := by simp
    rw [← events] at present
    simp at present
  safety := by
    intro input typed requires initial entry path member before formula after events
    simp only [earlyErrorCallee, List.mem_singleton] at member
    subst path
    cases before with
    | nil => simp at events
    | cons head tail =>
      simp only [List.cons_append, List.cons.injEq] at events
      have : tail ++ TimelineEvent.check formula :: after = [] := events.2.symm
      simp at this
  calls := by
    intro input typed requires initial entry path member before invocation after events
    simp only [earlyErrorCallee, List.mem_singleton] at member
    subst path
    cases before with
    | nil => simp at events
    | cons head tail =>
      simp only [List.cons_append, List.cons.injEq] at events
      have : tail ++ TimelineEvent.call invocation :: after = [] := events.2.symm
      simp at this
  returns := by
    intro input initial path result typed requires entry member trace
    simp only [earlyErrorCallee, List.mem_singleton] at member
    subst path
    cases trace with
    | cons event rest =>
      cases rest
      cases event with
      | child run =>
        rcases (errorLoopResult _ _ run).1 with ⟨bit, returned⟩
        exact ⟨{ outcome := .error 7, payload := [.bool bit] }, returned, rfl⟩

def errorLoopFinal : LoopState := { loopFinal with
  returned := some { outcome := .error 7, payload := [.bool true] } }

theorem errorLoopActuallyExecutes :
    (errorLoopVC zeroStore).ControlRelation .errorReturn loopInitial errorLoopFinal := by
  refine ⟨by decide, ⟨rfl, false, rfl⟩, loopFinal, errorExit zeroStore, ?_,
    by simp [errorLoopVC], rfl, trivial, ?_, ?_⟩
  · exact .next (edge := backedge) (.refl loopInitial)
      (by simp [errorLoopVC]) rfl rfl trivial rfl
  · simp [errorExit, timelineEdge, TimelineSafe]
  · exact ⟨{ state := { loop := loopFinal, entryStorage := zeroStore } }, [],
      some { outcome := .error 7, payload := [.bool true] }, .nil _, rfl, rfl, rfl⟩

example : earlyErrorCallee.execution { arguments := [], storage := zeroStore } {
    outcome := .error 7, payload := [.bool true], storage := writeResult } := by
  let initial : StorageActionState := { loop := loopInitial, entryStorage := zeroStore }
  let following : StorageActionState := { initial with loop := errorLoopFinal }
  have child : TimelineEventExec errorLoopSemantics (.child ()) initial {
      state := following, writes := fun location => location = flag } :=
    .child (control_relation_contained_in_certified_relation _
      (errorLoopProof _) _ _ _ errorLoopActuallyExecutes)
  exact ⟨initial, { events := [.child ()], returns := .loopExit },
    { state := following, writes := fun location => location = flag ∨ False },
    { outcome := .error 7, payload := [.bool true] }, rfl, by simp [earlyErrorCallee],
    .cons child (.nil following), rfl, rfl⟩

def concreteErrorSemantics : TimelineSemantics Formula StorageLocation Expr Unit := {
  errorLoopSemantics with
  childRelation := fun _ before after =>
    (errorLoopVC before.loop.storage).ControlRelation .errorReturn before.loop after
  childFrame := fun _ _ => {
    footprint := errorLoopFrame.footprint
    preservesOutside := fun before after run =>
      errorLoopFrame.preservesOutside before after
        (control_relation_contained_in_certified_relation _ (errorLoopProof _) _ _ _ run)
  }
}
def concreteErrorCallee := { earlyErrorCallee with semantics := fun _ => concreteErrorSemantics }

theorem errorEventsRefine (event : EventOf Formula StorageLocation Expr Unit)
    (initial : StorageActionState) (result : TimelineResult)
    (run : TimelineEventExec concreteErrorSemantics event initial result) :
    ∃ following, TimelineEventExec errorLoopSemantics event initial following ∧
      following.state = result.state := by
  cases run with
  | assume evaluated holds => exact ⟨_, .assume evaluated holds, rfl⟩
  | check evaluated => exact ⟨_, .check evaluated, rfl⟩
  | storage run => exact ⟨_, .storage run, rfl⟩
  | indexedRead run => exact ⟨_, .indexedRead run, rfl⟩
  | child run =>
    exact ⟨_, .child (control_relation_contained_in_certified_relation _
      (errorLoopProof _) _ _ _ run), rfl⟩
  | childResult run hResume =>
    exact ⟨_, .childResult (control_relation_contained_in_certified_relation _
      (errorLoopProof _) _ _ _ run) hResume, rfl⟩
  | call resolved => cases resolved

-- The existing induction theorem lifts ALL concrete executions and ALL
-- reachable check prefixes into the certified function relation.
example (input : CallInput) (output : CallOutput)
    (run : concreteErrorCallee.execution input output) : earlyErrorCallee.execution input output :=
  CallBody.execution_refines concreteErrorCallee earlyErrorCallee rfl rfl
    input rfl errorEventsRefine output run

example (input : CallInput) (typed : earlyErrorCallee.signature.inputTyped input = true)
    (requires : earlyErrorCallee.requires input) :
    concreteErrorCallee.safe input ∧ concreteErrorCallee.callsSafe input ∧ concreteErrorCallee.loopResultsSafe input :=
  CallBody.safe_refines concreteErrorCallee earlyErrorCallee rfl rfl input rfl rfl rfl
    errorEventsRefine (fun _ _ _ run => control_relation_contained_in_certified_relation _
      (errorLoopProof _) _ _ _ run) ⟨earlyErrorCertificate.safety input typed requires,
      earlyErrorCertificate.calls input typed requires, earlyErrorCertificate.loopResults input typed requires⟩

def prefixLoopSemantics : TimelineSemantics Formula StorageLocation Expr Unit := {
  storage := leafSemantics.storage
  evalFormula := leafSemantics.evalFormula
  calls := callingSemantics.calls
  childRelation := fun _ before after =>
    before.readValue? { file_id := 1, pattern_id := 1 } = some (.bool true) ∧
      loopRelation before.loop after
  childFrame := fun _ _ => {
    footprint := loopFrame.footprint
    preservesOutside := fun before after run => loopFrame.preservesOutside before after run.2
  }
}

-- A call before a loop feeds its actual result binding into the loop's
-- certified relation. The child observes the full preceding timeline state.
example (result : TimelineResult) (following : LoopState)
    (hCall : TimelineEventExec prefixLoopSemantics (.call invocation) caller result)
    (hLoop : loopRelation result.state.loop following) :
    TimelineEventExec prefixLoopSemantics (.child ()) result.state {
      state := { result.state with loop := following }, writes := loopFrame.footprint } := by
  cases hCall with
  | call resolved arguments relation outcome arity fresh =>
    simp only [prefixLoopSemantics, callingSemantics, Option.some.injEq] at resolved
    subst_vars
    have joint := leafJoint _ _ relation
    apply TimelineEventExec.child
    refine ⟨?_, hLoop⟩
    simp [StorageActionState.readValue?, CallInvocation.resume, invocation, caller,
      joint.1, lookupStorageRead]

end Ora.Loop.CallTests


namespace Ora.Loop.CallContextTests
open Ora.Obligation Ora.Loop.CallTests

-- This helper has no source requires. Its actual loop contains the checked
-- access i < 4, so its induction obligation must be established at invocation.
-- The obligation is derived from the body; it is not a new source assumption.
def stepEdge (limit : Nat) : Edge LoopState := {
  path := fun state => state.locals.length < limit
  safe := fun state => state.locals.length < 4
  next := fun before after => after.locals.length = before.locals.length + 1 ∧
    after.storage = before.storage
  target := .backedge
  targetAssertion := fun _ => True
}
def doneEdge (limit : Nat) : Edge LoopState := {
  path := fun state => state.locals.length = limit
  safe := fun _ => True
  next := Eq
  target := .successor
  targetAssertion := fun _ => True
}
def boundedVC (limit : Nat) (entry : StorageState) : LoopVC LoopState := {
  initial := fun state => state.locals = [] ∧ state.storage = entry
  invariant := fun state => state.locals.length ≤ limit ∧ state.storage = entry
  edges := [stepEdge limit, doneEdge limit]
}
def boundedProof (limit : Nat) (bound : limit ≤ 4) (entry : StorageState) :
    (boundedVC limit entry).InductionObligations where
  base := by
    rintro state ⟨locals, storage⟩
    exact ⟨by simp [locals], storage⟩
  backedge := by
    intro edge before after member target invariant path safe next
    simp only [boundedVC, List.mem_cons, List.not_mem_nil, or_false] at member
    rcases member with rfl | rfl
    · exact ⟨by have := next.1; have := path; dsimp [stepEdge] at *; omega,
        next.2.trans invariant.2⟩
    · cases target
  edgeSafety := by
    intro edge state member invariant path
    simp only [boundedVC, List.mem_cons, List.not_mem_nil, or_false] at member
    rcases member with rfl | rfl
    · dsimp [stepEdge] at path ⊢
      omega
    · trivial
  exit := by
    intro edge before after member target invariant path safe next
    simp only [boundedVC, List.mem_cons, List.not_mem_nil, or_false] at member
    rcases member with rfl | rfl <;> trivial

def limitOf (input : CallInput) : Nat := match input.arguments with
  | [.integer value] => value.bits.toNat
  | _ => 0
def contextSignature : CallSignature := {
  parameters := [({ file_id := 82, pattern_id := 0 }, .spelling "u8")]
  results := boolSignature.results
}
def contextRelation (input : CallInput) (before after : LoopState) : Prop :=
  (boundedVC (limitOf input) before.storage).CertifiedControlRelation .successor before after

theorem contextStorage (input : CallInput) (before after : LoopState)
    (run : contextRelation input before after) : after.storage = before.storage := by
  rcases run with ⟨_, _, current, edge, invariant, member, target, path, safe, next⟩
  simp only [boundedVC, List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl
  · cases target
  · cases next
    exact invariant.2

def contextFrame (input : CallInput) : CertifiedWriteFrame (contextRelation input) := {
  footprint := fun _ => False
  preservesOutside := fun before after run location _ => congrFun (contextStorage _ _ _ run) location
}
def inductionAt (input : CallInput) : Formula := fun state =>
  some (boundedVC (limitOf input) state.loop.storage).InductionObligations
def contextSemantics (input : CallInput) : TimelineSemantics Formula StorageLocation Expr Unit := {
  storage := leafSemantics.storage
  evalFormula := leafSemantics.evalFormula
  childRelation := fun _ before => contextRelation input before.loop
  childFrame := fun _ _ => contextFrame input
}
def contextBody : CallBody Formula StorageLocation Expr Unit := {
  signature := contextSignature
  semantics := contextSemantics
  paths := [{
    events := [.check (fun state => inductionAt {
      arguments := state.reads.map StorageReadBinding.value, storage := state.loop.storage } state), .child ()]
    returns := .direct .success [fun _ => some (.bool true)]
  }]
  requires := fun input => True ∧ (boundedVC (limitOf input) input.storage).InductionObligations
}

def contextCertificate : contextBody.Certificate where
  loopResults := by
    intro input typed required initial entry path member before child bindings after events
    simp only [contextBody, List.mem_singleton] at member
    subst path
    have present : TimelineEvent.childResult child bindings ∈ before ++ .childResult child bindings :: after := by simp
    rw [← events] at present
    simp at present
  safety := by
    intro input typed required initial entry path member
    simp only [contextBody, List.mem_singleton] at member
    subst path
    change contextSignature.inputTyped input = true at typed
    simp only [contextBody, CallSignature.entry?, typed, if_true, Option.some.injEq] at entry
    subst initial
    apply TimelineSafe.cons
    · intro formula equal
      cases equal
      refine ⟨_, rfl, ?_⟩
      have arity := typed
      simp only [contextBody, contextSignature, CallSignature.inputTyped,
        callValuesTyped, Bool.and_eq_true, beq_iff_eq, List.map, List.length] at arity
      have length : input.arguments.length = 1 := arity.2.1.symm
      cases arguments : input.arguments with
      | nil => simp [arguments] at length
      | cons head tail =>
        have tailLength : tail.length + 1 = 1 := by
          simpa only [arguments, List.length_cons] using length
        have tailEmpty : tail = [] := List.length_eq_zero.mp (Nat.succ.inj tailLength)
        subst tail
        simpa [inductionAt, contextSignature, limitOf, arguments, leafSemantics] using required.2
    · intro result trace
      cases trace
      intro before formula after events
      cases before with
      | nil => simp at events
      | cons head tail => simp at events
  calls := by
    intro input typed required initial entry path member before invocation after events
    simp only [contextBody, List.mem_singleton] at member
    subst path
    have contains : TimelineEvent.call invocation ∈ before ++ TimelineEvent.call invocation :: after := by simp
    rw [← events] at contains
    simp at contains
  returns := by
    intro input initial path result typed required entry member trace
    simp only [contextBody, List.mem_singleton] at member
    subst path
    exact ⟨{ outcome := .success, payload := [.bool true] }, rfl, rfl⟩

def contextCall (value : BitVec 8) : CallInvocation Expr := {
  callee := "bounded", arguments := [fun _ => some (.integer {
    shape := .unsigned .w8, bits := value })]
  outcome := .success, results := [{ file_id := 82, pattern_id := 1 }]
}
def resolver : String → Option CertifiedCall := fun _ => some (contextBody.certified contextCertificate)

theorem callerBoundEstablishesReadiness (value : BitVec 8) (bound : value.toNat ≤ 4)
    (store : StorageState) :
    (contextCall value).ready resolver leafSemantics.storage.evalValue {
      loop := { locals := [], storage := store }, entryStorage := zeroStore } := by
  refine ⟨contextBody.certified contextCertificate, [.integer {
    shape := .unsigned .w8, bits := value }], [.spelling "bool"], rfl, rfl, rfl,
    ⟨trivial, boundedProof _ bound _⟩, ?_, rfl, rfl⟩
  unfold CallInvocation.fresh
  simp [contextCall]

-- The same input-dependent induction certificate transfers complete loop
-- executions to the callee relation through the existing induction theorem.
example (input : CallInput) (bound : limitOf input ≤ 4) (before after : LoopState)
    (run : (boundedVC (limitOf input) before.storage).ControlRelation .successor before after) :
    contextRelation input before after :=
  control_relation_contained_in_certified_relation _ (boundedProof _ bound _) _ _ _ run

theorem fiveHasNoCertificate (store : StorageState) :
    ¬ (boundedVC 5 store).InductionObligations := by
  intro proof
  let state : LoopState := {
    locals := List.replicate 4 (Ora.Integer.Value.ofNat (.unsigned .w8) 0), storage := store }
  have failed := proof.edgeSafety (stepEdge 5) state (by simp [boundedVC])
    (by exact ⟨by change 4 ≤ 5; decide, rfl⟩) (by change 4 < 5; decide)
  change 4 < 4 at failed
  omega

theorem fiveNotReady (store : StorageState) :
    ¬ (contextCall 5).ready resolver leafSemantics.storage.evalValue {
      loop := { locals := [], storage := store }, entryStorage := zeroStore } := by
  rintro ⟨callee, arguments, types, resolved, evaluated, typed, required, _⟩
  simp only [resolver, Option.some.injEq] at resolved
  subst callee
  have actual : arguments = [.integer { shape := .unsigned .w8, bits := (5 : BitVec 8) }] :=
    (Option.some.inj evaluated).symm
  subst arguments
  exact fiveHasNoCertificate store required.2

-- A positive bound has a completed execution as well as universal safety.
-- Return and final storage come from that one nonempty loop execution.
example (store : StorageState) : contextBody.execution {
    arguments := [.integer (Ora.Integer.Value.ofNat (.unsigned .w8) 1)], storage := store } {
      outcome := .success, payload := [.bool true], storage := store } := by
  let input : CallInput := {
    arguments := [.integer (Ora.Integer.Value.ofNat (.unsigned .w8) 1)], storage := store }
  let initial : StorageActionState := {
    loop := { locals := [], storage := store }, entryStorage := store,
    reads := [{
      id := { file_id := 82, pattern_id := 0 }
      value := .integer (Ora.Integer.Value.ofNat (.unsigned .w8) 1) }] }
  let final : LoopState := {
    locals := [Ora.Integer.Value.ofNat (.unsigned .w8) 0], storage := store }
  have run : (boundedVC 1 store).ControlRelation .successor initial.loop final := by
    refine ⟨by decide, ⟨rfl, rfl⟩, final, doneEdge 1, ?_, by simp [boundedVC], rfl,
      rfl, trivial, rfl⟩
    exact .next (edge := stepEdge 1) (.refl initial.loop)
      (by simp [boundedVC]) rfl (by change 0 < 1; decide) (by change 0 < 4; decide) ⟨rfl, rfl⟩
  have child : TimelineEventExec (contextSemantics input) (.child ()) initial {
      state := { initial with loop := final }, writes := fun _ => False } :=
    .child (control_relation_contained_in_certified_relation _ (boundedProof 1 (by decide) store) _ _ _ run)
  refine ⟨initial, {
    events := [.check (fun state => inductionAt {
      arguments := state.reads.map StorageReadBinding.value, storage := state.loop.storage } state), .child ()]
    returns := .direct .success [fun _ => some (.bool true)]
  }, _, { outcome := .success, payload := [.bool true] },
    rfl, by simp [contextBody], .cons (.check rfl) (.cons child (.nil _)), rfl, rfl⟩

-- A later false condition cannot erase the unmet induction requirement.
example (store : StorageState) :
    ¬ TimelineSafe ({ leafSemantics with calls := resolver }).withCallChecks
      (checkedCallEvents (contextCall 5) ++ [.assume (.ordinary (fun _ => some False))]) {
        loop := { locals := [], storage := store }, entryStorage := zeroStore } := by
  intro safe
  exact fiveNotReady store (TimelineSafe.call_ready_at_prefix _ [] _ _ _ _ safe (.nil _))

end Ora.Loop.CallContextTests

namespace Ora.Loop.LoopResultTests
open Ora.Obligation Ora.Loop.CallTests

def id (n : Nat) : FreeVarId := { file_id := 90, pattern_id := n }
def byte (n : Nat) : Ora.Integer.Value := Ora.Integer.Value.ofNat (.unsigned .w8) n
def values : List Ora.Integer.Value := [byte 3, byte 7]
def bindings : List LoopResultBinding := [
  { id := id 2, index := 1, ty := .spelling "u8" },
  { id := id 3, index := 0, ty := .spelling "u8" }]

def vc (entry : StorageState) : LoopVC LoopState := {
  loopVC entry with
  initial := fun state => state.storage = entry ∧ state.locals = values
  invariant := fun state => (loopVC entry).invariant state ∧ state.locals = values
}
def proof (entry : StorageState) : (vc entry).InductionObligations where
  base := by
    intro state initial
    exact ⟨(loopProof entry).base state initial.1, initial.2⟩
  backedge := by
    intro edge current following member target invariant path safe next
    refine ⟨(loopProof entry).backedge edge current following member target invariant.1 path safe next, ?_⟩
    simp only [vc, loopVC, List.mem_cons, List.not_mem_nil, or_false] at member
    rcases member with rfl | rfl
    · cases next
      exact invariant.2
    · cases target
  edgeSafety := fun edge state member invariant path =>
    (loopProof entry).edgeSafety edge state member invariant.1 path
  exit := fun edge current following member target invariant path safe next =>
    (loopProof entry).exit edge current following member target invariant.1 path safe next

def relation (before : StorageState) (after : LoopState) : Prop :=
  (vc before).CertifiedControlRelation .successor { locals := values, storage := before } after

theorem results (before : StorageState) (after : LoopState) (run : relation before after) :
    after.locals = values ∧ after.storage flag = .bool true ∧
      ∀ location, location ≠ flag → after.storage location = before location := by
  rcases run with ⟨_, _, current, edge, invariant, member, target, path, _, next⟩
  simp only [vc, loopVC, List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl
  · cases target
  · cases next
    exact ⟨invariant.2, path, invariant.1⟩

def semantics : TimelineSemantics Formula StorageLocation Expr Unit := {
  storage := leafSemantics.storage
  evalFormula := leafSemantics.evalFormula
  childRelation := fun _ before => relation before.loop.storage
  childFrame := fun _ _ => {
    footprint := fun location => location = flag
    preservesOutside := fun before after run location outside =>
      (results _ _ run).2.2 location outside
  }
}

def start : StorageActionState := {
  loop := { locals := [byte 8], storage := fun _ => .bool false }
  entryStorage := fun _ => .bool true
  reads := [{ id := id 1, value := .bool false }]
}
def finish : LoopState := {
  locals := values, storage := start.loop.storage.set flag (.bool true)
}
def resumed : StorageActionState := {
  start with
  loop := { start.loop with storage := finish.storage }
  reads := start.reads ++ [{ id := id 2, value := .integer (byte 7) },
    { id := id 3, value := .integer (byte 3) }]
}

theorem nonemptyRun : (vc start.loop.storage).ControlRelation .successor
    { locals := values, storage := start.loop.storage } finish := by
  refine ⟨by decide, ⟨rfl, rfl⟩, finish, exitEdge, ?_, by simp [vc, loopVC], rfl, ?_, trivial, rfl⟩
  · exact .next (edge := backedge) (.refl _) (by simp [vc, loopVC]) rfl rfl trivial rfl
  · change finish.storage flag = .bool true
    simpa only [finish] using StorageState.set_self start.loop.storage flag (.bool true)

theorem ready : loopResultsReady semantics () bindings start := by
  refine ⟨by decide, ?_⟩
  intro following run
  simp only [loopResultBindings?, (results _ _ run).1]
  decide

theorem trace : TimelineExec semantics.withCallChecks
    (checkedLoopResultEvents () bindings) start {
      state := resumed, writes := fun location => False ∨ (location = flag ∨ False) } := by
  have certified := control_relation_contained_in_certified_relation _
    (proof start.loop.storage) _ _ _ nonemptyRun
  exact .cons (.check rfl) (.cons (.childResult certified rfl) (.nil _))

example : resumed.readValue? (id 2) = some (.integer (byte 7)) := rfl
example : resumed.loop.locals = start.loop.locals := rfl
example : resumed.oldValue flag = .bool true := rfl
example : resumed.readValue? (id 1) = some (.bool false) := rfl
example : resumeLoop? [{ id := id 2, index := 2, ty := .spelling "u8" }] start finish = none := rfl
example : resumeLoop? [{ id := id 2, index := 0, ty := .spelling "i8" }] start finish = none := rfl
example : resumeLoop? [{ id := id 2, index := 0, ty := .spelling "bool" }] start finish = none := rfl
example : (resumeLoop? [] start finish).map (·.reads) = some start.reads := rfl

-- The conflicting binding fails at the prefix even if there is no final loop
-- state or the following path is false.
example :
    ¬ TimelineSafe semantics.withCallChecks
      (checkedLoopResultEvents () [{ id := id 1, index := 0, ty := .spelling "u8" }] ++
        [.assume (.ordinary (fun _ => some False))]) start := by
  intro safe
  have failed := TimelineSafe.loop_results_ready_at_prefix semantics [] _ _ _ _ _ safe (.nil _)
  exact (by decide : ¬ (start.reads.map StorageReadBinding.id ++ [id 1]).Nodup) failed.1

#print axioms trace
#print axioms TimelineExec.preserves_outside
#print axioms TimelineSafe.loop_results_ready_at_prefix
end Ora.Loop.LoopResultTests

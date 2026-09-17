/- Joint typed function-call results over the existing timeline semantics. -/

import Ora.Loop.Timeline

namespace Ora.Loop
open Ora.Obligation

inductive CallFormula (FormulaExpr ValueExpr : Type) (Child : Type) where
  | ordinary : FormulaExpr → CallFormula FormulaExpr ValueExpr Child
  | ready : CallInvocation ValueExpr → CallFormula FormulaExpr ValueExpr Child
  | loopResultsReady : Child → List LoopResultBinding → CallFormula FormulaExpr ValueExpr Child

def loopResultsReady (semantics : TimelineSemantics F L V C) (child : C)
    (bindings : List LoopResultBinding) (state : StorageActionState) : Prop :=
  (state.reads.map StorageReadBinding.id ++ bindings.map (·.id)).Nodup ∧
    ∀ following, semantics.childRelation child state following →
      (loopResultBindings? bindings following).isSome = true

def TimelineSemantics.withCallChecks (semantics : TimelineSemantics F L V C) :
    TimelineSemantics (CallFormula F V C) L V C := {
  storage := semantics.storage
  indexedRegion := semantics.indexedRegion
  childRelation := semantics.childRelation
  childFrame := semantics.childFrame
  calls := semantics.calls
  evalFormula := fun formula state => match formula with
    | .ordinary value => semantics.evalFormula value state
    | .ready invocation => some (invocation.ready semantics.calls semantics.storage.evalValue state)
    | .loopResultsReady child bindings => some (loopResultsReady semantics child bindings state)
}

/-- Canonical call rows expand atomically to this checked invocation pair.
The check is present even when no matching return outcome exists. -/
def checkedCallEvents (invocation : CallInvocation V) :
    List (EventOf (CallFormula F V C) L V C) := [.check (.ready invocation), .call invocation]

theorem TimelineSafe.call_ready_at_prefix (semantics : TimelineSemantics F L V C)
    (before after : List (EventOf (CallFormula F V C) L V C))
    (invocation : CallInvocation V) (initial : StorageActionState) (result : TimelineResult)
    (safe : TimelineSafe semantics.withCallChecks
      (before ++ checkedCallEvents invocation ++ after) initial)
    (hPrefix : TimelineExec semantics.withCallChecks before initial result) :
    invocation.ready semantics.calls semantics.storage.evalValue result.state := by
  rcases safe before (.ready invocation) (.call invocation :: after)
    (by simp [checkedCallEvents, List.append_assoc]) result hPrefix with ⟨proposition, evaluated, holds⟩
  have equal := Option.some.inj evaluated
  exact equal ▸ holds

/-- Result typing and freshness are checked at the child prefix, even if the
child has no completed execution or a later condition is false. -/
def checkedLoopResultEvents (child : C) (bindings : List LoopResultBinding) :
    List (EventOf (CallFormula F V C) L V C) :=
  [.check (.loopResultsReady child bindings), .childResult child bindings]

theorem TimelineSafe.loop_results_ready_at_prefix (semantics : TimelineSemantics F L V C)
    (before after : List (EventOf (CallFormula F V C) L V C))
    (child : C) (bindings : List LoopResultBinding) (initial : StorageActionState)
    (result : TimelineResult)
    (safe : TimelineSafe semantics.withCallChecks
      (before ++ checkedLoopResultEvents child bindings ++ after) initial)
    (hPrefix : TimelineExec semantics.withCallChecks before initial result) :
    loopResultsReady semantics child bindings result.state := by
  rcases safe before (.loopResultsReady child bindings) (.childResult child bindings :: after)
    (by simp [checkedLoopResultEvents, List.append_assoc]) result hPrefix with ⟨proposition, evaluated, holds⟩
  exact Option.some.inj evaluated ▸ holds

inductive CallReturnExpr (ValueExpr : Type) where
  | direct : CallOutcome → List ValueExpr → CallReturnExpr ValueExpr
  | loopExit

def callEvaluateReturn? (evaluate : V → StorageActionState → Option Value)
    (expression : CallReturnExpr V) (state : StorageActionState) : Option CallReturn :=
  match expression with
  | .direct outcome values => do
      return { outcome, payload := ← callEvaluateValues? evaluate values state }
  | .loopExit => state.loop.returned

structure CallPath (FormulaExpr LocationExpr ValueExpr Child : Type) where
  events : List (EventOf FormulaExpr LocationExpr ValueExpr Child)
  returns : CallReturnExpr ValueExpr

/-- Complete projected paths use the existing timeline, including its checked
loop dependencies. This is a return adapter, not another execution system. -/
structure CallBody (FormulaExpr LocationExpr ValueExpr Child : Type) where
  signature : CallSignature
  semantics : CallInput → TimelineSemantics FormulaExpr LocationExpr ValueExpr Child
  paths : List (CallPath FormulaExpr LocationExpr ValueExpr Child)
  requires : CallInput → Prop

def CallBody.execution
    (body : CallBody F L V C) (input : CallInput) (output : CallOutput) : Prop :=
  ∃ initial path result returned,
    body.signature.entry? input = some initial ∧ path ∈ body.paths ∧
    TimelineExec (body.semantics input) path.events initial result ∧
    callEvaluateReturn? (body.semantics input).storage.evalValue path.returns result.state =
      some returned ∧
    output = { toCallReturn := returned, storage := result.state.loop.storage }

/-- Prefix safety is independent of completed executions. In particular, an
unreachable return does not erase a failing checkpoint on an earlier prefix. -/
def CallBody.safe (body : CallBody F L V C) (input : CallInput) : Prop :=
  ∀ initial, body.signature.entry? input = some initial →
    ∀ path, path ∈ body.paths → TimelineSafe (body.semantics input) path.events initial

/-- Callee preconditions and safety cannot be hidden by an absent return.
Canonical checked-call pairs discharge this from their preceding checkpoint. -/
def CallBody.callsSafe (body : CallBody F L V C) (input : CallInput) : Prop :=
  ∀ initial, body.signature.entry? input = some initial → ∀ path, path ∈ body.paths →
    ∀ before invocation after, path.events = before ++ .call invocation :: after →
      ∀ result, TimelineExec (body.semantics input) before initial result →
        invocation.ready (body.semantics input).calls
          (body.semantics input).storage.evalValue result.state

/-- Result binding must be defined for every related child outcome. A bad
binding cannot delete the execution that would reach a later check. -/
def CallBody.loopResultsSafe (body : CallBody F L V C) (input : CallInput) : Prop :=
  ∀ initial, body.signature.entry? input = some initial → ∀ path, path ∈ body.paths →
    ∀ before child bindings after, path.events = before ++ .childResult child bindings :: after →
      ∀ result, TimelineExec (body.semantics input) before initial result →
        loopResultsReady (body.semantics input) child bindings result.state

/-- The frame is obtained from executed direct actions and certified child
frames, never from declared `modifies` metadata. -/
def CallBody.footprint
    (body : CallBody F L V C) (input : CallInput) (location : StorageLocation) : Prop :=
  ∃ initial path result,
    body.signature.entry? input = some initial ∧ path ∈ body.paths ∧
    TimelineExec (body.semantics input) path.events initial result ∧ result.writes location

theorem CallBody.preservesOutside
    (body : CallBody F L V C) (input : CallInput) (output : CallOutput)
    (execution : body.execution input output) (location : StorageLocation)
    (outside : ¬ body.footprint input location) :
    output.storage location = input.storage location := by
  rcases execution with ⟨initial, path, result, payload, entry, member, trace, _, rfl⟩
  have preserved := TimelineExec.preserves_outside (body.semantics input) path.events
    initial result location trace (fun written =>
      outside ⟨initial, path, result, entry, member, trace, written⟩)
  simp only [CallSignature.entry?] at entry
  split at entry
  · cases Option.some.inj entry
    exact preserved
  · simp at entry

/-- Checking return evaluation separately prevents a missing or mistyped
return from removing the execution that should expose the error. -/
structure CallBody.Certificate (body : CallBody F L V C) : Prop where
  safety : ∀ input, body.signature.inputTyped input = true → body.requires input →
    body.safe input
  calls : ∀ input, body.signature.inputTyped input = true → body.requires input →
    body.callsSafe input
  loopResults : ∀ input, body.signature.inputTyped input = true → body.requires input →
    body.loopResultsSafe input
  returns : ∀ input initial path result,
    body.signature.inputTyped input = true → body.requires input →
    body.signature.entry? input = some initial → path ∈ body.paths →
    TimelineExec (body.semantics input) path.events initial result →
    ∃ returned,
      callEvaluateReturn? (body.semantics input).storage.evalValue path.returns result.state =
        some returned ∧
      body.signature.outputTyped {
        toCallReturn := returned, storage := result.state.loop.storage } = true

theorem CallBody.Certificate.outputTyped
    (body : CallBody F L V C) (certificate : body.Certificate)
    (input : CallInput) (output : CallOutput)
    (typed : body.signature.inputTyped input = true) (requires : body.requires input)
    (execution : body.execution input output) : body.signature.outputTyped output = true := by
  rcases execution with ⟨initial, path, result, payload, entry, member, trace, evaluated, rfl⟩
  rcases certificate.returns input initial path result typed requires entry member trace with
    ⟨actual, hActual, hTyped⟩
  have equal := Option.some.inj (evaluated.symm.trans hActual)
  subst actual
  exact hTyped

def CallBody.certified (body : CallBody F L V C) (certificate : body.Certificate) :
    CertifiedCall where
  signature := body.signature
  requires := body.requires
  relation := body.execution
  safe := fun input => body.safe input ∧ body.callsSafe input ∧ body.loopResultsSafe input
  footprint := body.footprint
  safety := fun input typed requires =>
    ⟨certificate.safety input typed requires, certificate.calls input typed requires,
      certificate.loopResults input typed requires⟩
  typed := certificate.outputTyped body
  preservesOutside := body.preservesOutside

/-- Event refinement composes with the same timeline execution. A certified
loop's invariant-exit relation therefore supplies the same function interface
as a leaf timeline, using `control_relation_contained_in_certified_relation`. -/
theorem TimelineExec.refine
    (concrete abstract : TimelineSemantics F L V C)
    (eventRefines : ∀ event initial result,
      TimelineEventExec concrete event initial result →
      ∃ following, TimelineEventExec abstract event initial following ∧
        following.state = result.state)
    (events : List (EventOf F L V C)) (initial : StorageActionState)
    (result : TimelineResult) (execution : TimelineExec concrete events initial result) :
    ∃ following, TimelineExec abstract events initial following ∧
      following.state = result.state := by
  induction execution with
  | nil state => exact ⟨{ state }, .nil state, rfl⟩
  | @cons event rest initial first following hEvent hRest ih =>
    rcases eventRefines event initial first hEvent with ⟨abstractFirst, hFirst, hState⟩
    rcases ih with ⟨abstractRest, hAbstractRest, hFinal⟩
    exact ⟨abstractFirst.comp abstractRest,
      .cons hFirst (hState ▸ hAbstractRest), hFinal⟩

theorem CallBody.execution_refines
    (concrete abstract : CallBody F L V C)
    (signature : concrete.signature = abstract.signature)
    (paths : concrete.paths = abstract.paths)
    (input : CallInput)
    (values : (concrete.semantics input).storage.evalValue =
      (abstract.semantics input).storage.evalValue)
    (eventRefines : ∀ event initial result,
      TimelineEventExec (concrete.semantics input) event initial result →
      ∃ following, TimelineEventExec (abstract.semantics input) event initial following ∧
        following.state = result.state)
    (output : CallOutput)
    (execution : concrete.execution input output) : abstract.execution input output := by
  rcases execution with ⟨initial, path, result, payload, entry, member, trace, evaluated, rfl⟩
  rcases TimelineExec.refine _ _ eventRefines _ _ _ trace with ⟨following, hTrace, hState⟩
  exact ⟨initial, path, following, payload, signature ▸ entry, paths ▸ member, hTrace,
    by simpa only [hState, ← values] using evaluated, by simp only [hState]⟩

theorem CallBody.safe_refines
    (concrete abstract : CallBody F L V C)
    (signature : concrete.signature = abstract.signature)
    (paths : concrete.paths = abstract.paths)
    (input : CallInput)
    (formulas : (concrete.semantics input).evalFormula = (abstract.semantics input).evalFormula)
    (values : (concrete.semantics input).storage.evalValue = (abstract.semantics input).storage.evalValue)
    (calls : (concrete.semantics input).calls = (abstract.semantics input).calls)
    (eventRefines : ∀ event initial result,
      TimelineEventExec (concrete.semantics input) event initial result →
      ∃ following, TimelineEventExec (abstract.semantics input) event initial following ∧
        following.state = result.state)
    (childRefines : ∀ child before after,
      (concrete.semantics input).childRelation child before after →
      (abstract.semantics input).childRelation child before after)
    (safe : abstract.safe input ∧ abstract.callsSafe input ∧ abstract.loopResultsSafe input) :
    concrete.safe input ∧ concrete.callsSafe input ∧ concrete.loopResultsSafe input := by
  constructor
  · intro initial entry path member before formula after events result trace
    rcases TimelineExec.refine _ _ eventRefines _ _ _ trace with ⟨following, hTrace, hState⟩
    have checked := safe.1 initial (signature ▸ entry) path (paths ▸ member)
      before formula after events following hTrace
    simpa only [timelineFormulaHolds, hState, ← formulas] using checked
  constructor
  · intro initial entry path member before invocation after events result trace
    rcases TimelineExec.refine _ _ eventRefines _ _ _ trace with ⟨following, hTrace, hState⟩
    have ready := safe.2.1 initial (signature ▸ entry) path (paths ▸ member)
      before invocation after events following hTrace
    simpa only [hState, ← values, ← calls] using ready

  · intro initial entry path member before child bindings after events result trace
    rcases TimelineExec.refine _ _ eventRefines _ _ _ trace with ⟨following, hTrace, hState⟩
    have ready := safe.2.2 initial (signature ▸ entry) path (paths ▸ member)
      before child bindings after events following hTrace
    rw [hState] at ready
    exact ⟨ready.1, fun final related => ready.2 final (childRefines _ _ _ related)⟩

end Ora.Loop

import Ora.Loop.CallModelTests
import Ora.Loop.StorageCertificate

namespace Ora.Loop.CallRowTests
open Ora.Obligation

def b : TyRef := .spelling "bool"
def ident : FreeVarId := { file_id := 81, pattern_id := 1 }
def resultVar : VariableRow := { index := 0, id := ident, ty := b }
def leaf : FunctionSummaryRow := {
  id := 1, owner := "leaf", structuralKey := "leaf",
  outcomes := [{ outcome := .success, types := [b] }, { outcome := .error 7, types := [] }],
  paths := [{ id := 0, structuralKey := "return", returns := .direct .success [{ value := { formula := .term 0 }, ty := b }] }]
}
def invocation : CallEventRow := {
  calleeSummaryId := 1, calleeVcSha256 := String.mk (List.replicate 64 '1'),
  outcome := .success, results := [resultVar]
}
example : leaf.shapeSupported = true := by decide
def lexicalLoop (bindings : Option (List VariableRow)) : FunctionSummaryRow := {
  leaf with paths := [{
    id := 0
    structuralKey := "loop"
    returns := .direct (.error 7) []
    timeline := [{
      id := 0
      structuralKey := "loop"
      payload := .summaryDependency {
        id := 0
        structuralKey := "loop"
        summaryId := 2
        loopVcSha256 := String.mk (List.replicate 64 '2')
        target := .successor
        writeFootprintComplete := true
        resultBindings := bindings } }] }] }
example : (lexicalLoop none).shapeSupported = false := by decide
example : (lexicalLoop (some [])).shapeSupported = true := by decide
example : callSignatureSupported leaf invocation = true := by decide
example : callSignatureSupported leaf { invocation with results := [] } = false := by decide
example : callSignatureSupported leaf { invocation with outcome := .error 7, results := [] } = true := by decide
example : callSignatureSupported leaf { invocation with outcome := .error 8, results := [] } = false := by decide
example : callSignatureSupported leaf { invocation with arguments := [
    { value := { formula := .term 0 }, ty := b }] } = false := by decide
example : callSignatureSupported leaf { invocation with results := [
    { resultVar with ty := .spelling "u8" }] } = false := by decide
example : ({ leaf with outcomes := leaf.outcomes ++ leaf.outcomes }).shapeSupported = false := by decide
example : ({ leaf with parameters := [resultVar, resultVar] }).shapeSupported = false := by decide
example : ({ leaf with paths := [] }).shapeSupported = false := by decide
example : ({ leaf with paths := [{ id := 0, structuralKey := "return", returns := .direct .success [] }] }).shapeSupported = false := by decide
example : ({ leaf with paths := [{ id := 0, structuralKey := "return", returns := .loopExit }] }).shapeSupported = false := by decide
example : callReturnReason? [] (.direct .success [{ value := { formula := .term 0, readResults := [ident] }, ty := b }]) = some .readIdentityForeign := by decide
example : directStorageTimeline? [{ id := 0, structuralKey := "call", payload := .call invocation }] = none := by rfl

def entryRow : FunctionSummaryRow := { leaf with
  parameters := [resultVar], moduleName := some "m", contractName := some "C"
  assumptionIds := [1], entryScope := some "contract:1", entryInvariantIds := [3, 4] }
def entryManifest : Manifest := {
  terms := [.variable (.free { id := ident, name := "arg", ty := some b }), .placeRead { root := "flag", region := .storage }, .old 1,
    .boolLit false, .variable (.free { id := { ident with pattern_id := 99 }, name := "foreign", ty := some b }), .boolLit true,
    .binary { op := .eq, lhs := 1, rhs := 5 },
    .binary { op := .eq, lhs := 2, rhs := 5 }]
  assumptions := [
    { id := 1, owner := "leaf", moduleName := some "m", contractName := some "C",
      kind := .requires, formula := some (.term 0) },
    { id := 2, owner := "leaf", moduleName := some "m", contractName := some "D",
      kind := .requires, formula := some (.term 3) }]
  obligations := [
    { id := 3, owner := "contract:1", contractOwner := true, kind := .logical .contractInvariant (.term 6) },
    { id := 4, owner := "contract:1", contractOwner := true, kind := .logical .contractInvariant (.term 7) },
    { id := 5, owner := "contract:2", contractOwner := true, kind := .logical .contractInvariant (.term 3) }]
}
example : entryRow.entryFormulas? entryManifest = some [.term 0, .term 6, .term 7] := by decide
example : ({ entryRow with assumptionIds := [] }).entryFormulas? entryManifest = none := by decide
example : ({ entryRow with assumptionIds := [2] }).entryFormulas? entryManifest = none := by decide
example : ({ entryRow with assumptionIds := [1, 1] }).entryFormulas? entryManifest = none := by decide
example : ({ entryRow with moduleName := none }).entryFormulas? entryManifest = none := by decide
example : ({ entryRow with entryInvariantIds := [] }).entryFormulas? entryManifest = none := by decide
example : ({ entryRow with entryInvariantIds := [3, 3] }).entryFormulas? entryManifest = none := by decide
example : ({ entryRow with entryInvariantIds := [4, 3] }).entryFormulas? entryManifest = none := by decide
example : ({ entryRow with entryInvariantIds := [3, 5] }).entryFormulas? entryManifest = none := by decide
example : ({ entryRow with entryInvariantIds := [3, 99] }).entryFormulas? entryManifest = none := by decide
example : ({ entryRow with entryScope := none }).entryFormulas? entryManifest = none := by decide
example : ({ entryRow with entryScope := some "" }).entryFormulas? entryManifest = none := by decide

def entryInput (argument store : Bool) : CallInput := {
  arguments := [.bool argument], storage := fun _ => .bool store }
example : entryRow.entryFactsHold entryManifest (entryInput true true) := by
  unfold FunctionSummaryRow.entryFactsHold
  change ∀ formula : FormulaRef, formula ∈ [.term 0, .term 6, .term 7] → _
  simp only [List.mem_cons, List.mem_singleton, List.not_mem_nil, forall_eq_or_imp, forall_eq, false_implies, implies_true, and_true]
  exact ⟨rfl, rfl, rfl⟩
example : ¬ entryRow.entryFactsHold entryManifest (entryInput false true) := by
  intro h
  have impossible : false = true := h (.term 0) (by simp)
  cases impossible
example : ¬ entryRow.entryFactsHold entryManifest (entryInput true false) := by
  intro h
  have impossible : false = true := h (.term 7) (by simp)
  cases impossible
-- A callee argument cannot be supplied by an ambient caller read binding.
example : ¬ entryRow.entryFactsHold
    { entryManifest with terms := [.variable (.free { id := { ident with pattern_id := 99 }, name := "foreign", ty := some b }),
      .boolLit true, .boolLit true] } (entryInput true true) := by
  intro h
  exact h (.term 0) (by simp)
example : ¬ entryRow.entryFactsHold entryManifest
    { arguments := [], storage := fun _ => .bool true } := by exact id
example : ¬ ({ entryRow with parameters := [{ resultVar with ty := .spelling "u8" }] }).entryFactsHold
    entryManifest (entryInput true true) := by exact id
example : ¬ entryRow.entryFactsHold
    { entryManifest with obligations := [
      { id := 3, owner := "contract:1", contractOwner := true, kind := .logical .contractInvariant (.term 3) },
      { id := 4, owner := "contract:1", contractOwner := true, kind := .logical .contractInvariant (.term 2) }] }
    (entryInput true true) := by
  intro h
  have impossible : false = true := h (.term 3) (by simp)
  cases impossible
#print axioms FunctionSummaryRow.entryFactsHold_requires_inputTyped
end Ora.Loop.CallRowTests

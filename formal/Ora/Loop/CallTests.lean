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
end Ora.Loop.CallRowTests

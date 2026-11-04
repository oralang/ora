import Ora.Loop.CallAdapter

namespace Ora.Loop.CallAdapterTests
open Ora.Obligation

def digest : String := String.mk (List.replicate 64 'a')
def stale : String := String.mk (List.replicate 64 'b')
def row : FunctionSummaryRow := {
  id := 1, owner := "leaf", structuralKey := "leaf"
  outcomes := [{ outcome := .success, types := [.spelling "bool"] }]
  paths := [{ id := 0, structuralKey := "return", returns := .direct .success [{ value := { formula := .term 0 }, ty := .spelling "bool" }] }]
  projectionComplete := true, writeFootprintComplete := true }
def manifest : CallAdapterManifest := {
  formulas := { terms := [.boolLit true] }, functions := [(digest, row)], loops := [] }
def body : AdapterBody := {
  signature := row.callSignature
  requires := row.entryFactsHold manifest.formulas
  paths := [{ events := [], returns := .direct .success [.term 0] }]
  semantics := fun _ => (adapterSemantics {}
    (directStorageContext manifest.formulas Env.empty [])).withCallChecks }

theorem bound : manifest.function {} digest row = .ok body := by rfl

theorem leafCertificate : body.Certificate where
  safety := by
    intro input typed requires initial entry path member before formula after events
    simp only [body, List.mem_singleton] at member
    subst path
    simp at events
  calls := by
    intro input typed requires initial entry path member before invocation after events
    simp only [body, List.mem_singleton] at member
    subst path
    simp at events
  loopResults := by
    intro input typed requires initial entry path member before child bindings after events
    simp only [body, List.mem_singleton] at member
    subst path
    simp at events
  returns := by
    intro input initial path result typed requires entry member trace
    simp only [body, List.mem_singleton] at member
    subst path
    cases trace
    change row.callSignature.inputTyped input = true at typed
    simp only [body, CallSignature.entry?, typed, if_true, Option.some.injEq] at entry
    subst initial
    refine ⟨{ outcome := .success, payload := [.bool true] }, ?_, ?_⟩
    · rfl
    · rfl

def installed : CallAdapterDependencies := {
  calls := [(adapterKey row.id digest, body.certified leafCertificate)] }
theorem installedBound : CallAdapterBuilt manifest installed :=
  .function .empty digest row body bound leafCertificate

def call : TimelineEventRow := {
  id := 0, structuralKey := "call", payload := .call {
    calleeSummaryId := 1, calleeVcSha256 := digest, outcome := .success
    results := [{ index := 0, id := { file_id := 1, pattern_id := 1 }, ty := .spelling "bool" }] } }

example : (manifest.function {} stale row).map (fun _ => ()) =
    .error "call_adapter_function_digest_mismatch" := by rfl
example : (manifest.function {} digest { row with owner := "foreign" }).map (fun _ => ()) =
    .error "call_adapter_function_foreign" := by rfl
example : (manifest.function {} digest { row with id := 2 }).map (fun _ => ()) =
    .error "call_adapter_function_missing" := by rfl
example : (manifest.event {} call).map (fun _ => ()) =
    .error "call_adapter_certificate_missing" := by rfl
example : (manifest.event installed call).map (fun _ => ()) = .ok () := by rfl
example : (manifest.function installed digest row).map (fun _ => ()) =
    .error "call_adapter_duplicate_certificate" := by rfl
example : ({ manifest with functions := [(stale, row)] }.event installed call).map (fun _ => ()) =
    .error "call_adapter_callee_digest_mismatch" := by rfl
example : ({ manifest with functions := [] }.event installed call).map (fun _ => ()) =
    .error "call_adapter_callee_missing" := by rfl
example : ({ manifest with functions := manifest.functions ++ manifest.functions }.function {} digest row).map (fun _ => ()) =
    .error "call_adapter_manifest_malformed" := by rfl
example : ({ manifest with functions := [("not-a-digest", row)] }.function {} "not-a-digest" row).map (fun _ => ()) =
    .error "call_adapter_manifest_malformed" := by rfl

#print axioms leafCertificate
#print axioms installedBound

end Ora.Loop.CallAdapterTests

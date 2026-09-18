import Ora.Loop.CallAdapterLoopTests
import Ora.Loop.CallTests

namespace Ora.Loop.CallAdapterSafetyTests
open Ora.Obligation
open CallAdapterTests

def falseFact : TimelineEventRow := {
  id := 0, structuralKey := "check:false"
  payload := .safetyCheckpoint { formulas := [{ formula := .term 1 }] } }
def falsePath : TimelineEventRow := {
  id := 1, structuralKey := "path:false"
  payload := .pathAssumption { formulas := [{ formula := .term 1 }] } }
def badRow : FunctionSummaryRow := { row with paths := [{
  id := 0, structuralKey := "checked-return", timeline := [falseFact, falsePath]
  returns := .direct .success [{ value := { formula := .term 0 }, ty := .spelling "bool" }] }] }
def badManifest : CallAdapterManifest := {
  formulas := { terms := [.boolLit true, .boolLit false] }
  functions := [(digest, badRow)], loops := [] }
def badBody : AdapterBody := { body with
  requires := badRow.entryFactsHold badManifest.formulas
  semantics := fun _ => (adapterSemantics {}
    (directStorageContext badManifest.formulas Env.empty [])).withCallChecks
  paths := [{
    events := [.check (.ordinary (.inl (.term 1))), .assume (.ordinary (.inl (.term 1)))]
    returns := .direct .success [.term 0] }] }
example : badManifest.function {} digest badRow = .ok badBody := by rfl

def input : CallInput := { arguments := [], storage := fun _ => .bool false }
def initial : StorageActionState := { loop := { locals := [], storage := input.storage }, entryStorage := input.storage }

theorem laterFalseCannotHideEarlierCheck : ¬ badBody.Certificate := by
  intro certificate
  have requires : badBody.requires input := by
    intro formula member
    cases member
  have safe := certificate.safety input rfl requires initial rfl _ (List.mem_singleton_self _)
  rcases safe [] (.ordinary (.inl (.term 1))) [.assume (.ordinary (.inl (.term 1)))] rfl
    { state := initial } (.nil initial) with ⟨proposition, evaluated, holds⟩
  change some (false = true) = some proposition at evaluated
  have impossible : false = true := Option.some.inj evaluated ▸ holds
  cases impossible

-- A declared wrong arm/payload never reaches the semantic certificate.
example : (manifest.event installed { call with payload := .call {
    calleeSummaryId := 1, calleeVcSha256 := digest, outcome := .error 7, results := [] } }).map (fun _ => ()) =
    .error "call_adapter_signature_mismatch" := by rfl

def restrictedRow : FunctionSummaryRow := { row with assumptionIds := [1] }
def restrictedManifest : CallAdapterManifest := {
  formulas := { terms := [.boolLit true, .boolLit false], assumptions := [
    { id := 1, owner := "leaf", kind := .requires, formula := some (.term 1) }] }
  functions := [(digest, restrictedRow)], loops := [] }
def restrictedBody : AdapterBody := { body with
  requires := restrictedRow.entryFactsHold restrictedManifest.formulas
  semantics := fun _ => (adapterSemantics {}
    (directStorageContext restrictedManifest.formulas Env.empty [])).withCallChecks }
theorem restrictedBound : restrictedManifest.function {} digest restrictedRow = .ok restrictedBody := by rfl
theorem restrictedRequiresFalse (input : CallInput)
    (typed : restrictedBody.signature.inputTyped input = true) : ¬ restrictedBody.requires input := by
  intro facts
  change restrictedRow.callSignature.inputTyped input = true at typed
  change restrictedRow.entryFactsHold restrictedManifest.formulas input at facts
  unfold FunctionSummaryRow.entryFactsHold at facts
  simp only [CallSignature.entry?, typed, if_true] at facts
  have impossible : false = true := facts (.term 1) (List.mem_singleton_self _)
  cases impossible
def restrictedCertificate : restrictedBody.Certificate where
  safety := fun input typed facts => (restrictedRequiresFalse input typed facts).elim
  calls := fun input typed facts => (restrictedRequiresFalse input typed facts).elim
  loopResults := fun input typed facts => (restrictedRequiresFalse input typed facts).elim
  returns := fun input _ _ _ typed facts => (restrictedRequiresFalse input typed facts).elim
def restricted := (CallAdapterCertificates.empty restrictedManifest).addFunction
  digest restrictedRow restrictedBody restrictedBound restrictedCertificate
def invocation : CallInvocation FormulaRef := {
  callee := adapterKey 1 digest, arguments := [], outcome := .success
  results := [{ file_id := 1, pattern_id := 1 }] }
def callerSemantics := adapterSemantics restricted.dependencies
  (directStorageContext restrictedManifest.formulas Env.empty [])
example : restrictedManifest.event restricted.dependencies call = .ok (checkedCallEvents invocation) := by rfl
theorem unmetPreconditionCannotHideBehindLaterFalse :
    ¬ TimelineSafe callerSemantics.withCallChecks
      (checkedCallEvents invocation ++ [.assume (.ordinary (.inl (.term 1)))]) initial := by
  intro safe
  have ready := TimelineSafe.call_ready_at_prefix callerSemantics []
    [.assume (.ordinary (.inl (.term 1)))] invocation initial { state := initial } safe (.nil initial)
  rcases ready with ⟨callee, arguments, types, resolved, evaluated, typed, required, _⟩
  change some (restrictedBody.certified restrictedCertificate) = some callee at resolved
  cases Option.some.inj resolved
  exact restrictedRequiresFalse _ typed required

-- The proof-binding equation rejects a body with an omitted path even at the same selector.
example : ¬ manifest.function {} digest row = .ok { body with paths := [] } := by
  intro forged
  have impossible := congrArg (fun result => result.map (fun body => body.paths.length)) forged
  change (Except.ok 1 : Except String Nat) = .ok 0 at impossible
  cases impossible
example : (manifest.event installed { call with payload := .call {
    calleeSummaryId := 1, calleeVcSha256 := digest, outcome := .success
    results := [{ index := 0, id := { file_id := 1, pattern_id := 1 }, ty := .spelling "u8" }] } }).map (fun _ => ()) =
    .error "call_adapter_signature_mismatch" := by rfl

-- Both installation orders of a mixed cycle fail before any certificate is installed.
def recursiveLoop : SummaryRow := { CallAdapterLoopTests.row with edges := [CallAdapterLoopTests.back, {
  CallAdapterLoopTests.done with timeline := [
    { id := 0, structuralKey := "path:true", payload := .pathAssumption { formulas := [{ formula := .term 0 }] } },
    { call with id := 1, payload := .call {
      calleeSummaryId := 3, calleeVcSha256 := CallAdapterLoopTests.functionDigest
      outcome := .success, results := [{ index := 0, id := { file_id := 91, pattern_id := 1 }, ty := .spelling "bool" }] } }] }] }
def cycleManifest : CallAdapterManifest := { CallAdapterLoopTests.manifest with
  loops := [(CallAdapterLoopTests.digest, recursiveLoop)] }
example : (cycleManifest.function {} CallAdapterLoopTests.functionDigest CallAdapterLoopTests.functionRow).map (fun _ => ()) =
    .error "call_adapter_loop_certificate_missing" := by rfl
example : (cycleManifest.loop {} CallAdapterLoopTests.digest recursiveLoop).map (fun _ => ()) =
    .error "call_adapter_certificate_missing" := by rfl

-- A full source requirement omitted from the loop's header-fact subset remains required.
def requiredOwner : FunctionSummaryRow := { CallAdapterLoopTests.functionRow with assumptionIds := [1] }
def requiredManifest : CallAdapterManifest := { CallAdapterLoopTests.manifest with
  functions := [(CallAdapterLoopTests.functionDigest, requiredOwner)]
  formulas := { CallAdapterLoopTests.manifest.formulas with assumptions := [
    { id := 1, owner := "looper", kind := .requires, formula := some (.term 2) }] } }
example : CallAdapterLoopTests.row.headerFacts = [] := by rfl
theorem falseSourceRequirementRetained (env : Env) (entry : StorageState) :
    ¬ adapterEntryFacts requiredManifest.formulas requiredOwner env entry := by
  rintro ⟨arguments, argumentsBound, facts⟩
  change some [] = some arguments at argumentsBound
  cases Option.some.inj argumentsBound
  have impossible : false = true := facts (.term 2) (List.mem_singleton_self _)
  cases impossible
example : ({ requiredManifest with functions := [(CallAdapterLoopTests.functionDigest,
    CallAdapterLoopTests.functionRow)] }.loop {} CallAdapterLoopTests.digest CallAdapterLoopTests.row).map (fun _ => ()) =
    .error "call_adapter_loop_owner_invalid" := by rfl

-- Invocation storage is the source-entry store even when the prefix changed storage.
def entryEnv : Env := bindStorageReads Env.empty [{ id := CallRowTests.ident, value := .bool true }]
example : adapterEntryFacts CallRowTests.entryManifest CallRowTests.entryRow entryEnv (fun _ => .bool true) := by
  refine ⟨[.bool true], rfl, ?_⟩
  change ∀ formula ∈ [FormulaRef.term 0, .term 6, .term 7], _
  simp only [List.mem_cons, List.mem_singleton, List.not_mem_nil, forall_eq_or_imp,
    forall_eq, false_implies, implies_true, and_true]
  exact ⟨rfl, rfl, rfl⟩
example : ¬ adapterEntryFacts CallRowTests.entryManifest CallRowTests.entryRow entryEnv (fun _ => .bool false) := by
  rintro ⟨arguments, argumentsBound, facts⟩
  change some [.bool true] = some arguments at argumentsBound
  cases Option.some.inj argumentsBound
  have impossible : false = true := facts (.term 7) (by simp)
  cases impossible

#print axioms laterFalseCannotHideEarlierCheck
#print axioms falseSourceRequirementRetained
#print axioms unmetPreconditionCannotHideBehindLaterFalse
end Ora.Loop.CallAdapterSafetyTests

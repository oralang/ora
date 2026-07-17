/-
Trusted checks for compiler-emitted source-accounting data.

The generated snapshot contributes primitive strings, booleans, and lists. It
does not contribute a theorem. This module independently enumerates the closed
vocabulary, executes the Lean policy table over its complete Cartesian product,
and replays the emitted kernel decision fixtures.
-/

import Ora.SourceAccounting.Theorems
import Ora.Generated.SourceAccountingSnapshot

namespace Ora.SourceAccountingSync

open Ora.SourceAccounting Ora.Generated

def allModes : List CompilationMode :=
  [.verifiedFull, .verifiedBasic, .unverifiedEmit]

def allFactKinds : List SourceFactKind :=
  [.requires, .guard, .ensures, .ensuresOk, .ensuresErr, .loopInvariant,
   .contractInvariant, .assert, .assume, .havoc, .modifies, .refinementGuard,
   .runtimeGuard]

def allFactOrigins : List FactOrigin := [.sourceSyntax, .semanticGenerated]

def allUseRoles : List UseRole :=
  [.proofTarget, .assumptionContext, .frameDirective, .stateDirective,
   .runtimeCondition]

def allTemplateActivations : List TemplateActivation :=
  [.runtimeBody, .symbolicCallBoundary, .comptimeBody]

def allHandlingKinds : List HandlingKind :=
  [.symbolic, .concreteTrue, .runtimeEnforced, .controlEliminated,
   .assumptionIncorporated, .frameValidated, .stateEffectIncorporated,
   .reducedScopeExcluded, .verificationDisabled, .rejected]

def allDispositions : List ExpansionDisposition :=
  [.symbolic, .foldCommitted, .foldAbandonedToSymbolic, .rejected]

def allFailureCodes : List FailureCode :=
  [.duplicateIdentity, .identityHashCollision, .unknownSite, .unknownExpansion,
   .unknownUse, .unknownEvidenceReference, .orphanEvidence,
   .evidenceCoverageMismatch, .missingSemanticSite, .unknownSemanticSite,
   .missingSemanticUse, .unexpectedSemanticUse, .invalidSiteRole,
   .invalidOwnerTemplate, .invalidExpansionActivation, .invalidExpansionParent,
   .invalidExpansionDisposition, .invalidControlGraph, .invalidFoldTrace,
   .abandonedFoldEvidence, .missingSymbolicReplacement,
   .invalidGeneratedFactDerivation, .missingHandling, .duplicateHandling,
   .rejectedHandling, .missingSymbolicObligation, .missingSymbolicQuery,
   .invalidSymbolicQueryKind, .missingAssumptionIncorporation,
   .missingRuntimeEnforcement, .missingFrameValidation, .missingStateEffect,
   .zeroConcreteChecks, .invalidConcreteCheckpoint, .falseConcreteTarget,
   .falseConcreteAssumption, .invalidControlElimination, .invalidFrameValidation,
   .invalidStateEffect, .handlingNotPermitted,
   .reducedScopeExcludedNotPermitted, .verificationDisabledNotPermitted]

def modeTag : CompilationMode → String
  | .verifiedFull => "verified_full"
  | .verifiedBasic => "verified_basic"
  | .unverifiedEmit => "unverified_emit"

def factKindTag : SourceFactKind → String
  | .requires => "requires" | .guard => "guard" | .ensures => "ensures"
  | .ensuresOk => "ensures_ok" | .ensuresErr => "ensures_err"
  | .loopInvariant => "loop_invariant" | .contractInvariant => "contract_invariant"
  | .assert => "assert" | .assume => "assume" | .havoc => "havoc"
  | .modifies => "modifies" | .refinementGuard => "refinement_guard"
  | .runtimeGuard => "runtime_guard"

def factOriginTag : FactOrigin → String
  | .sourceSyntax => "source_syntax"
  | .semanticGenerated => "semantic_generated"

def useRoleTag : UseRole → String
  | .proofTarget => "proof_target" | .assumptionContext => "assumption_context"
  | .frameDirective => "frame_directive" | .stateDirective => "state_directive"
  | .runtimeCondition => "runtime_condition"

def templateActivationTag : TemplateActivation → String
  | .runtimeBody => "runtime_body"
  | .symbolicCallBoundary => "symbolic_call_boundary"
  | .comptimeBody => "comptime_body"

def handlingKindTag : HandlingKind → String
  | .symbolic => "symbolic" | .concreteTrue => "concrete_true"
  | .runtimeEnforced => "runtime_enforced" | .controlEliminated => "control_eliminated"
  | .assumptionIncorporated => "assumption_incorporated"
  | .frameValidated => "frame_validated"
  | .stateEffectIncorporated => "state_effect_incorporated"
  | .reducedScopeExcluded => "reduced_scope_excluded"
  | .verificationDisabled => "verification_disabled" | .rejected => "rejected"

def dispositionTag : ExpansionDisposition → String
  | .symbolic => "symbolic" | .foldCommitted => "fold_committed"
  | .foldAbandonedToSymbolic => "fold_abandoned_to_symbolic"
  | .rejected => "rejected"

def failureCodeTag : FailureCode → String
  | .duplicateIdentity => "duplicate_identity"
  | .identityHashCollision => "identity_hash_collision"
  | .unknownSite => "unknown_site" | .unknownExpansion => "unknown_expansion"
  | .unknownUse => "unknown_use" | .unknownEvidenceReference => "unknown_evidence_reference"
  | .orphanEvidence => "orphan_evidence" | .evidenceCoverageMismatch => "evidence_coverage_mismatch"
  | .missingSemanticSite => "missing_semantic_site" | .unknownSemanticSite => "unknown_semantic_site"
  | .missingSemanticUse => "missing_semantic_use" | .unexpectedSemanticUse => "unexpected_semantic_use"
  | .invalidSiteRole => "invalid_site_role" | .invalidOwnerTemplate => "invalid_owner_template"
  | .invalidExpansionActivation => "invalid_expansion_activation"
  | .invalidExpansionParent => "invalid_expansion_parent"
  | .invalidExpansionDisposition => "invalid_expansion_disposition"
  | .invalidControlGraph => "invalid_control_graph" | .invalidFoldTrace => "invalid_fold_trace"
  | .abandonedFoldEvidence => "abandoned_fold_evidence"
  | .missingSymbolicReplacement => "missing_symbolic_replacement"
  | .invalidGeneratedFactDerivation => "invalid_generated_fact_derivation"
  | .missingHandling => "missing_handling" | .duplicateHandling => "duplicate_handling"
  | .rejectedHandling => "rejected_handling" | .missingSymbolicObligation => "missing_symbolic_obligation"
  | .missingSymbolicQuery => "missing_symbolic_query"
  | .invalidSymbolicQueryKind => "invalid_symbolic_query_kind"
  | .missingAssumptionIncorporation => "missing_assumption_incorporation"
  | .missingRuntimeEnforcement => "missing_runtime_enforcement"
  | .missingFrameValidation => "missing_frame_validation" | .missingStateEffect => "missing_state_effect"
  | .zeroConcreteChecks => "zero_concrete_checks" | .invalidConcreteCheckpoint => "invalid_concrete_checkpoint"
  | .falseConcreteTarget => "false_concrete_target" | .falseConcreteAssumption => "false_concrete_assumption"
  | .invalidControlElimination => "invalid_control_elimination"
  | .invalidFrameValidation => "invalid_frame_validation" | .invalidStateEffect => "invalid_state_effect"
  | .handlingNotPermitted => "handling_not_permitted"
  | .reducedScopeExcludedNotPermitted => "reduced_scope_excluded_not_permitted"
  | .verificationDisabledNotPermitted => "verification_disabled_not_permitted"

def decodeMode : String → Option CompilationMode
  | "verified_full" => some .verifiedFull
  | "verified_basic" => some .verifiedBasic
  | "unverified_emit" => some .unverifiedEmit
  | _ => none

def leanPolicyRow (mode : CompilationMode) (origin : FactOrigin) (kind : SourceFactKind) : String :=
  List.asString <|
  allUseRoles.flatMap fun role =>
  allHandlingKinds.flatMap fun handling =>
  allDispositions.map fun disposition =>
    if handlingPermitted mode origin kind role handling disposition then '1' else '0'

def decodeFactOrigin : String → Option FactOrigin
  | "source_syntax" => some .sourceSyntax
  | "semantic_generated" => some .semanticGenerated
  | _ => none

def decodeFactKind : String → Option SourceFactKind
  | "requires" => some .requires | "guard" => some .guard
  | "ensures" => some .ensures | "ensures_ok" => some .ensuresOk
  | "ensures_err" => some .ensuresErr | "loop_invariant" => some .loopInvariant
  | "contract_invariant" => some .contractInvariant | "assert" => some .assert
  | "assume" => some .assume | "havoc" => some .havoc
  | "modifies" => some .modifies | "refinement_guard" => some .refinementGuard
  | "runtime_guard" => some .runtimeGuard | _ => none

def policyRowMatches : String × String × String × String → Bool
  | (modeText, originText, kindText, expected) =>
      match decodeMode modeText, decodeFactOrigin originText, decodeFactKind kindText with
      | some mode, some origin, some kind => leanPolicyRow mode origin kind == expected
      | _, _, _ => false

def fixtureKey : SiteKey :=
  { path := "fixture.ora", owner := "function:run", startByte := 10,
    endByte := 20, kind := .loopInvariant, ordinal := 0 }

def fixtureDeclaredSite : DeclaredSite :=
  { id := 1, key := fixtureKey }

def fixtureTypedSite : TypedSite :=
  { id := 1, origin := .sourceSyntax, kind := .loopInvariant, key := fixtureKey,
    sourceFactId := some 10, declaredSiteId := some 1, derivationId := none }

def fixtureTemplate : OwnerTemplate :=
  { id := 1, activation := .comptimeBody, uses :=
    [{ siteId := 1, role := .proofTarget },
     { siteId := 1, role := .assumptionContext }] }

def fixtureExpansion : Expansion :=
  { id := 1, templateId := 1, activation := .speculativeFold,
    disposition := .foldCommitted, replacementExpansionId := none,
    rootRuntimeOwner := "function:run", identity := "call:run:0" }

def fixtureTargetUse : SourceUse :=
  { id := 1, siteId := 1, expansionId := 1, templateOrdinal := 0,
    role := .proofTarget }

def fixtureContextUse : SourceUse :=
  { id := 2, siteId := 1, expansionId := 1, templateOrdinal := 1,
    role := .assumptionContext }

def missingHandlingFixture : Manifest :=
  { declaredSites := [fixtureDeclaredSite],
    typedSites := [fixtureTypedSite],
    ownerTemplates := [fixtureTemplate],
    expansions := [fixtureExpansion],
    uses := [fixtureTargetUse, fixtureContextUse] }

def fixtureManifest : String → Option Manifest
  | "empty_verified_full" => some {}
  | "missing_invariant_handling" => some missingHandlingFixture
  | _ => none

def decisionRowMatches :
    String × String × Bool × Option String × List String → Bool
  | (name, modeText, expectedAccepted, expectedPrimary, expectedFailures) =>
      match fixtureManifest name, decodeMode modeText with
      | some manifest, some mode =>
          ((decide mode manifest == .accepted) == expectedAccepted) &&
          (primaryFailure mode manifest |>.map failureCodeTag) == expectedPrimary &&
          (failureList mode manifest |>.map failureCodeTag) == expectedFailures
      | _, _ => false

def failureWitnessRowMatches :
    String × Bool × Option String × List String → Bool
  | (target, accepted, primary, failures) =>
      sourceAccountingFailureCodeTags.contains target &&
      !accepted && !failures.isEmpty && failures.contains target &&
      primary == failures.head?

theorem generated_vocabulary_matches_closed_lean_vocabulary :
    sourceAccountingCompilationModeTags = allModes.map modeTag ∧
    sourceAccountingFactOriginTags = allFactOrigins.map factOriginTag ∧
    sourceAccountingFactKindTags = allFactKinds.map factKindTag ∧
    sourceAccountingUseRoleTags = allUseRoles.map useRoleTag ∧
    sourceAccountingTemplateActivationTags = allTemplateActivations.map templateActivationTag ∧
    sourceAccountingHandlingKindTags = allHandlingKinds.map handlingKindTag ∧
    sourceAccountingExpansionDispositionTags = allDispositions.map dispositionTag ∧
    sourceAccountingFailureCodeTags = allFailureCodes.map failureCodeTag ∧
    sourceAccountingFailureWitnessTags = allFailureCodes.map failureCodeTag := by
  decide

theorem generated_policy_matrix_matches_lean :
    sourceAccountingPolicyRows.all policyRowMatches = true := by
  set_option maxRecDepth 10000 in decide

theorem generated_failure_witnesses_are_complete_rejections :
    sourceAccountingFailureWitnessRows.all failureWitnessRowMatches = true ∧
    sourceAccountingFailureWitnessRows.map (fun row => row.1) =
      allFailureCodes.map failureCodeTag := by
  decide

theorem generated_decisions_match_lean :
    sourceAccountingDecisionRows.all decisionRowMatches = true := by
  set_option maxRecDepth 100000 in decide

end Ora.SourceAccountingSync

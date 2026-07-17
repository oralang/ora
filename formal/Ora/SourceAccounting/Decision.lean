import Ora.SourceAccounting.Policy

namespace Ora.SourceAccounting

inductive FailureCode where
  | duplicateIdentity | identityHashCollision | unknownSite | unknownExpansion
  | unknownUse | unknownEvidenceReference | orphanEvidence | evidenceCoverageMismatch
  | missingSemanticSite | unknownSemanticSite | missingSemanticUse
  | unexpectedSemanticUse | invalidSiteRole | invalidOwnerTemplate
  | invalidExpansionActivation | invalidExpansionParent | invalidExpansionDisposition
  | invalidControlGraph | invalidFoldTrace | abandonedFoldEvidence
  | missingSymbolicReplacement | invalidGeneratedFactDerivation | missingHandling
  | duplicateHandling | rejectedHandling | missingSymbolicObligation
  | missingSymbolicQuery | invalidSymbolicQueryKind | missingAssumptionIncorporation
  | missingRuntimeEnforcement | missingFrameValidation | missingStateEffect
  | zeroConcreteChecks | invalidConcreteCheckpoint | falseConcreteTarget
  | falseConcreteAssumption | invalidControlElimination | invalidFrameValidation
  | invalidStateEffect | handlingNotPermitted | reducedScopeExcludedNotPermitted
  | verificationDisabledNotPermitted
  deriving Repr, DecidableEq

inductive Decision where
  | accepted | rejected
  deriving Repr, DecidableEq

def accountsForAllBagB (mode : CompilationMode) (m : BagManifest) : Bool :=
  syntaxConservedBagB m && (usesCompleteBagB m && (evidenceClosedBagB m &&
    (traceValidBagB m && (handledOnceBagB m && policyCompatibleBagB mode m))))

def accountsForAllB (mode : CompilationMode) (m : Manifest) : Bool :=
  accountsForAllBagB mode m.bag

def AccountsForAll (mode : CompilationMode) (m : Manifest) : Prop :=
  SyntaxConserved m ∧ UsesComplete m ∧ EvidenceClosed m ∧
    TraceValid m ∧ HandledOnce m ∧ PolicyCompatible mode m

def decideBag (mode : CompilationMode) (m : BagManifest) : Decision :=
  if wellFormedBagB m && accountsForAllBagB mode m then .accepted else .rejected

def decide (mode : CompilationMode) (m : Manifest) : Decision :=
  decideBag mode m.bag

def failureList (mode : CompilationMode) (m : Manifest) : List FailureCode :=
  (if wellFormedB m then [] else [.duplicateIdentity]) ++
  (if syntaxConservedB m then [] else [.missingSemanticSite]) ++
  (if usesCompleteB m then [] else [.missingSemanticUse]) ++
  (if traceValidB m then [] else [.invalidFoldTrace]) ++
  (m.uses.filter (fun use =>
    !(m.handlings.any (fun handling => handling.useId == use.id))) |>.map
      (fun _ => .missingHandling)) ++
  (if evidenceClosedB m then [] else [.orphanEvidence]) ++
  (if policyCompatibleB mode m then [] else [.handlingNotPermitted])

def primaryFailure (mode : CompilationMode) (m : Manifest) : Option FailureCode :=
  (failureList mode m).head?

end Ora.SourceAccounting

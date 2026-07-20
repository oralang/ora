import Ora.SourceAccounting.Manifest

namespace Ora.SourceAccounting

def basicExcluded : SourceFactKind → Bool
  | .guard | .loopInvariant | .contractInvariant | .assert | .refinementGuard => true
  | _ => false

def handlingPermitted
    (mode : CompilationMode)
    (_origin : FactOrigin)
    (kind : SourceFactKind)
    (role : UseRole)
    (handling : HandlingKind)
    (disposition : ExpansionDisposition) : Bool :=
  if disposition == .rejected || disposition == .foldAbandonedToSymbolic then false
  else if role == .proofTarget && disposition == .symbolic && mode == .verifiedBasic && basicExcluded kind then
    handling == .reducedScopeExcluded
  else if role == .proofTarget && disposition == .symbolic && mode == .unverifiedEmit then
    handling == .verificationDisabled
  else if role == .assumptionContext && disposition == .symbolic && mode == .unverifiedEmit then
    handling == .assumptionIncorporated || handling == .verificationDisabled
  else match role with
  | .proofTarget => match disposition with
      | .symbolic => handling == .symbolic
      | .foldCommitted => handling == .concreteTrue || handling == .controlEliminated
      | .foldAbandonedToSymbolic | .rejected => false
  | .assumptionContext => match disposition with
      | .symbolic => handling == .assumptionIncorporated
      | .foldCommitted => handling == .concreteTrue || handling == .controlEliminated
      | .foldAbandonedToSymbolic | .rejected => false
  | .runtimeCondition => match disposition with
      | .symbolic => handling == .runtimeEnforced
      | .foldCommitted => handling == .concreteTrue || handling == .controlEliminated
      | .foldAbandonedToSymbolic | .rejected => false
  | .frameDirective => handling == .frameValidated ||
      (disposition == .foldCommitted && handling == .controlEliminated)
  | .stateDirective => match disposition with
      | .symbolic => handling == .stateEffectIncorporated
      | .foldCommitted => handling == .controlEliminated
      | .foldAbandonedToSymbolic | .rejected => false

def policyCompatibleBagB (mode : CompilationMode) (m : BagManifest) : Bool :=
  m.handlings.all fun handling =>
    m.uses.any fun use =>
      use.id == handling.useId &&
        m.typedSites.any (fun site =>
          site.id == use.siteId &&
            m.expansions.any (fun expansion =>
              expansion.id == use.expansionId &&
                handlingPermitted mode site.origin site.kind use.role handling.kind expansion.disposition))

def policyCompatibleB (mode : CompilationMode) (m : Manifest) : Bool :=
  policyCompatibleBagB mode m.bag

def PolicyCompatible (mode : CompilationMode) (m : Manifest) : Prop :=
  policyCompatibleB mode m = true

end Ora.SourceAccounting

/-
Handwritten data model for the compiler-kernel source-accounting law.

This model is deliberately structural. It does not model expression semantics,
SMT validity, or compiler execution. It models conservation of source sites,
activated semantic uses, terminal evidence, and concrete-fold trace authority.
-/

namespace Ora.SourceAccounting

inductive CompilationMode where
  | verifiedFull | verifiedBasic | unverifiedEmit
  deriving Repr, DecidableEq

inductive FactOrigin where
  | sourceSyntax | semanticGenerated
  deriving Repr, DecidableEq

inductive SourceFactKind where
  | requires | guard | ensures | ensuresOk | ensuresErr
  | loopInvariant | contractInvariant | assert | assume | havoc | modifies
  | refinementGuard | runtimeGuard
  deriving Repr, DecidableEq

inductive UseRole where
  | proofTarget | assumptionContext | frameDirective | stateDirective
  | runtimeCondition
  deriving Repr, DecidableEq

inductive ActivationReason where
  | runtimeOwner | symbolicCall | speculativeFold | requiredComptime
  deriving Repr, DecidableEq

inductive TemplateActivation where
  | runtimeBody | symbolicCallBoundary | comptimeBody
  deriving Repr, DecidableEq

inductive ExpansionDisposition where
  | symbolic | foldCommitted | foldAbandonedToSymbolic | rejected
  deriving Repr, DecidableEq

inductive HandlingKind where
  | symbolic | concreteTrue | runtimeEnforced | controlEliminated
  | assumptionIncorporated | frameValidated | stateEffectIncorporated
  | reducedScopeExcluded | verificationDisabled | rejected
  deriving Repr, DecidableEq

inductive QueryKind where
  | obligation | loopInvariantStep | loopBodySafety | loopInvariantPost
  | guardSatisfy | guardViolate | other
  deriving Repr, DecidableEq

inductive FoldDisposition where
  | committed | abandoned
  deriving Repr, DecidableEq

inductive ControlNodeKind where
  | entry | statement | branch | loopHead | loopBody | returnExit
  | successExit | errorExit
  deriving Repr, DecidableEq

inductive ControlEdgeKind where
  | next | branchTrue | branchFalse | loopBody | loopExit | backedge
  | breakExit | continueBackedge | returnExit | successExit | errorExit
  deriving Repr, DecidableEq

inductive TraceEventKind where
  | enterNode | takeEdge | predicateCheck | returnExit | breakExit
  | continueBackedge | successExit | errorExit
  deriving Repr, DecidableEq

structure SourceRange where
  file : String
  startByte : Nat
  endByte : Nat
  deriving Repr, DecidableEq

structure SiteKey where
  path : String
  owner : String
  startByte : Nat
  endByte : Nat
  kind : SourceFactKind
  ordinal : Nat
  deriving Repr, DecidableEq

structure DeclaredSite where
  id : Nat
  key : SiteKey
  deriving Repr, DecidableEq

structure TypedSite where
  id : Nat
  origin : FactOrigin
  kind : SourceFactKind
  key : SiteKey
  sourceFactId : Option Nat := none
  declaredSiteId : Option Nat
  derivationId : Option Nat
  deriving Repr, DecidableEq

structure GeneratedFactDerivation where
  id : Nat
  siteId : Nat
  semanticRule : String
  anchor : SourceRange
  parentIdentity : String
  ordinal : Nat
  deriving Repr, DecidableEq

structure UseTemplate where
  siteId : Nat
  role : UseRole
  controlNodeSlot : Option Nat := none
  deriving Repr, DecidableEq

structure ControlNodeTemplate where
  slot : Nat
  kind : ControlNodeKind
  range : SourceRange
  attachedUseOrdinals : List Nat := []
  deriving Repr, DecidableEq

structure ControlEdgeTemplate where
  slot : Nat
  fromSlot : Nat
  toSlot : Nat
  kind : ControlEdgeKind
  deriving Repr, DecidableEq

structure OwnerTemplate where
  id : Nat
  ownerKey : String := ""
  activation : TemplateActivation
  uses : List UseTemplate
  controlNodes : List ControlNodeTemplate := []
  controlEdges : List ControlEdgeTemplate := []
  entrySlot : Option Nat := none
  terminalSlots : List Nat := []
  deriving Repr, DecidableEq

structure Expansion where
  id : Nat
  templateId : Nat
  activation : ActivationReason
  disposition : ExpansionDisposition
  parentExpansionId : Option Nat := none
  replacementExpansionId : Option Nat
  rootRuntimeOwner : String := ""
  foldedCallSiteChain : List SourceRange := []
  importedModule : Option String := none
  genericBindings : List String := []
  traitImplementation : Option String := none
  traitMethod : Option String := none
  identity : String := ""
  deriving Repr, DecidableEq

structure SourceUse where
  id : Nat
  siteId : Nat
  expansionId : Nat
  templateOrdinal : Nat
  role : UseRole
  controlNodeId : Option Nat := none
  deriving Repr, DecidableEq

structure ControlNode where
  id : Nat
  expansionId : Nat
  slot : Nat
  kind : ControlNodeKind
  range : SourceRange
  attachedUseIds : List Nat := []
  deriving Repr, DecidableEq

structure ControlEdge where
  id : Nat
  expansionId : Nat
  fromNode : Nat
  toNode : Nat
  kind : ControlEdgeKind
  deriving Repr, DecidableEq

structure TraceEvent where
  kind : TraceEventKind
  nodeId : Option Nat := none
  edgeId : Option Nat := none
  useId : Option Nat := none
  predicateValue : Option Bool := none
  deriving Repr, DecidableEq

structure Fold where
  id : Nat
  expansionId : Nat
  entryNodeId : Nat
  terminalNodeId : Nat
  disposition : FoldDisposition
  events : List TraceEvent
  deriving Repr, DecidableEq

structure PredicateEvent where
  id : Nat
  foldId : Nat
  useId : Nat
  nodeId : Nat
  value : Bool
  deriving Repr, DecidableEq

structure CoveredEvidence where
  id : Nat
  producerId : Nat := 0
  coveredUseIds : List Nat
  deriving Repr, DecidableEq

structure QueryEvidence where
  id : Nat
  producerId : Nat := 0
  kind : QueryKind
  coveredUseIds : List Nat
  deriving Repr, DecidableEq

structure ValidationEvidence where
  id : Nat
  producerId : Nat := 0
  coveredUseIds : List Nat
  valid : Bool
  deriving Repr, DecidableEq

structure Handling where
  id : Nat
  useId : Nat
  kind : HandlingKind
  obligationIds : List Nat := []
  assumptionIds : List Nat := []
  queryIds : List Nat := []
  runtimeCheckIds : List Nat := []
  frameResultIds : List Nat := []
  stateEffectIds : List Nat := []
  predicateEventIds : List Nat := []
  foldId : Option Nat := none
  controlEventIndex : Option Nat := none
  rejectionReason : Option String := none
  deriving Repr, DecidableEq

structure Manifest where
  declaredSites : List DeclaredSite := []
  typedSites : List TypedSite := []
  generatedFactDerivations : List GeneratedFactDerivation := []
  ownerTemplates : List OwnerTemplate := []
  expansions : List Expansion := []
  uses : List SourceUse := []
  controlNodes : List ControlNode := []
  controlEdges : List ControlEdge := []
  folds : List Fold := []
  predicateEvents : List PredicateEvent := []
  obligations : List CoveredEvidence := []
  assumptions : List CoveredEvidence := []
  queries : List QueryEvidence := []
  runtimeChecks : List CoveredEvidence := []
  frameResults : List ValidationEvidence := []
  stateEffects : List ValidationEvidence := []
  handlings : List Handling := []
  deriving Repr, DecidableEq

private theorem permAllEq {α : Type} (predicate : α → Bool)
    {left right : List α} (h : left.Perm right) :
    left.all predicate = right.all predicate := by
  induction h with
  | nil => rfl
  | cons _ _ ih => simp [ih]
  | swap _ _ _ => simp [Bool.and_comm, Bool.and_left_comm]
  | trans _ _ ih₁ ih₂ => exact ih₁.trans ih₂

private theorem permAnyEq {α : Type} (predicate : α → Bool)
    {left right : List α} (h : left.Perm right) :
    left.any predicate = right.any predicate := by
  induction h with
  | nil => rfl
  | cons _ _ ih => simp [ih]
  | swap _ _ _ => simp [Bool.or_comm, Bool.or_left_comm]
  | trans _ _ ih₁ ih₂ => exact ih₁.trans ih₂

private theorem permFilterLengthEq {α : Type} (predicate : α → Bool)
    {left right : List α} (h : left.Perm right) :
    (left.filter predicate).length = (right.filter predicate).length :=
  (h.filter predicate).length_eq

def listPermSetoid (α : Type) : Setoid (List α) where
  r := List.Perm
  iseqv := {
    refl := List.Perm.refl
    symm := List.Perm.symm
    trans := List.Perm.trans
  }

/-- An unordered, multiplicity-preserving compiler row collection. -/
abbrev RowBag (α : Type) := Quotient (listPermSetoid α)

namespace RowBag

def ofList (rows : List α) : RowBag α := Quotient.mk _ rows

def all (rows : RowBag α) (predicate : α → Bool) : Bool :=
  Quotient.lift (fun values => values.all predicate)
    (fun _ _ h => permAllEq predicate h) rows

def any (rows : RowBag α) (predicate : α → Bool) : Bool :=
  Quotient.lift (fun values => values.any predicate)
    (fun _ _ h => permAnyEq predicate h) rows

def countP (rows : RowBag α) (predicate : α → Bool) : Nat :=
  Quotient.lift (fun values => (values.filter predicate).length)
    (fun _ _ h => permFilterLengthEq predicate h) rows

theorem ofList_eq_of_perm {left right : List α} (h : left.Perm right) :
    ofList left = ofList right := Quotient.sound h

end RowBag

structure BagManifest where
  declaredSites : RowBag DeclaredSite
  typedSites : RowBag TypedSite
  generatedFactDerivations : RowBag GeneratedFactDerivation
  ownerTemplates : RowBag OwnerTemplate
  expansions : RowBag Expansion
  uses : RowBag SourceUse
  controlNodes : RowBag ControlNode
  controlEdges : RowBag ControlEdge
  folds : RowBag Fold
  predicateEvents : RowBag PredicateEvent
  obligations : RowBag CoveredEvidence
  assumptions : RowBag CoveredEvidence
  queries : RowBag QueryEvidence
  runtimeChecks : RowBag CoveredEvidence
  frameResults : RowBag ValidationEvidence
  stateEffects : RowBag ValidationEvidence
  handlings : RowBag Handling

def Manifest.bag (m : Manifest) : BagManifest := {
  declaredSites := RowBag.ofList m.declaredSites
  typedSites := RowBag.ofList m.typedSites
  generatedFactDerivations := RowBag.ofList m.generatedFactDerivations
  ownerTemplates := RowBag.ofList m.ownerTemplates
  expansions := RowBag.ofList m.expansions
  uses := RowBag.ofList m.uses
  controlNodes := RowBag.ofList m.controlNodes
  controlEdges := RowBag.ofList m.controlEdges
  folds := RowBag.ofList m.folds
  predicateEvents := RowBag.ofList m.predicateEvents
  obligations := RowBag.ofList m.obligations
  assumptions := RowBag.ofList m.assumptions
  queries := RowBag.ofList m.queries
  runtimeChecks := RowBag.ofList m.runtimeChecks
  frameResults := RowBag.ofList m.frameResults
  stateEffects := RowBag.ofList m.stateEffects
  handlings := RowBag.ofList m.handlings
}

def noDuplicateNat (values : List Nat) : Bool :=
  values.all fun value => decide (values.count value = 1)

def exactlyOneNat (value : Nat) (values : List Nat) : Bool :=
  decide (values.count value = 1)

def uniqueBy (rows : RowBag α) (key : α → Nat) : Bool :=
  rows.all fun row => decide (rows.countP (fun other => key other == key row) = 1)

def activationDispositionValid
    (activation : ActivationReason) (disposition : ExpansionDisposition) : Bool :=
  match activation with
  | .runtimeOwner | .symbolicCall => disposition == .symbolic || disposition == .rejected
  | .speculativeFold => disposition == .foldCommitted ||
      disposition == .foldAbandonedToSymbolic || disposition == .rejected
  | .requiredComptime => disposition == .foldCommitted || disposition == .rejected

def templateActivation : ActivationReason → TemplateActivation
  | .runtimeOwner => .runtimeBody
  | .symbolicCall => .symbolicCallBoundary
  | .speculativeFold | .requiredComptime => .comptimeBody

def requiredRoles : SourceFactKind → List UseRole
  | .requires => [.proofTarget, .assumptionContext, .runtimeCondition]
  | .guard => [.proofTarget, .runtimeCondition]
  | .ensures | .ensuresOk | .ensuresErr => [.proofTarget, .assumptionContext]
  | .loopInvariant | .contractInvariant => [.proofTarget, .assumptionContext]
  | .assert => [.proofTarget, .runtimeCondition]
  | .assume => [.assumptionContext]
  | .havoc => [.stateDirective]
  | .modifies => [.frameDirective]
  | .refinementGuard => [.proofTarget, .runtimeCondition]
  | .runtimeGuard => [.runtimeCondition]

def templateRoles (kind : SourceFactKind) : TemplateActivation → List UseRole
  | .runtimeBody => match kind with
      | .requires => [.assumptionContext, .runtimeCondition]
      | .guard => [.proofTarget, .runtimeCondition]
      | .ensures | .ensuresOk | .ensuresErr => [.proofTarget]
      | .loopInvariant => [.proofTarget, .assumptionContext]
      | .contractInvariant => [.proofTarget, .proofTarget, .assumptionContext]
      | .assert => [.proofTarget, .runtimeCondition]
      | .assume => [.assumptionContext]
      | .havoc => [.stateDirective]
      | .modifies => [.frameDirective]
      | .refinementGuard => [.proofTarget, .runtimeCondition]
      | .runtimeGuard => [.runtimeCondition]
  | .symbolicCallBoundary => match kind with
      | .requires => [.proofTarget]
      | .ensures | .ensuresOk | .ensuresErr => [.assumptionContext]
      | _ => []
  | .comptimeBody => requiredRoles kind

def allUseRoles : List UseRole :=
  [.proofTarget, .assumptionContext, .frameDirective, .stateDirective, .runtimeCondition]

def ownerTemplateRolesValid (m : BagManifest) (template : OwnerTemplate) : Bool :=
  m.typedSites.all fun site =>
    if template.uses.any (fun use => use.siteId == site.id) then
      allUseRoles.all fun role =>
        decide ((template.uses.filter fun use =>
          use.siteId == site.id && use.role == role).length =
            ((templateRoles site.kind template.activation).filter (· == role)).length)
    else true

def sameExpansionInstance (left right : Expansion) : Bool :=
  left.parentExpansionId == right.parentExpansionId &&
  left.rootRuntimeOwner == right.rootRuntimeOwner &&
  left.foldedCallSiteChain == right.foldedCallSiteChain &&
  left.importedModule == right.importedModule &&
  left.genericBindings == right.genericBindings &&
  left.traitImplementation == right.traitImplementation &&
  left.traitMethod == right.traitMethod && left.identity == right.identity

def sourceTypedMatches (declared : DeclaredSite) (typed : TypedSite) : Bool :=
  typed.origin == .sourceSyntax &&
    typed.declaredSiteId == some declared.id &&
    typed.sourceFactId == some declared.key.startByte &&
    typed.kind == declared.key.kind && typed.key == declared.key

def syntaxConservedBagB (m : BagManifest) : Bool :=
  m.declaredSites.all (fun declared =>
    decide (m.typedSites.countP (sourceTypedMatches declared) = 1)) &&
  m.typedSites.all (fun typed =>
    match typed.origin with
    | .sourceSyntax => m.declaredSites.any (fun declared => sourceTypedMatches declared typed)
    | .semanticGenerated => typed.sourceFactId.isNone && typed.declaredSiteId.isNone &&
        match typed.derivationId with
        | none => false
        | some derivationId => m.generatedFactDerivations.any (fun row =>
            row.id == derivationId && row.siteId == typed.id))

def templateUseMatches (expansion : Expansion) (ordinal : Nat)
    (expected : UseTemplate) (use : SourceUse) : Bool :=
  use.expansionId == expansion.id && use.templateOrdinal == ordinal &&
    use.siteId == expected.siteId && use.role == expected.role

def expansionCarriesUses (expansion : Expansion) : Bool :=
  expansion.disposition != .foldAbandonedToSymbolic &&
    expansion.disposition != .rejected

def expansionUsesComplete (m : BagManifest) (expansion : Expansion) : Bool :=
  if !expansionCarriesUses expansion then
    decide (m.uses.countP (fun use => use.expansionId == expansion.id) = 0)
  else
    m.ownerTemplates.any fun template =>
      template.id == expansion.templateId &&
        (List.range template.uses.length).all fun ordinal =>
          match template.uses.get? ordinal with
          | none => false
          | some expected =>
              decide (m.uses.countP (templateUseMatches expansion ordinal expected) = 1)

def suppliedUseExpected (m : BagManifest) (use : SourceUse) : Bool :=
  m.expansions.any fun expansion =>
    expansion.id == use.expansionId && expansionCarriesUses expansion &&
      m.ownerTemplates.any fun template =>
        template.id == expansion.templateId &&
          match template.uses.get? use.templateOrdinal with
          | none => false
          | some expected => templateUseMatches expansion use.templateOrdinal expected use

def usesCompleteBagB (m : BagManifest) : Bool :=
  m.expansions.all (expansionUsesComplete m) && m.uses.all (suppliedUseExpected m)

def coveredReferenceValid (rows : RowBag CoveredEvidence)
    (useId : Nat) (ids : List Nat) : Bool :=
  ids.all fun id => rows.any fun row =>
    row.id == id && row.coveredUseIds.contains useId

def queryReferenceValid (rows : RowBag QueryEvidence)
    (useId : Nat) (ids : List Nat) : Bool :=
  ids.all fun id => rows.any fun row =>
    row.id == id && row.coveredUseIds.contains useId

def validationReferenceValid (rows : RowBag ValidationEvidence)
    (useId : Nat) (ids : List Nat) : Bool :=
  ids.all fun id => rows.any fun row =>
    row.id == id && row.coveredUseIds.contains useId

def coveredEvidenceConsumed (handlings : RowBag Handling)
    (row : CoveredEvidence) (references : Handling → List Nat) : Bool :=
  handlings.any (fun handling => (references handling).contains row.id) &&
    row.coveredUseIds.all (fun useId => handlings.any fun handling =>
      handling.useId == useId && (references handling).contains row.id) &&
    handlings.all (fun handling =>
      if (references handling).contains row.id then row.coveredUseIds.contains handling.useId else true)

def queryEvidenceConsumed (handlings : RowBag Handling) (row : QueryEvidence) : Bool :=
  handlings.any (fun handling => handling.queryIds.contains row.id) &&
    row.coveredUseIds.all (fun useId => handlings.any fun handling =>
      handling.useId == useId && handling.queryIds.contains row.id) &&
    handlings.all (fun handling =>
      if handling.queryIds.contains row.id then row.coveredUseIds.contains handling.useId else true)

def validationEvidenceConsumed (handlings : RowBag Handling)
    (row : ValidationEvidence) (references : Handling → List Nat) : Bool :=
  handlings.any (fun handling => (references handling).contains row.id) &&
    row.coveredUseIds.all (fun useId => handlings.any fun handling =>
      handling.useId == useId && (references handling).contains row.id) &&
    handlings.all (fun handling =>
      if (references handling).contains row.id then row.coveredUseIds.contains handling.useId else true)

def predicateReferenceValid (m : BagManifest) (handling : Handling) : Bool :=
  handling.predicateEventIds.all fun id => m.predicateEvents.any fun row =>
    row.id == id && row.useId == handling.useId

def handlingReferencesCompatible (handling : Handling) : Bool :=
  let hasObligations := !handling.obligationIds.isEmpty
  let hasAssumptions := !handling.assumptionIds.isEmpty
  let hasQueries := !handling.queryIds.isEmpty
  let hasRuntimeChecks := !handling.runtimeCheckIds.isEmpty
  let hasFrameResults := !handling.frameResultIds.isEmpty
  let hasStateEffects := !handling.stateEffectIds.isEmpty
  let hasPredicates := !handling.predicateEventIds.isEmpty
  let hasControlEvent := handling.controlEventIndex.isSome
  let hasRejection := handling.rejectionReason.isSome
  match handling.kind with
  | .symbolic => !hasAssumptions && !hasRuntimeChecks && !hasFrameResults &&
      !hasStateEffects && !hasPredicates && handling.foldId.isNone &&
      !hasControlEvent && !hasRejection
  | .concreteTrue => !hasObligations && !hasAssumptions && !hasQueries &&
      !hasRuntimeChecks && !hasFrameResults && !hasStateEffects &&
      handling.foldId.isSome && !hasControlEvent && !hasRejection
  | .runtimeEnforced => !hasObligations && !hasAssumptions && !hasQueries &&
      !hasFrameResults && !hasStateEffects && !hasPredicates &&
      handling.foldId.isNone && !hasControlEvent && !hasRejection
  | .controlEliminated => !hasObligations && !hasAssumptions && !hasQueries &&
      !hasRuntimeChecks && !hasFrameResults && !hasStateEffects && !hasPredicates &&
      handling.foldId.isSome && !hasRejection
  | .assumptionIncorporated => !hasObligations &&
      !hasRuntimeChecks && !hasFrameResults && !hasStateEffects && !hasPredicates &&
      handling.foldId.isNone && !hasControlEvent && !hasRejection
  | .frameValidated => !hasObligations && !hasAssumptions && !hasQueries &&
      !hasRuntimeChecks && !hasStateEffects && !hasPredicates &&
      handling.foldId.isNone && !hasControlEvent && !hasRejection
  | .stateEffectIncorporated => !hasObligations && !hasAssumptions && !hasQueries &&
      !hasRuntimeChecks && !hasFrameResults && !hasPredicates &&
      handling.foldId.isNone && !hasControlEvent && !hasRejection
  | .reducedScopeExcluded | .verificationDisabled => !hasObligations &&
      !hasAssumptions && !hasQueries && !hasRuntimeChecks && !hasFrameResults &&
      !hasStateEffects && !hasPredicates && handling.foldId.isNone &&
      !hasControlEvent && !hasRejection
  | .rejected => hasRejection

def queryKindAllowed (factKind : SourceFactKind) (queryKind : QueryKind) : Bool :=
  match factKind with
  | .loopInvariant => queryKind == .obligation || queryKind == .loopInvariantStep ||
      queryKind == .loopBodySafety || queryKind == .loopInvariantPost
  | .guard | .refinementGuard => queryKind == .guardSatisfy ||
      queryKind == .guardViolate || queryKind == .obligation
  | _ => queryKind == .obligation || queryKind == .other

def referencedQueryKindsValid (m : BagManifest) (site : TypedSite)
    (handling : Handling) : Bool :=
  handling.queryIds.all fun id => m.queries.any fun row =>
    row.id == id && queryKindAllowed site.kind row.kind

def symbolicTargetQuerySetComplete (m : BagManifest) (site : TypedSite)
    (handling : Handling) : Bool :=
  if site.kind == .loopInvariant then
    handling.queryIds.any (fun id => m.queries.any fun row =>
      row.id == id && row.kind == .obligation) &&
    handling.queryIds.any (fun id => m.queries.any fun row =>
      row.id == id && row.kind == .loopInvariantStep)
  else true

def handlingTerminalEvidenceValid (m : BagManifest) (handling : Handling) : Bool :=
  m.uses.any fun use => use.id == handling.useId &&
    m.typedSites.any fun site => site.id == use.siteId &&
      match handling.kind with
      | .symbolic => !handling.obligationIds.isEmpty && !handling.queryIds.isEmpty &&
          referencedQueryKindsValid m site handling &&
          symbolicTargetQuerySetComplete m site handling
      | .concreteTrue => !handling.predicateEventIds.isEmpty
      | .runtimeEnforced => !handling.runtimeCheckIds.isEmpty
      | .controlEliminated => handling.foldId.isSome
      | .assumptionIncorporated =>
          (!handling.assumptionIds.isEmpty || !handling.queryIds.isEmpty) &&
          referencedQueryKindsValid m site handling
      | .frameValidated => !handling.frameResultIds.isEmpty &&
          handling.frameResultIds.all (fun id => m.frameResults.any fun row =>
            row.id == id && row.valid)
      | .stateEffectIncorporated => !handling.stateEffectIds.isEmpty &&
          handling.stateEffectIds.all (fun id => m.stateEffects.any fun row =>
            row.id == id && row.valid)
      | .reducedScopeExcluded | .verificationDisabled => true
      | .rejected => false

def evidenceClosedBagB (m : BagManifest) : Bool :=
  m.handlings.all (fun handling =>
    handlingReferencesCompatible handling && handlingTerminalEvidenceValid m handling &&
    coveredReferenceValid m.obligations handling.useId handling.obligationIds &&
    coveredReferenceValid m.assumptions handling.useId handling.assumptionIds &&
    queryReferenceValid m.queries handling.useId handling.queryIds &&
    coveredReferenceValid m.runtimeChecks handling.useId handling.runtimeCheckIds &&
    validationReferenceValid m.frameResults handling.useId handling.frameResultIds &&
    validationReferenceValid m.stateEffects handling.useId handling.stateEffectIds &&
    predicateReferenceValid m handling) &&
  m.obligations.all (fun row => coveredEvidenceConsumed m.handlings row (·.obligationIds)) &&
  m.assumptions.all (fun row => coveredEvidenceConsumed m.handlings row (·.assumptionIds)) &&
  m.queries.all (queryEvidenceConsumed m.handlings) &&
  m.runtimeChecks.all (fun row => coveredEvidenceConsumed m.handlings row (·.runtimeCheckIds)) &&
  m.frameResults.all (fun row => validationEvidenceConsumed m.handlings row (·.frameResultIds)) &&
  m.stateEffects.all (fun row => validationEvidenceConsumed m.handlings row (·.stateEffectIds)) &&
  m.predicateEvents.all (fun row => m.handlings.any fun handling =>
    handling.useId == row.useId && handling.predicateEventIds.contains row.id)

def foldHasNoAuthorityBag (m : BagManifest) (fold : Fold) : Bool :=
  m.predicateEvents.all (fun evidence => evidence.foldId != fold.id) &&
    m.handlings.all (fun handling => handling.foldId != some fold.id)

def traceEventMatchesPredicate (event : TraceEvent) (predicate : PredicateEvent) : Bool :=
  event.kind == .predicateCheck && event.nodeId == some predicate.nodeId &&
    event.useId == some predicate.useId && event.predicateValue == some predicate.value

def edgeConnects (m : BagManifest) (fold : Fold)
    (fromNode toNode edgeId : Nat) (requiredKind : Option ControlEdgeKind := none) : Bool :=
  m.controlEdges.any fun edge =>
    edge.id == edgeId && edge.expansionId == fold.expansionId &&
      edge.fromNode == fromNode && edge.toNode == toNode &&
      match requiredKind with
      | none => true
      | some kind => edge.kind == kind

def terminalEventValid (m : BagManifest) (fold : Fold) (event : TraceEvent) : Bool :=
  event.nodeId == some fold.terminalNodeId &&
    m.controlNodes.any (fun node =>
      node.id == fold.terminalNodeId && node.expansionId == fold.expansionId &&
        match event.kind with
        | .successExit => node.kind == .successExit
        | .errorExit => node.kind == .errorExit
        | .returnExit => node.kind == .returnExit || node.kind == .successExit || node.kind == .errorExit
        | _ => false)

def validateTraceTail (m : BagManifest) (fold : Fold) :
    Nat → Option ControlEdgeKind → List TraceEvent → Bool
  | current, requiredEdge, [] => requiredEdge.isNone && current == fold.terminalNodeId
  | current, requiredEdge, event :: rest =>
      match event.kind with
      | .predicateCheck =>
          requiredEdge.isNone && event.nodeId == some current &&
            event.useId.isSome && event.predicateValue.isSome &&
            m.predicateEvents.any (fun predicate =>
              predicate.foldId == fold.id && traceEventMatchesPredicate event predicate) &&
            validateTraceTail m fold current none rest
      | .takeEdge =>
          match event.edgeId, rest with
          | some edgeId, nextEvent :: tail =>
              if nextEvent.kind == .enterNode then
                match nextEvent.nodeId with
                | some nextNode => edgeConnects m fold current nextNode edgeId requiredEdge &&
                    validateTraceTail m fold nextNode none tail
                | none => false
              else false
          | _, _ => false
      | .successExit | .errorExit =>
          requiredEdge.isNone && rest.isEmpty && terminalEventValid m fold event
      | .returnExit =>
          requiredEdge.isNone && event.nodeId == some current &&
            if current == fold.terminalNodeId then
              rest.isEmpty && terminalEventValid m fold event
            else validateTraceTail m fold current (some .returnExit) rest
      | .breakExit =>
          requiredEdge.isNone && event.nodeId == some current &&
            validateTraceTail m fold current (some .breakExit) rest
      | .continueBackedge =>
          requiredEdge.isNone && event.nodeId == some current &&
            validateTraceTail m fold current (some .continueBackedge) rest
      | .enterNode => false

def eventSegmentHasTrueCheck (events : List TraceEvent) (nodeId useId : Nat) : Bool :=
  (events.takeWhile fun event => event.kind != .enterNode).any fun event =>
    event.kind == .predicateCheck && event.nodeId == some nodeId &&
      event.useId == some useId && event.predicateValue == some true

def checkpointsValid (m : BagManifest) (events : List TraceEvent) : Bool :=
  match events with
  | [] => true
  | event :: rest =>
      (if event.kind == .enterNode then
        match event.nodeId with
        | none => false
        | some nodeId => m.controlNodes.any (fun node =>
            node.id == nodeId && node.attachedUseIds.all (eventSegmentHasTrueCheck rest nodeId))
       else true) && checkpointsValid m rest
def foldTraceValidBag (m : BagManifest) (fold : Fold) : Bool :=
  if fold.disposition == .abandoned then fold.events.isEmpty && foldHasNoAuthorityBag m fold
  else match fold.events with
    | [] => false
    | first :: rest =>
        first.kind == .enterNode && first.nodeId == some fold.entryNodeId &&
          m.controlNodes.any (fun node =>
            node.id == fold.entryNodeId && node.expansionId == fold.expansionId) &&
          checkpointsValid m fold.events && validateTraceTail m fold fold.entryNodeId none rest

def concreteHandlingTraceValidBag (m : BagManifest) (handling : Handling) : Bool :=
  if handling.kind == .concreteTrue then
    !handling.predicateEventIds.isEmpty &&
      match handling.foldId with
      | none => false
      | some foldId => m.folds.any fun fold =>
          fold.id == foldId && fold.disposition == .committed &&
            handling.predicateEventIds.all (fun predicateId =>
              m.predicateEvents.any fun predicate =>
                predicate.id == predicateId && predicate.foldId == fold.id &&
                  predicate.useId == handling.useId && predicate.value &&
                  fold.events.any (fun event => traceEventMatchesPredicate event predicate))
  else true

def noAbandonedFoldAuthorityBagB (m : BagManifest) : Bool :=
  m.folds.all fun fold =>
    if fold.disposition == .abandoned then foldHasNoAuthorityBag m fold else true

def traceValidBagB (m : BagManifest) : Bool :=
  noAbandonedFoldAuthorityBagB m &&
    (m.folds.all (foldTraceValidBag m) &&
      (m.predicateEvents.all (fun predicate => m.folds.any fun fold =>
        fold.id == predicate.foldId && fold.events.any (fun event => traceEventMatchesPredicate event predicate)) &&
       m.handlings.all (concreteHandlingTraceValidBag m)))

def expansionAbandoned (m : BagManifest) (expansionId : Nat) : Bool :=
  m.expansions.any fun expansion =>
    expansion.id == expansionId && expansion.disposition == .foldAbandonedToSymbolic

def handledOnceBagB (m : BagManifest) : Bool :=
  m.uses.all (fun use =>
    if expansionAbandoned m use.expansionId then
      decide (m.handlings.countP (fun handling => handling.useId == use.id) = 0)
    else
      decide (m.handlings.countP (fun handling => handling.useId == use.id) = 1)) &&
  m.handlings.all (fun handling =>
    m.uses.any (fun use => use.id == handling.useId && !expansionAbandoned m use.expansionId))

def terminalControlKind : ControlNodeKind → Bool
  | .returnExit | .successExit | .errorExit => true
  | _ => false

def ownerTemplateControlValid (template : OwnerTemplate) : Bool :=
  if template.controlNodes.isEmpty then
    template.controlEdges.isEmpty && template.entrySlot.isNone &&
      template.terminalSlots.isEmpty && template.uses.all (·.controlNodeSlot.isNone)
  else
    noDuplicateNat (template.controlNodes.map (·.slot)) &&
    noDuplicateNat (template.controlEdges.map (·.slot)) &&
    noDuplicateNat template.terminalSlots &&
    (match template.entrySlot with
     | none => false
     | some slot => template.controlNodes.any (fun node =>
         node.slot == slot && node.kind == .entry)) &&
    !template.terminalSlots.isEmpty &&
    template.terminalSlots.all (fun slot => template.controlNodes.any (fun node =>
      node.slot == slot && terminalControlKind node.kind)) &&
    template.controlEdges.all (fun edge =>
      template.controlNodes.any (fun node => node.slot == edge.fromSlot) &&
      template.controlNodes.any (fun node => node.slot == edge.toSlot)) &&
    template.controlNodes.all (fun node =>
      noDuplicateNat node.attachedUseOrdinals &&
      node.attachedUseOrdinals.all (fun ordinal =>
        match template.uses.get? ordinal with
        | none => false
        | some use => use.controlNodeSlot == some node.slot)) &&
    (List.range template.uses.length).all (fun ordinal =>
      match template.uses.get? ordinal with
      | none => false
      | some use => match use.controlNodeSlot with
        | none => true
        | some slot => template.controlNodes.any (fun node =>
            node.slot == slot && node.attachedUseOrdinals.contains ordinal))

def instantiatedNodeMatches (m : BagManifest) (expansion : Expansion)
    (expected : ControlNodeTemplate) (actual : ControlNode) : Bool :=
  actual.expansionId == expansion.id && actual.slot == expected.slot &&
  actual.kind == expected.kind && actual.range == expected.range &&
  noDuplicateNat actual.attachedUseIds &&
  actual.attachedUseIds.length == expected.attachedUseOrdinals.length &&
  expected.attachedUseOrdinals.all (fun ordinal =>
    m.uses.any (fun use =>
      use.expansionId == expansion.id && use.templateOrdinal == ordinal &&
        actual.attachedUseIds.contains use.id))

def instantiatedEdgeMatches (m : BagManifest) (expansion : Expansion)
    (expected : ControlEdgeTemplate) (actual : ControlEdge) : Bool :=
  actual.expansionId == expansion.id && actual.kind == expected.kind &&
  m.controlNodes.any (fun node =>
    node.id == actual.fromNode && node.expansionId == expansion.id &&
      node.slot == expected.fromSlot) &&
  m.controlNodes.any (fun node =>
    node.id == actual.toNode && node.expansionId == expansion.id &&
      node.slot == expected.toSlot)

def expansionControlComplete (m : BagManifest) (expansion : Expansion) : Bool :=
  if !expansionCarriesUses expansion then
    decide (m.controlNodes.countP (fun node => node.expansionId == expansion.id) = 0) &&
      decide (m.controlEdges.countP (fun edge => edge.expansionId == expansion.id) = 0)
  else m.ownerTemplates.any fun template =>
    template.id == expansion.templateId &&
    template.controlNodes.all (fun expected =>
      decide (m.controlNodes.countP (instantiatedNodeMatches m expansion expected) = 1)) &&
    m.controlNodes.all (fun actual =>
      if actual.expansionId == expansion.id then
        template.controlNodes.any (fun expected =>
          instantiatedNodeMatches m expansion expected actual)
      else true) &&
    template.controlEdges.all (fun expected =>
      decide (m.controlEdges.countP (instantiatedEdgeMatches m expansion expected) = 1)) &&
    m.controlEdges.all (fun actual =>
      if actual.expansionId == expansion.id then
        template.controlEdges.any (fun expected =>
          instantiatedEdgeMatches m expansion expected actual)
      else true)

def wellFormedBagB (m : BagManifest) : Bool :=
  uniqueBy m.declaredSites (·.id) &&
  uniqueBy m.typedSites (·.id) &&
  uniqueBy m.generatedFactDerivations (·.id) &&
  uniqueBy m.ownerTemplates (·.id) &&
  uniqueBy m.expansions (·.id) &&
  uniqueBy m.uses (·.id) &&
  uniqueBy m.controlNodes (·.id) &&
  uniqueBy m.controlEdges (·.id) &&
  uniqueBy m.folds (·.id) &&
  uniqueBy m.predicateEvents (·.id) &&
  uniqueBy m.obligations (·.id) &&
  uniqueBy m.assumptions (·.id) &&
  uniqueBy m.queries (·.id) &&
  uniqueBy m.runtimeChecks (·.id) &&
  uniqueBy m.frameResults (·.id) &&
  uniqueBy m.stateEffects (·.id) &&
  uniqueBy m.handlings (·.id) &&
  m.generatedFactDerivations.all (fun derivation =>
    !derivation.semanticRule.isEmpty && !derivation.parentIdentity.isEmpty &&
      m.typedSites.any (fun site =>
        site.id == derivation.siteId && site.origin == .semanticGenerated &&
          site.derivationId == some derivation.id)) &&
  m.ownerTemplates.all (fun template =>
    template.uses.all (fun row => m.typedSites.any (fun site => site.id == row.siteId)) &&
      ownerTemplateRolesValid m template &&
      noDuplicateNat (template.controlNodes.map (·.slot)) &&
      noDuplicateNat (template.controlEdges.map (·.slot)) &&
      ownerTemplateControlValid template) &&
  m.expansions.all (fun expansion =>
    !expansion.rootRuntimeOwner.isEmpty && !expansion.identity.isEmpty &&
      activationDispositionValid expansion.activation expansion.disposition &&
      expansion.disposition != .rejected &&
      m.ownerTemplates.any (fun template => template.id == expansion.templateId &&
        template.activation == templateActivation expansion.activation) &&
      (match expansion.parentExpansionId with
       | none => true
       | some parent => parent != expansion.id && m.expansions.any (fun row => row.id == parent)) &&
      (if expansion.disposition == .foldAbandonedToSymbolic then
        match expansion.replacementExpansionId with
        | none => false
        | some replacement => m.expansions.any (fun row =>
            row.id == replacement && row.disposition == .symbolic &&
              sameExpansionInstance expansion row)
       else expansion.replacementExpansionId.isNone) &&
      expansionControlComplete m expansion) &&
  m.typedSites.all (fun site => m.ownerTemplates.any (fun template =>
    template.uses.any (fun use => use.siteId == site.id))) &&
  m.uses.all (fun use =>
    m.typedSites.any (fun site => site.id == use.siteId) &&
      m.expansions.any (fun expansion => expansion.id == use.expansionId)) &&
  m.controlNodes.all (fun node =>
    node.range.startByte <= node.range.endByte &&
      m.expansions.any (fun expansion => expansion.id == node.expansionId)) &&
  m.controlEdges.all (fun edge =>
    m.controlNodes.any (fun node => node.id == edge.fromNode && node.expansionId == edge.expansionId) &&
      m.controlNodes.any (fun node => node.id == edge.toNode && node.expansionId == edge.expansionId)) &&
  m.folds.all (fun fold =>
    m.expansions.any (fun expansion => expansion.id == fold.expansionId) &&
      m.controlNodes.any (fun node => node.id == fold.entryNodeId && node.expansionId == fold.expansionId) &&
      m.controlNodes.any (fun node => node.id == fold.terminalNodeId && node.expansionId == fold.expansionId))

def syntaxConservedB (m : Manifest) : Bool := syntaxConservedBagB m.bag
def usesCompleteB (m : Manifest) : Bool := usesCompleteBagB m.bag
def evidenceClosedB (m : Manifest) : Bool := evidenceClosedBagB m.bag
def traceValidB (m : Manifest) : Bool := traceValidBagB m.bag
def handledOnceB (m : Manifest) : Bool := handledOnceBagB m.bag
def wellFormedB (m : Manifest) : Bool := wellFormedBagB m.bag

def foldHasNoAuthority (m : Manifest) (fold : Fold) : Bool := foldHasNoAuthorityBag m.bag fold
def concreteHandlingTraceValid (m : Manifest) (handling : Handling) : Bool :=
  concreteHandlingTraceValidBag m.bag handling

def SyntaxConserved (m : Manifest) : Prop := syntaxConservedB m = true
def UsesComplete (m : Manifest) : Prop := usesCompleteB m = true
def EvidenceClosed (m : Manifest) : Prop := evidenceClosedB m = true
def TraceValid (m : Manifest) : Prop := traceValidB m = true
def HandledOnce (m : Manifest) : Prop := handledOnceB m = true
def WellFormed (m : Manifest) : Prop := wellFormedB m = true

end Ora.SourceAccounting

/-
Canonical verification-obligation manifest model.

This is the Lean-side consumer for compiler-emitted obligation rows. It is not
the contract semantics layer yet; it checks that the generated manifest is
structurally well-formed and that every formula/term reference points at a real
row. Later slices attach semantic meaning to this same syntax for Z3/Lean
cross-checking.
-/

namespace Ora.Obligation

abbrev Id := Nat
abbrev TermId := Nat

inductive RegionRef where
  | none
  | storage
  | memory
  | transient
  | calldata
  deriving Repr, BEq, DecidableEq

inductive TyRef where
  | spelling : String → TyRef
  | compilerTypeId : Nat → TyRef
  deriving Repr, BEq, DecidableEq

structure FreeVarId where
  file_id : Nat
  pattern_id : Nat
  deriving Repr, BEq, DecidableEq

structure FreeVarRef where
  id : FreeVarId
  name : String
  ty : Option TyRef := none
  region : Option RegionRef := none
  deriving Repr, BEq, DecidableEq

structure BoundVarRef where
  /-- De Bruijn index: `0` is the nearest enclosing quantifier binder. -/
  index : Nat
  name : String := ""
  ty : Option TyRef := none
  region : Option RegionRef := none
  deriving Repr, BEq, DecidableEq

structure BinderRef where
  name : String
  ty : Option TyRef := none
  region : Option RegionRef := none
  deriving Repr, BEq, DecidableEq

inductive VarRef where
  | free : FreeVarRef → VarRef
  | bound : BoundVarRef → VarRef
  deriving Repr, BEq, DecidableEq

inductive PlaceKey where
  | parameter : FreeVarId → PlaceKey
  | comptimeParameter : Nat → PlaceKey
  | comptimeRangeParameter : Nat → PlaceKey
  | constant : String → PlaceKey
  | msgSender
  | txOrigin
  | unknown
  deriving Repr, BEq, DecidableEq

structure PlaceRef where
  root : String
  region : RegionRef
  fields : List String := []
  keys : List PlaceKey := []
  deriving Repr, BEq, DecidableEq

inductive UnaryOp where
  | not_
  | neg
  deriving Repr, BEq, DecidableEq

inductive BinaryOp where
  | eq
  | ne
  | lt
  | le
  | gt
  | ge
  | slt
  | sle
  | sgt
  | sge
  | add
  | sub
  | mul
  | div
  | mod_
  | and_
  | or_
  | implies
  deriving Repr, BEq, DecidableEq

inductive Quantifier where
  | forall_
  | exists_
  deriving Repr, BEq, DecidableEq

inductive VerificationQueryFragment where
  | unknown
  | qfBv
  | qfBvArray
  | aufbv
  | aufbvQuantifiers
  | other
  deriving Repr, BEq, DecidableEq

inductive QuantifierBinderSort where
  | bool
  | bitVector
  | byteSequence
  | opaqueUnknown
  deriving Repr, BEq, DecidableEq

inductive QuantifierPatternStatus where
  | explicit
  | synthesized
  | absent
  deriving Repr, BEq, DecidableEq

inductive QuantifierDegradation where
  | unsupportedBinderType
  | malformedBinderWidth
  deriving Repr, BEq, DecidableEq

structure IntegerLiteralTerm where
  value : String
  ty : Option TyRef := none
  deriving Repr, BEq, DecidableEq

structure UnaryTerm where
  op : UnaryOp
  operand : TermId
  deriving Repr, BEq, DecidableEq

structure BinaryTerm where
  op : BinaryOp
  lhs : TermId
  rhs : TermId
  ty : Option TyRef := none
  deriving Repr, BEq, DecidableEq

structure RefinementPredicateTerm where
  name : String
  value : TermId
  args : List TermId := []
  deriving Repr, BEq, DecidableEq

structure QuantifiedTerm where
  quantifier : Quantifier
  binder : BinderRef
  condition : Option TermId := none
  body : TermId
  deriving Repr, BEq, DecidableEq

inductive Term where
  | boolLit : Bool → Term
  | intLit : IntegerLiteralTerm → Term
  | variable : VarRef → Term
  | old : TermId → Term
  | result : Term
  | placeRead : PlaceRef → Term
  | unary : UnaryTerm → Term
  | binary : BinaryTerm → Term
  | refinementPredicate : RefinementPredicateTerm → Term
  | quantified : QuantifiedTerm → Term
  deriving Repr, BEq, DecidableEq

inductive FormulaRef where
  | term : TermId → FormulaRef
  deriving Repr, BEq, DecidableEq

inductive LogicalRole where
  | invariant
  | requires
  | calleePrecondition
  | ensures
  | ensuresOk
  | ensuresErr
  | assert_
  | guard
  | loopInvariant
  | contractInvariant
  | arithmeticSafety
  | refinement
  | importedCalleeObligation
  | importedCalleeEnsures
  deriving Repr, BEq, DecidableEq

inductive ResourceOperation where
  | move
  | create
  | destroy
  deriving Repr, BEq, DecidableEq

inductive ResourceProperty where
  | amountNonNegative
  | sourceSufficient
  | destinationNoOverflow
  | samePlaceIdentity
  | conservation
  deriving Repr, BEq, DecidableEq

inductive BackendComponent where
  | dispatcher
  | oratosir
  | sinora
  | artifactPolicy
  deriving Repr, BEq, DecidableEq

inductive BackendProperty where
  | complete
  | disjoint
  | ordered
  | preservesSelectorBehavior
  | noUnknownStrategy
  | dependencyValid
  deriving Repr, BEq, DecidableEq

inductive EffectFrameRelation where
  | writeCoveredByModifies
  | readPreservedByFrame
  | readPreservedByKeyEvidence
  | lockCoversWrite
  | externalCallFrame
  deriving Repr, BEq, DecidableEq

inductive KeyDisjointEvidenceKind where
  | freeVarDisequality
  deriving Repr, BEq, DecidableEq

structure KeyDisjointEvidence where
  kind : KeyDisjointEvidenceKind
  assumptionId : Id
  lhs : FreeVarId
  rhs : FreeVarId
  read : PlaceRef
  write : PlaceRef
  keyIndex : Nat
  deriving Repr, BEq, DecidableEq

structure EffectFrameGoal where
  relation : EffectFrameRelation
  declared : List PlaceRef := []
  actual : List PlaceRef := []
  evidence : List KeyDisjointEvidence := []
  deriving Repr, BEq, DecidableEq

structure ResourceGoal where
  op : ResourceOperation
  domain : String
  source : Option PlaceRef := none
  destination : Option PlaceRef := none
  amount : Option FormulaRef := none
  property : ResourceProperty
  deriving Repr, BEq, DecidableEq

structure QuantifierGoal where
  quantifier : Quantifier
  binderName : String
  binderType : TyRef
  binderSort : QuantifierBinderSort
  fragment : VerificationQueryFragment
  patternStatus : QuantifierPatternStatus
  degradation : Option QuantifierDegradation := none
  deriving Repr, BEq, DecidableEq

structure BackendFactGoal where
  component : BackendComponent
  property : BackendProperty
  deriving Repr, BEq, DecidableEq

inductive ObligationKind where
  | logical : LogicalRole → FormulaRef → ObligationKind
  | runtimeGuard : String → FormulaRef → ObligationKind
  | effectFrame : EffectFrameGoal → ObligationKind
  | resource : ResourceGoal → ObligationKind
  | quantifier : QuantifierGoal → ObligationKind
  | backendFact : BackendFactGoal → ObligationKind
  deriving Repr, BEq, DecidableEq

inductive AssumptionKind where
  | requires
  | assume
  | pathAssume
  | envAssume
  | binding
  | twoStateLinkage
  | frame
  | loopInvariant
  | calleeObligation
  | calleeEnsures
  | ghostAxiom
  | goal
  deriving Repr, BEq, DecidableEq

inductive ProofArtifactKind where
  | userlandLean
  deriving Repr, BEq, DecidableEq

structure AssumptionRow where
  id : Id
  owner : String
  kind : AssumptionKind
  formula : Option FormulaRef := none
  deriving Repr, BEq, DecidableEq

structure ObligationRow where
  id : Id
  owner : String
  kind : ObligationKind
  deriving Repr, BEq, DecidableEq

structure ProofArtifactRow where
  id : Id
  owner : String
  kind : ProofArtifactKind
  moduleName : String
  theoremName : String
  path : Option String := none
  contentHash : Option Nat := none
  obligationIds : List Id := []
  deriving Repr, BEq, DecidableEq

structure Manifest where
  terms : List Term := []
  assumptions : List AssumptionRow := []
  obligations : List ObligationRow := []
  proofArtifacts : List ProofArtifactRow := []
  deriving Repr, BEq, DecidableEq

def termRefInBounds (terms : List Term) (id : TermId) : Bool :=
  id < terms.length

def optionalTermRefInBounds (terms : List Term) : Option TermId → Bool
  | none => true
  | some id => termRefInBounds terms id

def FormulaRef.wf (terms : List Term) : FormulaRef → Bool
  | .term id => termRefInBounds terms id

def Term.wf (terms : List Term) : Term → Bool
  | .boolLit _ => true
  | .intLit _ => true
  | .variable _ => true
  | .old id => termRefInBounds terms id
  | .result => true
  | .placeRead _ => true
  | .unary u => termRefInBounds terms u.operand
  | .binary b =>
      termRefInBounds terms b.lhs &&
      termRefInBounds terms b.rhs
  | .refinementPredicate p =>
      termRefInBounds terms p.value &&
      p.args.all (termRefInBounds terms)
  | .quantified q =>
      optionalTermRefInBounds terms q.condition &&
      termRefInBounds terms q.body

def termRefBefore (id : TermId) (ref : TermId) : Bool :=
  ref < id

def optionalTermRefBefore (id : TermId) : Option TermId → Bool
  | none => true
  | some ref => termRefBefore id ref

def Term.refsBefore (id : TermId) : Term → Bool
  | .boolLit _ => true
  | .intLit _ => true
  | .variable _ => true
  | .old ref => termRefBefore id ref
  | .result => true
  | .placeRead _ => true
  | .unary u => termRefBefore id u.operand
  | .binary b =>
      termRefBefore id b.lhs &&
      termRefBefore id b.rhs
  | .refinementPredicate p =>
      termRefBefore id p.value &&
      p.args.all (termRefBefore id)
  | .quantified q =>
      optionalTermRefBefore id q.condition &&
      termRefBefore id q.body

def termsTopologicalFrom : TermId → List Term → Bool
  | _, [] => true
  | id, term :: rest =>
      term.refsBefore id &&
      termsTopologicalFrom (id + 1) rest

def ObligationKind.wf (terms : List Term) : ObligationKind → Bool
  | .logical _ formula => formula.wf terms
  | .runtimeGuard _ formula => formula.wf terms
  | .effectFrame _ => true
  | .resource goal =>
      match goal.amount with
      | none => true
      | some formula => formula.wf terms
  | .quantifier _ => true
  | .backendFact _ => true

def AssumptionRow.wf (terms : List Term) (row : AssumptionRow) : Bool :=
  match row.formula with
  | none => true
  | some formula => formula.wf terms

def ObligationRow.wf (terms : List Term) (row : ObligationRow) : Bool :=
  row.kind.wf terms

def obligationIdExists (obligations : List ObligationRow) (id : Id) : Bool :=
  obligations.any (fun row => row.id == id)

def ProofArtifactRow.wf (obligations : List ObligationRow) (row : ProofArtifactRow) : Bool :=
  row.obligationIds.all (obligationIdExists obligations)

def Manifest.wf (manifest : Manifest) : Bool :=
  manifest.terms.all (Term.wf manifest.terms) &&
  manifest.assumptions.all (AssumptionRow.wf manifest.terms) &&
  manifest.obligations.all (ObligationRow.wf manifest.terms) &&
  manifest.proofArtifacts.all (ProofArtifactRow.wf manifest.obligations)

def Manifest.termsTopological (manifest : Manifest) : Bool :=
  termsTopologicalFrom 0 manifest.terms

end Ora.Obligation

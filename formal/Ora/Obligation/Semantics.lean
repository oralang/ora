/-
Semantic interpretation for the supported obligation fragment.

`Manifest.lean` checks structural references. This file gives the supported
term fragment a fail-closed meaning. Unsupported syntax returns `none`;
`obligationDenotes` maps that to `False`, so Lean cannot silently discharge an
obligation whose manifest syntax is not interpreted here.

Storage terms are deliberately narrow. A bare `placeRead` is emitted only after
the compiler mutation gate proves that the root is stable across the query
span. `old(placeRead root)` denotes function-entry storage for that root. The
entry and stable place values are separate total-map keys; Lean does not assume
a frame equality between them.
-/

import Ora.Obligation.Manifest
import Ora.Obligation.BitVec
import Ora.Resource.Theorems
import Ora.Spec.Facts

namespace Ora.Obligation

inductive Value where
  | bool : Bool → Value
  | u256 : U256 → Value
  deriving Repr

inductive PlaceBindingKey where
  | stable : PlaceRef → PlaceBindingKey
  | entry : PlaceRef → PlaceBindingKey
  deriving Repr, BEq, DecidableEq

structure Env where
  freeBindings : List (FreeVarId × Value) := []
  placeValue : PlaceBindingKey → Value := fun _ => .u256 (BitVec.ofNat 256 0)
  /-- Bound variables use De Bruijn order: head is index `0`. -/
  boundBindings : List Value := []
  result : Option Value := none

def Env.empty : Env := {}

def lookupFreeBinding (bindings : List (FreeVarId × Value)) (id : FreeVarId) :
    Option Value :=
  match bindings with
  | [] => none
  | (candidate, value) :: rest =>
      if candidate == id then
        some value
      else
        lookupFreeBinding rest id

def Env.lookupFree (env : Env) (id : FreeVarId) : Option Value :=
  lookupFreeBinding env.freeBindings id

def Env.lookupPlace (env : Env) (place : PlaceRef) : Option Value :=
  some (env.placeValue (.stable place))

def Env.lookupEntryPlace (env : Env) (place : PlaceRef) : Option Value :=
  some (env.placeValue (.entry place))

def Env.lookupBound (env : Env) (index : Nat) : Option Value :=
  env.boundBindings[index]?

def Env.lookupVar (env : Env) : VarRef → Option Value
  | .free free => env.lookupFree free.id
  | .bound bound => env.lookupBound bound.index

def Env.setFree (env : Env) (id : FreeVarId) (value : Value) : Env :=
  { env with freeBindings := (id, value) :: env.freeBindings }

def Env.setPlace (env : Env) (place : PlaceRef) (value : Value) : Env :=
  { env with placeValue := fun key =>
      if key == .stable place then value else env.placeValue key }

def Env.setEntryPlace (env : Env) (place : PlaceRef) (value : Value) : Env :=
  { env with placeValue := fun key =>
      if key == .entry place then value else env.placeValue key }

def Env.pushBound (env : Env) (value : Value) : Env :=
  { env with boundBindings := value :: env.boundBindings }

theorem lookupBound_zero_after_pushBound (env : Env) (value : Value) :
    (env.pushBound value).lookupBound 0 = some value := by
  simp [Env.pushBound, Env.lookupBound]

theorem lookupBound_succ_after_pushBound (env : Env) (value : Value) (index : Nat) :
    (env.pushBound value).lookupBound (index + 1) = env.lookupBound index := by
  simp [Env.pushBound, Env.lookupBound, Nat.add_comm]

theorem lookupVar_uses_free_id_not_name (name : String) (lhs rhs : Value) :
    (((Env.empty.setFree { file_id := 0, pattern_id := 0 } lhs).setFree
      { file_id := 0, pattern_id := 1 } rhs).lookupVar
        (.free { id := { file_id := 0, pattern_id := 0 }, name := name })) =
      some lhs := by
  rfl

theorem lookupVar_uses_debruijn_index_not_name (name : String) (lhs rhs : Value) :
    (((Env.empty.pushBound lhs).pushBound rhs).lookupVar
      (.bound { index := 1, name := name })) = some lhs := by
  rfl

theorem lookupPlace_uses_place_identity (value : Value) :
    (Env.empty.setPlace { root := "reserve", region := .storage } value).lookupPlace
      { root := "reserve", region := .storage } = some value := by
  rfl

theorem lookupEntryPlace_uses_entry_place_identity (value : Value) :
    (Env.empty.setEntryPlace { root := "reserve", region := .storage } value).lookupEntryPlace
      { root := "reserve", region := .storage } = some value := by
  rfl

theorem stable_place_does_not_link_entry_place (value : Value) :
    (Env.empty.setPlace { root := "reserve", region := .storage } value).lookupEntryPlace
      { root := "reserve", region := .storage } =
        Env.empty.lookupEntryPlace { root := "reserve", region := .storage } := by
  rfl

def sampleStoragePlace : PlaceRef :=
  { root := "reserve", region := .storage }

def sampleOldPlaceReadManifest : Manifest :=
  { terms := [.placeRead sampleStoragePlace, .old 0] }

def compilerTypeIdU256 : Nat := Ora.Spec.expectedCompilerTypeIdU256
def compilerTypeIdI256 : Nat := Ora.Spec.expectedCompilerTypeIdI256
def compilerTypeIdBool : Nat := Ora.Spec.expectedCompilerTypeIdBool

def TyRef.isU256 : TyRef → Bool
  | .spelling name => name == "u256" || name == "uint256"
  | .compilerTypeId id => id == compilerTypeIdU256

def TyRef.isI256 : TyRef → Bool
  | .spelling name => name == "i256" || name == "int256"
  | .compilerTypeId id => id == compilerTypeIdI256

def TyRef.isBool : TyRef → Bool
  | .spelling name => name == "bool" || name == "i1"
  | .compilerTypeId id => id == compilerTypeIdBool

def TyRef.isU256Carrier (ty : TyRef) : Bool :=
  ty.isU256 || ty.isI256

def FreeVarRef.isU256 (var : FreeVarRef) : Bool :=
  match var.ty with
  | some ty => ty.isU256Carrier
  | none => false

def BoundVarRef.isU256 (var : BoundVarRef) : Bool :=
  match var.ty with
  | some ty => ty.isU256Carrier
  | none => false

def BinderRef.isU256 (binder : BinderRef) : Bool :=
  match binder.ty with
  | some ty => ty.isU256Carrier
  | none => false

def VarRef.isU256 : VarRef → Bool
  | VarRef.free var => FreeVarRef.isU256 var
  | VarRef.bound var => BoundVarRef.isU256 var

/-
Compiler manifests carry integer literals as normalized ASCII decimal text.
Keep their interpretation kernel-reducible: Lean 4.15's `String.toNat?` has an
opaque runtime implementation, which would make otherwise concrete proof
obligations depend on `native_decide`. Any non-decimal spelling fails closed.
-/
def decimalDigit? : Char → Option Nat
  | '0' => some 0
  | '1' => some 1
  | '2' => some 2
  | '3' => some 3
  | '4' => some 4
  | '5' => some 5
  | '6' => some 6
  | '7' => some 7
  | '8' => some 8
  | '9' => some 9
  | _ => none

def parseDecimalNatAux : List Char → Nat → Bool → Option Nat
  | [], value, seen => if seen then some value else none
  | digit :: rest, value, _ => do
      let parsed ← decimalDigit? digit
      parseDecimalNatAux rest (value * 10 + parsed) true

def parseDecimalNat? (text : String) : Option Nat :=
  parseDecimalNatAux text.data 0 false

example : parseDecimalNat? "3" = some 3 := by rfl
example : parseDecimalNat? "123" = some 123 := by rfl
example : parseDecimalNat? "" = none := by rfl
example : parseDecimalNat? "1_0" = none := by rfl

def IntegerLiteralTerm.asU256? (lit : IntegerLiteralTerm) : Option U256 :=
  match lit.ty with
  | some ty =>
      if ty.isU256Carrier then
        match lit.value with
        | "0" => some (BitVec.ofNat 256 0)
        | "1" => some (BitVec.ofNat 256 1)
        | _ => parseDecimalNat? lit.value |>.map (BitVec.ofNat 256)
      else
        none
  | none => none

def Value.eqProp? : Value → Value → Option Prop
  | .bool lhs, .bool rhs => some (lhs = rhs)
  | .u256 lhs, .u256 rhs => some (lhs = rhs)
  | _, _ => none

def Value.binaryU256? (op : U256 → U256 → U256) : Value → Value → Option Value
  | .u256 lhs, .u256 rhs => some (.u256 (op lhs rhs))
  | _, _ => none

def Value.refinementPredicate? (name : String) (value : Value) (args : List Value) :
    Option Prop :=
  match name, value, args with
  | "NonZero", .u256 value, [] =>
      some (value ≠ BitVec.ofNat 256 0)
  | "NonZeroAddress", .u256 value, [] =>
      some (value ≠ BitVec.ofNat 256 0)
  | "MinValue", .u256 value, [.u256 bound] =>
      some (U256.ule bound value)
  | "MaxValue", .u256 value, [.u256 bound] =>
      some (U256.ule value bound)
  | "InRange", .u256 value, [.u256 lower, .u256 upper] =>
      some (U256.ule lower value ∧ U256.ule value upper)
  | "BasisPoints", .u256 value, [] =>
      some (U256.ule (BitVec.ofNat 256 0) value ∧
        U256.ule value (BitVec.ofNat 256 10000))
  | _, _, _ => none

theorem unsupported_refinement_predicate_fails_closed (value : U256) :
    Value.refinementPredicate? "Exact" (.u256 value) [] = none := rfl

theorem malformed_refinement_arity_fails_closed (value : U256) :
    Value.refinementPredicate? "MinValue" (.u256 value) [] = none := rfl

mutual

def denoteValue? (manifest : Manifest) (env : Env) : Nat → TermId → Option Value
  | 0, _ => none
  | fuel + 1, id =>
      match manifest.terms[id]? with
      | some (.boolLit value) => some (.bool value)
      | some (.intLit lit) => lit.asU256?.map Value.u256
      | some (.variable var) => env.lookupVar var
      | some (.old operand) =>
          match manifest.terms[operand]? with
          | some (.placeRead place) => env.lookupEntryPlace place
          | _ => none
      | some .result => env.result
      | some (.placeRead place) => env.lookupPlace place
      | some (.binary binary) =>
          let binaryU256? (op : U256 → U256 → U256) :=
            match binary.ty with
            | some ty =>
                if ty.isU256Carrier then
                  match denoteValue? manifest env fuel binary.lhs,
                        denoteValue? manifest env fuel binary.rhs with
                  | some lhsValue, some rhsValue =>
                      lhsValue.binaryU256? op rhsValue
                  | _, _ => none
                else
                  none
            | none => none
          let div? :=
            match binary.ty with
            | some ty =>
                if ty.isU256 then
                  match denoteValue? manifest env fuel binary.lhs,
                        denoteValue? manifest env fuel binary.rhs with
                  | some lhsValue, some rhsValue =>
                      lhsValue.binaryU256? U256.udivTotal rhsValue
                  | _, _ => none
                else if ty.isI256 then
                  match denoteValue? manifest env fuel binary.lhs,
                        denoteValue? manifest env fuel binary.rhs with
                  | some lhsValue, some rhsValue =>
                      lhsValue.binaryU256? U256.sdivTotal rhsValue
                  | _, _ => none
                else
                  none
            | none => none
          let mod? :=
            match binary.ty with
            | some ty =>
                if ty.isU256 then
                  match denoteValue? manifest env fuel binary.lhs,
                        denoteValue? manifest env fuel binary.rhs with
                  | some lhsValue, some rhsValue =>
                      lhsValue.binaryU256? U256.uremTotal rhsValue
                  | _, _ => none
                else if ty.isI256 then
                  match denoteValue? manifest env fuel binary.lhs,
                        denoteValue? manifest env fuel binary.rhs with
                  | some lhsValue, some rhsValue =>
                      lhsValue.binaryU256? U256.sremTotal rhsValue
                  | _, _ => none
                else
                  none
            | none => none
          match binary.op with
          | .add => binaryU256? U256.add
          | .sub => binaryU256? U256.sub
          | .mul => binaryU256? U256.mul
          | .div => div?
          | .mod_ => mod?
          | _ => none
      | _ => none

def denoteValueList? (manifest : Manifest) (env : Env) (fuel : Nat) :
    List TermId → Option (List Value)
  | [] => some []
  | id :: rest =>
      match denoteValue? manifest env fuel id,
            denoteValueList? manifest env fuel rest with
      | some value, some values => some (value :: values)
      | _, _ => none

def denoteFormula? (manifest : Manifest) (env : Env) : Nat → TermId → Option Prop
  | 0, _ => none
  | fuel + 1, id =>
      match manifest.terms[id]? with
      | some (.boolLit value) => some (value = true)
      | some (.variable var) =>
          match env.lookupVar var with
          | some (.bool value) => some (value = true)
          | _ => none
      | some (.refinementPredicate predicate) =>
          match denoteValue? manifest env fuel predicate.value,
                denoteValueList? manifest env fuel predicate.args with
          | some value, some args => Value.refinementPredicate? predicate.name value args
          | _, _ => none
      | some (.unary unary) =>
          match unary.op with
          | .not_ =>
              match denoteFormula? manifest env fuel unary.operand with
              | some proposition => some (¬ proposition)
              | none => none
          | .neg => none
      | some (.binary binary) =>
          let binU256? (relation : U256 → U256 → Prop) :=
            match denoteValue? manifest env fuel binary.lhs,
                  denoteValue? manifest env fuel binary.rhs with
            | some (.u256 lhsValue), some (.u256 rhsValue) =>
                some (relation lhsValue rhsValue)
            | _, _ => none
          let binFormula? (relation : Prop → Prop → Prop) :=
            match denoteFormula? manifest env fuel binary.lhs,
                  denoteFormula? manifest env fuel binary.rhs with
            | some lhsProp, some rhsProp => some (relation lhsProp rhsProp)
            | _, _ => none
          match binary.op with
          | .eq =>
              match denoteValue? manifest env fuel binary.lhs,
                    denoteValue? manifest env fuel binary.rhs with
              | some lhs, some rhs => lhs.eqProp? rhs
              | _, _ => none
          | .ne =>
              match denoteValue? manifest env fuel binary.lhs,
                    denoteValue? manifest env fuel binary.rhs with
              | some lhs, some rhs =>
                  match lhs.eqProp? rhs with
                  | some eq => some (¬ eq)
                  | none => none
              | _, _ => none
          | .lt =>
              binU256? U256.ult
          | .le =>
              binU256? U256.ule
          | .gt =>
              binU256? U256.ugt
          | .ge =>
              binU256? U256.uge
          | .slt =>
              binU256? U256.slt
          | .sle =>
              binU256? U256.sle
          | .sgt =>
              binU256? U256.sgt
          | .sge =>
              binU256? U256.sge
          | .and_ =>
              binFormula? (fun lhs rhs => lhs ∧ rhs)
          | .or_ =>
              binFormula? (fun lhs rhs => lhs ∨ rhs)
          | .implies =>
              binFormula? (fun lhs rhs => lhs → rhs)
          | _ => none
      | some (.quantified quantified) =>
          if !quantified.binder.isU256 then
            none
          else
            match quantified.quantifier with
            | .forall_ =>
                some (
                  ∀ value : U256,
                    let localEnv := env.pushBound (.u256 value);
                    match quantified.condition with
                    | none =>
                        match denoteFormula? manifest localEnv fuel quantified.body with
                        | some body => body
                        | none => False
                    | some condition =>
                        match denoteFormula? manifest localEnv fuel condition,
                              denoteFormula? manifest localEnv fuel quantified.body with
                        | some antecedent, some consequent => antecedent → consequent
                        | _, _ => False
                )
            | .exists_ =>
                some (
                  ∃ value : U256,
                    let localEnv := env.pushBound (.u256 value);
                    match quantified.condition with
                    | none =>
                        match denoteFormula? manifest localEnv fuel quantified.body with
                        | some body => body
                        | none => False
                    | some condition =>
                        match denoteFormula? manifest localEnv fuel condition,
                              denoteFormula? manifest localEnv fuel quantified.body with
                        | some antecedent, some consequent => antecedent ∧ consequent
                        | _, _ => False
                )
      | _ => none

end

theorem denoteValue_old_placeRead_uses_entry_place (value : Value) :
    denoteValue? sampleOldPlaceReadManifest
      (Env.empty.setEntryPlace sampleStoragePlace value) 3 1 = some value := by
  rfl

theorem denoteValue_old_non_place_operand_fails_closed :
    denoteValue?
      { terms := [
          .intLit { value := "0", ty := some (.spelling "u256") },
          .old 0
        ] }
      Env.empty 3 1 = none := by
  rfl

def formulaDenotes? (manifest : Manifest) (env : Env) : FormulaRef → Option Prop
  | .term id => denoteFormula? manifest env (manifest.terms.length + 1) id

def formulaValue? (manifest : Manifest) (env : Env) : FormulaRef → Option Value
  | .term id => denoteValue? manifest env (manifest.terms.length + 1) id

def formulaU256? (manifest : Manifest) (env : Env) (formula : FormulaRef) : Option U256 :=
  match formulaValue? manifest env formula with
  | some (.u256 value) => some value
  | _ => none

def Env.resourceState (env : Env) : Ora.Resource.State PlaceRef :=
  fun place =>
    match env.placeValue (.stable place) with
    | .u256 value => value
    | .bool _ => U256.zero

def Env.resourcePlaceValue? (env : Env) (place : PlaceRef) : Option U256 :=
  match env.placeValue (.stable place) with
  | .u256 value => some value
  | .bool _ => none

def Env.resourcePlaceKnown? (env : Env) (place : PlaceRef) : Option PlaceRef :=
  match env.resourcePlaceValue? place with
  | some _ => some place
  | none => none

def placeListCovers (declared actual : List PlaceRef) : Bool :=
  actual.all (fun place => declared.contains place)

def RegionRef.isConcrete : RegionRef → Bool
  | .none => false
  | .storage
  | .memory
  | .transient
  | .calldata => true

def computedStorageRoot : String := "$computed_storage"

def placeKeyConstantNat? (value : String) : Option Nat :=
  match parseDecimalNat? value with
  | some parsed =>
      if parsed < 2^256 then some parsed else none
  | none => none

def placeKeysDefinitelyDistinct : PlaceKey → PlaceKey → Bool
  | .constant lhs, .constant rhs =>
      match placeKeyConstantNat? lhs, placeKeyConstantNat? rhs with
      | some lhsValue, some rhsValue => lhsValue != rhsValue
      | _, _ => false
  | _, _ => false

def placeKeyListsDefinitelyDisjoint : List PlaceKey → List PlaceKey → Bool
  | [], [] => false
  | [], _ :: _ => false
  | _ :: _, [] => false
  | lhs :: lhsRest, rhs :: rhsRest =>
      if lhs == rhs then
        placeKeyListsDefinitelyDisjoint lhsRest rhsRest
      else
        placeKeysDefinitelyDistinct lhs rhs

def placeDefinitelyDisjoint (lhs rhs : PlaceRef) : Bool :=
  if !lhs.region.isConcrete || !rhs.region.isConcrete then
    false
  else if lhs.root == computedStorageRoot || rhs.root == computedStorageRoot then
    false
  else if lhs.region != rhs.region then
    true
  else if lhs.root != rhs.root then
    true
  else if lhs.fields != rhs.fields then
    false
  else
    placeKeyListsDefinitelyDisjoint lhs.keys rhs.keys

def placeListDisjoint (declared actual : List PlaceRef) : Bool :=
  actual.all (fun read => declared.all (fun write =>
    placeDefinitelyDisjoint read write))

def Manifest.assumptionById (manifest : Manifest) (id : Id) : Option AssumptionRow :=
  manifest.assumptions.find? (fun row => row.id == id)

def optionPropAnd? (lhs rhs : Option Prop) : Option Prop :=
  match lhs, rhs with
  | some lhsProp, some rhsProp => some (lhsProp ∧ rhsProp)
  | _, _ => none

def termFreeVarId? (manifest : Manifest) (id : TermId) : Option FreeVarId :=
  match manifest.terms[id]? with
  | some (.variable (.free free)) => some free.id
  | _ => none

def freeVarPairMatches (lhs rhs first second : FreeVarId) : Bool :=
  (lhs == first && rhs == second) || (lhs == second && rhs == first)

def placeKeysEqualBefore : Nat → List PlaceKey → List PlaceKey → Bool
  | 0, _, _ => true
  | n + 1, lhs :: lhsRest, rhs :: rhsRest =>
      lhs == rhs && placeKeysEqualBefore n lhsRest rhsRest
  | _ + 1, _, _ => false

def keyEvidencePathMatches
    (read write : PlaceRef)
    (keyIndex : Nat)
    (lhs rhs : FreeVarId) : Bool :=
  if !read.region.isConcrete || !write.region.isConcrete then
    false
  else if read.root == computedStorageRoot || write.root == computedStorageRoot then
    false
  else if read.region != write.region then
    false
  else if read.root != write.root then
    false
  else if read.fields != write.fields then
    false
  else if read.keys.length != write.keys.length then
    false
  else if !placeKeysEqualBefore keyIndex read.keys write.keys then
    false
  else
    match read.keys[keyIndex]?, write.keys[keyIndex]? with
    | some (.parameter readId), some (.parameter writeId) =>
        readId != writeId && freeVarPairMatches lhs rhs readId writeId
    | _, _ => false

def keyDisjointEvidenceFormulaDenotes?
    (manifest : Manifest)
    (env : Env)
    (evidence : KeyDisjointEvidence) : Option Prop :=
  match manifest.assumptionById evidence.assumptionId with
  | some row =>
      if row.kind != .requires then
        none
      else
        match row.formula with
        | some (.term termId) =>
            match manifest.terms[termId]? with
            | some (.binary binary) =>
                if binary.op != .ne then
                  none
                else
                  match termFreeVarId? manifest binary.lhs,
                        termFreeVarId? manifest binary.rhs with
                  | some lhs, some rhs =>
                      if freeVarPairMatches lhs rhs evidence.lhs evidence.rhs then
                        formulaDenotes? manifest env (.term termId)
                      else
                        none
                  | _, _ => none
            | _ => none
        | _ => none
  | none => none

def keyDisjointEvidenceDenotes?
    (manifest : Manifest)
    (env : Env)
    (evidence : KeyDisjointEvidence) : Option Prop :=
  match evidence.kind with
  | .freeVarDisequality =>
      if keyEvidencePathMatches evidence.read evidence.write evidence.keyIndex
          evidence.lhs evidence.rhs then
        keyDisjointEvidenceFormulaDenotes? manifest env evidence
      else
        none

def evidenceListDenotes? (manifest : Manifest) (env : Env) :
    List KeyDisjointEvidence → Option Prop
  | [] => some True
  | evidence :: rest =>
      optionPropAnd?
        (keyDisjointEvidenceDenotes? manifest env evidence)
        (evidenceListDenotes? manifest env rest)

def evidenceMatchesPair (read write : PlaceRef) (evidence : KeyDisjointEvidence) : Bool :=
  evidence.read == read && evidence.write == write

def pairCoveredByEvidence (read write : PlaceRef) (evidence : List KeyDisjointEvidence) :
    Bool :=
  evidence.any (evidenceMatchesPair read write)

def placePairDisjointWithEvidence?
    (manifest : Manifest)
    (env : Env)
    (evidence : List KeyDisjointEvidence)
    (read write : PlaceRef) : Option Prop :=
  if placeDefinitelyDisjoint read write then
    some True
  else if pairCoveredByEvidence read write evidence then
    evidenceListDenotes? manifest env evidence
  else
    some False

def placeListDisjointWithEvidence? (manifest : Manifest) (env : Env)
    (declared actual : List PlaceRef) (evidence : List KeyDisjointEvidence) :
    Option Prop :=
  actual.foldl
    (fun acc read =>
      declared.foldl
        (fun inner write =>
          optionPropAnd? inner
            (placePairDisjointWithEvidence? manifest env evidence read write))
        acc)
    (some True)

def effectFrameGoalDenotes? (manifest : Manifest) (env : Env) (goal : EffectFrameGoal) :
    Option Prop :=
  match goal.relation with
  | .writeCoveredByModifies =>
      some (placeListCovers goal.declared goal.actual = true)
  | .readPreservedByFrame =>
      some (placeListDisjoint goal.declared goal.actual = true)
  | .readPreservedByKeyEvidence =>
      placeListDisjointWithEvidence? manifest env goal.declared goal.actual goal.evidence
  | .lockCoversWrite
  | .externalCallFrame =>
      none

def resourceGoalAmount? (manifest : Manifest) (env : Env) (goal : ResourceGoal) : Option U256 :=
  match goal.amount with
  | some formula => formulaU256? manifest env formula
  | none => none

def resourceGoalSource? (env : Env) (goal : ResourceGoal) : Option PlaceRef :=
  match goal.source with
  | some source => env.resourcePlaceKnown? source
  | none => none

def resourceGoalDestination? (env : Env) (goal : ResourceGoal) : Option PlaceRef :=
  match goal.destination with
  | some destination => env.resourcePlaceKnown? destination
  | none => none

def resourceGoalDenotes? (manifest : Manifest) (env : Env) (goal : ResourceGoal) :
    Option Prop :=
  let state := env.resourceState
  match goal.property with
  | .amountNonNegative =>
      -- U256 values are structurally non-negative. Keep this as a proposition
      -- rather than `some True` so the resource property remains explicitly
      -- denoted and cannot disappear through the semantics seam.
      match resourceGoalAmount? manifest env goal with
      | some amount => some (Ora.Resource.amountNonNegative amount)
      | none => none
  | .sourceSufficient =>
      match goal.op, resourceGoalSource? env goal, resourceGoalDestination? env goal,
          resourceGoalAmount? manifest env goal with
      | .move, some source, some destination, some amount =>
          some (Ora.Resource.moveSourceSufficient state source destination amount)
      | .destroy, some source, _, some amount =>
          some (Ora.Resource.sourceSufficient state source amount)
      | _, _, _, _ => none
  | .destinationNoOverflow =>
      match goal.op, resourceGoalSource? env goal, resourceGoalDestination? env goal,
          resourceGoalAmount? manifest env goal with
      | .move, some source, some destination, some amount =>
          some (Ora.Resource.moveDestinationNoOverflow state source destination amount)
      | .create, _, some destination, some amount =>
          some (Ora.Resource.destinationNoOverflow state destination amount)
      | _, _, _, _ => none
  | .samePlaceIdentity =>
      -- This is informative only in the alias case: for distinct places the
      -- implication is vacuous, while equal places cite the model's self-move
      -- identity behavior.
      match goal.op, resourceGoalSource? env goal, resourceGoalDestination? env goal,
          resourceGoalAmount? manifest env goal with
      | .move, some source, some destination, some amount =>
          some (source = destination →
            Ora.Resource.move state source destination amount = state)
      | _, _, _, _ => none
  | .conservation =>
      match goal.op, resourceGoalSource? env goal, resourceGoalDestination? env goal,
          resourceGoalAmount? manifest env goal with
      | .move, some source, some destination, some amount =>
          some (source ≠ destination →
            Ora.Resource.sourceSufficient state source amount →
              Ora.Resource.destinationNoOverflow state destination amount →
                (Ora.Resource.move state source destination amount source).toNat +
                    (Ora.Resource.move state source destination amount destination).toNat =
                  (state source).toNat + (state destination).toNat)
      | _, _, _, _ => none

def assumptionDenotesInEnv? (manifest : Manifest) (env : Env) (row : AssumptionRow) :
    Option Prop :=
  match row.formula with
  | some formula => formulaDenotes? manifest env formula
  | none => none

def obligationDenotesInEnv? (manifest : Manifest) (env : Env) (row : ObligationRow) :
    Option Prop :=
  match row.kind with
  | .logical _ formula => formulaDenotes? manifest env formula
  | .runtimeGuard _ formula => formulaDenotes? manifest env formula
  | .effectFrame goal => effectFrameGoalDenotes? manifest env goal
  | .resource goal => resourceGoalDenotes? manifest env goal
  | _ => none

def obligationDenotes? (manifest : Manifest) (row : ObligationRow) : Option Prop :=
  obligationDenotesInEnv? manifest Env.empty row

def assumptionAnd? (lhs rhs : Option Prop) : Option Prop :=
  match lhs, rhs with
  | some lhsProp, some rhsProp => some (lhsProp ∧ rhsProp)
  | _, _ => none

def assumptionsDenoteInEnv? (manifest : Manifest) (env : Env) :
    List AssumptionRow → Option Prop
  | [] => some True
  | row :: rest =>
      assumptionAnd?
        (assumptionDenotesInEnv? manifest env row)
        (assumptionsDenoteInEnv? manifest env rest)

def assumptionsDenoteInEnv (manifest : Manifest) (env : Env) (rows : List AssumptionRow) :
    Prop :=
  match assumptionsDenoteInEnv? manifest env rows with
  | some proposition => proposition
  | none => False

def assumptionsSatisfiable (manifest : Manifest) (rows : List AssumptionRow) : Prop :=
  ∃ env : Env, assumptionsDenoteInEnv manifest env rows

def obligationDenotesInEnv (manifest : Manifest) (env : Env) (row : ObligationRow) : Prop :=
  match obligationDenotesInEnv? manifest env row with
  | some proposition => proposition
  | none => False

def obligationDenotes (manifest : Manifest) (row : ObligationRow) : Prop :=
  obligationDenotesInEnv manifest Env.empty row

def obligationFollowsFromAssumptions
    (manifest : Manifest)
    (assumptions : List AssumptionRow)
    (row : ObligationRow) : Prop :=
  assumptionsSatisfiable manifest assumptions ∧
    ∀ env : Env,
      assumptionsDenoteInEnv manifest env assumptions →
        obligationDenotesInEnv manifest env row

end Ora.Obligation

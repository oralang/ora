/-
Semantic interpretation for the supported obligation fragment.

`Manifest.lean` checks structural references. This file gives the supported
term fragment a fail-closed meaning. Unsupported syntax returns `none`;
`obligationDenotes` maps that to `False`, so Lean cannot silently discharge an
obligation whose manifest syntax is not interpreted here.
-/

import Ora.Obligation.Manifest
import Ora.Obligation.BitVec

namespace Ora.Obligation

inductive Value where
  | bool : Bool → Value
  | u256 : U256 → Value
  deriving Repr

structure Env where
  freeBindings : List (FreeVarId × Value) := []
  /-- Bound variables use De Bruijn order: head is index `0`. -/
  boundBindings : List Value := []
  result : Option Value := none
  deriving Repr

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

def Env.lookupBound (env : Env) (index : Nat) : Option Value :=
  env.boundBindings[index]?

def Env.lookupVar (env : Env) : VarRef → Option Value
  | .free free => env.lookupFree free.id
  | .bound bound => env.lookupBound bound.index

def Env.setFree (env : Env) (id : FreeVarId) (value : Value) : Env :=
  { env with freeBindings := (id, value) :: env.freeBindings }

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

def TyRef.isU256 : TyRef → Bool
  | .spelling name => name == "u256" || name == "uint256"
  | .compilerTypeId _ => false

def FreeVarRef.isU256 (var : FreeVarRef) : Bool :=
  match var.ty with
  | some ty => ty.isU256
  | none => false

def BoundVarRef.isU256 (var : BoundVarRef) : Bool :=
  match var.ty with
  | some ty => ty.isU256
  | none => false

def BinderRef.isU256 (binder : BinderRef) : Bool :=
  match binder.ty with
  | some ty => ty.isU256
  | none => false

def VarRef.isU256 : VarRef → Bool
  | VarRef.free var => FreeVarRef.isU256 var
  | VarRef.bound var => BoundVarRef.isU256 var

def IntegerLiteralTerm.asU256? (lit : IntegerLiteralTerm) : Option U256 :=
  match lit.ty with
  | some ty =>
      if ty.isU256 then
        lit.value.toNat?.map (BitVec.ofNat 256)
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
      | some .result => env.result
      | some (.binary binary) =>
          match denoteValue? manifest env fuel binary.lhs,
                denoteValue? manifest env fuel binary.rhs with
          | some lhsValue, some rhsValue =>
              match binary.op with
              | .add => lhsValue.binaryU256? U256.add rhsValue
              | .sub => lhsValue.binaryU256? U256.sub rhsValue
              | _ => none
          | _, _ => none
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

def formulaDenotes? (manifest : Manifest) (env : Env) : FormulaRef → Option Prop
  | .term id => denoteFormula? manifest env (manifest.terms.length + 1) id

def placeListCovers (declared actual : List PlaceRef) : Bool :=
  actual.all (fun place => declared.contains place)

def placeListDisjoint (declared actual : List PlaceRef) : Bool :=
  actual.all (fun place => !(declared.contains place))

def EffectFrameGoal.denotes? (goal : EffectFrameGoal) : Option Prop :=
  match goal.relation with
  | .writeCoveredByModifies =>
      some (placeListCovers goal.declared goal.actual = true)
  | .readPreservedByFrame =>
      some (placeListDisjoint goal.declared goal.actual = true)
  | .lockCoversWrite
  | .externalCallFrame =>
      none

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
  | .effectFrame goal => goal.denotes?
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

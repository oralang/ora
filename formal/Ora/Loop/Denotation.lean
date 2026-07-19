/-
Fail-closed denotation of compiler-emitted scalar loop summaries.

The compiler emits only the data structures below. Unsupported identities,
types, references, or update layouts make `denoteLoopSummary?` return `none`.
-/

import Ora.Loop.Core
import Ora.Obligation.Semantics

namespace Ora.Loop

open Ora.Obligation

inductive LoopKind where
  | scfWhile
  | scfFor
  | other
  deriving Repr, BEq, DecidableEq

structure VariableRow where
  index : Nat
  id : FreeVarId
  name : String := ""
  ty : TyRef
  deriving Repr, BEq, DecidableEq

structure StepAssignmentRow where
  variableIndex : Nat
  target : FreeVarId
  value : FormulaRef
  deriving Repr, BEq, DecidableEq

structure SummaryRow where
  id : Nat
  owner : String
  kind : LoopKind
  contextVariables : List VariableRow := []
  variables : List VariableRow := []
  init : List FormulaRef := []
  guard : Option FormulaRef := none
  invariants : List FormulaRef := []
  step : List StepAssignmentRow := []
  post : List FormulaRef := []
  unsupportedReasons : List String := []
  deriving Repr, BEq, DecidableEq

def idsUnique (variables : List VariableRow) : Bool :=
  decide (variables.map VariableRow.id).Nodup

def variablesIndexedFrom : Nat → List VariableRow → Bool
  | _, [] => true
  | expected, loopVar :: rest =>
      loopVar.index == expected && variablesIndexedFrom (expected + 1) rest

def variablesAreU256 (variables : List VariableRow) : Bool :=
  variables.all (fun loopVar => loopVar.ty.isU256)

def variableSetsDisjoint (lhs rhs : List VariableRow) : Bool :=
  lhs.all (fun left => rhs.all (fun right => left.id != right.id))

def assignmentsMatchVariables : List VariableRow → List StepAssignmentRow → Bool
  | [], [] => true
  | loopVar :: variables, assignment :: assignments =>
      assignment.variableIndex == loopVar.index &&
        assignment.target == loopVar.id &&
        assignmentsMatchVariables variables assignments
  | _, _ => false

def valueForTy : TyRef → Value
  | ty => if ty.isBool then .bool true else .u256 (BitVec.ofNat 256 0)

def bindFreeVarsFromTerm (env : Env) : Term → Env
  | .variable (.free free) =>
      match free.ty with
      | some ty => env.setFree free.id (valueForTy ty)
      | none => env.setFree free.id (.u256 (BitVec.ofNat 256 0))
  | _ => env

def totalEnv (manifest : Manifest) : Env :=
  manifest.terms.foldl bindFreeVarsFromTerm Env.empty

def formulaDenotable (manifest : Manifest) (formula : FormulaRef) : Bool :=
  (formulaDenotes? manifest (totalEnv manifest) formula).isSome

def valueDenotable (manifest : Manifest) (formula : FormulaRef) : Bool :=
  (formulaValue? manifest (totalEnv manifest) formula).isSome

def SummaryRow.shapeSupported (row : SummaryRow) : Bool :=
  row.unsupportedReasons.isEmpty &&
    (row.kind == .scfWhile || row.kind == .scfFor) &&
    row.variables.length > 0 &&
    row.init.length == row.variables.length &&
    row.guard.isSome &&
    row.invariants.length > 0 &&
    row.post.length > 0 &&
    variablesIndexedFrom 0 row.variables &&
    idsUnique row.variables &&
    idsUnique row.contextVariables &&
    variablesAreU256 row.variables &&
    variablesAreU256 row.contextVariables &&
    variableSetsDisjoint row.variables row.contextVariables &&
    assignmentsMatchVariables row.variables row.step

def SummaryRow.formulasDenotable (manifest : Manifest) (row : SummaryRow) : Bool :=
  row.init.all (valueDenotable manifest) &&
    row.guard.all (formulaDenotable manifest) &&
    row.invariants.all (formulaDenotable manifest) &&
    row.step.all (fun assignment => valueDenotable manifest assignment.value) &&
    row.post.all (formulaDenotable manifest)

def SummaryRow.supported (manifest : Manifest) (row : SummaryRow) : Bool :=
  manifest.wf && manifest.termsTopological && row.shapeSupported && row.formulasDenotable manifest

def contextReady (env : Env) (variables : List VariableRow) : Prop :=
  ∀ loopVar ∈ variables, ∃ value, env.lookupFree loopVar.id = some (.u256 value)

def bindState : Env → List VariableRow → ScalarLoopState → Option Env
  | env, [], [] => some env
  | env, loopVar :: variables, value :: values =>
      bindState (env.setFree loopVar.id (.u256 value)) variables values
  | _, _, _ => none

def formulaHoldsInState
    (manifest : Manifest)
    (base : Env)
    (variables : List VariableRow)
    (state : ScalarLoopState)
    (formula : FormulaRef) : Prop :=
  match bindState base variables state with
  | some env =>
      match formulaDenotes? manifest env formula with
      | some proposition => proposition
      | none => False
  | none => False

def formulasHoldInState
    (manifest : Manifest)
    (base : Env)
    (variables : List VariableRow)
    (state : ScalarLoopState)
    (formulas : List FormulaRef) : Prop :=
  ∀ formula ∈ formulas, formulaHoldsInState manifest base variables state formula

def valuesInitializeState
    (manifest : Manifest)
    (base : Env) : List FormulaRef → ScalarLoopState → Prop
  | [], [] => True
  | formula :: formulas, value :: values =>
      formulaValue? manifest base formula = some (.u256 value) ∧
        valuesInitializeState manifest base formulas values
  | _, _ => False

def assignmentsProduceState
    (manifest : Manifest)
    (base : Env)
    (variables : List VariableRow)
    (current : ScalarLoopState) : List StepAssignmentRow → ScalarLoopState → Prop
  | [], [] => True
  | assignment :: assignments, value :: values =>
      match bindState base variables current with
      | some env =>
          formulaValue? manifest env assignment.value = some (.u256 value) ∧
            assignmentsProduceState manifest base variables current assignments values
      | none => False
  | _, _ => False

def denoteLoopSummary? (manifest : Manifest) (base : Env) (row : SummaryRow) :
    Option ScalarLoopSummary :=
  if row.supported manifest then
    match row.guard with
    | some guard => some {
        initial := fun state => valuesInitializeState manifest base row.init state
        guard := fun state => formulaHoldsInState manifest base row.variables state guard
        invariant := fun state =>
          formulasHoldInState manifest base row.variables state row.invariants
        step := fun current following =>
          assignmentsProduceState manifest base row.variables current row.step following
        post := fun state => formulasHoldInState manifest base row.variables state row.post
      }
    | none => none
  else
    none

def loopExitHolds (manifest : Manifest) (base : Env) (row : SummaryRow)
    (state : ScalarLoopState) : Prop :=
  match denoteLoopSummary? manifest base row with
  | some summary =>
      summary.invariant state ∧ ¬summary.guard state ∧ summary.post state
  | none => False

def loopExitPremises (manifest : Manifest) (base : Env) (row : SummaryRow)
    (state : ScalarLoopState) : Prop :=
  match denoteLoopSummary? manifest base row with
  | some summary => summary.invariant state ∧ ¬summary.guard state
  | none => False

def loopPostFollowsFromAssumptions
    (manifest : Manifest)
    (assumptions : List AssumptionRow)
    (row : SummaryRow) : Prop :=
  (∃ env state,
      assumptionsDenoteInEnv manifest env assumptions ∧
      contextReady env row.contextVariables ∧
      loopExitPremises manifest env row state) ∧
    ∀ env state,
      assumptionsDenoteInEnv manifest env assumptions →
      contextReady env row.contextVariables →
      loopExitPremises manifest env row state →
      loopExitHolds manifest env row state

theorem loop_post_proof_has_nonvacuous_witness
    (manifest : Manifest)
    (assumptions : List AssumptionRow)
    (row : SummaryRow)
    (proof : loopPostFollowsFromAssumptions manifest assumptions row) :
    ∃ env state,
      assumptionsDenoteInEnv manifest env assumptions ∧
      contextReady env row.contextVariables ∧
      loopExitPremises manifest env row state :=
  proof.1

theorem unsupported_kind_fails_closed
    (manifest : Manifest) (env : Env) (row : SummaryRow)
    (h : row.kind = .other) :
    denoteLoopSummary? manifest env row = none := by
  have hKind : (row.kind == .scfWhile || row.kind == .scfFor) = false := by
    rw [h]
    decide
  simp [denoteLoopSummary?, SummaryRow.supported, SummaryRow.shapeSupported, hKind]

end Ora.Loop

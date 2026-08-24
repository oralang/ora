/-
Fail-closed denotation of compiler-emitted loop verification conditions.

The compiler emits only data. Unsupported identities, types, references, edge
targets, or incomplete state bindings make `denoteLoopSummary?` return `none`.
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

inductive ControlTargetRow where
  | backedge
  | successor
  | functionReturn
  | errorReturn
  | explicitRevert
  | unresolved
  deriving Repr, BEq, DecidableEq

structure VariableRow where
  index : Nat
  id : FreeVarId
  name : String := ""
  ty : TyRef
  deriving Repr, BEq, DecidableEq

structure StateBindingRow where
  variableIndex : Nat
  target : FreeVarId
  sourceKey : String := ""
  value : FormulaRef
  deriving Repr, BEq, DecidableEq

structure EdgeRow where
  id : Nat
  structuralKey : String := ""
  path : List FormulaRef := []
  safety : List FormulaRef := []
  nextState : List StateBindingRow := []
  target : ControlTargetRow := .unresolved
  deriving Repr, BEq, DecidableEq

structure SummaryRow where
  id : Nat
  owner : String
  kind : LoopKind
  contextVariables : List VariableRow := []
  variables : List VariableRow := []
  init : List FormulaRef := []
  invariants : List FormulaRef := []
  edges : List EdgeRow := []
  projectionComplete : Bool := false
  post : List FormulaRef := []
  unsupportedReasons : List String := []
  deriving Repr, BEq, DecidableEq

def idsUnique (variables : List VariableRow) : Bool :=
  decide (variables.map VariableRow.id).Nodup

def variablesIndexedFrom : Nat → List VariableRow → Bool
  | _, [] => true
  | expected, loopVar :: rest =>
      loopVar.index == expected && variablesIndexedFrom (expected + 1) rest

def variablesAreIntegers (variables : List VariableRow) : Bool :=
  variables.all (fun loopVar => loopVar.ty.integerShape?.isSome)

def variableSetsDisjoint (lhs rhs : List VariableRow) : Bool :=
  lhs.all (fun left => rhs.all (fun right => left.id != right.id))

def bindingsMatchVariables : List VariableRow → List StateBindingRow → Bool
  | [], [] => true
  | loopVar :: variables, binding :: bindings =>
      binding.variableIndex == loopVar.index &&
        binding.target == loopVar.id &&
        !binding.sourceKey.isEmpty &&
        bindingsMatchVariables variables bindings
  | _, _ => false

def edgesIndexedFrom : Nat → List EdgeRow → Bool
  | _, [] => true
  | expected, edge :: rest => edge.id == expected && edgesIndexedFrom (expected + 1) rest

def edgeStructuralKeysValid (edges : List EdgeRow) : Bool :=
  edges.all (fun edge => !edge.structuralKey.isEmpty) &&
    decide (edges.map EdgeRow.structuralKey).Nodup

def edgeTargetSupported : ControlTargetRow → Bool
  | .backedge | .successor => true
  | .functionReturn | .errorReturn | .explicitRevert | .unresolved => false

def edgeShapeSupported (variables : List VariableRow) (edge : EdgeRow) : Bool :=
  !edge.path.isEmpty && edgeTargetSupported edge.target &&
    bindingsMatchVariables variables edge.nextState

def hasBackedge (edges : List EdgeRow) : Bool :=
  edges.any (fun edge => edge.target == .backedge)

def hasSuccessor (edges : List EdgeRow) : Bool :=
  edges.any (fun edge => edge.target == .successor)

def valueForTy : TyRef → Value
  | ty =>
      match ty.integerShape? with
      | some shape => .integer (Ora.Integer.Value.ofNat shape 0)
      | none => .bool true

def bindSupportVariable (env : Env) (rowVariable : VariableRow) : Env :=
  env.setFree rowVariable.id (valueForTy rowVariable.ty)

def SummaryRow.supportEnv (row : SummaryRow) : Env :=
  (row.contextVariables ++ row.variables).foldl bindSupportVariable Env.empty

def formulaDenotable (manifest : Manifest) (env : Env) (formula : FormulaRef) : Bool :=
  (formulaDenotes? manifest env formula).isSome

def valueDenotable (manifest : Manifest) (env : Env) (formula : FormulaRef) : Bool :=
  (formulaValue? manifest env formula).isSome

def edgeFormulasDenotable (manifest : Manifest) (env : Env) (edge : EdgeRow) : Bool :=
  edge.path.all (formulaDenotable manifest env) &&
    edge.safety.all (formulaDenotable manifest env) &&
    edge.nextState.all (fun binding => valueDenotable manifest env binding.value)

def SummaryRow.shapeSupported (row : SummaryRow) : Bool :=
  row.projectionComplete &&
    row.unsupportedReasons.isEmpty &&
    (row.kind == .scfWhile || row.kind == .scfFor) &&
    !row.variables.isEmpty &&
    row.init.length == row.variables.length &&
    !row.invariants.isEmpty &&
    !row.edges.isEmpty &&
    variablesIndexedFrom 0 row.variables &&
    idsUnique row.variables &&
    idsUnique row.contextVariables &&
    variablesAreIntegers row.variables &&
    variablesAreIntegers row.contextVariables &&
    variableSetsDisjoint row.variables row.contextVariables &&
    edgesIndexedFrom 0 row.edges &&
    edgeStructuralKeysValid row.edges &&
    row.edges.all (edgeShapeSupported row.variables) &&
    hasBackedge row.edges && hasSuccessor row.edges

def SummaryRow.formulasDenotable (manifest : Manifest) (row : SummaryRow) : Bool :=
  let env := row.supportEnv
  row.init.all (valueDenotable manifest env) &&
    row.invariants.all (formulaDenotable manifest env) &&
    row.edges.all (edgeFormulasDenotable manifest env) &&
    row.post.all (formulaDenotable manifest env)

def SummaryRow.supported (manifest : Manifest) (row : SummaryRow) : Bool :=
  manifest.wf && manifest.termsTopological && row.shapeSupported && row.formulasDenotable manifest

def contextReady (env : Env) (variables : List VariableRow) : Prop :=
  ∀ loopVar ∈ variables, ∃ value,
    loopVar.ty.integerShape? = some value.shape ∧
      env.lookupFree loopVar.id = some (.integer value)

/- Loop state contains only loop-carried integer values. The obligation
environment is captured by the denoted VC and remains unchanged; binding the
state shadows only the declared loop variables. -/
abbrev ScalarLoopState := List Ora.Integer.Value

def valuesMatchVariables : List VariableRow → ScalarLoopState → Bool
  | [], [] => true
  | loopVar :: variables, value :: values =>
      loopVar.ty.integerShape? == some value.shape &&
        valuesMatchVariables variables values
  | _, _ => false

def stateReady (variables : List VariableRow) (state : ScalarLoopState) : Prop :=
  valuesMatchVariables variables state = true

def bindState? : Env → List VariableRow → ScalarLoopState → Option Env
  | env, [], [] => some env
  | env, loopVar :: variables, value :: values =>
      if loopVar.ty.integerShape? = some value.shape then
        bindState? (env.setFree loopVar.id (.integer value)) variables values
      else
        none
  | _, _, _ => none

def formulaHoldsInState
    (manifest : Manifest)
    (base : Env)
    (variables : List VariableRow)
    (state : ScalarLoopState)
    (formula : FormulaRef) : Prop :=
  match bindState? base variables state with
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

def integerFormulaValue?
    (manifest : Manifest)
    (env : Env)
    (formula : FormulaRef) : Option Ora.Integer.Value :=
  match formulaValue? manifest env formula with
  | some (.integer value) => some value
  | _ => none

def evaluateIntegerFormulas?
    (manifest : Manifest)
    (env : Env) : List FormulaRef → Option (List Ora.Integer.Value)
  | [] => some []
  | formula :: formulas => do
      let value ← integerFormulaValue? manifest env formula
      let values ← evaluateIntegerFormulas? manifest env formulas
      pure (value :: values)

def initializeState?
    (manifest : Manifest)
    (base : Env)
    (variables : List VariableRow)
    (formulas : List FormulaRef) : Option ScalarLoopState := do
  let values ← evaluateIntegerFormulas? manifest base formulas
  if valuesMatchVariables variables values then
    pure values
  else
    none

def bindingFormulas (bindings : List StateBindingRow) : List FormulaRef :=
  bindings.map StateBindingRow.value

def applyStateBindings?
    (manifest : Manifest)
    (base : Env)
    (variables : List VariableRow)
    (current : ScalarLoopState)
    (bindings : List StateBindingRow) : Option ScalarLoopState :=
  if bindingsMatchVariables variables bindings then do
    let env ← bindState? base variables current
    let values ← evaluateIntegerFormulas? manifest env (bindingFormulas bindings)
    if valuesMatchVariables variables values then
      pure values
    else
      none
  else
    none

def valuesInitializeState
    (manifest : Manifest)
    (base : Env)
    (variables : List VariableRow)
    (formulas : List FormulaRef)
    (state : ScalarLoopState) : Prop :=
  initializeState? manifest base variables formulas = some state

theorem initialized_state_ready
    (manifest : Manifest)
    (base : Env)
    (variables : List VariableRow)
    (formulas : List FormulaRef)
    (state : ScalarLoopState)
    (hInitial : valuesInitializeState manifest base variables formulas state) :
    stateReady variables state := by
  unfold valuesInitializeState initializeState? at hInitial
  cases hValues : evaluateIntegerFormulas? manifest base formulas with
  | none => simp [hValues] at hInitial
  | some values =>
      cases hReady : valuesMatchVariables variables values with
      | false => simp [hValues, hReady] at hInitial
      | true =>
        simp [hValues, hReady] at hInitial
        subst state
        exact hReady

def stateBindingsProduceState
    (manifest : Manifest)
    (base : Env)
    (variables : List VariableRow)
    (current : ScalarLoopState)
    (bindings : List StateBindingRow)
    (following : ScalarLoopState) : Prop :=
  applyStateBindings? manifest base variables current bindings = some following

theorem bound_state_ready
    (manifest : Manifest)
    (base : Env)
    (variables : List VariableRow)
    (current following : ScalarLoopState)
    (bindings : List StateBindingRow)
    (hNext : stateBindingsProduceState manifest base variables current bindings following) :
    stateReady variables following := by
  unfold stateBindingsProduceState applyStateBindings? at hNext
  by_cases hBindings : bindingsMatchVariables variables bindings = true
  · cases hEnv : bindState? base variables current with
    | none => simp [hBindings, hEnv] at hNext
    | some env =>
        cases hValues : evaluateIntegerFormulas? manifest env (bindingFormulas bindings) with
        | none => simp [hBindings, hEnv, hValues] at hNext
        | some values =>
            by_cases hReady : valuesMatchVariables variables values = true
            · simp [hBindings, hEnv, hValues, hReady] at hNext
              subst following
              exact hReady
            · cases hReadyValue : valuesMatchVariables variables values <;>
                simp_all
  · cases hBindingsValue : bindingsMatchVariables variables bindings <;>
      simp_all

abbrev assignmentsProduceState := stateBindingsProduceState

def denoteControlTarget? : ControlTargetRow → Option ControlTarget
  | .backedge => some .backedge
  | .successor => some .successor
  | .functionReturn => some .functionReturn
  | .errorReturn => some .errorReturn
  | .explicitRevert => some .explicitRevert
  | .unresolved => none

def denoteEdge?
    (manifest : Manifest)
    (base : Env)
    (variables : List VariableRow)
    (row : EdgeRow) : Option (Edge ScalarLoopState) := do
  let target ← denoteControlTarget? row.target
  pure {
    path := fun state => formulasHoldInState manifest base variables state row.path
    safe := fun state => formulasHoldInState manifest base variables state row.safety
    next := fun current following =>
      stateBindingsProduceState manifest base variables current row.nextState following
    target := target
  }

def denoteEdges?
    (manifest : Manifest)
    (base : Env)
    (variables : List VariableRow) : List EdgeRow → Option (List (Edge ScalarLoopState))
  | [] => some []
  | edge :: rest => do
      let denoted ← denoteEdge? manifest base variables edge
      let denotedRest ← denoteEdges? manifest base variables rest
      pure (denoted :: denotedRest)

def denoteLoopSummary? (manifest : Manifest) (base : Env) (row : SummaryRow) :
    Option (LoopVC ScalarLoopState) :=
  if row.supported manifest then do
    let edges ← denoteEdges? manifest base row.variables row.edges
    pure {
      initial := fun state => valuesInitializeState manifest base row.variables row.init state
      invariant := fun state =>
        stateReady row.variables state ∧
          formulasHoldInState manifest base row.variables state row.invariants
      edges := edges
      targetAssertion := fun target state =>
        match target with
        | .successor => formulasHoldInState manifest base row.variables state row.post
        | .backedge | .functionReturn | .errorReturn | .explicitRevert => False
    }
  else
    none

def loopInitialPremises (manifest : Manifest) (base : Env) (row : SummaryRow)
    (state : ScalarLoopState) : Prop :=
  match denoteLoopSummary? manifest base row with
  | some summary => summary.initial state
  | none => False

def loopInductionObligationsAtEnv
    (manifest : Manifest)
    (base : Env)
    (row : SummaryRow) : Prop :=
  match denoteLoopSummary? manifest base row with
  | some summary => summary.InductionObligations
  | none => False

def loopVerifiedAtEnv
    (manifest : Manifest)
    (base : Env)
    (row : SummaryRow) : Prop :=
  match denoteLoopSummary? manifest base row with
  | some summary => summary.Verified
  | none => False

def loopSpecificationNonvacuous
    (manifest : Manifest)
    (assumptions : List AssumptionRow)
    (row : SummaryRow) : Prop :=
  ∃ env state,
    assumptionsDenoteInEnv manifest env assumptions ∧
    contextReady env row.contextVariables ∧
    loopInitialPremises manifest env row state

def loopInductionCertificateFromAssumptions
    (manifest : Manifest)
    (assumptions : List AssumptionRow)
    (row : SummaryRow) : Prop :=
  ∀ env,
    assumptionsDenoteInEnv manifest env assumptions →
    contextReady env row.contextVariables →
    loopInductionObligationsAtEnv manifest env row

def loopProofAuthorizationFromAssumptions
    (manifest : Manifest)
    (assumptions : List AssumptionRow)
    (row : SummaryRow) : Prop :=
  loopSpecificationNonvacuous manifest assumptions row ∧
    loopInductionCertificateFromAssumptions manifest assumptions row

def loopVerifiedFromAssumptions
    (manifest : Manifest)
    (assumptions : List AssumptionRow)
    (row : SummaryRow) : Prop :=
  ∀ env,
    assumptionsDenoteInEnv manifest env assumptions →
    contextReady env row.contextVariables →
    loopVerifiedAtEnv manifest env row

theorem loop_induction_certificate_sound
    (manifest : Manifest)
    (assumptions : List AssumptionRow)
    (row : SummaryRow)
    (proof : loopInductionCertificateFromAssumptions manifest assumptions row) :
    loopVerifiedFromAssumptions manifest assumptions row := by
  intro env hAssumptions hContext
  have hObligations := proof env hAssumptions hContext
  unfold loopInductionObligationsAtEnv at hObligations
  unfold loopVerifiedAtEnv
  cases hSummary : denoteLoopSummary? manifest env row with
  | none => simp [hSummary] at hObligations
  | some summary =>
      simp only [hSummary] at hObligations ⊢
      exact verified_of_induction summary hObligations

theorem loop_authorization_sound
    (manifest : Manifest)
    (assumptions : List AssumptionRow)
    (row : SummaryRow)
    (proof : loopProofAuthorizationFromAssumptions manifest assumptions row) :
    loopVerifiedFromAssumptions manifest assumptions row :=
  loop_induction_certificate_sound manifest assumptions row proof.2

theorem loop_authorization_has_nonvacuous_initial_state
    (manifest : Manifest)
    (assumptions : List AssumptionRow)
    (row : SummaryRow)
    (proof : loopProofAuthorizationFromAssumptions manifest assumptions row) :
    ∃ env state,
      assumptionsDenoteInEnv manifest env assumptions ∧
      contextReady env row.contextVariables ∧
      loopInitialPremises manifest env row state :=
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

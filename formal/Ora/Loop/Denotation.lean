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

structure SummaryDependencyRow where
  id : Nat
  structuralKey : String
  summaryId : Nat
  loopVcSha256 : String
  target : ControlTargetRow
  deriving Repr, BEq, DecidableEq

structure SafetyCheckpointRow where
  id : Nat
  structuralKey : String
  afterDependencyCount : Nat
  formulas : List FormulaRef
  deriving Repr, BEq, DecidableEq

structure EdgeRow where
  id : Nat
  structuralKey : String := ""
  path : List FormulaRef := []
  safetyCheckpoints : List SafetyCheckpointRow := []
  summaryDependencies : List SummaryDependencyRow := []
  nextState : List StateBindingRow := []
  target : ControlTargetRow := .unresolved
  targetFormulas : List FormulaRef := []
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
  | .backedge | .successor | .functionReturn | .errorReturn => true
  | .explicitRevert | .unresolved => false

def edgeTargetFormulasValid (edge : EdgeRow) : Bool :=
  match edge.target with
  | .backedge | .successor => edge.targetFormulas.isEmpty
  | .functionReturn | .errorReturn => true
  | .explicitRevert | .unresolved => false

def safetyCheckpointsIndexedAndStagedFrom : Nat → Nat → Nat →
    List SafetyCheckpointRow → Bool
  | _, _, _, [] => true
  | expected, previousStage, dependencyCount, checkpoint :: rest =>
      checkpoint.id == expected &&
        !checkpoint.structuralKey.isEmpty &&
        !checkpoint.formulas.isEmpty &&
        previousStage <= checkpoint.afterDependencyCount &&
        checkpoint.afterDependencyCount <= dependencyCount &&
        safetyCheckpointsIndexedAndStagedFrom (expected + 1)
          checkpoint.afterDependencyCount dependencyCount rest

def safetyCheckpointStructuralKeysValid (checkpoints : List SafetyCheckpointRow) : Bool :=
  decide (checkpoints.map SafetyCheckpointRow.structuralKey).Nodup

def safetyCheckpointsShapeSupported (edge : EdgeRow) : Bool :=
  safetyCheckpointsIndexedAndStagedFrom 0 0 edge.summaryDependencies.length
      edge.safetyCheckpoints &&
    safetyCheckpointStructuralKeysValid edge.safetyCheckpoints

def edgeBaseShapeSupported (variables : List VariableRow) (edge : EdgeRow) : Bool :=
  !edge.path.isEmpty && edgeTargetSupported edge.target &&
    edgeTargetFormulasValid edge &&
    safetyCheckpointsShapeSupported edge &&
    bindingsMatchVariables variables edge.nextState

def edgeShapeSupported (variables : List VariableRow) (edge : EdgeRow) : Bool :=
  edgeBaseShapeSupported variables edge && edge.summaryDependencies.isEmpty

def lowerHexDigit (value : Char) : Bool :=
  "0123456789abcdef".contains value

def dependencyDigestValid (dependency : SummaryDependencyRow) : Bool :=
  dependency.loopVcSha256.length == 64 &&
    dependency.loopVcSha256.toList.all lowerHexDigit

def dependenciesIndexedFrom : Nat → List SummaryDependencyRow → Bool
  | _, [] => true
  | expected, dependency :: rest =>
      dependency.id == expected && dependenciesIndexedFrom (expected + 1) rest

def dependencyStructuralKeysValid (dependencies : List SummaryDependencyRow) : Bool :=
  dependencies.all (fun dependency => !dependency.structuralKey.isEmpty) &&
    decide (dependencies.map SummaryDependencyRow.structuralKey).Nodup

def findSummaryRow? : List SummaryRow → Nat → Option SummaryRow
  | [], _ => none
  | candidate :: rest, id =>
      if candidate.id == id then some candidate else findSummaryRow? rest id

def summaryIdsUnique (rows : List SummaryRow) : Bool :=
  decide (rows.map SummaryRow.id).Nodup

def summaryHasTarget (row : SummaryRow) (target : ControlTargetRow) : Bool :=
  row.edges.any (fun edge => edge.target == target)

def dependencyShapeSupported
    (rows : List SummaryRow)
    (parent : SummaryRow)
    (dependency : SummaryDependencyRow) : Bool :=
  dependency.summaryId > parent.id &&
    edgeTargetSupported dependency.target &&
    dependency.target != .backedge &&
    match findSummaryRow? rows dependency.summaryId with
    | some inner => summaryHasTarget inner dependency.target
    | none => false

/- The exact dependency digest and its lowercase hexadecimal serialization are
checked by the compiler-kernel obligation and artifact gates before this
denotation can authorize output.  The digest remains part of the emitted row
and certificate identity, but it has no role in the child transition relation;
rechecking `String.contains` here would make semantic denotation depend on an
opaque string implementation rather than on proof-relevant loop data. -/

def edgeDependenciesSupported
    (rows : List SummaryRow)
    (parent : SummaryRow)
    (edge : EdgeRow) : Bool :=
  dependenciesIndexedFrom 0 edge.summaryDependencies &&
    dependencyStructuralKeysValid edge.summaryDependencies &&
    edge.summaryDependencies.all (dependencyShapeSupported rows parent)

def hasBackedge (edges : List EdgeRow) : Bool :=
  edges.any (fun edge => edge.target == .backedge)

def hasExit (edges : List EdgeRow) : Bool :=
  edges.any (fun edge => edge.target != .backedge && edgeTargetSupported edge.target)

def valueForTy : TyRef → Value
  | ty =>
      match ty.integerShape? with
      | some shape => .integer (Ora.Integer.Value.ofNat shape 0)
      | none => .bool true

def bindSupportVariable (env : Env) (rowVariable : VariableRow) : Env :=
  env.setFree rowVariable.id (valueForTy rowVariable.ty)

def SummaryRow.supportEnv (row : SummaryRow) : Env :=
  (row.contextVariables ++ row.variables).foldl bindSupportVariable Env.empty

def dependencyVariables (rows : List SummaryRow) (row : SummaryRow) : List VariableRow :=
  row.edges.flatMap fun edge =>
    edge.summaryDependencies.flatMap fun dependency =>
      match findSummaryRow? rows dependency.summaryId with
      | some inner => inner.variables
      | none => []

def dependencyVariablesFor
    (rows : List SummaryRow)
    (dependencies : List SummaryDependencyRow) : List VariableRow :=
  dependencies.flatMap fun dependency =>
    match findSummaryRow? rows dependency.summaryId with
    | some inner => inner.variables
    | none => []

/-- The support environment exposes only the parent variables and variables
declared by its exact summary dependencies.  Unrelated loop rows cannot make
an otherwise dangling formula reference denotable. -/
def SummaryRow.supportEnvIn (rows : List SummaryRow) (row : SummaryRow) : Env :=
  (row.contextVariables ++ row.variables ++ dependencyVariables rows row).foldl
    bindSupportVariable Env.empty

def formulaDenotable (manifest : Manifest) (env : Env) (formula : FormulaRef) : Bool :=
  (formulaDenotes? manifest env formula).isSome

def valueDenotable (manifest : Manifest) (env : Env) (formula : FormulaRef) : Bool :=
  (formulaValue? manifest env formula).isSome

def safetyCheckpointDenotable
    (manifest : Manifest)
    (rows : List SummaryRow)
    (row : SummaryRow)
    (edge : EdgeRow)
    (checkpoint : SafetyCheckpointRow) : Bool :=
  let variables := dependencyVariablesFor rows
    (edge.summaryDependencies.take checkpoint.afterDependencyCount)
  let env := (row.contextVariables ++ row.variables ++ variables).foldl
    bindSupportVariable Env.empty
  checkpoint.formulas.all (formulaDenotable manifest env)

def edgeFormulasDenotable (manifest : Manifest) (env : Env) (edge : EdgeRow) : Bool :=
  edge.path.all (formulaDenotable manifest env) &&
    edge.safetyCheckpoints.all (fun checkpoint =>
      checkpoint.formulas.all (formulaDenotable manifest env)) &&
    edge.targetFormulas.all (formulaDenotable manifest env) &&
    edge.nextState.all (fun binding => valueDenotable manifest env binding.value)

def edgeControlFormulasDenotable
    (manifest : Manifest)
    (rows : List SummaryRow)
    (row : SummaryRow)
    (env : Env)
    (edge : EdgeRow) : Bool :=
  edge.path.all (formulaDenotable manifest env) &&
    edge.safetyCheckpoints.all (safetyCheckpointDenotable manifest rows row edge) &&
    edge.targetFormulas.all (formulaDenotable manifest env)

def edgeBindingsDenotable
    (manifest : Manifest)
    (env : Env)
    (edge : EdgeRow) : Bool :=
  edge.nextState.all (fun binding => valueDenotable manifest env binding.value)

def SummaryRow.baseShapeSupported (row : SummaryRow) : Bool :=
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
    hasBackedge row.edges && hasExit row.edges

def SummaryRow.shapeSupported (row : SummaryRow) : Bool :=
  row.baseShapeSupported && row.edges.all (edgeShapeSupported row.variables)

def SummaryRow.shapeSupportedIn (rows : List SummaryRow) (row : SummaryRow) : Bool :=
  summaryIdsUnique rows &&
    row.baseShapeSupported &&
    row.edges.all (fun edge =>
      edgeBaseShapeSupported row.variables edge &&
        edgeDependenciesSupported rows row edge)

def SummaryRow.formulasDenotable (manifest : Manifest) (row : SummaryRow) : Bool :=
  let env := row.supportEnv
  row.init.all (valueDenotable manifest env) &&
    row.invariants.all (formulaDenotable manifest env) &&
    row.edges.all (edgeFormulasDenotable manifest env) &&
    row.post.all (formulaDenotable manifest env)

def SummaryRow.formulasDenotableIn
    (manifest : Manifest)
    (rows : List SummaryRow)
    (row : SummaryRow) : Bool :=
  let base := row.supportEnv
  let dependent := row.supportEnvIn rows
  row.init.all (valueDenotable manifest base) &&
    row.invariants.all (formulaDenotable manifest base) &&
    row.edges.all (edgeControlFormulasDenotable manifest rows row base) &&
    row.edges.all (edgeBindingsDenotable manifest dependent) &&
    row.post.all (formulaDenotable manifest base)

def SummaryRow.supported (manifest : Manifest) (row : SummaryRow) : Bool :=
  manifest.wf && manifest.termsTopological && row.shapeSupported && row.formulasDenotable manifest

def SummaryRow.supportedIn
    (manifest : Manifest)
    (rows : List SummaryRow)
    (row : SummaryRow) : Bool :=
  manifest.wf && manifest.termsTopological &&
    row.shapeSupportedIn rows && row.formulasDenotableIn manifest rows

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

def formulaHoldsInEnv
    (manifest : Manifest)
    (env : Env)
    (formula : FormulaRef) : Prop :=
  match formulaDenotes? manifest env formula with
  | some proposition => proposition
  | none => False

def formulasHoldInEnv
    (manifest : Manifest)
    (env : Env)
    (formulas : List FormulaRef) : Prop :=
  ∀ formula ∈ formulas, formulaHoldsInEnv manifest env formula

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

def applyStateBindingsInEnv?
    (manifest : Manifest)
    (variables : List VariableRow)
    (env : Env)
    (bindings : List StateBindingRow) : Option ScalarLoopState :=
  if bindingsMatchVariables variables bindings then do
    let values ← evaluateIntegerFormulas? manifest env (bindingFormulas bindings)
    if valuesMatchVariables variables values then
      pure values
    else
      none
  else
    none

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

theorem bound_state_in_env_ready
    (manifest : Manifest)
    (variables : List VariableRow)
    (env : Env)
    (bindings : List StateBindingRow)
    (following : ScalarLoopState)
    (hNext : applyStateBindingsInEnv? manifest variables env bindings = some following) :
    stateReady variables following := by
  unfold applyStateBindingsInEnv? at hNext
  by_cases hBindings : bindingsMatchVariables variables bindings = true
  · cases hValues : evaluateIntegerFormulas? manifest env (bindingFormulas bindings) with
    | none => simp [hBindings, hValues] at hNext
    | some values =>
        by_cases hReady : valuesMatchVariables variables values = true
        · simp [hBindings, hValues, hReady] at hNext
          subst following
          exact hReady
        · cases hReadyValue : valuesMatchVariables variables values <;>
            simp_all
  · cases hBindingsValue : bindingsMatchVariables variables bindings <;>
      simp_all

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

abbrev SummaryResolver := Env → SummaryRow → Option (LoopVC ScalarLoopState)

def summaryDependencyHolds
    (manifest : Manifest)
    (resolver : SummaryResolver)
    (inner : SummaryRow)
    (target : ControlTarget)
    (current following : Env) : Prop :=
  match resolver current inner with
  | none => False
  | some summary =>
      ∃ initial final bound,
        valuesInitializeState manifest current inner.variables inner.init initial ∧
        summary.CertifiedControlRelation target initial final ∧
        bindState? current inner.variables final = some bound ∧
        following = bound

structure EnvironmentRelation where
  holds : Env → Env → Prop

/-- Denote one certified inner-loop summary as an environment relation.  The
resolver is checked while constructing the parent VC, so an unavailable child
summary makes parent denotation fail instead of turning the edge relation into
`False` and making preservation vacuous. -/
def denoteSummaryDependency?
    (manifest : Manifest)
    (rows : List SummaryRow)
    (resolver : SummaryResolver)
    (supportBase : Env)
    (dependency : SummaryDependencyRow) : Option EnvironmentRelation := do
  let inner ← findSummaryRow? rows dependency.summaryId
  let target ← denoteControlTarget? dependency.target
  if (resolver supportBase inner).isSome then
    pure {
      holds := summaryDependencyHolds manifest resolver inner target
    }
  else
    none

def denoteSummaryDependencies?
    (manifest : Manifest)
    (rows : List SummaryRow)
    (resolver : SummaryResolver)
    (supportBase : Env) :
    List SummaryDependencyRow → Option (List EnvironmentRelation)
  | [] => some []
  | dependency :: rest => do
      let relation ← denoteSummaryDependency? manifest rows resolver supportBase dependency
      let relations ← denoteSummaryDependencies? manifest rows resolver supportBase rest
      pure (relation :: relations)

def environmentRelationsProduce :
    List EnvironmentRelation → Env → Env → Prop
  | [], current, following => current = following
  | relation :: rest, current, following =>
      ∃ intermediate,
        relation.holds current intermediate ∧
        environmentRelationsProduce rest intermediate following

def safetyCheckpointHolds
    (manifest : Manifest)
    (base : Env)
    (variables : List VariableRow)
    (current : ScalarLoopState)
    (dependencies : List EnvironmentRelation)
    (checkpoint : SafetyCheckpointRow) : Prop :=
  ∀ beforeDependencies checkpointEnv,
    bindState? base variables current = some beforeDependencies →
    environmentRelationsProduce
        (dependencies.take checkpoint.afterDependencyCount)
        beforeDependencies checkpointEnv →
    formulasHoldInEnv manifest checkpointEnv checkpoint.formulas

def safetyCheckpointsHold
    (manifest : Manifest)
    (base : Env)
    (variables : List VariableRow)
    (current : ScalarLoopState)
    (dependencies : List EnvironmentRelation)
    (checkpoints : List SafetyCheckpointRow) : Prop :=
  ∀ checkpoint ∈ checkpoints,
    safetyCheckpointHolds manifest base variables current dependencies checkpoint

/-- Partial correctness does not demand safety for code following a dependency
that has no terminating result. If the dependency does terminate, the
checkpoint remains universally quantified over every result. -/
theorem safety_checkpoint_after_nonterminating_dependency
    (manifest : Manifest)
    (base : Env)
    (variables : List VariableRow)
    (current : ScalarLoopState)
    (relation : EnvironmentRelation)
    (checkpoint : SafetyCheckpointRow)
    (before : Env)
    (hBound : bindState? base variables current = some before)
    (hStage : checkpoint.afterDependencyCount = 1)
    (hNever : ∀ following, ¬ relation.holds before following) :
    safetyCheckpointHolds manifest base variables current [relation] checkpoint := by
  intro actualBefore checkpointEnv hActualBound hProduce
  have hBefore : actualBefore = before := by
    exact Option.some.inj (hActualBound.symm.trans hBound)
  subst actualBefore
  rw [hStage] at hProduce
  simp only [List.take, environmentRelationsProduce] at hProduce
  rcases hProduce with ⟨following, hRelation, _⟩
  exact False.elim (hNever following hRelation)

def stateBindingsProduceAfterDependencies
    (manifest : Manifest)
    (base : Env)
    (variables : List VariableRow)
    (current : ScalarLoopState)
    (dependencies : List EnvironmentRelation)
    (bindings : List StateBindingRow)
    (following : ScalarLoopState) : Prop :=
  ∃ beforeDependencies afterDependencies,
    bindState? base variables current = some beforeDependencies ∧
    environmentRelationsProduce dependencies beforeDependencies afterDependencies ∧
    applyStateBindingsInEnv? manifest variables afterDependencies bindings = some following

theorem bound_state_after_dependencies_ready
    (manifest : Manifest)
    (base : Env)
    (variables : List VariableRow)
    (current : ScalarLoopState)
    (dependencies : List EnvironmentRelation)
    (bindings : List StateBindingRow)
    (following : ScalarLoopState)
    (hNext : stateBindingsProduceAfterDependencies manifest base variables current
      dependencies bindings following) :
    stateReady variables following := by
  rcases hNext with ⟨_, afterDependencies, _, _, hBindings⟩
  exact bound_state_in_env_ready manifest variables afterDependencies bindings following hBindings

theorem stateBindingsProduceAfterNoDependencies_iff
    (manifest : Manifest)
    (base : Env)
    (variables : List VariableRow)
    (current : ScalarLoopState)
    (bindings : List StateBindingRow)
    (following : ScalarLoopState) :
    stateBindingsProduceAfterDependencies manifest base variables current [] bindings following ↔
      stateBindingsProduceState manifest base variables current bindings following := by
  unfold stateBindingsProduceAfterDependencies stateBindingsProduceState
  cases hBind : bindState? base variables current <;>
    simp [environmentRelationsProduce, applyStateBindings?, applyStateBindingsInEnv?, hBind]

def denoteEdge?
    (manifest : Manifest)
    (base : Env)
    (variables : List VariableRow)
    (successorPost : List FormulaRef)
    (row : EdgeRow) : Option (Edge ScalarLoopState) := do
  let target ← denoteControlTarget? row.target
  let targetFormulas := match row.target with
    | .successor => successorPost
    | .functionReturn | .errorReturn => row.targetFormulas
    | .backedge | .explicitRevert | .unresolved => []
  pure {
    path := fun state => formulasHoldInState manifest base variables state row.path
    safe := fun state => safetyCheckpointsHold manifest base variables state []
      row.safetyCheckpoints
    next := fun current following =>
      stateBindingsProduceState manifest base variables current row.nextState following
    target := target
    targetAssertion := fun state =>
      formulasHoldInState manifest base variables state targetFormulas
  }

def denoteEdgeWithResolver?
    (manifest : Manifest)
    (rows : List SummaryRow)
    (resolver : SummaryResolver)
    (base : Env)
    (variables : List VariableRow)
    (successorPost : List FormulaRef)
    (row : EdgeRow) : Option (Edge ScalarLoopState) := do
  let target ← denoteControlTarget? row.target
  let dependencies ← denoteSummaryDependencies? manifest rows resolver base
    row.summaryDependencies
  let targetFormulas := match row.target with
    | .successor => successorPost
    | .functionReturn | .errorReturn => row.targetFormulas
    | .backedge | .explicitRevert | .unresolved => []
  pure {
    path := fun state => formulasHoldInState manifest base variables state row.path
    safe := fun state => safetyCheckpointsHold manifest base variables state dependencies
      row.safetyCheckpoints
    next := fun current following =>
      stateBindingsProduceAfterDependencies manifest base variables current
        dependencies row.nextState following
    target := target
    targetAssertion := fun state =>
      formulasHoldInState manifest base variables state targetFormulas
  }

def denoteEdges?
    (manifest : Manifest)
    (base : Env)
    (variables : List VariableRow)
    (successorPost : List FormulaRef) : List EdgeRow → Option (List (Edge ScalarLoopState))
  | [] => some []
  | edge :: rest => do
      let denoted ← denoteEdge? manifest base variables successorPost edge
      let denotedRest ← denoteEdges? manifest base variables successorPost rest
      pure (denoted :: denotedRest)

def denoteEdgesWithResolver?
    (manifest : Manifest)
    (rows : List SummaryRow)
    (resolver : SummaryResolver)
    (base : Env)
    (variables : List VariableRow)
    (successorPost : List FormulaRef) :
    List EdgeRow → Option (List (Edge ScalarLoopState))
  | [] => some []
  | edge :: rest => do
      let denoted ← denoteEdgeWithResolver? manifest rows resolver base variables
        successorPost edge
      let denotedRest ← denoteEdgesWithResolver? manifest rows resolver base variables
        successorPost rest
      pure (denoted :: denotedRest)

def denoteLoopSummary? (manifest : Manifest) (base : Env) (row : SummaryRow) :
    Option (LoopVC ScalarLoopState) :=
  if row.supported manifest then do
    let edges ← denoteEdges? manifest base row.variables row.post row.edges
    pure {
      initial := fun state => valuesInitializeState manifest base row.variables row.init state
      invariant := fun state =>
        stateReady row.variables state ∧
          formulasHoldInState manifest base row.variables state row.invariants
      edges := edges
    }
  else
    none

def denoteLoopSummaryWithResolver?
    (manifest : Manifest)
    (rows : List SummaryRow)
    (resolver : SummaryResolver)
    (base : Env)
    (row : SummaryRow) : Option (LoopVC ScalarLoopState) :=
  if row.supportedIn manifest rows then do
    let edges ← denoteEdgesWithResolver? manifest rows resolver base row.variables row.post
      row.edges
    pure {
      initial := fun state => valuesInitializeState manifest base row.variables row.init state
      invariant := fun state =>
        stateReady row.variables state ∧
          formulasHoldInState manifest base row.variables state row.invariants
      edges := edges
    }
  else
    none

/-- Fuel is consumed only when following a declared inner-summary dependency.
Unique, strictly-forward summary ids bound the required depth by `rows.length`;
the public entry point supplies one additional unit for the root summary. -/
def denoteLoopSummaryWithFuel?
    (fuel : Nat)
    (manifest : Manifest)
    (rows : List SummaryRow)
    (base : Env)
    (row : SummaryRow) : Option (LoopVC ScalarLoopState) :=
  match fuel with
  | 0 => none
  | remaining + 1 =>
      denoteLoopSummaryWithResolver? manifest rows
        (fun innerBase inner =>
          denoteLoopSummaryWithFuel? remaining manifest rows innerBase inner)
        base row
termination_by fuel

def denoteLoopSummaryIn?
    (manifest : Manifest)
    (rows : List SummaryRow)
    (base : Env)
    (row : SummaryRow) : Option (LoopVC ScalarLoopState) :=
  denoteLoopSummaryWithFuel? (rows.length + 1) manifest rows base row

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

def loopInitialPremisesIn
    (manifest : Manifest)
    (rows : List SummaryRow)
    (base : Env)
    (row : SummaryRow)
    (state : ScalarLoopState) : Prop :=
  match denoteLoopSummaryIn? manifest rows base row with
  | some summary => summary.initial state
  | none => False

def loopInductionObligationsAtEnvIn
    (manifest : Manifest)
    (rows : List SummaryRow)
    (base : Env)
    (row : SummaryRow) : Prop :=
  match denoteLoopSummaryIn? manifest rows base row with
  | some summary => summary.InductionObligations
  | none => False

def loopVerifiedAtEnvIn
    (manifest : Manifest)
    (rows : List SummaryRow)
    (base : Env)
    (row : SummaryRow) : Prop :=
  match denoteLoopSummaryIn? manifest rows base row with
  | some summary => summary.Verified
  | none => False

/-- The denoted pre/post relation for one exact loop control target.  Keeping
the target in the relation prevents an enclosing loop from treating a return
or error return as an ordinary successor. -/
def loopControlRelationAtEnv
    (manifest : Manifest)
    (base : Env)
    (row : SummaryRow)
    (target : ControlTarget)
    (initial final : ScalarLoopState) : Prop :=
  match denoteLoopSummary? manifest base row with
  | some summary => summary.ControlRelation target initial final
  | none => False

def loopCertifiedControlRelationAtEnv
    (manifest : Manifest)
    (base : Env)
    (row : SummaryRow)
    (target : ControlTarget)
    (initial final : ScalarLoopState) : Prop :=
  match denoteLoopSummary? manifest base row with
  | some summary => summary.CertifiedControlRelation target initial final
  | none => False

def loopControlTargetAssertionAtEnv
    (manifest : Manifest)
    (base : Env)
    (row : SummaryRow)
    (target : ControlTarget)
    (final : ScalarLoopState) : Prop :=
  match denoteLoopSummary? manifest base row with
  | some summary =>
      ∃ edge,
        edge ∈ summary.edges ∧
        edge.target = target ∧
        edge.targetAssertion final
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

def loopSpecificationNonvacuousIn
    (manifest : Manifest)
    (assumptions : List AssumptionRow)
    (rows : List SummaryRow)
    (row : SummaryRow) : Prop :=
  ∃ env state,
    assumptionsDenoteInEnv manifest env assumptions ∧
    contextReady env row.contextVariables ∧
    loopInitialPremisesIn manifest rows env row state

def loopInductionCertificateFromAssumptionsIn
    (manifest : Manifest)
    (assumptions : List AssumptionRow)
    (rows : List SummaryRow)
    (row : SummaryRow) : Prop :=
  ∀ env,
    assumptionsDenoteInEnv manifest env assumptions →
    contextReady env row.contextVariables →
    loopInductionObligationsAtEnvIn manifest rows env row

def loopProofAuthorizationFromAssumptionsIn
    (manifest : Manifest)
    (assumptions : List AssumptionRow)
    (rows : List SummaryRow)
    (row : SummaryRow) : Prop :=
  loopSpecificationNonvacuousIn manifest assumptions rows row ∧
    loopInductionCertificateFromAssumptionsIn manifest assumptions rows row

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

def loopVerifiedFromAssumptionsIn
    (manifest : Manifest)
    (assumptions : List AssumptionRow)
    (rows : List SummaryRow)
    (row : SummaryRow) : Prop :=
  ∀ env,
    assumptionsDenoteInEnv manifest env assumptions →
    contextReady env row.contextVariables →
    loopVerifiedAtEnvIn manifest rows env row

theorem loop_induction_certificate_sound_in
    (manifest : Manifest)
    (assumptions : List AssumptionRow)
    (rows : List SummaryRow)
    (row : SummaryRow)
    (proof : loopInductionCertificateFromAssumptionsIn manifest assumptions rows row) :
    loopVerifiedFromAssumptionsIn manifest assumptions rows row := by
  intro env hAssumptions hContext
  have hObligations := proof env hAssumptions hContext
  unfold loopInductionObligationsAtEnvIn at hObligations
  unfold loopVerifiedAtEnvIn
  cases hSummary : denoteLoopSummaryIn? manifest rows env row with
  | none => simp [hSummary] at hObligations
  | some summary =>
      simp only [hSummary] at hObligations ⊢
      exact verified_of_induction summary hObligations

theorem loop_authorization_sound_in
    (manifest : Manifest)
    (assumptions : List AssumptionRow)
    (rows : List SummaryRow)
    (row : SummaryRow)
    (proof : loopProofAuthorizationFromAssumptionsIn manifest assumptions rows row) :
    loopVerifiedFromAssumptionsIn manifest assumptions rows row :=
  loop_induction_certificate_sound_in manifest assumptions rows row proof.2

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

/-- An authorized compiler row exposes a sound binary relation for each exact
control target.  The proof reuses the ordinary loop induction certificate; it
does not introduce a nested-loop proof rule. -/
theorem loop_authorization_control_relation_sound
    (manifest : Manifest)
    (assumptions : List AssumptionRow)
    (row : SummaryRow)
    (proof : loopProofAuthorizationFromAssumptions manifest assumptions row)
    (env : Env)
    (hAssumptions : assumptionsDenoteInEnv manifest env assumptions)
    (hContext : contextReady env row.contextVariables)
    (target : ControlTarget)
    (initial final : ScalarLoopState)
    (hRelation : loopControlRelationAtEnv manifest env row target initial final) :
    loopControlTargetAssertionAtEnv manifest env row target final := by
  have hCertificate := proof.2 env hAssumptions hContext
  unfold loopInductionObligationsAtEnv at hCertificate
  unfold loopControlRelationAtEnv at hRelation
  unfold loopControlTargetAssertionAtEnv
  cases hSummary : denoteLoopSummary? manifest env row with
  | none => simp [hSummary] at hRelation
  | some summary =>
      simp only [hSummary] at hCertificate hRelation ⊢
      exact control_relation_target_holds summary hCertificate target initial final hRelation

theorem loop_authorization_control_relation_is_certified
    (manifest : Manifest)
    (assumptions : List AssumptionRow)
    (row : SummaryRow)
    (proof : loopProofAuthorizationFromAssumptions manifest assumptions row)
    (env : Env)
    (hAssumptions : assumptionsDenoteInEnv manifest env assumptions)
    (hContext : contextReady env row.contextVariables)
    (target : ControlTarget)
    (initial final : ScalarLoopState)
    (hRelation : loopControlRelationAtEnv manifest env row target initial final) :
    loopCertifiedControlRelationAtEnv manifest env row target initial final := by
  have hCertificate := proof.2 env hAssumptions hContext
  unfold loopInductionObligationsAtEnv at hCertificate
  unfold loopControlRelationAtEnv at hRelation
  unfold loopCertifiedControlRelationAtEnv
  cases hSummary : denoteLoopSummary? manifest env row with
  | none => simp [hSummary] at hRelation
  | some summary =>
      simp only [hSummary] at hCertificate hRelation ⊢
      exact control_relation_contained_in_certified_relation summary hCertificate
        target initial final hRelation

theorem unsupported_kind_fails_closed
    (manifest : Manifest) (env : Env) (row : SummaryRow)
    (h : row.kind = .other) :
    denoteLoopSummary? manifest env row = none := by
  have hKind : (row.kind == .scfWhile || row.kind == .scfFor) = false := by
    rw [h]
    decide
  simp [denoteLoopSummary?, SummaryRow.supported, SummaryRow.shapeSupported,
    SummaryRow.baseShapeSupported, hKind]

end Ora.Loop

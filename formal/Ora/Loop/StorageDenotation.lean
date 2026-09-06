/-
Executable manifest-formula views for the storage loop model.

The context owns only binding and semantic-place resolution.  Store selection
is fixed here: ordinary reads use the current store, `old(place)` uses the
immutable function-entry store, and timeline read identities are introduced
only after their read event.
-/

import Ora.FixedArray
import Ora.Loop.Timeline

namespace Ora.Loop

open Ora.Obligation

structure StorageFormulaContext where
  manifest : Manifest
  entryEnv : Env
  bindLocals : Env → List Ora.Integer.Value → Option Env
  resolvePlace : (FreeVarId → Option Value) → PlaceRef → Option StorageLocation

def bindStorageReads (env : Env) (reads : List StorageReadBinding) : Env :=
  reads.foldl (fun bound read => bound.setFree read.id read.value) env

theorem bindStorageReads_lookupFree_of_not_mem
    (env : Env)
    (reads : List StorageReadBinding)
    (id : FreeVarId)
    (hFresh : id ∉ reads.map StorageReadBinding.id) :
    (bindStorageReads env reads).lookupFree id = env.lookupFree id := by
  induction reads generalizing env with
  | nil => rfl
  | cons read rest ih =>
      simp only [List.map_cons, List.mem_cons, not_or] at hFresh
      simp only [bindStorageReads, List.foldl_cons]
      change (bindStorageReads (env.setFree read.id read.value) rest).lookupFree id = _
      rw [ih (env := env.setFree read.id read.value) hFresh.2]
      have hDifferent : (read.id == id) = false := by
        apply Bool.eq_false_iff.mpr
        intro hEqual
        exact hFresh.1 (beq_iff_eq.mp hEqual).symm
      change lookupFreeBinding ((read.id, read.value) :: env.freeBindings) id =
        lookupFreeBinding env.freeBindings id
      simp only [lookupFreeBinding]
      rw [hDifferent]
      rfl

def StorageFormulaContext.withStoreViews
    (context : StorageFormulaContext)
    (bindings : Env)
    (state : StorageActionState) : Env :=
  { bindings with placeValue := fun key =>
      match key with
      | .stable place =>
          match context.resolvePlace bindings.lookupFree place with
          | some location => state.currentValue location
          | none => bindings.placeValue key
      | .entry place =>
          match context.resolvePlace context.entryEnv.lookupFree place with
          | some location => state.oldValue location
          | none => bindings.placeValue key }

@[simp] theorem StorageFormulaContext.withStoreViews_lookupFree
    (context : StorageFormulaContext)
    (bindings : Env)
    (state : StorageActionState)
    (id : FreeVarId) :
    (context.withStoreViews bindings state).lookupFree id =
      bindings.lookupFree id := rfl

/-- Check storage-place resolution before constructing the total semantic
environment consumed by the shared obligation denotation. The fallback in the
environment constructor is therefore unreachable for every place read
reachable from the formula being evaluated. -/
def StorageFormulaContext.termStorageReady
    (context : StorageFormulaContext)
    (bindings : Env) : Nat → TermId → Bool
  | 0, _ => false
  | fuel + 1, id =>
      match context.manifest.terms[id]? with
      | none => false
      | some (.placeRead place) =>
          (context.resolvePlace bindings.lookupFree place).isSome
      | some (.old operand) =>
          match context.manifest.terms[operand]? with
          | some (.placeRead place) =>
              (context.resolvePlace context.entryEnv.lookupFree place).isSome
          | _ => false
      | some (.unary unary) =>
          context.termStorageReady bindings fuel unary.operand
      | some (.binary binary) =>
          context.termStorageReady bindings fuel binary.lhs &&
            context.termStorageReady bindings fuel binary.rhs
      | some (.refinementPredicate predicate) =>
          context.termStorageReady bindings fuel predicate.value &&
            predicate.args.all (context.termStorageReady bindings fuel)
      | some (.quantified quantified) =>
          (match quantified.condition with
           | none => true
           | some condition => context.termStorageReady bindings fuel condition) &&
            context.termStorageReady bindings fuel quantified.body
      | some (.boolLit _) => true
      | some (.intLit _) => true
      | some (.variable _) => true
      | some .result => true

def StorageFormulaContext.formulaStorageReady
    (context : StorageFormulaContext)
    (bindings : Env) : FormulaRef → Bool
  | .term id =>
      context.termStorageReady bindings (context.manifest.terms.length + 1) id

def StorageFormulaContext.preheaderEnv
    (context : StorageFormulaContext)
    (entryStorage currentStorage : StorageState) : Env :=
  context.withStoreViews context.entryEnv {
    loop := { locals := [], storage := currentStorage }
    entryStorage := entryStorage
  }

def StorageFormulaContext.stateEnv?
    (context : StorageFormulaContext)
    (state : StorageActionState) : Option Env := do
  let locals ← context.bindLocals context.entryEnv state.loop.locals
  pure (context.withStoreViews (bindStorageReads locals state.reads) state)

def StorageFormulaContext.evalPreheaderValue?
    (context : StorageFormulaContext)
    (entryStorage currentStorage : StorageState)
    (formula : FormulaRef) : Option Value := do
  let env := context.preheaderEnv entryStorage currentStorage
  if !context.formulaStorageReady env formula then none else pure ()
  formulaValue? context.manifest env formula

def StorageFormulaContext.evalFormula?
    (context : StorageFormulaContext)
    (formula : FormulaRef)
    (state : StorageActionState) : Option Prop := do
  let env ← context.stateEnv? state
  if !context.formulaStorageReady env formula then none else pure ()
  formulaDenotes? context.manifest env formula

theorem StorageFormulaContext.evalFormula_bool_true
    (context : StorageFormulaContext)
    (state : StorageActionState)
    (bound : Env)
    (termId : TermId)
    (hBound : context.stateEnv? state = some bound)
    (hTerm : context.manifest.terms[termId]? = some (.boolLit true)) :
    context.evalFormula? (.term termId) state = some True := by
  unfold evalFormula? formulaStorageReady termStorageReady
  simp [hBound, hTerm, formulaDenotes?, denoteFormula?]

def StorageFormulaContext.evalValue?
    (context : StorageFormulaContext)
    (formula : FormulaRef)
    (state : StorageActionState) : Option Value := do
  let env ← context.stateEnv? state
  if !context.formulaStorageReady env formula then none else pure ()
  formulaValue? context.manifest env formula

theorem StorageFormulaContext.evalValue_free_variable
    (context : StorageFormulaContext)
    (state : StorageActionState)
    (bound : Env)
    (termId : TermId)
    (reference : FreeVarRef)
    (value : Value)
    (hBound : context.stateEnv? state = some bound)
    (hTerm : context.manifest.terms[termId]? = some (.variable (.free reference)))
    (hValue : bound.lookupFree reference.id = some value) :
    context.evalValue? (.term termId) state = some value := by
  unfold evalValue? formulaStorageReady termStorageReady
  simp only [hBound, Option.bind_eq_bind, Option.some_bind, hTerm,
    Bool.not_true, Bool.false_eq_true, if_false]
  unfold formulaValue? denoteValue?
  simp only [hTerm, Env.lookupVar, hValue, Option.pure_def, Option.some_bind]

theorem StorageFormulaContext.evalFormula_place_eq_old_self
    (context : StorageFormulaContext)
    (state : StorageActionState)
    (bound : Env)
    (fuel : Nat)
    (equality current old entry : TermId)
    (ty : TyRef)
    (place : PlaceRef)
    (location : StorageLocation)
    (hBound : context.stateEnv? state = some bound)
    (hLength : context.manifest.terms.length = fuel + 1)
    (hEquality : context.manifest.terms[equality]? = some (.binary {
      op := .eq, lhs := current, rhs := old, ty := some ty }))
    (hCurrent : context.manifest.terms[current]? = some (.placeRead place))
    (hOld : context.manifest.terms[old]? = some (.old entry))
    (hEntry : context.manifest.terms[entry]? = some (.placeRead place))
    (hCurrentLocation : context.resolvePlace bound.lookupFree place = some location)
    (hEntryLocation : context.resolvePlace context.entryEnv.lookupFree place = some location)
    (hCurrentValue : bound.placeValue (.stable place) = state.loop.storage location)
    (hEntryValue : bound.placeValue (.entry place) = state.entryStorage location) :
    (∃ proposition,
      context.evalFormula? (.term equality) state = some proposition ∧ proposition) ↔
      match Value.eqProp? (state.loop.storage location)
          (state.entryStorage location) with
      | some proposition => proposition
      | none => False := by
  unfold evalFormula? formulaStorageReady
  rw [hLength]
  simp only [hBound, Option.bind_eq_bind, Option.some_bind, termStorageReady,
    hEquality, hCurrent, hOld, hEntry, hCurrentLocation, hEntryLocation,
    Option.isSome_some, Bool.true_and, Bool.not_true, Bool.false_eq_true,
    if_false]
  unfold formulaDenotes? denoteFormula?
  rw [hLength]
  simp only [hEquality, denoteValue?, hCurrent, hOld, hEntry,
    Env.lookupPlace, Env.lookupEntryPlace,
    hCurrentValue, hEntryValue, Option.bind_eq_bind,
    Option.some_bind]
  cases hEqualityValue : Value.eqProp? (state.loop.storage location)
      (state.entryStorage location) with
  | none => simp [hEqualityValue]
  | some proposition =>
      simp only [hEqualityValue, Option.pure_def, Option.some_bind,
        Option.some.injEq]
      constructor
      · rintro ⟨candidate, hCandidate, hHolds⟩
        exact hCandidate.mpr hHolds
      · intro hHolds
        exact ⟨proposition, rfl, hHolds⟩

def storageValueMatchesTy : Value → TyRef → Bool
  | .bool _, ty => ty.isBool
  | .integer value, ty => ty.integerShape? == some value.shape

structure TypedStorageFormula where
  formula : FormulaRef
  ty : TyRef
  deriving Repr, BEq, DecidableEq

structure FixedIndexProjectionExpr where
  index : TypedStorageFormula
  extent : Nat
  deriving Repr, BEq, DecidableEq

inductive StorageProjectionExpr where
  | field : String → StorageProjectionExpr
  | fixedIndex : FixedIndexProjectionExpr → StorageProjectionExpr
  | mappingKey : TypedStorageFormula → StorageProjectionExpr
  deriving Repr, BEq, DecidableEq

structure StorageLocationExpr where
  root : String
  projections : List StorageProjectionExpr := []
  valueTy : TyRef
  deriving Repr, BEq, DecidableEq

def StorageFormulaContext.evalTypedValue?
    (context : StorageFormulaContext)
    (state : StorageActionState)
    (expression : TypedStorageFormula) : Option Value := do
  let value ← context.evalValue? expression.formula state
  if storageValueMatchesTy value expression.ty then some value else none

def StorageFormulaContext.evalProjections?
    (context : StorageFormulaContext)
    (state : StorageActionState) :
    List StorageProjectionExpr → Option (List StorageProjection)
  | [] => some []
  | .field name :: rest => do
      if name.isEmpty then none else pure ()
      pure (.field name :: (← context.evalProjections? state rest))
  | .fixedIndex projection :: rest => do
      let value ← context.evalTypedValue? state projection.index
      match value with
      | .integer index =>
          let bits ← index.bitsFor? (.unsigned .w256)
          if decide (0 < projection.extent) &&
              decide (projection.extent < 2 ^ 256) then
            pure (.fixedIndex (.integer {
              shape := .unsigned .w256, bits
            }) :: (← context.evalProjections? state rest))
          else none
      | .bool _ => none
  | .mappingKey expression :: rest => do
      let value ← context.evalTypedValue? state expression
      pure (.mappingKey value :: (← context.evalProjections? state rest))

/-- A checked fixed-index projection resolves the storage action at the exact
index whose bounds predicate was proved; the extent affects safety and shape,
not the identity of the selected element. -/
theorem StorageFormulaContext.safe_fixedIndex_denotes_same_projection
    (context : StorageFormulaContext)
    (state : StorageActionState)
    (expression : TypedStorageFormula)
    (index : Ora.FixedArray.Index)
    (extent : Nat)
    (hValue : context.evalTypedValue? state expression = some (.integer {
      shape := .unsigned .w256, bits := index }))
    (hWellFormed : ({ index, extent } : Ora.FixedArray.CheckedIndex).WellFormed)
    (hSafe : ({ index, extent } : Ora.FixedArray.CheckedIndex).Safe) :
    ({ index, extent } : Ora.FixedArray.CheckedIndex).Safe ∧
      context.evalProjections? state [.fixedIndex { index := expression, extent }] =
        some [.fixedIndex (.integer { shape := .unsigned .w256, bits := index })] := by
  refine ⟨hSafe, ?_⟩
  simp only [StorageFormulaContext.evalProjections?, hValue, Option.bind_eq_bind,
    Option.some_bind, Ora.Integer.Value.bitsFor?, if_pos rfl,
    Option.pure_def]
  simp [hWellFormed.1, hWellFormed.2]

def StorageFormulaContext.evalLocation?
    (context : StorageFormulaContext)
    (expression : StorageLocationExpr)
    (state : StorageActionState) : Option StorageLocation := do
  if expression.root.isEmpty then none else pure ()
  if !(expression.valueTy.isBool || expression.valueTy.integerShape?.isSome) then none else pure ()
  pure {
    root := expression.root
    projections := ← context.evalProjections? state expression.projections
  }

/-- Resolve a one-key mapping location from the exact free-variable binding
used at this timeline point.  The key's Ora type and the stored value type are
both checked by the ordinary location evaluator. -/
theorem StorageFormulaContext.evalLocation_mapping_free_variable
    (context : StorageFormulaContext)
    (state : StorageActionState)
    (bound : Env)
    (termId : TermId)
    (reference : FreeVarRef)
    (key : Value)
    (keyTy valueTy : TyRef)
    (root : String)
    (hBound : context.stateEnv? state = some bound)
    (hTerm : context.manifest.terms[termId]? = some (.variable (.free reference)))
    (hLookup : bound.lookupFree reference.id = some key)
    (hKeyType : storageValueMatchesTy key keyTy = true)
    (hValueType : valueTy.isBool || valueTy.integerShape?.isSome)
    (hRoot : root.isEmpty = false) :
    context.evalLocation? {
      root
      projections := [.mappingKey { formula := .term termId, ty := keyTy }]
      valueTy
    } state = some { root, projections := [.mappingKey key] } := by
  unfold StorageFormulaContext.evalLocation?
    StorageFormulaContext.evalProjections?
    StorageFormulaContext.evalTypedValue?
  rw [context.evalValue_free_variable state bound termId reference key
    hBound hTerm hLookup]
  simp [StorageFormulaContext.evalProjections?, hRoot, hValueType, hKeyType]
  intro hBool
  have hSome : valueTy.integerShape?.isSome = true := by
    simpa [hBool] using hValueType
  exact Option.isSome_iff_ne_none.mp hSome

def StorageFormulaContext.actionSemantics
    (context : StorageFormulaContext) :
    StorageActionSemantics StorageLocationExpr FormulaRef := {
  evalLocation := context.evalLocation?
  evalValue := context.evalValue?
  valueMatchesLocation := fun location value => storageValueMatchesTy value location.valueTy
}

end Ora.Loop

import Ora.Loop.Storage

namespace Ora.Loop
open Ora.Obligation

structure CallSignature where
  parameters : List (FreeVarId × TyRef)
  results : CallOutcome → Option (List TyRef)

def callValuesTyped (types : List TyRef) (values : List Value) : Bool :=
  types.length == values.length &&
    (types.zip values).all (fun (ty, value) => storageValueMatchesTy value ty)

structure CallInput where
  arguments : List Value
  storage : StorageState

/-- Outcome, payload, and final store are selected by ONE execution. A void
return has an empty payload; no default return value is introduced. -/
structure CallOutput extends CallReturn where
  storage : StorageState

def CallSignature.inputTyped (signature : CallSignature) (input : CallInput) : Bool :=
  decide (signature.parameters.map Prod.fst).Nodup &&
    callValuesTyped (signature.parameters.map Prod.snd) input.arguments

def CallSignature.outputTyped (signature : CallSignature) (output : CallOutput) : Bool :=
  match signature.results output.outcome with
  | none => false
  | some types => callValuesTyped types output.payload

/-- F-SAFETYSCOPE: callee `old` snapshots the store at invocation. Arguments
bind in a fresh callee scope; caller locals, reads, and `old` are not imported. -/
def CallSignature.entry? (signature : CallSignature) (input : CallInput) :
    Option StorageActionState :=
  if signature.inputTyped input then some {
    loop := { locals := [], storage := input.storage }
    entryStorage := input.storage
    reads := (signature.parameters.zip input.arguments).map fun ((id, _), value) =>
      { id, value }
  } else none

/-- The bound callee certificate supplies both universal safety and the joint
relation. `safe` is required at invocation, regardless of whether it returns. -/
structure CertifiedCall where
  signature : CallSignature
  requires : CallInput → Prop
  relation : CallInput → CallOutput → Prop
  safe : CallInput → Prop
  footprint : CallInput → StorageLocation → Prop
  safety : ∀ input, signature.inputTyped input = true → requires input → safe input
  typed : ∀ input output, signature.inputTyped input = true → requires input →
    relation input output → signature.outputTyped output = true
  preservesOutside : ∀ input output, relation input output →
    ∀ location, ¬ footprint input location → output.storage location = input.storage location

structure CallInvocation (ValueExpr : Type) where
  callee : String
  arguments : List ValueExpr
  outcome : CallOutcome
  results : List FreeVarId

/-- Result identities live in the caller's scope. Callee locals and reads
never replace caller bindings; an error result retains that execution's store. -/
def CallInvocation.resume (call : CallInvocation V)
    (caller : StorageActionState) (output : CallOutput) : StorageActionState := {
  caller with
    loop := { caller.loop with storage := output.storage }
    reads := caller.reads ++ (call.results.zip output.payload).map fun (id, value) =>
      { id, value }
}

def CallInvocation.fresh (call : CallInvocation V) (caller : StorageActionState) : Prop :=
  (caller.reads.map StorageReadBinding.id ++ call.results).Nodup

theorem CallInvocation.resume_preserves_caller (call : CallInvocation V)
    (caller : StorageActionState) (output : CallOutput) :
    (call.resume caller output).entryStorage = caller.entryStorage ∧
    (call.resume caller output).loop.locals = caller.loop.locals ∧
    ∃ appended, (call.resume caller output).reads = caller.reads ++ appended :=
  ⟨rfl, rfl, _, rfl⟩

def callEvaluateValues?
    (evaluate : ValueExpr → StorageActionState → Option Value)
    (values : List ValueExpr) (state : StorageActionState) : Option (List Value) :=
  values.mapM (fun value => evaluate value state)

/-- F-SAFETYSCOPE: checked before invocation, without any returned result or
later path assumption. A missing certificate is a false checkpoint. -/
def CallInvocation.ready (call : CallInvocation V)
    (resolve : String → Option CertifiedCall)
    (evaluate : V → StorageActionState → Option Value) (caller : StorageActionState) : Prop :=
  ∃ callee arguments types,
    resolve call.callee = some callee ∧
    callEvaluateValues? evaluate call.arguments caller = some arguments ∧
    callee.signature.inputTyped { arguments, storage := caller.loop.storage } = true ∧
    callee.requires { arguments, storage := caller.loop.storage } ∧ call.fresh caller ∧
    callee.signature.results call.outcome = some types ∧ types.length = call.results.length

theorem CallInvocation.ready_safety (call : CallInvocation V)
    (resolve : String → Option CertifiedCall) (evaluate : V → StorageActionState → Option Value)
    (caller : StorageActionState) (ready : call.ready resolve evaluate caller) :
    ∃ callee arguments, resolve call.callee = some callee ∧
      callEvaluateValues? evaluate call.arguments caller = some arguments ∧
      callee.safe { arguments, storage := caller.loop.storage } := by
  rcases ready with ⟨callee, arguments, _, resolved, evaluated, typed, requires, _⟩
  exact ⟨callee, arguments, resolved, evaluated, callee.safety _ typed requires⟩

end Ora.Loop

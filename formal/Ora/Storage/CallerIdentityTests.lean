import Ora.Storage.CallerIdentity

namespace Ora.CallerIdentityTests
open Ora.Obligation Ora.Loop Ora.CallerIdentityModel

def key : InvocationKey := {moduleDigest := "live-module", functionId := 1, serial := 7}
def capture (word : BitVec 160) : Capture := {key, caller := word}
def env (word : BitVec 160) : Env := (capturedEnv (capture word)).toEnv key
def terms : Manifest := {terms := [.invocationCaller 1 (.compilerTypeId 14)]}

theorem live_leaf (word : BitVec 160) :
    denoteValue? terms (env word) 1 0 = some (Ora.Storage.addressValue word) := rfl

theorem missing_binding : denoteValue? terms Env.empty 1 0 = none := rfl

theorem wrong_owner (word : BitVec 160) :
    (env word).callerValue? 2 (.compilerTypeId 14) = none := rfl

theorem wrong_type (word : BitVec 160) :
    (env word).callerValue? 1 (.compilerTypeId 6) = none := rfl

theorem wrong_module (word : BitVec 160) :
    ((capturedEnv (capture word)).toEnv {key with moduleDigest := "foreign"}).callerValue?
      1 (.compilerTypeId 14) = none := rfl

theorem wrong_invocation (word : BitVec 160) :
    ((capturedEnv (capture word)).toEnv {key with serial := 8}).callerValue?
      1 (.compilerTypeId 14) = none := rfl

theorem source_shadow (word : BitVec 160) (id : FreeVarId) (value : Value) :
    denoteValue? terms ((env word).setFree id value) 1 0 =
      some (Ora.Storage.addressValue word) := rfl

theorem bound_shadow (word : BitVec 160) (value : Value) :
    denoteValue? terms ((env word).pushBound value) 1 0 =
      some (Ora.Storage.addressValue word) := rfl

theorem reads_preserve_capture (base : Env) (reads : List StorageReadBinding) :
    (bindStorageReads base reads).invocationCaller = base.invocationCaller := by
  induction reads generalizing base with
  | nil => rfl
  | cons read rest ih => exact ih (base.setFree read.id read.value)

theorem stores_preserve_capture (base : Env) (context : StorageFormulaContext)
    (state : StorageActionState) :
    (context.withStoreViews base state).invocationCaller = base.invocationCaller := rfl

theorem storage_is_not_a_capture (value : Value) :
    denoteValue? terms {Env.empty with placeValue := fun _ => value} 1 0 = none := rfl

theorem zero_has_no_nonzero_fact :
    (env 0).callerValue? 1 (.compilerTypeId 14) = some (Ora.Storage.addressValue 0) := rfl

end Ora.CallerIdentityTests

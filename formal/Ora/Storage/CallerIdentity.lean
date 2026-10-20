import Ora.Storage.LifecycleEntry
import Ora.Storage.Address

/- E2a isolated model. No production module is replaced. The capture is an
   explicit runtime premise; a digest, a source spelling, and a chosen witness
   cannot authenticate it. The existing timeline executor is reused. -/
namespace Ora.CallerIdentityModel
open Ora.Obligation Ora.Loop Ora.StorageLifecycle

structure InvocationKey where
  moduleDigest : String
  functionId : Nat
  serial : Nat
  deriving DecidableEq

structure Capture where
  key : InvocationKey
  caller : BitVec 160

/-- Context is a separate namespace; it is never a FreeVarId or parameter. -/
structure ContextEnv where
  ordinary : Env
  invocation : Option Capture

def callerValue? (expected : InvocationKey) (env : ContextEnv) : Option Value := do
  let captured ← env.invocation
  if captured.key = expected then some (Ora.Storage.addressValue captured.caller) else none

def capturedEnv (captured : Capture) : ContextEnv :=
  { ordinary := Env.empty, invocation := some captured }

def setSource (env : ContextEnv) (id : FreeVarId) (value : Value) : ContextEnv :=
  { env with ordinary := env.ordinary.setFree id value }

def bindReads (env : ContextEnv) (reads : List StorageReadBinding) : ContextEnv :=
  { env with ordinary := bindStorageReads env.ordinary reads }

def storeViews (env : ContextEnv) (context : StorageFormulaContext)
    (state : StorageActionState) : ContextEnv :=
  { env with ordinary := context.withStoreViews env.ordinary state }

theorem captured_caller_exact (captured : Capture) :
    callerValue? captured.key (capturedEnv captured) =
      some (Ora.Storage.addressValue captured.caller) := by
  simp [callerValue?, capturedEnv]

theorem absent_context_refuses (expected : InvocationKey) (ordinary : Env) :
    callerValue? expected {ordinary, invocation := none} = none := rfl

theorem stale_invocation_refuses (captured : Capture) (expected : InvocationKey)
    (different : captured.key ≠ expected) :
    callerValue? expected (capturedEnv captured) = none := by
  simp [callerValue?, capturedEnv, different]

theorem foreign_function_refuses (captured : Capture) (expected : InvocationKey)
    (different : captured.key.functionId ≠ expected.functionId) :
    callerValue? expected (capturedEnv captured) = none := by
  apply stale_invocation_refuses
  intro equal
  exact different (congrArg InvocationKey.functionId equal)

theorem foreign_module_refuses (captured : Capture) (expected : InvocationKey)
    (different : captured.key.moduleDigest ≠ expected.moduleDigest) :
    callerValue? expected (capturedEnv captured) = none := by
  apply stale_invocation_refuses
  intro equal
  exact different (congrArg InvocationKey.moduleDigest equal)

theorem source_binding_cannot_spoof_caller (expected : InvocationKey)
    (env : ContextEnv) (id : FreeVarId) (value : Value) :
    callerValue? expected (setSource env id value) = callerValue? expected env := rfl

theorem reads_cannot_change_caller (expected : InvocationKey)
    (env : ContextEnv) (reads : List StorageReadBinding) :
    callerValue? expected (bindReads env reads) = callerValue? expected env := rfl

theorem storage_views_cannot_change_caller (expected : InvocationKey)
    (env : ContextEnv) (context : StorageFormulaContext) (state : StorageActionState) :
    callerValue? expected (storeViews env context state) = callerValue? expected env := rfl

theorem caller_exact_address_sort (captured : Capture) :
    storageValueMatchesTy (Ora.Storage.addressValue captured.caller)
      (.compilerTypeId 14) = true := Ora.Storage.addressValueTyped _

theorem caller_identity_cannot_change_within_invocation
    (captured : Capture) (first second : Value)
    (a : callerValue? captured.key (capturedEnv captured) = some first)
    (b : callerValue? captured.key (capturedEnv captured) = some second) : first = second := by
  exact Option.some.inj (a.symm.trans b)

theorem different_callers_are_not_interchangeable (left right : Capture)
    (different : left.caller ≠ right.caller) :
    callerValue? left.key (capturedEnv left) ≠
      callerValue? right.key (capturedEnv right) := by
  rw [captured_caller_exact, captured_caller_exact]
  intro equal
  exact different ((Ora.Storage.addressValueInjective _ _).mp (Option.some.inj equal))

/-- Zero is admitted; this gate does not import the native encoder's ambient
    nonzero-caller premise. Identity alone never proves a NonZeroAddress. -/
theorem zero_caller_is_admitted (key : InvocationKey) :
    callerValue? key (capturedEnv {key, caller := 0}) =
      some (Ora.Storage.addressValue 0) := captured_caller_exact _

theorem caller_nonzero_is_not_automatic :
    ¬ (∀ caller : BitVec 160, caller ≠ 0) := by
  intro all
  exact all 0 rfl

theorem per_event_fresh_caller_is_unsound :
    (∃ caller : BitVec 160, caller = 0) ∧
    (∃ caller : BitVec 160, caller = 1) ∧
    ¬ (∃ caller : BitVec 160, caller = 0 ∧ caller = 1) := by
  refine ⟨⟨0, rfl⟩, ⟨1, rfl⟩, ?_⟩
  rintro ⟨caller, zero, one⟩
  have different : (0 : BitVec 160) ≠ 1 := by decide
  exact different (zero.symm.trans one)

theorem distinct_invocations_may_share_caller (key : InvocationKey) (word : BitVec 160) :
    callerValue? key (capturedEnv {key, caller := word}) =
      callerValue? {key with serial := key.serial + 1}
        (capturedEnv {key := {key with serial := key.serial + 1}, caller := word}) := by
  simp [callerValue?, capturedEnv]

theorem boolean_is_not_address :
    storageValueMatchesTy (.bool true) (.compilerTypeId 14) = false := rfl

theorem wide_word_is_not_address (word : BitVec 256) :
    storageValueMatchesTy (.integer {shape := .unsigned .w256, bits := word})
      (.compilerTypeId 14) = false := rfl

/-- A family closes over ONE captured invocation for its entry predicate and
    every event, reusing TimelineExec. New invocations may choose any caller. -/
structure Family (F L V C : Type) where
  semantics : Capture → TimelineSemantics F L V C
  events : List (EventOf F L V C)
  entry : Capture → StorageActionState → Prop

def Family.step (family : Family F L V C) : Transition := fun before after =>
  ∃ capture state count result, family.entry capture state ∧
    state.loop.storage = before ∧
    TimelineExec (family.semantics capture) (family.events.take count) state result ∧
    result.state.loop.storage = after

def Family.Checked (family : Family F L V C) (claims : List Claim) : Prop :=
  ∀ capture state, family.entry capture state → Invariant claims state.loop.storage →
    PrefixesHold claims (family.semantics capture) family.events state

theorem every_caller_every_prefix (family : Family F L V C)
    (checked : family.Checked claims) (valid : Invariant claims before)
    (executed : family.step before after) : Invariant claims after := by
  obtain ⟨capture, state, count, result, entry, start, trace, finish⟩ := executed
  rw [← finish]
  exact checked capture state entry (start.symm ▸ valid) count result trace

theorem complete_caller_coverage (family : Family F L V C) (actual : Transition)
    (authenticated : ∀ before after, actual before after →
      ∃ capture state count result, family.entry capture state ∧
        state.loop.storage = before ∧
        TimelineExec (family.semantics capture) (family.events.take count) state result ∧
        result.state.loop.storage = after) : Covers actual [family.step] := by
  intro before after executed
  exact ⟨family.step, by simp, authenticated before after executed⟩

theorem caller_family_certificate (families : List (Family F L V C))
    (witness : ∃ store, initial store)
    (bootstrap : ∀ store, initial store → Invariant claims store)
    (checked : ∀ family ∈ families, family.Checked claims) :
    Certificate claims initial (families.map Family.step) := by
  refine ⟨witness, bootstrap, ?_⟩
  intro step member before after valid executed
  obtain ⟨family, inFamilies, same⟩ := List.mem_map.mp member
  subst step
  exact every_caller_every_prefix family (checked family inFamilies) valid executed

/-- Existential witness success cannot restrict the universal caller domain. -/
theorem one_witness_does_not_cover_callers :
    (∃ caller : BitVec 160, caller = 0) ∧ ¬ (∀ caller : BitVec 160, caller = 0) := by
  constructor
  · exact ⟨0, rfl⟩
  · intro all
    have impossible := all 1
    have different : (1 : BitVec 160) ≠ 0 := by decide
    exact different impossible

theorem matching_caller_values_do_not_authenticate_identity (key : InvocationKey) :
    callerValue? key (capturedEnv {key := {key with serial := key.serial + 1}, caller := 0}) = none := by
  apply stale_invocation_refuses
  intro equal
  have bad := congrArg InvocationKey.serial equal
  exact Nat.succ_ne_self key.serial bad

/-- Only a checked capture supplies the interpreter's invocation namespace. -/
def ContextEnv.toEnv (expected : InvocationKey) (env : ContextEnv) : Env :=
  { env.ordinary with invocationCaller := do
      let capture ← env.invocation
      if capture.key = expected then some (expected.functionId, capture.caller) else none }

theorem interpreter_caller_agrees (expected : InvocationKey) (env : ContextEnv)
    (positive : expected.functionId > 0) :
    (env.toEnv expected).callerValue? expected.functionId (.compilerTypeId 14) =
      callerValue? expected env := by
  cases h : env.invocation with
  | none => simp [ContextEnv.toEnv, callerValue?, Env.callerValue?, h]
  | some capture =>
      by_cases equal : capture.key = expected <;>
        simp [ContextEnv.toEnv, callerValue?, Env.callerValue?, h, positive, equal,
          Ora.Storage.addressValue]
      rfl

end Ora.CallerIdentityModel

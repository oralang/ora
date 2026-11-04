import Ora.ContractInvariantBootstrap
import Ora.Storage.CallerIdentity

/- Model-only bridge for selected scalar storage bounds. The induction rule,
   storage semantics, caller families and prefix executor are all existing.
   This supplies no HIR coverage witness and activates no compiler authority.

   A live binding must enumerate every deployment/runtime writer and every
   observable prefix, authenticate retained checks, and prove exact frames.
   In particular, an unsupported claim at another slot is not evidence for it,
   and an error return never discards earlier storage writes. -/
namespace Ora.ScalarBoundAuthentication
open Ora.Obligation Ora.Loop Ora.StorageLifecycle Ora.CallerIdentityModel

abbrev Word := BitVec 256

def word (value : Word) : Value :=
  .integer { shape := .unsigned .w256, bits := value }

def capClaim (root : String) (cap : Word) : Claim :=
  { location := { root }, name := "MaxValue", bounds := [word cap] }

def zeroInitial (root : String) (store : StorageState) : Prop :=
  store { root } = word 0

theorem cap_denotation (root : String) (cap value : Word) :
    (capClaim root cap).holds (word value) ↔ value.toNat ≤ cap.toNat := Iff.rfl

theorem zero_bootstrap (root : String) (cap : Word) :
    ∀ store, zeroInitial root store → Invariant [capClaim root cap] store := by
  intro store zero claim member
  have same : claim = capClaim root cap := by simpa using member
  subst claim
  change (capClaim root cap).holds (store { root })
  rw [zero]
  exact (cap_denotation root cap 0).mpr (Nat.zero_le _)

theorem zero_witness (root : String) : ∃ store, zeroInitial root store :=
  ⟨fun _ => word 0, rfl⟩

/-- Includes every prefix of every path, with its captured invocation context.
    A proof about only the final store, successful outcome or one chosen caller
    cannot inhabit this premise. -/
theorem checked_paths_certificate
    (root : String) (cap : Word) (paths : List (Family F L V C))
    (checked : ∀ path ∈ paths, path.Checked [capClaim root cap]) :
    Certificate [capClaim root cap] (zeroInitial root) (paths.map Family.step) :=
  caller_family_certificate paths (zero_witness root) (zero_bootstrap root cap) checked

/-- Coverage is over the actual complete writer set, not a selected list of
    functions that happened to verify. Deployment writes belong to this set
    too: zero is already valid before their first write for these upper bounds. -/
def AllWritersCovered (actualWriters checkedSteps : List Transition) : Prop :=
  ∀ writer ∈ actualWriters, Covers writer checkedSteps

theorem base_preservation_coverage_entail_assumable
    (certificate : Certificate claims initial checkedSteps)
    (coverage : AllWritersCovered actualWriters checkedSteps) :
    Ora.InvariantBootstrap.EntrySound initial (Invariant claims) actualWriters := by
  apply Ora.InvariantBootstrap.inductive_rule_sound certificate.bootstrap
  intro writer member before after good executed
  exact covered_step_preserves certificate (coverage writer member) good executed

/-- The only fact exported is the authenticated claim at the reader's entry
    store. This theorem imports neither a source invariant nor a solver result. -/
theorem reader_bound
    (paths : List (Family F L V C))
    (checked : ∀ path ∈ paths, path.Checked [capClaim root cap])
    (coverage : AllWritersCovered actualWriters (paths.map Family.step))
    (reached : Ora.InvariantBootstrap.EntryReachable
      (zeroInitial root) actualWriters entry) :
    (capClaim root cap).holds (entry { root }) := by
  have sound := base_preservation_coverage_entail_assumable
    (checked_paths_certificate root cap paths checked) coverage
  exact sound entry reached (capClaim root cap) (by simp)

/-- Retained parameter/conversion checks establish this premise for init and
    setLimit. Their refinement type annotation alone is not this premise. -/
theorem bounded_store_preserves (root : String) (cap value : Word)
    (before : StorageState) (good : Invariant [capClaim root cap] before)
    (checked : value.toNat ≤ cap.toNat) :
    Invariant [capClaim root cap] (before.set { root } (word value)) := by
  apply checked_write_preserves _ _ _ _ good
  intro claim member _
  have same : claim = capClaim root cap := by simpa using member
  subst claim
  exact (cap_denotation root cap value).mpr checked

/-- The source's mint branch checks the computed word against the cap.
    Checked-add safety remains a separate obligation; this is not permission
    to erase an overflow guard. -/
theorem mint_store_preserves (root : String) (cap supply amount : Word)
    (before : StorageState) (good : Invariant [capClaim root cap] before)
    (branch : (supply + amount).toNat ≤ cap.toNat) :
    Invariant [capClaim root cap]
      (before.set { root } (word (supply + amount))) :=
  bounded_store_preserves root cap (supply + amount) before good branch

/-- burn needs the authenticated entry bound and its existing underflow
    precondition. No upper bound is assumed for an unrelated value. -/
theorem burn_store_preserves (root : String) (cap supply amount : Word)
    (before : StorageState) (good : Invariant [capClaim root cap] before)
    (loaded : before { root } = word supply)
    (noUnderflow : amount ≤ supply) :
    Invariant [capClaim root cap]
      (before.set { root } (word (supply - amount))) := by
  have bounded := good (capClaim root cap) (by simp)
  change (capClaim root cap).holds (before { root }) at bounded
  rw [loaded, cap_denotation] at bounded
  apply bounded_store_preserves root cap (supply - amount) before good
  rw [BitVec.toNat_sub_of_le noUnderflow]
  exact Nat.le_trans (Nat.sub_le _ _) bounded

/-- Mapping writes are framed by the actual resolved root, independently of
    key values. A guessed/substituted parameter key cannot prove this premise. -/
theorem different_root_write_preserves (root : String) (cap : Word)
    (before : StorageState) (written : StorageLocation) (value : Value)
    (good : Invariant [capClaim root cap] before)
    (different : root ≠ written.root) :
    Invariant [capClaim root cap] (before.set written value) := by
  apply frame_preserves _ _ _ good
  intro claim member
  have same : claim = capClaim root cap := by simpa using member
  subst claim
  apply StorageState.set_other
  intro equal
  exact different (congrArg StorageLocation.root equal)

theorem error_return_keeps_store (state : LoopState) (errorId : Nat)
    (payload : List Value) :
    ({ state with returned := some { outcome := .error errorId, payload } }).storage =
      state.storage := rfl

theorem error_return_cannot_rescue_bad_prefix (state : LoopState)
    (errorId : Nat) (payload : List Value)
    (bad : ¬ Invariant claims state.storage) :
    ¬ Invariant claims
      ({ state with returned := some { outcome := .error errorId, payload } }).storage :=
  bad

/-- A bound at one entry slot does not authorize the same bound at another. -/
theorem selected_claim_is_not_another_slot :
    ∃ store, Invariant [capClaim "limit" 1000000] store ∧
      ¬ (capClaim "other" 1000000).holds (store { root := "other" }) := by
  let store : StorageState := StorageState.set (fun _ => word 0) { root := "other" } (word 1000001)
  refine ⟨store, ?_, ?_⟩
  · apply zero_bootstrap
    simp [zeroInitial, store, StorageState.set]
  · change ¬ (capClaim "other" 1000000).holds (word 1000001)
    rw [cap_denotation]
    decide

/-- S1: a positive witness elsewhere never authenticates the actual initial
    store. This uses the committed assume-without-base-case characterization. -/
theorem unbootstrapped_cap_not_assumable :
    ¬ Ora.InvariantBootstrap.EntrySound
      (fun store : StorageState => store { root := "limit" } = word 1000001)
      (Invariant [capClaim "limit" 1000000]) writers := by
  apply Ora.InvariantBootstrap.missing_bootstrap_forbids_entry_assumption
    (state := fun _ => word 1000001)
  refine ⟨rfl, ?_⟩
  intro good
  have bounded := good (capClaim "limit" 1000000) (by simp)
  change (capClaim "limit" 1000000).holds (word 1000001) at bounded
  rw [cap_denotation] at bounded
  have impossible : ¬ (1000001 : Word).toNat ≤ (1000000 : Word).toNat := by decide
  exact impossible bounded

def badWriter : Transition := fun before after =>
  after = before.set { root := "limit" } (word 1000001)

theorem unpreserved_cap_not_assumable :
    ¬ Ora.InvariantBootstrap.EntrySound (zeroInitial "limit")
      (Invariant [capClaim "limit" 1000000]) [badWriter] := by
  apply Ora.InvariantBootstrap.bad_reached_writer_forbids_entry_assumption
    (before := fun _ => word 0)
    (after := StorageState.set (fun _ => word 0) { root := "limit" } (word 1000001))
    (writer := badWriter)
  · exact .initial rfl
  · simp
  · rfl
  · intro good
    have bounded := good (capClaim "limit" 1000000) (by simp)
    change (capClaim "limit" 1000000).holds (word 1000001) at bounded
    rw [cap_denotation] at bounded
    have impossible : ¬ (1000001 : Word).toNat ≤ (1000000 : Word).toNat := by decide
    exact impossible bounded

theorem omitted_bad_writer_cannot_be_authenticated
    (paths : List (Family F L V C))
    (checked : ∀ path ∈ paths, path.Checked [capClaim "limit" 1000000]) :
    ¬ AllWritersCovered [badWriter] (paths.map Family.step) := by
  intro covered
  exact unpreserved_cap_not_assumable
    (base_preservation_coverage_entail_assumable
      (checked_paths_certificate "limit" 1000000 paths checked) covered)

theorem dispatcher_base :
    ∀ store, zeroInitial "limit" store →
      Invariant [capClaim "limit" 1000000] store :=
  zero_bootstrap "limit" 1000000

theorem supply_base :
    ∀ store, zeroInitial "total_supply" store →
      Invariant [capClaim "total_supply" 1000000000] store :=
  zero_bootstrap "total_supply" 1000000000

end Ora.ScalarBoundAuthentication

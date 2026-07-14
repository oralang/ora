/-
Declarative specification of Sinora's dispatcher planner.

This module states the planner's typed input, candidate space, scoring policy,
tie breakers, hysteresis, and branch relation.  It is a specification of the
algorithm implemented by `sinora/src/switch_routing.zig`; it is not a theorem
about Zig source semantics.  The production compiler is tied to this model by
the per-contract trace checker in `Ora.DispatcherTableSync`.

Dense candidates carry their `PlanAdmissible` proof.  This makes an
inadmissible dense plan structurally unavailable to the reference planner.
-/

import Ora.Dispatcher
import Ora.Spec.DispatcherFacts

namespace Ora.DispatcherPlannerSpec

open Ora.Spec.DispatcherFacts

abbrev Selector := Ora.Dispatcher.Selector

inductive DispatchPolicy where
  | gas
  | balanced
  | size
  deriving Repr, DecidableEq

def DispatchPolicy.compilerName : DispatchPolicy → String
  | .gas => "gas"
  | .balanced => "balanced"
  | .size => "size"

def DispatchPolicy.lambdaX1000PerByte (policy : DispatchPolicy) : Nat :=
  ((expectedDispatchPolicyLambdasX1000.find? fun entry =>
    entry.1 == policy.compilerName).map Prod.snd).getD 0

inductive DenseKind where
  | bitWindow
  | multiplicative
  deriving Repr, DecidableEq

inductive DensePlan where
  | bitWindow (tableSlots indexBits indexShift : Nat)
  | multiplicative (tableSlots indexBits indexShift constant : Nat)
  deriving Repr, DecidableEq

structure SparsePlan where
  tableSlots : Nat
  bucketBits : Nat
  bucketShift : Nat
  deriving Repr, DecidableEq

inductive Plan where
  | linear
  | dense (plan : DensePlan)
  | sparse (plan : SparsePlan)
  deriving Repr, DecidableEq

structure Input (Label : Type) where
  cases : List (Selector × Label)
  hasDefault : Bool
  policy : DispatchPolicy

def selectors (input : Input Label) : List Selector :=
  input.cases.map Prod.fst

def pow2 (bits : Nat) : Nat := 2 ^ bits

def bitWindowIndex (selector bits shift : Nat) : Nat :=
  (selector / pow2 shift) % pow2 bits

def multiplicativeCandidate (index : Nat) : Nat :=
  let z0 : BitVec 32 := BitVec.ofNat 32 index + BitVec.ofNat 32 0x9E3779B9
  let z1 : BitVec 32 := (z0 ^^^ (z0 >>> 16)) * BitVec.ofNat 32 0x85EBCA6B
  let z2 : BitVec 32 := (z1 ^^^ (z1 >>> 13)) * BitVec.ofNat 32 0xC2B2AE35
  let z3 : BitVec 32 := z2 ^^^ (z2 >>> 16)
  (z3 ||| BitVec.ofNat 32 1).toNat

def multiplicativeIndex (selector constant bits : Nat) : Nat :=
  (selector * constant / pow2 (32 - bits)) % pow2 bits

def DensePlan.index : DensePlan → Selector → Nat
  | .bitWindow _ bits shift => fun selector => bitWindowIndex selector bits shift
  | .multiplicative _ bits _ constant =>
      fun selector => multiplicativeIndex selector constant bits

def DensePlan.kind : DensePlan → DenseKind
  | .bitWindow .. => .bitWindow
  | .multiplicative .. => .multiplicative

def DensePlan.tableSlots : DensePlan → Nat
  | .bitWindow slots .. => slots
  | .multiplicative slots .. => slots

def DensePlan.indexBits : DensePlan → Nat
  | .bitWindow _ bits _ => bits
  | .multiplicative _ bits _ _ => bits

def DensePlan.indexShift : DensePlan → Nat
  | .bitWindow _ _ shift => shift
  | .multiplicative _ _ shift _ => shift

def DensePlan.constant : DensePlan → Nat
  | .bitWindow .. => 0
  | .multiplicative _ _ _ constant => constant

def SparsePlan.index (plan : SparsePlan) : Selector → Nat :=
  fun selector => bitWindowIndex selector plan.bucketBits plan.bucketShift

def Plan.toBuilderPlan : Plan → Ora.Dispatcher.BuilderPlan Nat
  | .linear => .linear
  | .dense plan => .dense plan.index
  | .sparse plan => .sparse plan.index

def Plan.admissible (cases : List (Selector × Label)) (plan : Plan) : Bool :=
  Ora.Dispatcher.PlanAdmissible cases plan.toBuilderPlan

def DensePlan.wellShaped : DensePlan → Bool
  | .bitWindow slots bits shift =>
      decide (0 < bits) && decide (bits <= 8) && decide (slots = pow2 bits) &&
        decide (shift <= 24)
  | .multiplicative slots bits shift constant =>
      decide (0 < bits) && decide (bits <= 8) && decide (slots = pow2 bits) &&
        decide (shift = 32 - bits) && decide (0 < constant) && decide (constant % 2 = 1)

def SparsePlan.wellShaped (plan : SparsePlan) : Bool :=
  decide (0 < plan.bucketBits) && decide (plan.bucketBits <= 8) &&
    decide (plan.tableSlots = pow2 plan.bucketBits) && decide (plan.bucketShift <= 24)

def Plan.wellShaped : Plan → Bool
  | .linear => true
  | .dense plan => plan.wellShaped
  | .sparse plan => plan.wellShaped

structure DenseCandidate (cases : List (Selector × Label)) where
  plan : DensePlan
  admissible : Plan.admissible cases (.dense plan) = true
  wellShaped : Plan.wellShaped (.dense plan) = true

structure SparseCandidate where
  plan : SparsePlan
  wellShaped : Plan.wellShaped (.sparse plan) = true

structure CertifiedPlan (cases : List (Selector × Label)) where
  plan : Plan
  admissible : Plan.admissible cases plan = true
  wellShaped : Plan.wellShaped plan = true

def noDuplicateNat : List Nat → Bool
  | [] => true
  | value :: rest => !(rest.contains value) && noDuplicateNat rest

def collisionFree (cases : List (Selector × Label)) (index : Selector → Nat) : Bool :=
  noDuplicateNat ((cases.map Prod.fst).map index)

theorem noDuplicateNat_sound (values : List Nat)
    (h : noDuplicateNat values = true) : values.Nodup := by
  induction values with
  | nil => simp
  | cons value rest ih =>
      simp [noDuplicateNat] at h
      simp [List.nodup_cons, h.1, ih h.2]

theorem map_nodup_injective (values : List Nat) (index : Nat → Nat)
    (hnodup : (values.map index).Nodup) :
    ∀ first second,
      first ∈ values → second ∈ values → index first = index second → first = second := by
  induction values with
  | nil => simp
  | cons value rest ih =>
      intro first second hfirst hsecond heq
      simp only [List.map_cons, List.nodup_cons] at hnodup
      simp only [List.mem_cons] at hfirst hsecond
      rcases hfirst with rfl | hfirst
      · rcases hsecond with rfl | hsecond
        · rfl
        · exact False.elim
            (hnodup.1 (heq ▸ List.mem_map_of_mem index hsecond))
      · rcases hsecond with rfl | hsecond
        · exact False.elim
            (hnodup.1 (heq ▸ List.mem_map_of_mem index hfirst))
        · exact ih hnodup.2 first second hfirst hsecond heq

theorem collisionFree_admissible (cases : List (Selector × Label))
    (index : Selector → Nat) (h : collisionFree cases index = true) :
    Ora.Dispatcher.PlanAdmissible cases (.dense index) = true := by
  have hinjective := map_nodup_injective (cases.map Prod.fst) index
    (noDuplicateNat_sound _ h)
  simp [Ora.Dispatcher.PlanAdmissible]
  intro first firstLabel hfirst second secondLabel hsecond
  by_cases heq : index first = index second
  · exact Or.inr (hinjective first second
      (List.mem_map_of_mem Prod.fst hfirst)
      (List.mem_map_of_mem Prod.fst hsecond) heq)
  · exact Or.inl heq

def mkDenseCandidate? (cases : List (Selector × Label)) (plan : DensePlan) :
    Option (DenseCandidate cases) :=
  if hcollision : collisionFree cases plan.index = true then
    if hshape : Plan.wellShaped (.dense plan) = true then
      some ⟨plan, collisionFree_admissible cases plan.index hcollision, hshape⟩
    else none
  else none

def mkSparseCandidate? (plan : SparsePlan) : Option SparseCandidate :=
  if hshape : Plan.wellShaped (.sparse plan) = true then
    some ⟨plan, hshape⟩
  else none

def sparseBucketBits : List Nat := expectedSparseBucketBits
def sparseBucketShifts : List Nat := expectedSparseBucketShifts

def bitWindowCandidates (cases : List (Selector × Label)) : List (DenseCandidate cases) :=
  sparseBucketBits.flatMap fun bits =>
    sparseBucketShifts.filterMap fun shift =>
      mkDenseCandidate? cases (.bitWindow (pow2 bits) bits shift)

def ceilPowerOfTwoFrom : Nat → Nat → Nat → Nat
  | 0, current, _ => current
  | fuel + 1, current, minimum =>
      if minimum <= current then current
      else ceilPowerOfTwoFrom fuel (current * 2) minimum

def ceilPowerOfTwo (minimum : Nat) : Nat :=
  ceilPowerOfTwoFrom 8 2 minimum

def multiplicativeTableSlotsFrom : Nat → Nat → List Nat
  | 0, _ => []
  | fuel + 1, tableSlots =>
      if tableSlots <= expectedDenseMaxTableSlots then
        tableSlots :: multiplicativeTableSlotsFrom fuel (tableSlots * 2)
      else
        []

def multiplicativeTableSlots (casesLen : Nat) : List Nat :=
  multiplicativeTableSlotsFrom 9 (ceilPowerOfTwo (max casesLen 2))

def indexBitsForSlots? : Nat → Option Nat
  | 2 => some 1
  | 4 => some 2
  | 8 => some 3
  | 16 => some 4
  | 32 => some 5
  | 64 => some 6
  | 128 => some 7
  | 256 => some 8
  | _ => none

def firstMultiplicativeCandidate? (cases : List (Selector × Label))
    (tableSlots bits : Nat) : Nat → Nat → Option (DenseCandidate cases)
  | 0, _ => none
  | fuel + 1, candidateIndex =>
      let constant := multiplicativeCandidate candidateIndex
      let plan := DensePlan.multiplicative tableSlots bits (32 - bits) constant
      match mkDenseCandidate? cases plan with
      | some candidate => some candidate
      | none => firstMultiplicativeCandidate? cases tableSlots bits fuel (candidateIndex + 1)

def multiplicativeCandidates (cases : List (Selector × Label)) :
    List (DenseCandidate cases) :=
  (multiplicativeTableSlots cases.length).filterMap fun tableSlots =>
    match indexBitsForSlots? tableSlots with
    | some bits =>
        firstMultiplicativeCandidate? cases tableSlots bits
          expectedMultiplicativeSearchBudget 0
    | none => none

def denseCandidates (cases : List (Selector × Label)) : List (DenseCandidate cases) :=
  bitWindowCandidates cases ++ multiplicativeCandidates cases

def incrementBucket : Nat → List (Nat × Nat) → List (Nat × Nat)
  | bucket, [] => [(bucket, 1)]
  | bucket, entry :: rest =>
      if entry.1 == bucket then (entry.1, entry.2 + 1) :: rest
      else entry :: incrementBucket bucket rest

def bucketCounts (selectors : List Nat) (bits shift : Nat) : List Nat :=
  (selectors.foldl (fun counts selector =>
    incrementBucket (bitWindowIndex selector bits shift) counts) []).map Prod.snd

def usedBucketCount (counts : List Nat) : Nat := counts.length
def maxBucketSize (counts : List Nat) : Nat := (counts.max?).getD 0

def successfulScanChecks (counts : List Nat) : Nat :=
  counts.foldl (fun total count => total + count * (count + 1) / 2) 0

def divRound (numerator denominator : Nat) : Nat :=
  (numerator + denominator / 2) / denominator

def linearAverageChecks (casesLen : Nat) : Nat :=
  divRound
    ((casesLen * (casesLen + 1) / 2) * expectedExactSelectorCheckX1000)
    casesLen

def planScore (runtimeAverage codeBytes : Nat) (policy : DispatchPolicy) : Nat :=
  runtimeAverage + policy.lambdaX1000PerByte * codeBytes

def linearScore (casesLen : Nat) (policy : DispatchPolicy) : Nat :=
  if casesLen = 0 then 0
  else planScore (linearAverageChecks casesLen)
    (expectedLinearCaseCodeBytes * casesLen) policy

def denseRuntimeChecks (plan : DensePlan) : Nat :=
  expectedTableDispatchOverheadChecksX1000 + expectedExactSelectorCheckX1000 +
    match plan.kind with
    | .bitWindow => 0
    | .multiplicative => expectedDenseMultiplicativeExtraChecksX1000

def denseCodeBytes (casesLen : Nat) (plan : DensePlan) : Nat :=
  let preamble := match plan.kind with
    | .bitWindow => expectedDenseBitWindowPreambleCodeBytes
    | .multiplicative => expectedDenseMultiplicativePreambleCodeBytes
  preamble + expectedJumpTableEntryBytes * plan.tableSlots +
    expectedDenseUsedSlotCodeBytes * casesLen

def denseScore (casesLen : Nat) (policy : DispatchPolicy) (plan : DensePlan) : Nat :=
  planScore (denseRuntimeChecks plan) (denseCodeBytes casesLen plan) policy

def sparseScore (selectors : List Nat) (policy : DispatchPolicy) (plan : SparsePlan) : Nat :=
  let counts := bucketCounts selectors plan.bucketBits plan.bucketShift
  let exactAverage := divRound
    (successfulScanChecks counts * expectedExactSelectorCheckX1000) selectors.length
  let runtimeAverage := exactAverage + expectedTableDispatchOverheadChecksX1000
  let codeBytes := expectedSparsePreambleCodeBytes +
    expectedJumpTableEntryBytes * plan.tableSlots +
    expectedSparseUsedBucketCodeBytes * usedBucketCount counts +
    expectedSparseCaseCodeBytes * selectors.length
  planScore runtimeAverage codeBytes policy

def sparseCandidates (selectors : List Nat) : List SparseCandidate :=
  sparseBucketBits.flatMap fun bits =>
    sparseBucketShifts.filterMap fun shift =>
      let counts := bucketCounts selectors bits shift
      if maxBucketSize counts < selectors.length then
        mkSparseCandidate?
          { tableSlots := pow2 bits, bucketBits := bits, bucketShift := shift }
      else
        none

def denseCandidateBetter (casesLen : Nat) (policy : DispatchPolicy)
    (candidate best : DensePlan) : Bool :=
  let candidateScore := denseScore casesLen policy candidate
  let bestScore := denseScore casesLen policy best
  if candidateScore != bestScore then decide (candidateScore < bestScore)
  else if candidate.tableSlots != best.tableSlots then
    decide (candidate.tableSlots < best.tableSlots)
  else
    let candidateLoad := divRound (casesLen * expectedCheckScaleX1000) candidate.tableSlots
    let bestLoad := divRound (casesLen * expectedCheckScaleX1000) best.tableSlots
    if candidateLoad != bestLoad then decide (candidateLoad > bestLoad)
    else if candidate.kind != best.kind then candidate.kind == .bitWindow
    else if candidate.indexShift != best.indexShift then
      decide (candidate.indexShift < best.indexShift)
    else if candidate.indexBits != best.indexBits then
      decide (candidate.indexBits < best.indexBits)
    else false

def sparseCandidateBetter (selectors : List Nat) (policy : DispatchPolicy)
    (candidate best : SparseCandidate) : Bool :=
  let candidateScore := sparseScore selectors policy candidate.plan
  let bestScore := sparseScore selectors policy best.plan
  let candidateCounts :=
    bucketCounts selectors candidate.plan.bucketBits candidate.plan.bucketShift
  let bestCounts := bucketCounts selectors best.plan.bucketBits best.plan.bucketShift
  if candidateScore != bestScore then decide (candidateScore < bestScore)
  else if maxBucketSize candidateCounts != maxBucketSize bestCounts then
    decide (maxBucketSize candidateCounts < maxBucketSize bestCounts)
  else if usedBucketCount candidateCounts != usedBucketCount bestCounts then
    decide (usedBucketCount candidateCounts < usedBucketCount bestCounts)
  else if candidate.plan.tableSlots != best.plan.tableSlots then
    decide (candidate.plan.tableSlots < best.plan.tableSlots)
  else if candidate.plan.bucketShift != best.plan.bucketShift then
    decide (candidate.plan.bucketShift < best.plan.bucketShift)
  else decide (candidate.plan.bucketBits < best.plan.bucketBits)

def bestDenseCandidate (cases : List (Selector × Label)) (policy : DispatchPolicy) :
    Option (DenseCandidate cases) :=
  (denseCandidates cases).foldl (fun best candidate =>
    match best with
    | none => some candidate
    | some current =>
        if denseCandidateBetter cases.length policy candidate.plan current.plan then
          some candidate
        else best) none

def bestSparseCandidate (selectors : List Nat) (policy : DispatchPolicy) :
    Option SparseCandidate :=
  (sparseCandidates selectors).foldl (fun best candidate =>
    match best with
    | none => some candidate
    | some current =>
        if sparseCandidateBetter selectors policy candidate current then
          some candidate
        else best) none

def savesSelectorChecks (linear candidate : Nat) : Bool :=
  decide (candidate + expectedMinSelectorCheckSavingX1000 <= linear)

def preconditionsMet (input : Input Label) : Bool :=
  decide (input.cases.length >= 4) && input.hasDefault &&
    (selectors input).all (fun selector => decide (selector < pow2 32))

def acceptedDense? (input : Input Label) : Option (DenseCandidate input.cases) :=
  let linear := linearScore input.cases.length input.policy
  match bestDenseCandidate input.cases input.policy with
  | some candidate =>
      if savesSelectorChecks linear (denseScore input.cases.length input.policy candidate.plan)
      then some candidate else none
  | none => none

def acceptedSparse? (input : Input Label) : Option SparseCandidate :=
  let sels := selectors input
  let linear := linearScore input.cases.length input.policy
  match bestSparseCandidate sels input.policy with
  | some candidate =>
      if savesSelectorChecks linear (sparseScore sels input.policy candidate.plan)
      then some candidate else none
  | none => none

/-- Relational statement of the production planner's branch order. -/
inductive PlannerChooses (input : Input Label) : Plan → Prop where
  | preconditionFailure (h : preconditionsMet input = false) : PlannerChooses input .linear
  | dense (hpre : preconditionsMet input = true) {candidate : DenseCandidate input.cases}
      (h : acceptedDense? input = some candidate) : PlannerChooses input (.dense candidate.plan)
  | sparse (hpre : preconditionsMet input = true) (hdense : acceptedDense? input = none)
      {candidate : SparseCandidate} (h : acceptedSparse? input = some candidate) :
      PlannerChooses input (.sparse candidate.plan)
  | linear (hpre : preconditionsMet input = true) (hdense : acceptedDense? input = none)
      (hsparse : acceptedSparse? input = none) : PlannerChooses input .linear

end Ora.DispatcherPlannerSpec

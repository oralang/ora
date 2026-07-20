/-
Planner-model recomputation and dispatcher-row semantic checks.

Shared arithmetic is imported from PlannerArithmetic so the planner spec and
snapshot checker execute one definition rather than bridge duplicate helpers.
-/

import Ora.Dispatcher
import Ora.Dispatcher.PlannerArithmetic
import Ora.Dispatcher.Decode
import Ora.SinoraPlanner
import Ora.Spec.DispatcherFacts

namespace Ora.DispatcherTableSync

open Ora.Dispatcher.PlannerArithmetic Ora.Generated Ora.Spec.DispatcherFacts

def casePair (c : RawCase) : Nat × String :=
  (c.selector, c.label)

def rowCasePairs (row : RawRow) : List (Nat × String) :=
  row.cases.map casePair

def noDuplicateBy (key : RawCase → Nat) : List RawCase → Bool
  | [] => true
  | c :: rest => !(rest.any (fun d => key d == key c)) && noDuplicateBy key rest

def allHasNamedDefault (row : RawRow) : Bool :=
  row.cases.all (·.hasNamedDefault)

def routeIndicesInRange (row : RawRow) : Bool :=
  decide (0 < row.plan.tableSlots) &&
    row.cases.all (fun c => decide (c.index < row.plan.tableSlots))

def linearRouteIndicesSequentialFrom : Nat → List RawCase → Bool
  | _, [] => true
  | index, c :: rest =>
      (c.index == index) && linearRouteIndicesSequentialFrom (index + 1) rest

def linearRouteIndicesSequential (row : RawRow) : Bool :=
  linearRouteIndicesSequentialFrom 0 row.cases

def indexForSelector (cases : List RawCase) (selector : Nat) : Nat :=
  match cases.find? (fun c => c.selector == selector) with
  | some c => c.index
  | none => 0

def tableAtIndex (cases : List RawCase) (index : Nat) : Option (Nat × String) :=
  match cases.find? (fun c => c.index == index) with
  | some c => some (casePair c)
  | none => none

def planShapeValid (plan : RawPlan) (casesLen : Nat) : Bool :=
  match plan.strategy, plan.denseKind with
  | "linear", "" =>
      decide (plan.tableSlots = casesLen) &&
        decide (plan.indexBits = 0) &&
        decide (plan.indexShift = 0) &&
        decide (plan.mulConstant = 0)
  | "sparse", "" =>
      decide (0 < plan.indexBits) &&
        decide (plan.indexBits <= 8) &&
        decide (plan.tableSlots = pow2 plan.indexBits) &&
        decide (plan.indexShift <= 24) &&
        decide (plan.mulConstant = 0)
  | "dense", "bit_window" =>
      decide (0 < plan.indexBits) &&
        decide (plan.indexBits <= 8) &&
        decide (plan.tableSlots = pow2 plan.indexBits) &&
        decide (plan.indexShift <= 24) &&
        decide (plan.mulConstant = 0)
  | "dense", "multiplicative" =>
      decide (0 < plan.indexBits) &&
        decide (plan.indexBits <= 8) &&
        decide (plan.tableSlots = pow2 plan.indexBits) &&
        decide (plan.indexShift = 32 - plan.indexBits) &&
        decide (0 < plan.mulConstant) &&
        decide (plan.mulConstant % 2 = 1)
  | _, _ => false

def rowPlanShapeValid (row : RawRow) : Bool :=
  planShapeValid row.plan row.cases.length

def planIndex (plan : RawPlan) (selector : Nat) : Nat :=
  match plan.strategy, plan.denseKind with
  | "dense", "multiplicative" =>
      multiplicativeIndex selector plan.mulConstant plan.indexBits
  | "dense", "bit_window" =>
      bitWindowIndex selector plan.indexBits plan.indexShift
  | "sparse", _ =>
      bitWindowIndex selector plan.indexBits plan.indexShift
  | _, _ => 0

def rowPlanIndex (row : RawRow) (selector : Nat) : Nat :=
  planIndex row.plan selector

def planBuilder? (plan : RawPlan) (casesLen : Nat) : Option (Ora.Dispatcher.BuilderPlan Nat) :=
  if planShapeValid plan casesLen then
    match plan.strategy, plan.denseKind with
    | "linear", "" => some .linear
    | "dense", "multiplicative" =>
        some (.dense (fun selector =>
          multiplicativeIndex selector plan.mulConstant plan.indexBits))
    | "dense", "bit_window" =>
        some (.dense (fun selector =>
          bitWindowIndex selector plan.indexBits plan.indexShift))
    | "sparse", "" =>
        some (.sparse (fun selector =>
          bitWindowIndex selector plan.indexBits plan.indexShift))
    | _, _ => none
  else
    none

def rowBuilderPlan? (row : RawRow) : Option (Ora.Dispatcher.BuilderPlan Nat) :=
  planBuilder? row.plan row.cases.length

def indexedDispatcher (row : RawRow) : Ora.Dispatcher.Dispatcher Nat Nat String :=
  { index := fun selector => indexForSelector row.cases selector,
    table := fun index => tableAtIndex row.cases index }

def indexedRunKnownOk (row : RawRow) : Bool :=
  row.cases.all fun c =>
    (indexedDispatcher row).run c.selector == some c.label

def indexedRunSoundOk (row : RawRow) : Bool :=
  let d := indexedDispatcher row
  row.cases.all fun c =>
    match d.run c.selector with
    | some label => d.table (d.index c.selector) == some (c.selector, label)
    | none => false

def linearRunKnownOk (row : RawRow) : Bool :=
  let table := row.cases.map casePair
  row.cases.all fun c =>
    Ora.Dispatcher.linearRun table c.selector == some c.label

def linearRunSoundOk (row : RawRow) : Bool :=
  let table := row.cases.map casePair
  row.cases.all fun c =>
    match Ora.Dispatcher.linearRun table c.selector with
    | some label => table.any (fun entry => entry == (c.selector, label))
    | none => false

def bucketEntries (cases : List RawCase) (index : Nat) : List (Nat × String) :=
  (cases.filter (fun c => c.index == index)).map casePair

def sparseDispatcher (row : RawRow) : Ora.Dispatcher.SparseDispatcher Nat Nat String :=
  { index := fun selector => indexForSelector row.cases selector,
    buckets := fun index => bucketEntries row.cases index }

def sparseRunKnownOk (row : RawRow) : Bool :=
  row.cases.all fun c =>
    (sparseDispatcher row).run c.selector == some c.label

def sparseRunSoundOk (row : RawRow) : Bool :=
  let d := sparseDispatcher row
  row.cases.all fun c =>
    match d.run c.selector with
    | some label => (d.buckets (d.index c.selector)).any
        (fun entry => entry == (c.selector, label))
    | none => false

def strategyKnown (row : RawRow) : Bool :=
  match row.plan.strategy with
  | "linear" => row.plan.denseKind == ""
  | "sparse" => row.plan.denseKind == ""
  | "dense" =>
      row.plan.denseKind == "bit_window" || row.plan.denseKind == "multiplicative"
  | _ => false

def denseInjectiveIfDense (row : RawRow) : Bool :=
  if row.plan.strategy == "dense" then noDuplicateBy (·.index) row.cases else true

def selectorsDistinct (row : RawRow) : Bool :=
  noDuplicateBy (·.selector) row.cases

def modelRunKnownOk (row : RawRow) : Bool :=
  match row.plan.strategy with
  | "linear" => linearRunKnownOk row
  | "sparse" => sparseRunKnownOk row
  | "dense" => indexedRunKnownOk row
  | _ => false

def modelRunSoundOk (row : RawRow) : Bool :=
  match row.plan.strategy with
  | "linear" => linearRunSoundOk row
  | "sparse" => sparseRunSoundOk row
  | "dense" => indexedRunSoundOk row
  | _ => false

def rowMatchesDispatcherModel (row : RawRow) : Bool :=
  strategyKnown row &&
    selectorsDistinct row &&
    routeIndicesInRange row &&
    rowPlanShapeValid row &&
    allHasNamedDefault row &&
    denseInjectiveIfDense row &&
    modelRunKnownOk row &&
    modelRunSoundOk row

def rowPlanAdmissible (row : RawRow) : Bool :=
  match rowBuilderPlan? row with
  | some plan => Ora.Dispatcher.PlanAdmissible (rowCasePairs row) plan
  | none => false

def rawPlanAdmissible (cases : List (Nat × String)) (casesLen : Nat) (plan : RawPlan) : Bool :=
  match planBuilder? plan casesLen with
  | some builderPlan => Ora.Dispatcher.PlanAdmissible cases builderPlan
  | none => false

def plannerPolicy? : String → Option Ora.DispatcherPlannerSpec.DispatchPolicy
  | "gas" => some .gas
  | "balanced" => some .balanced
  | "size" => some .size
  | _ => none

def encodePlannerPlan (casesLen : Nat) : Ora.DispatcherPlannerSpec.Plan → RawPlan
  | .linear =>
      { strategy := "linear", denseKind := "", tableSlots := casesLen,
        indexBits := 0, indexShift := 0, mulConstant := 0 }
  | .dense (.bitWindow tableSlots bits shift) =>
      { strategy := "dense", denseKind := "bit_window", tableSlots := tableSlots,
        indexBits := bits, indexShift := shift, mulConstant := 0 }
  | .dense (.multiplicative tableSlots bits shift constant) =>
      { strategy := "dense", denseKind := "multiplicative", tableSlots := tableSlots,
        indexBits := bits, indexShift := shift, mulConstant := constant }
  | .sparse plan =>
      { strategy := "sparse", denseKind := "", tableSlots := plan.tableSlots,
        indexBits := plan.bucketBits, indexShift := plan.bucketShift, mulConstant := 0 }

/- The reference match binds the emitted plan to the universally-correct Lean
   planner and is the authority for builder correctness. The full recomputation
   below independently validates scores, search order, and collision witnesses.
   Both paths are intentionally permanent defense in depth. -/
def rowPlannerReferenceMatches (row : RawRow) : Bool :=
  match plannerPolicy? row.trace.policy with
  | none => false
  | some policy =>
      let input : Ora.DispatcherPlannerSpec.Input String :=
        { cases := rowCasePairs row,
          hasDefault := allHasNamedDefault row,
          policy := policy }
      encodePlannerPlan input.cases.length (Ora.SinoraPlanner.choosePlanReference input) ==
        row.plan

section EncodeBridge

/- One shared simp set for all three strategy branches of the encode bridge. -/
attribute [local simp]
  encodePlannerPlan planBuilder? planShapeValid
  Ora.DispatcherPlannerSpec.Plan.toBuilderPlan
  Ora.DispatcherPlannerSpec.Plan.wellShaped
  Ora.DispatcherPlannerSpec.DensePlan.wellShaped
  Ora.DispatcherPlannerSpec.SparsePlan.wellShaped
  Ora.DispatcherPlannerSpec.DensePlan.index
  Ora.DispatcherPlannerSpec.SparsePlan.index

theorem planBuilder_encodePlannerPlan
    (casesLen : Nat) (plan : Ora.DispatcherPlannerSpec.Plan)
    (hshape : Ora.DispatcherPlannerSpec.Plan.wellShaped plan = true) :
    planBuilder? (encodePlannerPlan casesLen plan) casesLen = some plan.toBuilderPlan := by
  cases plan with
  | linear => simp
  | dense dense => cases dense <;> simp_all
  | sparse sparse => cases sparse; simp_all; rfl

end EncodeBridge

def linearRawPlan (casesLen : Nat) : RawPlan :=
  { strategy := "linear", denseKind := "", tableSlots := casesLen,
    indexBits := 0, indexShift := 0, mulConstant := 0 }

def sparseBucketBits : List Nat := [1, 2, 3, 4, 5, 6, 7, 8]

def sparseBucketShifts : List Nat := [0, 4, 8, 12, 16, 20, 24]

def u32Modulus : Nat := pow2 32

def emittedMultiplicativeCandidate (index : Nat) : Nat :=
  compilerDispatcherMultiplicativeCandidates[index]?.getD 0

def denseBitWindowPlan? (selectors : List Nat) (bits shift : Nat) : Option RawPlan :=
  if collisionFree selectors (fun selector => bitWindowIndex selector bits shift) then
    some
      { strategy := "dense", denseKind := "bit_window", tableSlots := pow2 bits,
        indexBits := bits, indexShift := shift, mulConstant := 0 }
  else
    none

def collisionWitnessValid (selectors : List Nat) (bits : Nat)
    (witness : RawCollisionWitness) : Bool :=
  match selectors.get? witness.firstCase, selectors.get? witness.secondCase with
  | some first, some second =>
      decide (witness.firstCase < witness.secondCase) &&
        decide (multiplicativeIndex first witness.constant bits =
          multiplicativeIndex second witness.constant bits)
  | _, _ => false

def rejectedWitnessesValid (selectors : List Nat) (bits : Nat) : List RawCollisionWitness → Bool
  | [] => true
  | witness :: rest =>
      collisionWitnessValid selectors bits witness && rejectedWitnessesValid selectors bits rest

def rejectedConstantsValid (rejected : List RawCollisionWitness) : Bool :=
  rejected.map (·.constant) ==
    compilerDispatcherMultiplicativeCandidates.take rejected.length

def multiplicativeSearchValid (selectors : List Nat) (search : RawMultiplicativeSearch) : Bool :=
  match indexBitsForSlots? search.tableSlots with
  | none => false
  | some bits =>
      rejectedConstantsValid search.rejected &&
        rejectedWitnessesValid selectors bits search.rejected &&
        match search.selected with
        | some candidateIndex =>
            decide (candidateIndex < expectedMultiplicativeSearchBudget) &&
              decide (search.rejected.length = candidateIndex) &&
              collisionFree selectors (fun selector =>
                multiplicativeIndex selector (emittedMultiplicativeCandidate candidateIndex) bits)
        | none => decide (search.rejected.length = expectedMultiplicativeSearchBudget)

def multiplicativeSearchesValid (selectors : List Nat)
    (searches : List RawMultiplicativeSearch) : Bool :=
  searches.map (·.tableSlots) == multiplicativeTableSlots selectors.length &&
    searches.all (multiplicativeSearchValid selectors)

def policyLambda? : String → Option Nat
  | policy =>
      (expectedDispatchPolicyLambdasX1000.find? fun entry => entry.1 == policy).map Prod.snd

def planScore (runtimeAvgChecks codeBytes policyLambda : Nat) : Nat :=
  runtimeAvgChecks + policyLambda * codeBytes

def linearScore (casesLen policyLambda : Nat) : Nat :=
  if casesLen = 0 then 0
  else
    planScore
      (linearAverageChecks casesLen)
      (expectedLinearCaseCodeBytes * casesLen)
      policyLambda

def denseRuntimeChecks (plan : RawPlan) : Nat :=
  expectedTableDispatchOverheadChecksX1000 + expectedExactSelectorCheckX1000 +
    if plan.denseKind == "multiplicative" then
      expectedDenseMultiplicativeExtraChecksX1000
    else 0

def denseCodeBytes (plan : RawPlan) (casesLen : Nat) : Nat :=
  let preamble :=
    if plan.denseKind == "multiplicative" then
      expectedDenseMultiplicativePreambleCodeBytes
    else expectedDenseBitWindowPreambleCodeBytes
  preamble + expectedJumpTableEntryBytes * plan.tableSlots +
    expectedDenseUsedSlotCodeBytes * casesLen

def denseScore (plan : RawPlan) (casesLen policyLambda : Nat) : Nat :=
  planScore (denseRuntimeChecks plan) (denseCodeBytes plan casesLen) policyLambda

def multiplicativePlans (selectors : List Nat) (policyLambda : Nat)
    (searches : List RawMultiplicativeSearch) : List RawScoredPlan :=
  searches.filterMap fun search =>
    search.selected.map fun candidateIndex =>
      let bits := (indexBitsForSlots? search.tableSlots).getD 0
      let plan : RawPlan :=
        { strategy := "dense", denseKind := "multiplicative",
          tableSlots := search.tableSlots,
          indexBits := bits, indexShift := 32 - bits,
          mulConstant := emittedMultiplicativeCandidate candidateIndex }
      (plan, denseScore plan selectors.length policyLambda)

def sparseScoreFromCounts
    (selectors counts : List Nat) (plan : RawPlan) (policyLambda : Nat) : Nat :=
  let usedBuckets := usedBucketCount counts
  let exactAverage :=
    divRound
      (successfulScanChecks counts * expectedExactSelectorCheckX1000)
      selectors.length
  let runtimeAverage := exactAverage + expectedTableDispatchOverheadChecksX1000
  let codeBytes := expectedSparsePreambleCodeBytes +
    expectedJumpTableEntryBytes * plan.tableSlots +
    expectedSparseUsedBucketCodeBytes * usedBuckets +
    expectedSparseCaseCodeBytes * selectors.length
  planScore runtimeAverage codeBytes policyLambda

def sparseScore (selectors : List Nat) (plan : RawPlan) (policyLambda : Nat) : Nat :=
  sparseScoreFromCounts selectors
    (bucketCounts selectors plan.indexBits plan.indexShift) plan policyLambda

def denseCandidates (selectors : List Nat) (policyLambda : Nat)
    (searches : List RawMultiplicativeSearch) : List RawScoredPlan :=
  let bitWindow := sparseBucketBits.flatMap fun bits =>
    sparseBucketShifts.filterMap fun shift =>
      (denseBitWindowPlan? selectors bits shift).map fun plan =>
        (plan, denseScore plan selectors.length policyLambda)
  bitWindow ++ multiplicativePlans selectors policyLambda searches

def sparseCandidates (selectors : List Nat) (policyLambda : Nat) : List RawScoredPlan :=
  sparseBucketBits.flatMap fun bits =>
    sparseBucketShifts.filterMap fun shift =>
      let counts := bucketCounts selectors bits shift
      if maxBucketSize counts < selectors.length then
        let plan : RawPlan :=
          { strategy := "sparse", denseKind := "", tableSlots := pow2 bits,
            indexBits := bits, indexShift := shift, mulConstant := 0 }
        some (plan, sparseScoreFromCounts selectors counts plan policyLambda)
      else
        none

def denseCandidateBetter (casesLen : Nat) (candidate best : RawScoredPlan) : Bool :=
  if candidate.2 != best.2 then decide (candidate.2 < best.2)
  else if candidate.1.tableSlots != best.1.tableSlots then
    decide (candidate.1.tableSlots < best.1.tableSlots)
  else
    let candidateLoad :=
      divRound (casesLen * expectedCheckScaleX1000) candidate.1.tableSlots
    let bestLoad :=
      divRound (casesLen * expectedCheckScaleX1000) best.1.tableSlots
    if candidateLoad != bestLoad then decide (candidateLoad > bestLoad)
    else if candidate.1.denseKind != best.1.denseKind then
      candidate.1.denseKind == "bit_window"
    else if candidate.1.indexShift != best.1.indexShift then
      decide (candidate.1.indexShift < best.1.indexShift)
    else if candidate.1.indexBits != best.1.indexBits then
      decide (candidate.1.indexBits < best.1.indexBits)
    else false

def sparseCandidateBetter (selectors : List Nat) (candidate best : RawScoredPlan) : Bool :=
  let candidateCounts :=
    bucketCounts selectors candidate.1.indexBits candidate.1.indexShift
  let bestCounts := bucketCounts selectors best.1.indexBits best.1.indexShift
  if candidate.2 != best.2 then decide (candidate.2 < best.2)
  else if maxBucketSize candidateCounts != maxBucketSize bestCounts then
    decide (maxBucketSize candidateCounts < maxBucketSize bestCounts)
  else if usedBucketCount candidateCounts != usedBucketCount bestCounts then
    decide (usedBucketCount candidateCounts < usedBucketCount bestCounts)
  else if candidate.1.tableSlots != best.1.tableSlots then
    decide (candidate.1.tableSlots < best.1.tableSlots)
  else if candidate.1.indexShift != best.1.indexShift then
    decide (candidate.1.indexShift < best.1.indexShift)
  else decide (candidate.1.indexBits < best.1.indexBits)

structure RankedSparseCandidate where
  candidate : RawScoredPlan
  maxBucketSize : Nat
  usedBucketCount : Nat

def rankSparseCandidate
    (selectors : List Nat) (candidate : RawScoredPlan) : RankedSparseCandidate :=
  let counts := bucketCounts selectors candidate.1.indexBits candidate.1.indexShift
  { candidate
    maxBucketSize := maxBucketSize counts
    usedBucketCount := usedBucketCount counts }

def rankedSparseCandidateBetter
    (candidate best : RankedSparseCandidate) : Bool :=
  if candidate.candidate.2 != best.candidate.2 then
    decide (candidate.candidate.2 < best.candidate.2)
  else if candidate.maxBucketSize != best.maxBucketSize then
    decide (candidate.maxBucketSize < best.maxBucketSize)
  else if candidate.usedBucketCount != best.usedBucketCount then
    decide (candidate.usedBucketCount < best.usedBucketCount)
  else if candidate.candidate.1.tableSlots != best.candidate.1.tableSlots then
    decide (candidate.candidate.1.tableSlots < best.candidate.1.tableSlots)
  else if candidate.candidate.1.indexShift != best.candidate.1.indexShift then
    decide (candidate.candidate.1.indexShift < best.candidate.1.indexShift)
  else decide (candidate.candidate.1.indexBits < best.candidate.1.indexBits)

def bestDenseCandidate (casesLen : Nat) (candidates : List RawScoredPlan) : Option RawScoredPlan :=
  candidates.foldl (fun best candidate =>
    match best with
    | none => some candidate
    | some current =>
        if denseCandidateBetter casesLen candidate current then some candidate else best) none

def bestSparseCandidate
    (selectors : List Nat)
    (candidates : List RawScoredPlan) : Option RawScoredPlan :=
  ((candidates.map (rankSparseCandidate selectors)).foldl (fun best candidate =>
    match best with
    | none => some candidate
    | some current =>
        if rankedSparseCandidateBetter candidate current then some candidate else best) none).map
          (·.candidate)

def savesSelectorChecks (linear candidate : Nat) : Bool :=
  decide (candidate + expectedMinSelectorCheckSavingX1000 <= linear)

def chooseScoredPlan (casesLen linear : Nat)
    (bestDense bestSparse : Option RawScoredPlan) : RawPlan :=
  match bestDense with
  | some dense =>
      if savesSelectorChecks linear dense.2 then dense.1
      else
        match bestSparse with
        | some sparse =>
            if savesSelectorChecks linear sparse.2 then sparse.1
            else linearRawPlan casesLen
        | none => linearRawPlan casesLen
  | none =>
      match bestSparse with
      | some sparse =>
          if savesSelectorChecks linear sparse.2 then sparse.1
          else linearRawPlan casesLen
      | none => linearRawPlan casesLen

def plannerPreconditionsMet (selectors : List Nat) (hasDefault : Bool) : Bool :=
  decide (selectors.length >= 4) && hasDefault &&
    selectors.all (fun selector => decide (selector < u32Modulus))

def recomputePlannerCore? (row : RawRow) : Option (RawPlannerTrace × RawPlan) :=
  let selectors := row.cases.map (·.selector)
  let casesLen := selectors.length
  let policy := row.trace.policy
  match policyLambda? policy with
  | none => none
  | some policyLambda =>
      let preconditions := plannerPreconditionsMet selectors (allHasNamedDefault row)
      let linear := linearScore casesLen policyLambda
      let searches := row.trace.multiplicativeSearches
      if preconditions then
        let dense := denseCandidates selectors policyLambda searches
        let bestDense := bestDenseCandidate casesLen dense
        let sparse := sparseCandidates selectors policyLambda
        let bestSparse := bestSparseCandidate selectors sparse
        let chosen := chooseScoredPlan casesLen linear bestDense bestSparse
        let trace : RawPlannerTrace :=
          { policy := policy, preconditionsMet := preconditions,
            linearScore := linear, multiplicativeSearches := searches,
            denseCandidateCount := dense.length, bestDense := bestDense,
            sparseCandidateCount := sparse.length, bestSparse := bestSparse }
        some (trace, chosen)
      else
        let trace : RawPlannerTrace :=
          { policy := policy, preconditionsMet := preconditions,
            linearScore := linear, multiplicativeSearches := [],
            denseCandidateCount := 0, bestDense := none,
            sparseCandidateCount := 0, bestSparse := none }
        some (trace, linearRawPlan casesLen)

def rowMultiplicativeSearchesValid (row : RawRow) : Bool :=
  let selectors := row.cases.map (·.selector)
  if plannerPreconditionsMet selectors (allHasNamedDefault row) then
    multiplicativeSearchesValid selectors row.trace.multiplicativeSearches
  else row.trace.multiplicativeSearches.isEmpty

structure PlannerCoreChecks where
  policy : Bool
  preconditions : Bool
  linearScore : Bool
  denseCandidateCount : Bool
  bestDense : Bool
  sparseCandidateCount : Bool
  bestSparse : Bool
  plan : Bool

def PlannerCoreChecks.all (checks : PlannerCoreChecks) : Bool :=
  checks.policy && checks.preconditions && checks.linearScore &&
    checks.denseCandidateCount && checks.bestDense &&
    checks.sparseCandidateCount && checks.bestSparse && checks.plan

theorem PlannerCoreChecks.all_of_fields (checks : PlannerCoreChecks)
    (hpolicy : checks.policy = true)
    (hpreconditions : checks.preconditions = true)
    (hlinearScore : checks.linearScore = true)
    (hdenseCandidateCount : checks.denseCandidateCount = true)
    (hbestDense : checks.bestDense = true)
    (hsparseCandidateCount : checks.sparseCandidateCount = true)
    (hbestSparse : checks.bestSparse = true)
    (hplan : checks.plan = true) :
    checks.all = true := by
  simp [PlannerCoreChecks.all, hpolicy, hpreconditions, hlinearScore,
    hdenseCandidateCount, hbestDense, hsparseCandidateCount, hbestSparse, hplan]

def plannerCoreChecks (row : RawRow) : PlannerCoreChecks :=
  match recomputePlannerCore? row with
  | some (computed, plan) =>
      { policy := computed.policy == row.trace.policy
        preconditions := computed.preconditionsMet == row.trace.preconditionsMet
        linearScore := computed.linearScore == row.trace.linearScore
        denseCandidateCount := computed.denseCandidateCount == row.trace.denseCandidateCount
        bestDense := computed.bestDense == row.trace.bestDense
        sparseCandidateCount := computed.sparseCandidateCount == row.trace.sparseCandidateCount
        bestSparse := computed.bestSparse == row.trace.bestSparse
        plan := plan == row.plan }
  | none =>
      { policy := false, preconditions := false, linearScore := false
        denseCandidateCount := false, bestDense := false
        sparseCandidateCount := false, bestSparse := false, plan := false }

def rowPlannerCoreMatches (row : RawRow) : Bool :=
  (plannerCoreChecks row).all

def rowPlannerChosen? (row : RawRow) : Option RawPlan :=
  (recomputePlannerCore? row).map fun result => result.2

def rowPlannerMatches (row : RawRow) : Bool :=
  rowMultiplicativeSearchesValid row && rowPlannerCoreMatches row

def rowPlanIndicesMatch (row : RawRow) : Bool :=
  rowPlanShapeValid row &&
    match row.plan.strategy with
    | "linear" => linearRouteIndicesSequential row
    | "dense" =>
        row.cases.all (fun c => c.index == rowPlanIndex row c.selector)
    | "sparse" =>
        row.cases.all (fun c => c.index == rowPlanIndex row c.selector)
    | _ => false

end Ora.DispatcherTableSync

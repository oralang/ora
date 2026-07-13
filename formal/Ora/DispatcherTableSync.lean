/-
Trusted sync check for compiler-emitted dispatcher table facts.

`Ora/Generated/DispatcherTableSnapshot.lean` is data-only: each row is emitted
from actual Ora source compiled through `oraBuildSIRDispatcher`, then classified
with Sinora's real switch planner/index functions. This module decodes those
rows and checks that they satisfy the premises used by `Ora.Dispatcher`:

* known selectors resolve in the corresponding abstract dispatcher model;
* dense rows have injective route indices over known selectors;
* sparse rows resolve through bucket scan by exact selector equality;
* every switch has a named default destination (`guarded = true` in the
  raw snapshot schema);
* Lean independently regenerates the complete scored candidate search,
  including all 512 multiplicative constants per table size, and checks the
  emitted trace and chosen plan byte-for-byte as data.

This is representative fixture coverage for the repository snapshot and a
per-contract userland gate for every row emitted by `ora --lean-proofs`. It
proves the concrete contract's chosen plan equals an independent Lean execution
of the bounded planner model. It does not prove the Zig source implementation
equivalent for inputs that were never compiled through this gate.
-/

import Ora.Dispatcher
import Ora.Generated.DispatcherTableSnapshot
import Ora.SinoraPlanner
import Ora.Spec.DispatcherFacts

namespace Ora.DispatcherTableSync

open Ora.Generated Ora.Spec.DispatcherFacts

abbrev RawCase := Nat × String × Nat × Bool
abbrev GeneratedRawPlan := String × String × Nat × Nat × Nat × Nat
abbrev GeneratedRawScoredPlan := GeneratedRawPlan × Nat
abbrev GeneratedRawPlannerTrace :=
  String × Bool × Nat × List (Nat × Option Nat × List (Nat × Nat × Nat)) ×
    Nat × Option GeneratedRawScoredPlan × Nat × Option GeneratedRawScoredPlan
abbrev GeneratedRawRow := String × GeneratedRawPlan × GeneratedRawPlannerTrace × List RawCase

structure RawPlan where
  strategy : String
  denseKind : String
  tableSlots : Nat
  indexBits : Nat
  indexShift : Nat
  mulConstant : Nat
  deriving Repr, DecidableEq

abbrev RawScoredPlan := RawPlan × Nat
abbrev RawCollisionWitness := Nat × Nat × Nat
abbrev RawMultiplicativeSearch := Nat × Option Nat × List RawCollisionWitness

structure RawPlannerTrace where
  policy : String
  preconditionsMet : Bool
  linearScore : Nat
  multiplicativeSearches : List RawMultiplicativeSearch
  denseCandidateCount : Nat
  bestDense : Option RawScoredPlan
  sparseCandidateCount : Nat
  bestSparse : Option RawScoredPlan
  deriving Repr, DecidableEq

structure RawRow where
  name : String
  plan : RawPlan
  trace : RawPlannerTrace
  cases : List RawCase
  deriving Repr, DecidableEq

def decodeGeneratedPlan : GeneratedRawPlan → RawPlan
  | (strategy, denseKind, tableSlots, indexBits, indexShift, mulConstant) =>
      { strategy, denseKind, tableSlots, indexBits, indexShift, mulConstant }

def decodeGeneratedScoredPlan : GeneratedRawScoredPlan → RawScoredPlan
  | (plan, score) => (decodeGeneratedPlan plan, score)

def decodeGeneratedPlannerTrace : GeneratedRawPlannerTrace → RawPlannerTrace
  | (policy, preconditionsMet, linearScore, multiplicativeSearches,
      denseCandidateCount, bestDense, sparseCandidateCount, bestSparse) =>
      { policy, preconditionsMet, linearScore, multiplicativeSearches,
        denseCandidateCount, bestDense := bestDense.map decodeGeneratedScoredPlan,
        sparseCandidateCount, bestSparse := bestSparse.map decodeGeneratedScoredPlan }

def decodeGeneratedRow : GeneratedRawRow → RawRow
  | (name, plan, trace, cases) =>
      { name, plan := decodeGeneratedPlan plan,
        trace := decodeGeneratedPlannerTrace trace, cases }

def compilerDispatcherTableRows : List RawRow :=
  Ora.Generated.compilerDispatcherTableRows.map decodeGeneratedRow

abbrev RawIntent := Nat × String
abbrev RawNetworkSwitch := String × String × List (Nat × String)

def intentSelector : RawIntent → Nat := Prod.fst
def intentLabel : RawIntent → String := Prod.snd

def networkSwitchBlock : RawNetworkSwitch → String
  | (block, _, _) => block

def networkSwitchDefault : RawNetworkSwitch → String
  | (_, fallback, _) => fallback

def networkSwitchCases : RawNetworkSwitch → List (Nat × String)
  | (_, _, cases) => cases

def dispatchNetwork : Nat → List RawNetworkSwitch → String → Nat → String
  | 0, _, block, _ => block
  | fuel + 1, switches, block, selector =>
      match switches.find? (fun sw => networkSwitchBlock sw == block) with
      | none => block
      | some sw =>
          match (networkSwitchCases sw).find? (fun c => c.1 == selector) with
          | some c => c.2
          | none => dispatchNetwork fuel switches (networkSwitchDefault sw) selector

def networkNoDuplicateNat : List Nat → Bool
  | [] => true
  | value :: rest => !(rest.contains value) && networkNoDuplicateNat rest

def networkSelectorsDistinct (intents : List RawIntent) : Bool :=
  networkNoDuplicateNat (intents.map intentSelector)

def networkNoDuplicateString : List String → Bool
  | [] => true
  | value :: rest => !(rest.contains value) && networkNoDuplicateString rest

def networkBlocksDistinct (switches : List RawNetworkSwitch) : Bool :=
  networkNoDuplicateString (switches.map networkSwitchBlock)

def networkCasesAuthorized
    (intents : List RawIntent) (switches : List RawNetworkSwitch) : Bool :=
  switches.all fun sw =>
    (networkSwitchCases sw).all fun c => intents.contains c

def networkIntentsPresentExactlyOnce
    (intents : List RawIntent) (switches : List RawNetworkSwitch) : Bool :=
  intents.all fun intent =>
    decide ((switches.foldl (fun count sw =>
      count + (networkSwitchCases sw).count intent) 0) = 1)

def networkDefaultsKnown (switches : List RawNetworkSwitch) : Bool :=
  switches.all fun sw =>
    networkSwitchDefault sw == "revert_error" ||
      switches.any (fun target =>
        networkSwitchBlock target == networkSwitchDefault sw)

def networkKnownSelectorsReachIntent
    (intents : List RawIntent)
    (switches : List RawNetworkSwitch)
    (entry : String) : Bool :=
  intents.all fun intent =>
    dispatchNetwork (switches.length + 1) switches entry (intentSelector intent) ==
      intentLabel intent

def networkDefaultReachesRevert
    (switches : List RawNetworkSwitch) (entry : String) : Bool :=
  dispatchNetwork (switches.length + 1) switches entry 4294967296 ==
    "revert_error"

def dispatcherNetworkMatches
    (intents : List RawIntent)
    (switches : List RawNetworkSwitch)
    (entry : String) : Bool :=
  networkSelectorsDistinct intents &&
    networkBlocksDistinct switches &&
    networkCasesAuthorized intents switches &&
    networkIntentsPresentExactlyOnce intents switches &&
    networkDefaultsKnown switches &&
    networkKnownSelectorsReachIntent intents switches entry &&
    networkDefaultReachesRevert switches entry

def caseSelector : RawCase → Nat
  | (selector, _, _, _) => selector

def caseLabel : RawCase → String
  | (_, label, _, _) => label

def caseIndex : RawCase → Nat
  | (_, _, index, _) => index

def caseHasNamedDefault : RawCase → Bool
  | (_, _, _, hasNamedDefault) => hasNamedDefault

def rowName (row : RawRow) : String := row.name

def planStrategy (plan : RawPlan) : String := plan.strategy

def planDenseKind (plan : RawPlan) : String := plan.denseKind

def planTableSlots (plan : RawPlan) : Nat := plan.tableSlots

def planIndexBits (plan : RawPlan) : Nat := plan.indexBits

def planIndexShift (plan : RawPlan) : Nat := plan.indexShift

def planMulConstant (plan : RawPlan) : Nat := plan.mulConstant

def rowPlan (row : RawRow) : RawPlan := row.plan

def rowStrategy : RawRow → String
  | row => planStrategy (rowPlan row)

def rowDenseKind : RawRow → String
  | row => planDenseKind (rowPlan row)

def rowTableSlots : RawRow → Nat
  | row => planTableSlots (rowPlan row)

def rowIndexBits : RawRow → Nat
  | row => planIndexBits (rowPlan row)

def rowIndexShift : RawRow → Nat
  | row => planIndexShift (rowPlan row)

def rowMulConstant : RawRow → Nat
  | row => planMulConstant (rowPlan row)

def rowTrace (row : RawRow) : RawPlannerTrace := row.trace

def tracePolicy (trace : RawPlannerTrace) : String := trace.policy

def tracePreconditionsMet (trace : RawPlannerTrace) : Bool := trace.preconditionsMet

def traceLinearScore (trace : RawPlannerTrace) : Nat := trace.linearScore

def traceMultiplicativeSearches (trace : RawPlannerTrace) : List RawMultiplicativeSearch :=
  trace.multiplicativeSearches

def traceDenseCandidateCount (trace : RawPlannerTrace) : Nat := trace.denseCandidateCount

def traceBestDense (trace : RawPlannerTrace) : Option RawScoredPlan := trace.bestDense

def traceSparseCandidateCount (trace : RawPlannerTrace) : Nat := trace.sparseCandidateCount

def traceBestSparse (trace : RawPlannerTrace) : Option RawScoredPlan := trace.bestSparse

def rowCases (row : RawRow) : List RawCase := row.cases

def casePair (c : RawCase) : Nat × String :=
  (caseSelector c, caseLabel c)

def rowCasePairs (row : RawRow) : List (Nat × String) :=
  (rowCases row).map casePair

def noDuplicateBy (key : RawCase → Nat) : List RawCase → Bool
  | [] => true
  | c :: rest => !(rest.any (fun d => key d == key c)) && noDuplicateBy key rest

def allHasNamedDefault (row : RawRow) : Bool :=
  (rowCases row).all caseHasNamedDefault

def routeIndicesInRange (row : RawRow) : Bool :=
  decide (0 < rowTableSlots row) &&
    (rowCases row).all (fun c => decide (caseIndex c < rowTableSlots row))

def pow2 (bits : Nat) : Nat :=
  2 ^ bits

def indexedShapeValid (row : RawRow) : Bool :=
  decide (0 < rowIndexBits row) &&
    decide (rowIndexBits row <= 8) &&
    decide (rowTableSlots row = pow2 (rowIndexBits row))

def linearRouteIndicesSequentialFrom : Nat → List RawCase → Bool
  | _, [] => true
  | index, c :: rest =>
      (caseIndex c == index) && linearRouteIndicesSequentialFrom (index + 1) rest

def linearRouteIndicesSequential (row : RawRow) : Bool :=
  linearRouteIndicesSequentialFrom 0 (rowCases row)

def selectorKnown (cases : List RawCase) (selector : Nat) : Bool :=
  cases.any (fun c => caseSelector c == selector)

def indexForSelector (cases : List RawCase) (selector : Nat) : Nat :=
  match cases.find? (fun c => caseSelector c == selector) with
  | some c => caseIndex c
  | none => 0

def tableAtIndex (cases : List RawCase) (index : Nat) : Option (Nat × String) :=
  match cases.find? (fun c => caseIndex c == index) with
  | some c => some (casePair c)
  | none => none

def bitWindowIndex (selector bits shift : Nat) : Nat :=
  (selector / pow2 shift) % pow2 bits

def multiplicativeIndex (selector constant bits : Nat) : Nat :=
  (selector * constant / pow2 (32 - bits)) % pow2 bits

def planShapeValid (plan : RawPlan) (casesLen : Nat) : Bool :=
  match planStrategy plan, planDenseKind plan with
  | "linear", "" =>
      decide (planTableSlots plan = casesLen) &&
        decide (planIndexBits plan = 0) &&
        decide (planIndexShift plan = 0) &&
        decide (planMulConstant plan = 0)
  | "sparse", "" =>
      decide (0 < planIndexBits plan) &&
        decide (planIndexBits plan <= 8) &&
        decide (planTableSlots plan = pow2 (planIndexBits plan)) &&
        decide (planIndexShift plan <= 24) &&
        decide (planMulConstant plan = 0)
  | "dense", "bit_window" =>
      decide (0 < planIndexBits plan) &&
        decide (planIndexBits plan <= 8) &&
        decide (planTableSlots plan = pow2 (planIndexBits plan)) &&
        decide (planIndexShift plan <= 24) &&
        decide (planMulConstant plan = 0)
  | "dense", "multiplicative" =>
      decide (0 < planIndexBits plan) &&
        decide (planIndexBits plan <= 8) &&
        decide (planTableSlots plan = pow2 (planIndexBits plan)) &&
        decide (planIndexShift plan = 32 - planIndexBits plan) &&
        decide (0 < planMulConstant plan) &&
        decide (planMulConstant plan % 2 = 1)
  | _, _ => false

def rowPlanShapeValid (row : RawRow) : Bool :=
  planShapeValid (rowPlan row) (rowCases row).length

def planIndex (plan : RawPlan) (selector : Nat) : Nat :=
  match planStrategy plan, planDenseKind plan with
  | "dense", "multiplicative" =>
      multiplicativeIndex selector (planMulConstant plan) (planIndexBits plan)
  | "dense", "bit_window" =>
      bitWindowIndex selector (planIndexBits plan) (planIndexShift plan)
  | "sparse", _ =>
      bitWindowIndex selector (planIndexBits plan) (planIndexShift plan)
  | _, _ => 0

def rowPlanIndex (row : RawRow) (selector : Nat) : Nat :=
  planIndex (rowPlan row) selector

def planBuilder? (plan : RawPlan) (casesLen : Nat) : Option (Ora.Dispatcher.BuilderPlan Nat) :=
  if planShapeValid plan casesLen then
    match planStrategy plan, planDenseKind plan with
    | "linear", "" => some .linear
    | "dense", "multiplicative" =>
        some (.dense (fun selector =>
          multiplicativeIndex selector (planMulConstant plan) (planIndexBits plan)))
    | "dense", "bit_window" =>
        some (.dense (fun selector =>
          bitWindowIndex selector (planIndexBits plan) (planIndexShift plan)))
    | "sparse", "" =>
        some (.sparse (fun selector =>
          bitWindowIndex selector (planIndexBits plan) (planIndexShift plan)))
    | _, _ => none
  else
    none

def rowBuilderPlan? (row : RawRow) : Option (Ora.Dispatcher.BuilderPlan Nat) :=
  planBuilder? (rowPlan row) (rowCases row).length

def indexedDispatcher (row : RawRow) : Ora.Dispatcher.Dispatcher Nat Nat String :=
  let cases := rowCases row
  { index := fun selector => indexForSelector cases selector,
    table := fun index => tableAtIndex cases index }

def indexedRunKnownOk (row : RawRow) : Bool :=
  (rowCases row).all fun c =>
    (indexedDispatcher row).run (caseSelector c) == some (caseLabel c)

def indexedRunSoundOk (row : RawRow) : Bool :=
  let d := indexedDispatcher row
  (rowCases row).all fun c =>
    match d.run (caseSelector c) with
    | some label => d.table (d.index (caseSelector c)) == some (caseSelector c, label)
    | none => false

def linearRunKnownOk (row : RawRow) : Bool :=
  let table := (rowCases row).map casePair
  (rowCases row).all fun c =>
    Ora.Dispatcher.linearRun table (caseSelector c) == some (caseLabel c)

def linearRunSoundOk (row : RawRow) : Bool :=
  let table := (rowCases row).map casePair
  (rowCases row).all fun c =>
    match Ora.Dispatcher.linearRun table (caseSelector c) with
    | some label => table.any (fun entry => entry == (caseSelector c, label))
    | none => false

def bucketEntries (cases : List RawCase) (index : Nat) : List (Nat × String) :=
  (cases.filter (fun c => caseIndex c == index)).map casePair

def sparseDispatcher (row : RawRow) : Ora.Dispatcher.SparseDispatcher Nat Nat String :=
  let cases := rowCases row
  { index := fun selector => indexForSelector cases selector,
    buckets := fun index => bucketEntries cases index }

def sparseRunKnownOk (row : RawRow) : Bool :=
  (rowCases row).all fun c =>
    (sparseDispatcher row).run (caseSelector c) == some (caseLabel c)

def sparseRunSoundOk (row : RawRow) : Bool :=
  let d := sparseDispatcher row
  (rowCases row).all fun c =>
    match d.run (caseSelector c) with
    | some label => (d.buckets (d.index (caseSelector c))).any
        (fun entry => entry == (caseSelector c, label))
    | none => false

def strategyKnown (row : RawRow) : Bool :=
  match rowStrategy row with
  | "linear" => rowDenseKind row == ""
  | "sparse" => rowDenseKind row == ""
  | "dense" => rowDenseKind row == "bit_window" || rowDenseKind row == "multiplicative"
  | _ => false

def denseInjectiveIfDense (row : RawRow) : Bool :=
  if rowStrategy row == "dense" then noDuplicateBy caseIndex (rowCases row) else true

def selectorsDistinct (row : RawRow) : Bool :=
  noDuplicateBy caseSelector (rowCases row)

def modelRunKnownOk (row : RawRow) : Bool :=
  match rowStrategy row with
  | "linear" => linearRunKnownOk row
  | "sparse" => sparseRunKnownOk row
  | "dense" => indexedRunKnownOk row
  | _ => false

def modelRunSoundOk (row : RawRow) : Bool :=
  match rowStrategy row with
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
  match plannerPolicy? (tracePolicy (rowTrace row)) with
  | none => false
  | some policy =>
      let input : Ora.DispatcherPlannerSpec.Input String :=
        { cases := rowCasePairs row,
          hasDefault := allHasNamedDefault row,
          policy := policy }
      encodePlannerPlan input.cases.length (Ora.SinoraPlanner.choosePlanReference input) ==
        rowPlan row

theorem planBuilder_encodePlannerPlan
    (casesLen : Nat) (plan : Ora.DispatcherPlannerSpec.Plan)
    (hshape : Ora.DispatcherPlannerSpec.Plan.wellShaped plan = true) :
    planBuilder? (encodePlannerPlan casesLen plan) casesLen = some plan.toBuilderPlan := by
  cases plan with
  | linear =>
      simp [encodePlannerPlan, planBuilder?, planShapeValid, planStrategy, planDenseKind,
        planTableSlots, planIndexBits, planIndexShift, planMulConstant,
        Ora.DispatcherPlannerSpec.Plan.toBuilderPlan]
  | dense dense =>
      cases dense <;>
        simp_all [encodePlannerPlan, planBuilder?, planShapeValid, planStrategy, planDenseKind,
          planTableSlots, planIndexBits, planIndexShift, planMulConstant,
          Ora.DispatcherPlannerSpec.Plan.toBuilderPlan,
          Ora.DispatcherPlannerSpec.Plan.wellShaped,
          Ora.DispatcherPlannerSpec.DensePlan.wellShaped,
          Ora.DispatcherPlannerSpec.DensePlan.index, bitWindowIndex, multiplicativeIndex,
          Ora.DispatcherPlannerSpec.bitWindowIndex,
          Ora.DispatcherPlannerSpec.multiplicativeIndex, pow2,
          Ora.DispatcherPlannerSpec.pow2]
  | sparse sparse =>
      cases sparse
      simp_all [encodePlannerPlan, planBuilder?, planShapeValid, planStrategy, planDenseKind,
        planTableSlots, planIndexBits, planIndexShift, planMulConstant,
        Ora.DispatcherPlannerSpec.Plan.toBuilderPlan,
        Ora.DispatcherPlannerSpec.Plan.wellShaped,
        Ora.DispatcherPlannerSpec.SparsePlan.wellShaped,
        Ora.DispatcherPlannerSpec.SparsePlan.index, bitWindowIndex,
        Ora.DispatcherPlannerSpec.bitWindowIndex, pow2,
        Ora.DispatcherPlannerSpec.pow2]
      rfl

def linearRawPlan (casesLen : Nat) : RawPlan :=
  { strategy := "linear", denseKind := "", tableSlots := casesLen,
    indexBits := 0, indexShift := 0, mulConstant := 0 }

def sparseBucketBits : List Nat := [1, 2, 3, 4, 5, 6, 7, 8]

def sparseBucketShifts : List Nat := [0, 4, 8, 12, 16, 20, 24]

def u32Modulus : Nat := pow2 32

def multiplicativeCandidateFormula (index : Nat) : Nat :=
  let z0 : BitVec 32 := BitVec.ofNat 32 index + BitVec.ofNat 32 0x9E3779B9
  let z1 : BitVec 32 := (z0 ^^^ (z0 >>> 16)) * BitVec.ofNat 32 0x85EBCA6B
  let z2 : BitVec 32 := (z1 ^^^ (z1 >>> 13)) * BitVec.ofNat 32 0xC2B2AE35
  let z3 : BitVec 32 := z2 ^^^ (z2 >>> 16)
  (z3 ||| BitVec.ofNat 32 1).toNat

def multiplicativeCandidate (index : Nat) : Nat :=
  compilerDispatcherMultiplicativeCandidates[index]?.getD 0

def noDuplicateNat : List Nat → Bool
  | [] => true
  | value :: rest => !(rest.contains value) && noDuplicateNat rest

def collisionFree (selectors : List Nat) (index : Nat → Nat) : Bool :=
  noDuplicateNat (selectors.map index)

def denseBitWindowPlan? (selectors : List Nat) (bits shift : Nat) : Option RawPlan :=
  if collisionFree selectors (fun selector => bitWindowIndex selector bits shift) then
    some
      { strategy := "dense", denseKind := "bit_window", tableSlots := pow2 bits,
        indexBits := bits, indexShift := shift, mulConstant := 0 }
  else
    none

def multiplicativeTableSlotsFrom : Nat → Nat → List Nat
  | 0, _ => []
  | fuel + 1, tableSlots =>
      if tableSlots <= 256 then
        tableSlots :: multiplicativeTableSlotsFrom fuel (tableSlots * 2)
      else
        []

def ceilPowerOfTwoFrom : Nat → Nat → Nat → Nat
  | 0, current, _ => current
  | fuel + 1, current, minimum =>
      if minimum <= current then current
      else ceilPowerOfTwoFrom fuel (current * 2) minimum

def ceilPowerOfTwo (minimum : Nat) : Nat :=
  ceilPowerOfTwoFrom 8 2 minimum

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

def multiplicativeSearchTableSlots : RawMultiplicativeSearch → Nat
  | (tableSlots, _, _) => tableSlots

def multiplicativeSearchSelected : RawMultiplicativeSearch → Option Nat
  | (_, selected, _) => selected

def multiplicativeSearchRejected : RawMultiplicativeSearch → List RawCollisionWitness
  | (_, _, rejected) => rejected

def collisionWitnessConstant : RawCollisionWitness → Nat
  | (constant, _, _) => constant

def collisionWitnessFirstCase : RawCollisionWitness → Nat
  | (_, firstCase, _) => firstCase

def collisionWitnessSecondCase : RawCollisionWitness → Nat
  | (_, _, secondCase) => secondCase

def collisionWitnessValid (selectors : List Nat) (bits : Nat)
    (witness : RawCollisionWitness) : Bool :=
  match selectors.get? (collisionWitnessFirstCase witness),
      selectors.get? (collisionWitnessSecondCase witness) with
  | some first, some second =>
      decide (collisionWitnessFirstCase witness < collisionWitnessSecondCase witness) &&
        decide (multiplicativeIndex first (collisionWitnessConstant witness) bits =
          multiplicativeIndex second (collisionWitnessConstant witness) bits)
  | _, _ => false

def rejectedWitnessesValid (selectors : List Nat) (bits : Nat) : List RawCollisionWitness → Bool
  | [] => true
  | witness :: rest =>
      collisionWitnessValid selectors bits witness && rejectedWitnessesValid selectors bits rest

def rejectedConstantsValid (rejected : List RawCollisionWitness) : Bool :=
  rejected.map collisionWitnessConstant ==
    compilerDispatcherMultiplicativeCandidates.take rejected.length

def multiplicativeSearchValid (selectors : List Nat) (search : RawMultiplicativeSearch) : Bool :=
  match indexBitsForSlots? (multiplicativeSearchTableSlots search) with
  | none => false
  | some bits =>
      let rejected := multiplicativeSearchRejected search
      let selected := multiplicativeSearchSelected search
      rejectedConstantsValid rejected && rejectedWitnessesValid selectors bits rejected &&
        match selected with
        | some candidateIndex =>
            decide (candidateIndex < expectedMultiplicativeSearchBudget) &&
              decide (rejected.length = candidateIndex) &&
              collisionFree selectors (fun selector =>
                multiplicativeIndex selector (multiplicativeCandidate candidateIndex) bits)
        | none => decide (rejected.length = expectedMultiplicativeSearchBudget)

def multiplicativeSearchesValid (selectors : List Nat)
    (searches : List RawMultiplicativeSearch) : Bool :=
  searches.map multiplicativeSearchTableSlots == multiplicativeTableSlots selectors.length &&
    searches.all (multiplicativeSearchValid selectors)

def incrementBucket : Nat → List (Nat × Nat) → List (Nat × Nat)
  | bucket, [] => [(bucket, 1)]
  | bucket, entry :: rest =>
      if entry.1 == bucket then (entry.1, entry.2 + 1) :: rest
      else entry :: incrementBucket bucket rest

def bucketCounts (selectors : List Nat) (bits shift : Nat) : List Nat :=
  (selectors.foldl (fun counts selector =>
    incrementBucket (bitWindowIndex selector bits shift) counts) []).map Prod.snd

def usedBucketCount (counts : List Nat) : Nat :=
  counts.length

def maxBucketSize (counts : List Nat) : Nat :=
  (counts.max?).getD 0

def successfulScanChecks (counts : List Nat) : Nat :=
  counts.foldl (fun total count => total + count * (count + 1) / 2) 0

def divRound (numerator denominator : Nat) : Nat :=
  (numerator + denominator / 2) / denominator

def policyLambda? : String → Option Nat
  | policy =>
      (expectedDispatchPolicyLambdasX1000.find? fun entry => entry.1 == policy).map Prod.snd

def planScore (runtimeAvgChecks codeBytes policyLambda : Nat) : Nat :=
  runtimeAvgChecks + policyLambda * codeBytes

def linearAverageChecks (casesLen : Nat) : Nat :=
  divRound
    ((casesLen * (casesLen + 1) / 2) * expectedExactSelectorCheckX1000)
    casesLen

def linearScore (casesLen policyLambda : Nat) : Nat :=
  if casesLen = 0 then 0
  else
    planScore
      (linearAverageChecks casesLen)
      (expectedLinearCaseCodeBytes * casesLen)
      policyLambda

def denseRuntimeChecks (plan : RawPlan) : Nat :=
  expectedTableDispatchOverheadChecksX1000 + expectedExactSelectorCheckX1000 +
    if planDenseKind plan == "multiplicative" then
      expectedDenseMultiplicativeExtraChecksX1000
    else 0

def denseCodeBytes (plan : RawPlan) (casesLen : Nat) : Nat :=
  let preamble :=
    if planDenseKind plan == "multiplicative" then
      expectedDenseMultiplicativePreambleCodeBytes
    else expectedDenseBitWindowPreambleCodeBytes
  preamble + expectedJumpTableEntryBytes * planTableSlots plan +
    expectedDenseUsedSlotCodeBytes * casesLen

def denseScore (plan : RawPlan) (casesLen policyLambda : Nat) : Nat :=
  planScore (denseRuntimeChecks plan) (denseCodeBytes plan casesLen) policyLambda

def multiplicativePlans (selectors : List Nat) (policyLambda : Nat)
    (searches : List RawMultiplicativeSearch) : List RawScoredPlan :=
  searches.filterMap fun search =>
    (multiplicativeSearchSelected search).map fun candidateIndex =>
      let tableSlots := multiplicativeSearchTableSlots search
      let bits := (indexBitsForSlots? tableSlots).getD 0
      let plan : RawPlan :=
        { strategy := "dense", denseKind := "multiplicative", tableSlots := tableSlots,
          indexBits := bits, indexShift := 32 - bits,
          mulConstant := multiplicativeCandidate candidateIndex }
      (plan, denseScore plan selectors.length policyLambda)

def sparseScore (selectors : List Nat) (plan : RawPlan) (policyLambda : Nat) : Nat :=
  let counts := bucketCounts selectors (planIndexBits plan) (planIndexShift plan)
  let usedBuckets := usedBucketCount counts
  let exactAverage :=
    divRound
      (successfulScanChecks counts * expectedExactSelectorCheckX1000)
      selectors.length
  let runtimeAverage := exactAverage + expectedTableDispatchOverheadChecksX1000
  let codeBytes := expectedSparsePreambleCodeBytes +
    expectedJumpTableEntryBytes * planTableSlots plan +
    expectedSparseUsedBucketCodeBytes * usedBuckets +
    expectedSparseCaseCodeBytes * selectors.length
  planScore runtimeAverage codeBytes policyLambda

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
        some (plan, sparseScore selectors plan policyLambda)
      else
        none

def denseCandidateBetter (casesLen : Nat) (candidate best : RawScoredPlan) : Bool :=
  if candidate.2 != best.2 then decide (candidate.2 < best.2)
  else if planTableSlots candidate.1 != planTableSlots best.1 then
    decide (planTableSlots candidate.1 < planTableSlots best.1)
  else
    let candidateLoad :=
      divRound (casesLen * expectedCheckScaleX1000) (planTableSlots candidate.1)
    let bestLoad :=
      divRound (casesLen * expectedCheckScaleX1000) (planTableSlots best.1)
    if candidateLoad != bestLoad then decide (candidateLoad > bestLoad)
    else if planDenseKind candidate.1 != planDenseKind best.1 then
      planDenseKind candidate.1 == "bit_window"
    else if planIndexShift candidate.1 != planIndexShift best.1 then
      decide (planIndexShift candidate.1 < planIndexShift best.1)
    else if planIndexBits candidate.1 != planIndexBits best.1 then
      decide (planIndexBits candidate.1 < planIndexBits best.1)
    else false

def sparseCandidateBetter (selectors : List Nat) (candidate best : RawScoredPlan) : Bool :=
  let candidateCounts :=
    bucketCounts selectors (planIndexBits candidate.1) (planIndexShift candidate.1)
  let bestCounts := bucketCounts selectors (planIndexBits best.1) (planIndexShift best.1)
  if candidate.2 != best.2 then decide (candidate.2 < best.2)
  else if maxBucketSize candidateCounts != maxBucketSize bestCounts then
    decide (maxBucketSize candidateCounts < maxBucketSize bestCounts)
  else if usedBucketCount candidateCounts != usedBucketCount bestCounts then
    decide (usedBucketCount candidateCounts < usedBucketCount bestCounts)
  else if planTableSlots candidate.1 != planTableSlots best.1 then
    decide (planTableSlots candidate.1 < planTableSlots best.1)
  else if planIndexShift candidate.1 != planIndexShift best.1 then
    decide (planIndexShift candidate.1 < planIndexShift best.1)
  else decide (planIndexBits candidate.1 < planIndexBits best.1)

def bestDenseCandidate (casesLen : Nat) (candidates : List RawScoredPlan) : Option RawScoredPlan :=
  candidates.foldl (fun best candidate =>
    match best with
    | none => some candidate
    | some current =>
        if denseCandidateBetter casesLen candidate current then some candidate else best) none

def bestSparseCandidate
    (selectors : List Nat)
    (candidates : List RawScoredPlan) : Option RawScoredPlan :=
  candidates.foldl (fun best candidate =>
    match best with
    | none => some candidate
    | some current =>
        if sparseCandidateBetter selectors candidate current then some candidate else best) none

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
  let selectors := (rowCases row).map caseSelector
  let casesLen := selectors.length
  let policy := tracePolicy (rowTrace row)
  match policyLambda? policy with
  | none => none
  | some policyLambda =>
      let preconditions := plannerPreconditionsMet selectors (allHasNamedDefault row)
      let linear := linearScore casesLen policyLambda
      let searches := traceMultiplicativeSearches (rowTrace row)
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
  let selectors := (rowCases row).map caseSelector
  if plannerPreconditionsMet selectors (allHasNamedDefault row) then
    multiplicativeSearchesValid selectors (traceMultiplicativeSearches (rowTrace row))
  else (traceMultiplicativeSearches (rowTrace row)).isEmpty

def rowPlannerCoreMatches (row : RawRow) : Bool :=
  match recomputePlannerCore? row with
  | some (trace, plan) =>
      trace == rowTrace row && plan == rowPlan row
  | none => false

def rowPlannerChosen? (row : RawRow) : Option RawPlan :=
  (recomputePlannerCore? row).map fun result => result.2

def rowPlannerMatches (row : RawRow) : Bool :=
  rowMultiplicativeSearchesValid row && rowPlannerCoreMatches row

def rowPlanIndicesMatch (row : RawRow) : Bool :=
  rowPlanShapeValid row &&
    match rowStrategy row with
    | "linear" => linearRouteIndicesSequential row
    | "dense" =>
        (rowCases row).all (fun c => caseIndex c == rowPlanIndex row (caseSelector c))
    | "sparse" =>
        (rowCases row).all (fun c => caseIndex c == rowPlanIndex row (caseSelector c))
    | _ => false

def rowStrategyWF (row : RawRow) : Prop :=
  match rowBuilderPlan? row with
  | some plan => Ora.Dispatcher.StrategyWF (rowCasePairs row) plan
  | none => False

def rowManifestBaseOk (row : RawRow) : Bool :=
  strategyKnown row &&
    selectorsDistinct row &&
    routeIndicesInRange row &&
    rowPlanShapeValid row &&
    allHasNamedDefault row &&
    denseInjectiveIfDense row &&
    rowPlanIndicesMatch row

def rowManifestOk (row : RawRow) : Bool :=
  rowManifestBaseOk row && rowPlannerMatches row

theorem row_builder_correct_of_plan_admissible (row : RawRow)
    (h : rowPlanAdmissible row = true) :
    rowStrategyWF row :=
  by
    unfold rowPlanAdmissible at h
    unfold rowStrategyWF
    cases hplan : rowBuilderPlan? row with
    | none =>
        simp [hplan] at h
    | some plan =>
        simp [hplan] at h ⊢
        exact Ora.Dispatcher.builder_correct (rowCasePairs row) plan h

theorem row_builder_correct_of_reference_match (row : RawRow)
    (h : rowPlannerReferenceMatches row = true) :
    rowStrategyWF row := by
  unfold rowPlannerReferenceMatches at h
  cases hpolicy : plannerPolicy? (tracePolicy (rowTrace row)) with
  | none => simp [hpolicy] at h
  | some policy =>
      simp only [hpolicy] at h
      let input : Ora.DispatcherPlannerSpec.Input String :=
        { cases := rowCasePairs row,
          hasDefault := allHasNamedDefault row,
          policy := policy }
      have hplan :
          encodePlannerPlan input.cases.length
              (Ora.SinoraPlanner.choosePlanReference input) = rowPlan row := by
        simpa [input] using h
      have hwf := Ora.SinoraPlanner.planner_reference_builder_correct input
      have hshape := Ora.SinoraPlanner.choosePlanReference_wellShaped input
      have hlen : (rowCasePairs row).length = (rowCases row).length := by
        simp [rowCasePairs]
      dsimp [input] at hplan hwf hshape
      unfold rowStrategyWF rowBuilderPlan?
      rw [← hplan]
      rw [← hlen]
      rw [planBuilder_encodePlannerPlan _ _ hshape]
      exact hwf

theorem rows_builder_correct_of_plan_admissible (rows : List RawRow)
    (h : rows.all rowPlanAdmissible = true) :
    ∀ row, row ∈ rows → rowStrategyWF row := by
  intro row hmem
  exact row_builder_correct_of_plan_admissible row ((List.all_eq_true).mp h row hmem)

theorem rows_builder_correct_of_reference_match (rows : List RawRow)
    (h : rows.all rowPlannerReferenceMatches = true) :
    ∀ row, row ∈ rows → rowStrategyWF row := by
  intro row hmem
  exact row_builder_correct_of_reference_match row ((List.all_eq_true).mp h row hmem)

/- Stable surface consumed by the Zig-emitted per-contract Lean checker.
    Keep the runtime gate on these names so `lake build` catches helper renames
    here instead of letting string-generated Lean drift until a CLI run. -/
namespace Gate

def networkMatches
    (intents : List RawIntent)
    (switches : List RawNetworkSwitch)
    (entry : String) : Bool :=
  dispatcherNetworkMatches intents switches entry

def rowsHaveNamedDefault (rows : List RawRow) : Bool :=
  rows.all allHasNamedDefault

def rowsCovered (rows : List RawRow) : Bool :=
  rows.all modelRunKnownOk

def denseRowsInjective (rows : List RawRow) : Bool :=
  rows.all denseInjectiveIfDense

def rowsMatchModel (rows : List RawRow) : Bool :=
  rows.all rowMatchesDispatcherModel

def planShapesValid (rows : List RawRow) : Bool :=
  rows.all rowPlanShapeValid

def plansAdmissible (rows : List RawRow) : Bool :=
  rows.all fun row =>
    rawPlanAdmissible (rowCasePairs row) (rowCases row).length (rowPlan row)

def plannerMatches (rows : List RawRow) : Bool :=
  rows.all rowPlannerMatches

def plannerReferenceMatches (rows : List RawRow) : Bool :=
  rows.all rowPlannerReferenceMatches

def plannerSearchesValid (rows : List RawRow) : Bool :=
  rows.all rowMultiplicativeSearchesValid

def plannerCoreMatches (rows : List RawRow) : Bool :=
  rows.all rowPlannerCoreMatches

def planIndicesMatch (rows : List RawRow) : Bool :=
  rows.all rowPlanIndicesMatch

def manifestRowsMatch (rows : List RawRow) : Bool :=
  rows.all rowManifestOk

def manifestBaseRowsMatch (rows : List RawRow) : Bool :=
  rows.all rowManifestBaseOk

def rowStrategyWF := Ora.DispatcherTableSync.rowStrategyWF

theorem builderCorrect (rows : List RawRow)
    (h : plannerReferenceMatches rows = true) :
    ∀ row, row ∈ rows → rowStrategyWF row :=
  rows_builder_correct_of_reference_match rows h

theorem plannerMatchesOfParts (rows : List RawRow)
    (hsearches : plannerSearchesValid rows = true)
    (hcore : plannerCoreMatches rows = true) :
    plannerMatches rows = true := by
  induction rows with
  | nil => rfl
  | cons row rest ih =>
      simp only [plannerSearchesValid, plannerCoreMatches, plannerMatches,
        rowPlannerMatches, List.all_cons, Bool.and_eq_true] at hsearches hcore ⊢
      exact ⟨⟨hsearches.1, hcore.1⟩, ih hsearches.2 hcore.2⟩

theorem manifestRowsMatchOfParts (rows : List RawRow)
    (hbase : manifestBaseRowsMatch rows = true)
    (hplanner : plannerMatches rows = true) :
    manifestRowsMatch rows = true := by
  induction rows with
  | nil => rfl
  | cons row rest ih =>
      simp only [manifestBaseRowsMatch, plannerMatches, manifestRowsMatch,
        rowManifestOk, List.all_cons, Bool.and_eq_true] at hbase hplanner ⊢
      exact ⟨⟨hbase.1, hplanner.1⟩, ih hbase.2 hplanner.2⟩

end Gate

/- Zig emits these names into every per-contract checker. Keeping the complete
   public surface in this gate-collected block makes a Lean-side rename fail the
   repository build before it can become a runtime checker error. -/
#check @Gate.networkMatches
#check @Gate.rowsHaveNamedDefault
#check @Gate.rowsCovered
#check @Gate.denseRowsInjective
#check @Gate.rowsMatchModel
#check @Gate.planShapesValid
#check @Gate.plansAdmissible
#check @Gate.plannerMatches
#check @Gate.plannerReferenceMatches
#check @Gate.plannerSearchesValid
#check @Gate.plannerCoreMatches
#check @Gate.planIndicesMatch
#check @Gate.manifestRowsMatch
#check @Gate.manifestBaseRowsMatch
#check @Gate.rowStrategyWF
#check @Gate.builderCorrect
#check @Gate.plannerMatchesOfParts
#check @Gate.manifestRowsMatchOfParts

def compilerDispatcherRowsHaveNamedDefault : Bool :=
  Gate.rowsHaveNamedDefault compilerDispatcherTableRows

def compilerDispatcherRowsCovered : Bool :=
  Gate.rowsCovered compilerDispatcherTableRows

def compilerDispatcherRowsSound : Bool :=
  compilerDispatcherTableRows.all modelRunSoundOk

def compilerDenseRowsInjective : Bool :=
  Gate.denseRowsInjective compilerDispatcherTableRows

def compilerDispatcherPlanShapesValid : Bool :=
  Gate.planShapesValid compilerDispatcherTableRows

def compilerSparseRowsExactScan : Bool :=
  compilerDispatcherTableRows.all fun row =>
    if rowStrategy row == "sparse" then
      sparseRunKnownOk row && allHasNamedDefault row
    else true

def compilerDispatcherTableRowsMatch : Bool :=
  Gate.rowsMatchModel compilerDispatcherTableRows

def compilerDispatcherPlansAdmissible : Bool :=
  Gate.plansAdmissible compilerDispatcherTableRows

def compilerDispatcherPlannerMatches : Bool :=
  Gate.plannerMatches compilerDispatcherTableRows

def compilerDispatcherPlanIndicesMatch : Bool :=
  Gate.planIndicesMatch compilerDispatcherTableRows

def compilerDispatcherManifestRowsMatch : Bool :=
  Gate.manifestRowsMatch compilerDispatcherTableRows

def compilerDispatcherPlannerSearchesValid : Bool :=
  Gate.plannerSearchesValid compilerDispatcherTableRows

def compilerDispatcherPlannerCoreMatches : Bool :=
  Gate.plannerCoreMatches compilerDispatcherTableRows

def compilerDispatcherManifestBaseRowsMatch : Bool :=
  Gate.manifestBaseRowsMatch compilerDispatcherTableRows

set_option maxRecDepth 1000000 in
set_option maxHeartbeats 2000000 in
theorem dispatcher_multiplicative_candidates_match :
    compilerDispatcherMultiplicativeCandidates =
      (List.range expectedMultiplicativeSearchBudget).map
        multiplicativeCandidateFormula := by decide

theorem multiplicative_candidate_discriminators :
    multiplicativeCandidateFormula 0 = 2462723855 ∧
      multiplicativeCandidateFormula 1 = 2527132011 ∧
      multiplicativeCandidateFormula 2 = 3024231355 := by decide

theorem dispatcher_policy_lambda_discriminators :
    policyLambda? "gas" = some 0 ∧
      policyLambda? "balanced" = some 5 ∧
      policyLambda? "size" = some 50 ∧
      policyLambda? "unknown" = none := by decide

theorem multiplicative_search_rejects_false_collision_witness :
    multiplicativeSearchValid
      [1447852734, 832491607, 3309386683, 2561671559]
      (4, some 1, [(2462723855, 0, 1)]) = false := by decide

theorem dispatcher_table_rows_have_named_default :
    compilerDispatcherRowsHaveNamedDefault = true := by decide

theorem dispatcher_table_rows_covered :
    compilerDispatcherRowsCovered = true := by decide

theorem dispatcher_table_rows_sound :
    compilerDispatcherRowsSound = true := by decide

theorem dense_dispatcher_rows_injective :
    compilerDenseRowsInjective = true := by decide

theorem dispatcher_plan_shapes_valid :
    compilerDispatcherPlanShapesValid = true := by decide

theorem sparse_dispatcher_rows_exact_scan :
    compilerSparseRowsExactScan = true := by decide

theorem dispatcher_table_rows_match_model :
    compilerDispatcherTableRowsMatch = true := by decide

theorem dispatcher_plans_admissible :
    compilerDispatcherPlansAdmissible = true := by decide

set_option maxRecDepth 1000000 in
set_option maxHeartbeats 2000000 in
theorem dispatcher_planner_matches :
    compilerDispatcherPlannerMatches = true := by
  apply Gate.plannerMatchesOfParts
  · decide
  · decide

theorem dispatcher_plan_indices_match :
    compilerDispatcherPlanIndicesMatch = true := by decide

set_option maxRecDepth 1000000 in
set_option maxHeartbeats 2000000 in
theorem dispatcher_manifest_rows_match :
    compilerDispatcherManifestRowsMatch = true := by
  apply Gate.manifestRowsMatchOfParts
  · decide
  · exact dispatcher_planner_matches

end Ora.DispatcherTableSync

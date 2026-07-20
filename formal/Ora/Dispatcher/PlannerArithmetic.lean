/-
Shared arithmetic and list computations for the declarative dispatcher planner
and its compiler-table sync checker.

One definition is imported by both consumers. Sync proofs must compare planner
data and decisions, not bridge duplicated implementations of these helpers.
-/

import Ora.Spec.DispatcherFacts
import Ora.Util.List

namespace Ora.Dispatcher.PlannerArithmetic

open Ora.Spec.DispatcherFacts

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

def collisionFree (selectors : List Nat) (index : Nat → Nat) : Bool :=
  Ora.Util.noDuplicateNat (selectors.map index)

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

end Ora.Dispatcher.PlannerArithmetic

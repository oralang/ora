/-
Typed decode layer for compiler-emitted dispatcher table snapshot rows.

This module is the only place that binds positional snapshot tuples to named
fields. It contains data decoding only; planner and gate logic live in the sibling
modules.
-/

import Ora.Generated.DispatcherTableSnapshot

namespace Ora.DispatcherTableSync

open Ora.Generated

structure RawCase where
  selector : Nat
  label : String
  index : Nat
  hasNamedDefault : Bool
  deriving Repr, DecidableEq

structure RawPlan where
  strategy : String
  denseKind : String
  tableSlots : Nat
  indexBits : Nat
  indexShift : Nat
  mulConstant : Nat
  deriving Repr, DecidableEq

abbrev RawScoredPlan := RawPlan × Nat

structure RawCollisionWitness where
  constant : Nat
  firstCase : Nat
  secondCase : Nat
  deriving Repr, DecidableEq

structure RawMultiplicativeSearch where
  tableSlots : Nat
  selected : Option Nat
  rejected : List RawCollisionWitness
  deriving Repr, DecidableEq

/- The generated snapshot stays on positional tuples as its wire format; the
   decode layer below is the single place that binds tuple positions to named
   fields, so a transposed emitter column shows up here and nowhere else. -/
abbrev GeneratedRawCase := Nat × String × Nat × Bool
abbrev GeneratedRawPlan := String × String × Nat × Nat × Nat × Nat
abbrev GeneratedRawScoredPlan := GeneratedRawPlan × Nat
abbrev GeneratedRawCollisionWitness := Nat × Nat × Nat
abbrev GeneratedRawMultiplicativeSearch :=
  Nat × Option Nat × List GeneratedRawCollisionWitness
abbrev GeneratedRawPlannerTrace :=
  String × Bool × Nat × List GeneratedRawMultiplicativeSearch ×
    Nat × Option GeneratedRawScoredPlan × Nat × Option GeneratedRawScoredPlan
abbrev GeneratedRawRow :=
  String × GeneratedRawPlan × GeneratedRawPlannerTrace × List GeneratedRawCase

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

def decodeGeneratedCase : GeneratedRawCase → RawCase
  | (selector, label, index, hasNamedDefault) =>
      { selector, label, index, hasNamedDefault }

def decodeGeneratedPlan : GeneratedRawPlan → RawPlan
  | (strategy, denseKind, tableSlots, indexBits, indexShift, mulConstant) =>
      { strategy, denseKind, tableSlots, indexBits, indexShift, mulConstant }

def decodeGeneratedScoredPlan : GeneratedRawScoredPlan → RawScoredPlan
  | (plan, score) => (decodeGeneratedPlan plan, score)

def decodeGeneratedCollisionWitness : GeneratedRawCollisionWitness → RawCollisionWitness
  | (constant, firstCase, secondCase) => { constant, firstCase, secondCase }

def decodeGeneratedMultiplicativeSearch :
    GeneratedRawMultiplicativeSearch → RawMultiplicativeSearch
  | (tableSlots, selected, rejected) =>
      { tableSlots, selected,
        rejected := rejected.map decodeGeneratedCollisionWitness }

def decodeGeneratedPlannerTrace : GeneratedRawPlannerTrace → RawPlannerTrace
  | (policy, preconditionsMet, linearScore, multiplicativeSearches,
      denseCandidateCount, bestDense, sparseCandidateCount, bestSparse) =>
      { policy, preconditionsMet, linearScore,
        multiplicativeSearches :=
          multiplicativeSearches.map decodeGeneratedMultiplicativeSearch,
        denseCandidateCount, bestDense := bestDense.map decodeGeneratedScoredPlan,
        sparseCandidateCount, bestSparse := bestSparse.map decodeGeneratedScoredPlan }

def decodeGeneratedRow : GeneratedRawRow → RawRow
  | (name, plan, trace, cases) =>
      { name, plan := decodeGeneratedPlan plan,
        trace := decodeGeneratedPlannerTrace trace,
        cases := cases.map decodeGeneratedCase }

def compilerDispatcherTableRows : List RawRow :=
  Ora.Generated.compilerDispatcherTableRows.map decodeGeneratedRow


end Ora.DispatcherTableSync

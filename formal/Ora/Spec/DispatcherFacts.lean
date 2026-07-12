/-
Spec-side fact interface for Sinora dispatcher strategy sync.

This is the trusted representation of the dispatcher strategy table. Generated
Sinora data is allowed to use strings at the boundary, but this file keeps the
strategy identities and classifications named and typed.
-/

namespace Ora.Spec.DispatcherFacts

inductive DispatcherStrategy where
  | linear
  | sparse
  | dense
  deriving Repr, DecidableEq

structure DispatcherStrategyInfo where
  compilerName : String
  requiresExactSelectorValidation : Bool
  usesCompressedIndex : Bool
  deriving Repr, DecidableEq

def DispatcherStrategy.info : DispatcherStrategy → DispatcherStrategyInfo
  | .linear =>
      { compilerName := "linear",
        requiresExactSelectorValidation := true,
        usesCompressedIndex := false }
  | .sparse =>
      { compilerName := "sparse",
        requiresExactSelectorValidation := true,
        usesCompressedIndex := true }
  | .dense =>
      { compilerName := "dense",
        requiresExactSelectorValidation := true,
        usesCompressedIndex := true }

def allDispatcherStrategies : List DispatcherStrategy :=
  [.linear, .sparse, .dense]

def expectedDispatcherStrategyRows : List (String × Bool × Bool) :=
  allDispatcherStrategies.map fun strategy =>
    let info := strategy.info
    (info.compilerName, info.requiresExactSelectorValidation, info.usesCompressedIndex)

/-
The `range` kind was retired by the planner's policy-score model:
multiplicative perfect hashing dominates its entire natural domain, leaving
range with no reachable selection path (see sinora/src/switch_routing.zig).
-/
inductive DensePlanKind where
  | bitWindow
  | multiplicative
  deriving Repr, DecidableEq

def DensePlanKind.compilerName : DensePlanKind → String
  | .bitWindow => "bit_window"
  | .multiplicative => "multiplicative"

def allDensePlanKinds : List DensePlanKind :=
  [.bitWindow, .multiplicative]

def expectedDensePlanKinds : List String :=
  allDensePlanKinds.map DensePlanKind.compilerName

def expectedSparseBucketBits : List Nat :=
  [1, 2, 3, 4, 5, 6, 7, 8]

def expectedSparseBucketShifts : List Nat :=
  [0, 4, 8, 12, 16, 20, 24]

def expectedDenseMaxTableSlots : Nat := 256
/-
Reduced from 4000 when the planner moved to policy scores: the margin's
original job (absorbing unmodeled table-dispatch overhead) is now done by
explicit runtime + code-byte costs, leaving a one-check model-error margin.
-/
def expectedMinSelectorCheckSavingX1000 : Nat := 1000

/- Independent specification of the planner cost model. The compiler emits
   its own values into `DispatcherStrategySnapshot`; kernel-checked sync
   theorems reject any change made on only one side. -/
def expectedTableDispatchOverheadChecksX1000 : Nat := 2800
def expectedExactSelectorCheckX1000 : Nat := 1000
def expectedDenseMultiplicativeExtraChecksX1000 : Nat := 300
def expectedMultiplicativeSearchBudget : Nat := 512
def expectedDispatchPolicyLambdasX1000 : List (String × Nat) :=
  [("gas", 0), ("balanced", 5), ("size", 50)]
def expectedJumpTableEntryBytes : Nat := 2
def expectedLinearCaseCodeBytes : Nat := 13
def expectedDenseBitWindowPreambleCodeBytes : Nat := 30
def expectedDenseMultiplicativePreambleCodeBytes : Nat := 36
def expectedDenseUsedSlotCodeBytes : Nat := 18
def expectedSparsePreambleCodeBytes : Nat := 30
def expectedSparseUsedBucketCodeBytes : Nat := 5
def expectedSparseCaseCodeBytes : Nat := 17

def HasExactSelectorValidation (strategy : DispatcherStrategy) : Prop :=
  strategy.info.requiresExactSelectorValidation = true

theorem linear_has_exact_selector_validation :
    HasExactSelectorValidation .linear := rfl

theorem sparse_has_exact_selector_validation :
    HasExactSelectorValidation .sparse := rfl

theorem dense_has_exact_selector_validation :
    HasExactSelectorValidation .dense := rfl

end Ora.Spec.DispatcherFacts

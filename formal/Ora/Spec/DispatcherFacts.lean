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

inductive DensePlanKind where
  | bitWindow
  | range
  deriving Repr, DecidableEq

def DensePlanKind.compilerName : DensePlanKind → String
  | .bitWindow => "bit_window"
  | .range => "range"

def allDensePlanKinds : List DensePlanKind :=
  [.bitWindow, .range]

def expectedDensePlanKinds : List String :=
  allDensePlanKinds.map DensePlanKind.compilerName

def expectedSparseBucketBits : List Nat :=
  [1, 2, 3, 4, 5, 6, 7, 8]

def expectedSparseBucketShifts : List Nat :=
  [0, 4, 8, 12, 16, 20, 24]

def expectedDenseMaxTableSlots : Nat := 256
def expectedMinSelectorCheckSavingX1000 : Nat := 4000

def HasExactSelectorValidation (strategy : DispatcherStrategy) : Prop :=
  strategy.info.requiresExactSelectorValidation = true

theorem linear_has_exact_selector_validation :
    HasExactSelectorValidation .linear := rfl

theorem sparse_has_exact_selector_validation :
    HasExactSelectorValidation .sparse := rfl

theorem dense_has_exact_selector_validation :
    HasExactSelectorValidation .dense := rfl

end Ora.Spec.DispatcherFacts

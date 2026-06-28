/-
Trusted sync checks for Sinora dispatcher strategy facts.

Sinora emits data-only rows in `Ora/Generated/DispatcherStrategySnapshot.lean`.
This module proves those rows equal the typed strategy table in
`Ora/Spec/DispatcherFacts.lean`.
-/

import Ora.Generated.DispatcherStrategySnapshot
import Ora.Spec.DispatcherFacts

namespace Ora.DispatcherStrategySync

open Ora.Generated Ora.Spec.DispatcherFacts

theorem dispatcher_strategy_rows_match :
    compilerDispatcherStrategyRows = expectedDispatcherStrategyRows := by decide

theorem dense_plan_kinds_match :
    compilerDensePlanKinds = expectedDensePlanKinds := by decide

theorem sparse_bucket_bits_match :
    compilerSparseBucketBits = expectedSparseBucketBits := by decide

theorem sparse_bucket_shifts_match :
    compilerSparseBucketShifts = expectedSparseBucketShifts := by decide

theorem dense_max_table_slots_match :
    compilerDenseMaxTableSlots = expectedDenseMaxTableSlots := by decide

theorem min_selector_check_saving_match :
    compilerMinSelectorCheckSavingX1000 = expectedMinSelectorCheckSavingX1000 := by decide

end Ora.DispatcherStrategySync

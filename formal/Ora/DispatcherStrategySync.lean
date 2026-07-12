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

theorem table_dispatch_overhead_checks_match :
    compilerTableDispatchOverheadChecksX1000 =
      expectedTableDispatchOverheadChecksX1000 := by decide

theorem exact_selector_check_match :
    compilerExactSelectorCheckX1000 = expectedExactSelectorCheckX1000 := by decide

theorem dense_multiplicative_extra_checks_match :
    compilerDenseMultiplicativeExtraChecksX1000 =
      expectedDenseMultiplicativeExtraChecksX1000 := by decide

theorem multiplicative_search_budget_match :
    compilerMultiplicativeSearchBudget = expectedMultiplicativeSearchBudget := by decide

theorem dispatch_policy_lambdas_match :
    compilerDispatchPolicyLambdasX1000 = expectedDispatchPolicyLambdasX1000 := by decide

theorem jump_table_entry_bytes_match :
    compilerJumpTableEntryBytes = expectedJumpTableEntryBytes := by decide

theorem linear_case_code_bytes_match :
    compilerLinearCaseCodeBytes = expectedLinearCaseCodeBytes := by decide

theorem dense_bit_window_preamble_code_bytes_match :
    compilerDenseBitWindowPreambleCodeBytes =
      expectedDenseBitWindowPreambleCodeBytes := by decide

theorem dense_multiplicative_preamble_code_bytes_match :
    compilerDenseMultiplicativePreambleCodeBytes =
      expectedDenseMultiplicativePreambleCodeBytes := by decide

theorem dense_used_slot_code_bytes_match :
    compilerDenseUsedSlotCodeBytes = expectedDenseUsedSlotCodeBytes := by decide

theorem sparse_preamble_code_bytes_match :
    compilerSparsePreambleCodeBytes = expectedSparsePreambleCodeBytes := by decide

theorem sparse_used_bucket_code_bytes_match :
    compilerSparseUsedBucketCodeBytes = expectedSparseUsedBucketCodeBytes := by decide

theorem sparse_case_code_bytes_match :
    compilerSparseCaseCodeBytes = expectedSparseCaseCodeBytes := by decide

end Ora.DispatcherStrategySync

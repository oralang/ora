/-
Trusted sync checks for Sinora backend facts.

Sinora emits data-only rows in `Ora/Generated/SinoraBackendSnapshot.lean`.
This module proves those rows equal the typed backend facts in
`Ora/Spec/SinoraBackendFacts.lean`.
-/

import Ora.Generated.SinoraBackendSnapshot
import Ora.Spec.SinoraBackendFacts

namespace Ora.SinoraBackendSync

open Ora.Generated Ora.Spec.SinoraBackendFacts

theorem optimization_pass_rows_match :
    compilerSinoraOptimizationPassRows = expectedSinoraOptimizationPassRows := by decide

theorem optimization_final_legalizer_matches :
    compilerSinoraOptimizationPipelineRunsFinalLegalizer =
      expectedSinoraOptimizationPipelineRunsFinalLegalizer := by decide

theorem release_pipeline_stages_match :
    compilerSinoraReleasePipelineStages = expectedSinoraReleasePipelineStages := by decide

theorem release_splits_critical_edges_matches :
    compilerSinoraReleaseSplitsCriticalEdges =
      expectedSinoraReleaseSplitsCriticalEdges := by decide

theorem release_uses_effectful_scheduler_matches :
    compilerSinoraReleaseUsesEffectfulScheduler =
      expectedSinoraReleaseUsesEffectfulScheduler := by decide

theorem release_supports_source_maps_matches :
    compilerSinoraReleaseSupportsSourceMaps =
      expectedSinoraReleaseSupportsSourceMaps := by decide

end Ora.SinoraBackendSync

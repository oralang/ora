/-
Spec-side fact interface for Sinora backend sync.

This file records the backend pipeline facts that the formal sync gate expects
Sinora to publish. It is not a bytecode semantics proof yet; it is a kernel
checked drift guard for the backend vocabulary and mandatory release stages.
-/

namespace Ora.Spec.SinoraBackendFacts

inductive OptimizationPass where
  | sccp
  | copyPropagation
  | literalCommoning
  | unusedOperationElimination
  | defragment
  | switchPeephole
  deriving Repr, DecidableEq

structure OptimizationPassInfo where
  compilerName : String
  cliCode : String
  deriving Repr, DecidableEq

def OptimizationPass.info : OptimizationPass → OptimizationPassInfo
  | .sccp =>
      { compilerName := "sccp", cliCode := "s" }
  | .copyPropagation =>
      { compilerName := "copy_propagation", cliCode := "c" }
  | .literalCommoning =>
      { compilerName := "literal_commoning", cliCode := "p" }
  | .unusedOperationElimination =>
      { compilerName := "unused_operation_elimination", cliCode := "u" }
  | .defragment =>
      { compilerName := "defragment", cliCode := "d" }
  | .switchPeephole =>
      { compilerName := "switch_peephole", cliCode := "l" }

def allOptimizationPasses : List OptimizationPass :=
  [.sccp, .copyPropagation, .literalCommoning,
   .unusedOperationElimination, .defragment, .switchPeephole]

def expectedSinoraOptimizationPassRows : List (String × String) :=
  allOptimizationPasses.map fun pass =>
    let info := pass.info
    (info.compilerName, info.cliCode)

inductive ReleasePipelineStage where
  | literalCommoning
  | shortCircuitBranchThreading
  | criticalEdgeSplitting
  | effectAnalysis
  | effectfulScheduling
  | memoryLayout
  | codeToAsm
  deriving Repr, DecidableEq

def ReleasePipelineStage.compilerName : ReleasePipelineStage → String
  | .literalCommoning => "literal_commoning"
  | .shortCircuitBranchThreading => "short_circuit_branch_threading"
  | .criticalEdgeSplitting => "critical_edge_splitting"
  | .effectAnalysis => "effect_analysis"
  | .effectfulScheduling => "effectful_scheduling"
  | .memoryLayout => "memory_layout"
  | .codeToAsm => "code_to_asm"

def expectedSinoraReleasePipelineStages : List String :=
  [.literalCommoning, .shortCircuitBranchThreading, .criticalEdgeSplitting,
   .effectAnalysis, .effectfulScheduling, .memoryLayout, .codeToAsm].map
    ReleasePipelineStage.compilerName

def expectedSinoraOptimizationPipelineRunsFinalLegalizer : Bool := true
def expectedSinoraReleaseSplitsCriticalEdges : Bool := true
def expectedSinoraReleaseUsesEffectfulScheduler : Bool := true
def expectedSinoraReleaseSupportsSourceMaps : Bool := true

def HasFinalLegalizer : Prop :=
  expectedSinoraOptimizationPipelineRunsFinalLegalizer = true

def ReleaseSplitsCriticalEdges : Prop :=
  expectedSinoraReleaseSplitsCriticalEdges = true

def ReleaseUsesEffectfulScheduler : Prop :=
  expectedSinoraReleaseUsesEffectfulScheduler = true

theorem optimization_pipeline_has_final_legalizer :
    HasFinalLegalizer := rfl

theorem release_splits_critical_edges :
    ReleaseSplitsCriticalEdges := rfl

theorem release_uses_effectful_scheduler :
    ReleaseUsesEffectfulScheduler := rfl

end Ora.Spec.SinoraBackendFacts

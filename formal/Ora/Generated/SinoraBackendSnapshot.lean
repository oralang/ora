/-
GENERATED — DATA ONLY. Do NOT edit by hand and do NOT add any `theorem`,
`lemma`, `example`, `axiom`, `sorry`, `instance`, `macro`, attribute, or extra
`import` to this file. It contains only `def … := <literal>` Sinora backend
facts emitted from the compiler workspace. The TRUSTED checks live in
`Ora/SinoraBackendSync.lean`.

Regenerate with `scripts/check-formal-sync.sh`. Source:
src/formal/emit_sinora_backend_snapshot.zig,
sinora/src/passes.zig, sinora/src/release_generic_backend.zig.
-/

namespace Ora.Generated

def compilerSinoraOptimizationPassRows : List (String × String) :=
  [("sccp", "s"),
   ("copy_propagation", "c"),
   ("literal_commoning", "p"),
   ("unused_operation_elimination", "u"),
   ("defragment", "d"),
   ("switch_peephole", "l")]

def compilerSinoraOptimizationPipelineRunsFinalLegalizer : Bool := true

def compilerSinoraReleasePipelineStages : List String :=
  ["literal_commoning", "short_circuit_branch_threading", "critical_edge_splitting", "effect_analysis", "effectful_scheduling", "memory_layout", "code_to_asm"]

def compilerSinoraReleaseSplitsCriticalEdges : Bool := true
def compilerSinoraReleaseUsesEffectfulScheduler : Bool := true
def compilerSinoraReleaseSupportsSourceMaps : Bool := true

end Ora.Generated

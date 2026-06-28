/-
GENERATED — DATA ONLY. Do NOT edit by hand and do NOT add any `theorem`,
`lemma`, `example`, `axiom`, `sorry`, `instance`, `macro`, attribute, or extra
`import` to this file. It contains only `def … := <literal>` dispatcher
strategy rows emitted from Sinora. The TRUSTED checks live in
`Ora/DispatcherStrategySync.lean`.

Regenerate with `scripts/check-formal-sync.sh`. Source:
src/formal/emit_dispatcher_strategy_snapshot.zig,
sinora/src/switch_routing.zig.
-/

namespace Ora.Generated

def compilerDispatcherStrategyRows : List (String × Bool × Bool) :=
  [("linear", true, false),
   ("sparse", true, true),
   ("dense", true, true)]

def compilerDensePlanKinds : List String :=
  ["bit_window", "range"]

def compilerSparseBucketBits : List Nat := [1, 2, 3, 4, 5, 6, 7, 8]

def compilerSparseBucketShifts : List Nat := [0, 4, 8, 12, 16, 20, 24]

def compilerDenseMaxTableSlots : Nat := 256
def compilerMinSelectorCheckSavingX1000 : Nat := 4000

end Ora.Generated

/-
GENERATED — DATA ONLY. Do NOT edit by hand and do NOT add any `theorem`,
`lemma`, `example`, `axiom`, `sorry`, `instance`, `macro`, attribute, or extra
`import` to this file. It contains only primitive fixture rows emitted from
the compiler. The TRUSTED checks live in `Ora/StorageDisjointnessSync.lean`.

Regenerate with `scripts/check-formal-sync.sh`. Source:
src/formal/emit_storage_disjointness_snapshot.zig,
src/formal/obligation.zig.
-/

namespace Ora.Generated

def storageDisjointnessRows :
    List (String × (String × String × List String × List (String × String)) ×
      (String × String × List String × List (String × String)) × Bool) :=
  [("different_roots", ("balances", "storage", [], []), ("allowances", "storage", [], []), true),
   ("same_root_exact_path", ("balances", "storage", [], [("constant", "1")]), ("balances", "storage", [], [("constant", "1")]), false),
   ("whole_root_write_vs_keyed_read", ("balances", "storage", [], [("constant", "1")]), ("balances", "storage", [], []), false),
   ("normalized_unequal_constants", ("balances", "storage", [], [("constant", "1")]), ("balances", "storage", [], [("constant", "2")]), true),
   ("raw_noncanonical_equal_constants", ("balances", "storage", [], [("constant", "1")]), ("balances", "storage", [], [("constant", "01")]), false),
   ("unparseable_constant_blocks", ("balances", "storage", [], [("constant", "0..2")]), ("balances", "storage", [], [("constant", "2")]), false),
   ("underscore_constant_blocks", ("balances", "storage", [], [("constant", "1_000")]), ("balances", "storage", [], [("constant", "1001")]), false),
   ("parameter_vs_parameter", ("balances", "storage", [], [("parameter", "file:0:pattern:0")]), ("balances", "storage", [], [("parameter", "file:0:pattern:1")]), false),
   ("parameter_vs_constant", ("balances", "storage", [], [("parameter", "file:0:pattern:0")]), ("balances", "storage", [], [("constant", "2")]), false),
   ("msg_sender_vs_tx_origin", ("balances", "storage", [], [("msg_sender", "")]), ("balances", "storage", [], [("tx_origin", "")]), false),
   ("unknown_key_blocks", ("balances", "storage", [], [("unknown", "")]), ("balances", "storage", [], [("constant", "2")]), false),
   ("same_prefix_blocks", ("allowances", "storage", [], [("parameter", "file:0:pattern:0")]), ("allowances", "storage", [], [("parameter", "file:0:pattern:0"), ("constant", "1")]), false),
   ("different_concrete_regions", ("scratch", "transient", [], []), ("scratch", "storage", [], []), true),
   ("none_region_blocks", ("scratch", "none", [], []), ("scratch", "storage", [], []), false),
   ("computed_storage_blocks", ("$computed_storage", "storage", [], [("constant", "1")]), ("balances", "storage", [], [("constant", "2")]), false),
   ("different_fields_deferred", ("config", "storage", ["owner"], []), ("config", "storage", ["admin"], []), false)]

end Ora.Generated

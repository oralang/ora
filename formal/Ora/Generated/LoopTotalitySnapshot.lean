/-
GENERATED — DATA ONLY. The compiler emits the scalar-loop support
matrix; trusted checks live in Ora/LoopTotalitySync.lean.
Regenerate with scripts/check-formal-sync.sh.
-/

namespace Ora.Generated

def loopTotalityRows : List (String × String × Bool × String × Bool) := [
  ("supported_while", "supported_while", true, "", true),
  ("supported_for", "supported_for", true, "", true),
  ("storage_write", "storage_write", false, "loop_has_storage_write", false),
  ("external_call", "external_call", false, "loop_has_external_call", false),
  ("resource_operation", "resource_operation", false, "loop_has_resource_operation", false),
  ("break_or_continue", "break_or_continue", false, "loop_has_break_or_continue", false),
  ("error_control_flow", "error_control_flow", false, "loop_has_error_control_flow", false),
  ("nested_loop", "nested_loop", false, "loop_has_nested_loop", false),
  ("branching_body", "branching_body", false, "loop_has_branching_body", false),
  ("missing_guard", "missing_guard", false, "loop_guard_missing", false),
  ("missing_invariant", "missing_invariant", false, "loop_invariant_missing", false),
  ("unsupported_kind", "unsupported_kind", false, "loop_kind_unsupported", false),
  ("non_u256_variable", "non_u256_variable", false, "loop_variable_not_u256", false),
  ("bad_update_target", "bad_update_target", false, "loop_update_target_not_loop_variable", false),
  ("unsupported_formula", "unsupported_formula", false, "loop_formula_unsupported", false),
  ("identity_collision", "identity_collision", false, "loop_identity_missing", false),
  ("query_owner_mismatch", "query_owner_mismatch", false, "loop_summary_query_mismatch", true),
  ("query_id_mismatch", "query_id_mismatch", false, "loop_summary_query_mismatch", true)
]

end Ora.Generated

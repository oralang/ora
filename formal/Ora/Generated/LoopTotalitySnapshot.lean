/-
GENERATED — DATA ONLY. The compiler emits the scalar-loop support
matrix; trusted checks live in Ora/LoopTotalitySync.lean.
Regenerate with scripts/check-formal-sync.sh.
-/

namespace Ora.Generated

def loopTotalityRows : List (String × String × Bool × String × Bool) := [
  ("supported_while", "supported_while", true, "", true),
  ("supported_for", "supported_for", true, "", true),
  ("supported_u8", "supported_u8", true, "", true),
  ("supported_u16", "supported_u16", true, "", true),
  ("supported_u32", "supported_u32", true, "", true),
  ("supported_u64", "supported_u64", true, "", true),
  ("supported_u128", "supported_u128", true, "", true),
  ("supported_u160", "supported_u160", true, "", true),
  ("supported_i8", "supported_i8", true, "", true),
  ("supported_i16", "supported_i16", true, "", true),
  ("supported_i32", "supported_i32", true, "", true),
  ("supported_i64", "supported_i64", true, "", true),
  ("supported_i128", "supported_i128", true, "", true),
  ("supported_i256", "supported_i256", true, "", true),
  ("storage_write", "storage_write", false, "loop_has_storage_write", false),
  ("storage_read", "storage_read", false, "loop_has_storage_read", false),
  ("call", "call", false, "loop_has_call", false),
  ("external_call", "external_call", false, "loop_has_external_call", false),
  ("resource_operation", "resource_operation", false, "loop_has_resource_operation", false),
  ("break_or_continue", "break_or_continue", false, "loop_has_break_or_continue", false),
  ("error_control_flow", "error_control_flow", false, "loop_has_error_control_flow", false),
  ("nested_loop", "nested_loop", false, "loop_has_nested_loop", false),
  ("branching_body", "branching_body", false, "loop_has_branching_body", false),
  ("missing_guard", "missing_guard", false, "loop_guard_missing", false),
  ("missing_invariant", "missing_invariant", false, "loop_invariant_missing", false),
  ("missing_body_safety", "missing_body_safety", false, "loop_body_safety_missing", false),
  ("unsupported_kind", "unsupported_kind", false, "loop_kind_unsupported", false),
  ("unregistered_integer_variable", "unregistered_integer_variable", false, "loop_variable_not_registered_integer", false),
  ("bad_update_target", "bad_update_target", false, "loop_state_binding_foreign", false),
  ("unsupported_formula", "unsupported_formula", false, "loop_formula_unsupported", false),
  ("identity_collision", "identity_collision", false, "loop_identity_missing", false),
  ("query_owner_mismatch", "query_owner_mismatch", false, "loop_summary_query_mismatch", true),
  ("query_id_mismatch", "query_id_mismatch", false, "loop_summary_query_mismatch", true)
]

def loopTimelineEventTags : List String := ["path_assumption", "safety_checkpoint", "storage_read", "storage_write", "summary_dependency"]

def loopStorageProjectionTags : List String := ["field", "fixed_index", "mapping_key"]

def loopTimelineValidationRows : List (String × String) := [
  ("supported", ""),
  ("timeline_missing", "loop_timeline_missing"),
  ("event_order_invalid", "loop_timeline_event_order_invalid"),
  ("event_identity_missing", "loop_timeline_event_identity_missing"),
  ("event_identity_duplicate", "loop_timeline_event_identity_duplicate"),
  ("path_missing", "loop_timeline_path_missing"),
  ("storage_root_missing", "loop_storage_root_missing"),
  ("storage_projection_invalid", "loop_storage_projection_invalid"),
  ("storage_type_invalid", "loop_storage_type_invalid"),
  ("read_identity_duplicate", "loop_storage_read_identity_duplicate"),
  ("read_identity_foreign", "loop_storage_read_identity_foreign"),
  ("read_identity_forward", "loop_storage_read_identity_forward"),
  ("child_footprint_missing", "loop_storage_child_footprint_missing"),
  ("dependency_order_invalid", "loop_summary_dependency_order_invalid"),
  ("dependency_target_unsupported", "loop_summary_dependency_target_unsupported"),
  ("body_safety_missing", "loop_body_safety_missing")
]

def loopPreheaderValidationRows : List (String × String) := [
  ("supported", ""),
  ("read_identity_forward", "loop_storage_read_identity_forward"),
  ("dependency_unsupported", "loop_preheader_storage_relation_unsupported")
]

def loopFixedIndexBoundsRows : List (String × String) := [
  ("supported", ""),
  ("extent_invalid", "loop_fixed_index_extent_invalid"),
  ("checkpoint_missing", "loop_fixed_index_bounds_checkpoint_missing"),
  ("checkpoint_late", "loop_fixed_index_bounds_checkpoint_missing"),
  ("checkpoint_wrong_shape", "loop_fixed_index_bounds_formula_shape_mismatch"),
  ("checkpoint_wrong_index", "loop_fixed_index_bounds_index_identity_mismatch"),
  ("checkpoint_wrong_type", "loop_fixed_index_bounds_type_mismatch"),
  ("checkpoint_inconsistent", "loop_fixed_index_bounds_extent_literal_mismatch"),
  ("checkpoint_noncanonical_literal", "loop_fixed_index_bounds_extent_literal_mismatch")
]

end Ora.Generated

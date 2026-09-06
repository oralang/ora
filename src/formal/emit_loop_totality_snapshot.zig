//! Emits the data-only scalar-loop support matrix consumed by
//! `formal/Ora/LoopTotalitySync.lean`.

const std = @import("std");
const formal = @import("ora_formal");

const obligation = formal.obligation;
const obligation_to_lean = formal.obligation_to_lean;
const builtin = formal.builtin;

const Mutation = enum {
    supported_while,
    supported_for,
    supported_u8,
    supported_u16,
    supported_u32,
    supported_u64,
    supported_u128,
    supported_u160,
    supported_i8,
    supported_i16,
    supported_i32,
    supported_i64,
    supported_i128,
    supported_i256,
    storage_write,
    storage_read,
    call,
    external_call,
    resource_operation,
    break_or_continue,
    error_control_flow,
    nested_loop,
    branching_body,
    missing_guard,
    missing_invariant,
    missing_body_safety,
    unsupported_kind,
    unregistered_integer_variable,
    bad_update_target,
    unsupported_formula,
    identity_collision,
    query_owner_mismatch,
    query_id_mismatch,
};

const cases = [_]Mutation{
    .supported_while,
    .supported_for,
    .supported_u8,
    .supported_u16,
    .supported_u32,
    .supported_u64,
    .supported_u128,
    .supported_u160,
    .supported_i8,
    .supported_i16,
    .supported_i32,
    .supported_i64,
    .supported_i128,
    .supported_i256,
    .storage_write,
    .storage_read,
    .call,
    .external_call,
    .resource_operation,
    .break_or_continue,
    .error_control_flow,
    .nested_loop,
    .branching_body,
    .missing_guard,
    .missing_invariant,
    .missing_body_safety,
    .unsupported_kind,
    .unregistered_integer_variable,
    .bad_update_target,
    .unsupported_formula,
    .identity_collision,
    .query_owner_mismatch,
    .query_id_mismatch,
};

const TimelineMutation = enum {
    supported,
    timeline_missing,
    event_order_invalid,
    event_identity_missing,
    event_identity_duplicate,
    path_missing,
    storage_root_missing,
    storage_projection_invalid,
    storage_type_invalid,
    read_identity_duplicate,
    read_identity_foreign,
    read_identity_forward,
    child_footprint_missing,
    dependency_order_invalid,
    dependency_target_unsupported,
    body_safety_missing,
};

const timeline_cases = std.meta.tags(TimelineMutation);

// Common order, identity, duplicate, location, and read-reference mutations
// are pinned above against the same `validateTimeline` core used by preheaders.
// If edge and preheader validation are ever split, duplicate those rows here;
// these cases cover only the preheader-specific policy differences.
const PreheaderMutation = enum {
    supported,
    read_identity_forward,
    dependency_unsupported,
};

const preheader_cases = std.meta.tags(PreheaderMutation);

const FixedIndexMutation = enum {
    supported,
    extent_invalid,
    checkpoint_missing,
    checkpoint_late,
    checkpoint_wrong_shape,
    checkpoint_wrong_index,
    checkpoint_wrong_type,
    checkpoint_inconsistent,
    checkpoint_noncanonical_literal,
};

const FixedIndexTimelineShape = enum { normal, missing, late };
const FixedIndexFormulaShape = enum { canonical, wrong_operator, wrong_index, wrong_type };

const FixedIndexFixtureConfig = struct {
    literal_value: []const u8 = "4",
    extent: u64 = 4,
    timeline_shape: FixedIndexTimelineShape = .normal,
    formula_shape: FixedIndexFormulaShape = .canonical,
};

fn fixedIndexFixtureConfig(mutation: FixedIndexMutation) FixedIndexFixtureConfig {
    return switch (mutation) {
        .supported => .{},
        .extent_invalid => .{ .extent = 0 },
        .checkpoint_missing => .{ .timeline_shape = .missing },
        .checkpoint_late => .{ .timeline_shape = .late },
        .checkpoint_wrong_shape => .{ .formula_shape = .wrong_operator },
        .checkpoint_wrong_index => .{ .formula_shape = .wrong_index },
        .checkpoint_wrong_type => .{ .formula_shape = .wrong_type },
        .checkpoint_inconsistent => .{ .extent = 3 },
        .checkpoint_noncanonical_literal => .{ .literal_value = "04" },
    };
}

const fixed_index_cases = std.meta.tags(FixedIndexMutation);

const ty_u256: obligation.TypeRef = .{
    .compiler_type_id = builtin.lookupBuiltinById(.u256).comptime_type_id,
};
const ty_u8: obligation.TypeRef = .{ .compiler_type_id = builtin.lookupBuiltinById(.u8).comptime_type_id };
const ty_u16: obligation.TypeRef = .{ .compiler_type_id = builtin.lookupBuiltinById(.u16).comptime_type_id };
const ty_u32: obligation.TypeRef = .{ .compiler_type_id = builtin.lookupBuiltinById(.u32).comptime_type_id };
const ty_u64: obligation.TypeRef = .{ .compiler_type_id = builtin.lookupBuiltinById(.u64).comptime_type_id };
const ty_u128: obligation.TypeRef = .{ .compiler_type_id = builtin.lookupBuiltinById(.u128).comptime_type_id };
const ty_u160: obligation.TypeRef = .{ .compiler_type_id = builtin.lookupBuiltinById(.u160).comptime_type_id };
const ty_i8: obligation.TypeRef = .{ .compiler_type_id = builtin.lookupBuiltinById(.i8).comptime_type_id };
const ty_i16: obligation.TypeRef = .{ .compiler_type_id = builtin.lookupBuiltinById(.i16).comptime_type_id };
const ty_i32: obligation.TypeRef = .{ .compiler_type_id = builtin.lookupBuiltinById(.i32).comptime_type_id };
const ty_i64: obligation.TypeRef = .{ .compiler_type_id = builtin.lookupBuiltinById(.i64).comptime_type_id };
const ty_i128: obligation.TypeRef = .{ .compiler_type_id = builtin.lookupBuiltinById(.i128).comptime_type_id };
const ty_i256: obligation.TypeRef = .{ .compiler_type_id = builtin.lookupBuiltinById(.i256).comptime_type_id };
const ty_unregistered_integer: obligation.TypeRef = .{
    .compiler_type_id = 999,
};
const loop_id: obligation.FreeVarId = .{ .file_id = 10, .pattern_id = 1 };
const other_id: obligation.FreeVarId = .{ .file_id = 10, .pattern_id = 2 };
const query_id: obligation.Id = 200;
const summary_id: obligation.Id = 100;

const init_formulas = [_]obligation.FormulaRef{.{ .term = 0 }};
const invariant_formulas = [_]obligation.FormulaRef{.{ .term = 1 }};
const body_safety_formulas = [_]obligation.FormulaRef{.{ .term = 1 }};
const post_formulas = [_]obligation.FormulaRef{.{ .term = 1 }};
const induction_query_ids = [_]obligation.Id{query_id};

fn writeLeanString(writer: anytype, value: []const u8) !void {
    try writer.writeByte('"');
    for (value) |byte| switch (byte) {
        '\\' => try writer.writeAll("\\\\"),
        '"' => try writer.writeAll("\\\""),
        '\n' => try writer.writeAll("\\n"),
        '\r' => try writer.writeAll("\\r"),
        '\t' => try writer.writeAll("\\t"),
        else => try writer.writeByte(byte),
    };
    try writer.writeByte('"');
}

fn writeTagNames(writer: anytype, comptime T: type) !void {
    try writer.writeByte('[');
    inline for (std.meta.tags(T), 0..) |tag, index| {
        if (index != 0) try writer.writeAll(", ");
        try writeLeanString(writer, @tagName(tag));
    }
    try writer.writeByte(']');
}

fn reasonName(support: obligation_to_lean.SemanticSupport) []const u8 {
    return switch (support) {
        .supported => "",
        .unsupported => |reason| switch (reason) {
            .unsupported_loop_summary => |loop_reason| @tagName(loop_reason),
            else => @tagName(reason),
        },
    };
}

fn unsupportedReason(mutation: Mutation) ?obligation.LoopUnsupportedReason {
    return switch (mutation) {
        .storage_write => .loop_has_storage_write,
        .storage_read => .loop_has_storage_read,
        .call => .loop_has_call,
        .external_call => .loop_has_external_call,
        .resource_operation => .loop_has_resource_operation,
        .break_or_continue => .loop_has_break_or_continue,
        .error_control_flow => .loop_has_error_control_flow,
        .nested_loop => .loop_has_nested_loop,
        .branching_body => .loop_has_branching_body,
        .missing_guard => .loop_guard_missing,
        .missing_invariant => .loop_invariant_missing,
        .missing_body_safety => .loop_body_safety_missing,
        .unsupported_kind => .loop_kind_unsupported,
        else => null,
    };
}

fn leanDenotable(mutation: Mutation) bool {
    return switch (mutation) {
        .supported_while,
        .supported_for,
        .supported_u8,
        .supported_u16,
        .supported_u32,
        .supported_u64,
        .supported_u128,
        .supported_u160,
        .supported_i8,
        .supported_i16,
        .supported_i32,
        .supported_i64,
        .supported_i128,
        .supported_i256,
        .query_owner_mismatch,
        .query_id_mismatch,
        => true,
        else => false,
    };
}

fn integerTypeFor(mutation: Mutation) obligation.TypeRef {
    return switch (mutation) {
        .supported_u8 => ty_u8,
        .supported_u16 => ty_u16,
        .supported_u32 => ty_u32,
        .supported_u64 => ty_u64,
        .supported_u128 => ty_u128,
        .supported_u160 => ty_u160,
        .supported_i8 => ty_i8,
        .supported_i16 => ty_i16,
        .supported_i32 => ty_i32,
        .supported_i64 => ty_i64,
        .supported_i128 => ty_i128,
        .supported_i256 => ty_i256,
        .unregistered_integer_variable => ty_unregistered_integer,
        else => ty_u256,
    };
}

fn supportFor(mutation: Mutation) obligation_to_lean.SemanticSupport {
    const integer_ty = integerTypeFor(mutation);
    const terms = [_]obligation.Term{
        .{ .int_lit = .{ .value = "0", .ty = integer_ty } },
        .{ .bool_lit = true },
    };
    var variables = [_]obligation.LoopVariable{.{
        .index = 0,
        .id = loop_id,
        .name = "i",
        .ty = integer_ty,
    }};
    var context_variables = [_]obligation.LoopVariable{.{
        .index = 0,
        .id = loop_id,
        .name = "collision",
        .ty = ty_u256,
    }};
    const assignments = [_]obligation.LoopStepAssignment{.{
        .variable_index = 0,
        .target = if (mutation == .bad_update_target) other_id else loop_id,
        .source_key = "body/straight-line-yield/operand:0",
        .value = .{ .term = 0 },
    }};
    const exit_assignments = [_]obligation.LoopStepAssignment{.{
        .variable_index = 0,
        .target = loop_id,
        .source_key = "loop-state:0",
        .value = .{ .term = 0 },
    }};
    const invariant_path_exprs = [_]obligation.LoopFormulaExpr{.{ .formula = .{ .term = 1 } }};
    const unsupported_path_exprs = [_]obligation.LoopFormulaExpr{.{ .formula = .{ .term = 0 } }};
    const backedge_path_exprs: []const obligation.LoopFormulaExpr = if (mutation == .missing_guard)
        &.{}
    else if (mutation == .unsupported_formula)
        &unsupported_path_exprs
    else
        &invariant_path_exprs;
    const safety_exprs = [_]obligation.LoopFormulaExpr{.{ .formula = .{ .term = 1 } }};
    const backedge_timeline_with_safety = [_]obligation.LoopTimelineEvent{
        .{
            .id = 0,
            .structural_key = "body/straight-line-yield/path",
            .payload = .{ .path_assumption = .{ .formulas = backedge_path_exprs } },
        },
        .{
            .id = 1,
            .structural_key = "body/ora.assert:0",
            .payload = .{ .safety_checkpoint = .{ .formulas = &safety_exprs } },
        },
    };
    const backedge_timeline_without_safety = [_]obligation.LoopTimelineEvent{.{
        .id = 0,
        .structural_key = "body/straight-line-yield/path",
        .payload = .{ .path_assumption = .{ .formulas = backedge_path_exprs } },
    }};
    const backedge_timeline: []const obligation.LoopTimelineEvent = if (mutation == .missing_body_safety)
        &backedge_timeline_without_safety
    else
        &backedge_timeline_with_safety;
    const exit_timeline = [_]obligation.LoopTimelineEvent{.{
        .id = 0,
        .structural_key = "guard/false-successor/path",
        .payload = .{ .path_assumption = .{ .formulas = &invariant_path_exprs } },
    }};
    const edges = [_]obligation.LoopEdgeRow{
        .{
            .id = 0,
            .structural_key = "body/straight-line-yield",
            .timeline = backedge_timeline,
            .next_state = assignments[0..],
            .target = .backedge,
        },
        .{
            .id = 1,
            .structural_key = "guard/false-successor",
            .timeline = &exit_timeline,
            .next_state = exit_assignments[0..],
            .target = .successor,
        },
    };
    var reasons: [1]obligation.LoopUnsupportedReason = undefined;
    const reason_slice: []const obligation.LoopUnsupportedReason = if (unsupportedReason(mutation)) |reason| blk: {
        reasons[0] = reason;
        break :blk reasons[0..];
    } else &.{};

    const summary: obligation.LoopSummaryRow = .{
        .id = summary_id,
        .owner = .{ .statement = .{ .function_name = "run", .ordinal = 0 } },
        .source = .generated(),
        .phase = .report,
        .origin = .source,
        .loop_source_op_id = 0,
        .loop_kind = if (mutation == .supported_for) .scf_for else if (mutation == .unsupported_kind) .other else .scf_while,
        .context_variables = if (mutation == .identity_collision) context_variables[0..] else &.{},
        .variables = variables[0..],
        .init_formulas = &init_formulas,
        .invariant_formulas = if (mutation == .missing_invariant) &.{} else &invariant_formulas,
        .edges = &edges,
        .projection_completeness = .complete,
        .post_formulas = &post_formulas,
        .query_ids = .{ .induction = if (mutation == .query_id_mismatch) &.{} else &induction_query_ids },
        .unsupported_reasons = reason_slice,
    };
    const query: obligation.VerificationQuery = .{
        .id = query_id,
        .owner = if (mutation == .query_owner_mismatch)
            .{ .function = .{ .name = "other" } }
        else
            .{ .function = .{ .name = "run" } },
        .source = .generated(),
        .phase = .report,
        .origin = .source,
        .kind = .loop_induction,
        .loop_summary_id = summary_id,
    };
    const summaries = [_]obligation.LoopSummaryRow{summary};
    const set: obligation.ObligationSet = .{
        .terms = &terms,
        .loop_summaries = &summaries,
    };
    return obligation_to_lean.querySemanticSupport(set, query);
}

fn timelineReason(mutation: TimelineMutation) ?obligation.LoopUnsupportedReason {
    const path_exprs = [_]obligation.LoopFormulaExpr{.{ .formula = .{ .term = 0 } }};
    const dependency: obligation.LoopSummaryDependency = .{
        .id = 0,
        .structural_key = "body/child",
        .summary_id = 2,
        .loop_vc_sha256 = [_]u8{0} ** 32,
        .write_footprint_complete = true,
        .target = .successor,
    };
    const read_id: obligation.FreeVarId = .{ .file_id = 10, .pattern_id = 90 };
    const foreign_id: obligation.FreeVarId = .{ .file_id = 10, .pattern_id = 91 };
    const read_refs = [_]obligation.FreeVarId{read_id};
    const foreign_refs = [_]obligation.FreeVarId{foreign_id};
    const location: obligation.LoopStorageLocationExpr = .{ .root = "counter", .value_ty = ty_u256 };
    const path_event: obligation.LoopTimelineEvent = .{
        .id = 0,
        .structural_key = "body/path",
        .payload = .{ .path_assumption = .{ .formulas = &path_exprs } },
    };
    const read_event: obligation.LoopTimelineEvent = .{
        .id = 1,
        .structural_key = "body/read",
        .payload = .{ .storage_read = .{ .result = read_id, .location = location } },
    };
    const write_event: obligation.LoopTimelineEvent = .{
        .id = 2,
        .structural_key = "body/write",
        .payload = .{ .storage_write = .{
            .location = location,
            .value = .{ .formula = .{ .term = 0 }, .read_results = &read_refs },
        } },
    };
    const safety_event: obligation.LoopTimelineEvent = .{
        .id = 3,
        .structural_key = "body/safety",
        .payload = .{ .safety_checkpoint = .{ .formulas = &path_exprs } },
    };
    const dependency_event: obligation.LoopTimelineEvent = .{
        .id = 4,
        .structural_key = "body/child",
        .payload = .{ .summary_dependency = dependency },
    };
    var timeline = [_]obligation.LoopTimelineEvent{
        path_event,
        read_event,
        write_event,
        safety_event,
        dependency_event,
    };
    const no_path_timeline = [_]obligation.LoopTimelineEvent{.{
        .id = 0,
        .structural_key = "body/safety",
        .payload = .{ .safety_checkpoint = .{ .formulas = &path_exprs } },
    }};
    const forward_timeline = [_]obligation.LoopTimelineEvent{
        path_event,
        .{
            .id = 1,
            .structural_key = "body/write",
            .payload = write_event.payload,
        },
        .{
            .id = 2,
            .structural_key = "body/read",
            .payload = read_event.payload,
        },
        safety_event,
        dependency_event,
    };
    var timeline_slice: []const obligation.LoopTimelineEvent = &timeline;
    switch (mutation) {
        .supported => {},
        .timeline_missing => timeline_slice = &.{},
        .event_order_invalid => timeline[1].id = 7,
        .event_identity_missing => timeline[1].structural_key = "",
        .event_identity_duplicate => timeline[1].structural_key = timeline[0].structural_key,
        .path_missing => timeline_slice = &no_path_timeline,
        .storage_root_missing => timeline[1].payload.storage_read.location.root = "",
        .storage_projection_invalid => timeline[1].payload.storage_read.location.projections = &.{.{ .field = "" }},
        .storage_type_invalid => timeline[1].payload.storage_read.location.value_ty = .{ .compiler_type_id = 0 },
        .read_identity_duplicate => timeline[1].payload.storage_read.result = loop_id,
        .read_identity_foreign => timeline[2].payload.storage_write.value.read_results = &foreign_refs,
        .read_identity_forward => timeline_slice = &forward_timeline,
        .child_footprint_missing => timeline[4].payload.summary_dependency.write_footprint_complete = false,
        .dependency_order_invalid => timeline[4].payload.summary_dependency.id = 1,
        .dependency_target_unsupported => timeline[4].payload.summary_dependency.target = .backedge,
        .body_safety_missing => timeline[3].payload.safety_checkpoint.formulas = &.{},
    }

    const binding = [_]obligation.LoopStepAssignment{.{
        .variable_index = 0,
        .target = loop_id,
        .source_key = "loop-state:0",
        .value = .{ .term = 0 },
        .read_results = &read_refs,
    }};
    const exit_timeline = [_]obligation.LoopTimelineEvent{.{
        .id = 0,
        .structural_key = "exit/path",
        .payload = .{ .path_assumption = .{ .formulas = &path_exprs } },
    }};
    const edges = [_]obligation.LoopEdgeRow{
        .{
            .id = 0,
            .structural_key = "body",
            .timeline = timeline_slice,
            .next_state = &binding,
            .target = .backedge,
        },
        .{
            .id = 1,
            .structural_key = "exit",
            .timeline = &exit_timeline,
            .next_state = &.{.{
                .variable_index = 0,
                .target = loop_id,
                .source_key = "loop-state:0",
                .value = .{ .term = 0 },
            }},
            .target = .successor,
        },
    };
    const variables = [_]obligation.LoopVariable{.{
        .index = 0,
        .id = loop_id,
        .name = "i",
        .ty = ty_u256,
    }};
    const summary: obligation.LoopSummaryRow = .{
        .id = summary_id,
        .owner = .{ .statement = .{ .function_name = "run", .ordinal = 0 } },
        .source = .generated(),
        .phase = .report,
        .origin = .source,
        .loop_source_op_id = 0,
        .loop_kind = .scf_while,
        .variables = &variables,
        .edges = &edges,
        .projection_completeness = .complete,
    };
    return summary.validationReason();
}

fn preheaderReason(mutation: PreheaderMutation) ?obligation.LoopUnsupportedReason {
    const formula = [_]obligation.LoopFormulaExpr{.{ .formula = .{ .term = 0 } }};
    const read_id: obligation.FreeVarId = .{ .file_id = 10, .pattern_id = 90 };
    const read_refs = [_]obligation.FreeVarId{read_id};
    const location: obligation.LoopStorageLocationExpr = .{ .root = "counter", .value_ty = ty_u256 };
    var events = [_]obligation.LoopTimelineEvent{
        .{ .id = 0, .structural_key = "preheader/read", .payload = .{ .storage_read = .{ .result = read_id, .location = location } } },
        .{ .id = 1, .structural_key = "preheader/write", .payload = .{ .storage_write = .{ .location = location, .value = .{ .formula = .{ .term = 0 }, .read_results = &read_refs } } } },
        .{ .id = 2, .structural_key = "preheader/check", .payload = .{ .safety_checkpoint = .{ .formulas = &formula } } },
    };
    switch (mutation) {
        .supported => {},
        .read_identity_forward => std.mem.swap(obligation.LoopTimelineEvent, &events[0], &events[1]),
        .dependency_unsupported => events[0].payload = .{ .summary_dependency = .{
            .id = 0,
            .structural_key = "preheader/child",
            .summary_id = 2,
            .loop_vc_sha256 = [_]u8{0} ** 32,
            .write_footprint_complete = true,
            .target = .successor,
        } },
    }
    for (&events, 0..) |*event, index| event.id = @intCast(index);
    const variables = [_]obligation.LoopVariable{.{ .index = 0, .id = loop_id, .ty = ty_u256 }};
    return (obligation.LoopSummaryRow{
        .id = summary_id,
        .owner = .{ .function = .{ .name = "run" } },
        .source = .generated(),
        .phase = .report,
        .origin = .source,
        .loop_source_op_id = 0,
        .loop_kind = .scf_while,
        .variables = &variables,
        .preheader_timeline = &events,
    }).preheaderTimelineValidationReason();
}

fn fixedIndexReason(mutation: FixedIndexMutation) ?obligation.LoopUnsupportedReason {
    const config = fixedIndexFixtureConfig(mutation);
    const index_id: obligation.FreeVarId = .{ .file_id = 10, .pattern_id = 7 };
    const terms = [_]obligation.Term{
        .{ .variable = .{ .free = .{ .id = index_id, .name = "i", .ty = ty_u256 } } },
        .{ .int_lit = .{
            .value = config.literal_value,
            .ty = ty_u256,
        } },
        .{ .binary = .{
            .op = if (config.formula_shape == .wrong_operator) .le else .lt,
            .lhs = if (config.formula_shape == .wrong_index) 1 else 0,
            .rhs = 1,
            .ty = if (config.formula_shape == .wrong_type) ty_u8 else ty_u256,
        } },
    };
    const formulas = [_]obligation.LoopFormulaExpr{.{ .formula = .{ .term = 2 } }};
    const projection = [_]obligation.LoopStorageProjection{.{ .fixed_index = .{
        .index = .{ .value = .{ .formula = .{ .term = 0 } }, .ty = ty_u256 },
        .extent = config.extent,
    } }};
    const checkpoint: obligation.LoopTimelineEvent = .{
        .id = 0,
        .structural_key = "body/check",
        .payload = .{ .safety_checkpoint = .{ .formulas = &formulas } },
    };
    const write: obligation.LoopTimelineEvent = .{
        .id = 1,
        .structural_key = "body/write",
        .payload = .{ .storage_write = .{
            .location = .{ .root = "values", .projections = &projection, .value_ty = ty_u256 },
            .value = .{ .formula = .{ .term = 0 } },
        } },
    };
    const events: []const obligation.LoopTimelineEvent = switch (config.timeline_shape) {
        .missing => &.{write},
        .late => &.{ write, checkpoint },
        .normal => &.{ checkpoint, write },
    };
    return (obligation.ObligationSet{ .terms = &terms }).fixedIndexBoundsReason(events);
}

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    var stdout_buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &stdout_buffer);
    const writer = &stdout_writer.interface;

    try writer.writeAll(
        \\/-
        \\GENERATED — DATA ONLY. The compiler emits the scalar-loop support
        \\matrix; trusted checks live in Ora/LoopTotalitySync.lean.
        \\Regenerate with scripts/check-formal-sync.sh.
        \\-/
        \\
        \\namespace Ora.Generated
        \\
        \\def loopTotalityRows : List (String × String × Bool × String × Bool) := [
    );
    try writer.writeByte('\n');
    for (cases, 0..) |mutation, index| {
        const support = supportFor(mutation);
        try writer.writeAll("  (");
        try writeLeanString(writer, @tagName(mutation));
        try writer.writeAll(", ");
        try writeLeanString(writer, @tagName(mutation));
        try writer.writeAll(if (support == .supported) ", true, " else ", false, ");
        try writeLeanString(writer, reasonName(support));
        try writer.writeAll(if (leanDenotable(mutation)) ", true)" else ", false)");
        try writer.writeAll(if (index + 1 == cases.len) "\n" else ",\n");
    }
    try writer.writeAll("]\n\n");
    try writer.writeAll("def loopTimelineEventTags : List String := ");
    try writeTagNames(writer, obligation.LoopTimelineEventTag);
    try writer.writeAll("\n\ndef loopStorageProjectionTags : List String := ");
    try writeTagNames(writer, obligation.LoopStorageProjectionTag);
    try writer.writeAll("\n\ndef loopTimelineValidationRows : List (String × String) := [\n");
    for (timeline_cases, 0..) |mutation, index| {
        try writer.writeAll("  (");
        try writeLeanString(writer, @tagName(mutation));
        try writer.writeAll(", ");
        if (timelineReason(mutation)) |reason| {
            try writeLeanString(writer, @tagName(reason));
        } else {
            try writeLeanString(writer, "");
        }
        try writer.writeAll(if (index + 1 == timeline_cases.len) ")\n" else "),\n");
    }
    try writer.writeAll("]\n\ndef loopPreheaderValidationRows : List (String × String) := [\n");
    for (preheader_cases, 0..) |mutation, index| {
        try writer.writeAll("  (");
        try writeLeanString(writer, @tagName(mutation));
        try writer.writeAll(", ");
        try writeLeanString(writer, if (preheaderReason(mutation)) |reason| @tagName(reason) else "");
        try writer.writeAll(if (index + 1 == preheader_cases.len) ")\n" else "),\n");
    }
    try writer.writeAll("]\n\ndef loopFixedIndexBoundsRows : List (String × String) := [\n");
    for (fixed_index_cases, 0..) |mutation, index| {
        try writer.writeAll("  (");
        try writeLeanString(writer, @tagName(mutation));
        try writer.writeAll(", ");
        try writeLeanString(writer, if (fixedIndexReason(mutation)) |reason| @tagName(reason) else "");
        try writer.writeAll(if (index + 1 == fixed_index_cases.len) ")\n" else "),\n");
    }
    try writer.writeAll("]\n\nend Ora.Generated\n");
    try writer.flush();
}

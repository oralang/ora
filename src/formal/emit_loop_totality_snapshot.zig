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
    storage_write,
    external_call,
    resource_operation,
    break_or_continue,
    error_control_flow,
    nested_loop,
    branching_body,
    missing_guard,
    missing_invariant,
    unsupported_kind,
    non_u256_variable,
    bad_update_target,
    unsupported_formula,
    identity_collision,
    query_owner_mismatch,
    query_id_mismatch,
};

const cases = [_]Mutation{
    .supported_while,
    .supported_for,
    .storage_write,
    .external_call,
    .resource_operation,
    .break_or_continue,
    .error_control_flow,
    .nested_loop,
    .branching_body,
    .missing_guard,
    .missing_invariant,
    .unsupported_kind,
    .non_u256_variable,
    .bad_update_target,
    .unsupported_formula,
    .identity_collision,
    .query_owner_mismatch,
    .query_id_mismatch,
};

const ty_u256: obligation.TypeRef = .{
    .compiler_type_id = builtin.lookupBuiltinById(.u256).comptime_type_id,
};
const ty_u128: obligation.TypeRef = .{
    .compiler_type_id = builtin.lookupBuiltinById(.u128).comptime_type_id,
};
const loop_id: obligation.FreeVarId = .{ .file_id = 10, .pattern_id = 1 };
const other_id: obligation.FreeVarId = .{ .file_id = 10, .pattern_id = 2 };
const query_id: obligation.Id = 200;
const summary_id: obligation.Id = 100;

const terms = [_]obligation.Term{
    .{ .int_lit = .{ .value = "0", .ty = ty_u256 } },
    .{ .bool_lit = true },
};
const init_formulas = [_]obligation.FormulaRef{.{ .term = 0 }};
const invariant_formulas = [_]obligation.FormulaRef{.{ .term = 1 }};
const post_formulas = [_]obligation.FormulaRef{.{ .term = 1 }};
const post_query_ids = [_]obligation.Id{query_id};

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
        .external_call => .loop_has_external_call,
        .resource_operation => .loop_has_resource_operation,
        .break_or_continue => .loop_has_break_or_continue,
        .error_control_flow => .loop_has_error_control_flow,
        .nested_loop => .loop_has_nested_loop,
        .branching_body => .loop_has_branching_body,
        .missing_guard => .loop_guard_missing,
        .missing_invariant => .loop_invariant_missing,
        .unsupported_kind => .loop_kind_unsupported,
        else => null,
    };
}

fn leanDenotable(mutation: Mutation) bool {
    return switch (mutation) {
        .supported_while, .supported_for, .query_owner_mismatch, .query_id_mismatch => true,
        else => false,
    };
}

fn supportFor(mutation: Mutation) obligation_to_lean.SemanticSupport {
    var variables = [_]obligation.LoopVariable{.{
        .index = 0,
        .id = loop_id,
        .name = "i",
        .ty = if (mutation == .non_u256_variable) ty_u128 else ty_u256,
    }};
    var context_variables = [_]obligation.LoopVariable{.{
        .index = 0,
        .id = loop_id,
        .name = "collision",
        .ty = ty_u256,
    }};
    var assignments = [_]obligation.LoopStepAssignment{.{
        .variable_index = 0,
        .target = if (mutation == .bad_update_target) other_id else loop_id,
        .value = .{ .term = 0 },
    }};
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
        .guard_formula = if (mutation == .missing_guard) null else if (mutation == .unsupported_formula) .{ .term = 0 } else .{ .term = 1 },
        .invariant_formulas = if (mutation == .missing_invariant) &.{} else &invariant_formulas,
        .step_assignments = assignments[0..],
        .post_formulas = &post_formulas,
        .query_ids = .{ .post = if (mutation == .query_id_mismatch) &.{} else &post_query_ids },
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
        .kind = .loop_invariant_post,
        .loop_summary_id = summary_id,
    };
    const summaries = [_]obligation.LoopSummaryRow{summary};
    const set: obligation.ObligationSet = .{
        .terms = &terms,
        .loop_summaries = &summaries,
    };
    return obligation_to_lean.querySemanticSupport(set, query);
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
    try writer.writeAll("]\n\nend Ora.Generated\n");
    try writer.flush();
}

//! Shared constructors for formal-lane in-file tests.
//!
//! Keep this module test-shaped: it should remove boilerplate from fixtures,
//! not become a production manifest builder.

const obligation = @import("obligation.zig");

pub fn generatedFunctionOwner(function_name: []const u8) obligation.Owner {
    return .{ .function = .{ .name = function_name } };
}

pub fn logicalEnsuresObligation(
    id: obligation.Id,
    function_name: []const u8,
    target_term: obligation.TermId,
) obligation.Obligation {
    return .{
        .id = id,
        .owner = generatedFunctionOwner(function_name),
        .source = .generated(),
        .phase = .sema,
        .origin = .{ .sema_fact = .{ .kind = "ensures", .ordinal = 0 } },
        .kind = .{ .logical = .{ .role = .ensures, .formula = .{ .term = target_term } } },
    };
}

pub fn assumeAssumption(
    id: obligation.Id,
    function_name: []const u8,
    term: obligation.TermId,
) obligation.Assumption {
    return .{
        .id = id,
        .owner = generatedFunctionOwner(function_name),
        .source = .generated(),
        .phase = .sema,
        .origin = .{ .sema_fact = .{ .kind = "assume", .ordinal = 0 } },
        .kind = .assume,
        .formula = .{ .term = term },
    };
}

pub fn obligationQuery(
    id: obligation.Id,
    function_name: []const u8,
    obligation_ids: []const obligation.Id,
    assumption_ids: []const obligation.Id,
) obligation.VerificationQuery {
    return .{
        .id = id,
        .owner = generatedFunctionOwner(function_name),
        .source = .generated(),
        .phase = .report,
        .origin = .source,
        .kind = .obligation,
        .assumption_ids = assumption_ids,
        .obligation_ids = obligation_ids,
        .solver_logic = .all,
    };
}

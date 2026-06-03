const std = @import("std");

pub const FieldIssue = enum {
    missing,
    unknown,
    duplicate,
};

pub fn genericArity(
    allocator: std.mem.Allocator,
    kind: []const u8,
    name: []const u8,
    expected: usize,
    found: usize,
) ![]const u8 {
    return std.fmt.allocPrint(allocator, "generic {s} '{s}' expects {d} arguments, found {d}", .{
        kind,
        name,
        expected,
        found,
    });
}

pub fn invalidIntegerTypeName(
    allocator: std.mem.Allocator,
    name: []const u8,
    supported_names: []const u8,
) ![]const u8 {
    return std.fmt.allocPrint(allocator, "invalid integer type '{s}'; supported integer types are {s}", .{
        name,
        supported_names,
    });
}

pub fn expectedFound(
    allocator: std.mem.Allocator,
    subject: ?[]const u8,
    expected_noun: []const u8,
    expected: []const u8,
    found: []const u8,
) ![]const u8 {
    if (subject) |value| {
        return std.fmt.allocPrint(allocator, "{s} expects {s} '{s}', found '{s}'", .{
            value,
            expected_noun,
            expected,
            found,
        });
    }
    return std.fmt.allocPrint(allocator, "expected {s} '{s}', found '{s}'", .{
        expected_noun,
        expected,
        found,
    });
}

pub fn constantValueDoesNotFit(
    allocator: std.mem.Allocator,
    value_text: []const u8,
    type_name: []const u8,
) ![]const u8 {
    return std.fmt.allocPrint(allocator, "constant value {s} does not fit in type '{s}'", .{
        value_text,
        type_name,
    });
}

pub fn constantValueDoesNotFitCastTarget(
    allocator: std.mem.Allocator,
    value_text: []const u8,
    type_name: []const u8,
) ![]const u8 {
    return std.fmt.allocPrint(allocator, "constant value {s} does not fit in cast target type '{s}'", .{
        value_text,
        type_name,
    });
}

pub fn adtVariantField(
    allocator: std.mem.Allocator,
    issue: FieldIssue,
    field_name: []const u8,
    variant_name: []const u8,
) ![]const u8 {
    const issue_word = switch (issue) {
        .missing => "missing",
        .unknown => "unknown",
        .duplicate => "duplicate",
    };
    return std.fmt.allocPrint(allocator, "{s} field '{s}' for ADT variant '{s}'", .{
        issue_word,
        field_name,
        variant_name,
    });
}

pub fn lockedWrite(
    allocator: std.mem.Allocator,
    region_name: []const u8,
    slot_name: []const u8,
) ![]const u8 {
    return std.fmt.allocPrint(allocator, "cannot write locked {s} slot '{s}'", .{
        region_name,
        slot_name,
    });
}

pub fn externalCallWriteAfterPreWrite(
    allocator: std.mem.Allocator,
    slot_name: []const u8,
) ![]const u8 {
    return std.fmt.allocPrint(allocator, "cannot write storage slot '{s}' after external call because it was written before the call", .{
        slot_name,
    });
}

pub fn modifiesMissingWrite(
    allocator: std.mem.Allocator,
    slot_display: []const u8,
) ![]const u8 {
    return std.fmt.allocPrint(allocator, "storage write to '{s}' is not covered by this function's `modifies` clause", .{
        slot_display,
    });
}

test "diagnostic message builders preserve sema wording" {
    const allocator = std.testing.allocator;

    const arity = try genericArity(allocator, "struct", "Pair", 1, 2);
    defer allocator.free(arity);
    try std.testing.expectEqualStrings("generic struct 'Pair' expects 1 arguments, found 2", arity);

    const expected = try expectedFound(allocator, "declaration", "type", "u8", "bool");
    defer allocator.free(expected);
    try std.testing.expectEqualStrings("declaration expects type 'u8', found 'bool'", expected);

    const argument = try expectedFound(allocator, null, "argument type", "u256", "bool");
    defer allocator.free(argument);
    try std.testing.expectEqualStrings("expected argument type 'u256', found 'bool'", argument);

    const fit = try constantValueDoesNotFit(allocator, "300", "u8");
    defer allocator.free(fit);
    try std.testing.expectEqualStrings("constant value 300 does not fit in type 'u8'", fit);

    const field = try adtVariantField(allocator, .duplicate, "code", "Named");
    defer allocator.free(field);
    try std.testing.expectEqualStrings("duplicate field 'code' for ADT variant 'Named'", field);

    const locked = try lockedWrite(allocator, "storage", "total");
    defer allocator.free(locked);
    try std.testing.expectEqualStrings("cannot write locked storage slot 'total'", locked);
}

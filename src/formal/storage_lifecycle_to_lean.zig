//! Data emission only. A digest string is not a kernel certificate or a grant.
const std = @import("std");
const builtin = @import("ora_types").builtin;
const data = @import("storage_lifecycle.zig");
const obligation = @import("obligation.zig");
const lean = @import("obligation_to_lean.zig");

pub fn writeProjection(writer: anytype, allocator: std.mem.Allocator, inventory: data.Inventory, set: obligation.ObligationSet, bindings: []const data.FunctionBinding) !void {
    try data.validateBindings(inventory, bindings);
    if (bindings.len != set.function_summaries.len) return error.MissingLifecycleBinding;
    for (bindings) |binding| {
        const row = obligation.findById(set.function_summaries, binding.function_id) orelse return error.ForeignLifecycleBinding;
        const function = for (inventory.functions) |candidate| {
            if (candidate.ordinal == binding.function_ordinal) break candidate;
        } else return error.ForeignLifecycleBinding;
        if (row.owner != .function or !std.mem.eql(u8, row.owner.function.name, function.symbol)) return error.ForeignLifecycleBinding;
        if (!std.mem.eql(u8, &(try lean.functionVcSha256(set, row.id)), &binding.function_vc_sha256)) return error.LifecycleDigestMismatch;
    }
    try lean.writeCallCustodyModule(writer, set, "Ora.Generated.LifecycleFunctions");
    try writer.print("\nnamespace Ora.Generated.LifecycleFunctions\ndef lifecycleInventorySha256 : String := \"{s}\"\ndef lifecycleBindings : List (Nat × Nat × String) := [", .{std.fmt.bytesToHex(try inventory.sha256(allocator), .lower)});
    for (bindings, 0..) |binding, index| {
        if (index != 0) try writer.writeAll(", ");
        try writer.print("({d}, {d}, \"{s}\")", .{ binding.function_ordinal, binding.function_id, std.fmt.bytesToHex(binding.function_vc_sha256, .lower) });
    }
    try writer.writeAll("]\nend Ora.Generated.LifecycleFunctions\n");
}

pub fn writeClaim(writer: anytype, allocator: std.mem.Allocator, inventory: data.Inventory, index: usize) !void {
    try inventory.validate();
    if (index >= inventory.claims.len) return error.MissingLifecycleClaim;
    const claim = inventory.claims[index];
    if (claim.refusal != null) return error.UnsupportedLifecycleClaim;
    const info = builtin.integerInfoByComptimeTypeId(claim.type_id.?) orelse return error.InvalidLifecycleType;
    const bytes = try inventory.canonical(allocator);
    defer allocator.free(bytes);
    const digest = try inventory.sha256(allocator);
    try writer.writeAll("import Ora.Storage.Lifecycle\nnamespace Ora.Generated.LifecycleInputs\nopen Ora.Obligation Ora.Loop\n");
    try writer.print("def inventorySha256 : String := \"{s}\"\n", .{std.fmt.bytesToHex(digest, .lower)});
    // Hex preserves every canonical byte, including paths and unsupported rows.
    try writer.writeAll("def inventoryBytesHex : String := \"");
    for (bytes) |byte| try writer.print("{x:0>2}", .{byte});
    try writer.writeAll("\"\n");
    try writer.print("def claimIndex : Nat := {d}\ndef candidateClaim : Ora.StorageLifecycle.Claim := {{\n  location := {{ root := ", .{index});
    const root = try std.fmt.allocPrint(allocator, "{s}/slot:{s}", .{ claim.owner, claim.root });
    defer allocator.free(root);
    try string(writer, root);
    try writer.writeAll(" }\n  name := ");
    try string(writer, @tagName(claim.bounds.?.kind));
    try writer.writeAll("\n  bounds := [");
    const bounds = claim.bounds.?;
    if (bounds.kind != .MaxValue) try value(writer, info, bounds.minimum);
    if (bounds.kind == .InRange) try writer.writeAll(", ");
    if (bounds.kind != .MinValue) try value(writer, info, bounds.maximum);
    try writer.writeAll("]\n}\ndef zeroAtClaim : Value := ");
    try value(writer, info, 0);
    try writer.writeAll("\nend Ora.Generated.LifecycleInputs\n");
}

fn value(writer: anytype, info: builtin.BuiltinIntegerInfo, literal: u256) !void {
    try writer.print(".integer (.ofNat (.{s} .w{d}) {d})", .{ if (info.signed) "signed" else "unsigned", info.width, literal });
}

fn string(writer: anytype, text: []const u8) !void {
    try writer.writeByte('"');
    for (text) |byte| switch (byte) {
        '"' => try writer.writeAll("\\\""),
        '\\' => try writer.writeAll("\\\\"),
        '\n' => try writer.writeAll("\\n"),
        '\r' => try writer.writeAll("\\r"),
        '\t' => try writer.writeAll("\\t"),
        0...8, 11, 12, 14...31, 127 => return error.InvalidLifecycleString,
        else => try writer.writeByte(byte),
    };
    try writer.writeByte('"');
}

//! Deterministic text dump for `formal.obligation`.
//!
//! This module serializes an already-built obligation manifest. It does not
//! gather compiler facts, walk MLIR, encode Z3, emit Lean, or simplify terms.

const std = @import("std");
const obligation = @import("obligation.zig");

pub const Format = enum(u8) {
    json_lines,
};

pub fn write(writer: anytype, set: obligation.ObligationSet, comptime format: Format) !void {
    switch (format) {
        .json_lines => try writeJsonLines(writer, set),
    }
}

pub fn writeJsonLines(writer: anytype, set: obligation.ObligationSet) !void {
    for (set.assumptions) |item| {
        try writer.writeAll("{\"record\":\"assumption\"");
        try writeIdField(writer, "id", item.id);
        try writeOwnerField(writer, item.owner);
        try writeSourceField(writer, item.source);
        try writeStringField(writer, "phase", @tagName(item.phase));
        try writeOriginField(writer, item.origin);
        try writeStringField(writer, "kind", @tagName(item.kind));
        if (item.formula) |formula| try writeFormulaField(writer, formula);
        try writer.writeAll("}\n");
    }

    for (set.obligations) |item| {
        try writer.writeAll("{\"record\":\"obligation\"");
        try writeIdField(writer, "id", item.id);
        try writeOwnerField(writer, item.owner);
        try writeSourceField(writer, item.source);
        try writeStringField(writer, "phase", @tagName(item.phase));
        try writeOriginField(writer, item.origin);
        try writeKindField(writer, item.kind);
        try writeStringField(writer, "artifact_policy", @tagName(item.artifact_policy));
        try writeIdListField(writer, "dependencies", item.dependencies);
        try writeIdListField(writer, "derived_from", item.derived_from);
        try writer.writeAll("}\n");
    }

    for (set.queries) |item| {
        try writer.writeAll("{\"record\":\"query\"");
        try writeIdField(writer, "id", item.id);
        try writeOwnerField(writer, item.owner);
        try writeSourceField(writer, item.source);
        try writeStringField(writer, "phase", @tagName(item.phase));
        try writeOriginField(writer, item.origin);
        try writeStringField(writer, "backend", @tagName(item.backend));
        try writeStringField(writer, "kind", @tagName(item.kind));
        try writeOptionalEnumField(writer, "logical_role", obligation.LogicalRole, item.logical_role);
        try writeOptionalStringField(writer, "guard_id", item.guard_id);
        try writeIdListField(writer, "obligation_ids", item.obligation_ids);
        try writeIdListField(writer, "assumption_ids", item.assumption_ids);
        try writeStringField(writer, "fragment", @tagName(item.fragment));
        try writeStringField(writer, "solver_logic", @tagName(item.solver_logic));
        try writeNumberField(writer, "constraint_count", item.constraint_count);
        try writeOptionalNumberField(writer, "smtlib_hash", item.smtlib_hash);
        if (item.result) |result| try writeQueryResultField(writer, result);
        try writer.writeAll("}\n");
    }

    for (set.diagnostics) |item| {
        try writer.writeAll("{\"record\":\"diagnostic\"");
        try writeStringField(writer, "kind", @tagName(item.kind));
        try writeSourceField(writer, item.source);
        try writeStringField(writer, "message", item.message);
        try writeBoolField(writer, "blocks_artifacts", item.blocks_artifacts);
        try writer.writeAll("}\n");
    }

    for (set.terms, 0..) |item, index| {
        try writer.writeAll("{\"record\":\"term\"");
        try writeIdField(writer, "id", @intCast(index));
        try writeTermField(writer, item);
        try writer.writeAll("}\n");
    }
}

fn writeIdField(writer: anytype, comptime name: []const u8, value: obligation.Id) !void {
    try writeNumberField(writer, name, value);
}

fn writeNumberField(writer: anytype, comptime name: []const u8, value: anytype) !void {
    try writeFieldPrefix(writer, name);
    try writer.print("{d}", .{value});
}

fn writeBoolField(writer: anytype, comptime name: []const u8, value: bool) !void {
    try writeFieldPrefix(writer, name);
    try writer.writeAll(if (value) "true" else "false");
}

fn writeStringField(writer: anytype, comptime name: []const u8, value: []const u8) !void {
    try writeFieldPrefix(writer, name);
    try writeJsonString(writer, value);
}

fn writeOptionalStringField(writer: anytype, comptime name: []const u8, value: ?[]const u8) !void {
    try writeFieldPrefix(writer, name);
    if (value) |text| {
        try writeJsonString(writer, text);
    } else {
        try writer.writeAll("null");
    }
}

fn writeOptionalNumberField(writer: anytype, comptime name: []const u8, value: anytype) !void {
    try writeFieldPrefix(writer, name);
    if (value) |number| {
        try writer.print("{d}", .{number});
    } else {
        try writer.writeAll("null");
    }
}

fn writeOptionalEnumField(writer: anytype, comptime name: []const u8, comptime T: type, value: ?T) !void {
    try writeFieldPrefix(writer, name);
    if (value) |tag| {
        try writeJsonString(writer, @tagName(tag));
    } else {
        try writer.writeAll("null");
    }
}

fn writeFieldPrefix(writer: anytype, comptime name: []const u8) !void {
    try writer.writeAll(",\"");
    try writer.writeAll(name);
    try writer.writeAll("\":");
}

fn writeTaggedFieldPrefix(writer: anytype, comptime name: []const u8, tag: []const u8) !void {
    try writeFieldPrefix(writer, name);
    try writer.writeAll("{\"tag\":");
    try writeJsonString(writer, tag);
}

fn writeTaggedObjectPrefix(writer: anytype, tag: []const u8) !void {
    try writer.writeAll("{\"tag\":");
    try writeJsonString(writer, tag);
}

fn writeJsonString(writer: anytype, value: []const u8) !void {
    try writer.writeByte('"');
    for (value) |byte| {
        switch (byte) {
            '"' => try writer.writeAll("\\\""),
            '\\' => try writer.writeAll("\\\\"),
            '\n' => try writer.writeAll("\\n"),
            '\r' => try writer.writeAll("\\r"),
            '\t' => try writer.writeAll("\\t"),
            0x00...0x08, 0x0b...0x0c, 0x0e...0x1f => try writeControlEscape(writer, byte),
            else => try writer.writeByte(byte),
        }
    }
    try writer.writeByte('"');
}

fn writeControlEscape(writer: anytype, byte: u8) !void {
    const hex = "0123456789abcdef";
    try writer.writeAll("\\u00");
    try writer.writeByte(hex[byte >> 4]);
    try writer.writeByte(hex[byte & 0x0f]);
}

fn writeOwnerField(writer: anytype, owner: obligation.Owner) !void {
    try writeTaggedFieldPrefix(writer, "owner", @tagName(owner));
    switch (owner) {
        .module => |name| try writeStringField(writer, "name", name),
        .contract => |name| try writeStringField(writer, "name", name),
        .function => |function| {
            try writeOptionalStringField(writer, "module", function.module);
            try writeOptionalStringField(writer, "contract", function.contract);
            try writeStringField(writer, "name", function.name);
        },
        .trait_method => |method| {
            try writeStringField(writer, "trait_name", method.trait_name);
            try writeStringField(writer, "method_name", method.method_name);
            try writeOptionalStringField(writer, "impl_name", method.impl_name);
        },
        .statement => |statement| {
            try writeStringField(writer, "function_name", statement.function_name);
            try writeNumberField(writer, "ordinal", statement.ordinal);
        },
        .backend => |backend| {
            try writeStringField(writer, "component", @tagName(backend.component));
            try writeStringField(writer, "name", backend.name);
        },
    }
    try writer.writeByte('}');
}

fn writeSourceField(writer: anytype, source: obligation.SourceRef) !void {
    try writer.writeAll(",\"source\":{");
    try writer.writeAll("\"file\":");
    if (source.file) |file| {
        try writeJsonString(writer, file);
    } else {
        try writer.writeAll("null");
    }
    try writeNumberField(writer, "line", source.line);
    try writeNumberField(writer, "column", source.column);
    try writeNumberField(writer, "byte_start", source.byte_start);
    try writeNumberField(writer, "byte_end", source.byte_end);
    try writer.writeByte('}');
}

fn writeOriginField(writer: anytype, origin: obligation.Origin) !void {
    try writeTaggedFieldPrefix(writer, "origin", @tagName(origin));
    switch (origin) {
        .source => {},
        .sema_fact => |fact| {
            try writeStringField(writer, "kind", fact.kind);
            try writeNumberField(writer, "ordinal", fact.ordinal);
        },
        .mlir_op => |op| {
            try writeStringField(writer, "op_name", op.op_name);
            try writeOptionalStringField(writer, "symbol", op.symbol);
            try writeNumberField(writer, "ordinal", op.ordinal);
        },
        .effect_slot => |effect| {
            try writeStringField(writer, "access", @tagName(effect.access));
            try writePlaceFieldNamed(writer, "slot", effect.slot);
        },
        .resource_op => |resource| {
            try writeStringField(writer, "op", @tagName(resource.op));
            try writeStringField(writer, "domain", resource.domain);
            try writeNumberField(writer, "ordinal", resource.ordinal);
        },
        .backend_fact => |fact| {
            try writeStringField(writer, "component", @tagName(fact.component));
            try writeStringField(writer, "fact", fact.fact);
            try writeNumberField(writer, "ordinal", fact.ordinal);
        },
    }
    try writer.writeByte('}');
}

fn writeKindField(writer: anytype, kind: obligation.Kind) !void {
    try writeTaggedFieldPrefix(writer, "kind", @tagName(kind));
    switch (kind) {
        .logical => |logical| {
            try writeStringField(writer, "role", @tagName(logical.role));
            try writeFormulaField(writer, logical.formula);
        },
        .runtime_guard => |guard| {
            try writeStringField(writer, "guard_id", guard.guard_id);
            try writeFormulaField(writer, guard.formula);
            try writeStringField(writer, "erasure", @tagName(guard.erasure));
        },
        .type_wf => |type_wf| try writeTypeRefField(writer, "ty", type_wf.ty),
        .type_relation => |relation| {
            try writeStringField(writer, "relation", @tagName(relation.relation));
            try writeTypeRefField(writer, "lhs", relation.lhs);
            try writeTypeRefField(writer, "rhs", relation.rhs);
        },
        .region_relation => |relation| {
            try writeStringField(writer, "relation", @tagName(relation.relation));
            try writeStringField(writer, "from", @tagName(relation.from));
            try writeStringField(writer, "to", @tagName(relation.to));
        },
        .effect_frame => |effect| {
            try writeStringField(writer, "relation", @tagName(effect.relation));
            try writePlaceListField(writer, "declared", effect.declared);
            try writePlaceListField(writer, "actual", effect.actual);
        },
        .resource => |resource| {
            try writeStringField(writer, "op", @tagName(resource.op));
            try writeStringField(writer, "domain", resource.domain);
            if (resource.source) |place| try writePlaceFieldNamed(writer, "source", place);
            if (resource.destination) |place| try writePlaceFieldNamed(writer, "destination", place);
            if (resource.amount) |amount| try writeFormulaFieldNamed(writer, "amount", amount);
            try writeStringField(writer, "property", @tagName(resource.property));
        },
        .filtered_input => |filtered| {
            try writeVarRefField(writer, "value", filtered.value);
            try writePlaceFieldNamed(writer, "sink", filtered.sink);
            try writeIdListField(writer, "accepted_by", filtered.accepted_by);
        },
        .backend_fact => |fact| {
            try writeStringField(writer, "component", @tagName(fact.component));
            try writeStringField(writer, "property", @tagName(fact.property));
        },
    }
    try writer.writeByte('}');
}

fn writeFormulaField(writer: anytype, formula: obligation.FormulaRef) !void {
    try writeFormulaFieldNamed(writer, "formula", formula);
}

fn writeFormulaFieldNamed(writer: anytype, comptime name: []const u8, formula: obligation.FormulaRef) !void {
    try writeTaggedFieldPrefix(writer, name, @tagName(formula));
    switch (formula) {
        .origin_value => |value| {
            try writeOriginField(writer, value.origin);
            try writeStringField(writer, "value_kind", @tagName(value.kind));
            try writeNumberField(writer, "index", value.index);
        },
        .term => |term_id| try writeIdField(writer, "id", term_id),
    }
    try writer.writeByte('}');
}

fn writeTypeRefField(writer: anytype, comptime name: []const u8, ty: obligation.TypeRef) !void {
    try writeTaggedFieldPrefix(writer, name, @tagName(ty));
    switch (ty) {
        .spelling => |text| try writeStringField(writer, "value", text),
        .compiler_type_id => |id| try writeNumberField(writer, "value", id),
    }
    try writer.writeByte('}');
}

fn writeVarRefField(writer: anytype, comptime name: []const u8, value: obligation.VarRef) !void {
    try writeFieldPrefix(writer, name);
    try writer.writeAll("{\"name\":");
    try writeJsonString(writer, value.name);
    if (value.ty) |ty| try writeTypeRefField(writer, "ty", ty);
    if (value.region) |region| try writeStringField(writer, "region", @tagName(region));
    try writer.writeByte('}');
}

fn writePlaceFieldNamed(writer: anytype, comptime name: []const u8, place: obligation.PlaceRef) !void {
    try writeFieldPrefix(writer, name);
    try writePlaceObject(writer, place);
}

fn writePlaceListField(writer: anytype, comptime name: []const u8, places: []const obligation.PlaceRef) !void {
    try writeFieldPrefix(writer, name);
    try writer.writeByte('[');
    for (places, 0..) |place, index| {
        if (index != 0) try writer.writeByte(',');
        try writePlaceObject(writer, place);
    }
    try writer.writeByte(']');
}

fn writePlaceObject(writer: anytype, place: obligation.PlaceRef) !void {
    try writer.writeAll("{\"root\":");
    try writeJsonString(writer, place.root);
    try writeStringField(writer, "region", @tagName(place.region));
    try writer.writeAll(",\"fields\":[");
    for (place.fields, 0..) |field, index| {
        if (index != 0) try writer.writeByte(',');
        try writeJsonString(writer, field);
    }
    try writer.writeAll("],\"keys\":[");
    for (place.keys, 0..) |key, index| {
        if (index != 0) try writer.writeByte(',');
        try writePlaceKeyObject(writer, key);
    }
    try writer.writeAll("]}");
}

fn writePlaceKeyObject(writer: anytype, key: obligation.PlaceKey) !void {
    try writeTaggedObjectPrefix(writer, @tagName(key));
    switch (key) {
        .parameter, .comptime_parameter, .comptime_range_parameter => |index| try writeNumberField(writer, "index", index),
        .constant => |value| try writeStringField(writer, "value", value),
        .msg_sender, .tx_origin, .unknown => {},
    }
    try writer.writeByte('}');
}

fn writeTermField(writer: anytype, term: obligation.Term) !void {
    try writeTaggedFieldPrefix(writer, "term", @tagName(term));
    switch (term) {
        .bool_lit => |value| try writeBoolField(writer, "value", value),
        .int_lit => |value| try writeStringField(writer, "value", value),
        .variable => |value| try writeVarRefField(writer, "value", value),
        .old => |id| try writeIdField(writer, "operand", id),
        .result => {},
        .place_read => |place| try writePlaceFieldNamed(writer, "place", place),
        .unary => |unary| {
            try writeStringField(writer, "op", @tagName(unary.op));
            try writeIdField(writer, "operand", unary.operand);
        },
        .binary => |binary| {
            try writeStringField(writer, "op", @tagName(binary.op));
            try writeIdField(writer, "lhs", binary.lhs);
            try writeIdField(writer, "rhs", binary.rhs);
        },
        .quantified => |quantified| {
            try writeStringField(writer, "quantifier", @tagName(quantified.quantifier));
            try writeVarRefField(writer, "binder", quantified.binder);
            if (quantified.condition) |condition| try writeIdField(writer, "condition", condition);
            try writeIdField(writer, "body", quantified.body);
        },
    }
    try writer.writeByte('}');
}

fn writeIdListField(writer: anytype, comptime name: []const u8, ids: []const obligation.Id) !void {
    try writeFieldPrefix(writer, name);
    try writer.writeByte('[');
    for (ids, 0..) |id, index| {
        if (index != 0) try writer.writeByte(',');
        try writer.print("{d}", .{id});
    }
    try writer.writeByte(']');
}

fn writeQueryResultField(writer: anytype, result: obligation.VerificationQueryResult) !void {
    try writeTaggedFieldPrefix(writer, "result", @tagName(result.status));
    try writeBoolField(writer, "vacuous", result.vacuous);
    try writeBoolField(writer, "vacuity_unknown", result.vacuity_unknown);
    try writeBoolField(writer, "degraded", result.degraded);
    try writer.writeByte('}');
}

fn dumpToOwnedString(allocator: std.mem.Allocator, set: obligation.ObligationSet) ![]u8 {
    var buffer = std.Io.Writer.Allocating.init(allocator);
    errdefer buffer.deinit();
    try writeJsonLines(&buffer.writer, set);
    return try buffer.toOwnedSlice();
}

test "json-lines dump of empty manifest is empty" {
    const actual = try dumpToOwnedString(std.testing.allocator, .{});
    defer std.testing.allocator.free(actual);
    try std.testing.expectEqualStrings("", actual);
}

test "json-lines dump of mlir-origin logical obligation" {
    const item: obligation.Obligation = .{
        .id = 1,
        .owner = .{ .function = .{ .contract = "Token", .name = "transfer" } },
        .source = .{ .file = "erc20.ora", .line = 10, .column = 5, .byte_start = 100, .byte_end = 120 },
        .phase = .ora_mlir,
        .origin = .{ .mlir_op = .{ .op_name = "ora.ensures", .symbol = "transfer", .ordinal = 2 } },
        .kind = .{ .logical = .{
            .role = .ensures,
            .formula = .{ .origin_value = .{
                .origin = .{ .mlir_op = .{ .op_name = "ora.ensures", .symbol = "transfer", .ordinal = 2 } },
            } },
        } },
    };
    const set: obligation.ObligationSet = .{ .obligations = &.{item} };
    const actual = try dumpToOwnedString(std.testing.allocator, set);
    defer std.testing.allocator.free(actual);

    try std.testing.expectEqualStrings(
        "{\"record\":\"obligation\",\"id\":1,\"owner\":{\"tag\":\"function\",\"module\":null,\"contract\":\"Token\",\"name\":\"transfer\"},\"source\":{\"file\":\"erc20.ora\",\"line\":10,\"column\":5,\"byte_start\":100,\"byte_end\":120},\"phase\":\"ora_mlir\",\"origin\":{\"tag\":\"mlir_op\",\"op_name\":\"ora.ensures\",\"symbol\":\"transfer\",\"ordinal\":2},\"kind\":{\"tag\":\"logical\",\"role\":\"ensures\",\"formula\":{\"tag\":\"origin_value\",\"origin\":{\"tag\":\"mlir_op\",\"op_name\":\"ora.ensures\",\"symbol\":\"transfer\",\"ordinal\":2},\"value_kind\":\"result\",\"index\":0}},\"artifact_policy\":\"blocks_verified_artifacts\",\"dependencies\":[],\"derived_from\":[]}\n",
        actual,
    );
}

test "json-lines dump of blocking diagnostic" {
    const diagnostic: obligation.ObligationDiagnostic = .{
        .kind = .unsupported,
        .source = .{ .file = "test.ora", .line = 3, .column = 9 },
        .message = "unsupported quantified binder",
    };
    const set: obligation.ObligationSet = .{ .diagnostics = &.{diagnostic} };
    const actual = try dumpToOwnedString(std.testing.allocator, set);
    defer std.testing.allocator.free(actual);

    try std.testing.expectEqualStrings(
        "{\"record\":\"diagnostic\",\"kind\":\"unsupported\",\"source\":{\"file\":\"test.ora\",\"line\":3,\"column\":9,\"byte_start\":0,\"byte_end\":0},\"message\":\"unsupported quantified binder\",\"blocks_artifacts\":true}\n",
        actual,
    );
}

test "json-lines dump of derived backend fact" {
    const derived = [_]obligation.Id{ 1, 2 };
    const item: obligation.Obligation = .{
        .id = 3,
        .owner = .{ .backend = .{ .component = .dispatcher, .name = "erc20" } },
        .source = .generated(),
        .phase = .sinora,
        .origin = .{ .backend_fact = .{ .component = .dispatcher, .fact = "selector_table_complete" } },
        .kind = .{ .backend_fact = .{
            .component = .dispatcher,
            .property = .complete,
        } },
        .derived_from = &derived,
    };
    const set: obligation.ObligationSet = .{ .obligations = &.{item} };
    const actual = try dumpToOwnedString(std.testing.allocator, set);
    defer std.testing.allocator.free(actual);

    try std.testing.expectEqualStrings(
        "{\"record\":\"obligation\",\"id\":3,\"owner\":{\"tag\":\"backend\",\"component\":\"dispatcher\",\"name\":\"erc20\"},\"source\":{\"file\":null,\"line\":0,\"column\":0,\"byte_start\":0,\"byte_end\":0},\"phase\":\"sinora\",\"origin\":{\"tag\":\"backend_fact\",\"component\":\"dispatcher\",\"fact\":\"selector_table_complete\",\"ordinal\":0},\"kind\":{\"tag\":\"backend_fact\",\"component\":\"dispatcher\",\"property\":\"complete\"},\"artifact_policy\":\"blocks_verified_artifacts\",\"dependencies\":[],\"derived_from\":[1,2]}\n",
        actual,
    );
}

test "json-lines dump of projected verifier query" {
    const ids = [_]obligation.Id{7};
    const item: obligation.VerificationQuery = .{
        .id = 8,
        .owner = .{ .function = .{ .name = "transfer" } },
        .source = .generated(),
        .phase = .report,
        .origin = .{ .mlir_op = .{ .op_name = "ora.refinement_guard", .symbol = "transfer", .ordinal = 4 } },
        .backend = .z3,
        .kind = .guard_violate,
        .guard_id = "guard:transfer:amount",
        .obligation_ids = &ids,
        .fragment = .qf_bv_array,
        .solver_logic = .qf_aufbv,
        .constraint_count = 5,
        .smtlib_hash = 1234,
        .result = .{ .status = .unknown, .vacuity_unknown = true },
    };
    const set: obligation.ObligationSet = .{ .queries = &.{item} };
    const actual = try dumpToOwnedString(std.testing.allocator, set);
    defer std.testing.allocator.free(actual);

    try std.testing.expectEqualStrings(
        "{\"record\":\"query\",\"id\":8,\"owner\":{\"tag\":\"function\",\"module\":null,\"contract\":null,\"name\":\"transfer\"},\"source\":{\"file\":null,\"line\":0,\"column\":0,\"byte_start\":0,\"byte_end\":0},\"phase\":\"report\",\"origin\":{\"tag\":\"mlir_op\",\"op_name\":\"ora.refinement_guard\",\"symbol\":\"transfer\",\"ordinal\":4},\"backend\":\"z3\",\"kind\":\"guard_violate\",\"logical_role\":null,\"guard_id\":\"guard:transfer:amount\",\"obligation_ids\":[7],\"assumption_ids\":[],\"fragment\":\"qf_bv_array\",\"solver_logic\":\"qf_aufbv\",\"constraint_count\":5,\"smtlib_hash\":1234,\"result\":{\"tag\":\"unknown\",\"vacuous\":false,\"vacuity_unknown\":true,\"degraded\":false}}\n",
        actual,
    );
}

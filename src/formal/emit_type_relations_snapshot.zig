//! Emits `formal/Ora/Generated/CompilerTypeRelations.lean` — data-only rows for
//! compiler type relation functions (`typeEql`, `typesAssignable`,
//! `locatedTypeEql`, `isAssignable`). The trusted Lean checks live in
//! `formal/Ora/TypeRelationsSync.lean`.

const std = @import("std");
const formal = @import("ora_formal");

const LocatedType = formal.LocatedType;
const Type = formal.Type;

const header =
    \\/-
    \\GENERATED — DATA ONLY. Do NOT edit by hand and do NOT add any `theorem`,
    \\`lemma`, `example`, `axiom`, `sorry`, `instance`, `macro`, attribute, or extra
    \\`import` to this file. It contains only `def … := <literal>` relation rows
    \\emitted from the compiler. The TRUSTED checks live in
    \\`Ora/TypeRelationsSync.lean`.
    \\
    \\Regenerate with `scripts/check-formal-type-relations-sync.sh`. Source:
    \\src/formal/emit_type_relations_snapshot.zig,
    \\src/sema/type_descriptors.zig, and src/sema/region.zig.
    \\-/
    \\
    \\namespace Ora.Generated
    \\
    \\
;

const TypeRelationCase = struct {
    label: []const u8,
    lhs_name: []const u8,
    rhs_name: []const u8,
    lhs: Type,
    rhs: Type,
};

const LocatedRelationCase = struct {
    label: []const u8,
    lhs_name: []const u8,
    rhs_name: []const u8,
    lhs: LocatedType,
    rhs: LocatedType,
};

fn boolText(value: bool) []const u8 {
    return if (value) "true" else "false";
}

fn emitTypeRelationRows(
    out: anytype,
    comptime name: []const u8,
    cases: []const TypeRelationCase,
    relation: enum { eql, assignable },
) !void {
    try out.print("def {s} : List (String × String × String × Bool) :=\n  [", .{name});
    for (cases, 0..) |case, index| {
        if (index != 0) try out.writeAll(",\n   ");
        const result = switch (relation) {
            .eql => formal.typeEql(case.lhs, case.rhs),
            .assignable => formal.typesAssignable(case.lhs, case.rhs),
        };
        try out.print("(\"{s}\", \"{s}\", \"{s}\", {s})", .{
            case.label,
            case.lhs_name,
            case.rhs_name,
            boolText(result),
        });
    }
    try out.writeAll("]\n\n");
}

fn emitLocatedRelationRows(
    out: anytype,
    comptime name: []const u8,
    cases: []const LocatedRelationCase,
    relation: enum { eql, assignable },
) !void {
    try out.print("def {s} : List (String × String × String × Bool) :=\n  [", .{name});
    for (cases, 0..) |case, index| {
        if (index != 0) try out.writeAll(",\n   ");
        const result = switch (relation) {
            .eql => formal.locatedTypeEql(case.lhs, case.rhs),
            .assignable => formal.locatedTypeAssignable(case.lhs, case.rhs),
        };
        try out.print("(\"{s}\", \"{s}\", \"{s}\", {s})", .{
            case.label,
            case.lhs_name,
            case.rhs_name,
            boolText(result),
        });
    }
    try out.writeAll("]\n\n");
}

fn intType(comptime bits: u16, comptime signed: bool, comptime spelling: []const u8) Type {
    return .{ .integer = .{ .bits = bits, .signed = signed, .spelling = spelling } };
}

fn named(comptime tag: enum { struct_, external_proxy }, comptime name: []const u8) Type {
    return switch (tag) {
        .struct_ => .{ .struct_ = .{ .name = name } },
        .external_proxy => .{ .external_proxy = .{ .trait_name = name } },
    };
}

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    var out_buffer: [1 << 16]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &out_buffer);
    const out = &stdout_writer.interface;

    const u8_ty = intType(8, false, "u8");
    const u16_ty = intType(16, false, "u16");
    const u256_ty = intType(256, false, "u256");
    const i8_ty = intType(8, true, "i8");
    const bool_ty: Type = .{ .bool = {} };
    const address_ty: Type = .{ .address = {} };
    const bytes4_ty: Type = .{ .fixed_bytes = .{ .len = 4, .spelling = "bytes4" } };
    const bytes32_ty: Type = .{ .fixed_bytes = .{ .len = 32, .spelling = "bytes32" } };
    const point_ty = named(.struct_, "Point");
    const other_point_ty = named(.struct_, "OtherPoint");
    const token_resource_carrier_u8: Type = .{ .resource_domain = .{ .name = "TokenUnit", .carrier_type = &u8_ty } };
    const token_resource_carrier_u16: Type = .{ .resource_domain = .{ .name = "TokenUnit", .carrier_type = &u16_ty } };
    const token_place_u8: Type = .{ .resource_place = .{ .domain_type = &token_resource_carrier_u8 } };
    const token_place_u16: Type = .{ .resource_place = .{ .domain_type = &token_resource_carrier_u16 } };

    const tuple_u8_bool_items = [_]Type{ u8_ty, bool_ty };
    const tuple_u16_bool_items = [_]Type{ u16_ty, bool_ty };
    const tuple_u8_bool: Type = .{ .tuple = tuple_u8_bool_items[0..] };
    const tuple_u16_bool: Type = .{ .tuple = tuple_u16_bool_items[0..] };

    const array_u8_3: Type = .{ .array = .{ .element_type = &u8_ty, .len = 3 } };
    const array_u16_3: Type = .{ .array = .{ .element_type = &u16_ty, .len = 3 } };
    const slice_u8: Type = .{ .slice = .{ .element_type = &u8_ty } };
    const slice_u16: Type = .{ .slice = .{ .element_type = &u16_ty } };
    const map_address_u8: Type = .{ .map = .{ .key_type = &address_ty, .value_type = &u8_ty } };
    const map_address_u16: Type = .{ .map = .{ .key_type = &address_ty, .value_type = &u16_ty } };

    const min_one_args = [_]formal.RefinementArg{.{ .Integer = .{ .text = "1" } }};
    const min_hundred_args = [_]formal.RefinementArg{.{ .Integer = .{ .text = "100" } }};
    const min_one: Type = .{ .refinement = .{ .name = "MinValue", .base_type = &u256_ty, .args = min_one_args[0..] } };
    const min_hundred: Type = .{ .refinement = .{ .name = "MinValue", .base_type = &u256_ty, .args = min_hundred_args[0..] } };

    const fn_foo_params = [_]Type{u8_ty};
    const fn_foo_returns = [_]Type{bool_ty};
    const fn_foo: Type = .{ .function = .{ .name = "foo", .param_types = fn_foo_params[0..], .return_types = fn_foo_returns[0..] } };
    const fn_bar: Type = .{ .function = .{ .name = "bar", .param_types = fn_foo_params[0..], .return_types = fn_foo_returns[0..] } };

    try out.writeAll(header);

    const type_eql_cases = [_]TypeRelationCase{
        .{ .label = "primitive_u256_self", .lhs_name = "u256", .rhs_name = "u256", .lhs = u256_ty, .rhs = u256_ty },
        .{ .label = "primitive_u8_vs_u256", .lhs_name = "u8", .rhs_name = "u256", .lhs = u8_ty, .rhs = u256_ty },
        .{ .label = "fixed_bytes_same", .lhs_name = "bytes4", .rhs_name = "bytes4", .lhs = bytes4_ty, .rhs = bytes4_ty },
        .{ .label = "fixed_bytes_different", .lhs_name = "bytes4", .rhs_name = "bytes32", .lhs = bytes4_ty, .rhs = bytes32_ty },
        .{ .label = "tuple_same", .lhs_name = "(u8,bool)", .rhs_name = "(u8,bool)", .lhs = tuple_u8_bool, .rhs = tuple_u8_bool },
        .{ .label = "tuple_element_width_diff", .lhs_name = "(u8,bool)", .rhs_name = "(u16,bool)", .lhs = tuple_u8_bool, .rhs = tuple_u16_bool },
        .{ .label = "array_same", .lhs_name = "[3]u8", .rhs_name = "[3]u8", .lhs = array_u8_3, .rhs = array_u8_3 },
        .{ .label = "slice_element_diff", .lhs_name = "slice[u8]", .rhs_name = "slice[u16]", .lhs = slice_u8, .rhs = slice_u16 },
        .{ .label = "map_same", .lhs_name = "map[address,u8]", .rhs_name = "map[address,u8]", .lhs = map_address_u8, .rhs = map_address_u8 },
        .{ .label = "nominal_struct_same", .lhs_name = "Point", .rhs_name = "Point", .lhs = point_ty, .rhs = point_ty },
        .{ .label = "nominal_struct_different", .lhs_name = "Point", .rhs_name = "OtherPoint", .lhs = point_ty, .rhs = other_point_ty },
        .{ .label = "refinement_args_differ", .lhs_name = "MinValue<u256,1>", .rhs_name = "MinValue<u256,100>", .lhs = min_one, .rhs = min_hundred },
        .{ .label = "function_same_name", .lhs_name = "foo(u8)->bool", .rhs_name = "foo(u8)->bool", .lhs = fn_foo, .rhs = fn_foo },
        .{ .label = "function_different_name", .lhs_name = "foo(u8)->bool", .rhs_name = "bar(u8)->bool", .lhs = fn_foo, .rhs = fn_bar },
        .{ .label = "resource_domain_same", .lhs_name = "resource TokenUnit<u8>", .rhs_name = "resource TokenUnit<u8>", .lhs = token_resource_carrier_u8, .rhs = token_resource_carrier_u8 },
        .{ .label = "resource_domain_carrier_diff", .lhs_name = "resource TokenUnit<u8>", .rhs_name = "resource TokenUnit<u16>", .lhs = token_resource_carrier_u8, .rhs = token_resource_carrier_u16 },
    };
    try emitTypeRelationRows(out, "compilerTypeEqlRows", type_eql_cases[0..], .eql);

    const assignable_cases = [_]TypeRelationCase{
        .{ .label = "u8_to_u256", .lhs_name = "u256", .rhs_name = "u8", .lhs = u256_ty, .rhs = u8_ty },
        .{ .label = "u256_to_u8_rejected", .lhs_name = "u8", .rhs_name = "u256", .lhs = u8_ty, .rhs = u256_ty },
        .{ .label = "signedness_mismatch", .lhs_name = "u8", .rhs_name = "i8", .lhs = u8_ty, .rhs = i8_ty },
        .{ .label = "tuple_widening", .lhs_name = "(u16,bool)", .rhs_name = "(u8,bool)", .lhs = tuple_u16_bool, .rhs = tuple_u8_bool },
        .{ .label = "tuple_narrowing_rejected", .lhs_name = "(u8,bool)", .rhs_name = "(u16,bool)", .lhs = tuple_u8_bool, .rhs = tuple_u16_bool },
        .{ .label = "array_widening", .lhs_name = "[3]u16", .rhs_name = "[3]u8", .lhs = array_u16_3, .rhs = array_u8_3 },
        .{ .label = "slice_widening", .lhs_name = "slice[u16]", .rhs_name = "slice[u8]", .lhs = slice_u16, .rhs = slice_u8 },
        .{ .label = "map_value_widening", .lhs_name = "map[address,u16]", .rhs_name = "map[address,u8]", .lhs = map_address_u16, .rhs = map_address_u8 },
        .{ .label = "nominal_struct_same", .lhs_name = "Point", .rhs_name = "Point", .lhs = point_ty, .rhs = point_ty },
        .{ .label = "nominal_struct_different", .lhs_name = "Point", .rhs_name = "OtherPoint", .lhs = point_ty, .rhs = other_point_ty },
        .{ .label = "function_same_name_identity", .lhs_name = "foo(u8)->bool", .rhs_name = "foo(u8)->bool", .lhs = fn_foo, .rhs = fn_foo },
        .{ .label = "function_different_name_rejected", .lhs_name = "foo(u8)->bool", .rhs_name = "bar(u8)->bool", .lhs = fn_foo, .rhs = fn_bar },
        .{ .label = "resource_domain_same_identity", .lhs_name = "resource TokenUnit<u8>", .rhs_name = "resource TokenUnit<u8>", .lhs = token_resource_carrier_u8, .rhs = token_resource_carrier_u8 },
        .{ .label = "resource_domain_widening_rejected", .lhs_name = "resource TokenUnit<u16>", .rhs_name = "resource TokenUnit<u8>", .lhs = token_resource_carrier_u16, .rhs = token_resource_carrier_u8 },
        .{ .label = "resource_place_same_identity", .lhs_name = "Resource<TokenUnit<u8>>", .rhs_name = "Resource<TokenUnit<u8>>", .lhs = token_place_u8, .rhs = token_place_u8 },
        .{ .label = "resource_place_widening_rejected", .lhs_name = "Resource<TokenUnit<u16>>", .rhs_name = "Resource<TokenUnit<u8>>", .lhs = token_place_u16, .rhs = token_place_u8 },
    };
    try emitTypeRelationRows(out, "compilerTypesAssignableRows", assignable_cases[0..], .assignable);

    const storage_u256_local = LocatedType.withRegion(u256_ty, .storage);
    const storage_u256_storage_provenance = LocatedType.withRegionAndProvenance(u256_ty, .storage, .storage);
    const memory_u256 = LocatedType.withRegion(u256_ty, .memory);
    const storage_u8 = LocatedType.withRegion(u8_ty, .storage);
    const calldata_u8 = LocatedType.withRegion(u8_ty, .calldata);
    const storage_bool = LocatedType.withRegion(bool_ty, .storage);
    const memory_bool = LocatedType.withRegion(bool_ty, .memory);
    const memory_address = LocatedType.withRegion(address_ty, .memory);
    const calldata_u256 = LocatedType.withRegion(u256_ty, .calldata);

    const located_eql_cases = [_]LocatedRelationCase{
        .{ .label = "same_type_region_provenance", .lhs_name = "u256@storage/local", .rhs_name = "u256@storage/local", .lhs = storage_u256_local, .rhs = storage_u256_local },
        .{ .label = "different_region", .lhs_name = "u256@storage/local", .rhs_name = "u256@memory/local", .lhs = storage_u256_local, .rhs = memory_u256 },
        .{ .label = "different_type", .lhs_name = "u8@storage/local", .rhs_name = "u256@storage/local", .lhs = storage_u8, .rhs = storage_u256_local },
        .{ .label = "different_provenance", .lhs_name = "u256@storage/local", .rhs_name = "u256@storage/storage", .lhs = storage_u256_local, .rhs = storage_u256_storage_provenance },
    };
    try emitLocatedRelationRows(out, "compilerLocatedTypeEqlRows", located_eql_cases[0..], .eql);

    const located_assignable_cases = [_]LocatedRelationCase{
        .{ .label = "calldata_u8_to_memory_u256", .lhs_name = "u8@calldata/local", .rhs_name = "u256@memory/local", .lhs = calldata_u8, .rhs = memory_u256 },
        .{ .label = "memory_u256_to_calldata_u256_rejected", .lhs_name = "u256@memory/local", .rhs_name = "u256@calldata/local", .lhs = memory_u256, .rhs = calldata_u256 },
        .{ .label = "storage_bool_to_memory_bool", .lhs_name = "bool@storage/local", .rhs_name = "bool@memory/local", .lhs = storage_bool, .rhs = memory_bool },
        .{ .label = "storage_bool_to_memory_address_rejected", .lhs_name = "bool@storage/local", .rhs_name = "address@memory/local", .lhs = storage_bool, .rhs = memory_address },
        .{ .label = "storage_u8_to_storage_u256", .lhs_name = "u8@storage/local", .rhs_name = "u256@storage/local", .lhs = storage_u8, .rhs = storage_u256_local },
        .{ .label = "provenance_ignored_by_assignability", .lhs_name = "u256@storage/storage", .rhs_name = "u256@storage/local", .lhs = storage_u256_storage_provenance, .rhs = storage_u256_local },
    };
    try emitLocatedRelationRows(out, "compilerLocatedAssignableRows", located_assignable_cases[0..], .assignable);

    try out.writeAll("end Ora.Generated\n");
    try out.flush();
}

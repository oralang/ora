const std = @import("std");
const ast = @import("../ast/mod.zig");
const model = @import("model.zig");

const ItemIndexResult = model.ItemIndexResult;
const Type = model.Type;

pub fn descriptorFromTypeExpr(allocator: std.mem.Allocator, file: *const ast.AstFile, item_index: *const ItemIndexResult, type_expr_id: ast.TypeExprId) anyerror!Type {
    return switch (file.typeExpr(type_expr_id).*) {
        .Path => |path| if (std.mem.eql(u8, std.mem.trim(u8, path.name, " \t\n\r"), "NonZeroAddress"))
            .{ .refinement = .{
                .name = "NonZeroAddress",
                .base_type = try storeType(allocator, .{ .address = {} }),
                .args = &.{},
            } }
        else
            descriptorFromPathName(file, item_index, path.name),
        .Generic => |generic| try descriptorFromGenericType(allocator, file, item_index, generic),
        .Tuple => |tuple| blk: {
            const elements = try allocator.alloc(Type, tuple.elements.len);
            for (tuple.elements, 0..) |element, index| {
                elements[index] = try descriptorFromTypeExpr(allocator, file, item_index, element);
            }
            break :blk .{ .tuple = elements };
        },
        .AnonymousStruct => |struct_type| blk: {
            const fields = try allocator.alloc(model.AnonymousStructField, struct_type.fields.len);
            for (struct_type.fields, 0..) |field, index| {
                fields[index] = .{
                    .name = field.name,
                    .ty = try descriptorFromTypeExpr(allocator, file, item_index, field.type_expr),
                };
            }
            break :blk .{ .anonymous_struct = .{ .fields = fields } };
        },
        .Array => |array| .{ .array = .{
            .element_type = try storeType(allocator, try descriptorFromTypeExpr(allocator, file, item_index, array.element)),
            .len = parseArrayLen(array.size),
        } },
        .Slice => |slice| .{ .slice = .{
            .element_type = try storeType(allocator, try descriptorFromTypeExpr(allocator, file, item_index, slice.element)),
        } },
        .ErrorUnion => |error_union| blk: {
            const errors = try allocator.alloc(Type, error_union.errors.len);
            for (error_union.errors, 0..) |error_type, index| {
                errors[index] = try descriptorFromTypeExpr(allocator, file, item_index, error_type);
            }
            break :blk .{ .error_union = .{
                .payload_type = try storeType(allocator, try descriptorFromTypeExpr(allocator, file, item_index, error_union.payload)),
                .error_types = errors,
            } };
        },
        .Error => .{ .unknown = {} },
    };
}

pub fn descriptorFromPathName(file: *const ast.AstFile, item_index: *const ItemIndexResult, name: []const u8) Type {
    const trimmed = std.mem.trim(u8, name, " \t\n\r");
    if (std.mem.eql(u8, trimmed, "void")) return .{ .void = {} };
    if (std.mem.eql(u8, trimmed, "bool")) return .{ .bool = {} };
    if (std.mem.eql(u8, trimmed, "string")) return .{ .string = {} };
    if (std.mem.eql(u8, trimmed, "address")) return .{ .address = {} };
    if (std.mem.eql(u8, trimmed, "bytes")) return .{ .bytes = {} };
    if (parseIntegerType(trimmed)) |integer| return .{ .integer = integer };
    if (item_index.lookup(trimmed)) |item_id| {
        return switch (file.item(item_id).*) {
            .Contract => .{ .contract = .{ .name = trimmed } },
            .Struct => .{ .struct_ = .{ .name = trimmed } },
            .Bitfield => .{ .bitfield = .{ .name = trimmed } },
            .Enum => .{ .enum_ = .{ .name = trimmed } },
            else => .{ .named = .{ .name = trimmed } },
        };
    }
    return .{ .named = .{ .name = trimmed } };
}

pub fn descriptorFromGenericType(allocator: std.mem.Allocator, file: *const ast.AstFile, item_index: *const ItemIndexResult, generic: ast.GenericTypeExpr) anyerror!Type {
    if (std.mem.eql(u8, generic.name, "map")) {
        return .{ .map = .{
            .key_type = if (generic.args.len > 0 and generic.args[0] == .Type)
                try storeType(allocator, try descriptorFromTypeExpr(allocator, file, item_index, generic.args[0].Type))
            else
                null,
            .value_type = if (generic.args.len > 1 and generic.args[1] == .Type)
                try storeType(allocator, try descriptorFromTypeExpr(allocator, file, item_index, generic.args[1].Type))
            else
                null,
        } };
    }

    if (std.mem.eql(u8, generic.name, "MinValue") or
        std.mem.eql(u8, generic.name, "MaxValue") or
        std.mem.eql(u8, generic.name, "InRange") or
        std.mem.eql(u8, generic.name, "Scaled") or
        std.mem.eql(u8, generic.name, "Exact") or
        std.mem.eql(u8, generic.name, "NonZero") or
        std.mem.eql(u8, generic.name, "NonZeroAddress") or
        std.mem.eql(u8, generic.name, "BasisPoints"))
    {
        if (generic.args.len > 0) {
            return switch (generic.args[0]) {
                .Type => |type_expr_id| .{ .refinement = .{
                    .name = generic.name,
                    .base_type = try storeType(allocator, try descriptorFromTypeExpr(allocator, file, item_index, type_expr_id)),
                    .args = generic.args,
                } },
                else => .{ .unknown = {} },
            };
        }
    }

    if (std.mem.eql(u8, generic.name, "Result")) {
        if (generic.args.len != 2) return .{ .unknown = {} };
        if (generic.args[0] != .Type or generic.args[1] != .Type) return .{ .unknown = {} };

        const payload_type = try descriptorFromTypeExpr(allocator, file, item_index, generic.args[0].Type);
        const error_type = try descriptorFromTypeExpr(allocator, file, item_index, generic.args[1].Type);
        const error_types = try allocator.alloc(Type, 1);
        error_types[0] = error_type;
        return .{ .error_union = .{
            .payload_type = try storeType(allocator, payload_type),
            .error_types = error_types,
        } };
    }

    return descriptorFromPathName(file, item_index, generic.name);
}

pub fn inferItemType(allocator: std.mem.Allocator, file: *const ast.AstFile, item_index: *const ItemIndexResult, item: ast.Item) anyerror!Type {
    return switch (item) {
        .Contract => |contract| .{ .contract = .{ .name = contract.name } },
        .Function => |function| blk: {
            const params = try allocator.alloc(Type, function.parameters.len);
            for (function.parameters, 0..) |parameter, index| {
                params[index] = try descriptorFromTypeExpr(allocator, file, item_index, parameter.type_expr);
            }

            const returns = if (function.return_type) |type_expr| blk_returns: {
                const slice = try allocator.alloc(Type, 1);
                slice[0] = try descriptorFromTypeExpr(allocator, file, item_index, type_expr);
                break :blk_returns slice;
            } else &.{};

            break :blk .{ .function = .{
                .name = function.name,
                .param_types = params,
                .return_types = returns,
            } };
        },
        .Struct => |struct_item| .{ .struct_ = .{ .name = struct_item.name } },
        .Bitfield => |bitfield_item| .{ .bitfield = .{ .name = bitfield_item.name } },
        .Enum => |enum_item| .{ .enum_ = .{ .name = enum_item.name } },
        .ErrorDecl => |error_decl| .{ .named = .{ .name = error_decl.name } },
        .TypeAlias => |type_alias| try descriptorFromTypeExpr(allocator, file, item_index, type_alias.target_type),
        .GhostBlock => .{ .unknown = {} },
        .Field => |field| if (field.type_expr) |type_expr| try descriptorFromTypeExpr(allocator, file, item_index, type_expr) else .{ .unknown = {} },
        .Constant => |constant| if (constant.type_expr) |type_expr| try descriptorFromTypeExpr(allocator, file, item_index, type_expr) else .{ .unknown = {} },
        else => .{ .unknown = {} },
    };
}

pub fn mergeExprType(current: Type, next: Type) Type {
    if (current.kind() == .unknown) return next;
    if (next.kind() == .unknown) return current;
    if (typeEql(current, next)) return current;
    if (commonIntegerType(current, next)) |merged| return merged;
    return .{ .unknown = {} };
}

pub fn typeEql(lhs: Type, rhs: Type) bool {
    if (lhs.kind() != rhs.kind()) return false;
    return switch (lhs) {
        .unknown, .void, .bool, .string, .address, .bytes => true,
        .external_proxy => |left| std.mem.eql(u8, left.trait_name, rhs.external_proxy.trait_name),
        .integer => |left| blk: {
            const right = rhs.integer;
            break :blk left.bits == right.bits and left.signed == right.signed and std.meta.eql(left.spelling, right.spelling);
        },
        .named => |left| std.mem.eql(u8, left.name, rhs.named.name),
        .contract => |left| std.mem.eql(u8, left.name, rhs.contract.name),
        .struct_ => |left| std.mem.eql(u8, left.name, rhs.struct_.name),
        .bitfield => |left| std.mem.eql(u8, left.name, rhs.bitfield.name),
        .enum_ => |left| std.mem.eql(u8, left.name, rhs.enum_.name),
        .function => |left| blk: {
            const right = rhs.function;
            break :blk std.meta.eql(left.name, right.name) and typeSliceEql(left.param_types, right.param_types) and typeSliceEql(left.return_types, right.return_types);
        },
        .tuple => |left| typeSliceEql(left, rhs.tuple),
        .anonymous_struct => |left| anonymousStructFieldSliceEql(left.fields, rhs.anonymous_struct.fields),
        .array => |left| blk: {
            const right = rhs.array;
            break :blk left.len == right.len and typeEql(left.element_type.*, right.element_type.*);
        },
        .slice => |left| typeEql(left.element_type.*, rhs.slice.element_type.*),
        .map => |left| blk: {
            const right = rhs.map;
            break :blk optionalTypeEql(left.key_type, right.key_type) and optionalTypeEql(left.value_type, right.value_type);
        },
        .error_union => |left| blk: {
            const right = rhs.error_union;
            break :blk typeEql(left.payload_type.*, right.payload_type.*) and typeSliceEql(left.error_types, right.error_types);
        },
        .refinement => |left| blk: {
            const right = rhs.refinement;
            break :blk std.mem.eql(u8, left.name, right.name) and typeEql(left.base_type.*, right.base_type.*) and refinementArgSliceEql(left.args, right.args);
        },
    };
}

pub fn typesAssignable(expected_type: Type, actual_type: Type) bool {
    if (expected_type.kind() == .unknown or actual_type.kind() == .unknown) return true;
    if (typeEql(expected_type, actual_type)) return true;

    if (expected_type.kind() == .refinement and actual_type.kind() == .refinement) {
        return refinementSubtypeAssignable(expected_type.refinement, actual_type.refinement);
    }
    if (expected_type.kind() == .refinement) {
        return typesAssignable(expected_type.refinement.base_type.*, actual_type);
    }
    if (actual_type.kind() == .refinement) {
        return typesAssignable(expected_type, actual_type.refinement.base_type.*);
    }

    const expected_unwrapped = unwrapRefinement(expected_type);
    const actual_unwrapped = unwrapRefinement(actual_type);
    if (isIntegerType(expected_unwrapped) and isIntegerType(actual_unwrapped)) {
        return integerAssignable(expected_unwrapped.integer, actual_unwrapped.integer);
    }
    if (expected_unwrapped.kind() == .tuple and actual_unwrapped.kind() == .tuple) {
        const expected_tuple = expected_unwrapped.tuple;
        const actual_tuple = actual_unwrapped.tuple;
        if (expected_tuple.len != actual_tuple.len) return false;
        for (expected_tuple, actual_tuple) |expected_element, actual_element| {
            if (!typesAssignable(expected_element, actual_element)) return false;
        }
        return true;
    }
    if (expected_unwrapped.kind() == .anonymous_struct and actual_unwrapped.kind() == .anonymous_struct) {
        const expected_fields = expected_unwrapped.anonymous_struct.fields;
        const actual_fields = actual_unwrapped.anonymous_struct.fields;
        if (expected_fields.len != actual_fields.len) return false;
        for (expected_fields, actual_fields) |expected_field, actual_field| {
            if (!std.mem.eql(u8, expected_field.name, actual_field.name)) return false;
            if (!typesAssignable(expected_field.ty, actual_field.ty)) return false;
        }
        return true;
    }
    if (expected_unwrapped.kind() == .anonymous_struct and actual_unwrapped.kind() == .tuple) {
        return anonymousStructAssignableFromTuple(expected_unwrapped.anonymous_struct.fields, actual_unwrapped.tuple);
    }
    if (expected_unwrapped.kind() == .tuple and actual_unwrapped.kind() == .anonymous_struct) {
        return anonymousStructAssignableToTuple(expected_unwrapped.tuple, actual_unwrapped.anonymous_struct.fields);
    }
    if (expected_unwrapped.kind() == .array and actual_unwrapped.kind() == .array) {
        const expected_arr = expected_unwrapped.array;
        const actual_arr = actual_unwrapped.array;
        return expected_arr.len == actual_arr.len and
            typesAssignable(expected_arr.element_type.*, actual_arr.element_type.*);
    }
    if (expected_unwrapped.kind() == .slice and actual_unwrapped.kind() == .slice) {
        return typesAssignable(expected_unwrapped.slice.element_type.*, actual_unwrapped.slice.element_type.*);
    }
    if (expected_unwrapped.kind() == .map and actual_unwrapped.kind() == .map) {
        const expected_map = expected_unwrapped.map;
        const actual_map = actual_unwrapped.map;
        const key_ok = if (expected_map.key_type != null and actual_map.key_type != null)
            typesAssignable(expected_map.key_type.?.*, actual_map.key_type.?.*)
        else
            expected_map.key_type == null and actual_map.key_type == null;
        const val_ok = if (expected_map.value_type != null and actual_map.value_type != null)
            typesAssignable(expected_map.value_type.?.*, actual_map.value_type.?.*)
        else
            expected_map.value_type == null and actual_map.value_type == null;
        return key_ok and val_ok;
    }
    if (expected_unwrapped.kind() == .error_union) {
        const expected = expected_unwrapped.error_union;
        if (typesAssignable(expected.payload_type.*, actual_unwrapped)) return true;
        if (expected.error_types.len == 0 and actual_unwrapped.kind() == .named) return true;
        if (errorSetContains(expected.error_types, actual_unwrapped)) return true;
        if (actual_unwrapped.kind() == .error_union) {
            const actual = actual_unwrapped.error_union;
            return typesAssignable(expected.payload_type.*, actual.payload_type.*) and
                (expected.error_types.len == 0 or
                    errorSetContainsAll(expected.error_types, actual.error_types));
        }
    }
    return typeEql(expected_unwrapped, actual_unwrapped);
}

fn optionalTypeEql(lhs: ?*const Type, rhs: ?*const Type) bool {
    if (lhs == null or rhs == null) return lhs == null and rhs == null;
    return typeEql(lhs.?.*, rhs.?.*);
}

fn anonymousStructFieldSliceEql(lhs: []const model.AnonymousStructField, rhs: []const model.AnonymousStructField) bool {
    if (lhs.len != rhs.len) return false;
    for (lhs, rhs) |left, right| {
        if (!std.mem.eql(u8, left.name, right.name)) return false;
        if (!typeEql(left.ty, right.ty)) return false;
    }
    return true;
}

fn anonymousStructAssignableFromTuple(fields: []const model.AnonymousStructField, elements: []const Type) bool {
    if (fields.len != elements.len) return false;
    for (fields, elements) |field, element| {
        if (!typesAssignable(field.ty, element)) return false;
    }
    return true;
}

fn anonymousStructAssignableToTuple(elements: []const Type, fields: []const model.AnonymousStructField) bool {
    if (elements.len != fields.len) return false;
    for (elements, fields) |element, field| {
        if (!typesAssignable(element, field.ty)) return false;
    }
    return true;
}

fn isIntegerType(ty: Type) bool {
    return unwrapRefinement(ty).kind() == .integer;
}

fn unwrapRefinement(ty: Type) Type {
    return if (ty.refinementBaseType()) |base| base.* else ty;
}

const RefinementBounds = struct {
    base_type: Type,
    min_text: ?[]const u8 = null,
    max_text: ?[]const u8 = null,
};

fn typeSliceEql(lhs: []const Type, rhs: []const Type) bool {
    if (lhs.len != rhs.len) return false;
    for (lhs, rhs) |left, right| {
        if (!typeEql(left, right)) return false;
    }
    return true;
}

fn errorSetContains(errors: []const Type, needle: Type) bool {
    for (errors) |error_type| {
        if (typeEql(error_type, needle)) return true;
    }
    return false;
}

fn errorSetContainsAll(expected_errors: []const Type, actual_errors: []const Type) bool {
    for (actual_errors) |actual_error| {
        if (!errorSetContains(expected_errors, actual_error)) return false;
    }
    return true;
}

fn refinementArgSliceEql(lhs: []const ast.TypeArg, rhs: []const ast.TypeArg) bool {
    if (lhs.len != rhs.len) return false;
    for (lhs, rhs) |left, right| {
        switch (left) {
            .Type => |left_type| {
                if (right != .Type or left_type != right.Type) return false;
            },
            .Integer => |left_integer| {
                if (right != .Integer) return false;
                if (!std.mem.eql(u8, left_integer.text, right.Integer.text)) return false;
            },
        }
    }
    return true;
}

fn storeType(allocator: std.mem.Allocator, ty: Type) anyerror!*const Type {
    const stored = try allocator.create(Type);
    stored.* = ty;
    return stored;
}

fn integerAssignable(expected: model.IntegerType, actual: model.IntegerType) bool {
    if (expected.signed != null and actual.signed != null and expected.signed.? != actual.signed.?) return false;
    if (expected.bits == null or actual.bits == null) return true;
    return actual.bits.? <= expected.bits.?;
}

fn refinementSubtypeAssignable(expected: model.RefinementType, actual: model.RefinementType) bool {
    if (!typesAssignable(expected.base_type.*, actual.base_type.*) or !typesAssignable(actual.base_type.*, expected.base_type.*)) {
        return false;
    }
    if (std.mem.eql(u8, expected.name, "NonZeroAddress")) {
        return std.mem.eql(u8, actual.name, "NonZeroAddress");
    }
    if (refinementSemanticallyEqual(expected, actual)) return true;

    const expected_bounds = refinementBounds(expected) orelse return false;
    const actual_bounds = refinementBounds(actual) orelse return false;
    if (!typesAssignable(expected_bounds.base_type, actual_bounds.base_type) or
        !typesAssignable(actual_bounds.base_type, expected_bounds.base_type))
    {
        return false;
    }
    if (expected_bounds.min_text) |expected_min| {
        const actual_min = actual_bounds.min_text orelse return false;
        const cmp = compareIntegerText(actual_min, expected_min);
        if (cmp == .lt) return false;
    }
    if (expected_bounds.max_text) |expected_max| {
        const actual_max = actual_bounds.max_text orelse return false;
        const cmp = compareIntegerText(actual_max, expected_max);
        if (cmp == .gt) return false;
    }
    return true;
}

fn refinementSemanticallyEqual(lhs: model.RefinementType, rhs: model.RefinementType) bool {
    if (!std.mem.eql(u8, lhs.name, rhs.name)) return false;
    if (lhs.args.len != rhs.args.len) return false;
    for (lhs.args, rhs.args) |left, right| {
        switch (left) {
            .Integer => |left_integer| {
                if (right != .Integer) return false;
                if (compareIntegerText(left_integer.text, right.Integer.text) != .eq) return false;
            },
            .Type => {
                if (right != .Type) return false;
                // Refinement type arguments are semantically represented by base_type above.
            },
        }
    }
    return true;
}

fn refinementBounds(refinement: model.RefinementType) ?RefinementBounds {
    if (std.mem.eql(u8, refinement.name, "MinValue")) {
        if (refinement.args.len < 2 or refinement.args[1] != .Integer) return null;
        return .{
            .base_type = refinement.base_type.*,
            .min_text = refinement.args[1].Integer.text,
        };
    }
    if (std.mem.eql(u8, refinement.name, "MaxValue")) {
        if (refinement.args.len < 2 or refinement.args[1] != .Integer) return null;
        return .{
            .base_type = refinement.base_type.*,
            .max_text = refinement.args[1].Integer.text,
        };
    }
    if (std.mem.eql(u8, refinement.name, "InRange")) {
        if (refinement.args.len < 3 or refinement.args[1] != .Integer or refinement.args[2] != .Integer) return null;
        return .{
            .base_type = refinement.base_type.*,
            .min_text = refinement.args[1].Integer.text,
            .max_text = refinement.args[2].Integer.text,
        };
    }
    if (std.mem.eql(u8, refinement.name, "BasisPoints")) {
        return .{
            .base_type = refinement.base_type.*,
            .min_text = "0",
            .max_text = "10000",
        };
    }
    if (std.mem.eql(u8, refinement.name, "NonZero")) {
        return .{
            .base_type = refinement.base_type.*,
            .min_text = "1",
        };
    }
    if (std.mem.eql(u8, refinement.name, "Exact")) {
        if (refinement.args.len < 2 or refinement.args[1] != .Integer) return null;
        return .{
            .base_type = refinement.base_type.*,
            .min_text = refinement.args[1].Integer.text,
            .max_text = refinement.args[1].Integer.text,
        };
    }
    return null;
}

fn compareIntegerText(lhs_raw: []const u8, rhs_raw: []const u8) std.math.Order {
    const lhs = std.mem.trim(u8, lhs_raw, " \t\n\r");
    const rhs = std.mem.trim(u8, rhs_raw, " \t\n\r");
    const lhs_negative = lhs.len > 0 and lhs[0] == '-';
    const rhs_negative = rhs.len > 0 and rhs[0] == '-';
    if (lhs_negative != rhs_negative) return if (lhs_negative) .lt else .gt;

    const lhs_digits = trimLeadingZeros(if (lhs_negative) lhs[1..] else lhs);
    const rhs_digits = trimLeadingZeros(if (rhs_negative) rhs[1..] else rhs);
    if (lhs_digits.len != rhs_digits.len) {
        if (lhs_negative) {
            return if (lhs_digits.len > rhs_digits.len) .lt else .gt;
        }
        return if (lhs_digits.len < rhs_digits.len) .lt else .gt;
    }
    const ordered = std.mem.order(u8, lhs_digits, rhs_digits);
    return if (lhs_negative) switch (ordered) {
        .lt => .gt,
        .eq => .eq,
        .gt => .lt,
    } else ordered;
}

fn trimLeadingZeros(text: []const u8) []const u8 {
    var i: usize = 0;
    while (i < text.len and text[i] == '0') : (i += 1) {}
    return if (i == text.len) "0" else text[i..];
}

fn commonIntegerType(lhs: Type, rhs: Type) ?Type {
    if (lhs.kind() != .integer or rhs.kind() != .integer) return null;

    const left = lhs.integer;
    const right = rhs.integer;

    if (left.signed != null and right.signed != null and left.signed.? != right.signed.?) return null;

    if (left.bits == null) return rhs;
    if (right.bits == null) return lhs;

    if (left.bits.? > right.bits.?) return lhs;
    if (right.bits.? > left.bits.?) return rhs;

    if (left.signed == null and right.signed != null) return rhs;
    if (right.signed == null and left.signed != null) return lhs;
    return lhs;
}

fn parseArrayLen(size: ast.TypeArraySize) ?u32 {
    return switch (size) {
        .Integer => |literal| std.fmt.parseInt(u32, literal.text, 10) catch null,
        .Name => null,
    };
}

fn parseIntegerType(name: []const u8) ?model.IntegerType {
    if (name.len < 2) return null;
    const signed = switch (name[0]) {
        'u' => false,
        'i' => true,
        else => return null,
    };
    const bits = std.fmt.parseInt(u16, name[1..], 10) catch return null;
    return .{
        .bits = bits,
        .signed = signed,
        .spelling = name,
    };
}

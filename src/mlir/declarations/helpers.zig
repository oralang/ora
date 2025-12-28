// ============================================================================
// Declaration Lowering - Helper Functions
// ============================================================================

const std = @import("std");
const c = @import("mlir_c_api").c;
const lib = @import("ora_lib");
const h = @import("../helpers.zig");
const DeclarationLowerer = @import("mod.zig").DeclarationLowerer;

/// Create a file location from a source span
pub fn createFileLocation(self: *const DeclarationLowerer, span: lib.ast.SourceSpan) c.MlirLocation {
    return self.locations.createLocation(span);
}

/// Get the source span for any expression type
pub fn getExpressionSpan(_: *const DeclarationLowerer, expr: *const lib.ast.Expressions.ExprNode) lib.ast.SourceSpan {
    return switch (expr.*) {
        .Identifier => |ident| ident.span,
        .Literal => |lit| switch (lit) {
            .Integer => |int| int.span,
            .String => |str| str.span,
            .Bool => |bool_lit| bool_lit.span,
            .Address => |addr| addr.span,
            .Hex => |hex| hex.span,
            .Binary => |bin| bin.span,
            .Character => |char| char.span,
            .Bytes => |bytes| bytes.span,
        },
        .Binary => |bin| bin.span,
        .Unary => |unary| unary.span,
        .Assignment => |assign| assign.span,
        .CompoundAssignment => |comp_assign| comp_assign.span,
        .Call => |call| call.span,
        .Index => |index| index.span,
        .FieldAccess => |field| field.span,
        .Cast => |cast| cast.span,
        .Comptime => |comptime_expr| comptime_expr.span,
        .Old => |old| old.span,
        .Tuple => |tuple| tuple.span,
        .SwitchExpression => |switch_expr| switch_expr.span,
        .Quantified => |quantified| quantified.span,
        .Try => |try_expr| try_expr.span,
        .ErrorReturn => |error_ret| error_ret.span,
        .ErrorCast => |error_cast| error_cast.span,
        .Shift => |shift| shift.span,
        .StructInstantiation => |struct_inst| struct_inst.span,
        .AnonymousStruct => |anon_struct| anon_struct.span,
        .Range => |range| range.span,
        .LabeledBlock => |labeled_block| labeled_block.span,
        .Destructuring => |destructuring| destructuring.span,
        .EnumLiteral => |enum_lit| enum_lit.span,
        .ArrayLiteral => |array_lit| array_lit.span,
    };
}

/// Create a constant value from an attribute
pub fn createConstant(self: *const DeclarationLowerer, block: c.MlirBlock, attr: c.MlirAttribute, ty: c.MlirType, loc: c.MlirLocation) c.MlirValue {
    const const_op = c.oraArithConstantOpCreate(self.ctx, loc, ty, attr);
    if (const_op.ptr == null) {
        @panic("Failed to create declaration constant");
    }
    h.appendOp(block, const_op);
    return h.getResult(const_op, 0);
}

/// Create a placeholder operation for unsupported variable declarations
pub fn createVariablePlaceholder(self: *const DeclarationLowerer, var_decl: *const lib.ast.Statements.VariableDeclNode) c.MlirOperation {
    const loc = self.createFileLocation(var_decl.span);
    // add variable name as attribute
    const placeholder_ty = c.oraIntegerTypeCreate(self.ctx, 32);
    return c.oraVariablePlaceholderOpCreate(self.ctx, loc, h.strRef(var_decl.name), placeholder_ty);
}

/// Create a placeholder operation for module declarations
pub fn createModulePlaceholder(self: *const DeclarationLowerer, module_decl: *const lib.ast.ModuleNode) c.MlirOperation {
    const loc = self.createFileLocation(module_decl.span);
    var attrs = std.ArrayList(c.MlirNamedAttribute){};
    defer attrs.deinit(std.heap.page_allocator);

    if (module_decl.name) |name| {
        const name_ref = c.oraStringRefCreate(name.ptr, name.len);
        const name_attr = c.oraStringAttrCreate(self.ctx, name_ref);
        const name_id = h.identifier(self.ctx, "name");
        attrs.append(std.heap.page_allocator, c.oraNamedAttributeGet(name_id, name_attr)) catch {};
    }

    const module_name = if (module_decl.name) |name| name else "module";
    return c.oraModulePlaceholderOpCreate(
        self.ctx,
        loc,
        h.strRef(module_name),
        if (attrs.items.len == 0) null else attrs.items.ptr,
        attrs.items.len,
    );
}

/// Convert Ora TypeInfo to string representation for attributes
pub fn oraTypeToString(self: *const DeclarationLowerer, type_info: lib.ast.Types.TypeInfo, allocator: std.mem.Allocator) ![]const u8 {
    _ = self;
    if (type_info.ora_type) |ora_type| {
        return switch (ora_type) {
            .u8 => try std.fmt.allocPrint(allocator, "!ora.int<8,false>", .{}),
            .u16 => try std.fmt.allocPrint(allocator, "!ora.int<16,false>", .{}),
            .u32 => try std.fmt.allocPrint(allocator, "!ora.int<32,false>", .{}),
            .u64 => try std.fmt.allocPrint(allocator, "!ora.int<64,false>", .{}),
            .u128 => try std.fmt.allocPrint(allocator, "!ora.int<128,false>", .{}),
            .u256 => try std.fmt.allocPrint(allocator, "!ora.int<256,false>", .{}),
            .i8 => try std.fmt.allocPrint(allocator, "!ora.int<8,true>", .{}),
            .i16 => try std.fmt.allocPrint(allocator, "!ora.int<16,true>", .{}),
            .i32 => try std.fmt.allocPrint(allocator, "!ora.int<32,true>", .{}),
            .i64 => try std.fmt.allocPrint(allocator, "!ora.int<64,true>", .{}),
            .i128 => try std.fmt.allocPrint(allocator, "!ora.int<128,true>", .{}),
            .i256 => try std.fmt.allocPrint(allocator, "!ora.int<256,true>", .{}),
            .bool => try std.fmt.allocPrint(allocator, "!ora.bool", .{}),
            .address => try std.fmt.allocPrint(allocator, "!ora.address", .{}),
            .string => try std.fmt.allocPrint(allocator, "!ora.string", .{}),
            .bytes => try std.fmt.allocPrint(allocator, "!ora.bytes", .{}),
            .void => try std.fmt.allocPrint(allocator, "!ora.void", .{}),
            // for complex types, use simplified representation for now
            .struct_type => |name| try std.fmt.allocPrint(allocator, "!ora.struct<\"{s}\">", .{name}),
            .enum_type => |name| try std.fmt.allocPrint(allocator, "!ora.enum<\"{s}\">", .{name}),
            .contract_type => |name| try std.fmt.allocPrint(allocator, "!ora.contract<\"{s}\">", .{name}),
            else => try std.fmt.allocPrint(allocator, "!ora.unknown", .{}),
        };
    }
    return try std.fmt.allocPrint(allocator, "!ora.unknown", .{});
}

/// Enhanced function type creation with parameter default values (Requirements 6.3)
pub fn createFunctionType(self: *const DeclarationLowerer, func: *const lib.FunctionNode) c.MlirType {
    // create parameter types array
    var param_types = std.ArrayList(c.MlirType){};
    defer param_types.deinit(std.heap.page_allocator);

    for (func.parameters) |param| {
        const param_type = self.type_mapper.toMlirType(param.type_info);
        param_types.append(std.heap.page_allocator, param_type) catch {};
    }

    // create function type
    if (func.return_type_info) |ret_info| {
        const result_type = self.type_mapper.toMlirType(ret_info);
        return c.oraFunctionTypeGet(self.ctx, @intCast(param_types.items.len), if (param_types.items.len > 0) param_types.items.ptr else null, 1, @ptrCast(&result_type));
    } else {
        // functions with no return type should have 0 result types, not a 'none' type
        return c.oraFunctionTypeGet(self.ctx, @intCast(param_types.items.len), if (param_types.items.len > 0) param_types.items.ptr else null, 0, null);
    }
}

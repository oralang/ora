// ============================================================================
// Declaration Lowering - Globals and Constants
// ============================================================================

const std = @import("std");
const c = @import("mlir_c_api").c;
const lib = @import("ora_lib");
const constants = @import("../lower.zig");
const h = @import("../helpers.zig");
const DeclarationLowerer = @import("mod.zig").DeclarationLowerer;
const helpers = @import("helpers.zig");

/// Lower const declarations with global constant definitions (Requirements 7.6)
pub fn lowerConstDecl(self: *const DeclarationLowerer, const_decl: *const lib.ast.ConstantNode) c.MlirOperation {
    const loc = helpers.createFileLocation(self, const_decl.span);

    const name_ref = c.oraStringRefCreate(const_decl.name.ptr, const_decl.name.len);
    const declared_type = self.type_mapper.toMlirType(const_decl.typ);

    const result_type = if (const_decl.typ.ora_type) |ora_type| switch (ora_type) {
        .address, .non_zero_address => c.oraIntegerTypeCreate(self.ctx, 160),
        else => declared_type,
    } else declared_type;

    const value_attr = buildConstValueAttr(self, const_decl, result_type) orelse return c.MlirOperation{ .ptr = null };

    const const_op = c.oraConstOpCreate(self.ctx, loc, name_ref, value_attr, result_type);
    if (c.oraOperationIsNull(const_op)) {
        if (self.error_handler) |eh| {
            eh.reportError(.MlirOperationFailed, const_decl.span, "failed to create ora.const operation", null) catch {};
        }
        return c.MlirOperation{ .ptr = null };
    }

    // preserve legacy metadata for downstream tooling
    const sym_name_attr = c.oraStringAttrCreate(self.ctx, name_ref);
    c.oraOperationSetAttributeByName(const_op, h.strRef("sym_name"), sym_name_attr);
    const type_attr = c.oraTypeAttrCreateFromType(declared_type);
    c.oraOperationSetAttributeByName(const_op, h.strRef("type"), type_attr);

    const visibility_attr = switch (const_decl.visibility) {
        .Public => c.oraStringAttrCreate(self.ctx, h.strRef("pub")),
        .Private => c.oraStringAttrCreate(self.ctx, h.strRef("private")),
    };
    c.oraOperationSetAttributeByName(const_op, h.strRef("ora.visibility"), visibility_attr);

    const const_decl_attr = h.boolAttr(self.ctx, 1);
    c.oraOperationSetAttributeByName(const_op, h.strRef("ora.const_decl"), const_decl_attr);

    return const_op;
}

fn buildConstValueAttr(
    self: *const DeclarationLowerer,
    const_decl: *const lib.ast.ConstantNode,
    result_type: c.MlirType,
) ?c.MlirAttribute {
    const value_expr = const_decl.value;
    if (value_expr.* != .Literal) {
        if (self.error_handler) |eh| {
            eh.reportError(.UnsupportedFeature, const_decl.span, "const value must be a literal", "use a literal value in const declarations") catch {};
        }
        return null;
    }

    const literal = value_expr.Literal;
    return switch (literal) {
        .Integer => |int_lit| buildIntegerAttr(self, int_lit.value, result_type, const_decl.span),
        .Bool => |bool_lit| c.oraBoolAttrCreate(self.ctx, bool_lit.value),
        .Address => |addr_lit| buildAddressAttr(self, addr_lit.value, const_decl.span),
        .Character => |char_lit| c.oraIntegerAttrCreateI64FromType(result_type, @intCast(char_lit.value)),
        else => blk: {
            if (self.error_handler) |eh| {
                eh.reportError(.UnsupportedFeature, const_decl.span, "const literal type is not supported in MLIR lowering", "use an integer, bool, address, or character literal") catch {};
            }
            break :blk null;
        },
    };
}

fn buildIntegerAttr(
    self: *const DeclarationLowerer,
    value: []const u8,
    ty: c.MlirType,
    span: lib.ast.SourceSpan,
) ?c.MlirAttribute {
    var cleaned = std.ArrayList(u8){};
    defer cleaned.deinit(std.heap.page_allocator);

    for (value) |ch| {
        if (ch != '_') cleaned.append(std.heap.page_allocator, ch) catch {
            return null;
        };
    }

    const value_ref = c.oraStringRefCreate(cleaned.items.ptr, cleaned.items.len);
    const attr = c.oraIntegerAttrGetFromString(ty, value_ref);
    if (c.oraAttributeIsNull(attr)) {
        if (self.error_handler) |eh| {
            eh.reportError(.MalformedAst, span, "invalid integer constant literal", "use a valid integer literal") catch {};
        }
        return null;
    }
    return attr;
}

fn buildAddressAttr(
    self: *const DeclarationLowerer,
    value: []const u8,
    span: lib.ast.SourceSpan,
) ?c.MlirAttribute {
    const addr_str = if (std.mem.startsWith(u8, value, "0x")) value[2..] else value;
    if (addr_str.len != 40) {
        if (self.error_handler) |eh| {
            eh.reportError(.MalformedAst, span, "address literal must be 40 hex chars", "use a 20-byte hex address literal") catch {};
        }
        return null;
    }

    var parsed: u256 = 0;
    for (addr_str) |ch| {
        if (ch >= '0' and ch <= '9') {
            parsed = parsed * 16 + (ch - '0');
        } else if (ch >= 'a' and ch <= 'f') {
            parsed = parsed * 16 + (ch - 'a' + 10);
        } else if (ch >= 'A' and ch <= 'F') {
            parsed = parsed * 16 + (ch - 'A' + 10);
        } else {
            if (self.error_handler) |eh| {
                eh.reportError(.MalformedAst, span, "address literal contains non-hex characters", "use only 0-9 and a-f characters") catch {};
            }
            return null;
        }
    }

    const i160_ty = c.oraIntegerTypeCreate(self.ctx, 160);
    const attr = if (parsed <= std.math.maxInt(i64)) blk: {
        break :blk c.oraIntegerAttrCreateI64FromType(i160_ty, @intCast(parsed));
    } else blk: {
        var decimal_buf: [80]u8 = undefined;
        const decimal_str = std.fmt.bufPrint(&decimal_buf, "{}", .{parsed}) catch {
            if (self.error_handler) |eh| {
                eh.reportError(.MalformedAst, span, "failed to format address constant", null) catch {};
            }
            break :blk c.MlirAttribute{ .ptr = null };
        };
        const addr_ref = c.oraStringRefCreate(decimal_str.ptr, decimal_str.len);
        break :blk c.oraIntegerAttrGetFromString(i160_ty, addr_ref);
    };

    if (c.oraAttributeIsNull(attr)) {
        if (self.error_handler) |eh| {
            eh.reportError(.MalformedAst, span, "invalid address literal", "use a valid hex address") catch {};
        }
        return null;
    }
    return attr;
}

/// Lower immutable declarations with immutable global definitions and initialization constraints (Requirements 7.7)
pub fn lowerImmutableDecl(self: *const DeclarationLowerer, var_decl: *const lib.ast.Statements.VariableDeclNode) c.MlirOperation {
    const loc = helpers.createFileLocation(self, var_decl.span);

    // collect immutable attributes
    var attributes = std.ArrayList(c.MlirNamedAttribute){};
    defer attributes.deinit(std.heap.page_allocator);

    // add variable name
    const name_ref = c.oraStringRefCreate(var_decl.name.ptr, var_decl.name.len);
    const name_attr = c.oraStringAttrCreate(self.ctx, name_ref);
    const name_id = h.identifier(self.ctx, "sym_name");
    attributes.append(std.heap.page_allocator, c.oraNamedAttributeGet(name_id, name_attr)) catch {};

    // add variable type
    const var_type = self.type_mapper.toMlirType(var_decl.type_info);
    const type_attr = c.oraTypeAttrCreateFromType(var_type);
    const type_id = h.identifier(self.ctx, "type");
    attributes.append(std.heap.page_allocator, c.oraNamedAttributeGet(type_id, type_attr)) catch {};

    // add immutable constraint marker
    const immutable_attr = h.boolAttr(self.ctx, 1);
    const immutable_id = h.identifier(self.ctx, "ora.immutable");
    attributes.append(std.heap.page_allocator, c.oraNamedAttributeGet(immutable_id, immutable_attr)) catch {};

    // add initialization constraint - immutable variables must be initialized
    if (var_decl.value == null) {
        const requires_init_attr = h.boolAttr(self.ctx, 1);
        const requires_init_id = h.identifier(self.ctx, "ora.requires_init");
        attributes.append(std.heap.page_allocator, c.oraNamedAttributeGet(requires_init_id, requires_init_attr)) catch {};
    }

    const has_region = var_decl.value != null;
    const imm_op = c.oraImmutableDeclOpCreate(
        self.ctx,
        loc,
        &[_]c.MlirType{var_type},
        1,
        if (attributes.items.len == 0) null else attributes.items.ptr,
        attributes.items.len,
        if (has_region) 1 else 0,
        false,
    );

    if (has_region) {
        const block = c.oraOperationGetRegionBlock(imm_op, 0);
        if (c.oraBlockIsNull(block)) {
            @panic("ora.immutable missing body block");
        }
    }

    return imm_op;
}

/// Create global storage variable declaration
pub fn createGlobalDeclaration(self: *const DeclarationLowerer, var_decl: *const lib.ast.Statements.VariableDeclNode) c.MlirOperation {
    // use the type mapper to get the correct MLIR type (preserves signedness: u256 -> ui256, i256 -> si256)
    const var_type = self.type_mapper.toMlirType(var_decl.type_info);

    // the symbol table should already be populated in the first pass with the correct type
    // (using toMlirType which now returns i1 for booleans)
    // if it's not correct, that's a separate issue that needs to be fixed at the source

    // check if this is a map type - maps don't have scalar initializers
    const is_map_type = if (var_decl.type_info.ora_type) |ora_type| blk: {
        break :blk ora_type == .map;
    } else false;

    // determine the initial value based on the actual type
    // maps don't have initializers - use null attribute
    const init_attr = if (is_map_type) blk: {
        // maps are first-class types, no scalar initializer
        break :blk c.oraNullAttrCreate();
    } else if (var_decl.value) |_| blk: {
        // if there's an initial value expression, we need to lower it
        // for now, use a placeholder - the actual value should come from lowering the expression
        const zero_attr = c.oraIntegerAttrCreateI64FromType(var_type, 0);
        break :blk zero_attr;
    } else blk: {
        // no initial value - use zero of the correct type
        const zero_attr = c.oraIntegerAttrCreateI64FromType(var_type, 0);
        break :blk zero_attr;
    };

    // use the dialect helper function to create the global operation
    return self.ora_dialect.createGlobal(var_decl.name, var_type, init_attr, helpers.createFileLocation(self, var_decl.span));
}

/// Create memory global variable declaration
pub fn createMemoryGlobalDeclaration(self: *const DeclarationLowerer, var_decl: *const lib.ast.Statements.VariableDeclNode) c.MlirOperation {
    const loc = helpers.createFileLocation(self, var_decl.span);
    const var_type = c.oraIntegerTypeCreate(self.ctx, constants.DEFAULT_INTEGER_BITS); // default to i256
    return c.oraMemoryGlobalOpCreate(self.ctx, loc, h.strRef(var_decl.name), var_type);
}

/// Create transient storage global variable declaration
pub fn createTStoreGlobalDeclaration(self: *const DeclarationLowerer, var_decl: *const lib.ast.Statements.VariableDeclNode) c.MlirOperation {
    const loc = helpers.createFileLocation(self, var_decl.span);
    const var_type = c.oraIntegerTypeCreate(self.ctx, constants.DEFAULT_INTEGER_BITS); // default to i256
    return c.oraTStoreGlobalOpCreate(self.ctx, loc, h.strRef(var_decl.name), var_type);
}

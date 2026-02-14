// ============================================================================
// Monomorphization Engine
// ============================================================================
// Creates concrete copies of generic functions/structs by substituting
// type parameters with concrete types.
//
// Key: (function_name, [concrete_type_args...])
// Value: monomorphized FunctionNode with mangled name (e.g., max__u256)
// ============================================================================

const std = @import("std");
const ast = @import("../../ast.zig");
const type_info_mod = @import("../type_info.zig");
const TypeInfo = type_info_mod.TypeInfo;
const OraType = type_info_mod.OraType;
const comptime_eval = @import("../../comptime/mod.zig");
const log = @import("log");

const SourceSpan = ast.SourceSpan;
const ExprNode = ast.Expressions.ExprNode;
const StmtNode = ast.Statements.StmtNode;
const BlockNode = ast.Statements.BlockNode;
const FunctionNode = ast.FunctionNode;

pub const MonomorphError = error{OutOfMemory};

/// Monomorphization cache key
const CacheKey = struct {
    fn_name: []const u8,
    type_args: []const OraType,
};

/// Monomorphization engine
pub const Monomorphizer = struct {
    allocator: std.mem.Allocator,
    /// Cache: mangled name → monomorphized function
    cache: std.StringHashMap(FunctionNode),
    /// List of all monomorphized functions (for injection into AST)
    instances: std.ArrayList(FunctionNode),
    /// Ownership map: mangled function name -> owning contract name ("" for top-level)
    function_owners: std.StringHashMap([]const u8),
    /// Cache: mangled name → monomorphized struct
    struct_cache: std.StringHashMap(ast.StructDeclNode),
    /// List of all monomorphized structs (for injection into AST)
    struct_instances: std.ArrayList(ast.StructDeclNode),
    /// Ownership map: mangled struct name -> owning contract name ("" for top-level)
    struct_owners: std.StringHashMap([]const u8),

    pub fn init(allocator: std.mem.Allocator) Monomorphizer {
        return .{
            .allocator = allocator,
            .cache = std.StringHashMap(FunctionNode).init(allocator),
            .instances = std.ArrayList(FunctionNode){},
            .function_owners = std.StringHashMap([]const u8).init(allocator),
            .struct_cache = std.StringHashMap(ast.StructDeclNode).init(allocator),
            .struct_instances = std.ArrayList(ast.StructDeclNode){},
            .struct_owners = std.StringHashMap([]const u8).init(allocator),
        };
    }

    pub fn deinit(self: *Monomorphizer) void {
        self.cache.deinit();
        self.instances.deinit(self.allocator);
        self.function_owners.deinit();
        self.struct_cache.deinit();
        self.struct_instances.deinit(self.allocator);
        self.struct_owners.deinit();
    }

    /// Monomorphize a generic function with concrete type arguments.
    /// Returns the mangled name of the instantiated function.
    pub fn monomorphize(
        self: *Monomorphizer,
        generic_fn: *const FunctionNode,
        type_args: []const OraType,
        owner_contract: ?[]const u8,
    ) MonomorphError![]const u8 {
        const mangled = try mangleName(self.allocator, generic_fn.name, type_args);

        // Check cache
        if (self.cache.contains(mangled)) return mangled;

        // Build type substitution map: param_name → concrete OraType
        var type_map = std.StringHashMap(OraType).init(self.allocator);
        defer type_map.deinit();
        var ti: usize = 0;
        for (generic_fn.parameters) |param| {
            if (param.is_comptime) {
                if (param.type_info.ora_type) |ot| {
                    if (ot == .type) {
                        if (ti < type_args.len) {
                            try type_map.put(param.name, type_args[ti]);
                            ti += 1;
                        }
                    }
                }
            }
        }

        // Clone and substitute
        var mono_fn = try cloneFunction(self.allocator, generic_fn, mangled, &type_map);
        mono_fn.is_generic = false;
        mono_fn.type_param_names = &.{};
        // Remove comptime type params from the already-substituted parameter list
        mono_fn.parameters = try stripComptimeTypeParams(self.allocator, mono_fn.parameters);

        try self.cache.put(mangled, mono_fn);
        try self.function_owners.put(mangled, owner_contract orelse "");
        try self.instances.append(self.allocator, mono_fn);
        return mangled;
    }

    /// Monomorphize a generic struct with concrete type arguments.
    /// Returns the mangled name of the instantiated struct.
    pub fn monomorphizeStruct(
        self: *Monomorphizer,
        generic_struct: *const ast.StructDeclNode,
        type_args: []const OraType,
        owner_contract: ?[]const u8,
    ) MonomorphError![]const u8 {
        const mangled = try mangleName(self.allocator, generic_struct.name, type_args);

        // Check cache
        if (self.struct_cache.contains(mangled)) return mangled;

        // Build type substitution map: param_name → concrete OraType
        var type_map = std.StringHashMap(OraType).init(self.allocator);
        defer type_map.deinit();
        for (generic_struct.type_param_names, 0..) |name, i| {
            if (i < type_args.len) {
                try type_map.put(name, type_args[i]);
            }
        }

        // Clone and substitute field types
        const fields = self.allocator.alloc(ast.StructField, generic_struct.fields.len) catch
            return MonomorphError.OutOfMemory;
        for (generic_struct.fields, 0..) |field, i| {
            fields[i] = field;
            fields[i].type_info = substituteTypeInfo(field.type_info, &type_map);
        }

        const mono_struct = ast.StructDeclNode{
            .name = mangled,
            .fields = fields,
            .is_generic = false,
            .type_param_names = &.{},
            .span = generic_struct.span,
        };

        try self.struct_cache.put(mangled, mono_struct);
        try self.struct_owners.put(mangled, owner_contract orelse "");
        try self.struct_instances.append(self.allocator, mono_struct);
        return mangled;
    }

    /// Get all monomorphized function instances (for injection into the AST)
    pub fn getInstances(self: *const Monomorphizer) []const FunctionNode {
        return self.instances.items;
    }

    /// Get all monomorphized struct instances (for injection into the AST)
    pub fn getStructInstances(self: *const Monomorphizer) []const ast.StructDeclNode {
        return self.struct_instances.items;
    }

    /// Return the owning contract for a monomorphized function (null for top-level).
    pub fn getFunctionOwner(self: *const Monomorphizer, instance_name: []const u8) ?[]const u8 {
        const owner = self.function_owners.get(instance_name) orelse return null;
        if (owner.len == 0) return null;
        return owner;
    }

    /// Return the owning contract for a monomorphized struct (null for top-level).
    pub fn getStructOwner(self: *const Monomorphizer, instance_name: []const u8) ?[]const u8 {
        const owner = self.struct_owners.get(instance_name) orelse return null;
        if (owner.len == 0) return null;
        return owner;
    }
};

// ============================================================================
// Name Mangling
// ============================================================================

fn mangleName(allocator: std.mem.Allocator, base: []const u8, type_args: []const OraType) MonomorphError![]const u8 {
    var buf = std.ArrayList(u8){};
    buf.appendSlice(allocator, base) catch return MonomorphError.OutOfMemory;
    for (type_args) |t| {
        buf.appendSlice(allocator, "__") catch return MonomorphError.OutOfMemory;
        buf.appendSlice(allocator, t.toString()) catch return MonomorphError.OutOfMemory;
    }
    return buf.toOwnedSlice(allocator) catch return MonomorphError.OutOfMemory;
}

// ============================================================================
// AST Deep Cloning with Type Substitution
// ============================================================================

fn cloneFunction(
    alloc: std.mem.Allocator,
    src: *const FunctionNode,
    new_name: []const u8,
    type_map: *const std.StringHashMap(OraType),
) MonomorphError!FunctionNode {
    // Clone parameters (excluding comptime type params)
    const params = try cloneParams(alloc, src.parameters, type_map);

    // Clone return type
    var ret: ?TypeInfo = null;
    if (src.return_type_info) |ri| {
        ret = substituteTypeInfo(ri, type_map);
    }

    // Clone body
    const body = try cloneBlock(alloc, &src.body, type_map);

    // Clone requires/ensures
    const requires = try cloneExprSlice(alloc, src.requires_clauses, type_map);
    const ensures = try cloneExprSlice(alloc, src.ensures_clauses, type_map);
    const modifies = if (src.modifies_clause) |clause| try cloneExprSlice(alloc, clause, type_map) else null;

    return FunctionNode{
        .name = new_name,
        .parameters = params,
        .return_type_info = ret,
        .body = body,
        .visibility = src.visibility,
        .attributes = src.attributes,
        .requires_clauses = requires,
        .ensures_clauses = ensures,
        .modifies_clause = modifies,
        .is_ghost = src.is_ghost,
        .is_comptime_only = false,
        .is_generic = false,
        .type_param_names = &.{},
        .span = src.span,
    };
}

fn stripComptimeTypeParams(alloc: std.mem.Allocator, params: []const ast.ParameterNode) MonomorphError![]ast.ParameterNode {
    var count: usize = 0;
    for (params) |p| {
        if (p.is_comptime) {
            if (p.type_info.ora_type) |ot| {
                if (ot == .type) continue;
            }
        }
        count += 1;
    }
    const result = alloc.alloc(ast.ParameterNode, count) catch return MonomorphError.OutOfMemory;
    var idx: usize = 0;
    for (params) |p| {
        if (p.is_comptime) {
            if (p.type_info.ora_type) |ot| {
                if (ot == .type) continue;
            }
        }
        result[idx] = p;
        idx += 1;
    }
    return result;
}

fn cloneParams(
    alloc: std.mem.Allocator,
    params: []const ast.ParameterNode,
    type_map: *const std.StringHashMap(OraType),
) MonomorphError![]ast.ParameterNode {
    const result = alloc.alloc(ast.ParameterNode, params.len) catch return MonomorphError.OutOfMemory;
    for (params, 0..) |p, i| {
        result[i] = p;
        result[i].type_info = substituteTypeInfo(p.type_info, type_map);
    }
    return result;
}

fn substituteTypeInfo(ti: TypeInfo, type_map: *const std.StringHashMap(OraType)) TypeInfo {
    var result = ti;
    if (ti.ora_type) |ot| {
        if (ot == .type_parameter) {
            if (type_map.get(ot.type_parameter)) |concrete| {
                result.ora_type = concrete;
                result.category = concrete.getCategory();
            }
        }
    }
    return result;
}

// ============================================================================
// Expression Cloning
// ============================================================================

fn cloneExprSlice(
    alloc: std.mem.Allocator,
    exprs: []const *ExprNode,
    type_map: *const std.StringHashMap(OraType),
) MonomorphError![]*ExprNode {
    const result = alloc.alloc(*ExprNode, exprs.len) catch return MonomorphError.OutOfMemory;
    for (exprs, 0..) |e, i| {
        result[i] = cloneExpr(alloc, e, type_map) catch return MonomorphError.OutOfMemory;
    }
    return result;
}

fn cloneExprInline(
    alloc: std.mem.Allocator,
    expr: *const ExprNode,
    type_map: *const std.StringHashMap(OraType),
) MonomorphError!ExprNode {
    const cloned = try cloneExpr(alloc, expr, type_map);
    return cloned.*;
}

fn cloneExprInlineSlice(
    alloc: std.mem.Allocator,
    exprs: []const ExprNode,
    type_map: *const std.StringHashMap(OraType),
) MonomorphError![]ExprNode {
    const result = alloc.alloc(ExprNode, exprs.len) catch return MonomorphError.OutOfMemory;
    for (exprs, 0..) |*e, i| {
        result[i] = try cloneExprInline(alloc, e, type_map);
    }
    return result;
}

fn substituteLiteralTypeInfo(
    lit: *ast.Expressions.LiteralExpr,
    type_map: *const std.StringHashMap(OraType),
) void {
    switch (lit.*) {
        .Integer => |*v| v.type_info = substituteTypeInfo(v.type_info, type_map),
        .String => |*v| v.type_info = substituteTypeInfo(v.type_info, type_map),
        .Bool => |*v| v.type_info = substituteTypeInfo(v.type_info, type_map),
        .Address => |*v| v.type_info = substituteTypeInfo(v.type_info, type_map),
        .Hex => |*v| v.type_info = substituteTypeInfo(v.type_info, type_map),
        .Binary => |*v| v.type_info = substituteTypeInfo(v.type_info, type_map),
        .Character => |*v| v.type_info = substituteTypeInfo(v.type_info, type_map),
        .Bytes => |*v| v.type_info = substituteTypeInfo(v.type_info, type_map),
    }
}

fn cloneRangeExpr(
    alloc: std.mem.Allocator,
    r: ast.Expressions.RangeExpr,
    type_map: *const std.StringHashMap(OraType),
) MonomorphError!ast.Expressions.RangeExpr {
    var result = r;
    result.start = try cloneExpr(alloc, r.start, type_map);
    result.end = try cloneExpr(alloc, r.end, type_map);
    result.type_info = substituteTypeInfo(r.type_info, type_map);
    return result;
}

fn cloneSwitchPattern(
    alloc: std.mem.Allocator,
    pattern: ast.Expressions.SwitchPattern,
    type_map: *const std.StringHashMap(OraType),
) MonomorphError!ast.Expressions.SwitchPattern {
    return switch (pattern) {
        .Literal => |lit| blk: {
            var new_lit = lit;
            substituteLiteralTypeInfo(&new_lit.value, type_map);
            break :blk ast.Expressions.SwitchPattern{ .Literal = new_lit };
        },
        .Range => |r| ast.Expressions.SwitchPattern{ .Range = try cloneRangeExpr(alloc, r, type_map) },
        .EnumValue => |ev| ast.Expressions.SwitchPattern{ .EnumValue = ev },
        .Else => |sp| ast.Expressions.SwitchPattern{ .Else = sp },
    };
}

fn cloneSwitchBody(
    alloc: std.mem.Allocator,
    body: ast.Expressions.SwitchBody,
    type_map: *const std.StringHashMap(OraType),
) MonomorphError!ast.Expressions.SwitchBody {
    return switch (body) {
        .Expression => |expr_ptr| ast.Expressions.SwitchBody{ .Expression = try cloneExpr(alloc, expr_ptr, type_map) },
        .Block => |blk| blk: {
            const cloned = try cloneBlock(alloc, &blk, type_map);
            break :blk ast.Expressions.SwitchBody{ .Block = cloned };
        },
        .LabeledBlock => |lb| blk: {
            const cloned = try cloneBlock(alloc, &lb.block, type_map);
            break :blk ast.Expressions.SwitchBody{ .LabeledBlock = .{
                .label = lb.label,
                .block = cloned,
                .span = lb.span,
            } };
        },
    };
}

fn cloneSwitchCases(
    alloc: std.mem.Allocator,
    cases: []const ast.Expressions.SwitchCase,
    type_map: *const std.StringHashMap(OraType),
) MonomorphError![]ast.Expressions.SwitchCase {
    const result = alloc.alloc(ast.Expressions.SwitchCase, cases.len) catch return MonomorphError.OutOfMemory;
    for (cases, 0..) |c, i| {
        result[i] = .{
            .pattern = try cloneSwitchPattern(alloc, c.pattern, type_map),
            .body = try cloneSwitchBody(alloc, c.body, type_map),
            .span = c.span,
        };
    }
    return result;
}

fn cloneDestructuringPattern(
    alloc: std.mem.Allocator,
    pattern: ast.Expressions.DestructuringPattern,
) MonomorphError!ast.Expressions.DestructuringPattern {
    return switch (pattern) {
        .Struct => |fields| blk: {
            const copied = alloc.alloc(ast.Expressions.StructDestructureField, fields.len) catch return MonomorphError.OutOfMemory;
            @memcpy(copied, fields);
            break :blk ast.Expressions.DestructuringPattern{ .Struct = copied };
        },
        .Tuple => |names| blk: {
            const copied = alloc.alloc([]const u8, names.len) catch return MonomorphError.OutOfMemory;
            @memcpy(copied, names);
            break :blk ast.Expressions.DestructuringPattern{ .Tuple = copied };
        },
        .Array => |names| blk: {
            const copied = alloc.alloc([]const u8, names.len) catch return MonomorphError.OutOfMemory;
            @memcpy(copied, names);
            break :blk ast.Expressions.DestructuringPattern{ .Array = copied };
        },
    };
}

fn cloneLoopPattern(
    alloc: std.mem.Allocator,
    pattern: ast.Statements.LoopPattern,
) MonomorphError!ast.Statements.LoopPattern {
    return switch (pattern) {
        .Single => |single| ast.Statements.LoopPattern{ .Single = single },
        .IndexPair => |pair| ast.Statements.LoopPattern{ .IndexPair = pair },
        .Destructured => |destructured| ast.Statements.LoopPattern{ .Destructured = .{
            .pattern = try cloneDestructuringPattern(alloc, destructured.pattern),
            .span = destructured.span,
        } },
    };
}

fn cloneStructInstantiationFields(
    alloc: std.mem.Allocator,
    fields: []const ast.Expressions.StructInstantiationField,
    type_map: *const std.StringHashMap(OraType),
) MonomorphError![]ast.Expressions.StructInstantiationField {
    const result = alloc.alloc(ast.Expressions.StructInstantiationField, fields.len) catch return MonomorphError.OutOfMemory;
    for (fields, 0..) |field, i| {
        result[i] = field;
        result[i].value = try cloneExpr(alloc, field.value, type_map);
    }
    return result;
}

fn cloneAnonymousStructFields(
    alloc: std.mem.Allocator,
    fields: []const ast.Expressions.AnonymousStructField,
    type_map: *const std.StringHashMap(OraType),
) MonomorphError![]ast.Expressions.AnonymousStructField {
    const result = alloc.alloc(ast.Expressions.AnonymousStructField, fields.len) catch return MonomorphError.OutOfMemory;
    for (fields, 0..) |field, i| {
        result[i] = field;
        result[i].value = try cloneExpr(alloc, field.value, type_map);
    }
    return result;
}

fn cloneParameterNodes(
    alloc: std.mem.Allocator,
    params: []const ast.ParameterNode,
    type_map: *const std.StringHashMap(OraType),
) MonomorphError![]ast.ParameterNode {
    const result = alloc.alloc(ast.ParameterNode, params.len) catch return MonomorphError.OutOfMemory;
    for (params, 0..) |p, i| {
        result[i] = p;
        result[i].type_info = substituteTypeInfo(p.type_info, type_map);
    }
    return result;
}

fn cloneExpr(
    alloc: std.mem.Allocator,
    expr: *const ExprNode,
    type_map: *const std.StringHashMap(OraType),
) MonomorphError!*ExprNode {
    const new = alloc.create(ExprNode) catch return MonomorphError.OutOfMemory;
    new.* = expr.*;

    // Substitute type_info on expression variants that carry them
    switch (new.*) {
        .Identifier => |*id| {
            id.type_info = substituteTypeInfo(id.type_info, type_map);
        },
        .Binary => |*bin| {
            bin.lhs = try cloneExpr(alloc, bin.lhs, type_map);
            bin.rhs = try cloneExpr(alloc, bin.rhs, type_map);
            bin.type_info = substituteTypeInfo(bin.type_info, type_map);
        },
        .Unary => |*un| {
            un.operand = try cloneExpr(alloc, un.operand, type_map);
            un.type_info = substituteTypeInfo(un.type_info, type_map);
        },
        .Call => |*call| {
            const args = alloc.alloc(*ExprNode, call.arguments.len) catch return MonomorphError.OutOfMemory;
            for (call.arguments, 0..) |a, i| {
                args[i] = try cloneExpr(alloc, a, type_map);
            }
            call.arguments = args;
            call.callee = try cloneExpr(alloc, call.callee, type_map);
            call.type_info = substituteTypeInfo(call.type_info, type_map);
        },
        .FieldAccess => |*fa| {
            fa.target = try cloneExpr(alloc, fa.target, type_map);
            fa.type_info = substituteTypeInfo(fa.type_info, type_map);
        },
        .Index => |*idx| {
            idx.target = try cloneExpr(alloc, idx.target, type_map);
            idx.index = try cloneExpr(alloc, idx.index, type_map);
        },
        .Cast => |*cast| {
            cast.operand = try cloneExpr(alloc, cast.operand, type_map);
            cast.target_type = substituteTypeInfo(cast.target_type, type_map);
        },
        .Assignment => |*asgn| {
            asgn.target = try cloneExpr(alloc, asgn.target, type_map);
            asgn.value = try cloneExpr(alloc, asgn.value, type_map);
        },
        .CompoundAssignment => |*ca| {
            ca.target = try cloneExpr(alloc, ca.target, type_map);
            ca.value = try cloneExpr(alloc, ca.value, type_map);
        },
        .Comptime => |*ct| {
            ct.block = try cloneBlock(alloc, &ct.block, type_map);
            ct.type_info = substituteTypeInfo(ct.type_info, type_map);
        },
        .Old => |*old_expr| {
            old_expr.expr = try cloneExpr(alloc, old_expr.expr, type_map);
        },
        .Quantified => |*q| {
            q.variable_type = substituteTypeInfo(q.variable_type, type_map);
            if (q.condition) |cond| {
                q.condition = try cloneExpr(alloc, cond, type_map);
            }
            q.body = try cloneExpr(alloc, q.body, type_map);
        },
        .Tuple => |*t| {
            t.elements = try cloneExprSlice(alloc, t.elements, type_map);
        },
        .SwitchExpression => |*sw| {
            sw.condition = try cloneExpr(alloc, sw.condition, type_map);
            sw.cases = try cloneSwitchCases(alloc, sw.cases, type_map);
            if (sw.default_case) |default_block| {
                sw.default_case = try cloneBlock(alloc, &default_block, type_map);
            }
        },
        .Try => |*te| {
            te.expr = try cloneExpr(alloc, te.expr, type_map);
        },
        .ErrorReturn => |*er| {
            if (er.parameters) |params| {
                const cloned = alloc.alloc(*ExprNode, params.len) catch return MonomorphError.OutOfMemory;
                for (params, 0..) |p, i| {
                    cloned[i] = try cloneExpr(alloc, p, type_map);
                }
                er.parameters = cloned;
            }
        },
        .ErrorCast => |*ec| {
            ec.operand = try cloneExpr(alloc, ec.operand, type_map);
            ec.target_type = substituteTypeInfo(ec.target_type, type_map);
        },
        .Shift => |*sh| {
            sh.mapping = try cloneExpr(alloc, sh.mapping, type_map);
            sh.source = try cloneExpr(alloc, sh.source, type_map);
            sh.dest = try cloneExpr(alloc, sh.dest, type_map);
            sh.amount = try cloneExpr(alloc, sh.amount, type_map);
        },
        .StructInstantiation => |*si| {
            si.struct_name = try cloneExpr(alloc, si.struct_name, type_map);
            si.fields = try cloneStructInstantiationFields(alloc, si.fields, type_map);
        },
        .AnonymousStruct => |*anon| {
            anon.fields = try cloneAnonymousStructFields(alloc, anon.fields, type_map);
        },
        .Range => |*range_expr| {
            range_expr.start = try cloneExpr(alloc, range_expr.start, type_map);
            range_expr.end = try cloneExpr(alloc, range_expr.end, type_map);
            range_expr.type_info = substituteTypeInfo(range_expr.type_info, type_map);
        },
        .LabeledBlock => |*lb| {
            lb.block = try cloneBlock(alloc, &lb.block, type_map);
        },
        .Destructuring => |*destr| {
            destr.pattern = try cloneDestructuringPattern(alloc, destr.pattern);
            destr.value = try cloneExpr(alloc, destr.value, type_map);
        },
        .ArrayLiteral => |*arr| {
            arr.elements = try cloneExprSlice(alloc, arr.elements, type_map);
            if (arr.element_type) |et| {
                arr.element_type = substituteTypeInfo(et, type_map);
            }
        },
        .Literal => |*lit| {
            substituteLiteralTypeInfo(lit, type_map);
        },
        .EnumLiteral => {},
    }
    return new;
}

// ============================================================================
// Statement & Block Cloning
// ============================================================================

fn cloneBlock(
    alloc: std.mem.Allocator,
    block: *const BlockNode,
    type_map: *const std.StringHashMap(OraType),
) MonomorphError!BlockNode {
    const stmts = alloc.alloc(StmtNode, block.statements.len) catch return MonomorphError.OutOfMemory;
    for (block.statements, 0..) |*s, i| {
        stmts[i] = try cloneStmt(alloc, s, type_map);
    }
    return BlockNode{ .statements = stmts, .span = block.span };
}

fn cloneStmt(
    alloc: std.mem.Allocator,
    stmt: *const StmtNode,
    type_map: *const std.StringHashMap(OraType),
) MonomorphError!StmtNode {
    return switch (stmt.*) {
        .Expr => |*e| blk: {
            const cloned = try cloneExpr(alloc, e, type_map);
            break :blk StmtNode{ .Expr = cloned.* };
        },
        .Return => |*ret| blk: {
            var new_ret = ret.*;
            if (ret.value) |*v| {
                // value is ?ExprNode (inline), get pointer to it for cloneExpr
                const cloned = try cloneExpr(alloc, v, type_map);
                new_ret.value = cloned.*;
            }
            break :blk StmtNode{ .Return = new_ret };
        },
        .VariableDecl => |vd| blk: {
            var new_vd = vd;
            new_vd.type_info = substituteTypeInfo(vd.type_info, type_map);
            if (vd.value) |v| {
                // value is ?*ExprNode (pointer)
                new_vd.value = try cloneExpr(alloc, v, type_map);
            }
            break :blk StmtNode{ .VariableDecl = new_vd };
        },
        .CompoundAssignment => |ca| blk: {
            var new_ca = ca;
            new_ca.target = try cloneExpr(alloc, ca.target, type_map);
            new_ca.value = try cloneExpr(alloc, ca.value, type_map);
            break :blk StmtNode{ .CompoundAssignment = new_ca };
        },
        .If => |if_stmt| blk: {
            var new_if = if_stmt;
            // Deep-clone condition (inline ExprNode)
            const cond_clone = try cloneExpr(alloc, &if_stmt.condition, type_map);
            new_if.condition = cond_clone.*;
            new_if.then_branch = try cloneBlock(alloc, &if_stmt.then_branch, type_map);
            if (if_stmt.else_branch) |*eb| {
                new_if.else_branch = try cloneBlock(alloc, eb, type_map);
            }
            break :blk StmtNode{ .If = new_if };
        },
        .While => |wh| blk: {
            var new_wh = wh;
            const cond_clone = try cloneExpr(alloc, &wh.condition, type_map);
            new_wh.condition = cond_clone.*;
            new_wh.body = try cloneBlock(alloc, &wh.body, type_map);
            new_wh.invariants = try cloneExprInlineSlice(alloc, wh.invariants, type_map);
            if (wh.decreases) |dec| {
                new_wh.decreases = try cloneExpr(alloc, dec, type_map);
            }
            if (wh.increases) |inc| {
                new_wh.increases = try cloneExpr(alloc, inc, type_map);
            }
            break :blk StmtNode{ .While = new_wh };
        },
        .ForLoop => |for_loop| blk: {
            var new_for = for_loop;
            const iterable = try cloneExpr(alloc, &for_loop.iterable, type_map);
            new_for.iterable = iterable.*;
            new_for.pattern = try cloneLoopPattern(alloc, for_loop.pattern);
            new_for.body = try cloneBlock(alloc, &for_loop.body, type_map);
            new_for.invariants = try cloneExprInlineSlice(alloc, for_loop.invariants, type_map);
            if (for_loop.decreases) |dec| {
                new_for.decreases = try cloneExpr(alloc, dec, type_map);
            }
            if (for_loop.increases) |inc| {
                new_for.increases = try cloneExpr(alloc, inc, type_map);
            }
            break :blk StmtNode{ .ForLoop = new_for };
        },
        .Break => |br| blk: {
            var new_br = br;
            if (br.value) |value| {
                new_br.value = try cloneExpr(alloc, value, type_map);
            }
            break :blk StmtNode{ .Break = new_br };
        },
        .Continue => |cont| blk: {
            var new_cont = cont;
            if (cont.value) |value| {
                new_cont.value = try cloneExpr(alloc, value, type_map);
            }
            break :blk StmtNode{ .Continue = new_cont };
        },
        .Log => |log_stmt| blk: {
            var new_log = log_stmt;
            new_log.args = try cloneExprInlineSlice(alloc, log_stmt.args, type_map);
            break :blk StmtNode{ .Log = new_log };
        },
        .Lock => |lock_stmt| blk: {
            var new_lock = lock_stmt;
            const path = try cloneExpr(alloc, &lock_stmt.path, type_map);
            new_lock.path = path.*;
            break :blk StmtNode{ .Lock = new_lock };
        },
        .Unlock => |unlock_stmt| blk: {
            var new_unlock = unlock_stmt;
            const path = try cloneExpr(alloc, &unlock_stmt.path, type_map);
            new_unlock.path = path.*;
            break :blk StmtNode{ .Unlock = new_unlock };
        },
        .Assert => |assert_stmt| blk: {
            var new_assert = assert_stmt;
            const cond = try cloneExpr(alloc, &assert_stmt.condition, type_map);
            new_assert.condition = cond.*;
            break :blk StmtNode{ .Assert = new_assert };
        },
        .Invariant => |inv_stmt| blk: {
            var new_inv = inv_stmt;
            const cond = try cloneExpr(alloc, &inv_stmt.condition, type_map);
            new_inv.condition = cond.*;
            break :blk StmtNode{ .Invariant = new_inv };
        },
        .Requires => |req_stmt| blk: {
            var new_req = req_stmt;
            const cond = try cloneExpr(alloc, &req_stmt.condition, type_map);
            new_req.condition = cond.*;
            break :blk StmtNode{ .Requires = new_req };
        },
        .Ensures => |ens_stmt| blk: {
            var new_ens = ens_stmt;
            const cond = try cloneExpr(alloc, &ens_stmt.condition, type_map);
            new_ens.condition = cond.*;
            break :blk StmtNode{ .Ensures = new_ens };
        },
        .Assume => |assume_stmt| blk: {
            var new_assume = assume_stmt;
            const cond = try cloneExpr(alloc, &assume_stmt.condition, type_map);
            new_assume.condition = cond.*;
            break :blk StmtNode{ .Assume = new_assume };
        },
        .Havoc => stmt.*,
        .Switch => |sw| blk: {
            var new_sw = sw;
            const cond = try cloneExpr(alloc, &sw.condition, type_map);
            new_sw.condition = cond.*;
            new_sw.cases = try cloneSwitchCases(alloc, sw.cases, type_map);
            if (sw.default_case) |default_block| {
                new_sw.default_case = try cloneBlock(alloc, &default_block, type_map);
            }
            break :blk StmtNode{ .Switch = new_sw };
        },
        .LabeledBlock => |lb| blk: {
            var new_lb = lb;
            new_lb.block = try cloneBlock(alloc, &lb.block, type_map);
            break :blk StmtNode{ .LabeledBlock = new_lb };
        },
        .ErrorDecl => |ed| blk: {
            var new_ed = ed;
            if (ed.parameters) |params| {
                new_ed.parameters = try cloneParameterNodes(alloc, params, type_map);
            }
            break :blk StmtNode{ .ErrorDecl = new_ed };
        },
        .TryBlock => |tb| blk: {
            var new_tb = tb;
            new_tb.try_block = try cloneBlock(alloc, &tb.try_block, type_map);
            if (tb.catch_block) |catch_block| {
                new_tb.catch_block = ast.Statements.CatchBlock{
                    .error_variable = catch_block.error_variable,
                    .block = try cloneBlock(alloc, &catch_block.block, type_map),
                    .span = catch_block.span,
                };
            }
            break :blk StmtNode{ .TryBlock = new_tb };
        },
        .DestructuringAssignment => |da| blk: {
            var new_da = da;
            new_da.pattern = try cloneDestructuringPattern(alloc, da.pattern);
            new_da.value = try cloneExpr(alloc, da.value, type_map);
            break :blk StmtNode{ .DestructuringAssignment = new_da };
        },
    };
}

// ============================================================================
// Utility: Extract type args from a call to a generic function
// ============================================================================

/// Given a call expression and the generic function it targets, extract the
/// concrete OraType values for each comptime type parameter.
pub fn extractTypeArgs(
    generic_fn: *const FunctionNode,
    call: *const ast.Expressions.CallExpr,
    allocator: std.mem.Allocator,
) ?[]const OraType {
    var count: usize = 0;
    for (generic_fn.parameters) |p| {
        if (p.is_comptime) {
            if (p.type_info.ora_type) |ot| {
                if (ot == .type) count += 1;
            }
        }
    }
    if (count == 0) return null;

    const result = allocator.alloc(OraType, count) catch return null;
    var idx: usize = 0;
    var arg_idx: usize = 0;
    for (generic_fn.parameters) |p| {
        if (p.is_comptime) {
            if (p.type_info.ora_type) |ot| {
                if (ot == .type) {
                    if (arg_idx < call.arguments.len) {
                        if (resolveTypeFromExpr(call.arguments[arg_idx])) |concrete| {
                            if (idx < result.len) {
                                result[idx] = concrete;
                                idx += 1;
                            }
                        }
                    }
                }
            }
        }
        arg_idx += 1;
    }
    if (idx == count) {
        return result;
    }
    allocator.free(result);
    return null;
}

/// Resolve a concrete OraType from an expression that represents a type name.
pub fn resolveTypeFromExpr(expr: *const ExprNode) ?OraType {
    switch (expr.*) {
        .Identifier => |id| return nameToOraType(id.name),
        else => return null,
    }
}

pub fn nameToOraType(name: []const u8) ?OraType {
    if (std.mem.eql(u8, name, "u8")) return OraType{ .u8 = {} };
    if (std.mem.eql(u8, name, "u16")) return OraType{ .u16 = {} };
    if (std.mem.eql(u8, name, "u32")) return OraType{ .u32 = {} };
    if (std.mem.eql(u8, name, "u64")) return OraType{ .u64 = {} };
    if (std.mem.eql(u8, name, "u128")) return OraType{ .u128 = {} };
    if (std.mem.eql(u8, name, "u256")) return OraType{ .u256 = {} };
    if (std.mem.eql(u8, name, "i8")) return OraType{ .i8 = {} };
    if (std.mem.eql(u8, name, "i16")) return OraType{ .i16 = {} };
    if (std.mem.eql(u8, name, "i32")) return OraType{ .i32 = {} };
    if (std.mem.eql(u8, name, "i64")) return OraType{ .i64 = {} };
    if (std.mem.eql(u8, name, "i128")) return OraType{ .i128 = {} };
    if (std.mem.eql(u8, name, "i256")) return OraType{ .i256 = {} };
    if (std.mem.eql(u8, name, "bool")) return OraType{ .bool = {} };
    if (std.mem.eql(u8, name, "address")) return OraType{ .address = {} };
    if (std.mem.eql(u8, name, "string")) return OraType{ .string = {} };
    if (std.mem.eql(u8, name, "bytes")) return OraType{ .bytes = {} };
    return null;
}

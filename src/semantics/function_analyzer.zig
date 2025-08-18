const std = @import("std");
const ast = @import("../ast.zig");
const state = @import("state.zig");
const expr = @import("expression_analyzer.zig");
const locals = @import("locals_binder.zig");

fn copyOraTypeOwned(allocator: std.mem.Allocator, src: ast.OraType) !ast.OraType {
    switch (src) {
        ._union => |members| {
            const new_members = try allocator.alloc(ast.OraType, members.len);
            @memcpy(new_members, members);
            return ast.OraType{ ._union = new_members };
        },
        .error_union => |succ_ptr| {
            const new_succ_ptr = try allocator.create(ast.OraType);
            new_succ_ptr.* = succ_ptr.*;
            return ast.OraType{ .error_union = new_succ_ptr };
        },
        else => return src,
    }
}

pub fn collectFunctionSymbols(table: *state.SymbolTable, parent: *state.Scope, f: *const ast.FunctionNode) !void {
    const fn_scope = try table.allocator.create(state.Scope);
    fn_scope.* = state.Scope.init(table.allocator, parent, f.name);
    try table.scopes.append(fn_scope);
    try table.function_scopes.put(f.name, fn_scope);
    // Parameters
    for (f.parameters) |p| {
        const sym = state.Symbol{ .name = p.name, .kind = .Param, .typ = p.type_info, .span = p.span, .mutable = p.is_mutable };
        _ = try table.declare(fn_scope, sym);
    }

    // Create a function symbol type from parameters and return type
    var param_types = std.ArrayList(ast.OraType).init(table.allocator);
    defer param_types.deinit();
    for (f.parameters) |p| {
        if (p.type_info.ora_type) |ot| {
            try param_types.append(ot);
        } else {
            try param_types.append(.u256); // fallback for unknown param types
        }
    }
    const params_slice = try table.allocator.alloc(ast.OraType, param_types.items.len);
    for (param_types.items, 0..) |t, i| params_slice[i] = t;

    // Include return type in function type if present
    var ret_ptr: ?*const ast.OraType = null;
    if (f.return_type_info) |rt| {
        if (rt.ora_type) |ot| {
            const heap_rt = try table.allocator.create(ast.OraType);
            heap_rt.* = try copyOraTypeOwned(table.allocator, ot);
            ret_ptr = heap_rt;
        }
    }
    const fn_type = ast.type_info.FunctionType{ .params = params_slice, .return_type = ret_ptr };
    const new_ti = ast.TypeInfo.fromOraType(.{ .function = fn_type });
    // Record success type of error unions for quick checks
    if (f.return_type_info) |rt| {
        if (rt.ora_type) |ot| switch (ot) {
            .error_union => |succ_ptr| try table.function_success_types.put(f.name, @constCast(succ_ptr).*),
            ._union => |members| {
                var i: usize = 0;
                while (i < members.len) : (i += 1) {
                    switch (members[i]) {
                        .error_union => |succ_ptr| {
                            // Prefer the first error_union member
                            try table.function_success_types.put(f.name, @constCast(succ_ptr).*);
                            break;
                        },
                        else => {},
                    }
                }
            },
            else => {},
        };
    }
    // Update existing declaration if present; otherwise declare
    if (parent.findInCurrent(f.name)) |idx| {
        var existing = &parent.symbols.items[idx];
        if (existing.typ_owned) {
            @import("../ast/type_info.zig").deinitTypeInfo(table.allocator, &existing.typ.?);
        }
        existing.typ = new_ti;
        existing.typ_owned = true;
    } else {
        const fn_sym = state.Symbol{ .name = f.name, .kind = .Function, .typ = new_ti, .span = f.span, .typ_owned = true };
        _ = try table.declare(parent, fn_sym);
    }

    // Capture allowed error tags from the function return type if it is an error union with explicit union members
    if (f.return_type_info) |rt| {
        if (rt.ora_type) |ot| switch (ot) {
            ._union => |members| {
                // Collect members that are error tags by name (enum error-type modeling TBD)
                var list = std.ArrayList([]const u8).init(table.allocator);
                defer list.deinit();
                for (members) |m| switch (m) {
                    .struct_type, .enum_type, .contract_type => |name| {
                        // Tentative: treat named types as allowable error tags by name
                        try list.append(name);
                    },
                    else => {},
                };
                const slice = try list.toOwnedSlice();
                try table.function_allowed_errors.put(f.name, slice);
            },
            else => {},
        };
    }
}

// ============================================================================
// Function Analyzer
// ============================================================================
//
// Collects function symbols, creates scopes, and processes parameters.
//
// RESPONSIBILITIES:
//   • Create function scope
//   • Collect parameters as symbols
//   • Build function type (params + return)
//   • Handle error unions and union returns
//
// ============================================================================

const std = @import("std");
const ast = @import("../ast.zig");
const state = @import("state.zig");
const expr = @import("expression_analyzer.zig");
const locals = @import("locals_binder.zig");

pub fn copyOraTypeOwned(allocator: std.mem.Allocator, src: ast.Types.OraType) !ast.Types.OraType {
    switch (src) {
        ._union => |members| {
            const new_members = try allocator.alloc(ast.Types.OraType, members.len);
            for (members, 0..) |member, i| {
                new_members[i] = try copyOraTypeOwned(allocator, member);
            }
            return ast.Types.OraType{ ._union = new_members };
        },
        .error_union => |succ_ptr| {
            const new_succ_ptr = try allocator.create(ast.Types.OraType);
            new_succ_ptr.* = try copyOraTypeOwned(allocator, succ_ptr.*);
            return ast.Types.OraType{ .error_union = new_succ_ptr };
        },
        .slice => |elem_type_ptr| {
            const new_elem_type_ptr = try allocator.create(ast.Types.OraType);
            new_elem_type_ptr.* = try copyOraTypeOwned(allocator, elem_type_ptr.*);
            return ast.Types.OraType{ .slice = new_elem_type_ptr };
        },
        // refinement types need to copy the base pointer
        .min_value => |ref| {
            const new_base = try allocator.create(ast.Types.OraType);
            new_base.* = try copyOraTypeOwned(allocator, ref.base.*);
            return ast.Types.OraType{ .min_value = .{ .base = new_base, .min = ref.min } };
        },
        .max_value => |ref| {
            const new_base = try allocator.create(ast.Types.OraType);
            new_base.* = try copyOraTypeOwned(allocator, ref.base.*);
            return ast.Types.OraType{ .max_value = .{ .base = new_base, .max = ref.max } };
        },
        .in_range => |ref| {
            const new_base = try allocator.create(ast.Types.OraType);
            new_base.* = try copyOraTypeOwned(allocator, ref.base.*);
            return ast.Types.OraType{ .in_range = .{ .base = new_base, .min = ref.min, .max = ref.max } };
        },
        .scaled => |s| {
            const new_base = try allocator.create(ast.Types.OraType);
            new_base.* = try copyOraTypeOwned(allocator, s.base.*);
            return ast.Types.OraType{ .scaled = .{ .base = new_base, .decimals = s.decimals } };
        },
        .non_zero_address => {
            // non-zero address is a simple refinement, no base to copy
            return src;
        },
        .array => |arr| {
            // copy the element type recursively
            const new_elem = try allocator.create(ast.Types.OraType);
            new_elem.* = try copyOraTypeOwned(allocator, arr.elem.*);
            return ast.Types.OraType{ .array = .{ .elem = new_elem, .len = arr.len } };
        },
        .map => |mapping| {
            const new_key = try allocator.create(ast.Types.OraType);
            new_key.* = try copyOraTypeOwned(allocator, mapping.key.*);
            const new_value = try allocator.create(ast.Types.OraType);
            new_value.* = try copyOraTypeOwned(allocator, mapping.value.*);
            return ast.Types.OraType{ .map = .{ .key = new_key, .value = new_value } };
        },
        .tuple => |members| {
            const new_members = try allocator.alloc(ast.Types.OraType, members.len);
            for (members, 0..) |member, i| {
                new_members[i] = try copyOraTypeOwned(allocator, member);
            }
            return ast.Types.OraType{ .tuple = new_members };
        },
        .function => |func| {
            const new_params = try allocator.alloc(ast.Types.OraType, func.params.len);
            for (func.params, 0..) |param, i| {
                new_params[i] = try copyOraTypeOwned(allocator, param);
            }
            var new_ret: ?*const ast.Types.OraType = null;
            if (func.return_type) |ret_ptr| {
                const new_ret_ptr = try allocator.create(ast.Types.OraType);
                new_ret_ptr.* = try copyOraTypeOwned(allocator, ret_ptr.*);
                new_ret = new_ret_ptr;
            }
            return ast.Types.OraType{ .function = .{ .params = new_params, .return_type = new_ret } };
        },
        .anonymous_struct => |fields| {
            const new_fields = try allocator.alloc(ast.Types.AnonymousStructFieldType, fields.len);
            for (fields, 0..) |field, i| {
                const new_field_type = try allocator.create(ast.Types.OraType);
                new_field_type.* = try copyOraTypeOwned(allocator, field.typ.*);
                new_fields[i] = .{ .name = field.name, .typ = new_field_type };
            }
            return ast.Types.OraType{ .anonymous_struct = new_fields };
        },
        .exact => |e| {
            // copy the exact type recursively
            const new_e = try allocator.create(ast.Types.OraType);
            new_e.* = try copyOraTypeOwned(allocator, e.*);
            return ast.Types.OraType{ .exact = new_e };
        },
        else => return src,
    }
}

pub fn collectFunctionSymbols(table: *state.SymbolTable, parent: *state.Scope, f: *const ast.FunctionNode) !void {
    const fn_scope = try table.allocator.create(state.Scope);
    fn_scope.* = state.Scope.init(table.allocator, parent, f.name);
    try table.scopes.append(table.allocator, fn_scope);
    try table.function_scopes.put(f.name, fn_scope);
    // parameters
    for (f.parameters) |p| {
        var param_type = p.type_info;
        param_type.region = .Calldata;
        const sym = state.Symbol{ .name = p.name, .kind = .Param, .typ = param_type, .span = p.span, .mutable = p.is_mutable, .region = .Calldata };
        _ = try table.declare(fn_scope, sym);
    }

    // create a function symbol type from parameters and return type
    var param_types = std.ArrayListUnmanaged(ast.Types.OraType){};
    defer param_types.deinit(table.allocator);
    for (f.parameters) |p| {
        if (p.type_info.ora_type) |ot| {
            // copy the type to ensure refinement types are properly owned
            const copied_type = try copyOraTypeOwned(table.allocator, ot);
            try param_types.append(table.allocator, copied_type);
        } else {
            try param_types.append(table.allocator, .u256); // fallback for unknown param types
        }
    }
    const params_slice = try table.allocator.alloc(ast.Types.OraType, param_types.items.len);
    for (param_types.items, 0..) |t, i| params_slice[i] = t;

    // include return type in function type if present
    var ret_ptr: ?*const ast.Types.OraType = null;
    if (f.return_type_info) |rt| {
        if (rt.ora_type) |ot| {
            const heap_rt = try table.allocator.create(ast.Types.OraType);
            heap_rt.* = try copyOraTypeOwned(table.allocator, ot);
            ret_ptr = heap_rt;
        }
    }
    const fn_type = ast.type_info.FunctionType{ .params = params_slice, .return_type = ret_ptr };
    const new_ti = ast.Types.TypeInfo.fromOraType(.{ .function = fn_type });
    // record success type of error unions for quick checks
    // note: We need to deep-copy the success type if it contains pointers (refinement types)
    if (f.return_type_info) |rt| {
        if (rt.ora_type) |ot| switch (ot) {
            .error_union => |succ_ptr| {
                // deep-copy the success type to ensure refinement types are properly owned
                const copied_succ = try copyOraTypeOwned(table.allocator, succ_ptr.*);
                try table.function_success_types.put(f.name, copied_succ);
            },
            ._union => |members| {
                var i: usize = 0;
                while (i < members.len) : (i += 1) {
                    switch (members[i]) {
                        .error_union => |succ_ptr| {
                            // prefer the first error_union member
                            // deep-copy the success type to ensure refinement types are properly owned
                            const copied_succ = try copyOraTypeOwned(table.allocator, succ_ptr.*);
                            try table.function_success_types.put(f.name, copied_succ);
                            break;
                        },
                        else => {},
                    }
                }
            },
            else => {},
        };
    }
    // update existing declaration if present; otherwise declare
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

    // capture allowed error tags from the function return type if it is an error union with explicit union members
    if (f.return_type_info) |rt| {
        if (rt.ora_type) |ot| switch (ot) {
            ._union => |members| {
                // collect members that are error tags by name (enum error-type modeling TBD)
                var list = std.ArrayListUnmanaged([]const u8){};
                defer list.deinit(table.allocator);
                for (members) |m| switch (m) {
                    .struct_type, .enum_type, .contract_type => |name| {
                        // tentative: treat named types as allowable error tags by name
                        try list.append(table.allocator, name);
                    },
                    else => {},
                };
                const slice = try list.toOwnedSlice(table.allocator);
                try table.function_allowed_errors.put(f.name, slice);
            },
            else => {},
        };
    }
}

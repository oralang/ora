// ============================================================================
// MLIR Control Flow Graph Generation
// ============================================================================
//
// Uses Ora's MLIR C API to generate mode-specific Graphviz DOT CFGs.
//
// ============================================================================

const std = @import("std");
const mlir_c_api = @import("mlir_c_api");
const c = mlir_c_api.c;
const refinement_guards = @import("ora_lib").compiler.refinement_guards;

pub const Mode = enum {
    ora,
    sir,

    fn string(self: Mode) []const u8 {
        return switch (self) {
            .ora => "ora",
            .sir => "sir",
        };
    }
};

pub const Options = struct {
    mode: Mode,
    proven_guard_ids: ?*const std.StringHashMap(void) = null,
};

pub const SirOptimizationDiff = struct {
    before: []const u8,
    after: []const u8,

    pub fn deinit(self: SirOptimizationDiff, allocator: std.mem.Allocator) void {
        allocator.free(self.before);
        allocator.free(self.after);
    }
};

pub const FunctionCFG = struct {
    name: []const u8,
    dot: []const u8,

    pub fn deinit(self: FunctionCFG, allocator: std.mem.Allocator) void {
        allocator.free(self.name);
        allocator.free(self.dot);
    }
};

fn setModuleBoolAttr(ctx: c.MlirContext, module: c.MlirModule, name: []const u8) void {
    const attr = c.oraBoolAttrCreate(ctx, true);
    c.oraOperationSetAttributeByName(
        c.oraModuleGetOperation(module),
        c.oraStringRefCreate(name.ptr, name.len),
        attr,
    );
}

fn cloneModuleFromText(ctx: c.MlirContext, module: c.MlirModule) !c.MlirModule {
    const module_text = c.oraOperationPrintToString(c.oraModuleGetOperation(module));
    defer if (module_text.data != null) mlir_c_api.freeStringRef(module_text);

    if (module_text.data == null or module_text.length == 0)
        return error.CFGGenerationFailed;

    const clone = c.oraModuleCreateParse(ctx, c.oraStringRefCreate(module_text.data, module_text.length));
    if (c.oraModuleIsNull(clone))
        return error.CFGGenerationFailed;
    return clone;
}

/// Generate a Graphviz DOT CFG from an MLIR module.
/// `sir` mode emits a true SIR basic-block CFG. `ora` mode emits a structured
/// region/control view with FV guard overlay data when proven IDs are supplied.
pub fn generateCFG(
    ctx: c.MlirContext,
    module: c.MlirModule,
    allocator: std.mem.Allocator,
    options: Options,
) ![]const u8 {
    const mode_text = options.mode.string();
    var guard_refs: std.ArrayList(c.MlirStringRef) = .empty;
    defer guard_refs.deinit(allocator);

    if (options.proven_guard_ids) |ids| {
        var iter = ids.iterator();
        while (iter.next()) |entry| {
            if (refinement_guards.guardIdIsDuplicated(module, entry.key_ptr.*)) continue;
            try guard_refs.append(allocator, c.oraStringRefCreate(entry.key_ptr.*.ptr, entry.key_ptr.*.len));
        }
    }

    const dot_string_ref = c.oraGenerateCFGWithOptions(
        ctx,
        module,
        c.oraStringRefCreate(mode_text.ptr, mode_text.len),
        if (guard_refs.items.len > 0) guard_refs.items.ptr else null,
        guard_refs.items.len,
    );

    if (dot_string_ref.data == null) {
        std.log.err("CFG generation returned null (length: {d})", .{dot_string_ref.length});
        return error.CFGGenerationFailed;
    }

    if (dot_string_ref.length == 0) {
        std.log.err("CFG generation returned empty string", .{});
        return error.CFGGenerationFailed;
    }

    // copy the string to Zig-managed memory
    const dot_content = try allocator.dupe(u8, dot_string_ref.data[0..dot_string_ref.length]);

    mlir_c_api.freeStringRef(dot_string_ref);

    return dot_content;
}

pub fn generateFunctionCFGs(
    ctx: c.MlirContext,
    module: c.MlirModule,
    allocator: std.mem.Allocator,
    options: Options,
) ![]FunctionCFG {
    const count = c.oraCFGFunctionCount(module);
    var graphs = try std.ArrayList(FunctionCFG).initCapacity(allocator, count);
    errdefer {
        for (graphs.items) |graph| graph.deinit(allocator);
        graphs.deinit(allocator);
    }

    const mode_text = options.mode.string();
    for (0..count) |index| {
        const name_ref = c.oraCFGFunctionName(module, index);
        if (name_ref.data == null or name_ref.length == 0)
            return error.CFGGenerationFailed;
        defer mlir_c_api.freeStringRef(name_ref);

        const dot_ref = c.oraGenerateCFGForFunctionWithOptions(
            ctx,
            module,
            c.oraStringRefCreate(mode_text.ptr, mode_text.len),
            index,
        );
        if (dot_ref.data == null or dot_ref.length == 0)
            return error.CFGGenerationFailed;
        defer mlir_c_api.freeStringRef(dot_ref);

        const name = try allocator.dupe(u8, name_ref.data[0..name_ref.length]);
        errdefer allocator.free(name);
        const dot = try allocator.dupe(u8, dot_ref.data[0..dot_ref.length]);
        errdefer allocator.free(dot);

        graphs.appendAssumeCapacity(.{
            .name = name,
            .dot = dot,
        });
    }

    return graphs.toOwnedSlice(allocator);
}

/// Generate SIR CFG snapshots before and after the default MLIR framework
/// canonicalizer. This is a viewer/debugging helper: it lowers cloned modules
/// and never mutates the input module.
///
/// This intentionally returns two independently-rendered DOT graphs. It is not
/// a graph-isomorphism diff: canonicalization can merge, split, or renumber
/// blocks, so node IDs are only stable within each snapshot.
pub fn generateSirOptimizationDiff(
    ctx: c.MlirContext,
    module: c.MlirModule,
    allocator: std.mem.Allocator,
    debug_info: bool,
) !SirOptimizationDiff {
    const before_module = try cloneModuleFromText(ctx, module);
    defer c.oraModuleDestroy(before_module);
    const after_module = try cloneModuleFromText(ctx, module);
    defer c.oraModuleDestroy(after_module);

    setModuleBoolAttr(ctx, before_module, "ora.phase0.skip_sir_framework_canonicalizer");
    if (!c.oraConvertToSIR(ctx, before_module, debug_info))
        return error.CFGGenerationFailed;

    if (!c.oraConvertToSIR(ctx, after_module, debug_info))
        return error.CFGGenerationFailed;

    const before = try generateCFG(ctx, before_module, allocator, .{ .mode = .sir });
    errdefer allocator.free(before);
    const after = try generateCFG(ctx, after_module, allocator, .{ .mode = .sir });

    return .{
        .before = before,
        .after = after,
    };
}

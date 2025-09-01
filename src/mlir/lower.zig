// Main MLIR lowering orchestrator - coordinates modular components
// This file contains only the main lowerFunctionsToModule function and orchestration logic.
// All specific lowering functionality has been moved to modular files:
// - Type mapping: types.zig
// - Expression lowering: expressions.zig
// - Statement lowering: statements.zig
// - Declaration lowering: declarations.zig
// - Memory management: memory.zig
// - Symbol table: symbols.zig
// - Location tracking: locations.zig

const std = @import("std");
const lib = @import("ora_lib");
const c = @import("c.zig").c;

// Import modular components
const TypeMapper = @import("types.zig").TypeMapper;
const ExpressionLowerer = @import("expressions.zig").ExpressionLowerer;
const StatementLowerer = @import("statements.zig").StatementLowerer;
const DeclarationLowerer = @import("declarations.zig").DeclarationLowerer;
const MemoryManager = @import("memory.zig").MemoryManager;
const StorageMap = @import("memory.zig").StorageMap;
const SymbolTable = @import("symbols.zig").SymbolTable;
const ParamMap = @import("symbols.zig").ParamMap;
const LocalVarMap = @import("symbols.zig").LocalVarMap;
const LocationTracker = @import("locations.zig").LocationTracker;

/// Main entry point for lowering Ora AST nodes to MLIR module
/// This function orchestrates the modular lowering components
pub fn lowerFunctionsToModule(ctx: c.MlirContext, nodes: []lib.AstNode) c.MlirModule {
    const loc = c.mlirLocationUnknownGet(ctx);
    const module = c.mlirModuleCreateEmpty(loc);
    const body = c.mlirModuleGetBody(module);

    // Initialize modular components
    const type_mapper = TypeMapper.init(ctx);
    const locations = LocationTracker.init(ctx);
    const decl_lowerer = DeclarationLowerer.init(ctx, &type_mapper, locations);

    // Create global symbol table and storage map for the module
    var symbol_table = SymbolTable.init(std.heap.page_allocator);
    defer symbol_table.deinit();

    var global_storage_map = StorageMap.init(std.heap.page_allocator);
    defer global_storage_map.deinit();

    // Process all AST nodes using modular lowering components
    for (nodes) |node| {
        switch (node) {
            .Function => |func| {
                // Lower function declaration using the modular declaration lowerer
                var local_var_map = LocalVarMap.init(std.heap.page_allocator);
                defer local_var_map.deinit();

                const func_op = decl_lowerer.lowerFunction(&func, &global_storage_map, &local_var_map);
                c.mlirBlockAppendOwnedOperation(body, func_op);
            },
            .Contract => |contract| {
                // Lower contract declaration using the modular declaration lowerer
                const contract_op = decl_lowerer.lowerContract(&contract);
                c.mlirBlockAppendOwnedOperation(body, contract_op);
            },
            .VariableDecl => |var_decl| {
                // Lower global variable declarations
                switch (var_decl.region) {
                    .Storage => {
                        const global_op = decl_lowerer.createGlobalDeclaration(&var_decl);
                        c.mlirBlockAppendOwnedOperation(body, global_op);
                        _ = global_storage_map.getOrCreateAddress(var_decl.name) catch {};
                    },
                    .Memory => {
                        const memory_global_op = decl_lowerer.createMemoryGlobalDeclaration(&var_decl);
                        c.mlirBlockAppendOwnedOperation(body, memory_global_op);
                    },
                    .TStore => {
                        const tstore_global_op = decl_lowerer.createTStoreGlobalDeclaration(&var_decl);
                        c.mlirBlockAppendOwnedOperation(body, tstore_global_op);
                    },
                    .Stack => {
                        // Stack variables at module level are not allowed
                        std.debug.print("WARNING: Stack variable at module level: {s}\n", .{var_decl.name});
                    },
                }
            },
            .StructDecl => |struct_decl| {
                const struct_op = decl_lowerer.lowerStruct(&struct_decl);
                c.mlirBlockAppendOwnedOperation(body, struct_op);
            },
            .EnumDecl => |enum_decl| {
                const enum_op = decl_lowerer.lowerEnum(&enum_decl);
                c.mlirBlockAppendOwnedOperation(body, enum_op);
            },
            .Import => |import_decl| {
                const import_op = decl_lowerer.lowerImport(&import_decl);
                c.mlirBlockAppendOwnedOperation(body, import_op);
            },
            else => {
                // Handle other node types or report unsupported nodes
                std.debug.print("WARNING: Unsupported AST node type in MLIR lowering: {s}\n", .{@tagName(node)});
            },
        }
    }

    return module;
}

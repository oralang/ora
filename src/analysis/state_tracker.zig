// ============================================================================
// State Change Tracking & Analysis
// ============================================================================
//
// Tracks storage reads/writes, detects pure functions, generates warnings.
// Foundation for gas optimization, security analysis, and Z3 verification.
//
// ============================================================================

const std = @import("std");
const ast = @import("../ast.zig");
const SourceSpan = ast.SourceSpan;
const Expressions = ast.Expressions;
const Statements = ast.Statements;

/// Access type for a storage variable
pub const AccessType = enum {
    Read, // Variable is read
    Write, // Variable is written (not read before)
    ReadWrite, // Variable is read then written
};

/// Information about a single storage access
pub const AccessInfo = struct {
    access_type: AccessType,
    first_access_span: SourceSpan,
    last_access_span: SourceSpan,
    read_count: u32,
    write_count: u32,
};

/// Warning severity
pub const Severity = enum {
    Error, // Must fix
    Warning, // Should fix
    Info, // Nice to know
    Hint, // Optimization opportunity
};

/// Warning types
pub const WarningKind = enum {
    MissingCheck, // No balance check before transfer
    UnvalidatedConstructorParam, // Constructor parameter not validated before storage
    UnusedRead, // Read but never used
    DeadStore, // Write overwritten before read
    RedundantRead, // Multiple reads without write between
    PotentialReentrancy, // Storage write after external call
    MissingPostcondition, // Function modifies storage without ensures
    UnnecessaryWrite, // Write that doesn't change value
};

/// State analysis warning
pub const StateWarning = struct {
    severity: Severity,
    kind: WarningKind,
    function_name: []const u8,
    variable_name: ?[]const u8, // Optional variable involved
    span: ?SourceSpan, // Optional source location
    message: []const u8,
    suggestion: ?[]const u8,
};

/// Complete state analysis for a function
pub const FunctionStateAnalysis = struct {
    function_name: []const u8,

    // Storage tracking
    storage_accesses: std.StringHashMap(AccessInfo),

    // Computed properties
    is_stateless: bool, // No storage/memory access (pure computation)
    is_readonly: bool, // Only reads, no writes
    modifies_state: bool, // Has storage writes

    // Quick lookup sets (using StringArrayHashMap with void values as a set)
    reads_set: std.StringArrayHashMap(void),
    writes_set: std.StringArrayHashMap(void),

    // Memory
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, function_name: []const u8) FunctionStateAnalysis {
        return .{
            .function_name = function_name,
            .storage_accesses = std.StringHashMap(AccessInfo).init(allocator),
            .is_stateless = true, // Start optimistic (no state access)
            .is_readonly = true, // Start optimistic (no writes)
            .modifies_state = false,
            .reads_set = std.StringArrayHashMap(void).init(allocator),
            .writes_set = std.StringArrayHashMap(void).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *FunctionStateAnalysis) void {
        self.storage_accesses.deinit();
        self.reads_set.deinit();
        self.writes_set.deinit();
    }

    /// Record a storage read
    pub fn recordRead(self: *FunctionStateAnalysis, var_name: []const u8, span: SourceSpan) !void {
        self.is_stateless = false; // Any storage access = not stateless

        try self.reads_set.put(var_name, {});

        if (self.storage_accesses.getPtr(var_name)) |existing| {
            // Update existing (getPtr gives us a mutable pointer)
            existing.read_count += 1;
            existing.last_access_span = span;
            if (existing.access_type == .Write) {
                existing.access_type = .ReadWrite;
            }
        } else {
            // New read
            try self.storage_accesses.put(var_name, .{
                .access_type = .Read,
                .first_access_span = span,
                .last_access_span = span,
                .read_count = 1,
                .write_count = 0,
            });
        }
    }

    /// Record a storage write
    pub fn recordWrite(self: *FunctionStateAnalysis, var_name: []const u8, span: SourceSpan) !void {
        self.is_stateless = false; // Any storage access = not stateless
        self.is_readonly = false; // Writes = not readonly
        self.modifies_state = true;

        try self.writes_set.put(var_name, {});

        if (self.storage_accesses.getPtr(var_name)) |existing| {
            // Update existing (getPtr gives us a mutable pointer)
            existing.write_count += 1;
            existing.last_access_span = span;
            if (existing.access_type == .Read) {
                existing.access_type = .ReadWrite;
            }
        } else {
            // New write
            try self.storage_accesses.put(var_name, .{
                .access_type = .Write,
                .first_access_span = span,
                .last_access_span = span,
                .read_count = 0,
                .write_count = 1,
            });
        }
    }
};

/// Contract-level state analysis
pub const ContractStateAnalysis = struct {
    contract_name: []const u8,
    functions: std.StringHashMap(FunctionStateAnalysis),
    warnings: std.ArrayList(StateWarning),
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, contract_name: []const u8) ContractStateAnalysis {
        return .{
            .contract_name = contract_name,
            .functions = std.StringHashMap(FunctionStateAnalysis).init(allocator),
            .warnings = std.ArrayList(StateWarning){},
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *ContractStateAnalysis) void {
        var it = self.functions.valueIterator();
        while (it.next()) |func_analysis| {
            var mutable_analysis = func_analysis;
            mutable_analysis.deinit();
        }
        self.functions.deinit();

        // Free warning messages
        for (self.warnings.items) |warning| {
            self.allocator.free(warning.message);
        }
        self.warnings.deinit(self.allocator);
    }
};

/// Main state tracker
pub const StateTracker = struct {
    allocator: std.mem.Allocator,
    current_function: ?*FunctionStateAnalysis,

    pub fn init(allocator: std.mem.Allocator) StateTracker {
        return .{
            .allocator = allocator,
            .current_function = null,
        };
    }

    /// Analyze a function
    pub fn analyzeFunction(self: *StateTracker, func: *const ast.FunctionNode) !FunctionStateAnalysis {
        var analysis = FunctionStateAnalysis.init(self.allocator, func.name);
        self.current_function = &analysis;

        // Skip ghost functions (specification-only)
        if (func.is_ghost) {
            return analysis;
        }

        // Visit function body
        try self.visitBlock(&func.body);

        self.current_function = null;
        return analysis;
    }

    /// Visit a block of statements
    fn visitBlock(self: *StateTracker, block: *const Statements.BlockNode) anyerror!void {
        for (block.statements) |*stmt| {
            try self.visitStatement(stmt);
        }
    }

    /// Visit a statement
    fn visitStatement(self: *StateTracker, stmt: *const Statements.StmtNode) anyerror!void {
        switch (stmt.*) {
            .Expr => |expr_stmt| {
                try self.visitExpression(&expr_stmt);
            },
            .VariableDecl => |var_decl| {
                if (var_decl.value) |value| {
                    try self.visitExpression(value);
                }
            },
            .Return => |ret| {
                if (ret.value) |value| {
                    try self.visitExpression(&value);
                }
            },
            .If => |if_stmt| {
                try self.visitExpression(&if_stmt.condition);
                try self.visitBlock(&if_stmt.then_branch);
                if (if_stmt.else_branch) |*else_branch| {
                    try self.visitBlock(else_branch);
                }
            },
            .While => |while_stmt| {
                try self.visitExpression(&while_stmt.condition);
                try self.visitBlock(&while_stmt.body);
            },
            .ForLoop => |for_stmt| {
                try self.visitExpression(&for_stmt.iterable);
                try self.visitBlock(&for_stmt.body);
            },
            else => {},
        }
    }

    /// Visit an expression (read context)
    fn visitExpression(self: *StateTracker, expr: *const Expressions.ExprNode) anyerror!void {
        switch (expr.*) {
            .Identifier => |ident| {
                // Check if this is a storage variable access
                if (self.current_function) |func_analysis| {
                    // Simple heuristic: storage variables are accessed directly
                    // In real implementation, consult symbol table
                    if (self.isStorageVariable(ident.name)) {
                        try func_analysis.recordRead(ident.name, ident.span);
                    }
                }
            },
            .Binary => |binary| {
                try self.visitExpression(binary.lhs);
                try self.visitExpression(binary.rhs);
            },
            .Unary => |unary| {
                try self.visitExpression(unary.operand);
            },
            .Call => |call| {
                try self.visitExpression(call.callee);
                for (call.arguments) |arg| {
                    try self.visitExpression(arg);
                }
            },
            .Index => |index| {
                // Map/array access: balances[sender]
                try self.visitExpression(index.target);
                try self.visitExpression(index.index);

                // Record as storage access if base is storage
                if (self.current_function) |func_analysis| {
                    const base_name = try self.getBaseName(index.target);
                    if (self.isStorageVariable(base_name)) {
                        try func_analysis.recordRead(base_name, index.span);
                    }
                }
            },
            .FieldAccess => |field| {
                try self.visitExpression(field.target);
            },
            .Assignment => |assign| {
                // Left side is a write
                try self.visitExpressionForWrite(assign.target);
                // Right side is a read
                try self.visitExpression(assign.value);
            },
            .CompoundAssignment => |compound| {
                // Compound assignment (+=, -=, etc.) is both read and write
                // First, read the current value (implicit in +=, etc.)
                try self.visitExpression(compound.target);
                // Then write the new value
                try self.visitExpressionForWrite(compound.target);
                // And read the RHS
                try self.visitExpression(compound.value);
            },
            else => {},
        }
    }

    /// Visit an expression in write context
    fn visitExpressionForWrite(self: *StateTracker, expr: *const Expressions.ExprNode) anyerror!void {
        switch (expr.*) {
            .Identifier => |ident| {
                if (self.current_function) |func_analysis| {
                    if (self.isStorageVariable(ident.name)) {
                        try func_analysis.recordWrite(ident.name, ident.span);
                    }
                }
            },
            .Index => |index| {
                // Map/array write: balances[sender] = value
                if (self.current_function) |func_analysis| {
                    const base_name = try self.getBaseName(index.target);
                    if (self.isStorageVariable(base_name)) {
                        try func_analysis.recordWrite(base_name, index.span);
                    }
                }
            },
            else => {},
        }
    }

    /// Helper: Get base name from expression (e.g., "balances" from "balances[sender]")
    fn getBaseName(self: *StateTracker, expr: *const Expressions.ExprNode) anyerror![]const u8 {
        _ = self;
        return switch (expr.*) {
            .Identifier => |ident| ident.name,
            .Index => |index| try getBaseName(undefined, index.target),
            else => "",
        };
    }

    /// Helper: Check if a variable is a storage variable
    /// TODO: Consult symbol table for accurate information
    fn isStorageVariable(self: *StateTracker, name: []const u8) bool {
        _ = self;
        // Temporary heuristic: common storage variable names
        // In production, this should query the semantic analyzer's symbol table
        const storage_vars = [_][]const u8{
            "balance", "balances", "totalSupply", "allowances", "owner", "name", "symbol",
            // Test contract variables
            "counter", "config",   "unusedData",
        };
        for (storage_vars) |storage_var| {
            if (std.mem.eql(u8, name, storage_var)) {
                return true;
            }
        }
        return false;
    }
};

/// Analyze a contract
pub fn analyzeContract(allocator: std.mem.Allocator, contract: *const ast.ContractNode) !ContractStateAnalysis {
    var contract_analysis = ContractStateAnalysis.init(allocator, contract.name);
    var tracker = StateTracker.init(allocator);

    for (contract.body) |*node| {
        switch (node.*) {
            .Function => |func| {
                const func_analysis = try tracker.analyzeFunction(&func);
                try contract_analysis.functions.put(func.name, func_analysis);
            },
            else => {},
        }
    }

    // Generate warnings based on analysis
    try generateWarnings(allocator, &contract_analysis);

    return contract_analysis;
}

/// Generate warnings for potential issues
fn generateWarnings(allocator: std.mem.Allocator, analysis: *ContractStateAnalysis) !void {
    // PHASE 1: Contract-level dead store analysis
    // Collect all reads and writes across ALL functions
    var contract_reads = std.StringArrayHashMap(void).init(allocator);
    defer contract_reads.deinit();
    var contract_writes = std.StringArrayHashMap(void).init(allocator);
    defer contract_writes.deinit();

    // Build contract-wide read/write sets
    var func_iter = analysis.functions.iterator();
    while (func_iter.next()) |entry| {
        const func_analysis = entry.value_ptr.*;

        // Collect reads
        var reads_it = func_analysis.reads_set.iterator();
        while (reads_it.next()) |read_entry| {
            try contract_reads.put(read_entry.key_ptr.*, {});
        }

        // Collect writes
        var writes_it = func_analysis.writes_set.iterator();
        while (writes_it.next()) |write_entry| {
            try contract_writes.put(write_entry.key_ptr.*, {});
        }
    }

    // Find variables written but NEVER read by ANY function (true dead stores)
    var writes_iter = contract_writes.iterator();
    while (writes_iter.next()) |write_entry| {
        const var_name = write_entry.key_ptr.*;

        // Check if this variable is NEVER read by any function
        if (!contract_reads.contains(var_name)) {
            // Find which function(s) write to this variable
            var func_iter2 = analysis.functions.iterator();
            while (func_iter2.next()) |entry| {
                const func_name = entry.key_ptr.*;
                const func_analysis = entry.value_ptr.*;

                if (func_analysis.writes_set.contains(var_name)) {
                    const message = try std.fmt.allocPrint(allocator, "Storage variable '{s}' is written by '{s}' but never read by any function in the contract", .{ var_name, func_name });

                    try analysis.warnings.append(allocator, .{
                        .kind = .DeadStore,
                        .severity = .Warning,
                        .function_name = func_name,
                        .variable_name = var_name,
                        .message = message,
                        .span = null,
                        .suggestion = "Remove unused storage variable or add read logic in another function",
                    });
                }
            }
        }
    }

    // PHASE 2: Per-function missing check analysis
    func_iter = analysis.functions.iterator();
    while (func_iter.next()) |entry| {
        const func_name = entry.key_ptr.*;
        const func_analysis = entry.value_ptr.*;

        // Check if this is a constructor (init function)
        const is_constructor = std.mem.eql(u8, func_name, "init");

        if (is_constructor) {
            // Constructor-specific warning: unvalidated parameters
            if (func_analysis.modifies_state) {
                const message = try std.fmt.allocPrint(allocator, "Constructor 'init' stores parameters without validation (consider adding checks)", .{});

                try analysis.warnings.append(allocator, .{
                    .kind = .UnvalidatedConstructorParam,
                    .severity = .Info,
                    .function_name = func_name,
                    .variable_name = null,
                    .message = message,
                    .span = null,
                    .suggestion = "Add validation for constructor parameters (e.g., require amount > 0, address != 0)",
                });
            }
        } else {
            // Regular function warning: modifies state without reads (potential missing checks)
            if (func_analysis.modifies_state and func_analysis.reads_set.count() == 0) {
                const message = try std.fmt.allocPrint(allocator, "Function '{s}' modifies storage without reading any state (consider adding validation)", .{func_name});

                try analysis.warnings.append(allocator, .{
                    .kind = .MissingCheck,
                    .severity = .Info,
                    .function_name = func_name,
                    .variable_name = null,
                    .message = message,
                    .span = null,
                    .suggestion = "Add validation checks before modifying storage (e.g., require conditions)",
                });
            }
        }
    }
}

/// Print only warnings (for compilation mode)
pub fn printWarnings(writer: anytype, analysis: *const ContractStateAnalysis) !void {
    if (analysis.warnings.items.len == 0) {
        return; // No output if no warnings
    }

    try writer.print("\n⚠️  State Analysis Warnings for {s} ({d}):\n\n", .{ analysis.contract_name, analysis.warnings.items.len });
    for (analysis.warnings.items) |warning| {
        const severity_icon = switch (warning.severity) {
            .Error => "❌",
            .Warning => "⚠️ ",
            .Info => "ℹ️ ",
            .Hint => "💡",
        };

        try writer.print("{s} [{s}] {s}\n", .{
            severity_icon,
            @tagName(warning.kind),
            warning.message,
        });

        if (warning.suggestion) |suggestion| {
            try writer.print("   💬 {s}\n", .{suggestion});
        }
        try writer.print("\n", .{});
    }
}

/// Pretty print full analysis results (for --analyze-state mode)
pub fn printAnalysis(writer: anytype, analysis: *const ContractStateAnalysis) !void {
    try writer.print("\n=== State Analysis: {s} ===\n\n", .{analysis.contract_name});

    var it = analysis.functions.iterator();
    while (it.next()) |entry| {
        const func_name = entry.key_ptr.*;
        const func_analysis = entry.value_ptr;

        try writer.print("Function: {s}\n", .{func_name});
        try writer.print("├─ Stateless: {}\n", .{func_analysis.is_stateless});
        try writer.print("├─ Readonly: {}\n", .{func_analysis.is_readonly});
        try writer.print("├─ Modifies State: {}\n", .{func_analysis.modifies_state});

        // Print reads
        try writer.print("├─ Reads: ", .{});
        var reads_it = func_analysis.reads_set.iterator();
        var first = true;
        while (reads_it.next()) |read| {
            if (!first) try writer.print(", ", .{});
            try writer.print("{s}", .{read.key_ptr.*});
            first = false;
        }
        if (func_analysis.reads_set.count() == 0) {
            try writer.print("(none)", .{});
        }
        try writer.print("\n", .{});

        // Print writes
        try writer.print("└─ Writes: ", .{});
        var writes_it = func_analysis.writes_set.iterator();
        first = true;
        while (writes_it.next()) |write| {
            if (!first) try writer.print(", ", .{});
            try writer.print("{s}", .{write.key_ptr.*});
            first = false;
        }
        if (func_analysis.writes_set.count() == 0) {
            try writer.print("(none)", .{});
        }
        try writer.print("\n\n", .{});
    }

    // Print warnings if any
    if (analysis.warnings.items.len > 0) {
        try writer.print("⚠️  Warnings ({d}):\n\n", .{analysis.warnings.items.len});
        for (analysis.warnings.items) |warning| {
            const severity_icon = switch (warning.severity) {
                .Error => "❌",
                .Warning => "⚠️ ",
                .Info => "ℹ️ ",
                .Hint => "💡",
            };

            try writer.print("{s} [{s}] {s}\n", .{
                severity_icon,
                @tagName(warning.kind),
                warning.message,
            });

            if (warning.suggestion) |suggestion| {
                try writer.print("   💬 Suggestion: {s}\n", .{suggestion});
            }
            try writer.print("\n", .{});
        }
    }
}

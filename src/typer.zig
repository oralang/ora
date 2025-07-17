const std = @import("std");
const ast = @import("ast.zig");

/// Type system for Ora
pub const OraType = union(enum) {
    // Primitive types
    Bool: void,
    Address: void,
    U8: void,
    U16: void,
    U32: void,
    U64: void,
    U128: void,
    U256: void,
    I8: void,
    I16: void,
    I32: void,
    I64: void,
    I128: void,
    I256: void,
    String: void,
    Bytes: void,

    // Complex types
    Slice: *OraType,
    Mapping: struct {
        key: *OraType,
        value: *OraType,
    },
    DoubleMap: struct {
        key1: *OraType,
        key2: *OraType,
        value: *OraType,
    },

    // Function type
    Function: struct {
        params: []OraType,
        return_type: ?*OraType,
    },

    // Special types
    Void: void,
    Unknown: void,
    Error: void,
    Tuple: struct {
        types: []OraType,
    },
};

/// Type checking errors
pub const TyperError = error{
    UndeclaredVariable,
    TypeMismatch,
    InvalidOperation,
    UndeclaredFunction,
    ArgumentCountMismatch,
    InvalidMemoryRegion,
    OutOfMemory,
};

/// Symbol table entry
pub const Symbol = struct {
    name: []const u8,
    typ: OraType,
    region: ast.MemoryRegion,
    mutable: bool,
    span: ast.SourceSpan,
};

/// Symbol table for scope management (using ArrayList to avoid HashMap overflow in Zig 0.14.1)
pub const SymbolTable = struct {
    symbols: std.ArrayList(Symbol),
    parent: ?*SymbolTable,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, parent: ?*SymbolTable) SymbolTable {
        return SymbolTable{
            .symbols = std.ArrayList(Symbol).init(allocator),
            .parent = parent,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *SymbolTable) void {
        self.symbols.deinit();
    }

    pub fn declare(self: *SymbolTable, symbol: Symbol) !void {
        try self.symbols.append(symbol);
    }

    pub fn lookup(self: *SymbolTable, name: []const u8) ?Symbol {
        // Linear search - O(n) but fine for small symbol tables
        for (self.symbols.items) |symbol| {
            if (std.mem.eql(u8, symbol.name, name)) {
                return symbol;
            }
        }

        if (self.parent) |parent| {
            return parent.lookup(name);
        }

        return null;
    }
};

/// Type checker for ZigOra
pub const Typer = struct {
    allocator: std.mem.Allocator,
    global_scope: SymbolTable,
    current_scope: *SymbolTable,
    current_function: ?[]const u8,
    type_arena: std.heap.ArenaAllocator,
    /// Track allocated function parameter arrays for cleanup
    function_params: std.ArrayList([]OraType),

    pub fn init(allocator: std.mem.Allocator) Typer {
        return Typer{
            .allocator = allocator,
            .global_scope = SymbolTable.init(allocator, null),
            .current_scope = undefined, // Will be fixed in fixSelfReferences
            .current_function = null,
            .type_arena = std.heap.ArenaAllocator.init(allocator),
            .function_params = std.ArrayList([]OraType).init(allocator),
        };
    }

    /// Fix self-references after struct initialization
    pub fn fixSelfReferences(self: *Typer) void {
        self.current_scope = &self.global_scope;
        // Initialize standard library symbols after self-references are fixed
        self.initStandardLibrary() catch {};
    }

    /// Initialize standard library symbols in global scope
    pub fn initStandardLibrary(self: *Typer) TyperError!void {
        // Add 'std' as a module identifier
        const std_symbol = Symbol{
            .name = "std",
            .typ = OraType.Unknown, // Module type - will be refined later
            .region = .Stack,
            .mutable = false,
            .span = ast.SourceSpan{ .line = 0, .column = 0, .length = 3 },
        };
        try self.current_scope.declare(std_symbol);

        // Add built-in functions
        const require_params = try self.allocator.alloc(OraType, 1);
        require_params[0] = OraType.Bool;
        try self.function_params.append(require_params); // Track for cleanup

        const require_symbol = Symbol{
            .name = "require",
            .typ = OraType{ .Function = .{
                .params = require_params,
                .return_type = null,
            } },
            .region = .Stack,
            .mutable = false,
            .span = ast.SourceSpan{ .line = 0, .column = 0, .length = 7 },
        };
        try self.current_scope.declare(require_symbol);
    }

    pub fn deinit(self: *Typer) void {
        // Free all types at once with arena
        self.type_arena.deinit();
        self.global_scope.deinit();
        // Clean up function parameter arrays
        for (self.function_params.items) |params| {
            self.allocator.free(params);
        }
        self.function_params.deinit();
    }

    /// Type check a list of top-level AST nodes
    pub fn typeCheck(self: *Typer, nodes: []ast.AstNode) TyperError!void {
        // First pass: collect all declarations
        for (nodes) |*node| {
            try self.collectDeclarations(node);
        }

        // Second pass: type check implementations
        for (nodes) |*node| {
            try self.typeCheckNode(node);
        }
    }

    /// Collect all declarations for forward references
    fn collectDeclarations(self: *Typer, node: *ast.AstNode) TyperError!void {
        switch (node.*) {
            .Contract => |*contract| {
                // Create contract scope and collect members
                for (contract.body) |*member| {
                    try self.collectDeclarations(member);
                }
            },
            .Function => |*function| {
                // Add function to symbol table (simplified to avoid createFunctionType issues)
                const symbol = Symbol{
                    .name = function.name,
                    .typ = OraType.Unknown, // Simplified for now
                    .region = .Stack, // Functions don't have memory regions
                    .mutable = false,
                    .span = function.span,
                };
                try self.current_scope.declare(symbol);
            },
            .VariableDecl => |*var_decl| {
                // Add variable to symbol table
                const var_type = try self.convertAstTypeToOraType(&var_decl.typ);
                const symbol = Symbol{
                    .name = var_decl.name,
                    .typ = var_type,
                    .region = var_decl.region,
                    .mutable = var_decl.mutable,
                    .span = var_decl.span,
                };
                try self.current_scope.declare(symbol);
            },
            else => {
                // Skip other node types in declaration phase
            },
        }
    }

    /// Type check a single AST node
    fn typeCheckNode(self: *Typer, node: *ast.AstNode) TyperError!void {
        switch (node.*) {
            .Contract => |*contract| {
                for (contract.body) |*member| {
                    try self.typeCheckNode(member);
                }
            },
            .Function => |*function| {
                try self.typeCheckFunction(function);
            },
            .VariableDecl => |*var_decl| {
                try self.typeCheckVariableDecl(var_decl);
            },
            else => {
                // TODO: Add type checking for: StructDecl, EnumDecl, LogDecl, Import, ErrorDecl (top-level), Block, Expression, Statement, TryBlock
            },
        }
    }

    /// Type check a function
    fn typeCheckFunction(self: *Typer, function: *ast.FunctionNode) TyperError!void {
        // Create function scope
        var func_scope = SymbolTable.init(self.allocator, self.current_scope);
        defer func_scope.deinit();

        const prev_scope = self.current_scope;
        const prev_function = self.current_function;
        self.current_scope = &func_scope;
        self.current_function = function.name;
        defer {
            self.current_scope = prev_scope;
            self.current_function = prev_function;
        }

        // Add parameters to function scope
        for (function.parameters) |*param| {
            const param_type = try self.convertAstTypeToOraType(&param.typ);
            const symbol = Symbol{
                .name = param.name,
                .typ = param_type,
                .region = .Stack,
                .mutable = false, // Parameters are immutable by default
                .span = param.span,
            };
            try self.current_scope.declare(symbol);
        }

        // Type check function body
        try self.typeCheckBlock(&function.body);

        // Verify return type consistency
        if (function.return_type) |*return_type| {
            const expected_return = try self.convertAstTypeToOraType(return_type);
            // TODO: Verify all return statements match this type
            _ = expected_return;
        }
    }

    /// Type check a variable declaration
    fn typeCheckVariableDecl(self: *Typer, var_decl: *ast.VariableDeclNode) TyperError!void {
        // Handle tuple unpacking
        if (var_decl.tuple_names) |tuple_names| {
            // Tuple unpacking: let (a, b) = expr
            if (var_decl.value) |*init_expr| {
                const init_type = try self.typeCheckExpression(init_expr);

                // Ensure initializer is a tuple type
                if (init_type != .Tuple) {
                    return TyperError.TypeMismatch;
                }

                const tuple_type = init_type.Tuple;

                // Ensure tuple arity matches
                if (tuple_names.len != tuple_type.types.len) {
                    return TyperError.TypeMismatch;
                }

                // Declare each tuple variable
                for (tuple_names, tuple_type.types) |name, typ| {
                    const symbol = Symbol{
                        .name = name,
                        .typ = typ,
                        .region = var_decl.region,
                        .mutable = var_decl.mutable,
                        .span = var_decl.span,
                    };

                    try self.current_scope.declare(symbol);
                }
            } else {
                return TyperError.TypeMismatch; // Tuple unpacking requires initializer
            }
        } else {
            // Regular variable declaration
            const var_type = try self.convertAstTypeToOraType(&var_decl.typ);

            // Type check initializer if present
            if (var_decl.value) |*init_expr| {
                const init_type = try self.typeCheckExpression(init_expr);
                if (!self.typesCompatible(var_type, init_type)) {
                    return TyperError.TypeMismatch;
                }
            }

            // Validate memory region constraints
            try self.validateMemoryRegion(var_decl.region, var_type);

            // Add the variable to the symbol table
            const symbol = Symbol{
                .name = var_decl.name,
                .typ = var_type,
                .region = var_decl.region,
                .mutable = var_decl.mutable,
                .span = var_decl.span,
            };

            try self.current_scope.declare(symbol);
        }
    }

    /// Type check a block of statements
    fn typeCheckBlock(self: *Typer, block: *ast.BlockNode) TyperError!void {
        for (block.statements) |*stmt| {
            try self.typeCheckStatement(stmt);
        }
    }

    /// Type check a statement
    fn typeCheckStatement(self: *Typer, stmt: *ast.StmtNode) TyperError!void {
        switch (stmt.*) {
            .Expr => |*expr| {
                _ = try self.typeCheckExpression(expr);
            },
            .VariableDecl => |*var_decl| {
                try self.typeCheckVariableDecl(var_decl);
            },
            .Return => |*ret| {
                if (ret.value) |*value| {
                    _ = try self.typeCheckExpression(value);
                    // TODO: Verify return type matches function signature
                }
            },
            .Log => |*log| {
                // Type check log arguments
                for (log.args) |*arg| {
                    _ = try self.typeCheckExpression(arg);
                }
            },
            .Lock => |*lock| {
                // Type check lock path
                _ = try self.typeCheckExpression(&lock.path);
            },
            .ErrorDecl => |*error_decl| {
                // Error declarations don't need type checking
                _ = error_decl;
            },
            .TryBlock => |*try_block| {
                try self.typeCheckBlock(&try_block.try_block);
                if (try_block.catch_block) |*catch_block| {
                    try self.typeCheckBlock(&catch_block.block);
                }
            },
            .If => |*if_stmt| {
                // Type check condition
                const condition_type = try self.typeCheckExpression(&if_stmt.condition);
                if (!std.meta.eql(condition_type, OraType.Bool)) {
                    return TyperError.TypeMismatch;
                }

                // Type check then branch
                try self.typeCheckBlock(&if_stmt.then_branch);

                // Type check else branch if present
                if (if_stmt.else_branch) |*else_branch| {
                    try self.typeCheckBlock(else_branch);
                }
            },
            .While => |*while_stmt| {
                // Type check condition
                const condition_type = try self.typeCheckExpression(&while_stmt.condition);
                if (!std.meta.eql(condition_type, OraType.Bool)) {
                    return TyperError.TypeMismatch;
                }

                // Type check body
                try self.typeCheckBlock(&while_stmt.body);
            },
            .Break => |*break_stmt| {
                // Break statements are always valid (context validation happens elsewhere)
                _ = break_stmt;
            },
            .Continue => |*continue_stmt| {
                // Continue statements are always valid (context validation happens elsewhere)
                _ = continue_stmt;
            },
            .Invariant => |*invariant| {
                // Invariant condition must be boolean
                const condition_type = try self.typeCheckExpression(&invariant.condition);
                if (!std.meta.eql(condition_type, OraType.Bool)) {
                    return TyperError.TypeMismatch;
                }
            },
            .Requires => |*requires| {
                // Requires condition must be boolean
                const condition_type = try self.typeCheckExpression(&requires.condition);
                if (!std.meta.eql(condition_type, OraType.Bool)) {
                    return TyperError.TypeMismatch;
                }
            },
            .Ensures => |*ensures| {
                // Ensures condition must be boolean
                const condition_type = try self.typeCheckExpression(&ensures.condition);
                if (!std.meta.eql(condition_type, OraType.Bool)) {
                    return TyperError.TypeMismatch;
                }
            },
        }
    }

    /// Type check an expression and return its type
    fn typeCheckExpression(self: *Typer, expr: *ast.ExprNode) TyperError!OraType {
        switch (expr.*) {
            .Identifier => |*ident| {
                if (self.current_scope.lookup(ident.name)) |symbol| {
                    return symbol.typ;
                } else {
                    return TyperError.UndeclaredVariable;
                }
            },
            .Literal => |*literal| {
                return try self.getLiteralType(literal);
            },
            .Binary => |*binary| {
                const lhs_type = try self.typeCheckExpression(binary.lhs);
                const rhs_type = try self.typeCheckExpression(binary.rhs);

                return try self.typeCheckBinaryOp(binary.operator, lhs_type, rhs_type);
            },
            .Assignment => |*assign| {
                const target_type = try self.typeCheckExpression(assign.target);
                const value_type = try self.typeCheckExpression(assign.value);

                if (!self.typesCompatible(target_type, value_type)) {
                    return TyperError.TypeMismatch;
                }

                return target_type;
            },
            .Call => |*call| {
                return try self.typeCheckFunctionCall(call);
            },
            .Try => |*try_expr| {
                // Try expressions return the success type of the error union
                return try self.typeCheckExpression(try_expr.expr);
            },
            .ErrorReturn => |*error_return| {
                // Error returns should be validated elsewhere
                _ = error_return;
                return OraType.Error;
            },
            .ErrorCast => |*error_cast| {
                // Error casts convert to error union type
                _ = try self.typeCheckExpression(error_cast.operand);
                return try self.convertAstTypeToOraType(&error_cast.target_type);
            },
            .Shift => |*shift| {
                // Type check shift expression components
                _ = try self.typeCheckExpression(shift.mapping);
                _ = try self.typeCheckExpression(shift.source);
                _ = try self.typeCheckExpression(shift.dest);
                _ = try self.typeCheckExpression(shift.amount);
                // Shift operations return void
                return OraType.Void;
            },
            .Tuple => |*tuple| {
                // Type check tuple expressions
                var tuple_types = std.ArrayList(OraType).init(self.allocator);
                defer tuple_types.deinit();

                for (tuple.elements) |*element| {
                    const element_type = try self.typeCheckExpression(element);
                    try tuple_types.append(element_type);
                }

                return OraType{ .Tuple = .{
                    .types = try tuple_types.toOwnedSlice(),
                } };
            },
            .Unary => |*unary| {
                const operand_type = try self.typeCheckExpression(unary.operand);
                return try self.typeCheckUnaryOp(unary.operator, operand_type);
            },
            .CompoundAssignment => |*compound| {
                const target_type = try self.typeCheckExpression(compound.target);
                const value_type = try self.typeCheckExpression(compound.value);

                // Validate compound operation (e.g., += requires numeric types)
                const result_type = try self.typeCheckCompoundAssignmentOp(compound.operator, target_type, value_type);
                if (!self.typesCompatible(target_type, result_type)) {
                    return TyperError.TypeMismatch;
                }
                return target_type;
            },
            .Index => |*index| {
                const target_type = try self.typeCheckExpression(index.target);
                const index_type = try self.typeCheckExpression(index.index);

                return try self.typeCheckIndexAccess(target_type, index_type);
            },
            .FieldAccess => |*field| {
                const target_type = try self.typeCheckExpression(field.target);
                return try self.typeCheckFieldAccess(target_type, field.field);
            },
            .Cast => |*cast| {
                const operand_type = try self.typeCheckExpression(cast.operand);
                const target_type = try self.convertAstTypeToOraType(&cast.target_type);

                // Validate cast safety
                if (!self.isCastValid(operand_type, target_type)) {
                    return TyperError.TypeMismatch;
                }
                return target_type;
            },
            .Old => |*old| {
                // old() expressions have the same type as their inner expression
                return try self.typeCheckExpression(old.expr);
            },
            .Comptime => |*comptime_block| {
                // Comptime blocks return void (they're evaluated at compile time)
                try self.typeCheckBlock(&comptime_block.block);
                return OraType.Void;
            },
        }
    }

    /// Get the type of a literal
    fn getLiteralType(self: *Typer, literal: *ast.LiteralNode) TyperError!OraType {
        _ = self;
        return switch (literal.*) {
            .Integer => |*int_lit| {
                // Infer the smallest suitable integer type
                const value_str = int_lit.value;

                // Check if it's a negative number
                const is_negative = value_str.len > 0 and value_str[0] == '-';
                const abs_str = if (is_negative) value_str[1..] else value_str;

                if (is_negative) {
                    // Try parsing as different signed integer types
                    if (std.fmt.parseInt(i8, abs_str, 10)) |_| {
                        return OraType.I8;
                    } else |_| {}

                    if (std.fmt.parseInt(i16, abs_str, 10)) |_| {
                        return OraType.I16;
                    } else |_| {}

                    if (std.fmt.parseInt(i32, abs_str, 10)) |_| {
                        return OraType.I32;
                    } else |_| {}

                    if (std.fmt.parseInt(i64, abs_str, 10)) |_| {
                        return OraType.I64;
                    } else |_| {}

                    if (std.fmt.parseInt(i128, abs_str, 10)) |_| {
                        return OraType.I128;
                    } else |_| {}

                    // Default to i256 for very large negative numbers
                    return OraType.I256;
                } else {
                    // Try parsing as different unsigned integer types
                    if (std.fmt.parseInt(u8, value_str, 10)) |_| {
                        return OraType.U8;
                    } else |_| {}

                    if (std.fmt.parseInt(u16, value_str, 10)) |_| {
                        return OraType.U16;
                    } else |_| {}

                    if (std.fmt.parseInt(u32, value_str, 10)) |_| {
                        return OraType.U32;
                    } else |_| {}

                    if (std.fmt.parseInt(u64, value_str, 10)) |_| {
                        return OraType.U64;
                    } else |_| {}

                    if (std.fmt.parseInt(u128, value_str, 10)) |_| {
                        return OraType.U128;
                    } else |_| {}

                    // Default to u256 for very large numbers
                    return OraType.U256;
                }
            },
            .String => OraType.String,
            .Bool => OraType.Bool,
            .Address => OraType.Address,
            .Hex => OraType.U256, // Hex literals default to U256
        };
    }

    /// Type check a binary operation
    fn typeCheckBinaryOp(self: *Typer, op: ast.BinaryOp, lhs: OraType, rhs: OraType) TyperError!OraType {
        switch (op) {
            // Arithmetic operators
            .Plus, .Minus, .Star, .Slash, .Percent => {
                if (self.isNumericType(lhs) and self.isNumericType(rhs)) {
                    return self.commonNumericType(lhs, rhs);
                }
                return TyperError.TypeMismatch;
            },
            // Comparison operators
            .EqualEqual, .BangEqual, .Less, .LessEqual, .Greater, .GreaterEqual => {
                if (self.typesCompatible(lhs, rhs)) {
                    return OraType.Bool;
                }
                return TyperError.TypeMismatch;
            },
            // Logical operators
            .And, .Or => {
                if (std.meta.eql(lhs, OraType.Bool) and std.meta.eql(rhs, OraType.Bool)) {
                    return OraType.Bool;
                }
                return TyperError.TypeMismatch;
            },
            else => {
                return TyperError.InvalidOperation;
            },
        }
    }

    /// Type check a compound assignment operation
    fn typeCheckCompoundAssignmentOp(self: *Typer, op: ast.CompoundAssignmentOp, lhs: OraType, rhs: OraType) TyperError!OraType {
        switch (op) {
            .PlusEqual, .MinusEqual, .StarEqual, .SlashEqual, .PercentEqual => {
                if (self.isNumericType(lhs) and self.isNumericType(rhs)) {
                    return self.commonNumericType(lhs, rhs);
                }
                return TyperError.TypeMismatch;
            },
        }
    }

    /// Type check a function call
    fn typeCheckFunctionCall(self: *Typer, call: *ast.CallExpr) TyperError!OraType {
        // Extract function name from callee (assuming it's an identifier)
        const function_name = switch (call.callee.*) {
            .Identifier => |*ident| ident.name,
            else => return TyperError.InvalidOperation, // Complex callees not supported yet
        };

        // Check if function exists in symbol table
        if (self.current_scope.lookup(function_name)) |symbol| {
            switch (symbol.typ) {
                .Function => |func_type| {
                    // Validate argument count
                    if (call.arguments.len != func_type.params.len) {
                        return TyperError.ArgumentCountMismatch;
                    }

                    // Type check each argument
                    for (call.arguments, func_type.params) |*arg, expected_param| {
                        const arg_type = try self.typeCheckExpression(arg);
                        if (!self.typesCompatible(arg_type, expected_param)) {
                            return TyperError.TypeMismatch;
                        }
                    }

                    // Return function's return type
                    if (func_type.return_type) |return_type| {
                        return return_type.*;
                    } else {
                        return OraType.Void;
                    }
                },
                else => {
                    // Not a function - trying to call a variable
                    return TyperError.InvalidOperation;
                },
            }
        }

        // Check for built-in functions
        if (self.isBuiltinFunction(function_name)) {
            return try self.typeCheckBuiltinCall(call);
        }

        return TyperError.UndeclaredFunction;
    }

    /// Check if a function is a built-in function
    fn isBuiltinFunction(self: *Typer, name: []const u8) bool {
        _ = self;
        // Actual built-in functions in Ora language
        return std.mem.eql(u8, name, "requires") or
            std.mem.eql(u8, name, "ensures") or
            std.mem.eql(u8, name, "invariant") or
            std.mem.eql(u8, name, "old") or
            std.mem.eql(u8, name, "log") or
            // Division functions (with @ prefix)
            std.mem.eql(u8, name, "@divmod") or
            std.mem.eql(u8, name, "@divTrunc") or
            std.mem.eql(u8, name, "@divFloor") or
            std.mem.eql(u8, name, "@divCeil") or
            std.mem.eql(u8, name, "@divExact");
    }

    /// Type check built-in function calls
    fn typeCheckBuiltinCall(self: *Typer, call: *ast.CallExpr) TyperError!OraType {
        // Extract function name from callee
        const function_name = switch (call.callee.*) {
            .Identifier => |*ident| ident.name,
            else => return TyperError.InvalidOperation,
        };

        if (std.mem.eql(u8, function_name, "requires")) {
            // requires(condition, [message]) -> void
            if (call.arguments.len < 1 or call.arguments.len > 2) {
                return TyperError.ArgumentCountMismatch;
            }

            const condition_type = try self.typeCheckExpression(&call.arguments[0]);
            if (!std.meta.eql(condition_type, OraType.Bool)) {
                return TyperError.TypeMismatch;
            }

            if (call.arguments.len == 2) {
                const message_type = try self.typeCheckExpression(&call.arguments[1]);
                if (!std.meta.eql(message_type, OraType.String)) {
                    return TyperError.TypeMismatch;
                }
            }

            return OraType.Void;
        }

        if (std.mem.eql(u8, function_name, "ensures")) {
            // ensures(condition, [message]) -> void
            if (call.arguments.len < 1 or call.arguments.len > 2) {
                return TyperError.ArgumentCountMismatch;
            }

            const condition_type = try self.typeCheckExpression(&call.arguments[0]);
            if (!std.meta.eql(condition_type, OraType.Bool)) {
                return TyperError.TypeMismatch;
            }

            if (call.arguments.len == 2) {
                const message_type = try self.typeCheckExpression(&call.arguments[1]);
                if (!std.meta.eql(message_type, OraType.String)) {
                    return TyperError.TypeMismatch;
                }
            }

            return OraType.Void;
        }

        if (std.mem.eql(u8, function_name, "invariant")) {
            // invariant(condition, [message]) -> void
            if (call.arguments.len < 1 or call.arguments.len > 2) {
                return TyperError.ArgumentCountMismatch;
            }

            const condition_type = try self.typeCheckExpression(&call.arguments[0]);
            if (!std.meta.eql(condition_type, OraType.Bool)) {
                return TyperError.TypeMismatch;
            }

            if (call.arguments.len == 2) {
                const message_type = try self.typeCheckExpression(&call.arguments[1]);
                if (!std.meta.eql(message_type, OraType.String)) {
                    return TyperError.TypeMismatch;
                }
            }

            return OraType.Void;
        }

        if (std.mem.eql(u8, function_name, "old")) {
            // old(expression) -> same type as expression
            if (call.arguments.len != 1) {
                return TyperError.ArgumentCountMismatch;
            }

            // Return the same type as the argument
            return try self.typeCheckExpression(&call.arguments[0]);
        }

        if (std.mem.eql(u8, function_name, "log")) {
            // log is handled differently as it's a statement, not a function call
            // But if it appears in expression context, it returns void
            return OraType.Void;
        }

        // Division functions (Zig-inspired, with @ prefix)
        if (std.mem.eql(u8, function_name, "@divTrunc") or
            std.mem.eql(u8, function_name, "@divFloor") or
            std.mem.eql(u8, function_name, "@divCeil") or
            std.mem.eql(u8, function_name, "@divExact"))
        {
            // @divTrunc(a, b) -> same type as a and b (must be compatible)
            if (call.arguments.len != 2) {
                return TyperError.ArgumentCountMismatch;
            }

            const lhs_type = try self.typeCheckExpression(&call.arguments[0]);
            const rhs_type = try self.typeCheckExpression(&call.arguments[1]);

            if (!self.isNumericType(lhs_type) or !self.isNumericType(rhs_type)) {
                return TyperError.TypeMismatch;
            }

            return self.commonNumericType(lhs_type, rhs_type);
        }

        if (std.mem.eql(u8, function_name, "@divmod")) {
            // @divmod(a, b) -> (quotient, remainder) tuple
            if (call.arguments.len != 2) {
                return TyperError.ArgumentCountMismatch;
            }

            const lhs_type = try self.typeCheckExpression(&call.arguments[0]);
            const rhs_type = try self.typeCheckExpression(&call.arguments[1]);

            if (!self.isNumericType(lhs_type) or !self.isNumericType(rhs_type)) {
                return TyperError.TypeMismatch;
            }

            const common_type = self.commonNumericType(lhs_type, rhs_type);

            // Return tuple type (quotient, remainder) both same type
            var tuple_types = std.ArrayList(OraType).init(self.allocator);
            defer tuple_types.deinit();

            try tuple_types.append(common_type); // quotient
            try tuple_types.append(common_type); // remainder

            return OraType{ .Tuple = .{
                .types = try tuple_types.toOwnedSlice(),
            } };
        }

        // Default for other built-ins
        return OraType.Unknown;
    }

    /// Convert AST type reference to ZigOra type
    pub fn convertAstTypeToOraType(self: *Typer, ast_type: *ast.TypeRef) TyperError!OraType {
        return switch (ast_type.*) {
            .Bool => OraType.Bool,
            .Address => OraType.Address,
            .U8 => OraType.U8,
            .U16 => OraType.U16,
            .U32 => OraType.U32,
            .U64 => OraType.U64,
            .U128 => OraType.U128,
            .U256 => OraType.U256,
            .I8 => OraType.I8,
            .I16 => OraType.I16,
            .I32 => OraType.I32,
            .I64 => OraType.I64,
            .I128 => OraType.I128,
            .I256 => OraType.I256,
            .String => OraType.String,
            .Bytes => OraType.Bytes,
            .Slice => |slice_element_type| {
                // Use arena allocator for type lifetime management
                const element_type = try self.type_arena.allocator().create(OraType);
                element_type.* = try self.convertAstTypeToOraType(slice_element_type);
                return OraType{ .Slice = element_type };
            },
            .Mapping => |mapping_info| {
                // Use arena allocator for type lifetime management
                const key_type = try self.type_arena.allocator().create(OraType);
                key_type.* = try self.convertAstTypeToOraType(mapping_info.key);
                const value_type = try self.type_arena.allocator().create(OraType);
                value_type.* = try self.convertAstTypeToOraType(mapping_info.value);
                return OraType{ .Mapping = .{
                    .key = key_type,
                    .value = value_type,
                } };
            },
            .DoubleMap => |double_map_info| {
                // Use arena allocator for type lifetime management
                const key1_type = try self.type_arena.allocator().create(OraType);
                key1_type.* = try self.convertAstTypeToOraType(double_map_info.key1);
                const key2_type = try self.type_arena.allocator().create(OraType);
                key2_type.* = try self.convertAstTypeToOraType(double_map_info.key2);
                const value_type = try self.type_arena.allocator().create(OraType);
                value_type.* = try self.convertAstTypeToOraType(double_map_info.value);
                return OraType{ .DoubleMap = .{
                    .key1 = key1_type,
                    .key2 = key2_type,
                    .value = value_type,
                } };
            },
            .Identifier => |_| {
                // TODO: Look up custom types (structs, enums)
                return OraType.Unknown;
            },
            .ErrorUnion => |error_union| {
                // Error unions use the success type as the primary type
                return try self.convertAstTypeToOraType(error_union.success_type);
            },
            .Result => |result| {
                // Result types use the ok type as the primary type
                return try self.convertAstTypeToOraType(result.ok_type);
            },
            .Tuple => |tuple| {
                // Convert tuple elements to OraType
                const element_types = try self.type_arena.allocator().alloc(OraType, tuple.types.len);
                for (tuple.types, 0..) |*element_type, i| {
                    element_types[i] = try self.convertAstTypeToOraType(element_type);
                }
                return OraType{ .Tuple = .{
                    .types = element_types,
                } };
            },
            .Unknown => OraType.Unknown,
        };
    }

    /// Create function type from function node
    fn createFunctionType(self: *Typer, function: *ast.FunctionNode) TyperError!OraType {
        _ = self;
        _ = function;
        // TODO: Implement proper function type creation without pointer issues
        // For now, return a simple type to avoid the segmentation fault
        return OraType.Unknown;
    }

    /// Check if two types are compatible
    fn typesCompatible(self: *Typer, lhs: OraType, rhs: OraType) bool {
        // Exact type match
        if (self.typeEquals(lhs, rhs)) {
            return true;
        }

        // Allow compatible numeric conversions
        return self.isNumericConversionValid(rhs, lhs);
    }

    /// Check if a numeric conversion is valid (from -> to)
    fn isNumericConversionValid(self: *Typer, from: OraType, to: OraType) bool {
        // Allow promotion within unsigned types
        const unsigned_hierarchy = [_]OraType{ .U8, .U16, .U32, .U64, .U128, .U256 };
        if (self.isTypeInHierarchy(from, &unsigned_hierarchy) and self.isTypeInHierarchy(to, &unsigned_hierarchy)) {
            return self.getTypeHierarchyIndex(from, &unsigned_hierarchy) <= self.getTypeHierarchyIndex(to, &unsigned_hierarchy);
        }

        // Allow promotion within signed types
        const signed_hierarchy = [_]OraType{ .I8, .I16, .I32, .I64, .I128, .I256 };
        if (self.isTypeInHierarchy(from, &signed_hierarchy) and self.isTypeInHierarchy(to, &signed_hierarchy)) {
            return self.getTypeHierarchyIndex(from, &signed_hierarchy) <= self.getTypeHierarchyIndex(to, &signed_hierarchy);
        }

        // Allow unsigned to signed conversion if the signed type is larger or equal
        switch (from) {
            .U8 => switch (to) {
                .I8, .I16, .I32, .I64, .I128, .I256 => return true,
                else => return false,
            },
            .U16 => switch (to) {
                .I16, .I32, .I64, .I128, .I256 => return true,
                else => return false,
            },
            .U32 => switch (to) {
                .I32, .I64, .I128, .I256 => return true,
                else => return false,
            },
            .U64 => switch (to) {
                .I64, .I128, .I256 => return true,
                else => return false,
            },
            .U128 => switch (to) {
                .I128, .I256 => return true,
                else => return false,
            },
            .U256 => switch (to) {
                .I256 => return true,
                else => return false,
            },
            else => return false,
        }
    }

    /// Check if a type is in a hierarchy
    fn isTypeInHierarchy(self: *Typer, typ: OraType, hierarchy: []const OraType) bool {
        for (hierarchy) |h_type| {
            if (self.typeEquals(typ, h_type)) {
                return true;
            }
        }
        return false;
    }

    /// Get the index of a type in a hierarchy
    fn getTypeHierarchyIndex(self: *Typer, typ: OraType, hierarchy: []const OraType) usize {
        for (hierarchy, 0..) |h_type, i| {
            if (self.typeEquals(typ, h_type)) {
                return i;
            }
        }
        return hierarchy.len; // Not found
    }

    /// Check if two types are structurally equal
    fn typeEquals(self: *Typer, lhs: OraType, rhs: OraType) bool {
        return switch (lhs) {
            .Bool => switch (rhs) {
                .Bool => true,
                else => false,
            },
            .Address => switch (rhs) {
                .Address => true,
                else => false,
            },
            .U8 => switch (rhs) {
                .U8 => true,
                else => false,
            },
            .U16 => switch (rhs) {
                .U16 => true,
                else => false,
            },
            .U32 => switch (rhs) {
                .U32 => true,
                else => false,
            },
            .U64 => switch (rhs) {
                .U64 => true,
                else => false,
            },
            .U128 => switch (rhs) {
                .U128 => true,
                else => false,
            },
            .U256 => switch (rhs) {
                .U256 => true,
                else => false,
            },
            .I8 => switch (rhs) {
                .I8 => true,
                else => false,
            },
            .I16 => switch (rhs) {
                .I16 => true,
                else => false,
            },
            .I32 => switch (rhs) {
                .I32 => true,
                else => false,
            },
            .I64 => switch (rhs) {
                .I64 => true,
                else => false,
            },
            .I128 => switch (rhs) {
                .I128 => true,
                else => false,
            },
            .I256 => switch (rhs) {
                .I256 => true,
                else => false,
            },
            .String => switch (rhs) {
                .String => true,
                else => false,
            },
            .Bytes => switch (rhs) {
                .Bytes => true,
                else => false,
            },
            .Void => switch (rhs) {
                .Void => true,
                else => false,
            },
            .Unknown => true, // Unknown types are compatible with everything
            .Error => switch (rhs) {
                .Error => true,
                else => false,
            },
            .Slice => |lhs_elem| switch (rhs) {
                .Slice => |rhs_elem| self.typeEquals(lhs_elem.*, rhs_elem.*),
                else => false,
            },
            .Mapping => |lhs_map| switch (rhs) {
                .Mapping => |rhs_map| self.typeEquals(lhs_map.key.*, rhs_map.key.*) and
                    self.typeEquals(lhs_map.value.*, rhs_map.value.*),
                else => false,
            },
            .DoubleMap => |lhs_dmap| switch (rhs) {
                .DoubleMap => |rhs_dmap| self.typeEquals(lhs_dmap.key1.*, rhs_dmap.key1.*) and
                    self.typeEquals(lhs_dmap.key2.*, rhs_dmap.key2.*) and
                    self.typeEquals(lhs_dmap.value.*, rhs_dmap.value.*),
                else => false,
            },
            .Function => |lhs_func| switch (rhs) {
                .Function => |rhs_func| {
                    // Compare parameter count
                    if (lhs_func.params.len != rhs_func.params.len) return false;
                    // Compare each parameter type
                    for (lhs_func.params, rhs_func.params) |lhs_param, rhs_param| {
                        if (!self.typeEquals(lhs_param, rhs_param)) return false;
                    }
                    // Compare return types
                    if (lhs_func.return_type) |lhs_ret| {
                        if (rhs_func.return_type) |rhs_ret| {
                            return self.typeEquals(lhs_ret.*, rhs_ret.*);
                        } else {
                            return false; // lhs has return type, rhs doesn't
                        }
                    } else {
                        return rhs_func.return_type == null; // both should have no return type
                    }
                },
                else => false,
            },
            .Tuple => |lhs_tuple| switch (rhs) {
                .Tuple => |rhs_tuple| {
                    // Compare element count
                    if (lhs_tuple.types.len != rhs_tuple.types.len) return false;
                    // Compare each element type
                    for (lhs_tuple.types, rhs_tuple.types) |lhs_elem, rhs_elem| {
                        if (!self.typeEquals(lhs_elem, rhs_elem)) return false;
                    }
                    return true;
                },
                else => false,
            },
        };
    }

    /// Check if a type is numeric
    fn isNumericType(self: *Typer, typ: OraType) bool {
        _ = self;
        return switch (typ) {
            .U8, .U16, .U32, .U64, .U128, .U256, .I8, .I16, .I32, .I64, .I128, .I256 => true,
            else => false,
        };
    }

    /// Get common numeric type for operations
    fn commonNumericType(self: *Typer, lhs: OraType, rhs: OraType) OraType {
        // If both types are the same, return that type
        if (self.typeEquals(lhs, rhs)) {
            return lhs;
        }

        // Mixed signed/unsigned arithmetic: promote to the larger signed type
        const signed_hierarchy = [_]OraType{ .I8, .I16, .I32, .I64, .I128, .I256 };
        const unsigned_hierarchy = [_]OraType{ .U8, .U16, .U32, .U64, .U128, .U256 };

        const lhs_is_signed = self.isTypeInHierarchy(lhs, &signed_hierarchy);
        const rhs_is_signed = self.isTypeInHierarchy(rhs, &signed_hierarchy);

        // If both are signed, promote to the larger one
        if (lhs_is_signed and rhs_is_signed) {
            const lhs_idx = self.getTypeHierarchyIndex(lhs, &signed_hierarchy);
            const rhs_idx = self.getTypeHierarchyIndex(rhs, &signed_hierarchy);
            return signed_hierarchy[@max(lhs_idx, rhs_idx)];
        }

        // If both are unsigned, promote to the larger one
        if (!lhs_is_signed and !rhs_is_signed) {
            const lhs_idx = self.getTypeHierarchyIndex(lhs, &unsigned_hierarchy);
            const rhs_idx = self.getTypeHierarchyIndex(rhs, &unsigned_hierarchy);
            return unsigned_hierarchy[@max(lhs_idx, rhs_idx)];
        }

        // Mixed signed/unsigned: promote to a signed type that can hold both
        const signed_type = if (lhs_is_signed) lhs else rhs;
        const unsigned_type = if (lhs_is_signed) rhs else lhs;

        const signed_idx = self.getTypeHierarchyIndex(signed_type, &signed_hierarchy);
        const unsigned_idx = self.getTypeHierarchyIndex(unsigned_type, &unsigned_hierarchy);

        // Use the signed type if it's large enough, otherwise promote to a larger signed type
        const min_signed_idx = @max(signed_idx, unsigned_idx);
        return signed_hierarchy[@min(min_signed_idx, signed_hierarchy.len - 1)];
    }

    /// Validate memory region constraints
    fn validateMemoryRegion(self: *Typer, region: ast.MemoryRegion, typ: OraType) TyperError!void {
        _ = self;

        switch (region) {
            .Storage => {
                // Only certain types can be stored in storage
                switch (typ) {
                    .Mapping, .DoubleMap => {}, // OK
                    .Bool, .Address, .U8, .U16, .U32, .U64, .U128, .U256, .I8, .I16, .I32, .I64, .I128, .I256, .String => {}, // OK
                    else => return TyperError.InvalidMemoryRegion,
                }
            },
            else => {
                // Other regions are more permissive
            },
        }
    }

    /// Type check unary operations
    fn typeCheckUnaryOp(self: *Typer, op: ast.UnaryOp, operand_type: OraType) TyperError!OraType {
        return switch (op) {
            .Minus => {
                if (self.isNumericType(operand_type)) {
                    return operand_type;
                }
                return TyperError.TypeMismatch;
            },
            .Bang => {
                if (std.meta.eql(operand_type, OraType.Bool)) {
                    return OraType.Bool;
                }
                return TyperError.TypeMismatch;
            },
            .BitNot => {
                if (self.isIntegerType(operand_type)) {
                    return operand_type;
                }
                return TyperError.TypeMismatch;
            },
        };
    }

    /// Type check index access (arrays, mappings)
    fn typeCheckIndexAccess(self: *Typer, target_type: OraType, index_type: OraType) TyperError!OraType {
        return switch (target_type) {
            .Slice => |elem_type| {
                // Array/slice indexing requires integer index
                if (self.isIntegerType(index_type)) {
                    return elem_type.*;
                }
                return TyperError.TypeMismatch;
            },
            .Mapping => |mapping| {
                // Mapping access requires compatible key type
                if (self.typesCompatible(index_type, mapping.key.*)) {
                    return mapping.value.*;
                }
                return TyperError.TypeMismatch;
            },
            .DoubleMap => |_| {
                // DoubleMap requires special syntax - shouldn't reach here with single index
                return TyperError.InvalidOperation;
            },
            else => return TyperError.InvalidOperation,
        };
    }

    /// Type check field access
    fn typeCheckFieldAccess(self: *Typer, target_type: OraType, field_name: []const u8) TyperError!OraType {
        _ = self;
        _ = target_type;

        // Special case for std library access
        if (std.mem.eql(u8, field_name, "transaction") or
            std.mem.eql(u8, field_name, "block") or
            std.mem.eql(u8, field_name, "msg"))
        {
            // std.transaction, std.block, std.msg return special context types
            return OraType.Unknown; // TODO: Define proper context types
        }

        // TODO: Implement struct field access when structs are added
        return OraType.Unknown;
    }

    /// Check if cast is valid
    fn isCastValid(self: *Typer, from: OraType, to: OraType) bool {
        // Same type casts are always valid
        if (self.typeEquals(from, to)) {
            return true;
        }

        // Numeric type conversions
        if (self.isNumericType(from) and self.isNumericType(to)) {
            return true; // Allow all numeric conversions (with potential warnings)
        }

        // Address <-> U256 conversions
        if ((std.meta.eql(from, OraType.Address) and std.meta.eql(to, OraType.U256)) or
            (std.meta.eql(from, OraType.U256) and std.meta.eql(to, OraType.Address)))
        {
            return true;
        }

        // Unknown types can be cast to anything (for incomplete code)
        if (std.meta.eql(from, OraType.Unknown) or std.meta.eql(to, OraType.Unknown)) {
            return true;
        }

        return false;
    }

    /// Check if type is an integer type
    fn isIntegerType(self: *Typer, typ: OraType) bool {
        _ = self;
        return switch (typ) {
            .U8, .U16, .U32, .U64, .U128, .U256, .I8, .I16, .I32, .I64, .I128, .I256 => true,
            else => false,
        };
    }
};

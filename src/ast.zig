const std = @import("std");
const testing = std.testing;

// Import modular AST components
pub const expressions = @import("ast/expressions.zig");
pub const statements = @import("ast/statements.zig");
pub const type_info = @import("ast/type_info.zig");
pub const ast_visitor = @import("ast/ast_visitor.zig");

// Import serializer and type resolver
const ast_serializer = @import("ast/ast_serializer.zig");
pub const AstSerializer = ast_serializer.AstSerializer;

const type_resolver = @import("ast/type_resolver.zig");
pub const TypeResolver = type_resolver.TypeResolver;
pub const TypeResolutionError = type_resolver.TypeResolutionError;

/// Source span to track position info in the source code
pub const SourceSpan = struct {
    // File identity for multi-file projects (0 = unknown/default)
    file_id: u32 = 0,
    // 1-based caret position
    line: u32,
    column: u32,
    // Byte length of the span
    length: u32,
    // Start byte offset within file (for precise mapping)
    byte_offset: u32 = 0,
    // Optional original slice
    lexeme: ?[]const u8 = null,
};

// Namespace groupings for cleaner API
pub const Expressions = expressions;
pub const Statements = statements;
pub const Types = type_info;

// Memory and region types
pub const Memory = struct {
    pub const Region = statements.MemoryRegion;
    pub const VariableKind = statements.VariableKind;
};

// Operator types
pub const Operators = struct {
    pub const Binary = expressions.BinaryOp;
    pub const Unary = expressions.UnaryOp;
    pub const Compound = expressions.CompoundAssignmentOp;
    pub const Cast = expressions.CastType;
};

// Switch-related types
pub const Switch = struct {
    pub const Case = expressions.SwitchCase;
    pub const Pattern = expressions.SwitchPattern;
    pub const Body = expressions.SwitchBody;
    pub const ExprNode = expressions.SwitchExprNode;
};

// Literal types
pub const Literals = struct {
    pub const Integer = expressions.IntegerLiteral;
    pub const String = expressions.StringLiteral;
    pub const Array = expressions.ArrayLiteralExpr;
};

// Top-level AST nodes that were missing
pub const ModuleNode = struct {
    name: ?[]const u8, // Module name (optional, can be inferred from file)
    imports: []ImportNode, // All import declarations
    declarations: []AstNode, // All top-level declarations (contracts, functions, structs, etc.)
    span: SourceSpan,
};

pub const ContractNode = struct {
    name: []const u8,
    extends: ?[]const u8, // Base contract name for inheritance
    implements: [][]const u8, // Interface names this contract implements
    attributes: []u8, // Placeholder for future contract attributes
    body: []AstNode, // Contract body contains declarations
    span: SourceSpan,
};

pub const FunctionNode = struct {
    name: []const u8,
    parameters: []ParameterNode,
    return_type_info: ?Types.TypeInfo, // Unified type information for return type
    body: Statements.BlockNode,
    visibility: Visibility,
    attributes: []u8, // Placeholder for future function attributes
    is_inline: bool, // inline functions
    requires_clauses: []*Expressions.ExprNode, // Preconditions
    ensures_clauses: []*Expressions.ExprNode, // Postconditions
    span: SourceSpan,
};

pub const ParameterNode = struct {
    name: []const u8,
    type_info: Types.TypeInfo, // Unified type information
    is_mutable: bool, // mut parameter
    default_value: ?*Expressions.ExprNode, // Default parameter value
    span: SourceSpan,
};

pub const Visibility = enum { Public, Private };

// NOTE: Contract and function attributes were planned but not implemented yet
// These will be reintroduced when attributes are supported by the language

pub const StructDeclNode = struct {
    name: []const u8,
    fields: []StructField,
    span: SourceSpan,
};

pub const StructField = struct {
    name: []const u8,
    type_info: Types.TypeInfo, // Unified type information
    span: SourceSpan,
};

pub const EnumDeclNode = struct {
    name: []const u8,
    variants: []EnumVariant,
    underlying_type_info: ?Types.TypeInfo, // Unified type information for underlying type
    span: SourceSpan,
    has_explicit_values: bool = false, // Track if this enum has explicit values
};

pub const EnumVariant = struct {
    name: []const u8,
    value: ?Expressions.ExprNode, // Optional explicit value
    resolved_value: ?u256 = null, // Computed value after type resolution
    span: SourceSpan,
};

pub const ImportNode = struct {
    path: []const u8,
    alias: ?[]const u8,
    span: SourceSpan,

    // Current implementation supports basic import patterns
    // See ora-example/imports/basic_imports.ora for supported syntax
};

pub const LogDeclNode = struct {
    name: []const u8,
    fields: []LogField,
    span: SourceSpan,
};

pub const LogField = struct {
    name: []const u8,
    type_info: Types.TypeInfo, // Unified type information
    indexed: bool,
    span: SourceSpan,
};

pub const LockNode = struct {
    path: Expressions.ExprNode,
    span: SourceSpan,
};

pub const ConstantNode = struct {
    name: []const u8,
    typ: Types.TypeInfo,
    value: *Expressions.ExprNode,
    visibility: Visibility,
    span: SourceSpan,
};

// Unified AST Node type that the visitor expects
pub const AstNode = union(enum) {
    // Top-level program structure
    Module: ModuleNode,

    // Top-level declarations that exist in Ora
    Contract: ContractNode,
    Function: FunctionNode,
    Constant: ConstantNode,
    VariableDecl: Statements.VariableDeclNode,
    StructDecl: StructDeclNode,
    EnumDecl: EnumDeclNode,
    LogDecl: LogDeclNode,
    Import: ImportNode,
    ErrorDecl: Statements.ErrorDeclNode,

    // Structural nodes - use pointers to break circular dependency
    Block: Statements.BlockNode,
    Expression: *Expressions.ExprNode, // Use pointer to break circular dependency
    Statement: *Statements.StmtNode, // Use pointer to break circular dependency
    TryBlock: Statements.TryBlockNode,
};

// Utility functions
pub fn deinitExprNode(allocator: std.mem.Allocator, expr: *Expressions.ExprNode) void {
    expressions.deinitExprNode(allocator, expr);
}

pub fn deinitStmtNode(allocator: std.mem.Allocator, stmt: *Statements.StmtNode) void {
    statements.deinitStmtNode(allocator, stmt);
}

pub fn deinitBlockNode(allocator: std.mem.Allocator, block: *Statements.BlockNode) void {
    statements.deinitBlockNode(allocator, block);
}

pub fn deinitAstNode(allocator: std.mem.Allocator, node: *AstNode) void {
    switch (node.*) {
        .Module => |*module| {
            allocator.free(module.imports);
            for (module.declarations) |*decl| {
                deinitAstNode(allocator, decl);
            }
            allocator.free(module.declarations);
        },
        .Contract => |*contract| {
            for (contract.body) |*child| {
                deinitAstNode(allocator, child);
            }
            allocator.free(contract.body);
        },
        .Constant => |*constant| {
            deinitExprNode(allocator, constant.value);
            allocator.destroy(constant.value);
        },
        .Function => |*function| {
            for (function.parameters) |*param| {
                if (param.default_value) |default_val| {
                    deinitExprNode(allocator, default_val);
                    allocator.destroy(default_val);
                }
            }
            allocator.free(function.parameters);
            for (function.requires_clauses) |clause| {
                deinitExprNode(allocator, clause);
                allocator.destroy(clause);
            }
            for (function.ensures_clauses) |clause| {
                deinitExprNode(allocator, clause);
                allocator.destroy(clause);
            }
            allocator.free(function.requires_clauses);
            allocator.free(function.ensures_clauses);
            deinitBlockNode(allocator, &function.body);
        },
        .VariableDecl => |*var_decl| {
            if (var_decl.value) |value| {
                deinitExprNode(allocator, value);
            }
            if (var_decl.tuple_names) |names| {
                allocator.free(names);
            }
        },
        .StructDecl => |*struct_decl| {
            allocator.free(struct_decl.fields);
        },
        .EnumDecl => |*enum_decl| {
            for (enum_decl.variants) |*variant| {
                if (variant.value) |*value| {
                    deinitExprNode(allocator, value);
                }
            }
            allocator.free(enum_decl.variants);
        },
        .LogDecl => |*log_decl| {
            allocator.free(log_decl.fields);
        },
        .Import => {
            // No dynamic allocations to free
        },
        .ErrorDecl => |*error_decl| {
            // Clean up parameters if present
            if (error_decl.parameters) |params| {
                for (params) |*param| {
                    if (param.default_value) |default_val| {
                        deinitExprNode(allocator, default_val);
                        allocator.destroy(default_val);
                    }
                }
                allocator.free(params);
            }
        },
        .Block => |*block| {
            deinitBlockNode(allocator, block);
        },
        .Expression => |expr| {
            deinitExprNode(allocator, expr);
        },
        .Statement => |stmt| {
            deinitStmtNode(allocator, stmt);
        },
        .TryBlock => |*try_block| {
            deinitBlockNode(allocator, &try_block.try_block);
            if (try_block.catch_block) |*catch_block| {
                deinitBlockNode(allocator, &catch_block.block);
            }
        },
    }
}

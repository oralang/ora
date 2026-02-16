// ============================================================================
// AST Module
// ============================================================================
//
// aggregation layer for AST types, visitors, serializers, and utilities.
//
// ============================================================================

const std = @import("std");
const testing = std.testing;

// Import modular AST components
pub const expressions = @import("ast/expressions.zig");
pub const statements = @import("ast/statements.zig");
pub const type_info = @import("ast/type_info.zig");
pub const slot_key = @import("ast/slot_key.zig");
pub const ast_visitor = @import("ast/ast_visitor.zig");
pub const verification = @import("ast/verification.zig");

// Import serializer
const ast_serializer = @import("ast/ast_serializer.zig");
pub const AstSerializer = ast_serializer.AstSerializer;

// Note: TypeResolver is exported from ast/type_resolver/mod.zig directly
// to avoid circular dependency (ast.zig imports type_resolver, type_resolver imports ast.zig)

// Re-export SourceSpan from separate module to break circular dependencies
pub const SourceSpan = @import("ast/source_span.zig").SourceSpan;

// Namespace groupings for cleaner API
pub const Expressions = expressions;
pub const Statements = statements;
pub const Types = type_info;
pub const Verification = verification;

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
    attributes: []u8, // Placeholder for future contract attributes
    body: []AstNode, // Contract body contains declarations
    is_generic: bool = false,
    type_param_names: []const []const u8 = &.{},
    span: SourceSpan,
};

/// Contract invariant (class invariant) for formal verification
/// Example: invariant totalSupplyInvariant(totalSupply == sumOfBalances);
pub const ContractInvariant = struct {
    name: []const u8, // Name of the invariant
    condition: *Expressions.ExprNode, // Invariant condition
    span: SourceSpan,
    /// Metadata: Always specification-only
    is_specification: bool = true,

    /// Check if this invariant is specification-only (not compiled to bytecode)
    pub fn isSpecificationOnly(self: *const ContractInvariant) bool {
        return self.is_specification;
    }
};

pub const FunctionNode = struct {
    name: []const u8,
    parameters: []ParameterNode,
    return_type_info: ?Types.TypeInfo, // Unified type information for return type
    body: Statements.BlockNode,
    visibility: Visibility,
    attributes: []u8, // Placeholder for future function attributes
    requires_clauses: []*Expressions.ExprNode, // Preconditions (formal verification)
    ensures_clauses: []*Expressions.ExprNode, // Postconditions (formal verification)
    modifies_clause: ?[]*Expressions.ExprNode = null, // Frame conditions - what storage can be modified
    is_ghost: bool = false, // Is this a ghost function? (specification-only)
    is_comptime_only: bool = false, // Private fn with all call sites folded — skip MLIR lowering
    is_generic: bool = false, // Has comptime type parameters — needs monomorphization
    type_param_names: []const []const u8 = &.{}, // Names of comptime type parameters (e.g., ["T", "U"])
    span: SourceSpan,

    /// Check if this function is specification-only (not compiled to bytecode)
    pub fn isSpecificationOnly(self: *const FunctionNode) bool {
        return self.is_ghost;
    }
};

pub const ParameterNode = struct {
    name: []const u8,
    type_info: Types.TypeInfo, // Unified type information
    is_mutable: bool, // mut parameter
    is_comptime: bool = false, // comptime parameter (must be known at compile time)
    default_value: ?*Expressions.ExprNode, // Default parameter value
    span: SourceSpan,
};

pub const Visibility = enum { Public, Private };

// NOTE: Contract and function attributes were planned but not implemented yet
// These will be reintroduced when attributes are supported by the language

pub const StructDeclNode = struct {
    name: []const u8,
    fields: []StructField,
    is_generic: bool = false,
    type_param_names: []const []const u8 = &.{},
    span: SourceSpan,
};

pub const StructField = struct {
    name: []const u8,
    type_info: Types.TypeInfo, // Unified type information
    span: SourceSpan,
};

pub const BitfieldDeclNode = struct {
    name: []const u8,
    base_type_info: Types.TypeInfo, // The base integer type (e.g. u256)
    fields: []BitfieldField,
    auto_packed: bool, // true if no @at() annotations
    span: SourceSpan,
};

pub const BitfieldField = struct {
    name: []const u8,
    type_info: Types.TypeInfo, // Field type (bool, uN, iN)
    offset: ?u32, // Bit offset (null if auto-packed)
    width: ?u32, // Bit width (null to derive from type)
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

    // current implementation supports basic import patterns
    // see ora-example/imports/basic_imports.ora for supported syntax
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
    /// Is this a ghost constant (specification-only)?
    is_ghost: bool = false,

    pub fn isSpecificationOnly(self: *const ConstantNode) bool {
        return self.is_ghost;
    }
};

// Unified AST Node type that the visitor expects
pub const AstNode = union(enum) {
    // top-level program structure
    Module: ModuleNode,

    // top-level declarations that exist in Ora
    Contract: ContractNode,
    Function: FunctionNode,
    Constant: ConstantNode,
    VariableDecl: Statements.VariableDeclNode,
    StructDecl: StructDeclNode,
    BitfieldDecl: BitfieldDeclNode,
    EnumDecl: EnumDeclNode,
    LogDecl: LogDeclNode,
    Import: ImportNode,
    ErrorDecl: Statements.ErrorDeclNode,
    ContractInvariant: ContractInvariant, // Formal verification invariants

    // structural nodes - use pointers to break circular dependency
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
        .ContractInvariant => |*invariant| {
            deinitExprNode(allocator, invariant.condition);
            allocator.destroy(invariant.condition);
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
        .BitfieldDecl => |*bf_decl| {
            allocator.free(bf_decl.fields);
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
            // no dynamic allocations to free
        },
        .ErrorDecl => |*error_decl| {
            // clean up parameters if present
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

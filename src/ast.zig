const std = @import("std");
const testing = std.testing;

// Import modular AST components
pub const expressions = @import("ast/expressions.zig");
pub const statements = @import("ast/statements.zig");
pub const types = @import("ast/types.zig");
pub const type_info = @import("ast/type_info.zig");

// Import serializer and type resolver
const ast_serializer = @import("ast/ast_serializer.zig");
pub const AstSerializer = ast_serializer.AstSerializer;

const type_resolver = @import("ast/type_resolver.zig");
pub const TypeResolver = type_resolver.TypeResolver;
pub const TypeResolutionError = type_resolver.TypeResolutionError;

// Re-export main types for convenience
pub const ExprNode = expressions.ExprNode;
pub const StmtNode = statements.StmtNode;
pub const SourceSpan = types.SourceSpan;
pub const BlockNode = statements.BlockNode;
pub const BinaryOp = expressions.BinaryOp;
pub const UnaryOp = expressions.UnaryOp;
pub const CompoundAssignmentOp = expressions.CompoundAssignmentOp;
pub const CastType = expressions.CastType;

// Re-export new unified type system
pub const TypeInfo = type_info.TypeInfo;
pub const TypeCategory = type_info.TypeCategory;
pub const OraType = type_info.OraType;
pub const TypeSource = type_info.TypeSource;
pub const CommonTypes = type_info.CommonTypes;

// Memory and utilities re-exports
pub const MemoryRegion = statements.MemoryRegion;
pub const VariableKind = statements.VariableKind;

// Re-export specific node types needed by visitor
pub const IdentifierExpr = expressions.IdentifierExpr;
pub const LiteralExpr = expressions.LiteralExpr;
pub const BinaryExpr = expressions.BinaryExpr;
pub const UnaryExpr = expressions.UnaryExpr;
pub const AssignmentExpr = expressions.AssignmentExpr;
pub const CompoundAssignmentExpr = expressions.CompoundAssignmentExpr;
pub const CallExpr = expressions.CallExpr;
pub const IndexExpr = expressions.IndexExpr;
pub const FieldAccessExpr = expressions.FieldAccessExpr;
pub const CastExpr = expressions.CastExpr;
pub const ComptimeExpr = expressions.ComptimeExpr;
pub const OldExpr = expressions.OldExpr;
pub const TupleExpr = expressions.TupleExpr;
pub const QuantifiedExpr = expressions.QuantifiedExpr;
pub const QuantifierType = expressions.QuantifierType;
pub const TryExpr = expressions.TryExpr;
pub const ErrorReturnExpr = expressions.ErrorReturnExpr;
pub const ErrorCastExpr = expressions.ErrorCastExpr;
pub const ShiftExpr = expressions.ShiftExpr;
pub const StructInstantiationExpr = expressions.StructInstantiationExpr;
pub const AnonymousStructExpr = expressions.AnonymousStructExpr;
pub const AnonymousStructField = expressions.AnonymousStructField;
pub const RangeExpr = expressions.RangeExpr;
pub const LabeledBlockExpr = expressions.LabeledBlockExpr;
pub const DestructuringExpr = expressions.DestructuringExpr;
pub const DestructuringPattern = expressions.DestructuringPattern;
pub const StructDestructureField = expressions.StructDestructureField;
pub const EnumLiteralExpr = expressions.EnumLiteralExpr;
pub const AddressLiteral = expressions.AddressLiteral;
pub const HexLiteral = expressions.HexLiteral;
pub const BinaryLiteral = expressions.BinaryLiteral;
pub const StructInstantiationField = expressions.StructInstantiationField;
pub const SwitchCase = expressions.SwitchCase;
pub const SwitchPattern = expressions.SwitchPattern;
pub const SwitchBody = expressions.SwitchBody;

pub const ReturnNode = statements.ReturnNode;
pub const IfNode = statements.IfNode;
pub const WhileNode = statements.WhileNode;
pub const ForLoopNode = statements.ForLoopNode;
pub const LoopPattern = statements.LoopPattern;
pub const LogNode = statements.LogNode;
pub const InvariantNode = statements.InvariantNode;
pub const RequiresNode = statements.RequiresNode;
pub const EnsuresNode = statements.EnsuresNode;
pub const VariableDeclNode = statements.VariableDeclNode;
pub const ErrorDeclNode = statements.ErrorDeclNode;
pub const TryBlockNode = statements.TryBlockNode;
pub const CatchBlock = statements.CatchBlock;
pub const SwitchNode = statements.SwitchNode;
pub const BreakNode = statements.BreakNode;
pub const ContinueNode = statements.ContinueNode;
pub const DestructuringAssignmentNode = statements.DestructuringAssignmentNode;
pub const MoveNode = statements.MoveNode;
pub const UnlockNode = statements.UnlockNode;
pub const LabeledBlockNode = statements.LabeledBlockNode;

// Expression nodes that were missing
// Note: IdentifierNode doesn't exist - identifiers are handled as part of expressions
pub const SwitchExprNode = expressions.SwitchExprNode;
pub const IntegerLiteral = expressions.IntegerLiteral;
pub const StringLiteral = expressions.StringLiteral;
pub const ArrayLiteralExpr = expressions.ArrayLiteralExpr;

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
    return_type_info: ?TypeInfo, // Unified type information for return type
    body: BlockNode,
    visibility: Visibility,
    attributes: []u8, // Placeholder for future function attributes
    is_inline: bool, // inline functions
    requires_clauses: []*ExprNode, // Preconditions
    ensures_clauses: []*ExprNode, // Postconditions
    span: SourceSpan,
};

pub const ParameterNode = struct {
    name: []const u8,
    type_info: TypeInfo, // Unified type information
    is_mutable: bool, // mut parameter
    default_value: ?*ExprNode, // Default parameter value
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
    type_info: TypeInfo, // Unified type information
    span: SourceSpan,
};

pub const EnumDeclNode = struct {
    name: []const u8,
    variants: []EnumVariant,
    underlying_type_info: ?TypeInfo, // Unified type information for underlying type
    span: SourceSpan,
    has_explicit_values: bool = false, // Track if this enum has explicit values
};

pub const EnumVariant = struct {
    name: []const u8,
    value: ?ExprNode, // Optional explicit value
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
    type_info: TypeInfo, // Unified type information
    indexed: bool,
    span: SourceSpan,
};

pub const LockNode = struct {
    path: ExprNode,
    span: SourceSpan,
};

pub const ConstantNode = struct {
    name: []const u8,
    typ: TypeInfo,
    value: *ExprNode,
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
    VariableDecl: VariableDeclNode,
    StructDecl: StructDeclNode,
    EnumDecl: EnumDeclNode,
    LogDecl: LogDeclNode,
    Import: ImportNode,
    ErrorDecl: ErrorDeclNode,

    // Structural nodes - use pointers to break circular dependency
    Block: BlockNode,
    Expression: *ExprNode, // Use pointer to break circular dependency
    Statement: *StmtNode, // Use pointer to break circular dependency
    TryBlock: TryBlockNode,
};

// Utility functions
pub fn deinitExprNode(allocator: std.mem.Allocator, expr: *ExprNode) void {
    expressions.deinitExprNode(allocator, expr);
}

pub fn deinitStmtNode(allocator: std.mem.Allocator, stmt: *StmtNode) void {
    statements.deinitStmtNode(allocator, stmt);
}

pub fn deinitBlockNode(allocator: std.mem.Allocator, block: *BlockNode) void {
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

const std = @import("std");
const testing = std.testing;

/// Source span to track position info in the source code
pub const SourceSpan = struct {
    line: u32,
    column: u32,
    length: u32, // Added for better error reporting
};

/// Base AST Node type for Ora
pub const AstNode = union(enum) {
    // Top-level declarations
    Contract: ContractNode,
    Function: FunctionNode,
    VariableDecl: VariableDeclNode,
    StructDecl: StructDeclNode,
    EnumDecl: EnumDeclNode,
    LogDecl: LogDeclNode,
    Import: ImportNode,
    ErrorDecl: ErrorDeclNode,

    // Statements and expressions
    Block: BlockNode,
    Expression: ExprNode,
    Statement: StmtNode,
    TryBlock: TryBlockNode,
};

/// Import Declaration (import alias = "path")
pub const ImportNode = struct {
    name: []const u8,
    path: []const u8,
    span: SourceSpan,
};

/// Contract Declaration
pub const ContractNode = struct {
    name: []const u8,
    body: []AstNode,
    span: SourceSpan,
};

/// Function Declaration
pub const FunctionNode = struct {
    pub_: bool, // Whether `pub` is used
    name: []const u8,
    parameters: []ParamNode,
    return_type: ?TypeRef,
    requires_clauses: []ExprNode, // Preconditions
    ensures_clauses: []ExprNode, // Postconditions
    body: BlockNode,
    span: SourceSpan,
};

pub const ParamNode = struct {
    name: []const u8,
    typ: TypeRef,
    span: SourceSpan,
};

/// Enhanced type system to match Ora spec
pub const TypeRef = union(enum) {
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
    Slice: *TypeRef,
    Mapping: MappingType,
    DoubleMap: DoubleMapType,
    Identifier: []const u8, // For custom types (structs, enums)
    Tuple: TupleType, // Tuple types

    // Error handling types
    ErrorUnion: ErrorUnionType, // !T syntax
    Result: ResultType, // Result[T, E] syntax

    // Special types
    Unknown: void, // For type inference
};

/// Error union type (!T)
pub const ErrorUnionType = struct {
    success_type: *TypeRef,
};

/// Result type (Result[T, E])
pub const ResultType = struct {
    ok_type: *TypeRef,
    error_type: *TypeRef,
};

/// Tuple type for multiple values
pub const TupleType = struct {
    types: []TypeRef,
};

pub const MappingType = struct {
    key: *TypeRef,
    value: *TypeRef,
};

pub const DoubleMapType = struct {
    key1: *TypeRef,
    key2: *TypeRef,
    value: *TypeRef,
};

/// Variable Declaration with Ora's memory model
pub const VariableDeclNode = struct {
    name: []const u8,
    region: MemoryRegion,
    mutable: bool, // true for `var`, false for `let`
    locked: bool, // true if @lock annotation is present
    typ: TypeRef,
    value: ?ExprNode,
    span: SourceSpan,
    // Tuple unpacking support
    tuple_names: ?[][]const u8, // For tuple unpacking: let (a, b) = expr
};

/// Memory regions matching Ora specification
pub const MemoryRegion = enum {
    Stack, // let/var (default)
    Memory, // memory let/memory var
    Storage, // storage let/storage var
    TStore, // tstore let/tstore var
    Const, // const (compile-time)
    Immutable, // immutable (deploy-time)
};

/// Struct Declaration
pub const StructDeclNode = struct {
    name: []const u8,
    fields: []StructField,
    span: SourceSpan,
};

pub const StructField = struct {
    name: []const u8,
    typ: TypeRef,
    span: SourceSpan,
};

/// Enum Declaration
pub const EnumDeclNode = struct {
    name: []const u8,
    variants: []EnumVariant,
    span: SourceSpan,
};

pub const EnumVariant = struct {
    name: []const u8,
    span: SourceSpan,
};

/// Log Declaration
pub const LogDeclNode = struct {
    name: []const u8,
    fields: []LogField,
    span: SourceSpan,
};

pub const LogField = struct {
    name: []const u8,
    typ: TypeRef,
    span: SourceSpan,
};

/// Block
pub const BlockNode = struct {
    statements: []StmtNode,
    span: SourceSpan,
};

/// Statement
pub const StmtNode = union(enum) {
    Expr: ExprNode,
    VariableDecl: VariableDeclNode,
    Return: ReturnNode,
    If: IfNode,
    While: WhileNode,
    Break: SourceSpan,
    Continue: SourceSpan,
    Log: LogNode,
    Lock: LockNode, // @lock annotations
    Invariant: InvariantNode, // Loop invariants
    Requires: RequiresNode,
    Ensures: EnsuresNode,

    // Error handling statements
    ErrorDecl: ErrorDeclNode, // error MyError;
    TryBlock: TryBlockNode, // try { ... } catch { ... }
};

pub const ReturnNode = struct {
    value: ?ExprNode,
    span: SourceSpan,
};

pub const IfNode = struct {
    condition: ExprNode,
    then_branch: BlockNode,
    else_branch: ?BlockNode,
    span: SourceSpan,
};

pub const WhileNode = struct {
    condition: ExprNode,
    body: BlockNode,
    invariants: []ExprNode, // Loop invariants
    span: SourceSpan,
};

pub const LogNode = struct {
    event_name: []const u8,
    args: []ExprNode,
    span: SourceSpan,
};

pub const InvariantNode = struct {
    condition: ExprNode,
    span: SourceSpan,
};

pub const RequiresNode = struct {
    condition: ExprNode,
    span: SourceSpan,
};

pub const EnsuresNode = struct {
    condition: ExprNode,
    span: SourceSpan,
};

/// Error Declaration
pub const ErrorDeclNode = struct {
    name: []const u8,
    span: SourceSpan,
};

/// Try-Catch Block
pub const TryBlockNode = struct {
    try_block: BlockNode,
    catch_block: ?CatchBlock,
    span: SourceSpan,
};

pub const CatchBlock = struct {
    error_variable: ?[]const u8, // Optional: catch(e) { ... }
    block: BlockNode,
    span: SourceSpan,
};

pub const LockNode = struct {
    path: ExprNode, // e.g., balances[to]
    span: SourceSpan,
};

/// Expression
pub const ExprNode = union(enum) {
    Identifier: IdentifierExpr,
    Literal: LiteralNode,
    Binary: BinaryExpr,
    Unary: UnaryExpr,
    Assignment: AssignmentExpr,
    CompoundAssignment: CompoundAssignmentExpr,
    Call: CallExpr,
    Index: IndexExpr,
    FieldAccess: FieldAccessExpr,
    Cast: CastExpr,
    Comptime: ComptimeExpr,
    Old: OldExpr, // old() expressions in ensures clauses
    Tuple: TupleExpr, // tuple expressions (a, b, c)

    // Error handling expressions
    Try: TryExpr, // try expression
    ErrorReturn: ErrorReturnExpr, // error.SomeError
    ErrorCast: ErrorCastExpr, // value as !T

    // Shift operations
    Shift: ShiftExpr, // mapping from source -> dest : amount
};

pub const IdentifierExpr = struct {
    name: []const u8,
    span: SourceSpan,
};

pub const LiteralNode = union(enum) {
    Integer: IntegerLiteral,
    String: StringLiteral,
    Bool: BoolLiteral,
    Address: AddressLiteral,
    Hex: HexLiteral,
};

pub const IntegerLiteral = struct {
    value: []const u8, // Store as string to handle large numbers
    span: SourceSpan,
};

pub const StringLiteral = struct {
    value: []const u8,
    span: SourceSpan,
};

pub const BoolLiteral = struct {
    value: bool,
    span: SourceSpan,
};

pub const AddressLiteral = struct {
    value: []const u8,
    span: SourceSpan,
};

pub const HexLiteral = struct {
    value: []const u8,
    span: SourceSpan,
};

pub const BinaryExpr = struct {
    lhs: *ExprNode,
    operator: BinaryOp,
    rhs: *ExprNode,
    span: SourceSpan,
};

/// Extended binary operators for Ora
pub const BinaryOp = enum {
    // Arithmetic
    Plus,
    Minus,
    Star,
    Slash,
    Percent,

    // Comparison
    EqualEqual,
    BangEqual,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,

    // Logical
    And, // &&
    Or, // ||

    // Bitwise
    BitAnd, // &
    BitOr, // |
    BitXor, // ^
    ShiftLeft, // <<
    ShiftRight, // >>
};

pub const UnaryExpr = struct {
    operator: UnaryOp,
    operand: *ExprNode,
    span: SourceSpan,
};

pub const UnaryOp = enum {
    Minus, // -x
    Bang, // !x
    BitNot, // ~x (if supported)
};

pub const AssignmentExpr = struct {
    target: *ExprNode,
    value: *ExprNode,
    span: SourceSpan,
};

/// Compound assignment operators (+=, -=, etc.)
pub const CompoundAssignmentExpr = struct {
    target: *ExprNode,
    operator: CompoundAssignmentOp,
    value: *ExprNode,
    span: SourceSpan,
};

pub const CompoundAssignmentOp = enum {
    PlusEqual, // +=
    MinusEqual, // -=
    StarEqual, // *=
    SlashEqual, // /=
    PercentEqual, // %=
};

pub const CallExpr = struct {
    callee: *ExprNode,
    arguments: []ExprNode,
    span: SourceSpan,
};

pub const IndexExpr = struct {
    target: *ExprNode,
    index: *ExprNode,
    span: SourceSpan,
};

pub const FieldAccessExpr = struct {
    target: *ExprNode,
    field: []const u8,
    span: SourceSpan,
};

/// Casting expressions (as, as?, as!)
pub const CastExpr = struct {
    operand: *ExprNode,
    target_type: TypeRef,
    cast_type: CastType,
    span: SourceSpan,
};

pub const CastType = enum {
    Unsafe, // as
    Safe, // as?
    Forced, // as!
};

/// Compile-time expressions
pub const ComptimeExpr = struct {
    block: BlockNode,
    span: SourceSpan,
};

/// old() expressions for postconditions
pub const OldExpr = struct {
    expr: *ExprNode,
    span: SourceSpan,
};

/// Tuple expressions (a, b, c)
pub const TupleExpr = struct {
    elements: []ExprNode,
    span: SourceSpan,
};

/// Try expression
pub const TryExpr = struct {
    expr: *ExprNode,
    span: SourceSpan,
};

/// Error return expression
pub const ErrorReturnExpr = struct {
    error_name: []const u8,
    span: SourceSpan,
};

/// Error cast expression
pub const ErrorCastExpr = struct {
    operand: *ExprNode,
    target_type: TypeRef,
    span: SourceSpan,
};

/// Shift expression (mapping from source -> dest : amount)
pub const ShiftExpr = struct {
    mapping: *ExprNode, // The mapping being modified (e.g., balances)
    source: *ExprNode, // Source expression (e.g., std.transaction.sender)
    dest: *ExprNode, // Destination expression (e.g., to)
    amount: *ExprNode, // Amount expression (e.g., amount)
    span: SourceSpan,
};

/// Visitor pattern for AST traversal
pub const AstVisitor = struct {
    const Self = @This();

    // Visit functions for each node type
    visitContractFn: ?fn (self: *Self, node: *ContractNode) void = null,
    visitFunctionFn: ?fn (self: *Self, node: *FunctionNode) void = null,
    visitVariableDeclFn: ?fn (self: *Self, node: *VariableDeclNode) void = null,
    visitExprFn: ?fn (self: *Self, node: *ExprNode) void = null,
    // TODO: Add visitor functions for: StructDecl, EnumDecl, LogDecl, Import, ErrorDecl, Block, Statement, TryBlock

    pub fn visit(self: *Self, node: *AstNode) void {
        switch (node.*) {
            .Contract => |*contract| {
                if (self.visitContractFn) |visitFn| {
                    visitFn(self, contract);
                }
            },
            .Function => |*function| {
                if (self.visitFunctionFn) |visitFn| {
                    visitFn(self, function);
                }
            },
            .VariableDecl => |*var_decl| {
                if (self.visitVariableDeclFn) |visitFn| {
                    visitFn(self, var_decl);
                }
            },
            .Expression => |*expr| {
                if (self.visitExprFn) |visitFn| {
                    visitFn(self, expr);
                }
            },
            // TODO: Add visitor calls for: StructDecl, EnumDecl, LogDecl, Import, ErrorDecl, Block, Statement, TryBlock
            else => {},
        }
    }
};

/// Helper functions for AST construction
pub fn createIdentifier(allocator: std.mem.Allocator, name: []const u8, span: SourceSpan) !*ExprNode {
    const node = try allocator.create(ExprNode);
    node.* = .{ .Identifier = .{ .name = name, .span = span } };
    return node;
}

pub fn createBinaryExpr(allocator: std.mem.Allocator, lhs: *ExprNode, op: BinaryOp, rhs: *ExprNode, span: SourceSpan) !*ExprNode {
    const node = try allocator.create(ExprNode);
    node.* = .{ .Binary = .{ .lhs = lhs, .operator = op, .rhs = rhs, .span = span } };
    return node;
}

/// Pretty printer for debugging
pub fn printAst(node: *AstNode, writer: anytype, indent: u32) !void {
    const indent_str = "  " ** indent;

    switch (node.*) {
        .Contract => |contract| {
            try writer.print("{s}Contract: {s}\n", .{ indent_str, contract.name });
            for (contract.body) |*child| {
                try printAst(child, writer, indent + 1);
            }
        },
        .Function => |function| {
            try writer.print("{s}Function: {s} (pub: {})\n", .{ indent_str, function.name, function.pub_ });
        },
        .VariableDecl => |var_decl| {
            try writer.print("{s}Variable: {s} ({s}, mutable: {}, locked: {})\n", .{ indent_str, var_decl.name, @tagName(var_decl.region), var_decl.mutable, var_decl.locked });
        },
        // TODO: Add pretty printing for: StructDecl, EnumDecl, LogDecl, Import, ErrorDecl, Block, Statement, TryBlock
        else => {
            try writer.print("{s}{s}\n", .{ indent_str, @tagName(node.*) });
        },
    }
}

/// Recursively free an AST node and all its children
pub fn deinitAstNode(allocator: std.mem.Allocator, node: *AstNode) void {
    switch (node.*) {
        .Contract => |*contract| {
            // Free all contract members
            for (contract.body) |*member| {
                deinitAstNode(allocator, member);
            }
            allocator.free(contract.body);
        },
        .Function => |*function| {
            // Free parameters
            for (function.parameters) |*param| {
                deinitParamNode(allocator, param);
            }
            allocator.free(function.parameters);

            // Free return type if present
            if (function.return_type) |*ret_type| {
                deinitTypeRef(allocator, ret_type);
            }

            // Free requires/ensures clauses
            for (function.requires_clauses) |*clause| {
                deinitExprNode(allocator, clause);
            }
            allocator.free(function.requires_clauses);

            for (function.ensures_clauses) |*clause| {
                deinitExprNode(allocator, clause);
            }
            allocator.free(function.ensures_clauses);

            // Free function body
            deinitBlockNode(allocator, &function.body);
        },
        .VariableDecl => |*var_decl| {
            deinitTypeRef(allocator, &var_decl.typ);
            if (var_decl.value) |*value| {
                deinitExprNode(allocator, value);
            }
        },
        .LogDecl => |*log_decl| {
            for (log_decl.fields) |*field| {
                deinitTypeRef(allocator, &field.typ);
            }
            allocator.free(log_decl.fields);
        },
        .Expression => |*expr| {
            deinitExprNode(allocator, expr);
        },
        .Statement => |*stmt| {
            deinitStmtNode(allocator, stmt);
        },
        .Block => |*block| {
            deinitBlockNode(allocator, block);
        },
        .ErrorDecl => |*error_decl| {
            // Note: error_decl.name is a string literal from the parser,
            // not allocated memory, so we don't free it
            _ = error_decl;
        },
        .TryBlock => |*try_block| {
            deinitBlockNode(allocator, &try_block.try_block);
            if (try_block.catch_block) |*catch_block| {
                deinitBlockNode(allocator, &catch_block.block);
                if (catch_block.error_variable) |name| {
                    allocator.free(name);
                }
            }
        },
        else => {
            // TODO: Add deinitialization for: StructDecl, EnumDecl, Import
        },
    }
}

/// Free a parameter node
pub fn deinitParamNode(allocator: std.mem.Allocator, param: *ParamNode) void {
    deinitTypeRef(allocator, &param.typ);
}

/// Free a type reference
pub fn deinitTypeRef(allocator: std.mem.Allocator, type_ref: *TypeRef) void {
    switch (type_ref.*) {
        .Slice => |elem_type| {
            deinitTypeRef(allocator, elem_type);
            allocator.destroy(elem_type);
        },
        .Mapping => |mapping| {
            deinitTypeRef(allocator, mapping.key);
            deinitTypeRef(allocator, mapping.value);
            allocator.destroy(mapping.key);
            allocator.destroy(mapping.value);
        },
        .DoubleMap => |doublemap| {
            deinitTypeRef(allocator, doublemap.key1);
            deinitTypeRef(allocator, doublemap.key2);
            deinitTypeRef(allocator, doublemap.value);
            allocator.destroy(doublemap.key1);
            allocator.destroy(doublemap.key2);
            allocator.destroy(doublemap.value);
        },
        .ErrorUnion => |error_union| {
            deinitTypeRef(allocator, error_union.success_type);
            allocator.destroy(error_union.success_type);
        },
        .Result => |result| {
            deinitTypeRef(allocator, result.ok_type);
            deinitTypeRef(allocator, result.error_type);
            allocator.destroy(result.ok_type);
            allocator.destroy(result.error_type);
        },
        else => {
            // Primitive types don't need cleanup
        },
    }
}

/// Free an expression node
pub fn deinitExprNode(allocator: std.mem.Allocator, expr: *ExprNode) void {
    switch (expr.*) {
        .Binary => |*binary| {
            deinitExprNode(allocator, binary.lhs);
            deinitExprNode(allocator, binary.rhs);
            allocator.destroy(binary.lhs);
            allocator.destroy(binary.rhs);
        },
        .Unary => |*unary| {
            deinitExprNode(allocator, unary.operand);
            allocator.destroy(unary.operand);
        },
        .Assignment => |*assign| {
            deinitExprNode(allocator, assign.target);
            deinitExprNode(allocator, assign.value);
            allocator.destroy(assign.target);
            allocator.destroy(assign.value);
        },
        .CompoundAssignment => |*comp_assign| {
            deinitExprNode(allocator, comp_assign.target);
            deinitExprNode(allocator, comp_assign.value);
            allocator.destroy(comp_assign.target);
            allocator.destroy(comp_assign.value);
        },
        .Call => |*call| {
            deinitExprNode(allocator, call.callee);
            allocator.destroy(call.callee);
            for (call.arguments) |*arg| {
                deinitExprNode(allocator, arg);
            }
            allocator.free(call.arguments);
        },
        .Index => |*index| {
            deinitExprNode(allocator, index.target);
            deinitExprNode(allocator, index.index);
            allocator.destroy(index.target);
            allocator.destroy(index.index);
        },
        .FieldAccess => |*field_access| {
            deinitExprNode(allocator, field_access.target);
            allocator.destroy(field_access.target);
        },
        .Cast => |*cast| {
            deinitExprNode(allocator, cast.operand);
            allocator.destroy(cast.operand);
            deinitTypeRef(allocator, &cast.target_type);
        },
        .Comptime => |*comptime_expr| {
            deinitBlockNode(allocator, &comptime_expr.block);
        },
        .Old => |*old| {
            deinitExprNode(allocator, old.expr);
            allocator.destroy(old.expr);
        },
        .Try => |*try_expr| {
            deinitExprNode(allocator, try_expr.expr);
            allocator.destroy(try_expr.expr);
        },
        .ErrorReturn => |*error_return| {
            // Note: error_return.error_name is a string literal from the parser,
            // not allocated memory, so we don't free it
            _ = error_return;
        },
        .ErrorCast => |*error_cast| {
            deinitExprNode(allocator, error_cast.operand);
            allocator.destroy(error_cast.operand);
            deinitTypeRef(allocator, &error_cast.target_type);
        },
        .Shift => |*shift| {
            deinitExprNode(allocator, shift.mapping);
            deinitExprNode(allocator, shift.source);
            deinitExprNode(allocator, shift.dest);
            deinitExprNode(allocator, shift.amount);
            allocator.destroy(shift.mapping);
            allocator.destroy(shift.source);
            allocator.destroy(shift.dest);
            allocator.destroy(shift.amount);
        },
        else => {
            // Literals and identifiers don't need cleanup
        },
    }
}

/// Free a statement node
pub fn deinitStmtNode(allocator: std.mem.Allocator, stmt: *StmtNode) void {
    switch (stmt.*) {
        .Expr => |*expr| {
            deinitExprNode(allocator, expr);
        },
        .VariableDecl => |*var_decl| {
            deinitTypeRef(allocator, &var_decl.typ);
            if (var_decl.value) |*value| {
                deinitExprNode(allocator, value);
            }
        },
        .Return => |*ret| {
            if (ret.value) |*value| {
                deinitExprNode(allocator, value);
            }
        },
        .If => |*if_stmt| {
            deinitExprNode(allocator, &if_stmt.condition);
            deinitBlockNode(allocator, &if_stmt.then_branch);
            if (if_stmt.else_branch) |*else_branch| {
                deinitBlockNode(allocator, else_branch);
            }
        },
        .While => |*while_stmt| {
            deinitExprNode(allocator, &while_stmt.condition);
            deinitBlockNode(allocator, &while_stmt.body);
            for (while_stmt.invariants) |*inv| {
                deinitExprNode(allocator, inv);
            }
            allocator.free(while_stmt.invariants);
        },
        .Log => |*log| {
            for (log.args) |*arg| {
                deinitExprNode(allocator, arg);
            }
            allocator.free(log.args);
        },
        .Lock => |*lock| {
            deinitExprNode(allocator, &lock.path);
        },
        .Invariant => |*inv| {
            deinitExprNode(allocator, &inv.condition);
        },
        .Requires => |*req| {
            deinitExprNode(allocator, &req.condition);
        },
        .Ensures => |*ens| {
            deinitExprNode(allocator, &ens.condition);
        },
        .ErrorDecl => |*error_decl| {
            // Note: error_decl.name is a string literal from the parser,
            // not allocated memory, so we don't free it
            _ = error_decl;
        },
        .TryBlock => |*try_block| {
            deinitBlockNode(allocator, &try_block.try_block);
            if (try_block.catch_block) |*catch_block| {
                deinitBlockNode(allocator, &catch_block.block);
                if (catch_block.error_variable) |name| {
                    allocator.free(name);
                }
            }
        },
        else => {
            // Break and Continue don't need cleanup
        },
    }
}

/// Free a block node
fn deinitBlockNode(allocator: std.mem.Allocator, block: *BlockNode) void {
    for (block.statements) |*stmt| {
        deinitStmtNode(allocator, stmt);
    }
    allocator.free(block.statements);
}

/// JSON serialization for AST
pub const ASTSerializer = struct {
    const SerializationError = std.fmt.BufPrintError ||
        std.fs.File.Writer.Error ||
        std.ArrayList(u8).Writer.Error ||
        anyerror;

    pub fn serializeAST(nodes: []AstNode, writer: anytype) SerializationError!void {
        try writer.writeAll("{\n");
        try writer.writeAll("  \"type\": \"AST\",\n");
        try writer.writeAll("  \"nodes\": [\n");

        for (nodes, 0..) |*node, i| {
            if (i > 0) try writer.writeAll(",\n");
            try serializeAstNode(node, writer, 2);
        }

        try writer.writeAll("\n  ]\n");
        try writer.writeAll("}\n");
    }

    fn serializeAstNode(node: *AstNode, writer: anytype, indent: u32) SerializationError!void {
        try writeIndent(writer, indent);
        try writer.writeAll("{\n");

        switch (node.*) {
            .Contract => |*contract| {
                try writeIndent(writer, indent + 1);
                try writer.writeAll("\"type\": \"Contract\",\n");
                try writeIndent(writer, indent + 1);
                try writer.print("\"name\": \"{s}\",\n", .{contract.name});
                try writeIndent(writer, indent + 1);
                try writer.writeAll("\"members\": [\n");
                for (contract.body, 0..) |*member, i| {
                    if (i > 0) try writer.writeAll(",\n");
                    try serializeAstNode(member, writer, indent + 2);
                }
                try writer.writeAll("\n");
                try writeIndent(writer, indent + 1);
                try writer.writeAll("\"]\n");
            },
            .Function => |*function| {
                try writeIndent(writer, indent + 1);
                try writer.writeAll("\"type\": \"Function\",\n");
                try writeIndent(writer, indent + 1);
                try writer.print("\"name\": \"{s}\",\n", .{function.name});
                try writeIndent(writer, indent + 1);
                try writer.print("\"public\": {},\n", .{function.pub_});
                try writeIndent(writer, indent + 1);
                try writer.writeAll("\"parameters\": [\n");
                for (function.parameters, 0..) |*param, i| {
                    if (i > 0) try writer.writeAll(",\n");
                    try serializeFunctionParameter(param, writer, indent + 2);
                }
                try writer.writeAll("\n");
                try writeIndent(writer, indent + 1);
                try writer.writeAll("],\n");
                try writeIndent(writer, indent + 1);
                try writer.writeAll("\"return_type\": ");
                if (function.return_type) |*ret_type| {
                    try serializeTypeRef(ret_type, writer);
                } else {
                    try writer.writeAll("null");
                }
                try writer.writeAll(",\n");
                try writeIndent(writer, indent + 1);
                try writer.print("\"requires_count\": {},\n", .{function.requires_clauses.len});
                try writeIndent(writer, indent + 1);
                try writer.print("\"ensures_count\": {}\n", .{function.ensures_clauses.len});
            },
            .VariableDecl => |*var_decl| {
                try writeIndent(writer, indent + 1);
                try writer.writeAll("\"type\": \"VariableDecl\",\n");
                try writeIndent(writer, indent + 1);
                try writer.print("\"name\": \"{s}\",\n", .{var_decl.name});
                try writeIndent(writer, indent + 1);
                try writer.print("\"region\": \"{s}\",\n", .{@tagName(var_decl.region)});
                try writeIndent(writer, indent + 1);
                try writer.print("\"mutable\": {},\n", .{var_decl.mutable});
                try writeIndent(writer, indent + 1);
                try writer.print("\"locked\": {},\n", .{var_decl.locked});
                try writeIndent(writer, indent + 1);
                try writer.writeAll("\"var_type\": ");
                try serializeTypeRef(&var_decl.typ, writer);
                try writer.writeAll(",\n");
                try writeIndent(writer, indent + 1);
                try writer.print("\"has_initializer\": {}\n", .{var_decl.value != null});
            },
            .LogDecl => |*log_decl| {
                try writeIndent(writer, indent + 1);
                try writer.writeAll("\"type\": \"LogDecl\",\n");
                try writeIndent(writer, indent + 1);
                try writer.print("\"name\": \"{s}\",\n", .{log_decl.name});
                try writeIndent(writer, indent + 1);
                try writer.writeAll("\"fields\": [\n");
                for (log_decl.fields, 0..) |*field, i| {
                    if (i > 0) try writer.writeAll(",\n");
                    try serializeLogField(field, writer, indent + 2);
                }
                try writer.writeAll("\n");
                try writeIndent(writer, indent + 1);
                try writer.writeAll("\"]\n");
            },
            .ErrorDecl => |*error_decl| {
                try writeIndent(writer, indent + 1);
                try writer.writeAll("\"type\": \"ErrorDecl\",\n");
                try writeIndent(writer, indent + 1);
                try writer.print("\"name\": \"{s}\",\n", .{error_decl.name});
                try writeIndent(writer, indent + 1);
                try writer.writeAll("\"span\": ");
                try serializeSourceSpan(error_decl.span, writer);
                try writer.writeAll("\n");
            },
            .TryBlock => |*try_block| {
                try writeIndent(writer, indent + 1);
                try writer.writeAll("\"type\": \"TryBlock\",\n");
                try writeIndent(writer, indent + 1);
                try writer.writeAll("\"try_block\": ");
                try serializeBlockNode(&try_block.try_block, writer, indent + 2);
                try writer.writeAll(",\n");
                try writeIndent(writer, indent + 1);
                try writer.writeAll("\"catch_block\": ");
                if (try_block.catch_block) |*catch_block| {
                    try serializeCatchBlock(catch_block, writer, indent + 2);
                } else {
                    try writer.writeAll("null");
                }
                try writer.writeAll("\n");
            },
            else => {
                try writeIndent(writer, indent + 1);
                try writer.print("\"type\": \"{s}\"\n", .{@tagName(node.*)});
            },
        }

        try writeIndent(writer, indent);
        try writer.writeAll("}");
    }

    fn serializeFunctionParameter(param: *ParamNode, writer: anytype, indent: u32) SerializationError!void {
        try writeIndent(writer, indent);
        try writer.writeAll("{\n");
        try writeIndent(writer, indent + 1);
        try writer.print("\"name\": \"{s}\",\n", .{param.name});
        try writeIndent(writer, indent + 1);
        try writer.writeAll("\"type\": ");
        try serializeTypeRef(&param.typ, writer);
        try writer.writeAll("\n");
        try writeIndent(writer, indent);
        try writer.writeAll("}");
    }

    fn serializeLogField(field: *LogField, writer: anytype, indent: u32) SerializationError!void {
        try writeIndent(writer, indent);
        try writer.writeAll("{\n");
        try writeIndent(writer, indent + 1);
        try writer.print("\"name\": \"{s}\",\n", .{field.name});
        try writeIndent(writer, indent + 1);
        try writer.writeAll("\"type\": ");
        try serializeTypeRef(&field.typ, writer);
        try writer.writeAll("\n");
        try writeIndent(writer, indent);
        try writer.writeAll("}");
    }

    fn serializeTypeRef(typ: *TypeRef, writer: anytype) SerializationError!void {
        switch (typ.*) {
            .Bool => try writer.writeAll("\"bool\""),
            .Address => try writer.writeAll("\"address\""),
            .U8 => try writer.writeAll("\"u8\""),
            .U16 => try writer.writeAll("\"u16\""),
            .U32 => try writer.writeAll("\"u32\""),
            .U64 => try writer.writeAll("\"u64\""),
            .U128 => try writer.writeAll("\"u128\""),
            .U256 => try writer.writeAll("\"u256\""),
            .I8 => try writer.writeAll("\"i8\""),
            .I16 => try writer.writeAll("\"i16\""),
            .I32 => try writer.writeAll("\"i32\""),
            .I64 => try writer.writeAll("\"i64\""),
            .I128 => try writer.writeAll("\"i128\""),
            .I256 => try writer.writeAll("\"i256\""),
            .String => try writer.writeAll("\"string\""),
            .Bytes => try writer.writeAll("\"bytes\""),
            .Slice => |slice_element_type| {
                try writer.writeAll("{\"type\": \"slice\", \"element\": ");
                try serializeTypeRef(slice_element_type, writer);
                try writer.writeAll("}");
            },
            .Mapping => |*mapping| {
                try writer.writeAll("{\"type\": \"mapping\", \"key\": ");
                try serializeTypeRef(mapping.key, writer);
                try writer.writeAll(", \"value\": ");
                try serializeTypeRef(mapping.value, writer);
                try writer.writeAll("}");
            },
            .DoubleMap => |*double_map| {
                try writer.writeAll("{\"type\": \"doublemap\", \"key1\": ");
                try serializeTypeRef(double_map.key1, writer);
                try writer.writeAll(", \"key2\": ");
                try serializeTypeRef(double_map.key2, writer);
                try writer.writeAll(", \"value\": ");
                try serializeTypeRef(double_map.value, writer);
                try writer.writeAll("}");
            },
            .Identifier => |name| {
                try writer.writeAll("{\"type\": \"identifier\", \"name\": \"");
                try writer.writeAll(name);
                try writer.writeAll("\"}");
            },
            .ErrorUnion => |error_union| {
                try writer.writeAll("{\"type\": \"error_union\", \"success_type\": ");
                try serializeTypeRef(error_union.success_type, writer);
                try writer.writeAll("}");
            },
            .Result => |result| {
                try writer.writeAll("{\"type\": \"result\", \"ok_type\": ");
                try serializeTypeRef(result.ok_type, writer);
                try writer.writeAll(", \"error_type\": ");
                try serializeTypeRef(result.error_type, writer);
                try writer.writeAll("}");
            },
        }
    }

    fn serializeSourceSpan(span: SourceSpan, writer: anytype) SerializationError!void {
        try writer.writeAll("{\"line\": ");
        try writer.print("{}, \"column\": {}, \"length\": {}\n", .{ span.line, span.column, span.length });
    }

    fn serializeBlockNode(block: *BlockNode, writer: anytype, indent: u32) SerializationError!void {
        try writeIndent(writer, indent);
        try writer.writeAll("{\n");
        try writeIndent(writer, indent + 1);
        try writer.writeAll("\"statements\": [\n");
        for (block.statements, 0..) |*stmt, i| {
            if (i > 0) try writer.writeAll(",\n");
            try serializeStmtNode(stmt, writer, indent + 2);
        }
        try writer.writeAll("\n");
        try writeIndent(writer, indent + 1);
        try writer.writeAll("],\n");
        try writeIndent(writer, indent + 1);
        try writer.writeAll("\"span\": ");
        try serializeSourceSpan(block.span, writer);
        try writer.writeAll("\n");
        try writeIndent(writer, indent);
        try writer.writeAll("}");
    }

    fn serializeCatchBlock(catch_block: *CatchBlock, writer: anytype, indent: u32) SerializationError!void {
        try writeIndent(writer, indent);
        try writer.writeAll("{\n");
        try writeIndent(writer, indent + 1);
        try writer.writeAll("\"error_variable\": ");
        if (catch_block.error_variable) |name| {
            try writer.writeAll("\"");
            try writer.writeAll(name);
            try writer.writeAll("\"");
        } else {
            try writer.writeAll("null");
        }
        try writer.writeAll(",\n");
        try writeIndent(writer, indent + 1);
        try writer.writeAll("\"block\": ");
        try serializeBlockNode(&catch_block.block, writer, indent + 2);
        try writer.writeAll("\n");
        try writeIndent(writer, indent);
        try writer.writeAll("}");
    }

    fn serializeStmtNode(stmt: *StmtNode, writer: anytype, indent: u32) SerializationError!void {
        try writeIndent(writer, indent);
        try writer.writeAll("{\n");

        switch (stmt.*) {
            .Expr => |*expr| {
                try writeIndent(writer, indent + 1);
                try writer.writeAll("\"type\": \"Expr\",\n");
                try writeIndent(writer, indent + 1);
                try writer.writeAll("\"expr\": ");
                try serializeExprNode(expr, writer, indent + 2);
                try writer.writeAll("\n");
            },
            .VariableDecl => |*var_decl| {
                try writeIndent(writer, indent + 1);
                try writer.writeAll("\"type\": \"VariableDecl\",\n");
                try writeIndent(writer, indent + 1);
                try writer.print("\"name\": \"{s}\",\n", .{var_decl.name});
                try writeIndent(writer, indent + 1);
                try writer.print("\"region\": \"{s}\",\n", .{@tagName(var_decl.region)});
                try writeIndent(writer, indent + 1);
                try writer.print("\"mutable\": {},\n", .{var_decl.mutable});
                try writeIndent(writer, indent + 1);
                try writer.print("\"locked\": {},\n", .{var_decl.locked});
                try writeIndent(writer, indent + 1);
                try writer.writeAll("\"var_type\": ");
                try serializeTypeRef(&var_decl.typ, writer);
                try writer.writeAll(",\n");
                try writeIndent(writer, indent + 1);
                try writer.print("\"has_initializer\": {}\n", .{var_decl.value != null});
            },
            .Return => |*ret| {
                try writeIndent(writer, indent + 1);
                try writer.writeAll("\"type\": \"Return\",\n");
                try writeIndent(writer, indent + 1);
                try writer.writeAll("\"value\": ");
                if (ret.value) |*value| {
                    try serializeExprNode(value, writer, indent + 2);
                } else {
                    try writer.writeAll("null");
                }
                try writer.writeAll("\n");
            },
            .If => |*if_stmt| {
                try writeIndent(writer, indent + 1);
                try writer.writeAll("\"type\": \"If\",\n");
                try writeIndent(writer, indent + 1);
                try writer.writeAll("\"condition\": ");
                try serializeExprNode(&if_stmt.condition, writer, indent + 2);
                try writer.writeAll(",\n");
                try writeIndent(writer, indent + 1);
                try writer.writeAll("\"then_branch\": ");
                try serializeBlockNode(&if_stmt.then_branch, writer, indent + 2);
                try writer.writeAll(",\n");
                try writeIndent(writer, indent + 1);
                try writer.writeAll("\"else_branch\": ");
                if (if_stmt.else_branch) |*else_branch| {
                    try serializeBlockNode(else_branch, writer, indent + 2);
                } else {
                    try writer.writeAll("null");
                }
                try writer.writeAll("\n");
            },
            .While => |*while_stmt| {
                try writeIndent(writer, indent + 1);
                try writer.writeAll("\"type\": \"While\",\n");
                try writeIndent(writer, indent + 1);
                try writer.writeAll("\"condition\": ");
                try serializeExprNode(&while_stmt.condition, writer, indent + 2);
                try writer.writeAll(",\n");
                try writeIndent(writer, indent + 1);
                try writer.writeAll("\"body\": ");
                try serializeBlockNode(&while_stmt.body, writer, indent + 2);
                try writer.writeAll(",\n");
                try writeIndent(writer, indent + 1);
                try writer.writeAll("\"invariants\": [\n");
                for (while_stmt.invariants, 0..) |*inv, i| {
                    if (i > 0) try writer.writeAll(",\n");
                    try serializeExprNode(inv, writer, indent + 2);
                }
                try writer.writeAll("\n");
                try writeIndent(writer, indent + 1);
                try writer.writeAll("],\n");
            },
            .Log => |*log| {
                try writeIndent(writer, indent + 1);
                try writer.writeAll("\"type\": \"Log\",\n");
                try writeIndent(writer, indent + 1);
                try writer.print("\"event_name\": \"{s}\",\n", .{log.event_name});
                try writeIndent(writer, indent + 1);
                try writer.writeAll("\"args\": [\n");
                for (log.args, 0..) |*arg, i| {
                    if (i > 0) try writer.writeAll(",\n");
                    try serializeExprNode(arg, writer, indent + 2);
                }
                try writer.writeAll("\n");
                try writeIndent(writer, indent + 1);
                try writer.writeAll("],\n");
            },
            .Lock => |*lock| {
                try writeIndent(writer, indent + 1);
                try writer.writeAll("\"type\": \"Lock\",\n");
                try writeIndent(writer, indent + 1);
                try writer.writeAll("\"path\": ");
                try serializeExprNode(&lock.path, writer, indent + 2);
                try writer.writeAll("\n");
            },
            .Invariant => |*inv| {
                try writeIndent(writer, indent + 1);
                try writer.writeAll("\"type\": \"Invariant\",\n");
                try writeIndent(writer, indent + 1);
                try writer.writeAll("\"condition\": ");
                try serializeExprNode(&inv.condition, writer, indent + 2);
                try writer.writeAll("\n");
            },
            .Requires => |*req| {
                try writeIndent(writer, indent + 1);
                try writer.writeAll("\"type\": \"Requires\",\n");
                try writeIndent(writer, indent + 1);
                try writer.writeAll("\"condition\": ");
                try serializeExprNode(&req.condition, writer, indent + 2);
                try writer.writeAll("\n");
            },
            .Ensures => |*ens| {
                try writeIndent(writer, indent + 1);
                try writer.writeAll("\"type\": \"Ensures\",\n");
                try writeIndent(writer, indent + 1);
                try writer.writeAll("\"condition\": ");
                try serializeExprNode(&ens.condition, writer, indent + 2);
                try writer.writeAll("\n");
            },
            .ErrorDecl => |*error_decl| {
                try writeIndent(writer, indent + 1);
                try writer.writeAll("\"type\": \"ErrorDecl\",\n");
                try writeIndent(writer, indent + 1);
                try writer.print("\"name\": \"{s}\",\n", .{error_decl.name});
                try writeIndent(writer, indent + 1);
                try writer.writeAll("\"span\": ");
                try serializeSourceSpan(error_decl.span, writer);
                try writer.writeAll("\n");
            },
            .TryBlock => |*try_block| {
                try writeIndent(writer, indent + 1);
                try writer.writeAll("\"type\": \"TryBlock\",\n");
                try writeIndent(writer, indent + 1);
                try writer.writeAll("\"try_block\": ");
                try serializeBlockNode(&try_block.try_block, writer, indent + 2);
                try writer.writeAll(",\n");
                try writeIndent(writer, indent + 1);
                try writer.writeAll("\"catch_block\": ");
                if (try_block.catch_block) |*catch_block| {
                    try serializeCatchBlock(catch_block, writer, indent + 2);
                } else {
                    try writer.writeAll("null");
                }
                try writer.writeAll("\n");
            },
            .Break => |break_span| {
                try writeIndent(writer, indent + 1);
                try writer.writeAll("\"type\": \"Break\",\n");
                try writeIndent(writer, indent + 1);
                try writer.writeAll("\"span\": ");
                try serializeSourceSpan(break_span, writer);
                try writer.writeAll("\n");
            },
            .Continue => |continue_span| {
                try writeIndent(writer, indent + 1);
                try writer.writeAll("\"type\": \"Continue\",\n");
                try writeIndent(writer, indent + 1);
                try writer.writeAll("\"span\": ");
                try serializeSourceSpan(continue_span, writer);
                try writer.writeAll("\n");
            },
        }

        try writeIndent(writer, indent);
        try writer.writeAll("}");
    }

    fn serializeExprNode(expr: *ExprNode, writer: anytype, indent: u32) SerializationError!void {
        try writeIndent(writer, indent);
        try writer.writeAll("{\n");

        switch (expr.*) {
            .Binary => |*binary| {
                try writeIndent(writer, indent + 1);
                try writer.writeAll("\"type\": \"Binary\",\n");
                try writeIndent(writer, indent + 1);
                try writer.writeAll("\"lhs\": ");
                try serializeExprNode(binary.lhs, writer, indent + 2);
                try writer.writeAll(",\n");
                try writeIndent(writer, indent + 1);
                try writer.writeAll("\"operator\": \"");
                try writer.writeAll(@tagName(binary.operator));
                try writer.writeAll("\",\n");
                try writeIndent(writer, indent + 1);
                try writer.writeAll("\"rhs\": ");
                try serializeExprNode(binary.rhs, writer, indent + 2);
                try writer.writeAll("\n");
            },
            .Unary => |*unary| {
                try writeIndent(writer, indent + 1);
                try writer.writeAll("\"type\": \"Unary\",\n");
                try writeIndent(writer, indent + 1);
                try writer.writeAll("\"operator\": \"");
                try writer.writeAll(@tagName(unary.operator));
                try writer.writeAll("\",\n");
                try writeIndent(writer, indent + 1);
                try writer.writeAll("\"operand\": ");
                try serializeExprNode(unary.operand, writer, indent + 2);
                try writer.writeAll("\n");
            },
            .Assignment => |*assign| {
                try writeIndent(writer, indent + 1);
                try writer.writeAll("\"type\": \"Assignment\",\n");
                try writeIndent(writer, indent + 1);
                try writer.writeAll("\"target\": ");
                try serializeExprNode(assign.target, writer, indent + 2);
                try writer.writeAll(",\n");
                try writeIndent(writer, indent + 1);
                try writer.writeAll("\"value\": ");
                try serializeExprNode(assign.value, writer, indent + 2);
                try writer.writeAll("\n");
            },
            .CompoundAssignment => |*comp_assign| {
                try writeIndent(writer, indent + 1);
                try writer.writeAll("\"type\": \"CompoundAssignment\",\n");
                try writeIndent(writer, indent + 1);
                try writer.writeAll("\"target\": ");
                try serializeExprNode(comp_assign.target, writer, indent + 2);
                try writer.writeAll(",\n");
                try writeIndent(writer, indent + 1);
                try writer.writeAll("\"operator\": \"");
                try writer.writeAll(@tagName(comp_assign.operator));
                try writer.writeAll("\",\n");
                try writeIndent(writer, indent + 1);
                try writer.writeAll("\"value\": ");
                try serializeExprNode(comp_assign.value, writer, indent + 2);
                try writer.writeAll("\n");
            },
            .Call => |*call| {
                try writeIndent(writer, indent + 1);
                try writer.writeAll("\"type\": \"Call\",\n");
                try writeIndent(writer, indent + 1);
                try writer.writeAll("\"callee\": ");
                try serializeExprNode(call.callee, writer, indent + 2);
                try writer.writeAll(",\n");
                try writeIndent(writer, indent + 1);
                try writer.writeAll("\"arguments\": [\n");
                for (call.arguments, 0..) |*arg, i| {
                    if (i > 0) try writer.writeAll(",\n");
                    try serializeExprNode(arg, writer, indent + 2);
                }
                try writer.writeAll("\n");
                try writeIndent(writer, indent + 1);
                try writer.writeAll("]\n");
            },
            .Index => |*index| {
                try writeIndent(writer, indent + 1);
                try writer.writeAll("\"type\": \"Index\",\n");
                try writeIndent(writer, indent + 1);
                try writer.writeAll("\"target\": ");
                try serializeExprNode(index.target, writer, indent + 2);
                try writer.writeAll(",\n");
                try writeIndent(writer, indent + 1);
                try writer.writeAll("\"index\": ");
                try serializeExprNode(index.index, writer, indent + 2);
                try writer.writeAll("\n");
            },
            .FieldAccess => |*field_access| {
                try writeIndent(writer, indent + 1);
                try writer.writeAll("\"type\": \"FieldAccess\",\n");
                try writeIndent(writer, indent + 1);
                try writer.writeAll("\"target\": ");
                try serializeExprNode(field_access.target, writer, indent + 2);
                try writer.writeAll(",\n");
                try writeIndent(writer, indent + 1);
                try writer.writeAll("\"field\": \"");
                try writer.writeAll(field_access.field);
                try writer.writeAll("\"\n");
            },
            .Cast => |*cast| {
                try writeIndent(writer, indent + 1);
                try writer.writeAll("\"type\": \"Cast\",\n");
                try writeIndent(writer, indent + 1);
                try writer.writeAll("\"operand\": ");
                try serializeExprNode(cast.operand, writer, indent + 2);
                try writer.writeAll(",\n");
                try writeIndent(writer, indent + 1);
                try writer.writeAll("\"target_type\": ");
                try serializeTypeRef(&cast.target_type, writer);
                try writer.writeAll(",\n");
                try writeIndent(writer, indent + 1);
                try writer.writeAll("\"cast_type\": \"");
                try writer.writeAll(@tagName(cast.cast_type));
                try writer.writeAll("\"\n");
            },
            .Comptime => |*comptime_expr| {
                try writeIndent(writer, indent + 1);
                try writer.writeAll("\"type\": \"Comptime\",\n");
                try writeIndent(writer, indent + 1);
                try writer.writeAll("\"block\": ");
                try serializeBlockNode(&comptime_expr.block, writer, indent + 2);
                try writer.writeAll("\n");
            },
            .Old => |*old| {
                try writeIndent(writer, indent + 1);
                try writer.writeAll("\"type\": \"Old\",\n");
                try writeIndent(writer, indent + 1);
                try writer.writeAll("\"expr\": ");
                try serializeExprNode(old.expr, writer, indent + 2);
                try writer.writeAll("\n");
            },
            .Try => |*try_expr| {
                try writeIndent(writer, indent + 1);
                try writer.writeAll("\"type\": \"Try\",\n");
                try writeIndent(writer, indent + 1);
                try writer.writeAll("\"expr\": ");
                try serializeExprNode(try_expr.expr, writer, indent + 2);
                try writer.writeAll(",\n");
                try writeIndent(writer, indent + 1);
                try writer.writeAll("\"span\": ");
                try serializeSourceSpan(try_expr.span, writer);
                try writer.writeAll("\n");
            },
            .ErrorReturn => |*error_return| {
                try writeIndent(writer, indent + 1);
                try writer.writeAll("\"type\": \"ErrorReturn\",\n");
                try writeIndent(writer, indent + 1);
                try writer.print("\"error_name\": \"{s}\",\n", .{error_return.error_name});
                try writeIndent(writer, indent + 1);
                try writer.writeAll("\"span\": ");
                try serializeSourceSpan(error_return.span, writer);
                try writer.writeAll("\n");
            },
            .ErrorCast => |*error_cast| {
                try writeIndent(writer, indent + 1);
                try writer.writeAll("\"type\": \"ErrorCast\",\n");
                try writeIndent(writer, indent + 1);
                try writer.writeAll("\"operand\": ");
                try serializeExprNode(error_cast.operand, writer, indent + 2);
                try writer.writeAll(",\n");
                try writeIndent(writer, indent + 1);
                try writer.writeAll("\"target_type\": ");
                try serializeTypeRef(&error_cast.target_type, writer);
                try writer.writeAll(",\n");
                try writeIndent(writer, indent + 1);
                try writer.writeAll("\"span\": ");
                try serializeSourceSpan(error_cast.span, writer);
                try writer.writeAll("\n");
            },
            .Shift => |*shift| {
                try writeIndent(writer, indent + 1);
                try writer.writeAll("\"type\": \"Shift\",\n");
                try writeIndent(writer, indent + 1);
                try writer.writeAll("\"mapping\": ");
                try serializeExprNode(shift.mapping, writer, indent + 2);
                try writer.writeAll(",\n");
                try writeIndent(writer, indent + 1);
                try writer.writeAll("\"source\": ");
                try serializeExprNode(shift.source, writer, indent + 2);
                try writer.writeAll(",\n");
                try writeIndent(writer, indent + 1);
                try writer.writeAll("\"dest\": ");
                try serializeExprNode(shift.dest, writer, indent + 2);
                try writer.writeAll(",\n");
                try writeIndent(writer, indent + 1);
                try writer.writeAll("\"amount\": ");
                try serializeExprNode(shift.amount, writer, indent + 2);
                try writer.writeAll(",\n");
                try writeIndent(writer, indent + 1);
                try writer.writeAll("\"span\": ");
                try serializeSourceSpan(shift.span, writer);
                try writer.writeAll("\n");
            },
            .Identifier => |*ident| {
                try writeIndent(writer, indent + 1);
                try writer.writeAll("\"type\": \"Identifier\",\n");
                try writeIndent(writer, indent + 1);
                try writer.print("\"name\": \"{s}\",\n", .{ident.name});
                try writeIndent(writer, indent + 1);
                try writer.writeAll("\"span\": ");
                try serializeSourceSpan(ident.span, writer);
                try writer.writeAll("\n");
            },
            .Literal => |*literal| {
                try writeIndent(writer, indent + 1);
                try writer.writeAll("\"type\": \"Literal\",\n");
                try writeIndent(writer, indent + 1);
                try writer.writeAll("\"literal\": ");
                try serializeLiteralNode(literal, writer, indent + 2);
                try writer.writeAll("\n");
            },
        }

        try writeIndent(writer, indent);
        try writer.writeAll("}");
    }

    fn serializeLiteralNode(literal: *LiteralNode, writer: anytype, indent: u32) SerializationError!void {
        try writeIndent(writer, indent);
        try writer.writeAll("{\n");

        switch (literal.*) {
            .Integer => |*integer| {
                try writeIndent(writer, indent + 1);
                try writer.writeAll("\"type\": \"Integer\",\n");
                try writeIndent(writer, indent + 1);
                try writer.print("\"value\": \"{s}\",\n", .{integer.value});
                try writeIndent(writer, indent + 1);
                try writer.writeAll("\"span\": ");
                try serializeSourceSpan(integer.span, writer);
                try writer.writeAll("\n");
            },
            .String => |*string| {
                try writeIndent(writer, indent + 1);
                try writer.writeAll("\"type\": \"String\",\n");
                try writeIndent(writer, indent + 1);
                try writer.print("\"value\": \"{s}\",\n", .{string.value});
                try writeIndent(writer, indent + 1);
                try writer.writeAll("\"span\": ");
                try serializeSourceSpan(string.span, writer);
                try writer.writeAll("\n");
            },
            .Bool => |*bool_literal| {
                try writeIndent(writer, indent + 1);
                try writer.writeAll("\"type\": \"Bool\",\n");
                try writeIndent(writer, indent + 1);
                try writer.print("\"value\": {},\n", .{bool_literal.value});
                try writeIndent(writer, indent + 1);
                try writer.writeAll("\"span\": ");
                try serializeSourceSpan(bool_literal.span, writer);
                try writer.writeAll("\n");
            },
            .Address => |*address| {
                try writeIndent(writer, indent + 1);
                try writer.writeAll("\"type\": \"Address\",\n");
                try writeIndent(writer, indent + 1);
                try writer.print("\"value\": \"{s}\",\n", .{address.value});
                try writeIndent(writer, indent + 1);
                try writer.writeAll("\"span\": ");
                try serializeSourceSpan(address.span, writer);
                try writer.writeAll("\n");
            },
            .Hex => |*hex| {
                try writeIndent(writer, indent + 1);
                try writer.writeAll("\"type\": \"Hex\",\n");
                try writeIndent(writer, indent + 1);
                try writer.print("\"value\": \"{s}\",\n", .{hex.value});
                try writeIndent(writer, indent + 1);
                try writer.writeAll("\"span\": ");
                try serializeSourceSpan(hex.span, writer);
                try writer.writeAll("\n");
            },
        }

        try writeIndent(writer, indent);
        try writer.writeAll("}");
    }

    fn writeIndent(writer: anytype, indent: u32) SerializationError!void {
        var i: u32 = 0;
        while (i < indent) : (i += 1) {
            try writer.writeAll("  ");
        }
    }
};

/// Free an array of AST nodes
pub fn deinitAstNodes(allocator: std.mem.Allocator, nodes: []AstNode) void {
    for (nodes) |*node| {
        deinitAstNode(allocator, node);
    }
    allocator.free(nodes);
}

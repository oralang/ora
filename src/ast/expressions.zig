// ============================================================================
// AST Expressions
// ============================================================================
//
// expression node definitions for the Ora AST.
//
// ============================================================================

const std = @import("std");
const SourceSpan = @import("source_span.zig").SourceSpan;
const TypeInfo = @import("type_info.zig").TypeInfo;
const CommonTypes = @import("type_info.zig").CommonTypes;
const statements = @import("statements.zig");
const AstArena = @import("ast_arena.zig").AstArena;
const verification = @import("verification.zig");

/// Binary and unary operators
pub const BinaryOp = enum {
    // arithmetic
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    StarStar, // **

    // comparison
    EqualEqual,
    BangEqual,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,

    // logical
    And, // &&
    Or, // ||

    // bitwise
    BitwiseAnd, // &
    BitwiseOr, // |
    BitwiseXor, // ^
    LeftShift, // <<
    RightShift, // >>

    // comma operator
    Comma, // ,
};

pub const UnaryOp = enum {
    Minus, // -x
    Bang, // !x
    BitNot, // ~x (if supported)
};

pub const CompoundAssignmentOp = enum {
    PlusEqual, // +=
    MinusEqual, // -=
    StarEqual, // *=
    SlashEqual, // /=
    PercentEqual, // %=
};

pub const CastType = enum {
    Unsafe, // as
    Safe, // as?
    Forced, // as!
};

// Forward declaration for blocks
const BlockNode = statements.BlockNode;

/// Switch pattern matching (moved from statements.zig to break circular dependency)
pub const SwitchPattern = union(enum) {
    Literal: struct {
        value: LiteralExpr,
        span: SourceSpan,
    },
    Range: RangeExpr,
    EnumValue: struct {
        enum_name: []const u8,
        variant_name: []const u8,
        span: SourceSpan,
    },
    Else: SourceSpan,
};

/// Switch body types (moved from statements.zig to break circular dependency)
pub const SwitchBody = union(enum) {
    Expression: *ExprNode,
    Block: BlockNode,
    LabeledBlock: struct {
        label: []const u8,
        block: BlockNode,
        span: SourceSpan,
    },
};

/// Switch case for switch expressions (moved from statements.zig to break circular dependency)
pub const SwitchCase = struct {
    pattern: SwitchPattern,
    body: SwitchBody,
    span: SourceSpan,
};

/// Forward declaration for recursive expressions
pub const ExprNode = union(enum) {
    Identifier: IdentifierExpr, // identifier expressions (x)
    Literal: LiteralExpr, // literal expressions (1, "hello", true, etc.)
    Binary: BinaryExpr, // binary expressions (x + y)
    Unary: UnaryExpr, // unary expressions (-x, !x, ~x)
    Assignment: AssignmentExpr, // assignment expressions (x = y)
    CompoundAssignment: CompoundAssignmentExpr, // compound assignment expressions (x += y)
    Call: CallExpr, // function call expressions (f(x, y))
    Index: IndexExpr, // array/map indexing expressions (x[y])
    FieldAccess: FieldAccessExpr, // field access expressions (x.y)
    Cast: CastExpr, // casting expressions (x as T)
    Comptime: ComptimeExpr, // compile-time expressions (comptime { ... })
    Old: OldExpr, // old() expressions in ensures clauses
    Quantified: QuantifiedExpr, // quantified expressions (forall, exists)
    Tuple: TupleExpr, // tuple expressions (a, b, c)
    SwitchExpression: SwitchExprNode, // switch expressions (switch (x) { case y: ... default: ... })

    // error handling expressions
    Try: TryExpr, // try expression
    ErrorReturn: ErrorReturnExpr, // error.SomeError
    ErrorCast: ErrorCastExpr, // value as !T (value as !T)

    // shift operations
    Shift: ShiftExpr, // mapping from source -> dest : amount

    // struct instantiation
    StructInstantiation: StructInstantiationExpr, // StructName { field1: value1, field2: value2 }

    // anonymous struct literals
    AnonymousStruct: AnonymousStructExpr, // .{ field1: value1, field2: value2 }

    // range expressions
    Range: RangeExpr, // 1...1000 or 0..periods

    // labeled block expressions
    LabeledBlock: LabeledBlockExpr, // label: { ... break :label value; }

    // destructuring expressions
    Destructuring: DestructuringExpr, // let .{ balance, locked_until } = account

    // enum literal expressions
    EnumLiteral: EnumLiteralExpr, // EnumName.VariantName

    // array literal expressions
    ArrayLiteral: ArrayLiteralExpr, // [1, 2, 3] or []

    /// Check if this expression is specification-only (not compiled to bytecode)
    pub fn isSpecificationOnly(self: *const ExprNode) bool {
        return switch (self.*) {
            .Old => true,
            .Quantified => true,
            else => false,
        };
    }
};

/// Quantifier type for formal verification
pub const QuantifierType = enum {
    Forall, // forall
    Exists, // exists
};

/// Quantified expressions for formal verification (forall/exists)
/// Example: forall i: u256 where i < len => array[i] > 0
pub const QuantifiedExpr = struct {
    quantifier: QuantifierType, // forall | exists
    variable: []const u8, // bound variable name
    variable_type: TypeInfo, // type of bound variable
    condition: ?*ExprNode, // optional where clause
    body: *ExprNode, // the quantified expression
    span: SourceSpan,
    /// Metadata: Always specification-only
    is_specification: bool = true,
    /// Verification metadata for formal verification tools
    verification_metadata: ?*verification.QuantifiedMetadata = null,
    /// Verification attributes for this quantified expression
    verification_attributes: []verification.VerificationAttribute = &[_]verification.VerificationAttribute{},
};

/// Anonymous struct literal expression (.{ field1: value1, field2: value2 })
pub const AnonymousStructExpr = struct {
    fields: []AnonymousStructField,
    span: SourceSpan,
};

/// Field initializer for anonymous struct literals
pub const AnonymousStructField = struct {
    name: []const u8, // Field name
    value: *ExprNode, // Field value expression
    span: SourceSpan,
};

/// Range expression (e.g., 1...1000 or 0..periods)
pub const RangeExpr = struct {
    start: *ExprNode, // Start of range
    end: *ExprNode, // End of range
    inclusive: bool, // true for ..., false for ..
    span: SourceSpan,
    type_info: TypeInfo, // Type information
};

/// Labeled block expression (e.g., label: { ... break :label value; })
pub const LabeledBlockExpr = struct {
    label: []const u8, // Block label name
    block: BlockNode, // Block content
    span: SourceSpan,
};

/// Destructuring expression (e.g., let .{ balance, locked_until } = account)
pub const DestructuringExpr = struct {
    pattern: DestructuringPattern, // Pattern to match
    value: *ExprNode, // Expression to destructure
    span: SourceSpan,
};

/// Destructuring pattern for different kinds of destructuring
pub const DestructuringPattern = union(enum) {
    Struct: []StructDestructureField, // .{ field1, field2 }
    Tuple: [][]const u8, // (var1, var2)
    Array: [][]const u8, // [elem1, elem2]

    pub fn deinit(self: *DestructuringPattern, allocator: std.mem.Allocator) void {
        switch (self.*) {
            .Struct => |fields| {
                allocator.free(fields);
            },
            .Tuple => |names| {
                allocator.free(names);
            },
            .Array => |names| {
                allocator.free(names);
            },
        }
    }
};

/// Field specification for struct destructuring
pub const StructDestructureField = struct {
    name: []const u8, // field name in the struct
    variable: []const u8, // variable to bind to (can be same as name)
    span: SourceSpan,
};

pub const IdentifierExpr = struct {
    name: []const u8,
    type_info: TypeInfo,
    span: SourceSpan,
};

pub const LiteralExpr = union(enum) {
    Integer: IntegerLiteral,
    String: StringLiteral,
    Bool: BoolLiteral,
    Address: AddressLiteral,
    Hex: HexLiteral,
    Binary: BinaryLiteral,
    Character: CharacterLiteral,
    Bytes: BytesLiteral,
};

pub const IntegerType = enum {
    // unsigned integer types
    U8,
    U16,
    U32,
    U64,
    U128,
    U256,
    // signed integer types
    I8,
    I16,
    I32,
    I64,
    I128,
    I256,
    // used when type is not yet determined
    Unknown,
};

pub const IntegerLiteral = struct {
    value: []const u8, // Store as string to handle large numbers
    type_info: TypeInfo, // Unified type information
    span: SourceSpan,

    // check if the value fits within the specified type
    pub fn checkRange(self: *const IntegerLiteral) bool {
        const ora_type = self.type_info.ora_type orelse return true; // Cannot check range for unknown type

        // for the parsing context, we don't expect standalone negative literals here
        // as they would be parsed as unary minus operations in the AST.
        // this function is for checking if a value fits within a type's range.
        if (std.fmt.parseInt(u256, self.value, 0)) |val| {
            return switch (ora_type) {
                // unsigned types - just check upper bound
                .u8 => val <= std.math.maxInt(u8),
                .u16 => val <= std.math.maxInt(u16),
                .u32 => val <= std.math.maxInt(u32),
                .u64 => val <= std.math.maxInt(u64),
                .u128 => val <= std.math.maxInt(u128),
                .u256 => true, // All values fit in u256

                // signed types - check if value fits within positive range
                .i8 => val <= @as(u256, @intCast(std.math.maxInt(i8))),
                .i16 => val <= @as(u256, @intCast(std.math.maxInt(i16))),
                .i32 => val <= @as(u256, @intCast(std.math.maxInt(i32))),
                .i64 => val <= @as(u256, @intCast(std.math.maxInt(i64))),
                .i128 => val <= @as(u256, @intCast(std.math.maxInt(i128))),
                .i256 => val <= @as(u256, @intCast(std.math.maxInt(i256))),
                else => false, // Non-integer types
            };
        } else |_| {
            // could not parse as an unsigned integer
            // this could be due to:
            // 1. Malformed number
            // 2. Out of range for u256
            // in either case, we should consider it an error for our purposes
            return false;
        }
    }
};

pub const StringLiteral = struct {
    value: []const u8,
    type_info: TypeInfo,
    span: SourceSpan,
};

pub const BoolLiteral = struct {
    value: bool,
    type_info: TypeInfo,
    span: SourceSpan,
};

pub const AddressLiteral = struct {
    value: []const u8,
    type_info: TypeInfo,
    span: SourceSpan,
};

pub const HexLiteral = struct {
    value: []const u8,
    type_info: TypeInfo,
    span: SourceSpan,
};

pub const BinaryLiteral = struct {
    value: []const u8,
    type_info: TypeInfo,
    span: SourceSpan,
};

pub const CharacterLiteral = struct {
    value: u8,
    type_info: TypeInfo,
    span: SourceSpan,
};

pub const BytesLiteral = struct {
    value: []const u8,
    type_info: TypeInfo,
    span: SourceSpan,
};

pub const BinaryExpr = struct {
    lhs: *ExprNode,
    operator: BinaryOp,
    rhs: *ExprNode,
    type_info: TypeInfo,
    span: SourceSpan,
};

pub const UnaryExpr = struct {
    operator: UnaryOp,
    operand: *ExprNode,
    type_info: TypeInfo,
    span: SourceSpan,
};

pub const AssignmentExpr = struct {
    target: *ExprNode,
    value: *ExprNode,
    span: SourceSpan,
    /// Guard optimization: skip runtime guard if true (set during type resolution)
    /// True when: constant satisfies constraint, subtyping applies, or trusted builtin
    skip_guard: bool = false,
};

/// Compound assignment operators (+=, -=, etc.)
pub const CompoundAssignmentExpr = struct {
    target: *ExprNode,
    operator: CompoundAssignmentOp,
    value: *ExprNode,
    span: SourceSpan,
};

pub const CallExpr = struct {
    callee: *ExprNode,
    arguments: []*ExprNode, // Array of pointers to ExprNode
    type_info: TypeInfo, // Return type of the function call
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
    type_info: TypeInfo, // Type of the accessed field
    span: SourceSpan,
};

/// Casting expressions (as, as?, as!)
pub const CastExpr = struct {
    operand: *ExprNode,
    target_type: TypeInfo,
    cast_type: CastType,
    span: SourceSpan,
};

/// Compile-time expressions
pub const ComptimeExpr = struct {
    block: BlockNode,
    span: SourceSpan,
    type_info: TypeInfo = TypeInfo.unknown(),
};

/// old() expressions for postconditions
pub const OldExpr = struct {
    expr: *ExprNode,
    span: SourceSpan,
    /// Metadata: Always specification-only
    is_specification: bool = true,
};

/// Tuple expressions (a, b, c)
pub const TupleExpr = struct {
    elements: []*ExprNode, // Array of pointers to ExprNode
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
    parameters: ?[]*ExprNode = null, // Optional parameters for error data (e.g., error.TransferFailed(amount))
    span: SourceSpan,
};

/// Error cast expression
pub const ErrorCastExpr = struct {
    operand: *ExprNode,
    target_type: TypeInfo,
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

/// Struct instantiation expression (StructName { field1: value1, field2: value2 })
pub const StructInstantiationExpr = struct {
    struct_name: *ExprNode, // The struct type name (typically an Identifier)
    fields: []StructInstantiationField, // Field initializers
    span: SourceSpan,
};

/// Field initializer for struct instantiation
pub const StructInstantiationField = struct {
    name: []const u8, // Field name
    value: *ExprNode, // Field value expression
    span: SourceSpan,
};

/// Enum literal expression (e.g., Status.Active)
pub const EnumLiteralExpr = struct {
    enum_name: []const u8,
    variant_name: []const u8,
    span: SourceSpan,
};

/// Array literal expression (e.g., [1, 2, 3] or [])
pub const ArrayLiteralExpr = struct {
    elements: []*ExprNode, // Array of element expressions
    element_type: ?TypeInfo, // Optional explicit type (e.g., [u256])
    span: SourceSpan,
};

/// Switch expression node
pub const SwitchExprNode = struct {
    condition: *ExprNode, // Use pointer to break circular dependency
    cases: []SwitchCase,
    default_case: ?BlockNode,
    span: SourceSpan,
};

/// Helper functions for expression construction
/// Create an identifier expression with arena allocation
pub fn createIdentifier(allocator: std.mem.Allocator, name: []const u8, span: SourceSpan) !*ExprNode {
    const node = try allocator.create(ExprNode);
    node.* = ExprNode{
        .Identifier = IdentifierExpr{
            .name = name, // Note: name is expected to be arena-allocated already
            .type_info = TypeInfo.unknown(),
            .span = span,
        },
    };
    return node;
}

/// Create an identifier expression with explicit arena allocation
pub fn createIdentifierInArena(arena: *AstArena, name: []const u8, span: SourceSpan) !*ExprNode {
    // ensure the name is in the arena
    const name_copy = try arena.createString(name);

    // create the node in the arena
    const node = try arena.createNode(ExprNode);
    node.* = ExprNode{ .Identifier = IdentifierExpr{
        .name = name_copy,
        .type_info = TypeInfo.unknown(),
        .span = span,
    } };
    return node;
}

pub fn createBinaryExpr(allocator: std.mem.Allocator, lhs: *ExprNode, op: BinaryOp, rhs: *ExprNode, span: SourceSpan) !*ExprNode {
    const node = try allocator.create(ExprNode);
    node.* = ExprNode{ .Binary = BinaryExpr{
        .lhs = lhs,
        .operator = op,
        .rhs = rhs,
        .span = span,
        .type_info = TypeInfo.unknown(),
    } };
    return node;
}

/// Create a binary expression with explicit arena allocation
pub fn createBinaryExprInArena(arena: *AstArena, lhs: *ExprNode, op: BinaryOp, rhs: *ExprNode, span: SourceSpan) !*ExprNode {
    // create the node in the arena
    const node = try arena.createNode(ExprNode);
    node.* = ExprNode{ .Binary = BinaryExpr{
        .lhs = lhs,
        .operator = op,
        .rhs = rhs,
        .span = span,
        .type_info = TypeInfo.unknown(),
    } };
    return node;
}

/// Create a unary expression with standard allocator
pub fn createUnaryExpr(allocator: std.mem.Allocator, op: UnaryOp, operand: *ExprNode, span: SourceSpan) !*ExprNode {
    const node = try allocator.create(ExprNode);
    node.* = ExprNode{ .Unary = UnaryExpr{
        .operator = op,
        .operand = operand,
        .type_info = TypeInfo.unknown(),
        .span = span,
    } };
    return node;
}

/// Create a unary expression with explicit arena allocation
pub fn createUnaryExprInArena(arena: *AstArena, op: UnaryOp, operand: *ExprNode, span: SourceSpan) !*ExprNode {
    const node = try arena.createNode(ExprNode);
    node.* = ExprNode{ .Unary = UnaryExpr{
        .operator = op,
        .operand = operand,
        .type_info = TypeInfo.unknown(),
        .span = span,
    } };
    return node;
}

pub fn createQuantifiedExpr(allocator: std.mem.Allocator, quantifier: QuantifierType, variable: []const u8, variable_type: TypeInfo, condition: ?*ExprNode, body: *ExprNode, span: SourceSpan) !*ExprNode {
    const node = try allocator.create(ExprNode);
    node.* = .{ .Quantified = .{
        .quantifier = quantifier,
        .variable = variable,
        .variable_type = variable_type,
        .condition = condition,
        .body = body,
        .span = span,
        .verification_metadata = null,
        .verification_attributes = &[_]verification.VerificationAttribute{},
    } };
    return node;
}

/// Create a quantified expression with verification metadata
pub fn createQuantifiedExprWithVerification(allocator: std.mem.Allocator, quantifier: QuantifierType, variable: []const u8, variable_type: TypeInfo, condition: ?*ExprNode, body: *ExprNode, span: SourceSpan, verification_metadata: ?*verification.QuantifiedMetadata, verification_attributes: []verification.VerificationAttribute) !*ExprNode {
    const node = try allocator.create(ExprNode);
    node.* = .{ .Quantified = .{
        .quantifier = quantifier,
        .variable = variable,
        .variable_type = variable_type,
        .condition = condition,
        .body = body,
        .span = span,
        .verification_metadata = verification_metadata,
        .verification_attributes = verification_attributes,
    } };
    return node;
}

pub fn createAnonymousStructExpr(allocator: std.mem.Allocator, fields: []AnonymousStructField, span: SourceSpan) !*ExprNode {
    const node = try allocator.create(ExprNode);
    node.* = .{ .AnonymousStruct = .{
        .fields = fields,
        .span = span,
    } };
    return node;
}

/// Create an integer literal with the specified type info
pub fn createIntegerLiteral(allocator: std.mem.Allocator, value: []const u8, type_info: TypeInfo, span: SourceSpan) !*ExprNode {
    const node = try allocator.create(ExprNode);
    node.* = .{ .Literal = .{ .Integer = .{
        .value = value,
        .type_info = type_info,
        .span = span,
    } } };
    return node;
}

/// Create an integer literal with unknown type (to be determined during type checking)
pub fn createUntypedIntegerLiteral(allocator: std.mem.Allocator, value: []const u8, span: SourceSpan) !*ExprNode {
    return createIntegerLiteral(allocator, value, CommonTypes.unknown_integer(), span);
}

pub fn createRangeExpr(allocator: std.mem.Allocator, start: *ExprNode, end: *ExprNode, inclusive: bool, span: SourceSpan) !*ExprNode {
    const node = try allocator.create(ExprNode);
    node.* = .{ .Range = .{
        .start = start,
        .end = end,
        .inclusive = inclusive,
        .span = span,
    } };
    return node;
}

pub fn createArrayLiteralExpr(allocator: std.mem.Allocator, elements: []*ExprNode, element_type: ?TypeInfo, span: SourceSpan) !*ExprNode {
    const node = try allocator.create(ExprNode);
    node.* = .{ .ArrayLiteral = .{
        .elements = elements,
        .element_type = element_type,
        .span = span,
    } };
    return node;
}

pub fn createSwitchExprNode(allocator: std.mem.Allocator, condition: *ExprNode, cases: []SwitchCase, default_case: ?BlockNode, span: SourceSpan) !*ExprNode {
    const node = try allocator.create(ExprNode);
    node.* = .{ .SwitchExpression = .{
        .condition = condition,
        .cases = cases,
        .default_case = default_case,
        .span = span,
    } };
    return node;
}

pub fn createLabeledBlockExpr(allocator: std.mem.Allocator, label: []const u8, block: BlockNode, span: SourceSpan) !*ExprNode {
    const node = try allocator.create(ExprNode);
    node.* = .{ .LabeledBlock = .{
        .label = label,
        .block = block,
        .span = span,
    } };
    return node;
}

pub fn createDestructuringExpr(allocator: std.mem.Allocator, pattern: DestructuringPattern, value: *ExprNode, span: SourceSpan) !*ExprNode {
    const node = try allocator.create(ExprNode);
    node.* = .{ .Destructuring = .{
        .pattern = pattern,
        .value = value,
        .span = span,
    } };
    return node;
}

// L-value validation functions

/// Check if an expression is a valid L-value (can be assigned to)
/// According to grammar: lvalue ::= postfix_expression
/// Valid L-values: identifiers, field access, array/map indexing
pub fn isValidLValue(expr: *const ExprNode) bool {
    return switch (expr.*) {
        .Identifier => true, // Simple identifier (x)
        .FieldAccess => |field_access| {
            // field access (x.y) - target must also be valid L-value
            return isValidLValue(field_access.target);
        },
        .Index => |index_expr| {
            // array/map indexing (x[y]) - target must be valid L-value
            return isValidLValue(index_expr.target);
        },
        else => false, // All other expressions are not valid L-values
    };
}

/// Validate that an expression is a valid L-value, returning an error if not
pub const LValueError = error{
    InvalidLValue,
    LiteralNotAssignable,
    CallNotAssignable,
    BinaryExprNotAssignable,
    UnaryExprNotAssignable,
};

pub fn validateLValue(expr: *const ExprNode) LValueError!void {
    if (!isValidLValue(expr)) {
        return switch (expr.*) {
            .Literal => LValueError.LiteralNotAssignable,
            .Call => LValueError.CallNotAssignable,
            .Binary => LValueError.BinaryExprNotAssignable,
            .Unary => LValueError.UnaryExprNotAssignable,
            else => LValueError.InvalidLValue,
        };
    }
}

/// Recursively deinitialize child allocations for an expression node.
/// This does not destroy the node pointer itself; owners handle destruction.
pub fn deinitExprNode(allocator: std.mem.Allocator, expr: *ExprNode) void {
    switch (expr.*) {
        .Identifier => {},
        .Literal => {},
        .EnumLiteral => {},
        .ErrorReturn => {},
        .Quantified => |*q| {
            if (q.condition) |c| deinitExprNode(allocator, c);
            deinitExprNode(allocator, q.body);
        },

        .Unary => |*u| {
            deinitExprNode(allocator, u.operand);
        },
        .Binary => |*b| {
            deinitExprNode(allocator, b.lhs);
            deinitExprNode(allocator, b.rhs);
        },
        .Assignment => |*a| {
            deinitExprNode(allocator, a.target);
            deinitExprNode(allocator, a.value);
        },
        .CompoundAssignment => |*ca| {
            deinitExprNode(allocator, ca.target);
            deinitExprNode(allocator, ca.value);
        },
        .Call => |*c| {
            deinitExprNode(allocator, c.callee);
            for (c.arguments) |arg| deinitExprNode(allocator, arg);
            allocator.free(c.arguments);
        },
        .Index => |*ix| {
            deinitExprNode(allocator, ix.target);
            deinitExprNode(allocator, ix.index);
        },
        .FieldAccess => |*fa| {
            deinitExprNode(allocator, fa.target);
        },
        .Cast => |*c| {
            deinitExprNode(allocator, c.operand);
        },
        .Comptime => |*ct| {
            statements.deinitBlockNode(allocator, &ct.block);
        },
        .Old => |*o| {
            deinitExprNode(allocator, o.expr);
        },
        .Tuple => |*t| {
            for (t.elements) |el| deinitExprNode(allocator, el);
            allocator.free(t.elements);
        },
        .StructInstantiation => |*si| {
            deinitExprNode(allocator, si.struct_name);
            for (si.fields) |*f| deinitExprNode(allocator, f.value);
            allocator.free(si.fields);
        },
        .AnonymousStruct => |*as| {
            for (as.fields) |*f| deinitExprNode(allocator, f.value);
            allocator.free(as.fields);
        },
        .ArrayLiteral => |*al| {
            for (al.elements) |el| deinitExprNode(allocator, el);
            allocator.free(al.elements);
        },
        .SwitchExpression => |*sw| {
            deinitExprNode(allocator, sw.condition);
            // deinit patterns and bodies per case
            for (sw.cases) |*case| {
                switch (case.pattern) {
                    .Literal => {},
                    .EnumValue => {},
                    .Else => {},
                    .Range => |r| {
                        deinitExprNode(allocator, r.start);
                        deinitExprNode(allocator, r.end);
                    },
                }
                switch (case.body) {
                    .Expression => |e| deinitExprNode(allocator, e),
                    .Block => |b| statements.deinitBlockNode(allocator, @constCast(&b)),
                    .LabeledBlock => |lb| statements.deinitBlockNode(allocator, @constCast(&lb.block)),
                }
            }
            if (sw.default_case) |*db| {
                statements.deinitBlockNode(allocator, @constCast(db));
            }
        },
        .Range => |*r| {
            deinitExprNode(allocator, r.start);
            deinitExprNode(allocator, r.end);
        },
        .LabeledBlock => |*lb| {
            statements.deinitBlockNode(allocator, &lb.block);
        },
        .Destructuring => |*d| {
            d.pattern.deinit(allocator);
            deinitExprNode(allocator, d.value);
        },
        .Try => |*t| {
            deinitExprNode(allocator, t.expr);
        },
        .ErrorCast => |*ec| {
            deinitExprNode(allocator, ec.operand);
        },
        .Shift => |*sh| {
            deinitExprNode(allocator, sh.mapping);
            deinitExprNode(allocator, sh.source);
            deinitExprNode(allocator, sh.dest);
            deinitExprNode(allocator, sh.amount);
        },
    }
}

/// Deinit helper for switch cases (used by statement deinit)
pub fn deinitSwitchCase(self: *SwitchCase, allocator: std.mem.Allocator) void {
    switch (self.pattern) {
        .Range => |r| {
            deinitExprNode(allocator, r.start);
            deinitExprNode(allocator, r.end);
        },
        else => {},
    }
    switch (self.body) {
        .Expression => |e| deinitExprNode(allocator, e),
        .Block => |b| statements.deinitBlockNode(allocator, @constCast(&b)),
        .LabeledBlock => |lb| statements.deinitBlockNode(allocator, @constCast(&lb.block)),
    }
}

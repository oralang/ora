const std = @import("std");
const SourceSpan = @import("types.zig").SourceSpan;
const TypeInfo = @import("type_info.zig").TypeInfo;

/// Binary and unary operators
pub const BinaryOp = enum {
    // Arithmetic
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    StarStar, // **

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
    BitwiseAnd, // &
    BitwiseOr, // |
    BitwiseXor, // ^
    LeftShift, // <<
    RightShift, // >>

    // Comma operator
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

// Forward declaration for types and blocks
const TypeRef = @import("types.zig").TypeRef;
const BlockNode = @import("statements.zig").BlockNode;

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

    pub fn deinit(self: *SwitchPattern, allocator: std.mem.Allocator) void {
        switch (self.*) {
            .Range => |*range| {
                deinitExprNode(allocator, range.start);
                deinitExprNode(allocator, range.end);
                allocator.destroy(range.start);
                allocator.destroy(range.end);
            },
            else => {},
        }
    }
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

    pub fn deinit(self: *SwitchBody, allocator: std.mem.Allocator) void {
        switch (self.*) {
            .Expression => |expr| {
                deinitExprNode(allocator, expr);
                allocator.destroy(expr);
            },
            .Block => |*block| {
                // Import deinitBlockNode when needed
                @import("statements.zig").deinitBlockNode(allocator, block);
            },
            .LabeledBlock => |*labeled| {
                @import("statements.zig").deinitBlockNode(allocator, &labeled.block);
            },
        }
    }
};

/// Switch case for switch expressions (moved from statements.zig to break circular dependency)
pub const SwitchCase = struct {
    pattern: SwitchPattern,
    body: SwitchBody,
    span: SourceSpan,

    pub fn deinit(self: *SwitchCase, allocator: std.mem.Allocator) void {
        self.pattern.deinit(allocator);
        self.body.deinit(allocator);
    }
};

/// Forward declaration for recursive expressions
pub const ExprNode = union(enum) {
    Identifier: IdentifierExpr,
    Literal: LiteralExpr,
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
    SwitchExpression: SwitchExprNode, // switch expressions

    // Quantified expressions for formal verification
    Quantified: QuantifiedExpr, // forall/exists expressions

    // Error handling expressions
    Try: TryExpr, // try expression
    ErrorReturn: ErrorReturnExpr, // error.SomeError
    ErrorCast: ErrorCastExpr, // value as !T

    // Shift operations
    Shift: ShiftExpr, // mapping from source -> dest : amount

    // Struct instantiation
    StructInstantiation: StructInstantiationExpr, // StructName { field1: value1, field2: value2 }

    // Anonymous struct literals
    AnonymousStruct: AnonymousStructExpr, // .{ field1: value1, field2: value2 }

    // Range expressions
    Range: RangeExpr, // 1...1000 or 0..periods

    // Labeled block expressions
    LabeledBlock: LabeledBlockExpr, // label: { ... break :label value; }

    // Destructuring expressions
    Destructuring: DestructuringExpr, // let .{ balance, locked_until } = account

    // Enum literal expressions
    EnumLiteral: EnumLiteralExpr, // EnumName.VariantName

    // Array literal expressions
    ArrayLiteral: ArrayLiteralExpr, // [1, 2, 3] or []
};

/// Quantifier type for formal verification
pub const QuantifierType = enum {
    Forall, // forall
    Exists, // exists
};

/// Quantified expressions for formal verification (forall/exists)
pub const QuantifiedExpr = struct {
    quantifier: QuantifierType, // forall | exists
    variable: []const u8, // bound variable name
    variable_type: TypeInfo, // type of bound variable
    condition: ?*ExprNode, // optional where clause
    body: *ExprNode, // the quantified expression
    span: SourceSpan,
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
    type_info: @import("type_info.zig").TypeInfo, // Type information
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
    span: SourceSpan,
};

pub const LiteralExpr = union(enum) {
    Integer: IntegerLiteral,
    String: StringLiteral,
    Bool: BoolLiteral,
    Address: AddressLiteral,
    Hex: HexLiteral,
    Binary: BinaryLiteral,
};

pub const IntegerType = enum {
    // Unsigned integer types
    U8,
    U16,
    U32,
    U64,
    U128,
    U256,
    // Signed integer types
    I8,
    I16,
    I32,
    I64,
    I128,
    I256,
    // Used when type is not yet determined
    Unknown,
};

pub const IntegerLiteral = struct {
    value: []const u8, // Store as string to handle large numbers
    type_info: TypeInfo, // Unified type information
    span: SourceSpan,

    // Check if the value fits within the specified type
    pub fn checkRange(self: *const IntegerLiteral) bool {
        const ora_type = self.type_info.ora_type orelse return true; // Cannot check range for unknown type

        // For the parsing context, we don't expect standalone negative literals here
        // as they would be parsed as unary minus operations in the AST.
        // This function is for checking if a value fits within a type's range.
        if (std.fmt.parseInt(u256, self.value, 0)) |val| {
            return switch (ora_type) {
                // Unsigned types - just check upper bound
                .u8 => val <= std.math.maxInt(u8),
                .u16 => val <= std.math.maxInt(u16),
                .u32 => val <= std.math.maxInt(u32),
                .u64 => val <= std.math.maxInt(u64),
                .u128 => val <= std.math.maxInt(u128),
                .u256 => true, // All values fit in u256

                // Signed types - check if value fits within positive range
                .i8 => val <= @as(u256, @intCast(std.math.maxInt(i8))),
                .i16 => val <= @as(u256, @intCast(std.math.maxInt(i16))),
                .i32 => val <= @as(u256, @intCast(std.math.maxInt(i32))),
                .i64 => val <= @as(u256, @intCast(std.math.maxInt(i64))),
                .i128 => val <= @as(u256, @intCast(std.math.maxInt(i128))),
                .i256 => val <= @as(u256, @intCast(std.math.maxInt(i256))),
                else => false, // Non-integer types
            };
        } else |_| {
            // Could not parse as an unsigned integer
            // This could be due to:
            // 1. Malformed number
            // 2. Out of range for u256
            // In either case, we should consider it an error for our purposes
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
};

/// old() expressions for postconditions
pub const OldExpr = struct {
    expr: *ExprNode,
    span: SourceSpan,
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
            .span = span,
            .type_info = TypeInfo.unknown(),
        },
    };
    return node;
}

/// Create an identifier expression with explicit arena allocation
pub fn createIdentifierInArena(arena: *@import("ast_arena.zig").AstArena, name: []const u8, span: SourceSpan) !*ExprNode {
    // Ensure the name is in the arena
    const name_copy = try arena.createString(name);

    // Create the node in the arena
    const node = try arena.createNode(ExprNode);
    node.* = ExprNode{ .Identifier = IdentifierExpr{
        .name = name_copy,
        .span = span,
        .type_info = TypeInfo.unknown(),
    } };
    return node;
}

pub fn createBinaryExpr(allocator: std.mem.Allocator, lhs: *ExprNode, op: BinaryOp, rhs: *ExprNode, span: SourceSpan) !*ExprNode {
    const node = try allocator.create(ExprNode);
    node.* = ExprNode{ .Binary = BinaryExpr{
        .lhs = lhs,
        .op = op,
        .rhs = rhs,
        .span = span,
        .type_info = TypeInfo.unknown(),
    } };
    return node;
}

/// Create a binary expression with explicit arena allocation
pub fn createBinaryExprInArena(arena: *@import("ast_arena.zig").AstArena, lhs: *ExprNode, op: BinaryOp, rhs: *ExprNode, span: SourceSpan) !*ExprNode {
    // Create the node in the arena
    const node = try arena.createNode(ExprNode);
    node.* = ExprNode{ .Binary = BinaryExpr{
        .lhs = lhs,
        .op = op,
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
pub fn createUnaryExprInArena(arena: *@import("ast_arena.zig").AstArena, op: UnaryOp, operand: *ExprNode, span: SourceSpan) !*ExprNode {
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
    return createIntegerLiteral(allocator, value, @import("type_info.zig").CommonTypes.unknown_integer(), span);
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

/// Free an expression node and all its children
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
            for (call.arguments) |arg| {
                deinitExprNode(allocator, arg);
                allocator.destroy(arg);
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
            // Note: target_type cleanup handled by main AST
        },
        .Comptime => |*comptime_expr| {
            // Note: block cleanup handled by main AST
            _ = comptime_expr;
        },
        .Old => |*old| {
            deinitExprNode(allocator, old.expr);
            allocator.destroy(old.expr);
        },
        .Quantified => |*quantified| {
            if (quantified.condition) |condition| {
                deinitExprNode(allocator, condition);
                allocator.destroy(condition);
            }
            deinitExprNode(allocator, quantified.body);
            allocator.destroy(quantified.body);
        },
        .Try => |*try_expr| {
            deinitExprNode(allocator, try_expr.expr);
            allocator.destroy(try_expr.expr);
        },
        .ErrorReturn => {
            // String literals don't need cleanup
        },
        .ErrorCast => |*error_cast| {
            deinitExprNode(allocator, error_cast.operand);
            allocator.destroy(error_cast.operand);
            // Note: target_type cleanup handled by main AST
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
        .StructInstantiation => |*struct_inst| {
            deinitExprNode(allocator, struct_inst.struct_name);
            allocator.destroy(struct_inst.struct_name);
            for (struct_inst.fields) |*field| {
                deinitExprNode(allocator, field.value);
                allocator.destroy(field.value);
            }
            allocator.free(struct_inst.fields);
        },
        .AnonymousStruct => |*anon_struct| {
            for (anon_struct.fields) |*field| {
                deinitExprNode(allocator, field.value);
                allocator.destroy(field.value);
            }
            allocator.free(anon_struct.fields);
        },
        .Range => |*range| {
            deinitExprNode(allocator, range.start);
            deinitExprNode(allocator, range.end);
            allocator.destroy(range.start);
            allocator.destroy(range.end);
        },
        .LabeledBlock => |*labeled_block| {
            // Note: block cleanup handled by main AST
            _ = labeled_block;
        },
        .Destructuring => |*destructuring| {
            deinitExprNode(allocator, destructuring.value);
            allocator.destroy(destructuring.value);
            // Clean up pattern
            switch (destructuring.pattern) {
                .Struct => |fields| {
                    allocator.free(fields);
                },
                .Tuple => |vars| {
                    allocator.free(vars);
                },
                .Array => |vars| {
                    allocator.free(vars);
                },
            }
        },
        .EnumLiteral => {
            // String literals don't need cleanup
        },
        .ArrayLiteral => |*array_lit| {
            for (array_lit.elements) |element| {
                deinitExprNode(allocator, element);
                allocator.destroy(element);
            }
            allocator.free(array_lit.elements);
        },
        .Tuple => |*tuple| {
            for (tuple.elements) |element| {
                deinitExprNode(allocator, element);
                allocator.destroy(element);
            }
            allocator.free(tuple.elements);
        },
        .SwitchExpression => |*switch_expr| {
            deinitExprNode(allocator, switch_expr.condition);
            allocator.destroy(switch_expr.condition);
            for (switch_expr.cases) |*case| {
                // Note: switch case cleanup handled by statements module
                _ = case;
            }
            allocator.free(switch_expr.cases);
        },
        else => {
            // Literals and identifiers don't need cleanup
        },
    }
}

// L-value validation functions

/// Check if an expression is a valid L-value (can be assigned to)
/// According to grammar: lvalue ::= postfix_expression
/// Valid L-values: identifiers, field access, array/map indexing
pub fn isValidLValue(expr: *const ExprNode) bool {
    return switch (expr.*) {
        .Identifier => true, // Simple identifier (x)
        .FieldAccess => |field_access| {
            // Field access (x.y) - target must also be valid L-value
            return isValidLValue(field_access.target);
        },
        .Index => |index_expr| {
            // Array/map indexing (x[y]) - target must be valid L-value
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

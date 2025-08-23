const std = @import("std");
const SourceSpan = @import("../ast.zig").SourceSpan;

// Forward declaration for expressions
const expressions = @import("expressions.zig");
const ExprNode = expressions.ExprNode;
const LiteralExpr = expressions.LiteralExpr;
const RangeExpr = expressions.RangeExpr;
const SwitchCase = expressions.SwitchCase;
const SwitchPattern = expressions.SwitchPattern;
const SwitchBody = expressions.SwitchBody;

/// Block node to group statements
pub const BlockNode = struct {
    statements: []StmtNode,
    span: SourceSpan,
};

/// Statement node types
pub const StmtNode = union(enum) {
    Expr: ExprNode,
    VariableDecl: VariableDeclNode,
    DestructuringAssignment: DestructuringAssignmentNode, // let .{field1, field2} = expr
    Return: ReturnNode,
    If: IfNode,
    While: WhileNode,
    ForLoop: ForLoopNode, // for (expr) |var1, var2| stmt
    Break: BreakNode,
    Continue: ContinueNode,
    Log: LogNode,
    Lock: LockNode, // @lock annotations
    Unlock: UnlockNode, // @unlock annotations
    Invariant: InvariantNode, // Loop invariants
    Requires: RequiresNode,
    Ensures: EnsuresNode,
    Switch: SwitchNode, // switch statements
    Move: MoveNode, // expr from source -> dest : amount
    LabeledBlock: LabeledBlockNode, // label: { statements }

    // Error handling statements
    ErrorDecl: ErrorDeclNode, // error MyError;
    TryBlock: TryBlockNode, // try { ... } catch { ... }

    // Compound assignment statements
    CompoundAssignment: CompoundAssignmentNode, // a += b, a -= b, etc.
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

/// Loop pattern for different kinds of iteration variable binding
pub const LoopPattern = union(enum) {
    Single: struct {
        name: []const u8, // |item|
        span: SourceSpan,
    },
    IndexPair: struct { // |item, index|
        item: []const u8,
        index: []const u8,
        span: SourceSpan,
    },
    Destructured: struct { // |.{field1, field2}|
        pattern: DestructuringPattern,
        span: SourceSpan,
    },
};

/// Import destructuring pattern from expressions
const DestructuringPattern = expressions.DestructuringPattern;

pub const ForLoopNode = struct {
    iterable: ExprNode, // The expression to iterate over
    pattern: LoopPattern, // Enhanced from simple var names
    body: BlockNode, // Loop body block
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
    parameters: ?[]@import("../ast.zig").ParameterNode, // Optional parameters for error data
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

/// Break Statement with optional label and value
pub const BreakNode = struct {
    label: ?[]const u8, // Optional label for labeled break
    value: ?*ExprNode, // Optional value to return
    span: SourceSpan,
};

/// Continue Statement with optional label
pub const ContinueNode = struct {
    label: ?[]const u8, // Optional label for labeled continue
    value: ?*ExprNode, // Optional replacement operand (for labeled switch continue)
    span: SourceSpan,
};

/// Variable Declaration with Ora's memory model
pub const VariableDeclNode = struct {
    name: []const u8,
    region: MemoryRegion,
    kind: VariableKind, // var, let, const, immutable
    locked: bool, // true if @lock annotation is present
    type_info: @import("type_info.zig").TypeInfo, // Unified type information
    value: ?*ExprNode,
    span: SourceSpan,
    // Tuple unpacking support
    tuple_names: ?[][]const u8, // For tuple unpacking: let (a, b) = expr
};

/// Destructuring Assignment Statement
pub const DestructuringAssignmentNode = struct {
    pattern: DestructuringPattern, // Pattern to match (e.g., .{field1, field2})
    value: *ExprNode, // Expression to destructure
    span: SourceSpan,
};

/// Move Statement (expr from source -> dest : amount)
pub const MoveNode = struct {
    expr: ExprNode, // The expression being moved
    source: ExprNode, // Source location
    dest: ExprNode, // Destination location
    amount: ExprNode, // Amount to move
    span: SourceSpan,
};

/// Unlock Statement (@unlock(expression))
pub const UnlockNode = struct {
    path: ExprNode, // e.g., balances[to]
    span: SourceSpan,
};

/// Labeled Block Statement (label: { statements })
pub const LabeledBlockNode = struct {
    label: []const u8, // Block label
    block: BlockNode, // Block contents
    span: SourceSpan,
};

/// Memory regions matching Ora specification
pub const MemoryRegion = enum {
    Stack, // let/var (default)
    Memory, // memory let/memory var
    Storage, // storage let/storage var
    TStore, // tstore let/tstore var
};

/// Variable kinds for different mutability and initialization semantics
pub const VariableKind = enum {
    Var, // var - mutable variable
    Let, // let - immutable variable
    Const, // const - compile-time constant
    Immutable, // immutable - deploy-time constant
};

/// Switch statement
pub const SwitchNode = struct {
    condition: ExprNode,
    cases: []SwitchCase,
    default_case: ?BlockNode,
    span: SourceSpan,
};

/// Free a statement node
pub fn deinitStmtNode(allocator: std.mem.Allocator, stmt: *StmtNode) void {
    switch (stmt.*) {
        .Expr => |*expr| {
            expressions.deinitExprNode(allocator, expr);
        },
        .VariableDecl => |*var_decl| {
            // TypeInfo doesn't need explicit cleanup like TypeRef did
            if (var_decl.value) |value| {
                expressions.deinitExprNode(allocator, value);
                allocator.destroy(value);
            }
            if (var_decl.tuple_names) |names| {
                allocator.free(names);
            }
        },
        .DestructuringAssignment => |*dest_assign| {
            dest_assign.pattern.deinit(allocator);
            expressions.deinitExprNode(allocator, dest_assign.value);
            allocator.destroy(dest_assign.value);
        },
        .Move => |*move_stmt| {
            expressions.deinitExprNode(allocator, &move_stmt.expr);
            expressions.deinitExprNode(allocator, &move_stmt.source);
            expressions.deinitExprNode(allocator, &move_stmt.dest);
            expressions.deinitExprNode(allocator, &move_stmt.amount);
        },
        .Unlock => |*unlock| {
            expressions.deinitExprNode(allocator, &unlock.path);
        },
        .LabeledBlock => |*labeled| {
            deinitBlockNode(allocator, &labeled.block);
        },
        .Return => |*ret| {
            if (ret.value) |*value| {
                expressions.deinitExprNode(allocator, value);
            }
        },
        .If => |*if_stmt| {
            expressions.deinitExprNode(allocator, &if_stmt.condition);
            deinitBlockNode(allocator, &if_stmt.then_branch);
            if (if_stmt.else_branch) |*else_branch| {
                deinitBlockNode(allocator, else_branch);
            }
        },
        .While => |*while_stmt| {
            expressions.deinitExprNode(allocator, &while_stmt.condition);
            deinitBlockNode(allocator, &while_stmt.body);
            for (while_stmt.invariants) |*inv| {
                expressions.deinitExprNode(allocator, inv);
            }
            allocator.free(while_stmt.invariants);
        },
        .Log => |*log| {
            for (log.args) |*arg| {
                expressions.deinitExprNode(allocator, arg);
            }
            allocator.free(log.args);
        },
        .Lock => |*lock| {
            expressions.deinitExprNode(allocator, &lock.path);
        },
        .Invariant => |*inv| {
            expressions.deinitExprNode(allocator, &inv.condition);
        },
        .Requires => |*req| {
            expressions.deinitExprNode(allocator, &req.condition);
        },
        .Ensures => |*ens| {
            expressions.deinitExprNode(allocator, &ens.condition);
        },
        .ErrorDecl => |*error_decl| {
            // Clean up parameters if present
            if (error_decl.parameters) |params| {
                for (params) |*param| {
                    if (param.default_value) |default_val| {
                        expressions.deinitExprNode(allocator, default_val);
                        allocator.destroy(default_val);
                    }
                }
                allocator.free(params);
            }
        },
        .TryBlock => |*try_block| {
            deinitBlockNode(allocator, &try_block.try_block);
            if (try_block.catch_block) |*catch_block| {
                deinitBlockNode(allocator, &catch_block.block);
            }
        },
        .Switch => |*switch_stmt| {
            expressions.deinitExprNode(allocator, &switch_stmt.condition);
            for (switch_stmt.cases) |*case| {
                expressions.deinitSwitchCase(case, allocator);
            }
            allocator.free(switch_stmt.cases);
        },
        .Break => |*break_node| {
            if (break_node.value) |value| {
                expressions.deinitExprNode(allocator, value);
            }
        },
        .Continue => {
            // Continue statements only have a span and optional label, no cleanup needed
        },
        .ForLoop => |*for_loop| {
            expressions.deinitExprNode(allocator, &for_loop.iterable);
            deinitBlockNode(allocator, &for_loop.body);
        },
        .CompoundAssignment => |*compound| {
            expressions.deinitExprNode(allocator, compound.target);
            expressions.deinitExprNode(allocator, compound.value);
            allocator.destroy(compound.target);
            allocator.destroy(compound.value);
        },
    }
}

/// Compound assignment node (a += b, a -= b, etc.)
pub const CompoundAssignmentNode = struct {
    target: *ExprNode, // Target expression (left-hand side)
    operator: expressions.CompoundAssignmentOp, // Operation type
    value: *ExprNode, // Value expression (right-hand side)
    span: SourceSpan,
};

/// Free a block node - forward declaration to avoid circular imports
pub fn deinitBlockNode(allocator: std.mem.Allocator, block: *BlockNode) void {
    _ = allocator;
    _ = block;
    // Implementation will be moved to main ast.zig
}

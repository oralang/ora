//! Ora Comptime System
//!
//! Zig-style compile-time evaluation with:
//! - Two-layer value model (CtValue for evaluation, ConstValue for persistence)
//! - 3-way stage validity (comptime_only, comptime_ok, runtime_only)
//! - Slot+heap memory model with copy-on-write
//! - Deterministic limits (fuel/steps)
//!
//! ## Architecture
//!
//! ```
//! src/comptime/
//! ├── mod.zig      - Public API
//! ├── value.zig    - CtValue + ConstValue/ConstId
//! ├── pool.zig     - ConstPool (compiler-wide interning)
//! ├── env.zig      - CtEnv with slots + heap
//! ├── eval.zig     - Evaluator (typed IR interpreter)
//! ├── stage.zig    - Stage validity tags
//! ├── limits.zig   - EvalConfig + fuel
//! └── error.zig    - CtError types
//! ```
//!
//! ## Usage
//!
//! ```zig
//! // Create pool (lives for entire compilation)
//! var pool = ConstPool.init(allocator);
//! defer pool.deinit();
//!
//! // Create environment (per-evaluation)
//! var env = CtEnv.init(allocator, .{});
//! defer env.deinit();
//!
//! // Evaluate expression
//! const result = comptime.evaluate(&env, expr, .must_eval, .strict, diag);
//!
//! // Intern result for persistence
//! if (result == .value) {
//!     const const_id = try pool.intern(&env, result.value);
//!     // Store const_id in HIR/symbol table
//! }
//! ```

// Re-export value types
pub const CtValue = @import("value.zig").CtValue;
pub const CtEnum = @import("value.zig").CtEnum;
pub const ConstValue = @import("value.zig").ConstValue;
pub const ConstStruct = @import("value.zig").ConstStruct;
pub const ConstField = @import("value.zig").ConstField;
pub const ConstEnum = @import("value.zig").ConstEnum;
pub const ConstId = @import("value.zig").ConstId;
pub const SlotId = @import("value.zig").SlotId;
pub const HeapId = @import("value.zig").HeapId;
pub const TypeId = @import("value.zig").TypeId;
pub const FieldId = @import("value.zig").FieldId;
pub const VariantId = @import("value.zig").VariantId;

// Re-export pool
pub const ConstPool = @import("pool.zig").ConstPool;

// Re-export stage
pub const Stage = @import("stage.zig").Stage;
pub const classifyBinaryOp = @import("stage.zig").classifyBinaryOp;
pub const classifyUnaryOp = @import("stage.zig").classifyUnaryOp;
pub const isRuntimeOnlyIntrinsic = @import("stage.zig").isRuntimeOnlyIntrinsic;
pub const isComptimeOnlyIntrinsic = @import("stage.zig").isComptimeOnlyIntrinsic;

// Re-export errors
pub const CtError = @import("error.zig").CtError;
pub const CtErrorKind = @import("error.zig").CtErrorKind;
pub const TryEvalPolicy = @import("error.zig").TryEvalPolicy;

// Re-export limits
pub const EvalConfig = @import("limits.zig").EvalConfig;
pub const EvalStats = @import("limits.zig").EvalStats;
pub const LimitCheck = @import("limits.zig").LimitCheck;

// Re-export heap
pub const CtHeap = @import("heap.zig").CtHeap;
pub const CtAggregate = @import("heap.zig").CtAggregate;

// Re-export environment
pub const CtEnv = @import("env.zig").CtEnv;
pub const Scope = @import("env.zig").Scope;

// Re-export evaluator
pub const Evaluator = @import("eval.zig").Evaluator;
pub const BinaryOp = @import("eval.zig").BinaryOp;
pub const UnaryOp = @import("eval.zig").UnaryOp;

// Re-export AST evaluator (expressions)
pub const ast_eval = @import("ast_eval.zig");
pub const AstEvaluator = ast_eval.AstEvaluator;
pub const AstEvalResult = ast_eval.AstEvalResult;
pub const InternResult = ast_eval.AstEvaluator.InternResult;
pub const IdentifierLookup = ast_eval.IdentifierLookup;
pub const evaluateExpr = ast_eval.evaluateExpr;
pub const evaluateToInteger = ast_eval.evaluateToInteger;
pub const evaluateAndIntern = ast_eval.evaluateAndIntern;

// Re-export statement evaluator (loops, conditionals, blocks)
pub const StmtEvaluator = ast_eval.StmtEvaluator;
pub const AstStmtResult = ast_eval.AstStmtResult;
pub const evaluateBlock = ast_eval.evaluateBlock;
pub const evaluateBlockWithEnv = ast_eval.evaluateBlockWithEnv;

// Re-export evaluation mode and result from eval.zig
pub const EvalMode = @import("eval.zig").EvalMode;
pub const EvalResult = @import("eval.zig").EvalResult;
pub const ControlFlow = @import("eval.zig").ControlFlow;


// Run all module tests
test {
    const std = @import("std");
    std.testing.refAllDecls(@This());
    _ = @import("value.zig");
    _ = @import("stage.zig");
    _ = @import("error.zig");
    _ = @import("limits.zig");
    _ = @import("pool.zig");
    _ = @import("heap.zig");
    _ = @import("env.zig");
    _ = @import("eval.zig");
    _ = @import("ast_eval.zig");
}

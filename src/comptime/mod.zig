//! Ora Comptime System
//!
//! Zig-style compile-time evaluation with:
//! - Evaluator-local value model
//! - 3-way stage validity (comptime_only, comptime_ok, runtime_only)
//! - Slot+heap memory model with copy-on-write
//! - Deterministic limits (fuel/steps)
//!
//! ## Architecture
//!
//! ```
//! src/comptime/
//! ├── mod.zig      - Public API
//! ├── value.zig    - CtValue + evaluator handle types
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
//! // Create environment (per-evaluation)
//! var env = CtEnv.init(allocator, .{});
//! defer env.deinit();
//!
//! // Evaluate expression
//! const result = comptime.evaluate(&env, expr, .must_eval, .strict, diag);
//!
//! ```

// Re-export value types
pub const CtValue = @import("value.zig").CtValue;
pub const CtAdt = @import("value.zig").CtAdt;
pub const CtEnum = @import("value.zig").CtEnum;
pub const CtErrorUnion = @import("value.zig").CtErrorUnion;
pub const SlotId = @import("value.zig").SlotId;
pub const HeapId = @import("value.zig").HeapId;
pub const TypeId = @import("value.zig").TypeId;
pub const FieldId = @import("value.zig").FieldId;
pub const VariantId = @import("value.zig").VariantId;
pub const type_ids = @import("value.zig").type_ids;

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

// TEST: eval.zig renamed to __eval_todelete.zig to verify it's dead.
// Re-exports of Evaluator/BinaryOp/UnaryOp/EvalMode/EvalResult/ControlFlow removed for the test.

// The actual evaluator (AST walker used by sema, db, lsp)
pub const compiler_ast_eval = @import("compiler_ast_eval.zig");

// Run all module tests
test {
    const std = @import("std");
    std.testing.refAllDecls(@This());
    _ = @import("value.zig");
    _ = @import("stage.zig");
    _ = @import("error.zig");
    _ = @import("limits.zig");
    _ = @import("heap.zig");
    _ = @import("env.zig");
}

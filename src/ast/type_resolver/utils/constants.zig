// ============================================================================
// Constant Evaluation (wrapper)
// ============================================================================
// Delegates to the new comptime system in src/comptime/
// ============================================================================

const comptime_eval = @import("../../../comptime/mod.zig");

pub const CtValue = comptime_eval.CtValue;
pub const AstEvalResult = comptime_eval.AstEvalResult;
pub const AstEvaluator = comptime_eval.AstEvaluator;
pub const evaluateExpr = comptime_eval.evaluateExpr;
pub const evaluateToInteger = comptime_eval.evaluateToInteger;

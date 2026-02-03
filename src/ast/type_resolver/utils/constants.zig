// ============================================================================
// Constant Evaluation (wrapper)
// ============================================================================
// Delegates to shared comptime evaluator in src/const_eval.zig
// ============================================================================

const const_eval = @import("../../../const_eval.zig");

pub const ConstantValue = const_eval.ConstantValue;

pub const evaluateConstantExpression = const_eval.evaluateConstantExpression;

pub const extractIntegerValue = const_eval.extractIntegerValue;

const mlir = @import("mlir_c_api").c;
const ast = @import("../ast/mod.zig");
const support = @import("support.zig");

const appendOp = support.appendOp;
const exprRange = support.exprRange;

pub fn mixin(ContractLowerer: type, Lowerer: type, FunctionLowerer: type) type {
    _ = Lowerer;
    return struct {
        pub fn lowerInvariant(self: *ContractLowerer, expr_id: ast.ExprId) !void {
            var function_lowerer = FunctionLowerer.initContractContext(self.parent, self.block);
            const invariant_expr_id = switch (self.parent.file.expression(expr_id).*) {
                .Call => |call| if (call.args.len == 1) call.args[0] else expr_id,
                else => expr_id,
            };
            const value = try function_lowerer.lowerExpr(invariant_expr_id, &function_lowerer.locals);
            const op = mlir.oraInvariantOpCreate(self.parent.context, self.parent.location(exprRange(self.parent.file, expr_id)), value);
            if (mlir.oraOperationIsNull(op)) return error.MlirOperationCreationFailed;
            appendOp(self.block, op);
        }
    };
}

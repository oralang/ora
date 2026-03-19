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
            const value = try function_lowerer.lowerExpr(expr_id, &function_lowerer.locals);
            const op = mlir.oraInvariantOpCreate(self.parent.context, self.parent.location(exprRange(self.parent.file, expr_id)), value);
            if (mlir.oraOperationIsNull(op)) return error.MlirOperationCreationFailed;
            appendOp(self.block, op);
        }
    };
}

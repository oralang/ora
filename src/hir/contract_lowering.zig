const mlir = @import("mlir_c_api").c;
const ast = @import("../ast/mod.zig");
const support = @import("support.zig");

const appendOp = support.appendOp;
const exprRange = support.exprRange;
const namedStringAttr = support.namedStringAttr;
const strRef = support.strRef;

pub fn mixin(ContractLowerer: type, Lowerer: type, FunctionLowerer: type) type {
    _ = Lowerer;
    return struct {
        pub fn lowerInvariant(self: *ContractLowerer, expr_id: ast.ExprId) !void {
            var function_lowerer = FunctionLowerer.initContractContext(self.parent, self.block);
            const InvariantInfo = struct {
                expr_id: ast.ExprId,
                label: ?[]const u8,
            };
            const invariant_info: InvariantInfo = switch (self.parent.file.expression(expr_id).*) {
                .Call => |call| blk: {
                    if (call.args.len == 1) {
                        const label = switch (self.parent.file.expression(call.callee).*) {
                            .Name => |name| name.name,
                            else => null,
                        };
                        break :blk .{ .expr_id = call.args[0], .label = label };
                    }
                    break :blk .{ .expr_id = expr_id, .label = null };
                },
                else => .{ .expr_id = expr_id, .label = null },
            };
            const value = try function_lowerer.lowerExpr(invariant_info.expr_id, &function_lowerer.locals);
            const op = mlir.oraInvariantOpCreate(self.parent.context, self.parent.location(exprRange(self.parent.file, expr_id)), value);
            if (mlir.oraOperationIsNull(op)) return error.MlirOperationCreationFailed;
            if (invariant_info.label) |label| {
                mlir.oraOperationSetAttributeByName(op, strRef("ora.label"), namedStringAttr(self.parent.context, "ora.label", label).attribute);
            }
            appendOp(self.block, op);
        }
    };
}

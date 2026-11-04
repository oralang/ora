const mlir = @import("mlir_c_api").c;

pub fn getScfForUnsignedCmp(loop_op: mlir.MlirOperation) error{UnsupportedOperation}!bool {
    var unsigned_cmp = false;
    if (!mlir.oraScfForOpGetUnsignedCmp(loop_op, &unsigned_cmp)) {
        return error.UnsupportedOperation;
    }
    return unsigned_cmp;
}

const std = @import("std");
const mlir = @import("mlir_c_api").c;
const mlir_c = @import("mlir_c_api");

pub fn getScfForUnsignedCmp(loop_op: mlir.MlirOperation) bool {
    const printed = mlir.oraOperationPrintToString(loop_op);
    defer if (printed.data != null) {
        mlir_c.freeStringRef(printed);
    };

    if (printed.data != null and printed.length > 0) {
        const text = printed.data[0..printed.length];
        if (std.mem.indexOf(u8, text, "unsignedCmp = true") != null) return true;
        if (std.mem.indexOf(u8, text, "unsignedCmp = false") != null) return false;
    }

    const unsigned_attr = mlir.oraOperationGetAttributeByName(
        loop_op,
        mlir.oraStringRefCreate("unsignedCmp".ptr, "unsignedCmp".len),
    );
    if (mlir.oraAttributeIsNull(unsigned_attr)) return false;
    return mlir.oraIntegerAttrGetValueSInt(unsigned_attr) != 0;
}

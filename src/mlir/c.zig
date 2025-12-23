// ============================================================================
// MLIR C Bindings
// ============================================================================
//
// FFI bindings to MLIR C API for IR manipulation and dialect integration.
//
// ============================================================================

pub const c = @cImport({
    @cInclude("mlir-c/IR.h");
    @cInclude("mlir-c/BuiltinTypes.h");
    @cInclude("mlir-c/BuiltinAttributes.h");
    @cInclude("mlir-c/Support.h");
    @cInclude("mlir-c/Pass.h");
    @cInclude("mlir-c/RegisterEverything.h");

    // Ora dialect C interface
    @cInclude("ora/OraDialectC.h");
});

// Helper to free string returned from C API
pub fn freeStringRef(str: c.MlirStringRef) void {
    if (str.data != null) {
        // Note: We need mlirStringRefFree or similar - for now use free
        // The C API should provide a way to free this
        @import("std").c.free(@ptrCast(@constCast(str.data)));
    }
}

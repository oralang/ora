// ============================================================================
// MLIR C Bindings (Ora shims)
// ============================================================================
//
// FFI bindings to Ora's MLIR C++ shim interface.
//
// ============================================================================

pub const c = @cImport({
    @cInclude("ora/OraDialectC.h");
});

// Helper to free string returned from C API
pub fn freeStringRef(str: c.MlirStringRef) void {
    c.oraStringRefFree(str);
}

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
    // @cInclude("OraDialectC.h"); // Not needed for unregistered mode
});

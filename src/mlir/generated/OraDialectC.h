//===- OraDialectC.h - C Interface for Ora MLIR Dialect ----------------===//
//
// This file provides a C interface to the Ora MLIR dialect for use from Zig.
// This enables the gradual migration from unregistered to registered operations.
//
//===----------------------------------------------------------------------===//

#ifndef ORA_DIALECT_C_H
#define ORA_DIALECT_C_H

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C"
{
#endif

    //===----------------------------------------------------------------------===//
    // Ora Dialect Registration
    //===----------------------------------------------------------------------===//

    /// Register the Ora dialect with the given MLIR context
    /// Returns true if registration was successful, false otherwise
    MLIR_CAPI_EXPORTED bool oraDialectRegister(MlirContext ctx);

    /// Check if the Ora dialect is registered in the given context
    MLIR_CAPI_EXPORTED bool oraDialectIsRegistered(MlirContext ctx);

    /// Get the Ora dialect from the context (must be registered first)
    MLIR_CAPI_EXPORTED MlirDialect oraDialectGet(MlirContext ctx);

    //===----------------------------------------------------------------------===//
    // Ora Operation Creation (Registered Dialect)
    //===----------------------------------------------------------------------===//

    /// Create an ora.contract operation using the registered dialect
    /// Returns null operation if the dialect is not registered or creation fails
    MLIR_CAPI_EXPORTED MlirOperation oraContractOpCreate(
        MlirContext ctx,
        MlirLocation loc,
        MlirStringRef name);

    /// Create an ora.global operation using the registered dialect
    /// Returns null operation if the dialect is not registered or creation fails
    MLIR_CAPI_EXPORTED MlirOperation oraGlobalOpCreate(
        MlirContext ctx,
        MlirLocation loc,
        MlirStringRef name,
        MlirType type,
        MlirAttribute initValue);

    /// Create an ora.sload operation using the registered dialect
    /// Returns null operation if the dialect is not registered or creation fails
    MLIR_CAPI_EXPORTED MlirOperation oraSLoadOpCreate(
        MlirContext ctx,
        MlirLocation loc,
        MlirStringRef globalName,
        MlirType resultType);

    /// Create an ora.sstore operation using the registered dialect
    /// Returns null operation if the dialect is not registered or creation fails
    MLIR_CAPI_EXPORTED MlirOperation oraSStoreOpCreate(
        MlirContext ctx,
        MlirLocation loc,
        MlirValue value,
        MlirStringRef globalName);

#ifdef __cplusplus
}
#endif

#endif // ORA_DIALECT_C_H

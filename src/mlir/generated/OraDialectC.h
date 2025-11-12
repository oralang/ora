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

    /// Create an ora.if operation using the registered dialect
    /// Returns null operation if the dialect is not registered or creation fails
    MLIR_CAPI_EXPORTED MlirOperation oraIfOpCreate(
        MlirContext ctx,
        MlirLocation loc,
        MlirValue condition);

    /// Create an ora.while operation using the registered dialect
    /// Returns null operation if the dialect is not registered or creation fails
    MLIR_CAPI_EXPORTED MlirOperation oraWhileOpCreate(
        MlirContext ctx,
        MlirLocation loc,
        MlirValue condition);

    /// Create an ora.test operation (simple custom printer test)
    /// Returns null operation if the dialect is not registered or creation fails
    MLIR_CAPI_EXPORTED MlirOperation oraTestOpCreate(
        MlirContext ctx,
        MlirLocation loc);

    //===----------------------------------------------------------------------===//
    // Ora Type Creation
    //===----------------------------------------------------------------------===//

    /// Create an Ora integer type with the given width and signedness
    /// Returns null type if the dialect is not registered or creation fails
    MLIR_CAPI_EXPORTED MlirType oraIntegerTypeGet(
        MlirContext ctx,
        unsigned width,
        bool isSigned);

    /// Create an Ora boolean type
    /// Returns null type if the dialect is not registered or creation fails
    MLIR_CAPI_EXPORTED MlirType oraBoolTypeGet(MlirContext ctx);

    /// Create an Ora address type
    /// Returns null type if the dialect is not registered or creation fails
    MLIR_CAPI_EXPORTED MlirType oraAddressTypeGet(MlirContext ctx);

    /// Get the type of a global variable from an ora.global operation
    /// Returns null type if the operation is not an ora.global or if it fails
    MLIR_CAPI_EXPORTED MlirType oraGlobalOpGetType(MlirOperation globalOp);

    /// Create an ora.requires operation using the registered dialect
    MLIR_CAPI_EXPORTED MlirOperation oraRequiresOpCreate(
        MlirContext ctx,
        MlirLocation loc,
        MlirValue condition);

    /// Create an ora.ensures operation using the registered dialect
    MLIR_CAPI_EXPORTED MlirOperation oraEnsuresOpCreate(
        MlirContext ctx,
        MlirLocation loc,
        MlirValue condition);

    /// Create an ora.invariant operation using the registered dialect
    MLIR_CAPI_EXPORTED MlirOperation oraInvariantOpCreate(
        MlirContext ctx,
        MlirLocation loc,
        MlirValue condition);

    /// Create an ora.assert operation using the registered dialect
    MLIR_CAPI_EXPORTED MlirOperation oraAssertOpCreate(
        MlirContext ctx,
        MlirLocation loc,
        MlirValue condition,
        MlirStringRef message);

    /// Create an ora.yield operation using the registered dialect
    MLIR_CAPI_EXPORTED MlirOperation oraYieldOpCreate(
        MlirContext ctx,
        MlirLocation loc,
        const MlirValue *operands,
        size_t numOperands);

    /// Create an ora.mload operation using the registered dialect
    MLIR_CAPI_EXPORTED MlirOperation oraMLoadOpCreate(
        MlirContext ctx,
        MlirLocation loc,
        MlirStringRef variableName,
        MlirType resultType);

    /// Create an ora.mstore operation using the registered dialect
    MLIR_CAPI_EXPORTED MlirOperation oraMStoreOpCreate(
        MlirContext ctx,
        MlirLocation loc,
        MlirValue value,
        MlirStringRef variableName);

    /// Create an ora.tload operation using the registered dialect
    MLIR_CAPI_EXPORTED MlirOperation oraTLoadOpCreate(
        MlirContext ctx,
        MlirLocation loc,
        MlirStringRef key,
        MlirType resultType);

    /// Create an ora.tstore operation using the registered dialect
    MLIR_CAPI_EXPORTED MlirOperation oraTStoreOpCreate(
        MlirContext ctx,
        MlirLocation loc,
        MlirValue value,
        MlirStringRef key);

    /// Create an ora.map_get operation using the registered dialect
    MLIR_CAPI_EXPORTED MlirOperation oraMapGetOpCreate(
        MlirContext ctx,
        MlirLocation loc,
        MlirValue map,
        MlirValue key,
        MlirType resultType);

    /// Create an ora.map_store operation using the registered dialect
    MLIR_CAPI_EXPORTED MlirOperation oraMapStoreOpCreate(
        MlirContext ctx,
        MlirLocation loc,
        MlirValue map,
        MlirValue key,
        MlirValue value);

    /// Create an ora.continue operation using the registered dialect
    MLIR_CAPI_EXPORTED MlirOperation oraContinueOpCreate(
        MlirContext ctx,
        MlirLocation loc,
        MlirStringRef label);

    /// Create an ora.return operation using the registered dialect
    MLIR_CAPI_EXPORTED MlirOperation oraReturnOpCreate(
        MlirContext ctx,
        MlirLocation loc,
        const MlirValue *operands,
        size_t numOperands);

    /// Create an ora.decreases operation using the registered dialect
    MLIR_CAPI_EXPORTED MlirOperation oraDecreasesOpCreate(
        MlirContext ctx,
        MlirLocation loc,
        MlirValue measure);

    /// Create an ora.increases operation using the registered dialect
    MLIR_CAPI_EXPORTED MlirOperation oraIncreasesOpCreate(
        MlirContext ctx,
        MlirLocation loc,
        MlirValue measure);

    /// Create an ora.assume operation using the registered dialect
    MLIR_CAPI_EXPORTED MlirOperation oraAssumeOpCreate(
        MlirContext ctx,
        MlirLocation loc,
        MlirValue condition);

    /// Create an ora.havoc operation using the registered dialect
    MLIR_CAPI_EXPORTED MlirOperation oraHavocOpCreate(
        MlirContext ctx,
        MlirLocation loc,
        MlirStringRef variableName);

    /// Create an ora.old operation using the registered dialect
    MLIR_CAPI_EXPORTED MlirOperation oraOldOpCreate(
        MlirContext ctx,
        MlirLocation loc,
        MlirValue value,
        MlirType resultType);

    /// Create an ora.lock operation using the registered dialect
    MLIR_CAPI_EXPORTED MlirOperation oraLockOpCreate(
        MlirContext ctx,
        MlirLocation loc,
        MlirValue resource);

    /// Create an ora.unlock operation using the registered dialect
    MLIR_CAPI_EXPORTED MlirOperation oraUnlockOpCreate(
        MlirContext ctx,
        MlirLocation loc,
        MlirValue resource);

    /// Create an ora.string.constant operation using the registered dialect
    MLIR_CAPI_EXPORTED MlirOperation oraStringConstantOpCreate(
        MlirContext ctx,
        MlirLocation loc,
        MlirStringRef value,
        MlirType resultType);

    /// Create an ora.hex.constant operation using the registered dialect
    MLIR_CAPI_EXPORTED MlirOperation oraHexConstantOpCreate(
        MlirContext ctx,
        MlirLocation loc,
        MlirStringRef value,
        MlirType resultType);

    /// Create an ora.power operation using the registered dialect
    MLIR_CAPI_EXPORTED MlirOperation oraPowerOpCreate(
        MlirContext ctx,
        MlirLocation loc,
        MlirValue base,
        MlirValue exponent,
        MlirType resultType);

    /// Create an ora.const operation using the registered dialect
    MLIR_CAPI_EXPORTED MlirOperation oraConstOpCreate(
        MlirContext ctx,
        MlirLocation loc,
        MlirStringRef name,
        MlirAttribute value,
        MlirType resultType);

    /// Create an ora.immutable operation using the registered dialect
    MLIR_CAPI_EXPORTED MlirOperation oraImmutableOpCreate(
        MlirContext ctx,
        MlirLocation loc,
        MlirStringRef name,
        MlirValue value,
        MlirType resultType);

    /// Create an ora.struct_field_store operation using the registered dialect
    MLIR_CAPI_EXPORTED MlirOperation oraStructFieldStoreOpCreate(
        MlirContext ctx,
        MlirLocation loc,
        MlirValue structValue,
        MlirStringRef fieldName,
        MlirValue value);

    /// Create an ora.struct_init operation using the registered dialect
    MLIR_CAPI_EXPORTED MlirOperation oraStructInitOpCreate(
        MlirContext ctx,
        MlirLocation loc,
        const MlirValue *fieldValues,
        size_t numFieldValues,
        MlirType resultType);

    /// Create an ora.destructure operation
    MLIR_CAPI_EXPORTED MlirOperation oraDestructureOpCreate(
        MlirContext ctx,
        MlirLocation loc,
        MlirValue value,
        MlirStringRef patternType,
        MlirType resultType);

    /// Create an ora.enum.decl operation
    MLIR_CAPI_EXPORTED MlirOperation oraEnumDeclOpCreate(
        MlirContext ctx,
        MlirLocation loc,
        MlirStringRef name,
        MlirType reprType);

    /// Create an ora.enum_constant operation
    MLIR_CAPI_EXPORTED MlirOperation oraEnumConstantOpCreate(
        MlirContext ctx,
        MlirLocation loc,
        MlirStringRef enumName,
        MlirStringRef variantName,
        MlirType resultType);

    /// Create an ora.struct.decl operation
    MLIR_CAPI_EXPORTED MlirOperation oraStructDeclOpCreate(
        MlirContext ctx,
        MlirLocation loc,
        MlirStringRef name);

    /// Create an ora.struct_instantiate operation
    MLIR_CAPI_EXPORTED MlirOperation oraStructInstantiateOpCreate(
        MlirContext ctx,
        MlirLocation loc,
        MlirStringRef structName,
        const MlirValue *fieldValues,
        size_t numFieldValues,
        MlirType resultType);

    /// Create an ora.move operation
    MLIR_CAPI_EXPORTED MlirOperation oraMoveOpCreate(
        MlirContext ctx,
        MlirLocation loc,
        MlirValue amount,
        MlirValue source,
        MlirValue destination);

    /// Create an ora.log operation
    MLIR_CAPI_EXPORTED MlirOperation oraLogOpCreate(
        MlirContext ctx,
        MlirLocation loc,
        MlirStringRef eventName,
        const MlirValue *parameters,
        size_t numParameters);

    /// Create an ora.try_catch operation
    MLIR_CAPI_EXPORTED MlirOperation oraTryOpCreate(
        MlirContext ctx,
        MlirLocation loc,
        MlirValue tryOperation,
        MlirType resultType);

    /// Create an ora.for operation
    MLIR_CAPI_EXPORTED MlirOperation oraForOpCreate(
        MlirContext ctx,
        MlirLocation loc,
        MlirValue collection);

    /// Create an ora.break operation
    MLIR_CAPI_EXPORTED MlirOperation oraBreakOpCreate(
        MlirContext ctx,
        MlirLocation loc,
        MlirStringRef label,
        const MlirValue *values,
        size_t numValues);

    /// Create an ora.switch operation
    MLIR_CAPI_EXPORTED MlirOperation oraSwitchOpCreate(
        MlirContext ctx,
        MlirLocation loc,
        MlirValue value,
        MlirType resultType);

    //===----------------------------------------------------------------------===//
    // CFG Generation with Registered Dialect
    //===----------------------------------------------------------------------===//

    /// Generate Control Flow Graph (DOT format) from MLIR module
    /// Registers the Ora dialect, runs view-op-graph pass with control flow edges, and returns DOT content
    /// Returns null string ref on failure
    /// The caller must free the returned string using mlirStringRefFree
    /// @param includeControlFlow - If true, includes control flow edges (dashed lines showing dominance)
    MLIR_CAPI_EXPORTED MlirStringRef oraGenerateCFG(
        MlirContext ctx,
        MlirModule module,
        bool includeControlFlow);

    //===----------------------------------------------------------------------===//
    // MLIR Printing with Custom Assembly Formats
    //===----------------------------------------------------------------------===//

    /// Print an MLIR operation using the C++ API (ensures custom assembly formats are used)
    /// Registers the Ora dialect and uses OpAsmPrinter which respects custom assembly formats
    /// Returns null string ref on failure
    /// The caller must free the returned string using mlirStringRefFree
    MLIR_CAPI_EXPORTED MlirStringRef oraPrintOperation(
        MlirContext ctx,
        MlirOperation op);

#ifdef __cplusplus
}
#endif

#endif // ORA_DIALECT_C_H

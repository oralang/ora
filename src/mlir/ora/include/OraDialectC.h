//===- OraDialectC.h - C Interface for Ora MLIR Dialect ----------------===//
//
// This file provides a C interface to the Ora MLIR dialect for use from Zig.
// This enables the gradual migration from unregistered to registered operations.
//
//===----------------------------------------------------------------------===//

#ifndef ORA_DIALECT_C_H
#define ORA_DIALECT_C_H

#include "mlir-c/IR.h"
#include "mlir-c/Pass.h"
#include "mlir-c/Support.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

    //===----------------------------------------------------------------------===//
    // Core MLIR helpers (C++ API shims)
    //===----------------------------------------------------------------------===//

    MLIR_CAPI_EXPORTED MlirStringRef oraStringRefCreate(const char *data, size_t length);
    MLIR_CAPI_EXPORTED MlirStringRef oraStringRefCreateFromCString(const char *data);
    MLIR_CAPI_EXPORTED void oraStringRefFree(MlirStringRef ref);

    MLIR_CAPI_EXPORTED MlirIdentifier oraIdentifierGet(MlirContext ctx, MlirStringRef name);
    MLIR_CAPI_EXPORTED MlirStringRef oraIdentifierStr(MlirIdentifier id);
    MLIR_CAPI_EXPORTED MlirNamedAttribute oraNamedAttributeGet(MlirIdentifier name, MlirAttribute attr);

    MLIR_CAPI_EXPORTED MlirLocation oraLocationUnknownGet(MlirContext ctx);
    MLIR_CAPI_EXPORTED MlirLocation oraLocationFileLineColGet(
        MlirContext ctx,
        MlirStringRef filename,
        unsigned line,
        unsigned column);
    MLIR_CAPI_EXPORTED bool oraLocationIsNull(MlirLocation loc);
    /// Returns a newly allocated string; caller must free with oraStringRefFree.
    MLIR_CAPI_EXPORTED MlirStringRef oraLocationPrintToString(MlirLocation loc);

    MLIR_CAPI_EXPORTED MlirContext oraContextCreate();
    MLIR_CAPI_EXPORTED void oraContextDestroy(MlirContext ctx);
    MLIR_CAPI_EXPORTED void oraSetDebugEnabled(bool enabled);
    MLIR_CAPI_EXPORTED MlirDialectRegistry oraDialectRegistryCreate();
    MLIR_CAPI_EXPORTED void oraDialectRegistryDestroy(MlirDialectRegistry registry);
    MLIR_CAPI_EXPORTED void oraRegisterAllDialects(MlirDialectRegistry registry);
    MLIR_CAPI_EXPORTED void oraContextAppendDialectRegistry(MlirContext ctx, MlirDialectRegistry registry);
    MLIR_CAPI_EXPORTED void oraContextLoadAllAvailableDialects(MlirContext ctx);

    MLIR_CAPI_EXPORTED MlirModule oraModuleCreateEmpty(MlirLocation loc);
    MLIR_CAPI_EXPORTED MlirOperation oraModuleGetOperation(MlirModule module);
    MLIR_CAPI_EXPORTED MlirBlock oraModuleGetBody(MlirModule module);
    MLIR_CAPI_EXPORTED bool oraModuleIsNull(MlirModule module);
    MLIR_CAPI_EXPORTED void oraModuleDestroy(MlirModule module);

    MLIR_CAPI_EXPORTED void oraBlockAppendOwnedOperation(MlirBlock block, MlirOperation op);
    MLIR_CAPI_EXPORTED void oraBlockInsertOwnedOperationBefore(MlirBlock block, MlirOperation op, MlirOperation before);
    MLIR_CAPI_EXPORTED MlirOperation oraBlockGetFirstOperation(MlirBlock block);
    MLIR_CAPI_EXPORTED MlirOperation oraBlockGetTerminator(MlirBlock block);
    MLIR_CAPI_EXPORTED size_t oraBlockGetNumArguments(MlirBlock block);
    MLIR_CAPI_EXPORTED MlirValue oraBlockGetArgument(MlirBlock block, size_t index);
    MLIR_CAPI_EXPORTED bool oraBlockIsNull(MlirBlock block);

    MLIR_CAPI_EXPORTED MlirOperation oraOperationGetNextInBlock(MlirOperation op);
    MLIR_CAPI_EXPORTED void oraOperationErase(MlirOperation op);
    MLIR_CAPI_EXPORTED MlirValue oraOperationGetResult(MlirOperation op, size_t index);
    MLIR_CAPI_EXPORTED MlirValue oraOperationGetOperand(MlirOperation op, size_t index);
    MLIR_CAPI_EXPORTED size_t oraOperationGetNumOperands(MlirOperation op);
    MLIR_CAPI_EXPORTED size_t oraOperationGetNumResults(MlirOperation op);
    MLIR_CAPI_EXPORTED size_t oraOperationGetNumRegions(MlirOperation op);
    MLIR_CAPI_EXPORTED MlirStringRef oraOperationGetName(MlirOperation op);
    MLIR_CAPI_EXPORTED bool oraOperationIsNull(MlirOperation op);
    MLIR_CAPI_EXPORTED void oraOperationSetAttributeByName(
        MlirOperation op,
        MlirStringRef name,
        MlirAttribute attr);
    MLIR_CAPI_EXPORTED MlirAttribute oraOperationGetAttributeByName(MlirOperation op, MlirStringRef name);
    MLIR_CAPI_EXPORTED MlirLocation oraOperationGetLocation(MlirOperation op);
    MLIR_CAPI_EXPORTED MlirRegion oraOperationGetRegion(MlirOperation op, size_t index);

    MLIR_CAPI_EXPORTED MlirType oraValueGetType(MlirValue value);
    MLIR_CAPI_EXPORTED bool oraValueIsNull(MlirValue value);
    MLIR_CAPI_EXPORTED bool oraValueIsAOpResult(MlirValue value);
    MLIR_CAPI_EXPORTED MlirOperation oraOpResultGetOwner(MlirValue value);

    MLIR_CAPI_EXPORTED MlirBlock oraRegionGetFirstBlock(MlirRegion region);
    MLIR_CAPI_EXPORTED MlirBlock oraBlockGetNextInRegion(MlirBlock block);
    MLIR_CAPI_EXPORTED bool oraRegionIsNull(MlirRegion region);

    MLIR_CAPI_EXPORTED bool oraAttributeIsNull(MlirAttribute attr);
    MLIR_CAPI_EXPORTED MlirStringRef oraStringAttrGetValue(MlirAttribute attr);
    MLIR_CAPI_EXPORTED int64_t oraIntegerAttrGetValueSInt(MlirAttribute attr);
    MLIR_CAPI_EXPORTED size_t oraArrayAttrGetNumElements(MlirAttribute attr);
    MLIR_CAPI_EXPORTED MlirAttribute oraArrayAttrGetElement(MlirAttribute attr, size_t index);
    /// Returns the integer value as an unsigned decimal string preserving full APInt width.
    /// Returns a newly allocated string; caller must free with oraStringRefFree.
    MLIR_CAPI_EXPORTED MlirStringRef oraIntegerAttrGetValueString(MlirAttribute attr);

    MLIR_CAPI_EXPORTED MlirType oraFunctionTypeGet(
        MlirContext ctx,
        size_t numInputs,
        const MlirType *inputTypes,
        size_t numResults,
        const MlirType *resultTypes);

    /// Returns a newly allocated string; caller must free with oraStringRefFree.
    MLIR_CAPI_EXPORTED MlirStringRef oraOperationPrintToString(MlirOperation op);

    MLIR_CAPI_EXPORTED MlirPassManager oraPassManagerCreate(MlirContext ctx);
    MLIR_CAPI_EXPORTED void oraPassManagerDestroy(MlirPassManager pm);
    MLIR_CAPI_EXPORTED void oraPassManagerEnableVerifier(MlirPassManager pm, bool enable);
    MLIR_CAPI_EXPORTED void oraPassManagerEnableTiming(MlirPassManager pm);
    MLIR_CAPI_EXPORTED bool oraPassManagerParsePipeline(MlirPassManager pm, MlirStringRef pipeline);
    MLIR_CAPI_EXPORTED bool oraPassManagerRun(MlirPassManager pm, MlirOperation op);

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

    MLIR_CAPI_EXPORTED MlirBlock oraContractOpGetBodyBlock(MlirOperation op);

    /// Create a stub module with a contract and empty function (C++ lowering entrypoint)
    MLIR_CAPI_EXPORTED MlirModule oraLowerContractStub(
        MlirContext ctx,
        MlirLocation loc,
        MlirStringRef contractName,
        MlirStringRef funcName);

    typedef enum OraTypeTag
    {
        ORA_TYPE_VOID = 0,
        ORA_TYPE_U256 = 1,
        ORA_TYPE_I256 = 2,
        ORA_TYPE_BOOL = 3,
        ORA_TYPE_ADDRESS = 4
    } OraTypeTag;

    /// Create a stub module with a contract and function signature (C++ lowering entrypoint)
    MLIR_CAPI_EXPORTED MlirModule oraLowerContractStubWithSig(
        MlirContext ctx,
        MlirLocation loc,
        MlirStringRef contractName,
        MlirStringRef funcName,
        const uint32_t *paramTypes,
        size_t numParams,
        const MlirStringRef *paramNames,
        size_t numParamNames,
        uint32_t returnType);

    /// Create an ora.global operation using the registered dialect
    /// Returns null operation if the dialect is not registered or creation fails
    MLIR_CAPI_EXPORTED MlirOperation oraGlobalOpCreate(
        MlirContext ctx,
        MlirLocation loc,
        MlirStringRef name,
        MlirType type,
        MlirAttribute initValue);

    /// Set the name of an operation result (for meaningful SSA variable names)
    /// This improves readability of the generated MLIR
    MLIR_CAPI_EXPORTED void oraOperationSetResultName(
        MlirOperation op,
        unsigned resultIndex,
        MlirStringRef name);

    /// Create an ora.sload operation using the registered dialect
    /// Returns null operation if the dialect is not registered or creation fails
    MLIR_CAPI_EXPORTED MlirOperation oraSLoadOpCreate(
        MlirContext ctx,
        MlirLocation loc,
        MlirStringRef globalName,
        MlirType resultType);

    /// Create an ora.sload operation with a named result
    MLIR_CAPI_EXPORTED MlirOperation oraSLoadOpCreateWithName(
        MlirContext ctx,
        MlirLocation loc,
        MlirStringRef globalName,
        MlirType resultType,
        MlirStringRef resultName);

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

    MLIR_CAPI_EXPORTED MlirBlock oraIfOpGetThenBlock(MlirOperation ifOp);
    MLIR_CAPI_EXPORTED MlirBlock oraIfOpGetElseBlock(MlirOperation ifOp);

    /// Create an ora.isolated_if operation using the registered dialect
    /// Returns null operation if the dialect is not registered or creation fails
    MLIR_CAPI_EXPORTED MlirOperation oraIsolatedIfOpCreate(
        MlirContext ctx,
        MlirLocation loc,
        MlirValue condition);

    MLIR_CAPI_EXPORTED MlirBlock oraIsolatedIfOpGetThenBlock(MlirOperation ifOp);
    MLIR_CAPI_EXPORTED MlirBlock oraIsolatedIfOpGetElseBlock(MlirOperation ifOp);

    /// Create an ora.while operation using the registered dialect
    /// Returns null operation if the dialect is not registered or creation fails
    MLIR_CAPI_EXPORTED MlirOperation oraWhileOpCreate(
        MlirContext ctx,
        MlirLocation loc,
        MlirValue condition);

    MLIR_CAPI_EXPORTED MlirBlock oraWhileOpGetBodyBlock(MlirOperation whileOp);

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

    /// Create an Ora string type
    /// Returns null type if the dialect is not registered or creation fails
    MLIR_CAPI_EXPORTED MlirType oraStringTypeGet(MlirContext ctx);

    /// Create an Ora bytes type
    /// Returns null type if the dialect is not registered or creation fails
    MLIR_CAPI_EXPORTED MlirType oraBytesTypeGet(MlirContext ctx);

    /// Get the type of a global variable from an ora.global operation
    /// Returns null type if the operation is not an ora.global or if it fails
    MLIR_CAPI_EXPORTED MlirType oraGlobalOpGetType(MlirOperation globalOp);

    /// Convert Ora types to built-in MLIR types for arithmetic operations
    /// arith.* operations only accept built-in integer types, not dialect types
    /// Returns the built-in type equivalent (e.g., !ora.int<256, false> -> i256)
    MLIR_CAPI_EXPORTED MlirType oraTypeToBuiltin(MlirType type);

    /// Check if a type is an Ora integer type
    MLIR_CAPI_EXPORTED bool oraTypeIsIntegerType(MlirType type);

    /// Check if a type is an Ora address type
    MLIR_CAPI_EXPORTED bool oraTypeIsAddressType(MlirType type);

    /// Create an ora.addr.to.i160 operation to convert !ora.address to i160
    MLIR_CAPI_EXPORTED MlirOperation oraAddrToI160OpCreate(
        MlirContext ctx,
        MlirLocation loc,
        MlirValue addr);

    /// Create an ora.i160.to.addr operation to convert i160 to !ora.address
    MLIR_CAPI_EXPORTED MlirOperation oraI160ToAddrOpCreate(
        MlirContext ctx,
        MlirLocation loc,
        MlirValue value);

    /// Create an Ora map type !ora.map<keyType, valueType>
    MLIR_CAPI_EXPORTED MlirType oraMapTypeGet(
        MlirContext ctx,
        MlirType keyType,
        MlirType valueType);

    /// Create an Ora struct type !ora.struct<"struct_name">
    MLIR_CAPI_EXPORTED MlirType oraStructTypeGet(
        MlirContext ctx,
        MlirStringRef structName);

    /// Create an Ora error union type !ora.error_union<successType>
    MLIR_CAPI_EXPORTED MlirType oraErrorUnionTypeGet(
        MlirContext ctx,
        MlirType successType);

    /// Extract the success type from an Ora error union type
    /// Returns null type if the input is not an error union type
    MLIR_CAPI_EXPORTED MlirType oraErrorUnionTypeGetSuccessType(MlirType errorUnionType);

    /// Extract the value type from an Ora map type !ora.map<keyType, valueType>
    /// Returns null type if the input is not a map type
    MLIR_CAPI_EXPORTED MlirType oraMapTypeGetValueType(MlirType mapType);

    /// Extract the key type from an Ora map type !ora.map<keyType, valueType>
    /// Returns null type if the input is not a map type
    MLIR_CAPI_EXPORTED MlirType oraMapTypeGetKeyType(MlirType mapType);

    //===----------------------------------------------------------------------===//
    // Refinement Type Creation
    //===----------------------------------------------------------------------===//

    /// Create an Ora min_value refinement type !ora.min_value<baseType, min>
    /// min is passed as four uint64_t values to support full u256 precision:
    /// min = (minHighHigh << 192) | (minHighLow << 128) | (minLowHigh << 64) | minLowLow
    MLIR_CAPI_EXPORTED MlirType oraMinValueTypeGet(
        MlirContext ctx,
        MlirType baseType,
        uint64_t minHighHigh,
        uint64_t minHighLow,
        uint64_t minLowHigh,
        uint64_t minLowLow);

    /// Create an Ora max_value refinement type !ora.max_value<baseType, max>
    /// max is passed as four uint64_t values to support full u256 precision
    MLIR_CAPI_EXPORTED MlirType oraMaxValueTypeGet(
        MlirContext ctx,
        MlirType baseType,
        uint64_t maxHighHigh,
        uint64_t maxHighLow,
        uint64_t maxLowHigh,
        uint64_t maxLowLow);

    /// Create an Ora in_range refinement type !ora.in_range<baseType, min, max>
    /// min and max are passed as four uint64_t values each to support full u256 precision
    MLIR_CAPI_EXPORTED MlirType oraInRangeTypeGet(
        MlirContext ctx,
        MlirType baseType,
        uint64_t minHighHigh,
        uint64_t minHighLow,
        uint64_t minLowHigh,
        uint64_t minLowLow,
        uint64_t maxHighHigh,
        uint64_t maxHighLow,
        uint64_t maxLowHigh,
        uint64_t maxLowLow);

    /// Create an Ora scaled refinement type !ora.scaled<baseType, decimals>
    MLIR_CAPI_EXPORTED MlirType oraScaledTypeGet(
        MlirContext ctx,
        MlirType baseType,
        uint32_t decimals);

    /// Create an Ora exact refinement type !ora.exact<baseType>
    MLIR_CAPI_EXPORTED MlirType oraExactTypeGet(
        MlirContext ctx,
        MlirType baseType);

    /// Create an Ora non_zero_address refinement type !ora.non_zero_address
    MLIR_CAPI_EXPORTED MlirType oraNonZeroAddressTypeGet(MlirContext ctx);

    /// Extract the base type from a refinement type
    /// Returns null type if the input is not a refinement type
    MLIR_CAPI_EXPORTED MlirType oraRefinementTypeGetBaseType(MlirType refinementType);

    /// Create an ora.refinement_to_base operation
    /// Converts a refinement type to its base type
    /// block: The block to insert the operation into (for block arguments, use the block containing the argument)
    MLIR_CAPI_EXPORTED MlirOperation oraRefinementToBaseOpCreate(
        MlirContext ctx,
        MlirLocation loc,
        MlirValue value,
        MlirBlock block);

    /// Create an ora.base_to_refinement operation
    /// Converts a base type value to a refinement type
    /// block: The block to insert the operation into (for block arguments, use the block containing the argument)
    MLIR_CAPI_EXPORTED MlirOperation oraBaseToRefinementOpCreate(
        MlirContext ctx,
        MlirLocation loc,
        MlirValue value,
        MlirType resultType,
        MlirBlock block);

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

    /// Create an ora.refinement_guard operation using the registered dialect
    MLIR_CAPI_EXPORTED MlirOperation oraRefinementGuardOpCreate(
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

    //===----------------------------------------------------------------------===//
    // Standard MLIR Operations (C++ API shim)
    //===----------------------------------------------------------------------===//

    /// Create an arith.constant operation
    MLIR_CAPI_EXPORTED MlirOperation oraArithConstantOpCreate(
        MlirContext ctx,
        MlirLocation loc,
        MlirType resultType,
        MlirAttribute valueAttr);

    /// Create an arith.addi operation
    MLIR_CAPI_EXPORTED MlirOperation oraArithAddIOpCreate(
        MlirContext ctx,
        MlirLocation loc,
        MlirValue lhs,
        MlirValue rhs);

    /// Create an arith.subi operation
    MLIR_CAPI_EXPORTED MlirOperation oraArithSubIOpCreate(
        MlirContext ctx,
        MlirLocation loc,
        MlirValue lhs,
        MlirValue rhs);

    /// Create an arith.muli operation
    MLIR_CAPI_EXPORTED MlirOperation oraArithMulIOpCreate(
        MlirContext ctx,
        MlirLocation loc,
        MlirValue lhs,
        MlirValue rhs);

    /// Create an arith.divui operation
    MLIR_CAPI_EXPORTED MlirOperation oraArithDivUIOpCreate(
        MlirContext ctx,
        MlirLocation loc,
        MlirValue lhs,
        MlirValue rhs);

    /// Create an arith.divsi operation
    MLIR_CAPI_EXPORTED MlirOperation oraArithDivSIOpCreate(
        MlirContext ctx,
        MlirLocation loc,
        MlirValue lhs,
        MlirValue rhs);

    /// Create an arith.remui operation
    MLIR_CAPI_EXPORTED MlirOperation oraArithRemUIOpCreate(
        MlirContext ctx,
        MlirLocation loc,
        MlirValue lhs,
        MlirValue rhs);

    /// Create an arith.remsi operation
    MLIR_CAPI_EXPORTED MlirOperation oraArithRemSIOpCreate(
        MlirContext ctx,
        MlirLocation loc,
        MlirValue lhs,
        MlirValue rhs);

    /// Create an arith.andi operation
    MLIR_CAPI_EXPORTED MlirOperation oraArithAndIOpCreate(
        MlirContext ctx,
        MlirLocation loc,
        MlirValue lhs,
        MlirValue rhs);

    /// Create an arith.ori operation
    MLIR_CAPI_EXPORTED MlirOperation oraArithOrIOpCreate(
        MlirContext ctx,
        MlirLocation loc,
        MlirValue lhs,
        MlirValue rhs);

    /// Create an arith.xori operation
    MLIR_CAPI_EXPORTED MlirOperation oraArithXorIOpCreate(
        MlirContext ctx,
        MlirLocation loc,
        MlirValue lhs,
        MlirValue rhs);

    /// Create an arith.shli operation
    MLIR_CAPI_EXPORTED MlirOperation oraArithShlIOpCreate(
        MlirContext ctx,
        MlirLocation loc,
        MlirValue lhs,
        MlirValue rhs);

    /// Create an arith.shrui operation
    MLIR_CAPI_EXPORTED MlirOperation oraArithShrUIOpCreate(
        MlirContext ctx,
        MlirLocation loc,
        MlirValue lhs,
        MlirValue rhs);

    /// Create an arith.shrsi operation
    MLIR_CAPI_EXPORTED MlirOperation oraArithShrSIOpCreate(
        MlirContext ctx,
        MlirLocation loc,
        MlirValue lhs,
        MlirValue rhs);

    /// Create an arith.bitcast operation
    MLIR_CAPI_EXPORTED MlirOperation oraArithBitcastOpCreate(
        MlirContext ctx,
        MlirLocation loc,
        MlirValue operand,
        MlirType resultType);

    /// Create a builtin.unrealized_conversion_cast operation
    MLIR_CAPI_EXPORTED MlirOperation oraUnrealizedConversionCastOpCreate(
        MlirContext ctx,
        MlirLocation loc,
        MlirValue operand,
        MlirType resultType);

    /// Create an arith.extui operation
    MLIR_CAPI_EXPORTED MlirOperation oraArithExtUIOpCreate(
        MlirContext ctx,
        MlirLocation loc,
        MlirValue operand,
        MlirType resultType);

    /// Create an arith.trunci operation
    MLIR_CAPI_EXPORTED MlirOperation oraArithTruncIOpCreate(
        MlirContext ctx,
        MlirLocation loc,
        MlirValue operand,
        MlirType resultType);

    /// Create an arith.index_castui operation
    MLIR_CAPI_EXPORTED MlirOperation oraArithIndexCastUIOpCreate(
        MlirContext ctx,
        MlirLocation loc,
        MlirValue operand,
        MlirType resultType);

    /// Create an arith.cmpi operation
    MLIR_CAPI_EXPORTED MlirOperation oraArithCmpIOpCreate(
        MlirContext ctx,
        MlirLocation loc,
        int64_t predicate,
        MlirValue lhs,
        MlirValue rhs);

    /// Create a func.call operation
    MLIR_CAPI_EXPORTED MlirOperation oraFuncCallOpCreate(
        MlirContext ctx,
        MlirLocation loc,
        MlirStringRef callee,
        const MlirValue *operands,
        size_t numOperands,
        const MlirType *resultTypes,
        size_t numResults);

    /// Create a memref.alloca operation
    MLIR_CAPI_EXPORTED MlirOperation oraMemrefAllocaOpCreate(
        MlirContext ctx,
        MlirLocation loc,
        MlirType resultType);

    /// Create a memref.load operation
    MLIR_CAPI_EXPORTED MlirOperation oraMemrefLoadOpCreate(
        MlirContext ctx,
        MlirLocation loc,
        MlirValue memref,
        const MlirValue *indices,
        size_t numIndices,
        MlirType resultType);

    MLIR_CAPI_EXPORTED MlirOperation oraMemrefLoadOpCreateWithMemspace(
        MlirContext ctx,
        MlirLocation loc,
        MlirValue memref,
        const MlirValue *indices,
        size_t numIndices,
        MlirType resultType,
        MlirAttribute memspaceAttr);

    /// Create a memref.store operation
    MLIR_CAPI_EXPORTED MlirOperation oraMemrefStoreOpCreate(
        MlirContext ctx,
        MlirLocation loc,
        MlirValue value,
        MlirValue memref,
        const MlirValue *indices,
        size_t numIndices);

    MLIR_CAPI_EXPORTED MlirOperation oraMemrefStoreOpCreateWithMemspace(
        MlirContext ctx,
        MlirLocation loc,
        MlirValue value,
        MlirValue memref,
        const MlirValue *indices,
        size_t numIndices,
        MlirAttribute memspaceAttr);

    /// Create a memref.dim operation
    MLIR_CAPI_EXPORTED MlirOperation oraMemrefDimOpCreate(
        MlirContext ctx,
        MlirLocation loc,
        MlirValue memref,
        MlirValue index);

    /// Create a tensor.dim operation
    MLIR_CAPI_EXPORTED MlirOperation oraTensorDimOpCreate(
        MlirContext ctx,
        MlirLocation loc,
        MlirValue tensor,
        MlirValue index);

    /// Create a scf.yield operation
    MLIR_CAPI_EXPORTED MlirOperation oraScfYieldOpCreate(
        MlirContext ctx,
        MlirLocation loc,
        const MlirValue *operands,
        size_t numOperands);

    /// Create a scf.condition operation
    MLIR_CAPI_EXPORTED MlirOperation oraScfConditionOpCreate(
        MlirContext ctx,
        MlirLocation loc,
        MlirValue condition,
        const MlirValue *operands,
        size_t numOperands);

    /// Create a cf.br operation
    MLIR_CAPI_EXPORTED MlirOperation oraCfBrOpCreate(
        MlirContext ctx,
        MlirLocation loc,
        MlirBlock dest);

    /// Create a cf.cond_br operation
    MLIR_CAPI_EXPORTED MlirOperation oraCfCondBrOpCreate(
        MlirContext ctx,
        MlirLocation loc,
        MlirValue condition,
        MlirBlock true_block,
        MlirBlock false_block);

    /// Create a cf.assert operation
    MLIR_CAPI_EXPORTED MlirOperation oraCfAssertOpCreate(
        MlirContext ctx,
        MlirLocation loc,
        MlirValue condition,
        MlirStringRef message);

    /// Create a cf.assert operation with custom attributes
    MLIR_CAPI_EXPORTED MlirOperation oraCfAssertOpCreateWithAttrs(
        MlirContext ctx,
        MlirLocation loc,
        MlirValue condition,
        const MlirNamedAttribute *attrs,
        size_t numAttrs);

    /// Create a scf.break operation
    MLIR_CAPI_EXPORTED MlirOperation oraScfBreakOpCreate(
        MlirContext ctx,
        MlirLocation loc,
        const MlirValue *operands,
        size_t numOperands);

    /// Create a scf.continue operation
    MLIR_CAPI_EXPORTED MlirOperation oraScfContinueOpCreate(
        MlirContext ctx,
        MlirLocation loc,
        const MlirValue *operands,
        size_t numOperands);

    /// Create a scf.if operation
    MLIR_CAPI_EXPORTED MlirOperation oraScfIfOpCreate(
        MlirContext ctx,
        MlirLocation loc,
        MlirValue condition,
        const MlirType *resultTypes,
        size_t numResults,
        bool withElse);

    MLIR_CAPI_EXPORTED MlirBlock oraScfIfOpGetThenBlock(MlirOperation ifOp);
    MLIR_CAPI_EXPORTED MlirBlock oraScfIfOpGetElseBlock(MlirOperation ifOp);

    /// Create a scf.while operation
    MLIR_CAPI_EXPORTED MlirOperation oraScfWhileOpCreate(
        MlirContext ctx,
        MlirLocation loc,
        const MlirValue *operands,
        size_t numOperands,
        const MlirType *resultTypes,
        size_t numResults);

    MLIR_CAPI_EXPORTED MlirBlock oraScfWhileOpGetBeforeBlock(MlirOperation whileOp);
    MLIR_CAPI_EXPORTED MlirBlock oraScfWhileOpGetAfterBlock(MlirOperation whileOp);

    /// Create a scf.for operation
    MLIR_CAPI_EXPORTED MlirOperation oraScfForOpCreate(
        MlirContext ctx,
        MlirLocation loc,
        MlirValue lowerBound,
        MlirValue upperBound,
        MlirValue step,
        const MlirValue *initArgs,
        size_t numInitArgs,
        bool unsignedCmp);

    MLIR_CAPI_EXPORTED MlirBlock oraScfForOpGetBodyBlock(MlirOperation forOp);

    /// Create a scf.execute_region operation
    MLIR_CAPI_EXPORTED MlirOperation oraScfExecuteRegionOpCreate(
        MlirContext ctx,
        MlirLocation loc,
        const MlirType *resultTypes,
        size_t numResults,
        bool noInline);

    MLIR_CAPI_EXPORTED MlirBlock oraScfExecuteRegionOpGetBodyBlock(MlirOperation op);

    /// Create an llvm.mlir.undef operation
    MLIR_CAPI_EXPORTED MlirOperation oraLlvmUndefOpCreate(
        MlirContext ctx,
        MlirLocation loc,
        MlirType resultType);

    /// Create an llvm.insertvalue operation
    MLIR_CAPI_EXPORTED MlirOperation oraLlvmInsertValueOpCreate(
        MlirContext ctx,
        MlirLocation loc,
        MlirType resultType,
        MlirValue container,
        MlirValue value,
        const int64_t *positions,
        size_t numPositions);

    /// Create an llvm.extractvalue operation
    MLIR_CAPI_EXPORTED MlirOperation oraLlvmExtractValueOpCreate(
        MlirContext ctx,
        MlirLocation loc,
        MlirType resultType,
        MlirValue container,
        const int64_t *positions,
        size_t numPositions);

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

    /// Create an ora.bytes.constant operation using the registered dialect
    MLIR_CAPI_EXPORTED MlirOperation oraBytesConstantOpCreate(
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

    /// Create an ora.struct_field_extract operation using the registered dialect
    MLIR_CAPI_EXPORTED MlirOperation oraStructFieldExtractOpCreate(
        MlirContext ctx,
        MlirLocation loc,
        MlirValue structValue,
        MlirStringRef fieldName,
        MlirType resultType);

    /// Create an ora.struct_field_update operation using the registered dialect
    MLIR_CAPI_EXPORTED MlirOperation oraStructFieldUpdateOpCreate(
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

    /// Create an ora.move operation with mapping operand and result type
    MLIR_CAPI_EXPORTED MlirOperation oraMoveOpCreateWithMapping(
        MlirContext ctx,
        MlirLocation loc,
        MlirValue mapping,
        MlirValue source,
        MlirValue destination,
        MlirValue amount,
        MlirType resultType);

    /// Create an ora.cmp operation
    MLIR_CAPI_EXPORTED MlirOperation oraCmpOpCreate(
        MlirContext ctx,
        MlirLocation loc,
        MlirStringRef predicate,
        MlirValue lhs,
        MlirValue rhs,
        MlirType resultType);

    /// Create an ora.range operation
    MLIR_CAPI_EXPORTED MlirOperation oraRangeOpCreate(
        MlirContext ctx,
        MlirLocation loc,
        MlirValue start,
        MlirValue end,
        MlirType resultType,
        bool inclusive);

    /// Create an ora.quantified operation
    MLIR_CAPI_EXPORTED MlirOperation oraQuantifiedOpCreate(
        MlirContext ctx,
        MlirLocation loc,
        MlirStringRef quantifier,
        MlirStringRef variable,
        MlirStringRef variableType,
        MlirValue condition,
        bool hasCondition,
        MlirValue body,
        MlirType resultType);

    /// Create an ora.error.decl operation
    MLIR_CAPI_EXPORTED MlirOperation oraErrorDeclOpCreate(
        MlirContext ctx,
        MlirLocation loc,
        const MlirType *resultTypes,
        size_t numResults,
        const MlirNamedAttribute *attrs,
        size_t numAttrs);

    /// Create an ora.method_call operation
    MLIR_CAPI_EXPORTED MlirOperation oraMethodCallOpCreate(
        MlirContext ctx,
        MlirLocation loc,
        MlirStringRef methodName,
        const MlirValue *operands,
        size_t numOperands,
        MlirType resultType);

    /// Create an ora.binary.constant operation
    MLIR_CAPI_EXPORTED MlirOperation oraBinaryConstantOpCreate(
        MlirContext ctx,
        MlirLocation loc,
        MlirType resultType,
        const MlirNamedAttribute *attrs,
        size_t numAttrs);

    /// Create an ora.module operation
    MLIR_CAPI_EXPORTED MlirOperation oraModuleOpCreate(
        MlirContext ctx,
        MlirLocation loc,
        const MlirNamedAttribute *attrs,
        size_t numAttrs,
        size_t numRegions,
        bool addEmptyBlocks);

    /// Create an ora.block operation
    MLIR_CAPI_EXPORTED MlirOperation oraBlockOpCreate(
        MlirContext ctx,
        MlirLocation loc,
        const MlirNamedAttribute *attrs,
        size_t numAttrs,
        size_t numRegions,
        bool addEmptyBlocks);

    /// Create an ora.try_block operation
    MLIR_CAPI_EXPORTED MlirOperation oraTryBlockOpCreate(
        MlirContext ctx,
        MlirLocation loc,
        const MlirNamedAttribute *attrs,
        size_t numAttrs,
        size_t numRegions,
        bool addEmptyBlocks);

    /// Create an ora.import operation
    MLIR_CAPI_EXPORTED MlirOperation oraImportOpCreate(
        MlirContext ctx,
        MlirLocation loc,
        const MlirNamedAttribute *attrs,
        size_t numAttrs);

    /// Create an ora.log.decl operation
    MLIR_CAPI_EXPORTED MlirOperation oraLogDeclOpCreate(
        MlirContext ctx,
        MlirLocation loc,
        const MlirNamedAttribute *attrs,
        size_t numAttrs);

    /// Create an ora.quantified operation with regions
    MLIR_CAPI_EXPORTED MlirOperation oraQuantifiedDeclOpCreate(
        MlirContext ctx,
        MlirLocation loc,
        const MlirNamedAttribute *attrs,
        size_t numAttrs,
        size_t numRegions,
        bool addEmptyBlocks);

    /// Create an ora.variable_placeholder operation
    MLIR_CAPI_EXPORTED MlirOperation oraVariablePlaceholderOpCreate(
        MlirContext ctx,
        MlirLocation loc,
        MlirStringRef name,
        MlirType resultType);

    /// Create an ora.module_placeholder operation
    MLIR_CAPI_EXPORTED MlirOperation oraModulePlaceholderOpCreate(
        MlirContext ctx,
        MlirLocation loc,
        MlirStringRef name,
        const MlirNamedAttribute *attrs,
        size_t numAttrs);

    /// Create a tensor.extract operation
    MLIR_CAPI_EXPORTED MlirOperation oraTensorExtractOpCreate(
        MlirContext ctx,
        MlirLocation loc,
        MlirValue tensor,
        const MlirValue *indices,
        size_t numIndices,
        MlirType resultType);

    /// Create an ora.evm.* operation
    MLIR_CAPI_EXPORTED MlirOperation oraEvmOpCreate(
        MlirContext ctx,
        MlirLocation loc,
        MlirStringRef name,
        const MlirValue *operands,
        size_t numOperands,
        MlirType resultType);

    /// Create a StringAttr from a string reference
    MLIR_CAPI_EXPORTED MlirAttribute oraStringAttrCreate(
        MlirContext ctx,
        MlirStringRef value);

    /// Create a BoolAttr
    MLIR_CAPI_EXPORTED MlirAttribute oraBoolAttrCreate(
        MlirContext ctx,
        bool value);

    /// Create an IntegerAttr from a 64-bit value
    MLIR_CAPI_EXPORTED MlirAttribute oraIntegerAttrCreateI64(
        MlirContext ctx,
        MlirType type,
        int64_t value);

    /// Create an IntegerAttr from a 64-bit value using the type's context
    MLIR_CAPI_EXPORTED MlirAttribute oraIntegerAttrCreateI64FromType(
        MlirType type,
        int64_t value);

    /// Create a TypeAttr
    MLIR_CAPI_EXPORTED MlirAttribute oraTypeAttrCreate(
        MlirContext ctx,
        MlirType type);

    /// Create a TypeAttr using the type's context
    MLIR_CAPI_EXPORTED MlirAttribute oraTypeAttrCreateFromType(
        MlirType type);

    /// Create an ArrayAttr
    MLIR_CAPI_EXPORTED MlirAttribute oraArrayAttrCreate(
        MlirContext ctx,
        intptr_t numAttrs,
        const MlirAttribute *attrs);

    /// Create a null attribute
    MLIR_CAPI_EXPORTED MlirAttribute oraNullAttrCreate(void);

    /// Create a signless integer type
    MLIR_CAPI_EXPORTED MlirType oraIntegerTypeCreate(
        MlirContext ctx,
        uint32_t bits);

    /// Create an index type
    MLIR_CAPI_EXPORTED MlirType oraIndexTypeCreate(MlirContext ctx);

    /// Create a none type
    MLIR_CAPI_EXPORTED MlirType oraNoneTypeCreate(MlirContext ctx);

    /// Create a ranked tensor type
    MLIR_CAPI_EXPORTED MlirType oraRankedTensorTypeCreate(
        MlirContext ctx,
        intptr_t rank,
        const int64_t *shape,
        MlirType elementType,
        MlirAttribute encoding);

    /// Create a memref type
    MLIR_CAPI_EXPORTED MlirType oraMemRefTypeCreate(
        MlirContext ctx,
        MlirType elementType,
        intptr_t rank,
        const int64_t *shape,
        MlirAttribute layout,
        MlirAttribute memorySpace);

    /// Return the shaped dynamic size sentinel
    MLIR_CAPI_EXPORTED int64_t oraShapedTypeDynamicSize(void);

    /// Query: is integer type
    MLIR_CAPI_EXPORTED bool oraTypeIsAInteger(MlirType type);

    /// Query: type equality
    MLIR_CAPI_EXPORTED bool oraTypeEqual(MlirType a, MlirType b);

    /// Query: is shaped type
    MLIR_CAPI_EXPORTED bool oraTypeIsAShaped(MlirType type);

    /// Query: is memref type
    MLIR_CAPI_EXPORTED bool oraTypeIsAMemRef(MlirType type);

    /// Query: shaped element type
    MLIR_CAPI_EXPORTED MlirType oraShapedTypeGetElementType(MlirType type);

    /// Query: shaped rank
    MLIR_CAPI_EXPORTED intptr_t oraShapedTypeGetRank(MlirType type);

    /// Query: shaped dim size
    MLIR_CAPI_EXPORTED int64_t oraShapedTypeGetDimSize(MlirType type, intptr_t dim);

    /// Query: shaped has static shape
    MLIR_CAPI_EXPORTED bool oraShapedTypeHasStaticShape(MlirType type);

    /// Query: integer type width
    MLIR_CAPI_EXPORTED uint32_t oraIntegerTypeGetWidth(MlirType type);

    /// Query: type is null
    MLIR_CAPI_EXPORTED bool oraTypeIsNull(MlirType type);

    /// Query: type is none
    MLIR_CAPI_EXPORTED bool oraTypeIsANone(MlirType type);

    /// Query: is enum type
    MLIR_CAPI_EXPORTED bool oraTypeIsAEnum(MlirType type);

    /// Get representation type from enum type
    MLIR_CAPI_EXPORTED MlirType oraEnumTypeGetReprType(MlirType enumType);

    /// Query: is ora integer type
    MLIR_CAPI_EXPORTED bool oraTypeIsAOraInteger(MlirType type);

    /// Create an ora.const operation with optional regions
    MLIR_CAPI_EXPORTED MlirOperation oraConstDeclOpCreate(
        MlirContext ctx,
        MlirLocation loc,
        const MlirType *resultTypes,
        size_t numResults,
        const MlirNamedAttribute *attrs,
        size_t numAttrs,
        size_t numRegions,
        bool addEmptyBlocks);

    /// Create an ora.immutable operation with optional regions
    MLIR_CAPI_EXPORTED MlirOperation oraImmutableDeclOpCreate(
        MlirContext ctx,
        MlirLocation loc,
        const MlirType *resultTypes,
        size_t numResults,
        const MlirNamedAttribute *attrs,
        size_t numAttrs,
        size_t numRegions,
        bool addEmptyBlocks);

    /// Create an ora.memory.global operation
    MLIR_CAPI_EXPORTED MlirOperation oraMemoryGlobalOpCreate(
        MlirContext ctx,
        MlirLocation loc,
        MlirStringRef name,
        MlirType type);

    /// Create an ora.tstore.global operation
    MLIR_CAPI_EXPORTED MlirOperation oraTStoreGlobalOpCreate(
        MlirContext ctx,
        MlirLocation loc,
        MlirStringRef name,
        MlirType type);

    /// Create an ora.error placeholder operation
    MLIR_CAPI_EXPORTED MlirOperation oraErrorOpCreate(
        MlirContext ctx,
        MlirLocation loc,
        MlirType resultType);

    /// Create an ora.length operation
    MLIR_CAPI_EXPORTED MlirOperation oraLengthOpCreate(
        MlirContext ctx,
        MlirLocation loc,
        MlirValue value,
        MlirType resultType);

    /// Create an ora.expression_capture operation
    MLIR_CAPI_EXPORTED MlirOperation oraExpressionCaptureOpCreate(
        MlirContext ctx,
        MlirLocation loc,
        MlirValue value,
        MlirType resultType);

    /// Create an ora.log operation
    MLIR_CAPI_EXPORTED MlirOperation oraLogOpCreate(
        MlirContext ctx,
        MlirLocation loc,
        MlirStringRef eventName,
        const MlirValue *parameters,
        size_t numParameters);

    /// Create a func.func operation
    MLIR_CAPI_EXPORTED MlirOperation oraFuncFuncOpCreate(
        MlirContext ctx,
        MlirLocation loc,
        const MlirNamedAttribute *attrs,
        size_t numAttrs,
        const MlirType *paramTypes,
        const MlirLocation *paramLocs,
        size_t numParams);

    MLIR_CAPI_EXPORTED MlirBlock oraFuncOpGetBodyBlock(MlirOperation op);

    /// Create a generic operation (C++ API)
    MLIR_CAPI_EXPORTED MlirOperation oraOperationCreate(
        MlirContext ctx,
        MlirLocation loc,
        MlirStringRef name,
        const MlirValue *operands,
        size_t numOperands,
        const MlirType *resultTypes,
        size_t numResults,
        const MlirNamedAttribute *attrs,
        size_t numAttrs,
        size_t numRegions,
        bool addEmptyBlocks);

    MLIR_CAPI_EXPORTED MlirBlock oraOperationGetRegionBlock(MlirOperation op, size_t index);

    /// Create an ora.error.ok operation
    MLIR_CAPI_EXPORTED MlirOperation oraErrorOkOpCreate(
        MlirContext ctx,
        MlirLocation loc,
        MlirValue value,
        MlirType resultType);

    /// Create an ora.error.err operation
    MLIR_CAPI_EXPORTED MlirOperation oraErrorErrOpCreate(
        MlirContext ctx,
        MlirLocation loc,
        MlirValue value,
        MlirType resultType);

    /// Create an ora.error.is_error operation
    MLIR_CAPI_EXPORTED MlirOperation oraErrorIsErrorOpCreate(
        MlirContext ctx,
        MlirLocation loc,
        MlirValue value);

    /// Create an ora.error.unwrap operation
    MLIR_CAPI_EXPORTED MlirOperation oraErrorUnwrapOpCreate(
        MlirContext ctx,
        MlirLocation loc,
        MlirValue value,
        MlirType resultType);

    /// Create an ora.error.get_error operation
    MLIR_CAPI_EXPORTED MlirOperation oraErrorGetErrorOpCreate(
        MlirContext ctx,
        MlirLocation loc,
        MlirValue value,
        MlirType resultType);

    /// Create an ora.try_catch operation
    MLIR_CAPI_EXPORTED MlirOperation oraTryOpCreate(
        MlirContext ctx,
        MlirLocation loc,
        MlirValue tryOperation,
        MlirType resultType);

    MLIR_CAPI_EXPORTED MlirBlock oraTryOpGetTryBlock(MlirOperation tryOp);

MLIR_CAPI_EXPORTED MlirBlock oraTryOpGetCatchBlock(MlirOperation tryOp);

/// Create an ora.try_stmt operation
MLIR_CAPI_EXPORTED MlirOperation oraTryStmtOpCreate(
    MlirContext ctx,
    MlirLocation loc,
    const MlirType *resultTypes,
    size_t numResults);

MLIR_CAPI_EXPORTED MlirBlock oraTryStmtOpGetTryBlock(MlirOperation tryStmtOp);

MLIR_CAPI_EXPORTED MlirBlock oraTryStmtOpGetCatchBlock(MlirOperation tryStmtOp);

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

    MLIR_CAPI_EXPORTED MlirOperation oraSwitchOpCreateWithCases(
        MlirContext ctx,
        MlirLocation loc,
        MlirValue value,
        const MlirType *resultTypes,
        size_t numResults,
        size_t numCases);

    MLIR_CAPI_EXPORTED MlirBlock oraSwitchOpGetCaseBlock(MlirOperation switchOp, size_t index);

    /// Set case pattern attributes on an ora.switch or ora.switch_expr operation
    /// caseValues: array of literal case values (for literal patterns), length = numCases
    /// rangeStarts: array of range start values (for range patterns), length = numCases
    /// rangeEnds: array of range end values (for range patterns), length = numCases
    /// caseKinds: array indicating case type (0=literal, 1=range, 2=else), length = numCases
    /// defaultCaseIndex: index of the default/else case (-1 if none)
    /// numCases: number of cases
    MLIR_CAPI_EXPORTED void oraSwitchOpSetCasePatterns(
        MlirOperation op,
        const int64_t *caseValues,
        const int64_t *rangeStarts,
        const int64_t *rangeEnds,
        const int64_t *caseKinds,
        int64_t defaultCaseIndex,
        size_t numCases);

    /// Create an ora.switch_expr operation
    MLIR_CAPI_EXPORTED MlirOperation oraSwitchExprOpCreate(
        MlirContext ctx,
        MlirLocation loc,
        MlirValue value,
        MlirType resultType);

    MLIR_CAPI_EXPORTED MlirOperation oraSwitchExprOpCreateWithCases(
        MlirContext ctx,
        MlirLocation loc,
        MlirValue value,
        const MlirType *resultTypes,
        size_t numResults,
        size_t numCases);

    MLIR_CAPI_EXPORTED MlirBlock oraSwitchExprOpGetCaseBlock(MlirOperation switchOp, size_t index);

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
    //===----------------------------------------------------------------------===//
    // Ora Canonicalization
    //===----------------------------------------------------------------------===//

    /// Run canonicalization on Ora MLIR (folds constants in ora.add, ora.mul, etc.)
    /// Registers Ora dialect and runs canonicalization pass
    /// Returns true on success, false on failure
    MLIR_CAPI_EXPORTED bool oraCanonicalizeOraMLIR(
        MlirContext ctx,
        MlirModule module);

    //===----------------------------------------------------------------------===//
    // Ora to SIR Conversion
    //===----------------------------------------------------------------------===//

    /// Convert Ora dialect operations to SIR dialect
    /// Registers both Ora and SIR dialects, runs the conversion pass, and returns success status
    /// Returns true on success, false on failure
    MLIR_CAPI_EXPORTED bool oraConvertToSIR(
        MlirContext ctx,
        MlirModule module);

    //===----------------------------------------------------------------------===//
    // SIR Text Legalizer / Emitter
    //===----------------------------------------------------------------------===//

    /// Validate SIR MLIR for Sensei text emission
    /// Returns true on success, false on failure
    MLIR_CAPI_EXPORTED bool oraLegalizeSIRText(
        MlirContext ctx,
        MlirModule module);

    /// Build Solidity-style dispatcher for public functions
    /// Returns true on success, false on failure
    MLIR_CAPI_EXPORTED bool oraBuildSIRDispatcher(
        MlirContext ctx,
        MlirModule module);

    /// Emit Sensei SIR text from a SIR MLIR module
    /// Returns null string ref on failure
    /// The caller must free the returned string using oraStringRefFree
    MLIR_CAPI_EXPORTED MlirStringRef oraEmitSIRText(
        MlirContext ctx,
        MlirModule module);

    //===----------------------------------------------------------------------===//
    // Integer Attribute Creation with Full Precision (u256 support)
    //===----------------------------------------------------------------------===//

    /// Create an MLIR IntegerAttr from a string representation
    /// This supports full u256 precision (unlike mlirIntegerAttrGet which only accepts i64)
    /// The string should be a decimal representation of the integer value
    /// Returns null attribute if parsing fails or type is invalid
    MLIR_CAPI_EXPORTED MlirAttribute oraIntegerAttrGetFromString(
        MlirType type,
        MlirStringRef valueStr);

    //===----------------------------------------------------------------------===//
    // Function Argument and Result Attributes
    //===----------------------------------------------------------------------===//

    /// Set an attribute on a function argument
    /// This is used to attach semantic type information (e.g., ora.type) to function parameters
    /// Returns true on success, false on failure
    MLIR_CAPI_EXPORTED bool oraFuncSetArgAttr(
        MlirOperation funcOp,
        unsigned argIndex,
        MlirStringRef attrName,
        MlirAttribute attr);

    /// Set an attribute on a function result
    /// This is used to attach semantic type information (e.g., ora.type) to function return values
    /// Returns true on success, false on failure
    MLIR_CAPI_EXPORTED bool oraFuncSetResultAttr(
        MlirOperation funcOp,
        unsigned resultIndex,
        MlirStringRef attrName,
        MlirAttribute attr);

#ifdef __cplusplus
}
#endif

#endif // ORA_DIALECT_C_H

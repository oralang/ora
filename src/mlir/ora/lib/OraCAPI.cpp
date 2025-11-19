//===- OraDialectC.cpp - C Interface for Ora MLIR Dialect --------------===//
//
// This file implements the C interface to the Ora MLIR dialect.
//
//===----------------------------------------------------------------------===//

#include "OraDialectC.h"
#include "OraDialect.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "mlir/CAPI/Pass.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/AsmState.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include <sstream>
#include "mlir/Transforms/ViewOpGraph.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <memory>

using namespace mlir;
using namespace mlir::ora;

//===----------------------------------------------------------------------===//
// Ora Dialect Registration
//===----------------------------------------------------------------------===//

bool oraDialectRegister(MlirContext ctx)
{
    try
    {
        MLIRContext *context = unwrap(ctx);
        // Register the Ora dialect
        context->getOrLoadDialect<OraDialect>();
        return true;
    }
    catch (...)
    {
        return false;
    }
}

bool oraDialectIsRegistered(MlirContext ctx)
{
    MLIRContext *context = unwrap(ctx);
    return context->getLoadedDialect("ora") != nullptr;
}

MlirDialect oraDialectGet(MlirContext ctx)
{
    MLIRContext *context = unwrap(ctx);
    auto *dialect = context->getLoadedDialect("ora");
    if (!dialect)
    {
        return {nullptr};
    }
    return wrap(dialect);
}

//===----------------------------------------------------------------------===//
// Ora Operation Creation (Registered Dialect)
//===----------------------------------------------------------------------===//

MlirOperation oraContractOpCreate(MlirContext ctx, MlirLocation loc, MlirStringRef name)
{
    try
    {
        MLIRContext *context = unwrap(ctx);
        Location location = unwrap(loc);
        StringRef nameRef = unwrap(name);

        // Check if dialect is registered
        if (!oraDialectIsRegistered(ctx))
        {
            return {nullptr};
        }

        OpBuilder builder(context);

        // Create the contract operation
        auto contractOp = builder.create<ContractOp>(location, nameRef);

        // Ensure body region has a block (ContractOp requires a region)
        auto &bodyRegion = contractOp.getBody();
        if (bodyRegion.empty())
        {
            bodyRegion.push_back(new Block());
        }

        // Add gas_cost attribute (contract declaration has no runtime cost = 0)
        auto gasCostAttr = IntegerAttr::get(::mlir::IntegerType::get(context, 64), 0);
        contractOp->setAttr("gas_cost", gasCostAttr);

        return wrap(contractOp.getOperation());
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirOperation oraGlobalOpCreate(MlirContext ctx, MlirLocation loc, MlirStringRef name, MlirType type, MlirAttribute initValue)
{
    try
    {
        MLIRContext *context = unwrap(ctx);
        Location location = unwrap(loc);
        StringRef nameRef = unwrap(name);
        Type typeRef = unwrap(type);

        // Check if dialect is registered
        if (!oraDialectIsRegistered(ctx))
        {
            return {nullptr};
        }

        OpBuilder builder(context);

        // Create the global operation
        auto nameAttr = StringAttr::get(context, nameRef);
        auto typeAttr = TypeAttr::get(typeRef);

        // Handle optional initializer - use UnitAttr for null (maps don't have initializers)
        Attribute initAttr;
        if (mlirAttributeIsNull(initValue))
        {
            initAttr = UnitAttr::get(context);
        }
        else
        {
            initAttr = unwrap(initValue);
        }

        auto globalOp = builder.create<GlobalOp>(location, nameAttr, typeAttr, initAttr);

        // Add gas_cost attribute (global variable declaration has minimal cost = 0)
        // Actual access costs are handled by sload/sstore operations
        auto gasCostAttr = IntegerAttr::get(::mlir::IntegerType::get(context, 64), 0);
        globalOp->setAttr("gas_cost", gasCostAttr);

        return wrap(globalOp.getOperation());
    }
    catch (...)
    {
        return {nullptr};
    }
}

void oraOperationSetResultName(MlirOperation op, unsigned resultIndex, MlirStringRef name)
{
    try
    {
        Operation *operation = unwrap(op);
        StringRef nameRef = unwrap(name);

        if (resultIndex < operation->getNumResults())
        {
            // In MLIR, result names are set through attributes or OpAsmOpInterface
            // For now, we'll store the name as an attribute that can be used during printing
            // The actual name display is handled by AsmState during printing
            auto nameAttr = StringAttr::get(operation->getContext(), nameRef);
            std::string attrName = "ora.result_name_" + std::to_string(resultIndex);
            operation->setAttr(attrName, nameAttr);

            // Also try to set it through the result's name if the API exists
            // Note: OpResult doesn't have setName() directly, names are managed by AsmState
        }
    }
    catch (...)
    {
        // Ignore errors - name setting is optional
    }
}

MlirOperation oraSLoadOpCreate(MlirContext ctx, MlirLocation loc, MlirStringRef globalName, MlirType resultType)
{
    return oraSLoadOpCreateWithName(ctx, loc, globalName, resultType, {nullptr, 0});
}

MlirOperation oraSLoadOpCreateWithName(MlirContext ctx, MlirLocation loc, MlirStringRef globalName, MlirType resultType, MlirStringRef resultName)
{
    try
    {
        MLIRContext *context = unwrap(ctx);
        Location location = unwrap(loc);
        StringRef nameRef = unwrap(globalName);
        Type typeRef = unwrap(resultType);

        // Check if dialect is registered
        if (!oraDialectIsRegistered(ctx))
        {
            return {nullptr};
        }

        OpBuilder builder(context);

        // Create the sload operation
        auto sloadOp = builder.create<SLoadOp>(location, typeRef, StringAttr::get(context, nameRef));

        // Add gas_cost attribute (SLOAD_COLD = 2100, conservative estimate)
        auto gasCostAttr = IntegerAttr::get(::mlir::IntegerType::get(context, 64), 2100);
        sloadOp->setAttr("gas_cost", gasCostAttr);

        // Set result name if provided (stored as attribute for use during printing)
        if (resultName.data != nullptr && resultName.length > 0)
        {
            StringRef resultNameRef = unwrap(resultName);
            auto nameAttr = StringAttr::get(context, resultNameRef);
            sloadOp->setAttr("ora.result_name_0", nameAttr);
        }

        return wrap(sloadOp.getOperation());
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirOperation oraSStoreOpCreate(MlirContext ctx, MlirLocation loc, MlirValue value, MlirStringRef globalName)
{
    try
    {
        MLIRContext *context = unwrap(ctx);
        Location location = unwrap(loc);
        Value valueRef = unwrap(value);
        StringRef nameRef = unwrap(globalName);

        // Check if dialect is registered
        if (!oraDialectIsRegistered(ctx))
        {
            return {nullptr};
        }

        OpBuilder builder(context);

        // Create the sstore operation
        auto sstoreOp = builder.create<SStoreOp>(location, valueRef, StringAttr::get(context, nameRef));

        // Add gas_cost attribute (SSTORE_NEW = 20000, conservative worst-case estimate)
        auto gasCostAttr = IntegerAttr::get(::mlir::IntegerType::get(context, 64), 20000);
        sstoreOp->setAttr("gas_cost", gasCostAttr);

        return wrap(sstoreOp.getOperation());
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirOperation oraIfOpCreate(MlirContext ctx, MlirLocation loc, MlirValue condition)
{
    try
    {
        MLIRContext *context = unwrap(ctx);
        Location location = unwrap(loc);
        Value conditionRef = unwrap(condition);

        // Check if dialect is registered
        if (!oraDialectIsRegistered(ctx))
        {
            return {nullptr};
        }

        OpBuilder builder(context);

        // Create the if operation with two regions (then and else)
        // IfOp::create requires TypeRange for results (empty for now, can be added later)
        TypeRange emptyResults;
        auto ifOp = IfOp::create(builder, location, emptyResults, conditionRef);

        // Ensure regions have blocks (IfOp::create creates empty regions)
        // Get then and else regions
        auto &thenRegion = ifOp.getThenRegion();
        auto &elseRegion = ifOp.getElseRegion();

        // Create blocks for both regions if they don't exist
        if (thenRegion.empty())
        {
            thenRegion.push_back(new Block());
        }
        if (elseRegion.empty())
        {
            elseRegion.push_back(new Block());
        }

        // Add gas_cost attribute (branch operation has cost = 10)
        auto gasCostAttr = IntegerAttr::get(::mlir::IntegerType::get(context, 64), 10);
        ifOp->setAttr("gas_cost", gasCostAttr);

        return wrap(ifOp.getOperation());
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirOperation oraWhileOpCreate(MlirContext ctx, MlirLocation loc, MlirValue condition)
{
    try
    {
        MLIRContext *context = unwrap(ctx);
        Location location = unwrap(loc);
        Value conditionRef = unwrap(condition);

        // Check if dialect is registered
        if (!oraDialectIsRegistered(ctx))
        {
            return {nullptr};
        }

        OpBuilder builder(context);

        // Create the while operation with body region
        auto whileOp = WhileOp::create(builder, location, conditionRef);

        // Ensure body region has a block (WhileOp::create creates empty region)
        auto &bodyRegion = whileOp.getBody();
        if (bodyRegion.empty())
        {
            bodyRegion.push_back(new Block());
        }

        // Add gas_cost attribute (loop operation has cost = 10)
        auto gasCostAttr = IntegerAttr::get(::mlir::IntegerType::get(context, 64), 10);
        whileOp->setAttr("gas_cost", gasCostAttr);

        return wrap(whileOp.getOperation());
    }
    catch (...)
    {
        return {nullptr};
    }
}

//===----------------------------------------------------------------------===//
// TestOp Creation (Simple Test)
//===----------------------------------------------------------------------===//

MlirOperation oraTestOpCreate(MlirContext ctx, MlirLocation loc)
{
    try
    {
        MLIRContext *context = unwrap(ctx);
        Location location = unwrap(loc);

        // Check if dialect is registered
        if (!oraDialectIsRegistered(ctx))
        {
            return {nullptr};
        }

        OpBuilder builder(context);

        // Create the test operation (simplest possible - no operands, no results)
        auto testOp = builder.create<TestOp>(location);

        return wrap(testOp.getOperation());
    }
    catch (...)
    {
        return {nullptr};
    }
}

//===----------------------------------------------------------------------===//
// Ora Type Creation
//===----------------------------------------------------------------------===//

MlirType oraIntegerTypeGet(MlirContext ctx, unsigned width, bool isSigned)
{
    try
    {
        MLIRContext *context = unwrap(ctx);

        // Check if dialect is registered
        if (!oraDialectIsRegistered(ctx))
        {
            return {nullptr};
        }

        // Create the Ora integer type
        auto oraIntType = ora::IntegerType::get(context, width, isSigned);
        return wrap(oraIntType);
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirType oraBoolTypeGet(MlirContext ctx)
{
    try
    {
        MLIRContext *context = unwrap(ctx);

        // Check if dialect is registered
        if (!oraDialectIsRegistered(ctx))
        {
            return {nullptr};
        }

        // Create the Ora boolean type
        auto boolType = ora::BoolType::get(context);
        return wrap(boolType);
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirType oraAddressTypeGet(MlirContext ctx)
{
    try
    {
        MLIRContext *context = unwrap(ctx);

        // Check if dialect is registered
        if (!oraDialectIsRegistered(ctx))
        {
            return {nullptr};
        }

        // Create the Ora address type
        auto addressType = ora::AddressType::get(context);
        return wrap(addressType);
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirType oraGlobalOpGetType(MlirOperation globalOp)
{
    try
    {
        Operation *op = unwrap(globalOp);

        // Check if this is an ora.global operation
        if (op->getName().getStringRef() != "ora.global")
        {
            return {nullptr};
        }

        // Cast to GlobalOp and get the type
        if (auto globalOp = dyn_cast<ora::GlobalOp>(op))
        {
            Type globalType = globalOp.getGlobalType();
            return wrap(globalType);
        }

        return {nullptr};
    }
    catch (...)
    {
        return {nullptr};
    }
}

/// Convert Ora types to built-in MLIR types for arithmetic operations
/// arith.* operations only accept built-in integer types, not dialect types
MlirType oraTypeToBuiltin(MlirType type)
{
    try
    {
        Type mlirType = unwrap(type);

        // Check if it's an Ora integer type
        if (auto oraIntType = dyn_cast<ora::IntegerType>(mlirType))
        {
            // Convert to built-in integer type
            unsigned width = oraIntType.getWidth();
            bool isSigned = oraIntType.getIsSigned();
            return wrap(::mlir::IntegerType::get(mlirType.getContext(), width, isSigned ? ::mlir::IntegerType::Signed : ::mlir::IntegerType::Unsigned));
        }

        // Check if it's an Ora address type
        if (auto oraAddrType = dyn_cast<ora::AddressType>(mlirType))
        {
            // Convert to i160 (Ethereum address is 20 bytes = 160 bits)
            return wrap(::mlir::IntegerType::get(mlirType.getContext(), 160));
        }

        // Check if it's an Ora bool type
        if (auto oraBoolType = dyn_cast<ora::BoolType>(mlirType))
        {
            // Convert to i1
            return wrap(::mlir::IntegerType::get(mlirType.getContext(), 1));
        }

        // Not an Ora type, return as-is
        return type;
    }
    catch (...)
    {
        return type;
    }
}

/// Check if a type is an Ora integer type
bool oraTypeIsIntegerType(MlirType type)
{
    try
    {
        Type mlirType = unwrap(type);
        return dyn_cast<ora::IntegerType>(mlirType) != nullptr;
    }
    catch (...)
    {
        return false;
    }
}

/// Check if a type is an Ora address type
bool oraTypeIsAddressType(MlirType type)
{
    try
    {
        Type mlirType = unwrap(type);
        return dyn_cast<ora::AddressType>(mlirType) != nullptr;
    }
    catch (...)
    {
        return false;
    }
}

/// Create an ora.addr.to.i160 operation to convert !ora.address to i160
MlirOperation oraAddrToI160OpCreate(MlirContext ctx, MlirLocation loc, MlirValue addr)
{
    try
    {
        MLIRContext *context = unwrap(ctx);
        Location location = unwrap(loc);
        Value addrValue = unwrap(addr);

        if (!oraDialectIsRegistered(ctx))
        {
            return {nullptr};
        }

        OpBuilder builder(context);

        // Create i160 result type
        auto i160Type = ::mlir::IntegerType::get(context, 160);

        // Create the addr.to.i160 operation
        auto addrToI160Op = builder.create<ora::AddrToI160Op>(location, i160Type, addrValue);

        return wrap(addrToI160Op.getOperation());
    }
    catch (...)
    {
        return {nullptr};
    }
}

/// Create an Ora map type !ora.map<keyType, valueType>
MlirType oraMapTypeGet(MlirContext ctx, MlirType keyType, MlirType valueType)
{
    try
    {
        MLIRContext *context = unwrap(ctx);
        Type keyTypeRef = unwrap(keyType);
        Type valueTypeRef = unwrap(valueType);

        if (!oraDialectIsRegistered(ctx))
        {
            return {nullptr};
        }

        // Create the map type
        auto mapType = ora::MapType::get(context, keyTypeRef, valueTypeRef);
        return wrap(mapType);
    }
    catch (...)
    {
        return {nullptr};
    }
}

//===----------------------------------------------------------------------===//
// Verification Operations
//===----------------------------------------------------------------------===//

MlirOperation oraRequiresOpCreate(MlirContext ctx, MlirLocation loc, MlirValue condition)
{
    try
    {
        MLIRContext *context = unwrap(ctx);
        Location location = unwrap(loc);
        Value conditionRef = unwrap(condition);

        if (!oraDialectIsRegistered(ctx))
        {
            return {nullptr};
        }

        OpBuilder builder(context);
        auto requiresOp = builder.create<RequiresOp>(location, conditionRef);

        // Add gas_cost attribute (verification operation has no runtime cost = 0)
        auto gasCostAttr = IntegerAttr::get(::mlir::IntegerType::get(context, 64), 0);
        requiresOp->setAttr("gas_cost", gasCostAttr);

        return wrap(requiresOp.getOperation());
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirOperation oraEnsuresOpCreate(MlirContext ctx, MlirLocation loc, MlirValue condition)
{
    try
    {
        MLIRContext *context = unwrap(ctx);
        Location location = unwrap(loc);
        Value conditionRef = unwrap(condition);

        if (!oraDialectIsRegistered(ctx))
        {
            return {nullptr};
        }

        OpBuilder builder(context);
        auto ensuresOp = builder.create<EnsuresOp>(location, conditionRef);

        // Add gas_cost attribute (verification operation has no runtime cost = 0)
        auto gasCostAttr = IntegerAttr::get(::mlir::IntegerType::get(context, 64), 0);
        ensuresOp->setAttr("gas_cost", gasCostAttr);

        return wrap(ensuresOp.getOperation());
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirOperation oraInvariantOpCreate(MlirContext ctx, MlirLocation loc, MlirValue condition)
{
    try
    {
        MLIRContext *context = unwrap(ctx);
        Location location = unwrap(loc);
        Value conditionRef = unwrap(condition);

        if (!oraDialectIsRegistered(ctx))
        {
            return {nullptr};
        }

        OpBuilder builder(context);
        auto invariantOp = builder.create<InvariantOp>(location, conditionRef);

        // Add gas_cost attribute (verification operation has no runtime cost = 0)
        auto gasCostAttr = IntegerAttr::get(::mlir::IntegerType::get(context, 64), 0);
        invariantOp->setAttr("gas_cost", gasCostAttr);

        return wrap(invariantOp.getOperation());
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirOperation oraAssertOpCreate(MlirContext ctx, MlirLocation loc, MlirValue condition, MlirStringRef message)
{
    try
    {
        MLIRContext *context = unwrap(ctx);
        Location location = unwrap(loc);
        Value conditionRef = unwrap(condition);

        if (!oraDialectIsRegistered(ctx))
        {
            return {nullptr};
        }

        OpBuilder builder(context);

        // Create assert operation - message is optional
        StringAttr messageAttr = nullptr;
        if (message.data != nullptr && message.length > 0)
        {
            messageAttr = StringAttr::get(context, unwrap(message));
        }

        auto assertOp = builder.create<AssertOp>(location, conditionRef, messageAttr);

        // Add gas_cost attribute (assert operation has no runtime cost = 0)
        auto gasCostAttr = IntegerAttr::get(::mlir::IntegerType::get(context, 64), 0);
        assertOp->setAttr("gas_cost", gasCostAttr);

        return wrap(assertOp.getOperation());
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirOperation oraYieldOpCreate(MlirContext ctx, MlirLocation loc, const MlirValue *operands, size_t numOperands)
{
    try
    {
        MLIRContext *context = unwrap(ctx);
        Location location = unwrap(loc);

        if (!oraDialectIsRegistered(ctx))
        {
            return {nullptr};
        }

        OpBuilder builder(context);

        // Convert C array to ValueRange
        SmallVector<Value> operandValues;
        for (size_t i = 0; i < numOperands; ++i)
        {
            operandValues.push_back(unwrap(operands[i]));
        }

        auto yieldOp = builder.create<YieldOp>(location, operandValues);

        // Add gas_cost attribute (yield operation has minimal cost = 1)
        auto gasCostAttr = IntegerAttr::get(::mlir::IntegerType::get(context, 64), 1);
        yieldOp->setAttr("gas_cost", gasCostAttr);

        return wrap(yieldOp.getOperation());
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirOperation oraMLoadOpCreate(MlirContext ctx, MlirLocation loc, MlirStringRef variableName, MlirType resultType)
{
    try
    {
        MLIRContext *context = unwrap(ctx);
        Location location = unwrap(loc);
        StringRef varNameRef = unwrap(variableName);
        Type resultTypeRef = unwrap(resultType);

        if (!oraDialectIsRegistered(ctx))
        {
            return {nullptr};
        }

        OpBuilder builder(context);
        auto varNameAttr = StringAttr::get(context, varNameRef);
        auto mloadOp = builder.create<MLoadOp>(location, resultTypeRef, varNameAttr);

        // Add gas_cost attribute (memory load has minimal cost = 3)
        auto gasCostAttr = IntegerAttr::get(::mlir::IntegerType::get(context, 64), 3);
        mloadOp->setAttr("gas_cost", gasCostAttr);

        return wrap(mloadOp.getOperation());
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirOperation oraMStoreOpCreate(MlirContext ctx, MlirLocation loc, MlirValue value, MlirStringRef variableName)
{
    try
    {
        MLIRContext *context = unwrap(ctx);
        Location location = unwrap(loc);
        Value valueRef = unwrap(value);
        StringRef varNameRef = unwrap(variableName);

        if (!oraDialectIsRegistered(ctx))
        {
            return {nullptr};
        }

        OpBuilder builder(context);
        auto varNameAttr = StringAttr::get(context, varNameRef);
        auto mstoreOp = builder.create<MStoreOp>(location, valueRef, varNameAttr);

        // Add gas_cost attribute (memory store has minimal cost = 3)
        auto gasCostAttr = IntegerAttr::get(::mlir::IntegerType::get(context, 64), 3);
        mstoreOp->setAttr("gas_cost", gasCostAttr);

        return wrap(mstoreOp.getOperation());
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirOperation oraTLoadOpCreate(MlirContext ctx, MlirLocation loc, MlirStringRef key, MlirType resultType)
{
    try
    {
        MLIRContext *context = unwrap(ctx);
        Location location = unwrap(loc);
        StringRef keyRef = unwrap(key);
        Type resultTypeRef = unwrap(resultType);

        if (!oraDialectIsRegistered(ctx))
        {
            return {nullptr};
        }

        OpBuilder builder(context);
        auto keyAttr = StringAttr::get(context, keyRef);
        auto tloadOp = builder.create<TLoadOp>(location, resultTypeRef, keyAttr);

        // Add gas_cost attribute (transient load has minimal cost = 3)
        auto gasCostAttr = IntegerAttr::get(::mlir::IntegerType::get(context, 64), 3);
        tloadOp->setAttr("gas_cost", gasCostAttr);

        return wrap(tloadOp.getOperation());
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirOperation oraTStoreOpCreate(MlirContext ctx, MlirLocation loc, MlirValue value, MlirStringRef key)
{
    try
    {
        MLIRContext *context = unwrap(ctx);
        Location location = unwrap(loc);
        Value valueRef = unwrap(value);
        StringRef keyRef = unwrap(key);

        if (!oraDialectIsRegistered(ctx))
        {
            return {nullptr};
        }

        OpBuilder builder(context);
        auto keyAttr = StringAttr::get(context, keyRef);
        auto tstoreOp = builder.create<TStoreOp>(location, valueRef, keyAttr);

        // Add gas_cost attribute (transient store has minimal cost = 3)
        auto gasCostAttr = IntegerAttr::get(::mlir::IntegerType::get(context, 64), 3);
        tstoreOp->setAttr("gas_cost", gasCostAttr);

        return wrap(tstoreOp.getOperation());
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirOperation oraMapGetOpCreate(MlirContext ctx, MlirLocation loc, MlirValue map, MlirValue key, MlirType resultType)
{
    try
    {
        MLIRContext *context = unwrap(ctx);
        Location location = unwrap(loc);
        Value mapRef = unwrap(map);
        Value keyRef = unwrap(key);
        Type resultTypeRef = unwrap(resultType);

        if (!oraDialectIsRegistered(ctx))
        {
            return {nullptr};
        }

        OpBuilder builder(context);
        auto mapGetOp = builder.create<MapGetOp>(location, resultTypeRef, mapRef, keyRef);

        // Add gas_cost attribute (map get has cost = 2100, similar to sload)
        auto gasCostAttr = IntegerAttr::get(::mlir::IntegerType::get(context, 64), 2100);
        mapGetOp->setAttr("gas_cost", gasCostAttr);

        return wrap(mapGetOp.getOperation());
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirOperation oraMapStoreOpCreate(MlirContext ctx, MlirLocation loc, MlirValue map, MlirValue key, MlirValue value)
{
    try
    {
        MLIRContext *context = unwrap(ctx);
        Location location = unwrap(loc);
        Value mapRef = unwrap(map);
        Value keyRef = unwrap(key);
        Value valueRef = unwrap(value);

        if (!oraDialectIsRegistered(ctx))
        {
            return {nullptr};
        }

        OpBuilder builder(context);
        auto mapStoreOp = builder.create<MapStoreOp>(location, mapRef, keyRef, valueRef);

        // Add gas_cost attribute (map store has cost = 20000, similar to sstore)
        auto gasCostAttr = IntegerAttr::get(::mlir::IntegerType::get(context, 64), 20000);
        mapStoreOp->setAttr("gas_cost", gasCostAttr);

        return wrap(mapStoreOp.getOperation());
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirOperation oraContinueOpCreate(MlirContext ctx, MlirLocation loc, MlirStringRef label)
{
    try
    {
        MLIRContext *context = unwrap(ctx);
        Location location = unwrap(loc);

        if (!oraDialectIsRegistered(ctx))
        {
            return {nullptr};
        }

        OpBuilder builder(context);

        // Create continue operation - label is optional
        StringAttr labelAttr = nullptr;
        if (label.data != nullptr && label.length > 0)
        {
            labelAttr = StringAttr::get(context, unwrap(label));
        }

        auto continueOp = builder.create<ContinueOp>(location, labelAttr);

        // Add gas_cost attribute (continue operation has minimal cost = 1)
        auto gasCostAttr = IntegerAttr::get(::mlir::IntegerType::get(context, 64), 1);
        continueOp->setAttr("gas_cost", gasCostAttr);

        return wrap(continueOp.getOperation());
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirOperation oraReturnOpCreate(MlirContext ctx, MlirLocation loc, const MlirValue *operands, size_t numOperands)
{
    try
    {
        MLIRContext *context = unwrap(ctx);
        Location location = unwrap(loc);

        if (!oraDialectIsRegistered(ctx))
        {
            return {nullptr};
        }

        OpBuilder builder(context);

        // Convert C array to ValueRange
        SmallVector<Value> operandValues;
        for (size_t i = 0; i < numOperands; ++i)
        {
            operandValues.push_back(unwrap(operands[i]));
        }

        auto returnOp = builder.create<ReturnOp>(location, operandValues);

        // Add gas_cost attribute (return operation has cost = 8)
        auto gasCostAttr = IntegerAttr::get(::mlir::IntegerType::get(context, 64), 8);
        returnOp->setAttr("gas_cost", gasCostAttr);

        return wrap(returnOp.getOperation());
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirOperation oraDecreasesOpCreate(MlirContext ctx, MlirLocation loc, MlirValue measure)
{
    try
    {
        MLIRContext *context = unwrap(ctx);
        Location location = unwrap(loc);
        Value measureRef = unwrap(measure);

        if (!oraDialectIsRegistered(ctx))
        {
            return {nullptr};
        }

        OpBuilder builder(context);
        auto decreasesOp = builder.create<DecreasesOp>(location, measureRef);

        // Add gas_cost attribute (verification operation has no runtime cost = 0)
        auto gasCostAttr = IntegerAttr::get(::mlir::IntegerType::get(context, 64), 0);
        decreasesOp->setAttr("gas_cost", gasCostAttr);

        return wrap(decreasesOp.getOperation());
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirOperation oraIncreasesOpCreate(MlirContext ctx, MlirLocation loc, MlirValue measure)
{
    try
    {
        MLIRContext *context = unwrap(ctx);
        Location location = unwrap(loc);
        Value measureRef = unwrap(measure);

        if (!oraDialectIsRegistered(ctx))
        {
            return {nullptr};
        }

        OpBuilder builder(context);
        auto increasesOp = builder.create<IncreasesOp>(location, measureRef);

        // Add gas_cost attribute (verification operation has no runtime cost = 0)
        auto gasCostAttr = IntegerAttr::get(::mlir::IntegerType::get(context, 64), 0);
        increasesOp->setAttr("gas_cost", gasCostAttr);

        return wrap(increasesOp.getOperation());
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirOperation oraAssumeOpCreate(MlirContext ctx, MlirLocation loc, MlirValue condition)
{
    try
    {
        MLIRContext *context = unwrap(ctx);
        Location location = unwrap(loc);
        Value conditionRef = unwrap(condition);

        if (!oraDialectIsRegistered(ctx))
        {
            return {nullptr};
        }

        OpBuilder builder(context);
        auto assumeOp = builder.create<AssumeOp>(location, conditionRef);

        // Add gas_cost attribute (verification operation has no runtime cost = 0)
        auto gasCostAttr = IntegerAttr::get(::mlir::IntegerType::get(context, 64), 0);
        assumeOp->setAttr("gas_cost", gasCostAttr);

        return wrap(assumeOp.getOperation());
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirOperation oraHavocOpCreate(MlirContext ctx, MlirLocation loc, MlirStringRef variableName)
{
    try
    {
        MLIRContext *context = unwrap(ctx);
        Location location = unwrap(loc);
        StringRef varNameRef = unwrap(variableName);

        if (!oraDialectIsRegistered(ctx))
        {
            return {nullptr};
        }

        OpBuilder builder(context);
        auto varNameAttr = StringAttr::get(context, varNameRef);
        auto havocOp = builder.create<HavocOp>(location, varNameAttr);

        // Add gas_cost attribute (verification operation has no runtime cost = 0)
        auto gasCostAttr = IntegerAttr::get(::mlir::IntegerType::get(context, 64), 0);
        havocOp->setAttr("gas_cost", gasCostAttr);

        return wrap(havocOp.getOperation());
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirOperation oraOldOpCreate(MlirContext ctx, MlirLocation loc, MlirValue value, MlirType resultType)
{
    try
    {
        MLIRContext *context = unwrap(ctx);
        Location location = unwrap(loc);
        Value valueRef = unwrap(value);
        Type resultTypeRef = unwrap(resultType);

        if (!oraDialectIsRegistered(ctx))
        {
            return {nullptr};
        }

        OpBuilder builder(context);
        auto oldOp = builder.create<OldOp>(location, resultTypeRef, valueRef);

        // Add gas_cost attribute (verification operation has no runtime cost = 0)
        auto gasCostAttr = IntegerAttr::get(::mlir::IntegerType::get(context, 64), 0);
        oldOp->setAttr("gas_cost", gasCostAttr);

        return wrap(oldOp.getOperation());
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirOperation oraLockOpCreate(MlirContext ctx, MlirLocation loc, MlirValue resource)
{
    try
    {
        MLIRContext *context = unwrap(ctx);
        Location location = unwrap(loc);
        Value resourceRef = unwrap(resource);

        if (!oraDialectIsRegistered(ctx))
        {
            return {nullptr};
        }

        OpBuilder builder(context);
        auto lockOp = builder.create<LockOp>(location, resourceRef);

        // Add gas_cost attribute (locking operation has no runtime cost = 0)
        auto gasCostAttr = IntegerAttr::get(::mlir::IntegerType::get(context, 64), 0);
        lockOp->setAttr("gas_cost", gasCostAttr);

        return wrap(lockOp.getOperation());
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirOperation oraUnlockOpCreate(MlirContext ctx, MlirLocation loc, MlirValue resource)
{
    try
    {
        MLIRContext *context = unwrap(ctx);
        Location location = unwrap(loc);
        Value resourceRef = unwrap(resource);

        if (!oraDialectIsRegistered(ctx))
        {
            return {nullptr};
        }

        OpBuilder builder(context);
        auto unlockOp = builder.create<UnlockOp>(location, resourceRef);

        // Add gas_cost attribute (unlocking operation has no runtime cost = 0)
        auto gasCostAttr = IntegerAttr::get(::mlir::IntegerType::get(context, 64), 0);
        unlockOp->setAttr("gas_cost", gasCostAttr);

        return wrap(unlockOp.getOperation());
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirOperation oraStringConstantOpCreate(MlirContext ctx, MlirLocation loc, MlirStringRef value, MlirType resultType)
{
    try
    {
        MLIRContext *context = unwrap(ctx);
        Location location = unwrap(loc);
        StringRef valueRef = unwrap(value);
        Type resultTypeRef = unwrap(resultType);

        if (!oraDialectIsRegistered(ctx))
        {
            return {nullptr};
        }

        OpBuilder builder(context);
        auto valueAttr = StringAttr::get(context, valueRef);
        auto stringConstOp = builder.create<StringConstantOp>(location, resultTypeRef, valueAttr);

        // Add gas_cost attribute (constant has no runtime cost = 0)
        auto gasCostAttr = IntegerAttr::get(::mlir::IntegerType::get(context, 64), 0);
        stringConstOp->setAttr("gas_cost", gasCostAttr);

        return wrap(stringConstOp.getOperation());
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirOperation oraHexConstantOpCreate(MlirContext ctx, MlirLocation loc, MlirStringRef value, MlirType resultType)
{
    try
    {
        MLIRContext *context = unwrap(ctx);
        Location location = unwrap(loc);
        StringRef valueRef = unwrap(value);
        Type resultTypeRef = unwrap(resultType);

        if (!oraDialectIsRegistered(ctx))
        {
            return {nullptr};
        }

        OpBuilder builder(context);
        auto valueAttr = StringAttr::get(context, valueRef);
        auto hexConstOp = builder.create<HexConstantOp>(location, resultTypeRef, valueAttr);

        // Add gas_cost attribute (constant has no runtime cost = 0)
        auto gasCostAttr = IntegerAttr::get(::mlir::IntegerType::get(context, 64), 0);
        hexConstOp->setAttr("gas_cost", gasCostAttr);

        return wrap(hexConstOp.getOperation());
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirOperation oraPowerOpCreate(MlirContext ctx, MlirLocation loc, MlirValue base, MlirValue exponent, MlirType resultType)
{
    try
    {
        MLIRContext *context = unwrap(ctx);
        Location location = unwrap(loc);
        Value baseRef = unwrap(base);
        Value exponentRef = unwrap(exponent);
        Type resultTypeRef = unwrap(resultType);

        if (!oraDialectIsRegistered(ctx))
        {
            return {nullptr};
        }

        OpBuilder builder(context);
        auto powerOp = builder.create<PowerOp>(location, resultTypeRef, baseRef, exponentRef);

        // Add gas_cost attribute (power operation has cost = 5, similar to multiplication)
        auto gasCostAttr = IntegerAttr::get(::mlir::IntegerType::get(context, 64), 5);
        powerOp->setAttr("gas_cost", gasCostAttr);

        return wrap(powerOp.getOperation());
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirOperation oraConstOpCreate(MlirContext ctx, MlirLocation loc, MlirStringRef name, MlirAttribute value, MlirType resultType)
{
    try
    {
        MLIRContext *context = unwrap(ctx);
        Location location = unwrap(loc);
        StringRef nameRef = unwrap(name);
        Attribute valueAttr = unwrap(value);
        Type resultTypeRef = unwrap(resultType);

        if (!oraDialectIsRegistered(ctx))
        {
            return {nullptr};
        }

        OpBuilder builder(context);
        auto nameAttr = StringAttr::get(context, nameRef);
        auto constOp = builder.create<ConstOp>(location, resultTypeRef, nameAttr, valueAttr);

        // Add gas_cost attribute (constant declaration has no runtime cost = 0)
        auto gasCostAttr = IntegerAttr::get(::mlir::IntegerType::get(context, 64), 0);
        constOp->setAttr("gas_cost", gasCostAttr);

        return wrap(constOp.getOperation());
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirOperation oraImmutableOpCreate(MlirContext ctx, MlirLocation loc, MlirStringRef name, MlirValue value, MlirType resultType)
{
    try
    {
        MLIRContext *context = unwrap(ctx);
        Location location = unwrap(loc);
        StringRef nameRef = unwrap(name);
        Value valueRef = unwrap(value);
        Type resultTypeRef = unwrap(resultType);

        if (!oraDialectIsRegistered(ctx))
        {
            return {nullptr};
        }

        OpBuilder builder(context);
        auto nameAttr = StringAttr::get(context, nameRef);
        auto immutableOp = builder.create<ImmutableOp>(location, resultTypeRef, nameAttr, valueRef);

        // Add gas_cost attribute (immutable declaration has no runtime cost = 0)
        auto gasCostAttr = IntegerAttr::get(::mlir::IntegerType::get(context, 64), 0);
        immutableOp->setAttr("gas_cost", gasCostAttr);

        return wrap(immutableOp.getOperation());
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirOperation oraStructFieldStoreOpCreate(MlirContext ctx, MlirLocation loc, MlirValue structValue, MlirStringRef fieldName, MlirValue value)
{
    try
    {
        MLIRContext *context = unwrap(ctx);
        Location location = unwrap(loc);
        Value structValueRef = unwrap(structValue);
        StringRef fieldNameRef = unwrap(fieldName);
        Value valueRef = unwrap(value);

        if (!oraDialectIsRegistered(ctx))
        {
            return {nullptr};
        }

        OpBuilder builder(context);
        auto fieldNameAttr = StringAttr::get(context, fieldNameRef);
        auto structFieldStoreOp = builder.create<StructFieldStoreOp>(location, structValueRef, fieldNameAttr, valueRef);

        // Add gas_cost attribute (struct field store has cost = 3, similar to memory store)
        auto gasCostAttr = IntegerAttr::get(::mlir::IntegerType::get(context, 64), 3);
        structFieldStoreOp->setAttr("gas_cost", gasCostAttr);

        return wrap(structFieldStoreOp.getOperation());
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirOperation oraStructInitOpCreate(MlirContext ctx, MlirLocation loc, const MlirValue *fieldValues, size_t numFieldValues, MlirType resultType)
{
    try
    {
        MLIRContext *context = unwrap(ctx);
        Location location = unwrap(loc);
        Type resultTypeRef = unwrap(resultType);

        if (!oraDialectIsRegistered(ctx))
        {
            return {nullptr};
        }

        OpBuilder builder(context);

        // Convert C array to ValueRange
        SmallVector<Value> fieldValueRefs;
        for (size_t i = 0; i < numFieldValues; ++i)
        {
            fieldValueRefs.push_back(unwrap(fieldValues[i]));
        }

        auto structInitOp = builder.create<StructInitOp>(location, resultTypeRef, fieldValueRefs);

        // Add gas_cost attribute (struct initialization has cost = 3 per field, minimum 3)
        auto gasCostAttr = IntegerAttr::get(::mlir::IntegerType::get(context, 64), 3);
        structInitOp->setAttr("gas_cost", gasCostAttr);

        return wrap(structInitOp.getOperation());
    }
    catch (...)
    {
        return {nullptr};
    }
}

//===----------------------------------------------------------------------===//
// Additional Ora Operations
//===----------------------------------------------------------------------===//

MlirOperation oraDestructureOpCreate(MlirContext ctx, MlirLocation loc, MlirValue value, MlirStringRef patternType, MlirType resultType)
{
    try
    {
        MLIRContext *context = unwrap(ctx);
        Location location = unwrap(loc);
        Value val = unwrap(value);
        Type resultTy = unwrap(resultType);

        OpBuilder builder(context);

        // Create the destructure operation
        StringRef patternTypeRef = unwrap(patternType);
        auto patternTypeAttr = StringAttr::get(context, patternTypeRef);
        auto op = builder.create<ora::DestructureOp>(location, resultTy, val, patternTypeAttr);

        // Add gas cost attribute (destructuring has minimal cost)
        auto gasCostAttr = builder.getI64IntegerAttr(3);
        op->setAttr("gas_cost", gasCostAttr);

        return wrap(op.getOperation());
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirOperation oraEnumDeclOpCreate(MlirContext ctx, MlirLocation loc, MlirStringRef name, MlirType reprType)
{
    try
    {
        MLIRContext *context = unwrap(ctx);
        Location location = unwrap(loc);
        Type reprTy = unwrap(reprType);

        OpBuilder builder(context);

        // Create the enum.decl operation
        StringRef nameRef = unwrap(name);
        auto nameAttr = StringAttr::get(context, nameRef);
        auto op = builder.create<ora::EnumDeclOp>(location, nameAttr, TypeAttr::get(reprTy));

        // Ensure the variants region has at least one block
        if (op.getVariants().empty())
        {
            OpBuilder::InsertionGuard guard(builder);
            builder.createBlock(&op.getVariants());
        }

        // Add gas cost attribute (declaration has no runtime cost)
        auto gasCostAttr = builder.getI64IntegerAttr(0);
        op->setAttr("gas_cost", gasCostAttr);

        return wrap(op.getOperation());
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirOperation oraEnumConstantOpCreate(MlirContext ctx, MlirLocation loc, MlirStringRef enumName, MlirStringRef variantName, MlirType resultType)
{
    try
    {
        MLIRContext *context = unwrap(ctx);
        Location location = unwrap(loc);
        Type resultTy = unwrap(resultType);

        OpBuilder builder(context);

        // Create the enum_constant operation
        StringRef enumNameRef = unwrap(enumName);
        StringRef variantNameRef = unwrap(variantName);
        auto enumNameAttr = StringAttr::get(context, enumNameRef);
        auto variantNameAttr = StringAttr::get(context, variantNameRef);
        auto op = builder.create<ora::EnumConstantOp>(location, resultTy, enumNameAttr, variantNameAttr);

        // Add gas cost attribute (constant has no cost)
        auto gasCostAttr = builder.getI64IntegerAttr(0);
        op->setAttr("gas_cost", gasCostAttr);

        return wrap(op.getOperation());
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirOperation oraStructDeclOpCreate(MlirContext ctx, MlirLocation loc, MlirStringRef name)
{
    try
    {
        MLIRContext *context = unwrap(ctx);
        Location location = unwrap(loc);

        OpBuilder builder(context);

        // Create the struct.decl operation
        StringRef nameRef = unwrap(name);
        auto nameAttr = StringAttr::get(context, nameRef);
        auto op = builder.create<ora::StructDeclOp>(location, nameAttr);

        // Ensure the fields region has at least one block
        if (op.getFields().empty())
        {
            OpBuilder::InsertionGuard guard(builder);
            builder.createBlock(&op.getFields());
        }

        // Add gas cost attribute (declaration has no runtime cost)
        auto gasCostAttr = builder.getI64IntegerAttr(0);
        op->setAttr("gas_cost", gasCostAttr);

        return wrap(op.getOperation());
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirOperation oraStructInstantiateOpCreate(MlirContext ctx, MlirLocation loc, MlirStringRef structName, const MlirValue *fieldValues, size_t numFieldValues, MlirType resultType)
{
    try
    {
        MLIRContext *context = unwrap(ctx);
        Location location = unwrap(loc);
        Type resultTy = unwrap(resultType);

        OpBuilder builder(context);

        // Convert MlirValue array to SmallVector<Value>
        SmallVector<Value> fieldVals;
        fieldVals.reserve(numFieldValues);
        for (size_t i = 0; i < numFieldValues; ++i)
        {
            fieldVals.push_back(unwrap(fieldValues[i]));
        }

        // Create the struct_instantiate operation
        StringRef structNameRef = unwrap(structName);
        auto structNameAttr = StringAttr::get(context, structNameRef);
        auto op = builder.create<ora::StructInstantiateOp>(location, resultTy, structNameAttr, fieldVals);

        // Add gas cost attribute (struct instantiation has minimal cost)
        auto gasCostAttr = builder.getI64IntegerAttr(10);
        op->setAttr("gas_cost", gasCostAttr);

        return wrap(op.getOperation());
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirOperation oraMoveOpCreate(MlirContext ctx, MlirLocation loc, MlirValue amount, MlirValue source, MlirValue destination)
{
    try
    {
        MLIRContext *context = unwrap(ctx);
        Location location = unwrap(loc);
        Value amt = unwrap(amount);
        Value src = unwrap(source);
        Value dst = unwrap(destination);

        OpBuilder builder(context);

        // Create the move operation
        auto op = builder.create<ora::MoveOp>(location, amt, src, dst);

        // Add gas cost attribute (move operation has cost for balance updates)
        auto gasCostAttr = builder.getI64IntegerAttr(5000);
        op->setAttr("gas_cost", gasCostAttr);

        return wrap(op.getOperation());
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirOperation oraLogOpCreate(MlirContext ctx, MlirLocation loc, MlirStringRef eventName, const MlirValue *parameters, size_t numParameters)
{
    try
    {
        MLIRContext *context = unwrap(ctx);
        Location location = unwrap(loc);

        OpBuilder builder(context);

        // Convert MlirValue array to SmallVector<Value>
        SmallVector<Value> params;
        params.reserve(numParameters);
        for (size_t i = 0; i < numParameters; ++i)
        {
            params.push_back(unwrap(parameters[i]));
        }

        // Create the log operation
        StringRef eventNameRef = unwrap(eventName);
        auto eventNameAttr = StringAttr::get(context, eventNameRef);
        auto op = builder.create<ora::LogOp>(location, eventNameAttr, params);

        // Add gas cost attribute (logging has cost based on data size)
        // Base cost + per-byte cost (simplified to fixed cost for now)
        auto gasCostAttr = builder.getI64IntegerAttr(375 + (numParameters * 375));
        op->setAttr("gas_cost", gasCostAttr);

        return wrap(op.getOperation());
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirOperation oraTryOpCreate(MlirContext ctx, MlirLocation loc, MlirValue tryOperation, MlirType resultType)
{
    try
    {
        MLIRContext *context = unwrap(ctx);
        Location location = unwrap(loc);
        Value tryOp = unwrap(tryOperation);
        Type resultTy = unwrap(resultType);

        OpBuilder builder(context);

        // Create the try_catch operation
        auto op = builder.create<ora::TryOp>(location, resultTy, tryOp);

        // Ensure the catch region has at least one block
        if (op.getCatchRegion().empty())
        {
            OpBuilder::InsertionGuard guard(builder);
            builder.createBlock(&op.getCatchRegion());
        }

        // Add gas cost attribute (try-catch has minimal overhead)
        auto gasCostAttr = builder.getI64IntegerAttr(5);
        op->setAttr("gas_cost", gasCostAttr);

        return wrap(op.getOperation());
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirOperation oraForOpCreate(MlirContext ctx, MlirLocation loc, MlirValue collection)
{
    try
    {
        MLIRContext *context = unwrap(ctx);
        Location location = unwrap(loc);
        Value coll = unwrap(collection);

        OpBuilder builder(context);

        // Create the for operation
        auto op = builder.create<ora::ForOp>(location, coll);

        // Ensure the body region has at least one block
        if (op.getBody().empty())
        {
            OpBuilder::InsertionGuard guard(builder);
            builder.createBlock(&op.getBody());
        }

        // Add gas cost attribute (for loop has cost per iteration)
        auto gasCostAttr = builder.getI64IntegerAttr(10);
        op->setAttr("gas_cost", gasCostAttr);

        return wrap(op.getOperation());
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirOperation oraBreakOpCreate(MlirContext ctx, MlirLocation loc, MlirStringRef label, const MlirValue *values, size_t numValues)
{
    try
    {
        MLIRContext *context = unwrap(ctx);
        Location location = unwrap(loc);

        OpBuilder builder(context);

        // Convert MlirValue array to SmallVector<Value>
        SmallVector<Value> vals;
        vals.reserve(numValues);
        for (size_t i = 0; i < numValues; ++i)
        {
            vals.push_back(unwrap(values[i]));
        }

        // Create the break operation with optional label
        StringAttr labelAttr = label.data ? StringAttr::get(context, StringRef(label.data, label.length)) : StringAttr();
        auto op = builder.create<ora::BreakOp>(location, labelAttr, vals);

        // Add gas cost attribute (break has minimal cost)
        auto gasCostAttr = builder.getI64IntegerAttr(1);
        op->setAttr("gas_cost", gasCostAttr);

        return wrap(op.getOperation());
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirOperation oraSwitchOpCreate(MlirContext ctx, MlirLocation loc, MlirValue value, MlirType resultType)
{
    try
    {
        MLIRContext *context = unwrap(ctx);
        Location location = unwrap(loc);
        Value val = unwrap(value);
        Type resultTy = unwrap(resultType);

        OpBuilder builder(context);

        // Create the switch operation
        // SwitchOp expects TypeRange results, Value value, and unsigned casesCount
        TypeRange resultTypes = TypeRange(resultTy);
        auto op = builder.create<ora::SwitchOp>(location, resultTypes, val, /*casesCount=*/0);

        // SwitchOp has VariadicRegion for cases, caller will add cases as needed
        // No need to ensure a block exists initially

        // Add gas cost attribute (switch has cost per case)
        auto gasCostAttr = builder.getI64IntegerAttr(10);
        op->setAttr("gas_cost", gasCostAttr);

        return wrap(op.getOperation());
    }
    catch (...)
    {
        return {nullptr};
    }
}

//===----------------------------------------------------------------------===//
// CFG Generation with Registered Dialect
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// CFG Generation Implementation
//===----------------------------------------------------------------------===//

MlirStringRef oraGenerateCFG(MlirContext ctx, MlirModule module, bool includeControlFlow)
{
    try
    {
        MLIRContext *context = unwrap(ctx);
        (void)context; // Context is used implicitly by mlirPassManagerCreate
        ModuleOp moduleOp = unwrap(module);

        // Register the Ora dialect
        if (!oraDialectRegister(ctx))
        {
            return {nullptr, 0};
        }

        // Verify the module has content
        if (moduleOp.getBodyRegion().empty() || moduleOp.getBodyRegion().front().empty())
        {
            // Module is empty, nothing to graph
            return {nullptr, 0};
        }

        // Create a string stream to capture the DOT output from the pass
        std::string dotContent;
        llvm::raw_string_ostream dotStream(dotContent);

        // Create the view-op-graph pass with our custom output stream and control flow edges
        // The pass will write Graphviz DOT format to dotStream
        // Control flow edges (dashed lines) show dominance relationships
        auto graphPass = createPrintOpGraphPass(dotStream);

        // Note: createPrintOpGraphPass doesn't expose options directly
        // We'll use the default configuration which includes data flow edges
        // Control flow edges would need to be enabled via pass options if available
        // For now, the CFG shows operation structure and data dependencies
        (void)includeControlFlow; // Reserved for future use when API supports it

        // OperationPass<> works on any operation type, so add it directly to PassManager
        PassManager pm(context);
        pm.addPass(std::move(graphPass));

        // Run the pass on the module
        if (failed(pm.run(moduleOp)))
        {
            return {nullptr, 0};
        }

        // Flush the stream to ensure all content is written
        // The raw_string_ostream should have buffered all writes
        dotStream.flush();

        // Check if we got any content
        // If empty, the pass might have written to a different stream
        // or the module might not have any operations to graph
        if (dotContent.empty())
        {
            // The pass ran but produced no output
            // This could mean the module is empty or the pass isn't writing to our stream
            return {nullptr, 0};
        }

        // Allocate memory for the result (caller must free)
        char *result = (char *)malloc(dotContent.size() + 1);
        if (!result)
        {
            return {nullptr, 0};
        }
        memcpy(result, dotContent.c_str(), dotContent.size());
        result[dotContent.size()] = '\0';

        return {result, dotContent.size()};
    }
    catch (...)
    {
        return {nullptr, 0};
    }
}

//===----------------------------------------------------------------------===//
// MLIR Printing with Custom Assembly Formats
//===----------------------------------------------------------------------===//

MlirStringRef oraPrintOperation(MlirContext ctx, MlirOperation op)
{
    try
    {
        Operation *operation = unwrap(op);

        // Register the Ora dialect to ensure custom printers are available
        if (!oraDialectRegister(ctx))
        {
            return {nullptr, 0};
        }

        // Create a string stream to capture the printed output
        std::string mlirContent;
        llvm::raw_string_ostream mlirStream(mlirContent);

        // Use OpPrintingFlags - Operation::print(stream, flags) will automatically
        // dispatch to custom print(OpAsmPrinter&) methods when hasCustomAssemblyFormat = 1
        OpPrintingFlags flags;
        flags.enableDebugInfo(true, false);
        // CRITICAL: Explicitly disable generic op form to enable custom printers
        // The OpPrintingFlags constructor reads from command-line options, which
        // might enable generic form. We must explicitly disable it.
        flags.printGenericOpForm(false);
        // CRITICAL: Assume operations are verified to prevent MLIR from switching
        // to generic form if verification fails. This is necessary because
        // verifyOpAndAdjustFlags() will force generic printing if verification fails.
        flags.assumeVerified();
        // Print locations inline instead of using location references
        // This makes locations appear as loc("file":line:col) instead of loc(#locN)
        flags.useLocalScope();

        // Create AsmState to ensure custom printers are used
        // AsmState tracks the state of the printer and ensures dialect-specific
        // printers are invoked when operations have hasCustomAssemblyFormat = 1
        // CRITICAL: AsmState must be created with the flags to propagate them to nested operations
        AsmState state(operation, flags);

        // Print the operation using AsmState - this ensures custom printers are invoked
        // MLIR's Operation::print() should automatically call custom print() methods
        // when hasCustomAssemblyFormat = 1
        operation->print(mlirStream, state);

        // Flush the stream
        mlirStream.flush();

        // Check if we got any content
        if (mlirContent.empty())
        {
            return {nullptr, 0};
        }

        // Post-process the output to add line breaks for readability
        // Add blank lines after certain operations to improve readability
        std::string formattedContent;
        formattedContent.reserve(mlirContent.size() * 1.1); // Reserve slightly more space

        std::istringstream inputStream(mlirContent);
        std::string line;
        std::string prevLine;
        bool prevWasEmpty = false;

        while (std::getline(inputStream, line))
        {
            // Skip empty lines in input (we'll add our own)
            if (line.empty() || (line.find_first_not_of(" \t") == std::string::npos))
            {
                prevWasEmpty = true;
                continue;
            }

            // Add blank line after certain patterns for readability
            bool shouldAddBlankLine = false;

            // Add blank line before comments (if previous line wasn't empty and wasn't already a comment)
            if (line.find("//") != std::string::npos && line.find("//") < line.length() &&
                !prevLine.empty() && !prevWasEmpty && prevLine.find("//") == std::string::npos)
            {
                shouldAddBlankLine = true;
            }
            // Add blank line after return statements (before closing brace or next operation)
            else if (prevLine.find("return") != std::string::npos &&
                     prevLine.find("return") < prevLine.length() &&
                     !line.empty() && line.find("}") == std::string::npos)
            {
                shouldAddBlankLine = true;
            }
            // Add blank line after scf.if/else blocks close (before next operation)
            else if (prevLine.find("}") != std::string::npos &&
                     (prevLine.find("} {gas_cost") != std::string::npos ||
                      (prevLine.find("}") == prevLine.find_last_of("}") &&
                       prevLine.find("} else {") == std::string::npos)) &&
                     !line.empty() && line.find("}") == std::string::npos &&
                     line.find("func.func") == std::string::npos)
            {
                shouldAddBlankLine = true;
            }
            // Add blank line after comments (before next operation)
            else if (prevLine.find("//") != std::string::npos &&
                     prevLine.find("//") < prevLine.length() &&
                     !line.empty() && line.find("//") == std::string::npos)
            {
                shouldAddBlankLine = true;
            }

            if (shouldAddBlankLine && !prevLine.empty() && !prevWasEmpty)
            {
                formattedContent += "\n";
            }

            formattedContent += line;
            formattedContent += "\n";

            prevWasEmpty = false;
            prevLine = line;
        }

        // Allocate memory for the result (caller must free)
        char *result = (char *)malloc(formattedContent.size() + 1);
        if (!result)
        {
            return {nullptr, 0};
        }
        memcpy(result, formattedContent.c_str(), formattedContent.size());
        result[formattedContent.size()] = '\0';

        return {result, formattedContent.size()};
    }
    catch (...)
    {
        return {nullptr, 0};
    }
}

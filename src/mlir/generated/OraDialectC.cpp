//===- OraDialectC.cpp - C Interface for Ora MLIR Dialect --------------===//
//
// This file implements the C interface to the Ora MLIR dialect.
//
//===----------------------------------------------------------------------===//

#include "OraDialectC.h"
#include "OraDialect.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

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
        Attribute initAttr = unwrap(initValue);

        // Check if dialect is registered
        if (!oraDialectIsRegistered(ctx))
        {
            return {nullptr};
        }

        OpBuilder builder(context);

        // Create the global operation
        auto nameAttr = StringAttr::get(context, nameRef);
        auto typeAttr = TypeAttr::get(typeRef);
        auto globalOp = builder.create<GlobalOp>(location, nameAttr, typeAttr, initAttr);

        return wrap(globalOp.getOperation());
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirOperation oraSLoadOpCreate(MlirContext ctx, MlirLocation loc, MlirStringRef globalName, MlirType resultType)
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

        return wrap(sstoreOp.getOperation());
    }
    catch (...)
    {
        return {nullptr};
    }
}

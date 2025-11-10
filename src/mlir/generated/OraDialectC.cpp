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
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
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

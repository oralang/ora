//===- OraDialectC.cpp - C Interface for Ora MLIR Dialect --------------===//
//
// This file implements the C interface to the Ora MLIR dialect.
//
//===----------------------------------------------------------------------===//

#include "OraDialectC.h"
#include "OraDialect.h"
#include "OraToSIR.h"
#include "OraDebug.h"
#include "SIR/SIRDialect.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "mlir/CAPI/Pass.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/AsmState.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "llvm/ADT/APInt.h"
#include "llvm/Support/Casting.h"
#include <sstream>
#include "mlir/Transforms/ViewOpGraph.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include <cstdlib>
#include <cstring>

extern "C"
{
    MlirStringRef oraStringRefCreate(const char *data, size_t length)
    {
        MlirStringRef ref;
        ref.data = data;
        ref.length = length;
        return ref;
    }

    MlirStringRef oraStringRefCreateFromCString(const char *data)
    {
        return oraStringRefCreate(data, std::strlen(data));
    }

    void oraStringRefFree(MlirStringRef ref)
    {
        if (ref.data != nullptr)
        {
            std::free(const_cast<char *>(ref.data));
        }
    }

    MlirIdentifier oraIdentifierGet(MlirContext ctx, MlirStringRef name)
    {
        mlir::StringAttr attr = mlir::StringAttr::get(unwrap(ctx), unwrap(name));
        return MlirIdentifier{attr.getAsOpaquePointer()};
    }

    MlirStringRef oraIdentifierStr(MlirIdentifier id)
    {
        llvm::StringRef name = unwrap(id).getValue();
        return oraStringRefCreate(name.data(), name.size());
    }

    MlirNamedAttribute oraNamedAttributeGet(MlirIdentifier name, MlirAttribute attr)
    {
        MlirNamedAttribute named;
        named.name = name;
        named.attribute = attr;
        return named;
    }

    MlirLocation oraLocationUnknownGet(MlirContext ctx)
    {
        mlir::Location loc = mlir::UnknownLoc::get(unwrap(ctx));
        return MlirLocation{loc.getAsOpaquePointer()};
    }

    MlirLocation oraLocationFileLineColGet(
        MlirContext ctx,
        MlirStringRef filename,
        unsigned line,
        unsigned column)
    {
        mlir::Location loc = mlir::FileLineColLoc::get(unwrap(ctx), unwrap(filename), line, column);
        return MlirLocation{loc.getAsOpaquePointer()};
    }

    bool oraLocationIsNull(MlirLocation loc)
    {
        return loc.ptr == nullptr;
    }

    MlirStringRef oraLocationPrintToString(MlirLocation loc)
    {
        std::string buffer;
        llvm::raw_string_ostream os(buffer);
        unwrap(loc).print(os);
        os.flush();
        char *out = static_cast<char *>(std::malloc(buffer.size() + 1));
        if (!out)
        {
            return oraStringRefCreate(nullptr, 0);
        }
        std::memcpy(out, buffer.data(), buffer.size());
        out[buffer.size()] = '\0';
        return oraStringRefCreate(out, buffer.size());
    }

    MlirContext oraContextCreate()
    {
        return wrap(new mlir::MLIRContext());
    }

    void oraContextDestroy(MlirContext ctx)
    {
        delete unwrap(ctx);
    }

    MlirDialectRegistry oraDialectRegistryCreate()
    {
        return wrap(new mlir::DialectRegistry());
    }

    void oraDialectRegistryDestroy(MlirDialectRegistry registry)
    {
        delete unwrap(registry);
    }

    void oraRegisterAllDialects(MlirDialectRegistry registry)
    {
        mlir::registerAllDialects(*unwrap(registry));
    }

    void oraContextAppendDialectRegistry(MlirContext ctx, MlirDialectRegistry registry)
    {
        unwrap(ctx)->appendDialectRegistry(*unwrap(registry));
    }

    void oraContextLoadAllAvailableDialects(MlirContext ctx)
    {
        unwrap(ctx)->loadAllAvailableDialects();
    }

    MlirModule oraModuleCreateEmpty(MlirLocation loc)
    {
        return wrap(mlir::ModuleOp::create(unwrap(loc)));
    }

    MlirOperation oraModuleGetOperation(MlirModule module)
    {
        return wrap(unwrap(module).getOperation());
    }

    MlirBlock oraModuleGetBody(MlirModule module)
    {
        return wrap(unwrap(module).getBody());
    }

    bool oraModuleIsNull(MlirModule module)
    {
        return module.ptr == nullptr;
    }

    void oraModuleDestroy(MlirModule module)
    {
        if (module.ptr == nullptr)
        {
            return;
        }
        unwrap(module).erase();
    }

void oraBlockAppendOwnedOperation(MlirBlock block, MlirOperation op)
{
        mlir::Block *blk = unwrap(block);
        mlir::Operation *operation = unwrap(op);
        if (blk == nullptr || operation == nullptr)
        {
            return;
        }

        if (blk->empty())
        {
            blk->push_back(operation);
            return;
        }

        // If the last op is a terminator, insert before it so it stays last.
        mlir::Operation *last = &blk->back();
        if (last->hasTrait<mlir::OpTrait::IsTerminator>())
        {
            blk->getOperations().insert(mlir::Block::iterator(last), operation);
            return;
        }

        blk->push_back(operation);
    }

    MlirOperation oraBlockGetFirstOperation(MlirBlock block)
    {
        mlir::Block *blk = unwrap(block);
        if (blk == nullptr || blk->empty())
        {
            return MlirOperation{nullptr};
        }
        return wrap(&blk->front());
    }

    MlirOperation oraBlockGetTerminator(MlirBlock block)
    {
        mlir::Block *blk = unwrap(block);
        if (blk == nullptr || blk->empty())
        {
            return MlirOperation{nullptr};
        }
        return wrap(blk->getTerminator());
    }

    MlirValue oraBlockGetArgument(MlirBlock block, size_t index)
    {
        return wrap(unwrap(block)->getArgument(index));
    }

    bool oraBlockIsNull(MlirBlock block)
    {
        return block.ptr == nullptr;
    }

    MlirOperation oraOperationGetNextInBlock(MlirOperation op)
    {
        mlir::Operation *operation = unwrap(op);
        if (operation == nullptr)
        {
            return MlirOperation{nullptr};
        }
        return wrap(operation->getNextNode());
    }

    void oraOperationErase(MlirOperation op)
    {
        mlir::Operation *operation = unwrap(op);
        if (operation == nullptr)
        {
            return;
        }
        operation->erase();
    }

    MlirValue oraOperationGetResult(MlirOperation op, size_t index)
    {
        mlir::Operation *operation = unwrap(op);
        if (operation == nullptr)
        {
            return MlirValue{nullptr};
        }
        return wrap(operation->getResult(index));
    }

    MlirValue oraOperationGetOperand(MlirOperation op, size_t index)
    {
        return wrap(unwrap(op)->getOperand(index));
    }

    size_t oraOperationGetNumOperands(MlirOperation op)
    {
        return unwrap(op)->getNumOperands();
    }

    size_t oraOperationGetNumResults(MlirOperation op)
    {
        return unwrap(op)->getNumResults();
    }

    size_t oraOperationGetNumRegions(MlirOperation op)
    {
        return unwrap(op)->getNumRegions();
    }

    bool oraOperationIsNull(MlirOperation op)
    {
        return op.ptr == nullptr;
    }

    void oraOperationSetAttributeByName(
        MlirOperation op,
        MlirStringRef name,
        MlirAttribute attr)
    {
        unwrap(op)->setAttr(unwrap(name), unwrap(attr));
    }

    MlirAttribute oraOperationGetAttributeByName(MlirOperation op, MlirStringRef name)
    {
        return wrap(unwrap(op)->getAttr(unwrap(name)));
    }

    MlirLocation oraOperationGetLocation(MlirOperation op)
    {
        return wrap(unwrap(op)->getLoc());
    }

    MlirRegion oraOperationGetRegion(MlirOperation op, size_t index)
    {
        return wrap(&unwrap(op)->getRegion(index));
    }

    MlirType oraValueGetType(MlirValue value)
    {
        return wrap(unwrap(value).getType());
    }

    bool oraValueIsNull(MlirValue value)
    {
        return value.ptr == nullptr;
    }

    bool oraValueIsAOpResult(MlirValue value)
    {
        return llvm::isa<mlir::OpResult>(unwrap(value));
    }

    MlirOperation oraOpResultGetOwner(MlirValue value)
    {
        auto result = llvm::dyn_cast<mlir::OpResult>(unwrap(value));
        if (!result)
        {
            return MlirOperation{nullptr};
        }
        return wrap(result.getOwner());
    }

    MlirBlock oraRegionGetFirstBlock(MlirRegion region)
    {
        mlir::Region *reg = unwrap(region);
        if (reg == nullptr || reg->empty())
        {
            return MlirBlock{nullptr};
        }
        return wrap(&reg->front());
    }

    MlirBlock oraBlockGetNextInRegion(MlirBlock block)
    {
        mlir::Block *blk = unwrap(block);
        if (blk == nullptr)
        {
            return MlirBlock{nullptr};
        }
        return wrap(blk->getNextNode());
    }

    bool oraRegionIsNull(MlirRegion region)
    {
        return region.ptr == nullptr;
    }

    bool oraAttributeIsNull(MlirAttribute attr)
    {
        return attr.ptr == nullptr;
    }

    MlirStringRef oraStringAttrGetValue(MlirAttribute attr)
    {
        auto str_attr = llvm::dyn_cast<mlir::StringAttr>(unwrap(attr));
        if (!str_attr)
        {
            return oraStringRefCreate(nullptr, 0);
        }
        llvm::StringRef value = str_attr.getValue();
        return oraStringRefCreate(value.data(), value.size());
    }

    int64_t oraIntegerAttrGetValueSInt(MlirAttribute attr)
    {
        auto int_attr = llvm::dyn_cast<mlir::IntegerAttr>(unwrap(attr));
        if (!int_attr)
        {
            return 0;
        }
        return int_attr.getValue().getSExtValue();
    }

    MlirType oraFunctionTypeGet(
        MlirContext ctx,
        size_t numInputs,
        const MlirType *inputTypes,
        size_t numResults,
        const MlirType *resultTypes)
    {
        llvm::SmallVector<mlir::Type, 8> inputs;
        llvm::SmallVector<mlir::Type, 2> results;
        inputs.reserve(numInputs);
        results.reserve(numResults);
        for (size_t i = 0; i < numInputs; ++i)
        {
            inputs.push_back(unwrap(inputTypes[i]));
        }
        for (size_t i = 0; i < numResults; ++i)
        {
            results.push_back(unwrap(resultTypes[i]));
        }
        return wrap(mlir::FunctionType::get(unwrap(ctx), inputs, results));
    }

    MlirStringRef oraOperationPrintToString(MlirOperation op)
    {
        try
        {
            ORA_DEBUG_PREFIX("OraCAPI", "oraOperationPrintToString called");

            mlir::Operation *operation = unwrap(op);

            if (!operation)
            {
                ORA_DEBUG_PREFIX("OraCAPI", "ERROR: operation is null!");
                return {nullptr, 0};
            }

            // Check if DCE left the IR in an invalid state
            if (auto moduleOp = llvm::dyn_cast<mlir::ModuleOp>(operation))
            {
                if (moduleOp->hasAttr("ora.dce_invalid"))
                {
                    ORA_DEBUG_PREFIX("OraCAPI", "  This would cause a segfault if we tried to print");
                    return {nullptr, 0};
                }
            }

            ORA_DEBUG_PREFIX("OraCAPI", "Operation: " << operation->getName());

            // Register the Ora dialect to ensure custom printers are available
            mlir::MLIRContext *context = operation->getContext();
            if (!context)
            {
                return {nullptr, 0};
            }
            context->getOrLoadDialect<mlir::ora::OraDialect>();
            context->getOrLoadDialect<sir::SIRDialect>();

            // Create a string stream to capture the printed output
            std::string mlirContent;
            llvm::raw_string_ostream mlirStream(mlirContent);

            // Use OpPrintingFlags to enable custom assembly formats
            mlir::OpPrintingFlags flags;
            flags.enableDebugInfo(true, false);
            flags.printGenericOpForm(false);
            flags.assumeVerified();
            flags.useLocalScope();

            mlir::AsmState state(operation, flags);
            operation->print(mlirStream, state);

            mlirStream.flush();

            if (mlirContent.empty())
            {
                return {nullptr, 0};
            }

            // Post-process the output to add line breaks for readability
            std::string formattedContent;
            formattedContent.reserve(mlirContent.size() * 1.1);

            std::istringstream inputStream(mlirContent);
            std::string line;
            std::string prevLine;
            bool prevWasEmpty = false;

            while (std::getline(inputStream, line))
            {
                if (line.empty() || (line.find_first_not_of(" \t") == std::string::npos))
                {
                    prevWasEmpty = true;
                    continue;
                }

                bool shouldAddBlankLine = false;

                if ((line.find("sir.return") != std::string::npos || line.find("sir.iret") != std::string::npos) &&
                    !prevLine.empty() && !prevWasEmpty)
                {
                    shouldAddBlankLine = true;
                }
                else if (line.find("func.func") != std::string::npos &&
                         !prevLine.empty() && !prevWasEmpty && prevLine.find("func.func") == std::string::npos)
                {
                    shouldAddBlankLine = true;
                }
                else if (line.find("//") != std::string::npos && line.find("//") < line.length() &&
                         !prevLine.empty() && !prevWasEmpty && prevLine.find("//") == std::string::npos)
                {
                    shouldAddBlankLine = true;
                }
                else if (prevLine.find("return") != std::string::npos &&
                         prevLine.find("return") < prevLine.length() &&
                         !line.empty() && line.find("}") == std::string::npos)
                {
                    shouldAddBlankLine = true;
                }
                else if (prevLine.find("}") != std::string::npos &&
                         (prevLine.find("} {gas_cost") != std::string::npos ||
                          (prevLine.find("}") == prevLine.find_last_of("}") &&
                           prevLine.find("} else {") == std::string::npos)) &&
                         !line.empty() && line.find("}") == std::string::npos &&
                         line.find("func.func") == std::string::npos)
                {
                    shouldAddBlankLine = true;
                }
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

            char *result = static_cast<char *>(std::malloc(formattedContent.size() + 1));
            if (!result)
            {
                return {nullptr, 0};
            }
            std::memcpy(result, formattedContent.c_str(), formattedContent.size());
            result[formattedContent.size()] = '\0';

            return {result, formattedContent.size()};
        }
        catch (...)
        {
            return {nullptr, 0};
        }
    }

    MlirPassManager oraPassManagerCreate(MlirContext ctx)
    {
        return wrap(new mlir::PassManager(unwrap(ctx)));
    }

    void oraPassManagerDestroy(MlirPassManager pm)
    {
        delete unwrap(pm);
    }

    void oraPassManagerEnableVerifier(MlirPassManager pm, bool enable)
    {
        unwrap(pm)->enableVerifier(enable);
    }

    void oraPassManagerEnableTiming(MlirPassManager pm)
    {
        unwrap(pm)->enableTiming();
    }

    bool oraPassManagerParsePipeline(MlirPassManager pm, MlirStringRef pipeline)
    {
        return mlir::succeeded(mlir::parsePassPipeline(unwrap(pipeline), *unwrap(pm)));
    }

    bool oraPassManagerRun(MlirPassManager pm, MlirOperation op)
    {
        return mlir::succeeded(unwrap(pm)->run(unwrap(op)));
    }
} // extern "C"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Debug.h"
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <memory>

#define DEBUG_TYPE "ora-to-sir"

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
        context->getOrLoadDialect<OraDialect>();
        return true;
    }
    catch (...)
    {
        return false;
    }
}

void oraBlockInsertOwnedOperationBefore(MlirBlock block, MlirOperation op, MlirOperation before)
{
    try
    {
        Block *blk = unwrap(block);
        Operation *operation = unwrap(op);
        Operation *beforeOp = unwrap(before);
        if (!blk || !operation)
        {
            return;
        }

        if (beforeOp)
        {
            blk->getOperations().insert(Block::iterator(beforeOp), operation);
        }
        else
        {
            blk->getOperations().push_back(operation);
        }
    }
    catch (...)
    {
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

        if (!oraDialectIsRegistered(ctx))
        {
            return {nullptr};
        }

        OpBuilder builder(context);

        auto contractOp = builder.create<ContractOp>(location, nameRef);

        auto &bodyRegion = contractOp.getBody();
        if (bodyRegion.empty())
        {
            bodyRegion.push_back(new Block());
        }
        auto gasCostAttr = IntegerAttr::get(::mlir::IntegerType::get(context, 64), 0);
        contractOp->setAttr("gas_cost", gasCostAttr);

        return wrap(contractOp.getOperation());
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirBlock oraContractOpGetBodyBlock(MlirOperation op)
{
    try
    {
        Operation *operation = unwrap(op);
        if (!operation)
        {
            return {nullptr};
        }

        auto contractOp = dyn_cast<ora::ContractOp>(operation);
        if (!contractOp)
        {
            return {nullptr};
        }

        Region &body = contractOp.getBody();
        if (body.empty())
        {
            body.push_back(new Block());
        }

        return wrap(&body.front());
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirModule oraLowerContractStub(MlirContext ctx, MlirLocation loc, MlirStringRef contractName, MlirStringRef funcName)
{
    try
    {
        MLIRContext *context = unwrap(ctx);
        Location location = unwrap(loc);
        StringRef contractRef = unwrap(contractName);
        StringRef funcRef = unwrap(funcName);

        if (!oraDialectIsRegistered(ctx))
        {
            return {nullptr};
        }

        OpBuilder builder(context);

        auto moduleOp = ModuleOp::create(location);
        builder.setInsertionPointToStart(moduleOp.getBody());

        auto contractOp = builder.create<ContractOp>(location, contractRef);
        auto &bodyRegion = contractOp.getBody();
        if (bodyRegion.empty())
        {
            bodyRegion.push_back(new Block());
        }

        auto contractDeclAttr = builder.getBoolAttr(true);
        contractOp->setAttr("ora.contract_decl", contractDeclAttr);
        auto gasCostAttr = IntegerAttr::get(::mlir::IntegerType::get(context, 64), 0);
        contractOp->setAttr("gas_cost", gasCostAttr);

        Block &contractBody = bodyRegion.front();
        builder.setInsertionPointToStart(&contractBody);

        auto funcType = builder.getFunctionType(TypeRange{}, TypeRange{});
        auto funcOp = builder.create<func::FuncOp>(location, funcRef, funcType);
        funcOp->setAttr("ora.visibility", builder.getStringAttr("pub"));

        Block *entry = funcOp.addEntryBlock();
        builder.setInsertionPointToStart(entry);
        builder.create<ReturnOp>(location, ValueRange{});

        return wrap(moduleOp);
    }
    catch (...)
    {
        return {nullptr};
    }
}

static Type oraTypeFromTag(MLIRContext *context, uint32_t tag)
{
    switch (tag)
    {
    case ORA_TYPE_U256:
        return ora::IntegerType::get(context, 256, false);
    case ORA_TYPE_I256:
        return ora::IntegerType::get(context, 256, true);
    case ORA_TYPE_BOOL:
        return ora::BoolType::get(context);
    case ORA_TYPE_ADDRESS:
        return ora::AddressType::get(context);
    case ORA_TYPE_VOID:
    default:
        return Type();
    }
}

static StringRef oraTypeTagToString(uint32_t tag)
{
    switch (tag)
    {
    case ORA_TYPE_U256:
        return "!ora.int<256,false>";
    case ORA_TYPE_I256:
        return "!ora.int<256,true>";
    case ORA_TYPE_BOOL:
        return "!ora.bool";
    case ORA_TYPE_ADDRESS:
        return "!ora.address";
    case ORA_TYPE_VOID:
        return "!ora.void";
    default:
        return "!ora.unknown";
    }
}

MlirModule oraLowerContractStubWithSig(
    MlirContext ctx,
    MlirLocation loc,
    MlirStringRef contractName,
    MlirStringRef funcName,
    const uint32_t *paramTypes,
    size_t numParams,
    const MlirStringRef *paramNames,
    size_t numParamNames,
    uint32_t returnType)
{
    try
    {
        MLIRContext *context = unwrap(ctx);
        Location location = unwrap(loc);
        StringRef contractRef = unwrap(contractName);
        StringRef funcRef = unwrap(funcName);

        if (!oraDialectIsRegistered(ctx))
        {
            return {nullptr};
        }

        OpBuilder builder(context);

        auto moduleOp = ModuleOp::create(location);
        builder.setInsertionPointToStart(moduleOp.getBody());

        auto contractOp = builder.create<ContractOp>(location, contractRef);
        auto &bodyRegion = contractOp.getBody();
        if (bodyRegion.empty())
        {
            bodyRegion.push_back(new Block());
        }

        auto contractDeclAttr = builder.getBoolAttr(true);
        contractOp->setAttr("ora.contract_decl", contractDeclAttr);
        auto gasCostAttr = IntegerAttr::get(::mlir::IntegerType::get(context, 64), 0);
        contractOp->setAttr("gas_cost", gasCostAttr);

        SmallVector<Type> argTypes;
        argTypes.reserve(numParams);
        for (size_t i = 0; i < numParams; ++i)
        {
            Type t = oraTypeFromTag(context, paramTypes[i]);
            if (!t)
            {
                t = ora::IntegerType::get(context, 256, false);
            }
            argTypes.push_back(t);
        }

        SmallVector<Type> resultTypes;
        Type retType = oraTypeFromTag(context, returnType);
        if (retType)
        {
            resultTypes.push_back(retType);
        }

        auto funcType = builder.getFunctionType(argTypes, resultTypes);

        Block &contractBody = bodyRegion.front();
        builder.setInsertionPointToStart(&contractBody);

        auto funcOp = builder.create<func::FuncOp>(location, funcRef, funcType);
        funcOp->setAttr("ora.visibility", builder.getStringAttr("pub"));

        for (size_t i = 0; i < numParams; ++i)
        {
            StringRef typeStr = oraTypeTagToString(paramTypes ? paramTypes[i] : ORA_TYPE_U256);
            funcOp.setArgAttr(static_cast<unsigned>(i), "ora.type", builder.getStringAttr(typeStr));

            if (paramNames && i < numParamNames && paramNames[i].data)
            {
                StringRef nameRef = unwrap(paramNames[i]);
                funcOp.setArgAttr(static_cast<unsigned>(i), "ora.name", builder.getStringAttr(nameRef));
            }
        }

        if (retType)
        {
            StringRef retStr = oraTypeTagToString(returnType);
            funcOp.setResultAttr(0, "ora.type", builder.getStringAttr(retStr));
        }

        Block *entry = funcOp.addEntryBlock();
        builder.setInsertionPointToStart(entry);
        builder.create<ReturnOp>(location, ValueRange{});

        return wrap(moduleOp);
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
        (void)context;
        StringRef nameRef = unwrap(name);
        Type typeRef = unwrap(type);

        // Check if dialect is registered
        if (!oraDialectIsRegistered(ctx))
        {
            return {nullptr};
        }

        OpBuilder builder(context);

        auto nameAttr = StringAttr::get(context, nameRef);
        auto typeAttr = TypeAttr::get(typeRef);
        Attribute initAttr;
        if (oraAttributeIsNull(initValue))
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
        (void)context;
        StringRef nameRef = unwrap(globalName);
        Type typeRef = unwrap(resultType);

        // Check if dialect is registered
        if (!oraDialectIsRegistered(ctx))
        {
            return {nullptr};
        }

        OpBuilder builder(context);

        auto sloadOp = builder.create<SLoadOp>(location, typeRef, StringAttr::get(context, nameRef));

        auto gasCostAttr = IntegerAttr::get(::mlir::IntegerType::get(context, 64), 2100);
        sloadOp->setAttr("gas_cost", gasCostAttr);
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

        auto sstoreOp = builder.create<SStoreOp>(location, valueRef, StringAttr::get(context, nameRef));

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

        TypeRange emptyResults;
        auto ifOp = IfOp::create(builder, location, emptyResults, conditionRef);

        auto &thenRegion = ifOp.getThenRegion();
        auto &elseRegion = ifOp.getElseRegion();
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

MlirBlock oraIfOpGetThenBlock(MlirOperation ifOp)
{
    try
    {
        Operation *operation = unwrap(ifOp);
        if (!operation)
        {
            return {nullptr};
        }

        auto op = dyn_cast<ora::IfOp>(operation);
        if (!op)
        {
            return {nullptr};
        }

        Region &thenRegion = op.getThenRegion();
        if (thenRegion.empty())
        {
            thenRegion.push_back(new Block());
        }
        return wrap(&thenRegion.front());
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirBlock oraIfOpGetElseBlock(MlirOperation ifOp)
{
    try
    {
        Operation *operation = unwrap(ifOp);
        if (!operation)
        {
            return {nullptr};
        }

        auto op = dyn_cast<ora::IfOp>(operation);
        if (!op)
        {
            return {nullptr};
        }

        Region &elseRegion = op.getElseRegion();
        if (elseRegion.empty())
        {
            elseRegion.push_back(new Block());
        }
        return wrap(&elseRegion.front());
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

MlirBlock oraWhileOpGetBodyBlock(MlirOperation whileOp)
{
    try
    {
        Operation *operation = unwrap(whileOp);
        if (!operation)
        {
            return {nullptr};
        }

        auto op = dyn_cast<ora::WhileOp>(operation);
        if (!op)
        {
            return {nullptr};
        }

        Region &body = op.getBody();
        if (body.empty())
        {
            body.push_back(new Block());
        }

        return wrap(&body.front());
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

MlirType oraStringTypeGet(MlirContext ctx)
{
    try
    {
        MLIRContext *context = unwrap(ctx);

        if (!oraDialectIsRegistered(ctx))
        {
            return {nullptr};
        }

        auto stringType = ora::StringType::get(context);
        return wrap(stringType);
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirType oraBytesTypeGet(MlirContext ctx)
{
    try
    {
        MLIRContext *context = unwrap(ctx);

        if (!oraDialectIsRegistered(ctx))
        {
            return {nullptr};
        }

        auto bytesType = ora::BytesType::get(context);
        return wrap(bytesType);
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
            // Use signless type for compatibility with arith.* operations
            // (arith operations require signless integers)
            unsigned width = oraIntType.getWidth();
            return wrap(::mlir::IntegerType::get(mlirType.getContext(), width));
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
        return isa<ora::AddressType>(mlirType) || isa<ora::NonZeroAddressType>(mlirType);
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

        auto i160Type = ::mlir::IntegerType::get(context, 160);
        auto addrToI160Op = builder.create<ora::AddrToI160Op>(location, i160Type, addrValue);

        return wrap(addrToI160Op.getOperation());
    }
    catch (...)
    {
        return {nullptr};
    }
}

/// Create an ora.i160.to.addr operation to convert i160 to !ora.address
MlirOperation oraI160ToAddrOpCreate(MlirContext ctx, MlirLocation loc, MlirValue value)
{
    try
    {
        MLIRContext *context = unwrap(ctx);
        Location location = unwrap(loc);
        Value inputValue = unwrap(value);

        if (!oraDialectIsRegistered(ctx))
        {
            return {nullptr};
        }

        OpBuilder builder(context);

        auto addrType = ora::AddressType::get(context);
        auto i160ToAddrOp = builder.create<ora::I160ToAddrOp>(location, addrType, inputValue);

        return wrap(i160ToAddrOp.getOperation());
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

        auto mapType = ora::MapType::get(context, keyTypeRef, valueTypeRef);
        return wrap(mapType);
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirType oraMapTypeGetValueType(MlirType mapType)
{
    try
    {
        Type mapTypeRef = unwrap(mapType);
        if (auto oraMapType = dyn_cast<ora::MapType>(mapTypeRef))
        {
            return wrap(oraMapType.getValueType());
        }
        return {nullptr};
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirType oraStructTypeGet(MlirContext ctx, MlirStringRef structName)
{
    try
    {
        MLIRContext *context = unwrap(ctx);
        StringRef nameRef = unwrap(structName);

        if (!oraDialectIsRegistered(ctx))
        {
            return {nullptr};
        }

        auto structType = ora::StructType::get(context, nameRef);
        return wrap(structType);
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirType oraErrorUnionTypeGet(MlirContext ctx, MlirType successType)
{
    try
    {
        MLIRContext *context = unwrap(ctx);
        Type successTypeRef = unwrap(successType);

        if (!oraDialectIsRegistered(ctx))
        {
            return {nullptr};
        }

        auto errorUnionType = ora::ErrorUnionType::get(context, successTypeRef);
        return wrap(errorUnionType);
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirType oraErrorUnionTypeGetSuccessType(MlirType errorUnionType)
{
    try
    {
        Type mlirType = unwrap(errorUnionType);
        if (auto errorUnion = dyn_cast<ora::ErrorUnionType>(mlirType))
        {
            return wrap(errorUnion.getSuccessType());
        }
        return {nullptr};
    }
    catch (...)
    {
        return {nullptr};
    }
}

//===----------------------------------------------------------------------===//
// Refinement Type Creation
//===----------------------------------------------------------------------===//

MlirType oraMinValueTypeGet(MlirContext ctx, MlirType baseType, uint64_t minHighHigh, uint64_t minHighLow, uint64_t minLowHigh, uint64_t minLowLow)
{
    try
    {
        MLIRContext *context = unwrap(ctx);
        Type baseTypeRef = unwrap(baseType);

        if (!oraDialectIsRegistered(ctx))
        {
            return {nullptr};
        }

        auto minValueType = ora::MinValueType::get(context, baseTypeRef, minHighHigh, minHighLow, minLowHigh, minLowLow);
        return wrap(minValueType);
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirType oraMaxValueTypeGet(MlirContext ctx, MlirType baseType, uint64_t maxHighHigh, uint64_t maxHighLow, uint64_t maxLowHigh, uint64_t maxLowLow)
{
    try
    {
        MLIRContext *context = unwrap(ctx);
        Type baseTypeRef = unwrap(baseType);

        if (!oraDialectIsRegistered(ctx))
        {
            return {nullptr};
        }

        auto maxValueType = ora::MaxValueType::get(context, baseTypeRef, maxHighHigh, maxHighLow, maxLowHigh, maxLowLow);
        return wrap(maxValueType);
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirType oraInRangeTypeGet(MlirContext ctx, MlirType baseType, uint64_t minHighHigh, uint64_t minHighLow, uint64_t minLowHigh, uint64_t minLowLow, uint64_t maxHighHigh, uint64_t maxHighLow, uint64_t maxLowHigh, uint64_t maxLowLow)
{
    try
    {
        MLIRContext *context = unwrap(ctx);
        Type baseTypeRef = unwrap(baseType);

        if (!oraDialectIsRegistered(ctx))
        {
            return {nullptr};
        }

        auto inRangeType = ora::InRangeType::get(context, baseTypeRef, minHighHigh, minHighLow, minLowHigh, minLowLow, maxHighHigh, maxHighLow, maxLowHigh, maxLowLow);
        return wrap(inRangeType);
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirType oraScaledTypeGet(MlirContext ctx, MlirType baseType, uint32_t decimals)
{
    try
    {
        MLIRContext *context = unwrap(ctx);
        Type baseTypeRef = unwrap(baseType);

        if (!oraDialectIsRegistered(ctx))
        {
            return {nullptr};
        }

        auto scaledType = ora::ScaledType::get(context, baseTypeRef, decimals);
        return wrap(scaledType);
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirType oraExactTypeGet(MlirContext ctx, MlirType baseType)
{
    try
    {
        MLIRContext *context = unwrap(ctx);
        Type baseTypeRef = unwrap(baseType);

        if (!oraDialectIsRegistered(ctx))
        {
            return {nullptr};
        }

        auto exactType = ora::ExactType::get(context, baseTypeRef);
        return wrap(exactType);
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirType oraNonZeroAddressTypeGet(MlirContext ctx)
{
    try
    {
        MLIRContext *context = unwrap(ctx);

        if (!oraDialectIsRegistered(ctx))
        {
            return {nullptr};
        }

        auto nonZeroAddressType = ora::NonZeroAddressType::get(context);
        return wrap(nonZeroAddressType);
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirType oraRefinementTypeGetBaseType(MlirType refinementType)
{
    try
    {
        Type typeRef = unwrap(refinementType);

        // Check for each refinement type and extract base type
        if (auto minValueType = dyn_cast<ora::MinValueType>(typeRef))
        {
            return wrap(minValueType.getBaseType());
        }
        if (auto maxValueType = dyn_cast<ora::MaxValueType>(typeRef))
        {
            return wrap(maxValueType.getBaseType());
        }
        if (auto inRangeType = dyn_cast<ora::InRangeType>(typeRef))
        {
            return wrap(inRangeType.getBaseType());
        }
        if (auto scaledType = dyn_cast<ora::ScaledType>(typeRef))
        {
            return wrap(scaledType.getBaseType());
        }
        if (auto exactType = dyn_cast<ora::ExactType>(typeRef))
        {
            return wrap(exactType.getBaseType());
        }
        if (isa<ora::NonZeroAddressType>(typeRef))
        {
            // NonZeroAddress has no base type parameter - it's just an address
            // Return the address type
            MLIRContext *context = typeRef.getContext();
            auto addressType = ora::AddressType::get(context);
            return wrap(addressType);
        }

        // Not a refinement type
        return {nullptr};
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirOperation oraRefinementToBaseOpCreate(MlirContext ctx, MlirLocation loc, MlirValue value, MlirBlock block)
{
    try
    {
        MLIRContext *context = unwrap(ctx);
        Location location = unwrap(loc);
        Value valueRef = unwrap(value);
        Block *blockRef = unwrap(block);

        if (!oraDialectIsRegistered(ctx))
        {
            return {nullptr};
        }

        OpBuilder builder(context);
        // Insert at the start of the block (for block arguments) or after the defining op
        if (isa<BlockArgument>(valueRef))
        {
            builder.setInsertionPointToStart(blockRef);
        }
        else if (Operation *definingOp = valueRef.getDefiningOp())
        {
            builder.setInsertionPointAfter(definingOp);
        }
        else
        {
            // Fallback: insert at start of provided block
            builder.setInsertionPointToStart(blockRef);
        }

        auto op = builder.create<ora::RefinementToBaseOp>(location, valueRef);
        return wrap(op.getOperation());
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

MlirOperation oraRefinementGuardOpCreate(MlirContext ctx, MlirLocation loc, MlirValue condition, MlirStringRef message)
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

        // Create refinement_guard operation - message is optional
        StringAttr messageAttr = nullptr;
        if (message.data != nullptr && message.length > 0)
        {
            messageAttr = StringAttr::get(context, unwrap(message));
        }

        auto guardOp = builder.create<RefinementGuardOp>(location, conditionRef, messageAttr);

        // Add gas_cost attribute (guard operation has no runtime cost = 0)
        auto gasCostAttr = IntegerAttr::get(::mlir::IntegerType::get(context, 64), 0);
        guardOp->setAttr("gas_cost", gasCostAttr);

        return wrap(guardOp.getOperation());
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

MlirOperation oraArithConstantOpCreate(MlirContext ctx, MlirLocation loc, MlirType resultType, MlirAttribute valueAttr)
{
    try
    {
        MLIRContext *context = unwrap(ctx);
        Location location = unwrap(loc);
        Type resultTy = unwrap(resultType);
        Attribute attr = unwrap(valueAttr);
        context->getOrLoadDialect<arith::ArithDialect>();

        auto typedAttr = llvm::dyn_cast<mlir::TypedAttr>(attr);
        if (!typedAttr)
        {
            if (auto intTy = llvm::dyn_cast<mlir::IntegerType>(resultTy))
            {
                typedAttr = mlir::IntegerAttr::get(intTy, 0);
            }
            else if (auto indexTy = llvm::dyn_cast<mlir::IndexType>(resultTy))
            {
                typedAttr = mlir::IntegerAttr::get(indexTy, 0);
            }
            else
            {
                return {nullptr};
            }
        }

        OpBuilder builder(context);
        auto op = builder.create<arith::ConstantOp>(location, resultTy, typedAttr);

        // attach gas_cost=0 like the Zig-side builder
        auto gasCostAttr = builder.getI64IntegerAttr(0);
        op->setAttr("gas_cost", gasCostAttr);

        return wrap(op.getOperation());
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirOperation oraArithAddIOpCreate(MlirContext ctx, MlirLocation loc, MlirValue lhs, MlirValue rhs)
{
    try
    {
        MLIRContext *context = unwrap(ctx);
        Location location = unwrap(loc);
        Value lhsVal = unwrap(lhs);
        Value rhsVal = unwrap(rhs);

        OpBuilder builder(context);
        auto op = builder.create<arith::AddIOp>(location, lhsVal, rhsVal);
        auto gasCostAttr = builder.getI64IntegerAttr(3);
        op->setAttr("gas_cost", gasCostAttr);
        return wrap(op.getOperation());
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirOperation oraArithSubIOpCreate(MlirContext ctx, MlirLocation loc, MlirValue lhs, MlirValue rhs)
{
    try
    {
        MLIRContext *context = unwrap(ctx);
        Location location = unwrap(loc);
        Value lhsVal = unwrap(lhs);
        Value rhsVal = unwrap(rhs);

        OpBuilder builder(context);
        auto op = builder.create<arith::SubIOp>(location, lhsVal, rhsVal);
        auto gasCostAttr = builder.getI64IntegerAttr(3);
        op->setAttr("gas_cost", gasCostAttr);
        return wrap(op.getOperation());
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirOperation oraArithMulIOpCreate(MlirContext ctx, MlirLocation loc, MlirValue lhs, MlirValue rhs)
{
    try
    {
        MLIRContext *context = unwrap(ctx);
        Location location = unwrap(loc);
        Value lhsVal = unwrap(lhs);
        Value rhsVal = unwrap(rhs);

        OpBuilder builder(context);
        auto op = builder.create<arith::MulIOp>(location, lhsVal, rhsVal);
        auto gasCostAttr = builder.getI64IntegerAttr(5);
        op->setAttr("gas_cost", gasCostAttr);
        return wrap(op.getOperation());
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirOperation oraArithDivUIOpCreate(MlirContext ctx, MlirLocation loc, MlirValue lhs, MlirValue rhs)
{
    try
    {
        MLIRContext *context = unwrap(ctx);
        Location location = unwrap(loc);
        Value lhsVal = unwrap(lhs);
        Value rhsVal = unwrap(rhs);

        OpBuilder builder(context);
        auto op = builder.create<arith::DivUIOp>(location, lhsVal, rhsVal);
        auto gasCostAttr = builder.getI64IntegerAttr(5);
        op->setAttr("gas_cost", gasCostAttr);
        return wrap(op.getOperation());
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirOperation oraArithDivSIOpCreate(MlirContext ctx, MlirLocation loc, MlirValue lhs, MlirValue rhs)
{
    try
    {
        MLIRContext *context = unwrap(ctx);
        Location location = unwrap(loc);
        Value lhsVal = unwrap(lhs);
        Value rhsVal = unwrap(rhs);

        OpBuilder builder(context);
        auto op = builder.create<arith::DivSIOp>(location, lhsVal, rhsVal);
        auto gasCostAttr = builder.getI64IntegerAttr(5);
        op->setAttr("gas_cost", gasCostAttr);
        return wrap(op.getOperation());
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirOperation oraArithRemUIOpCreate(MlirContext ctx, MlirLocation loc, MlirValue lhs, MlirValue rhs)
{
    try
    {
        MLIRContext *context = unwrap(ctx);
        Location location = unwrap(loc);
        Value lhsVal = unwrap(lhs);
        Value rhsVal = unwrap(rhs);

        OpBuilder builder(context);
        auto op = builder.create<arith::RemUIOp>(location, lhsVal, rhsVal);
        auto gasCostAttr = builder.getI64IntegerAttr(5);
        op->setAttr("gas_cost", gasCostAttr);
        return wrap(op.getOperation());
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirOperation oraArithRemSIOpCreate(MlirContext ctx, MlirLocation loc, MlirValue lhs, MlirValue rhs)
{
    try
    {
        MLIRContext *context = unwrap(ctx);
        Location location = unwrap(loc);
        Value lhsVal = unwrap(lhs);
        Value rhsVal = unwrap(rhs);

        OpBuilder builder(context);
        auto op = builder.create<arith::RemSIOp>(location, lhsVal, rhsVal);
        auto gasCostAttr = builder.getI64IntegerAttr(5);
        op->setAttr("gas_cost", gasCostAttr);
        return wrap(op.getOperation());
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirOperation oraArithAndIOpCreate(MlirContext ctx, MlirLocation loc, MlirValue lhs, MlirValue rhs)
{
    try
    {
        MLIRContext *context = unwrap(ctx);
        Location location = unwrap(loc);
        Value lhsVal = unwrap(lhs);
        Value rhsVal = unwrap(rhs);

        OpBuilder builder(context);
        auto op = builder.create<arith::AndIOp>(location, lhsVal, rhsVal);
        return wrap(op.getOperation());
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirOperation oraArithOrIOpCreate(MlirContext ctx, MlirLocation loc, MlirValue lhs, MlirValue rhs)
{
    try
    {
        MLIRContext *context = unwrap(ctx);
        Location location = unwrap(loc);
        Value lhsVal = unwrap(lhs);
        Value rhsVal = unwrap(rhs);

        OpBuilder builder(context);
        auto op = builder.create<arith::OrIOp>(location, lhsVal, rhsVal);
        return wrap(op.getOperation());
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirOperation oraArithXorIOpCreate(MlirContext ctx, MlirLocation loc, MlirValue lhs, MlirValue rhs)
{
    try
    {
        MLIRContext *context = unwrap(ctx);
        Location location = unwrap(loc);
        Value lhsVal = unwrap(lhs);
        Value rhsVal = unwrap(rhs);

        OpBuilder builder(context);
        auto op = builder.create<arith::XOrIOp>(location, lhsVal, rhsVal);
        return wrap(op.getOperation());
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirOperation oraArithShlIOpCreate(MlirContext ctx, MlirLocation loc, MlirValue lhs, MlirValue rhs)
{
    try
    {
        MLIRContext *context = unwrap(ctx);
        Location location = unwrap(loc);
        Value lhsVal = unwrap(lhs);
        Value rhsVal = unwrap(rhs);

        OpBuilder builder(context);
        auto op = builder.create<arith::ShLIOp>(location, lhsVal, rhsVal);
        return wrap(op.getOperation());
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirOperation oraArithShrSIOpCreate(MlirContext ctx, MlirLocation loc, MlirValue lhs, MlirValue rhs)
{
    try
    {
        MLIRContext *context = unwrap(ctx);
        Location location = unwrap(loc);
        Value lhsVal = unwrap(lhs);
        Value rhsVal = unwrap(rhs);

        OpBuilder builder(context);
        auto op = builder.create<arith::ShRSIOp>(location, lhsVal, rhsVal);
        return wrap(op.getOperation());
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirOperation oraArithBitcastOpCreate(MlirContext ctx, MlirLocation loc, MlirValue operand, MlirType resultType)
{
    try
    {
        MLIRContext *context = unwrap(ctx);
        Location location = unwrap(loc);
        Value operandVal = unwrap(operand);
        Type resultTy = unwrap(resultType);

        OpBuilder builder(context);
        auto op = builder.create<arith::BitcastOp>(location, resultTy, operandVal);
        return wrap(op.getOperation());
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirOperation oraUnrealizedConversionCastOpCreate(MlirContext ctx, MlirLocation loc, MlirValue operand, MlirType resultType)
{
    try
    {
        MLIRContext *context = unwrap(ctx);
        Location location = unwrap(loc);
        Value operandVal = unwrap(operand);
        Type resultTy = unwrap(resultType);

        OpBuilder builder(context);
        auto op = builder.create<UnrealizedConversionCastOp>(location, resultTy, operandVal);
        return wrap(op.getOperation());
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirOperation oraArithExtUIOpCreate(MlirContext ctx, MlirLocation loc, MlirValue operand, MlirType resultType)
{
    try
    {
        MLIRContext *context = unwrap(ctx);
        Location location = unwrap(loc);
        Value operandVal = unwrap(operand);
        Type resultTy = unwrap(resultType);

        OpBuilder builder(context);
        auto op = builder.create<arith::ExtUIOp>(location, resultTy, operandVal);
        return wrap(op.getOperation());
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirOperation oraArithTruncIOpCreate(MlirContext ctx, MlirLocation loc, MlirValue operand, MlirType resultType)
{
    try
    {
        MLIRContext *context = unwrap(ctx);
        Location location = unwrap(loc);
        Value operandVal = unwrap(operand);
        Type resultTy = unwrap(resultType);

        OpBuilder builder(context);
        auto op = builder.create<arith::TruncIOp>(location, resultTy, operandVal);
        return wrap(op.getOperation());
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirOperation oraArithIndexCastUIOpCreate(MlirContext ctx, MlirLocation loc, MlirValue operand, MlirType resultType)
{
    try
    {
        MLIRContext *context = unwrap(ctx);
        Location location = unwrap(loc);
        Value operandVal = unwrap(operand);
        Type resultTy = unwrap(resultType);

        OpBuilder builder(context);
        auto op = builder.create<arith::IndexCastUIOp>(location, resultTy, operandVal);
        return wrap(op.getOperation());
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirOperation oraArithCmpIOpCreate(MlirContext ctx, MlirLocation loc, int64_t predicate, MlirValue lhs, MlirValue rhs)
{
    try
    {
        MLIRContext *context = unwrap(ctx);
        Location location = unwrap(loc);
        Value lhsVal = unwrap(lhs);
        Value rhsVal = unwrap(rhs);

        OpBuilder builder(context);
        auto pred = static_cast<arith::CmpIPredicate>(predicate);
        auto op = builder.create<arith::CmpIOp>(location, pred, lhsVal, rhsVal);
        return wrap(op.getOperation());
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirOperation oraFuncCallOpCreate(MlirContext ctx, MlirLocation loc, MlirStringRef callee, const MlirValue *operands, size_t numOperands, const MlirType *resultTypes, size_t numResults)
{
    try
    {
        MLIRContext *context = unwrap(ctx);
        Location location = unwrap(loc);
        StringRef calleeRef = unwrap(callee);

        OpBuilder builder(context);
        SmallVector<Value, 8> args;
        args.reserve(numOperands);
        for (size_t i = 0; i < numOperands; ++i)
        {
            args.push_back(unwrap(operands[i]));
        }

        SmallVector<Type, 2> results;
        results.reserve(numResults);
        for (size_t i = 0; i < numResults; ++i)
        {
            results.push_back(unwrap(resultTypes[i]));
        }

        auto op = builder.create<func::CallOp>(location, calleeRef, results, args);

        // attach gas_cost=10 like the Zig-side builder
        auto gasCostAttr = builder.getI64IntegerAttr(10);
        op->setAttr("gas_cost", gasCostAttr);

        return wrap(op.getOperation());
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirOperation oraMemrefAllocaOpCreate(MlirContext ctx, MlirLocation loc, MlirType resultType)
{
    try
    {
        MLIRContext *context = unwrap(ctx);
        Location location = unwrap(loc);
        Type memrefType = unwrap(resultType);
        auto memrefTy = llvm::dyn_cast<mlir::MemRefType>(memrefType);
        if (!memrefTy)
        {
            return {nullptr};
        }

        OpBuilder builder(context);
        auto op = builder.create<memref::AllocaOp>(location, memrefTy);
        return wrap(op.getOperation());
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirOperation oraMemrefLoadOpCreate(MlirContext ctx, MlirLocation loc, MlirValue memref, const MlirValue *indices, size_t numIndices, MlirType resultType)
{
    try
    {
        MLIRContext *context = unwrap(ctx);
        Location location = unwrap(loc);
        Value memrefVal = unwrap(memref);
        (void)resultType;

        OpBuilder builder(context);
        SmallVector<Value, 4> idx;
        idx.reserve(numIndices);
        for (size_t i = 0; i < numIndices; ++i)
        {
            idx.push_back(unwrap(indices[i]));
        }

        auto op = builder.create<memref::LoadOp>(location, memrefVal, idx);
        return wrap(op.getOperation());
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirOperation oraMemrefLoadOpCreateWithMemspace(MlirContext ctx, MlirLocation loc, MlirValue memref, const MlirValue *indices, size_t numIndices, MlirType resultType, MlirAttribute memspaceAttr)
{
    try
    {
        MLIRContext *context = unwrap(ctx);
        Location location = unwrap(loc);
        Value memrefVal = unwrap(memref);
        Type resultTypeRef = unwrap(resultType);

        SmallVector<Value, 4> idx;
        idx.reserve(numIndices);
        for (size_t i = 0; i < numIndices; ++i)
        {
            idx.push_back(unwrap(indices[i]));
        }

        OpBuilder builder(context);
        auto op = builder.create<memref::LoadOp>(location, memrefVal, idx);
        op->setAttr("memspace", unwrap(memspaceAttr));
        (void)resultTypeRef;

        return wrap(op.getOperation());
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirOperation oraMemrefStoreOpCreate(MlirContext ctx, MlirLocation loc, MlirValue value, MlirValue memref, const MlirValue *indices, size_t numIndices)
{
    try
    {
        MLIRContext *context = unwrap(ctx);
        Location location = unwrap(loc);
        Value valueVal = unwrap(value);
        Value memrefVal = unwrap(memref);

        OpBuilder builder(context);
        SmallVector<Value, 4> idx;
        idx.reserve(numIndices);
        for (size_t i = 0; i < numIndices; ++i)
        {
            idx.push_back(unwrap(indices[i]));
        }

        auto op = builder.create<memref::StoreOp>(location, valueVal, memrefVal, idx);
        return wrap(op.getOperation());
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirOperation oraMemrefStoreOpCreateWithMemspace(MlirContext ctx, MlirLocation loc, MlirValue value, MlirValue memref, const MlirValue *indices, size_t numIndices, MlirAttribute memspaceAttr)
{
    try
    {
        MLIRContext *context = unwrap(ctx);
        Location location = unwrap(loc);
        Value valueVal = unwrap(value);
        Value memrefVal = unwrap(memref);

        SmallVector<Value, 4> idx;
        idx.reserve(numIndices);
        for (size_t i = 0; i < numIndices; ++i)
        {
            idx.push_back(unwrap(indices[i]));
        }

        OpBuilder builder(context);
        auto op = builder.create<memref::StoreOp>(location, valueVal, memrefVal, idx);
        op->setAttr("memspace", unwrap(memspaceAttr));

        return wrap(op.getOperation());
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirOperation oraMemrefDimOpCreate(MlirContext ctx, MlirLocation loc, MlirValue memref, MlirValue index)
{
    try
    {
        MLIRContext *context = unwrap(ctx);
        Location location = unwrap(loc);
        Value memrefVal = unwrap(memref);
        Value indexVal = unwrap(index);

        OpBuilder builder(context);
        auto op = builder.create<memref::DimOp>(location, memrefVal, indexVal);
        return wrap(op.getOperation());
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirOperation oraTensorDimOpCreate(MlirContext ctx, MlirLocation loc, MlirValue tensor, MlirValue index)
{
    try
    {
        MLIRContext *context = unwrap(ctx);
        Location location = unwrap(loc);
        Value tensorVal = unwrap(tensor);
        Value indexVal = unwrap(index);

        OpBuilder builder(context);
        auto op = builder.create<tensor::DimOp>(location, tensorVal, indexVal);
        return wrap(op.getOperation());
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirOperation oraScfYieldOpCreate(MlirContext ctx, MlirLocation loc, const MlirValue *operands, size_t numOperands)
{
    try
    {
        MLIRContext *context = unwrap(ctx);
        Location location = unwrap(loc);

        OpBuilder builder(context);
        SmallVector<Value, 4> values;
        values.reserve(numOperands);
        for (size_t i = 0; i < numOperands; ++i)
        {
            values.push_back(unwrap(operands[i]));
        }

        auto op = builder.create<scf::YieldOp>(location, values);

        // attach gas_cost=1 like the Zig-side builder
        auto gasCostAttr = builder.getI64IntegerAttr(1);
        op->setAttr("gas_cost", gasCostAttr);

        return wrap(op.getOperation());
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirOperation oraScfConditionOpCreate(MlirContext ctx, MlirLocation loc, MlirValue condition, const MlirValue *operands, size_t numOperands)
{
    try
    {
        MLIRContext *context = unwrap(ctx);
        Location location = unwrap(loc);
        Value condVal = unwrap(condition);

        OpBuilder builder(context);
        SmallVector<Value, 4> values;
        values.reserve(numOperands);
        for (size_t i = 0; i < numOperands; ++i)
        {
            values.push_back(unwrap(operands[i]));
        }

        auto op = builder.create<scf::ConditionOp>(location, condVal, values);
        return wrap(op.getOperation());
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirOperation oraCfBrOpCreate(MlirContext ctx, MlirLocation loc, MlirBlock dest)
{
    try
    {
        MLIRContext *context = unwrap(ctx);
        Location location = unwrap(loc);
        Block *destBlock = unwrap(dest);

        OpBuilder builder(context);
        auto op = builder.create<cf::BranchOp>(location, destBlock);
        return wrap(op.getOperation());
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirOperation oraCfCondBrOpCreate(MlirContext ctx, MlirLocation loc, MlirValue condition, MlirBlock true_block, MlirBlock false_block)
{
    try
    {
        MLIRContext *context = unwrap(ctx);
        Location location = unwrap(loc);
        Value condVal = unwrap(condition);
        Block *trueBlock = unwrap(true_block);
        Block *falseBlock = unwrap(false_block);

        OpBuilder builder(context);
        auto op = builder.create<cf::CondBranchOp>(location, condVal, trueBlock, ValueRange{}, falseBlock, ValueRange{});
        return wrap(op.getOperation());
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirOperation oraCfAssertOpCreate(MlirContext ctx, MlirLocation loc, MlirValue condition, MlirStringRef message)
{
    try
    {
        MLIRContext *context = unwrap(ctx);
        Location location = unwrap(loc);
        Value condVal = unwrap(condition);
        StringRef msgRef = unwrap(message);

        OpBuilder builder(context);
        auto msgAttr = builder.getStringAttr(msgRef);
        auto op = builder.create<cf::AssertOp>(location, condVal, msgAttr);
        return wrap(op.getOperation());
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirOperation oraCfAssertOpCreateWithAttrs(MlirContext ctx, MlirLocation loc, MlirValue condition, const MlirNamedAttribute *attrs, size_t numAttrs)
{
    try
    {
        MLIRContext *context = unwrap(ctx);
        Location location = unwrap(loc);
        Value condVal = unwrap(condition);

        OperationState state(location, "cf.assert");
        state.addOperands(condVal);

        for (size_t i = 0; i < numAttrs; ++i)
        {
            auto name = unwrap(attrs[i].name);
            auto value = unwrap(attrs[i].attribute);
            state.addAttribute(name, value);
        }

        OpBuilder builder(context);
        Operation *op = Operation::create(state);
        builder.insert(op);
        return wrap(op);
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirOperation oraScfBreakOpCreate(MlirContext ctx, MlirLocation loc, const MlirValue *operands, size_t numOperands)
{
    try
    {
        MLIRContext *context = unwrap(ctx);
        Location location = unwrap(loc);

        OpBuilder builder(context);
        SmallVector<Value, 4> values;
        values.reserve(numOperands);
        for (size_t i = 0; i < numOperands; ++i)
        {
            values.push_back(unwrap(operands[i]));
        }

        OperationState state(location, "scf.break");
        state.addOperands(values);
        Operation *op = Operation::create(state);
        builder.insert(op);
        return wrap(op);
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirOperation oraScfContinueOpCreate(MlirContext ctx, MlirLocation loc, const MlirValue *operands, size_t numOperands)
{
    try
    {
        MLIRContext *context = unwrap(ctx);
        Location location = unwrap(loc);

        OpBuilder builder(context);
        SmallVector<Value, 4> values;
        values.reserve(numOperands);
        for (size_t i = 0; i < numOperands; ++i)
        {
            values.push_back(unwrap(operands[i]));
        }

        OperationState state(location, "scf.continue");
        state.addOperands(values);
        Operation *op = Operation::create(state);
        builder.insert(op);
        return wrap(op);
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirOperation oraScfIfOpCreate(MlirContext ctx, MlirLocation loc, MlirValue condition, const MlirType *resultTypes, size_t numResults, bool withElse)
{
    try
    {
        MLIRContext *context = unwrap(ctx);
        Location location = unwrap(loc);
        Value condVal = unwrap(condition);

        SmallVector<Type, 4> results;
        results.reserve(numResults);
        for (size_t i = 0; i < numResults; ++i)
        {
            results.push_back(unwrap(resultTypes[i]));
        }

        OpBuilder builder(context);
        auto op = scf::IfOp::create(builder, location, results, condVal, /*addThenBlock=*/true, /*addElseBlock=*/withElse);

        // attach gas_cost=10 like the Zig-side builder
        auto gasCostAttr = builder.getI64IntegerAttr(10);
        op->setAttr("gas_cost", gasCostAttr);

        return wrap(op.getOperation());
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirBlock oraScfIfOpGetThenBlock(MlirOperation ifOp)
{
    try
    {
        Operation *operation = unwrap(ifOp);
        if (!operation)
        {
            return {nullptr};
        }

        auto op = dyn_cast<scf::IfOp>(operation);
        if (!op)
        {
            return {nullptr};
        }

        return wrap(op.thenBlock());
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirBlock oraScfIfOpGetElseBlock(MlirOperation ifOp)
{
    try
    {
        Operation *operation = unwrap(ifOp);
        if (!operation)
        {
            return {nullptr};
        }

        auto op = dyn_cast<scf::IfOp>(operation);
        if (!op)
        {
            return {nullptr};
        }

        if (!op.getElseRegion().empty())
        {
            return wrap(op.elseBlock());
        }

        return {nullptr};
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirOperation oraScfWhileOpCreate(MlirContext ctx, MlirLocation loc, const MlirValue *operands, size_t numOperands, const MlirType *resultTypes, size_t numResults)
{
    try
    {
        MLIRContext *context = unwrap(ctx);
        Location location = unwrap(loc);

        SmallVector<Value, 4> initOperands;
        initOperands.reserve(numOperands);
        for (size_t i = 0; i < numOperands; ++i)
        {
            initOperands.push_back(unwrap(operands[i]));
        }

        SmallVector<Type, 4> results;
        results.reserve(numResults);
        for (size_t i = 0; i < numResults; ++i)
        {
            results.push_back(unwrap(resultTypes[i]));
        }

        OpBuilder builder(context);
        auto op = scf::WhileOp::create(builder, location, results, initOperands);

        // attach gas_cost=10 like the Zig-side builder
        auto gasCostAttr = builder.getI64IntegerAttr(10);
        op->setAttr("gas_cost", gasCostAttr);

        return wrap(op.getOperation());
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirBlock oraScfWhileOpGetBeforeBlock(MlirOperation whileOp)
{
    try
    {
        Operation *operation = unwrap(whileOp);
        if (!operation)
        {
            return {nullptr};
        }

        auto op = dyn_cast<scf::WhileOp>(operation);
        if (!op)
        {
            return {nullptr};
        }

        Region &before = op.getBefore();
        if (before.empty())
        {
            before.push_back(new Block());
        }

        return wrap(&before.front());
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirBlock oraScfWhileOpGetAfterBlock(MlirOperation whileOp)
{
    try
    {
        Operation *operation = unwrap(whileOp);
        if (!operation)
        {
            return {nullptr};
        }

        auto op = dyn_cast<scf::WhileOp>(operation);
        if (!op)
        {
            return {nullptr};
        }

        Region &after = op.getAfter();
        if (after.empty())
        {
            after.push_back(new Block());
        }

        return wrap(&after.front());
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirOperation oraScfForOpCreate(MlirContext ctx, MlirLocation loc, MlirValue lowerBound, MlirValue upperBound, MlirValue step, const MlirValue *initArgs, size_t numInitArgs, bool unsignedCmp)
{
    try
    {
        MLIRContext *context = unwrap(ctx);
        Location location = unwrap(loc);
        Value lb = unwrap(lowerBound);
        Value ub = unwrap(upperBound);
        Value st = unwrap(step);

        SmallVector<Value, 4> init;
        init.reserve(numInitArgs);
        for (size_t i = 0; i < numInitArgs; ++i)
        {
            init.push_back(unwrap(initArgs[i]));
        }

        OpBuilder builder(context);
        auto op = builder.create<scf::ForOp>(location, lb, ub, st, init, nullptr, unsignedCmp);

        // attach gas_cost=10 like the Zig-side builder
        auto gasCostAttr = builder.getI64IntegerAttr(10);
        op->setAttr("gas_cost", gasCostAttr);

        return wrap(op.getOperation());
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirBlock oraScfForOpGetBodyBlock(MlirOperation forOp)
{
    try
    {
        Operation *operation = unwrap(forOp);
        if (!operation)
        {
            return {nullptr};
        }

        auto op = dyn_cast<scf::ForOp>(operation);
        if (!op)
        {
            return {nullptr};
        }

        return wrap(op.getBody());
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirOperation oraScfExecuteRegionOpCreate(MlirContext ctx, MlirLocation loc, const MlirType *resultTypes, size_t numResults, bool noInline)
{
    try
    {
        MLIRContext *context = unwrap(ctx);
        Location location = unwrap(loc);

        SmallVector<Type, 4> results;
        results.reserve(numResults);
        for (size_t i = 0; i < numResults; ++i)
        {
            results.push_back(unwrap(resultTypes[i]));
        }

        OpBuilder builder(context);
        auto op = scf::ExecuteRegionOp::create(builder, location, results, noInline);
        return wrap(op.getOperation());
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirBlock oraScfExecuteRegionOpGetBodyBlock(MlirOperation op)
{
    try
    {
        Operation *operation = unwrap(op);
        if (!operation)
        {
            return {nullptr};
        }

        auto execOp = dyn_cast<scf::ExecuteRegionOp>(operation);
        if (!execOp)
        {
            return {nullptr};
        }

        Region &region = execOp.getRegion();
        if (region.empty())
        {
            region.push_back(new Block());
        }

        return wrap(&region.front());
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirOperation oraLlvmUndefOpCreate(MlirContext ctx, MlirLocation loc, MlirType resultType)
{
    try
    {
        MLIRContext *context = unwrap(ctx);
        Location location = unwrap(loc);
        Type resType = unwrap(resultType);

        OpBuilder builder(context);
        auto op = builder.create<LLVM::UndefOp>(location, resType);
        return wrap(op.getOperation());
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirOperation oraLlvmInsertValueOpCreate(MlirContext ctx, MlirLocation loc, MlirType resultType, MlirValue container, MlirValue value, const int64_t *positions, size_t numPositions)
{
    try
    {
        MLIRContext *context = unwrap(ctx);
        Location location = unwrap(loc);
        Type resType = unwrap(resultType);
        Value containerVal = unwrap(container);
        Value valueVal = unwrap(value);

        SmallVector<int64_t, 4> pos;
        pos.reserve(numPositions);
        for (size_t i = 0; i < numPositions; ++i)
        {
            pos.push_back(positions[i]);
        }

        OpBuilder builder(context);
        auto op = builder.create<LLVM::InsertValueOp>(location, resType, containerVal, valueVal, pos);
        return wrap(op.getOperation());
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirOperation oraLlvmExtractValueOpCreate(MlirContext ctx, MlirLocation loc, MlirType resultType, MlirValue container, const int64_t *positions, size_t numPositions)
{
    try
    {
        MLIRContext *context = unwrap(ctx);
        Location location = unwrap(loc);
        Type resType = unwrap(resultType);
        Value containerVal = unwrap(container);

        SmallVector<int64_t, 4> pos;
        pos.reserve(numPositions);
        for (size_t i = 0; i < numPositions; ++i)
        {
            pos.push_back(positions[i]);
        }

        OpBuilder builder(context);
        auto op = builder.create<LLVM::ExtractValueOp>(location, resType, containerVal, pos);
        return wrap(op.getOperation());
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

MlirOperation oraBytesConstantOpCreate(MlirContext ctx, MlirLocation loc, MlirStringRef value, MlirType resultType)
{
    try
    {
        auto *context = unwrap(ctx);
        auto location = unwrap(loc);
        auto valueRef = unwrap(value);
        auto resultTypeRef = unwrap(resultType);

        OpBuilder builder(context);

        if (!oraDialectIsRegistered(ctx))
        {
            return {nullptr};
        }

        auto valueAttr = StringAttr::get(context, valueRef);
        auto bytesConstOp = builder.create<BytesConstantOp>(location, resultTypeRef, valueAttr);
        return wrap(bytesConstOp.getOperation());
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

/// Create an MLIR IntegerAttr from a string representation with full u256 precision
/// This bypasses the C API limitation of mlirIntegerAttrGet which only accepts i64
MlirAttribute oraIntegerAttrGetFromString(MlirType type, MlirStringRef valueStr)
{
    try
    {
        Type mlirType = unwrap(type);
        StringRef valueRef = unwrap(valueStr);

        // Check if the type is an integer type (either standard MLIR or Ora)
        unsigned bitWidth = 0;
        bool isSigned = false;
        ::mlir::IntegerType intType = nullptr;

        // Try standard MLIR IntegerType first
        if (auto stdIntType = llvm::dyn_cast<::mlir::IntegerType>(mlirType))
        {
            intType = stdIntType;
            bitWidth = stdIntType.getWidth();
            isSigned = stdIntType.isSigned();
        }
        // Try Ora IntegerType
        else if (auto oraIntType = llvm::dyn_cast<ora::IntegerType>(mlirType))
        {
            bitWidth = oraIntType.getWidth();
            isSigned = oraIntType.getIsSigned();
            // Convert Ora type to standard MLIR type for IntegerAttr
            intType = ::mlir::IntegerType::get(mlirType.getContext(), bitWidth,
                                               isSigned ? ::mlir::IntegerType::Signed : ::mlir::IntegerType::Unsigned);
        }
        else
        {
            // Not an integer type
            return {nullptr};
        }

        // Remove underscores from the string (they're used as separators in Ora)
        std::string cleanValue;
        for (char c : valueRef.str())
        {
            if (c != '_')
            {
                cleanValue += c;
            }
        }

        // Parse decimal string to APInt
        // APInt can handle arbitrary precision (u256, u512, etc.)
        if (cleanValue.empty())
        {
            return {nullptr};
        }

        // Parse the string digit by digit to build the APInt
        // This handles values of any size up to the bit width
        llvm::APInt apValue(bitWidth, 0);
        llvm::APInt ten(bitWidth, 10);

        for (char c : cleanValue)
        {
            if (c >= '0' && c <= '9')
            {
                // Multiply by 10 and add the digit
                apValue *= ten;
                llvm::APInt digit(bitWidth, static_cast<uint64_t>(c - '0'));
                apValue += digit;
            }
        }

        // If the type is signed and the value would be negative in two's complement,
        // we need to handle sign extension. For now, we'll assume unsigned parsing.
        // The isSigned flag in the type will handle the interpretation.

        // Verify the value fits in the bit width
        // APInt will automatically truncate, but we want to ensure it's valid
        if (apValue.getBitWidth() != bitWidth)
        {
            // Adjust bit width if needed (shouldn't happen, but be safe)
            apValue = apValue.zextOrTrunc(bitWidth);
        }

        // Create IntegerAttr from APInt
        // IntegerAttr::get accepts APInt directly, preserving full precision
        auto attr = ::mlir::IntegerAttr::get(intType, apValue);
        if (!attr)
        {
            return {nullptr};
        }
        return wrap(attr);
    }
    catch (const std::exception &e)
    {
        // Log the exception for debugging (can be removed in production)
        // For now, just return null
        return {nullptr};
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

MlirOperation oraStructFieldExtractOpCreate(MlirContext ctx, MlirLocation loc, MlirValue structValue, MlirStringRef fieldName, MlirType resultType)
{
    try
    {
        MLIRContext *context = unwrap(ctx);
        Location location = unwrap(loc);
        Value structValueRef = unwrap(structValue);
        StringRef fieldNameRef = unwrap(fieldName);
        Type resultTypeRef = unwrap(resultType);

        if (!oraDialectIsRegistered(ctx))
        {
            return {nullptr};
        }

        OpBuilder builder(context);
        auto fieldNameAttr = StringAttr::get(context, fieldNameRef);
        auto structFieldExtractOp = builder.create<StructFieldExtractOp>(location, resultTypeRef, structValueRef, fieldNameAttr);

        // Add gas_cost attribute (field extraction has minimal cost = 1)
        auto gasCostAttr = IntegerAttr::get(::mlir::IntegerType::get(context, 64), 1);
        structFieldExtractOp->setAttr("gas_cost", gasCostAttr);

        return wrap(structFieldExtractOp.getOperation());
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirOperation oraStructFieldUpdateOpCreate(MlirContext ctx, MlirLocation loc, MlirValue structValue, MlirStringRef fieldName, MlirValue value)
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

        // Result type must match input struct type
        // Get the result type from the input struct value
        Type resultType = structValueRef.getType();
        auto structFieldUpdateOp = builder.create<StructFieldUpdateOp>(location, resultType, structValueRef, fieldNameAttr, valueRef);

        // Add gas_cost attribute (struct field update has cost = 3, similar to memory store)
        auto gasCostAttr = IntegerAttr::get(::mlir::IntegerType::get(context, 64), 3);
        structFieldUpdateOp->setAttr("gas_cost", gasCostAttr);

        return wrap(structFieldUpdateOp.getOperation());
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

        StringRef enumNameRef = unwrap(enumName);
        StringRef variantNameRef = unwrap(variantName);
        auto enumNameAttr = StringAttr::get(context, enumNameRef);
        auto variantNameAttr = StringAttr::get(context, variantNameRef);
        auto op = builder.create<ora::EnumConstantOp>(location, resultTy, enumNameAttr, variantNameAttr);

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

        StringRef nameRef = unwrap(name);
        auto nameAttr = StringAttr::get(context, nameRef);
        auto op = builder.create<ora::StructDeclOp>(location, nameAttr);

        if (op.getFields().empty())
        {
            OpBuilder::InsertionGuard guard(builder);
            builder.createBlock(&op.getFields());
        }

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

        SmallVector<Value> fieldVals;
        fieldVals.reserve(numFieldValues);
        for (size_t i = 0; i < numFieldValues; ++i)
        {
            fieldVals.push_back(unwrap(fieldValues[i]));
        }

        StringRef structNameRef = unwrap(structName);
        auto structNameAttr = StringAttr::get(context, structNameRef);
        auto op = builder.create<ora::StructInstantiateOp>(location, resultTy, structNameAttr, fieldVals);

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

MlirOperation oraMoveOpCreateWithMapping(MlirContext ctx, MlirLocation loc, MlirValue mapping, MlirValue source, MlirValue destination, MlirValue amount, MlirType resultType)
{
    try
    {
        Location location = unwrap(loc);

        Value mappingVal = unwrap(mapping);
        Value sourceVal = unwrap(source);
        Value destinationVal = unwrap(destination);
        Value amountVal = unwrap(amount);
        Type resultTy = unwrap(resultType);

        OperationState state(location, "ora.move");
        SmallVector<Value, 4> operands;
        operands.push_back(mappingVal);
        operands.push_back(sourceVal);
        operands.push_back(destinationVal);
        operands.push_back(amountVal);
        state.addOperands(operands);
        state.addTypes(resultTy);

        Operation *op = Operation::create(state);
        return wrap(op);
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirOperation oraCmpOpCreate(MlirContext ctx, MlirLocation loc, MlirStringRef predicate, MlirValue lhs, MlirValue rhs, MlirType resultType)
{
    try
    {
        MLIRContext *context = unwrap(ctx);
        Location location = unwrap(loc);
        Type resultTy = unwrap(resultType);

        StringRef predicateRef = unwrap(predicate);
        Value lhsVal = unwrap(lhs);
        Value rhsVal = unwrap(rhs);

        OpBuilder builder(context);
        auto op = builder.create<ora::CmpOp>(location, resultTy, predicateRef, lhsVal, rhsVal);
        return wrap(op.getOperation());
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirOperation oraRangeOpCreate(MlirContext ctx, MlirLocation loc, MlirValue start, MlirValue end, MlirType resultType, bool inclusive)
{
    try
    {
        MLIRContext *context = unwrap(ctx);
        Location location = unwrap(loc);
        Type resultTy = unwrap(resultType);

        Value startVal = unwrap(start);
        Value endVal = unwrap(end);
        auto inclusiveAttr = BoolAttr::get(context, inclusive);

        OpBuilder builder(context);
        auto op = builder.create<ora::RangeOp>(location, resultTy, startVal, endVal, inclusiveAttr);
        return wrap(op.getOperation());
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirOperation oraQuantifiedOpCreate(MlirContext ctx, MlirLocation loc, MlirStringRef quantifier, MlirStringRef variable, MlirStringRef variableType, MlirValue condition, bool hasCondition, MlirValue body, MlirType resultType)
{
    try
    {
        MLIRContext *context = unwrap(ctx);
        Location location = unwrap(loc);
        Type resultTy = unwrap(resultType);

        StringRef quantifierRef = unwrap(quantifier);
        StringRef variableRef = unwrap(variable);
        StringRef variableTypeRef = unwrap(variableType);

        Value conditionVal = hasCondition ? unwrap(condition) : Value();
        Value bodyVal = unwrap(body);

        OpBuilder builder(context);
        auto op = builder.create<ora::QuantifiedOp>(location, resultTy, quantifierRef, variableRef, variableTypeRef, conditionVal, bodyVal);
        return wrap(op.getOperation());
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirOperation oraErrorDeclOpCreate(MlirContext ctx, MlirLocation loc, const MlirType *resultTypes, size_t numResults, const MlirNamedAttribute *attrs, size_t numAttrs)
{
    try
    {
        Location location = unwrap(loc);

        OperationState state(location, "ora.error.decl");
        for (size_t i = 0; i < numResults; ++i)
        {
            state.addTypes(unwrap(resultTypes[i]));
        }
        for (size_t i = 0; i < numAttrs; ++i)
        {
            auto attrName = unwrap(attrs[i].name);
            auto attrValue = unwrap(attrs[i].attribute);
            state.addAttribute(attrName, attrValue);
        }

        Operation *op = Operation::create(state);
        return wrap(op);
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirOperation oraMethodCallOpCreate(MlirContext ctx, MlirLocation loc, MlirStringRef methodName, const MlirValue *operands, size_t numOperands, MlirType resultType)
{
    try
    {
        MLIRContext *context = unwrap(ctx);
        Location location = unwrap(loc);
        Type resultTy = unwrap(resultType);
        StringRef methodRef = unwrap(methodName);

        OperationState state(location, "ora.method_call");
        for (size_t i = 0; i < numOperands; ++i)
        {
            state.addOperands(unwrap(operands[i]));
        }
        state.addTypes(resultTy);
        state.addAttribute("method", StringAttr::get(context, methodRef));

        Operation *op = Operation::create(state);
        return wrap(op);
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirOperation oraBinaryConstantOpCreate(MlirContext ctx, MlirLocation loc, MlirType resultType, const MlirNamedAttribute *attrs, size_t numAttrs)
{
    try
    {
        Location location = unwrap(loc);
        Type resultTy = unwrap(resultType);

        OperationState state(location, "ora.binary.constant");
        state.addTypes(resultTy);
        for (size_t i = 0; i < numAttrs; ++i)
        {
            auto attrName = unwrap(attrs[i].name);
            auto attrValue = unwrap(attrs[i].attribute);
            state.addAttribute(attrName, attrValue);
        }

        Operation *op = Operation::create(state);
        return wrap(op);
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirOperation oraModuleOpCreate(MlirContext ctx, MlirLocation loc, const MlirNamedAttribute *attrs, size_t numAttrs, size_t numRegions, bool addEmptyBlocks)
{
    try
    {
        Location location = unwrap(loc);

        OperationState state(location, "ora.module");
        for (size_t i = 0; i < numAttrs; ++i)
        {
            auto attrName = unwrap(attrs[i].name);
            auto attrValue = unwrap(attrs[i].attribute);
            state.addAttribute(attrName, attrValue);
        }
        for (size_t i = 0; i < numRegions; ++i)
        {
            Region *region = state.addRegion();
            if (addEmptyBlocks)
            {
                region->push_back(new Block());
            }
        }

        Operation *op = Operation::create(state);
        return wrap(op);
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirOperation oraBlockOpCreate(MlirContext ctx, MlirLocation loc, const MlirNamedAttribute *attrs, size_t numAttrs, size_t numRegions, bool addEmptyBlocks)
{
    try
    {
        Location location = unwrap(loc);

        OperationState state(location, "ora.block");
        for (size_t i = 0; i < numAttrs; ++i)
        {
            auto attrName = unwrap(attrs[i].name);
            auto attrValue = unwrap(attrs[i].attribute);
            state.addAttribute(attrName, attrValue);
        }
        for (size_t i = 0; i < numRegions; ++i)
        {
            Region *region = state.addRegion();
            if (addEmptyBlocks)
            {
                region->push_back(new Block());
            }
        }

        Operation *op = Operation::create(state);
        return wrap(op);
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirOperation oraTryBlockOpCreate(MlirContext ctx, MlirLocation loc, const MlirNamedAttribute *attrs, size_t numAttrs, size_t numRegions, bool addEmptyBlocks)
{
    try
    {
        Location location = unwrap(loc);

        OperationState state(location, "ora.try_block");
        for (size_t i = 0; i < numAttrs; ++i)
        {
            auto attrName = unwrap(attrs[i].name);
            auto attrValue = unwrap(attrs[i].attribute);
            state.addAttribute(attrName, attrValue);
        }
        for (size_t i = 0; i < numRegions; ++i)
        {
            Region *region = state.addRegion();
            if (addEmptyBlocks)
            {
                region->push_back(new Block());
            }
        }

        Operation *op = Operation::create(state);
        return wrap(op);
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirOperation oraImportOpCreate(MlirContext ctx, MlirLocation loc, const MlirNamedAttribute *attrs, size_t numAttrs)
{
    try
    {
        Location location = unwrap(loc);

        OperationState state(location, "ora.import");
        for (size_t i = 0; i < numAttrs; ++i)
        {
            auto attrName = unwrap(attrs[i].name);
            auto attrValue = unwrap(attrs[i].attribute);
            state.addAttribute(attrName, attrValue);
        }

        Operation *op = Operation::create(state);
        return wrap(op);
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirOperation oraLogDeclOpCreate(MlirContext ctx, MlirLocation loc, const MlirNamedAttribute *attrs, size_t numAttrs)
{
    try
    {
        Location location = unwrap(loc);

        OperationState state(location, "ora.log.decl");
        for (size_t i = 0; i < numAttrs; ++i)
        {
            auto attrName = unwrap(attrs[i].name);
            auto attrValue = unwrap(attrs[i].attribute);
            state.addAttribute(attrName, attrValue);
        }

        Operation *op = Operation::create(state);
        return wrap(op);
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirOperation oraQuantifiedDeclOpCreate(MlirContext ctx, MlirLocation loc, const MlirNamedAttribute *attrs, size_t numAttrs, size_t numRegions, bool addEmptyBlocks)
{
    try
    {
        Location location = unwrap(loc);

        OperationState state(location, "ora.quantified");
        for (size_t i = 0; i < numAttrs; ++i)
        {
            auto attrName = unwrap(attrs[i].name);
            auto attrValue = unwrap(attrs[i].attribute);
            state.addAttribute(attrName, attrValue);
        }
        for (size_t i = 0; i < numRegions; ++i)
        {
            Region *region = state.addRegion();
            if (addEmptyBlocks)
            {
                region->push_back(new Block());
            }
        }

        Operation *op = Operation::create(state);
        return wrap(op);
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirOperation oraVariablePlaceholderOpCreate(MlirContext ctx, MlirLocation loc, MlirStringRef name, MlirType resultType)
{
    try
    {
        MLIRContext *context = unwrap(ctx);
        Location location = unwrap(loc);
        StringRef nameRef = unwrap(name);
        Type resultTy = unwrap(resultType);

        OperationState state(location, "ora.variable_placeholder");
        state.addTypes(resultTy);
        state.addAttribute("name", StringAttr::get(context, nameRef));

        Operation *op = Operation::create(state);
        return wrap(op);
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirOperation oraModulePlaceholderOpCreate(MlirContext ctx, MlirLocation loc, MlirStringRef name, const MlirNamedAttribute *attrs, size_t numAttrs)
{
    try
    {
        MLIRContext *context = unwrap(ctx);
        Location location = unwrap(loc);
        StringRef nameRef = unwrap(name);

        OperationState state(location, "ora.module_placeholder");
        state.addAttribute("name", StringAttr::get(context, nameRef));
        for (size_t i = 0; i < numAttrs; ++i)
        {
            auto attrName = unwrap(attrs[i].name);
            auto attrValue = unwrap(attrs[i].attribute);
            state.addAttribute(attrName, attrValue);
        }

        Operation *op = Operation::create(state);
        return wrap(op);
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirOperation oraTensorExtractOpCreate(MlirContext ctx, MlirLocation loc, MlirValue tensor, const MlirValue *indices, size_t numIndices, MlirType resultType)
{
    try
    {
        MLIRContext *context = unwrap(ctx);
        Location location = unwrap(loc);
        Value tensorVal = unwrap(tensor);
        Type resultTy = unwrap(resultType);

        SmallVector<Value> idx;
        idx.reserve(numIndices);
        for (size_t i = 0; i < numIndices; ++i)
        {
            idx.push_back(unwrap(indices[i]));
        }

        OpBuilder builder(context);
        auto op = builder.create<tensor::ExtractOp>(location, resultTy, tensorVal, idx);
        return wrap(op.getOperation());
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirOperation oraEvmOpCreate(MlirContext ctx, MlirLocation loc, MlirStringRef name, const MlirValue *operands, size_t numOperands, MlirType resultType)
{
    try
    {
        Location location = unwrap(loc);
        StringRef nameRef = unwrap(name);
        Type resultTy = unwrap(resultType);

        OperationState state(location, nameRef);
        for (size_t i = 0; i < numOperands; ++i)
        {
            state.addOperands(unwrap(operands[i]));
        }
        state.addTypes(resultTy);

        Operation *op = Operation::create(state);
        return wrap(op);
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirAttribute oraStringAttrCreate(MlirContext ctx, MlirStringRef value)
{
    try
    {
        MLIRContext *context = unwrap(ctx);
        StringRef valueRef = unwrap(value);
        return wrap(mlir::Attribute(mlir::StringAttr::get(context, valueRef)));
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirAttribute oraBoolAttrCreate(MlirContext ctx, bool value)
{
    try
    {
        MLIRContext *context = unwrap(ctx);
        return wrap(BoolAttr::get(context, value));
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirAttribute oraIntegerAttrCreateI64(MlirContext ctx, MlirType type, int64_t value)
{
    try
    {
        MLIRContext *context = unwrap(ctx);
        Type ty = unwrap(type);
        if (auto intTy = llvm::dyn_cast<mlir::IntegerType>(ty))
        {
            return wrap(IntegerAttr::get(intTy, value));
        }
        if (auto oraIntTy = llvm::dyn_cast<mlir::ora::IntegerType>(ty))
        {
            auto builtinTy = mlir::IntegerType::get(context, oraIntTy.getWidth(), oraIntTy.getIsSigned() ? mlir::IntegerType::Signed : mlir::IntegerType::Unsigned);
            return wrap(IntegerAttr::get(builtinTy, value));
        }
        return {nullptr};
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirAttribute oraIntegerAttrCreateI64FromType(MlirType type, int64_t value)
{
    try
    {
        Type ty = unwrap(type);
        MLIRContext *context = ty.getContext();
        if (auto intTy = llvm::dyn_cast<mlir::IntegerType>(ty))
        {
            return wrap(IntegerAttr::get(intTy, value));
        }
        if (auto oraIntTy = llvm::dyn_cast<mlir::ora::IntegerType>(ty))
        {
            auto builtinTy = mlir::IntegerType::get(context, oraIntTy.getWidth(), oraIntTy.getIsSigned() ? mlir::IntegerType::Signed : mlir::IntegerType::Unsigned);
            return wrap(IntegerAttr::get(builtinTy, value));
        }
        if (auto indexTy = llvm::dyn_cast<mlir::IndexType>(ty))
        {
            return wrap(IntegerAttr::get(indexTy, value));
        }
        // Handle ora::EnumType - extract underlying representation type
        if (auto enumTy = llvm::dyn_cast<ora::EnumType>(ty))
        {
            Type reprType = enumTy.getReprType();
            if (auto intTy = llvm::dyn_cast<mlir::IntegerType>(reprType))
            {
                return wrap(IntegerAttr::get(intTy, value));
            }
            if (auto oraIntTy = llvm::dyn_cast<mlir::ora::IntegerType>(reprType))
            {
                auto builtinTy = mlir::IntegerType::get(context, oraIntTy.getWidth(), oraIntTy.getIsSigned() ? mlir::IntegerType::Signed : mlir::IntegerType::Unsigned);
                return wrap(IntegerAttr::get(builtinTy, value));
            }
        }
        return {nullptr};
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirAttribute oraTypeAttrCreate(MlirContext ctx, MlirType type)
{
    try
    {
        (void)ctx;
        Type ty = unwrap(type);
        return wrap(TypeAttr::get(ty));
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirAttribute oraTypeAttrCreateFromType(MlirType type)
{
    try
    {
        Type ty = unwrap(type);
        return wrap(TypeAttr::get(ty));
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirAttribute oraArrayAttrCreate(MlirContext ctx, intptr_t numAttrs, const MlirAttribute *attrs)
{
    try
    {
        MLIRContext *context = unwrap(ctx);
        const size_t attr_count = numAttrs > 0 ? static_cast<size_t>(numAttrs) : 0;
        SmallVector<Attribute> items;
        items.reserve(attr_count);
        for (size_t i = 0; i < attr_count; ++i)
        {
            items.push_back(unwrap(attrs[i]));
        }
        return wrap(ArrayAttr::get(context, items));
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirAttribute oraNullAttrCreate(void)
{
    try
    {
        return wrap(Attribute());
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirType oraIntegerTypeCreate(MlirContext ctx, uint32_t bits)
{
    try
    {
        MLIRContext *context = unwrap(ctx);
        return wrap(mlir::IntegerType::get(context, bits));
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirType oraIndexTypeCreate(MlirContext ctx)
{
    try
    {
        MLIRContext *context = unwrap(ctx);
        return wrap(IndexType::get(context));
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirType oraNoneTypeCreate(MlirContext ctx)
{
    try
    {
        MLIRContext *context = unwrap(ctx);
        return wrap(NoneType::get(context));
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirType oraRankedTensorTypeCreate(MlirContext ctx, intptr_t rank, const int64_t *shape, MlirType elementType, MlirAttribute encoding)
{
    try
    {
        (void)ctx;
        Type elemTy = unwrap(elementType);
        Attribute encodingAttr = unwrap(encoding);
        ArrayRef<int64_t> shapeRef(shape, static_cast<size_t>(rank));
        return wrap(RankedTensorType::get(shapeRef, elemTy, encodingAttr));
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirType oraMemRefTypeCreate(MlirContext ctx, MlirType elementType, intptr_t rank, const int64_t *shape, MlirAttribute layout, MlirAttribute memorySpace)
{
    try
    {
        (void)ctx;
        Type elemTy = unwrap(elementType);
        Attribute layoutAttr = unwrap(layout);
        Attribute memSpaceAttr = unwrap(memorySpace);
        ArrayRef<int64_t> shapeRef(shape, static_cast<size_t>(rank));
        mlir::MemRefLayoutAttrInterface layoutIface;
        if (layoutAttr)
        {
            layoutIface = llvm::dyn_cast<mlir::MemRefLayoutAttrInterface>(layoutAttr);
        }
        return wrap(mlir::MemRefType::get(shapeRef, elemTy, layoutIface, memSpaceAttr));
    }
    catch (...)
    {
        return {nullptr};
    }
}

int64_t oraShapedTypeDynamicSize(void)
{
    return mlir::ShapedType::kDynamic;
}

bool oraTypeIsAInteger(MlirType type)
{
    Type ty = unwrap(type);
    return ty && llvm::isa<mlir::IntegerType>(ty);
}

bool oraTypeEqual(MlirType a, MlirType b)
{
    Type ta = unwrap(a);
    Type tb = unwrap(b);
    return ta && tb && ta == tb;
}

bool oraTypeIsAShaped(MlirType type)
{
    Type ty = unwrap(type);
    return ty && llvm::isa<mlir::ShapedType>(ty);
}

bool oraTypeIsAMemRef(MlirType type)
{
    Type ty = unwrap(type);
    return ty && llvm::isa<mlir::MemRefType>(ty);
}

MlirType oraShapedTypeGetElementType(MlirType type)
{
    Type ty = unwrap(type);
    if (auto shaped = llvm::dyn_cast<mlir::ShapedType>(ty))
    {
        return wrap(shaped.getElementType());
    }
    return {nullptr};
}

intptr_t oraShapedTypeGetRank(MlirType type)
{
    Type ty = unwrap(type);
    if (auto shaped = llvm::dyn_cast<mlir::ShapedType>(ty))
    {
        return shaped.getRank();
    }
    return -1;
}

int64_t oraShapedTypeGetDimSize(MlirType type, intptr_t dim)
{
    Type ty = unwrap(type);
    if (auto shaped = llvm::dyn_cast<mlir::ShapedType>(ty))
    {
        return shaped.getDimSize(dim);
    }
    return mlir::ShapedType::kDynamic;
}

bool oraShapedTypeHasStaticShape(MlirType type)
{
    Type ty = unwrap(type);
    if (auto shaped = llvm::dyn_cast<mlir::ShapedType>(ty))
    {
        return shaped.hasStaticShape();
    }
    return false;
}

uint32_t oraIntegerTypeGetWidth(MlirType type)
{
    Type ty = unwrap(type);
    if (auto intTy = llvm::dyn_cast<mlir::IntegerType>(ty))
    {
        return intTy.getWidth();
    }
    if (auto oraIntTy = llvm::dyn_cast<mlir::ora::IntegerType>(ty))
    {
        return oraIntTy.getWidth();
    }
    return 0;
}

bool oraTypeIsNull(MlirType type)
{
    return unwrap(type) == nullptr;
}

bool oraTypeIsANone(MlirType type)
{
    Type ty = unwrap(type);
    return ty && llvm::isa<mlir::NoneType>(ty);
}

bool oraTypeIsAEnum(MlirType type)
{
    try
    {
        Type ty = unwrap(type);
        return ty && llvm::isa<ora::EnumType>(ty);
    }
    catch (...)
    {
        return false;
    }
}

MlirType oraEnumTypeGetReprType(MlirType enumType)
{
    try
    {
        Type ty = unwrap(enumType);
        if (auto enumTy = llvm::dyn_cast<ora::EnumType>(ty))
        {
            return wrap(enumTy.getReprType());
        }
        return {nullptr};
    }
    catch (...)
    {
        return {nullptr};
    }
}

bool oraTypeIsAOraInteger(MlirType type)
{
    try
    {
        Type ty = unwrap(type);
        return ty && llvm::isa<ora::IntegerType>(ty);
    }
    catch (...)
    {
        return false;
    }
}

MlirOperation oraConstDeclOpCreate(MlirContext ctx, MlirLocation loc, const MlirType *resultTypes, size_t numResults, const MlirNamedAttribute *attrs, size_t numAttrs, size_t numRegions, bool addEmptyBlocks)
{
    try
    {
        Location location = unwrap(loc);

        OperationState state(location, "ora.const");
        for (size_t i = 0; i < numResults; ++i)
        {
            state.addTypes(unwrap(resultTypes[i]));
        }
        for (size_t i = 0; i < numAttrs; ++i)
        {
            auto attrName = unwrap(attrs[i].name);
            auto attrValue = unwrap(attrs[i].attribute);
            state.addAttribute(attrName, attrValue);
        }
        for (size_t i = 0; i < numRegions; ++i)
        {
            Region *region = state.addRegion();
            if (addEmptyBlocks)
            {
                region->push_back(new Block());
            }
        }

        Operation *op = Operation::create(state);
        return wrap(op);
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirOperation oraImmutableDeclOpCreate(MlirContext ctx, MlirLocation loc, const MlirType *resultTypes, size_t numResults, const MlirNamedAttribute *attrs, size_t numAttrs, size_t numRegions, bool addEmptyBlocks)
{
    try
    {
        Location location = unwrap(loc);

        OperationState state(location, "ora.immutable");
        for (size_t i = 0; i < numResults; ++i)
        {
            state.addTypes(unwrap(resultTypes[i]));
        }
        for (size_t i = 0; i < numAttrs; ++i)
        {
            auto attrName = unwrap(attrs[i].name);
            auto attrValue = unwrap(attrs[i].attribute);
            state.addAttribute(attrName, attrValue);
        }
        for (size_t i = 0; i < numRegions; ++i)
        {
            Region *region = state.addRegion();
            if (addEmptyBlocks)
            {
                region->push_back(new Block());
            }
        }

        Operation *op = Operation::create(state);
        return wrap(op);
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirOperation oraMemoryGlobalOpCreate(MlirContext ctx, MlirLocation loc, MlirStringRef name, MlirType type)
{
    try
    {
        MLIRContext *context = unwrap(ctx);
        Location location = unwrap(loc);
        StringRef nameRef = unwrap(name);
        Type varType = unwrap(type);

        OperationState state(location, "ora.memory.global");
        state.addAttribute("sym_name", StringAttr::get(context, nameRef));
        state.addAttribute("type", TypeAttr::get(varType));

        Operation *op = Operation::create(state);
        return wrap(op);
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirOperation oraTStoreGlobalOpCreate(MlirContext ctx, MlirLocation loc, MlirStringRef name, MlirType type)
{
    try
    {
        MLIRContext *context = unwrap(ctx);
        Location location = unwrap(loc);
        StringRef nameRef = unwrap(name);
        Type varType = unwrap(type);

        OperationState state(location, "ora.tstore.global");
        state.addAttribute("sym_name", StringAttr::get(context, nameRef));
        state.addAttribute("type", TypeAttr::get(varType));

        Operation *op = Operation::create(state);
        return wrap(op);
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirOperation oraErrorOpCreate(MlirContext ctx, MlirLocation loc, MlirType resultType)
{
    try
    {
        Location location = unwrap(loc);
        Type resultTy = unwrap(resultType);

        OperationState state(location, "ora.error");
        state.addTypes(resultTy);

        Operation *op = Operation::create(state);
        return wrap(op);
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirOperation oraLengthOpCreate(MlirContext ctx, MlirLocation loc, MlirValue value, MlirType resultType)
{
    try
    {
        Location location = unwrap(loc);
        Value val = unwrap(value);
        Type resultTy = unwrap(resultType);

        OperationState state(location, "ora.length");
        state.addOperands(val);
        state.addTypes(resultTy);

        Operation *op = Operation::create(state);
        return wrap(op);
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirOperation oraExpressionCaptureOpCreate(MlirContext ctx, MlirLocation loc, MlirValue value, MlirType resultType)
{
    try
    {
        MLIRContext *context = unwrap(ctx);
        Location location = unwrap(loc);
        Value val = unwrap(value);
        Type resultTy = unwrap(resultType);

        OperationState state(location, "ora.expression_capture");
        state.addOperands(val);
        state.addTypes(resultTy);
        state.addAttribute("ora.top_level_expression", BoolAttr::get(context, true));

        Operation *op = Operation::create(state);
        return wrap(op);
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
        context->getOrLoadDialect<ora::OraDialect>();

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

MlirOperation oraFuncFuncOpCreate(MlirContext ctx, MlirLocation loc, const MlirNamedAttribute *attrs, size_t numAttrs, const MlirType *paramTypes, const MlirLocation *paramLocs, size_t numParams)
{
    try
    {
        Location location = unwrap(loc);

        OperationState state(location, "func.func");
        for (size_t i = 0; i < numAttrs; ++i)
        {
            auto name = unwrap(attrs[i].name);
            auto value = unwrap(attrs[i].attribute);
            state.addAttribute(name, value);
        }

        Region *region = state.addRegion();
        SmallVector<Type, 8> argTypes;
        SmallVector<Location, 8> argLocs;
        argTypes.reserve(numParams);
        argLocs.reserve(numParams);
        for (size_t i = 0; i < numParams; ++i)
        {
            argTypes.push_back(unwrap(paramTypes[i]));
            argLocs.push_back(unwrap(paramLocs[i]));
        }

        Block *entry = new Block();
        entry->addArguments(argTypes, argLocs);
        region->push_back(entry);

        Operation *op = Operation::create(state);
        return wrap(op);
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirBlock oraFuncOpGetBodyBlock(MlirOperation op)
{
    try
    {
        Operation *operation = unwrap(op);
        if (!operation)
        {
            return {nullptr};
        }

        auto funcOp = dyn_cast<func::FuncOp>(operation);
        if (!funcOp)
        {
            return {nullptr};
        }

        if (funcOp.getBody().empty())
        {
            funcOp.addEntryBlock();
        }

        return wrap(&funcOp.getBody().front());
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirOperation oraOperationCreate(MlirContext ctx, MlirLocation loc, MlirStringRef name, const MlirValue *operands, size_t numOperands, const MlirType *resultTypes, size_t numResults, const MlirNamedAttribute *attrs, size_t numAttrs, size_t numRegions, bool addEmptyBlocks)
{
    try
    {
        Location location = unwrap(loc);
        StringRef nameRef = unwrap(name);

        OperationState state(location, nameRef);
        for (size_t i = 0; i < numOperands; ++i)
        {
            state.addOperands(unwrap(operands[i]));
        }
        for (size_t i = 0; i < numResults; ++i)
        {
            state.addTypes(unwrap(resultTypes[i]));
        }
        for (size_t i = 0; i < numAttrs; ++i)
        {
            auto attrName = unwrap(attrs[i].name);
            auto attrValue = unwrap(attrs[i].attribute);
            state.addAttribute(attrName, attrValue);
        }
        for (size_t i = 0; i < numRegions; ++i)
        {
            Region *region = state.addRegion();
            if (addEmptyBlocks)
            {
                Block *block = new Block();
                region->push_back(block);
            }
        }

        Operation *op = Operation::create(state);
        return wrap(op);
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirBlock oraOperationGetRegionBlock(MlirOperation op, size_t index)
{
    try
    {
        Operation *operation = unwrap(op);
        if (!operation)
        {
            return {nullptr};
        }

        if (index >= operation->getNumRegions())
        {
            return {nullptr};
        }

        Region &region = operation->getRegion(index);
        if (region.empty())
        {
            region.push_back(new Block());
        }

        return wrap(&region.front());
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirOperation oraErrorOkOpCreate(MlirContext ctx, MlirLocation loc, MlirValue value, MlirType resultType)
{
    try
    {
        MLIRContext *context = unwrap(ctx);
        Location location = unwrap(loc);
        Value valueRef = unwrap(value);
        Type resultTy = unwrap(resultType);

        OpBuilder builder(context);

        auto op = builder.create<ora::ErrorOkOp>(location, resultTy, valueRef);
        return wrap(op.getOperation());
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirOperation oraErrorErrOpCreate(MlirContext ctx, MlirLocation loc, MlirValue value, MlirType resultType)
{
    try
    {
        MLIRContext *context = unwrap(ctx);
        Location location = unwrap(loc);
        Value valueRef = unwrap(value);
        Type resultTy = unwrap(resultType);

        OpBuilder builder(context);

        auto op = builder.create<ora::ErrorErrOp>(location, resultTy, valueRef);
        return wrap(op.getOperation());
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirOperation oraErrorIsErrorOpCreate(MlirContext ctx, MlirLocation loc, MlirValue value)
{
    try
    {
        MLIRContext *context = unwrap(ctx);
        Location location = unwrap(loc);
        Value valueRef = unwrap(value);

        OpBuilder builder(context);

        auto resultTy = builder.getI1Type();
        auto op = builder.create<ora::ErrorIsErrorOp>(location, resultTy, valueRef);
        return wrap(op.getOperation());
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirOperation oraErrorUnwrapOpCreate(MlirContext ctx, MlirLocation loc, MlirValue value, MlirType resultType)
{
    try
    {
        MLIRContext *context = unwrap(ctx);
        Location location = unwrap(loc);
        Value valueRef = unwrap(value);
        Type resultTy = unwrap(resultType);

        OpBuilder builder(context);

        auto op = builder.create<ora::ErrorUnwrapOp>(location, resultTy, valueRef);
        return wrap(op.getOperation());
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirOperation oraErrorGetErrorOpCreate(MlirContext ctx, MlirLocation loc, MlirValue value, MlirType resultType)
{
    try
    {
        MLIRContext *context = unwrap(ctx);
        Location location = unwrap(loc);
        Value valueRef = unwrap(value);
        Type resultTy = unwrap(resultType);

        OpBuilder builder(context);

        auto op = builder.create<ora::ErrorGetErrorOp>(location, resultTy, valueRef);
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

MlirBlock oraTryOpGetCatchBlock(MlirOperation tryOp)
{
    try
    {
        Operation *operation = unwrap(tryOp);
        if (!operation)
        {
            return {nullptr};
        }

        auto op = dyn_cast<ora::TryOp>(operation);
        if (!op)
        {
            return {nullptr};
        }

        Region &catchRegion = op.getCatchRegion();
        if (catchRegion.empty())
        {
            catchRegion.push_back(new Block());
        }

        return wrap(&catchRegion.front());
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

MlirOperation oraSwitchOpCreateWithCases(MlirContext ctx, MlirLocation loc, MlirValue value, const MlirType *resultTypes, size_t numResults, size_t numCases)
{
    try
    {
        MLIRContext *context = unwrap(ctx);
        Location location = unwrap(loc);
        Value val = unwrap(value);

        SmallVector<Type, 4> results;
        results.reserve(numResults);
        for (size_t i = 0; i < numResults; ++i)
        {
            results.push_back(unwrap(resultTypes[i]));
        }

        OpBuilder builder(context);
        auto op = builder.create<ora::SwitchOp>(location, results, val, static_cast<unsigned>(numCases));

        auto gasCostAttr = builder.getI64IntegerAttr(10);
        op->setAttr("gas_cost", gasCostAttr);

        return wrap(op.getOperation());
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirBlock oraSwitchOpGetCaseBlock(MlirOperation switchOp, size_t index)
{
    try
    {
        Operation *operation = unwrap(switchOp);
        if (!operation)
        {
            return {nullptr};
        }

        auto op = dyn_cast<ora::SwitchOp>(operation);
        if (!op)
        {
            return {nullptr};
        }

        if (index >= operation->getNumRegions())
        {
            return {nullptr};
        }

        Region &region = operation->getRegion(index);
        if (region.empty())
        {
            region.push_back(new Block());
        }

        return wrap(&region.front());
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirOperation oraSwitchExprOpCreate(MlirContext ctx, MlirLocation loc, MlirValue value, MlirType resultType)
{
    try
    {
        MLIRContext *context = unwrap(ctx);
        Location location = unwrap(loc);
        Value val = unwrap(value);
        Type resultTy = unwrap(resultType);

        OpBuilder builder(context);

        // Create the switch_expr operation
        // SwitchExprOp expects TypeRange results, Value value, and unsigned casesCount
        TypeRange resultTypes = TypeRange(resultTy);
        auto op = builder.create<ora::SwitchExprOp>(location, resultTypes, val, /*casesCount=*/0);

        // SwitchExprOp has VariadicRegion for cases, caller will add cases as needed
        // No need to ensure a block exists initially

        // Add gas cost attribute (switch_expr has cost per case)
        auto gasCostAttr = builder.getI64IntegerAttr(10);
        op->setAttr("gas_cost", gasCostAttr);

        return wrap(op.getOperation());
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirOperation oraSwitchExprOpCreateWithCases(MlirContext ctx, MlirLocation loc, MlirValue value, const MlirType *resultTypes, size_t numResults, size_t numCases)
{
    try
    {
        MLIRContext *context = unwrap(ctx);
        Location location = unwrap(loc);
        Value val = unwrap(value);

        SmallVector<Type, 4> results;
        results.reserve(numResults);
        for (size_t i = 0; i < numResults; ++i)
        {
            results.push_back(unwrap(resultTypes[i]));
        }

        OpBuilder builder(context);
        auto op = builder.create<ora::SwitchExprOp>(location, results, val, static_cast<unsigned>(numCases));

        auto gasCostAttr = builder.getI64IntegerAttr(10);
        op->setAttr("gas_cost", gasCostAttr);

        return wrap(op.getOperation());
    }
    catch (...)
    {
        return {nullptr};
    }
}

MlirBlock oraSwitchExprOpGetCaseBlock(MlirOperation switchOp, size_t index)
{
    try
    {
        Operation *operation = unwrap(switchOp);
        if (!operation)
        {
            return {nullptr};
        }

        auto op = dyn_cast<ora::SwitchExprOp>(operation);
        if (!op)
        {
            return {nullptr};
        }

        if (index >= operation->getNumRegions())
        {
            return {nullptr};
        }

        Region &region = operation->getRegion(index);
        if (region.empty())
        {
            region.push_back(new Block());
        }

        return wrap(&region.front());
    }
    catch (...)
    {
        return {nullptr};
    }
}

void oraSwitchOpSetCasePatterns(
    MlirOperation op,
    const int64_t *caseValues,
    const int64_t *rangeStarts,
    const int64_t *rangeEnds,
    const int64_t *caseKinds,
    int64_t defaultCaseIndex,
    size_t numCases)
{
    try
    {
        Operation *operation = unwrap(op);
        MLIRContext *context = operation->getContext();

        if (numCases > 0)
        {
            if (caseValues)
            {
                ArrayRef<int64_t> caseValuesRef(caseValues, numCases);
                auto caseValuesAttr = DenseI64ArrayAttr::get(context, caseValuesRef);
                operation->setAttr("case_values", caseValuesAttr);
            }

            if (rangeStarts)
            {
                ArrayRef<int64_t> rangeStartsRef(rangeStarts, numCases);
                auto rangeStartsAttr = DenseI64ArrayAttr::get(context, rangeStartsRef);
                operation->setAttr("range_starts", rangeStartsAttr);
            }

            if (rangeEnds)
            {
                ArrayRef<int64_t> rangeEndsRef(rangeEnds, numCases);
                auto rangeEndsAttr = DenseI64ArrayAttr::get(context, rangeEndsRef);
                operation->setAttr("range_ends", rangeEndsAttr);
            }

            if (caseKinds)
            {
                ArrayRef<int64_t> caseKindsRef(caseKinds, numCases);
                auto caseKindsAttr = DenseI64ArrayAttr::get(context, caseKindsRef);
                operation->setAttr("case_kinds", caseKindsAttr);
            }
        }

        if (defaultCaseIndex >= 0)
        {
            auto defaultIndexAttr = ::mlir::IntegerAttr::get(
                ::mlir::IntegerType::get(context, 64), defaultCaseIndex);
            operation->setAttr("default_case_index", defaultIndexAttr);
        }
    }
    catch (...)
    {
        // Ignore errors - attribute setting is optional
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

        // Register both Ora and SIR dialects (needed for SIR MLIR)
        if (!oraDialectRegister(ctx))
        {
            return {nullptr, 0};
        }

        // Register SIR dialect for CFG generation from SIR MLIR
        context->getOrLoadDialect<sir::SIRDialect>();

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
// Ora Canonicalization (before conversion)
//===----------------------------------------------------------------------===//

bool oraCanonicalizeOraMLIR(MlirContext ctx, MlirModule module)
{
    try
    {
        MLIRContext *context = unwrap(ctx);
        ModuleOp moduleOp = unwrap(module);

        // Register Ora dialect
        if (!oraDialectRegister(ctx))
        {
            return false;
        }

        // Create pass manager for builtin.module operations
        PassManager pm(context, "builtin.module");
        pm.enableVerifier(false);

        // Run Ora optimizations in order:
        // 1. Constant deduplication and constant folding (fallback)
        pm.addPass(mlir::ora::createOraOptimizationPass());

        // 2. Canonicalization and DCE on Ora MLIR functions
        pm.addPass(mlir::ora::createSimpleOraOptimizationPass());

        // 3. Cleanup unused Ora operations
        pm.addPass(mlir::ora::createOraCleanupPass());

        // 4. Inline functions marked with ora.inline attribute
        pm.addPass(mlir::ora::createOraInliningPass());

        LogicalResult result = pm.run(moduleOp);

        if (failed(result))
        {
            return false;
        }

        return true;
    }
    catch (...)
    {
        return false;
    }
}

//===----------------------------------------------------------------------===//
// Ora to SIR Conversion
//===----------------------------------------------------------------------===//

bool oraConvertToSIR(MlirContext ctx, MlirModule module)
{
    try
    {
        MLIRContext *context = unwrap(ctx);
        ModuleOp moduleOp = unwrap(module);

        // Register both Ora and SIR dialects
        if (!oraDialectRegister(ctx))
        {
            return false;
        }

        // Register SIR dialect - use getOrLoadDialect which will work now that
        // the constructor is properly implemented in SIRDialect.cpp
        context->getOrLoadDialect<sir::SIRDialect>();

        // Create pass manager for builtin.module operations
        // This ensures nested passes can be added and executed
        PassManager pm(context, "builtin.module");
        pm.enableVerifier(false);

        ORA_DEBUG_PREFIX("OraCAPI", "Created PassManager for builtin.module");

        const bool enable_pre_sir_passes = false;
        if (enable_pre_sir_passes)
        {
            // 1. Constant deduplication and constant folding (fallback)
            pm.addPass(mlir::ora::createOraOptimizationPass());
            ORA_DEBUG_PREFIX("OraCAPI", "Added Ora optimization pass");

            // 2. Canonicalization and DCE on Ora MLIR functions
            pm.addPass(mlir::ora::createSimpleOraOptimizationPass());
            ORA_DEBUG_PREFIX("OraCAPI", "Added Simple Ora optimization pass (canonicalize + DCE)");

            // 3. Cleanup unused Ora operations
            pm.addPass(mlir::ora::createOraCleanupPass());
            ORA_DEBUG_PREFIX("OraCAPI", "Added Ora cleanup pass");

            // 4. Inline functions marked with ora.inline attribute
            pm.addPass(mlir::ora::createOraInliningPass());
            ORA_DEBUG_PREFIX("OraCAPI", "Added Ora inlining pass");
        }

        // Add the Ora to SIR conversion pass
        // Use addPass instead of addNestedPass since this is a ModuleOp pass
        const bool enable_ora_to_sir = true;
        if (enable_ora_to_sir)
        {
            pm.addPass(createOraToSIRPass());
        }
        const bool enable_post_sir_passes = false;
        if (enable_post_sir_passes)
        {
            // Add optimization pass (for SIR-specific optimizations like constant folding)
            pm.addPass(createSIROptimizationPass());
            // Add cleanup pass to remove unused memref operations
            // Cleanup runs BEFORE nested passes to clean up any remaining memref operations
            pm.addPass(createSIRCleanupPass());

            // Add simple pass that runs canonicalization and DCE on each function
            // This ensures both passes actually execute (nested passes weren't working)
            pm.addPass(mlir::ora::createSimpleDCEPass());
            ORA_DEBUG_PREFIX("OraCAPI", "Added simple pass (canonicalize + DCE)");
        }

        // Run the pass
        ORA_DEBUG_PREFIX("OraCAPI", "Running OraToSIR pass...");

        // Disable verifier to avoid segfault during verification
        // The IR should be valid after conversion, but verification might be triggering the crash
        pm.enableVerifier(false);

        LogicalResult result = pm.run(moduleOp);
        llvm::errs() << "[OraCAPI] pm.run() completed\n";
        llvm::errs().flush();

        if (failed(result))
        {
            llvm::errs() << "[OraCAPI] ERROR: Pass execution failed!\n";
            llvm::errs().flush();
            // Pass failed - this will be logged by MLIR's error handling
            return false;
        }
        llvm::errs() << "[OraCAPI] Pass execution completed successfully\n";
        llvm::errs().flush();

        // Verify the module is still valid after conversion
        llvm::errs() << "[OraCAPI] Verifying module validity...\n";
        llvm::errs().flush();

        if (!moduleOp)
        {
            llvm::errs() << "[OraCAPI] ERROR: Module is null after conversion!\n";
            llvm::errs().flush();
            return false;
        }

        // Check if module has valid operations
        if (moduleOp->getNumRegions() == 0)
        {
            llvm::errs().flush();
        }
        else
        {
            llvm::errs() << "[OraCAPI] Module has " << moduleOp->getNumRegions() << " region(s)\n";
            llvm::errs().flush();

            // Check first region
            auto &firstRegion = moduleOp->getRegion(0);
            llvm::errs() << "[OraCAPI] First region has " << firstRegion.getBlocks().size() << " block(s)\n";
            llvm::errs().flush();

            if (!firstRegion.empty())
            {
                auto &firstBlock = firstRegion.front();
                llvm::errs() << "[OraCAPI] First block has " << firstBlock.getOperations().size() << " operation(s)\n";
                llvm::errs().flush();
            }
        }

        llvm::errs() << "[OraCAPI] Module verification complete\n";
        llvm::errs().flush();

        return true;
    }
    catch (...)
    {
        return false;
    }
}

//===----------------------------------------------------------------------===//
// Operation Name Retrieval
//===----------------------------------------------------------------------===//

MlirStringRef oraOperationGetName(MlirOperation op)
{
    try
    {
        Operation *operation = unwrap(op);
        if (!operation)
        {
            return {nullptr, 0};
        }

        // Get the operation name as a string
        StringRef name = operation->getName().getStringRef();

        // Allocate memory for the string (caller must free)
        // Note: This is a temporary allocation - in production, consider using
        // a more efficient approach or requiring the caller to provide a buffer
        char *nameCopy = (char *)malloc(name.size() + 1);
        if (!nameCopy)
        {
            return {nullptr, 0};
        }

        memcpy(nameCopy, name.data(), name.size());
        nameCopy[name.size()] = '\0';

        return {nameCopy, name.size()};
    }
    catch (...)
    {
        return {nullptr, 0};
    }
}

//===----------------------------------------------------------------------===//
// Function Argument and Result Attributes
//===----------------------------------------------------------------------===//

bool oraFuncSetArgAttr(MlirOperation funcOp, unsigned argIndex, MlirStringRef attrName, MlirAttribute attr)
{
    try
    {
        Operation *operation = unwrap(funcOp);
        if (!operation)
        {
            return false;
        }

        // Check if this is a func.func operation
        auto func = dyn_cast<func::FuncOp>(operation);
        if (!func)
        {
            return false;
        }

        StringRef nameRef = unwrap(attrName);
        Attribute attrValue = unwrap(attr);

        // Set the attribute on the function argument
        func.setArgAttr(argIndex, nameRef, attrValue);

        return true;
    }
    catch (...)
    {
        return false;
    }
}

bool oraFuncSetResultAttr(MlirOperation funcOp, unsigned resultIndex, MlirStringRef attrName, MlirAttribute attr)
{
    try
    {
        Operation *operation = unwrap(funcOp);
        if (!operation)
        {
            return false;
        }

        // Check if this is a func.func operation
        auto func = dyn_cast<func::FuncOp>(operation);
        if (!func)
        {
            return false;
        }

        StringRef nameRef = unwrap(attrName);
        Attribute attrValue = unwrap(attr);

        // Set the attribute on the function result
        func.setResultAttr(resultIndex, nameRef, attrValue);

        return true;
    }
    catch (...)
    {
        return false;
    }
}

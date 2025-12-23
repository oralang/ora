#include "patterns/ControlFlow.h"
#include "OraToSIRTypeConverter.h"
#include "OraDebug.h"

#include "OraDialect.h"
#include "SIR/SIRDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "ora-to-sir"

using namespace mlir;
using namespace ora;

// Debug logging macro
#define DBG(msg) ORA_DEBUG_PREFIX("OraToSIR", msg)

// -----------------------------------------------------------------------------
// Lower func.func - convert function signature types (ora.int -> u256)
// -----------------------------------------------------------------------------
LogicalResult ConvertFuncOp::matchAndRewrite(
    mlir::func::FuncOp op,
    typename mlir::func::FuncOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    llvm::errs() << "[OraToSIR] ConvertFuncOp::matchAndRewrite() called for: " << op.getSymName() << "\n";
    llvm::errs().flush();

    // Always convert function signature - ensure all Ora types are converted to SIR types
    // This must happen before converting operations inside

    // Convert function signature types
    auto oldFuncType = op.getFunctionType();
    SmallVector<Type> newInputTypes;
    SmallVector<Type> newResultTypes;

    // Convert input types - ensure all Ora types become SIR u256
    for (Type inputType : oldFuncType.getInputs())
    {
        Type newType = this->getTypeConverter()->convertType(inputType);
        // If type converter didn't convert (returned same type), check if it's Ora type
        if (newType == inputType && llvm::isa<ora::IntegerType>(inputType))
        {
            // Force conversion to SIR u256
            newType = sir::U256Type::get(op.getContext());
        }
        newInputTypes.push_back(newType);
        llvm::errs() << "[OraToSIR]   Input type: " << inputType << " -> " << newType << "\n";
    }

    // Convert result types - ensure all Ora types become SIR u256
    // For functions that return values, SIR uses memory return: (ptr<1>, u256)
    // For void functions, keep them as void (no results)
    if (oldFuncType.getResults().empty())
    {
        // Void function - no results
        llvm::errs() << "[OraToSIR]   Result type: void -> void\n";
    }
    else
    {
    for (Type resultType : oldFuncType.getResults())
    {
        Type newType = this->getTypeConverter()->convertType(resultType);
        // If type converter didn't convert (returned same type), check if it's Ora type
        if (newType == resultType && llvm::isa<ora::IntegerType>(resultType))
        {
            // Force conversion to SIR u256
            newType = sir::U256Type::get(op.getContext());
        }

        // If the function returns a value (non-void), SIR returns via memory: (ptr<1>, len)
        // Convert single u256 return to (ptr<1>, u256)
        if (llvm::isa<sir::U256Type>(newType))
        {
            auto ptrType = sir::PtrType::get(op.getContext(), /*addrSpace*/ 1);
            newResultTypes.push_back(ptrType); // Return pointer
            newResultTypes.push_back(newType); // Return length (u256)
            llvm::errs() << "[OraToSIR]   Result type: " << resultType << " -> (" << ptrType << ", " << newType << ")\n";
        }
        else
        {
                // Other types - keep as is
            newResultTypes.push_back(newType);
            llvm::errs() << "[OraToSIR]   Result type: " << resultType << " -> " << newType << "\n";
            }
        }
    }
    llvm::errs().flush();

    // Check if any types changed
    bool typesChanged = false;
    if (newInputTypes.size() != oldFuncType.getInputs().size() ||
        newResultTypes.size() != oldFuncType.getResults().size())
    {
        typesChanged = true;
    }
    else
    {
        for (unsigned i = 0; i < newInputTypes.size(); ++i)
        {
            if (newInputTypes[i] != oldFuncType.getInput(i))
            {
                typesChanged = true;
                break;
            }
        }
        if (!typesChanged)
        {
            for (unsigned i = 0; i < newResultTypes.size(); ++i)
            {
                if (newResultTypes[i] != oldFuncType.getResult(i))
                {
                    typesChanged = true;
                    break;
                }
            }
        }
    }

    // Always convert function signature to ensure all Ora types are erased
    // Even if types appear unchanged, check if there are Ora types
    if (!typesChanged)
    {
        // Check if there are any Ora types remaining
        bool hasOraTypes = false;
        for (Type inputType : oldFuncType.getInputs())
        {
            if (llvm::isa<ora::IntegerType>(inputType))
            {
                hasOraTypes = true;
                break;
            }
        }
        if (!hasOraTypes)
        {
            for (Type resultType : oldFuncType.getResults())
            {
                if (llvm::isa<ora::IntegerType>(resultType))
                {
                    hasOraTypes = true;
                    break;
                }
            }
        }
        if (!hasOraTypes)
            return success(); // No Ora types, no conversion needed
        // Force conversion - types should have been converted but weren't
        llvm::errs() << "[OraToSIR] ConvertFuncOp: Forcing conversion despite typesChanged=false (Ora types detected)\n";
        llvm::errs().flush();
    }

    // Create new function type
    auto newFuncType = mlir::FunctionType::get(
        op.getContext(), newInputTypes, newResultTypes);

    // Update function signature and convert argument types
    rewriter.modifyOpInPlace(op, [&]()
                             { 
                                         op.setFunctionTypeAttr(TypeAttr::get(newFuncType));
                                         // Update function argument types to match new signature
                                         Block *entryBlock = &op.getBody().front();
                                         for (unsigned i = 0; i < newInputTypes.size(); ++i)
                                         {
                                             if (i < entryBlock->getNumArguments())
                                             {
                                                 entryBlock->getArgument(i).setType(newInputTypes[i]);
                                             }
                                         } });

    llvm::errs() << "[OraToSIR]   Updated function signature and arguments\n";
    llvm::errs().flush();
    return success();
}

// -----------------------------------------------------------------------------
// Keep ora.contract unchanged but allow recursion into its regions
// -----------------------------------------------------------------------------
LogicalResult ConvertContractOp::matchAndRewrite(
    ora::ContractOp op,
    typename ora::ContractOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    // Keep the contract unchanged - just mark as converted to allow recursion
    // The conversion will recurse into func.func operations inside
    DBG("ConvertContractOp: keeping ora.contract unchanged");
    return success(); // No changes needed, but allows recursion
}

// -----------------------------------------------------------------------------
// Lower ora.return to sir.return (EVM RETURN op)
// For void returns: sir.return with zero-sized allocation
// For non-void returns: sir.return %ptr, %len (memory return ABI)
// -----------------------------------------------------------------------------
LogicalResult ConvertReturnOp::matchAndRewrite(
    ora::ReturnOp op,
    typename ora::ReturnOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    DBG("ConvertReturnOp: matching return at " << op.getLoc());
    llvm::errs() << "[OraToSIR] ConvertReturnOp::matchAndRewrite() called at " << op.getLoc() << "\n";
    llvm::errs() << "[OraToSIR]   Operands count: " << op.getNumOperands() << "\n";
    if (op.getNumOperands() > 0)
    {
        llvm::errs() << "[OraToSIR]   Operand 0 type: " << op.getOperand(0).getType() << "\n";
    }
    llvm::errs().flush();
    auto loc = op.getLoc();
    auto ctx = rewriter.getContext();

    if (!ctx)
    {
        return rewriter.notifyMatchFailure(op, "null context");
    }

    auto operands = adaptor.getOperands();
    llvm::errs() << "[OraToSIR]   Adaptor operands count: " << operands.size() << "\n";
    if (operands.size() > 0)
    {
        llvm::errs() << "[OraToSIR]   Adaptor operand 0 type: " << operands[0].getType() << "\n";
    }
    llvm::errs().flush();
    auto u256Type = sir::U256Type::get(ctx);
    auto ptrType = sir::PtrType::get(ctx, /*addrSpace*/ 1);

    if (!u256Type || !ptrType)
    {
        return rewriter.notifyMatchFailure(op, "failed to create SIR types");
    }

    if (operands.empty())
    {
        // Void return - use sir.iret with no operands (internal return)
        // Note: sir.return requires ptr and len operands, so we use sir.iret for void
        // Create a distinct location for the return to add visual separation
        auto returnLoc = mlir::NameLoc::get(
            mlir::StringAttr::get(ctx, "return"),
            loc);
        rewriter.setInsertionPoint(op);
        rewriter.create<sir::IRetOp>(returnLoc, ValueRange{});
        rewriter.eraseOp(op);
        return success();
    }

    // Non-void return: get already-converted return value from adaptor
    Value retVal = operands[0];
    auto ui64Type = mlir::IntegerType::get(ctx, 64, mlir::IntegerType::Unsigned);

    // Ensure the value is SIR u256 type (adaptor should have already converted it)
    if (!llvm::isa<sir::U256Type>(retVal.getType()))
    {
        // Type converter should have handled this, but fallback to bitcast if needed
        Type convertedType = this->getTypeConverter()->convertType(retVal.getType());
        if (convertedType && convertedType != retVal.getType())
        {
            retVal = rewriter.create<sir::BitcastOp>(loc, convertedType, retVal);
        }
        else
        {
            // Force conversion to u256
            retVal = rewriter.create<sir::BitcastOp>(loc, u256Type, retVal);
        }
    }

    // Allocate 32 bytes for the return value
    auto sizeAttr = mlir::IntegerAttr::get(ui64Type, 32ULL);
    Value sizeConst = rewriter.create<sir::ConstOp>(loc, u256Type, sizeAttr);

    // Allocate memory
    Value mem = rewriter.create<sir::MallocOp>(loc, ptrType, sizeConst);

    // Store the return value into memory
    rewriter.create<sir::StoreOp>(loc, mem, retVal);

    // Create a distinct location for the return to add visual separation
    auto returnLoc = mlir::NameLoc::get(
        mlir::StringAttr::get(ctx, "return"),
        loc);

    // Replace func.return with sir.return %ptr, %len (EVM RETURN op)
    rewriter.setInsertionPoint(op);
    rewriter.create<sir::ReturnOp>(returnLoc, mem, sizeConst);
    rewriter.eraseOp(op);
    return success();
}

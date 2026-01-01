#include "patterns/ControlFlow.h"
#include "OraToSIRTypeConverter.h"
#include "OraDebug.h"

#include "OraDialect.h"
#include "SIR/SIRDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
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

    auto isOraType = [](Type type) -> bool {
        return type.getDialect().getNamespace() == "ora";
    };

    // Convert input types - ensure all Ora types become SIR types
    for (Type inputType : oldFuncType.getInputs())
    {
        Type newType = this->getTypeConverter()->convertType(inputType);
        if (!newType)
        {
            return rewriter.notifyMatchFailure(op, "failed to convert function input type");
        }
        // If type converter didn't convert (returned same type), check if it's Ora type
        if (newType == inputType && isOraType(inputType))
        {
            // Force conversion to SIR u256
            newType = sir::U256Type::get(op.getContext());
        }
        newInputTypes.push_back(newType);
        llvm::errs() << "[OraToSIR]   Input type: " << inputType << " -> " << newType << "\n";
    }

    // Convert result types - any non-void return uses memory return: (ptr<1>, u256)
    if (oldFuncType.getResults().empty())
    {
        // Void function - no results
        llvm::errs() << "[OraToSIR]   Result type: void -> void\n";
    }
    else
    {
        auto ptrType = sir::PtrType::get(op.getContext(), /*addrSpace*/ 1);
        auto u256Type = sir::U256Type::get(op.getContext());
        newResultTypes.push_back(ptrType);
        newResultTypes.push_back(u256Type);
        llvm::errs() << "[OraToSIR]   Result type: (non-void) -> (" << ptrType << ", " << u256Type << ")\n";
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
            if (isOraType(inputType))
            {
                hasOraTypes = true;
                break;
            }
        }
        if (!hasOraTypes)
        {
            for (Type resultType : oldFuncType.getResults())
            {
                if (isOraType(resultType))
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
                                                 op.removeArgAttr(i, rewriter.getStringAttr("ora.type"));
                                                 op.removeArgAttr(i, rewriter.getStringAttr("ora.name"));
                                             }
                                         } });
    for (unsigned i = 0; i < op.getNumResults(); ++i)
    {
        op.removeResultAttr(i, rewriter.getStringAttr("ora.type"));
    }

    llvm::errs() << "[OraToSIR]   Updated function signature and arguments\n";
    llvm::errs().flush();
    return success();
}

// -----------------------------------------------------------------------------
// Lower func.call - update call signature and materialize scalar return
// -----------------------------------------------------------------------------
LogicalResult ConvertCallOp::matchAndRewrite(
    mlir::func::CallOp op,
    typename mlir::func::CallOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    auto *typeConverter = getTypeConverter();
    if (!typeConverter)
    {
        return rewriter.notifyMatchFailure(op, "missing type converter");
    }

    SmallVector<Type> newResultTypes;
    auto oldResultTypes = op.getResultTypes();

    if (!oldResultTypes.empty())
    {
        auto ptrType = sir::PtrType::get(op.getContext(), /*addrSpace*/ 1);
        auto u256Type = sir::U256Type::get(op.getContext());
        newResultTypes.push_back(ptrType);
        newResultTypes.push_back(u256Type);
    }

    auto newCall = rewriter.create<mlir::func::CallOp>(
        op.getLoc(),
        op.getCallee(),
        newResultTypes,
        adaptor.getOperands());

    if (oldResultTypes.empty())
    {
        rewriter.replaceOp(op, newCall->getResults());
        return success();
    }

    if (newCall.getNumResults() < 1)
    {
        return rewriter.notifyMatchFailure(op, "expected pointer return for non-void call");
    }

    auto convertedType = typeConverter->convertType(oldResultTypes.front());
    if (!convertedType)
    {
        return rewriter.notifyMatchFailure(op, "unable to convert call result type");
    }

    auto loaded = rewriter.create<sir::LoadOp>(op.getLoc(), convertedType, newCall.getResult(0));
    rewriter.replaceOp(op, loaded.getResult());
    return success();
}

// -----------------------------------------------------------------------------
// Lower ora.contract by splicing its body into the parent module
// -----------------------------------------------------------------------------
LogicalResult ConvertContractOp::matchAndRewrite(
    ora::ContractOp op,
    typename ora::ContractOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    Region *parent = op->getParentRegion();
    if (!parent || parent->empty())
        return rewriter.notifyMatchFailure(op, "missing parent region");

    Block &moduleBlock = parent->front();
    Block &contractBlock = op.getBody().front();

    for (auto it = contractBlock.begin(); it != contractBlock.end();)
    {
        Operation *inner = &*it++;
        rewriter.moveOpBefore(inner, op);
    }

    rewriter.eraseOp(op);
    return success();
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

    ctx->getOrLoadDialect<sir::SIRDialect>();

    auto operands = adaptor.getOperands();
    llvm::errs() << "[OraToSIR]   Adaptor operands count: " << operands.size() << "\n";
    if (operands.size() > 0)
    {
        llvm::errs() << "[OraToSIR]   Adaptor operand 0 type: " << operands[0].getType() << "\n";
    }
    llvm::errs().flush();
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

    auto u256Type = sir::U256Type::get(ctx);
    auto ptrType = sir::PtrType::get(ctx, /*addrSpace*/ 1);

    if (!u256Type || !ptrType)
    {
        return rewriter.notifyMatchFailure(op, "failed to create SIR types");
    }

    // Non-void return: get already-converted return value from adaptor
    Value retVal = operands[0];
    auto ui64Type = mlir::IntegerType::get(ctx, 64, mlir::IntegerType::Unsigned);
    auto origType = op.getOperand(0).getType();
    const bool is_bytes_return = llvm::isa<ora::StringType, ora::BytesType>(origType);

    if (is_bytes_return)
    {
        // Return ABI for dynamic bytes: ptr + (len + 32)
        if (!llvm::isa<sir::PtrType>(retVal.getType()))
        {
            Type convertedType = this->getTypeConverter()->convertType(retVal.getType());
            if (convertedType && convertedType != retVal.getType())
            {
                retVal = rewriter.create<sir::BitcastOp>(loc, convertedType, retVal);
            }
        }

        Value length = rewriter.create<sir::LoadOp>(loc, u256Type, retVal);
        Value wordSize = rewriter.create<sir::ConstOp>(loc, u256Type, mlir::IntegerAttr::get(ui64Type, 32));
        Value sizeConst = rewriter.create<sir::AddOp>(loc, u256Type, length, wordSize);

        auto returnLoc = mlir::NameLoc::get(
            mlir::StringAttr::get(ctx, "return"),
            loc);
        rewriter.setInsertionPoint(op);
        rewriter.create<sir::ReturnOp>(returnLoc, retVal, sizeConst);
        rewriter.eraseOp(op);
        return success();
    }

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

    // Default return size is 32 bytes (EVM word)
    auto sizeAttr = mlir::IntegerAttr::get(ui64Type, 32);
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

// -----------------------------------------------------------------------------
// Lower ora.while → scf.while
// -----------------------------------------------------------------------------
LogicalResult ConvertWhileOp::matchAndRewrite(
    ora::WhileOp op,
    typename ora::WhileOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    auto loc = op.getLoc();
    auto ctx = rewriter.getContext();

    auto i1Type = mlir::IntegerType::get(ctx, 1);
    auto trueConst = rewriter.create<mlir::arith::ConstantIntOp>(loc, 1, 1);
    auto falseConst = rewriter.create<mlir::arith::ConstantIntOp>(loc, 0, 1);

    auto whileOp = rewriter.create<mlir::scf::WhileOp>(loc, TypeRange{i1Type}, ValueRange{trueConst});

    // Build before region: compute condition
    auto &beforeRegion = whileOp.getBefore();
    rewriter.createBlock(&beforeRegion);
    rewriter.setInsertionPointToEnd(&beforeRegion.front());

    Value cond = adaptor.getCondition();
    Value continueFlag = beforeRegion.front().addArgument(i1Type, loc);
    if (auto intType = dyn_cast<mlir::IntegerType>(cond.getType()))
    {
        if (intType.getWidth() != 1)
        {
            auto zero = rewriter.create<mlir::arith::ConstantOp>(
                loc,
                intType,
                rewriter.getIntegerAttr(intType, 0));
            cond = rewriter.create<mlir::arith::CmpIOp>(
                loc,
                mlir::arith::CmpIPredicate::ne,
                cond,
                zero);
        }
    }
    else if (llvm::isa<sir::U256Type>(cond.getType()))
    {
        auto i256Type = mlir::IntegerType::get(ctx, 256);
        auto zero = rewriter.create<mlir::arith::ConstantOp>(
            loc,
            i256Type,
            rewriter.getIntegerAttr(i256Type, 0));
        auto condI256 = rewriter.create<sir::BitcastOp>(loc, i256Type, cond);
        cond = rewriter.create<mlir::arith::CmpIOp>(
            loc,
            mlir::arith::CmpIPredicate::ne,
            condI256,
            zero);
    }

    if (cond.getType() == i1Type)
    {
        cond = rewriter.create<mlir::arith::AndIOp>(loc, cond, continueFlag);
    }

    rewriter.create<mlir::scf::ConditionOp>(loc, cond, ValueRange{});

    // Inline body region into after region
    auto &afterRegion = whileOp.getAfter();
    rewriter.inlineRegionBefore(op.getBody(), afterRegion, afterRegion.end());
    auto &afterEntry = afterRegion.front();
    afterEntry.addArgument(i1Type, loc);

    // Replace ora.break/ora.continue/ora.yield with scf.yield
    SmallVector<ora::BreakOp, 4> breaks;
    SmallVector<ora::ContinueOp, 4> continues;
    SmallVector<ora::YieldOp, 4> yields;
    afterRegion.walk([&](ora::BreakOp b) { breaks.push_back(b); });
    afterRegion.walk([&](ora::ContinueOp c) { continues.push_back(c); });
    afterRegion.walk([&](ora::YieldOp y) { yields.push_back(y); });
    for (auto b : breaks)
    {
        rewriter.setInsertionPoint(b);
        rewriter.replaceOpWithNewOp<mlir::scf::YieldOp>(b, ValueRange{falseConst});
    }
    for (auto c : continues)
    {
        rewriter.setInsertionPoint(c);
        rewriter.replaceOpWithNewOp<mlir::scf::YieldOp>(c, ValueRange{trueConst});
    }
    for (auto y : yields)
    {
        rewriter.setInsertionPoint(y);
        rewriter.replaceOpWithNewOp<mlir::scf::YieldOp>(y, ValueRange{trueConst});
    }

    rewriter.eraseOp(op);
    return success();
}

// -----------------------------------------------------------------------------
// Lower ora.if -> scf.if
// -----------------------------------------------------------------------------
LogicalResult ConvertIfOp::matchAndRewrite(
    ora::IfOp op,
    typename ora::IfOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    auto loc = op.getLoc();
    auto ctx = rewriter.getContext();

    SmallVector<Type> resultTypes;
    if (auto *tc = getTypeConverter())
    {
        for (Type t : op.getResultTypes())
        {
            Type converted = tc->convertType(t);
            if (!converted)
                return rewriter.notifyMatchFailure(op, "failed to convert if result type");
            resultTypes.push_back(converted);
        }
    }

    Value cond = adaptor.getCondition();
    auto i1Type = mlir::IntegerType::get(ctx, 1);
    if (auto intType = dyn_cast<mlir::IntegerType>(cond.getType()))
    {
        if (intType.getWidth() != 1)
        {
            auto zero = rewriter.create<mlir::arith::ConstantOp>(
                loc,
                intType,
                rewriter.getIntegerAttr(intType, 0));
            cond = rewriter.create<mlir::arith::CmpIOp>(
                loc,
                mlir::arith::CmpIPredicate::ne,
                cond,
                zero);
        }
    }
    else if (llvm::isa<sir::U256Type>(cond.getType()))
    {
        auto i256Type = mlir::IntegerType::get(ctx, 256);
        auto zero = rewriter.create<mlir::arith::ConstantOp>(
            loc,
            i256Type,
            rewriter.getIntegerAttr(i256Type, 0));
        auto condI256 = rewriter.create<sir::BitcastOp>(loc, i256Type, cond);
        cond = rewriter.create<mlir::arith::CmpIOp>(
            loc,
            mlir::arith::CmpIPredicate::ne,
            condI256,
            zero);
    }

    if (cond.getType() != i1Type)
    {
        return rewriter.notifyMatchFailure(op, "if condition is not i1 after conversion");
    }

    auto ifOp = rewriter.create<mlir::scf::IfOp>(loc, resultTypes, cond, /*withElseRegion=*/true);

    rewriter.inlineRegionBefore(op.getThenRegion(), ifOp.getThenRegion(), ifOp.getThenRegion().end());
    rewriter.inlineRegionBefore(op.getElseRegion(), ifOp.getElseRegion(), ifOp.getElseRegion().end());

    auto replaceYield = [&](mlir::Region &region) -> LogicalResult {
        SmallVector<ora::YieldOp, 4> yields;
        region.walk([&](ora::YieldOp y) { yields.push_back(y); });
        for (auto y : yields)
        {
            if (resultTypes.empty() && y.getNumOperands() != 0)
            {
                return rewriter.notifyMatchFailure(op, "if has yields but no results");
            }
            rewriter.setInsertionPoint(y);
            if (resultTypes.empty())
                rewriter.replaceOpWithNewOp<mlir::scf::YieldOp>(y, ValueRange{});
            else
                rewriter.replaceOpWithNewOp<mlir::scf::YieldOp>(y, y.getOperands());
        }
        return success();
    };

    if (failed(replaceYield(ifOp.getThenRegion())))
        return failure();
    if (failed(replaceYield(ifOp.getElseRegion())))
        return failure();

    rewriter.replaceOp(op, ifOp.getResults());
    return success();
}

LogicalResult ConvertBreakOp::matchAndRewrite(
    ora::BreakOp op,
    typename ora::BreakOp::Adaptor /*adaptor*/,
    ConversionPatternRewriter &rewriter) const
{
    Operation *parent = op->getParentOp();
    if (!parent)
        return rewriter.notifyMatchFailure(op, "break has no parent op");

    if (isa<mlir::scf::ExecuteRegionOp, mlir::scf::IfOp>(parent))
    {
        rewriter.replaceOpWithNewOp<mlir::scf::YieldOp>(op, ValueRange{});
        return success();
    }

    return rewriter.notifyMatchFailure(op, "break not in execute_region/if");
}

LogicalResult ConvertContinueOp::matchAndRewrite(
    ora::ContinueOp op,
    typename ora::ContinueOp::Adaptor /*adaptor*/,
    ConversionPatternRewriter &rewriter) const
{
    Operation *parent = op->getParentOp();
    if (!parent)
        return rewriter.notifyMatchFailure(op, "continue has no parent op");

    if (isa<mlir::scf::ExecuteRegionOp, mlir::scf::IfOp>(parent))
    {
        rewriter.replaceOpWithNewOp<mlir::scf::YieldOp>(op, ValueRange{});
        return success();
    }

    return rewriter.notifyMatchFailure(op, "continue not in execute_region/if");
}

LogicalResult ConvertSwitchExprOp::matchAndRewrite(
    ora::SwitchExprOp op,
    typename ora::SwitchExprOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    auto loc = op.getLoc();
    auto ctx = rewriter.getContext();

    // Convert result types
    SmallVector<Type> resultTypes;
    if (auto *tc = getTypeConverter())
    {
        for (Type t : op.getResultTypes())
        {
            Type converted = tc->convertType(t);
            if (!converted)
                return failure();
            resultTypes.push_back(converted);
        }
    }

    Value switchVal = adaptor.getValue();
    auto i256Type = mlir::IntegerType::get(ctx, 256);
    Value cmpVal = switchVal;
    if (llvm::isa<sir::U256Type>(switchVal.getType()))
    {
        cmpVal = rewriter.create<sir::BitcastOp>(loc, i256Type, switchVal);
    }
    else if (!llvm::isa<mlir::IntegerType>(switchVal.getType()))
    {
        return failure();
    }

    auto caseKinds = op.getCaseKindsAttr();
    auto caseValues = op.getCaseValuesAttr();
    auto rangeStarts = op.getRangeStartsAttr();
    auto rangeEnds = op.getRangeEndsAttr();
    auto defaultIdxAttr = op.getDefaultCaseIndexAttr();

    int64_t defaultIdx = -1;
    if (defaultIdxAttr)
        defaultIdx = defaultIdxAttr.getInt();

    SmallVector<int64_t> caseIdxs;
    int64_t numCases = static_cast<int64_t>(op.getCases().size());
    for (int64_t i = 0; i < numCases; ++i)
    {
        int64_t kind = 0;
        if (caseKinds && i < static_cast<int64_t>(caseKinds.size()))
            kind = caseKinds[i];
        if (kind == 2)
        {
            if (defaultIdx < 0)
                defaultIdx = i;
            continue;
        }
        caseIdxs.push_back(i);
    }

    if (defaultIdx < 0)
        return failure();

    auto inlineCaseRegion = [&](mlir::Region &src, mlir::Region &dest) -> bool {
        if (dest.empty())
            rewriter.createBlock(&dest);
        rewriter.inlineRegionBefore(src, dest, dest.end());
        SmallVector<ora::YieldOp, 4> yields;
        dest.walk([&](ora::YieldOp y) { yields.push_back(y); });
        if (yields.empty())
            return false;
        for (auto y : yields)
        {
            rewriter.setInsertionPoint(y);
            rewriter.replaceOpWithNewOp<mlir::scf::YieldOp>(y, y.getOperands());
        }
        return true;
    };

    std::function<mlir::scf::IfOp(int)> buildIf = [&](int idx) -> mlir::scf::IfOp {
        int64_t caseIdx = caseIdxs[idx];
        int64_t kind = 0;
        if (caseKinds && caseIdx < static_cast<int64_t>(caseKinds.size()))
            kind = caseKinds[caseIdx];

        Value cond;
        if (kind == 0 && caseValues && caseIdx < static_cast<int64_t>(caseValues.size()))
        {
            int64_t val = caseValues[caseIdx];
            auto cst = rewriter.create<mlir::arith::ConstantOp>(
                loc, i256Type, rewriter.getIntegerAttr(i256Type, val));
            cond = rewriter.create<mlir::arith::CmpIOp>(
                loc, mlir::arith::CmpIPredicate::eq, cmpVal, cst);
        }
        else if (kind == 1 && rangeStarts && rangeEnds &&
                 caseIdx < static_cast<int64_t>(rangeStarts.size()) &&
                 caseIdx < static_cast<int64_t>(rangeEnds.size()))
        {
            int64_t startVal = rangeStarts[caseIdx];
            int64_t endVal = rangeEnds[caseIdx];
            auto cStart = rewriter.create<mlir::arith::ConstantOp>(
                loc, i256Type, rewriter.getIntegerAttr(i256Type, startVal));
            auto cEnd = rewriter.create<mlir::arith::ConstantOp>(
                loc, i256Type, rewriter.getIntegerAttr(i256Type, endVal));
            auto ge = rewriter.create<mlir::arith::CmpIOp>(
                loc, mlir::arith::CmpIPredicate::uge, cmpVal, cStart);
            auto le = rewriter.create<mlir::arith::CmpIOp>(
                loc, mlir::arith::CmpIPredicate::ule, cmpVal, cEnd);
            cond = rewriter.create<mlir::arith::AndIOp>(loc, ge, le);
        }
        else
        {
            return mlir::scf::IfOp();
        }

        auto ifOp = rewriter.create<mlir::scf::IfOp>(loc, resultTypes, cond, true);
        if (!inlineCaseRegion(op.getCases()[caseIdx], ifOp.getThenRegion()))
            return mlir::scf::IfOp();

        if (idx + 1 >= static_cast<int>(caseIdxs.size()))
        {
            if (!inlineCaseRegion(op.getCases()[defaultIdx], ifOp.getElseRegion()))
                return mlir::scf::IfOp();
        }
        else
        {
            if (ifOp.getElseRegion().empty())
                rewriter.createBlock(&ifOp.getElseRegion());
            rewriter.setInsertionPointToStart(&ifOp.getElseRegion().front());
            auto nested = buildIf(idx + 1);
            if (!nested)
                return mlir::scf::IfOp();
            rewriter.create<mlir::scf::YieldOp>(loc, nested.getResults());
        }

        return ifOp;
    };

    if (caseIdxs.empty())
        return failure();

    rewriter.setInsertionPoint(op);
    auto rootIf = buildIf(0);
    if (!rootIf)
        return failure();

    rewriter.replaceOp(op, rootIf.getResults());
    return success();
}

LogicalResult ConvertSwitchOp::matchAndRewrite(
    ora::SwitchOp op,
    typename ora::SwitchOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    auto loc = op.getLoc();
    auto ctx = rewriter.getContext();

    Value switchVal = adaptor.getValue();
    auto i256Type = mlir::IntegerType::get(ctx, 256);
    Value cmpVal = switchVal;
    if (llvm::isa<sir::U256Type>(switchVal.getType()))
    {
        cmpVal = rewriter.create<sir::BitcastOp>(loc, i256Type, switchVal);
    }
    else if (!llvm::isa<mlir::IntegerType>(switchVal.getType()))
    {
        return failure();
    }

    auto caseKinds = op.getCaseKindsAttr();
    auto caseValues = op.getCaseValuesAttr();
    auto rangeStarts = op.getRangeStartsAttr();
    auto rangeEnds = op.getRangeEndsAttr();
    auto defaultIdxAttr = op.getDefaultCaseIndexAttr();

    int64_t defaultIdx = -1;
    if (defaultIdxAttr)
        defaultIdx = defaultIdxAttr.getInt();

    SmallVector<int64_t> caseIdxs;
    int64_t numCases = static_cast<int64_t>(op.getCases().size());
    for (int64_t i = 0; i < numCases; ++i)
    {
        int64_t kind = 0;
        if (caseKinds && i < static_cast<int64_t>(caseKinds.size()))
            kind = caseKinds[i];
        if (kind == 2)
        {
            if (defaultIdx < 0)
                defaultIdx = i;
            continue;
        }
        caseIdxs.push_back(i);
    }

    if (defaultIdx < 0)
        return failure();

    auto inlineCaseRegion = [&](mlir::Region &src, mlir::Region &dest) {
        if (dest.empty())
            rewriter.createBlock(&dest);
        rewriter.inlineRegionBefore(src, dest, dest.end());
        SmallVector<ora::YieldOp, 4> yields;
        dest.walk([&](ora::YieldOp y) { yields.push_back(y); });
        for (auto y : yields)
        {
            rewriter.setInsertionPoint(y);
            rewriter.replaceOpWithNewOp<mlir::scf::YieldOp>(y, y.getOperands());
        }
    };

    auto inlineCaseRegionStmt = [&](mlir::Region &src, mlir::Region &dest) {
        if (dest.empty())
            rewriter.createBlock(&dest);
        rewriter.inlineRegionBefore(src, dest, dest.end());
        SmallVector<ora::YieldOp, 4> yields;
        dest.walk([&](ora::YieldOp y) { yields.push_back(y); });
        for (auto y : yields)
        {
            rewriter.setInsertionPoint(y);
            rewriter.replaceOpWithNewOp<mlir::scf::YieldOp>(y, ValueRange{});
        }
        if (dest.front().empty() || !dest.front().back().hasTrait<mlir::OpTrait::IsTerminator>())
        {
            rewriter.setInsertionPointToEnd(&dest.front());
            rewriter.create<mlir::scf::YieldOp>(loc);
        }
    };

    std::function<mlir::scf::IfOp(int)> buildIf = [&](int idx) -> mlir::scf::IfOp {
        int64_t caseIdx = caseIdxs[idx];
        int64_t kind = 0;
        if (caseKinds && caseIdx < static_cast<int64_t>(caseKinds.size()))
            kind = caseKinds[caseIdx];

        Value cond;
        if (kind == 0 && caseValues && caseIdx < static_cast<int64_t>(caseValues.size()))
        {
            int64_t val = caseValues[caseIdx];
            auto cst = rewriter.create<mlir::arith::ConstantOp>(
                loc, i256Type, rewriter.getIntegerAttr(i256Type, val));
            cond = rewriter.create<mlir::arith::CmpIOp>(
                loc, mlir::arith::CmpIPredicate::eq, cmpVal, cst);
        }
        else if (kind == 1 && rangeStarts && rangeEnds &&
                 caseIdx < static_cast<int64_t>(rangeStarts.size()) &&
                 caseIdx < static_cast<int64_t>(rangeEnds.size()))
        {
            int64_t startVal = rangeStarts[caseIdx];
            int64_t endVal = rangeEnds[caseIdx];
            auto cStart = rewriter.create<mlir::arith::ConstantOp>(
                loc, i256Type, rewriter.getIntegerAttr(i256Type, startVal));
            auto cEnd = rewriter.create<mlir::arith::ConstantOp>(
                loc, i256Type, rewriter.getIntegerAttr(i256Type, endVal));
            auto ge = rewriter.create<mlir::arith::CmpIOp>(
                loc, mlir::arith::CmpIPredicate::uge, cmpVal, cStart);
            auto le = rewriter.create<mlir::arith::CmpIOp>(
                loc, mlir::arith::CmpIPredicate::ule, cmpVal, cEnd);
            cond = rewriter.create<mlir::arith::AndIOp>(loc, ge, le);
        }
        else
        {
            return mlir::scf::IfOp();
        }

        auto ifOp = rewriter.create<mlir::scf::IfOp>(loc, TypeRange{}, cond, true);
        inlineCaseRegionStmt(op.getCases()[caseIdx], ifOp.getThenRegion());

        if (idx + 1 >= static_cast<int>(caseIdxs.size()))
        {
            inlineCaseRegionStmt(op.getCases()[defaultIdx], ifOp.getElseRegion());
        }
        else
        {
            if (ifOp.getElseRegion().empty())
                rewriter.createBlock(&ifOp.getElseRegion());
            rewriter.setInsertionPointToStart(&ifOp.getElseRegion().front());
            auto nested = buildIf(idx + 1);
            if (!nested)
                return mlir::scf::IfOp();
        }

        return ifOp;
    };

    if (caseIdxs.empty())
        return failure();

    rewriter.setInsertionPoint(op);
    auto rootIf = buildIf(0);
    if (!rootIf)
        return failure();

    rewriter.eraseOp(op);
    return success();
}

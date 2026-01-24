#include "patterns/ControlFlow.h"
#include "OraToSIRTypeConverter.h"
#include "OraDebug.h"

#include "OraDialect.h"
#include "SIR/SIRDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <optional>

#define DEBUG_TYPE "ora-to-sir"

using namespace mlir;
using namespace ora;

// Debug logging macro
#define DBG(msg) ORA_DEBUG_PREFIX("OraToSIR", msg)

static Value ensureU256(PatternRewriter &rewriter, Location loc, Value value)
{
    auto u256Type = sir::U256Type::get(rewriter.getContext());
    if (llvm::isa<sir::U256Type>(value.getType()))
        return value;
    return rewriter.create<sir::BitcastOp>(loc, u256Type, value);
}

static Value toCondU256(PatternRewriter &rewriter, Location loc, Value value)
{
    auto u256Type = sir::U256Type::get(rewriter.getContext());
    Value v = ensureU256(rewriter, loc, value);
    Value isZero = rewriter.create<sir::IsZeroOp>(loc, u256Type, v);
    return rewriter.create<sir::IsZeroOp>(loc, u256Type, isZero);
}

static Value toIndex(ConversionPatternRewriter &rewriter, Location loc, Value value)
{
    if (llvm::isa<mlir::IndexType>(value.getType()))
        return value;
    return rewriter.create<sir::BitcastOp>(loc, rewriter.getIndexType(), value);
}

static std::optional<llvm::APInt> getConstValue(Value value)
{
    if (auto constOp = value.getDefiningOp<sir::ConstOp>())
    {
        if (auto intAttr = llvm::dyn_cast<IntegerAttr>(constOp.getValueAttr()))
            return intAttr.getValue();
        return std::nullopt;
    }

    if (auto arg = llvm::dyn_cast<BlockArgument>(value))
    {
        Block *block = arg.getOwner();
        Block *pred = block->getSinglePredecessor();
        if (!pred)
            return std::nullopt;

        auto *term = pred->getTerminator();
        auto br = llvm::dyn_cast<sir::BrOp>(term);
        if (!br)
            return std::nullopt;

        unsigned idx = arg.getArgNumber();
        if (idx >= br.getNumOperands())
            return std::nullopt;

        Value incoming = br.getOperand(idx);
        if (incoming == value)
            return std::nullopt;

        return getConstValue(incoming);
    }

    return std::nullopt;
}

static bool isNarrowErrorUnion(ora::ErrorUnionType type);

static LogicalResult materializeWideErrorUnion(
    PatternRewriter &rewriter,
    Location loc,
    Value operand,
    SmallVectorImpl<Value> &outValues)
{
    auto *ctx = rewriter.getContext();
    auto u256Type = sir::U256Type::get(ctx);
    auto u256IntType = mlir::IntegerType::get(ctx, 256, mlir::IntegerType::Unsigned);
    Value one = rewriter.create<sir::ConstOp>(loc, u256Type, mlir::IntegerAttr::get(u256IntType, 1));
    Value zero = rewriter.create<sir::ConstOp>(loc, u256Type, mlir::IntegerAttr::get(u256IntType, 0));

    auto toU256 = [&](Value v) -> Value { return ensureU256(rewriter, loc, v); };

    if (auto okOp = operand.getDefiningOp<ora::ErrorOkOp>())
    {
        outValues.push_back(zero);
        outValues.push_back(toU256(okOp.getValue()));
        return success();
    }
    if (auto errOp = operand.getDefiningOp<ora::ErrorErrOp>())
    {
        outValues.push_back(one);
        outValues.push_back(toU256(errOp.getValue()));
        return success();
    }
    if (auto castOp = operand.getDefiningOp<mlir::UnrealizedConversionCastOp>())
    {
        if (castOp.getNumOperands() == 2)
        {
            outValues.push_back(toU256(castOp.getOperand(0)));
            outValues.push_back(toU256(castOp.getOperand(1)));
            return success();
        }
    }

    if (auto errType = llvm::dyn_cast<ora::ErrorUnionType>(operand.getType()))
    {
        if (!isNarrowErrorUnion(errType))
        {
            llvm::errs() << "[OraToSIR] materializeWideErrorUnion: fallback split at " << loc << "\n";
            auto castOp = rewriter.create<mlir::UnrealizedConversionCastOp>(
                loc,
                TypeRange{u256Type, u256Type},
                operand);
            outValues.push_back(toU256(castOp.getResult(0)));
            outValues.push_back(toU256(castOp.getResult(1)));
            return success();
        }
    }

    return failure();
}

static std::optional<unsigned> getOraBitWidth(Type type)
{
    if (!type)
        return std::nullopt;
    if (auto builtinInt = llvm::dyn_cast<mlir::IntegerType>(type))
        return builtinInt.getWidth();
    if (auto intType = llvm::dyn_cast<ora::IntegerType>(type))
        return intType.getWidth();
    if (llvm::isa<ora::BoolType>(type))
        return 1u;
    if (llvm::isa<ora::AddressType>(type))
        return 160u;
    if (auto enumType = llvm::dyn_cast<ora::EnumType>(type))
        return getOraBitWidth(enumType.getReprType());
    if (auto errType = llvm::dyn_cast<ora::ErrorUnionType>(type))
        return getOraBitWidth(errType.getSuccessType());
    if (auto minType = llvm::dyn_cast<ora::MinValueType>(type))
        return getOraBitWidth(minType.getBaseType());
    if (auto maxType = llvm::dyn_cast<ora::MaxValueType>(type))
        return getOraBitWidth(maxType.getBaseType());
    if (auto rangeType = llvm::dyn_cast<ora::InRangeType>(type))
        return getOraBitWidth(rangeType.getBaseType());
    if (auto scaledType = llvm::dyn_cast<ora::ScaledType>(type))
        return getOraBitWidth(scaledType.getBaseType());
    if (auto exactType = llvm::dyn_cast<ora::ExactType>(type))
        return getOraBitWidth(exactType.getBaseType());
    if (llvm::isa<ora::StringType, ora::BytesType, ora::StructType, ora::MapType>(type))
        return 256u;
    return std::nullopt;
}

static bool isNarrowErrorUnion(ora::ErrorUnionType type)
{
    auto widthOpt = getOraBitWidth(type.getSuccessType());
    if (!widthOpt)
        return false;
    return *widthOpt <= 255;
}

static bool hasOpsAfterTerminator(Operation *op)
{
    return op && op->getNextNode();
}

static LogicalResult getErrorUnionEncodingTypes(const TypeConverter *typeConverter,
                                                Type resultType,
                                                SmallVector<Type> &convertedTypes)
{
    if (!typeConverter)
        return failure();
    if (failed(typeConverter->convertType(resultType, convertedTypes)))
    {
        if (auto errType = llvm::dyn_cast<ora::ErrorUnionType>(resultType))
        {
            auto *ctx = resultType.getContext();
            if (!ctx)
                return failure();
            auto u256 = sir::U256Type::get(ctx);
            if (isNarrowErrorUnion(errType))
            {
                convertedTypes.push_back(u256);
            }
            else
            {
                convertedTypes.push_back(u256);
                convertedTypes.push_back(u256);
            }
        }
        if (convertedTypes.empty())
            return failure();
    }
    if (convertedTypes.empty())
        return failure();
    return success();
}

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

    auto *typeConverter = this->getTypeConverter();
    if (!typeConverter)
    {
        return rewriter.notifyMatchFailure(op, "missing type converter");
    }

    // Convert function signature types (support 1->N for error unions).
    auto oldFuncType = op.getFunctionType();
    SmallVector<Type> newInputTypes;
    SmallVector<Type> newResultTypes;
    TypeConverter::SignatureConversion signatureConversion(oldFuncType.getNumInputs());

    auto isOraType = [](Type type) -> bool {
        return type.getDialect().getNamespace() == "ora";
    };

    // Convert input types - ensure all Ora types become SIR types
    for (auto [index, inputType] : llvm::enumerate(oldFuncType.getInputs()))
    {
        SmallVector<Type> convertedTypes;
        if (failed(typeConverter->convertType(inputType, convertedTypes)) || convertedTypes.empty())
        {
            return rewriter.notifyMatchFailure(op, "failed to convert function input type");
        }
        signatureConversion.addInputs(index, convertedTypes);
        newInputTypes.append(convertedTypes.begin(), convertedTypes.end());
        llvm::errs() << "[OraToSIR]   Input type: " << inputType << " -> " << convertedTypes.front() << "\n";
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
                             });
    if (failed(rewriter.convertRegionTypes(&op.getBody(), *typeConverter, &signatureConversion)))
        return rewriter.notifyMatchFailure(op, "failed to convert function body argument types");
    {
        Block *entryBlock = &op.getBody().front();
        for (unsigned i = 0; i < entryBlock->getNumArguments(); ++i)
        {
            op.removeArgAttr(i, rewriter.getStringAttr("ora.type"));
            op.removeArgAttr(i, rewriter.getStringAttr("ora.name"));
        }
    }
    for (unsigned i = 0; i < op.getNumResults(); ++i)
    {
        op.removeResultAttr(i, rewriter.getStringAttr("ora.type"));
    }

    llvm::errs() << "[OraToSIR]   Updated function signature and arguments\n";
    llvm::errs().flush();
    return success();
}

// -----------------------------------------------------------------------------
// Lower ora.range -> (end - start [+1 if inclusive])
// -----------------------------------------------------------------------------
LogicalResult ConvertRangeOp::matchAndRewrite(
    ora::RangeOp op,
    typename ora::RangeOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto u256Type = sir::U256Type::get(ctx);

    Value start = ensureU256(rewriter, loc, adaptor.getStart());
    Value end = ensureU256(rewriter, loc, adaptor.getEnd());

    Value diff = rewriter.create<sir::SubOp>(loc, u256Type, end, start);

    bool inclusive = false;
    if (auto inclusiveAttr = op.getInclusiveAttr())
        inclusive = inclusiveAttr.getValue();

    if (inclusive)
    {
        auto u256IntType = mlir::IntegerType::get(ctx, 256, mlir::IntegerType::Unsigned);
        auto oneAttr = mlir::IntegerAttr::get(u256IntType, 1);
        Value one = rewriter.create<sir::ConstOp>(loc, u256Type, oneAttr);
        diff = rewriter.create<sir::AddOp>(loc, u256Type, diff, one);
    }

    op->replaceAllUsesWith(ValueRange{diff});
    rewriter.eraseOp(op);
    return success();
}

// -----------------------------------------------------------------------------
// Lower ora.try_catch -> scf.if with catch region
// -----------------------------------------------------------------------------
LogicalResult ConvertTryCatchOp::matchAndRewrite(
    ora::TryOp op,
    typename ora::TryOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();

    auto resultType = op.getResult().getType();
    if (auto *tc = getTypeConverter())
    {
        if (auto converted = tc->convertType(resultType))
            resultType = converted;
    }

    auto operands = adaptor.getOperands();
    if (operands.empty())
        return failure();

    Value value = operands.front();
    if (!llvm::isa<ora::ErrorUnionType>(op.getTryOperation().getType()))
    {
        if (value.getType() != resultType)
        {
            if (llvm::isa<sir::PtrType>(resultType) && llvm::isa<sir::U256Type>(value.getType()))
            {
                Value casted = rewriter.create<sir::BitcastOp>(loc, resultType, value);
                op->replaceAllUsesWith(ValueRange{casted});
                rewriter.eraseOp(op);
                return success();
            }
            if (llvm::isa<sir::U256Type>(resultType) && llvm::isa<sir::PtrType>(value.getType()))
            {
                Value casted = rewriter.create<sir::BitcastOp>(loc, resultType, value);
                op->replaceAllUsesWith(ValueRange{casted});
                rewriter.eraseOp(op);
                return success();
            }
            return failure();
        }

        op->replaceAllUsesWith(ValueRange{value});
        rewriter.eraseOp(op);
        return success();
    }

    auto u256Type = sir::U256Type::get(ctx);
    auto u256IntType = mlir::IntegerType::get(ctx, 256, mlir::IntegerType::Unsigned);
    auto oneAttr = mlir::IntegerAttr::get(u256IntType, 1);
    Value one = rewriter.create<sir::ConstOp>(loc, u256Type, oneAttr);

    Value valueU256 = value;
    Value isErrU256;
    if (operands.size() == 2)
    {
        Value tag = ensureU256(rewriter, loc, operands[0]);
        isErrU256 = rewriter.create<sir::EqOp>(loc, u256Type, tag, one);
    }
    else
    {
        if (valueU256.getType() != u256Type)
            valueU256 = rewriter.create<sir::BitcastOp>(loc, u256Type, valueU256);
        Value masked = rewriter.create<sir::AndOp>(loc, u256Type, valueU256, one);
        isErrU256 = rewriter.create<sir::EqOp>(loc, u256Type, masked, one);
    }

    auto *parentBlock = op->getBlock();
    auto *parentRegion = parentBlock->getParent();
    auto mergeBlock = rewriter.splitBlock(parentBlock, Block::iterator(op));
    mergeBlock->addArgument(resultType, loc);

    SmallVector<ora::YieldOp, 4> tryYields;
    SmallVector<ora::YieldOp, 4> catchYields;
    op.getTryRegion().walk([&](ora::YieldOp y) {
        if (y->getParentOfType<ora::TryOp>() == op)
            tryYields.push_back(y);
    });
    op.getCatchRegion().walk([&](ora::YieldOp y) {
        if (y->getParentOfType<ora::TryOp>() == op)
            catchYields.push_back(y);
    });

    Block *catchBlock = nullptr;
    if (!op.getCatchRegion().empty())
        catchBlock = &op.getCatchRegion().front();
    rewriter.inlineRegionBefore(op.getCatchRegion(), *parentRegion, mergeBlock->getIterator());
    if (!catchBlock)
        catchBlock = rewriter.createBlock(parentRegion, mergeBlock->getIterator());

    Block *tryBlock = nullptr;
    if (!op.getTryRegion().empty())
        tryBlock = &op.getTryRegion().front();
    rewriter.inlineRegionBefore(op.getTryRegion(), *parentRegion, mergeBlock->getIterator());
    if (!tryBlock)
        tryBlock = rewriter.createBlock(parentRegion, mergeBlock->getIterator());

    auto *tc = getTypeConverter();
    auto replaceYieldWithBr = [&](ArrayRef<ora::YieldOp> yields) -> LogicalResult {
        for (auto y : yields)
        {
            if (y.getNumOperands() != 1)
                return failure();
            if (hasOpsAfterTerminator(y.getOperation()))
                return rewriter.notifyMatchFailure(y, "yield has trailing ops");
            rewriter.setInsertionPoint(y);
            Value operand = y.getOperand(0);
            Type targetType = mergeBlock->getArgument(0).getType();
            if (operand.getType() != targetType)
            {
                if (!tc)
                    return failure();
                Value converted = tc->materializeTargetConversion(rewriter, loc, targetType, operand);
                if (!converted)
                    return failure();
                operand = converted;
            }
            rewriter.replaceOpWithNewOp<sir::BrOp>(y, ValueRange{operand}, mergeBlock);
        }
        return success();
    };

    if (failed(replaceYieldWithBr(catchYields)))
        return failure();

    if (!op.getTryRegion().empty())
    {
        if (failed(replaceYieldWithBr(tryYields)))
            return failure();
    }
    else
    {
        rewriter.setInsertionPointToStart(tryBlock);
        Value unwrapped = valueU256;
        if (operands.size() == 2)
            unwrapped = ensureU256(rewriter, loc, operands[1]);
        else
            unwrapped = rewriter.create<sir::ShrOp>(loc, u256Type, one, valueU256);
        if (resultType != u256Type)
            unwrapped = rewriter.create<sir::BitcastOp>(loc, resultType, unwrapped);
        rewriter.create<sir::BrOp>(loc, ValueRange{unwrapped}, mergeBlock);
    }

    rewriter.setInsertionPointToEnd(parentBlock);
    rewriter.create<sir::CondBrOp>(loc, isErrU256, ValueRange{}, ValueRange{}, catchBlock, tryBlock);
    rewriter.setInsertionPointToStart(mergeBlock);
    op->replaceAllUsesWith(mergeBlock->getArguments());
    rewriter.eraseOp(op);
    return success();
}

// -----------------------------------------------------------------------------
// Lower ora.try_stmt -> CFG with error-aware unwrap
// -----------------------------------------------------------------------------
static LogicalResult rewriteErrorUnwrapInTryStmt(
    ArrayRef<ora::ErrorUnwrapOp> unwraps,
    Block *catchBlock,
    const TypeConverter *typeConverter,
    ConversionPatternRewriter &rewriter)
{
    auto *ctx = rewriter.getContext();
    auto u256Type = sir::U256Type::get(ctx);
    auto u256IntType = mlir::IntegerType::get(ctx, 256, mlir::IntegerType::Unsigned);
    auto oneAttr = mlir::IntegerAttr::get(u256IntType, 1);

    for (ora::ErrorUnwrapOp unwrap : unwraps)
    {
        auto loc = unwrap.getLoc();
        unsigned numOperands = unwrap->getNumOperands();
        if (numOperands == 0)
            return failure();

        rewriter.setInsertionPoint(unwrap);
        Value one = rewriter.create<sir::ConstOp>(loc, u256Type, oneAttr);
        Value payloadU256;
        Value isErrU256;
        if (numOperands == 2)
        {
            Value tag = ensureU256(rewriter, loc, unwrap->getOperand(0));
            payloadU256 = ensureU256(rewriter, loc, unwrap->getOperand(1));
            isErrU256 = rewriter.create<sir::EqOp>(loc, u256Type, tag, one);
        }
        else
        {
            Value valueU256 = ensureU256(rewriter, loc, unwrap->getOperand(0));
            Value masked = rewriter.create<sir::AndOp>(loc, u256Type, valueU256, one);
            isErrU256 = rewriter.create<sir::EqOp>(loc, u256Type, masked, one);
            payloadU256 = rewriter.create<sir::ShrOp>(loc, u256Type, one, valueU256);
        }

        Type resultType = unwrap.getResult().getType();
        if (typeConverter)
        {
            if (auto converted = typeConverter->convertType(resultType))
                resultType = converted;
        }

        Value payload = payloadU256;
        if (resultType != u256Type)
            payload = rewriter.create<sir::BitcastOp>(loc, resultType, payloadU256);

        Block *block = unwrap->getBlock();
        Block *contBlock = rewriter.splitBlock(block, Block::iterator(unwrap));
        contBlock->addArgument(resultType, loc);
        unwrap.getResult().replaceAllUsesWith(contBlock->getArgument(0));
        rewriter.eraseOp(unwrap);

        rewriter.setInsertionPointToEnd(block);
        SmallVector<Value> catchOperands;
        if (catchBlock->getNumArguments() > 1)
            return failure();
        if (catchBlock->getNumArguments() == 1)
        {
            Type catchType = catchBlock->getArgument(0).getType();
            Value errVal = payloadU256;
            if (catchType != u256Type)
                errVal = rewriter.create<sir::BitcastOp>(loc, catchType, payloadU256);
            catchOperands.push_back(errVal);
        }

        rewriter.setInsertionPointToEnd(block);
        rewriter.create<sir::CondBrOp>(
            loc,
            isErrU256,
            catchOperands,
            ValueRange{payload},
            catchBlock,
            contBlock);
    }

    return success();
}

LogicalResult ConvertTryStmtOp::matchAndRewrite(
    ora::TryStmtOp op,
    typename ora::TryStmtOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    (void)adaptor;
    auto loc = op.getLoc();
    auto *tc = getTypeConverter();
    llvm::errs() << "[OraToSIR] ConvertTryStmtOp: enter at " << loc << "\n";

    SmallVector<Type> resultTypes;
    for (Type t : op.getResultTypes())
    {
        SmallVector<Type> convertedTypes;
        if (!tc || failed(tc->convertType(t, convertedTypes)) || convertedTypes.empty())
            return rewriter.notifyMatchFailure(op, "failed to convert try_stmt result type");
        resultTypes.append(convertedTypes.begin(), convertedTypes.end());
    }

    SmallVector<ora::ErrorUnwrapOp, 4> unwraps;
    op.getTryRegion().walk([&](ora::ErrorUnwrapOp u) {
        if (u->getParentOp() == op.getOperation())
            unwraps.push_back(u);
    });

    SmallVector<ora::YieldOp, 4> tryYields;
    SmallVector<ora::YieldOp, 4> catchYields;
    op.getTryRegion().walk([&](ora::YieldOp y) {
        if (y->getParentOp() == op.getOperation())
            tryYields.push_back(y);
    });
    op.getCatchRegion().walk([&](ora::YieldOp y) {
        if (y->getParentOp() == op.getOperation())
            catchYields.push_back(y);
    });

    bool hasYields = !tryYields.empty() || !catchYields.empty();
    llvm::errs() << "[OraToSIR] ConvertTryStmtOp: results=" << resultTypes.size()
                 << " tryYields=" << tryYields.size()
                 << " catchYields=" << catchYields.size()
                 << " unwraps=" << unwraps.size() << "\n";

    Block *parentBlock = op->getBlock();
    Region *parentRegion = parentBlock->getParent();
    auto mergeBlock = rewriter.splitBlock(parentBlock, Block::iterator(op));
    if (hasYields)
    {
        for (Type t : resultTypes)
            mergeBlock->addArgument(t, loc);
    }

    Block *catchBlock = op.getCatchRegion().empty() ? nullptr : &op.getCatchRegion().front();
    Block *tryBlock = op.getTryRegion().empty() ? nullptr : &op.getTryRegion().front();
    rewriter.inlineRegionBefore(op.getCatchRegion(), *parentRegion, mergeBlock->getIterator());
    rewriter.inlineRegionBefore(op.getTryRegion(), *parentRegion, mergeBlock->getIterator());
    if (!catchBlock)
        catchBlock = rewriter.createBlock(parentRegion, mergeBlock->getIterator());
    if (!tryBlock)
        tryBlock = rewriter.createBlock(parentRegion, mergeBlock->getIterator());

    auto replaceYieldWithBr = [&](ArrayRef<ora::YieldOp> yields) -> LogicalResult {
        for (auto y : yields)
        {
            Block *yBlock = y->getBlock();
            bool hasLoopControl = false;
            for (auto br : yBlock->getOps<ora::BreakOp>())
            {
                (void)br;
                hasLoopControl = true;
                break;
            }
            if (!hasLoopControl)
            {
                for (auto cont : yBlock->getOps<ora::ContinueOp>())
                {
                    (void)cont;
                    hasLoopControl = true;
                    break;
                }
            }
            if (hasLoopControl)
            {
                if (!resultTypes.empty())
                    return rewriter.notifyMatchFailure(y, "try_stmt yield in loop-control block");
                rewriter.eraseOp(y);
                continue;
            }
            if (y.getNumOperands() != resultTypes.size())
                return rewriter.notifyMatchFailure(y, "try_stmt yield arity mismatch");
            if (hasOpsAfterTerminator(y.getOperation()))
                return rewriter.notifyMatchFailure(y, "yield has trailing ops");
            rewriter.setInsertionPoint(y);
            SmallVector<Value> convertedOperands;
            convertedOperands.reserve(y.getNumOperands());
            for (auto [idx, operand] : llvm::enumerate(y.getOperands()))
            {
                Type targetType = mergeBlock->getArgument(idx).getType();
                if (operand.getType() == targetType)
                {
                    convertedOperands.push_back(operand);
                    continue;
                }
                if (!tc)
                    return rewriter.notifyMatchFailure(y, "missing type converter");
                Value converted = tc->materializeTargetConversion(rewriter, loc, targetType, operand);
                if (!converted)
                    return rewriter.notifyMatchFailure(y, "failed to materialize try_stmt yield operand");
                convertedOperands.push_back(converted);
            }
            rewriter.replaceOpWithNewOp<sir::BrOp>(y, convertedOperands, mergeBlock);
        }
        return success();
    };

    if (failed(rewriteErrorUnwrapInTryStmt(unwraps, catchBlock, tc, rewriter)))
    {
        llvm::errs() << "[OraToSIR] ConvertTryStmtOp: unwrap rewrite failed at " << loc << "\n";
        return rewriter.notifyMatchFailure(op, "failed to rewrite error.unwrap in try_stmt");
    }
    if (failed(replaceYieldWithBr(tryYields)))
    {
        llvm::errs() << "[OraToSIR] ConvertTryStmtOp: try yields rewrite failed at " << loc << "\n";
        return rewriter.notifyMatchFailure(op, "failed to rewrite try_stmt try yields");
    }
    if (failed(replaceYieldWithBr(catchYields)))
    {
        llvm::errs() << "[OraToSIR] ConvertTryStmtOp: catch yields rewrite failed at " << loc << "\n";
        return rewriter.notifyMatchFailure(op, "failed to rewrite try_stmt catch yields");
    }
    rewriter.setInsertionPointToEnd(parentBlock);
    rewriter.create<sir::BrOp>(loc, ValueRange{}, tryBlock);

    if (!hasYields)
    {
        if (!op->use_empty())
            return rewriter.notifyMatchFailure(op, "try_stmt result used but no yields");
        rewriter.eraseOp(op);
        llvm::errs() << "[OraToSIR] ConvertTryStmtOp: erased statement-only try_stmt\n";
        return success();
    }

    if (resultTypes.empty())
        rewriter.eraseOp(op);
    else
    {
        if (static_cast<unsigned>(mergeBlock->getNumArguments()) != op.getNumResults())
        {
            llvm::errs() << "[OraToSIR] ConvertTryStmtOp: result mismatch op="
                         << op.getNumResults() << " merge="
                         << mergeBlock->getNumArguments() << "\n";
            return rewriter.notifyMatchFailure(op, "try_stmt result/merge mismatch");
        }
        llvm::errs() << "[OraToSIR] ConvertTryStmtOp: replace results="
                     << op.getNumResults() << "\n";
        rewriter.setInsertionPointToStart(mergeBlock);
        op->replaceAllUsesWith(mergeBlock->getArguments());
        rewriter.eraseOp(op);
    }

    return success();
}

// -----------------------------------------------------------------------------
// Lower ora.error.ok / ora.error.err -> pack payload into tagged value
// Layout: (payload << 1) | tag, where tag=0 for ok, tag=1 for error.
// -----------------------------------------------------------------------------
LogicalResult ConvertErrorOkOp::matchAndRewrite(
    ora::ErrorOkOp op,
    typename ora::ErrorOkOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    llvm::errs() << "[OraToSIR] ConvertErrorOkOp::matchAndRewrite() called at " << op.getLoc() << "\n";
    auto *ctx = rewriter.getContext();
    auto loc = op.getLoc();
    auto u256Type = sir::U256Type::get(ctx);
    auto u256IntType = mlir::IntegerType::get(ctx, 256, mlir::IntegerType::Unsigned);

    SmallVector<Type> resultTypes;
    if (failed(getErrorUnionEncodingTypes(getTypeConverter(), op.getResult().getType(), resultTypes)))
    {
        llvm::errs() << "[OraToSIR] ConvertErrorOkOp: failed to get encoding types at " << loc << "\n";
        return failure();
    }
    llvm::errs() << "[OraToSIR] ConvertErrorOkOp: resultTypes=" << resultTypes.size() << " at " << loc << "\n";

    Value value = adaptor.getValue();
    value = ensureU256(rewriter, loc, value);

    if (resultTypes.size() == 1)
    {
        auto oneAttr = mlir::IntegerAttr::get(u256IntType, 1);
        auto zeroAttr = mlir::IntegerAttr::get(u256IntType, 0);
        Value one = rewriter.create<sir::ConstOp>(loc, u256Type, oneAttr);
        Value zero = rewriter.create<sir::ConstOp>(loc, u256Type, zeroAttr);
        Value shifted = rewriter.create<sir::ShlOp>(loc, u256Type, one, value);
        Value packed = rewriter.create<sir::OrOp>(loc, u256Type, shifted, zero);
        rewriter.replaceOp(op, ValueRange{packed});
    }
    else
    {
        auto zeroAttr = mlir::IntegerAttr::get(u256IntType, 0);
        Value tag = rewriter.create<sir::ConstOp>(loc, u256Type, zeroAttr);
        rewriter.replaceOpWithMultiple(op, ArrayRef<ValueRange>{ValueRange{tag, value}});
    }
    return success();
}

LogicalResult ConvertErrorErrOp::matchAndRewrite(
    ora::ErrorErrOp op,
    typename ora::ErrorErrOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    llvm::errs() << "[OraToSIR] ConvertErrorErrOp::matchAndRewrite() called at " << op.getLoc() << "\n";
    auto *ctx = rewriter.getContext();
    auto loc = op.getLoc();
    auto u256Type = sir::U256Type::get(ctx);
    auto u256IntType = mlir::IntegerType::get(ctx, 256, mlir::IntegerType::Unsigned);

    SmallVector<Type> resultTypes;
    if (failed(getErrorUnionEncodingTypes(getTypeConverter(), op.getResult().getType(), resultTypes)))
    {
        llvm::errs() << "[OraToSIR] ConvertErrorErrOp: failed to get encoding types at " << loc << "\n";
        return failure();
    }
    llvm::errs() << "[OraToSIR] ConvertErrorErrOp: resultTypes=" << resultTypes.size() << " at " << loc << "\n";

    Value value = adaptor.getValue();
    value = ensureU256(rewriter, loc, value);

    if (resultTypes.size() == 1)
    {
        auto oneAttr = mlir::IntegerAttr::get(u256IntType, 1);
        Value one = rewriter.create<sir::ConstOp>(loc, u256Type, oneAttr);
        Value shifted = rewriter.create<sir::ShlOp>(loc, u256Type, one, value);
        Value packed = rewriter.create<sir::OrOp>(loc, u256Type, shifted, one);
        rewriter.replaceOp(op, ValueRange{packed});
    }
    else
    {
        auto oneAttr = mlir::IntegerAttr::get(u256IntType, 1);
        Value tag = rewriter.create<sir::ConstOp>(loc, u256Type, oneAttr);
        rewriter.replaceOpWithMultiple(op, ArrayRef<ValueRange>{ValueRange{tag, value}});
    }
    return success();
}

// -----------------------------------------------------------------------------
// Fold sir.cond_br with identical destinations/operands into sir.br
// -----------------------------------------------------------------------------
LogicalResult FoldCondBrSameDestOp::matchAndRewrite(
    sir::CondBrOp op,
    PatternRewriter &rewriter) const
{
    if (op.getTrueDest() != op.getFalseDest())
        return failure();

    if (!llvm::equal(op.getTrueOperands(), op.getFalseOperands()))
        return failure();

    rewriter.replaceOpWithNewOp<sir::BrOp>(op, op.getTrueOperands(), op.getTrueDest());
    return success();
}

// -----------------------------------------------------------------------------
// Fold sir.cond_br with iszero(iszero(x)) condition to use x directly
// -----------------------------------------------------------------------------
LogicalResult FoldCondBrDoubleIsZeroOp::matchAndRewrite(
    sir::CondBrOp op,
    PatternRewriter &rewriter) const
{
    auto outer = op.getCond().getDefiningOp<sir::IsZeroOp>();
    if (!outer)
        return failure();
    auto inner = outer.getX().getDefiningOp<sir::IsZeroOp>();
    if (!inner)
        return failure();

    Value newCond = inner.getX();
    rewriter.replaceOpWithNewOp<sir::CondBrOp>(
        op,
        newCond,
        op.getTrueOperands(),
        op.getFalseOperands(),
        op.getTrueDest(),
        op.getFalseDest());
    return success();
}

// -----------------------------------------------------------------------------
// Fold sir.cond_br with constant condition to sir.br
// -----------------------------------------------------------------------------
LogicalResult FoldCondBrConstOp::matchAndRewrite(
    sir::CondBrOp op,
    PatternRewriter &rewriter) const
{
    auto constValue = getConstValue(op.getCond());
    if (!constValue)
        return failure();

    bool condTrue = !constValue->isZero();
    if (condTrue)
    {
        rewriter.replaceOpWithNewOp<sir::BrOp>(op, op.getTrueOperands(), op.getTrueDest());
    }
    else
    {
        rewriter.replaceOpWithNewOp<sir::BrOp>(op, op.getFalseOperands(), op.getFalseDest());
    }
    return success();
}

// -----------------------------------------------------------------------------
// Fold sir.br to a destination that only forwards to another sir.br
// -----------------------------------------------------------------------------
LogicalResult FoldBrToBrOp::matchAndRewrite(
    sir::BrOp op,
    PatternRewriter &rewriter) const
{
    Block *dest = op.getDest();
    if (!dest || dest->getOperations().size() != 1)
        return failure();

    auto nextBr = llvm::dyn_cast<sir::BrOp>(dest->front());
    if (!nextBr)
        return failure();

    if (nextBr.getNumOperands() != dest->getNumArguments())
        return failure();

    SmallVector<Value, 4> forwardedOperands;
    forwardedOperands.reserve(nextBr.getNumOperands());
    for (auto [arg, incoming] : llvm::zip(dest->getArguments(), nextBr.getOperands()))
    {
        if (auto blockArg = llvm::dyn_cast<BlockArgument>(incoming))
        {
            if (blockArg.getOwner() != dest)
            {
                forwardedOperands.clear();
                break;
            }
            unsigned idx = blockArg.getArgNumber();
            if (idx >= op.getNumOperands())
            {
                forwardedOperands.clear();
                break;
            }
            forwardedOperands.push_back(op.getOperand(idx));
        }
        else
        {
            forwardedOperands.push_back(incoming);
        }
    }

    if (forwardedOperands.empty() && nextBr.getNumOperands() != 0)
        return failure();

    rewriter.replaceOpWithNewOp<sir::BrOp>(op, forwardedOperands, nextBr.getDest());
    return success();
}

// -----------------------------------------------------------------------------
// Lower ora.error.is_error -> check LSB of encoded value
// -----------------------------------------------------------------------------
LogicalResult ConvertErrorIsErrorOp::matchAndRewrite(
    ora::ErrorIsErrorOp op,
    typename ora::ErrorIsErrorOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    auto *ctx = rewriter.getContext();
    auto loc = op.getLoc();
    auto u256Type = sir::U256Type::get(ctx);
    auto i1Type = mlir::IntegerType::get(ctx, 1);
    auto u256IntType = mlir::IntegerType::get(ctx, 256, mlir::IntegerType::Unsigned);
    auto oneAttr = mlir::IntegerAttr::get(u256IntType, 1);
    Value one = rewriter.create<sir::ConstOp>(loc, u256Type, oneAttr);

    auto operands = adaptor.getOperands();
    if (operands.empty())
        return failure();

    Value isErrU256;
    if (operands.size() == 1)
    {
        Value valueU256 = ensureU256(rewriter, loc, operands[0]);
        Value masked = rewriter.create<sir::AndOp>(loc, u256Type, valueU256, one);
        isErrU256 = rewriter.create<sir::EqOp>(loc, u256Type, masked, one);
    }
    else
    {
        Value tag = ensureU256(rewriter, loc, operands[0]);
        isErrU256 = rewriter.create<sir::EqOp>(loc, u256Type, tag, one);
    }
    Value isErr = rewriter.create<sir::BitcastOp>(loc, i1Type, isErrU256);

    op->replaceAllUsesWith(ValueRange{isErr});
    rewriter.eraseOp(op);
    return success();
}

// -----------------------------------------------------------------------------
// Lower ora.error.unwrap / ora.error.get_error -> shift right by 1
// -----------------------------------------------------------------------------
LogicalResult ConvertErrorUnwrapOp::matchAndRewrite(
    ora::ErrorUnwrapOp op,
    typename ora::ErrorUnwrapOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    if (op->getParentOfType<ora::TryStmtOp>())
    {
        return rewriter.notifyMatchFailure(op, "handled by ora.try_stmt lowering");
    }

    auto *ctx = rewriter.getContext();
    auto loc = op.getLoc();
    auto u256Type = sir::U256Type::get(ctx);
    auto u256IntType = mlir::IntegerType::get(ctx, 256, mlir::IntegerType::Unsigned);
    auto oneAttr = mlir::IntegerAttr::get(u256IntType, 1);
    Value one = rewriter.create<sir::ConstOp>(loc, u256Type, oneAttr);

    auto resultType = op.getResult().getType();
    if (auto *tc = getTypeConverter())
        if (auto converted = tc->convertType(resultType))
            resultType = converted;

    auto operands = adaptor.getOperands();
    if (operands.empty())
        return failure();

    Value payload = ensureU256(rewriter, loc, operands[0]);
    if (operands.size() == 1)
        payload = rewriter.create<sir::ShrOp>(loc, u256Type, one, payload);
    else
        payload = ensureU256(rewriter, loc, operands[1]);

    if (resultType != u256Type)
        payload = rewriter.create<sir::BitcastOp>(loc, resultType, payload);

    op->replaceAllUsesWith(ValueRange{payload});
    rewriter.eraseOp(op);
    return success();
}

LogicalResult ConvertErrorGetErrorOp::matchAndRewrite(
    ora::ErrorGetErrorOp op,
    typename ora::ErrorGetErrorOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    auto *ctx = rewriter.getContext();
    auto loc = op.getLoc();
    auto u256Type = sir::U256Type::get(ctx);
    auto ui64Type = mlir::IntegerType::get(ctx, 64, mlir::IntegerType::Unsigned);
    auto oneAttr = mlir::IntegerAttr::get(ui64Type, 1);
    Value one = rewriter.create<sir::ConstOp>(loc, u256Type, oneAttr);

    auto resultType = op.getResult().getType();
    if (auto *tc = getTypeConverter())
        if (auto converted = tc->convertType(resultType))
            resultType = converted;

    auto operands = adaptor.getOperands();
    if (operands.empty())
        return failure();

    Value payload = ensureU256(rewriter, loc, operands[0]);
    if (operands.size() == 1)
        payload = rewriter.create<sir::ShrOp>(loc, u256Type, one, payload);
    else
        payload = ensureU256(rewriter, loc, operands[1]);

    if (resultType != u256Type)
        payload = rewriter.create<sir::BitcastOp>(loc, resultType, payload);

    op->replaceAllUsesWith(ValueRange{payload});
    rewriter.eraseOp(op);
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
    llvm::errs() << "[OraToSIR] ConvertCallOp: callee=" << op.getCallee()
                 << " results=" << op.getNumResults() << " at " << op.getLoc() << "\n";
    auto *typeConverter = getTypeConverter();
    if (!typeConverter)
    {
        return rewriter.notifyMatchFailure(op, "missing type converter");
    }

    SmallVector<Type> newResultTypes;
    auto oldResultTypes = op.getResultTypes();

    auto lowerErrorDeclCall = [&](ora::ErrorDeclOp errDecl) -> LogicalResult {
        auto errIdAttr = errDecl->getAttrOfType<mlir::IntegerAttr>("ora.error_id");
        if (!errIdAttr)
            return rewriter.notifyMatchFailure(op, "error.decl missing ora.error_id");

        auto *ctx = op.getContext();
        auto u256Type = sir::U256Type::get(ctx);
        auto u256IntType = mlir::IntegerType::get(ctx, 256, mlir::IntegerType::Unsigned);
        auto idVal = errIdAttr.getValue().zextOrTrunc(256);
        auto idAttr = mlir::IntegerAttr::get(u256IntType, idVal);
        Value idConst = rewriter.create<sir::ConstOp>(op.getLoc(), u256Type, idAttr);

        if (oldResultTypes.empty())
        {
            rewriter.eraseOp(op);
            return success();
        }

        SmallVector<Type> convertedTypes;
        if (failed(typeConverter->convertType(oldResultTypes.front(), convertedTypes)) || convertedTypes.empty())
        {
            if (Type converted = typeConverter->convertType(oldResultTypes.front()))
                convertedTypes.push_back(converted);
        }
        if (convertedTypes.empty())
            return rewriter.notifyMatchFailure(op, "unable to convert error.decl call result type");

        Value result = idConst;
        if (convertedTypes.front() != u256Type)
            result = rewriter.create<sir::BitcastOp>(op.getLoc(), convertedTypes.front(), idConst);

        op->replaceAllUsesWith(ValueRange{result});
        rewriter.eraseOp(op);
        return success();
    };

    StringRef calleeName = op.getCallee();
    if (!calleeName.empty())
    {
        auto calleeAttr = mlir::StringAttr::get(op.getContext(), calleeName);
        if (Operation *symbol = mlir::SymbolTable::lookupNearestSymbolFrom(op, calleeAttr))
        {
            if (auto errDecl = dyn_cast<ora::ErrorDeclOp>(symbol))
                return lowerErrorDeclCall(errDecl);
        }
        if (auto module = op->getParentOfType<mlir::ModuleOp>())
        {
            ora::ErrorDeclOp found;
            module.walk([&](ora::ErrorDeclOp decl) {
                auto sym = decl->getAttrOfType<mlir::StringAttr>("sym_name");
                if (sym && sym.getValue() == calleeName)
                    found = decl;
            });
            if (found)
                return lowerErrorDeclCall(found);
        }
    }

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
        op->replaceAllUsesWith(newCall->getResults());
        rewriter.eraseOp(op);
        return success();
    }

    if (newCall.getNumResults() < 1)
    {
        llvm::errs() << "[OraToSIR] ConvertCallOp: newCall has no results\n";
        return rewriter.notifyMatchFailure(op, "expected pointer return for non-void call");
    }

    SmallVector<Type> convertedTypes;
    if (llvm::isa<ora::ErrorUnionType>(oldResultTypes.front()))
    {
        auto ctx = op.getContext();
        auto u256Type = sir::U256Type::get(ctx);
        convertedTypes.push_back(u256Type);
        convertedTypes.push_back(u256Type);
    }
    else if (failed(typeConverter->convertType(oldResultTypes.front(), convertedTypes)) || convertedTypes.empty())
    {
        if (Type converted = typeConverter->convertType(oldResultTypes.front()))
            convertedTypes.push_back(converted);
    }
    if (convertedTypes.empty())
    {
        llvm::errs() << "[OraToSIR] ConvertCallOp: unable to convert result type " << oldResultTypes.front() << "\n";
        return rewriter.notifyMatchFailure(op, "unable to convert call result type");
    }

    auto ptr = newCall.getResult(0);
    if (convertedTypes.size() == 1)
    {
        if (llvm::isa<ora::BytesType>(oldResultTypes.front()) ||
            llvm::isa<ora::StringType>(oldResultTypes.front()))
        {
            op->replaceAllUsesWith(ValueRange{ptr});
        }
        else
        {
            auto loaded = rewriter.create<sir::LoadOp>(op.getLoc(), convertedTypes.front(), ptr);
            op->replaceAllUsesWith(ValueRange{loaded.getResult()});
        }
        rewriter.eraseOp(op);
        return success();
    }

    auto ctx = op.getContext();
    auto u256Type = sir::U256Type::get(ctx);
    auto u256IntType = mlir::IntegerType::get(ctx, 256, mlir::IntegerType::Unsigned);
    Value offset = rewriter.create<sir::ConstOp>(op.getLoc(), u256Type, mlir::IntegerAttr::get(u256IntType, 32));
    Value ptr2 = rewriter.create<sir::AddPtrOp>(op.getLoc(), ptr.getType(), ptr, offset);
    auto tag = rewriter.create<sir::LoadOp>(op.getLoc(), u256Type, ptr);
    auto payload = rewriter.create<sir::LoadOp>(op.getLoc(), u256Type, ptr2);
    op->replaceAllUsesWith(ValueRange{tag.getResult(), payload.getResult()});
    rewriter.eraseOp(op);
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

    Block &contractBlock = op.getBody().front();

    for (auto it = contractBlock.begin(); it != contractBlock.end();)
    {
        Operation *inner = &*it++;
        if (llvm::isa<ora::YieldOp>(inner))
        {
            rewriter.eraseOp(inner);
            continue;
        }
        rewriter.moveOpBefore(inner, op);
    }

    rewriter.eraseOp(op);
    return success();
}

// -----------------------------------------------------------------------------
// Lower ora.error.decl to sir.error.decl metadata
// -----------------------------------------------------------------------------
LogicalResult ConvertErrorDeclOp::matchAndRewrite(
    ora::ErrorDeclOp op,
    typename ora::ErrorDeclOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    auto *typeConverter = getTypeConverter();
    if (!typeConverter)
        return rewriter.notifyMatchFailure(op, "missing type converter");

    SmallVector<NamedAttribute> attrs;
    attrs.reserve(op->getAttrs().size());
    for (auto attr : op->getAttrs())
    {
        StringRef name = attr.getName();
        if (name == "ora.error_decl")
        {
            continue;
        }
        if (name == "ora.error_id")
        {
            attrs.push_back(rewriter.getNamedAttr("sir.error_id", attr.getValue()));
            continue;
        }
        if (name == "ora.param_names")
        {
            attrs.push_back(rewriter.getNamedAttr("sir.param_names", attr.getValue()));
            continue;
        }
        if (name == "ora.param_types")
        {
            auto arr = llvm::dyn_cast<ArrayAttr>(attr.getValue());
            if (!arr)
                return rewriter.notifyMatchFailure(op, "ora.param_types is not ArrayAttr");
            SmallVector<Attribute> converted;
            converted.reserve(arr.size());
            for (auto elem : arr)
            {
                auto typeAttr = llvm::dyn_cast<TypeAttr>(elem);
                if (!typeAttr)
                    return rewriter.notifyMatchFailure(op, "ora.param_types element is not TypeAttr");
                Type origType = typeAttr.getValue();
                Type convertedType = typeConverter->convertType(origType);
                if (!convertedType)
                    return rewriter.notifyMatchFailure(op, "unable to convert error param type");
                if (convertedType == origType)
                {
                    if (auto intType = llvm::dyn_cast<mlir::IntegerType>(origType))
                    {
                        if (intType.getWidth() == 256)
                            convertedType = sir::U256Type::get(op.getContext());
                    }
                }
                converted.push_back(TypeAttr::get(convertedType));
            }
            attrs.push_back(rewriter.getNamedAttr("sir.param_types", rewriter.getArrayAttr(converted)));
            continue;
        }
        if (name.starts_with("ora."))
            continue;
        attrs.push_back(attr);
    }

    OperationState state(op.getLoc(), sir::ErrorDeclOp::getOperationName());
    state.addAttributes(attrs);
    rewriter.create(state);
    rewriter.eraseOp(op);
    return success();
}

static LogicalResult convertOraReturn(
    ora::ReturnOp op,
    ArrayRef<Value> operands,
    const TypeConverter *tc,
    PatternRewriter &rewriter)
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
    rewriter.setInsertionPoint(op);
    llvm::errs() << "[OraToSIR]   Adaptor operands count: " << operands.size() << "\n";
    if (operands.size() > 0)
    {
        llvm::errs() << "[OraToSIR]   Adaptor operand 0 type: " << operands[0].getType() << "\n";
    }
    llvm::errs().flush();
    if (operands.empty())
    {
        llvm::errs() << "[OraToSIR] ConvertReturnOp: void return path at " << loc << "\n";
        // Void return - use sir.iret with no operands (internal return)
        // Note: sir.return requires ptr and len operands, so we use sir.iret for void
        rewriter.create<sir::IRetOp>(loc, ValueRange{});
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
    if (auto errType = llvm::dyn_cast<ora::ErrorUnionType>(origType))
    {
        if (!isNarrowErrorUnion(errType))
        {
            if (operands.size() == 1)
            {
                if (auto cast = operands[0].getDefiningOp<mlir::UnrealizedConversionCastOp>())
                {
                    if (cast.getNumOperands() == 2)
                    {
                        llvm::errs() << "[OraToSIR] ConvertReturnOp: wide error_union split (cast operands) at " << loc << "\n";
                        Value tag = ensureU256(rewriter, loc, cast.getOperand(0));
                        Value payload = ensureU256(rewriter, loc, cast.getOperand(1));

                        auto sizeAttr = mlir::IntegerAttr::get(ui64Type, 64);
                        Value sizeConst = rewriter.create<sir::ConstOp>(loc, u256Type, sizeAttr);
                        Value mem = rewriter.create<sir::MallocOp>(loc, ptrType, sizeConst);

                        rewriter.create<sir::StoreOp>(loc, mem, tag);
                        Value offset = rewriter.create<sir::ConstOp>(loc, u256Type, mlir::IntegerAttr::get(ui64Type, 32));
                        Value mem2 = rewriter.create<sir::AddPtrOp>(loc, ptrType, mem, offset);
                        rewriter.create<sir::StoreOp>(loc, mem2, payload);

                        rewriter.create<sir::ReturnOp>(loc, mem, sizeConst);
                        rewriter.eraseOp(op);
                        return success();
                    }
                }
            }
            if (operands.size() == 2)
            {
                llvm::errs() << "[OraToSIR] ConvertReturnOp: wide error_union split (2 operands) at " << loc << "\n";
                Value tag = ensureU256(rewriter, loc, operands[0]);
                Value payload = ensureU256(rewriter, loc, operands[1]);

                auto sizeAttr = mlir::IntegerAttr::get(ui64Type, 64);
                Value sizeConst = rewriter.create<sir::ConstOp>(loc, u256Type, sizeAttr);
                Value mem = rewriter.create<sir::MallocOp>(loc, ptrType, sizeConst);

                rewriter.create<sir::StoreOp>(loc, mem, tag);
                Value offset = rewriter.create<sir::ConstOp>(loc, u256Type, mlir::IntegerAttr::get(ui64Type, 32));
                Value mem2 = rewriter.create<sir::AddPtrOp>(loc, ptrType, mem, offset);
                rewriter.create<sir::StoreOp>(loc, mem2, payload);

                rewriter.create<sir::ReturnOp>(loc, mem, sizeConst);
                rewriter.eraseOp(op);
                return success();
            }

            SmallVector<Value> parts;
            if (failed(materializeWideErrorUnion(rewriter, loc, op.getOperand(0), parts)) || parts.size() != 2)
                return rewriter.notifyMatchFailure(op, "failed to materialize wide error_union return");

            llvm::errs() << "[OraToSIR] ConvertReturnOp: wide error_union split (materialized) at " << loc << "\n";
            Value tag = ensureU256(rewriter, loc, parts[0]);
            Value payload = ensureU256(rewriter, loc, parts[1]);

            auto sizeAttr = mlir::IntegerAttr::get(ui64Type, 64);
            Value sizeConst = rewriter.create<sir::ConstOp>(loc, u256Type, sizeAttr);
            Value mem = rewriter.create<sir::MallocOp>(loc, ptrType, sizeConst);

            rewriter.create<sir::StoreOp>(loc, mem, tag);
            Value offset = rewriter.create<sir::ConstOp>(loc, u256Type, mlir::IntegerAttr::get(ui64Type, 32));
            Value mem2 = rewriter.create<sir::AddPtrOp>(loc, ptrType, mem, offset);
            rewriter.create<sir::StoreOp>(loc, mem2, payload);

            rewriter.create<sir::ReturnOp>(loc, mem, sizeConst);
            rewriter.eraseOp(op);
            return success();
        }
    }

    if (operands.size() == 2)
    {
        llvm::errs() << "[OraToSIR] ConvertReturnOp: split return (2 operands) at " << loc << "\n";
        Value tag = ensureU256(rewriter, loc, operands[0]);
        Value payload = ensureU256(rewriter, loc, operands[1]);

        auto sizeAttr = mlir::IntegerAttr::get(ui64Type, 64);
        Value sizeConst = rewriter.create<sir::ConstOp>(loc, u256Type, sizeAttr);
        Value mem = rewriter.create<sir::MallocOp>(loc, ptrType, sizeConst);

        rewriter.create<sir::StoreOp>(loc, mem, tag);
        Value offset = rewriter.create<sir::ConstOp>(loc, u256Type, mlir::IntegerAttr::get(ui64Type, 32));
        Value mem2 = rewriter.create<sir::AddPtrOp>(loc, ptrType, mem, offset);
        rewriter.create<sir::StoreOp>(loc, mem2, payload);

        rewriter.create<sir::ReturnOp>(loc, mem, sizeConst);
        rewriter.eraseOp(op);
        return success();
    }

    if (is_bytes_return)
    {
        llvm::errs() << "[OraToSIR] ConvertReturnOp: bytes return path at " << loc << "\n";
        // Return ABI for dynamic bytes: ptr + (len + 32)
        if (!llvm::isa<sir::PtrType>(retVal.getType()))
        {
            if (!tc)
                return rewriter.notifyMatchFailure(op, "missing type converter");
            Type convertedType = tc->convertType(retVal.getType());
            if (convertedType && convertedType != retVal.getType())
            {
                retVal = rewriter.create<sir::BitcastOp>(loc, convertedType, retVal);
            }
        }

        Value length = rewriter.create<sir::LoadOp>(loc, u256Type, retVal);
        Value wordSize = rewriter.create<sir::ConstOp>(loc, u256Type, mlir::IntegerAttr::get(ui64Type, 32));
        Value sizeConst = rewriter.create<sir::AddOp>(loc, u256Type, length, wordSize);

        rewriter.create<sir::ReturnOp>(loc, retVal, sizeConst);
        rewriter.eraseOp(op);
        return success();
    }

    // Ensure the value is SIR u256 type (adaptor should have already converted it)
    if (auto intType = llvm::dyn_cast<mlir::IntegerType>(retVal.getType()))
    {
        if (intType.getWidth() == 1)
        {
            retVal = toCondU256(rewriter, loc, retVal);
        }
    }

    if (!llvm::isa<sir::U256Type>(retVal.getType()))
    {
        // Type converter should have handled this, but fallback to bitcast if needed
        if (!tc)
            return rewriter.notifyMatchFailure(op, "missing type converter");
        Type convertedType = tc->convertType(retVal.getType());
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

    // Replace func.return with sir.return %ptr, %len (EVM RETURN op)
    llvm::errs() << "[OraToSIR] ConvertReturnOp: default return path at " << loc << "\n";
    rewriter.create<sir::ReturnOp>(loc, mem, sizeConst);
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
    llvm::errs() << "[OraToSIR] ConvertReturnOp::matchAndRewrite() entry at " << op.getLoc() << "\n";
    llvm::errs() << "[OraToSIR]   Adaptor operand count: " << adaptor.getOperands().size() << "\n";
    for (auto it : llvm::enumerate(adaptor.getOperands()))
        llvm::errs() << "[OraToSIR]   Adaptor operand " << it.index() << " type: " << it.value().getType() << "\n";
    llvm::errs().flush();
    SmallVector<Value> operands;
    operands.append(adaptor.getOperands().begin(), adaptor.getOperands().end());
    auto result = convertOraReturn(op, operands, getTypeConverter(), rewriter);
    llvm::errs() << "[OraToSIR] ConvertReturnOp::matchAndRewrite() result="
                 << (succeeded(result) ? "success" : "failure")
                 << " at " << op.getLoc() << "\n";
    return result;
}

LogicalResult ConvertReturnOp::matchAndRewrite(
    ora::ReturnOp op,
    OneToNOpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    llvm::errs() << "[OraToSIR] ConvertReturnOp::matchAndRewrite(OneToN) entry at " << op.getLoc() << "\n";
    llvm::errs() << "[OraToSIR]   OneToN operand groups: " << adaptor.getOperands().size() << "\n";
    for (auto groupIt : llvm::enumerate(adaptor.getOperands()))
    {
        llvm::errs() << "[OraToSIR]   Group " << groupIt.index() << " size: " << groupIt.value().size() << "\n";
        for (auto valIt : llvm::enumerate(groupIt.value()))
            llvm::errs() << "[OraToSIR]     Group " << groupIt.index() << " operand " << valIt.index() << " type: " << valIt.value().getType() << "\n";
    }
    llvm::errs().flush();
    SmallVector<Value> flatOperands;
    for (ValueRange range : adaptor.getOperands())
        flatOperands.append(range.begin(), range.end());
    auto result = convertOraReturn(op, flatOperands, getTypeConverter(), rewriter);
    llvm::errs() << "[OraToSIR] ConvertReturnOp::matchAndRewrite(OneToN) result="
                 << (succeeded(result) ? "success" : "failure")
                 << " at " << op.getLoc() << "\n";
    return result;
}

LogicalResult ConvertReturnOpRaw::matchAndRewrite(
    ora::ReturnOp op,
    typename ora::ReturnOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    (void)adaptor;
    llvm::errs() << "[OraToSIR] ConvertReturnOpRaw::matchAndRewrite() entry at " << op.getLoc() << "\n";
    llvm::errs() << "[OraToSIR]   Raw operand count: " << op.getNumOperands() << "\n";
    for (auto it : llvm::enumerate(op.getOperands()))
        llvm::errs() << "[OraToSIR]   Raw operand " << it.index() << " type: " << it.value().getType() << "\n";
    llvm::errs().flush();
    if (op.getNumOperands() != 1)
        return rewriter.notifyMatchFailure(op, "raw return only for single operand");
    auto errType = llvm::dyn_cast<ora::ErrorUnionType>(op.getOperand(0).getType());
    if (!errType || isNarrowErrorUnion(errType))
        return rewriter.notifyMatchFailure(op, "raw return only for wide error_union");
    SmallVector<Value> operands;
    operands.append(op.getOperands().begin(), op.getOperands().end());
    return convertOraReturn(op, operands, getTypeConverter(), rewriter);
}

LogicalResult ConvertReturnOpFallback::matchAndRewrite(
    Operation *op,
    ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const
{
    (void)operands;
    auto retOp = llvm::dyn_cast<ora::ReturnOp>(op);
    if (!retOp)
        return failure();
    llvm::errs() << "[OraToSIR] ConvertReturnOpFallback::matchAndRewrite() entry at " << retOp.getLoc() << "\n";
    llvm::errs() << "[OraToSIR]   Fallback operand count: " << retOp.getNumOperands() << "\n";
    for (auto it : llvm::enumerate(retOp.getOperands()))
        llvm::errs() << "[OraToSIR]   Fallback operand " << it.index() << " type: " << it.value().getType() << "\n";
    llvm::errs().flush();
    SmallVector<Value> rawOperands;
    rawOperands.append(retOp.getOperands().begin(), retOp.getOperands().end());
    return convertOraReturn(retOp, rawOperands, typeConverter, rewriter);
}

LogicalResult ConvertReturnOpPre::matchAndRewrite(
    ora::ReturnOp op,
    PatternRewriter &rewriter) const
{
    llvm::errs() << "[OraToSIR] ConvertReturnOpPre::matchAndRewrite() entry at " << op.getLoc() << "\n";
    llvm::errs() << "[OraToSIR]   Pre operand count: " << op.getNumOperands() << "\n";
    for (auto it : llvm::enumerate(op.getOperands()))
        llvm::errs() << "[OraToSIR]   Pre operand " << it.index() << " type: " << it.value().getType() << "\n";
    llvm::errs().flush();

    if (op.getNumOperands() != 1)
        return failure();
    auto errType = llvm::dyn_cast<ora::ErrorUnionType>(op.getOperand(0).getType());
    if (!errType || isNarrowErrorUnion(errType))
        return failure();
    if (op.getOperand(0).getDefiningOp<scf::IfOp>())
        return failure();
    SmallVector<Value> operands;
    operands.append(op.getOperands().begin(), op.getOperands().end());
    return convertOraReturn(op, operands, typeConverter, rewriter);
}

// -----------------------------------------------------------------------------
// Lower ora.while → SIR CFG
// -----------------------------------------------------------------------------
LogicalResult ConvertWhileOp::matchAndRewrite(
    ora::WhileOp op,
    typename ora::WhileOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    auto loc = op.getLoc();

    SmallVector<ora::BreakOp, 4> breaks;
    SmallVector<ora::ContinueOp, 4> continues;
    SmallVector<ora::YieldOp, 4> yields;
    op.getBody().walk([&](ora::BreakOp b) {
        if (b->getParentOfType<ora::WhileOp>() == op)
            breaks.push_back(b);
    });
    op.getBody().walk([&](ora::ContinueOp c) {
        if (c->getParentOfType<ora::WhileOp>() == op)
            continues.push_back(c);
    });
    op.getBody().walk([&](ora::YieldOp y) {
        if (y->getParentOfType<ora::WhileOp>() == op)
            yields.push_back(y);
    });

    Block *parentBlock = op->getBlock();
    Region *parentRegion = parentBlock->getParent();
    auto afterBlock = rewriter.splitBlock(parentBlock, Block::iterator(op));
    auto condBlock = rewriter.createBlock(parentRegion, afterBlock->getIterator());

    Block *bodyBlock = op.getBody().empty() ? nullptr : &op.getBody().front();
    rewriter.inlineRegionBefore(op.getBody(), *parentRegion, afterBlock->getIterator());
    if (!bodyBlock)
        bodyBlock = rewriter.createBlock(parentRegion, afterBlock->getIterator());

    rewriter.setInsertionPointToEnd(parentBlock);
    rewriter.create<sir::BrOp>(loc, ValueRange{}, condBlock);

    rewriter.setInsertionPointToStart(condBlock);
    Value condU256 = toCondU256(rewriter, loc, adaptor.getCondition());
    rewriter.create<sir::CondBrOp>(loc, condU256, ValueRange{}, ValueRange{}, bodyBlock, afterBlock);

    for (auto b : breaks)
    {
        rewriter.setInsertionPoint(b);
        rewriter.create<sir::BrOp>(b.getLoc(), ValueRange{}, afterBlock);
        rewriter.eraseOp(b);
    }
    for (auto c : continues)
    {
        rewriter.setInsertionPoint(c);
        rewriter.create<sir::BrOp>(c.getLoc(), ValueRange{}, condBlock);
        rewriter.eraseOp(c);
    }
    for (auto y : yields)
    {
        if (hasOpsAfterTerminator(y.getOperation()))
            return rewriter.notifyMatchFailure(y, "yield has trailing ops");
        rewriter.setInsertionPoint(y);
        rewriter.create<sir::BrOp>(y.getLoc(), ValueRange{}, condBlock);
        rewriter.eraseOp(y);
    }

    if (bodyBlock->empty() || !bodyBlock->back().hasTrait<mlir::OpTrait::IsTerminator>())
    {
        rewriter.setInsertionPointToEnd(bodyBlock);
        rewriter.create<sir::BrOp>(loc, ValueRange{}, condBlock);
    }

    rewriter.eraseOp(op);
    return success();
}

// -----------------------------------------------------------------------------
// Lower ora.if -> SIR CFG
// -----------------------------------------------------------------------------
LogicalResult ConvertIfOp::matchAndRewrite(
    ora::IfOp op,
    typename ora::IfOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    auto loc = op.getLoc();
    auto *tc = getTypeConverter();
    if (!tc)
        return rewriter.notifyMatchFailure(op, "missing type converter");

    SmallVector<Type> resultTypes;
    for (Type t : op.getResultTypes())
    {
        SmallVector<Type> convertedTypes;
        if (failed(tc->convertType(t, convertedTypes)) || convertedTypes.empty())
            return rewriter.notifyMatchFailure(op, "failed to convert if result type");
        resultTypes.append(convertedTypes.begin(), convertedTypes.end());
    }

    Block *parentBlock = op->getBlock();
    Region *parentRegion = parentBlock->getParent();
    auto mergeBlock = rewriter.splitBlock(parentBlock, Block::iterator(op));
    for (Type t : resultTypes)
        mergeBlock->addArgument(t, loc);

    SmallVector<ora::YieldOp, 4> thenYields;
    SmallVector<ora::YieldOp, 4> elseYields;
    op.getThenRegion().walk([&](ora::YieldOp y) {
        if (y->getParentOfType<ora::IfOp>() == op)
            thenYields.push_back(y);
    });
    op.getElseRegion().walk([&](ora::YieldOp y) {
        if (y->getParentOfType<ora::IfOp>() == op)
            elseYields.push_back(y);
    });

    SmallVector<Block *> thenBlocks;
    SmallVector<Block *> elseBlocks;
    for (Block &b : op.getThenRegion())
        thenBlocks.push_back(&b);
    for (Block &b : op.getElseRegion())
        elseBlocks.push_back(&b);

    Block *thenBlock = thenBlocks.empty() ? nullptr : thenBlocks.front();
    Block *elseBlock = elseBlocks.empty() ? nullptr : elseBlocks.front();
    rewriter.inlineRegionBefore(op.getThenRegion(), *parentRegion, mergeBlock->getIterator());
    rewriter.inlineRegionBefore(op.getElseRegion(), *parentRegion, mergeBlock->getIterator());
    if (!thenBlock)
        thenBlock = rewriter.createBlock(parentRegion, mergeBlock->getIterator());
    if (!elseBlock)
        elseBlock = rewriter.createBlock(parentRegion, mergeBlock->getIterator());

    auto lowerReturnsInBlocks = [&](ArrayRef<Block *> blocks) -> LogicalResult {
        for (Block *block : blocks)
        {
            for (auto &blockOp : llvm::make_early_inc_range(*block))
            {
                auto retOp = llvm::dyn_cast<ora::ReturnOp>(&blockOp);
                if (!retOp)
                    continue;
                SmallVector<Value> rawOperands;
                rawOperands.append(retOp.getOperands().begin(), retOp.getOperands().end());
                if (failed(convertOraReturn(retOp, rawOperands, tc, rewriter)))
                    return rewriter.notifyMatchFailure(retOp, "failed to lower ora.return in if region");
            }
        }
        return success();
    };

    if (failed(lowerReturnsInBlocks(thenBlocks)))
        return failure();
    if (failed(lowerReturnsInBlocks(elseBlocks)))
        return failure();

    auto replaceYield = [&](ArrayRef<ora::YieldOp> yields, Block *block) -> LogicalResult {
        for (auto y : yields)
        {
            if (resultTypes.empty() && y.getNumOperands() != 0)
                return rewriter.notifyMatchFailure(op, "if has yields but no results");
            if (!resultTypes.empty() && y.getNumOperands() != resultTypes.size())
                return rewriter.notifyMatchFailure(op, "if yield arity mismatch");
            if (hasOpsAfterTerminator(y.getOperation()))
                return rewriter.notifyMatchFailure(y, "yield has trailing ops");
            rewriter.setInsertionPoint(y);
            SmallVector<Value> convertedOperands;
            convertedOperands.reserve(y.getNumOperands());
            for (auto [idx, operand] : llvm::enumerate(y.getOperands()))
            {
                Type targetType = mergeBlock->getArgument(idx).getType();
                if (operand.getType() == targetType)
                {
                    convertedOperands.push_back(operand);
                    continue;
                }
                Value converted = tc->materializeTargetConversion(rewriter, loc, targetType, operand);
                if (!converted)
                    return rewriter.notifyMatchFailure(op, "if yield conversion failed");
                convertedOperands.push_back(converted);
            }
            rewriter.replaceOpWithNewOp<sir::BrOp>(y, convertedOperands, mergeBlock);
        }
        if (yields.empty())
        {
            if (!resultTypes.empty())
                return rewriter.notifyMatchFailure(op, "if missing yield for result values");
            if (block->empty() || !block->back().hasTrait<mlir::OpTrait::IsTerminator>())
            {
                rewriter.setInsertionPointToEnd(block);
                rewriter.create<sir::BrOp>(loc, ValueRange{}, mergeBlock);
            }
        }
        return success();
    };

    if (failed(replaceYield(thenYields, thenBlock)))
        return failure();
    if (failed(replaceYield(elseYields, elseBlock)))
        return failure();

    rewriter.setInsertionPointToEnd(parentBlock);
    Value condU256 = toCondU256(rewriter, loc, adaptor.getCondition());
    rewriter.create<sir::CondBrOp>(loc, condU256, ValueRange{}, ValueRange{}, thenBlock, elseBlock);

    rewriter.setInsertionPointToStart(mergeBlock);
    op->replaceAllUsesWith(mergeBlock->getArguments());
    rewriter.eraseOp(op);
    return success();
}

LogicalResult ConvertIsolatedIfOp::matchAndRewrite(
    ora::IsolatedIfOp op,
    typename ora::IsolatedIfOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    auto loc = op.getLoc();
    auto *tc = getTypeConverter();
    if (!tc)
        return rewriter.notifyMatchFailure(op, "missing type converter");

    SmallVector<Type> resultTypes;
    for (Type t : op.getResultTypes())
    {
        SmallVector<Type> convertedTypes;
        if (failed(tc->convertType(t, convertedTypes)) || convertedTypes.empty())
            return rewriter.notifyMatchFailure(op, "failed to convert if result type");
        resultTypes.append(convertedTypes.begin(), convertedTypes.end());
    }

    SmallVector<ora::YieldOp, 4> thenYields;
    SmallVector<ora::YieldOp, 4> elseYields;
    op.getThenRegion().walk([&](ora::YieldOp y) {
        if (y->getParentOfType<ora::IsolatedIfOp>() == op)
            thenYields.push_back(y);
    });
    op.getElseRegion().walk([&](ora::YieldOp y) {
        if (y->getParentOfType<ora::IsolatedIfOp>() == op)
            elseYields.push_back(y);
    });

    Block *parentBlock = op->getBlock();
    Region *parentRegion = parentBlock->getParent();
    auto mergeBlock = rewriter.splitBlock(parentBlock, Block::iterator(op));
    for (Type t : resultTypes)
        mergeBlock->addArgument(t, loc);

    Block *thenBlock = op.getThenRegion().empty() ? nullptr : &op.getThenRegion().front();
    Block *elseBlock = op.getElseRegion().empty() ? nullptr : &op.getElseRegion().front();
    rewriter.inlineRegionBefore(op.getThenRegion(), *parentRegion, mergeBlock->getIterator());
    rewriter.inlineRegionBefore(op.getElseRegion(), *parentRegion, mergeBlock->getIterator());
    if (!thenBlock)
        thenBlock = rewriter.createBlock(parentRegion, mergeBlock->getIterator());
    if (!elseBlock)
        elseBlock = rewriter.createBlock(parentRegion, mergeBlock->getIterator());

    auto replaceYield = [&](ArrayRef<ora::YieldOp> yields, Block *block) -> LogicalResult {
        for (auto y : yields)
        {
            if (resultTypes.empty() && y.getNumOperands() != 0)
                return rewriter.notifyMatchFailure(op, "if has yields but no results");
            if (!resultTypes.empty() && y.getNumOperands() != resultTypes.size())
                return rewriter.notifyMatchFailure(op, "if yield arity mismatch");
            if (hasOpsAfterTerminator(y.getOperation()))
                return rewriter.notifyMatchFailure(y, "yield has trailing ops");
            rewriter.setInsertionPoint(y);
            SmallVector<Value> convertedOperands;
            convertedOperands.reserve(y.getNumOperands());
            for (auto [idx, operand] : llvm::enumerate(y.getOperands()))
            {
                Type targetType = mergeBlock->getArgument(idx).getType();
                if (operand.getType() == targetType)
                {
                    convertedOperands.push_back(operand);
                    continue;
                }
                Value converted = tc->materializeTargetConversion(rewriter, loc, targetType, operand);
                if (!converted)
                    return rewriter.notifyMatchFailure(op, "isolated_if yield conversion failed");
                convertedOperands.push_back(converted);
            }
            rewriter.replaceOpWithNewOp<sir::BrOp>(y, convertedOperands, mergeBlock);
        }
        if (yields.empty())
        {
            if (!resultTypes.empty())
                return rewriter.notifyMatchFailure(op, "if missing yield for result values");
            if (block->empty() || !block->back().hasTrait<mlir::OpTrait::IsTerminator>())
            {
                rewriter.setInsertionPointToEnd(block);
                rewriter.create<sir::BrOp>(loc, ValueRange{}, mergeBlock);
            }
        }
        return success();
    };

    if (failed(replaceYield(thenYields, thenBlock)))
        return failure();
    if (failed(replaceYield(elseYields, elseBlock)))
        return failure();

    rewriter.setInsertionPointToEnd(parentBlock);
    Value condU256 = toCondU256(rewriter, loc, adaptor.getCondition());
    rewriter.create<sir::CondBrOp>(loc, condU256, ValueRange{}, ValueRange{}, thenBlock, elseBlock);
    rewriter.setInsertionPointToStart(mergeBlock);
    op->replaceAllUsesWith(mergeBlock->getArguments());
    rewriter.eraseOp(op);
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

    return rewriter.notifyMatchFailure(op, "break must be lowered by loop conversion");
}

LogicalResult ConvertContinueOp::matchAndRewrite(
    ora::ContinueOp op,
    typename ora::ContinueOp::Adaptor /*adaptor*/,
    ConversionPatternRewriter &rewriter) const
{
    Operation *parent = op->getParentOp();
    if (!parent)
        return rewriter.notifyMatchFailure(op, "continue has no parent op");

    return rewriter.notifyMatchFailure(op, "continue must be lowered by loop conversion");
}

LogicalResult ConvertSwitchExprOp::matchAndRewrite(
    ora::SwitchExprOp op,
    typename ora::SwitchExprOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    auto loc = op.getLoc();
    auto ctx = rewriter.getContext();
    auto *tc = getTypeConverter();
    if (!tc)
        return rewriter.notifyMatchFailure(op, "missing type converter");

    SmallVector<Type> resultTypes;
    for (Type t : op.getResultTypes())
    {
        SmallVector<Type> convertedTypes;
        if (failed(tc->convertType(t, convertedTypes)) || convertedTypes.empty())
            return failure();
        resultTypes.append(convertedTypes.begin(), convertedTypes.end());
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

    Block *parentBlock = op->getBlock();
    Region *parentRegion = parentBlock->getParent();
    auto mergeBlock = rewriter.splitBlock(parentBlock, Block::iterator(op));
    for (Type t : resultTypes)
        mergeBlock->addArgument(t, loc);
    rewriter.setInsertionPointToEnd(parentBlock);
    Value selector = ensureU256(rewriter, loc, adaptor.getValue());

    SmallVector<Block *> caseBlocks(numCases, nullptr);
    SmallVector<SmallVector<ora::YieldOp, 4>, 4> caseYields(numCases);
    for (int64_t i = 0; i < numCases; ++i)
    {
        op.getCases()[i].walk([&](ora::YieldOp y) {
            if (y->getParentOfType<ora::SwitchExprOp>() == op)
                caseYields[i].push_back(y);
        });
    }
    for (int64_t i = 0; i < numCases; ++i)
    {
        Block *caseBlock = op.getCases()[i].empty() ? nullptr : &op.getCases()[i].front();
        rewriter.inlineRegionBefore(op.getCases()[i], *parentRegion, mergeBlock->getIterator());
        if (!caseBlock)
            caseBlock = rewriter.createBlock(parentRegion, mergeBlock->getIterator());
        caseBlocks[i] = caseBlock;

        auto &yields = caseYields[i];
        if (yields.empty())
            return failure();
        for (auto y : yields)
        {
            if (y.getNumOperands() != resultTypes.size())
                return failure();
            if (hasOpsAfterTerminator(y.getOperation()))
                return rewriter.notifyMatchFailure(y, "yield has trailing ops");
            rewriter.setInsertionPoint(y);
            SmallVector<Value> convertedOperands;
            convertedOperands.reserve(y.getNumOperands());
            for (auto [idx, operand] : llvm::enumerate(y.getOperands()))
            {
                Type targetType = mergeBlock->getArgument(idx).getType();
                if (operand.getType() == targetType)
                {
                    convertedOperands.push_back(operand);
                    continue;
                }
                if (!tc)
                    return failure();
                Value converted = tc->materializeTargetConversion(rewriter, loc, targetType, operand);
                if (!converted)
                    return failure();
                convertedOperands.push_back(converted);
            }
            rewriter.replaceOpWithNewOp<sir::BrOp>(y, convertedOperands, mergeBlock);
        }
    }

    auto u256Type = sir::U256Type::get(ctx);
    auto ui64Type = mlir::IntegerType::get(ctx, 64, mlir::IntegerType::Unsigned);
    auto makeConst = [&](int64_t v) -> Value {
        return rewriter.create<sir::ConstOp>(loc, u256Type, mlir::IntegerAttr::get(ui64Type, v));
    };

    Block *defaultBlock = caseBlocks[defaultIdx];
    if (caseIdxs.empty())
    {
        rewriter.setInsertionPointToEnd(parentBlock);
        rewriter.create<sir::BrOp>(loc, ValueRange{}, defaultBlock);
        rewriter.setInsertionPointToStart(mergeBlock);
        op->replaceAllUsesWith(mergeBlock->getArguments());
        rewriter.eraseOp(op);
        return success();
    }

    Block *currentCheck = parentBlock;
    for (size_t i = 0; i < caseIdxs.size(); ++i)
    {
        int64_t caseIdx = caseIdxs[i];
        int64_t kind = 0;
        if (caseKinds && caseIdx < static_cast<int64_t>(caseKinds.size()))
            kind = caseKinds[caseIdx];

        Block *nextCheck = (i + 1 < caseIdxs.size())
                               ? rewriter.createBlock(parentRegion, mergeBlock->getIterator())
                               : defaultBlock;

        rewriter.setInsertionPointToEnd(currentCheck);
        Value cond;
        if (kind == 0 && caseValues && caseIdx < static_cast<int64_t>(caseValues.size()))
        {
            Value cst = makeConst(caseValues[caseIdx]);
            cond = rewriter.create<sir::EqOp>(loc, u256Type, selector, cst);
        }
        else if (kind == 1 && rangeStarts && rangeEnds &&
                 caseIdx < static_cast<int64_t>(rangeStarts.size()) &&
                 caseIdx < static_cast<int64_t>(rangeEnds.size()))
        {
            Value start = makeConst(rangeStarts[caseIdx]);
            Value end = makeConst(rangeEnds[caseIdx]);
            Value lt = rewriter.create<sir::LtOp>(loc, u256Type, selector, start);
            Value gt = rewriter.create<sir::GtOp>(loc, u256Type, selector, end);
            Value ge = rewriter.create<sir::IsZeroOp>(loc, u256Type, lt);
            Value le = rewriter.create<sir::IsZeroOp>(loc, u256Type, gt);
            cond = rewriter.create<sir::AndOp>(loc, u256Type, ge, le);
        }
        else
        {
            return failure();
        }

        rewriter.create<sir::CondBrOp>(loc, cond, ValueRange{}, ValueRange{}, caseBlocks[caseIdx], nextCheck);
        currentCheck = nextCheck;
    }

    rewriter.setInsertionPointToStart(mergeBlock);
    op->replaceAllUsesWith(mergeBlock->getArguments());
    rewriter.eraseOp(op);
    return success();
}

LogicalResult ConvertSwitchOp::matchAndRewrite(
    ora::SwitchOp op,
    typename ora::SwitchOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    auto loc = op.getLoc();
    auto ctx = rewriter.getContext();
    auto *tc = getTypeConverter();

    if (op.getNumResults() > 0)
    {
        if (!tc)
            return rewriter.notifyMatchFailure(op, "missing type converter");

        SmallVector<Type> resultTypes;
        for (Type t : op.getResultTypes())
        {
            SmallVector<Type> convertedTypes;
            if (failed(tc->convertType(t, convertedTypes)) || convertedTypes.empty())
                return failure();
            resultTypes.append(convertedTypes.begin(), convertedTypes.end());
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

        Block *parentBlock = op->getBlock();
        Region *parentRegion = parentBlock->getParent();
        auto mergeBlock = rewriter.splitBlock(parentBlock, Block::iterator(op));
        for (Type t : resultTypes)
            mergeBlock->addArgument(t, loc);
        rewriter.setInsertionPointToEnd(parentBlock);
        Value selector = ensureU256(rewriter, loc, adaptor.getValue());

        SmallVector<Block *> caseBlocks(numCases, nullptr);
        SmallVector<SmallVector<ora::YieldOp, 4>, 4> caseYields(numCases);
        for (int64_t i = 0; i < numCases; ++i)
        {
            op.getCases()[i].walk([&](ora::YieldOp y) {
                if (y->getParentOfType<ora::SwitchOp>() == op)
                    caseYields[i].push_back(y);
            });
        }
        for (int64_t i = 0; i < numCases; ++i)
        {
            Block *caseBlock = op.getCases()[i].empty() ? nullptr : &op.getCases()[i].front();
            rewriter.inlineRegionBefore(op.getCases()[i], *parentRegion, mergeBlock->getIterator());
            if (!caseBlock)
                caseBlock = rewriter.createBlock(parentRegion, mergeBlock->getIterator());
            caseBlocks[i] = caseBlock;

            auto &yields = caseYields[i];
            if (yields.empty())
                return failure();
            for (auto y : yields)
            {
                if (y.getNumOperands() != resultTypes.size())
                    return failure();
                if (hasOpsAfterTerminator(y.getOperation()))
                    return rewriter.notifyMatchFailure(y, "yield has trailing ops");
                rewriter.setInsertionPoint(y);
                SmallVector<Value> convertedOperands;
                convertedOperands.reserve(y.getNumOperands());
                for (auto [idx, operand] : llvm::enumerate(y.getOperands()))
                {
                    Type targetType = mergeBlock->getArgument(idx).getType();
                    if (operand.getType() == targetType)
                    {
                        convertedOperands.push_back(operand);
                        continue;
                    }
                    Value converted = tc->materializeTargetConversion(rewriter, loc, targetType, operand);
                    if (!converted)
                        return failure();
                    convertedOperands.push_back(converted);
                }
                rewriter.replaceOpWithNewOp<sir::BrOp>(y, convertedOperands, mergeBlock);
            }
        }

        auto u256Type = sir::U256Type::get(ctx);
        auto ui64Type = mlir::IntegerType::get(ctx, 64, mlir::IntegerType::Unsigned);
        auto makeConst = [&](int64_t v) -> Value {
            return rewriter.create<sir::ConstOp>(loc, u256Type, mlir::IntegerAttr::get(ui64Type, v));
        };

        Block *defaultBlock = caseBlocks[defaultIdx];
        if (caseIdxs.empty())
        {
            rewriter.setInsertionPointToEnd(parentBlock);
            rewriter.create<sir::BrOp>(loc, ValueRange{}, defaultBlock);
            if (static_cast<unsigned>(mergeBlock->getNumArguments()) != op.getNumResults())
                return rewriter.notifyMatchFailure(op, "switch result/merge mismatch");
            rewriter.setInsertionPointToStart(mergeBlock);
            op->replaceAllUsesWith(mergeBlock->getArguments());
            rewriter.eraseOp(op);
            return success();
        }

        Block *currentCheck = parentBlock;
        for (size_t i = 0; i < caseIdxs.size(); ++i)
        {
            int64_t caseIdx = caseIdxs[i];
            int64_t kind = 0;
            if (caseKinds && caseIdx < static_cast<int64_t>(caseKinds.size()))
                kind = caseKinds[caseIdx];

            Block *nextCheck = (i + 1 < caseIdxs.size())
                                   ? rewriter.createBlock(parentRegion, mergeBlock->getIterator())
                                   : defaultBlock;

            rewriter.setInsertionPointToEnd(currentCheck);
            Value cond;
            if (kind == 0 && caseValues && caseIdx < static_cast<int64_t>(caseValues.size()))
            {
                Value cst = makeConst(caseValues[caseIdx]);
                cond = rewriter.create<sir::EqOp>(loc, u256Type, selector, cst);
            }
            else if (kind == 1 && rangeStarts && rangeEnds &&
                     caseIdx < static_cast<int64_t>(rangeStarts.size()) &&
                     caseIdx < static_cast<int64_t>(rangeEnds.size()))
            {
                Value start = makeConst(rangeStarts[caseIdx]);
                Value end = makeConst(rangeEnds[caseIdx]);
                Value lt = rewriter.create<sir::LtOp>(loc, u256Type, selector, start);
                Value gt = rewriter.create<sir::GtOp>(loc, u256Type, selector, end);
                Value ge = rewriter.create<sir::IsZeroOp>(loc, u256Type, lt);
                Value le = rewriter.create<sir::IsZeroOp>(loc, u256Type, gt);
                cond = rewriter.create<sir::AndOp>(loc, u256Type, ge, le);
            }
            else
            {
                return failure();
            }

            rewriter.create<sir::CondBrOp>(loc, cond, ValueRange{}, ValueRange{}, caseBlocks[caseIdx], nextCheck);
            currentCheck = nextCheck;
        }

        if (static_cast<unsigned>(mergeBlock->getNumArguments()) != op.getNumResults())
            return rewriter.notifyMatchFailure(op, "switch result/merge mismatch");
        rewriter.setInsertionPointToStart(mergeBlock);
        op->replaceAllUsesWith(mergeBlock->getArguments());
        rewriter.eraseOp(op);
        return success();
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

    Block *parentBlock = op->getBlock();
    Region *parentRegion = parentBlock->getParent();
    auto afterBlock = rewriter.splitBlock(parentBlock, Block::iterator(op));
    rewriter.setInsertionPointToEnd(parentBlock);
    Value selector = ensureU256(rewriter, loc, adaptor.getValue());

    SmallVector<Block *> caseBlocks(numCases, nullptr);
    SmallVector<SmallVector<ora::YieldOp, 4>, 4> caseYields(numCases);
    for (int64_t i = 0; i < numCases; ++i)
    {
        op.getCases()[i].walk([&](ora::YieldOp y) {
            if (y->getParentOfType<ora::SwitchOp>() == op)
                caseYields[i].push_back(y);
        });
    }
    for (int64_t i = 0; i < numCases; ++i)
    {
        Block *caseBlock = op.getCases()[i].empty() ? nullptr : &op.getCases()[i].front();
        rewriter.inlineRegionBefore(op.getCases()[i], *parentRegion, afterBlock->getIterator());
        if (!caseBlock)
            caseBlock = rewriter.createBlock(parentRegion, afterBlock->getIterator());
        caseBlocks[i] = caseBlock;

        auto &yields = caseYields[i];
        for (auto y : yields)
        {
            if (hasOpsAfterTerminator(y.getOperation()))
                return rewriter.notifyMatchFailure(y, "yield has trailing ops");
            rewriter.setInsertionPoint(y);
            rewriter.replaceOpWithNewOp<sir::BrOp>(y, ValueRange{}, afterBlock);
        }
        if (caseBlock->empty() || !caseBlock->back().hasTrait<mlir::OpTrait::IsTerminator>())
        {
            rewriter.setInsertionPointToEnd(caseBlock);
            rewriter.create<sir::BrOp>(loc, ValueRange{}, afterBlock);
        }
    }

    auto u256Type = sir::U256Type::get(ctx);
    auto ui64Type = mlir::IntegerType::get(ctx, 64, mlir::IntegerType::Unsigned);
    auto makeConst = [&](int64_t v) -> Value {
        return rewriter.create<sir::ConstOp>(loc, u256Type, mlir::IntegerAttr::get(ui64Type, v));
    };

    Block *defaultBlock = caseBlocks[defaultIdx];
    if (caseIdxs.empty())
    {
        rewriter.setInsertionPointToEnd(parentBlock);
        rewriter.create<sir::BrOp>(loc, ValueRange{}, defaultBlock);
        rewriter.eraseOp(op);
        return success();
    }

    Block *currentCheck = parentBlock;
    for (size_t i = 0; i < caseIdxs.size(); ++i)
    {
        int64_t caseIdx = caseIdxs[i];
        int64_t kind = 0;
        if (caseKinds && caseIdx < static_cast<int64_t>(caseKinds.size()))
            kind = caseKinds[caseIdx];

        Block *nextCheck = (i + 1 < caseIdxs.size())
                               ? rewriter.createBlock(parentRegion, afterBlock->getIterator())
                               : defaultBlock;

        rewriter.setInsertionPointToEnd(currentCheck);
        Value cond;
        if (kind == 0 && caseValues && caseIdx < static_cast<int64_t>(caseValues.size()))
        {
            Value cst = makeConst(caseValues[caseIdx]);
            cond = rewriter.create<sir::EqOp>(loc, u256Type, selector, cst);
        }
        else if (kind == 1 && rangeStarts && rangeEnds &&
                 caseIdx < static_cast<int64_t>(rangeStarts.size()) &&
                 caseIdx < static_cast<int64_t>(rangeEnds.size()))
        {
            Value start = makeConst(rangeStarts[caseIdx]);
            Value end = makeConst(rangeEnds[caseIdx]);
            Value lt = rewriter.create<sir::LtOp>(loc, u256Type, selector, start);
            Value gt = rewriter.create<sir::GtOp>(loc, u256Type, selector, end);
            Value ge = rewriter.create<sir::IsZeroOp>(loc, u256Type, lt);
            Value le = rewriter.create<sir::IsZeroOp>(loc, u256Type, gt);
            cond = rewriter.create<sir::AndOp>(loc, u256Type, ge, le);
        }
        else
        {
            return failure();
        }

        rewriter.create<sir::CondBrOp>(loc, cond, ValueRange{}, ValueRange{}, caseBlocks[caseIdx], nextCheck);
        currentCheck = nextCheck;
    }

    rewriter.eraseOp(op);
    return success();
}

// -----------------------------------------------------------------------------
// Lower cf.br -> sir.br
// -----------------------------------------------------------------------------
LogicalResult ConvertCfBrOp::matchAndRewrite(
    mlir::cf::BranchOp op,
    typename mlir::cf::BranchOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    rewriter.create<sir::BrOp>(op.getLoc(), adaptor.getDestOperands(), op.getDest());
    rewriter.eraseOp(op);
    return success();
}

// -----------------------------------------------------------------------------
// Lower cf.cond_br -> sir.cond_br
// -----------------------------------------------------------------------------
LogicalResult ConvertCfCondBrOp::matchAndRewrite(
    mlir::cf::CondBranchOp op,
    typename mlir::cf::CondBranchOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    auto loc = op.getLoc();
    Value cond = toCondU256(rewriter, loc, adaptor.getCondition());
    rewriter.create<sir::CondBrOp>(
        loc,
        cond,
        adaptor.getTrueDestOperands(),
        adaptor.getFalseDestOperands(),
        op.getTrueDest(),
        op.getFalseDest());
    rewriter.eraseOp(op);
    return success();
}

// -----------------------------------------------------------------------------
// Lower cf.assert -> sir.cond_br + sir.invalid
// -----------------------------------------------------------------------------
LogicalResult ConvertCfAssertOp::matchAndRewrite(
    mlir::cf::AssertOp op,
    typename mlir::cf::AssertOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    auto loc = op.getLoc();
    Block *parentBlock = op->getBlock();
    Region *parentRegion = parentBlock->getParent();

    auto afterBlock = rewriter.splitBlock(parentBlock, Block::iterator(op));
    auto failBlock = rewriter.createBlock(parentRegion, afterBlock->getIterator());

    rewriter.setInsertionPointToStart(failBlock);
    rewriter.create<sir::InvalidOp>(loc);

    rewriter.setInsertionPointToEnd(parentBlock);
    Value cond = toCondU256(rewriter, loc, adaptor.getArg());
    rewriter.create<sir::CondBrOp>(
        loc,
        cond,
        ValueRange{},
        ValueRange{},
        afterBlock,
        failBlock);
    rewriter.eraseOp(op);
    return success();
}

// -----------------------------------------------------------------------------
// Lower scf.if -> SIR CFG
// -----------------------------------------------------------------------------
LogicalResult ConvertScfIfOp::matchAndRewrite(
    mlir::scf::IfOp op,
    typename mlir::scf::IfOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    auto loc = op.getLoc();
    llvm::errs() << "[OraToSIR] ConvertScfIfOp::matchAndRewrite() called at " << loc << "\n";
    auto *tc = getTypeConverter();
    if (!tc)
        return rewriter.notifyMatchFailure(op, "missing type converter");

    SmallVector<Type> resultTypes;
    SmallVector<SmallVector<Type>> resultTypeGroups;
    for (Type t : op.getResultTypes())
    {
        SmallVector<Type> convertedTypes;
        if (failed(tc->convertType(t, convertedTypes)) || convertedTypes.empty())
        {
            if (auto errType = llvm::dyn_cast<ora::ErrorUnionType>(t))
            {
                auto *ctx = rewriter.getContext();
                auto u256 = sir::U256Type::get(ctx);
                if (isNarrowErrorUnion(errType))
                {
                    convertedTypes.push_back(u256);
                }
                else
                {
                    convertedTypes.push_back(u256);
                    convertedTypes.push_back(u256);
                }
            }
            if (convertedTypes.empty())
            {
                llvm::errs() << "[OraToSIR] ConvertScfIfOp: failed to convert result type "
                             << t << " at " << loc << "\n";
                return rewriter.notifyMatchFailure(op, "failed to convert scf.if result type");
            }
        }
        resultTypeGroups.push_back(convertedTypes);
        resultTypes.append(convertedTypes.begin(), convertedTypes.end());
    }
    llvm::errs() << "[OraToSIR] ConvertScfIfOp: results=" << op.getNumResults()
                 << " groups=" << resultTypeGroups.size() << " flat=" << resultTypes.size()
                 << " at " << loc << "\n";

    SmallVector<mlir::scf::YieldOp, 4> thenYields;
    SmallVector<mlir::scf::YieldOp, 4> elseYields;
    op.getThenRegion().walk([&](mlir::scf::YieldOp y) { thenYields.push_back(y); });
    op.getElseRegion().walk([&](mlir::scf::YieldOp y) { elseYields.push_back(y); });
    llvm::errs() << "[OraToSIR] ConvertScfIfOp: thenYields=" << thenYields.size()
                 << " elseYields=" << elseYields.size() << " at " << loc << "\n";

    Block *parentBlock = op->getBlock();
    Region *parentRegion = parentBlock->getParent();
    auto mergeBlock = rewriter.splitBlock(parentBlock, Block::iterator(op));
    for (Type t : resultTypes)
        mergeBlock->addArgument(t, loc);

    Block *thenBlock = op.getThenRegion().empty() ? nullptr : &op.getThenRegion().front();
    Block *elseBlock = op.getElseRegion().empty() ? nullptr : &op.getElseRegion().front();
    rewriter.inlineRegionBefore(op.getThenRegion(), *parentRegion, mergeBlock->getIterator());
    rewriter.inlineRegionBefore(op.getElseRegion(), *parentRegion, mergeBlock->getIterator());
    if (!thenBlock)
        thenBlock = rewriter.createBlock(parentRegion, mergeBlock->getIterator());
    if (!elseBlock)
        elseBlock = rewriter.createBlock(parentRegion, mergeBlock->getIterator());

    auto replaceYield = [&](ArrayRef<mlir::scf::YieldOp> yields) -> LogicalResult {
        for (auto y : yields)
        {
            if (y.getNumOperands() != resultTypeGroups.size())
            {
                llvm::errs() << "[OraToSIR] ConvertScfIfOp: yield/result count mismatch "
                             << y.getNumOperands() << " vs " << resultTypeGroups.size()
                             << " at " << y.getLoc() << "\n";
                return rewriter.notifyMatchFailure(y, "scf.if yield/result count mismatch");
            }
            if (hasOpsAfterTerminator(y.getOperation()))
                return rewriter.notifyMatchFailure(y, "yield has trailing ops");
            rewriter.setInsertionPoint(y);
            SmallVector<Value> convertedOperands;
            convertedOperands.reserve(resultTypes.size());
            for (auto it : llvm::enumerate(y.getOperands()))
            {
                unsigned idx = it.index();
                Value operand = it.value();
                auto &group = resultTypeGroups[idx];
                llvm::errs() << "[OraToSIR] ConvertScfIfOp: yield operand[" << idx
                             << "] type=" << operand.getType()
                             << " groupSize=" << group.size() << " at " << y.getLoc() << "\n";
                if (group.size() == 2 && llvm::isa<ora::ErrorUnionType>(operand.getType()))
                {
                    if (failed(materializeWideErrorUnion(rewriter, loc, operand, convertedOperands)))
                    {
                        llvm::errs() << "[OraToSIR] ConvertScfIfOp: failed wide error_union yield at " << y.getLoc() << "\n";
                        return rewriter.notifyMatchFailure(y, "failed to materialize wide error_union yield");
                    }
                    continue;
                }
                if (group.size() == 2)
                {
                    llvm::errs() << "[OraToSIR] ConvertScfIfOp: unexpected 1->2 conversion at " << y.getLoc() << "\n";
                    return rewriter.notifyMatchFailure(y, "unexpected 1->2 conversion for scf.if yield");
                }

                for (Type targetType : group)
                {
                    if (operand.getType() == targetType)
                    {
                        convertedOperands.push_back(operand);
                        continue;
                    }
                    if (!tc)
                        return rewriter.notifyMatchFailure(y, "missing type converter");
                    Value converted = tc->materializeTargetConversion(rewriter, loc, targetType, operand);
                    if (!converted)
                    {
                        llvm::errs() << "[OraToSIR] ConvertScfIfOp: materializeTargetConversion failed "
                                     << operand.getType() << " -> " << targetType
                                     << " at " << y.getLoc() << "\n";
                        return rewriter.notifyMatchFailure(y, "failed to materialize scf.if yield operand");
                    }
                    convertedOperands.push_back(converted);
                }
            }
            rewriter.replaceOpWithNewOp<sir::BrOp>(y, convertedOperands, mergeBlock);
        }
        return success();
    };

    if (failed(replaceYield(thenYields)))
        return failure();
    if (failed(replaceYield(elseYields)))
        return failure();

    if (thenBlock->empty() || !thenBlock->back().hasTrait<mlir::OpTrait::IsTerminator>())
    {
        rewriter.setInsertionPointToEnd(thenBlock);
        rewriter.create<sir::BrOp>(loc, ValueRange{}, mergeBlock);
    }
    if (elseBlock->empty() || !elseBlock->back().hasTrait<mlir::OpTrait::IsTerminator>())
    {
        rewriter.setInsertionPointToEnd(elseBlock);
        rewriter.create<sir::BrOp>(loc, ValueRange{}, mergeBlock);
    }

    rewriter.setInsertionPointToEnd(parentBlock);
    Value condU256 = toCondU256(rewriter, loc, adaptor.getCondition());
    rewriter.create<sir::CondBrOp>(loc, condU256, ValueRange{}, ValueRange{}, thenBlock, elseBlock);

    if (resultTypes.empty())
    {
        rewriter.eraseOp(op);
    }
    else
    {
        if (static_cast<unsigned>(mergeBlock->getNumArguments()) != resultTypes.size())
        {
            llvm::errs() << "[OraToSIR] ConvertScfIfOp: merge arg mismatch "
                         << mergeBlock->getNumArguments() << " vs " << resultTypes.size()
                         << " at " << loc << "\n";
            return rewriter.notifyMatchFailure(op, "if result/merge mismatch");
        }
        rewriter.setInsertionPointToStart(mergeBlock);
        SmallVector<SmallVector<Value>> replacementGroups;
        replacementGroups.reserve(resultTypeGroups.size());
        auto args = mergeBlock->getArguments();
        unsigned offset = 0;
        for (auto &group : resultTypeGroups)
        {
            SmallVector<Value> groupValues;
            groupValues.reserve(group.size());
            for (unsigned i = 0; i < group.size(); ++i)
                groupValues.push_back(args[offset + i]);
            replacementGroups.push_back(std::move(groupValues));
            offset += group.size();
        }
        rewriter.replaceOpWithMultiple(op, replacementGroups);
    }

    for (auto &blockOp : llvm::make_early_inc_range(*mergeBlock))
    {
        auto retOp = llvm::dyn_cast<ora::ReturnOp>(&blockOp);
        if (!retOp)
            continue;

        SmallVector<Value> rawOperands;
        if (retOp.getNumOperands() == 1 &&
            resultTypeGroups.size() == 1 &&
            resultTypeGroups[0].size() == 2)
        {
            rawOperands.push_back(mergeBlock->getArgument(0));
            rawOperands.push_back(mergeBlock->getArgument(1));
        }
        else
        {
            rawOperands.append(retOp.getOperands().begin(), retOp.getOperands().end());
        }

        if (failed(convertOraReturn(retOp, rawOperands, tc, rewriter)))
            return rewriter.notifyMatchFailure(retOp, "failed to lower ora.return in scf.if merge block");
    }

    return success();
}

// -----------------------------------------------------------------------------
// Lower scf.for -> SIR CFG (no iter_args)
// -----------------------------------------------------------------------------
LogicalResult ConvertScfForOp::matchAndRewrite(
    mlir::scf::ForOp op,
    typename mlir::scf::ForOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    if (!op.getInitArgs().empty())
        return rewriter.notifyMatchFailure(op, "scf.for iter_args not supported");

    auto loc = op.getLoc();
    Block *parentBlock = op->getBlock();
    Region *parentRegion = parentBlock->getParent();

    auto afterBlock = rewriter.splitBlock(parentBlock, Block::iterator(op));
    auto condBlock = rewriter.createBlock(parentRegion, afterBlock->getIterator(), {rewriter.getIndexType()}, {loc});

    Region &bodyRegion = op.getRegion();
    SmallVector<mlir::scf::YieldOp, 4> yields;
    SmallVector<ora::BreakOp, 4> breaks;
    SmallVector<ora::ContinueOp, 4> continues;
    op.getRegion().walk([&](mlir::scf::YieldOp y) {
        if (y->getParentOfType<mlir::scf::ForOp>() == op)
            yields.push_back(y);
    });
    op.getRegion().walk([&](ora::BreakOp br) {
        if (br->getParentOfType<mlir::scf::ForOp>() == op)
            breaks.push_back(br);
    });
    op.getRegion().walk([&](ora::ContinueOp cont) {
        if (cont->getParentOfType<mlir::scf::ForOp>() == op)
            continues.push_back(cont);
    });

    SmallVector<Block *, 4> movedBlocks;
    for (Block &b : bodyRegion)
        movedBlocks.push_back(&b);
    parentRegion->getBlocks().splice(afterBlock->getIterator(), bodyRegion.getBlocks());
    Block *bodyBlock = movedBlocks.empty() ? nullptr : movedBlocks.front();
    if (!bodyBlock)
        bodyBlock = rewriter.createBlock(parentRegion, afterBlock->getIterator(), {rewriter.getIndexType()}, {loc});

    Value lb = adaptor.getLowerBound();
    Value ub = adaptor.getUpperBound();
    Value step = adaptor.getStep();

    lb = toIndex(rewriter, loc, lb);
    ub = toIndex(rewriter, loc, ub);
    step = toIndex(rewriter, loc, step);

    rewriter.setInsertionPointToEnd(parentBlock);
    rewriter.create<sir::BrOp>(loc, ValueRange{lb}, condBlock);

    rewriter.setInsertionPointToStart(condBlock);
    Value iv = condBlock->getArgument(0);
    Value ivU256 = ensureU256(rewriter, loc, iv);
    Value ubU256 = ensureU256(rewriter, loc, ub);
    Value cond = rewriter.create<sir::LtOp>(loc, sir::U256Type::get(rewriter.getContext()), ivU256, ubU256);
    rewriter.create<sir::CondBrOp>(loc, cond, ValueRange{iv}, ValueRange{}, bodyBlock, afterBlock);

    if (yields.empty())
    {
        return rewriter.notifyMatchFailure(op, "scf.for body missing scf.yield");
    }
    for (auto y : yields)
    {
        Block *yBlock = y->getBlock();
        bool hasLoopControl = false;
        for (auto br : yBlock->getOps<ora::BreakOp>())
        {
            (void)br;
            hasLoopControl = true;
            break;
        }
        if (!hasLoopControl)
        {
            for (auto cont : yBlock->getOps<ora::ContinueOp>())
            {
                (void)cont;
                hasLoopControl = true;
                break;
            }
        }
        if (hasLoopControl)
        {
            rewriter.eraseOp(y);
            continue;
        }
        if (hasOpsAfterTerminator(y.getOperation()))
            return rewriter.notifyMatchFailure(y, "yield has trailing ops");
        rewriter.setInsertionPoint(y);
        Value bodyIv = bodyBlock->getArgument(0);
        Value nextU256 = rewriter.create<sir::AddOp>(
            loc,
            sir::U256Type::get(rewriter.getContext()),
            ensureU256(rewriter, loc, bodyIv),
            ensureU256(rewriter, loc, step));
        Value next = toIndex(rewriter, loc, nextU256);
        rewriter.replaceOpWithNewOp<sir::BrOp>(y, ValueRange{next}, condBlock);
    }

    for (auto br : breaks)
    {
        Block *brBlock = br->getBlock();
        rewriter.setInsertionPointToEnd(brBlock);
        rewriter.create<sir::BrOp>(br.getLoc(), ValueRange{}, afterBlock);
        rewriter.eraseOp(br);
    }
    for (auto cont : continues)
    {
        Block *contBlock = cont->getBlock();
        rewriter.setInsertionPointToEnd(contBlock);
        Value bodyIv = bodyBlock->getArgument(0);
        Value nextU256 = rewriter.create<sir::AddOp>(
            loc,
            sir::U256Type::get(rewriter.getContext()),
            ensureU256(rewriter, loc, bodyIv),
            ensureU256(rewriter, loc, step));
        Value next = toIndex(rewriter, loc, nextU256);
        rewriter.create<sir::BrOp>(cont.getLoc(), ValueRange{next}, condBlock);
        rewriter.eraseOp(cont);
    }

    for (Block *b : movedBlocks)
    {
        if (b->empty() || !b->back().hasTrait<mlir::OpTrait::IsTerminator>())
        {
            rewriter.setInsertionPointToEnd(b);
            Value nextU256 = rewriter.create<sir::AddOp>(
                loc,
                sir::U256Type::get(rewriter.getContext()),
                ensureU256(rewriter, loc, b->getArgument(0)),
                ensureU256(rewriter, loc, step));
            Value next = toIndex(rewriter, loc, nextU256);
            rewriter.create<sir::BrOp>(loc, ValueRange{next}, condBlock);
        }
    }

    rewriter.eraseOp(op);
    return success();
}

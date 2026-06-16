#include "patterns/Arithmetic.h"
#include "patterns/EVMConstants.h"
#include "patterns/LoweringHelpers.h"
#include "patterns/Naming.h"
#include "patterns/StorageLayout.h"
#include "OraMaterializationKinds.h"
#include "OraToSIRTypeConverter.h"
#include "OraDebug.h"

#include "OraDialect.h"
#include "SIR/SIRDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/APInt.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "ora-to-sir"

using namespace mlir;
using namespace ora;
using mlir::ora::lowering::coerceToU256;
using mlir::ora::lowering::constU256;
using mlir::ora::lowering::emitSelectorRevert;
using mlir::ora::lowering::parseHexSelector;
using mlir::ora::lowering::kStorageMemRefViewKind;

// Debug logging macro
#define DBG(msg) ORA_DEBUG_PREFIX("OraToSIR", msg)

static SIRNamingHelper arithmeticNamingHelper;

static Value getStorageViewRootSlot(Value value)
{
    if (!value)
        return Value();

    if (auto cast = value.getDefiningOp<mlir::UnrealizedConversionCastOp>())
    {
        if (cast.getNumOperands() == 1)
        {
            Value operand = cast.getOperand(0);
            auto viewKind = cast->getAttrOfType<StringAttr>(kOraMaterializationKindAttr);
            if (viewKind && viewKind.getValue() == kStorageMemRefViewKind &&
                llvm::isa<sir::U256Type>(operand.getType()))
                return operand;
            return getStorageViewRootSlot(operand);
        }
    }

    if (auto bitcast = value.getDefiningOp<sir::BitcastOp>())
    {
        Value operand = bitcast.getInput();
        auto viewKind = bitcast->getAttrOfType<StringAttr>(kOraMaterializationKindAttr);
        if (viewKind && viewKind.getValue() == kStorageMemRefViewKind &&
            llvm::isa<sir::U256Type>(operand.getType()))
            return operand;
        return getStorageViewRootSlot(operand);
    }

    return Value();
}

static SIRNamingHelper &getNamingHelper(Operation *op)
{
    return arithmeticNamingHelper;
}

static bool isU256IntegerCarrierType(Type type)
{
    if (llvm::isa<sir::U256Type>(type))
        return true;
    if (auto intType = llvm::dyn_cast<mlir::IntegerType>(type))
        return intType.getWidth() <= 256;
    return false;
}

static unsigned integerCarrierSourceWidth(Type type)
{
    if (auto intType = llvm::dyn_cast<mlir::IntegerType>(type))
        return intType.getWidth();
    if (auto oraIntType = llvm::dyn_cast<ora::IntegerType>(type))
        return oraIntType.getWidth();
    return 256;
}

static Value signExtendToU256(ConversionPatternRewriter &rewriter, Location loc, Value value, Type sourceType)
{
    auto u256Type = sir::U256Type::get(rewriter.getContext());
    Value input = coerceToU256(rewriter, loc, value);
    unsigned sourceWidth = integerCarrierSourceWidth(sourceType);
    if (sourceWidth >= 256)
        return input;

    llvm::APInt shiftAmount(256, 256 - sourceWidth);
    Value shift = constU256(rewriter, loc, shiftAmount);
    Value shiftedLeft = rewriter.create<sir::ShlOp>(loc, u256Type, shift, input).getResult();
    return rewriter.create<sir::SarOp>(loc, u256Type, shift, shiftedLeft).getResult();
}

template <typename SourceOp, typename SirOp, typename Adaptor>
static LogicalResult lowerBinaryOp(SourceOp op,
                                   Adaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   const TypeConverter *typeConverter,
                                   StringRef opName)
{
    auto loc = op.getLoc();
    auto u256Type = sir::U256Type::get(op.getContext());

    if (!typeConverter)
        return rewriter.notifyMatchFailure(op, "missing type converter");
    auto resultType = typeConverter->convertType(op.getResult().getType());
    if (!resultType)
        return rewriter.notifyMatchFailure(op, ("unable to convert " + opName + " result type").str());

    Value lhs = coerceToU256(rewriter, loc, adaptor.getLhs());
    Value rhs = coerceToU256(rewriter, loc, adaptor.getRhs());
    Value result = rewriter.create<SirOp>(loc, u256Type, lhs, rhs).getResult();
    if (!isU256IntegerCarrierType(resultType))
        result = rewriter.create<sir::BitcastOp>(loc, resultType, result).getResult();

    rewriter.replaceOp(op, result);
    return success();
}

template <typename SourceOp, typename SirOp, typename Adaptor>
static LogicalResult lowerSignedBinaryOp(SourceOp op,
                                         Adaptor adaptor,
                                         ConversionPatternRewriter &rewriter,
                                         const TypeConverter *typeConverter,
                                         StringRef opName)
{
    auto loc = op.getLoc();
    auto u256Type = sir::U256Type::get(op.getContext());

    if (!typeConverter)
        return rewriter.notifyMatchFailure(op, "missing type converter");
    auto resultType = typeConverter->convertType(op.getResult().getType());
    if (!resultType)
        return rewriter.notifyMatchFailure(op, ("unable to convert " + opName + " result type").str());

    Value lhs = signExtendToU256(rewriter, loc, adaptor.getLhs(), op.getLhs().getType());
    Value rhs = signExtendToU256(rewriter, loc, adaptor.getRhs(), op.getRhs().getType());
    Value result = rewriter.create<SirOp>(loc, u256Type, lhs, rhs).getResult();
    if (!isU256IntegerCarrierType(resultType))
        result = rewriter.create<sir::BitcastOp>(loc, resultType, result).getResult();

    rewriter.replaceOp(op, result);
    return success();
}

template <typename SourceOp, typename SirOp, typename Adaptor>
static LogicalResult lowerShiftOp(SourceOp op,
                                  Adaptor adaptor,
                                  ConversionPatternRewriter &rewriter,
                                  const TypeConverter *typeConverter,
                                  StringRef opName)
{
    auto loc = op.getLoc();
    Value shift = coerceToU256(rewriter, loc, adaptor.getRhs());
    Value value = coerceToU256(rewriter, loc, adaptor.getLhs());
    auto u256Type = sir::U256Type::get(op.getContext());
    Value shifted = rewriter.create<SirOp>(loc, u256Type, shift, value).getResult();

    if (!typeConverter)
        return rewriter.notifyMatchFailure(op, "missing type converter");
    auto resultType = typeConverter->convertType(op.getResult().getType());
    if (!resultType)
        return rewriter.notifyMatchFailure(op, ("unable to convert " + opName + " result type").str());

    if (isU256IntegerCarrierType(resultType))
    {
        rewriter.replaceOp(op, shifted);
    }
    else
    {
        auto casted = rewriter.create<sir::BitcastOp>(loc, resultType, shifted);
        rewriter.replaceOp(op, casted.getResult());
    }
    return success();
}

template <typename SourceOp, typename SirOp, typename Adaptor>
static LogicalResult lowerSignedRightShiftOp(SourceOp op,
                                             Adaptor adaptor,
                                             ConversionPatternRewriter &rewriter,
                                             const TypeConverter *typeConverter,
                                             StringRef opName)
{
    auto loc = op.getLoc();
    Value shift = coerceToU256(rewriter, loc, adaptor.getRhs());
    Value value = signExtendToU256(rewriter, loc, adaptor.getLhs(), op.getLhs().getType());
    auto u256Type = sir::U256Type::get(op.getContext());
    Value shifted = rewriter.create<SirOp>(loc, u256Type, shift, value).getResult();

    if (!typeConverter)
        return rewriter.notifyMatchFailure(op, "missing type converter");
    auto resultType = typeConverter->convertType(op.getResult().getType());
    if (!resultType)
        return rewriter.notifyMatchFailure(op, ("unable to convert " + opName + " result type").str());

    if (isU256IntegerCarrierType(resultType))
    {
        rewriter.replaceOp(op, shifted);
    }
    else
    {
        auto casted = rewriter.create<sir::BitcastOp>(loc, resultType, shifted);
        rewriter.replaceOp(op, casted.getResult());
    }
    return success();
}

void clearArithmeticNamingHelper()
{
    arithmeticNamingHelper.reset();
}
static sir::ConstOp findOriginalConstant(Value val)
{
    Operation *defOp = val.getDefiningOp();
    if (!defOp)
        return nullptr;

    if (auto constOp = dyn_cast<sir::ConstOp>(defOp))
        return constOp;

    if (auto bitcastOp = dyn_cast<sir::BitcastOp>(defOp))
    {
        return findOriginalConstant(bitcastOp.getInput());
    }
    if (auto castOp = dyn_cast<mlir::UnrealizedConversionCastOp>(defOp))
    {
        if (castOp.getInputs().size() == 1)
        {
            return findOriginalConstant(castOp.getInputs()[0]);
        }
    }

    return nullptr;
}

static bool isZeroConst(Value val)
{
    if (auto constOp = findOriginalConstant(val))
    {
        if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(constOp.getValueAttr()))
            return intAttr.getValue().isZero();
    }
    if (auto arithConst = val.getDefiningOp<mlir::arith::ConstantOp>())
    {
        if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(arithConst.getValue()))
            return intAttr.getValue().isZero();
    }
    if (auto addrCast = val.getDefiningOp<ora::I160ToAddrOp>())
        return isZeroConst(addrCast.getI160());
    if (auto bitcastOp = val.getDefiningOp<sir::BitcastOp>())
        return isZeroConst(bitcastOp.getInput());
    if (auto castOp = val.getDefiningOp<mlir::UnrealizedConversionCastOp>())
    {
        if (castOp.getInputs().size() == 1)
            return isZeroConst(castOp.getInputs()[0]);
    }
    return false;
}

static bool isMask160Const(Value val)
{
    if (auto constOp = findOriginalConstant(val))
    {
        if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(constOp.getValueAttr()))
        {
            llvm::APInt maskValue(256, 0);
            maskValue.setLowBits(160);
            return intAttr.getValue().zextOrTrunc(256) == maskValue;
        }
    }
    return false;
}

static bool isAlreadyMasked160(Value val)
{
    if (auto andOp = val.getDefiningOp<sir::AndOp>())
    {
        return isMask160Const(andOp.getLhs()) || isMask160Const(andOp.getRhs());
    }
    return false;
}

static Value constShiftedSelector(ConversionPatternRewriter &rewriter, Location loc, uint32_t selector)
{
    llvm::APInt selectorWord(256, selector);
    selectorWord = selectorWord.shl(224);
    return constU256(rewriter, loc, selectorWord);
}

static void emitPanicRevert(ConversionPatternRewriter &rewriter, Location loc, unsigned code)
{
    auto ptrType = sir::PtrType::get(rewriter.getContext(), /*addrSpace=*/1);
    Value totalSize = constU256(rewriter, loc, 4 + evm::kWordBytes);
    Value basePtr = rewriter.create<sir::MallocOp>(loc, ptrType, totalSize);
    rewriter.create<sir::StoreOp>(loc, basePtr, constShiftedSelector(rewriter, loc, 0x4e487b71));

    Value codePtr = rewriter.create<sir::AddPtrOp>(loc, ptrType, basePtr, constU256(rewriter, loc, 4));
    rewriter.create<sir::StoreOp>(loc, codePtr, constU256(rewriter, loc, code));

    rewriter.create<sir::RevertOp>(loc, basePtr, totalSize);
}

static void emitEmptyRevert(ConversionPatternRewriter &rewriter, Location loc)
{
    auto ptrType = sir::PtrType::get(rewriter.getContext(), /*addrSpace=*/1);
    Value zeroU256 = constU256(rewriter, loc, 0);
    Value zeroPtr = rewriter.create<sir::BitcastOp>(loc, ptrType, zeroU256);
    rewriter.create<sir::RevertOp>(loc, zeroPtr, zeroU256);
}

static bool isCleanUserRuntimeCheck(Operation *op)
{
    auto verificationType = op->getAttrOfType<StringAttr>("ora.verification_type");
    if (!verificationType)
        return false;
    StringRef type = verificationType.getValue();
    return type == "guard" ||
           type == "requires" ||
           type == "ensures" ||
           type == "invariant" ||
           type == "refinement_guard";
}

static LogicalResult lowerOraRuntimeCheck(
    Operation *op,
    Value condition,
    ConversionPatternRewriter &rewriter,
    StringAttr messageAttr = {},
    StringAttr selectorAttr = {})
{
    auto loc = op->getLoc();
    Block *parentBlock = op->getBlock();
    Region *parentRegion = parentBlock->getParent();

    auto *afterBlock = rewriter.splitBlock(parentBlock, Block::iterator(op));
    auto *revertBlock = rewriter.createBlock(parentRegion, afterBlock->getIterator());

    rewriter.setInsertionPointToStart(revertBlock);
    if (isCleanUserRuntimeCheck(op))
        emitEmptyRevert(rewriter, loc);
    else if (messageAttr)
    {
        if (!selectorAttr)
            return rewriter.notifyMatchFailure(op, "message assert missing ora.assert_selector");
        auto selector = parseHexSelector(selectorAttr.getValue());
        if (!selector)
            return rewriter.notifyMatchFailure(op, "invalid ora.assert_selector");
        emitSelectorRevert(rewriter, loc, *selector);
    }
    else
        emitPanicRevert(rewriter, loc, 1);

    rewriter.setInsertionPointToEnd(parentBlock);
    Value cond = coerceToU256(rewriter, loc, condition);
    Value isZero = rewriter.create<sir::IsZeroOp>(loc, sir::U256Type::get(rewriter.getContext()), cond);
    rewriter.create<sir::CondBrOp>(
        loc,
        isZero,
        ValueRange{},
        ValueRange{},
        revertBlock,
        afterBlock);
    rewriter.eraseOp(op);
    return success();
}

static Value maskAddressTo160(ConversionPatternRewriter &rewriter, Location loc, Value value)
{
    auto ctx = rewriter.getContext();
    auto u256Type = sir::U256Type::get(ctx);
    if (isZeroConst(value) || isAlreadyMasked160(value))
        return coerceToU256(rewriter, loc, value);
    Value v = coerceToU256(rewriter, loc, value);

    llvm::APInt maskValue(256, 0);
    maskValue.setLowBits(160);
    Value mask = constU256(rewriter, loc, maskValue);
    return rewriter.create<sir::AndOp>(loc, u256Type, v, mask);
}

// Note: convertOperandToU256, convertBinaryOperands, and
// ConvertAddOp/MulOp/SubOp/DivOp/RemOp have been removed.
// The Zig frontend now emits arith.addi/subi/muli/divui/remui directly,
// which are converted by ConvertArithAddIOp etc. below.

// -----------------------------------------------------------------------------
// Convert ora.cmp → sir.eq/lt/gt (+ not/or for derived predicates)
// -----------------------------------------------------------------------------
LogicalResult ConvertCmpOp::matchAndRewrite(
    ora::CmpOp op,
    typename ora::CmpOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    auto loc = op.getLoc();
    auto ctx = rewriter.getContext();

    auto predicate = op.getPredicate();
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();
    if (llvm::isa<ora::AddressType, ora::NonZeroAddressType>(op.getLhs().getType()))
        lhs = maskAddressTo160(rewriter, loc, lhs);
    if (llvm::isa<ora::AddressType, ora::NonZeroAddressType>(op.getRhs().getType()))
        rhs = maskAddressTo160(rewriter, loc, rhs);
    lhs = coerceToU256(rewriter, loc, lhs);
    rhs = coerceToU256(rewriter, loc, rhs);
    auto u256Type = sir::U256Type::get(ctx);

    auto makeEq = [&]() -> Value {
        return rewriter.create<sir::EqOp>(loc, u256Type, lhs, rhs);
    };
    auto makeLt = [&]() -> Value {
        return rewriter.create<sir::LtOp>(loc, u256Type, lhs, rhs);
    };
    auto makeGt = [&]() -> Value {
        return rewriter.create<sir::GtOp>(loc, u256Type, lhs, rhs);
    };
    auto makeSLt = [&]() -> Value {
        return rewriter.create<sir::SLtOp>(loc, u256Type, lhs, rhs);
    };
    auto makeSGt = [&]() -> Value {
        return rewriter.create<sir::SGtOp>(loc, u256Type, lhs, rhs);
    };

    Value result;
    if (predicate == "eq")
    {
        result = makeEq();
    }
    else if (predicate == "ne" || predicate == "neq")
    {
        auto eqVal = makeEq();
        result = rewriter.create<sir::NotOp>(loc, u256Type, eqVal);
    }
    else if (predicate == "lt" || predicate == "ult")
    {
        result = makeLt();
    }
    else if (predicate == "gt" || predicate == "ugt")
    {
        result = makeGt();
    }
    else if (predicate == "le" || predicate == "lte" || predicate == "ule")
    {
        auto ltVal = makeLt();
        auto eqVal = makeEq();
        result = rewriter.create<sir::OrOp>(loc, u256Type, ltVal, eqVal);
    }
    else if (predicate == "ge" || predicate == "gte" || predicate == "uge")
    {
        auto gtVal = makeGt();
        auto eqVal = makeEq();
        result = rewriter.create<sir::OrOp>(loc, u256Type, gtVal, eqVal);
    }
    else if (predicate == "slt")
    {
        result = makeSLt();
    }
    else if (predicate == "sgt")
    {
        result = makeSGt();
    }
    else if (predicate == "sle")
    {
        auto ltVal = makeSLt();
        auto eqVal = makeEq();
        result = rewriter.create<sir::OrOp>(loc, u256Type, ltVal, eqVal);
    }
    else if (predicate == "sge")
    {
        auto gtVal = makeSGt();
        auto eqVal = makeEq();
        result = rewriter.create<sir::OrOp>(loc, u256Type, gtVal, eqVal);
    }
    else
    {
        return rewriter.notifyMatchFailure(op, "unsupported cmp predicate");
    }

    rewriter.replaceOp(op, result);
    return success();
}

// -----------------------------------------------------------------------------
// Convert ora.const → sir.const
// -----------------------------------------------------------------------------
LogicalResult ConvertConstOp::matchAndRewrite(
    ora::ConstOp op,
    typename ora::ConstOp::Adaptor,
    ConversionPatternRewriter &rewriter) const
{
    if (op.getResult().use_empty())
    {
        rewriter.eraseOp(op);
        return success();
    }

    auto loc = op.getLoc();
    auto resultType = op.getResult().getType();
    auto valueAttr = op.getValueAttr();

    if (auto tupleType = llvm::dyn_cast<ora::TupleType>(resultType))
    {
        auto arrayAttr = llvm::dyn_cast<mlir::ArrayAttr>(valueAttr);
        if (!arrayAttr)
            return rewriter.notifyMatchFailure(op, "tuple const value is not an array attribute");

        auto elementTypes = tupleType.getElementTypes();
        if (arrayAttr.size() != elementTypes.size())
            return rewriter.notifyMatchFailure(op, "tuple const attribute arity does not match tuple type");

        std::function<FailureOr<Value>(Type, Attribute)> buildConstValue = [&](Type type, Attribute attr) -> FailureOr<Value>
        {
            if (auto nestedTupleType = llvm::dyn_cast<ora::TupleType>(type))
            {
                auto nestedArrayAttr = llvm::dyn_cast<mlir::ArrayAttr>(attr);
                if (!nestedArrayAttr)
                    return failure();
                auto nestedElementTypes = nestedTupleType.getElementTypes();
                if (nestedArrayAttr.size() != nestedElementTypes.size())
                    return failure();

                SmallVector<Value, 4> nestedElements;
                nestedElements.reserve(nestedElementTypes.size());
                for (auto [nestedType, nestedAttr] : llvm::zip(nestedElementTypes, nestedArrayAttr))
                {
                    auto nestedValue = buildConstValue(nestedType, nestedAttr);
                    if (failed(nestedValue))
                        return failure();
                    nestedElements.push_back(*nestedValue);
                }
                return rewriter.create<ora::TupleCreateOp>(loc, nestedTupleType, nestedElements).getResult();
            }

            auto typedAttr = llvm::dyn_cast<mlir::TypedAttr>(attr);
            if (!typedAttr)
                return failure();

            auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(typedAttr);
            if (!intAttr)
                return failure();

            auto nameAttr = rewriter.getStringAttr(op.getName());
            return rewriter.create<ora::ConstOp>(loc, type, nameAttr, intAttr).getResult();
        };

        SmallVector<Value, 4> elements;
        elements.reserve(elementTypes.size());
        for (auto [elementType, elementAttr] : llvm::zip(elementTypes, arrayAttr))
        {
            auto elementValue = buildConstValue(elementType, elementAttr);
            if (failed(elementValue))
                return rewriter.notifyMatchFailure(op, "unsupported tuple const element attribute");
            elements.push_back(*elementValue);
        }

        rewriter.replaceOpWithNewOp<ora::TupleCreateOp>(op, tupleType, elements);
        return success();
    }

    auto typedAttr = llvm::dyn_cast<mlir::TypedAttr>(valueAttr);
    if (!typedAttr)
    {
        return rewriter.notifyMatchFailure(op, "const value is not a typed attribute");
    }

    auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(typedAttr);
    if (!intAttr)
    {
        return rewriter.notifyMatchFailure(op, "const value is not an integer");
    }

    auto *typeConverter = getTypeConverter();
    if (!typeConverter)
    {
        return rewriter.notifyMatchFailure(op, "missing type converter");
    }

    auto convertedType = typeConverter->convertType(resultType);
    if (!convertedType)
    {
        return rewriter.notifyMatchFailure(op, "unable to convert const type");
    }

    auto u256Type = sir::U256Type::get(op.getContext());
    llvm::APInt val = intAttr.getValue();
    if (val.getBitWidth() < 256)
        val = val.zext(256);
    else if (val.getBitWidth() > 256)
        val = val.trunc(256);

    Value result = constU256(rewriter, loc, val);
    if (Operation *constOp = result.getDefiningOp())
        constOp->setAttr("ora.name", op.getNameAttr());
    if (convertedType != u256Type)
        result = rewriter.create<sir::BitcastOp>(loc, convertedType, result);

    rewriter.replaceOp(op, result);
    return success();
}

// -----------------------------------------------------------------------------
// Convert ora.length → sir.load of dynamic string/bytes length header
// -----------------------------------------------------------------------------
LogicalResult ConvertLengthOp::matchAndRewrite(
    Operation *op,
    ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const
{
    if (op->getName().getStringRef() != "ora.length")
        return failure();
    if (op->getNumResults() != 1 || operands.size() != 1)
        return rewriter.notifyMatchFailure(op, "ora.length expects one operand and one result");

    Value source = operands[0];
    auto loc = op->getLoc();
    auto u256Type = sir::U256Type::get(rewriter.getContext());

    if (Value storageRoot = getStorageViewRootSlot(op->getOperand(0)))
    {
        Value length = rewriter.create<sir::SLoadOp>(loc, u256Type, storageRoot);
        rewriter.replaceOp(op, length);
        return success();
    }
    if (Value storageRoot = getStorageViewRootSlot(source))
    {
        Value length = rewriter.create<sir::SLoadOp>(loc, u256Type, storageRoot);
        rewriter.replaceOp(op, length);
        return success();
    }

    if (!llvm::isa<sir::PtrType>(source.getType()))
        return rewriter.notifyMatchFailure(op, "ora.length source is not a lowered dynamic bytes pointer");

    Value length = rewriter.create<sir::LoadOp>(loc, u256Type, source);
    rewriter.replaceOp(op, length);
    return success();
}

// -----------------------------------------------------------------------------
// Convert ora.byte_at → sir.load8 from dynamic string/bytes payload
// Layout: [len: u256][bytes...]
// -----------------------------------------------------------------------------
LogicalResult ConvertByteAtOp::matchAndRewrite(
    Operation *op,
    ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const
{
    if (op->getName().getStringRef() != "ora.byte_at")
        return failure();
    if (op->getNumResults() != 1 || operands.size() != 2)
        return rewriter.notifyMatchFailure(op, "ora.byte_at expects two operands and one result");

    auto loc = op->getLoc();
    auto ctx = rewriter.getContext();
    auto u256Type = sir::U256Type::get(ctx);
    auto ptrType = sir::PtrType::get(ctx, /*addrSpace*/ 1);

    Value source = operands[0];
    if (!llvm::isa<sir::PtrType>(source.getType()))
        source = rewriter.create<sir::BitcastOp>(loc, ptrType, source);

    Value index = coerceToU256(rewriter, loc, operands[1]);
    Value headerSize = constU256(rewriter, loc, 32);
    Value offset = rewriter.create<sir::AddOp>(loc, u256Type, headerSize, index);
    Value addr = rewriter.create<sir::AddPtrOp>(loc, ptrType, source, offset);
    Value zero = constU256(rewriter, loc, 0);
    Value byteValue = rewriter.create<sir::Load8Op>(loc, u256Type, addr, zero);
    rewriter.replaceOp(op, byteValue);
    return success();
}

// -----------------------------------------------------------------------------
// Convert ora.concat → allocate [len][lhs bytes][rhs bytes]
// Layout: [len: u256][bytes...]
// -----------------------------------------------------------------------------
LogicalResult ConvertConcatOp::matchAndRewrite(
    Operation *op,
    ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const
{
    if (op->getName().getStringRef() != "ora.concat")
        return failure();
    if (op->getNumResults() != 1 || operands.size() != 2)
        return rewriter.notifyMatchFailure(op, "ora.concat expects two operands and one result");

    auto loc = op->getLoc();
    auto ctx = rewriter.getContext();
    auto u256Type = sir::U256Type::get(ctx);
    auto ptrType = sir::PtrType::get(ctx, /*addrSpace*/ 1);

    Value lhs = operands[0];
    Value rhs = operands[1];
    if (!llvm::isa<sir::PtrType>(lhs.getType()))
        lhs = rewriter.create<sir::BitcastOp>(loc, ptrType, lhs);
    if (!llvm::isa<sir::PtrType>(rhs.getType()))
        rhs = rewriter.create<sir::BitcastOp>(loc, ptrType, rhs);

    Value lhsLen = rewriter.create<sir::LoadOp>(loc, u256Type, lhs);
    Value rhsLen = rewriter.create<sir::LoadOp>(loc, u256Type, rhs);
    Value resultLen = rewriter.create<sir::AddOp>(loc, u256Type, lhsLen, rhsLen);
    Value headerSize = constU256(rewriter, loc, evm::kWordBytes);
    Value totalSize = rewriter.create<sir::AddOp>(loc, u256Type, resultLen, headerSize);

    Value base = rewriter.create<sir::MallocOp>(loc, ptrType, totalSize);
    rewriter.create<sir::StoreOp>(loc, base, resultLen);

    Value lhsPayload = rewriter.create<sir::AddPtrOp>(loc, ptrType, lhs, headerSize);
    Value rhsPayload = rewriter.create<sir::AddPtrOp>(loc, ptrType, rhs, headerSize);
    Value resultPayload = rewriter.create<sir::AddPtrOp>(loc, ptrType, base, headerSize);
    rewriter.create<sir::MCopyOp>(loc, resultPayload, lhsPayload, lhsLen);

    Value rhsDest = rewriter.create<sir::AddPtrOp>(loc, ptrType, resultPayload, lhsLen);
    rewriter.create<sir::MCopyOp>(loc, rhsDest, rhsPayload, rhsLen);

    rewriter.replaceOp(op, base);
    return success();
}

// -----------------------------------------------------------------------------
// Convert ora.slice → allocate a dynamic value.
// - string/bytes slices use byte lengths: [len: u256][bytes...]
// - memref slices use element counts: [len: u256][32-byte element slots...]
// -----------------------------------------------------------------------------
LogicalResult ConvertSliceOp::matchAndRewrite(
    Operation *op,
    ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const
{
    if (op->getName().getStringRef() != "ora.slice")
        return failure();
    if (op->getNumResults() != 1 || operands.size() != 3)
        return rewriter.notifyMatchFailure(op, "ora.slice expects three operands and one result");

    auto loc = op->getLoc();
    auto ctx = rewriter.getContext();
    auto u256Type = sir::U256Type::get(ctx);
    auto ptrType = sir::PtrType::get(ctx, /*addrSpace*/ 1);

    Value source = operands[0];
    if (!llvm::isa<sir::PtrType>(source.getType()))
        source = rewriter.create<sir::BitcastOp>(loc, ptrType, source);

    Value start = coerceToU256(rewriter, loc, operands[1]);
    Value length = coerceToU256(rewriter, loc, operands[2]);
    Value headerSize = constU256(rewriter, loc, evm::kWordBytes);

    if (auto resultMemRef = llvm::dyn_cast<mlir::MemRefType>(op->getResult(0).getType()))
    {
        if (resultMemRef.getRank() != 1 || !resultMemRef.isDynamicDim(0))
            return rewriter.notifyMatchFailure(op, "ora.slice memref result must be rank-1 dynamic");

        Value elementSlotBytes = constU256(rewriter, loc, evm::kWordBytes);
        Value startBytes = rewriter.create<sir::MulOp>(loc, u256Type, start, elementSlotBytes);
        Value payloadBytes = rewriter.create<sir::MulOp>(loc, u256Type, length, elementSlotBytes);
        Value totalSize = rewriter.create<sir::AddOp>(loc, u256Type, payloadBytes, headerSize);

        Value base = rewriter.create<sir::MallocOp>(loc, ptrType, totalSize);
        rewriter.create<sir::StoreOp>(loc, base, length);

        Value sourcePayload = source;
        if (auto sourceMemRef = llvm::dyn_cast<mlir::MemRefType>(op->getOperand(0).getType()))
        {
            if (!sourceMemRef.hasStaticShape())
                sourcePayload = rewriter.create<sir::AddPtrOp>(loc, ptrType, source, headerSize);
        }
        else
        {
            sourcePayload = rewriter.create<sir::AddPtrOp>(loc, ptrType, source, headerSize);
        }
        Value sourceStart = rewriter.create<sir::AddPtrOp>(loc, ptrType, sourcePayload, startBytes);
        Value resultPayload = rewriter.create<sir::AddPtrOp>(loc, ptrType, base, headerSize);
        rewriter.create<sir::MCopyOp>(loc, resultPayload, sourceStart, payloadBytes);

        rewriter.replaceOp(op, base);
        return success();
    }

    Value totalSize = rewriter.create<sir::AddOp>(loc, u256Type, length, headerSize);

    Value base = rewriter.create<sir::MallocOp>(loc, ptrType, totalSize);
    rewriter.create<sir::StoreOp>(loc, base, length);

    Value sourcePayload = rewriter.create<sir::AddPtrOp>(loc, ptrType, source, headerSize);
    Value sourceStart = rewriter.create<sir::AddPtrOp>(loc, ptrType, sourcePayload, start);
    Value resultPayload = rewriter.create<sir::AddPtrOp>(loc, ptrType, base, headerSize);
    rewriter.create<sir::MCopyOp>(loc, resultPayload, sourceStart, length);

    rewriter.replaceOp(op, base);
    return success();
}

// -----------------------------------------------------------------------------
// Convert ora.keccak256 → sir.keccak256 over dynamic string/bytes payload
// Layout: [len: u256][bytes...]
// -----------------------------------------------------------------------------
LogicalResult ConvertKeccak256Op::matchAndRewrite(
    Operation *op,
    ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const
{
    if (op->getName().getStringRef() != "ora.keccak256")
        return failure();
    if (op->getNumResults() != 1 || operands.size() != 1)
        return rewriter.notifyMatchFailure(op, "ora.keccak256 expects one operand and one result");

    auto loc = op->getLoc();
    auto ctx = rewriter.getContext();
    auto u256Type = sir::U256Type::get(ctx);
    auto ptrType = sir::PtrType::get(ctx, /*addrSpace*/ 1);

    Value source = operands[0];
    if (!llvm::isa<sir::PtrType>(source.getType()))
        source = rewriter.create<sir::BitcastOp>(loc, ptrType, source);

    Value length = rewriter.create<sir::LoadOp>(loc, u256Type, source);
    Value headerSize = constU256(rewriter, loc, 32);
    Value payload = rewriter.create<sir::AddPtrOp>(loc, ptrType, source, headerSize);
    Value hash = rewriter.create<sir::KeccakOp>(loc, u256Type, payload, length);
    rewriter.replaceOp(op, hash);
    return success();
}

// -----------------------------------------------------------------------------
// Convert ora.string.constant → ptr (dynamic bytes layout)
// Layout: [len: u256][bytes...]
// -----------------------------------------------------------------------------
LogicalResult ConvertStringConstantOp::matchAndRewrite(
    ora::StringConstantOp op,
    typename ora::StringConstantOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    auto loc = op.getLoc();
    auto valueAttr = op.getValueAttr();
    auto strAttr = llvm::dyn_cast<mlir::StringAttr>(valueAttr);
    if (!strAttr)
    {
        return rewriter.notifyMatchFailure(op, "string constant missing value attribute");
    }

    auto bytes = strAttr.getValue();
    auto ctx = op.getContext();
    auto u256Type = sir::U256Type::get(ctx);
    auto ptrType = sir::PtrType::get(ctx, /*addrSpace*/ 1);

    Value lengthConst = constU256(rewriter, loc, bytes.size());
    Value wordSize = constU256(rewriter, loc, 32);
    Value totalSize = rewriter.create<sir::AddOp>(loc, u256Type, lengthConst, wordSize);
    Value base = rewriter.create<sir::MallocOp>(loc, ptrType, totalSize);

    // store length word at offset 0
    rewriter.create<sir::StoreOp>(loc, base, lengthConst);

    // store payload as 32-byte words (EVM-friendly layout)
    const size_t word_count = (bytes.size() + 31) / 32;
    for (size_t w = 0; w < word_count; ++w)
    {
        llvm::APInt word(256, 0);
        for (size_t i = 0; i < 32; ++i)
        {
            const size_t idx = w * 32 + i;
            const uint8_t byte = idx < bytes.size() ? static_cast<uint8_t>(bytes[idx]) : 0;
            word = word.shl(8);
            word = word | llvm::APInt(256, byte);
        }
        Value wordConst = constU256(rewriter, loc, word);
        Value offsetConst = constU256(rewriter, loc, 32 + w * 32);
        Value dataPtr = rewriter.create<sir::AddPtrOp>(loc, ptrType, base, offsetConst);
        rewriter.create<sir::StoreOp>(loc, dataPtr, wordConst);
    }

    rewriter.replaceOp(op, base);
    return success();
}

// -----------------------------------------------------------------------------
// Convert ora.bytes.constant → ptr (dynamic bytes layout)
// -----------------------------------------------------------------------------
LogicalResult ConvertBytesConstantOp::matchAndRewrite(
    ora::BytesConstantOp op,
    typename ora::BytesConstantOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    auto loc = op.getLoc();
    auto valueAttr = op.getValueAttr();
    auto strAttr = llvm::dyn_cast<mlir::StringAttr>(valueAttr);
    if (!strAttr)
    {
        return rewriter.notifyMatchFailure(op, "bytes constant missing value attribute");
    }

    auto raw = strAttr.getValue();
    if (raw.starts_with("0x") || raw.starts_with("0X"))
        raw = raw.drop_front(2);

    if (raw.size() % 2 != 0)
    {
        return rewriter.notifyMatchFailure(op, "bytes constant has odd-length hex string");
    }

    SmallVector<uint8_t> bytes;
    bytes.reserve(raw.size() / 2);
    for (size_t i = 0; i < raw.size(); i += 2)
    {
        auto hi = llvm::hexDigitValue(raw[i]);
        auto lo = llvm::hexDigitValue(raw[i + 1]);
        if (hi < 0 || lo < 0)
        {
            return rewriter.notifyMatchFailure(op, "bytes constant has invalid hex characters");
        }
        bytes.push_back(static_cast<uint8_t>((hi << 4) | lo));
    }

    auto ctx = op.getContext();
    auto u256Type = sir::U256Type::get(ctx);
    auto ptrType = sir::PtrType::get(ctx, /*addrSpace*/ 1);

    Value lengthConst = constU256(rewriter, loc, bytes.size());
    Value wordSize = constU256(rewriter, loc, 32);
    Value totalSize = rewriter.create<sir::AddOp>(loc, u256Type, lengthConst, wordSize);
    Value base = rewriter.create<sir::MallocOp>(loc, ptrType, totalSize);

    rewriter.create<sir::StoreOp>(loc, base, lengthConst);

    // store payload as 32-byte words (EVM-friendly layout)
    const size_t word_count = (bytes.size() + 31) / 32;
    for (size_t w = 0; w < word_count; ++w)
    {
        llvm::APInt word(256, 0);
        for (size_t i = 0; i < 32; ++i)
        {
            const size_t idx = w * 32 + i;
            const uint8_t byte = idx < bytes.size() ? bytes[idx] : 0;
            word = word.shl(8);
            word = word | llvm::APInt(256, byte);
        }
        Value wordConst = constU256(rewriter, loc, word);
        Value offsetConst = constU256(rewriter, loc, 32 + w * 32);
        Value dataPtr = rewriter.create<sir::AddPtrOp>(loc, ptrType, base, offsetConst);
        rewriter.create<sir::StoreOp>(loc, dataPtr, wordConst);
    }

    rewriter.replaceOp(op, base);
    return success();
}

// -----------------------------------------------------------------------------
// Convert ora.hex.constant → sir.const (u256)
// -----------------------------------------------------------------------------
LogicalResult ConvertHexConstantOp::matchAndRewrite(
    ora::HexConstantOp op,
    typename ora::HexConstantOp::Adaptor,
    ConversionPatternRewriter &rewriter) const
{
    auto loc = op.getLoc();
    auto valueAttr = op.getValueAttr();
    auto strAttr = llvm::dyn_cast<mlir::StringAttr>(valueAttr);
    if (!strAttr)
    {
        return rewriter.notifyMatchFailure(op, "hex constant missing value attribute");
    }

    auto raw = strAttr.getValue();
    if (raw.starts_with("0x") || raw.starts_with("0X"))
        raw = raw.drop_front(2);

    if (raw.empty())
    {
        return rewriter.notifyMatchFailure(op, "hex constant has empty value");
    }

    llvm::APInt value(256, raw, 16);
    rewriter.replaceOp(op, constU256(rewriter, loc, value));
    return success();
}

// -----------------------------------------------------------------------------
// Lower ora.addr.to.i160 → masked u256 carrier
// -----------------------------------------------------------------------------
LogicalResult ConvertAddrToI160Op::matchAndRewrite(
    ora::AddrToI160Op op,
    typename ora::AddrToI160Op::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    Value input = adaptor.getAddr();

    rewriter.replaceOp(op, maskAddressTo160(rewriter, op.getLoc(), input));
    return success();
}

// -----------------------------------------------------------------------------
// Lower ora.i160.to.addr → sir.bitcast + mask to 160 bits
// -----------------------------------------------------------------------------
LogicalResult ConvertI160ToAddrOp::matchAndRewrite(
    ora::I160ToAddrOp op,
    typename ora::I160ToAddrOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    auto loc = op.getLoc();
    auto ctx = op.getContext();
    Value input = adaptor.getI160();

    auto u256Type = sir::U256Type::get(ctx);
    Value cast = rewriter.create<sir::BitcastOp>(loc, u256Type, input);

    if (isZeroConst(input))
    {
        rewriter.replaceOp(op, cast);
        return success();
    }

    llvm::APInt maskValue(256, 0);
    maskValue.setLowBits(160);
    Value mask = constU256(rewriter, loc, maskValue);
    Value masked = rewriter.create<sir::AndOp>(loc, u256Type, cast, mask);

    rewriter.replaceOp(op, masked);
    return success();
}

// -----------------------------------------------------------------------------
// Lower ora.old → passthrough value
// -----------------------------------------------------------------------------
LogicalResult ConvertOldOp::matchAndRewrite(
    ora::OldOp op,
    typename ora::OldOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    rewriter.replaceOp(op, adaptor.getValue());
    return success();
}

// -----------------------------------------------------------------------------
// Convert ora.refinement_to_base → passthrough value
// -----------------------------------------------------------------------------
LogicalResult ConvertRefinementToBaseOp::matchAndRewrite(
    ora::RefinementToBaseOp op,
    typename ora::RefinementToBaseOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    Type convertedType = this->getTypeConverter()->convertType(op.getType());
    if (!convertedType)
    {
        return rewriter.notifyMatchFailure(op, "failed to convert refinement base type");
    }

    Value value = adaptor.getValue();
    if (value.getType() != convertedType)
    {
        value = rewriter.create<sir::BitcastOp>(op.getLoc(), convertedType, value);
    }

    rewriter.replaceOp(op, value);
    return success();
}

// -----------------------------------------------------------------------------
// Convert ora.base_to_refinement → passthrough value
// -----------------------------------------------------------------------------
LogicalResult ConvertBaseToRefinementOp::matchAndRewrite(
    ora::BaseToRefinementOp op,
    typename ora::BaseToRefinementOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    Type convertedType = this->getTypeConverter()->convertType(op.getType());
    if (!convertedType)
    {
        return rewriter.notifyMatchFailure(op, "failed to convert base to refinement type");
    }

    Value value = adaptor.getValue();
    if (value.getType() != convertedType)
    {
        value = rewriter.create<sir::BitcastOp>(op.getLoc(), convertedType, value);
    }

    rewriter.replaceOp(op, value);
    return success();
}

// -----------------------------------------------------------------------------
// Verification ops lowered away for Ora → SIR
// -----------------------------------------------------------------------------
LogicalResult ConvertInvariantOp::matchAndRewrite(
    ora::InvariantOp op,
    typename ora::InvariantOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    rewriter.eraseOp(op);
    return success();
}

LogicalResult ConvertRequiresOp::matchAndRewrite(
    ora::RequiresOp op,
    typename ora::RequiresOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    rewriter.eraseOp(op);
    return success();
}

LogicalResult ConvertEnsuresOp::matchAndRewrite(
    ora::EnsuresOp op,
    typename ora::EnsuresOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    rewriter.eraseOp(op);
    return success();
}

LogicalResult ConvertAssertOp::matchAndRewrite(
    ora::AssertOp op,
    typename ora::AssertOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    return lowerOraRuntimeCheck(
        op.getOperation(),
        adaptor.getCondition(),
        rewriter,
        op->getAttrOfType<mlir::StringAttr>("message"),
        op->getAttrOfType<mlir::StringAttr>("ora.assert_selector"));
}

LogicalResult ConvertAssumeOp::matchAndRewrite(
    ora::AssumeOp op,
    typename ora::AssumeOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    rewriter.eraseOp(op);
    return success();
}

LogicalResult ConvertDecreasesOp::matchAndRewrite(
    ora::DecreasesOp op,
    typename ora::DecreasesOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    rewriter.eraseOp(op);
    return success();
}

LogicalResult ConvertIncreasesOp::matchAndRewrite(
    ora::IncreasesOp op,
    typename ora::IncreasesOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    rewriter.eraseOp(op);
    return success();
}

LogicalResult ConvertHavocOp::matchAndRewrite(
    ora::HavocOp op,
    typename ora::HavocOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    rewriter.eraseOp(op);
    return success();
}

LogicalResult ConvertQuantifiedOp::matchAndRewrite(
    ora::QuantifiedOp op,
    typename ora::QuantifiedOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    (void)adaptor;
    auto *typeConverter = getTypeConverter();
    if (!typeConverter)
        return rewriter.notifyMatchFailure(op, "missing type converter");

    Type resultType = op.getResult().getType();
    if (auto converted = typeConverter->convertType(resultType))
        resultType = converted;

    // Quantifiers are specification-only constructs. Runtime SIR has no SMT
    // domain, so erase the operator by replacing it with a trivially true value.
    if (llvm::isa<sir::U256Type>(resultType))
    {
        Value one = constU256(rewriter, op.getLoc(), 1);
        rewriter.replaceOp(op, ValueRange{one});
        return success();
    }

    if (auto intTy = llvm::dyn_cast<mlir::IntegerType>(resultType))
    {
        if (intTy.getWidth() == 1)
        {
            rewriter.replaceOpWithNewOp<mlir::arith::ConstantOp>(op, resultType, rewriter.getBoolAttr(true));
            return success();
        }
        auto oneAttr = rewriter.getIntegerAttr(resultType, 1);
        rewriter.replaceOpWithNewOp<mlir::arith::ConstantOp>(op, resultType, oneAttr);
        return success();
    }

    return rewriter.notifyMatchFailure(op, "unsupported converted result type for ora.quantified");
}

// -----------------------------------------------------------------------------
// Convert arith.constant with Ora types → sir.const with SIR types
// -----------------------------------------------------------------------------
LogicalResult ConvertArithConstantOp::matchAndRewrite(
    mlir::arith::ConstantOp op,
    typename mlir::arith::ConstantOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    auto loc = op.getLoc();
    auto resultType = op.getResult().getType();

    DBG("ConvertArithConstantOp: checking constant with type: " << resultType);

    auto *typeConverter = getTypeConverter();
    if (!typeConverter)
    {
        return rewriter.notifyMatchFailure(op, "missing type converter");
    }

    DBG("ConvertArithConstantOp: converting arith.constant to sir.const");

    auto valueAttr = op.getValue();
    if (!valueAttr)
    {
        return rewriter.notifyMatchFailure(op, "missing value attribute");
    }

    auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(valueAttr);
    if (!intAttr)
        return rewriter.notifyMatchFailure(op, "value attribute is not an integer");

    auto convertedType = typeConverter->convertType(resultType);
    if (!convertedType)
        return rewriter.notifyMatchFailure(op, "unable to convert constant type");

    // Preserve full APInt width — no truncation to uint64_t.
    llvm::APInt val = intAttr.getValue();
    if (val.getBitWidth() < 256)
        val = val.zext(256);
    else if (val.getBitWidth() > 256)
        val = val.trunc(256);

    Value constResult = constU256(rewriter, loc, val);

    auto &naming = getNamingHelper(op);
    if (val.ule(10000))
    {
        if (Operation *constOp = constResult.getDefiningOp())
            naming.nameConst(constOp, 0, val.getZExtValue());
    }
    rewriter.replaceOp(op, constResult);
    return success();
}

// -----------------------------------------------------------------------------
// Convert arith.cmpi → sir.{eq,lt,gt,sgt,slt} (+ combos for le/ge/ne)
// -----------------------------------------------------------------------------
LogicalResult ConvertArithCmpIOp::matchAndRewrite(
    mlir::arith::CmpIOp op,
    typename mlir::arith::CmpIOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    auto loc = op.getLoc();
    auto *typeConverter = getTypeConverter();
    if (!typeConverter)
    {
        return rewriter.notifyMatchFailure(op, "missing type converter");
    }

    auto resultType = typeConverter->convertType(op.getResult().getType());
    if (!resultType)
    {
        return rewriter.notifyMatchFailure(op, "unable to convert cmp result type");
    }

    auto u256Type = sir::U256Type::get(op.getContext());
    auto one = constU256(rewriter, loc, 1);

    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();
    if (!llvm::isa<sir::U256Type>(lhs.getType()))
        lhs = rewriter.create<sir::BitcastOp>(loc, u256Type, lhs);
    if (!llvm::isa<sir::U256Type>(rhs.getType()))
        rhs = rewriter.create<sir::BitcastOp>(loc, u256Type, rhs);

    // Address values must be masked to 160 bits before comparison to avoid
    // high-bit garbage affecting equality/ordering checks.
    if (llvm::isa<ora::AddressType>(op.getLhs().getType()))
        lhs = maskAddressTo160(rewriter, loc, lhs);
    if (llvm::isa<ora::AddressType>(op.getRhs().getType()))
        rhs = maskAddressTo160(rewriter, loc, rhs);

    const auto pred = op.getPredicate();
    switch (pred)
    {
    case mlir::arith::CmpIPredicate::slt:
    case mlir::arith::CmpIPredicate::sgt:
    case mlir::arith::CmpIPredicate::sle:
    case mlir::arith::CmpIPredicate::sge:
        lhs = signExtendToU256(rewriter, loc, lhs, op.getLhs().getType());
        rhs = signExtendToU256(rewriter, loc, rhs, op.getRhs().getType());
        break;
    default:
        break;
    }

    auto mkEq = [&]() { return rewriter.create<sir::EqOp>(loc, resultType, lhs, rhs).getResult(); };
    auto mkLt = [&]() { return rewriter.create<sir::LtOp>(loc, resultType, lhs, rhs).getResult(); };
    auto mkGt = [&]() { return rewriter.create<sir::GtOp>(loc, resultType, lhs, rhs).getResult(); };
    auto mkSLt = [&]() { return rewriter.create<sir::SLtOp>(loc, resultType, lhs, rhs).getResult(); };
    auto mkSGt = [&]() { return rewriter.create<sir::SGtOp>(loc, resultType, lhs, rhs).getResult(); };

    Value out;
    switch (pred)
    {
    case mlir::arith::CmpIPredicate::eq:
        out = mkEq();
        break;
    case mlir::arith::CmpIPredicate::ne:
    {
        auto eq = mkEq();
        out = rewriter.create<sir::XorOp>(loc, resultType, eq, one).getResult();
        break;
    }
    case mlir::arith::CmpIPredicate::ult:
        out = mkLt();
        break;
    case mlir::arith::CmpIPredicate::ugt:
        out = mkGt();
        break;
    case mlir::arith::CmpIPredicate::ule:
    {
        auto lt = mkLt();
        auto eq = mkEq();
        out = rewriter.create<sir::OrOp>(loc, resultType, lt, eq).getResult();
        break;
    }
    case mlir::arith::CmpIPredicate::uge:
    {
        auto gt = mkGt();
        auto eq = mkEq();
        out = rewriter.create<sir::OrOp>(loc, resultType, gt, eq).getResult();
        break;
    }
    case mlir::arith::CmpIPredicate::slt:
        out = mkSLt();
        break;
    case mlir::arith::CmpIPredicate::sgt:
        out = mkSGt();
        break;
    case mlir::arith::CmpIPredicate::sle:
    {
        auto lt = mkSLt();
        auto eq = mkEq();
        out = rewriter.create<sir::OrOp>(loc, resultType, lt, eq).getResult();
        break;
    }
    case mlir::arith::CmpIPredicate::sge:
    {
        auto gt = mkSGt();
        auto eq = mkEq();
        out = rewriter.create<sir::OrOp>(loc, resultType, gt, eq).getResult();
        break;
    }
    // All enum values are handled above, but keeping default for defensive programming
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wcovered-switch-default"
    default:
        return rewriter.notifyMatchFailure(op, "unsupported cmp predicate");
#pragma clang diagnostic pop
    }

    rewriter.replaceOp(op, out);
    return success();
}

// -----------------------------------------------------------------------------
// Convert arith.addi → sir.add
// -----------------------------------------------------------------------------
LogicalResult ConvertArithAddIOp::matchAndRewrite(
    mlir::arith::AddIOp op,
    typename mlir::arith::AddIOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    return lowerBinaryOp<mlir::arith::AddIOp, sir::AddOp>(op, adaptor, rewriter, getTypeConverter(), "addi");
}

// -----------------------------------------------------------------------------
// Convert ora.add_wrapping → sir.add
// -----------------------------------------------------------------------------
LogicalResult ConvertAddWrappingOp::matchAndRewrite(
    ora::AddWrappingOp op,
    typename ora::AddWrappingOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    return lowerBinaryOp<ora::AddWrappingOp, sir::AddOp>(op, adaptor, rewriter, getTypeConverter(), "add_wrapping");
}

LogicalResult ConvertSubWrappingOp::matchAndRewrite(
    ora::SubWrappingOp op,
    typename ora::SubWrappingOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    return lowerBinaryOp<ora::SubWrappingOp, sir::SubOp>(op, adaptor, rewriter, getTypeConverter(), "sub_wrapping");
}

LogicalResult ConvertMulWrappingOp::matchAndRewrite(
    ora::MulWrappingOp op,
    typename ora::MulWrappingOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    return lowerBinaryOp<ora::MulWrappingOp, sir::MulOp>(op, adaptor, rewriter, getTypeConverter(), "mul_wrapping");
}

// -----------------------------------------------------------------------------
// Convert arith.subi → sir.sub
// -----------------------------------------------------------------------------
LogicalResult ConvertArithSubIOp::matchAndRewrite(
    mlir::arith::SubIOp op,
    typename mlir::arith::SubIOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    return lowerBinaryOp<mlir::arith::SubIOp, sir::SubOp>(op, adaptor, rewriter, getTypeConverter(), "subi");
}

// -----------------------------------------------------------------------------
// Convert arith.muli → sir.mul
// -----------------------------------------------------------------------------
LogicalResult ConvertArithMulIOp::matchAndRewrite(
    mlir::arith::MulIOp op,
    typename mlir::arith::MulIOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    return lowerBinaryOp<mlir::arith::MulIOp, sir::MulOp>(op, adaptor, rewriter, getTypeConverter(), "muli");
}

// -----------------------------------------------------------------------------
// Convert arith.divui → sir.div
// -----------------------------------------------------------------------------
LogicalResult ConvertArithDivUIOp::matchAndRewrite(
    mlir::arith::DivUIOp op,
    typename mlir::arith::DivUIOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    return lowerBinaryOp<mlir::arith::DivUIOp, sir::DivOp>(op, adaptor, rewriter, getTypeConverter(), "divui");
}

// -----------------------------------------------------------------------------
// Convert arith.remui → sir.mod
// -----------------------------------------------------------------------------
LogicalResult ConvertArithRemUIOp::matchAndRewrite(
    mlir::arith::RemUIOp op,
    typename mlir::arith::RemUIOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    return lowerBinaryOp<mlir::arith::RemUIOp, sir::ModOp>(op, adaptor, rewriter, getTypeConverter(), "remui");
}

// -----------------------------------------------------------------------------
// Convert arith.divsi → sir.sdiv (signed division)
// -----------------------------------------------------------------------------
LogicalResult ConvertArithDivSIOp::matchAndRewrite(
    mlir::arith::DivSIOp op,
    typename mlir::arith::DivSIOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    return lowerSignedBinaryOp<mlir::arith::DivSIOp, sir::SDivOp>(op, adaptor, rewriter, getTypeConverter(), "divsi");
}

// -----------------------------------------------------------------------------
// Convert arith.remsi → sir.smod (signed remainder)
// -----------------------------------------------------------------------------
LogicalResult ConvertArithRemSIOp::matchAndRewrite(
    mlir::arith::RemSIOp op,
    typename mlir::arith::RemSIOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    return lowerSignedBinaryOp<mlir::arith::RemSIOp, sir::SModOp>(op, adaptor, rewriter, getTypeConverter(), "remsi");
}

// -----------------------------------------------------------------------------
// Convert arith.andi → sir.and
// -----------------------------------------------------------------------------
LogicalResult ConvertArithAndIOp::matchAndRewrite(
    mlir::arith::AndIOp op,
    typename mlir::arith::AndIOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    return lowerBinaryOp<mlir::arith::AndIOp, sir::AndOp>(op, adaptor, rewriter, getTypeConverter(), "andi");
}

// -----------------------------------------------------------------------------
// Convert arith.ori → sir.or
// -----------------------------------------------------------------------------
LogicalResult ConvertArithOrIOp::matchAndRewrite(
    mlir::arith::OrIOp op,
    typename mlir::arith::OrIOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    return lowerBinaryOp<mlir::arith::OrIOp, sir::OrOp>(op, adaptor, rewriter, getTypeConverter(), "ori");
}

// -----------------------------------------------------------------------------
// Convert arith.xori → sir.xor
// -----------------------------------------------------------------------------
LogicalResult ConvertArithXOrIOp::matchAndRewrite(
    mlir::arith::XOrIOp op,
    typename mlir::arith::XOrIOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    return lowerBinaryOp<mlir::arith::XOrIOp, sir::XorOp>(op, adaptor, rewriter, getTypeConverter(), "xori");
}

// -----------------------------------------------------------------------------
// Convert arith.shli → sir.shl
// -----------------------------------------------------------------------------
LogicalResult ConvertArithShlIOp::matchAndRewrite(
    mlir::arith::ShLIOp op,
    typename mlir::arith::ShLIOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    return lowerShiftOp<mlir::arith::ShLIOp, sir::ShlOp>(op, adaptor, rewriter, getTypeConverter(), "shli");
}

LogicalResult ConvertShlWrappingOp::matchAndRewrite(
    ora::ShlWrappingOp op,
    typename ora::ShlWrappingOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    return lowerShiftOp<ora::ShlWrappingOp, sir::ShlOp>(op, adaptor, rewriter, getTypeConverter(), "shl_wrapping");
}

// -----------------------------------------------------------------------------
// Convert arith.shrui → sir.shr
// -----------------------------------------------------------------------------
LogicalResult ConvertArithShrUIOp::matchAndRewrite(
    mlir::arith::ShRUIOp op,
    typename mlir::arith::ShRUIOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    return lowerShiftOp<mlir::arith::ShRUIOp, sir::ShrOp>(op, adaptor, rewriter, getTypeConverter(), "shrui");
}

// -----------------------------------------------------------------------------
// Convert arith.shrsi → sir.sar
// -----------------------------------------------------------------------------
LogicalResult ConvertArithShrSIOp::matchAndRewrite(
    mlir::arith::ShRSIOp op,
    typename mlir::arith::ShRSIOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    return lowerSignedRightShiftOp<mlir::arith::ShRSIOp, sir::SarOp>(op, adaptor, rewriter, getTypeConverter(), "shrsi");
}

LogicalResult ConvertShrWrappingOp::matchAndRewrite(
    ora::ShrWrappingOp op,
    typename ora::ShrWrappingOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    auto loc = op.getLoc();
    Value shift = coerceToU256(rewriter, loc, adaptor.getRhs());
    auto intType = llvm::dyn_cast<ora::IntegerType>(op.getLhs().getType());
    Value value = (intType && intType.getIsSigned())
        ? signExtendToU256(rewriter, loc, adaptor.getLhs(), op.getLhs().getType())
        : coerceToU256(rewriter, loc, adaptor.getLhs());
    auto u256Type = sir::U256Type::get(op.getContext());
    Value shifted = (intType && intType.getIsSigned())
        ? rewriter.create<sir::SarOp>(loc, u256Type, shift, value).getResult()
        : rewriter.create<sir::ShrOp>(loc, u256Type, shift, value).getResult();
    auto *typeConverter = getTypeConverter();
    if (!typeConverter) return rewriter.notifyMatchFailure(op, "missing type converter");
    auto resultType = typeConverter->convertType(op.getResult().getType());
    if (!resultType) return rewriter.notifyMatchFailure(op, "unable to convert shr_wrapping result type");
    if (isU256IntegerCarrierType(resultType)) rewriter.replaceOp(op, shifted);
    else rewriter.replaceOp(op, rewriter.create<sir::BitcastOp>(loc, resultType, shifted).getResult());
    return success();
}

// -----------------------------------------------------------------------------
// Convert arith.select → sir.select
// -----------------------------------------------------------------------------
LogicalResult ConvertArithSelectOp::matchAndRewrite(
    mlir::arith::SelectOp op,
    typename mlir::arith::SelectOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    auto *typeConverter = getTypeConverter();
    if (!typeConverter)
    {
        return rewriter.notifyMatchFailure(op, "missing type converter");
    }

    auto resultType = typeConverter->convertType(op.getResult().getType());
    if (!resultType)
    {
        return rewriter.notifyMatchFailure(op, "unable to convert select result type");
    }

    auto loc = op.getLoc();
    auto u256Type = sir::U256Type::get(op.getContext());
    Value cond = coerceToU256(rewriter, loc, adaptor.getCondition());
    Value trueVal = coerceToU256(rewriter, loc, adaptor.getTrueValue());
    Value falseVal = coerceToU256(rewriter, loc, adaptor.getFalseValue());

    // SIR select is strictly u256-typed. Cast the selected word back when the
    // converted target type is narrower/non-u256.
    Value selected = rewriter.create<sir::SelectOp>(loc, u256Type, cond, trueVal, falseVal).getResult();
    if (!isU256IntegerCarrierType(resultType))
    {
        selected = rewriter.create<sir::BitcastOp>(loc, resultType, selected).getResult();
    }
    rewriter.replaceOp(op, selected);
    return success();
}

// -----------------------------------------------------------------------------
// Convert arith.extui → sir.bitcast (u256)
// -----------------------------------------------------------------------------
LogicalResult ConvertArithExtUIOp::matchAndRewrite(
    mlir::arith::ExtUIOp op,
    typename mlir::arith::ExtUIOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    auto *typeConverter = getTypeConverter();
    if (!typeConverter)
    {
        return rewriter.notifyMatchFailure(op, "missing type converter");
    }
    auto resultType = typeConverter->convertType(op.getResult().getType());
    if (!resultType)
    {
        return rewriter.notifyMatchFailure(op, "unable to convert extui result type");
    }

    auto newOp = rewriter.create<sir::BitcastOp>(op.getLoc(), resultType, adaptor.getIn());
    rewriter.replaceOp(op, newOp.getResult());
    return success();
}

// -----------------------------------------------------------------------------
// Convert arith.extsi → sign extension through shl/sar
// -----------------------------------------------------------------------------
LogicalResult ConvertArithExtSIOp::matchAndRewrite(
    mlir::arith::ExtSIOp op,
    typename mlir::arith::ExtSIOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    auto *typeConverter = getTypeConverter();
    if (!typeConverter)
    {
        return rewriter.notifyMatchFailure(op, "missing type converter");
    }
    auto resultType = typeConverter->convertType(op.getResult().getType());
    if (!resultType)
    {
        return rewriter.notifyMatchFailure(op, "unable to convert extsi result type");
    }

    auto loc = op.getLoc();
    auto u256Type = sir::U256Type::get(rewriter.getContext());
    Value input = coerceToU256(rewriter, loc, adaptor.getIn());

    unsigned sourceWidth = op.getIn().getType().getIntOrFloatBitWidth();
    if (sourceWidth < 256)
    {
        llvm::APInt shiftAmount(256, 256 - sourceWidth);
        Value shift = constU256(rewriter, loc, shiftAmount);
        Value shiftedLeft = rewriter.create<sir::ShlOp>(loc, u256Type, shift, input).getResult();
        input = rewriter.create<sir::SarOp>(loc, u256Type, shift, shiftedLeft).getResult();
    }

    if (!isU256IntegerCarrierType(resultType))
    {
        input = rewriter.create<sir::BitcastOp>(loc, resultType, input).getResult();
    }
    rewriter.replaceOp(op, input);
    return success();
}

// -----------------------------------------------------------------------------
// Convert arith.index_castui → sir.bitcast
// -----------------------------------------------------------------------------
LogicalResult ConvertArithIndexCastUIOp::matchAndRewrite(
    mlir::arith::IndexCastUIOp op,
    typename mlir::arith::IndexCastUIOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    auto *typeConverter = getTypeConverter();
    if (!typeConverter)
    {
        return rewriter.notifyMatchFailure(op, "missing type converter");
    }
    auto resultType = typeConverter->convertType(op.getResult().getType());
    if (!resultType)
    {
        return rewriter.notifyMatchFailure(op, "unable to convert index_castui result type");
    }

    auto newOp = rewriter.create<sir::BitcastOp>(op.getLoc(), resultType, adaptor.getIn());
    rewriter.replaceOp(op, newOp.getResult());
    return success();
}

// -----------------------------------------------------------------------------
// Convert arith.index_cast → sir.bitcast
// -----------------------------------------------------------------------------
LogicalResult ConvertArithIndexCastOp::matchAndRewrite(
    mlir::arith::IndexCastOp op,
    typename mlir::arith::IndexCastOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    auto *typeConverter = getTypeConverter();
    if (!typeConverter)
    {
        return rewriter.notifyMatchFailure(op, "missing type converter");
    }
    auto resultType = typeConverter->convertType(op.getResult().getType());
    if (!resultType)
    {
        return rewriter.notifyMatchFailure(op, "unable to convert index_cast result type");
    }

    auto newOp = rewriter.create<sir::BitcastOp>(op.getLoc(), resultType, adaptor.getIn());
    rewriter.replaceOp(op, newOp.getResult());
    return success();
}

// -----------------------------------------------------------------------------
// Convert arith.trunci → sir.and with bitmask (clears high bits on EVM u256)
// -----------------------------------------------------------------------------
LogicalResult ConvertArithTruncIOp::matchAndRewrite(
    mlir::arith::TruncIOp op,
    typename mlir::arith::TruncIOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    auto *typeConverter = getTypeConverter();
    if (!typeConverter)
        return rewriter.notifyMatchFailure(op, "missing type converter");
    auto resultType = typeConverter->convertType(op.getResult().getType());
    if (!resultType)
        return rewriter.notifyMatchFailure(op, "unable to convert trunci result type");

    auto loc = op.getLoc();
    auto u256Type = sir::U256Type::get(rewriter.getContext());
    Value input = adaptor.getIn();

    // Determine the target bit-width to mask to.
    unsigned targetWidth = op.getType().getIntOrFloatBitWidth();
    if (targetWidth >= 256)
    {
        // No masking needed — just bitcast.
        rewriter.replaceOp(op, rewriter.create<sir::BitcastOp>(loc, resultType, input));
        return success();
    }

    // Emit AND with low-bits mask: (1 << targetWidth) - 1
    llvm::APInt mask = llvm::APInt::getLowBitsSet(256, targetWidth);
    Value maskConst = constU256(rewriter, loc, mask);
    Value inputU256 = coerceToU256(rewriter, loc, input);
    Value masked = rewriter.create<sir::AndOp>(loc, u256Type, inputU256, maskConst);

    if (!isU256IntegerCarrierType(resultType))
        masked = rewriter.create<sir::BitcastOp>(loc, resultType, masked);
    rewriter.replaceOp(op, masked);
    return success();
}

//===- SIRDialect.cpp - SIR Dialect implementation --------------------===//
//
// This file implements the SIR dialect and registers its types/ops.
//
//===----------------------------------------------------------------------===//

#include "SIRDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/TypeSwitch.h"

#include <optional>

using namespace mlir;
using namespace sir;

static constexpr llvm::StringLiteral kSIRResultNameAttr = "sir.result_name_0";

// Include the generated type definitions
#define GET_TYPEDEF_CLASSES
#include "SIRTypes.cpp.inc"
#undef GET_TYPEDEF_CLASSES

// Include the generated operation definitions
#define GET_OP_CLASSES
#include "SIROps.cpp.inc"
#undef GET_OP_CLASSES

//===----------------------------------------------------------------------===//
// ConstOp Custom Parser/Printer
//===----------------------------------------------------------------------===//

::mlir::ParseResult ConstOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result)
{
    // Parse format: value attr-dict : type
    llvm::APInt value;
    if (parser.parseInteger(value))
        return ::mlir::failure();

    // Parse optional attributes (including sir.result_name_0)
    if (parser.parseOptionalAttrDict(result.attributes))
        return ::mlir::failure();

    // Parse colon and type
    ::mlir::Type resultType;
    if (parser.parseColonType(resultType))
        return ::mlir::failure();
    result.addTypes(resultType);

    llvm::APInt value256 = value.isNegative() ? value.sextOrTrunc(256) : value.zextOrTrunc(256);
    auto intType = ::mlir::IntegerType::get(parser.getContext(), 256);
    result.addAttribute("value", ::mlir::IntegerAttr::get(intType, value256));

    return ::mlir::success();
}

void ConstOp::print(::mlir::OpAsmPrinter &p)
{
    p << " ";
    llvm::SmallString<80> text;
    getValueAttr().getValue().zextOrTrunc(256).toString(text, 10, false);
    p << text;

    // Print attributes, but elide "value" (property) and internal result name attribute
    SmallVector<StringRef> elidedAttrs = {"value", kSIRResultNameAttr};
    p.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);

    p << " : ";
    p << getResult().getType();
}

static void printAttrsElidingResultName(::mlir::OpAsmPrinter &p, Operation *op)
{
    SmallVector<StringRef> elidedAttrs = {kSIRResultNameAttr};
    p.printOptionalAttrDict(op->getAttrs(), elidedAttrs);
}

static void setNamedResult(Operation *op, Value result,
                           ::mlir::OpAsmSetValueNameFn setNameFn)
{
    if (auto nameAttr = op->getAttrOfType<::mlir::StringAttr>(kSIRResultNameAttr))
        setNameFn(result, nameAttr.getValue());
}

static ::mlir::ParseResult parseUnaryResultOp(::mlir::OpAsmParser &parser,
                                              ::mlir::OperationState &result)
{
    ::mlir::OpAsmParser::UnresolvedOperand operand;
    Type operandType, resultType;
    if (parser.parseOperand(operand) || parser.parseColonType(operandType) ||
        parser.parseOptionalAttrDict(result.attributes) ||
        parser.parseColonType(resultType))
        return ::mlir::failure();

    result.addTypes(resultType);
    return parser.resolveOperand(operand, operandType, result.operands);
}

static ::mlir::ParseResult parseBinaryResultOp(::mlir::OpAsmParser &parser,
                                               ::mlir::OperationState &result)
{
    ::mlir::OpAsmParser::UnresolvedOperand lhs, rhs;
    Type lhsType, rhsType, resultType;
    if (parser.parseOperand(lhs) || parser.parseColonType(lhsType) ||
        parser.parseComma() || parser.parseOperand(rhs) ||
        parser.parseColonType(rhsType) ||
        parser.parseOptionalAttrDict(result.attributes) ||
        parser.parseColonType(resultType))
        return ::mlir::failure();

    result.addTypes(resultType);
    if (parser.resolveOperand(lhs, lhsType, result.operands) ||
        parser.resolveOperand(rhs, rhsType, result.operands))
        return ::mlir::failure();
    return ::mlir::success();
}

static void printUnaryResultOp(::mlir::OpAsmPrinter &p, Operation *op, Value operand,
                               Type resultType)
{
    p << " ";
    p << operand;
    p << " : ";
    p << operand.getType();
    printAttrsElidingResultName(p, op);
    p << " : ";
    p << resultType;
}

static void printBinaryResultOp(::mlir::OpAsmPrinter &p, Operation *op, Value lhs,
                                Value rhs, Type resultType)
{
    p << " ";
    p << lhs;
    p << " : ";
    p << lhs.getType();
    p << ", ";
    p << rhs;
    p << " : ";
    p << rhs.getType();
    printAttrsElidingResultName(p, op);
    p << " : ";
    p << resultType;
}

#define DEFINE_SIR_SINGLE_RESULT_NAME(OpT)                                           \
    void OpT::getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn)               \
    {                                                                                \
        setNamedResult(*this, getResult(), setNameFn);                               \
    }

DEFINE_SIR_SINGLE_RESULT_NAME(ConstOp)
DEFINE_SIR_SINGLE_RESULT_NAME(AddOp)
DEFINE_SIR_SINGLE_RESULT_NAME(MulOp)
DEFINE_SIR_SINGLE_RESULT_NAME(MallocOp)
DEFINE_SIR_SINGLE_RESULT_NAME(AddPtrOp)
DEFINE_SIR_SINGLE_RESULT_NAME(KeccakOp)
DEFINE_SIR_SINGLE_RESULT_NAME(SLoadOp)

#undef DEFINE_SIR_SINGLE_RESULT_NAME

//===----------------------------------------------------------------------===//
// ConstOp Folding
//===----------------------------------------------------------------------===//

::mlir::OpFoldResult ConstOp::fold(FoldAdaptor adaptor)
{
    // Just return the constant attribute; this makes the op foldable and DCE-able
    return getValueAttr();
}

static std::optional<llvm::APInt> getSIRConstantAPInt(Value value)
{
    if (auto constOp = value.getDefiningOp<ConstOp>())
    {
        if (auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(constOp.getValueAttr()))
            return intAttr.getValue();
    }
    return std::nullopt;
}

static ::mlir::Attribute getSIRConstAttr(MLIRContext *ctx, const llvm::APInt &value)
{
    auto intType = ::mlir::IntegerType::get(ctx, 256);
    return ::mlir::IntegerAttr::get(intType, value.zextOrTrunc(256));
}

enum class BinaryConstFoldKind
{
    Add,
    Sub,
    Mul,
    Eq,
    Lt,
    Gt,
    SLt,
    SGt,
    And,
    Or,
    Xor,
};

enum class ShiftConstFoldKind
{
    Byte,
    Shl,
    Shr,
    Sar,
};

enum class UnaryConstFoldKind
{
    Not,
    IsZero,
};

static llvm::APInt foldSIRBinaryAPInts(BinaryConstFoldKind kind,
                                       const llvm::APInt &lhs,
                                       const llvm::APInt &rhs)
{
    llvm::APInt lhs256 = lhs.zextOrTrunc(256);
    llvm::APInt rhs256 = rhs.zextOrTrunc(256);

    switch (kind)
    {
    case BinaryConstFoldKind::Add:
        return lhs256 + rhs256;
    case BinaryConstFoldKind::Sub:
        return lhs256 - rhs256;
    case BinaryConstFoldKind::Mul:
        return lhs256 * rhs256;
    case BinaryConstFoldKind::Eq:
        return llvm::APInt(256, lhs256 == rhs256 ? 1 : 0);
    case BinaryConstFoldKind::Lt:
        return llvm::APInt(256, lhs256.ult(rhs256) ? 1 : 0);
    case BinaryConstFoldKind::Gt:
        return llvm::APInt(256, lhs256.ugt(rhs256) ? 1 : 0);
    case BinaryConstFoldKind::SLt:
        return llvm::APInt(256, lhs256.slt(rhs256) ? 1 : 0);
    case BinaryConstFoldKind::SGt:
        return llvm::APInt(256, lhs256.sgt(rhs256) ? 1 : 0);
    case BinaryConstFoldKind::And:
        return lhs256 & rhs256;
    case BinaryConstFoldKind::Or:
        return lhs256 | rhs256;
    case BinaryConstFoldKind::Xor:
        return lhs256 ^ rhs256;
    }

    llvm_unreachable("unknown SIR binary constant fold kind");
}

static llvm::APInt foldSIRShiftAPInts(ShiftConstFoldKind kind,
                                      const llvm::APInt &shift,
                                      const llvm::APInt &value)
{
    llvm::APInt shift256 = shift.zextOrTrunc(256);
    llvm::APInt value256 = value.zextOrTrunc(256);

    switch (kind)
    {
    case ShiftConstFoldKind::Byte:
        if (shift256.uge(llvm::APInt(256, 32)))
            return llvm::APInt(256, 0);
        return value256.lshr((31 - shift256.getZExtValue()) * 8)
            .zextOrTrunc(8)
            .zext(256);
    case ShiftConstFoldKind::Shl:
        if (shift256.uge(llvm::APInt(256, 256)))
            return llvm::APInt(256, 0);
        return value256.shl(shift256.getZExtValue());
    case ShiftConstFoldKind::Shr:
        if (shift256.uge(llvm::APInt(256, 256)))
            return llvm::APInt(256, 0);
        return value256.lshr(shift256.getZExtValue());
    case ShiftConstFoldKind::Sar:
        if (shift256.uge(llvm::APInt(256, 256)))
            return value256.isNegative()
                       ? llvm::APInt::getAllOnes(256)
                       : llvm::APInt(256, 0);
        return value256.ashr(shift256.getZExtValue());
    }

    llvm_unreachable("unknown SIR shift constant fold kind");
}

static llvm::APInt foldSIRUnaryAPInt(UnaryConstFoldKind kind,
                                     const llvm::APInt &value)
{
    llvm::APInt value256 = value.zextOrTrunc(256);

    switch (kind)
    {
    case UnaryConstFoldKind::Not:
        return ~value256;
    case UnaryConstFoldKind::IsZero:
        return llvm::APInt(256, value256.isZero() ? 1 : 0);
    }

    llvm_unreachable("unknown SIR unary constant fold kind");
}

static ::mlir::OpFoldResult foldSIRBinaryConstants(Value lhs, Value rhs,
                                                   MLIRContext *ctx,
                                                   BinaryConstFoldKind kind)
{
    auto lhsVal = getSIRConstantAPInt(lhs);
    auto rhsVal = getSIRConstantAPInt(rhs);
    if (!lhsVal || !rhsVal)
        return {};

    return getSIRConstAttr(ctx, foldSIRBinaryAPInts(kind, *lhsVal, *rhsVal));
}

static ::mlir::OpFoldResult foldSIRUnaryConstant(Value value, MLIRContext *ctx,
                                                 UnaryConstFoldKind kind)
{
    auto constant = getSIRConstantAPInt(value);
    if (!constant)
        return {};

    return getSIRConstAttr(ctx, foldSIRUnaryAPInt(kind, *constant));
}

static ::mlir::OpFoldResult foldSIRShiftConstants(Value shift, Value value,
                                                  MLIRContext *ctx,
                                                  ShiftConstFoldKind kind)
{
    auto shiftValue = getSIRConstantAPInt(shift);
    auto inputValue = getSIRConstantAPInt(value);
    if (!shiftValue || !inputValue)
        return {};

    return getSIRConstAttr(
        ctx,
        foldSIRShiftAPInts(kind, *shiftValue, *inputValue));
}

//===----------------------------------------------------------------------===//
// BitcastOp Folding
//===----------------------------------------------------------------------===//

static std::optional<unsigned> getSIRBitcastPayloadWidth(Type type)
{
    if (auto intType = llvm::dyn_cast<IntegerType>(type))
        return intType.getWidth();
    if (llvm::isa<U256Type, PtrType>(type))
        return 256u;
    return std::nullopt;
}

static bool hasSameKnownSIRBitcastPayloadWidth(Type lhs, Type rhs)
{
    auto lhsWidth = getSIRBitcastPayloadWidth(lhs);
    auto rhsWidth = getSIRBitcastPayloadWidth(rhs);
    return lhsWidth && rhsWidth && *lhsWidth == *rhsWidth;
}

static bool hasFoldBlockingAttrs(Operation *op)
{
    // Bitcast attrs are sometimes used as lowering breadcrumbs. The framework
    // folder is intentionally narrower than OraToSIR's final safety-net fold:
    // do not let a dialect-local fold erase metadata unless a later slice proves
    // that attribute class is discardable.
    return !op->getAttrs().empty();
}

::mlir::OpFoldResult BitcastOp::fold(FoldAdaptor adaptor)
{
    if (hasFoldBlockingAttrs(getOperation()))
        return {};

    Value input = getInput();
    Type inputType = input.getType();
    Type resultType = getResult().getType();

    if (inputType == resultType)
        return input;

    auto inner = input.getDefiningOp<BitcastOp>();
    if (!inner || hasFoldBlockingAttrs(inner.getOperation()))
        return {};

    Type startType = inner.getInput().getType();
    Type middleType = inputType;
    if (startType == resultType && hasSameKnownSIRBitcastPayloadWidth(startType, middleType))
        return inner.getInput();

    return {};
}

#define DEFINE_SIR_BINARY_CONST_FOLD(OpT, Kind)                                      \
    ::mlir::OpFoldResult OpT::fold(FoldAdaptor adaptor)                              \
    {                                                                                \
        return foldSIRBinaryConstants(getLhs(), getRhs(), getContext(),              \
                                      BinaryConstFoldKind::Kind);                    \
    }

#define DEFINE_SIR_UNARY_CONST_FOLD(OpT, Operand, Kind)                              \
    ::mlir::OpFoldResult OpT::fold(FoldAdaptor adaptor)                              \
    {                                                                                \
        return foldSIRUnaryConstant(Operand(), getContext(),                         \
                                    UnaryConstFoldKind::Kind);                       \
    }

#define DEFINE_SIR_SHIFT_CONST_FOLD(OpT, ShiftOperand, ValueOperand, Kind)            \
    ::mlir::OpFoldResult OpT::fold(FoldAdaptor adaptor)                              \
    {                                                                                \
        return foldSIRShiftConstants(ShiftOperand(), ValueOperand(), getContext(),    \
                                     ShiftConstFoldKind::Kind);                      \
    }

DEFINE_SIR_BINARY_CONST_FOLD(AddOp, Add)
DEFINE_SIR_BINARY_CONST_FOLD(SubOp, Sub)
DEFINE_SIR_BINARY_CONST_FOLD(MulOp, Mul)
DEFINE_SIR_BINARY_CONST_FOLD(LtOp, Lt)
DEFINE_SIR_BINARY_CONST_FOLD(GtOp, Gt)
DEFINE_SIR_BINARY_CONST_FOLD(SLtOp, SLt)
DEFINE_SIR_BINARY_CONST_FOLD(SGtOp, SGt)
DEFINE_SIR_BINARY_CONST_FOLD(AndOp, And)
DEFINE_SIR_BINARY_CONST_FOLD(OrOp, Or)
DEFINE_SIR_BINARY_CONST_FOLD(XorOp, Xor)
DEFINE_SIR_UNARY_CONST_FOLD(NotOp, getX, Not)
DEFINE_SIR_UNARY_CONST_FOLD(IsZeroOp, getX, IsZero)
DEFINE_SIR_SHIFT_CONST_FOLD(ByteOp, getI, getX, Byte)
DEFINE_SIR_SHIFT_CONST_FOLD(ShlOp, getShift, getX, Shl)
DEFINE_SIR_SHIFT_CONST_FOLD(ShrOp, getShift, getX, Shr)
DEFINE_SIR_SHIFT_CONST_FOLD(SarOp, getShift, getX, Sar)

#undef DEFINE_SIR_BINARY_CONST_FOLD
#undef DEFINE_SIR_UNARY_CONST_FOLD
#undef DEFINE_SIR_SHIFT_CONST_FOLD

//===----------------------------------------------------------------------===//
// Custom Print/Parse Methods to Elide sir.result_name_0 Attribute
//===----------------------------------------------------------------------===//

::mlir::ParseResult AddOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result)
{
    return parseBinaryResultOp(parser, result);
}

void AddOp::print(::mlir::OpAsmPrinter &p)
{
    printBinaryResultOp(p, *this, getLhs(), getRhs(), getResult().getType());
}

::mlir::ParseResult MulOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result)
{
    return parseBinaryResultOp(parser, result);
}

void MulOp::print(::mlir::OpAsmPrinter &p)
{
    printBinaryResultOp(p, *this, getLhs(), getRhs(), getResult().getType());
}

::mlir::ParseResult MallocOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result)
{
    return parseUnaryResultOp(parser, result);
}

void MallocOp::print(::mlir::OpAsmPrinter &p)
{
    printUnaryResultOp(p, *this, getSize(), getResult().getType());
}

::mlir::ParseResult AddPtrOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result)
{
    return parseBinaryResultOp(parser, result);
}

void AddPtrOp::print(::mlir::OpAsmPrinter &p)
{
    printBinaryResultOp(p, *this, getBase(), getOffset(), getResult().getType());
}

::mlir::ParseResult KeccakOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result)
{
    return parseBinaryResultOp(parser, result);
}

void KeccakOp::print(::mlir::OpAsmPrinter &p)
{
    printBinaryResultOp(p, *this, getPtr(), getLen(), getResult().getType());
}

::mlir::ParseResult SLoadOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result)
{
    return parseUnaryResultOp(parser, result);
}

void SLoadOp::print(::mlir::OpAsmPrinter &p)
{
    printUnaryResultOp(p, *this, getSlot(), getResult().getType());
}

::mlir::OpFoldResult EqOp::fold(FoldAdaptor adaptor)
{
    if (getLhs() == getRhs())
        return getSIRConstAttr(getContext(), llvm::APInt(256, 1));

    return foldSIRBinaryConstants(getLhs(), getRhs(), getContext(),
                                  BinaryConstFoldKind::Eq);
}

void SIRDialect::initialize()
{
    addTypes<
#define GET_TYPEDEF_LIST
#include "SIRTypes.cpp.inc"
        >();
    addOperations<
#define GET_OP_LIST
#include "SIROps.cpp.inc"
        >();
}

//===----------------------------------------------------------------------===//
// Canonicalization Patterns
//===----------------------------------------------------------------------===//

namespace
{
    template <typename OpT, BinaryConstFoldKind Kind>
    struct FoldBinaryConstants : public OpRewritePattern<OpT>
    {
        using OpRewritePattern<OpT>::OpRewritePattern;

        LogicalResult matchAndRewrite(OpT op, PatternRewriter &rewriter) const override
        {
            auto lhsVal = getSIRConstantAPInt(op.getLhs());
            auto rhsVal = getSIRConstantAPInt(op.getRhs());
            if (!lhsVal || !rhsVal)
                return failure();

            llvm::APInt result = foldSIRBinaryAPInts(Kind, *lhsVal, *rhsVal);
            rewriter.replaceOpWithNewOp<ConstOp>(op, result);
            return success();
        }
    };

    struct FoldCondBrSameDest : public OpRewritePattern<CondBrOp>
    {
        using OpRewritePattern<CondBrOp>::OpRewritePattern;

        LogicalResult matchAndRewrite(CondBrOp op, PatternRewriter &rewriter) const override
        {
            if (op.getTrueDest() != op.getFalseDest())
                return failure();
            if (!llvm::equal(op.getTrueOperands(), op.getFalseOperands()))
                return failure();

            rewriter.replaceOpWithNewOp<BrOp>(op, op.getTrueOperands(), op.getTrueDest());
            return success();
        }
    };

    struct FoldCondBrConstant : public OpRewritePattern<CondBrOp>
    {
        using OpRewritePattern<CondBrOp>::OpRewritePattern;

        LogicalResult matchAndRewrite(CondBrOp op, PatternRewriter &rewriter) const override
        {
            auto cond = getSIRConstantAPInt(op.getCond());
            if (!cond)
                return failure();

            if (cond->isZero())
                rewriter.replaceOpWithNewOp<BrOp>(op, op.getFalseOperands(), op.getFalseDest());
            else
                rewriter.replaceOpWithNewOp<BrOp>(op, op.getTrueOperands(), op.getTrueDest());
            return success();
        }
    };

    struct FoldCondBrDoubleIsZero : public OpRewritePattern<CondBrOp>
    {
        using OpRewritePattern<CondBrOp>::OpRewritePattern;

        LogicalResult matchAndRewrite(CondBrOp op, PatternRewriter &rewriter) const override
        {
            auto outer = op.getCond().getDefiningOp<IsZeroOp>();
            if (!outer)
                return failure();
            auto inner = outer.getX().getDefiningOp<IsZeroOp>();
            if (!inner)
                return failure();

            rewriter.replaceOpWithNewOp<CondBrOp>(
                op,
                inner.getX(),
                op.getTrueOperands(),
                op.getFalseOperands(),
                op.getTrueDest(),
                op.getFalseDest());
            return success();
        }
    };

    struct FoldBrToForwardingBr : public OpRewritePattern<BrOp>
    {
        using OpRewritePattern<BrOp>::OpRewritePattern;

        LogicalResult matchAndRewrite(BrOp op, PatternRewriter &rewriter) const override
        {
            Block *dest = op.getDest();
            if (!dest || dest->getOperations().size() != 1)
                return failure();

            auto nextBr = llvm::dyn_cast<BrOp>(dest->front());
            if (!nextBr)
                return failure();
            if (nextBr.getDestOperands().size() != dest->getNumArguments())
                return failure();

            SmallVector<Value, 4> forwardedOperands;
            forwardedOperands.reserve(nextBr.getDestOperands().size());
            for (Value incoming : nextBr.getDestOperands())
            {
                if (auto blockArg = llvm::dyn_cast<BlockArgument>(incoming))
                {
                    if (blockArg.getOwner() != dest)
                        return failure();
                    unsigned idx = blockArg.getArgNumber();
                    if (idx >= op.getDestOperands().size())
                        return failure();
                    forwardedOperands.push_back(op.getDestOperands()[idx]);
                    continue;
                }
                forwardedOperands.push_back(incoming);
            }

            rewriter.replaceOpWithNewOp<BrOp>(op, forwardedOperands, nextBr.getDest());
            return success();
        }
    };
}

#define DEFINE_SIR_BINARY_CONST_CANONICALIZER(OpT, Kind)                                 \
    void OpT::getCanonicalizationPatterns(RewritePatternSet &results, MLIRContext *context) \
    {                                                                                    \
        results.add<FoldBinaryConstants<OpT, BinaryConstFoldKind::Kind>>(context);       \
    }

DEFINE_SIR_BINARY_CONST_CANONICALIZER(AddOp, Add)
DEFINE_SIR_BINARY_CONST_CANONICALIZER(SubOp, Sub)
DEFINE_SIR_BINARY_CONST_CANONICALIZER(MulOp, Mul)
DEFINE_SIR_BINARY_CONST_CANONICALIZER(LtOp, Lt)
DEFINE_SIR_BINARY_CONST_CANONICALIZER(GtOp, Gt)
DEFINE_SIR_BINARY_CONST_CANONICALIZER(SLtOp, SLt)
DEFINE_SIR_BINARY_CONST_CANONICALIZER(SGtOp, SGt)
DEFINE_SIR_BINARY_CONST_CANONICALIZER(AndOp, And)
DEFINE_SIR_BINARY_CONST_CANONICALIZER(OrOp, Or)
DEFINE_SIR_BINARY_CONST_CANONICALIZER(XorOp, Xor)

#undef DEFINE_SIR_BINARY_CONST_CANONICALIZER

void BrOp::getCanonicalizationPatterns(RewritePatternSet &results, MLIRContext *context)
{
    results.add<FoldBrToForwardingBr>(context);
}

void CondBrOp::getCanonicalizationPatterns(RewritePatternSet &results, MLIRContext *context)
{
    results.add<FoldCondBrSameDest, FoldCondBrConstant, FoldCondBrDoubleIsZero>(context);
}

// Attribute parsing/printing - use default implementations
Attribute SIRDialect::parseAttribute(DialectAsmParser &parser, Type type) const
{
    return Dialect::parseAttribute(parser, type);
}

void SIRDialect::printAttribute(Attribute attr, DialectAsmPrinter &printer) const
{
    Dialect::printAttribute(attr, printer);
}

Operation *SIRDialect::materializeConstant(OpBuilder &builder,
                                           Attribute value,
                                           Type type,
                                           Location loc)
{
    if (!llvm::isa<U256Type>(type))
        return nullptr;

    auto intAttr = llvm::dyn_cast<IntegerAttr>(value);
    if (!intAttr)
        return nullptr;

    APInt normalized = intAttr.getValue().zextOrTrunc(256);
    auto attr = IntegerAttr::get(IntegerType::get(builder.getContext(), 256), normalized);
    return builder.create<ConstOp>(loc, attr);
}

// Include the generated dialect definition (TypeID, vtable, constructor, etc.)
#define GET_DIALECT_DEFINITION
#include "SIRDialect.cpp.inc"

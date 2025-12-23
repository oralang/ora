//===- SIRDialect.cpp - SIR Dialect implementation --------------------===//
//
// This file implements the SIR dialect and registers its types/ops.
//
//===----------------------------------------------------------------------===//

#include "SIRDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace sir;

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
    ::mlir::IntegerAttr valueAttr;
    if (parser.parseAttribute(valueAttr, "value", result.attributes))
        return ::mlir::failure();

    // Parse optional attributes (including sir.result_name_0)
    if (parser.parseOptionalAttrDict(result.attributes))
        return ::mlir::failure();

    // Parse colon and type
    ::mlir::Type resultType;
    if (parser.parseColonType(resultType))
        return ::mlir::failure();
    result.addTypes(resultType);

    // Remove "value" from attributes dict - it's a property, not a regular attribute
    // The generated code will convert it to a property, but we don't want it printed
    result.attributes.erase("value");

    return ::mlir::success();
}

void ConstOp::print(::mlir::OpAsmPrinter &p)
{
    p << " ";
    p.printAttributeWithoutType(getValueAttr());

    // Print attributes, but elide "value" (property) and internal result name attribute
    SmallVector<StringRef> elidedAttrs = {"value", "sir.result_name_0"};
    p.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);

    p << " : ";
    p << getResult().getType();
}

//===----------------------------------------------------------------------===//
// ConstOp Result Naming (via OpAsmOpInterface)
//===----------------------------------------------------------------------===//

void ConstOp::getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn)
{
    // Check if we have a stored result name attribute
    auto nameAttr = (*this)->getAttrOfType<::mlir::StringAttr>("sir.result_name_0");
    if (nameAttr)
    {
        setNameFn(getResult(), nameAttr.getValue());
    }
}

//===----------------------------------------------------------------------===//
// ConstOp Folding
//===----------------------------------------------------------------------===//

::mlir::OpFoldResult ConstOp::fold(FoldAdaptor adaptor)
{
    // Just return the constant attribute; this makes the op foldable and DCE-able
    return getValueAttr();
}

//===----------------------------------------------------------------------===//
// AddOp Result Naming
//===----------------------------------------------------------------------===//

void AddOp::getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn)
{
    auto nameAttr = (*this)->getAttrOfType<::mlir::StringAttr>("sir.result_name_0");
    if (nameAttr)
    {
        setNameFn(getResult(), nameAttr.getValue());
    }
}

//===----------------------------------------------------------------------===//
// MulOp Result Naming
//===----------------------------------------------------------------------===//

void MulOp::getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn)
{
    auto nameAttr = (*this)->getAttrOfType<::mlir::StringAttr>("sir.result_name_0");
    if (nameAttr)
    {
        setNameFn(getResult(), nameAttr.getValue());
    }
}

//===----------------------------------------------------------------------===//
// MallocOp Result Naming
//===----------------------------------------------------------------------===//

void MallocOp::getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn)
{
    auto nameAttr = (*this)->getAttrOfType<::mlir::StringAttr>("sir.result_name_0");
    if (nameAttr)
    {
        setNameFn(getResult(), nameAttr.getValue());
    }
}

//===----------------------------------------------------------------------===//
// AddPtrOp Result Naming
//===----------------------------------------------------------------------===//

void AddPtrOp::getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn)
{
    auto nameAttr = (*this)->getAttrOfType<::mlir::StringAttr>("sir.result_name_0");
    if (nameAttr)
    {
        setNameFn(getResult(), nameAttr.getValue());
    }
}

//===----------------------------------------------------------------------===//
// KeccakOp Result Naming
//===----------------------------------------------------------------------===//

void KeccakOp::getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn)
{
    auto nameAttr = (*this)->getAttrOfType<::mlir::StringAttr>("sir.result_name_0");
    if (nameAttr)
    {
        setNameFn(getResult(), nameAttr.getValue());
    }
}

//===----------------------------------------------------------------------===//
// SLoadOp Result Naming
//===----------------------------------------------------------------------===//

void SLoadOp::getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn)
{
    auto nameAttr = (*this)->getAttrOfType<::mlir::StringAttr>("sir.result_name_0");
    if (nameAttr)
    {
        setNameFn(getResult(), nameAttr.getValue());
    }
}

//===----------------------------------------------------------------------===//
// Custom Print/Parse Methods to Elide sir.result_name_0 Attribute
//===----------------------------------------------------------------------===//

::mlir::ParseResult AddOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result)
{
    ::mlir::OpAsmParser::UnresolvedOperand lhs, rhs;
    Type lhsType, rhsType, resType;
    if (parser.parseOperand(lhs) || parser.parseColonType(lhsType) ||
        parser.parseComma() || parser.parseOperand(rhs) ||
        parser.parseColonType(rhsType) ||
        parser.parseOptionalAttrDict(result.attributes) ||
        parser.parseColonType(resType))
        return ::mlir::failure();
    result.addTypes(resType);
    if (parser.resolveOperand(lhs, lhsType, result.operands) ||
        parser.resolveOperand(rhs, rhsType, result.operands))
        return ::mlir::failure();
    return ::mlir::success();
}

void AddOp::print(::mlir::OpAsmPrinter &p)
{
    p << " ";
    p << getLhs();
    p << " : ";
    p << getLhs().getType();
    p << ", ";
    p << getRhs();
    p << " : ";
    p << getRhs().getType();
    SmallVector<StringRef> elidedAttrs = {"sir.result_name_0"};
    p.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
    p << " : ";
    p << getResult().getType();
}

::mlir::ParseResult MulOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result)
{
    ::mlir::OpAsmParser::UnresolvedOperand lhs, rhs;
    Type lhsType, rhsType, resType;
    if (parser.parseOperand(lhs) || parser.parseColonType(lhsType) ||
        parser.parseComma() || parser.parseOperand(rhs) ||
        parser.parseColonType(rhsType) ||
        parser.parseOptionalAttrDict(result.attributes) ||
        parser.parseColonType(resType))
        return ::mlir::failure();
    result.addTypes(resType);
    if (parser.resolveOperand(lhs, lhsType, result.operands) ||
        parser.resolveOperand(rhs, rhsType, result.operands))
        return ::mlir::failure();
    return ::mlir::success();
}

void MulOp::print(::mlir::OpAsmPrinter &p)
{
    p << " ";
    p << getLhs();
    p << " : ";
    p << getLhs().getType();
    p << ", ";
    p << getRhs();
    p << " : ";
    p << getRhs().getType();
    SmallVector<StringRef> elidedAttrs = {"sir.result_name_0"};
    p.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
    p << " : ";
    p << getResult().getType();
}

::mlir::ParseResult MallocOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result)
{
    ::mlir::OpAsmParser::UnresolvedOperand size;
    Type sizeType, resType;
    if (parser.parseOperand(size) || parser.parseColonType(sizeType) ||
        parser.parseOptionalAttrDict(result.attributes) ||
        parser.parseColonType(resType))
        return ::mlir::failure();
    result.addTypes(resType);
    if (parser.resolveOperand(size, sizeType, result.operands))
        return ::mlir::failure();
    return ::mlir::success();
}

void MallocOp::print(::mlir::OpAsmPrinter &p)
{
    p << " ";
    p << getSize();
    p << " : ";
    p << getSize().getType();
    SmallVector<StringRef> elidedAttrs = {"sir.result_name_0"};
    p.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
    p << " : ";
    p << getResult().getType();
}

::mlir::ParseResult AddPtrOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result)
{
    ::mlir::OpAsmParser::UnresolvedOperand base, offset;
    Type baseType, offsetType, resType;
    if (parser.parseOperand(base) || parser.parseColonType(baseType) ||
        parser.parseComma() || parser.parseOperand(offset) ||
        parser.parseColonType(offsetType) ||
        parser.parseOptionalAttrDict(result.attributes) ||
        parser.parseColonType(resType))
        return ::mlir::failure();
    result.addTypes(resType);
    if (parser.resolveOperand(base, baseType, result.operands) ||
        parser.resolveOperand(offset, offsetType, result.operands))
        return ::mlir::failure();
    return ::mlir::success();
}

void AddPtrOp::print(::mlir::OpAsmPrinter &p)
{
    p << " ";
    p << getBase();
    p << " : ";
    p << getBase().getType();
    p << ", ";
    p << getOffset();
    p << " : ";
    p << getOffset().getType();
    SmallVector<StringRef> elidedAttrs = {"sir.result_name_0"};
    p.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
    p << " : ";
    p << getResult().getType();
}

::mlir::ParseResult KeccakOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result)
{
    ::mlir::OpAsmParser::UnresolvedOperand ptr, len;
    Type ptrType, lenType, resType;
    if (parser.parseOperand(ptr) || parser.parseColonType(ptrType) ||
        parser.parseComma() || parser.parseOperand(len) ||
        parser.parseColonType(lenType) ||
        parser.parseOptionalAttrDict(result.attributes) ||
        parser.parseColonType(resType))
        return ::mlir::failure();
    result.addTypes(resType);
    if (parser.resolveOperand(ptr, ptrType, result.operands) ||
        parser.resolveOperand(len, lenType, result.operands))
        return ::mlir::failure();
    return ::mlir::success();
}

void KeccakOp::print(::mlir::OpAsmPrinter &p)
{
    p << " ";
    p << getPtr();
    p << " : ";
    p << getPtr().getType();
    p << ", ";
    p << getLen();
    p << " : ";
    p << getLen().getType();
    SmallVector<StringRef> elidedAttrs = {"sir.result_name_0"};
    p.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
    p << " : ";
    p << getResult().getType();
}

::mlir::ParseResult SLoadOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result)
{
    ::mlir::OpAsmParser::UnresolvedOperand slot;
    Type slotType, resType;
    if (parser.parseOperand(slot) || parser.parseColonType(slotType) ||
        parser.parseOptionalAttrDict(result.attributes) ||
        parser.parseColonType(resType))
        return ::mlir::failure();
    result.addTypes(resType);
    if (parser.resolveOperand(slot, slotType, result.operands))
        return ::mlir::failure();
    return ::mlir::success();
}

void SLoadOp::print(::mlir::OpAsmPrinter &p)
{
    p << " ";
    p << getSlot();
    p << " : ";
    p << getSlot().getType();
    SmallVector<StringRef> elidedAttrs = {"sir.result_name_0"};
    p.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
    p << " : ";
    p << getResult().getType();
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
    // Pattern to fold constant addition: add(const, const) -> const
    struct FoldAddConstants : public OpRewritePattern<AddOp>
    {
        using OpRewritePattern<AddOp>::OpRewritePattern;

        LogicalResult matchAndRewrite(AddOp op, PatternRewriter &rewriter) const override
        {
            // Try to get constant values from operands
            // They might be direct ConstOp results or come from other operations
            auto getConstantValue = [](Value val) -> std::optional<uint64_t>
            {
                if (auto constOp = val.getDefiningOp<ConstOp>())
                {
                    auto attr = constOp.getValueAttr();
                    if (auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(attr))
                    {
                        return intAttr.getValue().getZExtValue();
                    }
                }
                return std::nullopt;
            };

            auto lhsVal = getConstantValue(op.getLhs());
            auto rhsVal = getConstantValue(op.getRhs());

            if (!lhsVal || !rhsVal)
                return failure();

            uint64_t result = *lhsVal + *rhsVal;
            auto u256Type = sir::U256Type::get(op.getContext());
            auto ui64Type = mlir::IntegerType::get(op.getContext(), 64, mlir::IntegerType::Unsigned);
            auto valueAttr = mlir::IntegerAttr::get(ui64Type, result);
            auto newConst = rewriter.create<ConstOp>(op.getLoc(), u256Type, valueAttr);
            rewriter.replaceOp(op, newConst.getResult());
            return success();
        }
    };

    // Pattern to fold constant multiplication: mul(const, const) -> const
    struct FoldMulConstants : public OpRewritePattern<MulOp>
    {
        using OpRewritePattern<MulOp>::OpRewritePattern;

        LogicalResult matchAndRewrite(MulOp op, PatternRewriter &rewriter) const override
        {
            // Try to get constant values from operands
            // They might be direct ConstOp results or come from other operations
            auto getConstantValue = [](Value val) -> std::optional<uint64_t>
            {
                if (auto constOp = val.getDefiningOp<ConstOp>())
                {
                    auto attr = constOp.getValueAttr();
                    if (auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(attr))
                    {
                        return intAttr.getValue().getZExtValue();
                    }
                }
                return std::nullopt;
            };

            auto lhsVal = getConstantValue(op.getLhs());
            auto rhsVal = getConstantValue(op.getRhs());

            if (!lhsVal || !rhsVal)
                return failure();

            uint64_t result = *lhsVal * *rhsVal;
            auto u256Type = sir::U256Type::get(op.getContext());
            auto ui64Type = mlir::IntegerType::get(op.getContext(), 64, mlir::IntegerType::Unsigned);
            auto valueAttr = mlir::IntegerAttr::get(ui64Type, result);
            auto newConst = rewriter.create<ConstOp>(op.getLoc(), u256Type, valueAttr);
            rewriter.replaceOp(op, newConst.getResult());
            return success();
        }
    };
}

void AddOp::getCanonicalizationPatterns(RewritePatternSet &results, MLIRContext *context)
{
    results.add<FoldAddConstants>(context);
}

void MulOp::getCanonicalizationPatterns(RewritePatternSet &results, MLIRContext *context)
{
    results.add<FoldMulConstants>(context);
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

// Include the generated dialect definition (TypeID, vtable, constructor, etc.)
#define GET_DIALECT_DEFINITION
#include "SIRDialect.cpp.inc"

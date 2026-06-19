//===- OraDialect.cpp - Ora dialect implementation ----------------------===//
//
// This file implements the Ora MLIR dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#include "OraDialect.h"

// Type definitions
#define GET_TYPEDEF_CLASSES
#include "OraTypes.cpp.inc"

namespace mlir
{
    namespace ora
    {
        /// Initialize the Ora dialect and add operations and types
        void OraDialect::initialize()
        {
            addOperations<
#define GET_OP_LIST
#include "OraOps.cpp.inc"
                >();
            addTypes<
#define GET_TYPEDEF_LIST
#include "OraTypes.cpp.inc"
                >();
        }

        ::mlir::Attribute OraDialect::parseAttribute(::mlir::DialectAsmParser &parser, ::mlir::Type type) const
        {
            return ::mlir::Dialect::parseAttribute(parser, type);
        }

        void OraDialect::printAttribute(::mlir::Attribute attr, ::mlir::DialectAsmPrinter &printer) const
        {
            ::mlir::Dialect::printAttribute(attr, printer);
        }

    } // namespace ora
} // namespace mlir

namespace mlir
{
    namespace ora
    {
        namespace
        {
            static bool nextTokenIsEllipsis(::mlir::OpAsmParser &parser)
            {
                const char *ptr = parser.getCurrentLocation().getPointer();
                return ptr[0] == '.' && ptr[1] == '.' && ptr[2] == '.';
            }

            static ::mlir::ParseResult parseFatArrow(::mlir::OpAsmParser &parser)
            {
                const char *ptr = parser.getCurrentLocation().getPointer();
                if (ptr[0] != '=' || ptr[1] != '>')
                {
                    parser.emitError(parser.getCurrentLocation(), "expected '=>'");
                    return ::mlir::failure();
                }
                if (parser.parseEqual() || parser.parseGreater())
                    return ::mlir::failure();
                return ::mlir::success();
            }

            static ::mlir::Type getRefinementBaseType(::mlir::Type type)
            {
                if (auto minType = llvm::dyn_cast<MinValueType>(type))
                    return minType.getBaseType();
                if (auto maxType = llvm::dyn_cast<MaxValueType>(type))
                    return maxType.getBaseType();
                if (auto rangeType = llvm::dyn_cast<InRangeType>(type))
                    return rangeType.getBaseType();
                if (auto scaledType = llvm::dyn_cast<ScaledType>(type))
                    return scaledType.getBaseType();
                if (auto exactType = llvm::dyn_cast<ExactType>(type))
                    return exactType.getBaseType();
                if (llvm::isa<NonZeroAddressType>(type))
                    return AddressType::get(type.getContext());
                return {};
            }

            static bool isIntegerWidth(::mlir::Type type, unsigned width)
            {
                if (auto base = getRefinementBaseType(type))
                    type = base;
                if (auto oraInt = llvm::dyn_cast<ora::IntegerType>(type))
                    return oraInt.getWidth() == width;
                if (auto builtinInt = llvm::dyn_cast<::mlir::IntegerType>(type))
                    return builtinInt.getWidth() == width;
                return false;
            }

            static bool isComputedStorageWordType(::mlir::Type type)
            {
                return isIntegerWidth(type, 256);
            }

            static bool isComputedStorageKeyType(::mlir::Type type)
            {
                if (auto base = getRefinementBaseType(type))
                    type = base;
                return isComputedStorageWordType(type) ||
                       llvm::isa<AddressType, NonZeroAddressType>(type);
            }

            static ::mlir::OpFoldResult foldAddressCarrierRoundTrip(::mlir::Value value, ::mlir::Type resultType)
            {
                if (value.getType() == resultType)
                    return value;
                return {};
            }

            template <typename SwitchLikeOp>
            static ::mlir::LogicalResult verifySwitchLikeOp(SwitchLikeOp op)
            {
                const size_t numCases = op.getCases().size();
                auto caseValuesAttr = op.getCaseValuesAttr();
                auto rangeStartsAttr = op.getRangeStartsAttr();
                auto rangeEndsAttr = op.getRangeEndsAttr();
                auto caseKindsAttr = op.getCaseKindsAttr();
                auto defaultIndexAttr = op.getDefaultCaseIndexAttr();

                auto requireSizedAttr = [&](auto attr, llvm::StringRef name) -> ::mlir::LogicalResult
                {
                    if (!attr)
                        return op.emitOpError() << "requires '" << name << "' metadata for all case regions";
                    if (static_cast<size_t>(attr.size()) != numCases)
                        return op.emitOpError() << "requires '" << name << "' to have " << numCases
                                                << " entries, but found " << attr.size();
                    return ::mlir::success();
                };

                if (numCases > 0)
                {
                    if (::mlir::failed(requireSizedAttr(caseValuesAttr, "case_values")) ||
                        ::mlir::failed(requireSizedAttr(rangeStartsAttr, "range_starts")) ||
                        ::mlir::failed(requireSizedAttr(rangeEndsAttr, "range_ends")) ||
                        ::mlir::failed(requireSizedAttr(caseKindsAttr, "case_kinds")))
                        return ::mlir::failure();
                }

                int64_t defaultIndex = -1;
                if (defaultIndexAttr)
                {
                    defaultIndex = defaultIndexAttr.getInt();
                    if (defaultIndex < 0 || static_cast<size_t>(defaultIndex) >= numCases)
                        return op.emitOpError() << "requires 'default_case_index' to be in [0, " << numCases
                                                << "), but found " << defaultIndex;
                }

                int64_t elseCount = 0;
                for (size_t i = 0; i < numCases; ++i)
                {
                    if (op.getCases()[i].empty())
                        return op.emitOpError() << "requires case region #" << i << " to contain a block";

                    const int64_t kind = caseKindsAttr[i];
                    if (kind < 0 || kind > 2)
                        return op.emitOpError() << "requires case_kinds[" << i << "] to be 0, 1, or 2, but found " << kind;

                    if (kind == 2)
                    {
                        ++elseCount;
                        if (defaultIndex < 0)
                            return op.emitOpError() << "requires 'default_case_index' when case_kinds[" << i << "] is else";
                        if (defaultIndex != static_cast<int64_t>(i))
                            return op.emitOpError() << "requires else case at index " << defaultIndex
                                                    << ", but case_kinds[" << i << "] is marked as else";
                    }
                }

                if (elseCount > 1)
                    return op.emitOpError() << "requires at most one else case, but found " << elseCount;

                if (defaultIndex >= 0 && caseKindsAttr[defaultIndex] != 2)
                    return op.emitOpError() << "requires case_kinds[" << defaultIndex << "] to be else (2)";

                return ::mlir::success();
            }

            template <typename SwitchLikeOp>
            static void printSwitchLikeCases(::mlir::OpAsmPrinter &p, SwitchLikeOp op)
            {
                auto caseValuesAttr = op.getCaseValuesAttr();
                auto rangeStartsAttr = op.getRangeStartsAttr();
                auto rangeEndsAttr = op.getRangeEndsAttr();
                auto caseKindsAttr = op.getCaseKindsAttr();
                auto defaultIndexAttr = op.getDefaultCaseIndexAttr();

                const size_t case_kinds_size = caseKindsAttr ? static_cast<size_t>(caseKindsAttr.size()) : 0;
                const size_t case_values_size = caseValuesAttr ? static_cast<size_t>(caseValuesAttr.size()) : 0;
                const size_t range_starts_size = rangeStartsAttr ? static_cast<size_t>(rangeStartsAttr.size()) : 0;
                const size_t range_ends_size = rangeEndsAttr ? static_cast<size_t>(rangeEndsAttr.size()) : 0;
                const int64_t defaultIndex = defaultIndexAttr ? defaultIndexAttr.getInt() : -1;

                const size_t numRegions = op->getNumRegions();
                for (size_t i = 0; i < numRegions; ++i)
                {
                    p.printNewline();
                    p << "  ";

                    if (defaultIndex >= 0 && static_cast<size_t>(defaultIndex) == i)
                    {
                        p << "else";
                    }
                    else
                    {
                        p << "case ";

                        if (caseKindsAttr && i < case_kinds_size)
                        {
                            int64_t kind = caseKindsAttr[i];
                            if (kind == 0 && caseValuesAttr && i < case_values_size)
                            {
                                p << caseValuesAttr[i];
                            }
                            else if (kind == 1 && rangeStartsAttr && rangeEndsAttr &&
                                     i < range_starts_size && i < range_ends_size)
                            {
                                p << rangeStartsAttr[i] << " ... " << rangeEndsAttr[i];
                            }
                        }
                    }

                    p << " => ";
                    auto &region = op->getRegion(i);
                    if (region.empty())
                    {
                        p << "{}";
                    }
                    else
                    {
                        p.printRegion(region);
                    }
                }
            }
        }

        // TupleType: !ora.tuple<type1, type2, ...>
        ::mlir::Type TupleType::parse(::mlir::AsmParser &parser)
        {
            if (parser.parseLess())
                return {};

            ::llvm::SmallVector<::mlir::Type> elementTypes;
            if (parser.parseTypeList(elementTypes))
                return {};

            if (parser.parseGreater())
                return {};

            return TupleType::get(parser.getContext(), elementTypes);
        }

        void TupleType::print(::mlir::AsmPrinter &printer) const
        {
            printer << "<";
            auto elementTypes = getElementTypes();
            for (size_t i = 0; i < elementTypes.size(); ++i)
            {
                if (i > 0)
                    printer << ", ";
                printer.printType(elementTypes[i]);
            }
            printer << ">";
        }

        // StructType: !ora.struct<"struct_name">
        ::mlir::Type StructType::parse(::mlir::AsmParser &parser)
        {
            if (parser.parseLess())
                return {};

            std::string name;
            if (parser.parseString(&name))
                return {};

            if (parser.parseGreater())
                return {};

            auto nameRef = parser.getBuilder().getStringAttr(name).getValue();
            return StructType::get(parser.getContext(), nameRef);
        }

        void StructType::print(::mlir::AsmPrinter &printer) const
        {
            printer << "<\"";
            printer << getName();
            printer << "\">";
        }

        // AdtType: !ora.adt<"adt_name", ("Variant", payload_type), ...>
        ::mlir::Type AdtType::parse(::mlir::AsmParser &parser)
        {
            if (parser.parseLess())
                return {};

            std::string name;
            if (parser.parseString(&name))
                return {};

            ::llvm::SmallVector<::llvm::StringRef> variantNames;
            ::llvm::SmallVector<::mlir::Type> payloadTypes;

            while (parser.parseOptionalComma().succeeded())
            {
                if (parser.parseLParen())
                    return {};

                std::string variantName;
                if (parser.parseString(&variantName))
                    return {};

                if (parser.parseComma())
                    return {};

                ::mlir::Type payloadType;
                if (parser.parseType(payloadType))
                    return {};

                if (parser.parseRParen())
                    return {};

                variantNames.push_back(parser.getBuilder().getStringAttr(variantName).getValue());
                payloadTypes.push_back(payloadType);
            }

            if (parser.parseGreater())
                return {};

            auto nameRef = parser.getBuilder().getStringAttr(name).getValue();
            return AdtType::get(parser.getContext(), nameRef, variantNames, payloadTypes);
        }

        void AdtType::print(::mlir::AsmPrinter &printer) const
        {
            printer << "<\"";
            printer << getName();
            printer << "\"";
            auto variantNames = getVariantNames();
            auto payloadTypes = getPayloadTypes();
            for (size_t i = 0; i < variantNames.size(); ++i)
            {
                printer << ", (\"" << variantNames[i] << "\", ";
                printer.printType(payloadTypes[i]);
                printer << ")";
            }
            printer << ">";
        }

        // AnonymousStructType: !ora.struct_anon<("field", type), ...>
        ::mlir::Type AnonymousStructType::parse(::mlir::AsmParser &parser)
        {
            if (parser.parseLess())
                return {};

            ::llvm::SmallVector<::llvm::StringRef> fieldNames;
            ::llvm::SmallVector<::mlir::Type> fieldTypes;

            if (parser.parseOptionalGreater().succeeded())
                return AnonymousStructType::get(parser.getContext(), fieldNames, fieldTypes);

            while (true)
            {
                if (parser.parseLParen())
                    return {};

                std::string name;
                if (parser.parseString(&name))
                    return {};

                if (parser.parseComma())
                    return {};

                ::mlir::Type fieldType;
                if (parser.parseType(fieldType))
                    return {};

                if (parser.parseRParen())
                    return {};

                fieldNames.push_back(parser.getBuilder().getStringAttr(name).getValue());
                fieldTypes.push_back(fieldType);

                if (parser.parseOptionalGreater().succeeded())
                    break;

                if (parser.parseComma())
                    return {};
            }

            return AnonymousStructType::get(parser.getContext(), fieldNames, fieldTypes);
        }

        void AnonymousStructType::print(::mlir::AsmPrinter &printer) const
        {
            printer << "<";
            auto fieldNames = getFieldNames();
            auto fieldTypes = getFieldTypes();
            for (size_t i = 0; i < fieldNames.size(); ++i)
            {
                if (i > 0)
                    printer << ", ";
                printer << "(\"" << fieldNames[i] << "\", ";
                printer.printType(fieldTypes[i]);
                printer << ")";
            }
            printer << ">";
        }

        // EnumType: !ora.enum<"enum_name", repr_type>
        ::mlir::Type EnumType::parse(::mlir::AsmParser &parser)
        {
            if (parser.parseLess())
                return {};

            std::string name;
            if (parser.parseString(&name))
                return {};

            if (parser.parseComma())
                return {};

            ::mlir::Type reprType;
            if (parser.parseType(reprType))
                return {};

            if (parser.parseGreater())
                return {};

            auto nameRef = parser.getBuilder().getStringAttr(name).getValue();
            return EnumType::get(parser.getContext(), nameRef, reprType);
        }

        void EnumType::print(::mlir::AsmPrinter &printer) const
        {
            printer << "<\"";
            printer << getName();
            printer << "\", ";
            printer.printType(getReprType());
            printer << ">";
        }

        // ContractType: !ora.contract<"contract_name">
        ::mlir::Type ContractType::parse(::mlir::AsmParser &parser)
        {
            if (parser.parseLess())
                return {};

            std::string name;
            if (parser.parseString(&name))
                return {};

            if (parser.parseGreater())
                return {};

            return ContractType::get(parser.getContext(), name);
        }

        void ContractType::print(::mlir::AsmPrinter &printer) const
        {
            printer << "<\"";
            printer << getName();
            printer << "\">";
        }

        // FunctionType: !ora.function<param_types, return_type>
        ::mlir::Type FunctionType::parse(::mlir::AsmParser &parser)
        {
            if (parser.parseLess())
                return {};

            ::llvm::SmallVector<::mlir::Type> paramTypes;
            if (parser.parseTypeList(paramTypes))
                return {};

            if (parser.parseComma())
                return {};

            ::mlir::Type returnType;
            if (parser.parseType(returnType))
                return {};

            if (parser.parseGreater())
                return {};

            return FunctionType::get(parser.getContext(), paramTypes, returnType);
        }

        void FunctionType::print(::mlir::AsmPrinter &printer) const
        {
            printer << "<";
            auto paramTypes = getParamTypes();
            for (size_t i = 0; i < paramTypes.size(); ++i)
            {
                if (i > 0)
                    printer << ", ";
                printer.printType(paramTypes[i]);
            }
            printer << ", ";
            printer.printType(getReturnType());
            printer << ">";
        }

        // ErrorUnionType: !ora.error_union<success_type[, error_type...]>
        ::mlir::Type ErrorUnionType::parse(::mlir::AsmParser &parser)
        {
            if (parser.parseLess())
                return {};

            ::mlir::Type successType;
            if (parser.parseType(successType))
                return {};

            ::llvm::SmallVector<::mlir::Type> errorTypes;
            while (parser.parseOptionalComma().succeeded())
            {
                ::mlir::Type errorType;
                if (parser.parseType(errorType))
                    return {};
                errorTypes.push_back(errorType);
            }

            if (parser.parseGreater())
                return {};

            return ErrorUnionType::get(parser.getContext(), successType, errorTypes);
        }

        void ErrorUnionType::print(::mlir::AsmPrinter &printer) const
        {
            printer << "<";
            printer.printType(getSuccessType());
            auto errorTypes = getErrorTypes();
            for (auto errorType : errorTypes)
            {
                printer << ", ";
                printer.printType(errorType);
            }
            printer << ">";
        }

        // MinValueType: !ora.min_value<base_type, min_high_high, min_high_low, min_low_high, min_low_low>
        ::mlir::Type MinValueType::parse(::mlir::AsmParser &parser)
        {
            if (parser.parseLess())
                return {};

            ::mlir::Type baseType;
            if (parser.parseType(baseType))
                return {};

            if (parser.parseComma())
                return {};

            uint64_t minHighHigh, minHighLow, minLowHigh, minLowLow;
            if (parser.parseInteger(minHighHigh))
                return {};
            if (parser.parseComma())
                return {};
            if (parser.parseInteger(minHighLow))
                return {};
            if (parser.parseComma())
                return {};
            if (parser.parseInteger(minLowHigh))
                return {};
            if (parser.parseComma())
                return {};
            if (parser.parseInteger(minLowLow))
                return {};

            if (parser.parseGreater())
                return {};

            return MinValueType::get(parser.getContext(), baseType, minHighHigh, minHighLow, minLowHigh, minLowLow);
        }

        void MinValueType::print(::mlir::AsmPrinter &printer) const
        {
            printer << "<";
            printer.printType(getBaseType());
            printer << ", ";
            printer << getMinHighHigh();
            printer << ", ";
            printer << getMinHighLow();
            printer << ", ";
            printer << getMinLowHigh();
            printer << ", ";
            printer << getMinLowLow();
            printer << ">";
        }

        // MaxValueType: !ora.max_value<base_type, max_high_high, max_high_low, max_low_high, max_low_low>
        ::mlir::Type MaxValueType::parse(::mlir::AsmParser &parser)
        {
            if (parser.parseLess())
                return {};

            ::mlir::Type baseType;
            if (parser.parseType(baseType))
                return {};

            if (parser.parseComma())
                return {};

            uint64_t maxHighHigh, maxHighLow, maxLowHigh, maxLowLow;
            if (parser.parseInteger(maxHighHigh))
                return {};
            if (parser.parseComma())
                return {};
            if (parser.parseInteger(maxHighLow))
                return {};
            if (parser.parseComma())
                return {};
            if (parser.parseInteger(maxLowHigh))
                return {};
            if (parser.parseComma())
                return {};
            if (parser.parseInteger(maxLowLow))
                return {};

            if (parser.parseGreater())
                return {};

            return MaxValueType::get(parser.getContext(), baseType, maxHighHigh, maxHighLow, maxLowHigh, maxLowLow);
        }

        void MaxValueType::print(::mlir::AsmPrinter &printer) const
        {
            printer << "<";
            printer.printType(getBaseType());
            printer << ", ";
            printer << getMaxHighHigh();
            printer << ", ";
            printer << getMaxHighLow();
            printer << ", ";
            printer << getMaxLowHigh();
            printer << ", ";
            printer << getMaxLowLow();
            printer << ">";
        }

        // InRangeType: !ora.in_range<base_type, min_high_high, min_high_low, min_low_high, min_low_low, max_high_high, max_high_low, max_low_high, max_low_low>
        ::mlir::Type InRangeType::parse(::mlir::AsmParser &parser)
        {
            if (parser.parseLess())
                return {};

            ::mlir::Type baseType;
            if (parser.parseType(baseType))
                return {};

            if (parser.parseComma())
                return {};

            uint64_t minHighHigh, minHighLow, minLowHigh, minLowLow;
            uint64_t maxHighHigh, maxHighLow, maxLowHigh, maxLowLow;
            if (parser.parseInteger(minHighHigh))
                return {};
            if (parser.parseComma())
                return {};
            if (parser.parseInteger(minHighLow))
                return {};
            if (parser.parseComma())
                return {};
            if (parser.parseInteger(minLowHigh))
                return {};
            if (parser.parseComma())
                return {};
            if (parser.parseInteger(minLowLow))
                return {};
            if (parser.parseComma())
                return {};
            if (parser.parseInteger(maxHighHigh))
                return {};
            if (parser.parseComma())
                return {};
            if (parser.parseInteger(maxHighLow))
                return {};
            if (parser.parseComma())
                return {};
            if (parser.parseInteger(maxLowHigh))
                return {};
            if (parser.parseComma())
                return {};
            if (parser.parseInteger(maxLowLow))
                return {};

            if (parser.parseGreater())
                return {};

            return InRangeType::get(parser.getContext(), baseType, minHighHigh, minHighLow, minLowHigh, minLowLow, maxHighHigh, maxHighLow, maxLowHigh, maxLowLow);
        }

        void InRangeType::print(::mlir::AsmPrinter &printer) const
        {
            printer << "<";
            printer.printType(getBaseType());
            printer << ", ";
            printer << getMinHighHigh();
            printer << ", ";
            printer << getMinHighLow();
            printer << ", ";
            printer << getMinLowHigh();
            printer << ", ";
            printer << getMinLowLow();
            printer << ", ";
            printer << getMaxHighHigh();
            printer << ", ";
            printer << getMaxHighLow();
            printer << ", ";
            printer << getMaxLowHigh();
            printer << ", ";
            printer << getMaxLowLow();
            printer << ">";
        }

        // RefinementToBaseOp: ora.refinement_to_base
        ::mlir::OpFoldResult AddrToI160Op::fold(FoldAdaptor)
        {
            if (auto inner = getAddr().getDefiningOp<I160ToAddrOp>())
                return foldAddressCarrierRoundTrip(inner.getI160(), getResult().getType());
            return {};
        }

        ::mlir::OpFoldResult I160ToAddrOp::fold(FoldAdaptor)
        {
            if (auto inner = getI160().getDefiningOp<AddrToI160Op>())
                return foldAddressCarrierRoundTrip(inner.getAddr(), getResult().getType());
            return {};
        }

        void RefinementToBaseOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value value)
        {
            Type valueType = value.getType();
            Type baseType = getRefinementBaseType(valueType);
            if (!baseType)
                baseType = valueType;

            odsState.addOperands(value);
            odsState.addTypes(baseType);
        }

        ::mlir::OpFoldResult RefinementToBaseOp::fold(FoldAdaptor)
        {
            if (auto inner = getValue().getDefiningOp<BaseToRefinementOp>())
                if (inner.getValue().getType() == getResult().getType())
                    return inner.getValue();
            return {};
        }

        ::mlir::OpFoldResult BaseToRefinementOp::fold(FoldAdaptor)
        {
            if (auto inner = getValue().getDefiningOp<RefinementToBaseOp>())
                if (inner.getValue().getType() == getResult().getType())
                    return inner.getValue();
            return {};
        }

    } // namespace ora
} // namespace mlir

// Include the generated operation method implementations
// This must come BEFORE custom print/parse methods so the operation classes are fully defined
#define GET_OP_CLASSES
#include "OraOps.cpp.inc"

namespace mlir
{
    namespace ora
    {
        namespace
        {
            static bool isStaticallyZeroIntegerValue(::mlir::Value value)
            {
                if (auto constOp = value.getDefiningOp<arith::ConstantOp>())
                {
                    if (auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(constOp.getValue()))
                    {
                        return intAttr.getValue().isZero();
                    }
                }
                return false;
            }

            static std::optional<mlir::IntegerAttr> getConstantIntegerAttr(::mlir::Value value)
            {
                if (auto constOp = value.getDefiningOp<arith::ConstantOp>())
                {
                    if (auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(constOp.getValue()))
                        return intAttr;
                }
                if (auto constOp = value.getDefiningOp<ConstOp>())
                {
                    if (auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(constOp.getValueAttr()))
                        return intAttr;
                }
                return std::nullopt;
            }

            static bool isIntegerConstant(::mlir::Value value, uint64_t expected)
            {
                auto attr = getConstantIntegerAttr(value);
                return attr && attr->getValue() == expected;
            }

            static bool isZeroIntegerConstant(::mlir::Value value)
            {
                auto attr = getConstantIntegerAttr(value);
                return attr && attr->getValue().isZero();
            }

            static std::optional<unsigned> getIntegerLikeBitWidth(::mlir::Type type)
            {
                if (auto oraInt = llvm::dyn_cast<ora::IntegerType>(type))
                    return oraInt.getWidth();
                if (auto builtinInt = llvm::dyn_cast<::mlir::IntegerType>(type))
                    return builtinInt.getWidth();
                return std::nullopt;
            }

            static std::optional<llvm::APInt> computeBoundedWrappingPower(
                const llvm::APInt &base,
                const llvm::APInt &exponent,
                unsigned bitWidth)
            {
                constexpr uint64_t kMaxFoldExponent = 1024;
                uint64_t exp = exponent.getLimitedValue(kMaxFoldExponent + 1);
                if (exp > kMaxFoldExponent)
                    return std::nullopt;

                llvm::APInt result(bitWidth, 1);
                llvm::APInt factor = base.zextOrTrunc(bitWidth);
                while (exp != 0)
                {
                    if (exp & 1)
                        result *= factor;
                    exp >>= 1;
                    if (exp != 0)
                        factor *= factor;
                }
                return result;
            }

            static std::optional<bool> compareSameOperand(StringRef predicate)
            {
                if (predicate == "eq" || predicate == "le" || predicate == "lte" ||
                    predicate == "ule" || predicate == "ge" || predicate == "gte" ||
                    predicate == "uge" || predicate == "sle" || predicate == "sge")
                    return true;
                if (predicate == "ne" || predicate == "neq" || predicate == "lt" ||
                    predicate == "ult" || predicate == "gt" || predicate == "ugt" ||
                    predicate == "slt" || predicate == "sgt")
                    return false;
                return std::nullopt;
            }

            static std::optional<bool> compareConstantIntegers(
                StringRef predicate,
                const llvm::APInt &lhs,
                const llvm::APInt &rhs)
            {
                if (predicate == "eq")
                    return lhs == rhs;
                if (predicate == "ne" || predicate == "neq")
                    return lhs != rhs;
                if (predicate == "lt" || predicate == "ult")
                    return lhs.ult(rhs);
                if (predicate == "le" || predicate == "lte" || predicate == "ule")
                    return lhs.ule(rhs);
                if (predicate == "gt" || predicate == "ugt")
                    return lhs.ugt(rhs);
                if (predicate == "ge" || predicate == "gte" || predicate == "uge")
                    return lhs.uge(rhs);
                if (predicate == "slt")
                    return lhs.slt(rhs);
                if (predicate == "sle")
                    return lhs.sle(rhs);
                if (predicate == "sgt")
                    return lhs.sgt(rhs);
                if (predicate == "sge")
                    return lhs.sge(rhs);
                return std::nullopt;
            }

            static LogicalResult replaceOpWithValueIfSameType(
                PatternRewriter &rewriter,
                Operation *op,
                Value replacement)
            {
                if (op->getNumResults() != 1)
                    return failure();
                if (replacement.getType() != op->getResult(0).getType())
                    return failure();

                rewriter.replaceOp(op, replacement);
                return success();
            }

            static GlobalOp lookupGlobalFor(Operation *op, StringRef globalName)
            {
                auto globalAttr = mlir::StringAttr::get(op->getContext(), globalName);
                if (Operation *symbol = mlir::SymbolTable::lookupNearestSymbolFrom(op, globalAttr))
                    return llvm::dyn_cast<GlobalOp>(symbol);
                return nullptr;
            }

            static StructDeclOp findStructDecl(Operation *op, StringRef structName)
            {
                ModuleOp module = op->getParentOfType<ModuleOp>();
                if (!module)
                    return nullptr;

                StructDeclOp structDecl = nullptr;
                module.walk([&](StructDeclOp declOp)
                            {
                    auto sym = declOp->getAttrOfType<StringAttr>("sym_name");
                    if (sym && sym.getValue() == structName)
                    {
                        structDecl = declOp;
                        return WalkResult::interrupt();
                    }
                    return WalkResult::advance(); });
                return structDecl;
            }

            static mlir::LogicalResult replaceOpWithConstant(
                PatternRewriter &rewriter,
                Operation *op,
                ::mlir::Type resultType,
                const llvm::APInt &value)
            {
                auto valueAttr = mlir::IntegerAttr::get(resultType, value);
                auto newConst = rewriter.create<arith::ConstantOp>(op->getLoc(), resultType, valueAttr);
                rewriter.replaceOp(op, newConst.getResult());
                return success();
            }

            static mlir::LogicalResult replaceOpWithBoolConstant(
                PatternRewriter &rewriter,
                Operation *op,
                bool value)
            {
                return replaceOpWithConstant(
                    rewriter,
                    op,
                    op->getResult(0).getType(),
                    llvm::APInt(1, value ? 1 : 0));
            }

            static ::mlir::LogicalResult getStructFieldInfo(
                Operation *op,
                StructType structType,
                StringRef fieldName,
                size_t &fieldIndex,
                ::mlir::Type &fieldType)
            {
                auto structDecl = findStructDecl(op, structType.getName());
                if (!structDecl)
                    return op->emitError() << "missing ora.struct.decl for struct '" << structType.getName() << "'";

                auto fieldNamesAttr = structDecl->getAttrOfType<ArrayAttr>("ora.field_names");
                auto fieldTypesAttr = structDecl->getAttrOfType<ArrayAttr>("ora.field_types");
                if (!fieldNamesAttr || !fieldTypesAttr || fieldNamesAttr.size() != fieldTypesAttr.size())
                    return op->emitError() << "struct declaration for '" << structType.getName() << "' has malformed field metadata";

                for (size_t i = 0; i < fieldNamesAttr.size(); ++i)
                {
                    auto nameAttr = llvm::dyn_cast<StringAttr>(fieldNamesAttr[i]);
                    auto typeAttr = llvm::dyn_cast<TypeAttr>(fieldTypesAttr[i]);
                    if (!nameAttr || !typeAttr)
                        return op->emitError() << "struct declaration for '" << structType.getName() << "' has invalid field metadata";
                    if (nameAttr.getValue() == fieldName)
                    {
                        fieldIndex = i;
                        fieldType = typeAttr.getValue();
                        return success();
                    }
                }

                return op->emitError() << "unknown field '" << fieldName << "' on struct '" << structType.getName() << "'";
            }

            static ::mlir::LogicalResult getStructFieldInfo(
                Operation *op,
                AnonymousStructType structType,
                StringRef fieldName,
                size_t &fieldIndex,
                ::mlir::Type &fieldType)
            {
                auto fieldNames = structType.getFieldNames();
                auto fieldTypes = structType.getFieldTypes();
                if (fieldNames.size() != fieldTypes.size())
                    return op->emitError("anonymous struct type has malformed field metadata");

                for (size_t i = 0; i < fieldNames.size(); ++i)
                {
                    if (fieldNames[i] == fieldName)
                    {
                        fieldIndex = i;
                        fieldType = fieldTypes[i];
                        return success();
                    }
                }

                return op->emitError() << "unknown field '" << fieldName << "' on anonymous struct";
            }

            static ::mlir::LogicalResult getStructFieldInfo(
                Operation *op,
                Type structValueType,
                StringRef fieldName,
                size_t &fieldIndex,
                ::mlir::Type &fieldType)
            {
                if (auto structType = llvm::dyn_cast<StructType>(structValueType))
                    return getStructFieldInfo(op, structType, fieldName, fieldIndex, fieldType);
                if (auto anonType = llvm::dyn_cast<AnonymousStructType>(structValueType))
                    return getStructFieldInfo(op, anonType, fieldName, fieldIndex, fieldType);
                return failure();
            }
        } // namespace

        ::mlir::LogicalResult DivOp::verify()
        {
            if (isStaticallyZeroIntegerValue(getRhs()))
                return emitOpError("divisor must not be a statically known zero constant");
            return success();
        }

        ::mlir::LogicalResult RemOp::verify()
        {
            if (isStaticallyZeroIntegerValue(getRhs()))
                return emitOpError("divisor must not be a statically known zero constant");
            return success();
        }

        ::mlir::LogicalResult SLoadOp::verify()
        {
            auto global = lookupGlobalFor(*this, getGlobalName());
            if (!global)
                return emitOpError() << "unknown storage global '" << getGlobalName() << "'";

            if (getResult().getType() != global.getGlobalType())
            {
                return emitOpError() << "result type " << getResult().getType()
                                     << " does not match storage global type " << global.getGlobalType()
                                     << " for '" << getGlobalName() << "'";
            }

            return success();
        }

        ::mlir::LogicalResult SStoreOp::verify()
        {
            auto global = lookupGlobalFor(*this, getGlobalName());
            if (!global)
                return emitOpError() << "unknown storage global '" << getGlobalName() << "'";

            if (getValue().getType() != global.getGlobalType())
            {
                return emitOpError() << "stored value type " << getValue().getType()
                                     << " does not match storage global type " << global.getGlobalType()
                                     << " for '" << getGlobalName() << "'";
            }

            return success();
        }

        ::mlir::LogicalResult StorageDeriveOp::verify()
        {
            auto hashAttr = (*this)->getAttrOfType<IntegerAttr>("namespace_hash");
            if (!hashAttr)
                return emitOpError("requires 256-bit integer 'namespace_hash' attribute");
            if (hashAttr.getValue().getBitWidth() != 256)
                return emitOpError("'namespace_hash' must be a 256-bit integer attribute");
            if (!isComputedStorageWordType(getSlot().getType()))
                return emitOpError() << "result type must be a 256-bit integer storage slot, got "
                                     << getSlot().getType();
            for (auto key : getKeys())
            {
                if (!isComputedStorageKeyType(key.getType()))
                    return emitOpError() << "key operand type must be address or 256-bit integer, got "
                                         << key.getType();
            }
            return success();
        }

        ::mlir::LogicalResult StorageWordLoadOp::verify()
        {
            if (!isComputedStorageWordType(getSlot().getType()))
                return emitOpError() << "slot operand must be a 256-bit integer storage slot, got "
                                     << getSlot().getType();
            if (!isComputedStorageWordType(getOffset().getType()))
                return emitOpError() << "offset operand must be a 256-bit integer word offset, got "
                                     << getOffset().getType();
            if (!isComputedStorageWordType(getResult().getType()))
                return emitOpError() << "result type must be a 256-bit integer word, got "
                                     << getResult().getType();
            return success();
        }

        ::mlir::LogicalResult StorageWordStoreOp::verify()
        {
            if (!isComputedStorageWordType(getSlot().getType()))
                return emitOpError() << "slot operand must be a 256-bit integer storage slot, got "
                                     << getSlot().getType();
            if (!isComputedStorageWordType(getOffset().getType()))
                return emitOpError() << "offset operand must be a 256-bit integer word offset, got "
                                     << getOffset().getType();
            if (!isComputedStorageWordType(getValue().getType()))
                return emitOpError() << "stored value must be a 256-bit integer word, got "
                                     << getValue().getType();
            return success();
        }

        ::mlir::LogicalResult StorageRangeEraseOp::verify()
        {
            if (!isComputedStorageWordType(getSlot().getType()))
                return emitOpError() << "slot operand must be a 256-bit integer storage slot, got "
                                     << getSlot().getType();
            auto wordCountAttr = (*this)->getAttrOfType<IntegerAttr>("word_count");
            if (!wordCountAttr)
                return emitOpError("requires 'word_count' attribute");
            if (wordCountAttr.getValue().getBitWidth() != 64)
                return emitOpError("'word_count' must be a 64-bit integer attribute");
            if (wordCountAttr.getValue().isNegative())
                return emitOpError("'word_count' must be non-negative");
            return success();
        }

        ::mlir::LogicalResult StructInitOp::verify()
        {
            SmallVector<Type, 8> expectedFieldTypes;
            std::string typeLabel;

            if (auto structType = llvm::dyn_cast<StructType>(getResult().getType()))
            {
                auto structDecl = findStructDecl(*this, structType.getName());
                if (!structDecl)
                    return emitOpError() << "missing ora.struct.decl for struct '" << structType.getName() << "'";

                auto fieldTypesAttr = structDecl->getAttrOfType<ArrayAttr>("ora.field_types");
                if (!fieldTypesAttr)
                    return emitOpError() << "struct declaration for '" << structType.getName() << "' is missing ora.field_types";

                for (auto attr : fieldTypesAttr)
                {
                    auto typeAttr = llvm::dyn_cast<TypeAttr>(attr);
                    if (!typeAttr)
                        return emitOpError() << "struct declaration for '" << structType.getName() << "' has invalid field type metadata";
                    expectedFieldTypes.push_back(typeAttr.getValue());
                }
                typeLabel = ("struct '" + structType.getName()).str();
                typeLabel += "'";
            }
            else if (auto anonType = llvm::dyn_cast<AnonymousStructType>(getResult().getType()))
            {
                auto fieldTypes = anonType.getFieldTypes();
                expectedFieldTypes.append(fieldTypes.begin(), fieldTypes.end());
                typeLabel = "anonymous struct";
            }
            else
            {
                return emitOpError("result type must be !ora.struct<...> or !ora.struct_anon<...>");
            }

            if (getFieldValues().size() != expectedFieldTypes.size())
            {
                return emitOpError() << "expected " << expectedFieldTypes.size()
                                     << " field values for " << typeLabel
                                     << ", got " << getFieldValues().size();
            }

            for (size_t i = 0; i < expectedFieldTypes.size(); ++i)
            {
                if (getFieldValues()[i].getType() != expectedFieldTypes[i])
                {
                    return emitOpError() << "field operand #" << i << " has type " << getFieldValues()[i].getType()
                                         << " but " << typeLabel << " expects " << expectedFieldTypes[i];
                }
            }

            return success();
        }

        ::mlir::LogicalResult TupleCreateOp::verify()
        {
            auto tupleType = llvm::dyn_cast<TupleType>(getResult().getType());
            if (!tupleType)
                return emitOpError("result type must be !ora.tuple<...>");

            auto elementTypes = tupleType.getElementTypes();
            if (getElements().size() != elementTypes.size())
            {
                return emitOpError() << "expected " << elementTypes.size()
                                     << " tuple elements, got " << getElements().size();
            }

            for (size_t i = 0; i < elementTypes.size(); ++i)
            {
                if (getElements()[i].getType() != elementTypes[i])
                {
                    return emitOpError() << "tuple element #" << i << " has type "
                                         << getElements()[i].getType() << " but tuple type expects "
                                         << elementTypes[i];
                }
            }

            return success();
        }

        ::mlir::LogicalResult TupleExtractOp::verify()
        {
            auto tupleType = llvm::dyn_cast<TupleType>(getTupleValue().getType());
            if (!tupleType)
                return emitOpError("tuple operand must have !ora.tuple<...> type");

            auto elementTypes = tupleType.getElementTypes();
            auto index = static_cast<size_t>(getIndex());
            if (index >= elementTypes.size())
            {
                return emitOpError() << "tuple index " << getIndex()
                                     << " is out of bounds for tuple of size "
                                     << elementTypes.size();
            }

            if (getResult().getType() != elementTypes[index])
            {
                return emitOpError() << "result type " << getResult().getType()
                                     << " does not match tuple element #" << index
                                     << " type " << elementTypes[index];
            }

            return success();
        }

        ::mlir::LogicalResult StructFieldExtractOp::verify()
        {
            size_t fieldIndex = 0;
            ::mlir::Type fieldType;
            if (failed(getStructFieldInfo(*this, getStructValue().getType(), getFieldName(), fieldIndex, fieldType)))
            {
                if (!llvm::isa<StructType, AnonymousStructType>(getStructValue().getType()))
                    return emitOpError("struct operand must have !ora.struct<...> or !ora.struct_anon<...> type");
                return failure();
            }

            if (getResult().getType() != fieldType)
            {
                return emitOpError() << "result type " << getResult().getType()
                                     << " does not match field '" << getFieldName()
                                     << "' type " << fieldType;
            }

            return success();
        }

        static ::mlir::LogicalResult getAdtVariantInfo(
            ::mlir::Operation *op,
            AdtType adtType,
            ::llvm::StringRef variantName,
            size_t &variantIndex,
            ::mlir::Type &payloadType)
        {
            auto variantNames = adtType.getVariantNames();
            auto payloadTypes = adtType.getPayloadTypes();
            if (variantNames.size() != payloadTypes.size())
            {
                return op->emitError() << "ADT type '" << adtType.getName()
                                       << "' has malformed variant metadata";
            }

            for (size_t i = 0; i < variantNames.size(); ++i)
            {
                if (variantNames[i] == variantName)
                {
                    variantIndex = i;
                    payloadType = payloadTypes[i];
                    return success();
                }
            }

            return op->emitError() << "unknown variant '" << variantName
                                   << "' on ADT '" << adtType.getName() << "'";
        }

        ::mlir::LogicalResult AdtConstructOp::verify()
        {
            auto adtType = llvm::dyn_cast<AdtType>(getResult().getType());
            if (!adtType)
                return emitOpError("result type must be !ora.adt<...>");

            size_t variantIndex = 0;
            ::mlir::Type payloadType;
            if (failed(getAdtVariantInfo(*this, adtType, getVariantName(), variantIndex, payloadType)))
                return failure();

            if (llvm::isa<::mlir::NoneType>(payloadType))
            {
                if (!getPayloadValues().empty())
                {
                    return emitOpError() << "unit variant '" << getVariantName()
                                         << "' expects no payload operands";
                }
                return success();
            }

            if (getPayloadValues().size() != 1)
            {
                return emitOpError() << "variant '" << getVariantName()
                                     << "' expects exactly one payload operand";
            }

            if (getPayloadValues().front().getType() != payloadType)
            {
                return emitOpError() << "payload operand type " << getPayloadValues().front().getType()
                                     << " does not match variant '" << getVariantName()
                                     << "' payload type " << payloadType;
            }

            return success();
        }

        ::mlir::LogicalResult AdtTagOp::verify()
        {
            auto resultType = getResult().getType();
            if (!llvm::isa<ora::IntegerType, ::mlir::IntegerType>(resultType))
            {
                return emitOpError() << "result type must be an integer tag type, got " << resultType;
            }

            return success();
        }

        ::mlir::LogicalResult AdtPayloadOp::verify()
        {
            auto adtType = llvm::dyn_cast<AdtType>(getValue().getType());
            if (!adtType)
                return emitOpError("operand must have !ora.adt<...> type");

            size_t variantIndex = 0;
            ::mlir::Type payloadType;
            if (failed(getAdtVariantInfo(*this, adtType, getVariantName(), variantIndex, payloadType)))
                return failure();

            if (llvm::isa<::mlir::NoneType>(payloadType))
            {
                return emitOpError() << "unit variant '" << getVariantName()
                                     << "' has no payload to extract";
            }

            if (getResult().getType() != payloadType)
            {
                return emitOpError() << "result type " << getResult().getType()
                                     << " does not match variant '" << getVariantName()
                                     << "' payload type " << payloadType;
            }

            return success();
        }

        ::mlir::LogicalResult AdtMatchArmOp::verify()
        {
            auto adtType = llvm::dyn_cast<AdtType>(getValue().getType());
            if (!adtType)
                return emitOpError("operand must have !ora.adt<...> type");

            size_t variantIndex = 0;
            ::mlir::Type payloadType;
            if (failed(getAdtVariantInfo(*this, adtType, getVariantName(), variantIndex, payloadType)))
                return failure();

            return success();
        }

        ::mlir::LogicalResult StructFieldUpdateOp::verify()
        {
            if (getResult().getType() != getStructValue().getType())
                return emitOpError("result type must match the input struct type");

            size_t fieldIndex = 0;
            ::mlir::Type fieldType;
            if (failed(getStructFieldInfo(*this, getStructValue().getType(), getFieldName(), fieldIndex, fieldType)))
            {
                if (!llvm::isa<StructType, AnonymousStructType>(getStructValue().getType()))
                    return emitOpError("struct operand must have !ora.struct<...> or !ora.struct_anon<...> type");
                return failure();
            }

            if (getValue().getType() != fieldType)
            {
                return emitOpError() << "updated value type " << getValue().getType()
                                     << " does not match field '" << getFieldName()
                                     << "' type " << fieldType;
            }

            return success();
        }

        ::mlir::LogicalResult MapGetOp::verify()
        {
            auto mapType = llvm::dyn_cast<MapType>(getMap().getType());
            if (!mapType)
                return emitOpError("map operand must have !ora.map<key, value> type");

            if (getKey().getType() != mapType.getKeyType())
                return emitOpError() << "key type " << getKey().getType()
                                     << " does not match map key type " << mapType.getKeyType();

            if (getResult().getType() != mapType.getValueType())
            {
                return emitOpError() << "result type " << getResult().getType()
                                     << " does not match map value type " << mapType.getValueType();
            }

            return success();
        }

        ::mlir::LogicalResult MapStoreOp::verify()
        {
            auto mapType = llvm::dyn_cast<MapType>(getMap().getType());
            if (!mapType)
                return emitOpError("map operand must have !ora.map<key, value> type");

            if (getKey().getType() != mapType.getKeyType())
                return emitOpError() << "key type " << getKey().getType()
                                     << " does not match map key type " << mapType.getKeyType();

            if (getValue().getType() != mapType.getValueType())
            {
                return emitOpError() << "value type " << getValue().getType()
                                     << " does not match map value type " << mapType.getValueType();
            }

            return success();
        }

        ::mlir::LogicalResult ErrorUnwrapOp::verify()
        {
            auto unionType = llvm::dyn_cast<ErrorUnionType>(getValue().getType());
            if (!unionType)
                return emitOpError("operand must have !ora.error_union<...> type");

            if (getResult().getType() != unionType.getSuccessType())
            {
                return emitOpError() << "result type " << getResult().getType()
                                     << " does not match error union success type "
                                     << unionType.getSuccessType();
            }

            return success();
        }

        ::mlir::LogicalResult AbiEncodeOp::verify()
        {
            auto layoutAttr = (*this)->getAttr("layout");
            if (!layoutAttr)
                return emitOpError("requires 'layout' attribute");
            if (!llvm::isa<::mlir::StringAttr>(layoutAttr))
                return emitOpError("'layout' must be a string attribute");

            return success();
        }

        ::mlir::LogicalResult AbiEncodeWithSelectorOp::verify()
        {
            auto selectorAttr = (*this)->getAttr("selector");
            if (!selectorAttr)
                return emitOpError("requires 'selector' attribute");
            auto selector = llvm::dyn_cast<::mlir::IntegerAttr>(selectorAttr);
            if (!selector)
                return emitOpError("'selector' must be an integer attribute");
            if (selector.getType().getIntOrFloatBitWidth() != 32)
                return emitOpError("'selector' must be a 32-bit integer attribute");

            auto argTypesAttr = (*this)->getAttrOfType<::mlir::ArrayAttr>("arg_types");
            if (!argTypesAttr)
                return emitOpError("requires 'arg_types' array attribute");
            if (argTypesAttr.size() != getOperands().size())
                return emitOpError("'arg_types' entry count must match operand count");
            for (auto attr : argTypesAttr)
            {
                if (!llvm::isa<::mlir::StringAttr>(attr))
                    return emitOpError("'arg_types' entries must be string attributes");
            }
            auto layoutAttr = (*this)->getAttr("layout");
            if (!layoutAttr)
                return emitOpError("requires 'layout' attribute");
            if (!llvm::isa<::mlir::StringAttr>(layoutAttr))
                return emitOpError("'layout' must be a string attribute");

            return success();
        }

        ::mlir::LogicalResult ExternalCallOp::verify()
        {
            auto callKind = getCallKindAttr();
            if (!callKind)
                return emitOpError("requires 'call_kind' attribute");
            auto callKindValue = callKind.getValue();
            if (callKindValue != "call" && callKindValue != "staticcall")
                return emitOpError("'call_kind' must be 'call' or 'staticcall'");
            if (!getTraitNameAttr())
                return emitOpError("requires 'trait_name' attribute");
            if (!getMethodNameAttr())
                return emitOpError("requires 'method_name' attribute");

            return success();
        }

        ::mlir::LogicalResult AbiDecodeOp::verify()
        {
            auto verifyStringEnumAttr = [&](llvm::StringRef attrName,
                                            llvm::ArrayRef<llvm::StringRef> allowed) -> ::mlir::LogicalResult
            {
                auto attr = (*this)->getAttrOfType<::mlir::StringAttr>(attrName);
                if (!attr)
                    return emitOpError() << "requires '" << attrName << "' string attribute";
                if (!llvm::is_contained(allowed, attr.getValue()))
                    return emitOpError() << "has unsupported '" << attrName << "' value '"
                                         << attr.getValue() << "'";
                return success();
            };

            auto returnTypesAttr = (*this)->getAttrOfType<::mlir::ArrayAttr>("return_types");
            if (!returnTypesAttr)
                return emitOpError("requires 'return_types' array attribute");
            if (returnTypesAttr.empty())
                return emitOpError("'return_types' must contain at least one entry");
            for (auto attr : returnTypesAttr)
            {
                if (!llvm::isa<::mlir::StringAttr>(attr))
                    return emitOpError("'return_types' entries must be string attributes");
            }
            auto layoutAttr = (*this)->getAttr("layout");
            if (!layoutAttr)
                return emitOpError("requires 'layout' attribute");
            if (!llvm::isa<::mlir::StringAttr>(layoutAttr))
                return emitOpError("'layout' must be a string attribute");
            if (failed(verifyStringEnumAttr("source", {"calldata", "returndata", "memory"})))
                return failure();
            if (failed(verifyStringEnumAttr("failure_mode", {"result", "revert", "error_union"})))
                return failure();
            if (auto modeAttr = (*this)->getAttrOfType<::mlir::StringAttr>("decode_mode"))
            {
                if (!llvm::is_contained({"strict", "permissive"}, modeAttr.getValue()))
                    return emitOpError() << "has unsupported 'decode_mode' value '"
                                         << modeAttr.getValue() << "'";
            }

            return success();
        }

        ::mlir::LogicalResult RefinementToBaseOp::verify()
        {
            auto expectedBaseType = getRefinementBaseType(getValue().getType());
            if (!expectedBaseType)
                return emitOpError("operand must have an Ora refinement type");

            if (getResult().getType() != expectedBaseType)
            {
                return emitOpError() << "result type " << getResult().getType()
                                     << " does not match refinement base type " << expectedBaseType;
            }

            return success();
        }

        ::mlir::LogicalResult BaseToRefinementOp::verify()
        {
            auto expectedBaseType = getRefinementBaseType(getResult().getType());
            if (!expectedBaseType)
                return emitOpError("result type must be an Ora refinement type");

            if (getValue().getType() != expectedBaseType)
            {
                return emitOpError() << "operand type " << getValue().getType()
                                     << " does not match refinement base type " << expectedBaseType;
            }

            return success();
        }

        ::mlir::LogicalResult ReturnOp::verify()
        {
            auto func = (*this)->getParentOfType<mlir::func::FuncOp>();
            if (!func)
                return emitOpError("must be nested inside func.func");

            auto resultTypes = func.getFunctionType().getResults();
            if (getOperands().size() != resultTypes.size())
            {
                return emitOpError() << "expected " << resultTypes.size()
                                     << " return operands to match function signature, got "
                                     << getOperands().size();
            }

            for (size_t i = 0; i < resultTypes.size(); ++i)
            {
                if (getOperands()[i].getType() != resultTypes[i])
                {
                    return emitOpError() << "return operand #" << i << " has type "
                                         << getOperands()[i].getType()
                                         << " but enclosing func.func expects " << resultTypes[i];
                }
            }

            return success();
        }

        ::mlir::LogicalResult FunctionRefOp::verify()
        {
            if (getSymName().empty())
                return emitOpError("requires non-empty function symbol name");
            if (!llvm::isa<ora::FunctionType>(getResult().getType()))
                return emitOpError() << "result must have !ora.function<...> type, got "
                                     << getResult().getType();
            return success();
        }

        ::mlir::LogicalResult ErrorReturnOp::verify()
        {
            if (getSymName().empty())
                return emitOpError("requires non-empty error symbol name");
            return success();
        }

        ::mlir::LogicalResult IfOp::verify()
        {
            constexpr llvm::StringLiteral kConditionalReturnContract =
                "ora.conditional_return is only valid for early-return control flow";

            auto verifySingleBlockRegion = [&](::mlir::Region &region, llvm::StringRef regionName) -> ::mlir::LogicalResult {
                if (!region.hasOneBlock())
                    return emitOpError() << kConditionalReturnContract << ": " << regionName
                                         << " region must contain exactly one block";
                return success();
            };

            if (failed(verifySingleBlockRegion(getThenRegion(), "then")))
                return failure();
            if (failed(verifySingleBlockRegion(getElseRegion(), "else")))
                return failure();

            auto &thenBlock = getThenRegion().front();
            if (thenBlock.empty())
                return emitOpError() << kConditionalReturnContract << ": then region must not be empty";
            auto *thenTerminator = thenBlock.getTerminator();
            if (!thenTerminator)
                return emitOpError() << kConditionalReturnContract
                                     << ": then region must terminate with func.return or ora.return";
            if (!llvm::isa<mlir::func::ReturnOp, ora::ReturnOp>(thenTerminator))
            {
                return emitOpError() << kConditionalReturnContract
                                     << ": then region must terminate with func.return or ora.return, found '"
                                     << thenTerminator->getName().getStringRef() << "'";
            }

            auto &elseBlock = getElseRegion().front();
            if (elseBlock.empty())
                return emitOpError() << kConditionalReturnContract << ": else region must not be empty";
            if (elseBlock.getOperations().size() != 1)
                return emitOpError() << kConditionalReturnContract
                                     << ": else region must contain only an empty ora.yield terminator";
            auto *elseTerminator = elseBlock.getTerminator();
            auto elseYield = llvm::dyn_cast_or_null<ora::YieldOp>(elseTerminator);
            if (!elseYield)
            {
                return emitOpError() << kConditionalReturnContract
                                     << ": else region must terminate with ora.yield, found '"
                                     << (elseTerminator ? elseTerminator->getName().getStringRef() : "<none>") << "'";
            }
            if (elseYield.getOperands().size() != 0)
                return emitOpError() << kConditionalReturnContract
                                     << ": else region ora.yield must not return values";

            return success();
        }

        ::mlir::ParseResult IfOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result)
        {
            ::mlir::OpAsmParser::UnresolvedOperand condition;
            if (parser.parseOperand(condition))
                return ::mlir::failure();

            // Parse "then" keyword
            if (parser.parseKeyword("then"))
                return ::mlir::failure();

            // Parse then region
            std::unique_ptr<::mlir::Region> thenRegion = std::make_unique<::mlir::Region>();
            if (parser.parseRegion(*thenRegion))
                return ::mlir::failure();

            // Parse "else" keyword
            if (parser.parseKeyword("else"))
                return ::mlir::failure();

            // Parse else region
            std::unique_ptr<::mlir::Region> elseRegion = std::make_unique<::mlir::Region>();
            if (parser.parseRegion(*elseRegion))
                return ::mlir::failure();

            // Parse optional attributes
            if (parser.parseOptionalAttrDict(result.attributes))
                return ::mlir::failure();

            // The printer emits the condition as an MLIR i1 value.
            if (parser.resolveOperand(condition, parser.getBuilder().getI1Type(), result.operands))
                return ::mlir::failure();

            // Add regions
            result.addRegion(std::move(thenRegion));
            result.addRegion(std::move(elseRegion));

            return ::mlir::success();
        }

        ::mlir::LogicalResult SwitchOp::verify()
        {
            return verifySwitchLikeOp(*this);
        }

        ::mlir::LogicalResult SwitchExprOp::verify()
        {
            return verifySwitchLikeOp(*this);
        }

        void IfOp::print(::mlir::OpAsmPrinter &p)
        {
            p << " ";
            p << getCondition();
            p << " ";

            p << "then ";
            p.printRegion(getThenRegion());

            p << " else ";
            p.printRegion(getElseRegion());

            SmallVector<StringRef> elidedAttrs;
            p.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
        }

        //===----------------------------------------------------------------------===//
        // ContractOp Custom Parser/Printer
        //===----------------------------------------------------------------------===//

        ::mlir::ParseResult ContractOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result)
        {
            // Parse @ symbol and symbol name (parseSymbolName handles @ automatically)
            ::mlir::StringAttr symName;
            if (parser.parseSymbolName(symName, "sym_name", result.attributes))
                return ::mlir::failure();

            // Parse body region
            std::unique_ptr<::mlir::Region> body = std::make_unique<::mlir::Region>();
            if (parser.parseRegion(*body))
                return ::mlir::failure();

            // Parse optional attributes
            if (parser.parseOptionalAttrDict(result.attributes))
                return ::mlir::failure();

            // Add region
            result.addRegion(std::move(body));

            return ::mlir::success();
        }

        void ContractOp::print(::mlir::OpAsmPrinter &p)
        {
            // Print symbol name with space before @ (printSymbolName already includes @)
            p << " ";
            p.printSymbolName(getSymName());

            // Print body region
            p << " ";
            p.printRegion(getBody());

            // Print attributes
            SmallVector<StringRef> elidedAttrs = {"sym_name"};
            p.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
        }

        //===----------------------------------------------------------------------===//
        // GlobalOp Custom Parser/Printer
        //===----------------------------------------------------------------------===//

        ::mlir::ParseResult GlobalOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result)
        {
            // Parse format: "name" : type (for maps) or "name" = init : type (for scalars)
            // Parse symbol name
            std::string symName;
            if (parser.parseKeywordOrString(&symName))
                return ::mlir::failure();
            result.addAttribute("sym_name", parser.getBuilder().getStringAttr(symName));

            // Parse optional = init (maps don't have initializers)
            ::mlir::Attribute init;
            if (succeeded(parser.parseOptionalEqual()))
            {
                // Found =, so parse the initializer
                if (parser.parseAttribute(init, "init", result.attributes))
                    return ::mlir::failure();
            }
            else
            {
                // No initializer - use UnitAttr to represent "no initializer"
                init = ::mlir::UnitAttr::get(parser.getContext());
                result.addAttribute("init", init);
            }

            // Parse : type
            ::mlir::Type type;
            if (parser.parseColonType(type))
                return ::mlir::failure();
            result.addAttribute("type", ::mlir::TypeAttr::get(type));

            // Parse optional attributes
            if (parser.parseOptionalAttrDict(result.attributes))
                return ::mlir::failure();

            return ::mlir::success();
        }

        void GlobalOp::print(::mlir::OpAsmPrinter &p)
        {
            // Print format: "name" : type (initializers are handled by storage, not shown in MLIR)
            p << " ";
            p.printAttributeWithoutType(getSymNameAttr());

            // Never print initializers - they are handled by storage
            // Format: "name" : type {attrs}
            p << " : ";
            p << getType();

            // Print attributes (excluding sym_name, type, init which are already printed)
            SmallVector<StringRef> elidedAttrs = {"sym_name", "type", "init"};
            p.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
        }

        //===----------------------------------------------------------------------===//
        // SwitchOp Custom Parser/Printer
        //===----------------------------------------------------------------------===//

        ::mlir::ParseResult SwitchOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result)
        {
            // Parse switch value operand
            ::mlir::OpAsmParser::UnresolvedOperand value;
            if (parser.parseOperand(value))
                return ::mlir::failure();

            // Parse : type(value)
            ::mlir::Type valueType;
            if (parser.parseColonType(valueType))
                return ::mlir::failure();

            // Parse optional -> result_types
            ::llvm::SmallVector<::mlir::Type> resultTypes;
            if (succeeded(parser.parseOptionalArrowTypeList(resultTypes)))
            {
                result.addTypes(resultTypes);
            }

            // Parse { cases }
            if (parser.parseLBrace())
                return ::mlir::failure();

            // Parse cases
            ::llvm::SmallVector<int64_t> caseValues;
            ::llvm::SmallVector<int64_t> rangeStarts;
            ::llvm::SmallVector<int64_t> rangeEnds;
            ::llvm::SmallVector<int64_t> caseKinds; // 0=literal, 1=range, 2=else
            int64_t defaultCaseIndex = -1;
            int caseIndex = 0;

            while (succeeded(parser.parseOptionalKeyword("case")))
            {
                // Try to parse literal value
                int64_t caseStart;
                if (succeeded(parser.parseInteger(caseStart)))
                {
                    int64_t rangeEnd;
                    if (nextTokenIsEllipsis(parser))
                    {
                        if (parser.parseEllipsis())
                            return ::mlir::failure();
                        if (parser.parseInteger(rangeEnd))
                            return ::mlir::failure();

                        rangeStarts.push_back(caseStart);
                        rangeEnds.push_back(rangeEnd);
                        caseValues.push_back(0);
                        caseKinds.push_back(1); // range
                    }
                    else
                    {
                        caseValues.push_back(caseStart);
                        rangeStarts.push_back(0);
                        rangeEnds.push_back(0);
                        caseKinds.push_back(0); // literal
                    }
                }
                else
                {
                    return ::mlir::failure();
                }

                // Parse =>
                if (failed(parseFatArrow(parser)))
                    return ::mlir::failure();

                // Parse case region
                std::unique_ptr<::mlir::Region> caseRegion = std::make_unique<::mlir::Region>();
                if (parser.parseRegion(*caseRegion))
                    return ::mlir::failure();

                result.addRegion(std::move(caseRegion));
                caseIndex++;
            }

            // Parse optional else/default case
            if (succeeded(parser.parseOptionalKeyword("else")))
            {
                if (failed(parseFatArrow(parser)))
                    return ::mlir::failure();

                std::unique_ptr<::mlir::Region> elseRegion = std::make_unique<::mlir::Region>();
                if (parser.parseRegion(*elseRegion))
                    return ::mlir::failure();

                result.addRegion(std::move(elseRegion));
                defaultCaseIndex = caseIndex;
                caseKinds.push_back(2); // else
                caseValues.push_back(0);
                rangeStarts.push_back(0);
                rangeEnds.push_back(0);
            }

            // Parse }
            if (parser.parseRBrace())
                return ::mlir::failure();

            // Parse optional attributes
            if (parser.parseOptionalAttrDict(result.attributes))
                return ::mlir::failure();

            // Resolve value operand
            if (parser.resolveOperand(value, valueType, result.operands))
                return ::mlir::failure();

            // Add case pattern attributes
            if (!caseValues.empty())
            {
                auto caseValuesAttr = ::mlir::DenseI64ArrayAttr::get(parser.getContext(), caseValues);
                result.addAttribute("case_values", caseValuesAttr);
            }

            if (!rangeStarts.empty())
            {
                auto rangeStartsAttr = ::mlir::DenseI64ArrayAttr::get(parser.getContext(), rangeStarts);
                result.addAttribute("range_starts", rangeStartsAttr);
            }

            if (!rangeEnds.empty())
            {
                auto rangeEndsAttr = ::mlir::DenseI64ArrayAttr::get(parser.getContext(), rangeEnds);
                result.addAttribute("range_ends", rangeEndsAttr);
            }

            if (!caseKinds.empty())
            {
                auto caseKindsAttr = ::mlir::DenseI64ArrayAttr::get(parser.getContext(), caseKinds);
                result.addAttribute("case_kinds", caseKindsAttr);
            }

            if (defaultCaseIndex >= 0)
            {
                auto defaultIndexAttr = ::mlir::IntegerAttr::get(
                    ::mlir::IntegerType::get(parser.getContext(), 64), defaultCaseIndex);
                result.addAttribute("default_case_index", defaultIndexAttr);
            }

            return ::mlir::success();
        }

        void SwitchOp::print(::mlir::OpAsmPrinter &p)
        {
            // Print value
            p << " ";
            p << getValue();
            p << " : ";
            p << getValue().getType();

            // Print result types if any
            if (!getResults().empty())
            {
                p << " -> ";
                llvm::interleaveComma(getResults().getTypes(), p, [&](::mlir::Type type)
                                      { p << type; });
            }

            // Print cases
            p << " {";

            printSwitchLikeCases(p, *this);

            p.printNewline();
            p << "}";

            // Print attributes (excluding case pattern attributes)
            SmallVector<StringRef> elidedAttrs = {
                "case_values", "range_starts", "range_ends", "case_kinds", "default_case_index"};
            p.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
        }

        //===----------------------------------------------------------------------===//
        // SwitchExprOp Custom Parser/Printer
        //===----------------------------------------------------------------------===//

        ::mlir::ParseResult SwitchExprOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result)
        {
            // Parse switch value operand
            ::mlir::OpAsmParser::UnresolvedOperand value;
            if (parser.parseOperand(value))
                return ::mlir::failure();

            // Parse : type(value)
            ::mlir::Type valueType;
            if (parser.parseColonType(valueType))
                return ::mlir::failure();

            // Parse -> result_types
            ::llvm::SmallVector<::mlir::Type> resultTypes;
            if (parser.parseArrowTypeList(resultTypes))
                return ::mlir::failure();
            result.addTypes(resultTypes);

            // Parse { cases }
            if (parser.parseLBrace())
                return ::mlir::failure();

            // Parse cases (same as SwitchOp)
            ::llvm::SmallVector<int64_t> caseValues;
            ::llvm::SmallVector<int64_t> rangeStarts;
            ::llvm::SmallVector<int64_t> rangeEnds;
            ::llvm::SmallVector<int64_t> caseKinds;
            int64_t defaultCaseIndex = -1;
            int caseIndex = 0;

            while (succeeded(parser.parseOptionalKeyword("case")))
            {
                int64_t caseStart;
                if (succeeded(parser.parseInteger(caseStart)))
                {
                    int64_t rangeEnd;
                    if (nextTokenIsEllipsis(parser))
                    {
                        if (parser.parseEllipsis())
                            return ::mlir::failure();
                        if (parser.parseInteger(rangeEnd))
                            return ::mlir::failure();

                        rangeStarts.push_back(caseStart);
                        rangeEnds.push_back(rangeEnd);
                        caseValues.push_back(0);
                        caseKinds.push_back(1);
                    }
                    else
                    {
                        caseValues.push_back(caseStart);
                        rangeStarts.push_back(0);
                        rangeEnds.push_back(0);
                        caseKinds.push_back(0);
                    }
                }
                else
                {
                    return ::mlir::failure();
                }

                if (failed(parseFatArrow(parser)))
                    return ::mlir::failure();

                std::unique_ptr<::mlir::Region> caseRegion = std::make_unique<::mlir::Region>();
                if (parser.parseRegion(*caseRegion))
                    return ::mlir::failure();

                result.addRegion(std::move(caseRegion));
                caseIndex++;
            }

            if (succeeded(parser.parseOptionalKeyword("else")))
            {
                if (failed(parseFatArrow(parser)))
                    return ::mlir::failure();

                std::unique_ptr<::mlir::Region> elseRegion = std::make_unique<::mlir::Region>();
                if (parser.parseRegion(*elseRegion))
                    return ::mlir::failure();

                result.addRegion(std::move(elseRegion));
                defaultCaseIndex = caseIndex;
                caseKinds.push_back(2);
                caseValues.push_back(0);
                rangeStarts.push_back(0);
                rangeEnds.push_back(0);
            }

            if (parser.parseRBrace())
                return ::mlir::failure();

            if (parser.parseOptionalAttrDict(result.attributes))
                return ::mlir::failure();

            if (parser.resolveOperand(value, valueType, result.operands))
                return ::mlir::failure();

            if (!caseValues.empty())
            {
                auto caseValuesAttr = ::mlir::DenseI64ArrayAttr::get(parser.getContext(), caseValues);
                result.addAttribute("case_values", caseValuesAttr);
            }

            if (!rangeStarts.empty())
            {
                auto rangeStartsAttr = ::mlir::DenseI64ArrayAttr::get(parser.getContext(), rangeStarts);
                result.addAttribute("range_starts", rangeStartsAttr);
            }

            if (!rangeEnds.empty())
            {
                auto rangeEndsAttr = ::mlir::DenseI64ArrayAttr::get(parser.getContext(), rangeEnds);
                result.addAttribute("range_ends", rangeEndsAttr);
            }

            if (!caseKinds.empty())
            {
                auto caseKindsAttr = ::mlir::DenseI64ArrayAttr::get(parser.getContext(), caseKinds);
                result.addAttribute("case_kinds", caseKindsAttr);
            }

            if (defaultCaseIndex >= 0)
            {
                auto defaultIndexAttr = ::mlir::IntegerAttr::get(
                    ::mlir::IntegerType::get(parser.getContext(), 64), defaultCaseIndex);
                result.addAttribute("default_case_index", defaultIndexAttr);
            }

            return ::mlir::success();
        }

        void SwitchExprOp::print(::mlir::OpAsmPrinter &p)
        {
            p << " ";
            p << getValue();
            p << " : ";
            p << getValue().getType();

            if (!getResults().empty())
            {
                p << " -> ";
                llvm::interleaveComma(getResults().getTypes(), p, [&](::mlir::Type type)
                                      { p << type; });
            }

            p << " {";

            printSwitchLikeCases(p, *this);

            p.printNewline();
            p << "}";

            SmallVector<StringRef> elidedAttrs = {
                "case_values", "range_starts", "range_ends", "case_kinds", "default_case_index"};
            p.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
        }

        //===----------------------------------------------------------------------===//
        // SLoadOp Custom Parser/Printer
        //===----------------------------------------------------------------------===//

        ::mlir::ParseResult SLoadOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result)
        {
            // Parse format: "global" : type attr-dict
            std::string globalName;
            if (parser.parseKeywordOrString(&globalName))
                return ::mlir::failure();
            result.addAttribute("global", parser.getBuilder().getStringAttr(globalName));

            ::mlir::Type resultType;
            if (parser.parseColonType(resultType))
                return ::mlir::failure();
            result.addTypes(resultType);

            // Parse optional attributes
            if (parser.parseOptionalAttrDict(result.attributes))
                return ::mlir::failure();

            return ::mlir::success();
        }

        void SLoadOp::print(::mlir::OpAsmPrinter &p)
        {
            // Print format: "global" : type attr-dict
            p << " ";
            p.printAttributeWithoutType(getGlobalAttr());
            p << " : ";
            p << getResult().getType();

            // Print attributes, but elide internal result name attributes
            SmallVector<StringRef> elidedAttrs = {"global", "ora.result_name_0"};
            p.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
        }

        //===----------------------------------------------------------------------===//
        // SLoadOp Result Naming (via OpAsmOpInterface)
        //===----------------------------------------------------------------------===//

        void SLoadOp::getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn)
        {
            // Check if we have a stored result name attribute (set via oraOperationSetResultName)
            auto nameAttr = (*this)->getAttrOfType<StringAttr>("ora.result_name_0");
            if (nameAttr)
            {
                // Use the provided name hint
                setNameFn(getResult(), nameAttr.getValue());
            }
            else
            {
                // Default: use the global name as the result name hint
                // MLIR will automatically uniquify if needed (e.g., %balances, %balances2)
                setNameFn(getResult(), getGlobal());
            }
        }

        namespace
        {
            enum class BinaryIntegerFoldKind
            {
                Add,
                Sub,
                Mul,
            };

            template <typename OpT, BinaryIntegerFoldKind Kind>
            struct FoldBinaryIntegerConstants : public OpRewritePattern<OpT>
            {
                using OpRewritePattern<OpT>::OpRewritePattern;

                LogicalResult matchAndRewrite(OpT op, PatternRewriter &rewriter) const override
                {
                    auto lhsVal = getConstantIntegerAttr(op.getLhs());
                    auto rhsVal = getConstantIntegerAttr(op.getRhs());
                    if (!lhsVal || !rhsVal)
                        return failure();

                    llvm::APInt result = lhsVal->getValue();
                    if constexpr (Kind == BinaryIntegerFoldKind::Add)
                        result += rhsVal->getValue();
                    else if constexpr (Kind == BinaryIntegerFoldKind::Sub)
                        result -= rhsVal->getValue();
                    else
                        result *= rhsVal->getValue();

                    return replaceOpWithConstant(
                        rewriter,
                        op,
                        op.getResult().getType(),
                        result);
                }
            };

            template <typename OpT, BinaryIntegerFoldKind Kind>
            struct FoldBinaryIntegerIdentity : public OpRewritePattern<OpT>
            {
                using OpRewritePattern<OpT>::OpRewritePattern;

                LogicalResult matchAndRewrite(OpT op, PatternRewriter &rewriter) const override
                {
                    if constexpr (Kind == BinaryIntegerFoldKind::Add)
                    {
                        if (isZeroIntegerConstant(op.getRhs()))
                            return replaceOpWithValueIfSameType(rewriter, op, op.getLhs());
                        if (isZeroIntegerConstant(op.getLhs()))
                            return replaceOpWithValueIfSameType(rewriter, op, op.getRhs());
                    }
                    else if constexpr (Kind == BinaryIntegerFoldKind::Sub)
                    {
                        if (isZeroIntegerConstant(op.getRhs()))
                            return replaceOpWithValueIfSameType(rewriter, op, op.getLhs());
                    }
                    else
                    {
                        if (isIntegerConstant(op.getRhs(), 1))
                            return replaceOpWithValueIfSameType(rewriter, op, op.getLhs());
                        if (isIntegerConstant(op.getLhs(), 1))
                            return replaceOpWithValueIfSameType(rewriter, op, op.getRhs());
                        if (isZeroIntegerConstant(op.getLhs()))
                            return replaceOpWithValueIfSameType(rewriter, op, op.getLhs());
                        if (isZeroIntegerConstant(op.getRhs()))
                            return replaceOpWithValueIfSameType(rewriter, op, op.getRhs());
                    }

                    return failure();
                }
            };

            enum class SignedBinaryFoldKind
            {
                Div,
                Rem,
            };

            template <typename OpT, SignedBinaryFoldKind Kind>
            struct FoldSignedAwareBinaryConstants : public OpRewritePattern<OpT>
            {
                using OpRewritePattern<OpT>::OpRewritePattern;

                LogicalResult matchAndRewrite(OpT op, PatternRewriter &rewriter) const override
                {
                    auto lhsVal = getConstantIntegerAttr(op.getLhs());
                    auto rhsVal = getConstantIntegerAttr(op.getRhs());
                    if (!lhsVal || !rhsVal || rhsVal->getValue().isZero())
                        return failure();

                    auto resultOraType = llvm::dyn_cast<ora::IntegerType>(op.getResult().getType());
                    if (!resultOraType)
                        return failure();

                    llvm::APInt result = lhsVal->getValue();
                    if constexpr (Kind == SignedBinaryFoldKind::Div)
                        result = resultOraType.getIsSigned()
                                     ? lhsVal->getValue().sdiv(rhsVal->getValue())
                                     : lhsVal->getValue().udiv(rhsVal->getValue());
                    else
                        result = resultOraType.getIsSigned()
                                     ? lhsVal->getValue().srem(rhsVal->getValue())
                                     : lhsVal->getValue().urem(rhsVal->getValue());

                    return replaceOpWithConstant(
                        rewriter,
                        op,
                        op.getResult().getType(),
                        result);
                }
            };

            struct FoldDivIdentity : public OpRewritePattern<DivOp>
            {
                using OpRewritePattern<DivOp>::OpRewritePattern;

                LogicalResult matchAndRewrite(DivOp op, PatternRewriter &rewriter) const override
                {
                    if (isIntegerConstant(op.getRhs(), 1))
                        return replaceOpWithValueIfSameType(rewriter, op, op.getLhs());
                    return failure();
                }
            };

            struct FoldRemByOne : public OpRewritePattern<RemOp>
            {
                using OpRewritePattern<RemOp>::OpRewritePattern;

                LogicalResult matchAndRewrite(RemOp op, PatternRewriter &rewriter) const override
                {
                    auto rhsVal = getConstantIntegerAttr(op.getRhs());
                    if (!rhsVal || rhsVal->getValue() != 1)
                        return failure();

                    return replaceOpWithConstant(
                        rewriter,
                        op,
                        op.getResult().getType(),
                        llvm::APInt::getZero(rhsVal->getValue().getBitWidth()));
                }
            };

            struct FoldPowerConstants : public OpRewritePattern<PowerOp>
            {
                using OpRewritePattern<PowerOp>::OpRewritePattern;

                LogicalResult matchAndRewrite(PowerOp op, PatternRewriter &rewriter) const override
                {
                    auto baseVal = getConstantIntegerAttr(op.getBase());
                    auto exponentVal = getConstantIntegerAttr(op.getExponent());
                    if (!baseVal || !exponentVal)
                        return failure();

                    auto bitWidth = getIntegerLikeBitWidth(op.getResult().getType());
                    if (!bitWidth)
                        return failure();

                    auto result = computeBoundedWrappingPower(
                        baseVal->getValue(),
                        exponentVal->getValue(),
                        *bitWidth);
                    if (!result)
                        return failure();

                    return replaceOpWithConstant(
                        rewriter,
                        op,
                        op.getResult().getType(),
                        *result);
                }
            };

            struct FoldPowerIdentity : public OpRewritePattern<PowerOp>
            {
                using OpRewritePattern<PowerOp>::OpRewritePattern;

                LogicalResult matchAndRewrite(PowerOp op, PatternRewriter &rewriter) const override
                {
                    if (isZeroIntegerConstant(op.getExponent()))
                    {
                        auto bitWidth = getIntegerLikeBitWidth(op.getResult().getType());
                        if (!bitWidth)
                            return failure();
                        return replaceOpWithConstant(
                            rewriter,
                            op,
                            op.getResult().getType(),
                            llvm::APInt(*bitWidth, 1));
                    }

                    if (isIntegerConstant(op.getExponent(), 1))
                        return replaceOpWithValueIfSameType(rewriter, op, op.getBase());

                    if (isIntegerConstant(op.getBase(), 1))
                        return replaceOpWithValueIfSameType(rewriter, op, op.getBase());

                    return failure();
                }
            };

            enum class ShiftFoldKind
            {
                Left,
                Right,
            };

            template <typename OpT, ShiftFoldKind Kind>
            struct FoldWrappingShiftConstants : public OpRewritePattern<OpT>
            {
                using OpRewritePattern<OpT>::OpRewritePattern;

                LogicalResult matchAndRewrite(OpT op, PatternRewriter &rewriter) const override
                {
                    auto lhsVal = getConstantIntegerAttr(op.getLhs());
                    auto rhsVal = getConstantIntegerAttr(op.getRhs());
                    if (!lhsVal || !rhsVal)
                        return failure();

                    uint64_t shift = rhsVal->getValue().getLimitedValue();
                    if (shift >= lhsVal->getValue().getBitWidth())
                        return failure();

                    llvm::APInt result = lhsVal->getValue();
                    if constexpr (Kind == ShiftFoldKind::Left)
                    {
                        result <<= shift;
                    }
                    else
                    {
                        bool isSigned = false;
                        if (auto resultOraType = llvm::dyn_cast<ora::IntegerType>(op.getResult().getType()))
                            isSigned = resultOraType.getIsSigned();
                        result = isSigned ? lhsVal->getValue().ashr(shift)
                                          : lhsVal->getValue().lshr(shift);
                    }

                    return replaceOpWithConstant(
                        rewriter,
                        op,
                        op.getResult().getType(),
                        result);
                }
            };

            template <typename OpT>
            struct FoldWrappingShiftByZero : public OpRewritePattern<OpT>
            {
                using OpRewritePattern<OpT>::OpRewritePattern;

                LogicalResult matchAndRewrite(OpT op, PatternRewriter &rewriter) const override
                {
                    if (isZeroIntegerConstant(op.getRhs()))
                        return replaceOpWithValueIfSameType(rewriter, op, op.getLhs());
                    return failure();
                }
            };

            template <typename OpT, BinaryIntegerFoldKind Kind>
            static void addBinaryIntegerCanonicalizers(RewritePatternSet &results, MLIRContext *context)
            {
                results.add<
                    FoldBinaryIntegerConstants<OpT, Kind>,
                    FoldBinaryIntegerIdentity<OpT, Kind>>(context);
            }

            template <typename OpT, SignedBinaryFoldKind Kind, typename IdentityPatternT>
            static void addSignedAwareBinaryCanonicalizers(RewritePatternSet &results, MLIRContext *context)
            {
                results.add<
                    FoldSignedAwareBinaryConstants<OpT, Kind>,
                    IdentityPatternT>(context);
            }

            template <typename OpT, ShiftFoldKind Kind>
            static void addWrappingShiftCanonicalizers(RewritePatternSet &results, MLIRContext *context)
            {
                results.add<
                    FoldWrappingShiftConstants<OpT, Kind>,
                    FoldWrappingShiftByZero<OpT>>(context);
            }

            struct FoldCmpSameOperand : public OpRewritePattern<CmpOp>
            {
                using OpRewritePattern<CmpOp>::OpRewritePattern;

                LogicalResult matchAndRewrite(CmpOp op, PatternRewriter &rewriter) const override
                {
                    if (op.getLhs() != op.getRhs())
                        return failure();
                    auto result = compareSameOperand(op.getPredicate());
                    if (!result)
                        return failure();
                    return replaceOpWithBoolConstant(rewriter, op, *result);
                }
            };

            struct FoldCmpConstants : public OpRewritePattern<CmpOp>
            {
                using OpRewritePattern<CmpOp>::OpRewritePattern;

                LogicalResult matchAndRewrite(CmpOp op, PatternRewriter &rewriter) const override
                {
                    auto lhsVal = getConstantIntegerAttr(op.getLhs());
                    auto rhsVal = getConstantIntegerAttr(op.getRhs());

                    if (!lhsVal || !rhsVal)
                        return failure();

                    auto result = compareConstantIntegers(
                        op.getPredicate(),
                        lhsVal->getValue(),
                        rhsVal->getValue());
                    if (!result)
                        return failure();
                    return replaceOpWithBoolConstant(rewriter, op, *result);
                }
            };

            struct FoldErrorIsErrorFromConstructor : public OpRewritePattern<ErrorIsErrorOp>
            {
                using OpRewritePattern<ErrorIsErrorOp>::OpRewritePattern;

                LogicalResult matchAndRewrite(ErrorIsErrorOp op, PatternRewriter &rewriter) const override
                {
                    if (op.getValue().getDefiningOp<ErrorOkOp>())
                        return replaceOpWithBoolConstant(rewriter, op, false);
                    if (op.getValue().getDefiningOp<ErrorErrOp>())
                        return replaceOpWithBoolConstant(rewriter, op, true);
                    return failure();
                }
            };

            template <typename ProjectionOpT, typename ConstructorOpT>
            struct FoldErrorProjectionFromConstructor : public OpRewritePattern<ProjectionOpT>
            {
                using OpRewritePattern<ProjectionOpT>::OpRewritePattern;

                LogicalResult matchAndRewrite(ProjectionOpT op, PatternRewriter &rewriter) const override
                {
                    auto constructor = op.getValue().template getDefiningOp<ConstructorOpT>();
                    if (!constructor || constructor.getValue().getType() != op.getResult().getType())
                        return failure();

                    rewriter.replaceOp(op, constructor.getValue());
                    return success();
                }
            };

            struct FoldAdtTagFromConstruct : public OpRewritePattern<AdtTagOp>
            {
                using OpRewritePattern<AdtTagOp>::OpRewritePattern;

                LogicalResult matchAndRewrite(AdtTagOp op, PatternRewriter &rewriter) const override
                {
                    auto construct = op.getValue().getDefiningOp<AdtConstructOp>();
                    if (!construct)
                        return failure();

                    auto adtType = llvm::dyn_cast<AdtType>(construct.getResult().getType());
                    if (!adtType)
                        return failure();

                    size_t variantIndex = 0;
                    Type payloadType;
                    if (failed(getAdtVariantInfo(op.getOperation(), adtType, construct.getVariantName(), variantIndex, payloadType)))
                        return failure();

                    auto bitWidth = getIntegerLikeBitWidth(op.getResult().getType());
                    if (!bitWidth)
                        return failure();

                    return replaceOpWithConstant(
                        rewriter,
                        op,
                        op.getResult().getType(),
                        llvm::APInt(*bitWidth, variantIndex));
                }
            };

            struct FoldAdtPayloadFromMatchingConstruct : public OpRewritePattern<AdtPayloadOp>
            {
                using OpRewritePattern<AdtPayloadOp>::OpRewritePattern;

                LogicalResult matchAndRewrite(AdtPayloadOp op, PatternRewriter &rewriter) const override
                {
                    auto construct = op.getValue().getDefiningOp<AdtConstructOp>();
                    if (!construct || construct.getVariantName() != op.getVariantName())
                        return failure();

                    auto payloadValues = construct.getPayloadValues();
                    if (payloadValues.size() != 1 || payloadValues.front().getType() != op.getResult().getType())
                        return failure();

                    rewriter.replaceOp(op, payloadValues.front());
                    return success();
                }
            };

            struct FoldTupleExtractFromCreate : public OpRewritePattern<TupleExtractOp>
            {
                using OpRewritePattern<TupleExtractOp>::OpRewritePattern;

                LogicalResult matchAndRewrite(TupleExtractOp op, PatternRewriter &rewriter) const override
                {
                    auto tupleCreate = op.getTupleValue().getDefiningOp<TupleCreateOp>();
                    if (!tupleCreate)
                        return failure();

                    auto index = static_cast<size_t>(op.getIndex());
                    auto elements = tupleCreate.getElements();
                    if (index >= elements.size())
                        return failure();

                    Value element = elements[index];
                    if (element.getType() != op.getResult().getType())
                        return failure();

                    rewriter.replaceOp(op, element);
                    return success();
                }
            };

            struct FoldTupleCreateFromExtracts : public OpRewritePattern<TupleCreateOp>
            {
                using OpRewritePattern<TupleCreateOp>::OpRewritePattern;

                LogicalResult matchAndRewrite(TupleCreateOp op, PatternRewriter &rewriter) const override
                {
                    Value sourceTuple;
                    for (auto [index, element] : llvm::enumerate(op.getElements()))
                    {
                        auto extract = element.getDefiningOp<TupleExtractOp>();
                        if (!extract)
                            return failure();
                        if (extract.getIndex() != static_cast<uint64_t>(index))
                            return failure();
                        if (!sourceTuple)
                        {
                            sourceTuple = extract.getTupleValue();
                            continue;
                        }
                        if (extract.getTupleValue() != sourceTuple)
                            return failure();
                    }

                    if (!sourceTuple || sourceTuple.getType() != op.getResult().getType())
                        return failure();

                    rewriter.replaceOp(op, sourceTuple);
                    return success();
                }
            };

            static LogicalResult getStructConstructorFieldValues(
                Value structValue,
                ValueRange &fieldValues,
                bool requireNamedInstantiateTypeMatch = false)
            {
                if (auto structInit = structValue.getDefiningOp<StructInitOp>())
                {
                    fieldValues = structInit.getFieldValues();
                    return success();
                }

                if (auto structInstantiate = structValue.getDefiningOp<StructInstantiateOp>())
                {
                    if (requireNamedInstantiateTypeMatch)
                    {
                        auto namedStructType = llvm::dyn_cast<StructType>(structValue.getType());
                        if (!namedStructType || namedStructType.getName() != structInstantiate.getStructName())
                            return failure();
                    }
                    fieldValues = structInstantiate.getFieldValues();
                    return success();
                }

                return failure();
            }

            struct FoldStructFieldExtractFromCreate : public OpRewritePattern<StructFieldExtractOp>
            {
                using OpRewritePattern<StructFieldExtractOp>::OpRewritePattern;

                LogicalResult matchAndRewrite(StructFieldExtractOp op, PatternRewriter &rewriter) const override
                {
                    size_t fieldIndex = 0;
                    Type fieldType;
                    if (failed(getStructFieldInfo(
                            op.getOperation(),
                            op.getStructValue().getType(),
                            op.getFieldName(),
                            fieldIndex,
                            fieldType)))
                        return failure();

                    ValueRange fieldValues;
                    if (failed(getStructConstructorFieldValues(op.getStructValue(), fieldValues)))
                        return failure();

                    if (fieldIndex >= fieldValues.size())
                        return failure();

                    Value fieldValue = fieldValues[fieldIndex];
                    if (fieldValue.getType() != op.getResult().getType())
                        return failure();

                    rewriter.replaceOp(op, fieldValue);
                    return success();
                }
            };

            static FailureOr<Value> getStructReconstructionSource(
                Operation *op,
                ValueRange fieldValues,
                Type resultType)
            {
                Value sourceStruct;
                for (auto [index, fieldValue] : llvm::enumerate(fieldValues))
                {
                    auto extract = fieldValue.getDefiningOp<StructFieldExtractOp>();
                    if (!extract)
                        return failure();

                    size_t fieldIndex = 0;
                    Type fieldType;
                    if (failed(getStructFieldInfo(
                            op,
                            extract.getStructValue().getType(),
                            extract.getFieldName(),
                            fieldIndex,
                            fieldType)))
                        return failure();

                    if (fieldIndex != static_cast<size_t>(index))
                        return failure();
                    if (fieldType != fieldValue.getType())
                        return failure();

                    if (!sourceStruct)
                    {
                        sourceStruct = extract.getStructValue();
                        continue;
                    }
                    if (extract.getStructValue() != sourceStruct)
                        return failure();
                }

                if (!sourceStruct || sourceStruct.getType() != resultType)
                    return failure();

                return sourceStruct;
            }

            static LogicalResult validateStructInstantiateResult(StructInstantiateOp op)
            {
                auto resultStructType = llvm::dyn_cast<StructType>(op.getResult().getType());
                if (!resultStructType || resultStructType.getName() != op.getStructName())
                    return failure();
                return success();
            }

            static LogicalResult validateStructInstantiateResult(StructInitOp)
            {
                return success();
            }

            template <typename OpT>
            struct FoldStructCreateFromExtracts : public OpRewritePattern<OpT>
            {
                using OpRewritePattern<OpT>::OpRewritePattern;

                LogicalResult matchAndRewrite(OpT op, PatternRewriter &rewriter) const override
                {
                    if (failed(validateStructInstantiateResult(op)))
                        return failure();

                    FailureOr<Value> sourceStruct = getStructReconstructionSource(
                        op.getOperation(),
                        op.getFieldValues(),
                        op.getResult().getType());
                    if (failed(sourceStruct))
                        return failure();

                    rewriter.replaceOp(op, *sourceStruct);
                    return success();
                }
            };

            struct FoldStructFieldExtractFromUpdate : public OpRewritePattern<StructFieldExtractOp>
            {
                using OpRewritePattern<StructFieldExtractOp>::OpRewritePattern;

                LogicalResult matchAndRewrite(StructFieldExtractOp op, PatternRewriter &rewriter) const override
                {
                    auto update = op.getStructValue().getDefiningOp<StructFieldUpdateOp>();
                    if (!update)
                        return failure();

                    if (update.getFieldName() == op.getFieldName())
                    {
                        if (update.getValue().getType() != op.getResult().getType())
                            return failure();
                        rewriter.replaceOp(op, update.getValue());
                        return success();
                    }

                    rewriter.replaceOpWithNewOp<StructFieldExtractOp>(
                        op,
                        op.getResult().getType(),
                        update.getStructValue(),
                        op.getFieldNameAttr());
                    return success();
                }
            };

            struct FoldStructFieldUpdateNoop : public OpRewritePattern<StructFieldUpdateOp>
            {
                using OpRewritePattern<StructFieldUpdateOp>::OpRewritePattern;

                LogicalResult matchAndRewrite(StructFieldUpdateOp op, PatternRewriter &rewriter) const override
                {
                    auto extract = op.getValue().getDefiningOp<StructFieldExtractOp>();
                    if (!extract)
                        return failure();
                    if (extract.getStructValue() != op.getStructValue())
                        return failure();
                    if (extract.getFieldName() != op.getFieldName())
                        return failure();
                    if (op.getStructValue().getType() != op.getResult().getType())
                        return failure();

                    rewriter.replaceOp(op, op.getStructValue());
                    return success();
                }
            };

            struct FoldStructFieldUpdateIntoInit : public OpRewritePattern<StructFieldUpdateOp>
            {
                using OpRewritePattern<StructFieldUpdateOp>::OpRewritePattern;

                LogicalResult matchAndRewrite(StructFieldUpdateOp op, PatternRewriter &rewriter) const override
                {
                    if (op.getStructValue().getType() != op.getResult().getType())
                        return failure();

                    size_t fieldIndex = 0;
                    Type fieldType;
                    if (failed(getStructFieldInfo(
                            op.getOperation(),
                            op.getStructValue().getType(),
                            op.getFieldName(),
                            fieldIndex,
                            fieldType)))
                        return failure();

                    if (fieldType != op.getValue().getType())
                        return failure();

                    ValueRange fieldValues;
                    if (failed(getStructConstructorFieldValues(
                            op.getStructValue(), fieldValues, /*requireNamedInstantiateTypeMatch=*/true)))
                        return failure();

                    if (fieldIndex >= fieldValues.size())
                        return failure();

                    SmallVector<Value> rebuiltFields(fieldValues.begin(), fieldValues.end());
                    rebuiltFields[fieldIndex] = op.getValue();
                    rewriter.replaceOpWithNewOp<StructInitOp>(
                        op,
                        op.getResult().getType(),
                        rebuiltFields);
                    return success();
                }
            };

            struct FoldStructFieldUpdateOverwrite : public OpRewritePattern<StructFieldUpdateOp>
            {
                using OpRewritePattern<StructFieldUpdateOp>::OpRewritePattern;

                LogicalResult matchAndRewrite(StructFieldUpdateOp op, PatternRewriter &rewriter) const override
                {
                    auto inner = op.getStructValue().getDefiningOp<StructFieldUpdateOp>();
                    if (!inner)
                        return failure();
                    if (inner.getFieldName() != op.getFieldName())
                        return failure();
                    if (inner.getStructValue().getType() != op.getResult().getType())
                        return failure();

                    rewriter.replaceOpWithNewOp<StructFieldUpdateOp>(
                        op,
                        op.getResult().getType(),
                        inner.getStructValue(),
                        op.getFieldNameAttr(),
                        op.getValue());
                    return success();
                }
            };
        }

#define DEFINE_ORA_BINARY_CANONICALIZER(OpT, Kind)                                 \
        void OpT::getCanonicalizationPatterns(RewritePatternSet &results,          \
                                              MLIRContext *context)                \
        {                                                                          \
            addBinaryIntegerCanonicalizers<OpT, BinaryIntegerFoldKind::Kind>(      \
                results, context);                                                 \
        }

#define DEFINE_ORA_SIGNED_BINARY_CANONICALIZER(OpT, Kind, IdentityPatternT)        \
        void OpT::getCanonicalizationPatterns(RewritePatternSet &results,          \
                                              MLIRContext *context)                \
        {                                                                          \
            addSignedAwareBinaryCanonicalizers<OpT, SignedBinaryFoldKind::Kind,    \
                                               IdentityPatternT>(results, context);\
        }

#define DEFINE_ORA_SHIFT_CANONICALIZER(OpT, Kind)                                  \
        void OpT::getCanonicalizationPatterns(RewritePatternSet &results,          \
                                              MLIRContext *context)                \
        {                                                                          \
            addWrappingShiftCanonicalizers<OpT, ShiftFoldKind::Kind>(              \
                results, context);                                                 \
        }

#define DEFINE_ORA_PATTERN_CANONICALIZER(OpT, ...)                                 \
        void OpT::getCanonicalizationPatterns(RewritePatternSet &results,          \
                                              MLIRContext *context)                \
        {                                                                          \
            results.add<__VA_ARGS__>(context);                                     \
        }

        DEFINE_ORA_BINARY_CANONICALIZER(AddOp, Add)
        DEFINE_ORA_BINARY_CANONICALIZER(AddWrappingOp, Add)
        DEFINE_ORA_BINARY_CANONICALIZER(MulOp, Mul)
        DEFINE_ORA_BINARY_CANONICALIZER(MulWrappingOp, Mul)
        DEFINE_ORA_BINARY_CANONICALIZER(SubOp, Sub)
        DEFINE_ORA_BINARY_CANONICALIZER(SubWrappingOp, Sub)
        DEFINE_ORA_SIGNED_BINARY_CANONICALIZER(DivOp, Div, FoldDivIdentity)
        DEFINE_ORA_SIGNED_BINARY_CANONICALIZER(RemOp, Rem, FoldRemByOne)
        DEFINE_ORA_PATTERN_CANONICALIZER(PowerOp, FoldPowerConstants, FoldPowerIdentity)
        DEFINE_ORA_SHIFT_CANONICALIZER(ShlWrappingOp, Left)
        DEFINE_ORA_SHIFT_CANONICALIZER(ShrWrappingOp, Right)
        DEFINE_ORA_PATTERN_CANONICALIZER(CmpOp, FoldCmpSameOperand, FoldCmpConstants)
        DEFINE_ORA_PATTERN_CANONICALIZER(ErrorIsErrorOp, FoldErrorIsErrorFromConstructor)
        DEFINE_ORA_PATTERN_CANONICALIZER(ErrorUnwrapOp, FoldErrorProjectionFromConstructor<ErrorUnwrapOp, ErrorOkOp>)
        DEFINE_ORA_PATTERN_CANONICALIZER(ErrorGetErrorOp, FoldErrorProjectionFromConstructor<ErrorGetErrorOp, ErrorErrOp>)
        DEFINE_ORA_PATTERN_CANONICALIZER(AdtTagOp, FoldAdtTagFromConstruct)
        DEFINE_ORA_PATTERN_CANONICALIZER(AdtPayloadOp, FoldAdtPayloadFromMatchingConstruct)
        DEFINE_ORA_PATTERN_CANONICALIZER(TupleExtractOp, FoldTupleExtractFromCreate)
        DEFINE_ORA_PATTERN_CANONICALIZER(TupleCreateOp, FoldTupleCreateFromExtracts)
        DEFINE_ORA_PATTERN_CANONICALIZER(StructInitOp, FoldStructCreateFromExtracts<StructInitOp>)
        DEFINE_ORA_PATTERN_CANONICALIZER(StructInstantiateOp, FoldStructCreateFromExtracts<StructInstantiateOp>)
        DEFINE_ORA_PATTERN_CANONICALIZER(StructFieldExtractOp, FoldStructFieldExtractFromCreate, FoldStructFieldExtractFromUpdate)
        DEFINE_ORA_PATTERN_CANONICALIZER(StructFieldUpdateOp, FoldStructFieldUpdateNoop, FoldStructFieldUpdateIntoInit, FoldStructFieldUpdateOverwrite)

#undef DEFINE_ORA_BINARY_CANONICALIZER
#undef DEFINE_ORA_SIGNED_BINARY_CANONICALIZER
#undef DEFINE_ORA_SHIFT_CANONICALIZER
#undef DEFINE_ORA_PATTERN_CANONICALIZER

        //===----------------------------------------------------------------------===//
        // Memory Effects
        //===----------------------------------------------------------------------===//

        // Storage load: reads from global storage
        void SLoadOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects)
        {
            effects.emplace_back(MemoryEffects::Read::get(), SideEffects::DefaultResource::get());
        }

        // Storage store: writes to global storage
        void SStoreOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects)
        {
            effects.emplace_back(MemoryEffects::Write::get(), SideEffects::DefaultResource::get());
        }

        // Memory load: reads from memory
        void MLoadOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects)
        {
            effects.emplace_back(MemoryEffects::Read::get(), SideEffects::DefaultResource::get());
        }

        // Memory store: writes to memory
        void MStoreOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects)
        {
            effects.emplace_back(MemoryEffects::Write::get(), SideEffects::DefaultResource::get());
        }

        // Byte memory load: reads from memory
        void MLoad8Op::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects)
        {
            effects.emplace_back(MemoryEffects::Read::get(), SideEffects::DefaultResource::get());
        }

        // Byte memory store: writes to memory
        void MStore8Op::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects)
        {
            effects.emplace_back(MemoryEffects::Write::get(), SideEffects::DefaultResource::get());
        }

        // Transient storage load: reads from transient storage
        void TLoadOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects)
        {
            effects.emplace_back(MemoryEffects::Read::get(), SideEffects::DefaultResource::get());
        }

        // Transient storage store: writes to transient storage
        void TStoreOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects)
        {
            effects.emplace_back(MemoryEffects::Write::get(), SideEffects::DefaultResource::get());
        }

        // Map get: reads from storage (maps are storage-based)
        void MapGetOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects)
        {
            // Map operations read from storage (the map is storage-based)
            effects.emplace_back(MemoryEffects::Read::get(), SideEffects::DefaultResource::get());
        }

        // Map store: writes to storage (maps are storage-based)
        void MapStoreOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects)
        {
            // Map operations write to storage (the map is storage-based)
            effects.emplace_back(MemoryEffects::Write::get(), SideEffects::DefaultResource::get());
        }

        //===----------------------------------------------------------------------===//
        // Result Naming (via OpAsmOpInterface)
        //===----------------------------------------------------------------------===//

        // Arithmetic operations: use semantic names
        void AddOp::getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn)
        {
            auto nameAttr = (*this)->getAttrOfType<StringAttr>("ora.result_name_0");
            if (nameAttr)
            {
                setNameFn(getResult(), nameAttr.getValue());
            }
            else
            {
                // Default semantic name for addition
                setNameFn(getResult(), "sum");
            }
        }

        void SubOp::getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn)
        {
            auto nameAttr = (*this)->getAttrOfType<StringAttr>("ora.result_name_0");
            if (nameAttr)
            {
                setNameFn(getResult(), nameAttr.getValue());
            }
            else
            {
                // Default semantic name for subtraction
                setNameFn(getResult(), "difference");
            }
        }

        void MulOp::getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn)
        {
            auto nameAttr = (*this)->getAttrOfType<StringAttr>("ora.result_name_0");
            if (nameAttr)
            {
                setNameFn(getResult(), nameAttr.getValue());
            }
            else
            {
                // Default semantic name for multiplication
                setNameFn(getResult(), "product");
            }
        }

        void DivOp::getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn)
        {
            auto nameAttr = (*this)->getAttrOfType<StringAttr>("ora.result_name_0");
            if (nameAttr)
            {
                setNameFn(getResult(), nameAttr.getValue());
            }
            else
            {
                // Default semantic name for division
                setNameFn(getResult(), "quotient");
            }
        }

        // Memory load: use variable name
        void MLoadOp::getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn)
        {
            auto nameAttr = (*this)->getAttrOfType<StringAttr>("ora.result_name_0");
            if (nameAttr)
            {
                setNameFn(getResult(), nameAttr.getValue());
            }
            else
            {
                // Default: use the variable name as the result name hint
                setNameFn(getResult(), getVariable());
            }
        }

        // Map get: use map name
        void MapGetOp::getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn)
        {
            auto nameAttr = (*this)->getAttrOfType<StringAttr>("ora.result_name_0");
            if (nameAttr)
            {
                setNameFn(getResult(), nameAttr.getValue());
            }
            else
            {
                // Default: use "value" as semantic name for map get
                setNameFn(getResult(), "value");
            }
        }

    } // namespace ora
} // namespace mlir

// Include the generated dialect definitions implementation
#include "OraDialect.cpp.inc"
